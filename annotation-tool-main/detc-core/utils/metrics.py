"""
Detection metrics: per-class AP, mAP@0.5, mAP@0.5:0.95.

Usage:
    meter = MeanAveragePrecision(nc=5)
    for batch in val_loader:
        boxes, scores = model(batch["imgs"])
        dets = nms_detections(boxes, scores, conf=0.25, iou=0.45)
        meter.update(dets, batch["labels"])
    results = meter.compute()
    print(results)
"""

from __future__ import annotations
from collections import defaultdict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms_detections(
    boxes:   torch.Tensor,   # (B, N, 4)  xyxy pixel
    scores:  torch.Tensor,   # (B, N, nc) sigmoid
    conf:    float = 0.25,
    iou_thr: float = 0.45,
    max_det: int   = 300,
) -> list[np.ndarray]:
    """Apply per-class NMS, return list of (M, 6) arrays per image.

    Each row: [x1, y1, x2, y2, score, class_id]
    """
    results = []
    B = boxes.shape[0]
    for bi in range(B):
        b = boxes[bi]     # (N, 4)
        s = scores[bi]    # (N, nc)

        cls_ids = s.argmax(1)                    # (N,)
        conf_s  = s.max(1).values                # (N,)

        keep = conf_s >= conf
        b, s_k, cls_ids = b[keep], conf_s[keep], cls_ids[keep]

        if b.shape[0] == 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue

        # Torchvision batched_nms
        from torchvision.ops import batched_nms
        idx = batched_nms(b.float(), s_k.float(), cls_ids, iou_thr)
        idx = idx[:max_det]

        det = torch.cat([
            b[idx],
            s_k[idx, None],
            cls_ids[idx, None].float(),
        ], 1).cpu().numpy()
        results.append(det)
    return results


# ---------------------------------------------------------------------------
# AP computation (VOC-style 11-point interpolation)
# ---------------------------------------------------------------------------

def _ap_from_pr(prec: np.ndarray, rec: np.ndarray) -> float:
    """Compute AP from precision-recall arrays using 101-point interpolation."""
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    # monotonically decreasing precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    # sum over threshold steps
    t = np.linspace(0, 1, 101)
    y = np.interp(t, mrec, mpre)
    # Manual trapezoidal rule — avoids depending on np.trapezoid (NumPy 2.0+
    # only) vs np.trapz (deprecated in 2.0, removed in later versions).
    # Works identically on any NumPy version.
    ap = float(np.sum((t[1:] - t[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return ap


def compute_ap(
    det_boxes:  np.ndarray,    # (M, 4)  xyxy detected
    det_scores: np.ndarray,    # (M,)
    gt_boxes:   np.ndarray,    # (G, 4)  xyxy ground-truth
    iou_thr:    float = 0.5,
) -> float:
    """Compute AP for one class at one IoU threshold."""
    if len(gt_boxes) == 0 and len(det_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 or len(det_boxes) == 0:
        return 0.0

    order   = np.argsort(-det_scores)
    det_boxes  = det_boxes[order]

    n_gt  = len(gt_boxes)
    used  = np.zeros(n_gt, dtype=bool)
    tp    = np.zeros(len(det_boxes))
    fp    = np.zeros(len(det_boxes))

    for di, db in enumerate(det_boxes):
        iou_best = -1.0
        best_gi  = -1
        for gi, gb in enumerate(gt_boxes):
            if used[gi]:
                continue
            ix1 = max(db[0], gb[0]); iy1 = max(db[1], gb[1])
            ix2 = min(db[2], gb[2]); iy2 = min(db[3], gb[3])
            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter   = inter_w * inter_h
            union   = ((db[2]-db[0])*(db[3]-db[1]) +
                       (gb[2]-gb[0])*(gb[3]-gb[1]) - inter)
            iou = inter / max(union, 1e-7)
            if iou > iou_best:
                iou_best = iou
                best_gi  = gi
        if iou_best >= iou_thr and best_gi >= 0:
            tp[di]        = 1
            used[best_gi] = True
        else:
            fp[di] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    prec   = tp_cum / (tp_cum + fp_cum + 1e-7)
    rec    = tp_cum / (n_gt + 1e-7)
    return _ap_from_pr(prec, rec)


# ---------------------------------------------------------------------------
# Mean Average Precision meter
# ---------------------------------------------------------------------------

class MeanAveragePrecision:
    """Streaming mAP computation over a validation dataset.

    Usage:
        meter = MeanAveragePrecision(nc=5)
        for det, gt in ...:
            meter.update(det, gt)
        results = meter.compute()
    """

    def __init__(
        self,
        nc: int = 5,
        iou_thrs: list[float] | None = None,
        classes: list[str] | None = None,
    ) -> None:
        self.nc       = nc
        self.iou_thrs = iou_thrs or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        # Class names for the CURRENT run — always passed in explicitly;
        # this package has no fixed class list.
        self.classes  = classes
        # Per-class accumulators: list of (det_boxes, det_scores, gt_boxes)
        self._dets: dict[int, list[tuple]] = defaultdict(list)
        self._gts:  dict[int, list[np.ndarray]] = defaultdict(list)

    def reset(self) -> None:
        self._dets.clear()
        self._gts.clear()

    def update(
        self,
        detections: list[np.ndarray],   # list of (M, 6) per image
        gt_labels:  torch.Tensor,        # (K, 6) [bi, cls, cx, cy, w, h] norm
        img_size:   int = 640,
    ) -> None:
        B = len(detections)
        for bi in range(B):
            det = detections[bi]           # (M, 6): x1,y1,x2,y2,score,cls
            gt  = gt_labels[gt_labels[:, 0] == bi]  # (G, 6)

            # Convert GT to xyxy pixel
            gt_xyxy = np.zeros((gt.shape[0], 5), dtype=np.float32)
            if gt.shape[0]:
                g = gt.cpu().numpy()
                g_cls = g[:, 1].astype(int)
                g_cx  = g[:, 2] * img_size
                g_cy  = g[:, 3] * img_size
                g_w   = g[:, 4] * img_size
                g_h   = g[:, 5] * img_size
                x1 = g_cx - g_w / 2;  y1 = g_cy - g_h / 2
                x2 = g_cx + g_w / 2;  y2 = g_cy + g_h / 2
                for gi, cls in enumerate(g_cls):
                    self._gts[cls].append(np.array([x1[gi], y1[gi], x2[gi], y2[gi]]))

            for row in det:
                cls = int(row[5])
                self._dets[cls].append((row[:4], row[4]))

    def compute(self) -> dict[str, float]:
        """Compute per-class AP and mAP@0.5, mAP@0.5:0.95."""
        ap50_per_cls   = {}
        ap5095_per_cls = {}

        for c in range(self.nc):
            gt_boxes = np.array(self._gts.get(c, []))
            dets     = self._dets.get(c, [])
            if not dets:
                det_boxes  = np.zeros((0, 4))
                det_scores = np.zeros(0)
            else:
                det_boxes  = np.stack([d[0] for d in dets])
                det_scores = np.array([d[1] for d in dets])

            if gt_boxes.ndim == 1:
                gt_boxes = gt_boxes.reshape(-1, 4)

            ap50 = compute_ap(det_boxes, det_scores, gt_boxes, 0.5)
            ap5095 = np.mean([
                compute_ap(det_boxes, det_scores, gt_boxes, t)
                for t in self.iou_thrs
            ])
            name = self.classes[c] if self.classes and c < len(self.classes) else str(c)
            ap50_per_cls[name]   = ap50
            ap5095_per_cls[name] = ap5095

        map50   = float(np.mean(list(ap50_per_cls.values())))
        map5095 = float(np.mean(list(ap5095_per_cls.values())))

        return {
            "mAP@0.5":      map50,
            "mAP@0.5:0.95": map5095,
            "per_class_AP50":   ap50_per_cls,
            "per_class_AP5095": ap5095_per_cls,
        }

    def __repr__(self) -> str:
        return f"MeanAveragePrecision(nc={self.nc})"
