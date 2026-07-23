"""
DETC_Loss — detection training loss.

Components:
  TaskAlignedAssigner  — assigns ground-truth boxes to anchor points
  BboxLoss             — box regression loss (CIoU) + distribution loss
  DetectionLoss        — combines BboxLoss with binary-cross-entropy class loss

Assignment strategy:
  For each ground-truth box, an alignment score is computed at every anchor:
    score = cls_score ^ alpha  *  IoU ^ beta
  The top-k anchors per ground-truth (by score) are marked positive.
  When one anchor is claimed by multiple ground-truths, the higher-scoring
  claim wins.

Loss terms:
  Box   — Complete-IoU loss between predicted and target boxes (positives only).
  Dist  — Cross-entropy between the predicted edge distribution and the ideal
          discrete distribution that encodes the target distance (positives only).
  Cls   — Binary cross-entropy at every anchor; positive anchors receive an
          IoU-weighted soft target rather than a hard 1.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor,
                 eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU between two sets of xyxy boxes.

    Args:
        box1: (M, 4)
        box2: (N, 4)

    Returns: (M, N) IoU matrix.
    """
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    inter    = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    union = area1[:, None] + area2[None, :] - inter + eps
    return inter / union


def ciou_loss(pred: torch.Tensor, gt: torch.Tensor,
              eps: float = 1e-7) -> torch.Tensor:
    """Element-wise Complete-IoU loss for matched xyxy box pairs.

    Args:
        pred: (N, 4)
        gt:   (N, 4)

    Returns: (N,) per-pair loss values.
    """
    pw = (pred[:, 2] - pred[:, 0]).clamp(0)
    ph = (pred[:, 3] - pred[:, 1]).clamp(0)
    gw = (gt[:, 2]   - gt[:, 0]).clamp(0)
    gh = (gt[:, 3]   - gt[:, 1]).clamp(0)

    ix1 = torch.max(pred[:, 0], gt[:, 0])
    iy1 = torch.max(pred[:, 1], gt[:, 1])
    ix2 = torch.min(pred[:, 2], gt[:, 2])
    iy2 = torch.min(pred[:, 3], gt[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    union = pw * ph + gw * gh - inter + eps
    iou   = inter / union

    # Diagonal of smallest enclosing box
    cw = torch.max(pred[:, 2], gt[:, 2]) - torch.min(pred[:, 0], gt[:, 0])
    ch = torch.max(pred[:, 3], gt[:, 3]) - torch.min(pred[:, 1], gt[:, 1])
    c2 = cw ** 2 + ch ** 2 + eps

    # Squared centre-point distance
    pcx = (pred[:, 0] + pred[:, 2]) / 2
    pcy = (pred[:, 1] + pred[:, 3]) / 2
    gcx = (gt[:, 0]   + gt[:, 2])   / 2
    gcy = (gt[:, 1]   + gt[:, 3])   / 2
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    # Aspect-ratio consistency term
    v = (4 / math.pi ** 2) * (
        torch.atan(gw / (gh + eps)) - torch.atan(pw / (ph + eps))
    ) ** 2
    with torch.no_grad():
        alpha_v = v / (1 - iou + v + eps)

    return 1.0 - (iou - (rho2 / c2 + v * alpha_v))


# ---------------------------------------------------------------------------
# Distribution loss
# ---------------------------------------------------------------------------

def dist_loss(pred_dist: torch.Tensor, gt_dist: torch.Tensor,
              num_bins: int) -> torch.Tensor:
    """Cross-entropy loss between predicted and target distance distributions.

    The target distance is represented as a mixture of two adjacent bin
    labels — the floor and the ceiling of the continuous target value,
    weighted by how close the target is to each.

    Args:
        pred_dist: (N, 4*num_bins) raw logits per anchor.
        gt_dist:   (N, 4)          target distances in feature-map units.
        num_bins:  Number of discrete bins.

    Returns: scalar mean loss.
    """
    n = pred_dist.shape[0]
    if n == 0:
        return pred_dist.sum() * 0

    gt_dist = gt_dist.clamp(0, num_bins - 1.01)
    tl = gt_dist.long()                          # lower bin index
    tr = (tl + 1).clamp(max=num_bins - 1)        # upper bin index
    wl = tr.float() - gt_dist                    # weight for lower
    wr = 1.0 - wl

    pred = pred_dist.view(n * 4, num_bins)        # (N*4, num_bins)
    tl_f = tl.reshape(n * 4)
    tr_f = tr.reshape(n * 4)
    wl_f = wl.reshape(n * 4)
    wr_f = wr.reshape(n * 4)

    loss = (
        F.cross_entropy(pred, tl_f, reduction="none") * wl_f +
        F.cross_entropy(pred, tr_f, reduction="none") * wr_f
    )
    return loss.mean()


# ---------------------------------------------------------------------------
# Box-coordinate conversion
# ---------------------------------------------------------------------------

def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) → (x1, y1, x2, y2)."""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], -1)


# ---------------------------------------------------------------------------
# Task-Aligned Assigner
# ---------------------------------------------------------------------------

class TaskAlignedAssigner:
    """Assign ground-truth boxes to anchor points.

    For each GT, the alignment score at every anchor is:
        score = cls_score ^ alpha * IoU ^ beta
    The top-k anchors (by score, restricted to anchors inside the GT box)
    are marked as positives for that GT.

    Args:
        topk:  Maximum positive anchors per GT box.
        alpha: Exponent on classification score.
        beta:  Exponent on IoU.
    """

    def __init__(self, topk: int = 10, alpha: float = 0.5,
                 beta: float = 6.0) -> None:
        self.topk  = topk
        self.alpha = alpha
        self.beta  = beta

    @torch.no_grad()
    def assign(
        self,
        pred_scores:  torch.Tensor,   # (B, N, nc)  sigmoid
        pred_boxes:   torch.Tensor,   # (B, N, 4)   xyxy pixel
        anchor_pts:   torch.Tensor,   # (N, 2)      cell-centre pixel coords
        gt_labels:    torch.Tensor,   # (M_total, 6) [batch_idx, cls, cx, cy, w, h] normalised
        img_size:     int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run assignment for a whole batch.

        Returns:
            pos_mask:   (B, N)     True at positive anchor positions.
            tgt_boxes:  (B, N, 4)  target xyxy boxes (zero at negatives).
            tgt_scores: (B, N, nc) soft class targets (IoU-weighted at positives).
            tgt_cls:    (B, N)     assigned class index, -1 at negatives.
        """
        B, N, nc = pred_scores.shape
        device   = pred_scores.device

        pos_mask   = torch.zeros(B, N, dtype=torch.bool,    device=device)
        tgt_boxes  = torch.zeros(B, N, 4,                   device=device)
        tgt_scores = torch.zeros(B, N, nc,                  device=device)
        tgt_cls    = torch.full((B, N), -1, dtype=torch.long, device=device)

        for bi in range(B):
            gt_bi = gt_labels[gt_labels[:, 0] == bi]       # (M, 6)
            if gt_bi.shape[0] == 0:
                continue

            gt_cls  = gt_bi[:, 1].long()
            gt_xyxy = xywh2xyxy(gt_bi[:, 2:6] * img_size) # (M, 4) pixel

            # Restrict to anchors whose centre lies inside each GT box
            ap = anchor_pts                                  # (N, 2)
            in_box = (
                (ap[None, :, 0] > gt_xyxy[:, 0:1]) &
                (ap[None, :, 1] > gt_xyxy[:, 1:2]) &
                (ap[None, :, 0] < gt_xyxy[:, 2:3]) &
                (ap[None, :, 1] < gt_xyxy[:, 3:4])
            )                                                # (M, N)

            iou_mat  = box_iou_xyxy(gt_xyxy, pred_boxes[bi])# (M, N)

            gt_cls_e     = gt_cls[:, None].expand(-1, N)
            gt_cls_score = pred_scores[bi].T[gt_cls_e,
                           torch.arange(N, device=device)[None]]  # (M, N)

            align = (
                gt_cls_score.clamp(0) ** self.alpha *
                iou_mat.clamp(0)      ** self.beta  *
                in_box.float()
            )

            M      = gt_xyxy.shape[0]
            topk_k = min(self.topk, N)
            _, topk_idx = align.topk(topk_k, dim=1)         # (M, topk)

            # Resolve conflicts: highest alignment score wins
            best_iou = torch.full((N,), -1.0, device=device)
            best_gt  = torch.full((N,), -1,   dtype=torch.long, device=device)
            for gi in range(M):
                for ai in topk_idx[gi]:
                    ai = ai.item()
                    if align[gi, ai] > best_iou[ai]:
                        best_iou[ai] = align[gi, ai]
                        best_gt[ai]  = gi

            pos_i = best_gt >= 0                             # (N,)
            pos_mask[bi] = pos_i

            if pos_i.any():
                gi_pos = best_gt[pos_i]
                tgt_boxes[bi, pos_i] = gt_xyxy[gi_pos]
                tgt_cls[bi, pos_i]   = gt_cls[gi_pos]
                for k, ai in enumerate(pos_i.nonzero(as_tuple=False)[:, 0]):
                    gi  = gi_pos[k]
                    c   = gt_cls[gi]
                    tgt_scores[bi, ai, c] = iou_mat[gi, ai].clamp(0)

        return pos_mask, tgt_boxes, tgt_scores, tgt_cls


# ---------------------------------------------------------------------------
# BboxLoss
# ---------------------------------------------------------------------------

class BboxLoss(nn.Module):
    """CIoU box loss + distribution loss for box-edge regression."""

    def __init__(self, num_bins: int = 16) -> None:
        super().__init__()
        self.num_bins = num_bins

    def forward(
        self,
        pred_dist:      torch.Tensor,   # (B, N, 4*num_bins) raw box logits
        pred_boxes:     torch.Tensor,   # (B, N, 4)   decoded xyxy
        anchor_pts:     torch.Tensor,   # (N, 2)      cell centres pixel
        strides:        torch.Tensor,   # (N, 1)      per-anchor stride
        tgt_boxes:      torch.Tensor,   # (B, N, 4)   xyxy GT
        pos_mask:       torch.Tensor,   # (B, N)      bool
        norm:           torch.Tensor,   # scalar      loss normalisation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pos_mask.sum() == 0:
            z = pred_dist.sum() * 0
            return z, z

        pb  = pred_boxes[pos_mask]      # (N_pos, 4)
        tb  = tgt_boxes[pos_mask]       # (N_pos, 4)

        loss_iou = ciou_loss(pb, tb).sum() / norm.clamp(1)

        # Convert pixel xyxy targets to distance targets (feature-map units)
        ap_b = anchor_pts[None].expand(pos_mask.shape[0], -1, -1)[pos_mask]
        st_b = strides.squeeze(1)[None].expand(pos_mask.shape[0], -1)[pos_mask]

        cx, cy = ap_b[:, 0], ap_b[:, 1]
        l_tgt  = (cx - tb[:, 0] / st_b).clamp(0, self.num_bins - 1)
        t_tgt  = (cy - tb[:, 1] / st_b).clamp(0, self.num_bins - 1)
        r_tgt  = (tb[:, 2] / st_b - cx).clamp(0, self.num_bins - 1)
        b_tgt  = (tb[:, 3] / st_b - cy).clamp(0, self.num_bins - 1)
        dist_tgt = torch.stack([l_tgt, t_tgt, r_tgt, b_tgt], -1)  # (N_pos, 4)

        pd = pred_dist[pos_mask]        # (N_pos, 4*num_bins)
        loss_dist = dist_loss(pd, dist_tgt, self.num_bins) / norm.clamp(1)

        return loss_iou, loss_dist


# ---------------------------------------------------------------------------
# DetectionLoss
# ---------------------------------------------------------------------------

class DetectionLoss(nn.Module):
    """Combined detection loss: box regression + class prediction.

    Args:
        nc:      Number of classes.
        num_bins: Distance-distribution bins (must match head).
        box_w:   Weight on the box loss term.
        cls_w:   Weight on the class loss term.
        dist_w:  Weight on the distribution loss term.
    """

    def __init__(
        self,
        nc:          int              = 5,
        num_bins:    int              = 16,
        box_w:       float            = 7.5,
        cls_w:       float            = 0.5,
        dist_w:      float            = 1.5,
        cls_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.nc       = nc
        self.num_bins = num_bins
        self.box_w    = box_w
        self.cls_w    = cls_w
        self.dist_w   = dist_w

        # Per-class classification weight — registered as a buffer so it
        # automatically moves to the right device (CPU/GPU) with the model.
        # Default: all classes weighted equally (1.0).
        if cls_weights is not None:
            self.register_buffer(
                "cls_pw", torch.tensor(cls_weights, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "cls_pw", torch.ones(nc, dtype=torch.float32)
            )

        self.assigner  = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(num_bins)
        self.bce       = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        raw_preds:   list[torch.Tensor],   # list[(B, 4*num_bins+nc, H_i, W_i)]
        gt_labels:   torch.Tensor,          # (M_total, 6) [bi, cls, cx, cy, w, h] norm
        anchor_pts:  torch.Tensor,          # (N_total, 2) pixel
        strides:     torch.Tensor,          # (N_total, 1) pixel
        head,                               # DETC_Head instance (for decode)
        img_size:    int,
    ) -> dict[str, torch.Tensor]:
        device = raw_preds[0].device
        B      = raw_preds[0].shape[0]

        # Flatten raw predictions across scales
        box_raws, cls_raws = [], []
        for t in raw_preds:
            b, c, h, w = t.shape
            t_flat = t.view(b, c, -1)
            box_raws.append(t_flat[:, :4 * self.num_bins])
            cls_raws.append(t_flat[:, 4 * self.num_bins:])

        box_raw = torch.cat(box_raws, 2)    # (B, 4*num_bins, N)
        cls_raw = torch.cat(cls_raws, 2)    # (B, nc, N)

        # Decode predictions for TAL assignment (no_grad — assigner is not differentiable)
        with torch.no_grad():
            dummy = [
                torch.empty(1, 1, t.shape[2], t.shape[3], device=device)
                for t in raw_preds
            ]
            anchor_pts, strides = head._make_anchors(
                dummy, dtype=raw_preds[0].dtype, device=device
            )
            dist_assign   = head.dfl(box_raw)
            pred_boxes_ng = head.dist2bbox(dist_assign, anchor_pts, strides)
            pred_scores   = cls_raw.permute(0, 2, 1).sigmoid()

        # anchor_pts are in feature-map units; convert to pixel space for in_box test
        anchor_pts_px = anchor_pts * strides   # (N, 2) pixel coords
        pos_mask, tgt_boxes, tgt_scores, _ = self.assigner.assign(
            pred_scores, pred_boxes_ng, anchor_pts_px, gt_labels, img_size
        )

        norm = tgt_scores.sum().clamp(1)

        # Localisation-quality scalar per anchor, from the box-distribution
        # shape. Computed WITH gradients so the quality sub-network actually learns.
        q_list = []
        for t in raw_preds:
            b_, _, h_, w_ = t.shape
            q = head.dgqp(t[:, :4 * self.num_bins])    # (B, 1, H, W)
            q_list.append(q.view(b_, 1, -1))           # (B, 1, H*W)
        q_raw = torch.cat(q_list, 2)                   # (B, 1, N)

        # Class loss — joint (classification x quality) vs IoU-soft target.
        # Manual BCE in fp32 so it is safe under mixed precision (AMP).
        cls_prob = cls_raw.permute(0, 2, 1).sigmoid().float()   # (B, N, nc)
        quality  = q_raw.permute(0, 2, 1).float()               # (B, N, 1)
        joint    = (cls_prob * quality).clamp(1e-6, 1.0 - 1e-6)  # (B, N, nc)
        # Per-element BCE, then multiply by per-class weight before summing.
        # cls_pw is (nc,) which broadcasts to (1, 1, nc) against (B, N, nc).
        bce      = -(tgt_scores * joint.log()
                     + (1.0 - tgt_scores) * (1.0 - joint).log())  # (B, N, nc)
        loss_cls = (bce * self.cls_pw).sum() / norm

        # Decode pred_boxes WITH gradients for CIoU loss
        pred_dist  = box_raw.permute(0, 2, 1)           # (B, N, 4*num_bins)
        dist_grad  = head.dfl(box_raw)
        pred_boxes = head.dist2bbox(dist_grad, anchor_pts, strides)

        # Box + distribution loss — positive anchors only
        loss_iou, loss_dist = self.bbox_loss(
            pred_dist, pred_boxes, anchor_pts, strides,
            tgt_boxes, pos_mask, norm,
        )

        total = self.box_w * loss_iou + self.cls_w * loss_cls + self.dist_w * loss_dist

        return {
            "loss":      total,
            "loss_box":  loss_iou.detach(),
            "loss_cls":  loss_cls.detach(),
            "loss_dist": loss_dist.detach(),
        }
