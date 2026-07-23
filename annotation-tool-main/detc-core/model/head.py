"""
DETC_Head — anchor-free detection head.

Two decoupled branches per detection scale:
  cv2 : box branch  → 4 * num_bins raw values per anchor (one distribution per edge)
  cv3 : class branch → nc logits per anchor

Training  → returns list[(B, 4*num_bins + nc, H_i, W_i)] per scale.
Inference → returns decoded (boxes_xyxy, class_scores) ready for NMS.

Box decoding:
  Each anchor sits at the centre of a grid cell. The network predicts how far
  each box edge is from that centre. Given cell centre (cx, cy) and predicted
  distances (l, t, r, b) in feature-map units:

    x1 = cx*stride − l*stride
    y1 = cy*stride − t*stride
    x2 = cx*stride + r*stride
    y2 = cy*stride + b*stride

  The distance-projection module converts the raw per-edge distribution
  (num_bins values) to a single expected-value distance before these
  formulae are applied.

Anchor points are built lazily on the first forward call and cached.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Conv, DistanceProjection, DistributionQuality


class DETC_Head(nn.Module):
    """Decoupled anchor-free detection head with distribution-based box regression.

    Args:
        nc:          Number of object classes.
        in_channels: Channel counts for each detection scale (from neck).
        num_bins:    Number of discrete distance bins per edge (default 16).
        strides:     Pixel stride for each scale — must match backbone downsampling.
    """

    def __init__(
        self,
        nc: int = 5,
        in_channels: tuple[int, ...] = (64, 128, 256),
        num_bins: int = 16,
        strides: tuple[int, ...] = (8, 16, 32),
    ) -> None:
        super().__init__()
        assert len(in_channels) == len(strides)
        self.nc       = nc
        self.num_bins = num_bins
        self.no       = nc + num_bins * 4     # outputs per anchor cell
        self.nl       = len(in_channels)
        self.strides  = list(strides)

        c2 = max(16, in_channels[0] // 4, num_bins * 4)  # box branch hidden dim
        c3 = max(nc, in_channels[0])                       # cls branch hidden dim

        # ── Box branch (per scale) ───────────────────────────────────────────
        # Two 3×3 convolutions extract spatial context; final 1×1 produces
        # 4 * num_bins raw values representing one distribution per box edge.
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c2, 3),
                Conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * num_bins, 1),
            )
            for ch in in_channels
        )

        # ── Class branch (per scale) ─────────────────────────────────────────
        # Two 3×3 convolutions; final 1×1 produces nc class logits.
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, nc, 1),
            )
            for ch in in_channels
        )

        self.dfl  = DistanceProjection(num_bins)
        # Localisation-quality predictor (shared across scales; tiny).
        self.dgqp = DistributionQuality(num_bins)
        self._anchor_cache: dict[tuple, torch.Tensor] = {}

        self._init_biases()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_biases(self) -> None:
        # Class branch: initialise biases so initial predicted probability ≈ 0.01,
        # preventing the network from being overwhelmed by negatives at the start.
        for seq in self.cv3:
            nn.init.constant_(seq[-1].bias, -math.log((1 - 0.01) / 0.01))
        for seq in self.cv2:
            nn.init.zeros_(seq[-1].bias)

    # ── Anchor-point grid construction ───────────────────────────────────────

    def _make_anchors(
        self,
        feats: list[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the grid of anchor-point centres for all detection scales.

        Returns:
            anchor_pts   (N_total, 2)  — (x, y) cell-centre in feature-map units.
            stride_tensor(N_total, 1)  — pixel stride for each anchor.
        """
        anchor_pts, stride_tensor = [], []
        for feat, stride in zip(feats, self.strides):
            H, W = feat.shape[2], feat.shape[3]
            key  = (H, W, stride)
            if key not in self._anchor_cache:
                sy = torch.arange(H, device=device, dtype=dtype)
                sx = torch.arange(W, device=device, dtype=dtype)
                grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij")
                pts = torch.stack(
                    [grid_x.flatten() + 0.5, grid_y.flatten() + 0.5], -1
                )                                            # (H*W, 2)
                self._anchor_cache[key] = pts
            anchor_pts.append(self._anchor_cache[key])
            stride_tensor.append(
                torch.full((H * W, 1), stride, dtype=dtype, device=device)
            )
        return (
            torch.cat(anchor_pts, 0),       # (N_total, 2)
            torch.cat(stride_tensor, 0),    # (N_total, 1)
        )

    # ── Box decoding ─────────────────────────────────────────────────────────

    @staticmethod
    def dist2bbox(
        dist:       torch.Tensor,    # (B, 4, N_total) edge distances in feat-map units
        anchor_pts: torch.Tensor,    # (N_total, 2)    cell centres in feat-map units
        stride:     torch.Tensor,    # (N_total, 1)    pixel stride per cell
    ) -> torch.Tensor:
        """Convert predicted distances to pixel-space xyxy boxes.

        Returns: (B, N_total, 4)
        """
        cx = anchor_pts[:, 0:1].T * stride.T   # (1, N) pixel x of cell centre
        cy = anchor_pts[:, 1:2].T * stride.T   # (1, N) pixel y of cell centre
        l, t, r, b = dist.unbind(1)             # each (B, N)
        x1 = cx - l * stride.T
        y1 = cy - t * stride.T
        x2 = cx + r * stride.T
        y2 = cy + b * stride.T
        return torch.stack([x1, y1, x2, y2], -1)   # (B, N, 4)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self, feats: tuple[torch.Tensor, ...]
    ) -> list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """Run detection head.

        Training : returns list of raw per-scale tensors (B, 4*num_bins+nc, H, W).
        Inference: returns (boxes (B, N, 4) xyxy pixels, scores (B, N, nc) sigmoid).
        """
        raw: list[torch.Tensor] = []
        for i, feat in enumerate(feats):
            box_logits = self.cv2[i](feat)          # (B, 4*num_bins, H, W)
            cls_logits = self.cv3[i](feat)          # (B, nc, H, W)
            raw.append(torch.cat([box_logits, cls_logits], 1))

        if self.training:
            return raw

        # ── Inference decoding ───────────────────────────────────────────────
        device = feats[0].device
        dtype  = feats[0].dtype

        anchor_pts, strides = self._make_anchors(list(feats), dtype, device)

        box_preds, cls_preds, q_preds = [], [], []
        for t in raw:
            B, _, H, W = t.shape
            t_flat = t.view(B, self.no, -1)                 # (B, no, H*W)
            box_preds.append(t_flat[:, :4 * self.num_bins])
            cls_preds.append(t_flat[:, 4 * self.num_bins:])
            # Localisation-quality scalar from the box-distribution shape.
            q = self.dgqp(t[:, :4 * self.num_bins])         # (B, 1, H, W)
            q_preds.append(q.view(B, 1, -1))                # (B, 1, H*W)

        box_raw = torch.cat(box_preds, 2)   # (B, 4*num_bins, N_total)
        cls_raw = torch.cat(cls_preds, 2)   # (B, nc, N_total)
        q_raw   = torch.cat(q_preds, 2)     # (B, 1, N_total)

        dist  = self.dfl(box_raw)                           # (B, 4, N_total)
        boxes = self.dist2bbox(dist, anchor_pts, strides)   # (B, N_total, 4)
        # Joint score: classification confidence x localisation quality.
        scores = (cls_raw.permute(0, 2, 1).sigmoid()
                  * q_raw.permute(0, 2, 1))                 # (B, N_total, nc)

        return boxes, scores
