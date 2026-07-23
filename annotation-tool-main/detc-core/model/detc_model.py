"""
DETCModel — full assembled detection model (backbone → neck → head).

The model is fully class-agnostic: the number of classes (nc) and their
names always come from the caller — in this project, from the annotation
tool's per-project class list. There is no built-in / default class list.

Model size variants (width_multiple, depth_multiple):
  n — (0.25, 0.50)   lightest,  ~2.4 M parameters
  s — (0.50, 0.50)              ~9.0 M parameters
  m — (1.00, 0.50)             ~20   M parameters
  l — (1.00, 1.00)             ~25   M parameters
  x — (1.50, 1.00)   heaviest, ~56   M parameters
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from .backbone  import DETC_Backbone
from .neck      import DETC_Neck
from .head      import DETC_Head


_SIZE_CONFIGS: dict[str, tuple[float, float]] = {
    "n": (0.25, 0.50),
    "s": (0.50, 0.50),
    "m": (1.00, 0.50),
    "l": (1.00, 1.00),
    "x": (1.50, 1.00),
}


class DETCModel(nn.Module):
    """End-to-end detector — single backbone/neck/head, configurable class count.

    Args:
        model_size: Capacity preset — one of 'n', 's', 'm', 'l', 'x'.
        num_bins:   Number of discrete bins used by the box-edge decoder.
        img_size:   Reference input resolution (informational, not enforced).
        nc:         Number of classes. Defaults to len(classes).
        classes:    Class name list — required (directly or via nc).
    """

    def __init__(
        self,
        model_size: str = "n",
        num_bins: int = 16,
        img_size: int = 640,
        nc: Optional[int] = None,
        classes: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        if model_size not in _SIZE_CONFIGS:
            raise ValueError(
                f"Unknown size '{model_size}'. Choose from {list(_SIZE_CONFIGS)}"
            )
        if nc is None and not classes:
            raise ValueError(
                "DETCModel requires 'classes' (a list of class names) and/or "
                "'nc'. Class names come from the annotation tool's project — "
                "there is no default class list."
            )
        width, depth = _SIZE_CONFIGS[model_size]

        self.classes    = list(classes) if classes else [str(i) for i in range(nc)]
        self.nc         = nc if nc is not None else len(self.classes)
        self.model_size = model_size
        self.img_size   = img_size

        self.backbone = DETC_Backbone(width, depth)
        self.neck     = DETC_Neck(self.backbone.out_channels, width, depth)
        self.head     = DETC_Head(
            nc=self.nc,
            in_channels=self.neck.out_channels,
            num_bins=num_bins,
        )

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """Run the full model.

        Training : returns list[(B, 4*num_bins+nc, H_i, W_i)] per scale.
        Inference: returns (boxes (B, N, 4) xyxy pixels, scores (B, N, nc)).
        """
        p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.neck(p3, p4, p5)
        return self.head((n3, n4, n5))

    # ── Utilities ─────────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        def _n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return "\n".join([
            f"DETCModel-{self.model_size}  nc={self.nc}  img={self.img_size}",
            f"  backbone : {_n(self.backbone):>12,}",
            f"  neck     : {_n(self.neck):>12,}",
            f"  head     : {_n(self.head):>12,}",
            f"  TOTAL    : {self.num_parameters():>12,}",
        ])
