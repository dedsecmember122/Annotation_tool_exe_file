"""
DETC_Neck — bidirectional multi-scale feature fusion.

Fuses the three backbone outputs (P3, P4, P5) through two passes:

  Top-down pass (high-level context flows toward small-object scales):
    P5 ──upsample──> merged with P4  ──DETC_Block──> td_p4
    td_p4 ──upsample──> merged with P3 ──DETC_Block──> N3  (/8)

  Bottom-up pass (fine spatial detail flows toward large-object scales):
    N3 ──Conv(stride=2)──> merged with td_p4 ──DETC_Block──> N4  (/16)
    N4 ──Conv(stride=2)──> merged with P5    ──DETC_Block──> N5  (/32)

Outputs (N3, N4, N5) carry context from all three scales and are passed
directly to the detection head.

Every sub-module is a named attribute so forward hooks see each layer.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .blocks import Conv, DETC_Block


class DETC_Neck(nn.Module):
    """Bidirectional feature-fusion neck.

    Args:
        in_channels:    (P3_ch, P4_ch, P5_ch) channel counts from backbone.
        width_multiple: Kept for API consistency; channel counts are
                        already pre-scaled by the backbone.
        depth_multiple: Controls how many blocks each fusion step uses.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        width_multiple: float = 0.25,
        depth_multiple: float = 0.50,
    ) -> None:
        super().__init__()
        p3_ch, p4_ch, p5_ch = in_channels

        def d(n: int) -> int:
            return max(1, round(n * depth_multiple))

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # ── Top-down ────────────────────────────────────────────────────────
        # Merge upsampled P5 with P4 → td_p4
        self.td_block_p4 = DETC_Block(
            p5_ch + p4_ch, p4_ch,
            n=d(2), deep_sub=False, e=0.5, shortcut=False,
        )
        # Merge upsampled td_p4 with P3 → N3
        self.td_block_p3 = DETC_Block(
            p4_ch + p3_ch, p3_ch,
            n=d(2), deep_sub=False, e=0.5, shortcut=False,
        )

        # ── Bottom-up ───────────────────────────────────────────────────────
        # Downsample N3, merge with td_p4 → N4
        self.bu_down_n3  = Conv(p3_ch, p3_ch, 3, 2)
        self.bu_block_n4 = DETC_Block(
            p3_ch + p4_ch, p4_ch,
            n=d(2), deep_sub=False, e=0.5, shortcut=False,
        )
        # Downsample N4, merge with P5 → N5
        self.bu_down_n4  = Conv(p4_ch, p4_ch, 3, 2)
        self.bu_block_n5 = DETC_Block(
            p4_ch + p5_ch, p5_ch,
            n=d(2), deep_sub=True, e=0.5, shortcut=False,
        )

        self.out_channels: tuple[int, int, int] = (p3_ch, p4_ch, p5_ch)

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (N3, N4, N5) fused feature maps."""

        # Top-down
        td_p4 = self.td_block_p4(torch.cat([self.upsample(p5), p4], 1))
        n3    = self.td_block_p3(torch.cat([self.upsample(td_p4), p3], 1))

        # Bottom-up
        n4 = self.bu_block_n4(torch.cat([self.bu_down_n3(n3), td_p4], 1))
        n5 = self.bu_block_n5(torch.cat([self.bu_down_n4(n4), p5], 1))

        return n3, n4, n5
