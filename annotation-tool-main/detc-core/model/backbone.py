"""
DETC_Backbone

Produces three multi-scale feature maps from a single input image:
  P3 (/8)   — small-object feature resolution
  P4 (/16)  — medium-object feature resolution
  P5 (/32)  — large-object feature resolution (with global attention)

Stage layout:
  stem       : Conv → Conv → DETC_Block
  P3 stage   : Conv(stride=2) → DETC_Block(light)
  P4 stage   : Conv(stride=2) → DETC_Block(deep)
  P5 stage   : Conv(stride=2) → DETC_Block(deep) → DETC_PyramidPool → DETC_AttnModule

Every sub-module is a named nn.Module attribute so forward hooks registered
by FeatureVisualizer can capture intermediate activations by name.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .blocks import Conv, DETC_Block, DETC_PyramidPool, DETC_AttnModule, make_divisible


_BASE_CHANNELS = [64, 128, 256, 512, 1024]


class _Stage(nn.Module):
    """Ordered sequence of named layers.

    Exposes each layer as a named attribute (rather than inside a
    nn.Sequential) so that FeatureVisualizer hooks can identify each
    layer individually by its full dotted path.
    """

    def __init__(self, layers: list[tuple[str, nn.Module]]) -> None:
        super().__init__()
        for name, module in layers:
            setattr(self, name, module)
        self._order = [n for n, _ in layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for name in self._order:
            x = getattr(self, name)(x)
        return x


class DETC_Backbone(nn.Module):
    """Multi-scale feature extractor with configurable width and depth.

    Args:
        width_multiple: Scales every channel count (0.25 = lightest, 1.50 = heaviest).
        depth_multiple: Scales how many blocks appear in each stage (0.50 – 1.00).
    """

    def __init__(self, width_multiple: float = 0.25,
                 depth_multiple: float = 0.50) -> None:
        super().__init__()
        ch = [make_divisible(c * width_multiple) for c in _BASE_CHANNELS]

        def d(n: int) -> int:
            return max(1, round(n * depth_multiple))

        # ── Stem (/4 of input resolution) ──────────────────────────────────
        # Two strided convolutions reduce spatial size 4×; DETC_Block then
        # extracts initial features without further downsampling.
        self.stem = _Stage([
            ("conv0", Conv(3,      ch[0], 3, 2)),                        # /2
            ("conv1", Conv(ch[0],  ch[1], 3, 2)),                        # /4
            ("block", DETC_Block(ch[1], ch[2], d(2), deep_sub=False, e=0.25)),
        ])

        # ── P3 stage (/8) — small-object features ──────────────────────────
        self.stage_p3 = _Stage([
            ("down",  Conv(ch[2],  ch[2], 3, 2)),                        # /8
            ("block", DETC_Block(ch[2], ch[3], d(2), deep_sub=False, e=0.25, bottleneck_e=0.5)),
        ])

        # ── P4 stage (/16) — medium-object features ────────────────────────
        self.stage_p4 = _Stage([
            ("down",  Conv(ch[3],  ch[3], 3, 2)),                        # /16
            ("block", DETC_Block(ch[3], ch[3], d(2), deep_sub=True, bottleneck_e=1.0)),
        ])

        # ── P5 stage (/32) — large-object features with global attention ───
        self.stage_p5 = _Stage([
            ("down",    Conv(ch[3],  ch[4], 3, 2)),                      # /32
            ("block",   DETC_Block(ch[4], ch[4], d(2), deep_sub=True, bottleneck_e=1.0)),
            ("pyramid", DETC_PyramidPool(ch[4], ch[4], 5)),
            ("attn",    DETC_AttnModule(ch[4], ch[4], d(2))),
        ])

        # Channel counts passed to the neck
        self.out_channels: tuple[int, int, int] = (ch[3], ch[3], ch[4])

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (P3, P4, P5) feature maps."""
        x  = self.stem(x)
        p3 = self.stage_p3(x)
        p4 = self.stage_p4(p3)
        p5 = self.stage_p5(p4)
        return p3, p4, p5
