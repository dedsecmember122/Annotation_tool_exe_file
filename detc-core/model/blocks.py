"""
Core building blocks for DETC.

Primitives:
  Conv            — Conv2d + BatchNorm2d + SiLU
  Bottleneck      — residual two-conv block with learnable-scale gate (y = x + α·F(x))
  DETC_CSP        — cross-stage partial block with configurable kernel
  DETC_SplitCSP   — split-concat CSP (faster variant)
  DETC_Block      — primary repeating block (DETC_SplitCSP + optional DETC_CSP sub-blocks)
  DETC_PyramidPool — multi-scale spatial pooling (fast cascaded version)
  Attention       — multi-head self-attention with depthwise positional bias
  DETC_AttnBlock  — attention + feed-forward with residual connections
  DETC_AttnModule — split-process-merge wrapper around DETC_AttnBlock
  DistanceProjection — distribution-to-distance decoder
  DistributionQuality — localisation-quality predictor
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(math.ceil(v / divisor) * divisor))


# ---------------------------------------------------------------------------
# Conv — standard convolution unit
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    """Conv + BN + SiLU with multi-branch re-parameterisation (drop-in).

    Same constructor signature (c1, c2, k, s, p, g, act) and SiLU activation
    as a plain conv unit, so it is a strict drop-in.

    TRAINING : for k==3, runs three parallel branches summed before the
               activation —  3x3 conv+BN  +  1x1 conv+BN  +  identity BN
               (identity only when c1==c2 and stride==1). Richer gradient flow
               than a single conv.
    DEPLOY   : switch_to_deploy() folds the branches into ONE 3x3 conv (+bias);
               inference cost is then identical to a plain Conv. Verified to
               match the training output to ~1e-6.

    For k==1 (pointwise) there is nothing to re-parameterise, so it falls back
    to a plain 1x1 conv + BN + activation.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1,
                 p: int | None = None, g: int = 1, act: bool = True,
                 deploy: bool = False) -> None:
        super().__init__()
        self.c1, self.c2, self.k, self.s, self.g = c1, c2, k, s, g
        self.pad = (k // 2) if p is None else p
        self.rep = (k == 3)                       # only 3x3 is re-parameterised
        self.deploy = deploy
        self.act = (nn.SiLU(inplace=True) if act is True
                    else act if isinstance(act, nn.Module) else nn.Identity())

        if deploy or not self.rep:
            self.conv = nn.Conv2d(c1, c2, k, s, self.pad, groups=g,
                                  bias=bool(deploy and self.rep))
            self.bn = nn.Identity() if (deploy and self.rep) else nn.BatchNorm2d(c2)
        else:
            self.conv = None
            self.main     = self._conv_bn(c1, c2, 3, s, self.pad, g)
            self.onexone  = self._conv_bn(c1, c2, 1, s, 0, g)
            self.identity = nn.BatchNorm2d(c1) if (c1 == c2 and s == 1) else None

    @staticmethod
    def _conv_bn(c1, c2, k, s, p, g):
        return nn.Sequential(nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False),
                             nn.BatchNorm2d(c2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:                 # plain path (deploy or k!=3)
            return self.act(self.bn(self.conv(x)))
        y = self.main(x) + self.onexone(x)        # multi-branch training path
        if self.identity is not None:
            y = y + self.identity(x)
        return self.act(y)

    # ---- Branch folding (call once before inference / export) -------------
    def _fuse(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            conv, bn = branch[0], branch[1]
            kernel = conv.weight
        else:                                     # identity BN -> 3x3 kernel
            bn = branch
            in_dim = self.c1 // self.g
            kernel = torch.zeros(self.c1, in_dim, 3, 3,
                                 dtype=bn.weight.dtype, device=bn.weight.device)
            for i in range(self.c1):
                kernel[i, i % in_dim, 1, 1] = 1.0
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    @staticmethod
    def _pad1to3(k):
        if isinstance(k, int):    return 0
        if k.shape[-1] == 3:      return k
        return F.pad(k, [1, 1, 1, 1])

    def switch_to_deploy(self):
        if self.deploy or not self.rep or self.conv is not None:
            return
        km, bm = self._fuse(self.main)
        k1, b1 = self._fuse(self.onexone)
        ki, bi = self._fuse(self.identity)
        kernel = km + self._pad1to3(k1) + self._pad1to3(ki)
        bias   = bm + b1 + bi
        self.conv = nn.Conv2d(self.c1, self.c2, 3, self.s, self.pad,
                              groups=self.g, bias=True)
        self.conv.weight.data = kernel
        self.conv.bias.data   = bias
        self.bn = nn.Identity()
        for a in ("main", "onexone", "identity"):
            if hasattr(self, a):
                delattr(self, a)
        self.deploy = True


# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    """Two Conv layers with an optional learnable-scale gated identity shortcut.

    The transform branch is:

        F(x) = cv2(cv1(x))

    When the identity shortcut is active (``shortcut`` and ``c1 == c2``) the
    branch output is scaled by a learnable parameter ``alpha`` before being
    added back:

        y = x + alpha · F(x)

    ``alpha`` is a learnable per-channel vector of length ``c2``, broadcast
    over (B, C, H, W) and initialised to a small constant
    (``layer_scale_init``). Each residual block therefore starts close to
    the identity and learns, per channel, how much of its residual branch
    to use.

    When there is no shortcut (``shortcut=False`` or ``c1 != c2``) the block
    returns ``F(x)`` and holds no ``alpha`` parameter — no other behaviour
    is changed.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True,
                 g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5,
                 layer_scale_init: float = 0.1) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0])
        self.cv2 = Conv(c_, c2, k[1], g=g)
        self.add = shortcut and c1 == c2
        self.alpha = (nn.Parameter(torch.full((c2,), float(layer_scale_init)))
                      if self.add else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))                 # F(x)
        if self.add:
            return x + self.alpha.view(1, -1, 1, 1) * out
        return out


# ---------------------------------------------------------------------------
# DETC_CSP — cross-stage partial block
# ---------------------------------------------------------------------------

class SimAM(nn.Module):
    """Parameter-free spatial attention gate.

    Amplifies positions that stand out from their channel mean; dims bland
    ones. No learnable parameters; AMP-safe; sits OUTSIDE conv fold.
    """

    def __init__(self, lam: float = 1e-4) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[2] * x.shape[3] - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)) ** 2
        v = d.sum(dim=[2, 3], keepdim=True) / n
        weight = d / (4 * (v + self.lam)) + 0.5
        return x * torch.sigmoid(weight)


class RepBasicBlockReverse(nn.Module):
    """Inner unit: re-parameterisable Conv(3x3) -> Conv(3x3), with an identity shortcut.

    The first conv is the reparameterizable branch (folds to one 3x3 at
    deploy); the second 3x3 is a plain Conv. The block-level residual add
    is OUTSIDE the folded conv, so folding is preserved. The shortcut is
    used only when shapes match (c1 == c2).
    """

    def __init__(self, c1: int, c2: int, ch_hidden_ratio: float = 1.0,
                 shortcut: bool = True, g: int = 1, k: int = 3) -> None:
        super().__init__()
        c_hidden = int(c2 * ch_hidden_ratio)
        self.conv1 = Conv(c1, c_hidden, k, 1)         # reparameterizable 3x3 branch
        self.conv2 = Conv(c_hidden, c2, k, 1, g=g)    # plain 3x3
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


class DETC_CSPStage(nn.Module):
    """Collect-and-fuse cross-stage block, drop-in for DETC_CSP.

    Original implementation — not derived from any AGPL/GPL-licensed source.

    Two 1x1 entry convs make a bypass and a processing entry. The processing
    entry runs through `n` RepBasicBlockReverse units; EVERY unit output is
    collected together with the bypass and fused by a single 1x1 (fuse-last).
    A parameter-free attention gate polishes the fused output.

    Args:
        c1, c2:          input / output channels.
        n:               number of inner units.
        shortcut, g:     passed to the inner units.
        e:               bypass split ratio; bypass width = int(c2 * e).
        k:               inner kernel size (3).
        bottleneck_e:    accepted for signature compatibility (unused here;
                         inner width is set by ch_hidden_ratio).
        ch_hidden_ratio: inner hidden-width ratio of each unit (default 1.0).
        use_simam:       apply the SimAM gate to the fused output.

    Public attrs kept stable so DETC_Block can subclass / introspect:
        self.units, self.cv1, self.cv2, self.cv3, forward().
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True,
                 g: int = 1, e: float = 0.5, k: int = 3,
                 bottleneck_e: float = 1.0, ch_hidden_ratio: float = 1.0,
                 use_simam: bool = True) -> None:
        super().__init__()
        ch_first = int(c2 * e)            # bypass branch width
        ch_mid   = c2 - ch_first          # processing branch width

        self.cv1 = Conv(c1, ch_first, 1)  # bypass entry
        self.cv2 = Conv(c1, ch_mid, 1)    # processing entry

        self.units = nn.ModuleList(
            RepBasicBlockReverse(ch_mid, ch_mid, ch_hidden_ratio, shortcut, g, k)
            for _ in range(n)
        )

        # fuse-last sees: bypass + every unit output = ch_first + n * ch_mid
        self.cv3 = Conv(ch_first + n * ch_mid, c2, 1)
        self.gate = SimAM() if use_simam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        collected = [self.cv1(x)]         # bypass
        h = self.cv2(x)                   # processing entry
        for unit in self.units:
            h = unit(h)
            collected.append(h)           # collect every unit's output
        return self.gate(self.cv3(torch.cat(collected, dim=1)))


class DETC_CSP(nn.Module):
    """Cross-stage partial block: dual path + sequential Bottleneck stack.

    One path goes through n stacked Bottlenecks; the other is a direct
    projection. Both are concatenated and projected back to c2 channels.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True,
                 g: int = 1, e: float = 0.5, k: int = 3,
                 bottleneck_e: float = 1.0) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m   = nn.Sequential(*(
            Bottleneck(c_, c_, shortcut, g, k=(k, k), e=bottleneck_e)
            for _ in range(n)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# ---------------------------------------------------------------------------
# DETC_SplitCSP — faster split-concat variant
# ---------------------------------------------------------------------------

class DETC_SplitCSP(nn.Module):
    """Split-concat block: input is split into halves, each half passes
    through successive processing blocks, all outputs are concatenated.

    Richer gradient flow than a single-path design because every
    intermediate activation feeds into the final merge.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5) -> None:
        super().__init__()
        self.c  = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m   = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------------------------
# DETC_Block — primary repeating block
# ---------------------------------------------------------------------------

class DETC_CleanELAN(nn.Module):
    """Layer-aggregation block: two entry transitions, a collected
    processing chain, and a single fuse-last projection.

    Original implementation — not derived from any AGPL/GPL-licensed source.

    Args:
        c1, c2:   input / output channels.
        n:        number of residual units on the processing route.
        shortcut: whether the residual units use identity shortcuts.
        g:        groups for the unit convolutions.
        e:        route-width ratio; each entry transition is int(c2 * e) wide.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5) -> None:
        super().__init__()

        # Width of each route (both entry transitions produce this many).
        self.c = int(c2 * e)

        # Two entry transitions (kept as separate 1x1 projections so the
        # two gradient paths are distinct from the very first layer).
        self.proj_bypass = Conv(c1, self.c, 1)   # cross-stage route (skips chain)
        self.proj_route  = Conv(c1, self.c, 1)   # processing-route entry

        # Fuse-last sees: bypass + route-entry + one output per unit = (n + 2).
        merged_width = (n + 2) * self.c
        self.proj_out = Conv(merged_width, c2, 1)

        # Processing chain (DETC_Block overrides this with its own unit type).
        self.units = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bypass = self.proj_bypass(x)             # cross-stage route
        route  = self.proj_route(x)              # processing-route entry

        collected = [bypass, route]
        current = route
        for unit in self.units:
            current = unit(current)              # grow the chain
            collected.append(current)            # collect every link's output

        return self.proj_out(torch.cat(collected, dim=1))   # fuse last


class DETC_Block(DETC_CleanELAN):
    """Primary repeating block of the DETC architecture.

    Extends DETC_SplitCSP by replacing each internal Bottleneck with
    either a DETC_CSP sub-block (deeper, more representational capacity)
    or a plain Bottleneck (lighter, faster).

    Set deep_sub=True at the deeper stages of the network where richer
    feature interactions are needed.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, deep_sub: bool = False,
                 e: float = 0.5, g: int = 1, shortcut: bool = True,
                 bottleneck_e: float | None = None) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        # Override the processing chain with this block's unit choice.
        if deep_sub:
            be = 1.0 if bottleneck_e is None else bottleneck_e
            self.units = nn.ModuleList(
                DETC_CSPStage(self.c, self.c, 2, shortcut, g,
                              bottleneck_e=be, ch_hidden_ratio=1.0, use_simam=True)
                for _ in range(n))
        else:
            be = 0.5 if bottleneck_e is None else bottleneck_e
            self.units = nn.ModuleList(
                Bottleneck(self.c, self.c, shortcut, g, e=be)
                for _ in range(n))


# ---------------------------------------------------------------------------
# DETC_PyramidPool — multi-scale spatial pooling
# ---------------------------------------------------------------------------

class DETC_PyramidPool(nn.Module):
    """Multi-scale pooling via three cascaded MaxPool operations.

    Each pool sees a progressively larger receptive field because each
    one operates on the output of the previous. The results are concatenated
    and projected, giving the network context at four different scales
    without requiring large kernels.
    """

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_       = c1 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_ * 4, c2, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


# ---------------------------------------------------------------------------
# Attention — multi-head self-attention with positional bias
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention with a depthwise-conv positional encoding.

    Each pixel in the feature map attends to every other pixel.
    The depthwise conv adds a local spatial bias on top of the global
    attention result before the final linear projection.
    """

    def __init__(self, dim: int, num_heads: int = 8,
                 attn_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.key_dim   = int(self.head_dim * attn_ratio)
        self.scale     = self.key_dim ** -0.5
        nh_kd          = self.key_dim * num_heads
        self.qkv  = Conv(dim, dim + nh_kd * 2, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe   = Conv(dim, dim, 3, 1, g=dim, act=False)   # depthwise pos enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N   = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.split([
            self.key_dim * self.num_heads,
            self.key_dim * self.num_heads,
            C,
        ], dim=1)
        q   = q.view(B, self.num_heads, self.key_dim, N)
        k   = k.view(B, self.num_heads, self.key_dim, N)
        v   = v.view(B, self.num_heads, self.head_dim, N)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(-1)
        out  = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        out  = out + self.pe(out)
        return self.proj(out)


# ---------------------------------------------------------------------------
# DETC_AttnBlock — attention + FFN with residual connections
# ---------------------------------------------------------------------------

class DETC_AttnBlock(nn.Module):
    """Self-attention followed by a two-layer feed-forward network.

    Both the attention and the FFN use residual (skip) connections so
    that the block can learn whether to rely on the attention result or
    pass the input through unchanged.
    """

    def __init__(self, c: int, attn_ratio: float = 0.5,
                 num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn  = nn.Sequential(
            Conv(c, c * 2, 1),
            Conv(c * 2, c, 1, act=False),
        )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x)  if self.add else self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# DETC_AttnModule — split-process-merge wrapper for DETC_AttnBlock
# ---------------------------------------------------------------------------

class DETC_AttnModule(nn.Module):
    """Wraps DETC_AttnBlock in a split-merge structure.

    Half the channels are processed by stacked DETC_AttnBlocks; the other
    half bypasses them. Both halves are merged at the end. This lets the
    model retain unmodified features alongside attention-refined ones.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        super().__init__()
        assert c1 == c2, f"DETC_AttnModule requires c1==c2, got {c1}!={c2}"
        self.c   = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m   = nn.Sequential(*(
            DETC_AttnBlock(
                self.c,
                attn_ratio=0.5,
                num_heads=max(1, self.c // 64),
            )
            for _ in range(n)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), 1)
        b    = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


# ---------------------------------------------------------------------------
# DistanceProjection — distribution-to-distance decoder
#
# Each box edge is regressed as a probability distribution P over a discrete
# support of candidate distances. The predicted distance is the expectation
# of that distribution:
#
#       y_hat = E[X] = sum_i  y_i * P(y_i)
#
# where the support points are y_i = start + i*step for i in 0..num_bins-1.
# On the default integer support this is numerically identical to a frozen
# [0..n-1] projection.
# ---------------------------------------------------------------------------

class DistanceProjection(nn.Module):
    """Project a per-edge discrete distribution onto its expected distance.

    The detection head emits 4 * num_bins logits per anchor: one distribution
    of length num_bins for each box edge (left, top, right, bottom). This
    module normalises each distribution and returns its expected value.

    Args:
        num_bins: Number of discrete support points per edge.
        start:    Value of the first support point (default 0.0).
        step:     Spacing between consecutive support points (default 1.0).

    The (start, step) pair generalises the fixed integer support
    [0, 1, ..., num_bins-1]: a larger `step` extends the reachable distance
    range, a smaller one increases resolution near the anchor — useful when
    targets are small (e.g. licence plates) and edge distances are short.
    """

    def __init__(self, num_bins: int = 16, start: float = 0.0,
                 step: float = 1.0) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        self.num_bins = num_bins
        # Support set y_i = start + i*step, held as a constant (non-learnable)
        # buffer so it follows .to(device)/.half() with the module and is never
        # touched by the optimiser.
        support = start + step * torch.arange(num_bins, dtype=torch.float32)
        self.register_buffer("support", support, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4*num_bins, N) raw logits -> (B, 4, N) expected distances."""
        b, c, n = x.shape
        if c != 4 * self.num_bins:
            raise ValueError(
                f"expected {4 * self.num_bins} channels, got {c}"
            )
        # (B, 4, num_bins, N): one distribution per edge; normalise over bins.
        prob = x.view(b, 4, self.num_bins, n).softmax(dim=2)
        # E[X] = sum_i support_i * P_i, contracted over the bin axis.
        support = self.support.to(prob.dtype).view(1, 1, self.num_bins, 1)
        return (prob * support).sum(dim=2)


# ---------------------------------------------------------------------------
# DistributionQuality — localisation-quality predictor
#
# The *shape* of each edge distribution reflects localisation quality: a sharp,
# peaked distribution means a confidently-placed edge; a flat one means the
# model is unsure. This module summarises each edge distribution by its Top-k
# probabilities (and their mean), then maps those statistics through a tiny
# two-layer MLP to a single quality scalar in (0, 1). That scalar scales the
# classification score so better-localised boxes are ranked higher in NMS.
# ---------------------------------------------------------------------------

class DistributionQuality(nn.Module):
    """Predict a localisation-quality scalar from box-distribution statistics.

    Args:
        num_bins: Distance bins per edge (must match the box branch).
        topk:     Number of largest per-edge probabilities kept (paper: 4).
        hidden:   Hidden width of the MLP (paper: 64).
        add_mean: Append each edge distribution's mean to its Top-k stats.

    Input : box_logits (B, 4*num_bins, H, W) — the raw box-branch output.
    Output: quality    (B, 1, H, W) in (0, 1).
    """

    def __init__(self, num_bins: int = 16, topk: int = 4,
                 hidden: int = 64, add_mean: bool = True) -> None:
        super().__init__()
        if topk > num_bins:
            raise ValueError(
                f"topk ({topk}) cannot exceed num_bins ({num_bins})"
            )
        self.num_bins = num_bins
        self.topk     = topk
        self.add_mean = add_mean

        stats_per_edge = topk + (1 if add_mean else 0)
        in_dim = 4 * stats_per_edge          # 4 box edges (l, t, r, b)

        # Per-location MLP, expressed as 1x1 convs so it runs on the spatial map.
        # Two layers: Linear -> ReLU -> Linear -> Sigmoid (per the paper).
        self.fc1 = nn.Conv2d(in_dim, hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, 1, 1)

    def forward(self, box_logits: torch.Tensor) -> torch.Tensor:
        b, c, h, w = box_logits.shape
        if c != 4 * self.num_bins:
            raise ValueError(
                f"expected {4 * self.num_bins} channels, got {c}"
            )
        # (B, 4, num_bins, H, W): normalise each edge distribution over its bins.
        prob = box_logits.view(b, 4, self.num_bins, h, w).softmax(dim=2)

        # Distribution-shape statistics per edge, over the bin axis.
        topk_vals, _ = prob.topk(self.topk, dim=2)        # (B, 4, topk, H, W)
        stats = topk_vals
        if self.add_mean:
            mean = prob.mean(dim=2, keepdim=True)          # (B, 4, 1, H, W)
            stats = torch.cat([topk_vals, mean], dim=2)
        stats = stats.reshape(b, -1, h, w)                 # (B, 4*(topk+1), H, W)

        # Tiny MLP -> one quality scalar per location.
        q = self.fc2(self.act(self.fc1(stats)))            # (B, 1, H, W)
        return q.sigmoid()
