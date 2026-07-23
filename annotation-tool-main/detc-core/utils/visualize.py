"""
FeatureVisualizer — forward-hook-based intermediate layer image output.

Registers a hook on every named nn.Module in the model. After a forward pass,
each hook's captured output tensor is converted to a jet-colourised heatmap
(mean absolute activation across channels) and saved to disk.

Usage during inference:
    vis = FeatureVisualizer(model, out_dir="feature_maps/step_0")
    with vis:
        boxes, scores = model(img_tensor)
    # PNG files written to out_dir/

Usage from training loop (every N steps):
    vis = FeatureVisualizer(model, out_dir=f"feature_maps/epoch{e}_step{s}")
    with torch.no_grad():
        with vis:
            _ = model(sample_img)
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


# Modules whose outputs are too small (parameters) or not spatial (1D)
# and would produce trivially uninteresting visualisations.
_SKIP_TYPES = (
    nn.BatchNorm2d,
    nn.SiLU,
    nn.Identity,
    nn.Upsample,
    nn.MaxPool2d,
    nn.ModuleList,
    nn.Sequential,
)


def _feat_to_heatmap(tensor: torch.Tensor, size: int = 256) -> np.ndarray:
    """Convert a (B, C, H, W) or (B, C) feature tensor to a jet heatmap.

    Steps:
      1. Use first batch item.
      2. Compute mean of absolute activations across the channel dim → (H, W).
      3. Normalise to [0, 255].
      4. Resize to `size × size`.
      5. Apply OpenCV jet colourmap.

    Returns uint8 BGR image (H, W, 3).
    """
    t = tensor.detach().cpu().float()

    if t.ndim == 4:          # (B, C, H, W)
        heatmap = t[0].abs().mean(0).numpy()          # (H, W)
    elif t.ndim == 3:        # (B, C, N)  — flattened spatial
        side = int(t.shape[-1] ** 0.5)
        if side * side == t.shape[-1]:
            heatmap = t[0].abs().mean(0).view(side, side).numpy()
        else:
            heatmap = t[0].abs().mean(0).numpy()      # 1-D
            heatmap = heatmap[None, :]                 # (1, N)
    elif t.ndim == 2:        # (B, C)
        heatmap = t[0].abs().numpy()[None, :]          # (1, C)
    else:
        return np.zeros((size, size, 3), dtype=np.uint8)

    # Normalise
    mn, mx = heatmap.min(), heatmap.max()
    if mx - mn > 1e-6:
        heatmap = (heatmap - mn) / (mx - mn)
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap_u8 = (heatmap * 255).astype(np.uint8)

    # Resize to fixed output size
    heatmap_u8 = cv2.resize(
        heatmap_u8, (size, size), interpolation=cv2.INTER_NEAREST
    )

    # Apply jet colourmap
    coloured = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    return coloured


def _sanitise(name: str) -> str:
    """Make layer name safe for use as a filename."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


class FeatureVisualizer:
    """Context-manager that hooks every named layer and saves heatmaps.

    Args:
        model:       The nn.Module to hook.
        out_dir:     Directory where PNG images are written.
        img_size:    Side length of each saved heatmap (square).
        skip_types:  Tuple of module types to ignore.
        batch_idx:   Which batch item to visualise (default 0).
    """

    def __init__(
        self,
        model: nn.Module,
        out_dir: str | Path = "feature_maps",
        img_size: int = 256,
        skip_types: tuple = _SKIP_TYPES,
        batch_idx: int = 0,
    ) -> None:
        self.model      = model
        self.out_dir    = Path(out_dir)
        self.img_size   = img_size
        self.skip_types = skip_types
        self.batch_idx  = batch_idx
        self._handles:  list[torch.utils.hooks.RemovableHook] = []
        self._captured: dict[str, torch.Tensor] = {}

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "FeatureVisualizer":
        self._register()
        return self

    def __exit__(self, *_) -> None:
        self._remove()
        self._save_all()

    # ── Hook management ───────────────────────────────────────────────────────

    def _register(self) -> None:
        self._handles.clear()
        self._captured.clear()
        for name, module in self.model.named_modules():
            if not name:                          # skip root module
                continue
            if isinstance(module, self.skip_types):
                continue
            # Capture by closure — each name is unique in the module tree
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            # output can be a tensor or a tuple (take first tensor)
            if isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self._captured[name] = item
                        break
            elif isinstance(output, torch.Tensor):
                self._captured[name] = output
        return hook

    # ── Saving ────────────────────────────────────────────────────────────────

    def _save_all(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for name, tensor in sorted(self._captured.items()):
            if tensor.ndim < 2:
                continue
            img    = _feat_to_heatmap(tensor, self.img_size)
            fname  = self.out_dir / f"{_sanitise(name)}.png"
            cv2.imwrite(str(fname), img)
            saved += 1
        if saved:
            print(f"[FeatureVisualizer] Saved {saved} heatmaps → {self.out_dir}")

    # ── One-shot helper ───────────────────────────────────────────────────────

    @staticmethod
    def run_once(
        model: nn.Module,
        img_tensor: torch.Tensor,
        out_dir: str | Path = "feature_maps",
        img_size: int = 256,
    ) -> None:
        """Convenience: run model once with all layers visualised.

        Temporarily sets model to eval mode.
        """
        training = model.training
        model.eval()
        vis = FeatureVisualizer(model, out_dir=out_dir, img_size=img_size)
        with torch.no_grad(), vis:
            model(img_tensor)
        if training:
            model.train()


# ---------------------------------------------------------------------------
# Grid composer — combine all heatmaps into a single overview image
# ---------------------------------------------------------------------------

def make_grid_image(
    heatmap_dir: str | Path,
    cols: int = 6,
    cell_size: int = 128,
    title_height: int = 16,
) -> np.ndarray | None:
    """Read all PNG files in `heatmap_dir` and arrange them in a grid.

    Returns a BGR numpy image or None if the directory is empty.
    """
    paths = sorted(Path(heatmap_dir).glob("*.png"))
    if not paths:
        return None

    rows = (len(paths) + cols - 1) // cols
    h = rows * (cell_size + title_height)
    w = cols * cell_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, p in enumerate(paths):
        row, col = divmod(idx, cols)
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (cell_size, cell_size))
        y0 = row * (cell_size + title_height) + title_height
        x0 = col * cell_size
        canvas[y0:y0 + cell_size, x0:x0 + cell_size] = img

        # Layer name as small text above the cell
        label = p.stem[:18]
        cv2.putText(
            canvas, label,
            (x0 + 2, y0 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1,
        )

    return canvas
