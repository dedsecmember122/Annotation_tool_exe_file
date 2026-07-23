"""
DetectionDataset — dataset loader for DETC model training.

Expected directory layout:
  <root>/
    images/
      train/  *.jpg / *.png / ...
      val/    ...
    labels/
      train/  *.txt   (one file per image, same stem)
      val/    ...

Label format (one row per object):  class_id  cx  cy  w  h   (all normalised to [0,1])

Class ids map to the user-defined class list supplied at training time —
there is no fixed class list in this package.

Augmentations (training only):
  • Horizontal flip  (p=0.5)
  • HSV jitter       (hue ±0.015, sat ±0.7, val ±0.4)
  • Random scale     (0.5 – 1.5×)
  • Mosaic 4-image   (p=0.5)
"""

from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_label(txt_path: Path) -> np.ndarray:
    """Load a label file → (N, 5) float32 array [cls, cx, cy, w, h]."""
    if not txt_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            rows.append([float(p) for p in parts])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def _augment_hsv(img: np.ndarray,
                 h_gain: float = 0.015,
                 s_gain: float = 0.7,
                 v_gain: float = 0.4) -> np.ndarray:
    """Random HSV colour jitter in-place."""
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv.astype(np.float32))
    h = np.clip(h * r[0], 0, 179).astype(np.uint8)
    s = np.clip(s * r[1], 0, 255).astype(np.uint8)
    v = np.clip(v * r[2], 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Letterbox-resize image to (size, size), return (img, scale, pad_x, pad_y)."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img
    return canvas, scale, pad_x, pad_y


def _adjust_labels(
    labels: np.ndarray,
    scale: float, pad_x: int, pad_y: int,
    img_w: int, img_h: int,
    canvas_size: int,
) -> np.ndarray:
    """Remap normalised labels after letterbox transform."""
    if labels.shape[0] == 0:
        return labels
    cls = labels[:, 0:1]
    cx  = labels[:, 1] * img_w * scale + pad_x
    cy  = labels[:, 2] * img_h * scale + pad_y
    bw  = labels[:, 3] * img_w * scale
    bh  = labels[:, 4] * img_h * scale
    # Re-normalise to canvas_size
    s = canvas_size
    return np.stack([
        cls[:, 0],
        cx / s, cy / s, bw / s, bh / s,
    ], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Mosaic augmentation
# ---------------------------------------------------------------------------

def _mosaic4(
    indices: list[int],
    imgs:    list[np.ndarray],
    labels:  list[np.ndarray],
    size:    int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine 4 images into a mosaic."""
    canvas = np.full((size * 2, size * 2, 3), 114, dtype=np.uint8)
    cut_x  = random.randint(size // 2, 3 * size // 2)
    cut_y  = random.randint(size // 2, 3 * size // 2)
    all_labels = []

    positions = [
        (0,         0,         cut_x,      cut_y),       # top-left
        (cut_x,     0,         size * 2,   cut_y),       # top-right
        (0,         cut_y,     cut_x,      size * 2),    # bottom-left
        (cut_x,     cut_y,     size * 2,   size * 2),    # bottom-right
    ]

    for i, (x1, y1, x2, y2) in enumerate(positions):
        img = imgs[indices[i]]
        lbl = labels[indices[i]].copy()
        h, w = img.shape[:2]
        rw, rh = x2 - x1, y2 - y1
        img_r = cv2.resize(img, (rw, rh))
        canvas[y1:y2, x1:x2] = img_r

        if lbl.shape[0]:
            lbl[:, 1] = (lbl[:, 1] * w * rw / w + x1) / (size * 2)
            lbl[:, 2] = (lbl[:, 2] * h * rh / h + y1) / (size * 2)
            lbl[:, 3] = lbl[:, 3] * rw / (size * 2)
            lbl[:, 4] = lbl[:, 4] * rh / (size * 2)
            all_labels.append(lbl)

    canvas = cv2.resize(canvas, (size, size))
    merged = np.concatenate(all_labels, 0) if all_labels else np.zeros((0, 5), np.float32)
    # Clip boxes
    if merged.shape[0]:
        merged[:, 1:] = np.clip(merged[:, 1:], 0, 1)
    return canvas, merged


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DetectionDataset(Dataset):
    """DETC detection dataset.

    Args:
        root:    Dataset root (contains images/ and labels/).
        split:   'train' or 'val'.
        img_size: Network input size (square).
        augment: Apply training augmentations.
        mosaic:  Mosaic probability (only when augment=True).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 640,
        augment: bool = True,
        mosaic: float = 0.5,
    ) -> None:
        self.root     = Path(root)
        self.split    = split
        self.img_size = img_size
        self.augment  = augment and split == "train"
        self.mosaic   = mosaic

        img_dir = self.root / "images" / split
        lbl_dir = self.root / "labels" / split

        self.img_paths: list[Path] = sorted(
            p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
        )
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in {img_dir}")

        # Pre-load labels (fast, small)
        self.labels: list[np.ndarray] = []
        for p in self.img_paths:
            txt = lbl_dir / (p.stem + ".txt")
            self.labels.append(_load_label(txt))

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_img(self, idx: int) -> np.ndarray:
        img = cv2.imread(str(self.img_paths[idx]))
        if img is None:
            raise IOError(f"Cannot read image: {self.img_paths[idx]}")
        return img

    def __getitem__(self, idx: int) -> dict:
        # ── Mosaic ────────────────────────────────────────────────────────────
        if self.augment and random.random() < self.mosaic and len(self) >= 4:
            indices = [idx] + random.choices(range(len(self)), k=3)
            imgs   = [self._load_img(i) for i in indices]
            lbls   = [self.labels[i].copy() for i in indices]
            img, label = _mosaic4(list(range(4)), imgs, lbls, self.img_size)
        else:
            img   = self._load_img(idx)
            label = self.labels[idx].copy()
            img, scale, px, py = _letterbox(img, self.img_size)
            h, w = self._load_img(idx).shape[:2]   # original dims
            label = _adjust_labels(label, scale, px, py, w, h, self.img_size)

        # ── Augmentations ─────────────────────────────────────────────────────
        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
                if label.shape[0]:
                    label[:, 1] = 1.0 - label[:, 1]

            # HSV jitter
            img = _augment_hsv(img)

        # ── To tensor ─────────────────────────────────────────────────────────
        img_t = torch.from_numpy(
            img[:, :, ::-1].copy()             # BGR → RGB
        ).permute(2, 0, 1).float() / 255.0    # (3, H, W)  [0,1]

        return {"img": img_t, "labels": label, "path": str(self.img_paths[idx])}


# ---------------------------------------------------------------------------
# Collate function — handles variable-length label arrays
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    imgs   = torch.stack([b["img"] for b in batch])
    paths  = [b["path"] for b in batch]
    # Labels: concatenate with batch index prepended
    label_list = []
    for bi, b in enumerate(batch):
        lbl = torch.from_numpy(b["labels"]).float()   # (N, 5)
        if lbl.shape[0]:
            bi_col = torch.full((lbl.shape[0], 1), bi)
            label_list.append(torch.cat([bi_col, lbl], 1))  # (N, 6)
    labels = torch.cat(label_list, 0) if label_list else torch.zeros((0, 6))
    return {"imgs": imgs, "labels": labels, "paths": paths}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_loader(
    root: str | Path,
    split: str = "train",
    img_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 4,
    augment: bool = True,
    mosaic: float = 0.5,
) -> DataLoader:
    ds = DetectionDataset(root, split, img_size, augment, mosaic)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )
