"""
Custom model adapter — wires in the DETC detection model (backbone/neck/head
architecture living in the custom model directory, see CUSTOM_MODEL_DIR).

Implements:
  load()    — load best.pt / last.pt from the model dir, auto-detect nc
  predict() — letterbox → tensorize → NMS → return standard dicts
  train()   — prepare YOLO dataset, call train.py via subprocess, return best.pt path
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from backend.app.ml.base_detector import BaseDetector

# Default epochs for the 50-image bootstrap training (range: 50-70)
DEFAULT_EPOCHS = 60


class CustomModelAdapter(BaseDetector):

    def __init__(self) -> None:
        self._model = None
        self._model_dir: str = ""
        self._device = None
        self._class_names: list[str] = []
        self._nc: int = 0
        self._model_size: str = "n"
        self._img_size: int = 640

    # ────────────────────────────────────────────────────────────────────────
    # load()
    # ────────────────────────────────────────────────────────────────────────

    def load(
        self,
        model_dir: str,
        class_names: Optional[list[str]] = None,
        nc: Optional[int] = None,
        weights_path: Optional[str] = None,
        model_size: Optional[str] = None,
        img_size: Optional[int] = None,
    ) -> None:
        """
        Load the DETC model from *model_dir*.

        Priority for weights:
          1. weights_path (explicit override)
          2. model_dir/runs/train/best.pt
          3. model_dir/runs/train/last.pt
          4. Any *.pt found recursively in model_dir

        nc and class_names are auto-detected from the checkpoint when not
        explicitly provided.
        """
        import torch

        self._model_dir = str(model_dir)
        if self._model_dir not in sys.path:
            sys.path.insert(0, self._model_dir)

        # ── Locate weights ────────────────────────────────────────────────────
        if weights_path and Path(weights_path).is_file():
            ckpt_path = Path(weights_path)
        else:
            candidates = [
                Path(self._model_dir) / "runs" / "train" / "best.pt",
                Path(self._model_dir) / "runs" / "train" / "last.pt",
            ]
            ckpt_path = next((p for p in candidates if p.is_file()), None)
            if ckpt_path is None:
                pts = list(Path(self._model_dir).rglob("best.pt"))
                if not pts:
                    pts = list(Path(self._model_dir).rglob("*.pt"))
                if pts:
                    ckpt_path = pts[0]

        # ── Choose device ─────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        # ── Import the model package ──────────────────────────────────────────
        try:
            from model import DETCModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"Could not import DETCModel from {self._model_dir}. "
                f"Make sure the model directory is correctly set. Error: {exc}"
            )

        # ── Auto-detect nc from checkpoint (only when caller didn't specify) ───
        # If nc/class_names were passed in explicitly, they represent the
        # project's real, user-defined classes and must not be overridden by
        # whatever an unrelated checkpoint (e.g. a different project's warm
        # start, or the original base model) happens to contain.
        detected_nc = nc
        detected_classes = class_names
        detected_model_size = model_size
        detected_img_size = img_size

        if ckpt_path and ckpt_path.is_file() and (
            nc is None and not class_names or model_size is None or img_size is None
        ):
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            if nc is None and not class_names:
                # Detect nc from final classification head weight
                for key in ("head.cv3.0.2.weight", "head.cls.0.weight"):
                    if key in sd:
                        detected_nc = sd[key].shape[0]
                        break
            # Recover class list / model size / img size from saved args
            if isinstance(ckpt, dict) and "args" in ckpt:
                saved_args = ckpt["args"]
                if isinstance(saved_args, dict):
                    if nc is None and not class_names and saved_args.get("classes"):
                        detected_classes = [
                            c.strip() for c in saved_args["classes"].split(",") if c.strip()
                        ]
                    if model_size is None and saved_args.get("model"):
                        detected_model_size = str(saved_args["model"])
                    if img_size is None and saved_args.get("imgsz"):
                        detected_img_size = int(saved_args["imgsz"])

        self._nc = detected_nc or (len(detected_classes) if detected_classes else 0)
        self._class_names = detected_classes or []
        self._model_size = detected_model_size or "n"
        self._img_size = int(detected_img_size or 640)

        if not self._class_names:
            raise ValueError(
                f"Could not resolve class names for {model_dir}. "
                "Pass class_names explicitly or ensure the checkpoint has them saved."
            )
        if len(self._class_names) != self._nc:
            # A real mismatch here means this checkpoint's architecture is
            # incompatible with the requested classes (e.g. it belongs to a
            # different project). Fail loudly rather than padding the class
            # list with fabricated names like "class_2" to force a fit.
            raise ValueError(
                f"Checkpoint at {ckpt_path} has nc={self._nc} but "
                f"{len(self._class_names)} class names were resolved "
                f"({self._class_names}). This checkpoint is not compatible "
                "with the requested class list — use a fresh checkpoint or "
                "correct the class names."
            )

        # ── Build model ───────────────────────────────────────────────────────
        model = DETCModel(
            model_size=self._model_size,
            img_size=self._img_size,
            nc=self._nc,
            classes=self._class_names,
        ).to(device)

        if ckpt_path and ckpt_path.is_file():
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(sd, strict=False)

        model.eval()
        self._model = model

    # ────────────────────────────────────────────────────────────────────────
    # predict()
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _suppress_contained(dets_np, containment_thr: float = 0.5):
        """Drop lower-confidence boxes that are mostly covered by another,
        same-class box — regardless of standard IoU.

        Standard NMS compares intersection-over-union, which stays low when
        two boxes differ a lot in size even if the smaller one sits almost
        entirely inside the larger one (e.g. several tight boxes nested
        inside one wide box around the same cluster of foliage). That lets
        near-duplicate, nested boxes survive NMS and stack up on screen.
        This pass instead compares intersection-over-smaller-area, which
        catches containment regardless of the size mismatch.
        """
        import numpy as np

        n = len(dets_np)
        if n <= 1:
            return dets_np

        order = np.argsort(-dets_np[:, 4])
        dets_sorted = dets_np[order]
        keep = np.ones(n, dtype=bool)

        x1, y1, x2, y2 = (dets_sorted[:, i] for i in range(4))
        areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if not keep[j] or dets_sorted[i, 5] != dets_sorted[j, 5]:
                    continue
                ix1, iy1 = max(x1[i], x1[j]), max(y1[i], y1[j])
                ix2, iy2 = min(x2[i], x2[j]), min(y2[i], y2[j])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                smaller = min(areas[i], areas[j])
                if smaller > 0 and inter / smaller > containment_thr:
                    keep[j] = False

        return dets_sorted[keep]

    def predict(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[dict]:
        """
        Run inference on *image_path*.

        Returns a list of dicts:
          [{"class": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
        Coordinates are in original image pixel space.
        """
        import cv2
        import numpy as np
        import torch

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        device = self._device or torch.device("cpu")
        tensor, scale, px, py, orig_h, orig_w = self._preprocess(
            img_bgr, self._img_size, device
        )

        with torch.no_grad():
            boxes, scores = self._model(tensor)

        try:
            from utils.metrics import nms_detections  # type: ignore
        except ImportError:
            sys.path.insert(0, self._model_dir)
            from utils.metrics import nms_detections  # type: ignore

        dets = nms_detections(boxes, scores, conf=conf, iou_thr=iou)[0]
        if len(dets) == 0:
            return []

        # Rescale to original image coordinates
        dets_np = dets.numpy() if hasattr(dets, "numpy") else dets
        dets_np = self._suppress_contained(dets_np)
        dets_np[:, [0, 2]] = (dets_np[:, [0, 2]] - px) / scale
        dets_np[:, [1, 3]] = (dets_np[:, [1, 3]] - py) / scale
        dets_np[:, [0, 2]] = dets_np[:, [0, 2]].clip(0, orig_w)
        dets_np[:, [1, 3]] = dets_np[:, [1, 3]].clip(0, orig_h)

        results = []
        for row in dets_np:
            cls_id = int(row[5])
            results.append({
                "class": self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id),
                "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                "confidence": float(row[4]),
            })
        return results

    # ────────────────────────────────────────────────────────────────────────
    # train()
    # ────────────────────────────────────────────────────────────────────────

    def train(
        self,
        annotated_images: list[dict],
        output_dir: str,
        class_names: Optional[list[str]] = None,
        epochs: int = DEFAULT_EPOCHS,
        train_split: float = 0.8,
        model_size: str = "n",
        img_size: int = 640,
        log_callback=None,
        on_process_start=None,
    ) -> str:
        """
        Prepare a YOLO-format dataset from *annotated_images* and launch
        train.py as a subprocess.

        annotated_images format:
          [{"image_path": str, "annotations": [{"class": str, "bbox": [x1,y1,x2,y2]}, ...]}, ...]

        train_split is the fraction of images used for training (the rest go
        to validation), clamped to [0.5, 0.95].

        Returns:
          Path to the best.pt checkpoint.
        """
        import random
        import shutil

        import cv2

        if not annotated_images:
            raise ValueError("No annotated images provided for training.")

        effective_classes = class_names or self._class_names
        if not effective_classes:
            raise ValueError(
                "No class names provided for training. Classes must be "
                "created and used in the annotation tool before training."
            )
        nc = len(effective_classes)
        cls_to_id = {name: i for i, name in enumerate(effective_classes)}

        # ── Build YOLO dataset in a temp directory ─────────────────────────────
        dataset_dir = Path(tempfile.mkdtemp(prefix="insiso_train_"))
        for split in ("train", "val"):
            (dataset_dir / "images" / split).mkdir(parents=True)
            (dataset_dir / "labels" / split).mkdir(parents=True)

        # User-configurable train/val split (default 80/20)
        train_split = max(0.5, min(0.95, float(train_split)))
        items = [img for img in annotated_images if img.get("annotations")]
        random.shuffle(items)
        split_idx = max(1, int(len(items) * train_split))
        splits = {"train": items[:split_idx], "val": items[split_idx:]}
        if not splits["val"]:
            splits["val"] = items[:1]  # at least 1 val image

        for split, split_items in splits.items():
            img_dir = dataset_dir / "images" / split
            lbl_dir = dataset_dir / "labels" / split
            for entry in split_items:
                src = Path(entry["image_path"])
                if not src.is_file():
                    continue
                dst_img = img_dir / src.name
                shutil.copy2(src, dst_img)

                # Read dimensions for normalisation
                img = cv2.imread(str(src))
                if img is None:
                    continue
                ih, iw = img.shape[:2]

                # Write YOLO label file
                lines = []
                for ann in entry.get("annotations", []):
                    cls_name = ann.get("class", "")
                    cls_id = cls_to_id.get(cls_name, 0)
                    bbox = ann.get("bbox", [])
                    if len(bbox) < 4:
                        continue
                    x1, y1, x2, y2 = bbox
                    cx = ((x1 + x2) / 2) / iw
                    cy = ((y1 + y2) / 2) / ih
                    bw = (x2 - x1) / iw
                    bh = (y2 - y1) / ih
                    # Clamp to [0,1]
                    cx, cy, bw, bh = (max(0.0, min(1.0, v)) for v in (cx, cy, bw, bh))
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                lbl_file = lbl_dir / (src.stem + ".txt")
                lbl_file.write_text("\n".join(lines))

        # ── Run train.py as subprocess ─────────────────────────────────────────
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if model_size not in ("n", "s", "m", "l", "x"):
            model_size = "n"
        img_size = int(img_size)
        img_size = max(320, min(1280, img_size))
        img_size -= img_size % 32  # network stride requires a multiple of 32

        # self._model_dir (settings.CUSTOM_MODEL_DIR) must exist as a real
        # directory before we hand it to subprocess.Popen(cwd=...) below —
        # a missing/typo'd path there fails as a cryptic
        # "[WinError 267] The directory name is invalid" with no indication
        # of what's actually wrong. Fail with an actionable message instead.
        if not Path(self._model_dir).is_dir():
            raise RuntimeError(
                f"Training code directory not found: {self._model_dir}. "
                "This usually means the application was not installed "
                "correctly (a required folder is missing from the install). "
                "Try reinstalling the application; if the problem persists, "
                "contact support."
            )

        train_args = [
            "--data", str(dataset_dir),
            "--nc", str(nc),
            "--classes", ",".join(effective_classes),
            "--model", model_size,
            "--epochs", str(epochs),
            "--batch", "8",
            "--imgsz", str(img_size),
            "--save-dir", str(output_dir),
            "--no-amp",  # safe fallback; AMP enabled automatically on CUDA by train.py
        ]

        if getattr(sys, "frozen", False):
            # In a packaged build, sys.executable is this app's own exe —
            # there is no separate python.exe bundled alongside it to hand
            # train.py to. Re-invoke this same exe with a hidden flag that
            # frontend/main.py recognizes and dispatches straight to
            # train.py's training loop instead of launching the GUI again.
            cmd = [sys.executable, "--train-worker", *train_args]
        else:
            train_script = Path(self._model_dir) / "train.py"
            cmd = [sys.executable, str(train_script), *train_args]

        raw_tail: list[str] = []  # last N raw lines, for diagnostics on failure

        try:
            import time
            process = subprocess.Popen(
                cmd,
                cwd=self._model_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if on_process_start:
                on_process_start(process)

            for line in iter(process.stdout.readline, ''):
                if not line:
                    continue
                raw_tail.append(line)
                del raw_tail[:-40]
                if log_callback:
                    log_callback(line)

            process.wait()
            returncode = process.returncode
        finally:
            # Clean up tmp dataset
            shutil.rmtree(dataset_dir, ignore_errors=True)

        # interrupt.pt only exists when a graceful cancel (SIGINT) was sent —
        # train.py writes it for whatever epoch it stopped at, which is more
        # recent than a best/last checkpoint from an earlier periodic save,
        # so it takes priority when present.
        interrupt_pt = output_dir / "interrupt.pt"
        best_pt = output_dir / "best.pt"
        last_pt = output_dir / "last.pt"
        if interrupt_pt.is_file():
            return str(interrupt_pt)
        if best_pt.is_file():
            return str(best_pt)
        if last_pt.is_file():
            return str(last_pt)
        detail = "".join(raw_tail[-20:]).strip()
        raise RuntimeError(
            f"Training finished but no checkpoint found in {output_dir}. "
            f"Return code: {returncode}"
            + (f"\nLast output:\n{detail}" if detail else "")
        )

    # ────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(img_bgr, size: int, device):
        """Letterbox → tensor. Returns (tensor, scale, px, py, orig_h, orig_w)."""
        import numpy as np
        import torch

        orig_h, orig_w = img_bgr.shape[:2]
        scale = min(size / orig_h, size / orig_w)
        nh, nw = int(round(orig_h * scale)), int(round(orig_w * scale))

        import cv2
        img_r = cv2.resize(img_bgr, (nw, nh))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        px = (size - nw) // 2
        py = (size - nh) // 2
        canvas[py:py + nh, px:px + nw] = img_r

        t = torch.from_numpy(
            canvas[:, :, ::-1].copy()
        ).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(device), scale, px, py, orig_h, orig_w
