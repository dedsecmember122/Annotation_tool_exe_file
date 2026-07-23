#!/usr/bin/env python3
"""
DETC Training Script
=====================
Trains the DETC detection model on a user-defined class list.

Classes are always supplied via --classes (comma-separated) — there is no
default or fixed class list. Class names come from the annotation tool's
project data.

Dataset layout expected:
  <data-dir>/
    images/train/   *.jpg|*.png|...
    images/val/     ...
    labels/train/   *.txt  (one row per object: cls cx cy w h, normalised to [0,1])
    labels/val/     ...

Quick start:
  python train.py --data /path/to/dataset --classes moringa_plant --model n --epochs 100

Full example:
  uv run python train.py --data /path/to/dataset --classes moringa_plant --model m --epochs 200 --batch 8 --imgsz 640 --lr0 0.01 --device cuda --workers 8 --save-dir runs/train --vis-every 10
         # save feature-map images every N epochs
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time
from pathlib import Path

import signal

# ── Force UTF-8 stdout ──────────────────────────────────────────────────────
# Windows consoles default to cp1252, which crashes on characters like '→'
# in log messages below. This previously killed training mid-run (e.g. right
# after saving a "new best" checkpoint), silently truncating the run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

# ── Graceful interrupt flag ────────────────────────────────────────────────────
_INTERRUPTED = False

def _handle_sigint(_sig, _frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    print("\n[DETC] Interrupt received — will save checkpoint after this epoch …")

signal.signal(signal.SIGINT, _handle_sigint)

# ── Local imports ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model import DETCModel
from utils.dataset  import build_loader
from utils.loss     import DetectionLoss
from utils.metrics  import MeanAveragePrecision, nms_detections
from utils.visualize import FeatureVisualizer, make_grid_image


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("DETC trainer")
    p.add_argument("--data",       required=True,         help="Dataset root dir")
    p.add_argument("--model",      default="n",           help="Model size: n/s/m/l/x")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch",      type=int, default=16)
    p.add_argument("--imgsz",      type=int, default=640)
    p.add_argument("--lr0",        type=float, default=0.01,   help="Initial LR")
    p.add_argument("--lrf",        type=float, default=0.01,   help="Final LR fraction")
    p.add_argument("--momentum",   type=float, default=0.937)
    p.add_argument("--wd",         type=float, default=5e-4,   help="Weight decay")
    p.add_argument("--warmup-ep",  type=float, default=3.0,    help="Warmup epochs")
    p.add_argument("--reg-max",    type=int,   default=16,     help="Distance-distribution bins")
    p.add_argument("--device",     default="",               help="cuda / cpu / 0,1,...")
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--save-dir",   default="runs/train",    help="Output directory")
    p.add_argument("--resume",     default="",              help="Resume from checkpoint .pt")
    p.add_argument("--vis-every",  type=int, default=10,
                   help="Save layer-visualisation every N epochs (0 = disabled)")
    p.add_argument("--no-amp",     action="store_true",     help="Disable mixed precision")
    p.add_argument("--conf",       type=float, default=0.25, help="NMS conf for val")
    p.add_argument("--iou",        type=float, default=0.45, help="NMS IoU for val")
    p.add_argument("--cls-weights", default="",
                   help="Comma-separated per-class loss weights, e.g. 2.0,1.0,1.0 "
                        "(must match --classes length). Default: uniform (all 1.0). "
                        "There is no default bias toward any particular class.")
    p.add_argument("--nc",         type=int, default=None,
                   help="Override number of classes (default: inferred from --classes)")
    p.add_argument("--classes",    required=True,
                   help="Comma-separated class names, e.g. moringa_plant,weed. Required — "
                        "there is no default class list.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Learning-rate schedulers
# ---------------------------------------------------------------------------

def warmup_cosine_lrf(
    epoch: int,
    total_epochs: int,
    warmup_epochs: float,
    lrf: float,
) -> float:
    """Warmup linear then cosine decay; returns scale factor for LR."""
    if epoch < warmup_epochs:
        return epoch / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lrf + (1.0 - lrf) * cosine


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_ckpt(path: Path, model: nn.Module, optimizer, scaler,
              epoch: int, best_map: float, args: argparse.Namespace) -> None:
    torch.save({
        "epoch":     epoch,
        "best_map":  best_map,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler":    scaler.state_dict() if scaler else None,
        "args":      vars(args),
    }, path)


def load_ckpt(path: str, model: nn.Module, optimizer, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("best_map", 0.0)


# ---------------------------------------------------------------------------
# Visualise feature maps for a sample batch
# ---------------------------------------------------------------------------

def visualise_layers(
    model: nn.Module,
    sample_img: torch.Tensor,
    out_dir: Path,
    img_size: int = 256,
) -> None:
    model.eval()
    vis_dir = out_dir / "feature_maps"
    FeatureVisualizer.run_once(model, sample_img.unsqueeze(0), vis_dir, img_size)

    # Compose grid overview
    import cv2
    from utils.visualize import make_grid_image
    grid = make_grid_image(vis_dir, cols=8, cell_size=128)
    if grid is not None:
        cv2.imwrite(str(out_dir / "feature_grid.png"), grid)
    model.train()


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:   nn.Module,
    loader,
    device:  torch.device,
    img_size: int,
    conf:    float,
    iou_thr: float,
    classes: list[str] | None = None,
) -> dict[str, float]:
    model.eval()
    meter = MeanAveragePrecision(nc=model.nc, classes=classes)
    max_score_seen = 0.0
    total_dets = 0

    for batch in loader:
        imgs   = batch["imgs"].to(device)
        labels = batch["labels"].to(device)

        boxes, scores = model(imgs)
        max_score_seen = max(max_score_seen, scores.max().item())
        dets = nms_detections(boxes, scores, conf=conf, iou_thr=iou_thr)
        total_dets += sum(d.shape[0] for d in dets)
        meter.update(dets, labels.cpu(), img_size)

    print(f"  [Val] max_conf={max_score_seen:.4f}  total_dets={total_dets}  conf_thr={conf}")
    model.train()
    return meter.compute()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DETC] Device: {device}   Save dir: {save_dir}")

    # ── Resolve nc / class names ──────────────────────────────────────────────
    # Classes are always user-defined in the annotation tool — there is no
    # sensible default class list, so --classes is required.
    if not args.classes:
        raise ValueError(
            "--classes is required (comma-separated class names). "
            "There is no default class list — classes must come from "
            "the annotation tool's project data."
        )
    cls_list = [c.strip() for c in args.classes.split(",") if c.strip()]
    nc = args.nc or len(cls_list)
    print(f"[DETC] nc={nc}  classes={cls_list}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DETCModel(model_size=args.model, num_bins=args.reg_max,
                      img_size=args.imgsz, nc=nc, classes=cls_list).to(device)
    print(model.summary())

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = build_loader(
        args.data, "train", args.imgsz, args.batch, args.workers, augment=True
    )
    val_loader = build_loader(
        args.data, "val", args.imgsz, args.batch, args.workers, augment=False
    )
    print(f"[DETC] Train: {len(train_loader.dataset)} imgs  "
          f"Val: {len(val_loader.dataset)} imgs")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Per-class classification weight. Uniform by default — classes are
    # user-defined per project, so there is no general reason to bias
    # training toward whichever class happens to be first.
    if args.cls_weights:
        _cls_w = [float(x) for x in args.cls_weights.split(",") if x.strip()]
        if len(_cls_w) != model.nc:
            raise ValueError(
                f"--cls-weights has {len(_cls_w)} values but nc={model.nc}. "
                "Provide one weight per class, or omit for uniform weighting."
            )
    else:
        _cls_w = [1.0] * model.nc
    criterion = DetectionLoss(
        nc=model.nc, num_bins=args.reg_max, cls_weights=_cls_w
    ).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Separate weight-decay groups (BN / bias ← no decay)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.SGD(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr0,
        momentum=args.momentum,
        nesterov=True,
    )

    # ── AMP scaler ────────────────────────────────────────────────────────────
    use_amp = not args.no_amp and device.type == "cuda"
    scaler  = GradScaler("cuda") if use_amp else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch, best_map = 0, 0.0
    if args.resume:
        start_epoch, best_map = load_ckpt(args.resume, model, optimizer, scaler)
        print(f"[DETC] Resumed from epoch {start_epoch}, best mAP@0.5={best_map:.4f}")

    # ── LR scheduler (per-epoch lambda) ───────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: warmup_cosine_lrf(
            e + start_epoch, args.epochs, args.warmup_ep, args.lrf
        ),
    )

    # Keep a sample image for feature visualisation
    sample_batch = next(iter(val_loader))
    sample_img   = sample_batch["imgs"][0].to(device)

    # ── Training epochs ───────────────────────────────────────────────────────
    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        running = {"loss": 0.0, "loss_box": 0.0, "loss_cls": 0.0, "loss_dist": 0.0}
        n_batches = 0

        for batch_i, batch in enumerate(train_loader):
            imgs   = batch["imgs"].to(device)
            labels = batch["labels"].to(device)

            # Build anchor points from head (first forward to populate cache)
            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                raw_preds = model(imgs)          # list[(B, no, H_i, W_i)]

                # Get anchor points matching current feature-map sizes
                feats = []
                for i, t in enumerate(raw_preds):
                    B, _, H, W = t.shape
                    feats.append(t.view(B, -1, H, W))

                with torch.no_grad():
                    anchor_pts, strides = model.head._make_anchors(
                        [t.view(t.shape[0], -1, t.shape[2], t.shape[3])
                         for t in raw_preds],
                        dtype=imgs.dtype, device=device,
                    )
                    # _make_anchors needs (B,C,H,W) inputs; pass dummy tensors
                    dummy = [
                        torch.empty(1, 1, t.shape[2], t.shape[3], device=device)
                        for t in raw_preds
                    ]
                    anchor_pts, strides = model.head._make_anchors(
                        dummy, dtype=imgs.dtype, device=device
                    )

                losses = criterion(
                    raw_preds, labels, anchor_pts, strides,
                    model.head, args.imgsz,
                )
                loss = losses["loss"]

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

            for k in running:
                running[k] += losses[k].item() if k in losses else 0
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        print(
            f"Epoch {epoch+1:>4}/{args.epochs}  "
            f"loss={avg['loss']:.4f}  "
            f"box={avg['loss_box']:.4f}  "
            f"cls={avg['loss_cls']:.4f}  "
            f"dfl={avg['loss_dist']:.4f}  "
            f"lr={lr_now:.6f}  "
            f"time={elapsed:.1f}s"
        )

        # ── Feature-map visualisation ─────────────────────────────────────────
        if args.vis_every > 0 and (epoch + 1) % args.vis_every == 0:
            ep_vis_dir = save_dir / f"epoch_{epoch+1:04d}"
            ep_vis_dir.mkdir(parents=True, exist_ok=True)
            visualise_layers(model, sample_img, ep_vis_dir, img_size=256)

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_res = validate(model, val_loader, device, args.imgsz,
                               args.conf, args.iou, classes=cls_list)
            map50 = val_res["mAP@0.5"]
            print(f"  [Val] mAP@0.5={map50:.4f}  mAP@.5:.95={val_res['mAP@0.5:0.95']:.4f}")
            for cls_name, ap in val_res["per_class_AP50"].items():
                print(f"        {cls_name:<16} AP@0.5={ap:.4f}")

            # ── Checkpoint ────────────────────────────────────────────────────
            save_ckpt(save_dir / "last.pt", model, optimizer, scaler,
                      epoch + 1, best_map, args)
            if map50 > best_map:
                best_map = map50
                save_ckpt(save_dir / "best.pt", model, optimizer, scaler,
                          epoch + 1, best_map, args)
                print(f"  [*] New best mAP@0.5 = {best_map:.4f} → saved to {save_dir/'best.pt'}")

        if _INTERRUPTED:
            print(f"[DETC] Saving interrupt checkpoint at epoch {epoch + 1} …")
            save_ckpt(save_dir / "interrupt.pt", model, optimizer, scaler,
                      epoch + 1, best_map, args)
            print(f"[DETC] Saved → {save_dir / 'interrupt.pt'}")
            break

    # ── Final layer visualisation ─────────────────────────────────────────────
    print("\n[DETC] Training complete. Saving final feature maps …")
    visualise_layers(model, sample_img, save_dir / "final", img_size=256)
    print(f"[DETC] Best mAP@0.5 = {best_map:.4f}")
    print(f"[DETC] Weights saved to {save_dir}/best.pt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    train(args)
