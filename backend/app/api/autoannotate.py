"""
Auto-annotate router:
  POST /api/projects/{id}/train                — kick off training job
  POST /api/projects/{id}/train/cancel         — gracefully stop the active job
  GET  /api/projects/{id}/train/status         — poll latest job
  GET  /api/projects/{id}/annotation-stats     — counts for loop decisions
  GET  /api/projects/{id}/model/download       — download latest best.pt
  POST /api/images/{id}/auto-annotate          — run inference on one image
  POST /api/projects/{id}/auto-annotate-batch  — run inference on all unannotated images
"""
import json
import re
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.api.auth import current_user_dep
from backend.app.core.config import get_settings
from backend.app.db import SessionLocal, get_db
from backend.app.models.models import Annotation, Image, LabelClass, ModelVersion, TrainingJob, User
from backend.app.schemas.schemas import AutoAnnotateRequest, TrainRequest, TrainingJobOut

router = APIRouter(tags=["autoannotate"])
settings = get_settings()

# train.py's stdout is meant for developers running it standalone from the
# CLI — device, save paths, model architecture dumps, per-batch loss
# components, torch warnings. None of that belongs in front of a customer,
# so only these shapes are translated into a customer-facing progress line;
# everything else from the subprocess is dropped before it ever reaches
# TrainingJob.log / the UI.
_EPOCH_LOG_RE = re.compile(r"^Epoch\s*(\d+)\s*/\s*(\d+)")
_BEST_CKPT_LOG_RE = re.compile(r"^\s*\[\*\]\s*New best")
_CANCEL_REQUESTED_LOG_RE = re.compile(r"^\[DETC\] Interrupt received")
_CANCEL_SAVED_LOG_RE = re.compile(r"^\[DETC\] Saving interrupt checkpoint")


def _simplify_training_log_line(line: str) -> str | None:
    """Translate a raw train.py stdout line into a customer-facing progress
    message, or return None to drop it. Keeps the log a simple progress
    indicator instead of leaking internal implementation details."""
    stripped = line.strip()
    m = _EPOCH_LOG_RE.match(stripped)
    if m:
        current, total = int(m.group(1)), int(m.group(2))
        pct = int(current / total * 100) if total else 0
        return f"Training… epoch {current}/{total} ({pct}%)\n"
    if _BEST_CKPT_LOG_RE.match(stripped):
        return "New best model checkpoint saved.\n"
    if _CANCEL_REQUESTED_LOG_RE.match(stripped):
        return "Cancelling… finishing the current epoch, then stopping.\n"
    if _CANCEL_SAVED_LOG_RE.match(stripped):
        return "Checkpoint saved. Stopping…\n"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Active-job registry — lets the /train/cancel endpoint reach the subprocess
# a background training thread is running. Keyed by TrainingJob.id.
# ─────────────────────────────────────────────────────────────────────────────

_jobs_lock = threading.Lock()
_active_jobs: dict[int, dict] = {}  # job_id -> {"process": Popen | None, "cancel_requested": bool}


def _send_graceful_interrupt(process) -> None:
    """Ask the train.py subprocess to stop after its current epoch and save
    a checkpoint. train.py has a SIGINT handler that does exactly this.

    Windows can't deliver SIGINT to a subprocess the way POSIX can, so there
    we fall back to terminate() — an immediate stop with no interrupt
    checkpoint (whatever best.pt/last.pt already exists from the last
    periodic save, if any, is what a cancel there recovers).
    """
    if sys.platform == "win32":
        process.terminate()
    else:
        process.send_signal(signal.SIGINT)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_latest_model_version(project_id: int, db: Session):
    """Return latest ModelVersion for this project, or None."""
    return (
        db.query(ModelVersion)
        .filter(ModelVersion.project_id == project_id)
        .order_by(ModelVersion.version_number.desc())
        .first()
    )


def _build_adapter(model_version: ModelVersion):
    """Load the CustomModelAdapter from checkpoint stored in model_version."""
    from backend.app.ml.custom_model_adapter import CustomModelAdapter

    adapter = CustomModelAdapter()
    meta = json.loads(model_version.metrics or "{}")
    class_names = meta.get("classes") or []
    nc = meta.get("nc") or len(class_names)

    if settings.CUSTOM_MODEL_DIR:
        adapter.load(
            settings.CUSTOM_MODEL_DIR,
            class_names=class_names or None,
            nc=nc or None,
            weights_path=model_version.checkpoint_path or None,
            model_size=meta.get("model_size"),
            img_size=meta.get("img_size"),
        )
    else:
        raise RuntimeError("CUSTOM_MODEL_DIR not configured in settings.")
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Background training
# ─────────────────────────────────────────────────────────────────────────────

def _run_training(
    job_id: int,
    project_id: int,
    epochs: int = 60,
    train_split: float = 0.8,
    model_size: str = "n",
    img_size: int = 640,
) -> None:
    log_buffer: list[str] = ["Training job started...\n"]

    with _jobs_lock:
        _active_jobs[job_id] = {"process": None, "cancel_requested": False}

    try:
        # Short-lived session just to mark the job "running" — released
        # immediately rather than held open for the whole training run.
        with SessionLocal() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if not job:
                return
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            db.commit()

        def on_process_start(process):
            with _jobs_lock:
                entry = _active_jobs.get(job_id)
                if entry is None:
                    return
                entry["process"] = process
                already_cancelled = entry["cancel_requested"]
            if already_cancelled:
                # /train/cancel was called during dataset prep, before the
                # subprocess existed to signal — catch up now.
                _send_graceful_interrupt(process)

        def handle_log(line: str):
            display = _simplify_training_log_line(line)
            if display is None:
                return  # internal detail — not shown to the customer
            log_buffer.append(display)
            # Update DB every 2 lines to stream logs live to UI. Previously
            # every 10 lines, and failures here were silently swallowed —
            # combined with a long-lived session elsewhere colliding with
            # these writes on SQLite, that made a working training run look
            # completely frozen in the UI. Failures are now logged instead
            # of hidden, so a real problem is visible instead of silent.
            if len(log_buffer) % 2 == 0:
                try:
                    with SessionLocal() as local_db:
                        j = local_db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                        if j:
                            j.log = "".join(log_buffer)
                            local_db.commit()
                except Exception as log_exc:
                    print(f"[training job {job_id}] failed to write log update: {log_exc}")

        # TrainingManager manages its own session lifecycle internally —
        # it closes this one before the long subprocess call and opens a
        # fresh one afterward, so nothing holds a lock for the training
        # duration.
        from backend.app.ml.training_manager import TrainingManager
        train_db = SessionLocal()
        tm = TrainingManager(project_id, train_db)
        result = tm.train(
            epochs=epochs,
            train_split=train_split,
            model_size=model_size,
            img_size=img_size,
            log_callback=handle_log,
            on_process_start=on_process_start,
        )

        with _jobs_lock:
            was_cancelled = _active_jobs.get(job_id, {}).get("cancel_requested", False)

        if was_cancelled:
            summary = (
                f"Training cancelled by user.\n"
                f"Partial checkpoint saved from {result['trained_on']} images.\n"
                f"Classes ({result['nc']}): {', '.join(result['classes'])}\n"
                f"Checkpoint: {result['checkpoint']}"
            )
        else:
            summary = (
                f"Training completed.\n"
                f"Trained on {result['trained_on']} images.\n"
                f"Classes ({result['nc']}): {', '.join(result['classes'])}\n"
                f"Checkpoint: {result['checkpoint']}"
            )
        log = "".join(log_buffer) + "\n\n" + summary

        # Fresh session + re-fetch the job row for the final write. The
        # object from the first session above is detached by now (that
        # session was closed before the long subprocess ran) — committing
        # against a detached object silently persists nothing, so re-query
        # it here rather than reusing the stale reference.
        with SessionLocal() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                job.status = "cancelled" if was_cancelled else "completed"
                job.completed_at = datetime.now(timezone.utc)
                job.log = log
                db.commit()

    except Exception as exc:
        with _jobs_lock:
            was_cancelled = _active_jobs.get(job_id, {}).get("cancel_requested", False)
        with SessionLocal() as db:
            job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            if job:
                job.status = "failed"
                # Append the exception to what we already captured instead of replacing it
                note = "\n\n[Cancelled before any checkpoint could be saved]" if was_cancelled else ""
                job.log = "".join(log_buffer) + f"\n\n[ERROR]: {str(exc)}" + note
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
    finally:
        with _jobs_lock:
            _active_jobs.pop(job_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/projects/{project_id}/train", response_model=TrainingJobOut, status_code=202)
def start_training(
    project_id: int,
    req: TrainRequest = TrainRequest(),
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> TrainingJob:
    """Kick off a background training job for the project."""
    epochs = req.epochs
    train_split = max(0.5, min(0.95, req.train_split))
    model_size = req.model_size if req.model_size in ("n", "s", "m", "l", "x") else "n"
    img_size = max(320, min(1280, int(req.img_size)))
    img_size -= img_size % 32
    # Check if there's already a running job
    running = db.query(TrainingJob).filter(
        TrainingJob.project_id == project_id,
        TrainingJob.status == "running",
    ).first()
    if running:
        raise HTTPException(409, "A training job is already running for this project")

    # Enforce minimum annotation count (≥ 50)
    reviewed_count = (
        db.query(Annotation)
        .join(Image, Image.id == Annotation.image_id)
        .filter(
            Image.project_id == project_id,
            Annotation.reviewed == True,  # noqa: E712
        )
        .count()
    )
    if reviewed_count < 50:
        raise HTTPException(
            400,
            f"Need at least 50 reviewed annotations to train (you have {reviewed_count}). "
            "Please annotate more images and mark them as reviewed."
        )

    # Clamp epochs to a sane range
    epochs = max(10, min(300, epochs))

    job = TrainingJob(project_id=project_id, status="pending")
    db.add(job)
    db.commit()
    db.refresh(job)

    t = threading.Thread(
        target=_run_training,
        args=(job.id, project_id, epochs, train_split, model_size, img_size),
        daemon=True,
    )
    t.start()
    return job


@router.post("/projects/{project_id}/train/cancel", response_model=TrainingJobOut)
def cancel_training(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> TrainingJob:
    """Gracefully stop the active training job for this project. train.py
    finishes its current epoch and saves a checkpoint before exiting, so
    whatever progress was made up to that point isn't lost."""
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.project_id == project_id)
        .order_by(TrainingJob.created_at.desc())
        .first()
    )
    if not job or job.status not in ("pending", "running"):
        raise HTTPException(400, "No active training job to cancel.")

    with _jobs_lock:
        entry = _active_jobs.get(job.id)
        if entry is None:
            # The background thread hasn't registered the job yet — this is
            # a very small window right after submission.
            raise HTTPException(409, "Training job is still starting up — try again in a moment.")
        entry["cancel_requested"] = True
        process = entry["process"]

    if process is not None:
        _send_graceful_interrupt(process)
    # else: dataset prep is still running: on_process_start() will see
    # cancel_requested=True and signal the subprocess the moment it starts.

    return job


@router.get("/projects/{project_id}/model/download")
def download_model(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> FileResponse:
    """Download the latest trained best.pt checkpoint for this project."""
    latest_version = _load_latest_model_version(project_id, db)
    if not latest_version:
        raise HTTPException(404, "No trained model found for this project.")
    ckpt = Path(latest_version.checkpoint_path or "")
    if not ckpt.is_file():
        raise HTTPException(404, f"Checkpoint file missing on disk: {ckpt}")
    return FileResponse(
        path=str(ckpt),
        media_type="application/octet-stream",
        filename=f"best_v{latest_version.version_number}.pt",
    )


@router.get("/projects/{project_id}/train/status", response_model=TrainingJobOut)
def get_training_status(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> TrainingJob:
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.project_id == project_id)
        .order_by(TrainingJob.created_at.desc())
        .first()
    )
    if not job:
        raise HTTPException(404, "No training jobs found for this project")
    return job


@router.get("/projects/{project_id}/annotation-stats")
def get_annotation_stats(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> dict:
    """
    Return per-status image counts and reviewed annotation count.
    Used by the frontend to decide if training threshold is met.
    """
    all_images = db.query(Image).filter(Image.project_id == project_id).all()
    total = len(all_images)
    counts: dict = {}
    for img in all_images:
        counts[img.status] = counts.get(img.status, 0) + 1

    reviewed_count = (
        db.query(Annotation)
        .join(Image, Image.id == Annotation.image_id)
        .filter(
            Image.project_id == project_id,
            Annotation.reviewed == True,  # noqa: E712
        )
        .count()
    )

    latest_version = _load_latest_model_version(project_id, db)
    has_trained_model = latest_version is not None

    return {
        "total": total,
        "unannotated": counts.get("unannotated", 0),
        "in_progress": counts.get("in_progress", 0),
        "annotated": counts.get("annotated", 0),
        "auto_annotated": counts.get("auto_annotated", 0),
        "reviewed_annotation_count": reviewed_count,
        "can_train": reviewed_count >= 50,
        "has_trained_model": has_trained_model,
        "latest_model_version": latest_version.version_number if latest_version else None,
    }


@router.post("/images/{image_id}/auto-annotate", status_code=202)
def auto_annotate_image(
    image_id: int,
    req: AutoAnnotateRequest = AutoAnnotateRequest(),
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> dict:
    """Run inference with the latest trained model on a single image."""
    confidence_threshold = req.confidence_threshold
    img = db.query(Image).filter(Image.id == image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")

    latest_version = _load_latest_model_version(img.project_id, db)
    if not latest_version:
        raise HTTPException(
            404,
            "No trained model found for this project. "
            "Train the model first (annotate ≥50 images → Train Model)."
        )

    if not settings.CUSTOM_MODEL_DIR:
        raise HTTPException(500, "CUSTOM_MODEL_DIR not configured.")

    try:
        adapter = _build_adapter(latest_version)
        detections = adapter.predict(img.storage_path, conf=confidence_threshold)
    except Exception as exc:
        raise HTTPException(500, f"Inference error: {exc}")

    classes = db.query(LabelClass).filter(LabelClass.project_id == img.project_id).all()
    name_to_id = {c.name: c.id for c in classes}

    # Remove previous auto-annotations on this image
    db.query(Annotation).filter(
        Annotation.image_id == image_id,
        Annotation.source == "auto",
    ).delete()

    created = 0
    for det in detections:
        ann = Annotation(
            image_id=image_id,
            class_id=name_to_id.get(det.get("class", ""), None),
            shape_type="bbox",
            source="auto",
            confidence=det.get("confidence", 0.0),
            reviewed=False,
            created_by=user.id,
        )
        ann.set_coordinates({
            "x1": det["bbox"][0], "y1": det["bbox"][1],
            "x2": det["bbox"][2], "y2": det["bbox"][3],
        })
        db.add(ann)
        created += 1

    if created > 0:
        img.status = "auto_annotated"
    db.commit()
    return {"created": created, "message": f"{created} auto-annotations created"}


@router.post("/projects/{project_id}/auto-annotate-batch", status_code=202)
def auto_annotate_batch(
    project_id: int,
    req: AutoAnnotateRequest = AutoAnnotateRequest(),
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> dict:
    """
    Run inference on ALL unannotated images in the project using the latest
    trained model. This is the core of the auto-annotation loop.
    """
    confidence_threshold = req.confidence_threshold
    latest_version = _load_latest_model_version(project_id, db)
    if not latest_version:
        raise HTTPException(
            404,
            "No trained model found for this project. "
            "Train the model first (annotate ≥50 images → Train Model)."
        )

    if not settings.CUSTOM_MODEL_DIR:
        raise HTTPException(500, "CUSTOM_MODEL_DIR not configured.")

    try:
        adapter = _build_adapter(latest_version)
    except Exception as exc:
        raise HTTPException(500, f"Model load error: {exc}")

    unannotated_images = (
        db.query(Image)
        .filter(
            Image.project_id == project_id,
            Image.status == "unannotated",
        )
        .all()
    )

    if not unannotated_images:
        return {"processed": 0, "total_annotations": 0, "message": "No unannotated images found."}

    classes = db.query(LabelClass).filter(LabelClass.project_id == project_id).all()
    name_to_id = {c.name: c.id for c in classes}

    processed = 0
    total_anns = 0

    for img in unannotated_images:
        try:
            detections = adapter.predict(img.storage_path, conf=confidence_threshold)
        except Exception:
            continue  # skip unreadable images

        created = 0
        for det in detections:
            ann = Annotation(
                image_id=img.id,
                class_id=name_to_id.get(det.get("class", ""), None),
                shape_type="bbox",
                source="auto",
                confidence=det.get("confidence", 0.0),
                reviewed=False,
                created_by=user.id,
            )
            ann.set_coordinates({
                "x1": det["bbox"][0], "y1": det["bbox"][1],
                "x2": det["bbox"][2], "y2": det["bbox"][3],
            })
            db.add(ann)
            created += 1

        if created > 0:
            img.status = "auto_annotated"

        total_anns += created
        processed += 1

    db.commit()
    return {
        "processed": processed,
        "total_annotations": total_anns,
        "message": (
            f"Auto-annotated {processed} images with {total_anns} detections "
            f"using model v{latest_version.version_number}."
        ),
    }
