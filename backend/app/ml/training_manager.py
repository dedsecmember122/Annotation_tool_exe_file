"""
Training manager — orchestrates data preparation, adapter training, and
model version bookkeeping.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.models.models import Annotation, Image, LabelClass, ModelVersion

settings = get_settings()


class TrainingManager:

    def __init__(self, project_id: int, db: Session) -> None:
        self._project_id = project_id
        self._db = db

    def _prepare_dataset(self) -> tuple[list[dict], list[str]]:
        """Return (annotated_images, class_names) for training.

        class_names is derived only from classes that appear on at least one
        reviewed annotation in this project — not from every LabelClass row
        that exists for the project. Querying all LabelClass rows previously
        pulled in unused/placeholder classes (e.g. leftover template slots
        like "class_1", "class_2") into training even though the user never
        annotated anything with them.
        """
        images = (
            self._db.query(Image)
            .filter(
                Image.project_id == self._project_id,
                Image.status.in_(["annotated", "auto_annotated"]),
            )
            .all()
        )

        dataset: list[dict] = []
        used_class_ids: set[int] = set()
        for img in images:
            annotations_orm = (
                self._db.query(Annotation)
                .filter(
                    Annotation.image_id == img.id,
                    Annotation.reviewed == True,  # noqa: E712
                )
                .all()
            )
            if not annotations_orm:
                continue
            for ann in annotations_orm:
                used_class_ids.add(ann.class_id)
            dataset.append({"image_path": img.storage_path, "annotations": annotations_orm})

        classes = (
            self._db.query(LabelClass)
            .filter(
                LabelClass.project_id == self._project_id,
                LabelClass.id.in_(used_class_ids),
            )
            .all()
            if used_class_ids else []
        )
        id_to_name = {c.id: c.name for c in classes}
        class_names = [c.name for c in classes]

        # Now build the final annotation dicts using only resolved names.
        # (Second pass keeps the id->name lookup scoped to used classes only.)
        final_dataset: list[dict] = []
        for entry in dataset:
            anns = []
            for ann in entry["annotations"]:
                coords = ann.get_coordinates()
                name = id_to_name.get(ann.class_id)
                if name is None:
                    continue  # class not in the used/resolved set — skip defensively
                anns.append({
                    "class": name,
                    "bbox": [
                        coords.get("x1", 0), coords.get("y1", 0),
                        coords.get("x2", 0), coords.get("y2", 0),
                    ],
                })
            final_dataset.append({"image_path": entry["image_path"], "annotations": anns})

        return final_dataset, class_names

    def _next_version(self) -> int:
        latest = (
            self._db.query(ModelVersion)
            .filter(ModelVersion.project_id == self._project_id)
            .order_by(ModelVersion.version_number.desc())
            .first()
        )
        return (latest.version_number + 1) if latest else 1

    def train(
        self,
        epochs: int = 60,
        train_split: float = 0.8,
        model_size: str = "n",
        img_size: int = 640,
        log_callback=None,
        on_process_start=None,
    ) -> dict:
        from backend.app.db import SessionLocal
        from backend.app.ml.custom_model_adapter import CustomModelAdapter

        dataset, class_names = self._prepare_dataset()
        if not dataset:
            raise ValueError("No reviewed annotations found for training.")
        if not class_names:
            raise ValueError(
                "No labeled classes found for this project. "
                "Create at least one class and annotate images before training."
            )

        nc = len(class_names)

        # Version number needs to be read before we release the session,
        # since we recompute it again below after re-opening a fresh one.
        next_version = self._next_version()

        output_dir = (
            Path.home()
            / "AnnotationTool"
            / "models"
            / str(self._project_id)
            / f"v{next_version}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        adapter = CustomModelAdapter()
        if settings.CUSTOM_MODEL_DIR:
            # Load existing weights for warm-start if available
            try:
                adapter.load(
                    settings.CUSTOM_MODEL_DIR,
                    class_names=class_names,
                    nc=nc,
                )
            except Exception:
                # No prior weights — train from scratch (model will init fresh)
                adapter._model_dir = settings.CUSTOM_MODEL_DIR
                adapter._class_names = class_names
                adapter._nc = nc
                import sys
                if settings.CUSTOM_MODEL_DIR not in sys.path:
                    sys.path.insert(0, settings.CUSTOM_MODEL_DIR)

        # Release the DB session before the long-running subprocess call.
        # Holding a session open for the ~tens of minutes training takes
        # previously left a transaction sitting on SQLite for the whole
        # run, colliding with the periodic log-progress writes and status
        # polls happening concurrently on other connections.
        self._db.close()

        checkpoint = adapter.train(
            dataset,
            str(output_dir),
            class_names=class_names,
            epochs=epochs,
            train_split=train_split,
            model_size=model_size,
            img_size=img_size,
            log_callback=log_callback,
            on_process_start=on_process_start,
        )

        # Fresh session for the final write — the old one may be stale
        # after being closed for the duration of training.
        self._db = SessionLocal()
        version = ModelVersion(
            project_id=self._project_id,
            version_number=self._next_version(),
            checkpoint_path=checkpoint,
            trained_on_count=len(dataset),
            metrics=json.dumps({
                "nc": nc,
                "classes": class_names,
                "epochs": epochs,
                "train_split": train_split,
                "model_size": model_size,
                "img_size": img_size,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }),
        )
        self._db.add(version)
        self._db.commit()
        return {
            "checkpoint": checkpoint,
            "trained_on": len(dataset),
            "nc": nc,
            "classes": class_names,
        }
