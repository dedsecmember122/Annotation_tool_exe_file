"""
Train Model Dialog — dedicated training workflow, separate from auto-annotation.

Steps:
  1. Dataset check    — verifies images are annotated & reviewed (≥50 required)
  2. Dataset split    — user picks train/validation percentage
  3. Model settings   — model size (n/s/m/l/x, default n), image size, epochs
  4. Train            — background job with live log; save best.pt locally when done
"""
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from frontend.api_client import APIClient, APIError

MIN_REVIEWED = 50  # minimum reviewed annotations before training is allowed

MODEL_SIZES = [
    ("n", "n  —  nano (fastest, default)"),
    ("s", "s  —  small"),
    ("m", "m  —  medium"),
    ("l", "l  —  large"),
    ("x", "x  —  extra large (most accurate, slowest)"),
]

IMAGE_SIZES = [320, 416, 512, 640, 768, 960, 1280]


# ── Background workers ────────────────────────────────────────────────────────

class TrainWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, api: APIClient, project_id: int, epochs: int,
                 train_split: float, model_size: str, img_size: int) -> None:
        super().__init__()
        self._api = api
        self._project_id = project_id
        self._epochs = epochs
        self._train_split = train_split
        self._model_size = model_size
        self._img_size = img_size

    def run(self) -> None:
        try:
            result = self._api.start_training(
                self._project_id,
                epochs=self._epochs,
                train_split=self._train_split,
                model_size=self._model_size,
                img_size=self._img_size,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class DownloadModelWorker(QObject):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, api: APIClient, project_id: int, dest_path: str) -> None:
        super().__init__()
        self._api = api
        self._project_id = project_id
        self._dest_path = dest_path

    def run(self) -> None:
        try:
            data = self._api.download_model(self._project_id)
            Path(self._dest_path).write_bytes(data)
            self.finished.emit(self._dest_path)
        except Exception as e:
            self.error.emit(str(e))


# ── Main dialog ───────────────────────────────────────────────────────────────

class TrainModelDialog(QDialog):

    def __init__(self, api: APIClient, project: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._project = project
        self._stats: dict = {}
        self._training_active = False  # a pending/running job exists for this project
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_training_status)
        self._setup_ui()
        self._resume_active_job()
        self._refresh_stats()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Train Model")
        self.setMinimumWidth(640)
        self.setMinimumHeight(700)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        main_layout.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)

        root = QVBoxLayout(container)
        root.setSpacing(14)
        root.setContentsMargins(22, 22, 22, 22)

        # Header
        title = QLabel(f"Train Model — {self._project['name']}")
        title.setObjectName("dialogTitle")
        root.addWidget(title)

        subtitle = QLabel(
            "Train the custom detection model on this project's reviewed annotations. "
            "Classes are taken automatically from the project."
        )
        subtitle.setObjectName("mutedLabel")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # ── Step 1: Dataset check ─────────────────────────────────────────────
        check_group = QGroupBox("Step 1 — Dataset Check")
        check_layout = QVBoxLayout(check_group)
        check_layout.setSpacing(8)

        self._stats_label = QLabel("Checking annotations…")
        self._stats_label.setWordWrap(True)
        check_layout.addWidget(self._stats_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setMaximum(MIN_REVIEWED)
        self._progress_bar.setValue(0)
        check_layout.addWidget(self._progress_bar)

        self._check_result_label = QLabel("")
        self._check_result_label.setObjectName("warningLabel")
        self._check_result_label.setWordWrap(True)
        check_layout.addWidget(self._check_result_label)

        refresh_btn = QPushButton("↻  Re-check")
        refresh_btn.setFixedWidth(130)
        refresh_btn.clicked.connect(self._refresh_stats)
        check_layout.addWidget(refresh_btn)
        root.addWidget(check_group)

        # ── Step 2: Dataset split ─────────────────────────────────────────────
        split_group = QGroupBox("Step 2 — Dataset Split (Training / Validation)")
        split_layout = QVBoxLayout(split_group)
        split_layout.setSpacing(8)

        split_desc = QLabel(
            "Choose how much of the annotated data is used for training; "
            "the rest is used for validation."
        )
        split_desc.setObjectName("mutedLabel")
        split_desc.setWordWrap(True)
        split_layout.addWidget(split_desc)

        split_row = QHBoxLayout()
        self._split_slider = QSlider(Qt.Horizontal)
        self._split_slider.setRange(50, 95)
        self._split_slider.setValue(80)
        self._split_slider.setSingleStep(5)
        self._split_slider.setPageStep(5)
        self._split_slider.valueChanged.connect(self._update_split_label)
        split_row.addWidget(self._split_slider, stretch=1)
        self._split_label = QLabel("")
        self._split_label.setFixedWidth(230)
        self._split_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        split_row.addWidget(self._split_label)
        split_layout.addLayout(split_row)
        root.addWidget(split_group)

        # ── Step 3: Model settings ────────────────────────────────────────────
        model_group = QGroupBox("Step 3 — Model Settings")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(10)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Model size:"))
        self._model_size_combo = QComboBox()
        for value, label in MODEL_SIZES:
            self._model_size_combo.addItem(label, value)
        self._model_size_combo.setCurrentIndex(0)  # default: n
        size_row.addWidget(self._model_size_combo, stretch=1)
        model_layout.addLayout(size_row)

        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Image size:"))
        self._img_size_combo = QComboBox()
        for s in IMAGE_SIZES:
            self._img_size_combo.addItem(f"{s} × {s}", s)
        self._img_size_combo.setCurrentIndex(IMAGE_SIZES.index(640))  # default: 640
        img_row.addWidget(self._img_size_combo, stretch=1)
        model_layout.addLayout(img_row)

        epochs_row = QHBoxLayout()
        epochs_row.addWidget(QLabel("Epochs:"))
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(10, 300)
        self._epochs_spin.setValue(60)
        self._epochs_spin.setFixedWidth(90)
        epochs_row.addWidget(self._epochs_spin)
        epochs_row.addStretch()
        model_layout.addLayout(epochs_row)

        note = QLabel(
            "Larger model sizes and image sizes are more accurate but train and run slower. "
            "The default (n @ 640) is a good starting point."
        )
        note.setObjectName("mutedLabel")
        note.setWordWrap(True)
        model_layout.addWidget(note)
        root.addWidget(model_group)

        # ── Step 4: Train ─────────────────────────────────────────────────────
        train_group = QGroupBox("Step 4 — Train")
        train_layout = QVBoxLayout(train_group)
        train_layout.setSpacing(10)

        train_btn_row = QHBoxLayout()
        self._train_btn = QPushButton("🚀  Train Model")
        self._train_btn.setObjectName("primaryButton")
        self._train_btn.setFixedHeight(42)
        self._train_btn.setEnabled(False)
        self._train_btn.clicked.connect(self._start_training)
        train_btn_row.addWidget(self._train_btn, stretch=1)

        self._stop_btn = QPushButton("⏹  Stop Training")
        self._stop_btn.setObjectName("dangerButton")
        self._stop_btn.setFixedHeight(42)
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._stop_training)
        train_btn_row.addWidget(self._stop_btn)
        train_layout.addLayout(train_btn_row)

        self._train_status_label = QLabel("")
        self._train_status_label.setObjectName("warningLabel")
        self._train_status_label.setWordWrap(True)
        train_layout.addWidget(self._train_status_label)

        self._save_model_btn = QPushButton("💾  Save best.pt to Computer…")
        self._save_model_btn.setFixedHeight(34)
        self._save_model_btn.setEnabled(False)
        self._save_model_btn.clicked.connect(self._save_model_locally)
        train_layout.addWidget(self._save_model_btn)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFixedHeight(160)
        self._log_text.setObjectName("logView")
        train_layout.addWidget(self._log_text)
        root.addWidget(train_group)

        root.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn)

        self._update_split_label()

    # ── Stats / dataset check ─────────────────────────────────────────────────

    def _annotated_image_count(self) -> int:
        s = self._stats
        return s.get("annotated", 0) + s.get("auto_annotated", 0)

    def _update_split_label(self) -> None:
        pct = self._split_slider.value()
        n = self._annotated_image_count()
        if n > 0:
            n_train = max(1, int(n * pct / 100))
            n_val = max(1, n - n_train)
            self._split_label.setText(
                f"{pct}% train / {100 - pct}% val  (~{n_train} / {n_val} images)"
            )
        else:
            self._split_label.setText(f"{pct}% train / {100 - pct}% val")

    def _refresh_stats(self) -> None:
        try:
            self._stats = self._api.get_annotation_stats(self._project["id"])
        except APIError as e:
            self._stats = {}
            self._stats_label.setText(f"Could not load stats: {e.detail}")
            self._check_result_label.setText("Dataset check failed — is the backend running?")
            self._train_btn.setEnabled(False)
            return

        s = self._stats
        reviewed = s.get("reviewed_annotation_count", 0)
        can_train = s.get("can_train", False)
        has_model = s.get("has_trained_model", False)
        model_ver = s.get("latest_model_version")

        self._stats_label.setText(
            f"Total images: {s.get('total', 0)}   |   "
            f"Annotated: {s.get('annotated', 0)}   |   "
            f"Auto (review): {s.get('auto_annotated', 0)}   |   "
            f"Unannotated: {s.get('unannotated', 0)}\n"
            f"Reviewed annotations: {reviewed}   |   "
            f"Trained model: {'v' + str(model_ver) if model_ver else 'none yet'}"
        )

        self._progress_bar.setValue(min(reviewed, MIN_REVIEWED))
        if can_train:
            self._progress_bar.setFormat(f"{reviewed} reviewed ✓")
            self._check_result_label.setObjectName("successLabel")
            self._check_result_label.setText(
                "✓ Dataset check passed — enough reviewed annotations to train."
            )
        else:
            remaining = MIN_REVIEWED - reviewed
            self._progress_bar.setFormat(f"{reviewed} / {MIN_REVIEWED} reviewed")
            self._check_result_label.setObjectName("warningLabel")
            self._check_result_label.setText(
                f"✗ Not enough annotated data: review at least {remaining} more "
                f"annotation(s) before training (minimum {MIN_REVIEWED})."
            )
        # Re-polish so the objectName-based color change takes effect
        self._check_result_label.style().unpolish(self._check_result_label)
        self._check_result_label.style().polish(self._check_result_label)

        self._train_btn.setEnabled(can_train and not self._training_active)
        self._save_model_btn.setEnabled(has_model)
        self._update_split_label()

    # ── Training ──────────────────────────────────────────────────────────────

    def _resume_active_job(self) -> None:
        """Pick up an already pending/running training job for this project
        so reopening the dialog (or the app) shows live progress instead of
        the static dataset-check screen as if nothing were happening."""
        try:
            job = self._api.get_training_status(self._project["id"])
        except APIError:
            return  # no training job has ever been submitted for this project

        status = job.get("status", "")
        if status not in ("pending", "running"):
            return

        self._training_active = True
        self._train_btn.setEnabled(False)
        self._train_btn.setText("⏳  Training in progress…")
        self._stop_btn.setVisible(True)
        self._stop_btn.setEnabled(True)
        self._train_status_label.setText(
            f"Resumed: a training job is already {status} for this project…"
        )
        log = job.get("log", "")
        if log:
            self._log_text.setText(log)
            self._log_text.verticalScrollBar().setValue(
                self._log_text.verticalScrollBar().maximum()
            )
        self._poll_timer.start(5000)

    def _start_training(self) -> None:
        self._train_btn.setEnabled(False)
        self._train_btn.setText("⏳  Submitting…")
        self._train_status_label.setText("Submitting training job…")

        self._train_thread = QThread()
        self._train_worker = TrainWorker(
            self._api,
            self._project["id"],
            epochs=self._epochs_spin.value(),
            train_split=self._split_slider.value() / 100.0,
            model_size=self._model_size_combo.currentData(),
            img_size=self._img_size_combo.currentData(),
        )
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.finished.connect(self._on_train_submitted)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.finished.connect(self._train_thread.quit)
        self._train_worker.error.connect(self._train_thread.quit)
        self._train_thread.start()

    def _on_train_submitted(self, result: dict) -> None:
        self._training_active = True
        self._train_btn.setText("🚀  Train Model")
        self._stop_btn.setVisible(True)
        self._stop_btn.setEnabled(True)
        status = result.get("status", "pending")
        cfg = (
            f"model={self._model_size_combo.currentData()}, "
            f"imgsz={self._img_size_combo.currentData()}, "
            f"epochs={self._epochs_spin.value()}, "
            f"split={self._split_slider.value()}/{100 - self._split_slider.value()}"
        )
        self._train_status_label.setText(
            f"Training job submitted ({cfg}). Status: {status}. Polling for progress…"
        )
        self._log_text.append(f"Training job started ({cfg})…")
        self._poll_timer.start(5000)

    def _on_train_error(self, msg: str) -> None:
        self._train_btn.setText("🚀  Train Model")
        self._train_btn.setEnabled(True)
        self._train_status_label.setText(f"Error: {msg}")

    def _poll_training_status(self) -> None:
        try:
            job = self._api.get_training_status(self._project["id"])
        except APIError:
            return

        status = job.get("status", "")
        log = job.get("log", "")

        if status in ("completed", "failed", "cancelled"):
            self._poll_timer.stop()
            self._training_active = False
            self._train_btn.setText("🚀  Train Model")
            self._train_btn.setEnabled(True)
            self._stop_btn.setVisible(False)

            if status == "completed":
                self._train_status_label.setText(
                    "Training complete! Save best.pt below, or use Auto-Annotate "
                    "to label the remaining images."
                )
            elif status == "cancelled":
                self._train_status_label.setText(
                    "Training stopped. A checkpoint from the last completed epoch "
                    "was saved, if any — see log below."
                )
            else:
                self._train_status_label.setText("Training failed — see log below.")

            self._log_text.setText(log)
            self._log_text.verticalScrollBar().setValue(
                self._log_text.verticalScrollBar().maximum()
            )
            self._refresh_stats()
        else:
            self._train_status_label.setText(f"Training {status}…")
            if log:
                self._log_text.setText(log)
                self._log_text.verticalScrollBar().setValue(
                    self._log_text.verticalScrollBar().maximum()
                )

    def _stop_training(self) -> None:
        reply = QMessageBox.question(
            self, "Stop Training",
            "Stop the current training job? The model will finish its current "
            "epoch and save a checkpoint before stopping — progress so far isn't lost.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._stop_btn.setEnabled(False)
        self._stop_btn.setText("⏳  Stopping…")
        try:
            self._api.cancel_training(self._project["id"])
            self._train_status_label.setText(
                "Cancelling… finishing the current epoch, then stopping."
            )
        except APIError as e:
            QMessageBox.warning(self, "Stop Failed", f"Could not stop training:\n{e.detail}")
            self._stop_btn.setEnabled(True)
            self._stop_btn.setText("⏹  Stop Training")

    # ── Save best.pt locally ──────────────────────────────────────────────────

    def _save_model_locally(self) -> None:
        model_ver = self._stats.get("latest_model_version")
        suggested = f"best_v{model_ver}.pt" if model_ver else "best.pt"
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Trained Model", str(Path.home() / suggested),
            "PyTorch checkpoint (*.pt)"
        )
        if not dest:
            return

        self._save_model_btn.setEnabled(False)
        self._save_model_btn.setText("⏳  Saving…")

        self._dl_thread = QThread()
        self._dl_worker = DownloadModelWorker(self._api, self._project["id"], dest)
        self._dl_worker.moveToThread(self._dl_thread)
        self._dl_thread.started.connect(self._dl_worker.run)
        self._dl_worker.finished.connect(self._on_model_saved)
        self._dl_worker.error.connect(self._on_model_save_error)
        self._dl_worker.finished.connect(self._dl_thread.quit)
        self._dl_worker.error.connect(self._dl_thread.quit)
        self._dl_thread.start()

    def _on_model_saved(self, path: str) -> None:
        self._save_model_btn.setText("💾  Save best.pt to Computer…")
        self._save_model_btn.setEnabled(True)
        QMessageBox.information(self, "Model Saved", f"Trained model saved to:\n{path}")

    def _on_model_save_error(self, msg: str) -> None:
        self._save_model_btn.setText("💾  Save best.pt to Computer…")
        self._save_model_btn.setEnabled(True)
        QMessageBox.critical(self, "Save Failed", f"Could not save model:\n{msg}")

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        super().closeEvent(event)
