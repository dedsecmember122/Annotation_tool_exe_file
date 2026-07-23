"""
Auto-Annotate Dialog — provides two modes:
  1. Manual Only  — informational, no action (user annotates by hand)
  2. Auto-Annotation Loop — stats display + Train Model + Auto-Annotate Remaining

The training step lets the user choose epochs and the train/validation split,
and a trained best.pt can be saved anywhere on the local computer.
"""
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QObject, Signal, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from frontend.api_client import APIClient, APIError


BOOTSTRAP_THRESHOLD = 50  # minimum reviewed annotations before training is allowed


# ── Background workers ────────────────────────────────────────────────────────

class TrainWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, api: APIClient, project_id: int, epochs: int, train_split: float) -> None:
        super().__init__()
        self._api = api
        self._project_id = project_id
        self._epochs = epochs
        self._train_split = train_split

    def run(self) -> None:
        try:
            result = self._api.start_training(
                self._project_id, epochs=self._epochs, train_split=self._train_split
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class BatchAnnotateWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, api: APIClient, project_id: int, confidence: float) -> None:
        super().__init__()
        self._api = api
        self._project_id = project_id
        self._confidence = confidence

    def run(self) -> None:
        try:
            result = self._api.auto_annotate_batch(self._project_id, self._confidence)
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

class AutoAnnotateDialog(QDialog):

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
        self.setWindowTitle("Auto-Annotation & Training")
        self.setMinimumWidth(620)
        self.setMinimumHeight(680)

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
        title = QLabel(f"Auto-Annotation — {self._project['name']}")
        title.setObjectName("dialogTitle")
        root.addWidget(title)

        subtitle = QLabel(
            "Annotate a small set of images by hand, train the detection model on them, "
            "then let the model annotate the rest for you."
        )
        subtitle.setObjectName("mutedLabel")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # Mode selector
        mode_group = QGroupBox("Annotation Mode")
        mode_layout = QVBoxLayout(mode_group)

        self._btn_manual = QRadioButton("Manual only — annotate every image by hand")
        self._btn_auto = QRadioButton(
            f"Auto-annotation loop — annotate ≥{BOOTSTRAP_THRESHOLD} images → train → "
            "auto-annotate the rest → review → repeat"
        )
        self._btn_manual.setChecked(True)

        bg = QButtonGroup(self)
        bg.addButton(self._btn_manual)
        bg.addButton(self._btn_auto)

        mode_layout.addWidget(self._btn_manual)
        mode_layout.addWidget(self._btn_auto)
        root.addWidget(mode_group)

        self._btn_manual.toggled.connect(self._on_mode_changed)
        self._btn_auto.toggled.connect(self._on_mode_changed)

        # ── Manual mode info panel ────────────────────────────────────────────
        self._manual_panel = QFrame()
        manual_layout = QVBoxLayout(self._manual_panel)
        info = QLabel(
            "In Manual mode, annotate each image yourself using the BBox or Polygon tools.\n"
            "All annotations are fully controlled by you — no automated predictions are made."
        )
        info.setWordWrap(True)
        info.setObjectName("mutedLabel")
        manual_layout.addWidget(info)
        root.addWidget(self._manual_panel)

        # ── Auto loop panel ───────────────────────────────────────────────────
        self._auto_panel = QFrame()
        self._auto_panel.setVisible(False)
        auto_layout = QVBoxLayout(self._auto_panel)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        auto_layout.setSpacing(12)

        # Stats box
        stats_group = QGroupBox("Annotation Progress")
        stats_layout = QVBoxLayout(stats_group)
        self._stats_label = QLabel("Loading stats…")
        stats_layout.addWidget(self._stats_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat(f"%v / {BOOTSTRAP_THRESHOLD} reviewed")
        self._progress_bar.setMaximum(BOOTSTRAP_THRESHOLD)
        self._progress_bar.setValue(0)
        stats_layout.addWidget(self._progress_bar)

        refresh_btn = QPushButton("↻  Refresh Stats")
        refresh_btn.setFixedWidth(150)
        refresh_btn.clicked.connect(self._refresh_stats)
        stats_layout.addWidget(refresh_btn)
        auto_layout.addWidget(stats_group)

        # ── Step 1: Train ─────────────────────────────────────────────────────
        step1_group = QGroupBox("Step 1 — Train the Model")
        step1_layout = QVBoxLayout(step1_group)
        step1_layout.setSpacing(10)
        step1_desc = QLabel(
            "Trains the custom detection model on all reviewed annotations. "
            "Class names and count are taken automatically from this project's classes."
        )
        step1_desc.setWordWrap(True)
        step1_desc.setObjectName("mutedLabel")
        step1_layout.addWidget(step1_desc)

        # Epochs row
        epochs_row = QHBoxLayout()
        epochs_row.addWidget(QLabel("Epochs (50–70):"))
        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(50, 70)
        self._epochs_spin.setValue(60)
        self._epochs_spin.setFixedWidth(80)
        epochs_row.addWidget(self._epochs_spin)
        epochs_row.addStretch()
        step1_layout.addLayout(epochs_row)

        # Train/val split row
        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Train / Validation split:"))
        self._split_slider = QSlider(Qt.Horizontal)
        self._split_slider.setRange(50, 95)
        self._split_slider.setValue(80)
        self._split_slider.setSingleStep(5)
        self._split_slider.setPageStep(5)
        self._split_slider.valueChanged.connect(self._on_split_changed)
        split_row.addWidget(self._split_slider, stretch=1)
        self._split_label = QLabel("80% train / 20% val")
        self._split_label.setFixedWidth(150)
        self._split_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        split_row.addWidget(self._split_label)
        step1_layout.addLayout(split_row)

        train_btn_row = QHBoxLayout()
        self._train_btn = QPushButton("🚀  Train Model")
        self._train_btn.setObjectName("primaryButton")
        self._train_btn.setFixedHeight(40)
        self._train_btn.clicked.connect(self._start_training)
        train_btn_row.addWidget(self._train_btn, stretch=1)

        self._stop_btn = QPushButton("⏹  Stop Training")
        self._stop_btn.setObjectName("dangerButton")
        self._stop_btn.setFixedHeight(40)
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._stop_training)
        train_btn_row.addWidget(self._stop_btn)
        step1_layout.addLayout(train_btn_row)

        self._train_status_label = QLabel("")
        self._train_status_label.setObjectName("warningLabel")
        self._train_status_label.setWordWrap(True)
        step1_layout.addWidget(self._train_status_label)

        # Save best.pt row
        self._save_model_btn = QPushButton("💾  Save best.pt to Computer…")
        self._save_model_btn.setFixedHeight(34)
        self._save_model_btn.setEnabled(False)
        self._save_model_btn.clicked.connect(self._save_model_locally)
        step1_layout.addWidget(self._save_model_btn)

        auto_layout.addWidget(step1_group)

        # ── Step 2: Auto-annotate batch ───────────────────────────────────────
        step2_group = QGroupBox("Step 2 — Auto-Annotate Remaining Images")
        step2_layout = QVBoxLayout(step2_group)
        step2_layout.setSpacing(10)
        step2_desc = QLabel(
            "Uses the latest trained model to auto-annotate all unannotated images. "
            "Auto-annotations are marked for your review before they are used in the next training round."
        )
        step2_desc.setWordWrap(True)
        step2_desc.setObjectName("mutedLabel")
        step2_layout.addWidget(step2_desc)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence threshold:"))
        self._conf_spin = QSpinBox()
        self._conf_spin.setRange(10, 90)
        self._conf_spin.setValue(15)
        self._conf_spin.setSuffix(" %")
        self._conf_spin.setFixedWidth(90)
        conf_row.addWidget(self._conf_spin)
        conf_row.addStretch()
        step2_layout.addLayout(conf_row)

        self._batch_btn = QPushButton("⚡  Auto-Annotate All Unannotated Images")
        self._batch_btn.setObjectName("primaryButton")
        self._batch_btn.setFixedHeight(40)
        self._batch_btn.clicked.connect(self._start_batch_annotate)
        step2_layout.addWidget(self._batch_btn)

        self._batch_status_label = QLabel("")
        self._batch_status_label.setObjectName("successLabel")
        self._batch_status_label.setWordWrap(True)
        step2_layout.addWidget(self._batch_status_label)
        auto_layout.addWidget(step2_group)

        # Log area
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFixedHeight(140)
        self._log_text.setObjectName("logView")
        log_layout.addWidget(self._log_text)
        auto_layout.addWidget(log_group)

        root.addWidget(self._auto_panel)

        # Loop description
        loop_note = QLabel(
            "<b>Loop workflow:</b>  "
            "Upload images → manually annotate ≥50 → mark as reviewed → "
            "<b>Train Model</b> → <b>Auto-Annotate Remaining</b> → "
            "review auto-annotations → repeat for higher accuracy."
        )
        loop_note.setWordWrap(True)
        loop_note.setObjectName("mutedLabel")
        root.addWidget(loop_note)

        root.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        root.addWidget(close_btn)

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def _on_mode_changed(self) -> None:
        auto_mode = self._btn_auto.isChecked()
        self._manual_panel.setVisible(not auto_mode)
        self._auto_panel.setVisible(auto_mode)
        if auto_mode:
            self._refresh_stats()

    def _on_split_changed(self, value: int) -> None:
        self._split_label.setText(f"{value}% train / {100 - value}% val")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _refresh_stats(self) -> None:
        try:
            self._stats = self._api.get_annotation_stats(self._project["id"])
        except APIError as e:
            self._stats = {}
            self._stats_label.setText(f"Could not load stats: {e.detail}")
            return

        s = self._stats
        reviewed = s.get("reviewed_annotation_count", 0)
        can_train = s.get("can_train", False)
        has_model = s.get("has_trained_model", False)
        model_ver = s.get("latest_model_version")

        self._stats_label.setText(
            f"Total images: {s.get('total', 0)}   |   "
            f"Unannotated: {s.get('unannotated', 0)}   |   "
            f"Annotated: {s.get('annotated', 0)}   |   "
            f"Auto (review): {s.get('auto_annotated', 0)}\n"
            f"Reviewed annotations: {reviewed}   |   "
            f"Model: {'v' + str(model_ver) if model_ver else 'none trained yet'}"
        )

        progress = min(reviewed, BOOTSTRAP_THRESHOLD)
        self._progress_bar.setValue(progress)
        if reviewed >= BOOTSTRAP_THRESHOLD:
            self._progress_bar.setFormat(f"{reviewed} reviewed ✓  (ready to train)")
        else:
            remaining = BOOTSTRAP_THRESHOLD - reviewed
            self._progress_bar.setFormat(
                f"{reviewed} / {BOOTSTRAP_THRESHOLD} reviewed  (need {remaining} more)"
            )

        self._train_btn.setEnabled(can_train and not self._training_active)
        self._batch_btn.setEnabled(has_model)
        self._save_model_btn.setEnabled(has_model)

        if self._training_active:
            pass  # leave the live status from polling in place
        elif not can_train:
            self._train_status_label.setText(
                f"Annotate and review at least {BOOTSTRAP_THRESHOLD} images before training."
            )
        else:
            self._train_status_label.setText("Enough data to train — click 'Train Model' to start.")

        if not has_model:
            self._batch_status_label.setText("No trained model yet. Train first.")
        else:
            self._batch_status_label.setText(f"Model v{model_ver} ready. Click to auto-annotate.")

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
        # The training controls live in the auto-loop panel, which is
        # hidden by default (manual mode) — switch to it so the resumed
        # progress is actually visible instead of silently running behind
        # the manual-mode screen.
        self._btn_auto.setChecked(True)
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

        epochs = self._epochs_spin.value()
        train_split = self._split_slider.value() / 100.0

        self._train_thread = QThread()
        self._train_worker = TrainWorker(self._api, self._project["id"], epochs, train_split)
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
        self._train_status_label.setText(
            f"Training job submitted (status: {status}). Polling for completion…"
        )
        self._log_text.append("Training job started…")
        self._poll_timer.start(5000)  # poll every 5 seconds

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
                    "Training complete! You can now auto-annotate, or save best.pt to your computer."
                )
            elif status == "cancelled":
                self._train_status_label.setText(
                    "Training stopped. A checkpoint from the last completed epoch "
                    "was saved, if any — see log below."
                )
            else:
                self._train_status_label.setText("Training failed — see log below.")

            self._log_text.setText(log)
            self._log_text.verticalScrollBar().setValue(self._log_text.verticalScrollBar().maximum())
            self._refresh_stats()
        else:
            self._train_status_label.setText(f"Training {status}…")
            if log:
                self._log_text.setText(log)
                self._log_text.verticalScrollBar().setValue(self._log_text.verticalScrollBar().maximum())

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

    # ── Batch auto-annotate ───────────────────────────────────────────────────

    def _start_batch_annotate(self) -> None:
        self._batch_btn.setEnabled(False)
        self._batch_btn.setText("⏳  Auto-annotating…")
        self._batch_status_label.setText("Running inference on unannotated images…")

        confidence = self._conf_spin.value() / 100.0

        self._batch_thread = QThread()
        self._batch_worker = BatchAnnotateWorker(self._api, self._project["id"], confidence)
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.finished.connect(self._on_batch_done)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.finished.connect(self._batch_thread.quit)
        self._batch_worker.error.connect(self._batch_thread.quit)
        self._batch_thread.start()

    def _on_batch_done(self, result: dict) -> None:
        self._batch_btn.setText("⚡  Auto-Annotate All Unannotated Images")
        self._batch_btn.setEnabled(True)
        msg = result.get("message", "Done.")
        processed = result.get("processed", 0)
        total_anns = result.get("total_annotations", 0)
        self._batch_status_label.setText(
            f"{msg}\n"
            "Review the auto-annotations, correct any errors, mark as reviewed, "
            "then train again for higher accuracy."
        )
        self._log_text.append(
            f"Batch auto-annotation complete: {processed} images processed, "
            f"{total_anns} annotations created."
        )
        self._refresh_stats()

    def _on_batch_error(self, msg: str) -> None:
        self._batch_btn.setText("⚡  Auto-Annotate All Unannotated Images")
        self._batch_btn.setEnabled(True)
        self._batch_status_label.setText(f"Error: {msg}")

    def closeEvent(self, event) -> None:
        self._poll_timer.stop()
        super().closeEvent(event)
