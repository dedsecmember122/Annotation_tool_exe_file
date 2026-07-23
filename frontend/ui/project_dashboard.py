"""
Project Dashboard widget — shows project list and thumbnail gallery.
"""
import io
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal, QThread, QObject
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QProgressBar,
    QComboBox,
    QAbstractItemView,
)

from frontend.api_client import APIClient, APIError

STATUS_COLORS = {
    "unannotated": "#607D8B",
    "in_progress": "#FFA726",
    "annotated": "#43A047",
    "auto_annotated": "#FF8C00",
}

STATUS_LABELS = {
    "unannotated": "Unannotated",
    "in_progress": "In Progress",
    "annotated": "Annotated",
    "auto_annotated": "Auto (Review)",
}


class UploadWorker(QObject):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, api: APIClient, project_id: int, paths: list[str]) -> None:
        super().__init__()
        self._api = api
        self._project_id = project_id
        self._paths = paths

    def run(self) -> None:
        try:
            result = self._api.upload_images(self._project_id, self._paths)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ThumbnailWorker(QObject):
    """Fetches pre-resized thumbnails on a background thread, one at a time,
    emitting each as it arrives so the gallery fills in progressively instead
    of blocking the UI thread until every image is done."""

    thumbnail_ready = Signal(int, bytes)  # image_id, jpeg bytes
    finished = Signal()

    def __init__(self, api: APIClient, image_ids: list[int]) -> None:
        super().__init__()
        self._api = api
        self._image_ids = image_ids
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        for image_id in self._image_ids:
            if self._stop:
                break
            try:
                data = self._api.get_image_thumbnail(image_id)
            except Exception:
                continue
            if self._stop:
                break
            self.thumbnail_ready.emit(image_id, data)
        self.finished.emit()


class ProjectDashboard(QWidget):
    open_annotation = Signal(int, int)  # project_id, image_id

    def __init__(self, api: APIClient, user: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._user = user
        self._current_project: dict | None = None
        self._images: list[dict] = []
        self._gallery_items: dict[int, QListWidgetItem] = {}
        self._thumb_thread: QThread | None = None
        self._thumb_worker: ThumbnailWorker | None = None
        self._setup_ui()
        self._load_projects()

    def _setup_ui(self) -> None:
        main = QHBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        main.addWidget(splitter)

        # ── Left: project list ────────────────────────────────────────────────
        left = QFrame()
        left.setObjectName("leftPanel")
        left.setFixedWidth(250)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        proj_header = QHBoxLayout()
        proj_title = QLabel("Projects")
        proj_title.setStyleSheet("font-size: 15px; font-weight: bold;")
        proj_header.addWidget(proj_title)
        new_proj_btn = QPushButton("+ New")
        new_proj_btn.setFixedHeight(28)
        new_proj_btn.clicked.connect(self._create_project)
        proj_header.addWidget(new_proj_btn)
        left_layout.addLayout(proj_header)

        self._project_list = QListWidget()
        self._project_list.currentItemChanged.connect(self._on_project_selected)
        left_layout.addWidget(self._project_list)

        del_btn = QPushButton("Delete Project")
        del_btn.setObjectName("dangerButton")
        del_btn.clicked.connect(self._delete_project)
        left_layout.addWidget(del_btn)

        splitter.addWidget(left)

        # ── Right: image gallery + toolbar ────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        # Toolbar row 1
        toolbar = QHBoxLayout()
        self._proj_name_label = QLabel("Select a project")
        self._proj_name_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        toolbar.addWidget(self._proj_name_label)
        toolbar.addStretch()

        self._status_filter = QComboBox()
        self._status_filter.addItem("All statuses", None)
        for k, v in STATUS_LABELS.items():
            self._status_filter.addItem(v, k)
        self._status_filter.currentIndexChanged.connect(self._apply_filter)
        toolbar.addWidget(self._status_filter)

        upload_btn = QPushButton("⬆ Upload Images")
        upload_btn.setObjectName("primaryButton")
        upload_btn.clicked.connect(self._upload_images)
        toolbar.addWidget(upload_btn)

        manage_classes_btn = QPushButton("🏷 Classes")
        manage_classes_btn.clicked.connect(self._manage_classes)
        toolbar.addWidget(manage_classes_btn)

        export_btn = QPushButton("⬇ Export")
        export_btn.clicked.connect(self._export_project)
        toolbar.addWidget(export_btn)

        right_layout.addLayout(toolbar)

        # Toolbar row 2 — image management
        img_toolbar = QHBoxLayout()
        img_toolbar.addStretch()

        self._del_images_btn = QPushButton("🗑 Delete Selected Image(s)")
        self._del_images_btn.setObjectName("dangerButton")
        self._del_images_btn.clicked.connect(self._delete_selected_images)
        img_toolbar.addWidget(self._del_images_btn)

        self._auto_annotate_btn = QPushButton("🤖 Auto-Annotate")
        self._auto_annotate_btn.setObjectName("primaryButton")
        self._auto_annotate_btn.clicked.connect(self._open_auto_annotate)
        img_toolbar.addWidget(self._auto_annotate_btn)

        self._train_model_btn = QPushButton("🎯 Train Model")
        self._train_model_btn.setObjectName("primaryButton")
        self._train_model_btn.clicked.connect(self._open_train_model)
        img_toolbar.addWidget(self._train_model_btn)

        right_layout.addLayout(img_toolbar)

        # Stats bar
        self._stats_label = QLabel("")
        self._stats_label.setObjectName("mutedLabel")
        right_layout.addWidget(self._stats_label)

        # Upload progress
        self._upload_progress = QProgressBar()
        self._upload_progress.setVisible(False)
        self._upload_progress.setRange(0, 0)  # indeterminate
        right_layout.addWidget(self._upload_progress)

        # Gallery — multi-selection enabled
        self._gallery = QListWidget()
        self._gallery.setViewMode(QListWidget.IconMode)
        self._gallery.setIconSize(QSize(160, 120))
        self._gallery.setSpacing(8)
        self._gallery.setResizeMode(QListWidget.Adjust)
        self._gallery.setMovement(QListWidget.Static)
        self._gallery.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._gallery.itemDoubleClicked.connect(self._open_image)
        right_layout.addWidget(self._gallery)

        # Selection hint
        hint = QLabel("Tip: Click to select · Ctrl+Click / Shift+Click for multi-select · Double-click to annotate")
        hint.setObjectName("mutedLabel")
        right_layout.addWidget(hint)

        splitter.addWidget(right)
        splitter.setSizes([250, 900])

    # ── Projects ─────────────────────────────────────────────────────────────

    def _load_projects(self) -> None:
        try:
            projects = self._api.list_projects()
            self._project_list.clear()
            for p in projects:
                item = QListWidgetItem(p["name"])
                item.setData(Qt.UserRole, p)
                self._project_list.addItem(item)
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _create_project(self) -> None:
        name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if not ok or not name.strip():
            return
        try:
            p = self._api.create_project(name.strip())
            item = QListWidgetItem(p["name"])
            item.setData(Qt.UserRole, p)
            self._project_list.addItem(item)
            self._project_list.setCurrentItem(item)
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _delete_project(self) -> None:
        item = self._project_list.currentItem()
        if not item:
            return
        p = item.data(Qt.UserRole)
        reply = QMessageBox.question(self, "Delete Project",
                                     f"Delete project '{p['name']}' and all its data?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                self._api.delete_project(p["id"])
                self._project_list.takeItem(self._project_list.row(item))
                self._gallery.clear()
                self._proj_name_label.setText("Select a project")
                self._current_project = None
            except APIError as e:
                QMessageBox.critical(self, "Error", str(e))

    def _on_project_selected(self, current: QListWidgetItem, _: QListWidgetItem) -> None:
        if not current:
            return
        self._current_project = current.data(Qt.UserRole)
        self._proj_name_label.setText(self._current_project["name"])
        self._load_images()

    # ── Images ────────────────────────────────────────────────────────────────

    def _load_images(self, status_filter: str | None = None) -> None:
        if not self._current_project:
            return
        try:
            self._images = self._api.list_images(
                self._current_project["id"],
                status=status_filter,
            )
            self._refresh_gallery()
            self._refresh_stats()
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _apply_filter(self, index: int) -> None:
        status = self._status_filter.itemData(index)
        self._load_images(status_filter=status)

    def _refresh_gallery(self) -> None:
        # Previously this fetched the full-resolution original for every
        # image, synchronously on the UI thread, just to build a 160x120
        # icon — and this method reruns on every project open, filter
        # change, and return-to-dashboard. That's what made the app feel
        # slow/heavy: it blocked the UI and decoded full-size images into
        # memory over and over. Now it builds the (fast, metadata-only)
        # list immediately and fills in thumbnails asynchronously via the
        # server-resized /thumbnail endpoint.
        self._stop_thumbnail_loading()

        self._gallery.clear()
        self._gallery_items = {}
        for img in self._images:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, img)

            status = img.get("status", "unannotated")
            count = img.get("annotation_count", 0)
            label = f"{Path(img['filename']).stem}\n{STATUS_LABELS.get(status, status)} ({count})"
            item.setText(label)
            item.setTextAlignment(Qt.AlignCenter)

            color = STATUS_COLORS.get(status, "#607D8B")
            item.setForeground(QColor(color))

            self._gallery.addItem(item)
            self._gallery_items[img["id"]] = item

        if self._images:
            self._start_thumbnail_loading([img["id"] for img in self._images])

    def _stop_thumbnail_loading(self) -> None:
        if self._thumb_worker is not None:
            self._thumb_worker.stop()
        self._thumb_thread = None
        self._thumb_worker = None

    def _start_thumbnail_loading(self, image_ids: list[int]) -> None:
        thread = QThread()
        worker = ThumbnailWorker(self._api, image_ids)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thumb_thread = thread
        self._thumb_worker = worker
        thread.start()

    def _on_thumbnail_ready(self, image_id: int, data: bytes) -> None:
        item = self._gallery_items.get(image_id)
        if item is None:
            return
        pixmap = QPixmap()
        if pixmap.loadFromData(data) and not pixmap.isNull():
            # The server thumbnail is capped at 240px on its longest side
            # (see images.py's /thumbnail endpoint) but the gallery's icon
            # box is a fixed 160x120 - without scaling down to match, a
            # thumbnail wider or taller than that box overflows past its
            # grid cell into neighboring items, making the gallery look like
            # photos are overlapping/merging together.
            scaled = pixmap.scaled(self._gallery.iconSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item.setIcon(QIcon(scaled))

    def _refresh_stats(self) -> None:
        total = len(self._images)
        counts = {}
        for img in self._images:
            s = img.get("status", "unannotated")
            counts[s] = counts.get(s, 0) + 1
        parts = [f"Total: {total}"]
        for k, v in counts.items():
            parts.append(f"{STATUS_LABELS.get(k, k)}: {v}")
        self._stats_label.setText("  |  ".join(parts))

    def _upload_images(self) -> None:
        if not self._current_project:
            QMessageBox.warning(self, "No project", "Select a project first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select images", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tiff *.webp)"
        )
        if not paths:
            return

        self._upload_progress.setVisible(True)

        self._thread = QThread()
        self._worker = UploadWorker(self._api, self._current_project["id"], paths)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_upload_done)
        self._worker.error.connect(self._on_upload_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_upload_done(self, images: list) -> None:
        self._upload_progress.setVisible(False)
        self._load_images()

    def _on_upload_error(self, msg: str) -> None:
        self._upload_progress.setVisible(False)
        QMessageBox.critical(self, "Upload failed", msg)

    def _delete_selected_images(self) -> None:
        if not self._current_project:
            QMessageBox.warning(self, "No project", "Select a project first.")
            return
        selected = self._gallery.selectedItems()
        if not selected:
            QMessageBox.information(self, "No selection", "Select one or more images first.")
            return

        count = len(selected)
        reply = QMessageBox.question(
            self, "Delete Images",
            f"Permanently delete {count} selected image(s) and all their annotations?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        errors = []
        for item in selected:
            img = item.data(Qt.UserRole)
            try:
                self._api.delete_image(img["id"])
            except APIError as e:
                errors.append(f"{img['filename']}: {e.detail}")

        if errors:
            QMessageBox.warning(self, "Some deletions failed", "\n".join(errors))

        self._load_images()

    def _open_image(self, item: QListWidgetItem) -> None:
        img = item.data(Qt.UserRole)
        if self._current_project:
            self.open_annotation.emit(self._current_project["id"], img["id"])

    # ── Classes ───────────────────────────────────────────────────────────────

    def _manage_classes(self) -> None:
        if not self._current_project:
            QMessageBox.warning(self, "No project", "Select a project first.")
            return
        from frontend.ui.class_manager_dialog import ClassManagerDialog
        dlg = ClassManagerDialog(self._api, self._current_project["id"], self)
        dlg.exec()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_project(self) -> None:
        if not self._current_project:
            QMessageBox.warning(self, "No project", "Select a project first.")
            return
        from frontend.ui.export_dialog import ExportDialog
        dlg = ExportDialog(self._api, self._current_project["id"], self)
        dlg.exec()

    # ── Auto-Annotate ─────────────────────────────────────────────────────────

    def _open_auto_annotate(self) -> None:
        if not self._current_project:
            QMessageBox.warning(self, "No project", "Select a project first.")
            return
        from frontend.ui.auto_annotate_dialog import AutoAnnotateDialog
        dlg = AutoAnnotateDialog(self._api, self._current_project, self)
        dlg.exec()
        self._load_images()

    # ── Train Model ───────────────────────────────────────────────────────────

    def _open_train_model(self) -> None:
        if not self._current_project:
            QMessageBox.warning(self, "No project", "Select a project first.")
            return
        from frontend.ui.train_model_dialog import TrainModelDialog
        dlg = TrainModelDialog(self._api, self._current_project, self)
        dlg.exec()
        self._load_images()

    def refresh(self) -> None:
        self._load_images()
