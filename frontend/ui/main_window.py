"""
Main application window.
Hosts the project dashboard and annotation canvas, handles navigation.
"""
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from frontend.api_client import APIClient, APIError
from frontend.ui.annotation_canvas import AnnotationCanvas, MODE_BBOX, MODE_DRAG, MODE_POLYGON
from frontend.ui.project_dashboard import ProjectDashboard


class AnnotationView(QWidget):
    """Full annotation workspace — canvas + side panels."""

    def __init__(self, api: APIClient, parent=None) -> None:
        super().__init__(parent)
        self._api = api
        self._project_id = None
        self._image_id = None
        self._images: list[dict] = []
        self._current_idx = 0
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # ── Left: tool panel ──────────────────────────────────────────────────
        left = QFrame()
        left.setObjectName("leftPanel")
        left.setFixedWidth(56)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 8, 4, 8)
        left_layout.setSpacing(4)

        self._tool_btns = {}
        for icon, mode, tip in [
            ("✋", MODE_DRAG, "Select/Pan (Esc)"),
            ("▭", MODE_BBOX, "Bounding Box (B)"),
            ("⬡", MODE_POLYGON, "Polygon (P)"),
        ]:
            btn = QPushButton(icon)
            btn.setObjectName("toolButton")
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setFixedSize(44, 44)
            btn.clicked.connect(lambda _, m=mode: self._set_tool(m))
            self._tool_btns[mode] = btn
            left_layout.addWidget(btn)

        left_layout.addStretch()

        # Zoom controls
        for icon, action in [("🔍+", "in"), ("🔍-", "out"), ("⊡", "fit")]:
            btn = QPushButton(icon)
            btn.setObjectName("toolButton")
            btn.setFixedSize(44, 32)
            btn.clicked.connect(lambda _, a=action: self._zoom(a))
            left_layout.addWidget(btn)

        splitter.addWidget(left)

        # ── Center: canvas ────────────────────────────────────────────────────
        self.canvas = AnnotationCanvas(self._api)
        self.canvas.annotation_changed.connect(self._on_annotation_changed)
        self.canvas.status_message.connect(self._show_status)
        self.canvas.scene().selectionChanged.connect(self._on_canvas_selection_changed)
        splitter.addWidget(self.canvas)

        # ── Right: layers + classes ────────────────────────────────────────────
        right = QFrame()
        right.setObjectName("rightPanel")
        right.setFixedWidth(220)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        # Image navigation
        nav_layout = QHBoxLayout()
        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(36)
        self._prev_btn.clicked.connect(self._prev_image)
        nav_layout.addWidget(self._prev_btn)

        self._img_label = QLabel("0 / 0")
        self._img_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self._img_label)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(36)
        self._next_btn.clicked.connect(self._next_image)
        nav_layout.addWidget(self._next_btn)
        right_layout.addLayout(nav_layout)

        # Class selector
        right_layout.addWidget(QLabel("Active Class"))
        self._class_combo = QComboBox()
        self._class_combo.addItem("(no class)", None)
        self._class_combo.currentIndexChanged.connect(self._on_class_changed)
        right_layout.addWidget(self._class_combo)

        # Layers (annotations list)
        right_layout.addWidget(QLabel("Annotations"))
        self._layers_list = QListWidget()
        self._layers_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._layers_list.itemSelectionChanged.connect(self._on_layers_selection_changed)
        right_layout.addWidget(self._layers_list)

        review_btn = QPushButton("✓ Mark Reviewed")
        review_btn.setToolTip("Mark selected auto-annotations reviewed without editing them")
        review_btn.clicked.connect(self.canvas.mark_selected_reviewed)
        right_layout.addWidget(review_btn)

        del_ann_btn = QPushButton("🗑 Delete Selected")
        del_ann_btn.setObjectName("dangerButton")
        del_ann_btn.clicked.connect(self.canvas.delete_selected)
        right_layout.addWidget(del_ann_btn)

        # Status
        self._status_label = QLabel("Ready")
        self._status_label.setObjectName("mutedLabel")
        self._status_label.setWordWrap(True)
        right_layout.addWidget(self._status_label)

        splitter.addWidget(right)
        splitter.setSizes([56, 800, 220])

        self._set_tool(MODE_DRAG)

    # ── Data loading ──────────────────────────────────────────────────────────

    def load(self, project_id: int, image_id: int) -> None:
        self._project_id = project_id
        # Fetch image list
        try:
            self._images = self._api.list_images(project_id)
        except APIError:
            self._images = []
        self._current_idx = next(
            (i for i, img in enumerate(self._images) if img["id"] == image_id), 0
        )
        # Load classes
        try:
            self.canvas.classes = self._api.list_classes(project_id)
        except APIError:
            self.canvas.classes = []
        self._refresh_class_combo()
        self._load_current_image()

    def _load_current_image(self) -> None:
        if not self._images:
            return
        img = self._images[self._current_idx]
        self._image_id = img["id"]
        self._img_label.setText(f"{self._current_idx + 1} / {len(self._images)}")
        self.canvas.load_image(self._project_id, self._image_id)
        self._refresh_layers()

    def _refresh_class_combo(self) -> None:
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItem("(no class)", None)
        for cls in self.canvas.classes:
            self._class_combo.addItem(cls["name"], cls)
        self._class_combo.blockSignals(False)

    def _refresh_layers(self) -> None:
        # Blocked because clear()/addItem() fire itemSelectionChanged, which
        # would otherwise cascade into _on_layers_selection_changed and wipe
        # the canvas selection this list is just trying to mirror.
        self._layers_list.blockSignals(True)
        self._layers_list.clear()
        for ann in self.canvas.annotations:
            item = QListWidgetItem()
            cls_name = ann.class_info["name"] if ann.class_info else "Unlabeled"
            source = getattr(ann, "source", "manual")
            tag = source
            if source == "auto":
                tag = "auto ✓" if getattr(ann, "reviewed", False) else "auto ⚠ needs review"
            item.setText(f"[{ann.shape_type.upper()}] {cls_name} ({tag})")
            item.setData(Qt.UserRole, ann)
            self._layers_list.addItem(item)
            item.setSelected(ann.isSelected())
        self._layers_list.blockSignals(False)

    # ── Navigation ─────────────────────────────────────────────────────────────

    def _prev_image(self) -> None:
        if self._current_idx > 0:
            self.canvas.save_and_unload()
            self._current_idx -= 1
            self._load_current_image()

    def _next_image(self) -> None:
        if self._current_idx < len(self._images) - 1:
            self.canvas.save_and_unload()
            self._current_idx += 1
            self._load_current_image()

    # ── Tools ─────────────────────────────────────────────────────────────────

    def _set_tool(self, mode: str) -> None:
        for m, btn in self._tool_btns.items():
            btn.setChecked(m == mode)
        self.canvas.set_mode(mode)

    def _zoom(self, action: str) -> None:
        if action == "in":
            self.canvas.scale(1.2, 1.2)
        elif action == "out":
            self.canvas.scale(1 / 1.2, 1 / 1.2)
        elif action == "fit":
            self.canvas.fit_to_view()

    # ── Signals ───────────────────────────────────────────────────────────────

    def _on_annotation_changed(self) -> None:
        self._refresh_layers()

    def _on_class_changed(self, idx: int) -> None:
        cls = self._class_combo.itemData(idx)
        self.canvas.set_current_class(cls)

    def _on_layers_selection_changed(self) -> None:
        """Layers list → canvas: selecting rows (with Ctrl/Shift) selects the
        matching shapes on the canvas."""
        selected = {item.data(Qt.UserRole) for item in self._layers_list.selectedItems()}
        scene = self.canvas.scene()
        scene.blockSignals(True)
        for ann in self.canvas.annotations:
            ann.setSelected(ann in selected)
        scene.blockSignals(False)

    def _on_canvas_selection_changed(self) -> None:
        """Canvas → layers list: Ctrl/Shift-selecting shapes on the canvas
        highlights the matching rows in the layers list."""
        selected = set(self.canvas.scene().selectedItems())
        self._layers_list.blockSignals(True)
        for i in range(self._layers_list.count()):
            item = self._layers_list.item(i)
            item.setSelected(item.data(Qt.UserRole) in selected)
        self._layers_list.blockSignals(False)

    def _show_status(self, msg: str) -> None:
        self._status_label.setText(msg)


class MainWindow(QMainWindow):
    def __init__(self, api: APIClient, user: dict) -> None:
        super().__init__()
        self._api = api
        self._user = user
        self._current_theme = "dark"
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("InSiSo Model Bench")
        self.setMinimumSize(1280, 800)

        # ── Menu bar ──────────────────────────────────────────────────────────
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction(self._make_action("Back to Dashboard", self._go_dashboard, "Ctrl+D"))
        if self._user.get("role") == "admin":
            file_menu.addAction(self._make_action("Admin Dashboard", self._open_admin_dashboard))
        file_menu.addSeparator()
        file_menu.addAction(self._make_action("Logout", self._logout))
        file_menu.addAction(self._make_action("Exit", self.close, "Ctrl+Q"))

        view_menu = menubar.addMenu("View")
        view_menu.addAction(self._make_action("Toggle Theme", self._toggle_theme, "Ctrl+T"))
        view_menu.addAction(self._make_action("Fit Image", self._fit_image, "Ctrl+F"))

        annotation_menu = menubar.addMenu("Annotation")
        annotation_menu.addAction(self._make_action("BBox Tool", lambda: self._set_tool("bbox"), "B"))
        annotation_menu.addAction(self._make_action("Polygon Tool", lambda: self._set_tool("polygon"), "P"))
        annotation_menu.addAction(self._make_action("Select/Pan", lambda: self._set_tool("drag"), "Esc"))
        annotation_menu.addSeparator()
        annotation_menu.addAction(self._make_action("Undo", self._undo, "Ctrl+Z"))
        annotation_menu.addAction(self._make_action("Redo", self._redo, "Ctrl+Y"))
        annotation_menu.addSeparator()
        annotation_menu.addAction(self._make_action("Delete Selected", self._delete_selected, "Del"))
        annotation_menu.addAction(self._make_action("Save", self._save, "Ctrl+S"))

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QToolBar("Main toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self._dash_btn = QPushButton("📁 Dashboard")
        self._dash_btn.clicked.connect(self._go_dashboard)
        toolbar.addWidget(self._dash_btn)
        toolbar.addSeparator()

        self._user_label = QLabel(f"  👤 {self._user.get('username', '')}  [{self._user.get('role', '')}]")
        toolbar.addWidget(self._user_label)
        toolbar.addSeparator()

        theme_btn = QPushButton("🌙 Dark")
        theme_btn.setCheckable(True)
        theme_btn.setChecked(True)
        theme_btn.clicked.connect(self._toggle_theme)
        self._theme_btn = theme_btn
        toolbar.addWidget(theme_btn)

        # ── Stacked widget ────────────────────────────────────────────────────
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._dashboard = ProjectDashboard(self._api, self._user)
        self._dashboard.open_annotation.connect(self._open_annotation)
        self._stack.addWidget(self._dashboard)

        self._annotation_view = AnnotationView(self._api)
        self._stack.addWidget(self._annotation_view)

        # ── Status bar ────────────────────────────────────────────────────────
        self.statusBar().showMessage("Ready")

    def _make_action(self, text: str, slot, shortcut: str | None = None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        return action

    def _go_dashboard(self) -> None:
        if self._stack.currentWidget() == self._annotation_view:
            self._annotation_view.canvas.save_and_unload()
        self._dashboard.refresh()
        self._stack.setCurrentWidget(self._dashboard)
        self.statusBar().showMessage("Dashboard")

    def _open_admin_dashboard(self) -> None:
        from frontend.ui.admin_dashboard import AdminDashboardDialog
        dlg = AdminDashboardDialog(self._api, self._user, self)
        dlg.exec()

    def _open_annotation(self, project_id: int, image_id: int) -> None:
        self._annotation_view.load(project_id, image_id)
        self._stack.setCurrentWidget(self._annotation_view)

    def _set_tool(self, mode: str) -> None:
        if self._stack.currentWidget() == self._annotation_view:
            self._annotation_view._set_tool(mode)

    def _undo(self) -> None:
        if self._stack.currentWidget() == self._annotation_view:
            canvas = self._annotation_view.canvas
            desc = canvas.undo_redo.undo()
            if desc:
                canvas.status_message.emit(f"Undo: {desc}")
                canvas._dirty = True

    def _redo(self) -> None:
        if self._stack.currentWidget() == self._annotation_view:
            canvas = self._annotation_view.canvas
            desc = canvas.undo_redo.redo()
            if desc:
                canvas.status_message.emit(f"Redo: {desc}")
                canvas._dirty = True

    def _delete_selected(self) -> None:
        if self._stack.currentWidget() == self._annotation_view:
            self._annotation_view.canvas.delete_selected()

    def _save(self) -> None:
        if self._stack.currentWidget() == self._annotation_view:
            self._annotation_view.canvas._save_annotations()

    def _fit_image(self) -> None:
        if self._stack.currentWidget() == self._annotation_view:
            self._annotation_view.canvas.fit_to_view()

    def _toggle_theme(self) -> None:
        from frontend.main import _apply_theme
        if self._current_theme == "dark":
            self._current_theme = "light"
            self._theme_btn.setText("☀ Light")
        else:
            self._current_theme = "dark"
            self._theme_btn.setText("🌙 Dark")
        _apply_theme(self._current_theme)

    def _logout(self) -> None:
        self._api.logout()
        self.close()

    def closeEvent(self, event) -> None:
        if self._stack.currentWidget() == self._annotation_view:
            self._annotation_view.canvas.save_and_unload()
        event.accept()
