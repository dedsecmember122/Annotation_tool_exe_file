"""
Annotation Canvas — the main QGraphicsView for drawing and editing annotations.
"""
import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    QPointF,
    QRectF,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QMessageBox,
)

from frontend.api_client import APIClient, APIError
from frontend.tools.bbox_tool import BBoxItem, BBoxTool
from frontend.tools.drag_tool import DragTool
from frontend.tools.polygon_tool import PolygonItem, PolygonTool
from frontend.tools.undo_redo_manager import (
    AddAnnotationCommand,
    DeleteAnnotationCommand,
    UndoRedoManager,
)

MODE_DRAG = "drag"
MODE_BBOX = "bbox"
MODE_POLYGON = "polygon"


class AnnotationCanvas(QGraphicsView):
    """
    Zoomable / pannable canvas built on QGraphicsView.
    Supports bbox drawing, polygon drawing, selection, undo/redo, autosave.
    """

    annotation_changed = Signal()  # emitted when anything changes
    status_message = Signal(str)

    def __init__(self, api: APIClient, parent=None) -> None:
        super().__init__(parent)
        self._api = api
        self._project_id: Optional[int] = None
        self._image_id: Optional[int] = None
        self._image_data: Optional[dict] = None
        self.annotations: list = []  # list of BBoxItem | PolygonItem
        self.current_class: Optional[dict] = None
        self.classes: list[dict] = []

        self.undo_redo = UndoRedoManager()
        self._mode = MODE_DRAG
        self._bbox_tool = BBoxTool(self)
        self._polygon_tool = PolygonTool(self)
        self._drag_tool = DragTool(self)
        self._multiselect_drag = False

        self._autosave_timer = QTimer()
        self._autosave_timer.setInterval(30_000)  # 30s
        self._autosave_timer.timeout.connect(self._save_annotations)
        self._dirty = False

        self._setup_view()

    def _setup_view(self) -> None:
        scene = QGraphicsScene(self)
        scene.setBackgroundBrush(QColor("#111122"))
        self.setScene(scene)
        self.setRenderHints(
            self.renderHints() |
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_image(self, project_id: int, image_id: int) -> None:
        """Load an image and its annotations from the backend."""
        if self._dirty:
            self._save_annotations()

        self._project_id = project_id
        self._image_id = image_id
        self.undo_redo.clear()
        # Deselect before destroying items: QGraphicsScene.clear() deletes
        # items in C++ immediately, and if one is still selected, Qt emits
        # selectionChanged() reentrantly from inside that destructor loop.
        # Listeners (e.g. the layers panel) that resolve the old selection
        # back to a still-being-destroyed item can then crash the process.
        # Clearing selection first lets that signal fire while everything
        # is still alive.
        self.scene().clearSelection()
        self.annotations.clear()
        self.scene().clear()
        self._dirty = False

        # Fetch image bytes
        try:
            data = self._api.get_image_data(image_id)
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            pix_item = QGraphicsPixmapItem(pixmap)
            pix_item.setZValue(-1)
            self.scene().addItem(pix_item)
            self.scene().setSceneRect(QRectF(pixmap.rect()))
            self.fitInView(pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        except APIError as e:
            self.status_message.emit(f"Error loading image: {e}")
            return

        # Load existing annotations
        try:
            anns = self._api.get_annotations(image_id)
            for ann in anns:
                self._add_annotation_from_data(ann)
        except APIError:
            pass

        self._autosave_timer.start()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if mode == MODE_DRAG:
            self._drag_tool.activate()
        else:
            self._drag_tool.deactivate()
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def set_current_class(self, class_info: Optional[dict]) -> None:
        self.current_class = class_info

    def current_class_color(self) -> QColor:
        if self.current_class:
            return QColor(self.current_class.get("color_hex", "#4FC3F7"))
        return QColor("#4FC3F7")

    def delete_selected(self) -> None:
        for item in self.scene().selectedItems():
            if item in self.annotations:
                self.undo_redo.execute(DeleteAnnotationCommand(self, item))
        self._dirty = True
        self.annotation_changed.emit()

    def mark_selected_reviewed(self) -> None:
        """Mark selected auto-annotations reviewed without editing them —
        for the case where the auto box/polygon is already correct.
        item.mark_reviewed() schedules the save and emits annotation_changed."""
        for item in self.scene().selectedItems():
            if item in self.annotations and getattr(item, "source", "manual") == "auto":
                item.mark_reviewed()

    # ── Internal: build annotation items from API data ────────────────────────

    def _add_annotation_from_data(self, ann: dict) -> None:
        cls_info = next((c for c in self.classes if c["id"] == ann.get("class_id")), None)
        coords = ann.get("coordinates", {})

        if ann["shape_type"] == "bbox":
            rect = QRectF(
                coords.get("x1", 0), coords.get("y1", 0),
                coords.get("x2", 0) - coords.get("x1", 0),
                coords.get("y2", 0) - coords.get("y1", 0),
            )
            item = BBoxItem(rect, class_info=cls_info,
                            annotation_id=ann["id"], source=ann.get("source", "manual"),
                            reviewed=ann.get("reviewed", False))
        else:
            pts = [QPointF(p[0], p[1]) for p in (coords if isinstance(coords, list) else [])]
            item = PolygonItem(pts, class_info=cls_info,
                               annotation_id=ann["id"], source=ann.get("source", "manual"),
                               reviewed=ann.get("reviewed", False))

        self.scene().addItem(item)
        self.annotations.append(item)

    # ── Autosave ──────────────────────────────────────────────────────────────

    def _schedule_save(self) -> None:
        self._dirty = True
        self.annotation_changed.emit()

    def _save_annotations(self) -> None:
        if not self._dirty or self._image_id is None:
            return
        try:
            existing = self._api.get_annotations(self._image_id)
            existing_ids = {a["id"] for a in existing}

            # Collect current on-screen items
            current_items = list(self.annotations)

            # Delete removed annotations (those with IDs not in current items)
            current_ann_ids = {
                item.annotation_id for item in current_items
                if hasattr(item, "annotation_id") and item.annotation_id
            }
            for ann_id in existing_ids - current_ann_ids:
                try:
                    self._api.delete_annotation(ann_id)
                except APIError:
                    pass

            # Create or update
            for item in current_items:
                coords = item.to_dict()
                class_id = item.class_info["id"] if item.class_info else None

                if hasattr(item, "annotation_id") and item.annotation_id:
                    self._api.update_annotation(
                        item.annotation_id, coordinates=coords, class_id=class_id,
                        reviewed=getattr(item, "reviewed", False),
                    )
                else:
                    ann = self._api.create_annotation(
                        self._image_id, class_id, item.shape_type, coords,
                        source=getattr(item, "source", "manual"),
                    )
                    item.annotation_id = ann["id"]
                    # The backend auto-marks manual annotations reviewed on
                    # creation — sync that back so the next save (e.g. the
                    # 30s autosave) doesn't stomp it with the item's stale
                    # local default of False.
                    item.reviewed = ann.get("reviewed", False)

            self._dirty = False
            self.status_message.emit("Saved")
        except APIError as e:
            self.status_message.emit(f"Save error: {e}")

    def save_and_unload(self) -> None:
        """Call before navigating away from this image."""
        self._save_annotations()
        self._autosave_timer.stop()

    # ── Mouse events ──────────────────────────────────────────────────────────

    @staticmethod
    def _force_ctrl(event: QMouseEvent) -> QMouseEvent:
        """Qt only extends click/rubber-band selection for Ctrl, not Shift.
        Re-stamp the event with Ctrl added so Shift behaves the same way."""
        return QMouseEvent(
            event.type(), event.position(), event.globalPosition(),
            event.button(), event.buttons(),
            event.modifiers() | Qt.KeyboardModifier.ControlModifier,
        )

    def mousePressEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.pos())
        if self._mode == MODE_BBOX and event.button() == Qt.MouseButton.LeftButton:
            self._bbox_tool.mouse_press(scene_pos)
        elif self._mode == MODE_POLYGON and event.button() == Qt.MouseButton.LeftButton:
            double = event.type().name == "MouseButtonDblClick"
            self._polygon_tool.mouse_press(scene_pos, double=double)
        elif (
            self._mode == MODE_DRAG
            and event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier)
        ):
            # Shift or Ctrl (+drag) adds annotations to the current selection
            # instead of panning/replacing it.
            self._multiselect_drag = True
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            super().mousePressEvent(self._force_ctrl(event))
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.pos())
        if self._mode == MODE_POLYGON:
            self._polygon_tool.mouse_press(scene_pos, double=True)
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.pos())
        if self._mode == MODE_BBOX:
            self._bbox_tool.mouse_move(scene_pos)
        elif self._mode == MODE_POLYGON:
            self._polygon_tool.mouse_move(scene_pos)
        elif self._multiselect_drag:
            super().mouseMoveEvent(self._force_ctrl(event))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        scene_pos = self.mapToScene(event.pos())
        if self._mode == MODE_BBOX:
            self._bbox_tool.mouse_release(scene_pos)
        elif self._multiselect_drag:
            super().mouseReleaseEvent(self._force_ctrl(event))
        else:
            super().mouseReleaseEvent(event)

        if self._multiselect_drag and event.button() == Qt.MouseButton.LeftButton:
            self._multiselect_drag = False
            self.setDragMode(
                QGraphicsView.DragMode.ScrollHandDrag if self._mode == MODE_DRAG
                else QGraphicsView.DragMode.NoDrag
            )

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()

        if event.matches(QKeySequence.StandardKey.Undo):
            desc = self.undo_redo.undo()
            if desc:
                self.status_message.emit(f"Undo: {desc}")
                self._dirty = True
        elif event.matches(QKeySequence.StandardKey.Redo):
            desc = self.undo_redo.redo()
            if desc:
                self.status_message.emit(f"Redo: {desc}")
                self._dirty = True
        elif key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace:
            if self._mode != MODE_POLYGON:
                self.delete_selected()
        elif key == Qt.Key.Key_B:
            self.set_mode(MODE_BBOX)
            self.status_message.emit("BBox tool active")
        elif key == Qt.Key.Key_P:
            self.set_mode(MODE_POLYGON)
            self.status_message.emit("Polygon tool active")
        elif key == Qt.Key.Key_Escape:
            self.set_mode(MODE_DRAG)
            self._polygon_tool._clear_temporaries()
            self.status_message.emit("Select/drag tool active")
        elif self._mode == MODE_POLYGON:
            self._polygon_tool.key_press(key)
        else:
            super().keyPressEvent(event)

    # ── Zoom ──────────────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def fit_to_view(self) -> None:
        if self.scene().items():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
