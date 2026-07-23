"""
Polygon drawing tool.
Click to add vertices, double-click / Enter to close the polygon.
Vertices can be dragged after creation.
"""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPolygonItem


VERTEX_RADIUS = 5


class PolygonItem(QGraphicsPolygonItem):
    """A closed polygon annotation with draggable vertices."""

    def __init__(self, points: list[QPointF], class_info: dict | None = None,
                 annotation_id: int | None = None, source: str = "manual",
                 reviewed: bool = False) -> None:
        polygon = QPolygonF(points)
        super().__init__(polygon)
        self.shape_type = "polygon"
        self.class_info = class_info
        self.annotation_id = annotation_id
        self.source = source
        self.reviewed = reviewed
        self.points = list(points)
        self._move_start_pos: QPointF | None = None

        self.setFlags(
            QGraphicsPolygonItem.ItemIsSelectable |
            QGraphicsPolygonItem.ItemIsMovable |
            QGraphicsPolygonItem.ItemSendsGeometryChanges
        )
        self._update_style()

    def set_class(self, class_info: dict | None) -> None:
        self.class_info = class_info
        self._update_style()

    def _update_style(self) -> None:
        if self.source == "auto" and not self.reviewed:
            color = QColor("#FF8C00")
        elif self.class_info:
            color = QColor(self.class_info.get("color_hex", "#81C784"))
        else:
            color = QColor("#81C784")
        pen = QPen(color, 2, Qt.SolidLine)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(40)
        self.setBrush(fill)

    # ── Move → dirty/reviewed ────────────────────────────────────────────────
    # Mirrors BBoxItem: moving the whole polygon (drag) previously never
    # marked the canvas dirty, so the edit was silently dropped on autosave —
    # same bug class as the bbox one fixed on 2026-07-19, just never hit here
    # because polygons had no resize handles to make editing common.

    def mousePressEvent(self, event) -> None:
        self._move_start_pos = self.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if self._move_start_pos is not None and self.pos() != self._move_start_pos:
            self._mark_edited()
        self._move_start_pos = None

    def _mark_edited(self) -> None:
        """Call whenever the user moves this polygon. Correcting an
        auto-annotation is itself the review step, so it counts as reviewed."""
        if self.source == "auto" and not self.reviewed:
            self.reviewed = True
            self._update_style()
        canvas = self._canvas()
        if canvas is not None:
            canvas._schedule_save()

    def mark_reviewed(self) -> None:
        """Explicitly mark this polygon reviewed without editing it (it was
        already correct as auto-annotated)."""
        if self.reviewed:
            return
        self.reviewed = True
        self._update_style()
        canvas = self._canvas()
        if canvas is not None:
            canvas._schedule_save()

    def _canvas(self):
        scene = self.scene()
        views = scene.views() if scene else []
        return views[0] if views else None

    def to_dict(self) -> list:
        """Returns list of [x, y] scene-coordinate pairs."""
        scene_poly = self.mapToScene(self.polygon())
        return [[p.x(), p.y()] for p in scene_poly]


class PolygonTool:
    """
    Handles mouse events for polygon drawing.
    - Left click: add vertex
    - Double-click or Enter: close polygon
    - Right-click: remove last vertex
    """

    def __init__(self, canvas: "AnnotationCanvas") -> None:
        self._canvas = canvas
        self._points: list[QPointF] = []
        self._preview_item: QGraphicsPolygonItem | None = None
        self._vertex_items: list[QGraphicsEllipseItem] = []

    def _draw_preview(self) -> None:
        if self._preview_item:
            self._canvas.scene().removeItem(self._preview_item)
        if len(self._points) < 2:
            return
        polygon = QPolygonF(self._points)
        self._preview_item = QGraphicsPolygonItem(polygon)
        pen = QPen(QColor("#81C784"), 2, Qt.DashLine)
        self._preview_item.setPen(pen)
        self._canvas.scene().addItem(self._preview_item)

    def _add_vertex_marker(self, pt: QPointF) -> None:
        r = VERTEX_RADIUS
        ellipse = QGraphicsEllipseItem(pt.x() - r, pt.y() - r, r * 2, r * 2)
        ellipse.setBrush(QColor("#81C784"))
        ellipse.setPen(QPen(Qt.NoPen))
        self._canvas.scene().addItem(ellipse)
        self._vertex_items.append(ellipse)

    def _clear_temporaries(self) -> None:
        if self._preview_item:
            self._canvas.scene().removeItem(self._preview_item)
            self._preview_item = None
        for v in self._vertex_items:
            self._canvas.scene().removeItem(v)
        self._vertex_items.clear()
        self._points.clear()

    def mouse_press(self, scene_pos: QPointF, double: bool = False) -> None:
        if double:
            self.close_polygon()
            return
        self._points.append(scene_pos)
        self._add_vertex_marker(scene_pos)
        self._draw_preview()

    def mouse_move(self, scene_pos: QPointF) -> None:
        pass  # Could draw a "rubber band" from last point to cursor

    def mouse_release(self, scene_pos: QPointF) -> None:
        pass

    def close_polygon(self) -> None:
        if len(self._points) < 3:
            self._clear_temporaries()
            return
        item = PolygonItem(self._points, class_info=self._canvas.current_class)
        self._clear_temporaries()
        from frontend.tools.undo_redo_manager import AddAnnotationCommand
        self._canvas.undo_redo.execute(AddAnnotationCommand(self._canvas, item))
        self._canvas._schedule_save()

    def key_press(self, key: int) -> None:
        if key == Qt.Key_Return or key == Qt.Key_Enter:
            self.close_polygon()
        elif key == Qt.Key_Escape:
            self._clear_temporaries()
        elif key == Qt.Key_Backspace and self._points:
            self._points.pop()
            if self._vertex_items:
                old = self._vertex_items.pop()
                self._canvas.scene().removeItem(old)
            self._draw_preview()
