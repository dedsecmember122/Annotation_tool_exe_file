"""
Bounding box drawing tool.
Used by AnnotationCanvas when mode == "bbox".
"""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QCursor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsScene


HANDLE_SIZE = 8

_CURSOR_FOR_HANDLE = {
    "tl": Qt.CursorShape.SizeFDiagCursor,
    "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor,
    "bl": Qt.CursorShape.SizeBDiagCursor,
    "tm": Qt.CursorShape.SizeVerCursor,
    "bm": Qt.CursorShape.SizeVerCursor,
    "lm": Qt.CursorShape.SizeHorCursor,
    "rm": Qt.CursorShape.SizeHorCursor,
}


class BBoxItem(QGraphicsRectItem):
    """A resizable, selectable bounding box annotation."""

    HANDLES = ["tl", "tr", "bl", "br", "tm", "bm", "lm", "rm"]

    def __init__(self, rect: QRectF, class_info: dict | None = None,
                 annotation_id: int | None = None, source: str = "manual",
                 reviewed: bool = False) -> None:
        super().__init__(rect)
        self.shape_type = "bbox"
        self.class_info = class_info
        self.annotation_id = annotation_id
        self.source = source
        self.reviewed = reviewed
        self._dragging_handle: str | None = None
        self._orig_rect: QRectF | None = None
        self._move_start_pos: QPointF | None = None

        self.setFlags(
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self._update_style()

    def set_class(self, class_info: dict | None) -> None:
        self.class_info = class_info
        self._update_style()

    def _update_style(self) -> None:
        if self.source == "auto" and not self.reviewed:
            color = QColor("#FF8C00")
        elif self.class_info:
            color = QColor(self.class_info.get("color_hex", "#4FC3F7"))
        else:
            color = QColor("#4FC3F7")
        pen = QPen(color, 2, Qt.SolidLine)
        self.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(40)
        self.setBrush(fill)

    def _handle_rects(self) -> dict[str, QRectF]:
        r = self.rect()
        s = HANDLE_SIZE
        hs = s / 2
        cx, cy = r.center().x(), r.center().y()
        return {
            "tl": QRectF(r.left() - hs, r.top() - hs, s, s),
            "tr": QRectF(r.right() - hs, r.top() - hs, s, s),
            "bl": QRectF(r.left() - hs, r.bottom() - hs, s, s),
            "br": QRectF(r.right() - hs, r.bottom() - hs, s, s),
            "tm": QRectF(cx - hs, r.top() - hs, s, s),
            "bm": QRectF(cx - hs, r.bottom() - hs, s, s),
            "lm": QRectF(r.left() - hs, cy - hs, s, s),
            "rm": QRectF(r.right() - hs, cy - hs, s, s),
        }

    def _handle_at(self, pos: QPointF) -> str | None:
        if not self.isSelected():
            return None
        for name, rect in self._handle_rects().items():
            if rect.contains(pos):
                return name
        return None

    # ── Hit-testing / painting ─────────────────────────────────────────────
    # boundingRect/shape must include the handle squares (they poke out past
    # the plain rect) — Qt uses these for both repaint clipping and mouse
    # hit-testing, so without this the handles would be invisible and unclickable.

    def boundingRect(self) -> QRectF:
        hs = HANDLE_SIZE
        return super().boundingRect().adjusted(-hs, -hs, hs, hs)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(self.rect())
        if self.isSelected():
            for handle_rect in self._handle_rects().values():
                path.addRect(handle_rect)
        return path

    def paint(self, painter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        if self.isSelected():
            painter.setPen(QPen(QColor("#222222"), 1))
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            for handle_rect in self._handle_rects().values():
                painter.drawRect(handle_rect)

    # ── Hover feedback ──────────────────────────────────────────────────────

    def hoverMoveEvent(self, event) -> None:
        handle = self._handle_at(event.pos())
        if handle:
            self.setCursor(QCursor(_CURSOR_FOR_HANDLE[handle]))
        else:
            self.unsetCursor()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self.unsetCursor()
        super().hoverLeaveEvent(event)

    # ── Resize / move interaction ───────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        handle = self._handle_at(event.pos())
        if handle and event.button() == Qt.MouseButton.LeftButton:
            self._dragging_handle = handle
            self._orig_rect = QRectF(self.rect())
            event.accept()
            return
        self._move_start_pos = self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging_handle:
            self._resize_to(event.pos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._dragging_handle:
            new_rect = QRectF(self.rect())
            old_rect = self._orig_rect
            self._dragging_handle = None
            self._orig_rect = None
            if old_rect is not None and new_rect != old_rect:
                self._commit_resize(old_rect, new_rect)
            event.accept()
            return

        super().mouseReleaseEvent(event)
        if self._move_start_pos is not None and self.pos() != self._move_start_pos:
            self._mark_edited()
        self._move_start_pos = None

    def _resize_to(self, pos: QPointF) -> None:
        r = QRectF(self.rect())
        handle = self._dragging_handle
        if "l" in handle:
            r.setLeft(pos.x())
        if "r" in handle:
            r.setRight(pos.x())
        if "t" in handle:
            r.setTop(pos.y())
        if "b" in handle:
            r.setBottom(pos.y())
        self.setRect(r.normalized())

    def _canvas(self):
        scene = self.scene()
        views = scene.views() if scene else []
        return views[0] if views else None

    def _commit_resize(self, old_rect: QRectF, new_rect: QRectF) -> None:
        canvas = self._canvas()
        if canvas is None:
            return
        from frontend.tools.undo_redo_manager import ResizeBBoxCommand
        canvas.undo_redo.execute(ResizeBBoxCommand(self, old_rect, new_rect))
        self._mark_edited()

    def _mark_edited(self) -> None:
        """Call whenever the user moves/resizes this box. Correcting an
        auto-annotation is itself the review step, so it counts as reviewed."""
        if self.source == "auto" and not self.reviewed:
            self.reviewed = True
            self._update_style()
        canvas = self._canvas()
        if canvas is not None:
            canvas._schedule_save()

    def mark_reviewed(self) -> None:
        """Explicitly mark this box reviewed without editing it (it was
        already correct as auto-annotated)."""
        if self.reviewed:
            return
        self.reviewed = True
        self._update_style()
        canvas = self._canvas()
        if canvas is not None:
            canvas._schedule_save()

    def to_dict(self) -> dict:
        r = self.mapToScene(self.rect()).boundingRect()
        return {"x1": r.left(), "y1": r.top(), "x2": r.right(), "y2": r.bottom()}


class BBoxTool:
    """
    Handles mouse events on the QGraphicsView to draw bounding boxes.
    Attach to AnnotationCanvas via canvas.set_tool(BBoxTool(canvas)).
    """

    def __init__(self, canvas: "AnnotationCanvas") -> None:
        self._canvas = canvas
        self._start: QPointF | None = None
        self._current_item: BBoxItem | None = None

    def mouse_press(self, scene_pos: QPointF) -> None:
        self._start = scene_pos
        r = QRectF(scene_pos, scene_pos)
        color = self._canvas.current_class_color()
        self._current_item = BBoxItem(r, class_info=self._canvas.current_class)
        self._canvas.scene().addItem(self._current_item)

    def mouse_move(self, scene_pos: QPointF) -> None:
        if self._current_item and self._start:
            rect = QRectF(self._start, scene_pos).normalized()
            self._current_item.setRect(rect)

    def mouse_release(self, scene_pos: QPointF) -> None:
        if self._current_item:
            rect = self._current_item.rect()
            if rect.width() < 4 or rect.height() < 4:
                # Too small — discard
                self._canvas.scene().removeItem(self._current_item)
            else:
                from frontend.tools.undo_redo_manager import AddAnnotationCommand
                self._canvas.undo_redo.execute(
                    AddAnnotationCommand(self._canvas, self._current_item)
                )
                self._canvas._schedule_save()
            self._current_item = None
            self._start = None
