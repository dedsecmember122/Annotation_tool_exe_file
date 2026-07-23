"""
Drag/Select tool — pan canvas, select/move existing annotations.
"""
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QGraphicsItem


class DragTool:
    """
    In drag mode:
    - Click on empty space → pan (handled by QGraphicsView's DragMode)
    - Click on an item → select it (Qt handles this natively)
    - The canvas sets ViewportDragMode based on this tool's state
    """

    def __init__(self, canvas: "AnnotationCanvas") -> None:
        self._canvas = canvas

    def activate(self) -> None:
        from PySide6.QtWidgets import QGraphicsView
        self._canvas.setDragMode(QGraphicsView.ScrollHandDrag)

    def deactivate(self) -> None:
        from PySide6.QtWidgets import QGraphicsView
        self._canvas.setDragMode(QGraphicsView.NoDrag)

    def mouse_press(self, scene_pos: QPointF) -> None:
        pass

    def mouse_move(self, scene_pos: QPointF) -> None:
        pass

    def mouse_release(self, scene_pos: QPointF) -> None:
        pass
