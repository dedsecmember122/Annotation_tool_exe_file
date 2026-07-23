"""
Command-pattern Undo/Redo manager.
Every reversible action implements Command and is pushed onto the stack.
"""
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional


class Command(ABC):
    """Base class for all reversible operations."""

    @abstractmethod
    def execute(self) -> None:
        """Apply the command."""
        ...

    @abstractmethod
    def undo(self) -> None:
        """Reverse the command."""
        ...

    @property
    def description(self) -> str:
        return self.__class__.__name__


# ── Concrete Commands ─────────────────────────────────────────────────────────

class AddAnnotationCommand(Command):
    def __init__(self, canvas: "AnnotationCanvas", item: "AnnotationItem") -> None:
        self._canvas = canvas
        self._item = item

    def execute(self) -> None:
        self._canvas.scene().addItem(self._item)
        self._canvas.annotations.append(self._item)

    def undo(self) -> None:
        self._canvas.scene().removeItem(self._item)
        self._canvas.annotations.remove(self._item)

    @property
    def description(self) -> str:
        return f"Add {self._item.shape_type}"


class DeleteAnnotationCommand(Command):
    def __init__(self, canvas: "AnnotationCanvas", item: "AnnotationItem") -> None:
        self._canvas = canvas
        self._item = item

    def execute(self) -> None:
        self._canvas.scene().removeItem(self._item)
        if self._item in self._canvas.annotations:
            self._canvas.annotations.remove(self._item)

    def undo(self) -> None:
        self._canvas.scene().addItem(self._item)
        self._canvas.annotations.append(self._item)

    @property
    def description(self) -> str:
        return f"Delete {self._item.shape_type}"


class MoveAnnotationCommand(Command):
    def __init__(self, item: "AnnotationItem", old_pos: object, new_pos: object) -> None:
        self._item = item
        self._old_pos = old_pos
        self._new_pos = new_pos

    def execute(self) -> None:
        self._item.setPos(self._new_pos)

    def undo(self) -> None:
        self._item.setPos(self._old_pos)

    @property
    def description(self) -> str:
        return "Move annotation"


class RelabelCommand(Command):
    def __init__(self, item: "AnnotationItem", old_class: Optional[dict], new_class: Optional[dict]) -> None:
        self._item = item
        self._old = old_class
        self._new = new_class

    def execute(self) -> None:
        self._item.set_class(self._new)

    def undo(self) -> None:
        self._item.set_class(self._old)

    @property
    def description(self) -> str:
        return "Relabel annotation"


class ResizeBBoxCommand(Command):
    def __init__(self, item: "AnnotationItem", old_rect: object, new_rect: object) -> None:
        self._item = item
        self._old = old_rect
        self._new = new_rect

    def execute(self) -> None:
        self._item.setRect(self._new)

    def undo(self) -> None:
        self._item.setRect(self._old)

    @property
    def description(self) -> str:
        return "Resize bounding box"


# ── Manager ───────────────────────────────────────────────────────────────────

class UndoRedoManager:
    MAX_HISTORY = 200

    def __init__(self) -> None:
        self._undo_stack: deque[Command] = deque(maxlen=self.MAX_HISTORY)
        self._redo_stack: deque[Command] = deque(maxlen=self.MAX_HISTORY)

    def execute(self, command: Command) -> None:
        command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()

    def undo(self) -> Optional[str]:
        if not self._undo_stack:
            return None
        cmd = self._undo_stack.pop()
        cmd.undo()
        self._redo_stack.append(cmd)
        return cmd.description

    def redo(self) -> Optional[str]:
        if not self._redo_stack:
            return None
        cmd = self._redo_stack.pop()
        cmd.execute()
        self._undo_stack.append(cmd)
        return cmd.description

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    @property
    def undo_description(self) -> Optional[str]:
        return self._undo_stack[-1].description if self._undo_stack else None

    @property
    def redo_description(self) -> Optional[str]:
        return self._redo_stack[-1].description if self._redo_stack else None
