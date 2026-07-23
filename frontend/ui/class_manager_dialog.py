"""
Class manager dialog.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from frontend.api_client import APIClient, APIError


class ClassManagerDialog(QDialog):
    def __init__(self, api: APIClient, project_id: int, parent=None) -> None:
        super().__init__(parent)
        self._api = api
        self._project_id = project_id
        self.setWindowTitle("Manage Classes")
        self.setMinimumSize(360, 480)
        self._setup_ui()
        self._load()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(QLabel("Label Classes"))

        self._list = QListWidget()
        layout.addWidget(self._list)

        btns = QHBoxLayout()
        add_btn = QPushButton("+ Add Class")
        add_btn.setObjectName("primaryButton")
        add_btn.clicked.connect(self._add_class)
        btns.addWidget(add_btn)

        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._rename_class)
        btns.addWidget(rename_btn)

        color_btn = QPushButton("Change Color")
        color_btn.clicked.connect(self._change_color)
        btns.addWidget(color_btn)

        del_btn = QPushButton("Delete")
        del_btn.setObjectName("dangerButton")
        del_btn.clicked.connect(self._delete_class)
        btns.addWidget(del_btn)

        layout.addLayout(btns)

    def _load(self) -> None:
        try:
            classes = self._api.list_classes(self._project_id)
            self._list.clear()
            for cls in classes:
                item = QListWidgetItem()
                item.setText(cls["name"])
                item.setData(Qt.UserRole, cls)
                item.setForeground(QColor(cls["color_hex"]))
                self._list.addItem(item)
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _add_class(self) -> None:
        name, ok = QInputDialog.getText(self, "New Class", "Class name:")
        if not ok or not name.strip():
            return
        color = QColorDialog.getColor(QColor("#FF6B35"), self, "Pick color")
        color_hex = color.name() if color.isValid() else "#FF6B35"
        try:
            self._api.create_class(self._project_id, name.strip(), color_hex)
            self._load()
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _rename_class(self) -> None:
        item = self._list.currentItem()
        if not item:
            return
        cls = item.data(Qt.UserRole)
        name, ok = QInputDialog.getText(self, "Rename", "New name:", text=cls["name"])
        if not ok or not name.strip():
            return
        try:
            self._api.update_class(self._project_id, cls["id"], name=name.strip())
            self._load()
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _change_color(self) -> None:
        item = self._list.currentItem()
        if not item:
            return
        cls = item.data(Qt.UserRole)
        color = QColorDialog.getColor(QColor(cls["color_hex"]), self, "Pick color")
        if not color.isValid():
            return
        try:
            self._api.update_class(self._project_id, cls["id"], color_hex=color.name())
            self._load()
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _delete_class(self) -> None:
        item = self._list.currentItem()
        if not item:
            return
        cls = item.data(Qt.UserRole)
        reply = QMessageBox.question(self, "Delete", f"Delete class '{cls['name']}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                self._api.delete_class(self._project_id, cls["id"])
                self._load()
            except APIError as e:
                QMessageBox.critical(self, "Error", str(e))
