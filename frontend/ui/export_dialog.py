"""
Export dialog — supports COCO JSON, YOLO TXT, Pascal VOC XML, and full zip.
"""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from frontend.api_client import APIClient, APIError


class ExportDialog(QDialog):
    def __init__(self, api: APIClient, project_id: int, parent=None) -> None:
        super().__init__(parent)
        self._api = api
        self._project_id = project_id
        self.setWindowTitle("Export Project")
        self.setFixedSize(380, 420)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        layout.addWidget(QLabel("Export Format"))

        self._group = QButtonGroup(self)
        for label, value in [
            ("COCO JSON (.json)", "coco"),
            ("YOLO TXT (.zip)", "yolo"),
            ("Pascal VOC XML (.zip)", "voc"),
            ("Full dataset ZIP (images + COCO)", "zip"),
        ]:
            rb = QRadioButton(label)
            rb.setProperty("format", value)
            self._group.addButton(rb)
            layout.addWidget(rb)
            if value == "coco":
                rb.setChecked(True)

        layout.addSpacing(8)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        layout.addWidget(sep)
        layout.addSpacing(4)

        self._split_check = QCheckBox("Split into Train / Validation / Test sets")
        self._split_check.toggled.connect(self._on_split_toggled)
        layout.addWidget(self._split_check)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Train %"))
        self._train_spin = QSpinBox()
        self._train_spin.setRange(1, 98)
        self._train_spin.setValue(80)
        self._train_spin.valueChanged.connect(self._update_test_pct)
        split_row.addWidget(self._train_spin)

        split_row.addWidget(QLabel("Val %"))
        self._val_spin = QSpinBox()
        self._val_spin.setRange(1, 98)
        self._val_spin.setValue(10)
        self._val_spin.valueChanged.connect(self._update_test_pct)
        split_row.addWidget(self._val_spin)

        self._test_label = QLabel("Test: 10%")
        self._test_label.setObjectName("mutedLabel")
        split_row.addWidget(self._test_label)
        layout.addLayout(split_row)

        self._split_widgets = [self._train_spin, self._val_spin]
        self._on_split_toggled(False)  # start collapsed/disabled

        layout.addStretch()

        btns = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)

        export_btn = QPushButton("Export")
        export_btn.setObjectName("primaryButton")
        export_btn.clicked.connect(self._do_export)
        btns.addWidget(export_btn)
        layout.addLayout(btns)

    def _on_split_toggled(self, checked: bool) -> None:
        for w in self._split_widgets:
            w.setEnabled(checked)
        self._update_test_pct()

    def _update_test_pct(self) -> None:
        test_pct = 100 - self._train_spin.value() - self._val_spin.value()
        if test_pct < 0:
            self._test_label.setText("Test: invalid (over 100%)")
            self._test_label.setStyleSheet("color: #E5484D;")
        else:
            self._test_label.setText(f"Test: {test_pct}%")
            self._test_label.setStyleSheet("")

    def _do_export(self) -> None:
        checked = self._group.checkedButton()
        if not checked:
            return
        fmt = checked.property("format")

        split = self._split_check.isChecked()
        train_pct = self._train_spin.value() / 100.0
        val_pct = self._val_spin.value() / 100.0
        if split and train_pct + val_pct > 1.0:
            QMessageBox.warning(self, "Invalid Split",
                               "Train % + Val % cannot exceed 100%.")
            return

        # A split export is always a zip (train/val/test subfolders or
        # files), even for formats that are normally a single JSON (COCO).
        ext_map = {
            "coco": ("JSON Files (*.json)", "export_coco.json"),
            "yolo": ("ZIP Files (*.zip)", "export_yolo.zip"),
            "voc": ("ZIP Files (*.zip)", "export_voc.zip"),
            "zip": ("ZIP Files (*.zip)", "dataset.zip"),
        }
        filter_str, default_name = ext_map.get(fmt, ("All Files (*)", "export"))
        if split and fmt == "coco":
            filter_str, default_name = "ZIP Files (*.zip)", "export_coco_split.zip"

        path, _ = QFileDialog.getSaveFileName(self, "Save Export", default_name, filter_str)
        if not path:
            return

        try:
            if fmt == "zip":
                data = self._api.export_zip(self._project_id, split=split, train_pct=train_pct, val_pct=val_pct)
            else:
                data = self._api.export_project(self._project_id, fmt, split=split, train_pct=train_pct, val_pct=val_pct)

            with open(path, "wb") as f:
                f.write(data)
            QMessageBox.information(self, "Export Complete",
                                    f"Exported successfully to:\n{path}")
            self.accept()
        except APIError as e:
            QMessageBox.critical(self, "Export Failed", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
