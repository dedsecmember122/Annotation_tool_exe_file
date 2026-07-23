"""
Minimize/close controls for frameless dialogs (login/signup use a custom
rounded-card look via Qt.FramelessWindowHint, which drops the native title
bar — and with it, the native minimize/close buttons — entirely).
"""
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


def make_window_controls(window: QWidget, on_close: Callable[[], None]) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addStretch()

    min_btn = QPushButton("−")
    min_btn.setFixedSize(28, 28)
    min_btn.setCursor(Qt.PointingHandCursor)
    min_btn.setStyleSheet("""
        QPushButton { color: #8C91A5; background: transparent; border: none; font-size: 18px; font-weight: bold; }
        QPushButton:hover { color: #E5E7EE; background: #2C3140; border-radius: 6px; }
    """)
    min_btn.clicked.connect(window.showMinimized)
    row.addWidget(min_btn)

    close_btn = QPushButton("✕")
    close_btn.setFixedSize(28, 28)
    close_btn.setCursor(Qt.PointingHandCursor)
    close_btn.setStyleSheet("""
        QPushButton { color: #8C91A5; background: transparent; border: none; font-size: 13px; font-weight: bold; }
        QPushButton:hover { color: white; background: #E5484D; border-radius: 6px; }
    """)
    close_btn.clicked.connect(on_close)
    row.addWidget(close_btn)

    return row
