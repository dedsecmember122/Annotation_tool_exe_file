"""
Signup window.
"""
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from frontend.api_client import APIClient, APIError
from frontend.ui.window_controls import make_window_controls

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _password_strength(pw: str) -> str:
    if len(pw) < 8:
        return "Too short (min 8 chars)"
    has_upper = any(c.isupper() for c in pw)
    has_digit = any(c.isdigit() for c in pw)
    has_special = any(c in "!@#$%^&*()-_=+[]{}|;':\",./<>?" for c in pw)
    score = sum([has_upper, has_digit, has_special])
    return ["Weak", "Fair", "Good", "Strong"][score]


STRENGTH_COLORS = {"Too short (min 8 chars)": "#E5484D", "Weak": "#E07A3F",
                   "Fair": "#E5B454", "Good": "#7CB342", "Strong": "#4CAF7D"}


class SignupWindow(QDialog):
    signup_successful = Signal(dict)

    def __init__(self, api: APIClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("InSiSo Model Bench — Create Account")
        self.setFixedSize(420, 690)  # +50 vs. before the custom title-bar row was added
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)

        card = QWidget()
        card.setObjectName("loginCard")
        card.setStyleSheet("""
            QWidget#loginCard {
                background: rgba(26, 29, 40, 0.98);
                border-radius: 18px;
                border: 1px solid #2C3140;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(36, 36, 36, 36)
        layout.setSpacing(14)

        layout.addLayout(make_window_controls(self, on_close=self.reject))

        title = QLabel("Create Account")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(8)

        for label_text, attr, placeholder, echo in [
            ("Username", "_uname", "Choose a username", QLineEdit.Normal),
            ("Email", "_email", "your@email.com", QLineEdit.Normal),
            ("Password", "_pw", "Min 8 characters", QLineEdit.Password),
            ("Confirm Password", "_pw2", "Repeat password", QLineEdit.Password),
        ]:
            layout.addWidget(QLabel(label_text))
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setEchoMode(echo)
            edit.setFixedHeight(40)
            setattr(self, attr, edit)
            layout.addWidget(edit)

        # Password strength indicator
        self._strength_label = QLabel("")
        self._strength_label.setAlignment(Qt.AlignRight)
        self._pw.textChanged.connect(self._on_pw_change)
        layout.addWidget(self._strength_label)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #E5767A; font-size: 12px;")
        self._error_label.setAlignment(Qt.AlignCenter)
        self._error_label.setWordWrap(True)
        layout.addWidget(self._error_label)

        btn = QPushButton("Create Account")
        btn.setObjectName("primaryButton")
        btn.setFixedHeight(44)
        btn.clicked.connect(self._on_signup)
        layout.addWidget(btn)

        layout.addSpacing(16)

        back_row = QHBoxLayout()
        back_lbl = QLabel("Already have an account?")
        back_row.addWidget(back_lbl)
        back_btn = QPushButton("Sign in")
        back_btn.setFlat(True)
        back_btn.setStyleSheet("color: #7C6FF0; font-weight: bold; border: none; background: transparent;")
        back_btn.clicked.connect(self.reject)
        back_row.addWidget(back_btn)
        back_row.setAlignment(Qt.AlignCenter)
        layout.addLayout(back_row)

        root.addWidget(card)

    def _on_pw_change(self, pw: str) -> None:
        strength = _password_strength(pw)
        color = STRENGTH_COLORS.get(strength, "#E0E0E0")
        self._strength_label.setText(f"<span style='color:{color}'>{strength}</span>")

    def _on_signup(self) -> None:
        uname = self._uname.text().strip()
        email = self._email.text().strip()
        pw = self._pw.text()
        pw2 = self._pw2.text()

        if not all([uname, email, pw, pw2]):
            self._error_label.setText("All fields are required.")
            return
        if not EMAIL_RE.match(email):
            self._error_label.setText("Invalid email address.")
            return
        strength = _password_strength(pw)
        if strength in ("Too short (min 8 chars)", "Weak"):
            self._error_label.setText(f"Password is {strength.lower()}. Choose a stronger password.")
            return
        if pw != pw2:
            self._error_label.setText("Passwords do not match.")
            return

        try:
            user = self._api.signup(uname, email, pw, pw2)
            self.signup_successful.emit(user)
            self.accept()
        except APIError as e:
            self._error_label.setText(str(e.detail))
        except Exception as e:
            self._error_label.setText(f"Connection error: {e}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and hasattr(self, "_drag_pos"):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
