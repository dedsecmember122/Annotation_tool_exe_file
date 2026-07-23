"""
Login window — dark/light themed PySide6 dialog.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from frontend.api_client import APIClient, APIError
from frontend.ui.window_controls import make_window_controls


class LoginWindow(QDialog):
    login_successful = Signal(dict)  # emits user info dict

    def __init__(self, api: APIClient, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._api = api
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle("InSiSo Model Bench — Login")
        self.setFixedSize(420, 630)  # +50 vs. before the custom title-bar row was added
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(0)

        # Card
        card = QWidget()
        card.setObjectName("loginCard")
        card.setStyleSheet("""
            QWidget#loginCard {
                background: rgba(26, 29, 40, 0.98);
                border-radius: 18px;
                border: 1px solid #2C3140;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(36, 40, 36, 40)
        card_layout.setSpacing(20)

        # Custom title bar — this window is frameless (for the rounded-card
        # look), so it has no native minimize/close buttons unless we add
        # our own.
        card_layout.addLayout(make_window_controls(self, on_close=self.reject))

        # Header
        title = QLabel("InSiSo Model Bench")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel("Image Annotation Tool")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(subtitle)

        tagline = QLabel("Sign in to your workspace")
        tagline.setStyleSheet("color: #8C91A5; font-size: 11px;")
        tagline.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(tagline)

        card_layout.addSpacing(16)

        # Form
        self._username_edit = QLineEdit()
        self._username_edit.setPlaceholderText("Username or email")
        self._username_edit.setFixedHeight(42)
        card_layout.addWidget(QLabel("Username / Email"))
        card_layout.addWidget(self._username_edit)

        self._password_edit = QLineEdit()
        self._password_edit.setEchoMode(QLineEdit.Password)
        self._password_edit.setPlaceholderText("Password")
        self._password_edit.setFixedHeight(42)
        card_layout.addWidget(QLabel("Password"))
        card_layout.addWidget(self._password_edit)

        card_layout.addSpacing(8)

        # Trial notice (hidden by default, shown after login)
        self._trial_label = QLabel("")
        self._trial_label.setStyleSheet("color: #E5B454; font-size: 11px;")
        self._trial_label.setAlignment(Qt.AlignCenter)
        self._trial_label.setWordWrap(True)
        card_layout.addWidget(self._trial_label)

        # Error label
        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #E5767A; font-size: 12px;")
        self._error_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self._error_label)

        # Login button
        self._login_btn = QPushButton("Sign In")
        self._login_btn.setObjectName("primaryButton")
        self._login_btn.setFixedHeight(44)
        self._login_btn.clicked.connect(self._on_login)
        card_layout.addWidget(self._login_btn)

        card_layout.addSpacing(16)

        # Switch to signup
        switch_row = QHBoxLayout()
        switch_row.addWidget(QLabel("Don't have an account?"))
        signup_btn = QPushButton("Create account")
        signup_btn.setFlat(True)
        signup_btn.setStyleSheet("color: #7C6FF0; font-weight: bold; border: none; background: transparent;")
        signup_btn.clicked.connect(self._open_signup)
        switch_row.addWidget(signup_btn)
        switch_row.setAlignment(Qt.AlignCenter)
        card_layout.addLayout(switch_row)

        root.addWidget(card)

        # Enter key support
        self._password_edit.returnPressed.connect(self._on_login)
        self._username_edit.returnPressed.connect(self._password_edit.setFocus)

    def _on_login(self) -> None:
        username = self._username_edit.text().strip()
        password = self._password_edit.text()
        if not username or not password:
            self._error_label.setText("Please fill in all fields.")
            return
        self._login_btn.setEnabled(False)
        self._login_btn.setText("Signing in…")
        self._trial_label.setText("")
        try:
            self._api.login(username, password)
            user = self._api.me()
            # Show trial notice if applicable
            days = user.get("trial_days_remaining")
            if days is not None and user.get("role") != "admin":
                if days <= 1:
                    self._trial_label.setText(
                        f"⚠ Your free trial expires {'today' if days == 0 else 'tomorrow'}! "
                        "Contact admin to extend access."
                    )
                else:
                    self._trial_label.setText(f"ℹ Free trial: {days} day(s) remaining.")
            self.login_successful.emit(user)
            self.accept()
        except APIError as e:
            self._error_label.setText(str(e.detail))
        except Exception as e:
            self._error_label.setText(f"Connection error: {e}")
        finally:
            self._login_btn.setEnabled(True)
            self._login_btn.setText("Sign In")

    def _open_signup(self) -> None:
        from frontend.ui.signup_window import SignupWindow
        dlg = SignupWindow(self._api, self)
        dlg.signup_successful.connect(self._on_signup_done)
        dlg.exec()

    def _on_signup_done(self, user: dict) -> None:
        self._username_edit.setText(user.get("username", ""))
        self._error_label.setText("")

    # ── Draggable frameless window ────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and hasattr(self, "_drag_pos"):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
