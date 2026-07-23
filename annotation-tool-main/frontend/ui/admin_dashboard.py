"""
Admin dashboard — manage users, roles, and trial access (admin only).
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from frontend.api_client import APIClient, APIError


class _AddUserDialog(QDialog):
    """Collects username/email/password/role for admin_create_user().
    Field validation happens in the caller after exec(), same pattern as
    the other simple dialogs in this app (e.g. ClassManagerDialog)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add User")
        self.setFixedSize(340, 260)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        form = QFormLayout()
        self.username_edit = QLineEdit()
        self.email_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.role_combo = QComboBox()
        self.role_combo.addItems(["annotator", "admin"])
        form.addRow("Username", self.username_edit)
        form.addRow("Email", self.email_edit)
        form.addRow("Password", self.password_edit)
        form.addRow("Role", self.role_combo)
        layout.addLayout(form)

        layout.addStretch()

        btns = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        create_btn = QPushButton("Create")
        create_btn.setObjectName("primaryButton")
        create_btn.clicked.connect(self.accept)
        btns.addWidget(create_btn)
        layout.addLayout(btns)

    def values(self) -> tuple[str, str, str, str]:
        return (
            self.username_edit.text().strip(),
            self.email_edit.text().strip(),
            self.password_edit.text(),
            self.role_combo.currentText(),
        )


class AdminDashboardDialog(QDialog):
    def __init__(self, api: APIClient, current_user: dict, parent=None) -> None:
        super().__init__(parent)
        self._api = api
        self._current_user = current_user
        self.setWindowTitle("Admin Dashboard")
        self.setMinimumSize(680, 440)
        self._setup_ui()
        self._load()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        layout.addWidget(QLabel("Users"))

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Username", "Email", "Role", "Trial", "Joined"])
        self._tree.setRootIsDecorated(False)
        self._tree.setColumnWidth(0, 140)
        self._tree.setColumnWidth(1, 220)
        self._tree.setColumnWidth(2, 90)
        self._tree.setColumnWidth(3, 110)
        self._tree.itemSelectionChanged.connect(self._update_button_state)
        layout.addWidget(self._tree)

        btns = QHBoxLayout()

        add_user_btn = QPushButton("+ Add User")
        add_user_btn.setObjectName("primaryButton")
        add_user_btn.clicked.connect(self._add_user)
        btns.addWidget(add_user_btn)

        self._role_btn = QPushButton("Make Admin")
        self._role_btn.setObjectName("primaryButton")
        self._role_btn.clicked.connect(self._toggle_role)
        btns.addWidget(self._role_btn)

        self._extend_btn = QPushButton("Extend Trial +7 days")
        self._extend_btn.clicked.connect(self._extend_trial)
        btns.addWidget(self._extend_btn)

        self._reset_btn = QPushButton("Reset Trial")
        self._reset_btn.clicked.connect(self._reset_trial)
        btns.addWidget(self._reset_btn)

        btns.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load)
        btns.addWidget(refresh_btn)

        layout.addLayout(btns)
        self._update_button_state()

    # ── Data ─────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            users = self._api.admin_list_users()
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        self._tree.clear()
        for u in users:
            role = u.get("role", "")
            if role == "admin":
                trial_text = "—"
            else:
                days = u.get("trial_days_remaining")
                trial_text = "not started" if days is None else f"{days}d left"
            item = QTreeWidgetItem([
                u.get("username", ""),
                u.get("email", ""),
                role,
                trial_text,
                (u.get("created_at") or "")[:10],
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, u)
            self._tree.addTopLevelItem(item)
        self._update_button_state()

    def _selected_user(self) -> dict | None:
        item = self._tree.currentItem()
        if not item:
            return None
        return item.data(0, Qt.ItemDataRole.UserRole)

    def _update_button_state(self) -> None:
        user = self._selected_user()
        if user is None:
            self._role_btn.setEnabled(False)
            self._extend_btn.setEnabled(False)
            self._reset_btn.setEnabled(False)
            self._role_btn.setText("Make Admin")
            return
        is_self = user["id"] == self._current_user.get("id")
        is_admin = user.get("role") == "admin"
        self._role_btn.setEnabled(not is_self)
        self._role_btn.setText("Make Annotator" if is_admin else "Make Admin")
        self._extend_btn.setEnabled(not is_admin)
        self._reset_btn.setEnabled(not is_admin)

    # ── Actions ──────────────────────────────────────────────────────────────

    def _add_user(self) -> None:
        dlg = _AddUserDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        username, email, password, role = dlg.values()
        if not username or not email or not password:
            QMessageBox.warning(self, "Missing Fields", "All fields are required.")
            return
        if len(password) < 8:
            QMessageBox.warning(self, "Weak Password", "Password must be at least 8 characters.")
            return
        try:
            self._api.admin_create_user(username, email, password, role)
            self._load()
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _toggle_role(self) -> None:
        user = self._selected_user()
        if not user:
            return
        new_role = "annotator" if user.get("role") == "admin" else "admin"
        reply = QMessageBox.question(
            self, "Change Role",
            f"Set '{user['username']}' as {new_role}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            self._api.admin_set_role(user["id"], new_role)
            self._load()
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _extend_trial(self) -> None:
        user = self._selected_user()
        if not user:
            return
        try:
            self._api.admin_extend_trial(user["id"], 7)
            self._load()
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))

    def _reset_trial(self) -> None:
        user = self._selected_user()
        if not user:
            return
        reply = QMessageBox.question(
            self, "Reset Trial",
            f"Reset '{user['username']}' trial to a fresh 7 days from now?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            self._api.admin_reset_trial(user["id"])
            self._load()
        except APIError as e:
            QMessageBox.critical(self, "Error", str(e))
