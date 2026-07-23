"""
Application entry point.

Dev mode (APP_ENV=development, default):
  - Starts the FastAPI backend in a background thread on 127.0.0.1:8765
  - Launches the PySide6 GUI
  - Everything in one process — no server setup needed

Prod mode (APP_ENV=production):
  - Skips the embedded backend
  - Reads BACKEND_URL from .env or config to point at the remote server
  - Ships as a lightweight frontend-only .exe
"""
import os
import sys
import threading
import time
from pathlib import Path

# In a windowed (console=False) PyInstaller build there is no console, so
# sys.stdout/stderr are None. Anything that writes to them or probes them
# (uvicorn's logging setup calls sys.stdout.isatty() to decide on color
# output) crashes with AttributeError. Patch them to harmless no-op streams
# before any other import gets a chance to touch them.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

# ── Make the project root importable ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import get_settings
from frontend.api_client import APIClient

settings = get_settings()

# ── Globals ────────────────────────────────────────────────────────────────────
_app = None  # QApplication


# ── Hidden training-worker mode ─────────────────────────────────────────────────

def _run_train_worker(argv: list[str]) -> int:
    """Run detc-core/train.py's training loop directly in this process.

    A packaged onefile PyInstaller build has no separate python.exe to hand
    train.py to — sys.executable there is this very exe, so trying to launch
    it as `sys.executable train.py ...` (the dev-mode command) just starts a
    second copy of the GUI. backend/app/ml/custom_model_adapter.py works
    around this by re-invoking this exe itself with a hidden
    "--train-worker" flag as a real child process (subprocess.Popen); when
    that flag is present we skip the GUI/backend entirely and dispatch
    straight into train.py instead (the same trick multiprocessing's
    freeze_support() uses for worker processes).
    """
    model_dir = str(settings.CUSTOM_MODEL_DIR)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    import importlib
    train_mod = importlib.import_module("train")  # detc-core/train.py
    args = train_mod.parse_args(argv)
    train_mod.train(args)
    return 0


# ── Resource resolution ────────────────────────────────────────────────────────

def _resource_path(name: str) -> Path:
    """Locate a file under frontend/resources/, whether running from source
    or as a frozen PyInstaller exe.

    PyInstaller has a real footgun here: for the *entry* script (main.py,
    the one passed to Analysis) specifically, __file__ in a frozen build
    resolves to a flattened path inside the extraction bundle that's
    missing the frontend/ prefix the bundled data actually lives under —
    unlike __file__ for normally-imported modules, which keeps the full
    package path. Path(__file__).parent / "resources" therefore silently
    fails to find anything in the packaged exe (no exception — callers
    like _apply_theme() just skip applying the file if it's not found),
    while working fine in dev mode. sys._MEIPASS is the one anchor
    PyInstaller guarantees regardless of that, so use it directly when
    frozen instead of relying on this module's own __file__.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "frontend" / "resources" / name
    return Path(__file__).parent / "resources" / name


# ── Theme helper ──────────────────────────────────────────────────────────────

def _apply_theme(theme: str = "dark") -> None:
    from PySide6.QtWidgets import QApplication

    # Use the live singleton rather than the module-level `_app` global:
    # when this module is re-imported as "frontend.main" from within the
    # package (e.g. by main_window.py) while the app was launched as a
    # top-level script (`python frontend/main.py`, running as "__main__"),
    # that import creates a second, distinct module object whose own
    # `_app` was never assigned — QApplication.instance() is shared no
    # matter which module identity asks for it.
    app = QApplication.instance()
    if app is None:
        return
    qss_file = _resource_path(f"style_{theme}.qss")
    if qss_file.exists():
        app.setStyleSheet(qss_file.read_text(encoding="utf-8"))


# ── Embedded backend ──────────────────────────────────────────────────────────

def _backend_log_path() -> Path:
    log_dir = Path.home() / "AnnotationTool"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "backend_error.log"


def _start_backend() -> None:
    try:
        import uvicorn
        from backend.app.main import app as fastapi_app
        uvicorn.run(
            fastapi_app,
            host=settings.API_HOST,
            port=settings.API_PORT,
            log_level="warning",
            access_log=False,
        )
    except Exception:
        # Runs in a background thread inside a windowed (console=False) exe,
        # so an uncaught exception here would otherwise vanish silently —
        # write it somewhere the user (or support) can actually find it.
        import traceback
        _backend_log_path().write_text(traceback.format_exc(), encoding="utf-8")


# PyInstaller sets sys.frozen=True on the packaged exe. A frozen, onefile
# build re-extracts the whole bundle (incl. torch) on every launch and is
# often scanned by antivirus on first run, so cold start is much slower
# than in dev — give it much more room before declaring failure.
BACKEND_START_TIMEOUT = 90.0 if getattr(sys, "frozen", False) else 15.0


def _wait_for_backend(timeout: float = BACKEND_START_TIMEOUT, thread: threading.Thread | None = None) -> bool:
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{settings.API_HOST}:{settings.API_PORT}/health", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        if thread is not None and not thread.is_alive():
            # Backend thread already crashed — no point waiting out the
            # rest of the timeout, the exception is in backend_error.log.
            return False
        time.sleep(0.3)
    return False


# ── GUI entry ─────────────────────────────────────────────────────────────────

def main() -> int:
    global _app

    from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen
    from PySide6.QtGui import QPixmap, QFont, QIcon
    from PySide6.QtCore import Qt

    _app = QApplication(sys.argv)
    _app.setApplicationName("InSiSo Model Bench")
    _app.setOrganizationName("InSiSo Technologies")

    # Sets the default icon for every window that doesn't set its own
    # (taskbar, title bar, Alt-Tab) — the exe itself is also built with this
    # icon (see annotation_tool.spec), so it's consistent everywhere.
    icon_path = _resource_path("icon.ico")
    if icon_path.exists():
        _app.setWindowIcon(QIcon(str(icon_path)))

    # Set default font — pick a native UI font per platform
    if sys.platform == "darwin":
        font = QFont("Helvetica Neue", 12)
    elif sys.platform.startswith("linux"):
        font = QFont("Ubuntu", 10)
    else:
        font = QFont("Segoe UI", 10)
    _app.setFont(font)

    # Apply dark theme by default
    _apply_theme("dark")

    # ── Splash screen ─────────────────────────────────────────────────────────
    splash_pix = QPixmap(400, 220)
    splash_pix.fill(Qt.transparent)
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setAttribute(Qt.WA_TranslucentBackground)

    from PySide6.QtGui import QPainter, QColor, QPen
    from PySide6.QtCore import QRect
    painter = QPainter(splash_pix)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(26, 29, 40, 245))
    painter.setPen(QPen(QColor(124, 111, 240), 2))
    painter.drawRoundedRect(0, 0, 400, 220, 18, 18)
    title_font = QFont(font.family(), 18, QFont.Bold)
    painter.setFont(title_font)
    painter.setPen(QColor(124, 111, 240))
    painter.drawText(QRect(0, 50, 400, 60), Qt.AlignCenter, "InSiSo Model Bench")
    sub_font = QFont(font.family(), 11)
    painter.setFont(sub_font)
    painter.setPen(QColor(169, 174, 196))
    painter.drawText(QRect(0, 120, 400, 40), Qt.AlignCenter, "Starting services…")
    painter.end()
    splash.setPixmap(splash_pix)
    splash.show()
    _app.processEvents()

    # ── Start embedded backend (dev mode) ─────────────────────────────────────
    if settings.is_development:
        backend_thread = threading.Thread(target=_start_backend, daemon=True)
        backend_thread.start()

        if not _wait_for_backend(timeout=BACKEND_START_TIMEOUT, thread=backend_thread):
            splash.close()
            log_path = _backend_log_path()
            detail = ""
            if log_path.exists() and log_path.stat().st_size > 0:
                detail = f"\n\nError details saved to:\n{log_path}"
            QMessageBox.critical(None, "Startup Error",
                                 f"The embedded backend failed to start within "
                                 f"{int(BACKEND_START_TIMEOUT)} seconds.\n"
                                 "This can happen if port 8765 is already in use, "
                                 "or antivirus is scanning the app on first launch."
                                 + detail)
            return 1

    api = APIClient()

    splash.close()

    # ── Login ─────────────────────────────────────────────────────────────────
    from frontend.ui.login_window import LoginWindow
    login_win = LoginWindow(api)
    user: dict | None = None

    def on_login(u: dict) -> None:
        nonlocal user
        user = u

    login_win.login_successful.connect(on_login)
    result = login_win.exec()

    if result != LoginWindow.DialogCode.Accepted or user is None:
        return 0

    # ── Main window ───────────────────────────────────────────────────────────
    from frontend.ui.main_window import MainWindow
    win = MainWindow(api, user)
    win.show()

    return _app.exec()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train-worker":
        sys.exit(_run_train_worker(sys.argv[2:]))
    sys.exit(main())
