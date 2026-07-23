# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for AnnotationTool. Shared by both platforms - PyInstaller
can't cross-compile, so this same spec is run once on windows-latest and
once on macos-latest in CI, each producing that platform's native output.

Build command (from project root):
  pyinstaller build/annotation_tool.spec

Outputs:
  Windows: dist/AnnotationTool.exe        (single file)
  macOS:   dist/AnnotationTool.app        (bundle, wrapping a onefile binary)
"""

import sys
from pathlib import Path

ROOT = Path(".").resolve()
IS_MACOS = sys.platform == "darwin"
ICON_FILE = str(ROOT / "frontend" / "resources" / ("icon.icns" if IS_MACOS else "icon.ico"))

a = Analysis(
    [str(ROOT / "frontend" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "frontend" / "resources"), "frontend/resources"),
        (str(ROOT / "backend" / "app" / "core" / "storage"), "backend/app/core/storage"),
        # Training code (model/, utils/, train.py) — invoked as a subprocess
        # at training time (see custom_model_adapter.py). It's loaded
        # dynamically via sys.path, not statically imported, so PyInstaller's
        # analysis can't discover it on its own; without this entry the
        # packaged exe's CUSTOM_MODEL_DIR (backend/app/core/config.py) points
        # at a folder that was never actually bundled, which is what caused
        # training to fail with "[WinError 267] The directory name is
        # invalid" for every installed build.
        (str(ROOT / "detc-core"), "detc-core"),
    ],
    hiddenimports=[
        # FastAPI / Uvicorn
        "uvicorn",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        # Pydantic
        "pydantic",
        "pydantic.v1",
        "pydantic_settings",
        # SQLAlchemy dialects
        "sqlalchemy.dialects.sqlite",
        "sqlalchemy.dialects.postgresql",
        # passlib
        "passlib",
        "passlib.handlers",
        "passlib.handlers.bcrypt",
        # jose
        "jose",
        # Pillow
        "PIL",
        "PIL.Image",
        # PyTorch / TorchVision (imported lazily inside ml adapter functions,
        # so PyInstaller's static analysis can miss them without this)
        "torch",
        "torchvision",
        "torchvision.ops",
        # PySide6
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        # Application modules
        "backend.app.main",
        "backend.app.api.auth",
        "backend.app.api.projects",
        "backend.app.api.images",
        "backend.app.api.annotations",
        "backend.app.api.autoannotate",
        "backend.app.api.export",
        "backend.app.core.config",
        "backend.app.core.security",
        "backend.app.core.storage.base",
        "backend.app.core.storage.local_storage",
        "backend.app.models.models",
        "backend.app.schemas.schemas",
        "backend.app.db",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["transformers", "boto3"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="AnnotationTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,         # windowed mode — no terminal popup
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Must be an absolute path (via ROOT) - PyInstaller resolves a bare
    # relative icon path against the .spec file's own directory (build/),
    # not the invocation cwd, so "frontend/resources/icon.ico" looked for
    # build/frontend/resources/icon.ico and failed with FileNotFoundError.
    icon=ICON_FILE,
)

if IS_MACOS:
    # Wraps the onefile binary above in a proper .app bundle - without
    # this, macOS would just show a bare Unix executable, not something
    # Finder/Dock treat as a real application (no icon, no name in the
    # menu bar, no double-click-to-launch semantics).
    app = BUNDLE(
        exe,
        name="AnnotationTool.app",
        icon=ICON_FILE,
        bundle_identifier="com.insisotech.annotationtool",
        info_plist={
            "CFBundleName": "InSiSo Model Bench",
            "CFBundleDisplayName": "InSiSo Model Bench",
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )
