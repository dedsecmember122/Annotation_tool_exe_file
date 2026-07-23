"""
Local-disk storage backend (development / single-user mode).
"""
import shutil
from pathlib import Path
from typing import BinaryIO

from backend.app.core.config import get_settings
from backend.app.core.storage.base import StorageBackend

settings = get_settings()


class LocalStorage(StorageBackend):

    def __init__(self) -> None:
        self.root = Path(settings.LOCAL_STORAGE_PATH)
        self.root.mkdir(parents=True, exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        d = self.root / project_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_image(self, project_id: str, filename: str, file: BinaryIO) -> str:
        dest = self._project_dir(project_id) / filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file, f)
        return str(dest)

    def get_image_url(self, path: str) -> str:
        # For local storage the "URL" is the absolute path on disk.
        return str(Path(path).resolve())

    def delete_image(self, path: str) -> None:
        p = Path(path)
        if p.exists():
            p.unlink()

    def get_image_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()
