"""
Abstract storage backend — all concrete stores implement this interface.
Nothing else in the codebase should import a concrete storage class directly;
use get_storage() from this module instead.
"""
from abc import ABC, abstractmethod
from typing import BinaryIO


class StorageBackend(ABC):

    @abstractmethod
    def save_image(self, project_id: str, filename: str, file: BinaryIO) -> str:
        """
        Persist image bytes and return the storage path/key.
        """
        ...

    @abstractmethod
    def get_image_url(self, path: str) -> str:
        """
        Return a URL (or absolute local path) that can be used to serve the image.
        """
        ...

    @abstractmethod
    def delete_image(self, path: str) -> None:
        """Remove the stored image."""
        ...

    @abstractmethod
    def get_image_bytes(self, path: str) -> bytes:
        """Return raw bytes for the image at *path*."""
        ...
