"""
Central configuration module.
Reads APP_ENV (development | production) and exposes a Settings singleton.
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_custom_model_dir() -> str:
    """Directory holding the DETC training/inference code (model/, utils/, train.py).

    In a frozen PyInstaller build, walking a fixed number of .parent hops
    off this module's own __file__ lands inside the temporary extraction
    folder (sys._MEIPASS), not next to a real "detc-core" folder — that
    folder only exists there because build/annotation_tool.spec explicitly
    bundles it as data. Anchor on sys._MEIPASS directly when frozen instead
    of relying on __file__, the same fix already applied to resource
    lookups in frontend/main.py's _resource_path().
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return str(Path(meipass) / "detc-core")
    return str(Path(__file__).resolve().parent.parent.parent.parent / "detc-core")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Environment ──────────────────────────────────────────────────────────
    APP_ENV: str = "development"  # development | production

    # ── API ──────────────────────────────────────────────────────────────────
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8765
    API_PREFIX: str = "/api"

    # ── Security ─────────────────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_USE_STRONG_RANDOM_KEY"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── Database ─────────────────────────────────────────────────────────────
    # Dev: SQLite (auto-created in user data dir)
    # Prod: set DATABASE_URL=postgresql+asyncpg://user:pass@host/db
    DATABASE_URL: str = ""

    @property
    def resolved_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        data_dir = Path.home() / "AnnotationTool" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{data_dir / 'annotation_tool.db'}"

    # ── Storage ──────────────────────────────────────────────────────────────
    LOCAL_STORAGE_PATH: str = str(Path.home() / "AnnotationTool" / "storage")

    # S3 (production only)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = ""

    # ── ML ───────────────────────────────────────────────────────────────────
    CUSTOM_MODEL_DIR: str = _default_custom_model_dir()
    HF_MODEL_NAME: str = "google/owlvit-base-patch32"
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5

    # ── Auto-train ───────────────────────────────────────────────────────────
    DEFAULT_AUTOTRAIN_THRESHOLD: int = 100  # images before first auto-train

    @property
    def is_development(self) -> bool:
        return self.APP_ENV.lower() == "development"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
