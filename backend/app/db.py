"""
SQLAlchemy engine + session factory.
Uses a sync engine (SQLite in dev, PostgreSQL in prod).
"""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import get_settings

settings = get_settings()

# SQLite requires connect_args to avoid threading issues in the embedded backend
_is_sqlite = settings.resolved_database_url.startswith("sqlite")
connect_args = {"check_same_thread": False} if _is_sqlite else {}

engine = create_engine(
    settings.resolved_database_url,
    connect_args=connect_args,
    echo=False,
)

if _is_sqlite:
    # WAL mode allows concurrent readers alongside a writer (default SQLite
    # journal mode only allows one connection to touch the DB at a time).
    # busy_timeout makes a connection retry for up to 30s instead of
    # immediately raising "database is locked" when it hits a writer.
    # Without these, a long-running background training job holding a
    # session open collides with periodic log-progress writes and status
    # polls, and those failures were being silently swallowed elsewhere —
    # this fixes the contention at the source instead of masking symptoms.
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context-manager version — for use outside FastAPI dependency injection."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (idempotent)."""
    from backend.app.models import models  # noqa: F401 — registers mapped classes
    Base.metadata.create_all(bind=engine)
