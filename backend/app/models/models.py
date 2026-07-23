"""
SQLAlchemy ORM models for all tables.
"""
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Shortened from 7 to 1 day for testing, so trial-expiry behavior can
# actually be exercised without waiting a week — change back to 7 (or
# whatever the real policy is) before rolling out to real customers.
TRIAL_PERIOD_DAYS = 1


# ── Users ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(256), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="annotator")  # admin|annotator|reviewer
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # ── Trial fields (non-admin only) ──────────────────────────────────────────
    trial_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    trial_extended_until: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    projects: Mapped[list["Project"]] = relationship("Project", back_populates="owner")
    annotations: Mapped[list["Annotation"]] = relationship("Annotation", back_populates="creator")

    @property
    def trial_expires_at(self) -> Optional[datetime]:
        """Returns the effective trial expiry datetime (None if no trial active).

        Always returns an aware (UTC) datetime — SQLite discards tzinfo on
        storage, so values read back are naive and must be re-tagged as UTC
        before comparing against datetime.now(timezone.utc).
        """
        if self.role == "admin":
            return None  # admins never expire
        expires = None
        if self.trial_extended_until:
            expires = self.trial_extended_until
        elif self.trial_started_at:
            expires = self.trial_started_at + timedelta(days=TRIAL_PERIOD_DAYS)
        if expires is not None and expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return expires

    @property
    def is_trial_expired(self) -> bool:
        expires = self.trial_expires_at
        if expires is None:
            return False
        return datetime.now(timezone.utc) > expires

    @property
    def trial_days_remaining(self) -> Optional[int]:
        expires = self.trial_expires_at
        if expires is None:
            return None
        delta = expires - datetime.now(timezone.utc)
        return max(0, delta.days)


# ── Projects ──────────────────────────────────────────────────────────────────

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    owner_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    class_list: Mapped[str] = mapped_column(Text, default="[]")  # JSON list of class names
    storage_mode: Mapped[str] = mapped_column(String(32), default="local")  # local|cloud
    autotrain_threshold: Mapped[int] = mapped_column(Integer, default=100)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    owner: Mapped["User"] = relationship("User", back_populates="projects")
    images: Mapped[list["Image"]] = relationship("Image", back_populates="project", cascade="all, delete-orphan")
    classes: Mapped[list["LabelClass"]] = relationship("LabelClass", back_populates="project", cascade="all, delete-orphan")
    model_versions: Mapped[list["ModelVersion"]] = relationship("ModelVersion", back_populates="project", cascade="all, delete-orphan")
    training_jobs: Mapped[list["TrainingJob"]] = relationship("TrainingJob", back_populates="project", cascade="all, delete-orphan")

    def get_class_list(self) -> list:
        return json.loads(self.class_list or "[]")

    def set_class_list(self, classes: list) -> None:
        self.class_list = json.dumps(classes)


# ── Images ────────────────────────────────────────────────────────────────────

class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    # unannotated | in_progress | annotated | auto_annotated
    status: Mapped[str] = mapped_column(String(32), default="unannotated")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped["Project"] = relationship("Project", back_populates="images")
    annotations: Mapped[list["Annotation"]] = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")


# ── Label Classes ─────────────────────────────────────────────────────────────

class LabelClass(Base):
    __tablename__ = "classes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    color_hex: Mapped[str] = mapped_column(String(7), default="#FF6B35")

    project: Mapped["Project"] = relationship("Project", back_populates="classes")
    annotations: Mapped[list["Annotation"]] = relationship("Annotation", back_populates="label_class")


# ── Annotations ───────────────────────────────────────────────────────────────

class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    image_id: Mapped[int] = mapped_column(Integer, ForeignKey("images.id"), nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, ForeignKey("classes.id"), nullable=True)
    shape_type: Mapped[str] = mapped_column(String(16), nullable=False)  # bbox | polygon
    coordinates: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    source: Mapped[str] = mapped_column(String(16), default="manual")  # manual | auto
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    image: Mapped["Image"] = relationship("Image", back_populates="annotations")
    label_class: Mapped["LabelClass"] = relationship("LabelClass", back_populates="annotations")
    creator: Mapped["User"] = relationship("User", back_populates="annotations")

    def get_coordinates(self) -> dict:
        return json.loads(self.coordinates or "{}")

    def set_coordinates(self, coords: dict | list) -> None:
        self.coordinates = json.dumps(coords)


# ── Model Versions ────────────────────────────────────────────────────────────

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    checkpoint_path: Mapped[str] = mapped_column(String(1024), default="")
    trained_on_count: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped["Project"] = relationship("Project", back_populates="model_versions")


# ── Training Jobs ─────────────────────────────────────────────────────────────

class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending|running|completed|failed
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    log: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    project: Mapped["Project"] = relationship("Project", back_populates="training_jobs")
