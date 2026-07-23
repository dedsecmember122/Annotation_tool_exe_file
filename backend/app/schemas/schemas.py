"""
Pydantic schemas for request/response validation.
"""
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Auth ──────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=8)
    confirm_password: str

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info: Any) -> str:
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match")
        return v


class LoginRequest(BaseModel):
    username: str  # username or email accepted
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: str
    created_at: datetime
    trial_started_at: Optional[datetime] = None
    trial_days_remaining: Optional[int] = None

    model_config = {"from_attributes": True}


class TrialExtendRequest(BaseModel):
    """Admin-only: extend or reset a user's trial."""
    extra_days: int = Field(7, ge=1, le=365, description="Days from now to set as new expiry")


class RoleUpdateRequest(BaseModel):
    """Admin-only: change a user's role."""
    role: Literal["admin", "annotator"]


class AdminCreateUserRequest(BaseModel):
    """Admin-only: create a user account directly, without them
    self-signing-up. No confirm_password — this isn't filled in by the
    account's own owner."""
    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: Literal["admin", "annotator"] = "annotator"


# ── Projects ──────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: str = ""
    autotrain_threshold: int = 100


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    autotrain_threshold: Optional[int] = None


class ProjectOut(BaseModel):
    id: int
    name: str
    description: str
    owner_id: int
    class_list: list[str]
    storage_mode: str
    autotrain_threshold: int
    created_at: datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_project(cls, proj: Any) -> "ProjectOut":
        return cls(
            id=proj.id,
            name=proj.name,
            description=proj.description,
            owner_id=proj.owner_id,
            class_list=proj.get_class_list(),
            storage_mode=proj.storage_mode,
            autotrain_threshold=proj.autotrain_threshold,
            created_at=proj.created_at,
        )


# ── Images ────────────────────────────────────────────────────────────────────

class ImageOut(BaseModel):
    id: int
    project_id: int
    filename: str
    storage_path: str
    width: int
    height: int
    status: str
    uploaded_at: datetime
    annotation_count: int = 0

    model_config = {"from_attributes": True}


# ── Label Classes ─────────────────────────────────────────────────────────────

class ClassCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    color_hex: str = "#FF6B35"


class ClassUpdate(BaseModel):
    name: Optional[str] = None
    color_hex: Optional[str] = None


class ClassOut(BaseModel):
    id: int
    project_id: int
    name: str
    color_hex: str

    model_config = {"from_attributes": True}


# ── Annotations ───────────────────────────────────────────────────────────────

class AnnotationCreate(BaseModel):
    class_id: Optional[int] = None
    shape_type: str  # bbox | polygon
    coordinates: dict | list  # {"x1","y1","x2","y2"} or [[x,y], ...]
    source: str = "manual"
    confidence: float = 1.0


class AnnotationUpdate(BaseModel):
    class_id: Optional[int] = None
    coordinates: Optional[dict | list] = None
    reviewed: Optional[bool] = None


class AnnotationOut(BaseModel):
    id: int
    image_id: int
    class_id: Optional[int]
    shape_type: str
    coordinates: dict | list
    source: str
    confidence: float
    reviewed: bool
    created_by: Optional[int]
    created_at: datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_annotation(cls, ann: Any) -> "AnnotationOut":
        return cls(
            id=ann.id,
            image_id=ann.image_id,
            class_id=ann.class_id,
            shape_type=ann.shape_type,
            coordinates=ann.get_coordinates(),
            source=ann.source,
            confidence=ann.confidence,
            reviewed=ann.reviewed,
            created_by=ann.created_by,
            created_at=ann.created_at,
        )


# ── Training ──────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    epochs: int = 60
    # Fraction of images used for training; the rest go to validation.
    train_split: float = 0.8
    # Model capacity preset: n / s / m / l / x
    model_size: str = "n"
    # Network input resolution (square, multiple of 32)
    img_size: int = 640


class AutoAnnotateRequest(BaseModel):
    confidence_threshold: float = 0.15


class TrainingJobOut(BaseModel):
    id: int
    project_id: int
    status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    log: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ModelVersionOut(BaseModel):
    id: int
    project_id: int
    version_number: int
    checkpoint_path: str
    trained_on_count: int
    metrics: dict
    created_at: datetime

    model_config = {"from_attributes": True}
