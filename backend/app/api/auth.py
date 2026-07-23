"""
Auth router: /api/auth/signup | /login | /refresh | /me
Admin: /api/admin/users (GET list, POST create) | /api/admin/users/{id}/role
       | /api/admin/users/{id}/extend-trial | /api/admin/users/{id}/reset-trial
"""
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from backend.app.db import get_db
from backend.app.models.models import TRIAL_PERIOD_DAYS, User
from backend.app.schemas.schemas import (
    AdminCreateUserRequest,
    LoginRequest,
    RefreshRequest,
    RoleUpdateRequest,
    SignupRequest,
    TokenResponse,
    TrialExtendRequest,
    UserOut,
)

router = APIRouter(prefix="/auth", tags=["auth"])
admin_router = APIRouter(prefix="/admin", tags=["admin"])


def _get_current_user(token: str, db: Session) -> User:
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


def get_token_from_header(authorization: str = "") -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    return authorization[7:]


from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def current_user_dep(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return _get_current_user(credentials.credentials, db)


def admin_only_dep(user: User = Depends(current_user_dep)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(body: SignupRequest, db: Session = Depends(get_db)) -> User:
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(400, "Username already taken")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    is_first = db.query(User).count() == 0
    user = User(
        username=body.username,
        email=body.email,
        password_hash=hash_password(body.password),
        role="admin" if is_first else "annotator",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = (
        db.query(User)
        .filter((User.username == body.username) | (User.email == body.username))
        .first()
    )
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    if not user.is_active:
        raise HTTPException(403, "Account deactivated")

    # ── Trial management ──────────────────────────────────────────────────────
    if user.role != "admin":
        # Stamp trial start on first login
        if user.trial_started_at is None:
            user.trial_started_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(user)
        # Enforce expiry
        if user.is_trial_expired:
            raise HTTPException(
                403,
                f"Your {TRIAL_PERIOD_DAYS}-day free trial has expired. Please contact the administrator to extend access."
            )

    return TokenResponse(
        access_token=create_access_token(user.id, user.role),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)) -> TokenResponse:
    payload = decode_token(body.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(401, "Invalid refresh token")
    user = db.query(User).filter(User.id == int(payload["sub"])).first()
    if not user or not user.is_active:
        raise HTTPException(401, "User not found")
    # Without this, a user logged in before their trial expired could keep
    # refreshing (refresh tokens last 7 days) straight through expiry —
    # login() enforces it, but refresh() previously didn't, so expiry was
    # only ever checked once, at the very first login.
    if user.role != "admin" and user.is_trial_expired:
        raise HTTPException(
            403,
            f"Your {TRIAL_PERIOD_DAYS}-day free trial has expired. Please contact the administrator to extend access."
        )
    return TokenResponse(
        access_token=create_access_token(user.id, user.role),
        refresh_token=create_refresh_token(user.id),
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user_dep)) -> User:
    return user


# ── Admin endpoints ───────────────────────────────────────────────────────────

@admin_router.get("/users", response_model=list[UserOut])
def list_users(
    db: Session = Depends(get_db),
    admin: User = Depends(admin_only_dep),
) -> list[User]:
    """List all users (admin only)."""
    return db.query(User).order_by(User.created_at.asc()).all()


@admin_router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def admin_create_user(
    body: AdminCreateUserRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(admin_only_dep),
) -> User:
    """Create a user account directly (admin only) — provisions a teammate
    without them having to self-signup first."""
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(400, "Username already taken")
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(400, "Email already registered")
    user = User(
        username=body.username,
        email=body.email,
        password_hash=hash_password(body.password),
        role=body.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@admin_router.post("/users/{user_id}/role", response_model=UserOut)
def set_user_role(
    user_id: int,
    body: RoleUpdateRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(admin_only_dep),
) -> User:
    """Promote/demote a user's role (admin only). An admin cannot change
    their own role — that has to be done by a different admin, so an
    account can never demote itself into having no admins left."""
    if user_id == admin.id:
        raise HTTPException(400, "You cannot change your own role. Ask another admin.")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.role = body.role
    if body.role == "admin":
        # Admins never expire — clear any leftover trial state so
        # trial_expires_at reads as None rather than a stale date.
        user.trial_started_at = None
        user.trial_extended_until = None
    db.commit()
    db.refresh(user)
    return user


@admin_router.post("/users/{user_id}/extend-trial", response_model=UserOut)
def extend_trial(
    user_id: int,
    body: TrialExtendRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(admin_only_dep),
) -> User:
    """Extend a user's trial by N days from now (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    if user.role == "admin":
        raise HTTPException(400, "Admin accounts do not have trials")
    user.trial_extended_until = datetime.now(timezone.utc) + timedelta(days=body.extra_days)
    db.commit()
    db.refresh(user)
    return user


@admin_router.post("/users/{user_id}/reset-trial", response_model=UserOut)
def reset_trial(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(admin_only_dep),
) -> User:
    """Reset a user's trial to start fresh from now (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    if user.role == "admin":
        raise HTTPException(400, "Admin accounts do not have trials")
    user.trial_started_at = datetime.now(timezone.utc)
    user.trial_extended_until = None
    db.commit()
    db.refresh(user)
    return user
