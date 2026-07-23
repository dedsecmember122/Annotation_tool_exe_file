"""
Projects router: GET/POST /api/projects, GET/PUT/DELETE /api/projects/{id}
Classes router: GET/POST /api/projects/{id}/classes
"""
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.auth import current_user_dep
from backend.app.db import get_db
from backend.app.models.models import LabelClass, Project, User
from backend.app.schemas.schemas import (
    ClassCreate,
    ClassOut,
    ClassUpdate,
    ProjectCreate,
    ProjectOut,
    ProjectUpdate,
)

router = APIRouter(prefix="/projects", tags=["projects"])


def _get_project_or_404(project_id: int, db: Session) -> Project:
    p = db.query(Project).filter(Project.id == project_id).first()
    if not p:
        raise HTTPException(404, "Project not found")
    return p


# ── Projects ──────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ProjectOut])
def list_projects(
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> list[ProjectOut]:
    if user.role == "admin":
        projects = db.query(Project).all()
    else:
        projects = db.query(Project).filter(Project.owner_id == user.id).all()
    return [ProjectOut.from_orm_project(p) for p in projects]


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(
    body: ProjectCreate,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> ProjectOut:
    p = Project(
        name=body.name,
        description=body.description,
        owner_id=user.id,
        autotrain_threshold=body.autotrain_threshold,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return ProjectOut.from_orm_project(p)


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> ProjectOut:
    return ProjectOut.from_orm_project(_get_project_or_404(project_id, db))


@router.put("/{project_id}", response_model=ProjectOut)
def update_project(
    project_id: int,
    body: ProjectUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> ProjectOut:
    p = _get_project_or_404(project_id, db)
    if user.role != "admin" and p.owner_id != user.id:
        raise HTTPException(403, "Not authorized")
    if body.name is not None:
        p.name = body.name
    if body.description is not None:
        p.description = body.description
    if body.autotrain_threshold is not None:
        p.autotrain_threshold = body.autotrain_threshold
    db.commit()
    db.refresh(p)
    return ProjectOut.from_orm_project(p)


@router.delete("/{project_id}", status_code=204)
def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> None:
    p = _get_project_or_404(project_id, db)
    if user.role != "admin" and p.owner_id != user.id:
        raise HTTPException(403, "Not authorized")
    db.delete(p)
    db.commit()


# ── Classes ───────────────────────────────────────────────────────────────────

@router.get("/{project_id}/classes", response_model=list[ClassOut])
def list_classes(
    project_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> list[LabelClass]:
    _get_project_or_404(project_id, db)
    return db.query(LabelClass).filter(LabelClass.project_id == project_id).all()


@router.post("/{project_id}/classes", response_model=ClassOut, status_code=201)
def create_class(
    project_id: int,
    body: ClassCreate,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> LabelClass:
    _get_project_or_404(project_id, db)
    cls = LabelClass(project_id=project_id, name=body.name, color_hex=body.color_hex)
    db.add(cls)
    db.commit()
    db.refresh(cls)
    return cls


@router.put("/{project_id}/classes/{class_id}", response_model=ClassOut)
def update_class(
    project_id: int,
    class_id: int,
    body: ClassUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> LabelClass:
    cls = db.query(LabelClass).filter(
        LabelClass.id == class_id, LabelClass.project_id == project_id
    ).first()
    if not cls:
        raise HTTPException(404, "Class not found")
    if body.name is not None:
        cls.name = body.name
    if body.color_hex is not None:
        cls.color_hex = body.color_hex
    db.commit()
    db.refresh(cls)
    return cls


@router.delete("/{project_id}/classes/{class_id}", status_code=204)
def delete_class(
    project_id: int,
    class_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> None:
    cls = db.query(LabelClass).filter(
        LabelClass.id == class_id, LabelClass.project_id == project_id
    ).first()
    if not cls:
        raise HTTPException(404, "Class not found")
    db.delete(cls)
    db.commit()
