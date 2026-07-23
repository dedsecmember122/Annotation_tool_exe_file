"""
Annotations router:
  GET/POST  /api/images/{id}/annotations
  PUT/DELETE /api/annotations/{id}
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.api.auth import current_user_dep
from backend.app.db import get_db
from backend.app.models.models import Annotation, Image, User
from backend.app.schemas.schemas import AnnotationCreate, AnnotationOut, AnnotationUpdate

router = APIRouter(tags=["annotations"])


def _update_image_status(image_id: int, db: Session) -> None:
    img = db.query(Image).filter(Image.id == image_id).first()
    if not img:
        return
    annotations = db.query(Annotation).filter(Annotation.image_id == image_id).all()
    if not annotations:
        img.status = "unannotated"
    elif any(a.source == "auto" and not a.reviewed for a in annotations):
        img.status = "auto_annotated"
    else:
        img.status = "annotated"
    db.flush()


@router.get("/images/{image_id}/annotations", response_model=list[AnnotationOut])
def get_annotations(
    image_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> list[AnnotationOut]:
    img = db.query(Image).filter(Image.id == image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")
    annotations = db.query(Annotation).filter(Annotation.image_id == image_id).all()
    return [AnnotationOut.from_orm_annotation(a) for a in annotations]


@router.post("/images/{image_id}/annotations", response_model=AnnotationOut, status_code=201)
def create_annotation(
    image_id: int,
    body: AnnotationCreate,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> AnnotationOut:
    img = db.query(Image).filter(Image.id == image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")
    ann = Annotation(
        image_id=image_id,
        class_id=body.class_id,
        shape_type=body.shape_type,
        source=body.source,
        confidence=body.confidence,
        created_by=user.id,
        reviewed=(body.source == "manual"),
    )
    ann.set_coordinates(body.coordinates)
    db.add(ann)
    db.flush()
    _update_image_status(image_id, db)
    db.commit()
    db.refresh(ann)
    return AnnotationOut.from_orm_annotation(ann)


@router.put("/annotations/{annotation_id}", response_model=AnnotationOut)
def update_annotation(
    annotation_id: int,
    body: AnnotationUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> AnnotationOut:
    ann = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not ann:
        raise HTTPException(404, "Annotation not found")
    if body.class_id is not None:
        ann.class_id = body.class_id
    if body.coordinates is not None:
        ann.set_coordinates(body.coordinates)
    if body.reviewed is not None:
        ann.reviewed = body.reviewed
    db.flush()
    _update_image_status(ann.image_id, db)
    db.commit()
    db.refresh(ann)
    return AnnotationOut.from_orm_annotation(ann)


@router.delete("/annotations/{annotation_id}", status_code=204)
def delete_annotation(
    annotation_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> None:
    ann = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not ann:
        raise HTTPException(404, "Annotation not found")
    image_id = ann.image_id
    db.delete(ann)
    db.flush()
    _update_image_status(image_id, db)
    db.commit()
