"""
Images router:
  POST   /api/projects/{id}/images   — bulk upload
  GET    /api/projects/{id}/images   — list (with status filter)
  DELETE /api/images/{id}            — remove
  GET    /api/images/{id}/data       — serve raw bytes
"""
import io
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from backend.app.api.auth import current_user_dep
from backend.app.core.config import get_settings
from backend.app.core.storage.local_storage import LocalStorage
from backend.app.db import get_db
from backend.app.models.models import Annotation, Image, User
from backend.app.schemas.schemas import ImageOut

settings = get_settings()

router = APIRouter(tags=["images"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _get_storage():
    if settings.is_production:
        from backend.app.core.storage.cloud_storage import CloudStorage
        return CloudStorage()
    return LocalStorage()


def _image_out(img: Image, db: Session) -> ImageOut:
    count = db.query(Annotation).filter(Annotation.image_id == img.id).count()
    return ImageOut(
        id=img.id,
        project_id=img.project_id,
        filename=img.filename,
        storage_path=img.storage_path,
        width=img.width,
        height=img.height,
        status=img.status,
        uploaded_at=img.uploaded_at,
        annotation_count=count,
    )


@router.post("/projects/{project_id}/images", response_model=list[ImageOut], status_code=201)
def upload_images(
    project_id: int,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> list[ImageOut]:
    storage = _get_storage()
    created: list[Image] = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {suffix}")

        # Detect duplicates
        existing = db.query(Image).filter(
            Image.project_id == project_id,
            Image.filename == upload.filename,
        ).first()
        if existing:
            created.append(existing)
            continue

        file_bytes = upload.file.read()
        # Try to read image dimensions
        width, height = 0, 0
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(file_bytes))
            width, height = pil_img.size
        except Exception:
            pass

        path = storage.save_image(str(project_id), upload.filename or "image", io.BytesIO(file_bytes))
        img = Image(
            project_id=project_id,
            filename=upload.filename,
            storage_path=path,
            width=width,
            height=height,
            status="unannotated",
        )
        db.add(img)
        db.flush()
        created.append(img)

    db.commit()
    for img in created:
        db.refresh(img)
    return [_image_out(img, db) for img in created]


@router.get("/projects/{project_id}/images", response_model=list[ImageOut])
def list_images(
    project_id: int,
    status: str | None = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> list[ImageOut]:
    q = db.query(Image).filter(Image.project_id == project_id)
    if status:
        q = q.filter(Image.status == status)
    images = q.order_by(Image.uploaded_at.asc()).all()
    return [_image_out(img, db) for img in images]


@router.delete("/images/{image_id}", status_code=204)
def delete_image(
    image_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> None:
    img = db.query(Image).filter(Image.id == image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")
    storage = _get_storage()
    try:
        storage.delete_image(img.storage_path)
    except Exception:
        pass
    db.delete(img)
    db.commit()


@router.get("/images/{image_id}/data")
def get_image_data(
    image_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> Response:
    img = db.query(Image).filter(Image.id == image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")
    storage = _get_storage()
    data = storage.get_image_bytes(img.storage_path)
    suffix = Path(img.filename).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".webp": "image/webp",
    }
    return Response(content=data, media_type=media_types.get(suffix, "application/octet-stream"))


THUMBNAIL_MAX_SIZE = (240, 240)


@router.get("/images/{image_id}/thumbnail")
def get_image_thumbnail(
    image_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> Response:
    """Small, pre-resized preview for gallery views. Downloading and
    decoding the full-resolution original just to shrink it to a 160x120
    icon is what was making the project gallery slow and memory-heavy on
    every open/filter/refresh — this resizes server-side instead so the
    client only ever transfers and decodes a few KB per image."""
    img = db.query(Image).filter(Image.id == image_id).first()
    if not img:
        raise HTTPException(404, "Image not found")
    storage = _get_storage()
    data = storage.get_image_bytes(img.storage_path)

    from PIL import Image as PILImage
    pil_img = PILImage.open(io.BytesIO(data))
    pil_img.thumbnail(THUMBNAIL_MAX_SIZE, PILImage.LANCZOS)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=80)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
