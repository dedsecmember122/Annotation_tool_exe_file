"""
Export router:
  GET /api/projects/{id}/export?format=coco|yolo|voc
  GET /api/projects/{id}/export/zip
"""
import io
import json
import random
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from backend.app.api.auth import current_user_dep
from backend.app.core.config import get_settings
from backend.app.core.storage.local_storage import LocalStorage
from backend.app.db import get_db
from backend.app.models.models import Annotation, Image, LabelClass, Project, User

settings = get_settings()
router = APIRouter(tags=["export"])


def _get_project(project_id: int, db: Session) -> Project:
    p = db.query(Project).filter(Project.id == project_id).first()
    if not p:
        raise HTTPException(404, "Project not found")
    return p


def _split_images(images: list[Image], train_pct: float, val_pct: float) -> dict[str, list[Image]]:
    """Randomly partition images into train/val/test buckets by the given
    percentages (test gets whatever's left). Same shuffle+cutoff approach
    CustomModelAdapter already uses for its internal train/val split,
    extended to a third bucket for dataset export."""
    items = list(images)
    random.shuffle(items)
    n = len(items)
    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def _get_storage():
    if settings.is_production:
        from backend.app.core.storage.cloud_storage import CloudStorage
        return CloudStorage()
    return LocalStorage()


# ── COCO JSON ─────────────────────────────────────────────────────────────────

def _export_coco(project: Project, images: list[Image], db: Session) -> dict:
    classes = db.query(LabelClass).filter(LabelClass.project_id == project.id).all()
    class_map = {c.id: i + 1 for i, c in enumerate(classes)}

    coco = {
        "info": {"description": project.name, "version": "1.0"},
        "licenses": [],
        "categories": [
            {"id": class_map[c.id], "name": c.name, "supercategory": "object"}
            for c in classes
        ],
        "images": [],
        "annotations": [],
    }
    ann_id = 1
    for img in images:
        coco["images"].append({
            "id": img.id,
            "file_name": img.filename,
            "width": img.width,
            "height": img.height,
        })
        annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
        for ann in annotations:
            coords = ann.get_coordinates()
            if ann.shape_type == "bbox":
                x1, y1, x2, y2 = coords.get("x1", 0), coords.get("y1", 0), coords.get("x2", 0), coords.get("y2", 0)
                w, h = x2 - x1, y2 - y1
                segmentation = [[x1, y1, x2, y1, x2, y2, x1, y2]]
            else:
                pts = coords if isinstance(coords, list) else coords.get("points", [])
                segmentation = [[c for pt in pts for c in pt]]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1, y1 = min(xs), min(ys)
                w, h = max(xs) - x1, max(ys) - y1

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img.id,
                "category_id": class_map.get(ann.class_id, 0),
                "segmentation": segmentation,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1
    return coco


# ── YOLO TXT ──────────────────────────────────────────────────────────────────

def _export_yolo(project: Project, images: list[Image], db: Session) -> dict[str, str]:
    classes = db.query(LabelClass).filter(LabelClass.project_id == project.id).all()
    class_index = {c.id: i for i, c in enumerate(classes)}
    files: dict[str, str] = {}
    files["classes.txt"] = "\n".join(c.name for c in classes)
    for img in images:
        lines: list[str] = []
        annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
        iw, ih = max(img.width, 1), max(img.height, 1)
        for ann in annotations:
            coords = ann.get_coordinates()
            cidx = class_index.get(ann.class_id, 0)
            if ann.shape_type == "bbox":
                x1, y1 = coords.get("x1", 0), coords.get("y1", 0)
                x2, y2 = coords.get("x2", 0), coords.get("y2", 0)
                cx = ((x1 + x2) / 2) / iw
                cy = ((y1 + y2) / 2) / ih
                bw = (x2 - x1) / iw
                bh = (y2 - y1) / ih
                lines.append(f"{cidx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        stem = Path(img.filename).stem
        files[f"{stem}.txt"] = "\n".join(lines)
    return files


# ── Pascal VOC XML ────────────────────────────────────────────────────────────

def _export_voc_single(img: Image, classes: dict, db: Session) -> str:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = img.filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img.width)
    ET.SubElement(size, "height").text = str(img.height)
    ET.SubElement(size, "depth").text = "3"

    annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
    for ann in annotations:
        coords = ann.get_coordinates()
        obj = ET.SubElement(root, "object")
        cls_name = classes.get(ann.class_id, "unknown")
        ET.SubElement(obj, "name").text = cls_name
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        if ann.shape_type == "bbox":
            ET.SubElement(bndbox, "xmin").text = str(int(coords.get("x1", 0)))
            ET.SubElement(bndbox, "ymin").text = str(int(coords.get("y1", 0)))
            ET.SubElement(bndbox, "xmax").text = str(int(coords.get("x2", 0)))
            ET.SubElement(bndbox, "ymax").text = str(int(coords.get("y2", 0)))
        else:
            pts = coords if isinstance(coords, list) else coords.get("points", [])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ET.SubElement(bndbox, "xmin").text = str(int(min(xs)))
            ET.SubElement(bndbox, "ymin").text = str(int(min(ys)))
            ET.SubElement(bndbox, "xmax").text = str(int(max(xs)))
            ET.SubElement(bndbox, "ymax").text = str(int(max(ys)))
    return ET.tostring(root, encoding="unicode")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/export")
def export_project(
    project_id: int,
    format: str = Query("coco", pattern="^(coco|yolo|voc)$"),
    split: bool = Query(False, description="Split into train/val/test sets"),
    train_pct: float = Query(0.8, ge=0.0, le=1.0),
    val_pct: float = Query(0.1, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> Response:
    if split and train_pct + val_pct > 1.0:
        raise HTTPException(400, "train_pct + val_pct cannot exceed 1.0")

    project = _get_project(project_id, db)
    images = db.query(Image).filter(Image.project_id == project_id).all()

    if not split:
        if format == "coco":
            data = json.dumps(_export_coco(project, images, db), indent=2)
            return Response(
                content=data,
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="{project.name}_coco.json"'},
            )
        elif format == "yolo":
            files = _export_yolo(project, images, db)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for fname, content in files.items():
                    zf.writestr(fname, content)
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{project.name}_yolo.zip"'},
            )
        else:  # voc
            classes = db.query(LabelClass).filter(LabelClass.project_id == project_id).all()
            cls_map = {c.id: c.name for c in classes}
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for img in images:
                    xml_str = _export_voc_single(img, cls_map, db)
                    stem = Path(img.filename).stem
                    zf.writestr(f"{stem}.xml", xml_str)
            buf.seek(0)
            return StreamingResponse(
                buf,
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{project.name}_voc.zip"'},
            )

    # ── Split export: always a zip, one subset per train/val/test ──────────────
    buckets = _split_images(images, train_pct, val_pct)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if format == "coco":
            for subset, subset_images in buckets.items():
                if not subset_images:
                    continue
                data = json.dumps(_export_coco(project, subset_images, db), indent=2)
                zf.writestr(f"{subset}.json", data)
        elif format == "yolo":
            # classes.txt is project-wide, independent of which images are
            # passed in — write it once at the zip root rather than once
            # per subset.
            classes = db.query(LabelClass).filter(LabelClass.project_id == project_id).all()
            zf.writestr("classes.txt", "\n".join(c.name for c in classes))
            for subset, subset_images in buckets.items():
                if not subset_images:
                    continue
                files = _export_yolo(project, subset_images, db)
                for fname, content in files.items():
                    if fname != "classes.txt":
                        zf.writestr(f"{subset}/{fname}", content)
        else:  # voc
            classes = db.query(LabelClass).filter(LabelClass.project_id == project_id).all()
            cls_map = {c.id: c.name for c in classes}
            for subset, subset_images in buckets.items():
                for img in subset_images:
                    xml_str = _export_voc_single(img, cls_map, db)
                    stem = Path(img.filename).stem
                    zf.writestr(f"{subset}/{stem}.xml", xml_str)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{project.name}_{format}_split.zip"'},
    )


@router.get("/projects/{project_id}/export/zip")
def export_zip(
    project_id: int,
    split: bool = Query(False, description="Split into train/val/test sets"),
    train_pct: float = Query(0.8, ge=0.0, le=1.0),
    val_pct: float = Query(0.1, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
    user: User = Depends(current_user_dep),
) -> StreamingResponse:
    if split and train_pct + val_pct > 1.0:
        raise HTTPException(400, "train_pct + val_pct cannot exceed 1.0")

    project = _get_project(project_id, db)
    images = db.query(Image).filter(Image.project_id == project_id).all()
    storage = _get_storage()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Class list (shared, project-wide — not split)
        classes = db.query(LabelClass).filter(LabelClass.project_id == project_id).all()
        cls_info = [{"id": c.id, "name": c.name, "color": c.color_hex} for c in classes]
        zf.writestr("classes.json", json.dumps(cls_info, indent=2))

        if not split:
            coco_data = json.dumps(_export_coco(project, images, db), indent=2)
            zf.writestr("annotations/coco.json", coco_data)
            for img in images:
                try:
                    img_bytes = storage.get_image_bytes(img.storage_path)
                    zf.writestr(f"images/{img.filename}", img_bytes)
                except Exception:
                    pass
        else:
            buckets = _split_images(images, train_pct, val_pct)
            for subset, subset_images in buckets.items():
                if not subset_images:
                    continue
                coco_data = json.dumps(_export_coco(project, subset_images, db), indent=2)
                zf.writestr(f"annotations/{subset}.json", coco_data)
                for img in subset_images:
                    try:
                        img_bytes = storage.get_image_bytes(img.storage_path)
                        zf.writestr(f"images/{subset}/{img.filename}", img_bytes)
                    except Exception:
                        pass

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{project.name}_dataset.zip"'},
    )
