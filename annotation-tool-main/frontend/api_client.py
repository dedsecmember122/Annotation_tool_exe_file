"""
API client — thin wrapper around requests that handles auth tokens,
retries, and type-safe helpers for every backend endpoint.
"""
import io
import threading
from pathlib import Path
from typing import Any, Optional

import requests

from backend.app.core.config import get_settings

settings = get_settings()

# Endpoints that must never trigger a refresh-and-retry (avoids infinite
# recursion through refresh() itself, and login/signup have no token yet).
_NO_REFRESH_PATHS = {"/auth/login", "/auth/signup", "/auth/refresh"}


class APIError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"[{status}] {detail}")


class APIClient:
    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = base_url or f"http://{settings.API_HOST}:{settings.API_PORT}{settings.API_PREFIX}"
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._session = requests.Session()
        self._token_lock = threading.Lock()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _auth_headers(self) -> dict:
        if not self._access_token:
            return {}
        return {"Authorization": f"Bearer {self._access_token}"}

    def _handle(self, resp: requests.Response) -> Any:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(resp.status_code, str(detail))
        if resp.status_code == 204 or not resp.content:
            return None
        try:
            return resp.json()
        except Exception:
            return resp.content

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        """Send a request, transparently refreshing an expired access token
        and retrying once on a 401. Safe to call from multiple threads
        (training/download workers run on background QThreads)."""
        token_used = self._access_token
        resp = self._session.request(
            method, f"{self.base_url}{path}", headers=self._auth_headers(), **kwargs
        )
        if resp.status_code == 401 and self._refresh_token and path not in _NO_REFRESH_PATHS:
            with self._token_lock:
                # Only refresh if another thread hasn't already done so
                # since this request was sent.
                if self._access_token == token_used:
                    try:
                        self.refresh()
                    except APIError:
                        return resp  # refresh token itself is dead — surface original 401
            self._rewind_streams(kwargs.get("files"))
            resp = self._session.request(
                method, f"{self.base_url}{path}", headers=self._auth_headers(), **kwargs
            )
        return resp

    @staticmethod
    def _rewind_streams(files: Any) -> None:
        """Multipart file streams are consumed by the first attempt — seek
        them back to 0 so a retry after token refresh sends full content."""
        if not files:
            return
        entries = files.values() if isinstance(files, dict) else (v for _, v in files)
        for entry in entries:
            stream = entry[1] if isinstance(entry, tuple) else entry
            if hasattr(stream, "seek"):
                stream.seek(0)

    def _get(self, path: str, **kwargs: Any) -> Any:
        return self._handle(self._request("GET", path, **kwargs))

    def _post(self, path: str, json: Any = None, **kwargs: Any) -> Any:
        return self._handle(self._request("POST", path, json=json, **kwargs))

    def _put(self, path: str, json: Any = None, **kwargs: Any) -> Any:
        return self._handle(self._request("PUT", path, json=json, **kwargs))

    def _delete(self, path: str, **kwargs: Any) -> Any:
        return self._handle(self._request("DELETE", path, **kwargs))

    # ── Auth ─────────────────────────────────────────────────────────────────

    def signup(self, username: str, email: str, password: str, confirm: str) -> dict:
        data = self._post("/auth/signup", json={
            "username": username, "email": email,
            "password": password, "confirm_password": confirm,
        })
        return data

    def login(self, username: str, password: str) -> dict:
        data = self._post("/auth/login", json={"username": username, "password": password})
        self._access_token = data["access_token"]
        self._refresh_token = data["refresh_token"]
        return data

    def refresh(self) -> None:
        data = self._post("/auth/refresh", json={"refresh_token": self._refresh_token})
        self._access_token = data["access_token"]
        self._refresh_token = data["refresh_token"]

    def me(self) -> dict:
        return self._get("/auth/me")

    def logout(self) -> None:
        self._access_token = None
        self._refresh_token = None

    # ── Projects ─────────────────────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        return self._get("/projects")

    def create_project(self, name: str, description: str = "", threshold: int = 100) -> dict:
        return self._post("/projects", json={"name": name, "description": description, "autotrain_threshold": threshold})

    def get_project(self, project_id: int) -> dict:
        return self._get(f"/projects/{project_id}")

    def update_project(self, project_id: int, **kwargs: Any) -> dict:
        return self._put(f"/projects/{project_id}", json=kwargs)

    def delete_project(self, project_id: int) -> None:
        self._delete(f"/projects/{project_id}")

    # ── Classes ──────────────────────────────────────────────────────────────

    def list_classes(self, project_id: int) -> list[dict]:
        return self._get(f"/projects/{project_id}/classes")

    def create_class(self, project_id: int, name: str, color: str = "#FF6B35") -> dict:
        return self._post(f"/projects/{project_id}/classes", json={"name": name, "color_hex": color})

    def update_class(self, project_id: int, class_id: int, **kwargs: Any) -> dict:
        return self._put(f"/projects/{project_id}/classes/{class_id}", json=kwargs)

    def delete_class(self, project_id: int, class_id: int) -> None:
        self._delete(f"/projects/{project_id}/classes/{class_id}")

    # ── Images ────────────────────────────────────────────────────────────────

    def upload_images(self, project_id: int, file_paths: list[str]) -> list[dict]:
        files = []
        opened = []
        for fp in file_paths:
            f = open(fp, "rb")
            opened.append(f)
            files.append(("files", (Path(fp).name, f, "image/jpeg")))
        try:
            resp = self._request("POST", f"/projects/{project_id}/images", files=files)
            return self._handle(resp)
        finally:
            for f in opened:
                f.close()

    def list_images(self, project_id: int, status: Optional[str] = None) -> list[dict]:
        params = {}
        if status:
            params["status"] = status
        return self._get(f"/projects/{project_id}/images", params=params)

    def delete_image(self, image_id: int) -> None:
        self._delete(f"/images/{image_id}")

    def get_image_data(self, image_id: int) -> bytes:
        resp = self._request("GET", f"/images/{image_id}/data")
        if resp.status_code >= 400:
            raise APIError(resp.status_code, "Failed to fetch image data")
        return resp.content

    def get_image_thumbnail(self, image_id: int) -> bytes:
        resp = self._request("GET", f"/images/{image_id}/thumbnail")
        if resp.status_code >= 400:
            raise APIError(resp.status_code, "Failed to fetch image thumbnail")
        return resp.content

    # ── Annotations ───────────────────────────────────────────────────────────

    def get_annotations(self, image_id: int) -> list[dict]:
        return self._get(f"/images/{image_id}/annotations")

    def create_annotation(self, image_id: int, class_id: Optional[int],
                          shape_type: str, coordinates: Any,
                          source: str = "manual", confidence: float = 1.0) -> dict:
        return self._post(f"/images/{image_id}/annotations", json={
            "class_id": class_id,
            "shape_type": shape_type,
            "coordinates": coordinates,
            "source": source,
            "confidence": confidence,
        })

    def update_annotation(self, annotation_id: int, **kwargs: Any) -> dict:
        return self._put(f"/annotations/{annotation_id}", json=kwargs)

    def delete_annotation(self, annotation_id: int) -> None:
        self._delete(f"/annotations/{annotation_id}")

    # ── Export ────────────────────────────────────────────────────────────────

    @staticmethod
    def _raise_for_export_error(resp: requests.Response, fallback: str) -> None:
        try:
            detail = resp.json().get("detail", fallback)
        except Exception:
            detail = fallback
        raise APIError(resp.status_code, str(detail))

    def export_project(
        self, project_id: int, fmt: str = "coco",
        split: bool = False, train_pct: float = 0.8, val_pct: float = 0.1,
    ) -> bytes:
        params = {"format": fmt}
        if split:
            params.update({"split": "true", "train_pct": train_pct, "val_pct": val_pct})
        resp = self._request("GET", f"/projects/{project_id}/export", params=params)
        if resp.status_code >= 400:
            self._raise_for_export_error(resp, "Export failed")
        return resp.content

    def export_zip(
        self, project_id: int,
        split: bool = False, train_pct: float = 0.8, val_pct: float = 0.1,
    ) -> bytes:
        params = {}
        if split:
            params.update({"split": "true", "train_pct": train_pct, "val_pct": val_pct})
        resp = self._request("GET", f"/projects/{project_id}/export/zip", params=params)
        if resp.status_code >= 400:
            self._raise_for_export_error(resp, "Export zip failed")
        return resp.content

    # ── Training ──────────────────────────────────────────────────────────────

    def start_training(
        self,
        project_id: int,
        epochs: int = 60,
        train_split: float = 0.8,
        model_size: str = "n",
        img_size: int = 640,
    ) -> dict:
        return self._post(
            f"/projects/{project_id}/train",
            json={
                "epochs": epochs,
                "train_split": train_split,
                "model_size": model_size,
                "img_size": img_size,
            },
        )

    def cancel_training(self, project_id: int) -> dict:
        return self._post(f"/projects/{project_id}/train/cancel")

    def download_model(self, project_id: int) -> bytes:
        """Download the latest trained best.pt checkpoint as raw bytes."""
        resp = self._request("GET", f"/projects/{project_id}/model/download")
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise APIError(resp.status_code, str(detail))
        return resp.content

    def get_training_status(self, project_id: int) -> dict:
        return self._get(f"/projects/{project_id}/train/status")

    def get_annotation_stats(self, project_id: int) -> dict:
        return self._get(f"/projects/{project_id}/annotation-stats")

    def auto_annotate_image(self, image_id: int, confidence: float = 0.25) -> dict:
        return self._post(
            f"/images/{image_id}/auto-annotate",
            json={"confidence_threshold": confidence},
        )

    def auto_annotate_batch(self, project_id: int, confidence: float = 0.25) -> dict:
        return self._post(
            f"/projects/{project_id}/auto-annotate-batch",
            json={"confidence_threshold": confidence},
        )

    # ── Admin ───────────────────────────────────────────────────────────

    def admin_list_users(self) -> list[dict]:
        return self._get("/admin/users")

    def admin_create_user(self, username: str, email: str, password: str, role: str = "annotator") -> dict:
        return self._post("/admin/users", json={
            "username": username, "email": email, "password": password, "role": role,
        })

    def admin_set_role(self, user_id: int, role: str) -> dict:
        return self._post(f"/admin/users/{user_id}/role", json={"role": role})

    def admin_extend_trial(self, user_id: int, extra_days: int = 7) -> dict:
        return self._post(f"/admin/users/{user_id}/extend-trial", json={"extra_days": extra_days})

    def admin_reset_trial(self, user_id: int) -> dict:
        return self._post(f"/admin/users/{user_id}/reset-trial")

    # ── Health ─────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        resp = self._session.get(
            f"http://{settings.API_HOST}:{settings.API_PORT}/health"
        )
        return resp.json()
