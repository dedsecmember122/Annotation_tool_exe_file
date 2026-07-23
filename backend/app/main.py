"""
FastAPI application factory.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import annotations, auth, autoannotate, export, images, projects
from backend.app.core.config import get_settings
from backend.app.db import init_db

settings = get_settings()


def create_app() -> FastAPI:
    app = FastAPI(
        title="InSiSo Image Annotation Tool API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow localhost connections from the embedded frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.API_PREFIX

    app.include_router(auth.router, prefix=prefix)
    app.include_router(auth.admin_router, prefix=prefix)
    app.include_router(projects.router, prefix=prefix)
    app.include_router(images.router, prefix=prefix)
    app.include_router(annotations.router, prefix=prefix)
    app.include_router(autoannotate.router, prefix=prefix)
    app.include_router(export.router, prefix=prefix)

    @app.on_event("startup")
    def startup() -> None:
        init_db()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "env": settings.APP_ENV}

    return app


app = create_app()
