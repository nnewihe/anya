"""
config.py
=========
Central settings, read from environment variables (12-factor).  Defaults are
geared for local docker-compose dev; production overrides them via the
environment / ECS task definition.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


class Settings:
    # ── Service ────────────────────────────────────────────────────────────
    APP_NAME: str = "rally-predictor-api"
    ENV: str = os.environ.get("ENV", "dev")

    # ── Redis (Celery broker + job-state store) ────────────────────────────
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # ── Storage ────────────────────────────────────────────────────────────
    # STORAGE_BACKEND = "s3" (production) or "local" (dev without AWS).
    STORAGE_BACKEND: str = os.environ.get("STORAGE_BACKEND", "local")
    S3_BUCKET: str = os.environ.get("S3_BUCKET", "rally-predictor-media")
    S3_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
    # When using a local-stack / MinIO endpoint for testing.
    S3_ENDPOINT_URL: str | None = os.environ.get("S3_ENDPOINT_URL") or None
    PRESIGN_EXPIRY_SEC: int = int(os.environ.get("PRESIGN_EXPIRY_SEC", "3600"))

    # Local-storage root (only used when STORAGE_BACKEND=local).
    LOCAL_STORAGE_DIR: Path = Path(
        os.environ.get("LOCAL_STORAGE_DIR", "/data/rally-media")
    )

    # ── Pipeline ───────────────────────────────────────────────────────────
    # Where rally_detector.py + helpers + models/ live.
    # Defaults to pipeline/ at the repo root (three levels up from this file).
    PIPELINE_DIR: Path = Path(
        os.environ.get(
            "PIPELINE_DIR",
            str(Path(__file__).resolve().parents[2] / "pipeline"),
        )
    )
    # Process every Nth frame.  1 = every frame (most accurate, slowest).
    FRAME_STRIDE: int = int(os.environ.get("FRAME_STRIDE", "1"))

    # ── Uploads ────────────────────────────────────────────────────────────
    MAX_UPLOAD_BYTES: int = int(
        os.environ.get("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024 * 1024))  # 8 GiB
    )

    # ── CORS ───────────────────────────────────────────────────────────────
    # Comma-separated list; "*" allows all (fine for a mobile app with no
    # cookie auth, tighten in production if you add a web client).
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.environ.get("CORS_ORIGINS", "*").split(",")
        if o.strip()
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
