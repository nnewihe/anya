"""
config.py
=========
Central settings, read from environment variables (12-factor).  Defaults are
geared for local docker-compose dev; a deployment overrides them via the
environment.
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
    # Uploads and finished reels live on the local filesystem.
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
