"""
storage.py
==========
Where raw uploads and finished rally reels live.

Storage is filesystem-backed: files live under LOCAL_STORAGE_DIR and are served
by the API itself (see the /local-storage routes in main.py). Object *keys* are
opaque strings the worker resolves to a concrete local file path.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import Settings, get_settings


def input_key(job_id: str, filename: str) -> str:
    suffix = Path(filename).suffix or ".mp4"
    return f"inputs/{job_id}{suffix}"


def output_key(job_id: str) -> str:
    return f"outputs/{job_id}_rallies.mp4"


# Calibration side-car files stored alongside the input video.
# The pipeline writes these next to whatever file it processes; we mirror them
# into storage so subsequent runs (of the same job's video) start warm.
_CACHE_SUFFIXES = ("_court_cache.json", "_exclusion_cache.json", "_active_zone_config.json")


def cache_key(job_id: str, suffix: str) -> str:
    return f"inputs/{job_id}{suffix}"


class LocalStorage:
    """Filesystem-backed storage. Upload/download URLs are API routes."""

    def __init__(self, settings: Settings):
        self.root = settings.LOCAL_STORAGE_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def upload_url(self, key: str, content_type: str) -> str:
        # The client PUTs raw bytes to this API route (see main.py).
        return f"/local-storage/{key}"

    def download_url(self, key: str) -> str:
        return f"/local-storage/{key}"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def download_to(self, key: str, dest: Path) -> Path:
        src = self._path(key)
        if src.resolve() == dest.resolve():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest

    def upload_from(self, src: Path, key: str, content_type: str = "video/mp4") -> None:
        dst = self._path(key)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    # Local-only helpers used by the API's PUT/GET passthrough routes.
    def local_path(self, key: str) -> Path:
        return self._path(key)


_storage: LocalStorage | None = None


def get_storage() -> LocalStorage:
    global _storage
    if _storage is None:
        _storage = LocalStorage(get_settings())
    return _storage
