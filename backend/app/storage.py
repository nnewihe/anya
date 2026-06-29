"""
storage.py
==========
Abstraction over where raw uploads and finished rally reels live.

Two backends:
  • s3    — production.  Clients upload/download directly via presigned URLs so
            multi-gigabyte videos never stream through the API process.
  • local — dev.  Files live under LOCAL_STORAGE_DIR and are served by the API
            itself (see the /local-storage routes in main.py).

Both expose the same interface: object *keys* are opaque strings the worker
resolves to a concrete local file path (downloading from S3 first if needed).
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
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


class Storage(ABC):
    @abstractmethod
    def presigned_put(self, key: str, content_type: str) -> str: ...

    @abstractmethod
    def presigned_get(self, key: str) -> str: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def download_to(self, key: str, dest: Path) -> Path:
        """Make the object available as a local file at `dest`; return dest."""

    @abstractmethod
    def upload_from(self, src: Path, key: str, content_type: str = "video/mp4") -> None: ...


class LocalStorage(Storage):
    """Filesystem-backed storage for dev. Presigned URLs are API routes."""

    def __init__(self, settings: Settings):
        self.root = settings.LOCAL_STORAGE_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def presigned_put(self, key: str, content_type: str) -> str:
        # The client PUTs raw bytes to this API route (see main.py).
        return f"/local-storage/{key}"

    def presigned_get(self, key: str) -> str:
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


class S3Storage(Storage):
    """Production storage backed by AWS S3 with presigned URLs."""

    def __init__(self, settings: Settings):
        import boto3  # imported lazily so dev doesn't need boto3 installed

        self.settings = settings
        self.bucket = settings.S3_BUCKET
        self.client = boto3.client(
            "s3",
            region_name=settings.S3_REGION,
            endpoint_url=settings.S3_ENDPOINT_URL,
        )

    def presigned_put(self, key: str, content_type: str) -> str:
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=self.settings.PRESIGN_EXPIRY_SEC,
        )

    def presigned_get(self, key: str) -> str:
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.settings.PRESIGN_EXPIRY_SEC,
        )

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def download_to(self, key: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(dest))
        return dest

    def upload_from(self, src: Path, key: str, content_type: str = "video/mp4") -> None:
        self.client.upload_file(
            str(src), self.bucket, key, ExtraArgs={"ContentType": content_type}
        )


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        settings = get_settings()
        _storage = (
            S3Storage(settings)
            if settings.STORAGE_BACKEND == "s3"
            else LocalStorage(settings)
        )
    return _storage
