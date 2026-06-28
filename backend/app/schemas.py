"""
schemas.py
==========
Pydantic request/response models — the wire contract shared with the Flutter
client.  Keep field names stable; the Dart models mirror them.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"        # created, awaiting upload completion
    QUEUED = "queued"          # upload done, waiting for a worker
    PROCESSING = "processing"  # worker running rally_detector
    COMPLETED = "completed"    # rally reel ready
    FAILED = "failed"


class Segment(BaseModel):
    """One detected rally, in source-video seconds."""

    start: float
    end: float
    origin: str  # "near" | "far"


class CreateJobRequest(BaseModel):
    filename: str
    content_type: str = "video/mp4"
    source: str = "upload"  # "upload" | "live" | "gopro" — informational


class CreateJobResponse(BaseModel):
    job_id: str
    # Presigned PUT URL (S3) the client uploads the raw video to directly.
    # For local-storage dev this is a relative API path instead.
    upload_url: str
    # Storage key/path the server will read from once the upload completes.
    upload_method: str = "PUT"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    source: str = "upload"
    filename: Optional[str] = None
    progress: float = 0.0          # 0.0 – 1.0
    message: Optional[str] = None
    segments: list[Segment] = []
    # Presigned GET URL for the finished rally reel (None until completed).
    result_url: Optional[str] = None
    created_at: float
    updated_at: float
    error: Optional[str] = None
