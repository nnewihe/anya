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
    # URL the client PUTs the raw video to.
    # For local-storage dev this is a relative API path instead.
    upload_url: str
    # Storage key/path the server will read from once the upload completes.
    upload_method: str = "PUT"


class AddClipRequest(BaseModel):
    """Add one clip to a multi-clip job. Returns an upload URL for that clip."""
    filename: str
    content_type: str = "video/mp4"
    clip_index: int  # ordering index so the worker concatenates in the right order


class AddClipResponse(BaseModel):
    clip_key: str
    upload_url: str
    upload_method: str = "PUT"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    source: str = "upload"
    filename: Optional[str] = None
    # Storage keys for individual clips (multi-clip flow); empty = single-file flow.
    clip_keys: list[str] = []
    progress: float = 0.0          # 0.0 – 1.0
    message: Optional[str] = None
    segments: list[Segment] = []
    # Download URL for the finished rally reel (None until completed).
    result_url: Optional[str] = None
    created_at: float
    updated_at: float
    error: Optional[str] = None
