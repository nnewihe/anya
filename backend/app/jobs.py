"""
jobs.py
=======
Redis-backed job-state store.  A job is a single JSON blob keyed by job_id.
The API process and the Celery worker both read/write it; the API also
SUBSCRIBEs to a per-job pub/sub channel to push live progress over WebSocket.
"""

from __future__ import annotations

import json
import time
import uuid

import redis

from .config import get_settings
from .schemas import Job, JobStatus, Segment

_KEY_PREFIX = "job:"
_CHANNEL_PREFIX = "job-events:"
_JOB_TTL_SEC = 7 * 24 * 3600  # keep finished jobs queryable for a week


def _client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().REDIS_URL, decode_responses=True)


def _key(job_id: str) -> str:
    return f"{_KEY_PREFIX}{job_id}"


def channel(job_id: str) -> str:
    return f"{_CHANNEL_PREFIX}{job_id}"


def new_job_id() -> str:
    return uuid.uuid4().hex


def create(job_id: str, source: str, filename: str) -> Job:
    now = time.time()
    job = Job(
        job_id=job_id,
        status=JobStatus.PENDING,
        source=source,
        filename=filename,
        created_at=now,
        updated_at=now,
    )
    _save(job)
    return job


def get(job_id: str) -> Job | None:
    raw = _client().get(_key(job_id))
    if raw is None:
        return None
    return Job.model_validate_json(raw)


def _save(job: Job) -> None:
    job.updated_at = time.time()
    r = _client()
    r.set(_key(job.job_id), job.model_dump_json(), ex=_JOB_TTL_SEC)
    # Notify any WebSocket subscribers of the new state.
    r.publish(channel(job.job_id), job.model_dump_json())


def update(
    job_id: str,
    *,
    status: JobStatus | None = None,
    progress: float | None = None,
    message: str | None = None,
    segments: list[Segment] | None = None,
    clip_keys: list[str] | None = None,
    result_url: str | None = None,
    error: str | None = None,
) -> Job | None:
    job = get(job_id)
    if job is None:
        return None
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if message is not None:
        job.message = message
    if segments is not None:
        job.segments = segments
    if clip_keys is not None:
        job.clip_keys = clip_keys
    if result_url is not None:
        job.result_url = result_url
    if error is not None:
        job.error = error
    _save(job)
    return job


def subscribe(job_id: str) -> redis.client.PubSub:
    """Return a PubSub already subscribed to this job's event channel."""
    pubsub = _client().pubsub()
    pubsub.subscribe(channel(job_id))
    return pubsub
