"""
tasks.py
========
Celery application + the long-running rally-detection task.

Video analysis takes minutes and pins a GPU, so it must never run inside an API
request.  The API enqueues `process_rally_job`; a separate worker process (see
docker-compose / the `celery worker` command) executes it and streams progress
back into the Redis job store, which the API relays over WebSocket.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from celery import Celery

from . import jobs, storage
from .config import get_settings
from .pipeline_runner import run_rally_job
from .schemas import JobStatus, Segment

settings = get_settings()

celery_app = Celery(
    "rally_predictor",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)
celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,   # one heavy video at a time per worker
    task_acks_late=True,            # re-queue if a worker dies mid-job
)


@celery_app.task(name="process_rally_job", bind=True)
def process_rally_job(self, job_id: str) -> dict:
    store = storage.get_storage()
    job = jobs.get(job_id)
    if job is None:
        return {"job_id": job_id, "error": "job not found"}

    in_key = storage.input_key(job_id, job.filename or "match.mp4")
    out_key = storage.output_key(job_id)

    jobs.update(job_id, status=JobStatus.PROCESSING, progress=0.0,
                message="Starting analysis…")

    workdir = Path(tempfile.mkdtemp(prefix=f"rally_{job_id}_"))
    try:
        local_in = store.download_to(in_key, workdir / "input.mp4")
        local_out = workdir / "rallies.mp4"

        def _progress(frac: float, msg: str) -> None:
            jobs.update(job_id, progress=frac, message=msg)

        segments = run_rally_job(local_in, local_out, progress=_progress)

        store.upload_from(local_out, out_key, content_type="video/mp4")
        result_url = store.presigned_get(out_key)

        jobs.update(
            job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            message=f"{len(segments)} rallies detected",
            segments=[Segment(**s) for s in segments],
            result_url=result_url,
        )
        return {"job_id": job_id, "segments": len(segments)}

    except Exception as exc:  # noqa: BLE001 — surface any failure to the client
        jobs.update(
            job_id,
            status=JobStatus.FAILED,
            message="Processing failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
