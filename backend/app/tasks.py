"""
tasks.py
========
Celery application + the long-running rally-detection task.

Video analysis takes minutes and pins a GPU, so it must never run inside an API
request.  The API enqueues `process_rally_job`; a separate worker process (see
docker-compose / the `celery worker` command) executes it and streams progress
back into the Redis job store, which the API relays over WebSocket.

Multi-clip support
------------------
If the job has `clip_keys` set (a list of storage keys for individual GoPro
clips), the worker concatenates them with ffmpeg (stream-copy, lossless) before
running the detector.  This mirrors what run_pipeline.py does locally.

Calibration cache persistence
------------------------------
Court-corner and exclusion-zone caches are stored as side-car files alongside
the input video in storage.  They are restored into the work dir before analysis
and saved back after, so subsequent runs of the same video start warm.
"""

from __future__ import annotations

import json
import shutil
import subprocess
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

# Calibration side-car file stems (relative to the processed video's name stem).
_CACHE_STEMS = ("_court_cache.json", "_exclusion_cache.json")


def _restore_caches(store: storage.Storage, job_id: str, workdir: Path, stem: str) -> None:
    """Download any cached calibration files from storage into the work dir."""
    for sfx in _CACHE_STEMS:
        key = storage.cache_key(job_id, sfx)
        if store.exists(key):
            try:
                store.download_to(key, workdir / f"{stem}{sfx}")
            except Exception:
                pass  # missing cache is fine — pipeline will recompute


def _save_caches(store: storage.Storage, job_id: str, workdir: Path, stem: str) -> None:
    """Upload freshly computed calibration files back to storage."""
    for sfx in _CACHE_STEMS:
        src = workdir / f"{stem}{sfx}"
        if src.exists():
            try:
                store.upload_from(src, storage.cache_key(job_id, sfx), content_type="application/json")
            except Exception:
                pass  # non-fatal: next run will recompute


def _ffmpeg_concat(clip_paths: list[Path], output: Path) -> None:
    """Stream-copy concatenate clips in order using ffmpeg (lossless, fast)."""
    list_file = output.parent / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            safe = str(p).replace("'", r"'\''")
            f.write(f"file '{safe}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(output),
    ]
    result = subprocess.run(cmd, capture_output=True)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr.decode()[:500]}")


@celery_app.task(name="process_rally_job", bind=True)
def process_rally_job(self, job_id: str) -> dict:
    store = storage.get_storage()
    job = jobs.get(job_id)
    if job is None:
        return {"job_id": job_id, "error": "job not found"}

    out_key = storage.output_key(job_id)
    jobs.update(job_id, status=JobStatus.PROCESSING, progress=0.0,
                message="Starting analysis…")

    workdir = Path(tempfile.mkdtemp(prefix=f"rally_{job_id}_"))
    try:
        # ── Assemble input video ─────────────────────────────────────────
        clip_keys = job.clip_keys or []
        if clip_keys:
            # Multi-clip: download each clip and concatenate.
            jobs.update(job_id, message=f"Downloading {len(clip_keys)} clip(s)…")
            clip_paths = []
            for i, ck in enumerate(clip_keys):
                dest = workdir / f"clip_{i:03d}.mp4"
                store.download_to(ck, dest)
                clip_paths.append(dest)
            jobs.update(job_id, message="Concatenating clips…")
            local_in = workdir / "input.mp4"
            _ffmpeg_concat(clip_paths, local_in)
            # Free clip copies after concat.
            for p in clip_paths:
                p.unlink(missing_ok=True)
        else:
            # Single-file upload (original flow).
            in_key = storage.input_key(job_id, job.filename or "match.mp4")
            local_in = store.download_to(in_key, workdir / "input.mp4")

        # ── Restore calibration caches ───────────────────────────────────
        _restore_caches(store, job_id, workdir, stem="input")

        # ── Run the detector ─────────────────────────────────────────────
        local_out = workdir / "rallies.mp4"

        def _progress(frac: float, msg: str) -> None:
            jobs.update(job_id, progress=frac, message=msg)

        segments = run_rally_job(local_in, local_out, progress=_progress)

        # ── Persist calibration caches for next run ──────────────────────
        _save_caches(store, job_id, workdir, stem="input")

        # ── Upload result ────────────────────────────────────────────────
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
        shutil.rmtree(workdir, ignore_errors=True)
