"""
main.py
=======
FastAPI surface for the rally-predictor service.

Flow (upload):
  1. POST /jobs                → {job_id, upload_url}
  2. PUT  <upload_url>         → client uploads raw video (via the
                                 /local-storage passthrough route)
  3. POST /jobs/{id}/start     → enqueues the Celery analysis task
  4. GET  /jobs/{id}           → poll status / segments / result_url
     WS  /jobs/{id}/events     → live progress push (preferred)

Flow (live):
  WS /live/{id}               → client streams encoded video chunks; on close
                                the server runs the same pipeline on the
                                assembled file.  See live.py.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from . import jobs, storage
from .config import get_settings
from .live import register_live_routes
from .schemas import AddClipRequest, AddClipResponse, CreateJobRequest, CreateJobResponse, Job, JobStatus
from .tasks import process_rally_job

settings = get_settings()
app = FastAPI(title="Rally Predictor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": settings.ENV, "storage": "local"}


# ── Job lifecycle ───────────────────────────────────────────────────────────
@app.post("/jobs", response_model=CreateJobResponse)
def create_job(req: CreateJobRequest) -> CreateJobResponse:
    job_id = jobs.new_job_id()
    jobs.create(job_id, source=req.source, filename=req.filename)

    store = storage.get_storage()
    key = storage.input_key(job_id, req.filename)
    upload_url = store.upload_url(key, req.content_type)

    return CreateJobResponse(job_id=job_id, upload_url=upload_url)


@app.post("/jobs/{job_id}/clips", response_model=AddClipResponse)
def add_clip(job_id: str, req: AddClipRequest) -> AddClipResponse:
    """Register one clip for a multi-clip job and return its upload URL.

    Call once per GoPro clip, in any order (clip_index controls concatenation
    order).  After all clips are uploaded, call POST /jobs/{id}/start.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    store = storage.get_storage()
    clip_key = f"inputs/{job_id}_clip{req.clip_index:03d}.mp4"
    upload_url = store.upload_url(clip_key, req.content_type)

    # Record the clip key on the job so the worker knows what to concatenate.
    existing = list(job.clip_keys)
    # Insert at the right position so the list stays ordered by clip_index.
    # We store the key directly; the worker sorts by the embedded index.
    if clip_key not in existing:
        existing.append(clip_key)
        existing.sort()  # lexicographic sort preserves the clip_000/001/... order
    jobs.update(job_id, clip_keys=existing)

    return AddClipResponse(clip_key=clip_key, upload_url=upload_url)


@app.post("/jobs/{job_id}/start", response_model=Job)
def start_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")

    store = storage.get_storage()
    key = storage.input_key(job_id, job.filename or "match.mp4")
    if not store.exists(key):
        raise HTTPException(409, "upload not found — PUT the video first")

    jobs.update(job_id, status=JobStatus.QUEUED, message="Queued for analysis")
    process_rally_job.delay(job_id)
    return jobs.get(job_id)  # type: ignore[return-value]


@app.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.websocket("/jobs/{job_id}/events")
async def job_events(ws: WebSocket, job_id: str) -> None:
    """Push job-state JSON to the client whenever it changes, until terminal."""
    await ws.accept()

    job = jobs.get(job_id)
    if job is None:
        await ws.send_text(json.dumps({"error": "job not found"}))
        await ws.close()
        return

    # Send current state immediately, then stream updates.
    await ws.send_text(job.model_dump_json())
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        await ws.close()
        return

    pubsub = jobs.subscribe(job_id)
    try:
        while True:
            msg = await asyncio.to_thread(
                pubsub.get_message, ignore_subscribe_messages=True, timeout=1.0
            )
            if msg and msg.get("type") == "message":
                payload = msg["data"]
                await ws.send_text(payload)
                state = json.loads(payload)
                if state.get("status") in ("completed", "failed"):
                    break
            else:
                # Keepalive so proxies don't drop an idle socket.
                await ws.send_text('{"keepalive":true}')
    except WebSocketDisconnect:
        pass
    finally:
        pubsub.close()
        try:
            await ws.close()
        except RuntimeError:
            pass


# ── Local-storage passthrough ────────────────────────────────────────────────
@app.put("/local-storage/{key:path}")
async def local_put(key: str, request: Request) -> Response:
    store = storage.get_storage()
    dest = store.local_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    return Response(status_code=200)


@app.get("/local-storage/{key:path}")
def local_get(key: str) -> FileResponse:
    store = storage.get_storage()
    path = store.local_path(key)
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/mp4")


# ── Live streaming routes ────────────────────────────────────────────────────
register_live_routes(app)
