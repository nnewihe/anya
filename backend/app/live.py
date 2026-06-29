"""
live.py
=======
Live-streaming ingest.

A phone has no GoPro-style file to upload, so the Flutter client opens a
WebSocket, records the camera, and streams the encoded video as it goes.  The
server appends every binary frame/chunk to a single file.  When the socket
closes (the user ends the match), the assembled recording is handed to the
exact same rally_detector pipeline as an uploaded file.

This gives "near-real-time" results — the full match is analyzed the moment
recording stops, not frame-by-frame during play.  True per-point streaming
detection (running the detector on a rolling frame buffer live) is a larger
change to rally_detector's file-based loop and is documented as future work in
the deploy README.

Wire protocol on /live/{job_id}:
  • binary message  → raw encoded video chunk, appended in order
  • text  "EOS"     → end of stream; server finalizes and enqueues analysis
  (closing the socket also finalizes.)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from . import jobs, storage
from .schemas import JobStatus
from .tasks import process_rally_job


def register_live_routes(app: FastAPI) -> None:
    @app.websocket("/live/{job_id}")
    async def live_ingest(ws: WebSocket, job_id: str) -> None:
        await ws.accept()

        # The client may create the job first (POST /jobs with source=live) or
        # let the stream create it implicitly.
        job = jobs.get(job_id)
        if job is None:
            jobs.create(job_id, source="live", filename=f"{job_id}.mp4")

        jobs.update(job_id, status=JobStatus.PENDING,
                    message="Receiving live stream…")

        tmp = Path(tempfile.mkdtemp(prefix=f"live_{job_id}_")) / "stream.mp4"
        bytes_received = 0
        finalized = False

        with open(tmp, "wb") as f:
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    if (data := msg.get("bytes")) is not None:
                        f.write(data)
                        bytes_received += len(data)
                    elif (text := msg.get("text")) is not None:
                        if text.strip().upper() == "EOS":
                            break
            except WebSocketDisconnect:
                pass

        if bytes_received == 0:
            jobs.update(job_id, status=JobStatus.FAILED,
                        error="empty live stream")
            return

        # Push the assembled recording into storage under the normal input key,
        # then enqueue the standard analysis task.
        store = storage.get_storage()
        in_key = storage.input_key(job_id, f"{job_id}.mp4")
        store.upload_from(tmp, in_key, content_type="video/mp4")

        jobs.update(job_id, status=JobStatus.QUEUED,
                    message=f"Stream complete ({bytes_received} bytes) — queued")
        process_rally_job.delay(job_id)
        finalized = True

        try:
            await ws.send_text('{"status":"queued"}' if finalized else "{}")
            await ws.close()
        except RuntimeError:
            pass
