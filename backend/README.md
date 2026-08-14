# Rally Predictor — backend

A FastAPI service that wraps the existing `rally_detector.py` pipeline behind an
async job API, so the Flutter app (or any client) can upload a match, track
progress, and download the detected rally reel.

## Architecture

```
 Flutter app
    │  POST /jobs            (create job, get upload URL)
    │  PUT  <upload_url> ───────────────►  media volume  (raw match.mp4)
    │  POST /jobs/{id}/start ──┐
    │                          ▼
    │                      Redis  ◄──► Celery worker  (rally_detector.py + YOLO)
    │                          │              │ reads input from the volume
    │  WS /jobs/{id}/events ◄──┘              │ runs collect_rally_segments()
    │                                         │ create_highlights_ffmpeg()
    │  GET result_url  ◄──────────────────────┘ writes reel to the volume
```

| Concern        | Choice                              | Why |
|----------------|-------------------------------------|-----|
| API            | FastAPI + Uvicorn                   | async, native WebSocket, imports the pipeline directly |
| Job queue      | Celery + Redis                      | analysis takes minutes; never block a request |
| Job state      | Redis JSON + pub/sub                | one store the API and worker share; pub/sub drives WS |
| Storage        | Local filesystem volume             | no third-party service in the media path |
| Inference      | Ultralytics YOLO (`models/*.pt`)    | GPU strongly recommended |

### Files

```
backend/
  app/
    main.py            FastAPI routes (jobs, websocket, local-storage dev)
    tasks.py           Celery app + process_rally_job
    pipeline_runner.py adapter → rally_detector.collect_rally_segments
    jobs.py            Redis job store + pub/sub
    storage.py         local filesystem storage
    live.py            WS live-ingest → same pipeline
    schemas.py         wire contract (mirrored by mobile/lib/models/job.dart)
    config.py          env-driven settings
  Dockerfile           bundles pipeline files + models/
  docker-compose.yml   api + worker + redis
  requirements.txt
```

The pipeline source (`rally_detector.py`, `anya_base.py`, `ball_tracker.py`,
`utilities.py`) and `models/` stay at the repo root; the Dockerfile copies them
into the image and `pipeline_runner.py` imports them.

## Local development

From the **repo root**:

```bash
docker compose -f backend/docker-compose.yml up --build
```

This runs the API, a Celery worker, and Redis, with media files on a shared
volume served by the API. Then:

- API docs:   http://localhost:8000/docs
- Health:     http://localhost:8000/health

### Smoke test end-to-end

```bash
# 1. create a job
JOB=$(curl -s -XPOST localhost:8000/jobs \
  -H 'content-type: application/json' \
  -d '{"filename":"match.mp4"}')
ID=$(echo "$JOB" | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
URL=$(echo "$JOB" | python3 -c 'import sys,json;print(json.load(sys.stdin)["upload_url"])')

# 2. upload a video (local-storage URL is relative → prefix the host)
curl -XPUT "localhost:8000$URL" --data-binary @/path/to/match.mp4 \
  -H 'content-type: video/mp4'

# 3. start analysis
curl -XPOST localhost:8000/jobs/$ID/start

# 4. poll
curl -s localhost:8000/jobs/$ID | python3 -m json.tool
```

Running without Docker: `pip install -r backend/requirements.txt`, start a
Redis, then in two shells:
```bash
uvicorn app.main:app --reload                     # from backend/
celery -A app.tasks.celery_app worker --loglevel=info
```
(`PIPELINE_DIR` defaults to the repo root, so the import just works.)

## Deployment

The service is deployed as the same two containers you run locally: the API and
a Celery worker, backed by a Redis instance and a filesystem volume for media.

- **Storage** — uploads and finished reels live under `LOCAL_STORAGE_DIR`.
  Point it at a persistent volume that both the API and the worker mount, and
  size it for the largest match videos you expect.
- **Redis** — set `REDIS_URL` to your Redis host.
- **Compute** — YOLO inference wants a GPU for the worker. Build the image from
  the GPU Dockerfile variant (swap the base to
  `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, install `python3`/`pip`, keep the
  rest) and add the `deploy.resources` GPU block (already stubbed, commented, in
  `docker-compose.yml`). Run the host with the NVIDIA Container Toolkit.
- **Environment (production)**
  ```
  ENV=prod
  LOCAL_STORAGE_DIR=/data/rally-media
  REDIS_URL=redis://<redis-host>:6379/0
  CORS_ORIGINS=*
  ```
- **API exposure** — terminate TLS at a reverse proxy in front of the API.
  WebSockets (`/jobs/{id}/events`, `/live/{id}`) need an idle timeout long
  enough for analysis (e.g. 300 s; the WS also emits keepalives).

## Live streaming

`live.py` accepts a WebSocket stream of encoded video chunks, assembles them,
and runs the **same** pipeline once the stream ends — "near-real-time" (results
seconds after the match ends). The Flutter `live_screen` records with the camera
and streams the file on stop.

**Future work — true real-time per-point detection:** `rally_detector.py` drives
its loop from `cv2.VideoCapture(file)`. Real-time would mean feeding it a
rolling frame buffer instead and emitting segments as points close. Path:
WebRTC/HLS ingest → decode to frames → adapt `collect_rally_segments` to consume
a frame generator (it already takes a `progress_cb` and processes frame-by-frame,
so the loop body is reusable) → push each closed segment over the events socket.

## Notes / scaling knobs

- **One video per worker** (`worker_prefetch_multiplier=1`, `--concurrency=1`):
  each job pins the GPU. Scale by adding worker instances/tasks.
- `task_acks_late=True` re-queues a job if a worker dies mid-analysis.
- Finished jobs persist in Redis for 7 days (`_JOB_TTL_SEC`).
- `FRAME_STRIDE` is wired in config for a future speed/accuracy trade-off.
