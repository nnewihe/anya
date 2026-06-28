# Rally Predictor — backend

A FastAPI service that wraps the existing `rally_detector.py` pipeline behind an
async job API, so the Flutter app (or any client) can upload a match, track
progress, and download the detected rally reel.

## Architecture

```
 Flutter app
    │  POST /jobs            (create job, get presigned upload URL)
    │  PUT  <presigned>  ───────────────►  S3  (raw match.mp4)
    │  POST /jobs/{id}/start ──┐
    │                          ▼
    │                      Redis  ◄──► Celery worker  (rally_detector.py + YOLO)
    │                          │              │ downloads input from S3
    │  WS /jobs/{id}/events ◄──┘              │ runs collect_rally_segments()
    │                                         │ create_highlights_ffmpeg()
    │  GET result_url  ◄──────────────────────┘ uploads reel to S3
```

| Concern        | Choice                              | Why |
|----------------|-------------------------------------|-----|
| API            | FastAPI + Uvicorn                   | async, native WebSocket, imports the pipeline directly |
| Job queue      | Celery + Redis                      | analysis takes minutes; never block a request |
| Job state      | Redis JSON + pub/sub                | one store the API and worker share; pub/sub drives WS |
| Storage        | S3 (presigned PUT/GET)              | multi-GB videos skip the API process entirely |
| Inference      | Ultralytics YOLO (`models/*.pt`)    | GPU strongly recommended |

### Files

```
backend/
  app/
    main.py            FastAPI routes (jobs, websocket, local-storage dev)
    tasks.py           Celery app + process_rally_job
    pipeline_runner.py adapter → rally_detector.collect_rally_segments
    jobs.py            Redis job store + pub/sub
    storage.py         S3 / local storage with presigned URLs
    live.py            WS live-ingest → same pipeline
    schemas.py         wire contract (mirrored by mobile/lib/models/job.dart)
    config.py          env-driven settings
  Dockerfile           bundles pipeline files + models/
  docker-compose.yml   api + worker + redis (local, no AWS needed)
  requirements.txt
```

The pipeline source (`rally_detector.py`, `anya_base.py`, `ball_tracker.py`,
`utilities.py`) and `models/` stay at the repo root; the Dockerfile copies them
into the image and `pipeline_runner.py` imports them.

## Local development (no AWS)

From the **repo root**:

```bash
docker compose -f backend/docker-compose.yml up --build
```

This runs the API, a Celery worker, and Redis with `STORAGE_BACKEND=local`
(files under a shared volume, served by the API). Then:

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

## AWS deployment

### 1. S3 — media bucket
```bash
aws s3 mb s3://rally-predictor-media
```
Add a CORS policy so the Flutter app can PUT/GET directly:
```json
[{"AllowedMethods":["PUT","GET"],"AllowedOrigins":["*"],
  "AllowedHeaders":["*"],"ExposeHeaders":["ETag"]}]
```

### 2. ElastiCache — Redis
Create a Redis (single node is fine to start). Note the primary endpoint →
`REDIS_URL=redis://<endpoint>:6379/0`.

### 3. ECR — image
```bash
aws ecr create-repository --repository-name rally-predictor-backend
docker build -f backend/Dockerfile -t rally-predictor-backend .   # context = repo root
# tag + push to the ECR URI (see `aws ecr get-login-password`)
```

### 4. Compute — GPU for the worker
YOLO inference wants a GPU. Recommended: **`g4dn.xlarge`** (T4) for the worker.

- **Simplest:** one GPU EC2 instance running both containers via the same
  compose file, with `STORAGE_BACKEND=s3`. Build the image from the **GPU
  Dockerfile variant** (swap the base to `nvidia/cuda:12.4.1-runtime-ubuntu22.04`,
  install `python3`/`pip`, keep the rest) and add the `deploy.resources` GPU
  block (already stubbed, commented, in `docker-compose.yml`). Run the host with
  the NVIDIA Container Toolkit.
- **Scalable:** ECS — API task on Fargate (CPU), worker task on a GPU-backed
  ECS/EC2 capacity provider. Both pull the same image; the worker overrides the
  command to `celery -A app.tasks.celery_app worker`.

Give the task/instance an **IAM role** with `s3:GetObject`/`s3:PutObject` on the
bucket — no static keys.

### 5. Environment (production)
```
ENV=prod
STORAGE_BACKEND=s3
S3_BUCKET=rally-predictor-media
AWS_REGION=us-east-1
REDIS_URL=redis://<elasticache-endpoint>:6379/0
CORS_ORIGINS=*
```

### 6. API exposure
Put the API behind an ALB (TLS via ACM). WebSockets (`/jobs/{id}/events`,
`/live/{id}`) work through ALB — no special config beyond an idle-timeout long
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
