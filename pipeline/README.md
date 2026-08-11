# Rally Predictor — Pipeline

The core detection logic. Edit files here and the change propagates to both
the desktop app and the backend server automatically.

## Files

| File | Purpose |
|---|---|
| `deadtime_cutter.py` | **Dead-time cutter entry point** — serve-anchored point detection, cuts everything between points |
| `match_telemetry.py` | Dead-time cutter stage 1: one offline perception pass → cached JSONL telemetry |
| `point_segmenter.py` | Dead-time cutter stage 2: serve events (near + far) + fused point ends (no video/models needed) |
| `serve_stgcn.py` | Far-side ST-GCN serve classifier (MediaPipe pose + graph conv) |
| `rally_detector.py` | Trace-driven rally detector — segment detection, HMM filter, highlight export |
| `anya_telemetry.py` | Rally-reel stage 1: general-purpose perception pass (both players + raw whole-court balls) → cached JSONL |
| `extract_far_pose.py` | Rally-reel stage 2: pose over the far-player crop, keyed by frame |
| `anya_far_serve.py` | Far-side serve starts: baseline arming + hand-raise gate, ball trace as confirmation only |
| `anya_near_serve.py` | Near-side serve starts: dwell / toss / jerk scorer |
| `anya_far_telemetry.py` | **Far fast path** — far-only extractor (band proxy + 5 fps player + batched pose + gated ball); replaces stages 1-2 for `anya_far_serve` at ~5x |
| `anya_near_telemetry.py` | **Near fast path** — near-only extractor (540p proxy + 5 fps player + toss-ROI ball) at ~12x |
| `proxy.py` | One-time frame-exact ffmpeg proxies (downscaled whole frame, or native-resolution crop) shared by both fast paths |
| `anya_base.py` | YOLO player + ball telemetry provider |
| `ball_tracker.py` | IMM Kalman single-ball tracker |
| `utilities.py` | ffmpeg highlight cutter, court homography helpers |
| `run_pipeline.py` | CLI: concatenate GoPro clips → run detector |
| `models/` | Model weights (`ball_best.pt`, `yolo26n.pt`, `serve_stgcn.pt`, `pose_landmarker_full.task`) |

## Dead-time cutter

Removes everything from the end of each point to the start of the next
service motion. Point **starts** come from serve detection on both ends of
the court (near: ball toss + trophy pose; far: native-resolution ball toss
blended with ST-GCN pose kinematics, 0.65/0.35). Serve candidates that never
produce a serve-like ball trace (e.g. an aborted toss the server catches)
are displaced by the trace-confirmed serve that follows.
Point **ends** are found inside the bounded window to the next serve by
fusing the replayed ball trace with player kinematics (direction reversals /
both-players-moving = rally; steady walking = ball retrieval), so weak
far-side ball tracking degrades gracefully instead of truncating points.

```bash
# Full run (first run pays the slow perception pass; it is cached as
# <match>_match_telemetry.jsonl next to the video)
python -m pipeline.deadtime_cutter match.mp4

# Re-segment + report only (seconds, uses the cache; tune SegmenterConfig
# in point_segmenter.py and re-run freely)
python -m pipeline.deadtime_cutter match.mp4 --dry-run

# Stages individually
python -m pipeline.match_telemetry match.mp4            # stage 1 only
python -m pipeline.point_segmenter match_match_telemetry.jsonl   # stage 2 only

# Synthetic self-test (no video or models needed)
python -m pipeline.point_segmenter --self-test
```

Outputs: `<match>_no_deadtime.mp4`, plus `<match>_points.csv` /
`<match>_points.json` with per-point serve time, end time, serve side, and
which evidence decided the end (`trace`, `trace+activity`, `activity`,
`fallback`) for review.

First run on a new video prompts once for the 4 court corners and the
8-point active zone (cached beside the video, shared with the rest of the
pipeline).

## Run the rally detector from the command line

```bash
# Single file
python pipeline/rally_detector.py match.mp4 --headless

# Multiple GoPro clips (concatenates first)
python pipeline/run_pipeline.py /path/to/folder/with/clips
```

## How consumers import this

**Desktop app** (`desktop/app.py`):
```python
sys.path.insert(0, "../pipeline")
from rally_detector import collect_rally_segments
```

**Backend worker** (`backend/app/pipeline_runner.py`):
```python
# PIPELINE_DIR env var, defaults to pipeline/ at repo root
sys.path.insert(0, PIPELINE_DIR)
from rally_detector import collect_rally_segments
```
