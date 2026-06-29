# Rally Predictor — Pipeline

The core detection logic. Edit files here and the change propagates to both
the desktop app and the backend server automatically.

## Files

| File | Purpose |
|---|---|
| `rally_detector.py` | **Main entry point** — segment detection, HMM filter, highlight export |
| `anya_base.py` | YOLO player + ball telemetry provider |
| `ball_tracker.py` | IMM Kalman single-ball tracker |
| `utilities.py` | ffmpeg highlight cutter, court homography helpers |
| `run_pipeline.py` | CLI: concatenate GoPro clips → run detector |
| `models/` | YOLO model weights (`ball_best.pt`, `yolo26n.pt`) |

## Run from the command line

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
