# Anya — Desktop app

PyQt6 GUI wrapper around the pipeline. Runs locally; no server required.
Two modes, selectable in the UI:

- **Rally Reel** — `rally_detector.collect_rally_segments`: a highlights reel
  of the rallies (self-calibrating, no court setup).
- **Remove Dead Time** — `deadtime_cutter.cut_dead_time`: keeps
  `[serve motion .. point end]` for every point and drops the gaps. The first
  run on a new video asks you to click the four court corners once (bottom-left,
  bottom-right, top-right, top-left); the calibration is cached next to the
  video, and stage-1 perception is cached too, so re-runs are fast.

## Run

```bash
cd desktop
pip install -r requirements.txt
python app.py
```

Pick a mode, **Browse** to a `match.mp4` or GoPro clip, and run. Progress
streams in real-time; the output is written alongside the input file
(`*_rallies.mp4` or `*_no_deadtime.mp4`).

## Build a distributable .app (macOS)

```bash
pip install pyinstaller
pyinstaller rally_app.spec
# → dist/RallyDetector.app
```

## How it connects to the pipeline

`app.py` puts the **repo root** on `sys.path` and imports the pipeline as a
package (its modules use relative imports, so bare top-level imports don't
work):

```python
from pipeline.rally_detector import collect_rally_segments
from pipeline.deadtime_cutter import cut_dead_time
from pipeline.utilities import create_highlights_ffmpeg, init_court, Config
```

Editing anything in `pipeline/` takes effect immediately on the next run —
no rebuild needed. Both modes share the same two model weights
(`ball_best.pt`, `yolo26n.pt`).
