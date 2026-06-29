# Rally Predictor — Desktop app

PyQt6 GUI wrapper around the pipeline. Runs locally; no server required.

## Run

```bash
cd desktop
pip install -r requirements.txt
python app.py
```

Click **Open Video**, select a `match.mp4` or GoPro clip. Progress streams in
real-time; the output reel is written alongside the input file.

## Build a distributable .app (macOS)

```bash
pip install pyinstaller
pyinstaller rally_app.spec
# → dist/Rally Detector.app
```

## How it connects to the pipeline

`app.py` adds `../pipeline` to `sys.path` at startup and imports directly:

```python
from rally_detector import collect_rally_segments
from utilities import create_highlights_ffmpeg
```

Editing anything in `pipeline/` takes effect immediately on the next run —
no rebuild needed.
