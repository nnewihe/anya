# Anya Tennis — desktop app

> Watch your matches in minutes, not hours.

PyQt6 desktop app around `pipeline.rally_reel`. Runs entirely locally — no
server, no upload. Pick a match video, click the four court corners once, get
back a reel containing only the rallies.

Runs on **macOS, Linux and Windows**.

## Run

```bash
pip install -r desktop/requirements.txt
python desktop/app.py
```

`ffmpeg` must be on PATH — the final cut shells out to it:

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
winget install Gyan.FFmpeg   # Windows
```

## What it does

One pipeline, seven stages, all reported live in the progress bar:

| Stage | Work | Cached as |
|---|---|---|
| 0 | Court corners (interactive, first run only) | `<video>_court_cache.json` |
| 1 | Telemetry — players + raw ball detections | `<video>_anya_telemetry.jsonl` |
| 2 | Far-player pose | `<video>_far_pose.jsonl` |
| 3 | Far-side serve starts | — |
| 4 | Near-side serve starts | `<video>_..._near_serve_events.json` |
| 5 | Walking → dead time | `<video>_walk_pose.npz` |
| 6 | Segment assembly | `<video>_rally_segments.json` |
| 7 | Cut + concatenate | `<video>_rally_reel.mp4` |

Every stage caches next to the input video, so a second run on the same match
skips straight to whatever changed. Expect roughly **3x the clip length** on a
first run (a 7-minute clip takes ~20 minutes on an M-series Mac); later runs are
seconds plus the ffmpeg cut.

The court click happens on the main thread before analysis starts — `init_court`
opens an OpenCV window, which is unsafe from a worker thread.

## How it connects to the pipeline

`app.py` puts the **repo root** on `sys.path` and imports the pipeline as a
package (its modules use relative imports, so bare top-level imports don't
work):

```python
from pipeline.rally_reel import ReelConfig, build_reel
```

**Editing anything under `pipeline/` takes effect on the next run — no rebuild.**
That extends to the stage list itself: stage count, labels and progress all come
from `rally_reel.reel`, and the GUI renders whatever it is handed, so adding or
reordering a stage there needs no change in `app.py`.

Tuning lives in `pipeline/rally_reel/config.py` (`ReelConfig`) — thresholds, roll
padding, point-end policy. The app instantiates a default `ReelConfig`; edit that
file to change behaviour globally, or use the CLI for one-off runs:

```bash
python -m pipeline.rally_reel match.mp4 --dry-run --ball-quiet-mode off
```

## Build a distributable

```bash
pip install pyinstaller
cd desktop && pyinstaller rally_app.spec
```

- macOS → `dist/Anya Tennis.app`
- Windows / Linux → `dist/AnyaTennis/`

Weights are pulled from the repo automatically (`pipeline/models/`,
`walking/outputs/`); nothing to copy by hand. Expect a **large bundle (~2 GB)** —
torch and ultralytics are required by the telemetry and pose stages and cannot be
excluded. `ffmpeg` is not bundled and must be present on the target machine.

**Only the macOS path has been exercised.** The spec targets all three platforms,
but Windows and Linux builds are untested; the likely friction points are torch's
platform wheels and PyQt6 plugin collection.

## Design

Colours, tagline and logo follow [`DESIGN.md`](../DESIGN.md) and are shared with
the mobile app — black ground, `#E8FF3D` yellow as the single focal accent, and
the logo loaded from `mobile/assets/images/anya_logo_black.svg` (the
white-on-dark variant, named for its target background).
