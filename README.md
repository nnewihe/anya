# Anya — Rally Predictor

Automatically detects and extracts rally segments from tennis match footage using YOLO ball tracking and an HMM serving-pattern filter.

Three ways to use it:

| Mode | Best for |
|---|---|
| **Pipeline (CLI)** | Batch processing, scripting, server use |
| **Desktop app** | Quick analysis on a local Mac/Windows machine |
| **Mobile app** | Upload from phone or stream live; results on the cloud |

---

## Repo layout

```
anya/
  pipeline/          — core detection logic (edit here to change behavior)
    rally_detector.py
    anya_base.py
    ball_tracker.py
    utilities.py
    models/
    run_pipeline.py

  desktop/           — PyQt6 GUI for local use
    app.py
    requirements.txt

  backend/           — FastAPI + Celery server (wraps pipeline/)
    app/
    Dockerfile
    docker-compose.yml

  mobile/            — Flutter app (Android, iOS, macOS)
    lib/
    pubspec.yaml
```

Both the desktop app and the backend server import directly from `pipeline/`. Changing a file in `pipeline/` immediately updates the desktop app and takes effect in the backend after the next Docker build.

---

## 1. Pipeline — command line

### Requirements

```bash
pip install opencv-python numpy ultralytics filterpy scikit-learn
```

### Single video file

```bash
python pipeline/rally_detector.py match.mp4 --headless
```

Output is written to `match_rallies.mp4` alongside the input file.

| Flag | Default | Description |
|---|---|---|
| `--output PATH` | `<input>_rallies.mp4` | Where to write the rally reel |
| `--headless` | off | Skip all interactive OpenCV windows (required on servers) |
| `--start-frame N` | 0 | Resume from frame N (useful after a crash) |

### Multiple GoPro clips

GoPro splits long recordings into ~4 GB chunks. `run_pipeline.py` finds them, concatenates them with ffmpeg (no re-encode), then runs the detector on the combined file.

```bash
python pipeline/run_pipeline.py /path/to/gopro/folder
```

Clips are matched by name pattern `GH0<digit>0897.MP4` and sorted before concatenation.

### Output

The detector writes a highlight reel MP4 containing only the detected rally segments. Each segment is trimmed with a short pre-roll so the first shot is never cut off.

---

## 2. Desktop app

A PyQt6 GUI — no server required. Runs everything locally.

### Install

```bash
pip install -r desktop/requirements.txt
```

### Run

```bash
python desktop/app.py
```

Click **Open Video**, select a match file or a single GoPro clip. A progress bar streams frame-by-frame updates. When analysis finishes, the output reel path is shown and can be opened directly from the app.

### Build a standalone .app (macOS)

```bash
pip install pyinstaller
cd desktop
pyinstaller rally_app.spec
# → dist/Rally Detector.app
```

The `.app` bundles Python and all dependencies; no separate install needed on the target machine.

---

## 3. Mobile app (Flutter)

Upload a match video from your phone, upload GoPro clips, or stream live from the camera. Results are processed on the backend server and the rally reel is played back in-app.

### Prerequisites

- [Flutter SDK](https://docs.flutter.dev/get-started/install) ≥ 3.19
- Android Studio or Xcode (for device builds)

### Run locally (against local backend)

Start the backend first (see section 4), then:

```bash
cd mobile
flutter run --dart-define=API_BASE_URL=http://<your-mac-ip>:8000
```

Use `10.0.2.2` as the IP for an Android emulator pointing at your Mac host.

### Deploy to Android phone

```bash
cd mobile
flutter run --dart-define=API_BASE_URL=http://44.203.32.208:8000
```

Or build a release APK:

```bash
flutter build apk --dart-define=API_BASE_URL=http://44.203.32.208:8000
adb install build/app/outputs/flutter-apk/app-release.apk
```

### Deploy to iOS

```bash
cd mobile
flutter run -d <your-iphone-device-id> \
  --dart-define=API_BASE_URL=http://44.203.32.208:8000
```

### Three upload modes

| Mode | How to use |
|---|---|
| **Upload a match** | Pick a single video from the phone's library; upload, then tap Analyze |
| **Upload GoPro clips** | Pick multiple clips (sorted by filename); they are concatenated server-side before analysis |
| **Go live** | Records from the phone camera and streams to the server in real time; analysis runs when you stop recording |

---

## 4. Backend server

### Local development (no AWS)

Requires Docker Desktop.

```bash
# From the repo root
docker compose -f backend/docker-compose.yml up --build
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### AWS deployment

The backend is running on:

```
http://44.203.32.208:8000
```

SSH access:

```bash
ssh -i ~/Documents/rally-predictor-key.pem ubuntu@44.203.32.208
# Check setup progress:
tail -f /var/log/rally-setup.log
```

Full AWS deployment guide (S3, ECR, EC2, ALB): [`backend/README.md`](backend/README.md)

---

## How the pipeline works

1. **Ball tracking** — YOLO (`models/ball_best.pt`) detects the ball each frame; an IMM Kalman filter (`ball_tracker.py`) smooths the trajectory.
2. **Player tracking** — YOLO (`models/yolo26n.pt`) detects players; court homography maps pixel positions to court coordinates (`anya_base.py`).
3. **Rally detection** — `rally_detector.py` watches for sustained ball activity (velocity, continuity, court zone) to open and close rally segments.
4. **HMM filter** — a hidden Markov model over serve-side alternation prunes false positives caused by ball-retrieval walks between points.
5. **Highlight reel** — `ffmpeg` stream-copies the detected windows (no re-encode) into a single output file (`utilities.py`).
