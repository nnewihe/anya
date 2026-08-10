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

### Dead-time cutter (recommended for full matches)

Cuts everything from the end of each point to the start of the next service
motion. Serve detection on both court ends anchors the point starts; point
ends fuse ball traces with player kinematics. See
[`pipeline/README.md`](pipeline/README.md) for details.

```bash
python -m pipeline.deadtime_cutter match.mp4            # full run
python -m pipeline.deadtime_cutter match.mp4 --dry-run  # segments + report only
```

The slow perception pass is cached next to the video, so re-tuning the
segmentation re-runs in seconds.

### Single video file (rally detector)

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
5. **Highlight reel** — `ffmpeg` cuts and concatenates the detected windows into a single output file (`utilities.py`). Segments are re-encoded, not stream-copied, so cuts land on exact timestamps rather than keyframes; see Performance below for the encoder.

---

## Performance

All figures measured on an Apple M4 over `Data/21` (7:00 of 4K, 12,594 frames
at 29.97fps, source on an external USB drive), for the `rally_reel` path that
`desktop/app.py` runs.

### Where the time goes

| Stage | Before | Now | |
|---|---|---|---|
| 1 · telemetry (3 model calls/frame) | 954.7s | 411.9s | 2.32x |
| 2 · far-player pose | ~195s | ~195s | unchanged |
| 5 · walking pose | 428.6s | 179.8s | 2.38x |
| 7 · cut + encode | 167.7s | 68.6s | 2.44x |
| **total** | **~29 min** | **~14.4 min** | **~2.0x** |

That is 4.15x realtime down to 2.05x. For a 20-minute 4K match: **~83 min of
processing down to ~41 min.**

### What made the difference

**Batching** (`ExtractorConfig.batch_size`, `extract_pose.BATCH`, both 16) —
each per-frame model call paid a fixed cost (Python-side preprocess, MPS
dispatch, postprocess) that batching amortises. Almost all of the gain is the
far-ROI player call, 10.09 -> 3.25 ms/frame, because a 556x540 crop at
imgsz=384 was nearly all fixed cost. The two full-frame 960px calls are
compute-bound and gave back only ~15%.

**Threaded decode** (`ExtractorConfig.prefetch`, `extract_pose.extract(prefetch=)`)
— 4K decode is 6.4-6.9 ms/frame of pure CPU against ~12.6 ms of GPU, and both
passes ran them strictly in sequence. Overlapping is free. This mattered more
than expected: telemetry's wall time fell 954 -> 412s while its CPU time only
fell 757 -> 637s, so much of the win was hidden I/O wait.

**Hardware H.264** (`utilities.video_encode_args`) — stage 7 re-encodes every
kept second at source resolution. `h264_videotoolbox -b:v 40M` replaces
`libx264 -crf 18`, 2.44x faster for ~23% more bytes. Falls back to x264 where
VideoToolbox is unavailable, so non-Apple platforms are unchanged.

Each has an escape hatch: `batch_size=1`, `--batch 1`, `prefetch=False`, or a
forced `_VIDEO_ENCODER_CACHE`.

### Accuracy

Batching and prefetch are behaviour-preserving, not approximations.
Ultralytics only switches letterbox mode for mixed-shape batches
(`pre_transform`: `auto=same_shapes and ...`), and every batch built here is
shape-uniform. Verified by full-clip A/B, same code, one variable changed:

- telemetry: 18 of 12,594 records differ (0.14%), player boxes bit-identical,
  ball detection count identical (123,278)
- walking pose: bit-identical, zero detection-slot mismatches
- **serve event times identical; final segments byte-identical**

### Things that do NOT work here

Measured and rejected — recorded so they are not retried from scratch. See
`DESIGN.md` section 8 for the full numbers.

- **CoreML export is slower**, not faster: ball 18.8 ms/frame vs 11.0 batched
  torch, player 20.3 vs 9.7. Not merely wrapper overhead — a raw
  `coremltools.predict` forward is 8.3 ms against 5.0 ms for batched
  torch-MPS, and `compute_units` makes no difference. `ios_tracker`'s ~5
  ms/frame does not transfer because that is native Swift feeding
  MLMultiArray directly.
- **Subsampling (stride 2 / 15fps)** saves ~1.4x but breaks detectors whose
  thresholds are coupled to the frame rate. On Data/23 it loses **all 20 far
  serves**, because `SMOOTH_MIN_SAMPLES = 3` is a sample count applied to a
  duration-defined window. Near-serve recall drops 8/8 -> 3/8 from the same
  class of bug in `ratio_smooth_n` (3 taps span 0.1s at 30fps but 0.2s at
  15fps, flattening the curvature `jerk` measures). Both are fixable, neither
  is fixed.
- **Coarse-to-fine gating** reaches 100% recall only at a threshold that flags
  83-92% of the clip, leaving ~10% to save. No operating point has both
  recall and selectivity: "a player is standing still" is true most of the
  time in tennis footage.
- **Gating the walking pass** to where its output is consulted is worth only
  2-7% once padded for the 8s feature window, and needs a correctness
  argument spanning four constants in three files.

### Benchmarking notes

`time.time()` counts machine sleep. An unattended run that sleeps mid-pass
reports absurd figures — one such run reported 11,113s and 21,746s for passes
that take ~250s — **and can appear to change detection counts.** Run long
benchmarks under `caffeinate -i`, and include a same-config control arm: these
passes are deterministic, so two identical runs must produce bit-identical
`.npz` output. Several conclusions in this work were wrong until a control arm
was added.

### Where the remaining time is

For a 20-minute 4K match, roughly: telemetry 19.6 min, far pose 9.6 min,
walking 8.6 min, encode 3.3 min. The largest structural redundancy left is
that stage 5 re-detects the near player with `yolov8n-pose` that stage 1
already found with `yolo26n` — it needs keypoints the latter does not emit, so
collapsing them is a model change, not scheduling. A further ~4-5x looks
reachable in Python; 10x would need the native/on-device path.
