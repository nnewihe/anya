# Anya Tennis — Flutter app

Cross-platform (Android + iOS + macOS) front end for the rally detector. Pick a
match video and the whole analysis runs **on-device** — no upload, no server,
and no network calls.

## What's here

```
lib/
  main.dart                    app entry
  screens/
    home_screen.dart           pick a match video
    match_setup_screen.dart    progress → segments + reel playback + share
  engine/                      the on-device pipeline (Dart port of pipeline/)
    engine.dart                model loading + per-frame orchestration
    inference.dart             ONNX Runtime sessions (CoreML / NNAPI / CPU)
    ball_tracker.dart          IMM Kalman ball tracking
    rally_detector.dart        rally segmentation
    deadtime_engine.dart       dead-time cutter
    point_segmenter.dart       serve-anchored point starts / ends
    reel.dart                  segment merge + reel cut
    frame_source.dart          ffmpeg_kit (mobile) / system ffmpeg (desktop)
  services/
    background_analysis.dart   foreground-service wrapper so long runs survive
    gallery_export.dart        save the reel to Photos / Gallery
    youtube_upload.dart        optional, user-initiated share to YouTube
assets/models/
  ball_best.onnx               ball detector
  yolo26n.onnx                 player detector
```

## Setup

```bash
cd mobile
flutter pub get
```

## Running

```bash
flutter run                              # attached device or simulator
flutter run -d <your-iphone-device-id>   # specific iOS device
```

Release builds:

```bash
flutter build apk       # Android
flutter build ios       # iOS
```

## How it runs on-device

The two ONNX models ship as bundled assets and are loaded once per process via
`OrtSession.fromBuffer` ([`engine/inference.dart`](lib/engine/inference.dart)),
with the CoreML execution provider on iOS, NNAPI on Android, and a CPU fallback
everywhere. Video decode and the final reel cut go through the bundled
`ffmpeg_kit_flutter_new` on mobile; on desktop the same code path shells out to
the system `ffmpeg` (see [`engine/platform.dart`](lib/engine/platform.dart)).

Analysis of a long match can outlive the foreground, so it runs under
`flutter_foreground_task` ([`services/background_analysis.dart`](lib/services/background_analysis.dart)).
iOS caps background CPU, so this is best-effort there — a long match may pause
and resume when the app returns to the foreground.

## Flow

1. **Pick** — `home_screen` picks a video from the device library.
2. **Analyze** — `match_setup_screen` runs the engine locally, streaming
   progress into the UI. Nothing leaves the device.
3. **Result** — the reel plays inline via `video_player`, alongside the detected
   segment list.
4. **Share** — save the reel to Photos/Gallery (local), or optionally upload it
   to YouTube. The YouTube upload is the app's only network call and only fires
   when the user taps it; see [`docs/sharing_setup.md`](docs/sharing_setup.md).
