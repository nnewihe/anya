# Rally Predictor — Flutter app

Cross-platform (Android + iOS) front end for the `rally_detector.py` pipeline.
Upload a match (or capture live), watch analysis progress, and play back the
detected rally reel.

## What's here

```
lib/
  main.dart              app entry
  config.dart            backend URL (override with --dart-define)
  api/api_client.dart    REST + WebSocket client
  models/job.dart        mirrors backend/app/schemas.py
  screens/
    home_screen.dart     pick a video / go live
    job_screen.dart      upload → progress → segments + reel playback
    live_screen.dart     camera capture → stream to backend
```

Only the Dart source lives in the repo. The native `android/` and `ios/`
projects are generated locally (they're large and machine-specific).

## First-time setup

```bash
cd mobile
flutter create .          # generates android/ + ios/ around the existing lib/
flutter pub get
```

`flutter create .` will NOT overwrite `lib/`, `pubspec.yaml`, or this README.

### Permissions to add after `flutter create`

**iOS — `ios/Runner/Info.plist`** (camera, mic, and—for dev over http—ATS):

```xml
<key>NSCameraUsageDescription</key>
<string>Capture matches for rally detection.</string>
<key>NSMicrophoneUsageDescription</key>
<string>Record match audio.</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>Pick a match video to analyze.</string>
<!-- DEV ONLY: allow plain http to your local backend -->
<key>NSAppTransportSecurity</key>
<dict><key>NSAllowsArbitraryLoads</key><true/></dict>
```

**Android — `android/app/src/main/AndroidManifest.xml`**:

```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
```
Add `android:usesCleartextTraffic="true"` to the `<application>` tag for dev
(plain http to a local server).

## Running

Point the app at your backend. Defaults to `http://10.0.2.2:8000` (the Android
emulator's alias for the host machine).

```bash
# Android emulator → host backend on :8000
flutter run

# iOS simulator
flutter run --dart-define=API_BASE_URL=http://localhost:8000

# Against a deployed backend
flutter run --dart-define=API_BASE_URL=https://api.yourdomain.com
```

## Flow

1. **Upload** — `home_screen` picks a video, creates a job (`POST /jobs`),
   then `job_screen` PUTs the file to the returned upload URL, calls
   `POST /jobs/{id}/start`, and streams progress over
   `WS /jobs/{id}/events` (with a 5 s polling fallback).
2. **Live** — `live_screen` records with the device camera and streams the
   recording to `WS /live/{id}` in chunks; the backend assembles it and runs
   the same pipeline.
3. **Result** — on completion the job carries `result_url` (a presigned S3
   link) and the segment list; the reel plays inline via `video_player`.
