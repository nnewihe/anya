# BallTracker — dedicated Android tennis ball tracker

A standalone native Android app (Kotlin + Jetpack Compose) that tracks the
tennis ball using **Android's built-in on-device ML acceleration**. No server,
no Flutter, no ONNX Runtime — the ball model runs as a TensorFlow Lite (LiteRT)
graph on the **NNAPI** delegate (the device NPU/DSP), falling back to the **GPU**
delegate and then multithreaded **CPU (XNNPACK)**.

This is the Android counterpart of `ios_tracker/` (which runs the same model on
the Apple Neural Engine via Core ML). The tracking brain — the Kalman/IMM online
tracker and the offline Viterbi solver — is a line-by-line Kotlin port of the
same validated engine (`mobile/lib/engine`, `pipeline/ball_tracker.py`), so both
apps behave identically frame-for-frame.

Two modes:

| Mode | What it does |
|---|---|
| **Live** | Point the phone at the court; the ball is detected, Kalman-tracked, and drawn as a fading trail over the CameraX preview, with state / speed / latency / accelerator HUD. |
| **Video** | Pick a match video; every frame is decoded and tracked offline (global Viterbi solve), then played back with the trajectory overlaid. Export a stitched highlights reel of the tracked rallies. |

## Why it uses Android's built-in ML

Android's on-device inference stack is **TensorFlow Lite / LiteRT** plus the
hardware delegates the platform ships:

- **NNAPI delegate** (API 27+) — routes the conv net to the vendor NPU / DSP.
  This is the direct analog of iOS running the model on the ANE.
- **GPU delegate** — fallback when NNAPI can't take the whole graph.
- **CPU / XNNPACK** — last-resort fallback (and what the emulator uses).

`BallDetector` picks the best available at load time and reports which one is
live in the HUD. As on iOS, the model is exported **NMS-free**: baked
NonMaxSuppression / TopK ops partition the graph and knock it off the
accelerators, so decode + single-class NMS run in Kotlin over the raw `[1,5,N]`
head — which also keeps the ultra-low conf thresholds the tennis pipeline needs
tunable per call-site.

## Layout

```
android_tracker/
  app/src/main/java/com/build2launch/balltracker/
    tracking/    Matrix, KalmanIMM, BallTrackManager, ViterbiTracker,
                 TrackerEngine — pure-Kotlin port of the validated engine
                 (no Android deps, JVM-unit-tested)
    detection/   Letterbox (Bitmap -> NHWC tensor), BallDetector (TFLite +
                 NNAPI/GPU/CPU delegate + decode/NMS), ExclusionZones (DBSCAN)
    live/        CameraManager — CameraX capture pipeline
    video/       VideoDecoder (MediaCodec), VideoProcessor (2-pass offline
                 solve), HighlightsExporter (MediaMuxer reel)
    ui/          Compose screens, shared BallOverlay, theme
    MainActivity.kt
  app/src/main/assets/ball_best.tflite   the exported model (see below; git-ignored)
  app/src/test/                          JVM tracker-parity harness
  run_tracker_tests.sh
```

## Building the model

The model is a build artifact, not source. Generate it with the export script
(run it in the same venv as the Core ML export — torch 2.7 / numpy 1.26 /
ultralytics 8.4 — plus `tensorflow` for the parity read-back):

```bash
python spikes/export_tflite.py
#  -> android_tracker/app/src/main/assets/ball_best.tflite   (fp16, 544x960, NMS-free)
#  -> spikes/fixtures/tflite_probe.json   (output shape + coord scale)
#  -> spikes/fixtures/tflite_parity.json  (.pt vs TFLite boxes)
```

`BallDetector` auto-detects the two things that vary between exporter versions —
the output layout (`[1,5,N]` vs `[1,N,5]`) and whether box coords are normalized
to `[0,1]` or in pixels — so a re-export won't silently break decoding. The
probe JSON records what the current export actually produced.

## Running

```bash
# JVM tracker-parity harness (no device needed):
android_tracker/run_tracker_tests.sh
#  10-scenario tracker oracle (same as pipeline/ball_tracker.py) + DBSCAN + Viterbi

# Build/install the app on a device (NNAPI/GPU need real hardware; the emulator
# falls back to CPU):
cd android_tracker
./gradlew installDebug     # or open the folder in Android Studio and Run
```

Requirements: Android Studio (bundled JDK 17+), Android SDK platform 35, a
device on API 26+. Set your signing team in Android Studio to run on hardware.

## Verification

`run_tracker_tests.sh` compiles the **actual app sources** and runs the tracker
against the same 10 self-test scenarios as the iOS parity harness and the Python
pipeline (moving/stopped/lost/occlusion/racket-reversal/court-bounce/serve
re-acquire/sparse-net-crossing/false-positive-rejection). All 10 pass, plus
DBSCAN clustering and the Viterbi solver. The detection-parity half (letterbox +
accelerator inference + decode vs the Ultralytics golden boxes) runs on-device —
add it as an instrumentation test once you have the `.tflite` bundled.

## Notes & known limitations

- Tracking quality is tuned for landscape, court-behind-baseline framing at
  960-wide analysis scale — the same regime as the anya pipeline. Gate constants
  and `activeBallConf = 0.10` come straight from the validated config.
- Speed is reported in px/s (no court calibration in this app by design).
- Video mode decodes frames headlessly with MediaCodec (YUV_420_888 -> ARGB in
  software) and rotates by the track's rotation metadata; 4K files work but
  process slower than 1080p.
- The highlights reel is a stream copy (MediaExtractor -> MediaMuxer), so there's
  no picture re-encode; the kept time ranges match `ios_tracker/make_highlights.py`.
```
