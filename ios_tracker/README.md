# BallTracker — dedicated iOS tennis ball tracker

A standalone native SwiftUI app that tracks the tennis ball on Apple's Neural
Engine. No server, no ONNX Runtime — the ball model runs as a Core ML
`.mlpackage` compiled straight into the app, and the whole graph stays
ANE-resident.

Two modes:

| Mode | What it does |
|---|---|
| **Live** | Point the phone at the court; the ball is detected, Kalman-tracked, and drawn as a fading trail over the camera preview at 60 fps, with state / speed / latency HUD. |
| **Video** | Pick a match video from the library; every frame is processed offline (ball detection + Vision player boxes), the rally detector (port of `pipeline/rally_detector.py`) cuts the active segments, and the video plays back with the tracked trajectory overlaid (scrubbing stays in sync). Detected rallies export as a stitched highlights reel. |

## Why it's fast

- `ball_best.pt` (YOLO11n) exported via `spikes/export_coreml.py` as an
  **NMS-free** fp16 mlprogram with a 960×544 image input. No
  NonMaxSuppression/TopK in the graph → no CPU fallback, pure conv net on the
  ANE (verified by op histogram).
- Decode + single-class NMS run in Swift over the raw `[1,5,10710]` output
  read through `MLMultiArray.withUnsafeBufferPointer` — no marshalling cost.
  **Important:** the output buffer is padded (channel stride 10720 ≠ 10710);
  always index via `strides`, never densely.
- Letterbox is GPU (CoreImage) into a pooled BGRA pixel buffer; Core ML takes
  the CVPixelBuffer directly.

Measured on an M-series Mac (same ANE architecture as recent iPhones):
**~5 ms/frame** for letterbox + inference + decode; the 30 s / 60 fps
end-to-end video check runs ~2.9× realtime including H.264 decode.

## Layout

```
ios_tracker/
  BallTracker.xcodeproj      Xcode 16 project (filesystem-synchronized — new
                             files under BallTracker/ are picked up automatically)
  BallTracker/
    App/                     entry point, tab root
    Detection/               Letterbox (CoreImage), BallDetector (Core ML + NMS),
                             PlayerDetector (Vision human rectangles → near/far boxes)
    Tracking/                Mat/Kalman/IMM/BallTrackManager — line-by-line port
                             of mobile/lib/engine (the validated Dart engine) —
                             plus RallyDetector (port of pipeline/rally_detector.py:
                             carry suppression, segment cutting, serving-pattern
                             HMM) and TrackerEngine glue
    Live/                    AVCaptureSession camera pipeline + overlay UI
    Video/                   AVAssetReader batch processing + synced playback overlay
    Resources/
      ball_best.mlpackage    the exported model (regenerate with spikes/export_coreml.py)
  ParityCheck/               macOS harness (not in the app target)
  run_parity_check.sh        detection parity vs Python golden + 10 tracker scenarios
  run_video_check.sh         end-to-end video run on real footage
```

## Verification

```bash
ios_tracker/run_parity_check.sh
#  PASS box count / IoU 0.976 / conf Δ0.001 vs the Ultralytics .pt golden boxes
#  PASS all 10 tracker oracle scenarios (same as pipeline/ball_tracker.py self-test)

ios_tracker/run_video_check.sh [video.mp4]   # end-to-end on real footage
```

Both harnesses compile the *actual app sources*, so they exercise the shipped
letterbox/decode/NMS/tracker code paths on the Mac's ANE.

## Running on an iPhone

1. `open ios_tracker/BallTracker.xcodeproj`
2. Select the BallTracker target → Signing & Capabilities → set your Team
   (bundle id `com.build2launch.balltracker`).
3. Build & run on a device (iOS 17+). The ANE is not available in the
   simulator — Core ML falls back to CPU/GPU there.

Tracking quality expectations: detection is tuned for landscape, court-behind-
baseline framing at 960-wide analysis scale (the same regime as the anya
pipeline). The tracker's `activeBallConf = 0.10` operating threshold and all
gate constants come from the validated pipeline config.

## Known limitations

- One legacy MPEG-4 Part 2 test file (`archive/out.mp4`) fails mid-read in
  AVFoundation's video-composition output on macOS; H.264/HEVC (all iPhone
  footage) is unaffected.
- Speed is reported in px/s (no court calibration in this app by design).
- Video mode keeps buffers at the composition render size; 4K files work but
  process slower than 1080p.
