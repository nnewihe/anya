# Anya Tennis — native iOS app

Record a match with the phone on a tripod. anya2 runs live, about 10 s behind
the camera, and the highlight reel is ready seconds after you tap Stop. The app
is pure Swift: Core ML for pose, AVFoundation for capture and cutting, and
OpenCV only for camera-bump registration.

| Milestone | What | Status |
|---|---|---|
| M0 | Spikes; Python changes on main (causal far band, …) | far-band comparison run (below); no PR to main yet |
| **M1** | **Batch engine: anya2 stages 5–8 in Swift** | **done: matches main exactly on all 13 corpus clips** |
| **M2** | **Live engine: about 7 s behind, finalised at Stop** | **done: identical to batch on all 13 clips, under random chunking** |
| **M3** | **Perception: letterbox, Core ML pose, camera track, far band** | **done: see "Perception"** |
| **M4** | **Capture: calibrate → record → stop, interruptions, recovery after a kill** | **built; simulator replay end to end; not yet run on a phone** |
| **M5** | **Live analysis while recording, and the reel** | **built; simulator replay end to end** |
| **M6** | **Match info, Photos, Share, YouTube** | **built; YouTube waits for the OAuth client** |

## The app (`AnyaTennis/`)

| Folder | What |
|---|---|
| `App/` | `AnyaTennisApp` (screens, app model), `MatchController` (one match from calibration to reel) |
| `Screens/` | Calibrate (tap the 4 singles corners), Record (live status), Paused, Finish (match info while the reel is cut), match list, match detail |
| `Capture/` | `CameraSession` (4K30, stabilisation off), `MasterRecorder` (fragmented HEVC .mov), `ProxyRecorder` (2 s near/far proxy segments at 15 Hz), `FrameScaler`, `FileReplay` (Debug: a file instead of the camera) |
| `Analysis/` | `AnalysisScheduler`: segment → pose → camera track → live anya2 → perception log |
| `Storage/` | `MatchStore` (one folder per match), `MatchManifest`, `CalibrationFrame` |
| `Output/` | `ReelExporter` (passthrough cut across parts), `PhotosSaver`, `YouTubeUploader` (resumable, background) |

How a match flows:
1. **Calibrate.** Tap the 4 singles corners on the live preview. The far band (`FarBand`) comes from that frame alone.
2. **Record.** The camera thread writes the master `.mov`, plus two 15 Hz proxies cut into 2 s segments. As each segment lands, `AnalysisScheduler` runs pose on it, then the camera track, then `LiveMatchAnalysis`, and appends to `perception.log`.
3. **Interruption** (a call, or leaving the app). The part is closed and the match is *paused*; Resume starts a new part on the same timeline.
4. **Kill.** On the next launch the master's last fragment gives the part's length and the match is paused. Analysis replays `perception.log` in seconds instead of re-running pose.
5. **Stop.** The backlog drains, `finish()` runs the whole-match steps, and `ReelExporter` cuts the reel by passthrough.

Build: `xcodegen` (spec in `project.yml`; the generated project is checked in), and `tools/build_opencv.sh` once (OpenCV is not checked in).

To replay a corpus clip in the simulator through the real recording path:
```bash
xcrun simctl launch --console-pty booted com.build2launch.anyatennis \
  -AnyaReplay /path/snippet.mp4 -AnyaCorners "82,384,877,378,579,250,419,245"
```

YouTube needs one manual step. Create an iOS OAuth client for `com.build2launch.anyatennis` in Google Cloud. Put its client ID in `project.yml` as `GIDClientID`, and add its reversed ID as a URL scheme. Until then the button is hidden.

## Perception

| Piece | Checked against | Result |
|---|---|---|
| `UltralyticsLetterbox` | `cv2.resize(INTER_LINEAR)` + Ultralytics padding, 30 real frames | bit-exact (fixed-point weights, NEON vertical pass) |
| `PoseModel` decode + NMS, **fp32** | Ultralytics predict, same frames | same people; IoU 1.000, conf 0.0000, keypoints 0.55 px |
| `PoseModel`, **fp16** (shipped) | same | CPU: IoU 0.988, conf 0.016, 2.7 px; judged on the corpus instead |
| `CameraRegistration` | `camera.py` self-test scenes | jostles undone to < 0.3 px |
| `LiveCameraTrack` | `camera.py` run fresh on Data/78's frames | 0.000 px. Main's cached track is one random FLANN-LSH draw, 3–6 px from any re-run |
| live camera bookkeeping | batch fill + smooth | identical under random failures |

**One model instance per pass.** Core ML re-plans whenever an instance's input shape changes. Alternating 640×384 and 960×H on one instance cost 60 ms per sample on the Neural Engine; with two instances it's 6.6 ms (M-series Mac).

**Why ±1 grey matters.** With a floating-point bilinear resize, far confidences moved by up to 0.02 and a far keypoint by 9.7 px, because the far player is small and borderline.

## Anya2Engine (`Packages/Anya2Engine`)

The anya2 pipeline from `pipeline/anya2` on main, one Swift file per Python
module. It's a SwiftPM package, so it builds and tests on macOS as well as iOS.

| If you want to change… | Swift | Python twin |
|---|---|---|
| near serves | `NearServe.swift` | `near_serve.py` |
| far serves | `FarServe.swift` | `far_serve.py` |
| the shared serve skeleton | `ServeCommon.swift` | `serve_common.py` |
| point ends, the live score | `PointEnd.swift` | `point_end.py` |
| the four "between points" pose signals | `NearEnd.swift` | `pipeline/near_end.py` |
| the walking classifier | `Walking.swift` (+ `Resources/walking_model.json`) | `walking/features.py`, `walking/predict.py` |
| reel shape and rules | `Orchestrator.swift` (`ReelConfig`) | `orchestrator.py` |
| player slots | `Tracks.swift` | `tracks.py` |
| court geometry, the two gates | `Court.swift` | `court.py`, `walking/court.py` |
| ramps, running max, gap filling | `Signals.swift` (read the "two families" note) | `signals.py` |
| numpy behaviours we depend on | `NumpyCompat.swift` | — |
| stages 5–8 end to end | `BatchRun.swift` | `run.py` |

Constants keep their Python names (`TROPHY_MIN`, `W_HOME`, …), grouped per
module (`NearServeRules`, `TrackRules`, …). Search
[docs/anya2/NOTES.md](../../docs/anya2/NOTES.md) for a name to find why it has
its value. When you change a constant, change it in both languages, or the
parity tests will tell you.

### Things that have to stay exactly as they are

- **Python's `round` is half-to-even.** Window sizes like `0.7 s × 15 Hz = 10.5`
  round to 10, so use `pythonRound`, never `.rounded()`.
- **The homography is OpenCV's recipe, not just any solve.** The 8×8 system is
  ill-conditioned, and Python maps onto a float32 court (8.22999954 m). Any
  other correct solver differs in the 7th digit, which was enough to flip a few
  walking-classifier splits. See `Homography.init(from:to:)`.
- **Tracks and near_end values are stored as float32**, as Python stores them.
- **`Data.withUnsafeBytes` needs an explicit `UnsafeRawBufferPointer` type.**
  Otherwise a byte read can bind to the deprecated typed overload and silently
  read the wrong bytes.

## Checking it against Python

```bash
# 1. Export main's own stage outputs for the corpus (needs /Volumes/Anya/Data)
~/.pyenv/versions/anya-build/bin/python ios/AnyaTennis/tools/export_replay.py /tmp/anya2_replay

# 2. Compare (release build; ANYA2_CLIPS="22 25" limits the clips)
cd ios/AnyaTennis/Packages/Anya2Engine
ANYA2_REPLAY=/tmp/anya2_replay swift test -c release -Xswiftc -enable-testing
swift run -c release anya2-replay /tmp/anya2_replay
```

`CorpusParityTests` checks, on every clip:
- the homography
- tracks, bit for bit
- walking probability and near_end signals
- the live score
- every detector's events, and the reel's segments, notes included

If you change the walking model, re-export it with `tools/export_walking_model.py`.
