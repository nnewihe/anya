# Anya Tennis — native iOS app

Record a match with the phone on a tripod. anya2 runs live, about 10 s behind
the camera, and the highlight reel is ready seconds after you tap Stop. The app
is pure Swift: Core ML for pose, AVFoundation for capture and cutting, and
OpenCV only for camera-bump registration.

| Milestone | What | Status |
|---|---|---|
| M0 | Spikes (pose on the ANE, capture soak); Python changes on main (causal far band, reference still, proxy replay) | not started |
| **M1** | **Batch engine: anya2 stages 5–8 in Swift** | **done: matches main exactly on all 13 corpus clips** |
| M2 | Incremental engine (live, about 7 s behind, finalise at Stop) | next |
| M3 | Perception: letterbox, Core ML pose, camera track, far band | |
| M4 | Capture shell: calibrate → record → stop, resume after a kill | |
| M5 | Live analysis and the reel | |
| M6 | Match info, Save/Share, YouTube | |

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
