# anya2: finding the points in a tennis video (pose only, no ball)

Video in, highlight reel out. The desktop app calls `run.build_reel`.

```
 video(s)
   │  1 join        several GoPro chapters → one file          (pipeline/join.py)
   │  2 calibrate   4 court corners, clicked once, cached       (run.ensure_court)
   │  3 camera      per-frame warp back to the calibration view (camera.py)
   │  4 perceive    pose of every person: near pass + far band  (perceive.py)
   │  5 tracks      ≤2 near + ≤2 far player slots               (tracks.py)
   │  6 end signals walking prob. + near_end's 4 signals        (run._compute_end_signals,
   │                                                             pipeline/near_end.py)
   │  7 detect      near serves, far serves, point ends         (near_serve.py, far_serve.py,
   │                                                             point_end.py)
   │  8 assemble    serves + ends → segments                    (orchestrator.py)
   ▼  9 cut         ffmpeg                                      (run.cut)
 <stem>_anya2_reel.mp4
```

## Where things live

| If you want to change… | Look in | Key names |
|---|---|---|
| what counts as a near serve | `near_serve.py` | `TROPHY_*`, `TOGETHER_BH`, `SERVE_ZONE_*_FT`, `THRESHOLD` |
| what counts as a far serve | `far_serve.py` | `TROPHY_*`, `READY_MEAN_*`, `STILL_*`, `W_STILL`, `THRESHOLD` |
| the shared serve skeleton (candidates, swing, score) | `serve_common.py` | `trophy_candidates`, `swing`, `combine` |
| when a point ends | `point_end.py` | `LIVE_HI`, `LIVE_LO`, `LIVE_SMOOTH_S`, `UNION_NAMES` |
| the four "between points" pose signals | `pipeline/near_end.py` | `SETTLE_*`, `TURN_*`, `READY_KNEE_DEG`, `POCKET_R_BH` |
| reel shape: pre/post roll, merging, min length | `orchestrator.py` | `ReelConfig` |
| how serves become segments | `orchestrator.py` | `build_reel` (numbered recipe) |
| who counts as a player, slot assignment | `tracks.py` | `MIN_H_PX`, `*_BACK_M`, `W_HOME`, `W_CONF`, `W_CONT` |
| court geometry and the two player gates | `court.py` | `X_LO`, `X_HI`, `is_near`, `in_bounds` |
| camera-bump correction | `camera.py` | `MIN_GROUND_INLIERS`, `SMOOTH_SAMPLES`; off with `ANYA_CAMERA_TRACK=0` |
| pose model, sizes, the far crop | `perceive.py` | `NEAR_IMGSZ`, `FAR_IMGSZ`, `FAR_BAND_*` |
| per-run overrides (what the app may set) | `config.py` | `Anya2Config` |
| ramps, running max, gap filling | `signals.py` | read the "two families" note first |

Each module's docstring explains its method in plain terms, and each constant
has a one-line comment.

- **Why each number has its value** (sweeps, per-clip measurements, dead ends):
  [docs/anya2/NOTES.md](../../docs/anya2/NOTES.md). Search it for the constant's name.
- **Behaviour that looks wrong but was kept on purpose:**
  [docs/anya2/QUIRKS.md](../../docs/anya2/QUIRKS.md).

## Conventions

- **Time** in events is always seconds on the source video. Pose arrays are
  indexed by *pose sample* at `fps` (15 Hz). The source frame is `sample * stride`.
- **Image y grows downward.** Elevations are written so that positive means
  higher, in body heights ("`_bh`").
- **Slots:** 0, 1 are near and 2, 3 are far. Singles leaves 1 and 3 empty (NaN).
- **NaN means "not seen".** A gate given NaN answers False. A score with no
  evidence is 0, except where noted (e.g. far `STILL_UNKNOWN`).

## Artifacts (in `workdir.artifact_dir(video)`)

| file | written by | contents |
|---|---|---|
| `<stem>_court_cache.json` | stage 2 | clicked corners |
| `<stem>_anya2_camera.npz` | camera | per-sample warps |
| `<stem>_proxy540.mp4`, `<stem>_farband.mp4` | perceive | decode proxies |
| `<stem>_anya2_near_dets.npz`, `_far_dets.npz` | perceive | all-person poses |
| `<stem>_anya2_tracks.npz` | tracks | 4 player slots |
| `<stem>_anya2_walk_pose.npz`, `_walk.npz`, `_endsig.npz` | run (stage 6) | near-player pose, walking prob., near_end signals |
| `<stem>_anya2_near_serve.json`, `_far_serve.json`, `_point_end.json` | detectors | events |
| `<stem>_anya2_reel.json` | orchestrator | starts, segments, diagnostics |

Every stage reuses its cached file if one exists. Pass `force` (or delete the
file) to recompute. Stages 5, 7 and 8 always rerun and take seconds.

## Running

```bash
python -m pipeline.anya2.camera VIDEO            # or --self-test
python -m pipeline.anya2.perceive VIDEO --roi both
python -m pipeline.anya2.tracks VIDEO --far-dets <stem>_anya2_far_dets.npz
python -m pipeline.anya2.near_serve VIDEO        # also far_serve, point_end
python -m pipeline.anya2.orchestrator VIDEO      # one or more chapter files
python -m pipeline.anya2.eval --mode near_serve --arm mine:_anya2_near_serve.json
```

Use the `anya-build` pyenv environment: Python 3.13 hits a torch import error.

### Headless: one command, a fixed camera, other pose runtimes

```bash
python -m pipeline.anya2 CHAPTER1.MP4 CHAPTER2.MP4 \
    --site site/ --backend ncnn --device cpu -o reel.mp4
```

This never prompts (`headless.py`). The court corners come from a **site profile**:
corners clicked once plus the frame they were clicked on (`site.py`, made with
`python -m pipeline.anya2.site save`). Each new recording is registered against the
profile frame with the camera track's ORB/ground-plane fit, and the corners move with
that warp. A move of more than 15 px raises `NeedsCalibration` instead of guessing.

`ANYA_POSE_BACKEND=ncnn|onnx` (`pose_backend.py`) swaps the pose runtime without
touching the passes. NCNN exports are made at exactly the rectangle PyTorch infers
at; `export_pose parity` measured IoU p5 ≥ 0.968 and conf p95 ≤ 0.015 there. That is
parity only: an `eval.py` corpus run with the backend set is still owed.
`ANYA_FFMPEG_HWACCEL` and `ANYA_SINGLE_DECODE_PROXIES` cut source-decode cost
(`proxy.py`), and `Anya2Config.copy_video` cuts the reel as a lossless stream copy.
The Raspberry Pi service that uses all of this is in `pi/`.

## Changing things safely

1. **If you expect the output to stay the same** (a refactor or a rename),
   snapshot an artifact dir before the change and compare after:

   ```bash
   python -m pipeline.anya2.tools.compare_artifacts BEFORE_DIR AFTER_DIR
   ```

   Every `.npz` array and `.json` file must be identical. From the pose passes
   onward the pipeline is fully deterministic, so any difference is real.
2. **If you expect the output to change** (a threshold, a rule), score it with
   `pipeline.anya2.eval` (serves and ends against `ground_truth.json`, per clip)
   and `orchestrator.score_reel` (whole points kept, dead time, cuts per minute).
   Read the per-clip rows, not only the pooled total.
3. Clips 37 and 63 have labels on disk but are untrusted. Don't score them.
