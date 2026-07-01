# On-Device Migration — Spike Findings (A & B)

De-risking spikes for moving the rally-predictor pipeline on-device in Flutter.
All runs on macOS arm64 (Apple Silicon), Flutter 3.44.4 / Dart 3.12.2,
ONNX Runtime 1.15.1 via the `onnxruntime` 1.4.1 pub package.

## Verdict

**Both spikes pass. The on-device Flutter path is feasible.** The full chain —
real mp4 → in-Dart frame decode → in-Dart letterbox → ONNX inference (CoreML) →
detections matching the Python pipeline — runs end-to-end with no server.

One real bottleneck was found (output marshalling, not inference or decode) with
clear, known fixes for the build phase.

## What was proven

| # | Question | Result |
|---|----------|--------|
| 0 | Do the YOLO models export to ONNX faithfully? | ✅ Both export at `imgsz=960`, ~10 MB each. Ball box matches Ultralytics to sub-pixel, conf Δ0.008. |
| B | Does ONNX Runtime run inside Flutter on macOS (hardest cross-platform case)? | ✅ Loads and runs. CoreML EP engages. |
| B | Does the Dart pre/post-processing reproduce the Python golden boxes? | ✅ Exact match: IoU 0.9999 (player) / 0.9995 (ball), conf Δ 0.0 on CPU. |
| A | Can we get 960×540 frames into Dart, and is decode a bottleneck? | ✅ Frames stream in; decode ≈ 2–3 ms/frame — **negligible** vs inference. |
| A+B | Does the whole chain compose on a real video? | ✅ 300 real frames processed; ball detected in 186/300. |

## Model export decisions

- **Player (`yolo26n`)**: YOLO26 is **natively end-to-end / NMS-free** → always
  outputs `[1,300,6]` (x1,y1,x2,y2,conf,cls). Trivial Dart decode.
- **Ball (`ball_best`, YOLO11n)**: export **NMS-free / raw** → `[1,5,18900]`.
  Decode + NMS in Dart (~40 lines, validated). This keeps the ultra-low conf
  thresholds (0.04–0.05) the pipeline relies on **tunable per call-site**.
  - `nms=True` bakes a fixed conf threshold into the graph and **drops** the
    low-conf ball detections the pipeline needs. Rejected.
  - `nms=True, conf=0.001` produces compact `[1,300,6]` and keeps low-conf
    detections, BUT the baked `NonMaxSuppression`/`TopK` ops are **not
    CoreML-supported** → they fall back to CPU and force graph partitioning,
    making inference *slower*. Rejected. **Keep the ball graph NMS-free so it
    runs entirely on the ANE; do NMS in Dart.**

## Performance findings

Controlled bench (pre-baked input tensor, 8 iterations, cool machine):

| Model | EP | ms/frame (pure inference) |
|-------|----|--------------------------:|
| `yolo26n` (players, end2end) | CoreML | ~32 |
| `ball_best` (ball, raw) | CoreML | ~26 |

- **CoreML gives ~3× over debug CPU** (95 → ~30 ms). fp16 on the ANE introduces
  tiny drift (IoU 0.997, conf Δ0.006) — well within tolerance, but **parity
  tests must use tolerances, not exact equality.**
- **Decode is cheap** (~2–3 ms/frame); **preprocess/letterbox in Dart is cheap**
  (~2–3 ms/frame).
- **Bottleneck = output marshalling.** The `onnxruntime` package returns results
  as nested Dart `List`s via `.value`. Converting a raw `[1,5,18900]` output
  (94,500 floats) to nested lists every frame costs ~100 ms/frame — this, not
  inference, dominates the current end-to-end fps. The player output (1,800
  floats) does not suffer this.
- Absolute end-to-end fps in the capstone (5–8 fps) is **thermally throttled and
  marshalling-bound** — not representative of achievable throughput.

### Throughput outlook (with build-phase fixes)
Even unoptimized, a short rally-dense clip processes in minutes — already far
better than uploading a ~1 GB file. Headroom before this is a UX problem:
1. **Read outputs as a typed `Float32List`** (raw buffer) instead of nested
   `List`s — removes the ~100 ms/frame marshalling cost. Highest-priority fix.
2. **Isolate** the decode+inference loop so the UI never blocks.
3. **Exploit the pipeline's existing striding** (players every 4th frame; ball
   selectively by state) — big multiplier on effective per-video-frame cost.
4. GPU/ANE already in use via CoreML; NNAPI/GPU on Android is the analogue.

## Cross-platform decode note

Desktop spike used `Process.start(ffmpeg …)` piping `rgb24` as the `FrameSource`.
Mobile can't spawn processes; the same frames come from `ffmpeg_kit` or a native
`AVAssetReader` (Apple) / `MediaCodec` (Android) plugin behind an identical
`FrameSource` interface. Since decode is not the bottleneck, any of these works;
the native plugin is the cleanest production path.

## Artifacts

- `spikes/export_onnx.py` — export + Ultralytics-vs-ONNX parity check.
- `spikes/make_fixture.py` — builds the golden fixture (input tensor + expected boxes).
- `spikes/models/*.onnx` — exported models (`ball_best` raw, `yolo26n` end2end).
- `spikes/fixtures/` — `input_960.f32`, `*_meta.json`, `spikeB_result_macos_coreml.json`.
- `spikes/onnx_dart/` — Flutter macOS spike app.
  - `lib/main.dart` — Spike B (parity + timed inference).
  - `lib/decode_main.dart` — Spike A capstone (video → frames → inference).

Reproduce:
```bash
python spikes/export_onnx.py            # export + parity
python spikes/make_fixture.py           # golden fixture
cd spikes/onnx_dart
flutter run -d macos --release                          # Spike B
flutter run -d macos --release -t lib/decode_main.dart  # Spike A capstone
```

## Updated risk register

| Prior risk | Status after spikes |
|-----------|--------------------|
| Cross-platform inference runtime (esp. desktop) | **Cleared** — ORT runs in Flutter on macOS with CoreML; package covers all 5 native platforms. |
| Video decode into Dart | **Cleared** — frames stream in; decode negligible. Native plugin is the production path. |
| Numerical parity (Dart vs Python) | **Cleared for inference** — exact on CPU, tolerance-level on ANE. Kalman/homography/HMM ports still to be validated against golden fixtures (Phases 2–3). |
| Mobile throughput | **Reduced** — inference ~30 ms/frame on ANE; real remaining work is the output-marshalling fix + isolate, not a fundamental wall. |

## Recommended next step

Begin **Phase 1** (inference + decode foundation) by turning the spike into a
reusable `FrameSource` + `InferenceSession` Dart module, **leading with the
typed-buffer output fix** and an isolate-based analysis loop.
