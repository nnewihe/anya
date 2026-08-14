"""Export ball_best.pt to TensorFlow Lite (LiteRT) for the native Android ball
tracker (android_tracker/).

Android counterpart of spikes/export_coreml.py. The same export decisions carry
over (see spikes/FINDINGS.md and the ball-model NMS regression):

- NMS-free raw output ``[1,5,N]`` (or its transpose): baked NonMaxSuppression /
  TopK ops force graph partitioning and knock the model off the NNAPI/GPU
  accelerators. Decode + single-class NMS happen in Kotlin (BallDetector.kt),
  keeping the ultra-low conf thresholds the tennis pipeline relies on tunable
  per call-site — exactly like the iOS path.
- fp16 weights (``half=True``). NNAPI/GPU delegates are fp16-native; parity uses
  tolerances.
- Rectangular 544x960 input to match 16:9 camera/video frames.

Unlike Core ML, the Ultralytics TFLite export takes a float32 NHWC tensor in
[0,1] (the Kotlin Letterbox does the /255), and — depending on the exporter
version — may emit box coords **normalized to [0,1]** rather than in pixels.
BallDetector.kt auto-detects which, but this script records the actual output
shape and coordinate scale so the Kotlin decode can be reconciled if a future
Ultralytics changes the layout.

Outputs
-------
android_tracker/app/src/main/assets/ball_best.tflite   model bundled into the app
spikes/fixtures/tflite_probe.json                      output shape + coord scale
spikes/fixtures/tflite_parity.json                     .pt vs TFLite parity summary

Run (in the export venv — torch 2.7 / numpy 1.26 / ultralytics 8.4, plus
`tensorflow` for the parity read-back):  python spikes/export_tflite.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS_SRC = REPO / "pipeline" / "models"
ASSETS_DIR = REPO / "android_tracker" / "app" / "src" / "main" / "assets"
FIX_DIR = Path(__file__).resolve().parent / "fixtures"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
FIX_DIR.mkdir(parents=True, exist_ok=True)

IMGSZ = (544, 960)  # (h, w) — 16:9 letterbox, matches export_coreml.py
CONF = 0.05         # ball parity threshold, same as export_onnx.py / export_coreml.py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import boxes_of, match_sets, get_frame, TEST_VIDEO  # noqa: E402


def _find_tflite(export_path: Path) -> Path:
    """Ultralytics returns the SavedModel dir or the .tflite; normalise to the
    fp16 .tflite file."""
    p = Path(export_path)
    if p.is_file() and p.suffix == ".tflite":
        return p
    # Prefer the float16 export, then any tflite in the produced dir.
    cands = sorted(p.rglob("*float16.tflite")) or sorted(p.rglob("*.tflite"))
    if not cands:
        raise SystemExit(f"no .tflite found under {p}")
    return cands[0]


def main() -> int:
    from ultralytics import YOLO

    frame_png = FIX_DIR / "frame_960x540.png"
    if frame_png.exists():
        frame = cv2.imread(str(frame_png))
    else:
        frame = get_frame(TEST_VIDEO)
        cv2.imwrite(str(frame_png), frame)
    print(f"[frame] {frame.shape}")

    pt = YOLO(str(MODELS_SRC / "ball_best.pt"))
    exported = pt.export(format="tflite", nms=False, half=True, imgsz=list(IMGSZ))
    tflite_src = _find_tflite(Path(exported))
    dest = ASSETS_DIR / "ball_best.tflite"
    if dest.exists():
        dest.unlink()
    shutil.copyfile(tflite_src, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"[export] {dest.relative_to(REPO)}  ({size_mb:.1f} MB)  <- {tflite_src.name}")

    # The SavedModel dir and intermediate ONNX are export scratch onnx2tf leaves
    # behind in pipeline/models — drop them so the repo keeps only the source
    # weights and the bundled .tflite (which now lives in the app assets).
    for scratch in (MODELS_SRC / "ball_best_saved_model", MODELS_SRC / "ball_best.onnx"):
        if scratch.is_dir():
            shutil.rmtree(scratch, ignore_errors=True)
        elif scratch.exists():
            scratch.unlink()

    # Probe the exported model directly so the Kotlin decode assumptions are on
    # record: output tensor shape, and whether box coords are normalized.
    probe = {"imgsz": list(IMGSZ), "tflite": tflite_src.name}
    try:
        import tensorflow as tf  # noqa: F401
        interp = tf.lite.Interpreter(model_path=str(dest))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        probe["input_shape"] = [int(x) for x in inp["shape"]]
        probe["output_shape"] = [int(x) for x in out["shape"]]

        # Ultralytics letterbox to (h, w), 0..1 RGB NHWC.
        h, w = IMGSZ
        r = min(w / frame.shape[1], h / frame.shape[0])
        nw, nh = int(round(frame.shape[1] * r)), int(round(frame.shape[0] * r))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((h, w, 3), 114, np.uint8)
        px, py = (w - nw) // 2, (h - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        interp.set_tensor(inp["index"], rgb[None])
        interp.invoke()
        raw = interp.get_tensor(out["index"])[0]           # [5,N] or [N,5]
        chan = raw if raw.shape[0] == 5 else raw.T          # -> [5,N]
        coords = chan[:4]
        max_coord = float(np.max(coords))
        probe["coords_are_pixels"] = bool(max_coord > 4.0)
        probe["max_coord"] = max_coord
        print(f"[probe] output={probe['output_shape']}  "
              f"coords_are_pixels={probe['coords_are_pixels']} (max={max_coord:.3f})")
    except Exception as e:  # tensorflow not installed in every venv
        probe["probe_error"] = str(e)
        print(f"[probe] skipped read-back ({e}); BallDetector auto-detects at runtime")

    (FIX_DIR / "tflite_probe.json").write_text(json.dumps(probe, indent=2))

    # Parity: same frame, same Ultralytics letterbox+NMS, .pt vs TFLite backend.
    r_pt = pt(frame, verbose=False, conf=CONF, imgsz=list(IMGSZ))[0]
    tfl = YOLO(str(dest), task="detect")
    r_tf = tfl(frame, verbose=False, conf=CONF, imgsz=list(IMGSZ))[0]
    b_pt, b_tf = boxes_of(r_pt), boxes_of(r_tf)
    # fp16 + TF's letterbox resize drift a little more than ONNX fp32 did.
    cmp = match_sets(b_pt, b_tf, iou_tol=0.85, conf_tol=0.05)
    print(f"[parity] {cmp}")
    print(f"[pt    ] {b_pt}")
    print(f"[tflite] {b_tf}")
    (FIX_DIR / "tflite_parity.json").write_text(json.dumps(cmp, indent=2))

    print("\nRESULT:", "PASS ✅" if cmp["pass"] else "REVIEW ⚠️")
    return 0 if cmp["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
