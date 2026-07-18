"""Export ball_best.pt to Core ML (.mlpackage) for the dedicated iOS ball tracker.

Export decisions (carried over from spikes/FINDINGS.md and the ball-model NMS
regression):
- NMS-free raw output ``[1,5,N]``: baked NonMaxSuppression/TopK ops are not
  ANE-supported and force graph partitioning + CPU fallback. Decode and NMS
  happen in Swift, keeping conf thresholds tunable per call-site.
- fp16 weights (the ANE is fp16-native anyway; parity uses tolerances).
- Rectangular 544x960 input to match 16:9 camera/video frames (~1.77x fewer
  pixels than a padded 960x960 square).
- Image input type: on iOS the model takes a CVPixelBuffer directly, with the
  1/255 scaling baked into the graph — no manual tensor packing in Swift.

Outputs
-------
spikes/models/ball_best.mlpackage      the model bundled into the iOS app
spikes/fixtures/coreml_ball_boxes.json golden decoded boxes for the Swift tests
spikes/fixtures/coreml_parity.json     .pt vs CoreML parity summary

Run:  python spikes/export_coreml.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
MODELS_SRC = REPO / "pipeline" / "models"
OUT_DIR = Path(__file__).resolve().parent / "models"
FIX_DIR = Path(__file__).resolve().parent / "fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIX_DIR.mkdir(parents=True, exist_ok=True)

IMGSZ = (1088, 1920)  # (h, w) — 16:9 letterbox, matches the mobile ONNX re-export
CONF = 0.1         # ball parity threshold, same as export_onnx.py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_onnx import boxes_of, match_sets, get_frame, TEST_VIDEO  # noqa: E402


def main() -> int:
    from ultralytics import YOLO

    frame_png = FIX_DIR / "frame_1920x1080.png"
    if frame_png.exists():
        frame = cv2.imread(str(frame_png))
    else:
        frame = get_frame(TEST_VIDEO)
        cv2.imwrite(str(frame_png), frame)
    print(f"[frame] {frame.shape}")

    pt = YOLO(str(MODELS_SRC / "ball_best.pt"))
    exported = Path(pt.export(format="coreml", nms=False, half=True, imgsz=list(IMGSZ)))
    dest = OUT_DIR / "ball_best.mlpackage"
    if dest.exists():
        shutil.rmtree(dest)
    exported.replace(dest)
    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"[export] {dest.relative_to(REPO)}  ({size_mb:.1f} MB)")

    # The converter auto-names the output (e.g. "var_1223"), which shifts
    # between exports. Rename it to a stable name the Swift code can rely on.
    import coremltools as ct
    m = ct.models.MLModel(str(dest))
    spec = m.get_spec()
    old_name = spec.description.output[0].name
    if old_name != "detections":
        ct.utils.rename_feature(spec, old_name, "detections")
        ct.models.MLModel(spec, weights_dir=m.weights_dir).save(str(dest))
        print(f"[rename] output {old_name!r} -> 'detections'")

    # Parity: same frame, same Ultralytics letterbox+NMS, two backends.
    r_pt = pt(frame, verbose=False, conf=CONF, imgsz=list(IMGSZ))[0]
    cm = YOLO(str(dest), task="detect")
    r_cm = cm(frame, verbose=False, conf=CONF, imgsz=list(IMGSZ))[0]

    b_pt, b_cm = boxes_of(r_pt), boxes_of(r_cm)
    # fp16 + Apple's letterbox resize drift a little more than ONNX fp32 did.
    cmp = match_sets(b_pt, b_cm, iou_tol=0.90, conf_tol=0.05)
    print(f"[parity] {cmp}")
    print(f"[pt   ] {b_pt}")
    print(f"[coreml] {b_cm}")

    (FIX_DIR / "coreml_ball_boxes.json").write_text(json.dumps({
        "frame": frame_png.name,
        "imgsz": list(IMGSZ),
        "conf": CONF,
        "source_size": [frame.shape[1], frame.shape[0]],
        "boxes_pt": b_pt,
        "boxes_coreml": b_cm,
    }, indent=2))
    (FIX_DIR / "coreml_parity.json").write_text(json.dumps(cmp, indent=2))

    print("\nRESULT:", "PASS ✅" if cmp["pass"] else "REVIEW ⚠️")
    return 0 if cmp["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
