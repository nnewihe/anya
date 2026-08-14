"""Export ball_best.pt to Core ML at 480x288 for the tracked-ROI path.

Same export decisions as export_coreml.py (NMS-free raw [1,5,N] head, fp16,
image input with baked 1/255 scaling) at a small input the iOS tracker runs on
crops: a 320x192 source crop letterboxed to 480x288 sees the ball at exactly
the same effective resolution as the shipped 1920x1088 full-frame model, so
per-crop accuracy carries over. 288 (not 270/272) because YOLO inputs must be
multiples of the max stride, 32.

Output: spikes/models/ball_best_roi.mlpackage

Run:  python spikes/export_coreml_roi.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS_SRC = REPO / "pipeline" / "models"
OUT_DIR = Path(__file__).resolve().parent / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMGSZ = (288, 480)  # (h, w), multiples of 32


def main() -> int:
    from ultralytics import YOLO

    pt = YOLO(str(MODELS_SRC / "ball_best.pt"))
    exported = Path(pt.export(format="coreml", nms=False, half=True, imgsz=list(IMGSZ)))
    dest = OUT_DIR / "ball_best_roi.mlpackage"
    if dest.exists():
        shutil.rmtree(dest)
    exported.replace(dest)
    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"[export] {dest.relative_to(REPO)}  ({size_mb:.1f} MB)")

    # Stable output name for the Swift decoder (the converter auto-names it).
    import coremltools as ct
    m = ct.models.MLModel(str(dest))
    spec = m.get_spec()
    old_name = spec.description.output[0].name
    if old_name != "detections":
        ct.utils.rename_feature(spec, old_name, "detections")
        ct.models.MLModel(spec, weights_dir=m.weights_dir).save(str(dest))
        print(f"[rename] output {old_name!r} -> 'detections'")

    inp = spec.description.input[0]
    print(f"[check] input {inp.type.imageType.width}x{inp.type.imageType.height}, "
          f"output '{spec.description.output[0].name}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
