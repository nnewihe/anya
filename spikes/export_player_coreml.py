"""Export yolo26n.pt (person detector) to Core ML (.mlpackage) for the iOS
player detector used by the ball tracker's carry-suppression.

Mirrors export_coreml.py's decisions (see spikes/FINDINGS.md):
- NMS-free raw output ``[1, 4+nc, N]`` so the graph stays ANE-friendly; decode
  (person class only) + NMS happen in Swift, matching the ball model.
- fp16 weights (ANE is fp16-native).
- 16:9 rectangular input; players are large objects so a modest 384x640 is
  ample and keeps the second per-frame model cheap.
- Image input (CVPixelBuffer) with 1/255 baked into the graph.

Output
------
spikes/models/yolo26n.mlpackage   the player model bundled into the iOS app

Run:  python spikes/export_player_coreml.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS_SRC = REPO / "pipeline" / "models"
OUT_DIR = Path(__file__).resolve().parent / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMGSZ = (384, 640)  # (h, w) — 16:9-ish, multiples of 32; plenty for large players
CONF = 0.5          # person confidence, matches the pipeline's yolo26n conf=0.5
PERSON_CLASS = 0    # COCO person index


def main() -> int:
    from ultralytics import YOLO

    pt = YOLO(str(MODELS_SRC / "yolo26n.pt"))
    exported = Path(pt.export(format="coreml", nms=False, half=True, imgsz=list(IMGSZ)))
    dest = OUT_DIR / "yolo26n.mlpackage"
    if dest.exists():
        shutil.rmtree(dest)
    exported.replace(dest)
    size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"[export] {dest.relative_to(REPO)}  ({size_mb:.1f} MB)")

    # The converter auto-names the output (e.g. "var_1223"); rename to a stable
    # name the Swift code relies on, exactly as the ball export does.
    import coremltools as ct
    m = ct.models.MLModel(str(dest))
    spec = m.get_spec()
    out = spec.description.output[0]
    old_name = out.name
    if old_name != "detections":
        ct.utils.rename_feature(spec, old_name, "detections")
        m = ct.models.MLModel(spec, weights_dir=m.weights_dir)
        m.save(str(dest))
        print(f"[rename] output {old_name!r} -> 'detections'")

    # Report the output shape + a sample person decode so the Swift decode can be
    # matched against the exported layout.
    spec = ct.models.MLModel(str(dest)).get_spec()
    inp = spec.description.input[0]
    print(f"[input ] {inp.name}: "
          f"{inp.type.imageType.width}x{inp.type.imageType.height}")
    shape = list(spec.description.output[0].type.multiArrayType.shape)
    print(f"[output] {spec.description.output[0].name}: shape={shape} "
          f"(expected [1, 4+nc, N]; person = channel {4 + PERSON_CLASS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
