"""
Re-export the models shipped in mobile/assets/models/, fixing a regression
from the validated spike decision (spikes/FINDINGS.md) and adding a
rectangular-input optimization.

Two problems fixed here:

1. **NMS regression.** The shipped `ball_best.onnx` has a baked-in
   NonMaxSuppression node ([1,300,6] output). spikes/export_onnx.py /
   FINDINGS.md explicitly tested and REJECTED this variant: NMS/TopK ops
   aren't CoreML/NNAPI-supported, so ONNX Runtime partitions the graph and
   runs the NMS subgraph (and everything gated behind its dynamic-shaped
   outputs) on CPU instead of the ANE/NPU. Ultralytics' export defaults to
   `nms=False`; the shipped asset was produced with `nms=True` explicitly,
   silently reversing the spike's conclusion. This script re-exports NMS-free
   (raw [1,5,N] ball output; Dart does NMS — see mobile/lib/engine/inference.dart).

2. **Wasted letterbox padding.** The pipeline's analysis frame is 960x540;
   the models were exported square at imgsz=960, so every inference pads
   43.75% of the tensor with gray. Re-exporting at the stride-32-rounded
   rectangular size (960x544) removes nearly all the padding: ~1.77x fewer
   pixels per inference, same weights, same accuracy (validated below).

Three exports are produced:
  - yolo26n.onnx        (players, RECT 960x544) -> mobile/assets/models/
  - ball_best.onnx       (ball, RECT 960x544)     -> mobile/assets/models/
                          used for whole-frame ball detection (every frame,
                          both the rally Engine and DeadTimeEngine stage 1).
  - ball_best_far_crop.onnx (ball, SQUARE 960x960) -> mobile/assets/models/
                          used ONLY by DeadTimeEngine's far-region native
                          crop (match_telemetry.dart FixedFarCropSource),
                          which pads an arbitrary-aspect crop into a square
                          tensor via letterboxCropToTensor. That code path
                          isn't wired into the shipped UI yet, but keeping
                          it on the ORIGINAL square geometry avoids silently
                          breaking its tuned crop-based far-serve detection
                          (see CutterConfig.farCropTopExtendFrac) when it
                          does ship.

For every export, this script validates parity against the source .pt
weights using the exact match_sets() method from spikes/export_onnx.py
(greedy IoU match, tolerances iou>=0.98 conf_delta<=0.02) on the SAME
reference frame the original spike used, so the parity bar is unchanged.

Run:  python spikes/export_mobile_models.py [/path/to/video.mp4]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS_SRC = REPO / "pipeline" / "models"
MOBILE_ASSETS = REPO / "mobile" / "assets" / "models"
FIX_DIR = Path(__file__).resolve().parent / "fixtures"
FIX_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_W, ANALYSIS_H = 960, 540

_CANDIDATES = [
    REPO / "archive" / "out.mp4",
    Path("/Users/tennis/Documents/Code/Laptop/src/anya/archive/out.mp4"),
    Path("/Users/tennis/Documents/Code/Laptop/src/anya/archive/farside_serve_viz.mp4"),
]
TEST_VIDEO = next((p for p in (
    [Path(sys.argv[1])] if len(sys.argv) > 1 else []) + _CANDIDATES if p.exists()),
    _CANDIDATES[0])


def get_frame(path: Path, idx: int = 300) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    target = min(idx, max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {target} of {path}")
    return cv2.resize(frame, (ANALYSIS_W, ANALYSIS_H), interpolation=cv2.INTER_AREA)


def boxes_of(result) -> list[dict]:
    out = []
    if result.boxes is None:
        return out
    for b in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        out.append({
            "cls": int(b.cls[0]),
            "conf": round(float(b.conf[0]), 5),
            "xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })
    out.sort(key=lambda d: (-d["conf"], d["cls"]))
    return out


def match_sets(a: list[dict], b: list[dict], iou_tol=0.98, conf_tol=0.02) -> dict:
    """Greedy match by IoU; report worst deltas. Verbatim from export_onnx.py."""
    def iou(p, q):
        ax1, ay1, ax2, ay2 = p; bx1, by1, bx2, by2 = q
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    used = set()
    worst_iou, worst_conf = 1.0, 0.0
    for da in a:
        best, bj = 0.0, -1
        for j, db in enumerate(b):
            if j in used or db["cls"] != da["cls"]:
                continue
            v = iou(da["xyxy"], db["xyxy"])
            if v > best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj)
            worst_iou = min(worst_iou, best)
            worst_conf = max(worst_conf, abs(da["conf"] - b[bj]["conf"]))
    return {
        "n_pt": len(a), "n_onnx": len(b), "matched": len(used),
        "worst_iou": round(worst_iou, 4), "worst_conf_delta": round(worst_conf, 4),
        "pass": len(a) == len(b) == len(used)
                and worst_iou >= iou_tol and worst_conf <= conf_tol,
    }


def graph_info(onnx_path: Path) -> dict:
    import onnx
    m = onnx.load(str(onnx_path))
    g = m.graph
    ops = {}
    for n in g.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return {
        "input_shape": [d.dim_value or d.dim_param for d in g.input[0].type.tensor_type.shape.dim],
        "output_shape": [d.dim_value or d.dim_param for d in g.output[0].type.tensor_type.shape.dim],
        "has_nms": ops.get("NonMaxSuppression", 0) > 0,
        "node_count": len(g.node),
    }


def export_one(name: str, imgsz, dest: Path, conf: float, frame: np.ndarray) -> dict:
    from ultralytics import YOLO

    pt_path = MODELS_SRC / f"{name}.pt"
    print(f"\n=== {name}  imgsz={imgsz}  -> {dest.name} (conf={conf}) ===")

    pt = YOLO(str(pt_path))
    onnx_path = Path(pt.export(format="onnx", imgsz=imgsz, opset=12,
                               dynamic=False, simplify=True, nms=False))
    dest.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.replace(dest)
    print(f"[export] {dest}  ({dest.stat().st_size // 1024} KB)")

    r_pt = pt(frame, verbose=False, conf=conf, imgsz=imgsz)[0]
    onnx_model = YOLO(str(dest))
    r_ox = onnx_model(frame, verbose=False, conf=conf, imgsz=imgsz)[0]

    b_pt, b_ox = boxes_of(r_pt), boxes_of(r_ox)
    cmp = match_sets(b_pt, b_ox)
    info = graph_info(dest)
    cmp.update(info)
    print(f"[parity] {cmp}")
    return cmp


def main() -> int:
    frame = get_frame(TEST_VIDEO)
    cv2.imwrite(str(FIX_DIR / "frame_960x540.png"), frame)
    print(f"[frame] {TEST_VIDEO.name} -> {frame.shape} saved")

    summary = {}
    # 1. Player, rectangular 960x544 (only ever used on the fixed 960x540
    #    analysis frame -> no far-crop coupling, safe to reshape).
    summary["yolo26n_rect"] = export_one(
        "yolo26n", [544, 960], MOBILE_ASSETS / "yolo26n.onnx", 0.25, frame)

    # 2. Ball, rectangular 960x544 (whole-frame ball detection path).
    summary["ball_best_rect"] = export_one(
        "ball_best", [544, 960], MOBILE_ASSETS / "ball_best.onnx", 0.05, frame)

    # 3. Ball, square 960x960 (far-crop path only; unchanged geometry).
    summary["ball_best_square"] = export_one(
        "ball_best", 960, MOBILE_ASSETS / "ball_best_far_crop.onnx", 0.05, frame)

    (FIX_DIR / "mobile_export_parity.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    all_pass = all(v["pass"] for v in summary.values())
    no_nms = all(not v["has_nms"] for v in summary.values())
    print(f"\nparity: {'PASS' if all_pass else 'REVIEW'}   nms-free: {'YES' if no_nms else 'NO — REGRESSION STILL PRESENT'}")
    return 0 if (all_pass and no_nms) else 1


if __name__ == "__main__":
    sys.exit(main())
