"""
Phase-0 / Spike-B prerequisite: export the pipeline's YOLO models to ONNX and
prove the exported graph is numerically faithful to the original .pt weights.

Steps
-----
1. Export pipeline/models/{ball_best,yolo26n}.pt -> spikes/models/*.onnx
   at the exact imgsz the pipeline uses (960).
2. Grab a real analysis frame (960x540) from a test video.
3. Run detection twice on that frame:
     a) the original .pt via Ultralytics
     b) the exported .onnx via Ultralytics (loads onnxruntime under the hood)
   and assert the two box/conf sets match within tolerance.
4. Additionally dump a raw golden fixture for the Dart spike:
     - the exact letterboxed input tensor Ultralytics feeds the net
     - the raw ONNX output tensor
     - the decoded boxes
   so the Dart pre/post-processing can be checked against exact numbers.

Run:  python spikes/export_onnx.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODELS_SRC = REPO / "pipeline" / "models"
OUT_DIR = Path(__file__).resolve().parent / "models"
FIX_DIR = Path(__file__).resolve().parent / "fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIX_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_W, ANALYSIS_H = 960, 540
IMGSZ = 960  # Config.PLAYER_IMGSZ / ACTIVE_BALL_IMGSZ

# Test video may live in the main checkout (large, untracked in the worktree).
# Override with:  python spikes/export_onnx.py /path/to/video.mp4
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
    """Greedy match by IoU; report worst deltas."""
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


def main() -> int:
    from ultralytics import YOLO

    frame = get_frame(TEST_VIDEO)
    cv2.imwrite(str(FIX_DIR / "frame_960x540.png"), frame)
    print(f"[frame] {TEST_VIDEO.name} -> {frame.shape} saved")

    summary = {}
    for name, conf in (("ball_best", 0.05), ("yolo26n", 0.25)):
        pt_path = MODELS_SRC / f"{name}.pt"
        print(f"\n=== {name} (conf={conf}) ===")

        pt = YOLO(str(pt_path))
        onnx_path = Path(pt.export(format="onnx", imgsz=IMGSZ, opset=12,
                                   dynamic=False, simplify=True))
        dest = OUT_DIR / f"{name}.onnx"
        onnx_path.replace(dest)
        print(f"[export] {dest.relative_to(REPO)}  ({dest.stat().st_size//1024} KB)")

        r_pt = pt(frame, verbose=False, conf=conf, imgsz=IMGSZ)[0]
        onnx_model = YOLO(str(dest))
        r_ox = onnx_model(frame, verbose=False, conf=conf, imgsz=IMGSZ)[0]

        b_pt, b_ox = boxes_of(r_pt), boxes_of(r_ox)
        cmp = match_sets(b_pt, b_ox)
        summary[name] = cmp
        print(f"[parity] {cmp}")

        (FIX_DIR / f"{name}_boxes_pt.json").write_text(json.dumps(b_pt, indent=2))
        (FIX_DIR / f"{name}_boxes_onnx.json").write_text(json.dumps(b_ox, indent=2))

    (FIX_DIR / "parity_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    all_pass = all(v["pass"] for v in summary.values())
    print("\nRESULT:", "PASS ✅" if all_pass else "REVIEW ⚠️")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
