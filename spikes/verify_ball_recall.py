"""
Verify the model re-export didn't change ball-detection recall at the
pipeline's actual operating threshold (ACTIVE_BALL_CONF=0.10), across many
frames -- not just the single conf=0.05 frame export_mobile_models.py checks.

The rally-segment count changed from 1 to 2 after the re-export (baseline
had one continuous 1.74-30.01s segment; the re-exported models produce two
shorter segments). This script determines whether that's a REGRESSION
(new pipeline is missing ball detections the old one found) or a FIX
(the old pipeline's baked NMS was dropping legitimate low-conf detections,
consistent with the audit's prediction that a baked conf threshold could be
above 0.10).

Method: for N sample frames across clip30.mp4, decode ball detections with
plain onnxruntime (no Ultralytics wrapper, so preprocessing exactly matches
what's tested) from:
  (a) OLD model (/tmp/anya_model_backup/ball_best.onnx, square 960x960,
      baked NMS, output [1,300,6])
  (b) NEW model (mobile/assets/models/ball_best.onnx, rect 960x544,
      NMS-free, output [1,5,10710]) with raw-grid decode + NMS(iou=0.45),
      a faithful Python port of inference.dart's _decodeRawGrid/_nms.
Both thresholded at conf >= 0.10 (EngineConfig.activeBallConf), which is
the operating point collect_rally_segments actually uses for the whole-court
ball trace that determines segment boundaries.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
CLIP = REPO / "spikes" / "fixtures" / "clip" / "clip30.mp4"
OLD_MODEL = Path("/tmp/anya_model_backup/ball_best.onnx")
NEW_MODEL = REPO / "mobile" / "assets" / "models" / "ball_best.onnx"
CONF = 0.10
# Dense, not sparse: 60 evenly-spaced samples across 30s (~1 every 0.5s) missed
# every true-positive frame on a first pass -- ball detections are bursty, not
# uniform. Instead densely cover the exact window where baseline (1 segment,
# continuous to 30.01s) and the re-exported models (2 segments, second one
# ending at 15.16s) disagree: run EVERY frame from t=0 to t=16s.
DENSE_START_T = 0.0
DENSE_END_T = 16.0
FPS = 59.94


def letterbox_square(frame_960x540: np.ndarray) -> np.ndarray:
    """Pad 960x540 -> 960x960 (dh=210 top/bottom), matching the OLD model's
    geometry (imgsz=960 square), gray-114 pad, no resize (r=1.0, dw=0)."""
    out = np.full((960, 960, 3), 114, dtype=np.uint8)
    out[210:210 + 540, :, :] = frame_960x540
    tensor = out.astype(np.float32)[:, :, ::-1] / 255.0  # BGR->RGB
    return np.transpose(tensor, (2, 0, 1))[None, ...].copy()


def letterbox_rect(frame_960x540: np.ndarray) -> np.ndarray:
    """Pad 960x540 -> 960x544 (dh=2 top/bottom), matching the NEW model."""
    out = np.full((544, 960, 3), 114, dtype=np.uint8)
    out[2:2 + 540, :, :] = frame_960x540
    tensor = out.astype(np.float32)[:, :, ::-1] / 255.0
    return np.transpose(tensor, (2, 0, 1))[None, ...].copy()


def decode_end2end_nms_baked(out: np.ndarray, conf: float, dh: int) -> list:
    # out: [1,300,6] = x1,y1,x2,y2,conf,cls (already NMS'd by the graph)
    rows = out[0]
    boxes = []
    for r in rows:
        if r[4] < conf:
            continue
        boxes.append((r[0], r[1] - dh, r[2], r[3] - dh, float(r[4])))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2, _ = a
    bx1, by1, bx2, by2, _ = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def nms(boxes, thr=0.45):
    boxes = sorted(boxes, key=lambda b: -b[4])
    keep = []
    rest = list(boxes)
    while rest:
        b = rest.pop(0)
        keep.append(b)
        rest = [o for o in rest if iou(b, o) < thr]
    return keep


def decode_raw_grid(out: np.ndarray, conf: float, dh: int) -> list:
    # out: [1,5,N] = cx,cy,w,h,score (single class, NMS-free)
    grid = out[0]  # [5,N]
    n = grid.shape[1]
    cand = []
    scores = grid[4]
    idx = np.where(scores >= conf)[0]
    for j in idx:
        cx, cy, w, h, s = grid[0, j], grid[1, j], grid[2, j], grid[3, j], grid[4, j]
        cand.append((cx - w / 2, cy - h / 2 - dh, cx + w / 2, cy + h / 2 - dh, float(s)))
    return nms(cand, 0.45)


def main():
    cap = cv2.VideoCapture(str(CLIP))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_f = int(DENSE_START_T * FPS)
    end_f = min(int(DENSE_END_T * FPS), total - 1)
    sample_idx = np.arange(start_f, end_f + 1)

    sess_old = ort.InferenceSession(str(OLD_MODEL), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(str(NEW_MODEL), providers=["CPUExecutionProvider"])
    in_old = sess_old.get_inputs()[0].name
    in_new = sess_new.get_inputs()[0].name

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_f))
    rows = []
    old_total = new_total = 0
    for idx in sample_idx:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t_old = letterbox_square(frame_rgb)
        out_old = sess_old.run(None, {in_old: t_old})[0]
        boxes_old = decode_end2end_nms_baked(out_old, CONF, 210)

        t_new = letterbox_rect(frame_rgb)
        out_new = sess_new.run(None, {in_new: t_new})[0]
        boxes_new = decode_raw_grid(out_new, CONF, 2)

        old_total += len(boxes_old)
        new_total += len(boxes_new)
        rows.append({
            "frame": int(idx),
            "t": round(idx / 59.94, 2),
            "n_old": len(boxes_old),
            "n_new": len(boxes_new),
            "old_confs": [round(b[4], 3) for b in boxes_old],
            "new_confs": [round(b[4], 3) for b in boxes_new],
        })

    summary = {
        "conf_threshold": CONF,
        "n_samples": len(rows),
        "old_total_detections": old_total,
        "new_total_detections": new_total,
        "frames_old_only": sum(1 for r in rows if r["n_old"] > 0 and r["n_new"] == 0),
        "frames_new_only": sum(1 for r in rows if r["n_new"] > 0 and r["n_old"] == 0),
        "frames_both": sum(1 for r in rows if r["n_old"] > 0 and r["n_new"] > 0),
        "frames_neither": sum(1 for r in rows if r["n_old"] == 0 and r["n_new"] == 0),
        "rows": rows,
    }
    out_path = REPO / "spikes" / "fixtures" / "ball_recall_verification.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
