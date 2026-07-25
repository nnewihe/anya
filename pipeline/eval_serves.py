"""
eval_serves.py
=============
Validate the current serve detector — the foundation of the planned two-pass
(find all serves, then find each point-end). Runs the PointStartSystem serve
logic (dwell -> toss -> ratio-shift) over the WHOLE video and scores the detected
serve frames against the near-side GT `start` frames.

Serve-only scan: each time the state machine reaches ACTIVE we record the frame
and reset to WAITING to look for the next serve (deduped by MIN_GAP_SEC).

Reports, per clip and overall: recall (GT serves found), precision (detections
that matched a GT serve), and timing error (detected - GT, seconds) for matches
within TOL_SEC.

Usage:
    python pipeline/eval_serves.py --clips 36        # smoke test
    python pipeline/eval_serves.py                   # all GT clips with court cache
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anya_near_serve import PointStartSystem, TrackData, MatchState
from optimize_energy import _near_rallies, _video_path, _court_cache_path, ANALYSIS_SIZE

TOL_SEC     = 1.5    # a detection within this of a GT serve counts as a match
MIN_GAP_SEC = 3.0    # min spacing between accepted serves (dedupe)
PLAYER_STRIDE = 10
PLAYER_IMGSZ  = 640
BALL_IMGSZ    = 256


def _load_corners(video_path):
    return np.array(json.load(open(_court_cache_path(video_path)))["points"], dtype=np.float32)


def detect_serves(clip_dir, player_model, ball_model, device):
    vid = _video_path(clip_dir)
    corners = _load_corners(vid)
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = ANALYSIS_SIZE
    system = PointStartSystem(corners, W, H, fps=int(round(fps)), verbose=False)

    detected, cached_bbox, last_serve = [], None, -10**9
    prev_state = MatchState.WAITING
    min_gap = int(fps * MIN_GAP_SEC)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)

        if (frame_idx - 1) % PLAYER_STRIDE == 0:
            res = player_model.predict(frame, classes=[0], imgsz=PLAYER_IMGSZ, device=device, verbose=False)[0]
            best, best_key = None, None
            for b in res.boxes:
                x, y, x2, y2 = b.xyxy[0].cpu().numpy()
                w, h = x2 - x, y2 - y
                wx, wy = system._get_world_coords(x + w/2, y + h)
                if -2.0 <= wx <= 29.0 and wy <= 38.0:
                    key = abs(wy)
                    if best_key is None or key < best_key:
                        best_key, best = key, (float(x), float(y), float(w), float(h))
            cached_bbox = best

        ball_pos = None
        if system.current_toss_roi is not None:
            rl, rt, rr, rb = system.current_toss_roi
            cl, ct, cr, cb = max(0, int(rl)), max(0, int(rt)), min(W, int(rr)), min(H, int(rb))
            if cr > cl and cb > ct:
                bres = ball_model.predict(frame[ct:cb, cl:cr], imgsz=BALL_IMGSZ, device=device, verbose=False)[0]
                if len(bres.boxes) > 0:
                    bx1, by1, bx2, by2 = bres.boxes[0].xyxy[0].cpu().numpy()
                    ball_pos = (bx1 + cl + (bx2-bx1)/2, by1 + ct + (by2-by1)/2)

        state = system.process_frame(TrackData(frame_idx, cached_bbox, ball_pos))
        if prev_state != MatchState.ACTIVE and state == MatchState.ACTIVE:
            if frame_idx - last_serve >= min_gap:
                detected.append(frame_idx); last_serve = frame_idx
            # reset to hunt for the next serve
            system.state = MatchState.WAITING
            system.active_frame_counter = 0
            system.current_point_start = None
        prev_state = system.state

    cap.release()
    return detected, fps


def score(detected, gt, fps):
    tol = fps * TOL_SEC
    gt_used, errs, tp = set(), [], 0
    for d in detected:
        j, best = -1, tol
        for i, g in enumerate(gt):
            if i in gt_used:
                continue
            if abs(d - g) <= best:
                best, j = abs(d - g), i
        if j >= 0:
            gt_used.add(j); tp += 1; errs.append((d - gt[j]) / fps)
    recall = tp / len(gt) if gt else 0.0
    prec = tp / len(detected) if detected else 0.0
    return recall, prec, tp, errs


def main():
    ap = argparse.ArgumentParser(description="Validate serve detection vs GT")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--ball_model", default="/Users/tennis/Documents/Code/Laptop/src/anya/pipeline/models/ball_best.pt")
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args()

    if args.clips:
        clip_dirs = [os.path.join(args.data_root, c) for c in args.clips]
    else:
        clip_dirs = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
                     if _near_rallies(os.path.join(args.data_root, d))
                     and os.path.isfile(_court_cache_path(_video_path(os.path.join(args.data_root, d)) or ""))]

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[init] {len(clip_dirs)} clips on {device}")
    player_model = YOLO("yolov8n.pt")
    ball_model = YOLO(args.ball_model)

    all_rec, all_prec, all_err = [], [], []
    tot_gt = tot_det = tot_tp = 0
    for c in clip_dirs:
        name = os.path.basename(c)
        gt = sorted(r["start"] for r in _near_rallies(c))
        try:
            detected, fps = detect_serves(c, player_model, ball_model, device)
        except Exception as e:
            print(f"  {name}: FAILED ({e})"); continue
        rec, prec, tp, errs = score(detected, gt, fps)
        all_rec.append(rec); all_prec.append(prec); all_err += errs
        tot_gt += len(gt); tot_det += len(detected); tot_tp += tp
        me = np.median([abs(e) for e in errs]) if errs else float("nan")
        print(f"  {name:>4}: GT={len(gt):2d} detected={len(detected):2d} matched={tp:2d} "
              f"recall={rec:.2f} prec={prec:.2f} |err|med={me:.2f}s")

    print(f"\n[OVERALL] recall={tot_tp/tot_gt if tot_gt else 0:.3f} "
          f"precision={tot_tp/tot_det if tot_det else 0:.3f}  "
          f"(GT={tot_gt} detected={tot_det} matched={tot_tp})")
    if all_err:
        print(f"[OVERALL] timing error: mean {np.mean(all_err):+.2f}s  "
              f"median {np.median(all_err):+.2f}s  |err| median {np.median(np.abs(all_err)):.2f}s")


if __name__ == "__main__":
    main()
