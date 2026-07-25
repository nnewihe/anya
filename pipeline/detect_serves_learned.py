"""
detect_serves_learned.py
=======================
Whole-video serve detection with the learned serve GRU (Pass 1 of the two-pass).
Runs near-player pose over the video, slides the serve model to get P(serve) per
frame, and peak-picks above a threshold (min-gap deduped) => serve frames. In
--eval mode, scores recall/precision vs the GT near serves.

This is the real serve-detection metric (window-level AUC != detection quality,
since between-points motion only appears here).

Usage:
    python pipeline/detect_serves_learned.py --clips 36 43 21 --eval
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
from extract_serve_pose import _select_near_by_zone
from extract_pose import _normalize, N_KP
from make_windows import _window, WIN_SEC
from train_active import make_model, featurize
from eval_serves import score
from anya_near_serve import PointStartSystem
from optimize_energy import _near_rallies, _video_path, _court_cache_path, ANALYSIS_SIZE

POSE_STRIDE = 2      # run pose every N frames (P interpolated between)
THR         = 0.6
MIN_GAP_SEC = 3.0


def load_gru(path):
    ck = torch.load(path, map_location="cpu")
    m = make_model(); m.load_state_dict(ck["state"]); m.eval()
    return m


def p_serve_series(clip_dir, pose_model, gru, device):
    vid = _video_path(clip_dir)
    corners = np.array(json.load(open(_court_cache_path(vid)))["points"], dtype=np.float32)
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    geom = PointStartSystem(corners, ANALYSIS_SIZE[0], ANALYSIS_SIZE[1], fps=int(round(fps)), verbose=False)
    win = max(2, round(fps * WIN_SEC))
    n_samp = win // POSE_STRIDE

    samp_frames, samp_pose = [], []   # sampled (every POSE_STRIDE) normalized pose (or NaN)
    frame_idx = 0
    P = np.zeros(total + 1, dtype=np.float32)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if (frame_idx - 1) % POSE_STRIDE != 0:
            continue
        frame = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        res = pose_model.predict(frame, imgsz=640, device=device, verbose=False)[0]
        kp, nb = _select_near_by_zone(res, geom)
        samp_frames.append(frame_idx)
        samp_pose.append(_normalize(kp, nb) if kp is not None else np.full(N_KP*3, np.nan, np.float32))

        if len(samp_pose) >= n_samp:
            W = np.stack(samp_pose[-n_samp:])                       # [n_samp,51]
            idx = np.linspace(0, n_samp - 1, 60).round().astype(int)
            Xf = featurize(W[idx][None, ...])
            with torch.no_grad():
                P[frame_idx] = float(torch.sigmoid(gru(torch.tensor(Xf))).item())
    cap.release()
    # fill P between sampled frames
    xs = [f for f in samp_frames if P[f] > 0 or True]
    P = np.interp(np.arange(len(P)), samp_frames, P[samp_frames])
    return P, fps


def peak_pick(P, fps, thr, min_gap_sec):
    min_gap = int(fps * min_gap_sec)
    picks, i, n = [], 0, len(P)
    while i < n:
        if P[i] >= thr:
            j = i
            while j < n and P[j] >= thr:
                j += 1
            seg = P[i:j]
            picks.append(i + int(np.argmax(seg)))
            i = j + min_gap
        else:
            i += 1
    return picks


def main():
    ap = argparse.ArgumentParser(description="Learned whole-video serve detection")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--model", default="/Volumes/Anya/Data/serve_model.pt")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--thr", type=float, default=THR)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args()

    if args.clips:
        clip_dirs = [os.path.join(args.data_root, c) for c in args.clips]
    else:
        clip_dirs = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
                     if _near_rallies(os.path.join(args.data_root, d))]

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    pose_model = YOLO("yolov8n-pose.pt")
    gru = load_gru(args.model)

    tot_gt = tot_det = tot_tp = 0; all_err = []
    for c in clip_dirs:
        name = os.path.basename(c)
        P, fps = p_serve_series(c, pose_model, gru, device)
        det = peak_pick(P, fps, args.thr, MIN_GAP_SEC)
        if args.eval:
            gt = sorted(r["start"] for r in _near_rallies(c))
            rec, prec, tp, errs = score(det, gt, fps)
            tot_gt += len(gt); tot_det += len(det); tot_tp += tp; all_err += errs
            me = np.median([abs(e) for e in errs]) if errs else float("nan")
            print(f"  {name:>4}: GT={len(gt):2d} detected={len(det):2d} matched={tp:2d} "
                  f"recall={rec:.2f} prec={prec:.2f} |err|med={me:.2f}s")
        else:
            print(f"  {name}: {len(det)} serves @ {det}")

    if args.eval and tot_gt:
        print(f"\n[OVERALL] recall={tot_tp/tot_gt:.3f} precision={tot_tp/max(1,tot_det):.3f} "
              f"(GT={tot_gt} detected={tot_det} matched={tot_tp})")
        if all_err:
            print(f"[OVERALL] |err| median {np.median(np.abs(all_err)):.2f}s")


if __name__ == "__main__":
    main()
