"""
integrate_active.py
==================
Fuse the trained pose GRU (primary, 0.8) with the ball trace (secondary, 0.2)
into a per-frame activity signal and derive a point-end, evaluated on the cached
pose + telemetry (no live YOLO). Lets us pick the fusion threshold / offset
before wiring the rule into anya_near_serve.py.

Per rally:
    P_active(f)  = GRU over the 2s pose window ending at f   (slid every STRIDE)
    ball_live(f) = ball seen within BALL_TRACE_SEC
    A(f)         = W_PLAYER * P_active + W_BALL * ball_live
    point_end    = first f (after the start grace) where A < THR for SUSTAIN,
                   then + OFFSET_SEC
Reports predicted end vs the marked player transition AND the ball GT end.

Usage:
    python pipeline/integrate_active.py
    python pipeline/integrate_active.py --thr 0.5 --offset_sec 1.7
"""

import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_active import featurize, make_model
from make_windows import _window, WIN_SEC
from extract_pose import POSE_NPZ, POSE_META
from optimize_energy import _near_rallies, TELEMETRY_CACHE

W_PLAYER, W_BALL = 0.8, 0.2
BALL_TRACE_SEC   = 0.7
STRIDE           = 3      # evaluate P_active every STRIDE frames (interp between)
START_GRACE_SEC  = 2.0    # skip the leading window's-worth (incomplete pose windows read as unsure)
SMOOTH_SEC       = 0.4    # centered moving-average on the fused signal


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = make_model()
    m.load_state_dict(ck["state"]); m.eval()
    return m


def p_active_series(model, arr, fps):
    """P_active at each frame offset (0..T-1), slid every STRIDE and interpolated."""
    T = arr.shape[0]
    win = max(2, round(fps * WIN_SEC))
    offs = list(range(0, T, STRIDE))
    batch = np.stack([_window(arr, 0, o, win) for o in offs])   # [M,60,51]
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(featurize(batch)))).numpy()
    return np.interp(np.arange(T), offs, p)


def ball_live_series(f_by_id, start, T, fps):
    win = max(1, round(fps * BALL_TRACE_SEC))
    seen = np.array([1.0 if (f_by_id.get(start + t) and f_by_id[start + t]["ball"]) else 0.0
                     for t in range(T)])
    out = np.zeros(T)
    for t in range(T):
        out[t] = 1.0 if seen[max(0, t - win + 1): t + 1].any() else 0.0
    return out


def predict_end(A, fps, thr, sustain_sec, start_off):
    sustain = max(1, round(fps * sustain_sec))
    grace = round(fps * START_GRACE_SEC)
    run = 0
    for t in range(start_off + grace, len(A)):
        run = run + 1 if A[t] < thr else 0
        if run >= sustain:
            return t - sustain + 1   # first frame of the sustained-dead run
    return len(A) - 1


def main():
    ap = argparse.ArgumentParser(description="Evaluate pose+ball fusion point-end on cached data")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--model", default="/Volumes/Anya/Data/active_model.pt")
    ap.add_argument("--transitions", default="/Volumes/Anya/Data/transitions.json")
    ap.add_argument("--thr", type=float, default=0.45)          # tuned operating point
    ap.add_argument("--sustain_sec", type=float, default=1.0)   # sustained-dead before ending
    ap.add_argument("--offset_sec", type=float, default=0.0)    # add ~1.7s to align player-end -> ball-end
    args = ap.parse_args()

    model = load_model(args.model)
    trans = json.load(open(args.transitions)) if os.path.isfile(args.transitions) else {}

    clips = [d for d in sorted(os.listdir(args.data_root))
             if os.path.isfile(os.path.join(args.data_root, d, POSE_NPZ)) and _near_rallies(os.path.join(args.data_root, d))]

    err_player, err_ball = [], []
    for name in clips:
        cdir = os.path.join(args.data_root, name)
        pose = np.load(os.path.join(cdir, POSE_NPZ))
        pm = json.load(open(os.path.join(cdir, POSE_META)))
        tel = json.load(open(os.path.join(cdir, TELEMETRY_CACHE)))
        fps = pm["fps"]
        f_by_id = {}
        for r in tel["rallies"]:
            for k, v in r["frames"].items():
                f_by_id[int(k)] = v

        for ri, r in enumerate(pm["rallies"]):
            arr = pose[f"r{ri}"]
            start, end, span_end = r["start"], r["end"], r["span_end"]
            T = arr.shape[0]
            P = p_active_series(model, arr, fps)
            B = ball_live_series(f_by_id, start, T, fps)
            A = W_PLAYER * P + W_BALL * B
            sm = max(1, round(fps * SMOOTH_SEC))
            A = np.convolve(A, np.ones(sm) / sm, mode="same")
            t_end = predict_end(A, fps, args.thr, args.sustain_sec, 0)
            pred = start + t_end + round(args.offset_sec * fps)
            err_ball.append((pred - end) / fps)
            key = f"{name}_r{ri}"
            if key in trans:
                err_player.append((pred - trans[key]) / fps)

    def summ(e, lab):
        if not e:
            print(f"  {lab}: n/a"); return
        e = np.array(e)
        print(f"  {lab}: MAE {np.abs(e).mean():.2f}s  mean {e.mean():+.2f}s  "
              f"median {np.median(e):+.2f}s  n={len(e)}")

    print(f"[fusion] W_player={W_PLAYER} W_ball={W_BALL} thr={args.thr} "
          f"sustain={args.sustain_sec}s offset={args.offset_sec}s")
    summ(err_player, "vs player transition")
    summ(err_ball,   "vs ball GT end      ")


if __name__ == "__main__":
    main()
