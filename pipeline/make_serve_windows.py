"""
make_serve_windows.py
====================
Stage 2 of the learned serve detector: build labeled 2s pose windows for a
binary serve / not-serve classifier, emitted in the same format train_active.py
consumes (so the GRU trainer is reused unchanged).

POSITIVES — from serve_pose_cache.npz (the run-ups): windows whose end sits in
    [serve - POS_PRE, serve + POS_POST] so the trailing 2s contains the toss +
    contact.
NEGATIVES — from the point-end pose_cache.npz (rally spans): windows ending well
    after the rally start (clean rally / dead-time motion, no serve), sampled to
    roughly balance the positives.

Output: serve_windows.npz + serve_labels.json (all audited=True; label 1=serve).

Usage:
    python pipeline/make_serve_windows.py
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_windows import _window, L, WIN_SEC
from extract_serve_pose import SERVE_NPZ, SERVE_META
from extract_pose import POSE_NPZ, POSE_META
from optimize_energy import _near_rallies

POS_PRE, POS_POST = 0.3, 0.7     # window-end offset (s) around the serve for positives
POS_STEP          = 0.15
NEG_AFTER_SEC     = 2.5          # negatives must end this long after the rally start
NEG_PER_RALLY     = 7            # ~balance the ~7 positive windows per serve


def _win_frames(fps):
    return max(2, round(fps * WIN_SEC))


def build(clip_dirs):
    X, meta = [], []
    for c in clip_dirs:
        name = os.path.basename(c)
        # ── positives ──
        sp, sm = os.path.join(c, SERVE_NPZ), os.path.join(c, SERVE_META)
        if os.path.isfile(sp) and os.path.isfile(sm):
            spose = np.load(sp); smeta = json.load(open(sm)); fps = smeta["fps"]
            wf = _win_frames(fps)
            step = max(1, round(fps * POS_STEP))
            for si, s in enumerate(smeta["serves"]):
                arr = spose[f"s{si}"]; start, f0 = s["start"], s["f0"]
                lo = start - round(fps * POS_PRE); hi = start + round(fps * POS_POST)
                for E in range(lo, hi + 1, step):
                    W = _window(arr, f0, E, wf)
                    if np.sum(~np.isnan(W[:, 0])) == 0:
                        continue
                    X.append(W); meta.append((f"{name}_serve{si}_e{E}", name, 1))

        # ── negatives (rally / dead motion) ──
        pp, pm = os.path.join(c, POSE_NPZ), os.path.join(c, POSE_META)
        if os.path.isfile(pp) and os.path.isfile(pm):
            rpose = np.load(pp); rmeta = json.load(open(pm)); fps = rmeta["fps"]
            wf = _win_frames(fps)
            after = round(fps * NEG_AFTER_SEC)
            for ri, r in enumerate(rmeta["rallies"]):
                arr = rpose[f"r{ri}"]; start, end, span_end = r["start"], r["end"], r["span_end"]
                lo = start + after; hi = span_end
                if hi - lo < wf:
                    continue
                for E in np.linspace(lo, hi, NEG_PER_RALLY).round().astype(int):
                    W = _window(arr, start, int(E), wf)
                    if np.sum(~np.isnan(W[:, 0])) == 0:
                        continue
                    X.append(W); meta.append((f"{name}_neg{ri}_e{int(E)}", name, 0))

        n = sum(1 for m in meta if m[1] == name)
        print(f"[serve-windows] {name}: {n} windows")
    return np.asarray(X, dtype=np.float32), meta


def main():
    ap = argparse.ArgumentParser(description="Build serve/not-serve pose windows")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", default="/Volumes/Anya/Data/serve_windows.npz")
    ap.add_argument("--labels", default="/Volumes/Anya/Data/serve_labels.json")
    args = ap.parse_args()

    if args.clips:
        clip_dirs = [os.path.join(args.data_root, c) for c in args.clips]
    else:
        clip_dirs = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
                     if os.path.isfile(os.path.join(args.data_root, d, SERVE_NPZ))]

    X, meta = build(clip_dirs)
    if len(X) == 0:
        raise SystemExit("No serve windows — run extract_serve_pose.py first.")

    arrs = {"X": X,
            "wid": np.array([m[0] for m in meta]),
            "clip": np.array([m[1] for m in meta]),
            "auto": np.array([m[2] for m in meta]),
            "boundary": np.zeros(len(meta), dtype=int)}
    np.savez_compressed(args.out, **arrs)
    labels = {m[0]: {"auto": m[2], "label": m[2], "audited": True} for m in meta}
    json.dump(labels, open(args.labels, "w"), indent=0)

    npos = sum(m[2] for m in meta)
    print(f"\n[done] {len(meta)} windows  serve={npos} not-serve={len(meta)-npos}  -> {args.out}")


if __name__ == "__main__":
    main()
