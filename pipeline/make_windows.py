"""
make_windows.py
==============
Stage 2 of the active/dead classifier: turn cached pose sequences into labeled
2-second windows.

Per near rally we emit, keyed by the window's END frame E:
  - ~11 windows slid across the transition (E from gt_end - 1.5s to +1.5s),
  - an early-rally ACTIVE anchor and a late-tail DEAD anchor.
Each window covers [E - 2s, E], is resampled to a fixed L=60 timesteps (so 30
and 60 fps clips align), and is auto-labeled by its END frame:
    ACTIVE (1) if E <= gt_end, else DEAD (0).
Windows whose end is within a ±DEADBAND of gt_end are flagged `boundary` — the
genuinely ambiguous ones the audit tool surfaces for review.

Output (at data_root):
    windows.npz    X [N, 60, 51] float32 (NaN where pose missing), plus parallel
                   metadata arrays (clip, rally, end_off, gt_end_off, auto,
                   boundary, anchor, n_present)
    labels.json    {wid: {auto, label, audited}} seeded from auto labels; the
                   audit tool edits this, training reads it.

Usage:
    python pipeline/make_windows.py
    python pipeline/make_windows.py --clips 36
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_pose import POSE_NPZ, POSE_META
from optimize_energy import _near_rallies, TELEMETRY_CACHE
from global_features import (bbox_stream, bboxes_from_telemetry_rally,
                             N_GLOBAL, NAMES as GLOBAL_NAMES)

L           = 60      # timesteps per window after resampling
WIN_SEC     = 2.0
N_SIDE      = 5       # slides each side of the transition
STEP_SEC    = 0.33
DEADBAND_SEC = 0.3    # default +/- band around gt_end flagged as boundary (overridable)


def _window(arr, start, E, win_frames):
    """Rows for frames [E-win_frames+1 .. E] (NaN where outside the rally),
    resampled to L timesteps by nearest index."""
    W = np.full((win_frames, arr.shape[1]), np.nan, dtype=np.float32)
    for i in range(win_frames):
        t = (E - win_frames + 1 + i) - start
        if 0 <= t < arr.shape[0]:
            W[i] = arr[t]
    idx = np.linspace(0, win_frames - 1, L).round().astype(int)
    return W[idx]


def _rally_ends(fps, start, center, span_end, win_frames):
    """(E, is_anchor) window end frames: slides centered on the transition
    `center`, plus an early-rally active anchor and a late-tail dead anchor."""
    step = max(1, round(fps * STEP_SEC))
    ends = [(center + k * step, False) for k in range(-N_SIDE, N_SIDE + 1)]
    ends.append((start + win_frames - 1, True))   # early-rally ACTIVE anchor
    ends.append((span_end, True))                 # late-tail DEAD anchor
    out, seen = [], set()
    for E, anc in ends:
        E = int(max(start, min(span_end, E)))
        if E not in seen:
            seen.add(E); out.append((E, anc))
    return out


def build(clip_dirs, deadband_sec=DEADBAND_SEC, transitions=None, global_stream=True):
    X, meta = [], []
    for c in clip_dirs:
        name = os.path.basename(c)
        npz_p, meta_p = os.path.join(c, POSE_NPZ), os.path.join(c, POSE_META)
        if not (os.path.isfile(npz_p) and os.path.isfile(meta_p)):
            print(f"[skip] {name}: no pose cache")
            continue
        pose = np.load(npz_p)
        pm = json.load(open(meta_p))
        fps = pm["fps"]
        win_frames = max(2, round(fps * WIN_SEC))
        deadband = round(fps * deadband_sec)

        # Stream B source. Telemetry rallies are enumerated in the same order as
        # the pose arrays (both come from _near_rallies), so index ri lines up.
        tel = None
        if global_stream:
            tel_p = os.path.join(c, TELEMETRY_CACHE)
            if os.path.isfile(tel_p):
                tel = json.load(open(tel_p))["rallies"]
            else:
                print(f"[skip] {name}: no telemetry cache for global stream")
                continue

        for ri, r in enumerate(pm["rallies"]):
            arr = pose[f"r{ri}"]
            start, end, span_end = r["start"], r["end"], r["span_end"]
            if tel is not None:
                # Concat Stream B (global bbox trajectory) onto Stream A (pose).
                bb = bboxes_from_telemetry_rally(tel[ri], start, span_end)
                G = bbox_stream(bb, fps)
                if G.shape[0] != arr.shape[0]:                # defensive: align lengths
                    T = min(G.shape[0], arr.shape[0])
                    arr, G = arr[:T], G[:T]
                arr = np.concatenate([arr, G], axis=1)
            # Label boundary = human transition frame. With --transitions, skip
            # rallies with no marked transition (avoid mixing player-based and
            # ball-based labels, which sit ~1.7s apart).
            ref = end
            if transitions is not None:
                key = f"{name}_r{ri}"
                if key not in transitions:
                    continue
                ref = int(transitions[key])
            for E, anchor in _rally_ends(fps, start, ref, span_end, win_frames):
                W = _window(arr, start, E, win_frames)
                n_present = int(np.sum(~np.isnan(W[:, 0])))
                if n_present == 0:
                    continue
                auto = 1 if E < ref else 0            # ACTIVE strictly before the transition
                boundary = abs(E - ref) <= deadband
                X.append(W)
                meta.append({
                    "wid": f"{name}_r{ri}_e{E}",
                    "clip": name, "rally": ri,
                    "end_off": E - start, "gt_end_off": ref - start,
                    "auto": auto, "boundary": int(boundary),
                    "anchor": int(anchor), "n_present": n_present,
                })
        print(f"[windows] {name}: {sum(1 for m in meta if m['clip']==name)} windows")
    return np.asarray(X, dtype=np.float32), meta


def main():
    ap = argparse.ArgumentParser(description="Build labeled 2s pose windows from pose caches")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", default="/Volumes/Anya/Data/windows.npz")
    ap.add_argument("--labels", default="/Volumes/Anya/Data/labels.json")
    ap.add_argument("--deadband_sec", type=float, default=DEADBAND_SEC,
                    help="+/- band around the transition flagged boundary")
    ap.add_argument("--transitions", default=None,
                    help="transitions.json from mark_transitions.py — label windows by the human "
                         "player-transition frame instead of the ball-based GT end")
    ap.add_argument("--no_global", action="store_true",
                    help="omit Stream B (global bbox trajectory) — pose-only, 51 dims, "
                         "the pre-two-stream feature set")
    args = ap.parse_args()

    if args.clips:
        clip_dirs = [os.path.join(args.data_root, c) for c in args.clips]
    else:
        clip_dirs = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
                     if os.path.isfile(os.path.join(args.data_root, d, POSE_NPZ))]

    transitions = json.load(open(args.transitions)) if args.transitions else None
    X, meta = build(clip_dirs, deadband_sec=args.deadband_sec, transitions=transitions,
                    global_stream=not args.no_global)
    if len(X) == 0:
        raise SystemExit("No windows built — run extract_pose.py first.")

    keys = ["clip", "rally", "end_off", "gt_end_off", "auto", "boundary", "anchor", "n_present"]
    arrs = {"X": X}
    # Feature layout travels with the tensor so train/eval never guess the dims.
    arrs["n_pose"] = np.array(51)
    arrs["n_global"] = np.array(0 if args.no_global else N_GLOBAL)
    arrs["global_names"] = np.array([] if args.no_global else GLOBAL_NAMES)
    arrs["wid"] = np.array([m["wid"] for m in meta])
    for k in keys:
        arrs[k] = np.array([m[k] for m in meta])
    np.savez_compressed(args.out, **arrs)

    # Seed labels.json. With --transitions the auto label IS the human-derived
    # label (from the marked transition), so mark it audited. Otherwise preserve
    # any prior per-window audited labels.
    existing = json.load(open(args.labels)) if os.path.isfile(args.labels) else {}
    use_trans = transitions is not None
    labels = {}
    for m in meta:
        prev = existing.get(m["wid"])
        if use_trans:
            labels[m["wid"]] = {"auto": m["auto"], "label": m["auto"], "audited": True}
        else:
            labels[m["wid"]] = {
                "auto": m["auto"],
                "label": prev["label"] if prev and prev.get("audited") else m["auto"],
                "audited": bool(prev["audited"]) if prev else False,
            }
    json.dump(labels, open(args.labels, "w"), indent=0)

    n = len(meta)
    n_bound = sum(m["boundary"] for m in meta)
    n_active = sum(m["auto"] for m in meta)
    print(f"\n[done] {n} windows  active={n_active} dead={n-n_active}  "
          f"boundary={n_bound}  -> {args.out}")
    print(f"[done] labels seeded -> {args.labels}")


if __name__ == "__main__":
    main()
