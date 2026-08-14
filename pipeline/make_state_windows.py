"""
make_state_windows.py
=====================
Three-class window builder: {dead=0, transition=1, active=2}.

Sources windows from `timeline_cache.npz` (extract_timeline.py) rather than the
rally-span pose cache, which buys two things the binary pipeline could not have:

  * FAR-serve rallies. `energy_telemetry_cache.json` was built from
    `_near_rallies`, so no bbox exists for far rallies and they were untrainable.
    They are live time — the near player is receiving and rallying — and they are
    55% of all rallies.
  * Genuine deep-dead. The telemetry cache stopped 6s after each rally end, so
    every dead sample was the most live-looking dead that exists.

WHY A TRANSITION CLASS
----------------------
The boundary is measurably non-discriminative. Mean bbox speed 1s AFTER a point
ends (0.123) is HIGHER than 1s before it (0.091); the streams only separate ~3s
out. Forcing a hard active/dead call there trains the model against noise. The
binary pipeline half-acknowledged this with a `boundary` flag that merely
EXCLUDED those windows from training — discarding the hardest examples. Here the
ambiguous band is a label the model can actually predict.

Labels are assigned by the state at the window's END frame, matching how the
binary pipeline and evaluate_events.py both score:

    E within [t - band_before, t + band_after]  -> transition
    else live_mask[E]                      -> active
    else                                   -> dead

Sampling is transition-anchored with seeded quotas (the spec's stage 2), and
every sampled window is written to a manifest CSV so the draw is reproducible
and hand-auditable.

Usage:
    python pipeline/make_state_windows.py --out /Volumes/Anya/Data/state_windows.npz
    python pipeline/make_state_windows.py --band_after_sec 2.5 --n_end 600
"""

import os
import sys
import csv
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from global_features import bbox_stream, N_GLOBAL, NAMES as GLOBAL_NAMES
from parse_ground_truth import load_rallies, live_mask, transitions, gt_path
from extract_timeline import TIMELINE_NPZ

L = 60             # timesteps per window after resampling
WIN_SEC = 2.0      # window length in seconds
CLASSES = {0: "dead", 1: "transition", 2: "active"}


def contiguous_runs(frame_idx):
    """[(row_start, row_end, frame_start, frame_end)] for each contiguous run."""
    runs, s = [], 0
    for i in range(1, len(frame_idx) + 1):
        if i == len(frame_idx) or frame_idx[i] != frame_idx[i - 1] + 1:
            runs.append((s, i - 1, int(frame_idx[s]), int(frame_idx[i - 1])))
            s = i
    return runs


def clip_features(clip_dir, want_global=True):
    """Per-row features [N, D] plus frame_idx/fps, Stream B computed per run.

    Velocity must not be differenced across a seek boundary, so Stream B is
    built independently within each contiguous run.
    """
    d = np.load(os.path.join(clip_dir, TIMELINE_NPZ))
    pose, bbox = d["pose"], d["bbox"]
    fps = float(d["fps"])
    frame_idx = (d["frame_idx"] if "frame_idx" in d.files
                 else np.arange(int(d["start"]), int(d["start"]) + len(pose)))
    if not want_global:
        return pose, frame_idx, fps
    G = np.full((len(pose), N_GLOBAL), np.nan, dtype=np.float32)
    for r0, r1, _, _ in contiguous_runs(frame_idx):
        seg = bbox[r0:r1 + 1]
        bl = [None if np.isnan(b[0]) else tuple(b) for b in seg]
        G[r0:r1 + 1] = bbox_stream(bl, fps)
    return np.concatenate([pose, G], axis=1), frame_idx, fps


def label_frames(clip_dir, fps, total, band_before_sec, band_after_sec):
    """(live mask, transition mask, transitions, band_after_frames) over [0, total).

    The band is ASYMMETRIC, for two independent reasons that agree:

    1. Geometry. A window is labelled by its END frame and spans [E-2s, E], so a
       window ending before a transition is PURE — it contains no post-transition
       frames. Only windows ending in (t, t+2s) are genuinely mixed. Nothing
       before t needs to be called ambiguous.
    2. Measurement. Held-out P(active) against offset from the point-end runs
       0.79-0.91 from -4s to -0.5s (confidently live), crosses 0.5 at ~+1.1s, and
       settles to 0.22-0.33 by +2s. The uncertainty lives entirely after the
       event.

    A symmetric +/-1.5s band therefore mislabels a second of confidently-live
    windows as transition, while cutting off before the model has settled. The
    default is 0.5s before (absorbing GT annotation jitter) and 2.0s after
    (matching both the window length and the measured settling time).
    """
    r = load_rallies(clip_dir, fps)
    lm = live_mask(r, total)
    tr = transitions(r)
    tf = sorted([e["frame"] for e in tr["point_end"]] +
                [e["frame"] for e in tr["point_start"]])
    b0 = int(round(band_before_sec * fps))
    b1 = int(round(band_after_sec * fps))
    tm = np.zeros(total, dtype=bool)
    for f in tf:
        tm[max(0, f - b0):min(total, f + b1 + 1)] = True
    return lm, tm, tr, b1


def sample_ends(rng, pool, k):
    """Draw up to k frames from a pool without replacement."""
    if len(pool) == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    if len(pool) <= k:
        return np.asarray(pool, dtype=np.int64)
    return rng.choice(np.asarray(pool, dtype=np.int64), size=k, replace=False)


def build_clip(clip_dir, args, rng):
    name = os.path.basename(clip_dir)
    feat, frame_idx, fps = clip_features(clip_dir, want_global=not args.no_global)
    total = int(frame_idx[-1]) + 1
    win_frames = max(2, round(fps * WIN_SEC))
    idx_grid = np.linspace(0, win_frames - 1, L).round().astype(int)

    lm, tm, tr, band = label_frames(clip_dir, fps, total,
                                    args.band_before_sec, args.band_after_sec)

    # row lookup: absolute frame -> row, -1 if uncovered
    pos = np.full(total, -1, dtype=np.int64)
    inr = frame_idx < total
    pos[frame_idx[inr]] = np.flatnonzero(inr)

    def usable(E):
        s = E - win_frames + 1
        if s < 0 or E >= total:
            return False
        return pos[s] >= 0 and pos[E] >= 0 and pos[E] - pos[s] == win_frames - 1

    bracket = int(round(args.bracket_sec * fps))
    guard = band + int(round(args.guard_sec * fps))

    ends = [e["frame"] for e in tr["point_end"]]
    starts = [e["frame"] for e in tr["point_start"]]

    # Transition-anchored pools: every offset in the bracket around each event.
    pool_end = [f + o for f in ends for o in range(-bracket, bracket + 1) if usable(f + o)]
    pool_start = [f + o for f in starts for o in range(-bracket, bracket + 1) if usable(f + o)]
    # Interior pools: at least guard away from any transition.
    cand = np.flatnonzero(~tm)
    deep_live = [int(f) for f in cand if lm[f] and usable(int(f))
                 and (not len(tr["point_end"]) or
                      min(abs(int(f) - t) for t in ends + starts) >= guard)]
    deep_dead = [int(f) for f in cand if not lm[f] and usable(int(f))
                 and (not len(tr["point_end"]) or
                      min(abs(int(f) - t) for t in ends + starts) >= guard)]

    def per(n):
        """Split a global quota evenly across clips."""
        return max(1, int(round(n / max(args.n_clips_hint, 1))))

    picks = np.unique(np.concatenate([
        sample_ends(rng, pool_end, per(args.n_end)),
        sample_ends(rng, pool_start, per(args.n_start)),
        sample_ends(rng, deep_live, per(args.n_live)),
        sample_ends(rng, deep_dead, per(args.n_dead)),
    ]).astype(np.int64))

    X, meta = [], []
    for E in picks:
        E = int(E)
        s = E - win_frames + 1
        rows = feat[pos[s]:pos[E] + 1]
        W = rows[idx_grid]
        n_present = int(np.sum(~np.isnan(W[:, 0])))
        if n_present < args.min_present * L:
            continue
        if tm[E]:
            y = 1
        elif lm[E]:
            y = 2
        else:
            y = 0
        # distance to the nearest transition, signed (+ = after the event)
        near = min(ends + starts, key=lambda t: abs(E - t)) if (ends or starts) else None
        X.append(W)
        meta.append({
            "wid": f"{name}_e{E}", "clip": name, "end_frame": E,
            "label": y, "class": CLASSES[y],
            "live_frac": float(lm[max(0, s):E + 1].mean()),
            "d_transition_s": (float(E - near) / fps) if near is not None else float("nan"),
            "n_present": n_present, "fps": round(fps, 3),
        })
    print(f"[state] {name}: {len(meta)} windows  "
          f"dead={sum(m['label']==0 for m in meta)} "
          f"trans={sum(m['label']==1 for m in meta)} "
          f"active={sum(m['label']==2 for m in meta)}")
    return np.asarray(X, dtype=np.float32), meta


def main():
    ap = argparse.ArgumentParser(description="Build 3-class dead/transition/active windows")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", default="/Volumes/Anya/Data/state_windows.npz")
    ap.add_argument("--manifest", default="/Volumes/Anya/Data/state_windows.csv")
    ap.add_argument("--band_before_sec", type=float, default=0.5,
                    help="seconds BEFORE a transition labelled `transition` — small, "
                         "since windows ending before a transition are pure (see "
                         "label_frames); this only absorbs GT annotation jitter")
    ap.add_argument("--band_after_sec", type=float, default=2.0,
                    help="seconds AFTER a transition labelled `transition` — this is "
                         "where the ambiguity actually lives; matches the 2s window "
                         "length and the measured settling time")
    ap.add_argument("--bracket_sec", type=float, default=3.0,
                    help="+/- seconds around a transition to draw anchored windows from")
    ap.add_argument("--guard_sec", type=float, default=2.0,
                    help="extra distance beyond the band required for deep live/dead")
    ap.add_argument("--n_end", type=int, default=1600)
    ap.add_argument("--n_start", type=int, default=800)
    ap.add_argument("--n_live", type=int, default=800)
    ap.add_argument("--n_dead", type=int, default=800)
    ap.add_argument("--min_present", type=float, default=0.5,
                    help="drop windows with less than this fraction of frames posed")
    ap.add_argument("--no_global", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_derived", action="store_true",
                    help="also accept derived_ground_truth.json (model-inferred labels)")
    args = ap.parse_args()

    import parse_ground_truth as _pgt
    _pgt.ALLOW_DERIVED = args.allow_derived

    if args.clips:
        dirs = [os.path.join(args.data_root, c) for c in args.clips]
    else:
        dirs = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
                if os.path.isfile(os.path.join(args.data_root, d, TIMELINE_NPZ))
                and gt_path(os.path.join(args.data_root, d))]
    # Enforce label trust even for an explicit --clips list, and say what was
    # dropped rather than silently shrinking the dataset. A stale timeline cache
    # from a previous run is exactly how an excluded clip sneaks back in.
    kept = []
    for d in dirs:
        name = os.path.basename(d)
        if gt_path(d) is None:
            why = _pgt.EXCLUDED.get(name, "no ground_truth.json")
            print(f"[exclude] {name}: {why}")
            continue
        kept.append(d)
    dirs = kept
    if not dirs:
        raise SystemExit(f"no clips with {TIMELINE_NPZ} — run extract_timeline.py first")
    args.n_clips_hint = len(dirs)
    print(f"[init] {len(dirs)} clips, seed={args.seed}, "
          f"band=-{args.band_before_sec}s/+{args.band_after_sec}s")

    rng = np.random.default_rng(args.seed)
    Xs, metas = [], []
    for d in dirs:
        try:
            X, m = build_clip(d, args, rng)
        except Exception as e:
            print(f"[WARN] {os.path.basename(d)}: {e}")
            continue
        if len(X):
            Xs.append(X); metas.extend(m)
    if not Xs:
        raise SystemExit("no windows built")

    X = np.concatenate(Xs)
    n_global = 0 if args.no_global else N_GLOBAL
    arrs = {"X": X,
            "wid": np.array([m["wid"] for m in metas]),
            "clip": np.array([m["clip"] for m in metas]),
            "y": np.array([m["label"] for m in metas], dtype=np.int64),
            "end_frame": np.array([m["end_frame"] for m in metas]),
            "n_present": np.array([m["n_present"] for m in metas]),
            "n_pose": np.array(51), "n_global": np.array(n_global),
            "global_names": np.array([] if args.no_global else GLOBAL_NAMES),
            "band_before_sec": np.array(args.band_before_sec),
            "band_after_sec": np.array(args.band_after_sec),
            "seed": np.array(args.seed)}
    np.savez_compressed(args.out, **arrs)

    with open(args.manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metas[0].keys()) + ["hand_label"])
        w.writeheader()
        for m in metas:
            w.writerow({**m, "hand_label": ""})

    y = arrs["y"]
    print(f"\n[done] {len(X)} windows {X.shape}  "
          f"dead={int((y==0).sum())} transition={int((y==1).sum())} active={int((y==2).sum())}")
    print(f"[done] -> {args.out}\n[done] manifest (hand_label column blank) -> {args.manifest}")


if __name__ == "__main__":
    main()
