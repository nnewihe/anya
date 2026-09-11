"""
rally_eval.py
=============
Frame-level scoring for the rally confidence curve.

Why this exists beside `eval.py` rather than inside it
------------------------------------------------------
`eval.py` matches EVENTS: greedy one-to-one inside a tolerance, each detection
credited once.  That is the right metric for a detector that emits timestamps
and it is deliberately shared by all three of them.

A curve has no timestamps.  Scored through `eval.py` it would first have to be
collapsed into events, which is exactly the step RALLY_CONFIDENCE.md argues is
throwing the signal away -- an AUC 86.7% curve became a 49.6%-recall event
stream.  Measuring the collapse cannot tell you whether the curve improved.

So this harness scores the curve directly against the per-frame live/dead
timeline, and `eval.py` keeps scoring whatever the orchestrator finally emits.
Two metrics, two files, neither pretending to be the other.

What is being asked
-------------------
    Does this curve separate "a point is in play" from "it is not",
    frame by frame, on footage where the labels can answer?

Reported as AUC (threshold-free, so it cannot be flattered by tuning a cut) and
best-F1 with the threshold that achieved it (which a downstream hysteresis has
to live with).  AUC is the headline: RALLY_CONFIDENCE.md's target is to beat
86.7%, and a change that does not move AUC is not an improvement however good
the reel looks.

Three rules inherited from `eval.py`, for the same reasons
---------------------------------------------------------
PER-CLIP IS THE RESULT, POOLED IS A SUMMARY.  A curve that is excellent on the
clips it can see and silent elsewhere pools well and is not better.  The
per-clip table is always printed and is always the thing to read.

ONLY SCORE THE LABELLED SPAN.  Ground truth labels LIVE intervals and dead time
is their complement, which is a valid inference only BETWEEN the first and last
rally.  Clip 38 is labelled to 206 s of 420 s; counting that tail as 214 s of
correctly-identified dead time would be scoring the curve against the absence
of a label.  `eval.labelled_span` is reused rather than re-derived.

LABEL TRUST IS NOT THIS FILE'S DECISION.  Clips are found with
`parse_ground_truth.discover` and read with `load_rallies`, which is where the
EXCLUDED set (37, 63 -- incompletely labelled) is enforced.  A hand-rolled clip
list would silently score them.

What it does NOT yet handle
---------------------------
`valid` -- the coverage channel RALLY_CONFIDENCE.md defect 3 calls for -- does
not exist on the current curve, so unmeasurable frames are scored as though
they were confidently dead.  That is the behaviour being measured, not an
oversight here: the baseline has to be reproduced as it stands before it is
changed.  When the channel lands, frames where `valid` is false leave the
denominator, and THAT number is not comparable to the ones this prints today.

Usage:
    python -m pipeline.anya2.rally_eval
    python -m pipeline.anya2.rally_eval --clips 22 24 --smooth 4 5 8
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from parse_ground_truth import DATA_ROOT, discover, load_rallies  # noqa: E402

from pipeline.anya2 import rally as RC  # noqa: E402
from pipeline.anya2.eval import clip_video, labelled_span  # noqa: E402

# Clip 58 is 46% of every scored frame in the corpus, so a pooled row that
# includes it is largely clip 58's row.  Excluded by default at the user's
# direction; pass --include-58 to score it.
DEFAULT_EXCLUDE = ("58",)


def live_timeline(clip_dir, fps, n):
    """(live, scored) boolean arrays on the curve's own sample grid.

    `live` is True inside a labelled rally.  `scored` is True inside the
    labelled span -- everything outside it is dropped from both classes rather
    than counted as dead.
    """
    rallies = load_rallies(clip_dir)
    live = np.zeros(n, dtype=bool)
    for r in rallies:
        a = max(0, int(round(r["start_s"] * fps)))
        b = min(n, int(round(r["end_s"] * fps)) + 1)
        if b > a:
            live[a:b] = True
    lo, hi = labelled_span(clip_dir, margin_s=0.0)
    scored = np.zeros(n, dtype=bool)
    a, b = max(0, int(round(lo * fps))), min(n, int(round(hi * fps)) + 1)
    if b > a:
        scored[a:b] = True
    return live, scored


def auc(x, y):
    """Rank AUC (Mann-Whitney U), NaN when either class is empty.

    Rank-based rather than a threshold sweep so ties are handled correctly:
    the curve is smoothed and normalised, and long dead stretches sit at
    identical values.  A sweep would credit those arbitrarily.
    """
    pos, neg = x[y], x[~y]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = np.empty(len(x), dtype=float)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    i = 0
    while i < len(sx):                      # average ranks within a tie group
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[y].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def best_f1(x, y, n_thr=200):
    """Best F1 over a quantile sweep, and the threshold that achieved it."""
    if not y.any() or y.all():
        return float("nan"), float("nan")
    thr = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_thr)))
    best, bt = 0.0, float("nan")
    for t in thr:
        p = x >= t
        tp = float((p & y).sum())
        if tp == 0:
            continue
        f = 2 * tp / (p.sum() + y.sum())
        if f > best:
            best, bt = f, float(t)
    return best, bt


def curve(video, smooth_s=None, tracks_npz=None, arm="current"):
    """One arm's curve. Returns (x, fps).

    Both arms run off the SAME cached pose passes, so the construction is the
    only variable -- the discipline the camera-tracking A/B used.
    """
    if arm == "current":
        # The pre-redesign construction, reconstructed exactly: 4 s smoothing,
        # the single-player shim union, no absence term.  point_end.live_score
        # is gone, but it WAS this, so the historical baseline stays runnable.
        r = RC.compute(video, tracks_npz, smooth_s=(smooth_s or 4.0),
                       per_slot_union=False, w_absence=0.0)
        return np.asarray(r["conf"], dtype=float), float(r["fps"])
    if arm == "rally":
        r = RC.compute(video, tracks_npz, smooth_s=smooth_s)
        return np.asarray(r["conf"], dtype=float), float(r["fps"])
    if arm == "rally_min":
        r = RC.compute(video, tracks_npz, smooth_s=smooth_s, per_slot_union="min")
        return np.asarray(r["conf"], dtype=float), float(r["fps"])
    if arm == "rally_shim":
        # rally.py with the SINGLE-PLAYER union from run._end_signals, i.e.
        # everything except the per-slot team union.  Isolates that change.
        r = RC.compute(video, tracks_npz, smooth_s=smooth_s, per_slot_union=False)
        return np.asarray(r["conf"], dtype=float), float(r["fps"])
    if arm == "rally_noabs":
        # The ablation: rally.py's plumbing with the absence term switched off,
        # so a difference between this and `rally` is the TERM rather than any
        # other change made on the way out of point_end.py.
        r = RC.compute(video, tracks_npz, smooth_s=smooth_s, w_absence=0.0)
        return np.asarray(r["conf"], dtype=float), float(r["fps"])
    raise ValueError(arm)


def score_clip(clip_dir, smooth_s=None, arm="current"):
    video = clip_video(clip_dir)
    if video is None:
        return None, "no video"
    try:
        x, fps = curve(video, smooth_s=smooth_s, arm=arm)
    except FileNotFoundError as e:
        return None, f"missing artifact: {os.path.basename(str(e).split(': ')[-1])}"
    except Exception as e:                       # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    live, scored = live_timeline(clip_dir, fps, len(x))
    xs, ys = x[scored], live[scored]
    a = auc(xs, ys)
    f1, t = best_f1(xs, ys)
    return {"n": int(scored.sum()), "live_frac": float(ys.mean()) if ys.size else float("nan"),
            "auc": a, "f1": f1, "thr": t, "x": xs, "y": ys}, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--smooth", nargs="*", type=float, default=[None],
                    help="smoothing values to sweep (default: the module's own)")
    ap.add_argument("--arm", nargs="*", default=["current"],
                    choices=["current", "rally", "rally_noabs", "rally_shim", "rally_min"])
    ap.add_argument("--include-58", action="store_true",
                    help="score clip 58 too; it is 46%% of the corpus by frames")
    a = ap.parse_args()

    dirs = ([os.path.join(a.data_root, c) for c in a.clips]
            if a.clips else discover(a.data_root))
    if not a.clips and not a.include_58:
        dirs = [d for d in dirs
                if os.path.basename(d.rstrip("/")) not in DEFAULT_EXCLUDE]

    for arm in a.arm:
     for sm in a.smooth:
         label = "module default" if sm is None else f"{sm:g} s"
         print(f"\n=== arm {arm!r}, smoothing {label} ===")
         print(f"  {'clip':>5} {'frames':>7} {'live%':>6} {'AUC':>7} {'bestF1':>7} {'@thr':>6}")
         rows, skipped = [], []
         for d in dirs:
             c = os.path.basename(d.rstrip("/"))
             r, err = score_clip(d, smooth_s=sm, arm=arm)
             if r is None:
                 skipped.append((c, err))
                 continue
             rows.append((c, r))
             print(f"  {c:>5} {r['n']:7d} {100 * r['live_frac']:5.1f}% "
                   f"{100 * r['auc']:6.1f}% {100 * r['f1']:6.1f}% {r['thr']:6.3f}")
         if rows:
             # Pooled over CONCATENATED frames, not an average of per-clip AUCs:
             # an 81-rally clip and a 6-rally clip do not get equal votes.
             X = np.concatenate([r["x"] for _, r in rows])
             Y = np.concatenate([r["y"] for _, r in rows])
             pa = auc(X, Y)
             pf, pt = best_f1(X, Y)
             print(f"  {'POOL':>5} {len(X):7d} {100 * Y.mean():5.1f}% "
                   f"{100 * pa:6.1f}% {100 * pf:6.1f}% {pt:6.3f}   ({len(rows)} clips)")
             mean_auc = float(np.nanmean([r["auc"] for _, r in rows]))
             print(f"  mean per-clip AUC {100 * mean_auc:.1f}%")
         for c, err in skipped:
             print(f"  {c:>5}   SKIPPED -- {err}")


if __name__ == "__main__":
    main()
