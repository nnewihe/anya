"""
reel_eval.py
============
Scores the ORCHESTRATOR: how well it places point ends, and what the reel looks
like as a result.

Two metrics, reported side by side, because they answer different questions and
a change can move them in opposite directions:

  ENDS, as events.  `eval.score` in `point_end` mode -- the same greedy
  one-to-one matcher the detectors are scored with, so an orchestrator end and
  a detector end are directly comparable.  **Truncations are the headline.**

  THE REEL, as footage.  `orchestrator.score_reel` -- whole points, live
  retained, how much dead time survived, how often it cuts.

WHY TRUNCATIONS LEAD.  The brief is conservative ends: overrunning into dead
time is cheap, truncating live tennis is not.  Those are not symmetric errors
and a mean absolute error would average them into one number that hides the
only one that matters.  The shipped pose-only detector achieves ZERO
truncations on every clip at 49.6% recall; recall is what a change is allowed
to buy, and zero truncations is what it must not spend.

Both arms read the SAME cached serve detections and the same pose passes, so
the end policy is the only variable.

Usage:
    python -m pipeline.anya2.reel_eval --arm events curve
    python -m pipeline.anya2.reel_eval --clips 22 24 --lo 0.25 0.35 --dwell 2 3
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from parse_ground_truth import DATA_ROOT, discover  # noqa: E402

from pipeline.anya2.contract import POINT_END  # noqa: E402
from pipeline.anya2.eval import (clip_video, gt_times, labelled_span,  # noqa: E402
                                 score)
from pipeline.anya2.orchestrator import (ReelConfig, build_reel,  # noqa: E402
                                         score_reel)
from pipeline.anya2.rally_eval import DEFAULT_EXCLUDE  # noqa: E402


def run_clip(clip_dir, arm, lo=None, dwell=None):
    video = clip_video(clip_dir)
    cfg = ReelConfig()
    cfg.end_policy = "events" if arm == "events" else "curve"
    cfg.union_per_slot = (arm != "curve_shim")
    if lo is not None:
        cfg.end_lo = lo
    if dwell is not None:
        cfg.end_dwell_s = dwell
    res = build_reel(video, cfg, verbose=False)

    # Ends scored as events.  Only ends the orchestrator actually FOUND are
    # scored -- an "estimated" end is a fallback, not a detection, and crediting
    # it would score the duration prior rather than the policy.
    found = sorted(float(s["end_t"]) for s in res["segments"]
                   if s["end_source"] in ("detected", "curve"))
    gt = gt_times(clip_dir, POINT_END)
    span = labelled_span(clip_dir)
    if span:
        found = [t for t in found if span[0] <= t <= span[1]]
        gt = [t for t in gt if span[0] <= t <= span[1]]
    ev = score(found, gt, POINT_END)
    reel = score_reel(res, clip_dir)
    ev["n_estimated"] = sum(1 for s in res["segments"]
                            if s["end_source"] == "estimated")
    return ev, reel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--arm", nargs="*", default=["events", "curve"])
    ap.add_argument("--lo", nargs="*", type=float, default=[None])
    ap.add_argument("--dwell", nargs="*", type=float, default=[None])
    ap.add_argument("--include-58", action="store_true")
    ap.add_argument("--brief", action="store_true", help="totals only")
    a = ap.parse_args()

    dirs = ([os.path.join(a.data_root, c) for c in a.clips]
            if a.clips else discover(a.data_root))
    if not a.clips and not a.include_58:
        dirs = [d for d in dirs
                if os.path.basename(d.rstrip("/")) not in DEFAULT_EXCLUDE]

    for arm in a.arm:
        los = a.lo if arm == "curve" else [None]
        dws = a.dwell if arm == "curve" else [None]
        for lo in los:
            for dw in dws:
                tag = arm if arm != "curve" else (
                    f"curve lo={lo if lo is not None else 'dflt'} "
                    f"dwell={dw if dw is not None else 'dflt'}")
                E, R = [], []
                if a.brief:
                    print(f"{tag:>34} |", end=" ")
                else:
                    print(f"\n=== {tag} ===")
                    print(f"  {'clip':>4} {'gt':>4} {'found':>5} {'hit':>4} "
                          f"{'R':>6} {'P':>6} {'bias':>6} {'TRUNC':>5} {'est':>4} | "
                          f"{'whole':>5} {'live%':>6} {'reel%':>6} {'dead/pt':>7}")
                for d in dirs:
                    c = os.path.basename(d.rstrip("/"))
                    try:
                        ev, reel = run_clip(d, arm, lo, dw)
                    except Exception as e:            # noqa: BLE001
                        print(f"  {c:>4}  SKIPPED -- {type(e).__name__}: {e}")
                        continue
                    E.append(ev)
                    R.append(reel)
                    if not a.brief:
                        print(f"  {c:>4} {ev['n_gt']:4d} {ev['n_det']:5d} "
                              f"{ev['hit']:4d} {100*ev['recall']:5.1f}% "
                              f"{100*ev['precision']:5.1f}% {ev['bias']:+6.2f} "
                              f"{ev['trunc']:5d} {ev['n_estimated']:4d} | "
                              f"{reel['points_whole']:5d} "
                              f"{100*reel['live_retained']:5.1f}% "
                              f"{100*reel['compression']:5.1f}% "
                              f"{reel['dead_per_point_s']:6.1f}s")
                if E:
                    g = sum(e["n_gt"] for e in E)
                    dt = sum(e["n_det"] for e in E)
                    h = sum(e["hit"] for e in E)
                    tr = sum(e["trunc"] for e in E)
                    es = sum(e["n_estimated"] for e in E)
                    lr = float(np.mean([r["live_retained"] for r in R]))
                    wh = sum(r["points_whole"] for r in R)
                    pts = sum(r["n_points"] for r in R)
                    cm = float(np.mean([r["compression"] for r in R]))
                    dp = float(np.mean([r["dead_per_point_s"] for r in R]))
                    print(f"  {'TOT' if not a.brief else '':>4} {g:4d} {dt:5d} {h:4d} "
                          f"{100*h/max(g,1):5.1f}% {100*h/max(dt,1):5.1f}% "
                          f"{'':6} {tr:5d} {es:4d} | {wh:5d} "
                          f"{100*lr:5.1f}% {100*cm:5.1f}% {dp:6.1f}s"
                          f"   [{wh}/{pts} whole]")


if __name__ == "__main__":
    main()
