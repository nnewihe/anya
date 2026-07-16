#!/usr/bin/env python3
"""Score a tracker trace against hand labels.

    ./run_video_check.sh clip.mov            # with DUMP_CSV=/tmp/trace.csv
    python3 score_trace.py --labels labels.json --trace /tmp/trace.csv

Only labelled frames count; unlabelled ones are ignored, so you can label a
couple of rallies and still get an honest read on them.

The metric that matters here is not "live trace %". A tracker that draws a
confident trace across empty court scores 100% coverage and is useless. So we
score against truth:

  recall     of frames where the ball IS visible, how many did we place within
             --tol of it
  precision  of frames where we drew something, how many were actually on the
             ball (a trace on empty court is punished here)
  ghost rate of frames where the ball is NOT visible, how often we drew anyway
  err        localisation error on the hits

`f1` is the headline: it cannot be gamed by drawing more or drawing less.
"""
import argparse
import csv
import json
import math
import sys


def load_trace(path):
    """Row order == decoded frame index (VideoProcessor emits one row/frame)."""
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            x, y = float(r["x"]), float(r["y"])
            live = r["state"] in ("moving", "coasting") and not math.isnan(x) and x >= 0
            out.append((x, y, r["state"]) if live else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--tol", type=float, default=25.0,
                    help="hit radius in SOURCE px (default 25 ~ one ball width)")
    ap.add_argument("--from", dest="lo", type=int, default=None,
                    help="only score frames >= this")
    ap.add_argument("--to", dest="hi", type=int, default=None,
                    help="only score frames <= this. Use --from/--to to exclude "
                         "stretches the tracker is not meant to handle (e.g. a "
                         "stationary ball, which it refuses to track by design).")
    ap.add_argument("--quiet", action="store_true", help="one line, for sweeps")
    a = ap.parse_args()

    with open(a.labels) as f:
        labels = {int(k): v for k, v in json.load(f)["labels"].items()}
    if a.lo is not None:
        labels = {k: v for k, v in labels.items() if k >= a.lo}
    if a.hi is not None:
        labels = {k: v for k, v in labels.items() if k <= a.hi}
    trace = load_trace(a.trace)
    if not labels:
        sys.exit("no labels")

    tp = fn = ghost = quiet_ok = 0
    mislocated = 0
    errs = []
    for fr, lab in sorted(labels.items()):
        got = trace[fr] if 0 <= fr < len(trace) else None
        if lab is None:                       # ball not visible
            if got is None:
                quiet_ok += 1
            else:
                ghost += 1
            continue
        if got is None:                       # ball visible, nothing drawn
            fn += 1
            continue
        d = math.hypot(got[0] - lab["x"], got[1] - lab["y"])
        if d <= a.tol:
            tp += 1
            errs.append(d)
        else:
            mislocated += 1                   # drew, but not on the ball

    ball_frames = tp + fn + mislocated
    drawn = tp + mislocated + ghost
    recall = tp / ball_frames if ball_frames else 0.0
    precision = tp / drawn if drawn else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    empty_frames = ghost + quiet_ok
    ghost_rate = ghost / empty_frames if empty_frames else 0.0
    errs.sort()
    med = errs[len(errs) // 2] if errs else float("nan")
    p90 = errs[int(len(errs) * 0.9)] if errs else float("nan")

    if a.quiet:
        print(f"f1={f1:.3f} recall={recall:.3f} prec={precision:.3f} "
              f"ghost={ghost_rate:.3f} med_err={med:.1f}")
        return

    print(f"labelled frames : {len(labels)}  ({ball_frames} with ball, {empty_frames} without)")
    print(f"tolerance       : {a.tol:.0f} px")
    print()
    print(f"  F1            : {f1:.3f}   <- headline")
    print(f"  recall        : {recall:.3f}   ({tp}/{ball_frames} visible balls found)")
    print(f"  precision     : {precision:.3f}   ({tp}/{drawn} drawn points on the ball)")
    print(f"  ghost rate    : {ghost_rate:.3f}   ({ghost}/{empty_frames} traces drawn with no ball)")
    print()
    print(f"  mislocated    : {mislocated}  (drew, but >{a.tol:.0f}px off)")
    print(f"  missed        : {fn}  (ball visible, drew nothing)")
    print(f"  loc error     : median {med:.1f}px   p90 {p90:.1f}px")


if __name__ == "__main__":
    main()
