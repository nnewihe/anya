#!/usr/bin/env python3
"""Tune the Viterbi solver against hand labels.

Runs the REAL Swift solver (run_video_check.sh) once per trial with VITERBI_*
overrides, scores each trace against the labels, and ranks by F1. No rebuild per
trial — the weights come from the environment.

    python3 sweep.py --video /tmp/clip.mov --labels labels.json \
        --param ACCEL_WEIGHT=0.0015,0.004,0.01 \
        --param FRAME_REWARD=0.5,1.0,2.0

With no --param it runs a small default grid over the weights most likely to be
mistuned: how harshly unexplained acceleration is charged, and how strongly long
paths are rewarded. Those two trade off directly — a high frame reward with a
low accel weight is exactly what let the solver chain noise into fiction.

Each trial costs one full decode+solve of the clip (~15 s for 60 s of video), so
keep grids small: 12 trials ≈ 3 minutes.
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "..", "run_video_check.sh")
SCORER = os.path.join(HERE, "score_trace.py")

DEFAULT_GRID = {
    "ACCEL_WEIGHT": ["0.0015", "0.004", "0.01"],
    "FRAME_REWARD": ["0.5", "1.0", "2.0"],
}


def run_trial(video, labels, overrides, tol, lo=None, hi=None):
    env = dict(os.environ)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        csv_path = tf.name
    env["DUMP_CSV"] = csv_path
    for k, v in overrides.items():
        env["VITERBI_" + k] = v
    try:
        proc = subprocess.run([CHECK, video], env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            return None, f"harness failed: {proc.stderr.strip().splitlines()[-1:] or proc.stdout[-200:]}"
        live = re.search(r"live trace:\s*(\d+)%", proc.stdout)
        cmd = [sys.executable, SCORER, "--labels", labels,
               "--trace", csv_path, "--tol", str(tol), "--quiet"]
        if lo is not None:
            cmd += ["--from", str(lo)]
        if hi is not None:
            cmd += ["--to", str(hi)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            return None, out.stderr.strip()
        m = dict(kv.split("=") for kv in out.stdout.split())
        m["live"] = (live.group(1) + "%") if live else "?"
        return m, None
    finally:
        os.unlink(csv_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tol", type=float, default=25.0)
    ap.add_argument("--from", dest="lo", type=int, default=None)
    ap.add_argument("--to", dest="hi", type=int, default=None)
    ap.add_argument("--param", action="append", default=[],
                    metavar="NAME=v1,v2", help="e.g. ACCEL_WEIGHT=0.002,0.01")
    a = ap.parse_args()

    grid = {}
    for p in a.param:
        if "=" not in p:
            sys.exit(f"bad --param {p!r}, want NAME=v1,v2")
        k, vs = p.split("=", 1)
        grid[k.strip().upper()] = [v.strip() for v in vs.split(",")]
    if not grid:
        grid = DEFAULT_GRID
        print("no --param given; using default grid over ACCEL_WEIGHT x FRAME_REWARD\n")

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"{len(combos)} trials over {', '.join(keys)}\n")

    rows = []
    for i, combo in enumerate(combos, 1):
        ov = dict(zip(keys, combo))
        desc = " ".join(f"{k}={v}" for k, v in ov.items())
        print(f"[{i}/{len(combos)}] {desc} …", end=" ", flush=True)
        m, err = run_trial(a.video, a.labels, ov, a.tol, a.lo, a.hi)
        if err:
            print("FAILED:", err)
            continue
        print(f"F1={m['f1']}  recall={m['recall']}  prec={m['prec']}  ghost={m['ghost']}")
        rows.append((float(m["f1"]), ov, m))

    if not rows:
        sys.exit("no successful trials")
    rows.sort(key=lambda r: -r[0])
    print("\n=== ranked by F1 ===")
    print(f"{'F1':>6} {'recall':>7} {'prec':>6} {'ghost':>6} {'err':>6} {'live':>5}  params")
    for f1, ov, m in rows:
        print(f"{m['f1']:>6} {m['recall']:>7} {m['prec']:>6} {m['ghost']:>6} "
              f"{m['med_err']:>6} {m['live']:>5}  " + " ".join(f"{k}={v}" for k, v in ov.items()))
    best = rows[0]
    print("\nbest:", " ".join(f"{k}={v}" for k, v in best[1].items()), f"-> F1 {best[2]['f1']}")
    print("Apply by editing the matching defaults in ViterbiConfig (ViterbiTracker.swift).")


if __name__ == "__main__":
    main()
