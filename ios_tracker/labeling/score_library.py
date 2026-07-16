#!/usr/bin/env python3
"""Score the tracker across the whole labelled library.

    python3 score_library.py --lib library
    python3 score_library.py --lib library --param ACCEL_WEIGHT=0.01   # one-off
    python3 score_library.py --lib library --sweep ACCEL_WEIGHT=0.002,0.01,0.03

Runs the REAL Swift solver on every labelled clip and reports per-clip and
aggregate scores. Tuning against one video overfits that court; the whole point
of the library is that a weight has to earn its keep on clay and hard, indoor
and outdoor, 30 and 120 fps.

The headline is **macro-F1** — the mean of per-clip F1, not F1 pooled over all
frames. Pooling would let a couple of long, easy clips drown out a court where
the tracker fails completely, which is exactly the failure we want to see.
`worst` matters as much as the mean: a tracker that is great on 9 courts and
blind on the 10th is not robust.
"""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "..", "run_video_check.sh")


def load_trace(path):
    import csv
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            x, y = float(r["x"]), float(r["y"])
            live = r["state"] in ("moving", "coasting") and not math.isnan(x) and x >= 0
            out.append((x, y) if live else None)
    return out


def score_one(labels, trace, tol):
    tp = fn = ghost = quiet = mis = 0
    errs = []
    for fr, lab in labels.items():
        got = trace[fr] if 0 <= fr < len(trace) else None
        if lab is None:
            if got:
                ghost += 1
            else:
                quiet += 1
            continue
        if got is None:
            fn += 1
            continue
        d = math.hypot(got[0] - lab["x"], got[1] - lab["y"])
        if d <= tol:
            tp += 1
            errs.append(d)
        else:
            mis += 1
    ball = tp + fn + mis
    drawn = tp + mis + ghost
    rec = tp / ball if ball else 0.0
    pre = tp / drawn if drawn else 0.0
    f1 = 2 * rec * pre / (rec + pre) if (rec + pre) else 0.0
    empty = ghost + quiet
    return {"f1": f1, "recall": rec, "prec": pre,
            "ghost": ghost / empty if empty else 0.0,
            "err": sorted(errs)[len(errs) // 2] if errs else float("nan"),
            "n": len(labels), "ball": ball}


def run_clip(clip_path, overrides):
    env = dict(os.environ)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        csv_path = tf.name
    env["DUMP_CSV"] = csv_path
    for k, v in overrides.items():
        env["VITERBI_" + k] = str(v)
    try:
        p = subprocess.run([CHECK, clip_path], env=env, capture_output=True, text=True)
        if p.returncode != 0:
            return None
        return load_trace(csv_path)
    finally:
        if os.path.exists(csv_path):
            os.unlink(csv_path)


def evaluate(lib, man, overrides, tol, verbose):
    rows = []
    for c in man["clips"]:
        lp = os.path.join(lib, "labels", c["name"] + ".json")
        if not os.path.exists(lp):
            continue
        labels = {int(k): v for k, v in json.load(open(lp))["labels"].items()}
        if not labels:
            continue
        trace = run_clip(c["clip"], overrides)
        if trace is None:
            if verbose:
                print(f"  {c['name']}: solver failed")
            continue
        s = score_one(labels, trace, tol)
        s["name"] = c["name"]
        s["match"] = c["match"]
        rows.append(s)
        if verbose:
            print(f"  {s['name']:<10} F1={s['f1']:.3f} rec={s['recall']:.3f} "
                  f"prec={s['prec']:.3f} ghost={s['ghost']:.3f} n={s['n']}")
    if not rows:
        return None, []
    macro = sum(r["f1"] for r in rows) / len(rows)
    agg = {
        "macro_f1": macro,
        "worst_f1": min(r["f1"] for r in rows),
        "recall": sum(r["recall"] for r in rows) / len(rows),
        "prec": sum(r["prec"] for r in rows) / len(rows),
        "ghost": sum(r["ghost"] for r in rows) / len(rows),
        "clips": len(rows),
        "matches": len({r["match"] for r in rows}),
    }
    return agg, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", default="library")
    ap.add_argument("--tol", type=float, default=25.0)
    ap.add_argument("--param", action="append", default=[], metavar="K=V")
    ap.add_argument("--sweep", action="append", default=[], metavar="K=v1,v2")
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.lib, "manifest.json")))
    base = {}
    for p in a.param:
        k, v = p.split("=", 1)
        base[k.strip().upper()] = v.strip()

    if not a.sweep:
        print(f"scoring {len(man['clips'])} clips…")
        agg, rows = evaluate(a.lib, man, base, a.tol, True)
        if not agg:
            sys.exit("no labelled clips yet — run label_library.py first")
        print(f"\n  macro-F1 : {agg['macro_f1']:.3f}   <- headline (mean over clips)")
        print(f"  worst F1 : {agg['worst_f1']:.3f}   <- the court we are worst on")
        print(f"  recall   : {agg['recall']:.3f}")
        print(f"  precision: {agg['prec']:.3f}")
        print(f"  ghost    : {agg['ghost']:.3f}")
        print(f"  over {agg['clips']} clips / {agg['matches']} matches")
        worst = sorted(rows, key=lambda r: r["f1"])[:3]
        print("\n  weakest clips: " + ", ".join(f"{r['name']}({r['f1']:.2f})" for r in worst))
        return

    grid = {}
    for s in a.sweep:
        k, vs = s.split("=", 1)
        grid[k.strip().upper()] = [v.strip() for v in vs.split(",")]
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"{len(combos)} trials x {len(man['clips'])} clips over {', '.join(keys)}\n")
    results = []
    for i, combo in enumerate(combos, 1):
        ov = dict(base)
        ov.update(dict(zip(keys, combo)))
        desc = " ".join(f"{k}={v}" for k, v in zip(keys, combo))
        print(f"[{i}/{len(combos)}] {desc} …", end=" ", flush=True)
        agg, _ = evaluate(a.lib, man, ov, a.tol, False)
        if not agg:
            print("no data")
            continue
        print(f"macroF1={agg['macro_f1']:.3f} worst={agg['worst_f1']:.3f} "
              f"rec={agg['recall']:.3f} ghost={agg['ghost']:.3f}")
        results.append((agg["macro_f1"], desc, agg))
    if not results:
        sys.exit("nothing scored")
    results.sort(reverse=True, key=lambda r: r[0])
    print("\n=== ranked by macro-F1 ===")
    print(f"{'macroF1':>8} {'worst':>6} {'recall':>7} {'prec':>6} {'ghost':>6}  params")
    for f1, desc, agg in results:
        print(f"{agg['macro_f1']:>8.3f} {agg['worst_f1']:>6.3f} {agg['recall']:>7.3f} "
              f"{agg['prec']:>6.3f} {agg['ghost']:>6.3f}  {desc}")
    print(f"\nbest: {results[0][1]} -> macro-F1 {results[0][0]:.3f}")
    print("Apply by editing the matching default in ViterbiConfig (ViterbiTracker.swift).")


if __name__ == "__main__":
    main()
