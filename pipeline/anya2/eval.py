"""
eval.py
=======
One event-matching harness for all three anya2 detectors.

The detectors are independent; the corpus is not.  Every one of them is scored
against the same `<clip>/ground_truth.json`, in the same way, so their numbers
are comparable and none of them gets to invent a favourable metric.

    --mode near_serve   GT = `start` of rallies with serve == "near"
    --mode far_serve    GT = `start` of rallies with serve == "far"
    --mode point_end    GT = `end` of every rally

Matching reuses `eval_point_end.match_events`: greedy one-to-one by absolute
error inside +/- tol, each detection credited at most once.  Sharing the matcher
matters more than it looks -- a detector that fires twice per serve should be
punished for it, and a matcher that credits both would hide exactly the failure
mode a refractory window exists to prevent.

Two reporting rules, both learned the hard way and both enforced here rather
than left to the reader:

  PER-CLIP IS THE RESULT, POOLED IS A SUMMARY.  A corpus total rewards a
  detector that stays quiet on the clips it cannot solve at all: it loses a
  little recall and pays no precision, and the pooled row goes up.  That is the
  clip-25 trap `anya_far_telemetry` documents.  The per-clip table is always
  printed and is always the thing to read.

  EARLY AND LATE ARE NOT THE SAME ERROR, so point_end never averages them.  An
  end more than `--trunc` seconds EARLY deletes live tennis from the reel; a
  late one only wastes footage.  Truncations are counted separately.

Serve modes carry the mirror of that asymmetry, reported as `bias`: a serve
detected LATE has already cut off the start of the point.

Usage:
    python -m pipeline.anya2.eval --mode near_serve --arm ported:_anya2_near_serve.json
    python -m pipeline.anya2.eval --mode point_end --clips 21 22 \
        --arm a:_run_a.json --arm b:_run_b.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from parse_ground_truth import DATA_ROOT, discover, load_rallies  # noqa: E402
from eval_point_end import match_events  # noqa: E402

from pipeline.anya2.contract import FAR_SERVE, NEAR_SERVE, POINT_END  # noqa: E402

TOL_S = 2.0
TRUNC_S = 2.0


def clip_video(clip_dir):
    for name in ("snippet.mp4", "match.mp4"):
        p = os.path.join(clip_dir, name)
        if os.path.isfile(p):
            return p
    return None


def labelled_span(clip_dir, margin_s=TOL_S):
    """The window over which this clip's labels actually say anything.

    Ground truth labels LIVE intervals only, and dead time is inferred as their
    complement -- but that inference is only valid BETWEEN the first and last
    rally.  Outside them, "not labelled live" and "no tennis happened" are
    indistinguishable.

    Clip 38 is the case that forces this: it is labelled from 26 s to 206 s and
    then stops, leaving 214 s -- more than half the clip -- unlabelled.  All
    three of the near detector's apparent false positives there fall in that
    tail, and scoring them as errors would be scoring the detector against the
    absence of a label rather than against a label.

    Detections outside the span are neither credited nor penalised; they are
    counted and reported separately, because a large number of them is itself
    worth seeing.
    """
    r = load_rallies(clip_dir)
    if not r:
        return None
    return r[0]["start_s"] - margin_s, r[-1]["end_s"] + margin_s


def gt_times(clip_dir, mode):
    """Labelled event times in seconds for this mode."""
    rallies = load_rallies(clip_dir)
    if mode == NEAR_SERVE:
        return [r["start_s"] for r in rallies if r["serve"] == "near"]
    if mode == FAR_SERVE:
        return [r["start_s"] for r in rallies if r["serve"] == "far"]
    if mode == POINT_END:
        return [r["end_s"] for r in rallies]
    raise ValueError(mode)


def run_path(clip_dir, suffix):
    v = clip_video(clip_dir)
    if v is None:
        return None
    stem = os.path.splitext(os.path.basename(v))[0]
    p = os.path.join(clip_dir, f"{stem}{suffix}")
    return p if os.path.isfile(p) else None


def load_det(path, mode, min_p=0.0):
    """Detection times from a contract-format events JSON, filtered to `mode`.

    Files that carry other kinds too (a fused run) are filtered rather than
    rejected, so one run file can be scored in all three modes.
    """
    with open(path) as fh:
        rows = json.load(fh).get("events", [])
    return sorted(float(r["t"]) for r in rows
                  if r.get("kind", mode) == mode and float(r.get("p", 1.0)) >= min_p)


def score(det_t, gt_t, mode, tol_s=TOL_S, trunc_s=TRUNC_S):
    pairs, miss, extra = match_events(det_t, gt_t, tol_s)
    err = np.array([det_t[i] - gt_t[j] for i, j in pairs], dtype=float)
    r = {
        "n_gt": len(gt_t), "n_det": len(det_t), "hit": len(pairs),
        "miss": len(miss), "fp": len(extra),
        "recall": len(pairs) / len(gt_t) if gt_t else float("nan"),
        "precision": len(pairs) / len(det_t) if det_t else float("nan"),
        "bias": float(np.median(err)) if err.size else float("nan"),
        "abs": float(np.median(np.abs(err))) if err.size else float("nan"),
    }
    if mode == POINT_END:
        # Counted over MATCHED ends only: an end that missed entirely is a
        # miss, and calling it a truncation as well would double-count it.
        r["trunc"] = int((err < -trunc_s).sum()) if err.size else 0
    return r


def _fmt(name, r, mode):
    def pct(x):
        return "  n/a" if not np.isfinite(x) else f"{100 * x:5.1f}"
    tail = f"  trunc {r['trunc']:3d}" if mode == POINT_END else ""
    if r.get("outside"):
        tail += f"  [{r['outside']} outside labelled span]"
    bias = "   n/a" if not np.isfinite(r["bias"]) else f"{r['bias']:+6.2f}"
    return (f"  {name:<10} gt {r['n_gt']:4d}  det {r['n_det']:4d}  "
            f"hit {r['hit']:4d}  miss {r['miss']:4d}  fp {r['fp']:4d}  "
            f"R {pct(r['recall'])}%  P {pct(r['precision'])}%  "
            f"bias {bias}s{tail}")


def pool(rows):
    """Pooled = summed counts, NOT averaged rates -- an 81-rally clip and a
    6-rally clip do not get equal votes."""
    g = sum(r["n_gt"] for r in rows)
    d = sum(r["n_det"] for r in rows)
    h = sum(r["hit"] for r in rows)
    out = {"n_gt": g, "n_det": d, "hit": h,
           "miss": sum(r["miss"] for r in rows), "fp": sum(r["fp"] for r in rows),
           "recall": h / g if g else float("nan"),
           "precision": h / d if d else float("nan"),
           "outside": sum(r.get("outside", 0) for r in rows),
           "bias": float(np.median([r["bias"] for r in rows
                                    if np.isfinite(r["bias"])] or [np.nan])),
           "abs": float(np.median([r["abs"] for r in rows
                                   if np.isfinite(r["abs"])] or [np.nan]))}
    if "trunc" in rows[0]:
        out["trunc"] = sum(r["trunc"] for r in rows)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--mode", required=True,
                    choices=[NEAR_SERVE, FAR_SERVE, POINT_END])
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME:SUFFIX",
                    help="Run to score, e.g. ported:_anya2_near_serve.json")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--tol", type=float, default=TOL_S)
    ap.add_argument("--trunc", type=float, default=TRUNC_S)
    ap.add_argument("--no-span", action="store_true",
                    help="Score detections outside the labelled span too "
                         "(see labelled_span: usually wrong, occasionally "
                         "what you want when auditing a detector's chatter)")
    ap.add_argument("--min-p", type=float, default=0.0,
                    help="Ignore detections below this confidence")
    a = ap.parse_args(argv)

    dirs = discover(a.data_root)
    if a.clips:
        want = set(a.clips)
        dirs = [d for d in dirs if os.path.basename(d) in want]
        missing = want - {os.path.basename(d) for d in dirs}
        if missing:
            print(f"[eval] skipping untrusted/absent clips: {sorted(missing)}")
    arms = [tuple(s.split(":", 1)) for s in a.arm]

    print(f"\n=== {a.mode}  tol +/-{a.tol}s  "
          f"{len(dirs)} trusted clips ===")
    results = {name: [] for name, _ in arms}
    for d in dirs:
        clip = os.path.basename(d)
        gt = gt_times(d, a.mode)
        # A clip with ZERO labels for this mode is scored, not skipped.  Clips
        # 23 and 40 have no near serves at all, so for --mode near_serve every
        # detection there is a false positive -- and skipping them would hide
        # exactly the clips where a serve detector is most likely to be fooled
        # (a far-serve rally still has a near player standing at the baseline
        # between points, and clip 40 has two of them).  Precision is undefined
        # on such a row, so it prints as n/a and the FP count carries it.
        n_lab = f"{len(gt)} labelled" if gt else "NO labels for this mode"
        print(f"\n{clip}  ({n_lab})")
        for name, suffix in arms:
            p = run_path(d, suffix)
            if p is None:
                print(f"  {name:<10} -- no run file ({suffix})")
                continue
            det = load_det(p, a.mode, a.min_p)
            span = labelled_span(d, a.tol)
            outside = 0
            if span and not a.no_span:
                lo, hi = span
                keep = [t for t in det if lo <= t <= hi]
                outside = len(det) - len(keep)
                det = keep
            r = score(det, gt, a.mode, a.tol, a.trunc)
            r["clip"] = clip
            r["outside"] = outside
            results[name].append(r)
            print(_fmt(name, r, a.mode))

    print("\n" + "=" * 100)
    print("POOLED (a summary, not the result -- read the per-clip table above)")
    for name, _ in arms:
        rows = results[name]
        if rows:
            print(_fmt(name, pool(rows), a.mode) + f"   [{len(rows)} clips]")
    print()
    return results


if __name__ == "__main__":
    main()
