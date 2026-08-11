"""
eval_point_end.py
=================
Score point ENDS against ground truth, per clip and pooled.

Until now the only record of point-end accuracy anywhere was a 15-point prose
table in `rally_reel/config.py` for Data/23.  The labels to do better have been
on disk the whole time: `<clip>/ground_truth.json` carries `end` for every
rally, 216 of them across the 12 trusted clips.

Three views of the same run, because they fail differently:

  events    Did an end fire near each labelled end?  recall / precision /
            timing.  This is the view that catches a detector that never
            fires and the one that fires constantly.

  points    For each labelled rally, what did the segment covering it actually
            do to its end?  Median error, and TRUNCATIONS — ends more than
            `trunc_s` EARLY, which cut live tennis out of the reel.  A late end
            wastes footage; an early one loses the point, so they are not the
            same error and are not averaged together.

  mid-rally False fires per live-minute: ends that landed inside a labelled
            rally and matched nothing.  Window/frame accuracy hides these
            (dead and live are both long, easy states) which is why
            `evaluate_events.py` scores the transition instead.

Scoring rules worth knowing before reading the numbers:

  * Matching is greedy by absolute error, one detection to one label, inside
    ±`tol_s`.  A detection can only be credited once.
  * `end_t` is scored, NOT `end`: the segment's `end` includes post-roll, which
    is a framing choice, not a claim about when the point stopped.
  * Clips 35, 37, 63 and 68 are excluded by `parse_ground_truth.gt_path` as
    incompletely labelled or model-derived.  Do not route around it — clip 63
    has 1.4% of its frames marked live.
  * Per-clip numbers are always printed and the pooled row is a summary, never
    the target.  A corpus total rewards a detector that stays quiet on clips it
    cannot solve at all; see the clip-25 trap in `anya_far_telemetry`.

Usage:
    # score the default run of every trusted clip
    python pipeline/eval_point_end.py

    # A/B two arms saved under different suffixes
    python pipeline/eval_point_end.py --clips 21 22 \
        --arm baseline:_rally_segments_baseline.json \
        --arm fast:_rally_segments_fast.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ground_truth import (DATA_ROOT, _fps_for, _n_frames, discover,
                                gt_path, load_rallies, transitions)

SEGMENTS_SUFFIX = "_rally_segments.json"
TOL_S = 2.0        # a point end this far from the label still counts as found
TRUNC_S = 2.0      # early by more than this = a truncation, the harmful case


# ── loading a run ────────────────────────────────────────────────────────

def clip_video(clip_dir: str) -> Optional[str]:
    for name in ("snippet.mp4", "match.mp4"):
        p = os.path.join(clip_dir, name)
        if os.path.isfile(p):
            return p
    return None


def segments_path(clip_dir: str, suffix: str = SEGMENTS_SUFFIX) -> Optional[str]:
    video = clip_video(clip_dir)
    if video is None:
        return None
    stem = os.path.splitext(os.path.basename(video))[0]
    p = os.path.join(clip_dir, f"{stem}{suffix}")
    return p if os.path.isfile(p) else None


def load_run(path: str) -> Tuple[List[Dict], Dict]:
    """Point ends from a rally_reel segments file, or a raw onset list.

    Returns (ends, meta) where each end is {t, method, serve_t, side}.  A raw
    list of onsets — [{t, method}] or bare numbers — is accepted too, so an
    onset source can be scored before it is wired into segment assembly.
    """
    payload = json.load(open(path))
    if isinstance(payload, dict) and "segments" in payload:
        ends = [{"t": float(s["end_t"]), "method": s.get("end_method"),
                 "serve_t": float(s["serve_t"]), "side": s.get("side")}
                for s in payload["segments"]]
        return ends, {"config": payload.get("config", {}),
                      "n_serve_starts": payload.get("n_serve_starts")}
    raw = payload.get("onsets", payload) if isinstance(payload, dict) else payload
    ends = [{"t": float(x if not isinstance(x, dict) else x["t"]),
             "method": (x.get("method") if isinstance(x, dict) else None),
             "serve_t": None, "side": None} for x in raw]
    return ends, {}


# ── matching ─────────────────────────────────────────────────────────────

def match_events(det_t: Sequence[float], gt_t: Sequence[float],
                 tol_s: float = TOL_S) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """One-to-one greedy match by absolute error. Returns (pairs, miss, extra)."""
    pairs: List[Tuple[int, int]] = []
    cand = sorted(((abs(d - g), i, j) for i, d in enumerate(det_t)
                   for j, g in enumerate(gt_t) if abs(d - g) <= tol_s))
    used_d, used_g = set(), set()
    for _, i, j in cand:
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        pairs.append((i, j))
    miss = [j for j in range(len(gt_t)) if j not in used_g]
    extra = [i for i in range(len(det_t)) if i not in used_d]
    return sorted(pairs, key=lambda p: p[1]), miss, extra


def _pct(a: int, b: int) -> str:
    return f"{a}/{b}" + (f" ({100.0 * a / b:.0f}%)" if b else "")


# ── per-clip scoring ─────────────────────────────────────────────────────

def score_clip(clip_dir: str, run_path: str, tol_s: float = TOL_S,
               trunc_s: float = TRUNC_S) -> Optional[Dict]:
    rallies = load_rallies(clip_dir)
    if not rallies:
        return None
    ends, meta = load_run(run_path)
    fps = _fps_for(clip_dir)
    n_frames = _n_frames(clip_dir) or (rallies[-1]["end"] + 1)

    # Only rallies that are followed by real dead time are live->dead
    # transitions; back-to-back rallies would otherwise demand an end that no
    # correct detector should emit.
    tr = transitions(rallies)
    gt_t = [e["frame"] / fps for e in tr["point_end"]]
    det_t = [e["t"] for e in ends]

    pairs, miss, extra = match_events(det_t, gt_t, tol_s)
    err = np.array([det_t[i] - gt_t[j] for i, j in pairs]) if pairs else np.array([])

    # points view: what happened to the end of each labelled rally, judged
    # through the segment that actually covers it.
    per_point, truncations, uncovered = [], 0, 0
    for r in rallies:
        best, best_ov = None, 0.0
        for e in ends:
            if e["serve_t"] is None:
                continue
            ov = min(e["t"], r["end_s"]) - max(e["serve_t"], r["start_s"])
            if ov > best_ov:
                best, best_ov = e, ov
        if best is None:
            uncovered += 1
            continue
        d = best["t"] - r["end_s"]
        per_point.append(d)
        if d < -trunc_s:
            truncations += 1

    # mid-rally false fires: an unmatched end inside a labelled rally is the
    # expensive kind of false positive — it cuts a point in half.
    live = np.zeros(n_frames, bool)
    for r in rallies:
        live[max(0, r["start"]):min(n_frames, r["end"] + 1)] = True
    mid = sum(1 for i in extra
              if 0 <= int(det_t[i] * fps) < n_frames and live[int(det_t[i] * fps)])
    live_min = live.sum() / fps / 60.0

    by_method: Dict[str, int] = {}
    for e in ends:
        by_method[e["method"] or "?"] = by_method.get(e["method"] or "?", 0) + 1

    return {
        "clip": os.path.basename(clip_dir.rstrip("/")),
        "n_gt": len(gt_t), "n_det": len(det_t), "n_match": len(pairs),
        "recall": len(pairs) / len(gt_t) if gt_t else 0.0,
        "precision": len(pairs) / len(det_t) if det_t else 0.0,
        "err": err, "point_err": np.array(per_point),
        "truncations": truncations, "uncovered": uncovered,
        "mid_rally_fp": mid, "live_min": live_min,
        "by_method": by_method, "config": meta.get("config", {}),
    }


def _agg(rows: List[Dict]) -> Dict:
    err = np.concatenate([r["err"] for r in rows]) if rows else np.array([])
    pe = np.concatenate([r["point_err"] for r in rows]) if rows else np.array([])
    n_gt = sum(r["n_gt"] for r in rows)
    n_det = sum(r["n_det"] for r in rows)
    n_ok = sum(r["n_match"] for r in rows)
    return {"n_gt": n_gt, "n_det": n_det, "n_match": n_ok,
            "recall": n_ok / n_gt if n_gt else 0.0,
            "precision": n_ok / n_det if n_det else 0.0,
            "err": err, "point_err": pe,
            "truncations": sum(r["truncations"] for r in rows),
            "uncovered": sum(r["uncovered"] for r in rows),
            "mid_rally_fp": sum(r["mid_rally_fp"] for r in rows),
            "live_min": sum(r["live_min"] for r in rows)}


def _row(name: str, r: Dict) -> str:
    e, pe = r["err"], r["point_err"]
    med = f"{np.median(e):+.2f}" if len(e) else "  -  "
    p90 = f"{np.percentile(np.abs(e), 90):.2f}" if len(e) else "  -  "
    pmed = f"{np.median(pe):+.2f}" if len(pe) else "  -  "
    ffr = r["mid_rally_fp"] / r["live_min"] if r["live_min"] else 0.0
    return (f"{name:>9} {r['n_gt']:>4} {r['n_det']:>5} "
            f"{r['recall']*100:>6.0f}% {r['precision']*100:>6.0f}% "
            f"{med:>7} {p90:>6} {pmed:>7} {r['truncations']:>6} "
            f"{r['mid_rally_fp']:>5} {ffr:>6.2f}")


HEADER = (f"{'clip':>9} {'gt':>4} {'det':>5} {'recall':>7} {'prec':>7} "
          f"{'med e':>7} {'p90':>6} {'pt med':>7} {'trunc':>6} {'midFP':>5} "
          f"{'/min':>6}")


def run(clip_dirs: List[str], arms: List[Tuple[str, str]], tol_s: float,
        trunc_s: float) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for name, suffix in arms:
        rows = []
        print(f"\n=== arm: {name}   ({suffix}) ===")
        print(HEADER)
        for d in clip_dirs:
            p = segments_path(d, suffix)
            if p is None:
                print(f"{os.path.basename(d.rstrip('/')):>9}   — no run on disk")
                continue
            r = score_clip(d, p, tol_s, trunc_s)
            if r is None:
                continue
            rows.append(r)
            print(_row(r["clip"], r))
        if rows:
            print("-" * len(HEADER))
            print(_row("POOLED", _agg(rows)))
            methods: Dict[str, int] = {}
            for r in rows:
                for k, v in r["by_method"].items():
                    methods[k] = methods.get(k, 0) + v
            print(f"[{name}] end methods: " +
                  ", ".join(f"{k}={v}" for k, v in sorted(methods.items())))
        out[name] = rows
    return out


def compare(results: Dict[str, List[Dict]]) -> None:
    """Per-clip deltas between exactly two arms — the comparison that counts.

    A pooled delta can hide a fast path that gained six ends on one clip and
    lost four on another; those are different bugs and only the per-clip view
    separates them.
    """
    names = list(results)
    if len(names) != 2:
        return
    a, b = names
    ra = {r["clip"]: r for r in results[a]}
    rb = {r["clip"]: r for r in results[b]}
    both = [c for c in ra if c in rb]
    if not both:
        return
    print(f"\n=== per-clip delta: {b} - {a} ===")
    print(f"{'clip':>9} {'recall':>16} {'prec':>16} {'trunc':>12} {'midFP':>12}")
    for c in sorted(both, key=lambda x: int(x) if x.isdigit() else 0):
        x, y = ra[c], rb[c]
        print(f"{c:>9} "
              f"{_pct(x['n_match'], x['n_gt']):>7} -> {_pct(y['n_match'], y['n_gt']):>7} "
              f"{_pct(x['n_match'], x['n_det']):>7} -> {_pct(y['n_match'], y['n_det']):>7} "
              f"{x['truncations']:>5} -> {y['truncations']:>4} "
              f"{x['mid_rally_fp']:>5} -> {y['mid_rally_fp']:>4}")
    pa, pb = _agg([ra[c] for c in both]), _agg([rb[c] for c in both])
    print("-" * 70)
    print(f"{'POOLED':>9} "
          f"{_pct(pa['n_match'], pa['n_gt']):>7} -> {_pct(pb['n_match'], pb['n_gt']):>7} "
          f"{_pct(pa['n_match'], pa['n_det']):>7} -> {_pct(pb['n_match'], pb['n_det']):>7} "
          f"{pa['truncations']:>5} -> {pb['truncations']:>4} "
          f"{pa['mid_rally_fp']:>5} -> {pb['mid_rally_fp']:>4}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score rally_reel point ends against ground_truth.json")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--clips", nargs="*", default=None,
                    help="Clip folder names (default: every trusted clip)")
    ap.add_argument("--arm", action="append", default=None, metavar="NAME:SUFFIX",
                    help="Run to score, as name:segments-suffix. Repeatable; "
                         "two arms print a per-clip delta table.")
    ap.add_argument("--tol", type=float, default=TOL_S,
                    help=f"Match window in seconds (default {TOL_S})")
    ap.add_argument("--trunc", type=float, default=TRUNC_S,
                    help=f"Early by more than this = truncation (default {TRUNC_S})")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    dirs = ([os.path.join(args.data_root, c) for c in args.clips]
            if args.clips else discover(args.data_root))
    dirs = [d for d in dirs if gt_path(d)]
    if not dirs:
        raise SystemExit("no clips with trusted ground truth")

    arms = [tuple(a.split(":", 1)) for a in (args.arm or [f"run:{SEGMENTS_SUFFIX}"])]
    results = run(dirs, arms, args.tol, args.trunc)
    compare(results)

    if args.json_out:
        dump = {n: [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in r.items()} for r in rows]
                for n, rows in results.items()}
        json.dump(dump, open(args.json_out, "w"), indent=1)
        print(f"\n[eval] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
