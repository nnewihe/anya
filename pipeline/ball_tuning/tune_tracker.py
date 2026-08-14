"""
tune_tracker.py
===============
Optimize ``ParabolicBallTracker`` parameters against hand labels.

Loads the detection cache (``cache_detections.py``) and the labels
(``label_ball.py``), then runs coordinate descent over the tracker's parameters
plus the confidence floor.  Because the cache holds every detection down to a low
conf floor, changing ``ball_conf`` is just a filter on the cache — YOLO never
re-runs, so each evaluation is a fast pure-numpy ``resolve()`` call.

Objective (lower is better), averaged over labeled frames only:
  * ball-labeled frame, trace present  -> min(pixel error, MISS_CAP)
  * ball-labeled frame, trace absent   -> MISS_CAP           (false negative)
  * no-ball frame,      trace present  -> FP_CAP             (false positive)
  * no-ball frame,      trace absent   -> 0                  (correct)

All caps are in full-resolution pixels.  The tuner prints the baseline (current
defaults) vs best score with matched/FN/FP breakdowns, and writes
``<stem>_best_params.json`` for the caller to apply to ball_detector.py.

    python pipeline/ball_tuning/tune_tracker.py /Volumes/Anya/Data/69/snippet_1min.mp4
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from ball_detector import ParabolicBallTracker, make_image_row_perspective  # noqa: E402
from utilities import Config  # noqa: E402

MISS_CAP = 100.0   # px penalty for a missed ball / capped position error
FP_CAP = 100.0     # px penalty for a spurious ball in dead time


def _stem_path(video_path: str, suffix: str) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}_{suffix}")


def build_homography(court_points):
    """Image->court-feet homography, mirroring AnyaBallDetector._build_homography."""
    if not court_points or len(court_points) != 4:
        return None
    pts = sorted(court_points, key=lambda p: p[1])
    far_pair, near_pair = pts[:2], pts[2:]
    TL, TR = sorted(far_pair, key=lambda p: p[0])
    BL, BR = sorted(near_pair, key=lambda p: p[0])
    src = np.array([BL, BR, TR, TL], dtype=np.float32)
    dst = np.array([
        [0, 0], [Config.COURT_WIDTH_FT, 0],
        [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT],
        [0, Config.COURT_LENGTH_FT],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


# Parameters that get scaled internally by px_scale (calibrated at 960-wide).
# We tune the *base* (960-px) values, exactly what the constructor takes.
TUNABLE = {
    "conf_floor":      [0.02, 0.05, 0.08, 0.12, 0.18, 0.25],
    "min_arc":         [3, 4, 5],
    "resid_tol":       [8.0, 12.0, 16.0, 20.0],
    "grow_resid_tol":  [12.0, 18.0, 24.0],
    "seed_gate_px":    [60.0, 90.0, 120.0],
    "two_pt_gate_px":  [40.0, 60.0, 80.0],
    "gate_px":         [25.0, 40.0, 55.0],
    "max_gap":         [4, 6, 9],
    "move_thresh_px":  [15.0, 22.0, 30.0, 40.0],
    "max_arc_s":       [1.5, 2.5, 3.5],
    "zone_reject_frac": [0.35, 0.5, 0.65],
    "court_gate":      [True, False],
}

DEFAULTS = {
    "conf_floor": 0.05, "min_arc": 3, "resid_tol": 12.0, "grow_resid_tol": 18.0,
    "seed_gate_px": 90.0, "two_pt_gate_px": 60.0, "gate_px": 40.0, "max_gap": 6,
    "move_thresh_px": 30.0, "max_arc_s": 2.5, "zone_reject_frac": 0.5,
    "court_gate": True,
}


class Evaluator:
    def __init__(self, cache, labels):
        self.width = cache["width"]
        self.height = cache["height"]
        self.fps = cache["fps"]
        self.px_scale = self.width / float(Config.ANALYSIS_WIDTH)
        self.persp = make_image_row_perspective(self.height)
        self.H = build_homography(cache["court_points"])
        self.zones = [tuple(z) for z in cache["exclusion_zones"]]
        self.raw_dets = cache["dets"]  # list per frame of [x,y,conf]
        # labels: {frame:int -> (x,y) | None}
        self.labels = labels

    def _filtered(self, conf_floor):
        return [[(d[0], d[1], d[2]) for d in fr if d[2] >= conf_floor]
                for fr in self.raw_dets]

    def resolve(self, p):
        dets = self._filtered(p["conf_floor"])
        tracker = ParabolicBallTracker(
            fps=self.fps, px_scale=self.px_scale,
            perspective_scale=self.persp,
            homography=(self.H if p["court_gate"] else None),
            exclusion_zones=self.zones,
            min_arc=p["min_arc"], resid_tol=p["resid_tol"],
            grow_resid_tol=p["grow_resid_tol"], seed_gate_px=p["seed_gate_px"],
            two_pt_gate_px=p["two_pt_gate_px"], gate_px=p["gate_px"],
            max_gap=p["max_gap"], move_thresh_px=p["move_thresh_px"],
            max_arc_s=p["max_arc_s"], zone_reject_frac=p["zone_reject_frac"],
        )
        positions, _states, _segs = tracker.resolve(dets)
        return positions

    def score(self, p):
        positions = self.resolve(p)
        n = len(positions)
        err = 0.0
        matched = fn = fp = tn = 0
        dist_sum = 0.0
        for f, lab in self.labels.items():
            if f >= n:
                continue
            pos = positions[f]
            if lab is not None:  # ball should be here
                if pos is not None:
                    d = min(MISS_CAP, ((pos[0] - lab[0]) ** 2 +
                                       (pos[1] - lab[1]) ** 2) ** 0.5)
                    err += d
                    dist_sum += d
                    matched += 1
                else:
                    err += MISS_CAP
                    fn += 1
            else:               # dead time: trace should be empty
                if pos is not None:
                    err += FP_CAP
                    fp += 1
                else:
                    tn += 1
        n_lab = matched + fn + fp + tn
        mean = err / max(1, n_lab)
        stats = dict(mean=mean, matched=matched, fn=fn, fp=fp, tn=tn,
                     mean_hit_px=(dist_sum / matched if matched else float("nan")))
        return mean, stats


def coordinate_descent(ev, start, order=None, passes=3):
    best = dict(start)
    best_score, best_stats = ev.score(best)
    print(f"[start] score={best_score:.2f}  "
          f"matched={best_stats['matched']} fn={best_stats['fn']} "
          f"fp={best_stats['fp']} hit_px={best_stats['mean_hit_px']:.1f}")
    order = order or list(TUNABLE.keys())
    for it in range(passes):
        improved = False
        for name in order:
            cur = best[name]
            for val in TUNABLE[name]:
                if val == best[name]:
                    continue
                trial = dict(best)
                trial[name] = val
                sc, _ = ev.score(trial)
                if sc < best_score - 1e-9:
                    best_score, best[name] = sc, val
            if best[name] != cur:
                improved = True
                print(f"  pass {it+1}: {name}: {cur} -> {best[name]}  "
                      f"score={best_score:.2f}")
        if not improved:
            print(f"[converged after pass {it+1}]")
            break
    _, best_stats = ev.score(best)
    return best, best_score, best_stats


def main(video_path):
    cache_p = _stem_path(video_path, "dets.json")
    labels_p = _stem_path(video_path, "labels.json")
    if not os.path.isfile(cache_p):
        sys.exit(f"missing detection cache: {cache_p} (run cache_detections.py)")
    if not os.path.isfile(labels_p):
        sys.exit(f"missing labels: {labels_p} (run label_ball.py)")

    with open(cache_p) as f:
        cache = json.load(f)
    with open(labels_p) as f:
        raw = json.load(f)
    labels = {int(k): (tuple(v) if v is not None else None)
              for k, v in raw.items()}
    n_ball = sum(1 for v in labels.values() if v is not None)
    n_none = sum(1 for v in labels.values() if v is None)
    print(f"[INFO] {len(labels)} labels ({n_ball} ball, {n_none} no-ball); "
          f"{sum(len(d) for d in cache['dets'])} cached detections")

    ev = Evaluator(cache, labels)

    print("\n=== BASELINE (current ball_detector.py defaults) ===")
    base_score, base_stats = ev.score(DEFAULTS)
    print(f"score={base_score:.2f}  matched={base_stats['matched']} "
          f"fn={base_stats['fn']} fp={base_stats['fp']} "
          f"hit_px={base_stats['mean_hit_px']:.1f}")

    print("\n=== COORDINATE DESCENT ===")
    best, best_score, best_stats = coordinate_descent(ev, DEFAULTS)

    print("\n=== RESULT ===")
    print(f"baseline score : {base_score:.2f}")
    print(f"tuned score    : {best_score:.2f}  "
          f"({100 * (base_score - best_score) / max(base_score, 1e-6):+.1f}%)")
    print(f"tuned stats    : matched={best_stats['matched']} fn={best_stats['fn']} "
          f"fp={best_stats['fp']} tn={best_stats['tn']} "
          f"hit_px={best_stats['mean_hit_px']:.1f}")
    print("\nchanged params:")
    for k in TUNABLE:
        if best[k] != DEFAULTS[k]:
            print(f"  {k}: {DEFAULTS[k]} -> {best[k]}")
    if all(best[k] == DEFAULTS[k] for k in TUNABLE):
        print("  (none — defaults already optimal on these labels)")

    out = _stem_path(video_path, "best_params.json")
    with open(out, "w") as f:
        json.dump({"params": best, "baseline_score": base_score,
                   "tuned_score": best_score, "stats": best_stats,
                   "defaults": DEFAULTS}, f, indent=2)
    print(f"\n[INFO] wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tune_tracker.py <video.mp4>")
        sys.exit(1)
    main(sys.argv[1])
