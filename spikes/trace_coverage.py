"""
trace_coverage.py — Phase 0 follow-up: separate the two questions.

Onset-lag told us raw ball-onset can't reliably mark the SERVE (late/sparse on
68, swamped by dead-time ball activity on 23).  But point-END only needs the
LAST ball trace of a rally.  So measure, per GT rally:

  * coverage   = fraction of the rally's frames with a live ball trace
  * end_lag    = (last live frame in rally window) - gt_end   [for the +1s tail]
And globally:
  * dead_live  = fraction of NON-rally (dead-time) frames that are live
                 -> how badly between-point ball activity defeats an onset gate.
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_onset_lag import (load_ball_dets, load_telemetry, homography_from_points,
                             build_homography, load_ground_truth)
from ball_detector import ParabolicBallTracker
from utilities import Config


def run(folder: str):
    data = f"/Volumes/Anya/Data/{folder}"
    dets = [f for f in os.listdir(data) if f.endswith("_ball_dets.jsonl")]
    tel = [f for f in os.listdir(data) if f.endswith("_match_telemetry.jsonl")]
    if dets:
        fps, width, cp, balls = load_ball_dets(os.path.join(data, dets[0]))
        H = homography_from_points(cp); px = width / float(Config.ANALYSIS_WIDTH)
    else:
        t = load_telemetry(os.path.join(data, tel[0]))
        fps, balls, px = t.fps, t.balls, 1.0
        stem = tel[0].replace("_match_telemetry.jsonl", "")
        H = build_homography(os.path.join(data, f"{stem}_court_cache.json"))
    tr = ParabolicBallTracker(fps=fps, px_scale=px, homography=H)
    _, states, _ = tr.resolve(balls)
    live = np.array([s != "none" for s in states])
    n = len(live)
    gt = load_ground_truth(data, fps)

    print(f"\n===== folder {folder}: fps={fps:.2f} frames={n} "
          f"global-live={live.mean()*100:.0f}% rallies={len(gt)} =====")
    covs, endlags = [], []
    in_rally = np.zeros(n, dtype=bool)
    for r in gt:
        a, b = int(r.start_s * fps), min(n - 1, int(r.end_s * fps))
        in_rally[a:b + 1] = True
        seg = live[a:b + 1]
        cov = seg.mean() if len(seg) else 0.0
        covs.append(cov)
        live_idx = np.where(seg)[0]
        endlag = (a + live_idx[-1]) / fps - r.end_s if len(live_idx) else None
        if endlag is not None:
            endlags.append(endlag)
    dead_live = live[~in_rally].mean() if (~in_rally).any() else 0.0
    covs = np.array(covs)
    print(f"  per-rally trace coverage: mean={covs.mean()*100:.0f}% "
          f"median={np.median(covs)*100:.0f}% "
          f">50%:{(covs>0.5).sum()}/{len(covs)}  >20%:{(covs>0.2).sum()}/{len(covs)}")
    if endlags:
        e = np.array(endlags)
        print(f"  last-trace vs gt_end: mean={e.mean():+.2f}s median={np.median(e):+.2f}s "
              f"p10={np.percentile(e,10):+.2f} p90={np.percentile(e,90):+.2f}")
    print(f"  DEAD-TIME live fraction: {dead_live*100:.0f}%  "
          f"(high => between-point ball activity defeats a quiet-gap onset gate)")


if __name__ == "__main__":
    for f in sys.argv[1:] or ["21", "23", "68"]:
        run(f)
