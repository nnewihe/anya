"""
trace_eval.py
=============
Trace-quality benchmark for the dead-time cutter: how faithfully does the
replayed ball trace cover the labeled rallies?  Every trace-accuracy change
(detection enrichment, tracker tuning, fitting) should move these numbers,
measured before/after on the ground-truth folders.

Metrics per labeled rally:
  coverage     fraction of [start, end] covered by genuine alive intervals
  fragments    number of alive intervals overlapping the rally (1 = ideal)
  max_gap_s    longest uncovered stretch strictly inside the rally
  onset_lag_s  first alive-interval onset within [start-1, end] minus start
               (how long after the labeled serve the trace comes alive)

Aggregates across rallies: mean coverage, worst rally, total fragments,
median onset lag, and stray alive time OUTSIDE any rally (±5 s margin) —
the precision counterweight when detection recall is turned up.

GT formats (auto-detected):
  ground_truth.json          {"rallies": [{"start": F, "end": F, ...}]}    frames
  derived_ground_truth.json  {"rallies": [{"start_s": S, "end_s": S}]}     seconds
  tags.json                  {"points":  [{"start": S, "end": S, ...}]}    seconds

Frame-based GT is converted with the VIDEO fps from telemetry meta (stride
does not change GT frame timestamps).

Run:
    python -m pipeline.trace_eval <telemetry.jsonl> <gt.json>
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional, Tuple

from .point_segmenter import (SegmenterConfig, MatchTelemetry, load_telemetry,
                              suppress_static_candidates, replay_ball_tracker,
                              alive_intervals, segment_match)


def load_gt_rallies(path: str, video_fps: float) -> List[dict]:
    """Normalize any of the three GT formats to [{start_s, end_s, serve?}].

    Frame vs seconds is ambiguous from values alone in short videos, so
    decide per format: ground_truth.json uses frames, everything else uses
    seconds."""
    with open(path, "r") as fh:
        data = json.load(fh)
    if "points" in data:                                     # tags.json
        return [{"start_s": float(r["start"]), "end_s": float(r["end"]),
                 "serve": r.get("serve")} for r in data["points"]]
    rallies = data.get("rallies", [])
    if rallies and "start_s" in rallies[0]:                  # derived (seconds)
        return [{"start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
                 "serve": r.get("serve")} for r in rallies]
    return [{"start_s": float(r["start"]) / video_fps,      # frames
             "end_s": float(r["end"]) / video_fps,
             "serve": r.get("serve")} for r in rallies]


def evaluate_trace(match: MatchTelemetry, rallies: List[dict],
                   cfg: Optional[SegmenterConfig] = None,
                   replay=None, verbose: bool = True) -> dict:
    """Score the replayed ball trace against labeled rallies.

    Pass a precomputed `replay` (from replay_ball_tracker over the whole
    match) to A/B different replay settings without re-loading telemetry.
    """
    cfg = cfg or SegmenterConfig()
    if match.meta:
        size = match.meta.get("analysis_size")
        if size:
            cfg.frame_height_px = float(size[1])
    suppress_static_candidates(match, cfg)

    if replay is None:
        replay = replay_ball_tracker(match, 0.0, match.duration + 1.0, cfg)
    intervals = alive_intervals(replay, cfg.alive_merge_gap_s)

    rows = []
    for r in rallies:
        gs, ge = r["start_s"], r["end_s"]
        span = max(ge - gs, 1e-9)
        overlaps = [(max(s, gs), min(e, ge)) for s, e in intervals
                    if e > gs and s < ge]
        covered = sum(e - s for s, e in overlaps)
        # longest uncovered stretch inside the rally
        max_gap, cursor = 0.0, gs
        for s, e in overlaps:
            max_gap = max(max_gap, s - cursor)
            cursor = max(cursor, e)
        max_gap = max(max_gap, ge - cursor)
        onset = next((s for s, e in intervals if gs - 1.0 <= s <= ge), None)
        rows.append({
            "start_s": gs, "end_s": ge, "serve": r.get("serve"),
            "coverage": covered / span,
            "fragments": len(overlaps),
            "max_gap_s": max_gap,
            "onset_lag_s": None if onset is None else onset - gs,
        })

    # stray alive time outside any rally (±5 s margin)
    margin = 5.0
    stray = 0.0
    for s, e in intervals:
        t = s
        while t < e:
            inside = any(r["start_s"] - margin <= t <= r["end_s"] + margin
                         for r in rallies)
            step = 0.5
            if not inside:
                stray += step
            t += step

    covs = [x["coverage"] for x in rows]
    lags = sorted(x["onset_lag_s"] for x in rows if x["onset_lag_s"] is not None)
    summary = {
        "n_rallies": len(rows),
        "mean_coverage": sum(covs) / len(covs) if covs else 0.0,
        "min_coverage": min(covs) if covs else 0.0,
        "rallies_over_80pct": sum(c >= 0.8 for c in covs),
        "total_fragments": sum(x["fragments"] for x in rows),
        "no_trace_rallies": sum(x["fragments"] == 0 for x in rows),
        "median_onset_lag_s": lags[len(lags) // 2] if lags else None,
        "stray_alive_s": stray,
        "rows": rows,
    }

    if verbose:
        print(f"{'rally':>18} {'serve':>5} {'cover':>6} {'frags':>5} "
              f"{'maxGap':>7} {'onsetLag':>8}")
        for x in rows:
            lag = "  --" if x["onset_lag_s"] is None else f"{x['onset_lag_s']:+7.2f}"
            print(f"{x['start_s']:8.1f}–{x['end_s']:7.1f} {str(x['serve'] or '?'):>5} "
                  f"{100*x['coverage']:5.0f}% {x['fragments']:5d} "
                  f"{x['max_gap_s']:6.1f}s {lag:>8}")
        s = summary
        lag_txt = ("--" if s["median_onset_lag_s"] is None
                   else f"{s['median_onset_lag_s']:+.2f}s")
        print(f"\nmean coverage {100*s['mean_coverage']:.0f}%  "
              f"(min {100*s['min_coverage']:.0f}%, "
              f"{s['rallies_over_80pct']}/{s['n_rallies']} rallies >=80%)  "
              f"fragments {s['total_fragments']}  "
              f"no-trace rallies {s['no_trace_rallies']}  "
              f"median onset lag {lag_txt}  "
              f"stray alive {s['stray_alive_s']:.0f}s")
    return summary


def _merge_intervals(ivs: List[Tuple[float, float]], gap: float
                     ) -> List[Tuple[float, float]]:
    """Union of intervals, folding gaps <= `gap` — mirrors the ffmpeg export's
    merge_gap_sec so coverage reflects the actual kept video."""
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def evaluate_segments(match: MatchTelemetry, rallies: List[dict],
                      cfg: Optional[SegmenterConfig] = None,
                      merge_gap_s: float = 1.0,
                      verbose: bool = True) -> dict:
    """Product-level metric under the deadtime-cutter framing: does the KEPT
    VIDEO span the labeled rallies?  Runs the full segmenter and measures GT
    rally coverage by the final [start, end] segments (merged as the export
    would), plus the cost side — kept seconds that fall in true dead time.
    """
    cfg = cfg or SegmenterConfig()
    segments = segment_match(match, cfg, verbose=False)
    kept = _merge_intervals([(s.start, s.end) for s in segments], merge_gap_s)

    rows = []
    for r in rallies:
        gs, ge = r["start_s"], r["end_s"]
        span = max(ge - gs, 1e-9)
        overlaps = [(max(s, gs), min(e, ge)) for s, e in kept
                    if e > gs and s < ge]
        covered = sum(e - s for s, e in overlaps)
        # a point "started" if a kept segment covers the labeled serve moment
        started = any(s <= gs + 1.0 and e >= gs for s, e in kept)
        rows.append({
            "start_s": gs, "end_s": ge, "serve": r.get("serve"),
            "coverage": covered / span,
            "started": started,
            "full": covered / span >= 0.95,
        })

    # kept seconds outside any labeled rally (±margin) = dead time NOT removed
    margin = 2.0
    gt_union = _merge_intervals([(r["start_s"] - margin, r["end_s"] + margin)
                                 for r in rallies], 0.0)
    kept_total = sum(e - s for s, e in kept)
    stray = 0.0
    for s, e in kept:
        rem = e - s
        for gs, ge in gt_union:
            rem -= max(0.0, min(e, ge) - max(s, gs))
        stray += max(0.0, rem)

    covs = [x["coverage"] for x in rows]
    summary = {
        "n_rallies": len(rows),
        "rallies_started": sum(x["started"] for x in rows),
        "rallies_full": sum(x["full"] for x in rows),
        "mean_coverage": sum(covs) / len(covs) if covs else 0.0,
        "min_coverage": min(covs) if covs else 0.0,
        "n_segments": len(segments),
        "kept_total_s": kept_total,
        "stray_kept_s": stray,
        "rows": rows,
    }

    if verbose:
        print(f"{'rally':>18} {'serve':>5} {'cover':>6} {'start':>6} {'full':>5}")
        for x in rows:
            print(f"{x['start_s']:8.1f}–{x['end_s']:7.1f} {str(x['serve'] or '?'):>5} "
                  f"{100*x['coverage']:5.0f}% {'yes' if x['started'] else ' NO':>6} "
                  f"{'yes' if x['full'] else '  -':>5}")
        s = summary
        print(f"\nrallies started {s['rallies_started']}/{s['n_rallies']}  "
              f"full-cover {s['rallies_full']}/{s['n_rallies']}  "
              f"mean coverage {100*s['mean_coverage']:.0f}% (min {100*s['min_coverage']:.0f}%)  "
              f"segments {s['n_segments']}  "
              f"kept {s['kept_total_s']:.0f}s (stray {s['stray_kept_s']:.0f}s in dead time)")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace-quality benchmark against labeled rallies")
    parser.add_argument("telemetry", help="match telemetry JSONL")
    parser.add_argument("gt", help="ground truth JSON (frames or seconds)")
    parser.add_argument("--mode", choices=["trace", "segments", "both"],
                        default="both",
                        help="trace = raw ball-trace coverage; segments = "
                             "kept-video coverage (the product metric)")
    args = parser.parse_args()

    match = load_telemetry(args.telemetry)
    video_fps = float(match.meta.get("fps", 30.0))
    rallies = load_gt_rallies(args.gt, video_fps)
    print(f"[EVAL] {len(rallies)} labeled rallies, video fps {video_fps:.2f}\n")
    if args.mode in ("trace", "both"):
        print("── RAW TRACE COVERAGE ──")
        evaluate_trace(match, rallies)
    if args.mode in ("segments", "both"):
        print("\n── KEPT-VIDEO (SEGMENT) COVERAGE ──")
        evaluate_segments(match, rallies)
