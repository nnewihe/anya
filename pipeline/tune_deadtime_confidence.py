"""Tune monotonic dead-time-confidence weights against labelled clips.

Uses the cached end telemetry and walking pose cache, so experiments do not
rerun detection.  Labels are only used here: production scores from detected
point starts, while this evaluator starts at each labelled rally to isolate the
quality of the dead-time score from serve-start errors.

Example:
  python -m pipeline.tune_deadtime_confidence /Volumes/Anya/Data/21 \
      /Volumes/Anya/Data/22 /Volumes/Anya/Data/23
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from .anya_end_telemetry import end_dets_path_for, end_pose_path_for
from .anya_far_serve import load_telemetry
from .rally_reel.config import ReelConfig
from .rally_reel.deadtime_confidence import (score_deadtime_from_evidence,
                                             telemetry_evidence_by_second,
                                             walking_evidence_by_second)
from .rally_reel.reel import _walk_intervals, _walk_model_path


def _load_clip(directory: Path, cfg: ReelConfig):
    video = directory / "snippet.mp4"
    telemetry = directory / "snippet_anya_end_telemetry.jsonl"
    labels = json.loads((directory / "ground_truth.json").read_text())["rallies"]
    meta, _ = load_telemetry(str(telemetry))
    fps = float(meta["fps"])
    starts = [SimpleNamespace(t=float(r["start"]) / fps) for r in labels]
    _, walk_result = _walk_intervals(
        str(video), device="cpu", dets_npz=end_dets_path_for(str(video)),
        pose_npz=end_pose_path_for(str(video)), model_path=_walk_model_path(cfg),
        return_result=True)
    duration = float(meta["total_frames"]) / fps
    return {
        "starts": starts,
        "duration": duration,
        "walk": walking_evidence_by_second(walk_result),
        "telemetry": telemetry_evidence_by_second(str(telemetry)),
        "ends": [float(label["end"]) / fps for label in labels],
    }


def _score_clips(clips, cfg):
    rows = []
    for clip in clips:
        scores = score_deadtime_from_evidence(
            clip["starts"], clip["duration"], clip["walk"], clip["telemetry"], cfg)
        by_point = {}
        for row in scores:
            by_point.setdefault(row["point_index"], []).append(row)
        rows.extend((by_point[i], end) for i, end in enumerate(clip["ends"]))
    return rows


def _evaluate(rows, threshold):
    errors = []
    for samples, true_end in rows:
        predicted = next((r["timestamp_s"] for r in samples
                          if r["confidence"] >= threshold), samples[-1]["timestamp_s"])
        errors.append(predicted - true_end)
    early = [e for e in errors if e < -2.0]
    # A premature cut loses tennis, whereas a late cut merely retains footage.
    loss = sum(abs(e) + 3.0 * max(0.0, -e) for e in errors) / len(errors)
    ordered = sorted(errors)
    return {
        "loss": loss,
        "median_error_s": ordered[len(ordered) // 2],
        "mae_s": sum(abs(e) for e in errors) / len(errors),
        "early_cuts": len(early),
        "n": len(errors),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="+", type=Path,
                    help="clip folders containing snippet.mp4, end telemetry, and ground_truth.json")
    ap.add_argument("--threshold", type=float, default=0.60)
    args = ap.parse_args(argv)

    base = ReelConfig()
    for clip in args.clips:
        print(f"[TUNE] loading {clip}")
    # Feature construction is independent per clip and CPU-bound.  Parallel
    # cache reads keep a three-clip calibration within one interactive run.
    with ThreadPoolExecutor(max_workers=min(3, len(args.clips))) as pool:
        loaded_clips = list(pool.map(lambda clip: _load_clip(clip, base), args.clips))

    candidates = []
    for elapsed in (0.04, 0.08, 0.12):
        for walking in (0.20, 0.34, 0.48):
            for ball in (0.12, 0.24, 0.36):
                cfg = replace(base, deadtime_base_per_s=elapsed,
                              deadtime_walking_weight=walking,
                              deadtime_ball_trace_weight=ball,
                              deadtime_score_threshold=args.threshold)
                result = _evaluate(_score_clips(loaded_clips, cfg), args.threshold)
                candidates.append((result["loss"], cfg, result))
    _, best, result = min(candidates, key=lambda x: x[0])
    print("[TUNE] best configuration")
    print(json.dumps({
        "deadtime_base_per_s": best.deadtime_base_per_s,
        "deadtime_walking_weight": best.deadtime_walking_weight,
        "deadtime_ball_trace_weight": best.deadtime_ball_trace_weight,
        "deadtime_score_threshold": args.threshold,
        "metrics": result,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
