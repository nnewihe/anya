"""Tune monotonic dead-time-confidence weights against labelled clips.

Uses the cached end telemetry and walking pose cache, so experiments do not
rerun detection.  Labels are only used here: production scores from detected
point starts, while this evaluator starts at each labelled rally to isolate the
quality of the dead-time score from serve-start errors.

--leave-one-out is the number that matters.  A plain sweep reports its own
training-set argmin, which on this problem is badly optimistic below roughly
eight clips: at three clips every fold picks a different config, and at four the
walking weight fits to 1.00 against the 0.34 that ten clips settle on.  The
held-out pass refits per fold and scores each clip with weights it never saw,
so a config that only fits one clip cannot look good.

Example:
  python -m pipeline.tune_deadtime_confidence --leave-one-out \
      /Volumes/Anya/Data/2{1,2,3,4,5,6} /Volumes/Anya/Data/{36,38,40,43}
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

from .anya_end_telemetry import end_dets_path_for, end_pose_path_for
from .rally_reel.config import ReelConfig
from .rally_reel.deadtime_confidence import (score_deadtime_from_evidence,
                                             telemetry_evidence_by_second,
                                             walking_evidence_by_second)
from .rally_reel.reel import _ball_stream, _walk_intervals, _walk_model_path


def _load_clip(directory: Path, cfg: ReelConfig):
    video = directory / "snippet.mp4"
    telemetry = directory / "snippet_anya_end_telemetry.jsonl"
    labels = json.loads((directory / "ground_truth.json").read_text())["rallies"]
    meta, records, is_ball = _ball_stream(str(telemetry))
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
        "telemetry": telemetry_evidence_by_second(records, is_ball),
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


def _errors(rows, threshold) -> List[float]:
    """Signed seconds between the predicted end and the labelled one."""
    out = []
    for samples, true_end in rows:
        predicted = next((r["timestamp_s"] for r in samples
                          if r["confidence"] >= threshold), samples[-1]["timestamp_s"])
        out.append(predicted - true_end)
    return out


def _metrics(errors: Sequence[float]) -> Dict:
    ordered = sorted(errors)
    return {
        # A premature cut loses tennis, whereas a late cut merely retains footage.
        "loss": sum(abs(e) + 3.0 * max(0.0, -e) for e in errors) / len(errors),
        "median_error_s": ordered[len(ordered) // 2],
        "mae_s": sum(abs(e) for e in errors) / len(errors),
        "early_cuts": sum(1 for e in errors if e < -2.0),
        "n": len(errors),
    }


def _evaluate(rows, threshold) -> Dict:
    return _metrics(_errors(rows, threshold))


# Swept ranges, widened until the optimum stopped landing on an edge.  Walking
# in particular pinned to whichever end of a narrow grid it was given before
# the corpus was large enough to constrain it.
BASE_GRID = (0.02, 0.04, 0.06, 0.08, 0.12)
WALK_GRID = (0.10, 0.20, 0.27, 0.34, 0.48, 0.62)
BALL_GRID = (0.12, 0.18, 0.24, 0.30, 0.36)


def _grid(base: ReelConfig, threshold: float):
    for elapsed in BASE_GRID:
        for walking in WALK_GRID:
            for ball in BALL_GRID:
                yield replace(base, deadtime_base_per_s=elapsed,
                              deadtime_walking_weight=walking,
                              deadtime_ball_trace_weight=ball,
                              deadtime_score_threshold=threshold)


def _weights(cfg: ReelConfig) -> Dict[str, float]:
    return {
        "deadtime_base_per_s": cfg.deadtime_base_per_s,
        "deadtime_walking_weight": cfg.deadtime_walking_weight,
        "deadtime_ball_trace_weight": cfg.deadtime_ball_trace_weight,
    }


def _error_table(loaded, threshold, base: ReelConfig) -> Dict:
    """Per-(config, clip) error lists for the whole grid.

    Scoring dominates the sweep and every fold reuses the same per-clip numbers,
    so score each pair once here and let folds be aggregations over this table.
    Recomputing per fold instead costs a factor of len(clips) and puts a ten-clip
    held-out run out of interactive reach.
    """
    table = {}
    for i, cfg in enumerate(_grid(base, threshold)):
        table[tuple(_weights(cfg).values())] = [
            _errors(_score_clips([clip], cfg), threshold) for clip in loaded]
        if (i + 1) % 25 == 0:
            print(f"[TUNE]   scored {i + 1} configs", flush=True)
    return table


def _pick(table: Dict, idxs: Sequence[int]):
    """Training-set argmin over the grid, restricted to `idxs`."""
    def loss(key):
        return _metrics([e for i in idxs for e in table[key][i]])["loss"]
    return min(table, key=loss)


def _leave_one_out(table: Dict, names: Sequence[str]) -> Dict:
    """Refit per fold and score each clip with weights fitted without it."""
    folds, picks = [], []
    for i, name in enumerate(names):
        key = _pick(table, [j for j in range(len(names)) if j != i])
        result = _metrics(table[key][i])
        picks.append(key)
        folds.append((name, key, result))
        print(f"[TUNE]   hold out {name}: base={key[0]:.2f} walk={key[1]:.2f} "
              f"ball={key[2]:.2f} -> loss={result['loss']:.3f} "
              f"mae={result['mae_s']:.2f} early={result['early_cuts']}/{result['n']}")
    total = sum(r["n"] for _, _, r in folds)
    pooled = sum(r["loss"] * r["n"] for _, _, r in folds) / total
    distinct = len(set(picks))
    print(f"[TUNE]   {distinct} distinct config(s) across {len(names)} folds; "
          f"pooled held-out loss={pooled:.3f}, "
          f"early={sum(r['early_cuts'] for _, _, r in folds)}/{total}")
    if distinct > max(2, len(names) // 4):
        print("[TUNE]   WARNING: folds disagree — the corpus is too small to pin "
              "these weights.  Add clips before trusting them.")
    return {
        "pooled_loss": pooled,
        "distinct_configs": distinct,
        "folds": [{"clip": n, "deadtime_base_per_s": k[0],
                   "deadtime_walking_weight": k[1],
                   "deadtime_ball_trace_weight": k[2], "metrics": r}
                  for n, k, r in folds],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="+", type=Path,
                    help="clip folders containing snippet.mp4, end telemetry, and ground_truth.json")
    ap.add_argument("--threshold", type=float,
                    default=ReelConfig().deadtime_score_threshold)
    ap.add_argument("--leave-one-out", action="store_true",
                    help="also refit per fold and report held-out loss")
    args = ap.parse_args(argv)

    base = ReelConfig()
    for clip in args.clips:
        print(f"[TUNE] loading {clip}")
    # Cache reads and the walking forward pass both release the GIL, so threads
    # do overlap here despite the work being nominally CPU-bound.
    with ThreadPoolExecutor(max_workers=min(4, len(args.clips))) as pool:
        loaded_clips = list(pool.map(lambda clip: _load_clip(clip, base), args.clips))
    names = [clip.name for clip in args.clips]

    payload: Dict = {"threshold": args.threshold, "n_clips": len(loaded_clips)}
    table = _error_table(loaded_clips, args.threshold, base)

    if args.leave_one_out:
        if len(loaded_clips) < 3:
            print("[TUNE] --leave-one-out needs at least 3 clips")
            return 2
        print("[TUNE] leave-one-clip-out")
        payload["held_out"] = _leave_one_out(table, names)

    best = _pick(table, range(len(loaded_clips)))
    result = _metrics([e for per_clip in table[best] for e in per_clip])
    payload.update(deadtime_base_per_s=best[0], deadtime_walking_weight=best[1],
                   deadtime_ball_trace_weight=best[2],
                   deadtime_score_threshold=args.threshold, metrics=result)
    edges = [k for k, v, grid in (("deadtime_base_per_s", best[0], BASE_GRID),
                                  ("deadtime_walking_weight", best[1], WALK_GRID),
                                  ("deadtime_ball_trace_weight", best[2], BALL_GRID))
             if v in (grid[0], grid[-1])]
    if edges:
        payload["at_grid_edge"] = edges
        print(f"[TUNE] WARNING: {', '.join(edges)} landed on a sweep edge — widen "
              f"the grid, the optimum may lie outside it.")
    if not args.leave_one_out:
        print("[TUNE] NOTE: this is a training-set argmin.  Re-run with "
              "--leave-one-out before adopting these weights.")
    print("[TUNE] best configuration")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
