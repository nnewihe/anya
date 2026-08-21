"""Tune the energy-bar point-end weights against labelled clips.

Reads the cached 30 Hz end telemetry, the cached serve detections and the
walking pose cache, so a sweep never reruns perception.

THE OBJECTIVE IS EVAL_POINT_END, NOT A PROXY FOR IT.  Configurations are scored
through `build_segments` and `eval_point_end.score_ends`, so recall, precision,
truncations and mid-rally false fires mean exactly what that file prints:

    loss = (1 - recall) + 4.0 * trunc_rate + 1.0 * midfp_rate
                        + 0.10 * mean(min(|point_err|, 10)) / 10

One truncation must buy four recovered ends — the standing acceptance gate,
stated as an objective.  The timing term is only a tiebreak: the first three are
piecewise constant with large flat regions, and without it the argmin is decided
by dictionary order across dozens of exact ties.

AND IT FITS AGAINST DETECTED STARTS BY DEFAULT.  `tune_deadtime_confidence`
starts at each labelled rally to isolate end quality from serve-start error, and
that premise does not survive contact with this policy: the first fit here
scored 82% recall and zero truncations from labelled starts, then 57% and three
truncations in production.  The energy bar RESETS at every start and integrates
forward from it, so a late or missing start corrupts the end in a way a
globally-computed trace onset simply cannot.  Serve-start error is part of what
this policy has to survive, so it is part of what the tuner scores.  Pass
--labelled-starts for the old, more flattering view.

The corpus is limited to clips with `snippet_anya_end_telemetry_b30_i1920.jsonl`
cached — the trace needs that stream and `trace_intervals` refuses anything
slower.  As of writing that is Data/21,22,23,24,43: five clips, still below the
~8-clip floor this project learned the hard way, so --leave-one-out is the
number that matters and the shipped defaults are hand-set from the structural
argument with the sweep as confirmation, not the other way round.

Example:
  python -m pipeline.tune_energy --leave-one-out /Volumes/Anya/Data/{21,22,23,24,43}
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .anya_end_telemetry import (end_dets_path_for, end_pose_path_for,
                                 end_telemetry_path_for)
from .anya_far_serve import detect_far_serves
from .anya_far_telemetry import far_telemetry_path_for
from .anya_near_serve import events_path_for
from .anya_near_telemetry import near_telemetry_path_for
from .eval_point_end import score_ends, _agg
from .parse_ground_truth import load_rallies
from .rally_reel import ball_trace, energy
from .rally_reel.config import ReelConfig
from .rally_reel.points import (PointStart, build_segments, enforce_service_runs,
                                merge_serve_starts)
from .rally_reel.reel import _ball_stream, _walk_intervals, _walk_model_path


# The swept knobs, in the order the staged sweep visits them.  Every weight is
# per second (see ReelConfig), so none of these grids moves with energy_step_s.
GRIDS = {
    "energy_ball_weight":         (0.10, 0.15, 0.20, 0.28, 0.40, 0.55),
    # Widened twice.  The first pass pinned the boost to 4.0, but at ball 0.28
    # and walk 1.0 the boosted drain is already 1.4/s against a 0.60 cap, so the
    # CAP was binding and the boost above it is unidentifiable — which is why
    # energy_max_drop_per_s is swept alongside it rather than left fixed.
    "energy_walk_boost":          (0.0, 0.75, 1.5, 2.5, 4.0, 6.0, 9.0),
    "energy_motion_weight":       (0.0, 0.15, 0.30, 0.50, 0.80),   # 0 ablates recovery
    "energy_reversal_weight":     (0.0, 0.08, 0.15, 0.30, 0.50),
    "energy_near_missing_weight": (0.0, 0.05, 0.10, 0.20, 0.35),
    "energy_hold_s":              (1.0, 1.5, 2.0, 3.0, 4.0),
    "energy_stamp_s":             (1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
    "energy_confirm_s":           (0.0, 0.25, 0.5, 1.0),
    "energy_max_drop_per_s":      (0.40, 0.60, 0.90, 1.40),
}
# These grids are where widening STOPS PAYING, not where the edges disappear.
# One further round (boost to 14, near to 0.80, hold to 6.0, stamp to 5.0)
# improved the training argmin by 0.0006 and cost held-out recall 79% -> 77%
# with the first truncation of the whole exercise — the extra room was spent on
# five clips' worth of noise.  energy_near_missing_weight is therefore left
# sitting on its edge deliberately; read that as the term wanting more than this
# corpus can justify giving it, not as a grid that needs another pass.

# Staged coordinate sweep: pairs that interact are swept together, since the
# whole risk of coordinate descent here is a ridge between a gain and a drain.
STAGES = (("energy_ball_weight", "energy_walk_boost"),
          ("energy_motion_weight", "energy_reversal_weight"),
          ("energy_near_missing_weight", "energy_hold_s"),
          ("energy_stamp_s", "energy_confirm_s"),
          ("energy_walk_boost", "energy_max_drop_per_s"))

TRUNC_PENALTY = 4.0
MIDFP_PENALTY = 1.0   # an end inside a labelled rally cuts a point in half


# ── loading ─────────────────────────────────────────────────────────────

def _detected_starts(directory: Path, cfg: ReelConfig) -> List[PointStart]:
    """The production start timeline, rebuilt from cached detector output.

    Stages 3 and 4 of the reel, and nothing else: `detect_far_serves` parses a
    cached far telemetry JSONL and the near events are already scored on disk,
    so this costs a couple of seconds and cannot drift from what `build_reel`
    would produce — it calls the same two functions on the same two files.
    """
    video = str(directory / "snippet.mp4")
    far_serves = detect_far_serves(far_telemetry_path_for(video)) if cfg.use_far else []
    near_events = []
    if cfg.use_near:
        path = Path(events_path_for(near_telemetry_path_for(video)))
        if path.exists():
            payload = json.loads(path.read_text())
            near_events = payload["events"] if isinstance(payload, dict) else payload
    starts = merge_serve_starts(far_serves, near_events, cfg)
    if cfg.enforce_service_runs:
        starts = enforce_service_runs(starts, cfg)
        if cfg.drop_side_conflicts:
            starts = [p for p in starts if not p.side_conflict]
    return starts


def _load_clip(directory: Path, cfg: ReelConfig, labelled_starts: bool) -> Dict:
    video = str(directory / "snippet.mp4")
    telemetry = end_telemetry_path_for(video, cfg.trace_ball_fps, cfg.trace_ball_imgsz)
    if not Path(telemetry).exists():
        raise SystemExit(
            f"[TUNE] {directory.name}: missing {Path(telemetry).name}.  The energy "
            f"policy reads the same {cfg.trace_ball_fps:.0f} Hz stream the trace "
            f"policy does; re-extract with\n"
            f"  python -m pipeline.rally_reel {video} --dry-run --end-policy trace")

    meta, records, filter_balls = _ball_stream(telemetry)
    intervals, _ = ball_trace.trace_intervals(
        meta, records, filter_balls,
        str(directory / "snippet_court_cache.json"), cfg)
    _, walk_result = _walk_intervals(
        video, device="cpu", dets_npz=end_dets_path_for(video),
        pose_npz=end_pose_path_for(video), model_path=_walk_model_path(cfg),
        return_result=True)

    ev = energy.build_evidence(meta, records, walk_result, intervals)
    fps = float(meta["fps"])
    n_frames = int(meta["total_frames"])
    duration = n_frames / fps
    rallies = load_rallies(str(directory))

    if labelled_starts:
        starts = [PointStart(t=r["start_s"], side=r.get("serve", ""), confidence="")
                  for r in rallies]
    else:
        starts = _detected_starts(directory, cfg)

    # Bin once.  The bins depend only on the starts, energy_step_s and the two
    # reference scales — none of which the sweep moves — so every configuration
    # in the grid reads these same arrays.
    bins = []
    for i, ps in enumerate(starts):
        stop = starts[i + 1].t if i + 1 < len(starts) else duration
        bins.append(energy.bin_point(ev, ps.t, max(stop, ps.t + cfg.energy_step_s), cfg))
    return {"name": directory.name, "starts": starts, "bins": bins,
            "rallies": rallies, "fps": fps, "n_frames": n_frames,
            "duration": duration}


# ── scoring ─────────────────────────────────────────────────────────────

def _score(clip: Dict, cfg: ReelConfig) -> Dict:
    """One configuration through the production path, scored by the eval.

    `build_segments` rather than a local end rule: point_min_s, point_max_s and
    next_serve_guard_s decide a real share of these ends, and a tuner that
    skipped them would be fitting weights against a policy that does not ship.
    """
    details = []
    for bins, ps in zip(clip["bins"], clip["starts"]):
        res = energy.integrate(bins, cfg, stop_at_end=True)
        found = energy.point_end(bins, res, ps.t, cfg)
        if found is not None:
            details.append(found)
    segments = build_segments(clip["starts"], energy.energy_onsets(details),
                              clip["duration"], cfg,
                              min_point_s=cfg.energy_point_min_s)
    ends = [{"t": s.end_t, "method": s.end_method, "serve_t": s.serve_t,
             "side": s.side} for s in segments]
    return score_ends(clip["rallies"], ends, clip["fps"], clip["n_frames"])


def _metrics(rows: Sequence[Dict]) -> Dict:
    """Pool per-clip scores and attach the loss.  `_agg` is eval_point_end's."""
    if not rows:
        return {"loss": float("inf"), "n_gt": 0}
    a = _agg(list(rows))
    n_points = len(a["point_err"]) + a["uncovered"]
    pe = a["point_err"]
    timing = float(np.mean(np.minimum(np.abs(pe), 10.0)) / 10.0) if pe.size else 1.0
    trunc_rate = a["truncations"] / n_points if n_points else 0.0
    midfp_rate = a["mid_rally_fp"] / a["n_gt"] if a["n_gt"] else 0.0
    return {
        "loss": ((1.0 - a["recall"]) + TRUNC_PENALTY * trunc_rate
                 + MIDFP_PENALTY * midfp_rate + 0.10 * timing),
        "recall": a["recall"], "precision": a["precision"],
        "truncations": a["truncations"], "mid_rally_fp": a["mid_rally_fp"],
        "uncovered": a["uncovered"],
        "point_median_s": float(np.median(pe)) if pe.size else float("nan"),
        "event_median_s": float(np.median(a["err"])) if a["err"].size else float("nan"),
        "n_gt": a["n_gt"], "n_det": a["n_det"], "n_points": n_points,
    }


def _scores_for(clips: Sequence[Dict], over: Dict, base: ReelConfig,
                table: Dict) -> List[Dict]:
    """Per-clip score dicts for one configuration, memoised into `table`.

    Scoring dominates and every fold reuses the same per-clip numbers, so score
    each (config, clip) pair once and let folds be aggregations over the table.
    """
    key = _key(over)
    if key not in table:
        cfg = replace(base, **over)
        table[key] = [_score(clip, cfg) for clip in clips]
    return table[key]


def _key(over: Dict) -> Tuple:
    return tuple(over[k] for k in sorted(GRIDS))


def _unkey(key: Tuple) -> Dict:
    return dict(zip(sorted(GRIDS), key))


def _pick(table: Dict, idxs: Sequence[int]) -> Tuple:
    return min(table, key=lambda k: _metrics([table[k][i] for i in idxs])["loss"])


def _line(name: str, m: Dict) -> str:
    return (f"{name:>12}: loss={m['loss']:.3f} recall={m['recall']:.0%} "
            f"prec={m['precision']:.0%} trunc={m['truncations']}/{m['n_points']} "
            f"midFP={m['mid_rally_fp']} ptmed={m['point_median_s']:+.2f}")


def _leave_one_out(table: Dict, names: Sequence[str]) -> Dict:
    folds, picks = [], []
    for i, name in enumerate(names):
        key = _pick(table, [j for j in range(len(names)) if j != i])
        m = _metrics([table[key][i]])
        picks.append(key)
        folds.append((name, key, m))
        print("[TUNE]   hold out " + _line(name, m))
    pooled = _metrics([table[k][i] for i, (_, k, _) in enumerate(folds)])
    distinct = len(set(picks))
    print("[TUNE]   " + _line(f"pooled/{distinct} cfg", pooled))
    if distinct > max(2, len(names) // 4):
        print("[TUNE]   WARNING: folds disagree — the corpus is too small to pin "
              "these weights.  Read the sweep as direction, not as a default.")
    return {"pooled": pooled, "distinct_configs": distinct,
            "folds": [{"clip": n, **_unkey(k), "metrics": m} for n, k, m in folds]}


# ── sweeps ───────────────────────────────────────────────────────────────

def _defaults(base: ReelConfig) -> Dict:
    return {k: getattr(base, k) for k in GRIDS}


FULL_GRID_LIMIT = 20000


def _full_grid(clips, base: ReelConfig, table: Dict) -> None:
    """The whole cross product, as a check on the coordinate sweep.

    Disagreement between the two is itself the finding: it means the loss
    surface has interactions the staged descent walked past.  With the grids as
    widened they multiply out past two million configurations, though, so this
    refuses rather than appearing to hang — narrow GRIDS before asking for it.
    """
    configs = [{}]
    for field, grid in sorted(GRIDS.items()):
        configs = [dict(c, **{field: v}) for c in configs for v in grid]
    if len(configs) > FULL_GRID_LIMIT:
        raise SystemExit(
            f"[TUNE] the full grid is {len(configs)} configurations, over the "
            f"{FULL_GRID_LIMIT} limit.  Narrow GRIDS in this file, or drop "
            f"--full-grid and read the staged sweep.")
    print(f"[TUNE] full grid: {len(configs)} configuration(s)")
    for i, over in enumerate(configs):
        _scores_for(clips, over, base, table)
        if (i + 1) % 500 == 0:
            print(f"[TUNE]   scored {i + 1}/{len(configs)}", flush=True)


def _staged_search(clips, base: ReelConfig, table: Dict, passes: int = 2) -> Dict:
    """Greedy coordinate descent over interacting pairs, `passes` times round.

    Pairs rather than single knobs because the one real risk here is a ridge
    between a gain and a drain: raising energy_motion_weight and
    energy_ball_weight together is nearly a no-op, so a one-knob-at-a-time walk
    stalls on exactly the axes that matter.

    The candidate set this visits is chosen against the WHOLE corpus, so the
    leave-one-out pass that follows holds out a clip from the fit but not from
    the enumeration.  That is mildly optimistic and worth saying out loud; the
    alternative — re-running the descent inside every fold — refits the search
    path per fold and costs a factor of len(clips) for a number that is harder,
    not easier, to interpret.
    """
    def loss(over):
        return _metrics(_scores_for(clips, over, base, table))["loss"]

    current = _defaults(base)
    best_loss = loss(current)
    for p in range(passes):
        for stage in STAGES:
            for combo in _product([GRIDS[f] for f in stage]):
                over = dict(current, **dict(zip(stage, combo)))
                value = loss(over)
                if value < best_loss - 1e-12:
                    current, best_loss = over, value
        print(f"[TUNE]   pass {p + 1}: loss={best_loss:.4f} "
              f"after {len(table)} scored config(s)", flush=True)
    return current


def _product(grids):
    out = [()]
    for g in grids:
        out = [c + (v,) for c in out for v in g]
    return out


def _dump(payload: Dict, path: Optional[str]) -> None:
    if path:
        Path(path).write_text(json.dumps(payload, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--leave-one-out", action="store_true",
                    help="also refit per fold and report held-out loss")
    ap.add_argument("--full-grid", action="store_true",
                    help="score the whole cross product instead of the staged "
                         "coordinate sweep; disagreement between the two is "
                         "itself the finding")
    ap.add_argument("--labelled-starts", action="store_true",
                    help="start each bar at a LABELLED rally instead of the "
                         "detected serve.  Isolates end quality from serve-start "
                         "error, and consequently overstates this policy — see "
                         "the module docstring")
    ap.add_argument("--report-only", action="store_true",
                    help="score the current defaults, no sweep")
    ap.add_argument("--step", type=float, default=None,
                    help="override energy_step_s (re-bins; use for the "
                         "resolution ablation)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    base = ReelConfig()
    base.end_policy = "energy"
    if args.step is not None:
        base.energy_step_s = args.step

    for clip in args.clips:
        print(f"[TUNE] loading {clip}")
    # Cache reads and the walking forward pass both release the GIL.
    with ThreadPoolExecutor(max_workers=min(4, len(args.clips))) as pool:
        clips = list(pool.map(
            lambda c: _load_clip(c, base, args.labelled_starts), args.clips))
    names = [c["name"] for c in clips]
    n_ends = sum(len(c["rallies"]) for c in clips)
    n_starts = sum(len(c["starts"]) for c in clips)
    kind = "labelled" if args.labelled_starts else "detected"
    print(f"[TUNE] {len(clips)} clip(s), {n_ends} labelled rall(ies), "
          f"{n_starts} {kind} start(s), step {base.energy_step_s}s")

    payload: Dict = {"clips": names, "n_rallies": n_ends, "starts": kind,
                     "energy_step_s": base.energy_step_s}

    if args.report_only:
        rows = [_score(c, base) for c in clips]
        for name, row in zip(names, rows):
            print("[TUNE]   " + _line(name, _metrics([row])))
        print("[TUNE]   " + _line("POOLED", _metrics(rows)))
        payload.update(defaults=_defaults(base), metrics=_metrics(rows),
                       per_clip={n: _metrics([r]) for n, r in zip(names, rows)})
        _dump(payload, args.json_out)
        return 0

    table: Dict = {}
    if args.full_grid:
        _full_grid(clips, base, table)
    else:
        _staged_search(clips, base, table)
    print(f"[TUNE] {len(table)} configuration(s) scored")

    if args.leave_one_out:
        if len(clips) < 3:
            print("[TUNE] --leave-one-out needs at least 3 clips")
            return 2
        print("[TUNE] leave-one-clip-out")
        payload["held_out"] = _leave_one_out(table, names)

    best = _pick(table, range(len(clips)))
    over = _unkey(best)
    result = _metrics(table[best])
    for i, n in enumerate(names):
        print("[TUNE]   " + _line(n, _metrics([table[best][i]])))
    print("[TUNE]   " + _line("POOLED", result))
    payload.update(best=over, metrics=result,
                   per_clip={n: _metrics([table[best][i]]) for i, n in enumerate(names)})
    edges = [k for k, v in over.items() if v in (GRIDS[k][0], GRIDS[k][-1])]
    if edges:
        payload["at_grid_edge"] = edges
        print(f"[TUNE] WARNING: {', '.join(edges)} landed on a sweep edge — widen "
              f"the grid, the optimum may lie outside it.")
    if not args.leave_one_out:
        print("[TUNE] NOTE: this is a training-set argmin.  Re-run with "
              "--leave-one-out before adopting these weights.")
    print("[TUNE] best configuration")
    print(json.dumps(over, indent=2))
    _dump(payload, args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
