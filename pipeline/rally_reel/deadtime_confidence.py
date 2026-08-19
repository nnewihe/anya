"""Monotonic per-second dead-time confidence from cached reel telemetry.

The score begins at zero at every detected point start.  Every subsequent
second adds elapsed-time and walking evidence, then subtracts ball-trace
evidence.  Negative increments are clipped to zero, so confidence cannot fall
before the next point start.

A second in which the ball model never looked is not evidence either way, so
it holds the score instead of incrementing it.  Treating an absent look as a
ball-free second would make patchy detection the strongest dead-time signal
we have, which is exactly backwards.
"""

from typing import Callable, Dict, List, Sequence

import numpy as np

from .config import ReelConfig


def walking_evidence_by_second(walk_result: Dict) -> Dict[int, Dict[str, float]]:
    """Mean walking probability per second."""
    fps = float(walk_result["fps"])
    prob = np.asarray(walk_result["prob"], dtype=float)
    out: Dict[int, Dict[str, float]] = {}
    for second in range(int(np.ceil(len(prob) / fps))):
        a, b = int(round(second * fps)), min(int(round((second + 1) * fps)), len(prob))
        if b > a:
            out[second] = {"walking_probability": float(np.mean(prob[a:b]))}
    return out


def telemetry_evidence_by_second(records: Sequence[Dict],
                                 is_ball: Callable[[Dict], bool]) -> Dict[int, Dict[str, float]]:
    """Filtered ball-trace coverage per second.

    Takes the caller's ball stream rather than building one.  `reel._ball_stream`
    already constructs exactly this — confidence floor, rescaled exclusion zones,
    self-calibrated static blobs — for the onset rules, and it exists so that a
    difference between two point-end policies is never a difference in what
    counts as a ball.  Rebuilding a second detector here would reintroduce the
    divergence it was factored out to prevent; the duplicated parse was only
    ~0.1 s, so the drift was always the real cost.
    """
    bins: Dict[int, Dict[str, List[float]]] = {}
    for record in records:
        t = float(record["t"])
        second = int(t)
        bucket = bins.setdefault(second, {"ball": []})
        if record.get("bn", True):
            bucket["ball"].append(float(bool(is_ball(record))))

    out: Dict[int, Dict[str, float]] = {}
    for second, values in bins.items():
        if not values["ball"]:
            continue    # no ball-model look landed here; absent, not empty
        out[second] = {
            "ball_trace": float(np.mean(values["ball"])),
        }
    return out


def score_deadtime(starts: Sequence, duration: float, walk_result: Dict,
                   records: Sequence[Dict], is_ball: Callable[[Dict], bool],
                   cfg: ReelConfig) -> List[Dict]:
    """Return monotonic confidence samples for intervals between point starts."""
    walk = walking_evidence_by_second(walk_result)
    evidence = telemetry_evidence_by_second(records, is_ball)
    return score_deadtime_from_evidence(starts, duration, walk, evidence, cfg)


def score_deadtime_from_evidence(starts: Sequence, duration: float,
                                 walk: Dict[int, Dict[str, float]],
                                 evidence: Dict[int, Dict[str, float]],
                                 cfg: ReelConfig) -> List[Dict]:
    """Score precomputed evidence; used by the tuner to avoid reprocessing."""
    rows: List[Dict] = []
    for i, start in enumerate(starts):
        start_t = float(start.t)
        stop_t = float(starts[i + 1].t) if i + 1 < len(starts) else duration
        confidence = 0.0
        elapsed_second = 0
        while start_t + elapsed_second < stop_t:
            t = start_t + elapsed_second
            absolute_second = int(t)
            e = evidence.get(absolute_second)
            w = walk.get(absolute_second, {})
            walking = w.get("walking_probability", 0.0)
            ball_seen = e is not None
            ball_trace = e["ball_trace"] if ball_seen else 0.0
            if elapsed_second and ball_seen:
                raw_increment = (cfg.deadtime_base_per_s
                                 + cfg.deadtime_walking_weight * walking
                                 - cfg.deadtime_ball_trace_weight * ball_trace)
                increment = min(cfg.deadtime_max_increment_per_s, max(0.0, raw_increment))
                confidence = min(1.0, confidence + increment)
            else:
                increment = 0.0
            rows.append({
                "point_index": i,
                "point_start_s": round(start_t, 3),
                "elapsed_s": elapsed_second,
                "timestamp_s": round(t, 3),
                "confidence": round(confidence, 4),
                "increment": round(increment, 4),
                "walking_probability": round(walking, 4),
                "ball_trace_coverage": round(ball_trace, 4) if ball_seen else None,
            })
            elapsed_second += 1
    return rows


def deadtime_onsets(rows: Sequence[Dict], cfg: ReelConfig) -> List[tuple]:
    """First threshold crossing per point, as (t, "confidence") dead onsets.

    Emitting onsets rather than ends keeps `find_point_end` in the loop, so the
    confidence policy inherits point_min_s, point_max_s and next_serve_guard_s
    unchanged instead of quietly reimplementing them.  A point whose score never
    reaches the threshold contributes nothing and falls through to those guards,
    which is the honest reading: the evidence never became convincing.
    """
    onsets: List[tuple] = []
    seen = set()
    for row in rows:
        i = row["point_index"]
        if i in seen or row["confidence"] < cfg.deadtime_score_threshold:
            continue
        seen.add(i)
        onsets.append((float(row["timestamp_s"]), "confidence"))
    return onsets
