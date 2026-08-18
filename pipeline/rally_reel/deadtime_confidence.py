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

from typing import Dict, List, Sequence

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


def telemetry_evidence_by_second(telemetry_path: str) -> Dict[int, Dict[str, float]]:
    """Filtered ball-trace coverage per second.

    The ball definition is intentionally the same filtered stream used by the
    existing ball-quiet logic.
    """
    from ..anya_far_serve import (FarServeDetector, FarServeDetectorConfig,
                                  calibrate_static_blobs, load_telemetry,
                                  scale_exclusion_zones)

    meta, records = load_telemetry(telemetry_path)
    detector = FarServeDetector(FarServeDetectorConfig())
    detector.set_exclusion_zones(
        scale_exclusion_zones(meta.get("exclusion_zones", []), meta))
    detector.set_static_cells(calibrate_static_blobs(records, detector.cfg))

    bins: Dict[int, Dict[str, List[float]]] = {}
    for record in records:
        t = float(record["t"])
        second = int(t)
        bucket = bins.setdefault(second, {"ball": []})
        if record.get("bn", True):
            bucket["ball"].append(float(bool(detector._filter_balls(record.get("all_balls", [])))))

    out: Dict[int, Dict[str, float]] = {}
    for second, values in bins.items():
        if not values["ball"]:
            continue    # no ball-model look landed here; absent, not empty
        out[second] = {
            "ball_trace": float(np.mean(values["ball"])),
        }
    return out


def score_deadtime(starts: Sequence, duration: float, walk_result: Dict,
                   telemetry_path: str, cfg: ReelConfig) -> List[Dict]:
    """Return monotonic confidence samples for intervals between point starts."""
    walk = walking_evidence_by_second(walk_result)
    evidence = telemetry_evidence_by_second(telemetry_path)
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
