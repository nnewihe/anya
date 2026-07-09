"""
trace_enrich.py
===============
Tracker-guided native-resolution ball re-detection ("detect by tracking").

The stage-1 whole-court ball pass runs on a 960x540 downscale where the ball
is 1-3 px on the far side — traces fragment and die exactly where speed
measurement needs them.  This pass exploits the offline advantage: replay
the IMM tracker over the recorded detections, and wherever the tracker has a
prediction but no detection, decode THAT native-res frame, crop a window
around the prediction, and re-run the ball model at full detail with a low
confidence floor.  Accepted detections are written back to the telemetry as
a new optional channel `rballs`, which the stage-2 replay merges exactly
like the native far-crop channel (fballs).

Four target classes per alive trace interval:
  GAP       coasting frames inside a tracked span (no detection that frame)
  BRIDGE    frames between two intervals separated by <= bridge_max_gap_s,
            along a linear interpolation between the endpoints — the
            highest-value class: the ball was demonstrably in flight on
            both sides of the gap, so the position prior is strong
  BACKFILL  frames BEFORE the interval onset, along a backward linear
            extrapolation — attacks the ~2 s onset lag (tracker needs 3
            hits to confirm, so the trace starts late; the ball was visible
            earlier)
  EXTEND    frames after the interval dies, along a forward extrapolation

Because added detections improve the track, the pass iterates (default 2).
The telemetry file is rewritten atomically with all original channels
preserved; `rballs` is replaced wholesale each run (idempotent).

Run:
    python -m pipeline.trace_enrich /path/to/match.mp4 [--iterations 2]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .ball_tracker import make_image_row_perspective
from .match_telemetry import telemetry_path_for
from .point_segmenter import (SegmenterConfig, load_telemetry,
                              suppress_static_candidates, alive_intervals,
                              _replay_core)
from .utilities import load_cached_exclusion_zones, _is_in_exclusion_zone

_MODELS_DIR = Path(__file__).parent / "models"


@dataclass
class EnrichConfig:
    iterations:   int   = 2
    conf:         float = 0.06    # recall-heavy: the prediction gate does the vetting
    crop_px:      int   = 320     # native-res crop side around the prediction
    accept_px:    float = 45.0    # accept radius in ANALYSIS px (perspective-scaled)
    backfill_s:   float = 2.5     # extrapolate this far before each trace onset
    extend_s:     float = 2.0     # ... and after each trace death
    bridge_max_gap_s: float = 6.0 # interpolate between intervals up to this far apart
    widen_per_s:  float = 1.6     # accept radius + crop growth per extrapolated second
    max_targets:  int   = 25000   # safety cap per iteration
    seek_gap:     int   = 40      # grab() forward instead of seek for gaps under this


def _replay_with_predictions(match, cfg: SegmenterConfig):
    """Full-match replay that also returns per-record track predictions —
    delegates to point_segmenter._replay_core so the detection filtering
    stays identical to the stage-2 replay."""
    return _replay_core(match, 0.0, match.duration + 1.0, cfg, collect=True)


def _interval_velocity(replay, t0: float, t1: float, from_start: bool
                       ) -> Optional[Tuple[float, float, float, float]]:
    """(x, y, vx, vy) at an interval endpoint from ~0.4 s of genuine points."""
    pts = [(fr.t, fr.position[0], fr.position[1]) for fr in replay
           if fr.genuine and fr.position is not None and t0 <= fr.t <= t1]
    if len(pts) < 2:
        return None
    pts = pts[:12] if from_start else pts[-12:]
    (ta, xa, ya), (tb, xb, yb) = pts[0], pts[-1]
    if tb - ta < 1e-3:
        return None
    vx, vy = (xb - xa) / (tb - ta), (yb - ya) / (tb - ta)
    if from_start:
        return xa, ya, vx, vy
    return xb, yb, vx, vy


def build_targets(match, replay, preds, cfg: SegmenterConfig,
                  ecfg: EnrichConfig) -> Dict[int, Tuple[float, float, float]]:
    """frame_index -> (pred_x, pred_y, extrapolation_seconds)."""
    dt = 1.0 / max(match.fps, 1e-6)
    targets: Dict[int, Tuple[float, float, float]] = {}

    # GAP: coasting frames inside tracked spans
    for rec, pred in zip(match.records, preds):
        if pred is None:
            continue
        x, y, tsd = pred
        if tsd > 1.5 * dt:                      # no detection this frame
            targets[rec.f] = (x, y, 0.0)

    intervals = alive_intervals(replay, cfg.alive_merge_gap_s)

    # BRIDGE: interpolate across the gap between consecutive intervals —
    # the ball was tracked on both sides, so the linear prior is strong
    # (widen still grows toward the middle to absorb bounce kinks).
    for (s0, e0), (s1, e1) in zip(intervals, intervals[1:]):
        gap = s1 - e0
        if not 0.0 < gap <= ecfg.bridge_max_gap_s:
            continue
        a = _interval_velocity(replay, max(s0, e0 - 0.4), e0, False)
        b = _interval_velocity(replay, s1, min(e1, s1 + 0.4), True)
        if a is None or b is None:
            continue
        xa, ya = a[0], a[1]
        xb, yb = b[0], b[1]
        i0, _ = match.index_range(e0, e0)
        _, i1 = match.index_range(s1, s1)
        for i in range(i0, min(i1, len(match.records))):
            rec = match.records[i]
            if not e0 < rec.t < s1:
                continue
            u = (rec.t - e0) / gap
            px, py = xa + (xb - xa) * u, ya + (yb - ya) * u
            extrap = min(rec.t - e0, s1 - rec.t)
            if 0 <= px < 960 and 0 <= py < cfg.frame_height_px:
                targets[rec.f] = (px, py, extrap)

    for start, end in intervals:
        # BACKFILL before onset
        ep = _interval_velocity(replay, start, min(end, start + 0.4), True)
        if ep is not None:
            x0, y0, vx, vy = ep
            i1, _ = match.index_range(start, start)
            i0, _ = match.index_range(start - ecfg.backfill_s,
                                      start - ecfg.backfill_s)
            for i in range(max(0, i0), min(i1, len(match.records))):
                rec = match.records[i]
                back = start - rec.t
                if back <= 0:
                    continue
                px, py = x0 - vx * back, y0 - vy * back
                if 0 <= px < 960 and 0 <= py < cfg.frame_height_px:
                    targets.setdefault(rec.f, (px, py, back))
        # EXTEND after death
        ep = _interval_velocity(replay, max(start, end - 0.4), end, False)
        if ep is not None:
            x1, y1, vx, vy = ep
            _, i0 = match.index_range(end, end)
            _, i1 = match.index_range(end + ecfg.extend_s, end + ecfg.extend_s)
            for i in range(i0, min(i1, len(match.records))):
                rec = match.records[i]
                fwd = rec.t - end
                if fwd <= 0:
                    continue
                px, py = x1 + vx * fwd, y1 + vy * fwd
                if 0 <= px < 960 and 0 <= py < cfg.frame_height_px:
                    targets.setdefault(rec.f, (px, py, fwd))

    if len(targets) > ecfg.max_targets:
        # keep the least-extrapolated targets
        keep = sorted(targets.items(), key=lambda kv: kv[1][2])[:ecfg.max_targets]
        targets = dict(keep)
    return targets


def detect_targets(video_path: str, targets: Dict[int, Tuple[float, float, float]],
                   analysis_size: Tuple[int, int], ecfg: EnrichConfig,
                   persp, exclusion_zones) -> Dict[int, List[Tuple[float, float, float]]]:
    """Decode target frames at native res, run the ball model on a crop
    around each prediction, return accepted detections in analysis coords."""
    from ultralytics import YOLO
    model = YOLO(str(_MODELS_DIR / "ball_best.pt"))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    nat_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    nat_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sx, sy = nat_w / analysis_size[0], nat_h / analysis_size[1]

    found: Dict[int, List[Tuple[float, float, float]]] = {}
    cur = -10**9
    t_start = time.time()
    order = sorted(targets)
    for n, f in enumerate(order):
        px, py, extrap = targets[f]
        if 0 < f - cur <= ecfg.seek_gap:
            for _ in range(f - cur - 1):
                cap.grab()
            ok, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
        cur = f
        if not ok or frame is None:
            continue

        widen = 1.0 + ecfg.widen_per_s * extrap
        half = int(ecfg.crop_px * min(widen, 3.0) / 2)
        cxn, cyn = px * sx, py * sy
        x1 = max(0, int(cxn) - half); x2 = min(int(nat_w), int(cxn) + half)
        y1 = max(0, int(cyn) - half); y2 = min(int(nat_h), int(cyn) + half)
        if x2 - x1 < 32 or y2 - y1 < 32:
            continue
        crop = frame[y1:y2, x1:x2]
        res = model(crop, verbose=False, conf=ecfg.conf,
                    imgsz=max(320, min(640, ((x2 - x1) // 32) * 32)))
        if not (res and res[0].boxes):
            continue
        accept = ecfg.accept_px * max(persp(py), 0.35) * widen
        dets = []
        for b in res[0].boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            ax = (x1 + (bx1 + bx2) / 2.0) / sx
            ay = (y1 + (by1 + by2) / 2.0) / sy
            if ((ax - px) ** 2 + (ay - py) ** 2) > accept ** 2:
                continue
            if _is_in_exclusion_zone(ax, ay, exclusion_zones):
                continue
            dets.append((round(ax, 1), round(ay, 1), round(float(b.conf[0]), 3)))
        if dets:
            found[f] = dets
        if n and n % 2000 == 0:
            rate = n / max(time.time() - t_start, 1e-9)
            print(f"[ENRICH]   {n}/{len(order)} frames ({rate:.0f} f/s), "
                  f"{sum(len(v) for v in found.values())} detections")
    cap.release()
    return found


def _rewrite_telemetry(telemetry_path: str,
                       rballs: Dict[int, List[Tuple[float, float, float]]],
                       stats: dict) -> None:
    """Rewrite the JSONL preserving all original fields; replace `rballs`."""
    tmp = telemetry_path + ".enrich.tmp"
    with open(telemetry_path, "r") as src, open(tmp, "w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj:
                obj["meta"]["rballs"] = stats
            else:
                obj.pop("rballs", None)
                dets = rballs.get(obj["f"])
                if dets:
                    obj["rballs"] = [list(d) for d in dets]
            dst.write(json.dumps(obj) + "\n")
    os.replace(tmp, telemetry_path)


def enrich(video_path: str, telemetry_path: Optional[str] = None,
           ecfg: Optional[EnrichConfig] = None) -> dict:
    ecfg = ecfg or EnrichConfig()
    telemetry_path = telemetry_path or telemetry_path_for(video_path)
    exclusion_zones = load_cached_exclusion_zones(video_path) or []

    # Seed with detections already in the file — a re-run refines them
    # rather than silently dropping the previous pass's work.
    all_rballs: Dict[int, List[Tuple[float, float, float]]] = {}
    for rec_obj in open(telemetry_path):
        rec_obj = rec_obj.strip()
        if not rec_obj:
            continue
        obj = json.loads(rec_obj)
        if "meta" not in obj and obj.get("rballs"):
            all_rballs[obj["f"]] = [tuple(d) for d in obj["rballs"]]

    stats = {"iterations": [], "conf": ecfg.conf}
    for it in range(1, ecfg.iterations + 1):
        match = load_telemetry(telemetry_path)
        cfg = SegmenterConfig()
        size = match.meta.get("analysis_size") or [960, 540]
        cfg.frame_height_px = float(size[1])
        suppress_static_candidates(match, cfg)
        persp = make_image_row_perspective(cfg.frame_height_px)

        replay, preds = _replay_with_predictions(match, cfg)
        targets = build_targets(match, replay, preds, cfg, ecfg)
        # don't re-decode frames that already produced rballs this run
        targets = {f: v for f, v in targets.items() if f not in all_rballs}
        print(f"[ENRICH] iteration {it}: {len(targets)} target frames")
        if not targets:
            break
        found = detect_targets(video_path, targets, tuple(size), ecfg,
                               persp, exclusion_zones)
        n_det = sum(len(v) for v in found.values())
        print(f"[ENRICH] iteration {it}: +{n_det} detections "
              f"on {len(found)} frames")
        stats["iterations"].append({"targets": len(targets),
                                    "frames_hit": len(found),
                                    "detections": n_det})
        if not found:
            break
        all_rballs.update(found)
        _rewrite_telemetry(telemetry_path, all_rballs, stats)

    total = sum(len(v) for v in all_rballs.values())
    print(f"[ENRICH] Done: {total} re-detections on {len(all_rballs)} frames "
          f"→ {telemetry_path}")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tracker-guided native-res ball re-detection")
    parser.add_argument("video", help="source video (telemetry sits beside it)")
    parser.add_argument("--telemetry", default=None)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    ecfg = EnrichConfig(iterations=args.iterations)
    enrich(args.video, args.telemetry, ecfg)
