"""In-court ball TRACE as the point-end evidence, replacing ball presence.

Presence — "did any detection survive the conf floor, exclusion zones and static
blobs" — is what `walk-ball` ends points on, and it is not robust: per-look
detection rates run 9.7%-43% across the corpus, so that policy has to buy safety
with a 5.0 s veto and a 5.0 s quiet trigger and leaves ~8.6 s of dead time past
each point.  A trace is stricter: an IMM-tracked ball that is actually MOVING,
inside the court, corroborated across frames.  A single glint cannot make one.

Nothing here re-implements tracking.  `point_segmenter.replay_ball_tracker` and
`alive_intervals` already turn a telemetry stream into "when was a real ball in
flight", and `ball_tracker.BallTrackManager` is the IMM behind them; this module
adapts the rally-reel's end telemetry into the `MatchTelemetry` shape they
expect, adds the court gate, and expresses the two point-end rules over the
resulting intervals.

SAMPLE RATE IS A CORRECTNESS PRECONDITION, not a quality knob.  `BallTrackManager`
builds a constant-dt state transition from `fps` (ball_tracker.py:120) and calls
`predict()` once per `update()`, so the timebase must be uniform and the rate
must be real.  At the shipped 10 Hz, dt is 0.1 s and a 1000 px/s ball moves
~100 px between samples against `gate_base_px = 50` — association fails and
`confirm_hits = 3` inside `confirm_window_s = 0.6` has only 6 looks to work
with, so the tracker essentially never confirms.  At ball_fps 30 every corpus
clip lands on a uniform 29.97 Hz (stride 1 on 29.97 fps sources, stride 2 on
59.94), which is the rate those constants were tuned against — so none of them
is rescaled here.
"""

import bisect
import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..utilities import Config
from .config import ReelConfig
from .points import usable_walk_intervals


# ── court gate ───────────────────────────────────────────────────────────

def court_gate(court_cache_path: str, cfg: ReelConfig,
               frame_size: Tuple[int, int] = (960, 540)):
    """`(x, y) -> bool`: is this detection on the court, in ANALYSIS PIXELS.

    Pixel space on purpose, with no homography.  A ground-plane homography
    describes points ON the ground, and a ball in flight is not: mapping an
    airborne ball projects it past the far baseline, and above the horizon line
    the sign of w flips and the mapped point is garbage.  Gating on that would
    silently delete every high far-side ball — the exact detections a far-serve
    rally depends on.  The cached corners are already in the 960x540 analysis
    space the detections use, so no conversion is needed at all.

    The polygon is derived from the quad rather than hardcoded.  `anya_base.
    _compute_active_zone_polygon` uses 150/200 px literals, which do not
    transfer between a court spanning 149 px of depth and one spanning 200+,
    and its near edge is the baseline itself — excluding the bottom of the
    frame where near-side serves and deep returns actually happen.  Here the
    near edge runs to the frame bottom and every margin is expressed in court
    units at the far baseline's own px/ft scale.

    Returns a gate that admits everything if `cfg.trace_court_gate` is False, or
    if the cache is missing or fails the sanity checks below — a mis-clicked or
    stale court cache would otherwise wipe out the trace invisibly.
    """
    if not cfg.trace_court_gate:
        return lambda x, y: True
    if not os.path.isfile(court_cache_path):
        print(f"[TRACE]   WARN no court cache at {os.path.basename(court_cache_path)}; "
              f"court gate OFF")
        return lambda x, y: True

    with open(court_cache_path) as fh:
        pts = json.load(fh).get("points") or []
    if len(pts) != 4:
        print(f"[TRACE]   WARN court cache has {len(pts)} points, expected 4; gate OFF")
        return lambda x, y: True

    W, H = float(frame_size[0]), float(frame_size[1])
    (blx, bly), (brx, bry), (trx, try_), (tlx, tly) = [(float(a), float(b)) for a, b in pts]

    near_w = brx - blx
    far_w = trx - tlx
    # Corner order is a contract (COURT_CORNER_ORDER); a violated one means the
    # calibration is not what we think it is, and a silent pass is worse here.
    if near_w <= 0 or far_w <= 0 or bly <= try_ or bry <= tly:
        print("[TRACE]   WARN court quad is not near-below-far / left-to-right; gate OFF")
        return lambda x, y: True
    area = cv2.contourArea(np.array([[blx, bly], [brx, bry], [trx, try_], [tlx, tly]],
                                    dtype=np.float32))
    if area < 0.05 * W * H:
        print(f"[TRACE]   WARN court quad is {area / (W * H):.1%} of frame; gate OFF")
        return lambda x, y: True

    ppf_far = far_w / float(Config.COURT_WIDTH_FT)     # px per court-foot, far end
    pad = cfg.trace_court_pad_ft * ppf_far
    lat_far = cfg.trace_court_lateral_frac * far_w
    lat_near = cfg.trace_court_near_frac * near_w

    poly = np.array([
        [0.0, H], [W, H],                                  # near edge = frame bottom
        [brx + lat_near, bry],
        [trx + lat_far, try_ - pad],
        [tlx - lat_far, tly - pad],
        [blx - lat_near, bly],
    ], dtype=np.float32)

    def gate(x: float, y: float) -> bool:
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    return gate


# ── telemetry -> trace intervals ─────────────────────────────────────────

def _near_box_lookup(records: Sequence[Dict], pose_tol_s: float):
    """`t -> near box`, matched to the NEAREST pose sample within a tolerance.

    Pose runs at 15 Hz and the ball at 30, so half the ball records carry no
    `np` of their own.  Nearest-sample rather than a forward hold: a box carried
    arbitrarily far forward applies stale exclusion geometry to a player who has
    since moved, and the in-box test is what suppresses racket/arm false
    positives at contact.
    """
    ts = [float(r["t"]) for r in records if r.get("pn") and r.get("np")]
    boxes = [r["np"] for r in records if r.get("pn") and r.get("np")]

    def lookup(t: float):
        if not ts:
            return None
        i = bisect.bisect_left(ts, t)
        best, best_d = None, None
        for j in (i - 1, i):
            if 0 <= j < len(ts):
                d = abs(ts[j] - t)
                if best_d is None or d < best_d:
                    best, best_d = boxes[j], d
        return best if best_d is not None and best_d <= pose_tol_s else None

    return lookup


def trace_intervals(meta: Dict, records: Sequence[Dict], filter_balls: Callable,
                    court_cache_path: str, cfg: ReelConfig
                    ) -> Tuple[List[Tuple[float, float]], Dict]:
    """Intervals of genuine in-court ball motion, plus stats for the log line.

    Takes the caller's `_ball_stream` output so the trace and the rule-based
    policies cannot disagree about what counts as a ball.
    """
    from ..point_segmenter import (FrameRecord, MatchTelemetry, SegmenterConfig,
                                   alive_intervals, replay_ball_tracker)

    ball_fps = float(meta.get("ball_fps") or 0.0)
    if ball_fps < cfg.trace_min_ball_fps:
        raise ValueError(
            f"end telemetry samples the ball at {ball_fps:.1f} Hz, below "
            f"trace_min_ball_fps={cfg.trace_min_ball_fps}.  The IMM tracker "
            f"cannot confirm a moving ball at that rate — re-extract with "
            f"ball_fps={cfg.trace_ball_fps} (see ReelConfig.trace_ball_fps).")

    gate = court_gate(court_cache_path, cfg,
                      tuple(meta.get("analysis_size") or (960, 540)))
    pose_fps = float(meta.get("pose_fps") or 15.0)
    near_box_at = _near_box_lookup(records, 0.5 / max(pose_fps, 1e-6))

    frames: List[FrameRecord] = []
    n_looks = n_det = n_gated = 0
    for r in records:
        if not r.get("bn"):           # ball never looked here; uniform dt needs this
            continue
        n_looks += 1
        dets = filter_balls(r)
        n_det += len(dets)
        # Gate at TRACKER INPUT, not on the output track: the IMM must never
        # lock onto off-court clutter in the first place.  Gating output
        # positions instead would also kill legitimate coasting through a gap.
        kept = [d for d in dets if gate(d[0], d[1])]
        n_gated += len(kept)
        frames.append(FrameRecord(
            f=int(r.get("f", 0)), t=float(r["t"]),
            near_box=near_box_at(float(r["t"])), near_world=r.get("npw"),
            far_box=None, far_held=False, far_world=None,   # end telemetry has no far box
            balls=[tuple(d) for d in kept], toss=[], trophy=0.0, stgcn=0.0))

    if not frames:
        return [], {"looks": 0, "dets": 0, "in_court": 0, "intervals": 0, "alive_s": 0.0}

    # MatchTelemetry computes fps = meta["fps"] / meta["stride"].  End telemetry
    # writes the SOURCE fps with stride 1, which on a 60 fps clip would hand the
    # tracker 59.94 while records arrive at 29.97 — a silent 2x dt error.
    match = MatchTelemetry({"fps": ball_fps, "stride": 1,
                            "analysis_size": meta.get("analysis_size", [960, 540])},
                           frames)

    scfg = SegmenterConfig()
    scfg.frame_height_px = float((meta.get("analysis_size") or [960, 540])[1])
    scfg.alive_merge_gap_s = cfg.trace_merge_gap_s

    replay = replay_ball_tracker(match, 0.0, match.duration + 1.0, scfg)
    ivals = alive_intervals(replay, scfg.alive_merge_gap_s)
    # Drop micro-blips.  One confirmed frame of clutter mid-dead-time otherwise
    # resets rule 2's quiet clock; SegmenterConfig.far_trace_min_interval_s
    # exists for the same reason on the far-serve path.
    ivals = [(a, b) for a, b in ivals if b - a >= cfg.trace_min_interval_s]

    stats = {
        "looks": n_looks, "dets": n_det, "in_court": n_gated,
        "gate_rate": (n_gated / n_det) if n_det else 1.0,
        "intervals": len(ivals),
        "alive_s": float(sum(b - a for a, b in ivals)),
    }
    if n_det and stats["gate_rate"] < 0.20:
        print(f"[TRACE]   WARN court gate passed only {stats['gate_rate']:.0%} of "
              f"detections — check the court cache")
    return ivals, stats


# ── interval algebra ─────────────────────────────────────────────────────

def any_trace_in(intervals: Sequence[Tuple[float, float]],
                 t0: float, t1: float) -> bool:
    """Does any interval overlap [t0, t1]?"""
    if not intervals:
        return False
    starts = [a for a, _ in intervals]
    i = bisect.bisect_right(starts, t1)
    for j in range(max(0, i - 1), -1, -1):
        a, b = intervals[j]
        if b >= t0:
            if a <= t1:
                return True
        if b < t0 and a < t0:
            break
    return False


def trace_gaps(intervals: Sequence[Tuple[float, float]],
               t_start: float, t_end: float) -> List[Tuple[float, float]]:
    """Maximal spans in [t_start, t_end] covered by no interval."""
    gaps, cursor = [], t_start
    for a, b in intervals:
        if b <= t_start:
            continue
        if a > t_end:
            break
        if a > cursor:
            gaps.append((cursor, min(a, t_end)))
        cursor = max(cursor, b)
        if cursor >= t_end:
            return gaps
    if cursor < t_end:
        gaps.append((cursor, t_end))
    return gaps


def _merge_spans(spans: Sequence[Tuple[float, float]],
                 gap: float) -> List[Tuple[float, float]]:
    """Join spans separated by <= gap."""
    out: List[List[float]] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


# ── the two rules ────────────────────────────────────────────────────────

def trace_onsets(intervals: Sequence[Tuple[float, float]],
                 walks: Sequence[Dict], duration: float, cfg: ReelConfig,
                 look_times: Optional[Sequence[float]] = None,
                 last_record_t: Optional[float] = None) -> List[Tuple[float, str]]:
    """Point ends under end_policy="trace".  Returns [(t, source)].

      1  "walk-trace"   a walk span starts at w0 and NO trace appears in
                        [w0, w0 + trace_walk_confirm_s].  Stamped at
                        w0 + trace_walk_stamp_s — the point stopped when the
                        player turned away, and the lookahead is confirmation,
                        not duration.
      2  "trace-quiet"  trace_quiet_s with no trace at all, where no usable walk
                        span covers the TRIGGER INSTANT.  Stamped at the gap's
                        start + trace_quiet_stamp_s, measured from the last
                        trace rather than from the confirmation.

    Emitting onsets rather than ends keeps `find_point_end` in the loop, so this
    policy inherits point_min_s, point_max_s and next_serve_guard_s instead of
    quietly reimplementing them.
    """
    looks = sorted(look_times or [])
    last_t = duration if last_record_t is None else float(last_record_t)

    def enough_looks(t0: float, t1: float) -> bool:
        if not looks:
            return True
        n = bisect.bisect_right(looks, t1) - bisect.bisect_left(looks, t0)
        return n >= cfg.ball_quiet_min_looks

    # usable_walk_intervals filters but never MERGES, and the classifier emits
    # runs split by sub-second gaps.  _walk_ball_onsets collapses them with its
    # pending_a latch; an interval rule has no such latch and would emit one
    # onset per fragment inside a single dead period, which costs precision
    # directly because eval_point_end divides matches by the detection count.
    usable = usable_walk_intervals(walks, cfg)
    spans = _merge_spans([(float(w["start_second"]), float(w["end_second"]))
                          for w in usable], cfg.trace_walk_merge_gap_s)

    onsets: List[Tuple[float, str]] = []

    # ── rule 1 ──
    for w0, _w1 in spans:
        w_end = w0 + cfg.trace_walk_confirm_s
        # "No trace in the next 3 s" is trivially true past the last record.
        # Confirming on absence-of-data is the truncation-shaped failure.
        if w_end > last_t:
            continue
        if not enough_looks(w0, w_end):
            continue
        if any_trace_in(intervals, w0, w_end):
            continue
        onsets.append((w0 + cfg.trace_walk_stamp_s, "walk-trace"))

    # ── rule 2 ──
    def walk_covers(t: float) -> bool:
        return any(a <= t <= b for a, b in spans)

    if intervals:
        # Gaps before the first trace are skipped, mirroring _walk_ball_onsets'
        # `if last_ball is None: continue` — silence before any ball was ever
        # seen is not evidence that a point ended.
        for g0, g1 in trace_gaps(intervals, intervals[0][1], last_t):
            if g1 - g0 < cfg.trace_quiet_s:
                continue
            t_trig = g0 + cfg.trace_quiet_s
            if not enough_looks(g0, t_trig):
                continue
            if walk_covers(t_trig):        # rule 1's territory
                continue
            onsets.append((g0 + cfg.trace_quiet_stamp_s, "trace-quiet"))

    # Collapse onsets landing in the same moment.  Both rules can legitimately
    # fire inside one long gap (a walk beginning after the quiet trigger), and
    # a duplicate is a plain false positive against n_det.  The corroborated
    # label wins the tie.
    onsets.sort(key=lambda x: (x[0], x[1] != "walk-trace"))
    deduped: List[Tuple[float, str]] = []
    for t, src in onsets:
        if deduped and t - deduped[-1][0] <= cfg.trace_onset_dedupe_s:
            continue
        deduped.append((t, src))
    return deduped
