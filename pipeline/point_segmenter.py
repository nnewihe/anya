"""
point_segmenter.py
==================
Stage 2 of the dead-time cutter: turn a match-telemetry JSONL file
(match_telemetry.py) into point segments [serve start .. point end], so the
cutter can drop everything in between (the "dead time" from the end of one
point to the start of the next service motion).

Runs on telemetry only — no video, no ML models — so it re-runs in seconds
while tuning.  All knobs live in SegmenterConfig.

Pipeline
--------
1. SERVE EVENTS (point starts)
     Near side: ready-band dwell behind the near baseline, then the
       toss+trophy weighted score (port of TransitionEngine's ARMED logic).
     Far side:  ready-band dwell behind the far baseline, then the ST-GCN
       serve probability recorded in the telemetry.
     Events are deduped (two serves can't happen within a few seconds; the
     near detector wins conflicts since it is the better-tuned signal).

2. SERVE VALIDATION
     Each event is checked for a serve-like ball trace (downward +
     horizontal motion shortly after the event, perspective-scaled) by
     replaying the recorded ball detections through the IMM tracker.
     A Viterbi-decoded serving-side HMM (serves are sticky — the same player
     serves a whole game) then drops unconfirmed events whose side disagrees
     with the inferred pattern.

3. POINT ENDS
     Offline advantage: point i's end must lie in (serve_i, serve_{i+1}), a
     bounded search window.  Within it we fuse:
       • Ball-trace chain — maximal "genuinely moving" trace intervals from
         the tracker replay, chained across gaps; larger gaps are bridged
         only when player kinematics look rally-like.
       • Player kinematics — direction reversals and simultaneous two-player
         movement are rally signatures; steady one-player walking (ball
         retrieval) is not.  They extend a trace chain that died early (weak
         far-side ball tracking) and are the sole authority when no usable
         trace exists at all.
       • Carried-ball suppression — a ball whose velocity is coupled to the
         walking near player is being carried, not played.

Run:
    python -m pipeline.point_segmenter match_match_telemetry.jsonl [--csv out.csv]
    python -m pipeline.point_segmenter --self-test
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np

from .ball_tracker import BallTrackManager, make_image_row_perspective


# =====================================================================
# Configuration — every tunable knob for segmentation lives here.
# =====================================================================
@dataclass
class SegmenterConfig:
    # ---- Court constants (overridden from telemetry meta) ----
    court_length_ft: float = 78.0
    frame_height_px: float = 540.0

    # ---- Ready-band gating (WAITING->ARMED port) ----
    near_band_ft: Tuple[float, float] = (-0.5, 3.5)
    far_band_ft:  Tuple[float, float] = (-6.0, 6.0)
    ready_dwell_s:    float = 0.4
    band_window_s:    float = 2.0
    band_out_ratio:   float = 0.25

    # ---- Near serve scoring (toss + trophy) ----
    toss_conf_floor:     float = 0.5
    toss_gap_tolerance:  int   = 1
    toss_confirm_frames: int   = 2
    trophy_weight: float = 0.2
    toss_weight:   float = 0.8
    serve_score_threshold: float = 0.55
    serve_event_window_s:  float = 1.2

    # ---- Far serve scoring ----
    stgcn_threshold: float = 0.55

    # ---- Serve-event bookkeeping ----
    min_serve_separation_s: float = 8.0

    # ---- Serve-trace confirmation (perspective-scaled at the trace) ----
    confirm_window_s:      float = 4.0
    trace_downward_px_s:   float = 40.0
    trace_horizontal_px_s: float = 30.0

    # ---- Serving-side HMM (fitted on 15 labeled matches, see rally_detector) ----
    hmm_p_stay:    float = 0.9355
    hmm_p_correct: float = 0.85

    # ---- Ball-trace liveness during replay ----
    move_velocity_floor_px_s: float = 20.0   # perspective-scaled
    alive_merge_gap_s: float = 0.6           # micro-gaps folded into one interval
    racket_spike_thresh: float = 0.25

    # ---- Carried-ball suppression (port of rally_detector coupling test) ----
    coupling_window_s:        float = 0.40
    coupling_min_player_speed: float = 25.0   # px/s
    coupling_ratio_max:        float = 0.50

    # ---- Point-end chaining ----
    serve_chain_window_s: float = 5.0   # first trace interval must start this soon after serve
    chain_gap_s:          float = 2.5   # always bridge trace gaps up to this
    chain_gap_active_s:   float = 6.0   # bridge up to this when players look rally-like
    activity_gap_s:       float = 2.5   # gap allowed between rally cues when chaining activity
    activity_extend_max_s: float = 12.0 # cap on extending a trace chain via activity alone
    fallback_point_s:     float = 6.0   # assumed length when no evidence at all (ace/short point)
    max_point_s:          float = 60.0
    min_point_s:          float = 1.5
    next_serve_guard_s:   float = 1.5

    # ---- Player-kinematics rally cues ----
    speed_window_s:       float = 0.4
    speed_min_dt_s:       float = 0.15
    reversal_speed_ft_s:  float = 3.0   # |vx| needed to count as a significant direction
    both_moving_ft_s:     float = 3.0   # both players at/above this = rally cue
    speed_stale_s:        float = 0.6   # forward-fill limit when sampling speeds

    # ---- Output segments ----
    pre_roll_s:     float = 2.0    # before a near serve event (captures the motion start)
    far_pre_roll_s: float = 3.0    # ST-GCN fires later into the motion — wider pre-roll
    end_pad_s:      float = 1.0


# =====================================================================
# Telemetry data model
# =====================================================================
@dataclass
class FrameRecord:
    f: int
    t: float
    near_box:   Optional[Tuple[int, int, int, int]]
    near_world: Optional[Tuple[float, float]]
    far_box:    Optional[Tuple[int, int, int, int]]
    far_held:   bool
    far_world:  Optional[Tuple[float, float]]
    balls: List[Tuple[float, float, float]]
    toss:  List[Tuple[float, float, float]]
    trophy: float
    stgcn:  float


class MatchTelemetry:
    """Loaded telemetry: meta header + time-indexed frame records."""

    def __init__(self, meta: dict, records: List[FrameRecord]):
        self.meta = meta
        self.records = records
        self.ts = [r.t for r in records]
        stride = max(1, int(meta.get("stride", 1)))
        self.fps = float(meta.get("fps", 30.0)) / stride

    @property
    def duration(self) -> float:
        return self.ts[-1] if self.ts else 0.0

    def index_range(self, t0: float, t1: float) -> Tuple[int, int]:
        """Record indices covering [t0, t1)."""
        return (bisect.bisect_left(self.ts, t0),
                bisect.bisect_left(self.ts, t1))

    def slice(self, t0: float, t1: float) -> Sequence[FrameRecord]:
        i0, i1 = self.index_range(t0, t1)
        return self.records[i0:i1]


def load_telemetry(path: str) -> MatchTelemetry:
    meta: dict = {}
    records: List[FrameRecord] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj:
                meta = obj["meta"]
                continue
            records.append(FrameRecord(
                f=obj["f"], t=obj["t"],
                near_box=tuple(obj["np"]) if obj.get("np") else None,
                near_world=tuple(obj["npw"]) if obj.get("npw") else None,
                far_box=tuple(obj["fp"]) if obj.get("fp") else None,
                far_held=bool(obj.get("fph", 0)),
                far_world=tuple(obj["fpw"]) if obj.get("fpw") else None,
                balls=[tuple(b) for b in obj.get("balls", [])],
                toss=[tuple(b) for b in obj.get("toss", [])],
                trophy=float(obj.get("trophy", 0.0)),
                stgcn=float(obj.get("stgcn", 0.0)),
            ))
    return MatchTelemetry(meta, records)


# =====================================================================
# Serve events
# =====================================================================
@dataclass
class ServeEvent:
    t: float
    side: str            # "near" | "far"
    score: float
    trace_confirmed: bool = False


class _NearServeScorer:
    """Port of TransitionEngine's ARMED toss/trophy scoring, replayed offline."""

    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.toss_consecutive = 0
        self.toss_gap = 0
        self.toss_above_head = False
        self.toss_min_y: Optional[float] = None
        self.last_toss: Optional[dict] = None
        self._trophy_scores: Deque[Tuple[float, float]] = deque()
        self._toss_scores:   Deque[Tuple[float, float]] = deque()

    def _toss_score(self, rec: FrameRecord, now: float) -> float:
        ny1 = rec.near_box[1]
        candidates = [c for c in rec.toss if c[2] >= self.cfg.toss_conf_floor]
        if not candidates:
            self.last_toss = None
            self.toss_gap += 1
            if self.toss_gap > self.cfg.toss_gap_tolerance:
                self.toss_consecutive = 0
                self.toss_above_head = False
            return 0.0

        best = max(candidates, key=lambda c: c[2])
        cy = best[1]

        moving_up  = (self.last_toss is not None and
                      cy < self.last_toss["y"] and now > self.last_toss["time"])
        above_head = cy < ny1

        if above_head and (self.toss_min_y is None or cy < self.toss_min_y):
            self.toss_min_y = cy
        self.last_toss = {"y": cy, "time": now}

        if moving_up and above_head:
            self.toss_gap = 0
            self.toss_consecutive += 1
            self.toss_above_head = True
        else:
            self.toss_gap += 1
            if self.toss_gap > self.cfg.toss_gap_tolerance:
                self.toss_consecutive = 0
                self.toss_above_head = False

        if not self.toss_above_head:
            return 0.0
        if self.toss_consecutive >= self.cfg.toss_confirm_frames:
            return 1.0
        if self.toss_consecutive >= 1:
            return 0.5
        return 0.0

    def update(self, rec: FrameRecord, now: float) -> float:
        if rec.near_box is None:
            return 0.0
        if rec.trophy > 0:
            self._trophy_scores.append((rec.trophy, now))
        ts = self._toss_score(rec, now)
        if ts > 0:
            self._toss_scores.append((ts, now))
        for buf in (self._trophy_scores, self._toss_scores):
            while buf and now - buf[0][1] > self.cfg.serve_event_window_s:
                buf.popleft()
        max_trophy = max((s for s, _ in self._trophy_scores), default=0.0)
        max_toss   = max((s for s, _ in self._toss_scores),   default=0.0)
        return (self.cfg.trophy_weight * max_trophy +
                self.cfg.toss_weight * max_toss)

    def validate(self, rec: FrameRecord) -> bool:
        """The toss must have peaked above the player's head."""
        if rec.near_box is None:
            return False
        return self.toss_min_y is not None and self.toss_min_y < rec.near_box[1]


class _FarServeScorer:
    """Far serves are scored directly by the recorded ST-GCN probability."""

    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg

    def reset(self) -> None:
        pass

    def update(self, rec: FrameRecord, now: float) -> float:
        if rec.far_box is None:
            return 0.0
        return rec.stgcn

    def validate(self, rec: FrameRecord) -> bool:
        return True


def detect_serve_events(match: MatchTelemetry, side: str,
                        cfg: SegmenterConfig) -> List[ServeEvent]:
    """
    Ready-band dwell + score threshold, replayed over the telemetry.

    A serve fires when the player has settled behind their baseline
    (ready_dwell_s inside the band), the recent out-of-band ratio stays low,
    and the side's serve score crosses its threshold.
    """
    if side == "near":
        baseline, direction, band = 0.0, -1.0, cfg.near_band_ft
        scorer = _NearServeScorer(cfg)
        threshold = cfg.serve_score_threshold
        world_of = lambda r: r.near_world
    else:
        baseline, direction, band = cfg.court_length_ft, 1.0, cfg.far_band_ft
        scorer = _FarServeScorer(cfg)
        threshold = cfg.stgcn_threshold
        world_of = lambda r: r.far_world

    events: List[ServeEvent] = []
    ready_start: Optional[float] = None
    armed = False
    band_hist: Deque[Tuple[float, bool]] = deque()
    cooldown_until = -math.inf

    for rec in match.records:
        now = rec.t
        if now < cooldown_until:
            continue

        world = world_of(rec)
        in_band = False
        if world is not None:
            dist = (world[1] - baseline) * direction
            in_band = band[0] <= dist <= band[1]

        if not armed:
            if in_band:
                if ready_start is None:
                    ready_start = now
                elif now - ready_start > cfg.ready_dwell_s:
                    armed = True
                    scorer.reset()
                    band_hist.clear()
            else:
                ready_start = None
            continue

        # ---- armed: watch the out-of-band ratio, then score ----
        band_hist.append((now, in_band))
        while band_hist and now - band_hist[0][0] > cfg.band_window_s:
            band_hist.popleft()
        if len(band_hist) > 1:
            total = band_hist[-1][0] - band_hist[0][0]
            if total > 1.0:
                t_out = sum(band_hist[i + 1][0] - band_hist[i][0]
                            for i in range(len(band_hist) - 1)
                            if not band_hist[i][1])
                if t_out / total > cfg.band_out_ratio:
                    armed = False
                    ready_start = None
                    continue

        if not in_band:
            continue

        score = scorer.update(rec, now)
        if score >= threshold and scorer.validate(rec):
            events.append(ServeEvent(t=now, side=side, score=score))
            armed = False
            ready_start = None
            cooldown_until = now + cfg.min_serve_separation_s

    return events


def dedupe_serve_events(events: List[ServeEvent],
                        cfg: SegmenterConfig) -> List[ServeEvent]:
    """
    Collapse events closer than min_serve_separation_s.  Two serves cannot
    happen that close together, so one of a conflicting pair is false: the
    near detector (toss-anchored, well tuned) wins over the far detector
    (ST-GCN only); within the same side the earlier event wins (it marks the
    serve-motion onset).
    """
    events = sorted(events, key=lambda e: e.t)
    kept: List[ServeEvent] = []
    for evt in events:
        if kept and evt.t - kept[-1].t < cfg.min_serve_separation_s:
            prev = kept[-1]
            if prev.side == evt.side or prev.side == "near":
                continue                       # keep prev, drop evt
            kept[-1] = evt                     # near beats an earlier far event
        else:
            kept.append(evt)
    return kept


# =====================================================================
# Serving-side HMM (Viterbi) — copied from rally_detector.py so this
# module stays free of the model-loading import chain.
# =====================================================================
def _viterbi(obs_sides: List[str], p_stay: float, p_correct: float) -> List[str]:
    n = len(obs_sides)
    if n == 0:
        return []
    if n == 1:
        return list(obs_sides)

    sides = ["near", "far"]
    log_trans = np.log(np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]]))
    log_emit  = np.log(np.array([[p_correct, 1 - p_correct],
                                 [1 - p_correct, p_correct]]))
    obs = [0 if o == "near" else 1 for o in obs_sides]

    delta = np.log(np.array([0.5, 0.5])) + log_emit[:, obs[0]]
    psi = np.zeros((n, 2), dtype=int)
    for t in range(1, n):
        new_delta = np.empty(2)
        for s in range(2):
            scores = delta + log_trans[:, s]
            best = int(np.argmax(scores))
            psi[t, s] = best
            new_delta[s] = scores[best] + log_emit[s, obs[t]]
        delta = new_delta
    path = [0] * n
    path[-1] = int(np.argmax(delta))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return [sides[s] for s in path]


def hmm_filter_events(events: List[ServeEvent],
                      cfg: SegmenterConfig, verbose: bool = True) -> List[ServeEvent]:
    """Drop serve events whose side disagrees with the Viterbi-decoded serving
    pattern AND that lack a confirmed serve trace (weak anomalies)."""
    if len(events) < 2:
        return events
    decoded = _viterbi([e.side for e in events], cfg.hmm_p_stay, cfg.hmm_p_correct)
    kept = []
    for evt, dec in zip(events, decoded):
        if evt.side != dec and not evt.trace_confirmed:
            if verbose:
                print(f"[HMM] Dropped serve @ {evt.t:.2f}s side={evt.side} "
                      f"(decoded={dec}, unconfirmed trace)")
            continue
        kept.append(evt)
    return kept


# =====================================================================
# Ball-trace replay
# =====================================================================
# _SmoothedVelocity / carried-ball coupling are ported from rally_detector.py
# (kept local so this module never imports the YOLO-loading chain).
class _SmoothedVelocity:
    def __init__(self, window_sec: float):
        self.window_sec = float(window_sec)
        self._pts: Deque[Tuple[float, float, float]] = deque()

    def add(self, t: float, x: float, y: float) -> None:
        self._pts.append((float(t), float(x), float(y)))
        cutoff = float(t) - self.window_sec
        while self._pts and self._pts[0][0] < cutoff:
            self._pts.popleft()

    def velocity(self) -> Optional[Tuple[float, float]]:
        if len(self._pts) < 2:
            return None
        t0, x0, y0 = self._pts[0]
        t1, x1, y1 = self._pts[-1]
        if t1 <= t0:
            return None
        return ((x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0))


def _is_carried(v_ball, v_player, cfg: SegmenterConfig) -> bool:
    if v_ball is None or v_player is None:
        return False
    ball_speed = math.hypot(*v_ball)
    player_speed = math.hypot(*v_player)
    if ball_speed < 1e-6:
        return False
    ratio = math.hypot(v_ball[0] - v_player[0], v_ball[1] - v_player[1]) / ball_speed
    return ratio < cfg.coupling_ratio_max and player_speed >= cfg.coupling_min_player_speed


@dataclass
class ReplayFrame:
    t: float
    genuine: bool                      # trace alive, above velocity floor, not carried
    racket_prob: float
    position: Optional[Tuple[float, float]]
    tsd: float = 0.0                   # seconds since the last real detection


def replay_ball_tracker(match: MatchTelemetry, t0: float, t1: float,
                        cfg: SegmenterConfig) -> List[ReplayFrame]:
    """
    Re-run the IMM ball tracker over the recorded detections in [t0, t1).

    Detections inside either player's box are excluded (racket/arm/body false
    positives); the tracker's coasting bridges the resulting occlusions.
    """
    persp = make_image_row_perspective(cfg.frame_height_px)
    mgr = BallTrackManager(fps=match.fps, perspective_scale=persp)
    ball_vel = _SmoothedVelocity(cfg.coupling_window_s)
    player_vel = _SmoothedVelocity(cfg.coupling_window_s)

    out: List[ReplayFrame] = []
    for rec in match.slice(t0, t1):
        dets = []
        for bx, by, conf in rec.balls:
            inside = False
            for box in (rec.near_box, rec.far_box):
                if box and box[0] <= bx <= box[2] and box[1] <= by <= box[3]:
                    inside = True
                    break
            if not inside:
                dets.append((bx, by, conf))

        status = mgr.update(dets, rec.t)

        if status.position is not None:
            ball_vel.add(rec.t, *status.position)
        if rec.near_box is not None:
            player_vel.add(rec.t,
                           (rec.near_box[0] + rec.near_box[2]) / 2.0,
                           (rec.near_box[1] + rec.near_box[3]) / 2.0)

        genuine = False
        if status.has_moving_trace and status.position is not None:
            floor = cfg.move_velocity_floor_px_s * persp(status.position[1])
            if status.speed_px_s >= floor:
                genuine = not _is_carried(ball_vel.velocity(),
                                          player_vel.velocity(), cfg)
        out.append(ReplayFrame(rec.t, genuine, status.racket_prob, status.position,
                               status.time_since_detection))
    return out


def alive_intervals(replay: List[ReplayFrame],
                    merge_gap_s: float) -> List[Tuple[float, float]]:
    """
    Maximal [start, end] intervals of genuine trace motion, folding gaps
    <= merge_gap_s into a single interval.

    Interval ends are anchored to the last REAL detection (t - tsd), not the
    coasted prediction — the tracker keeps a dying trace nominally alive for
    up to miss_timeout_s after the ball disappears, and that tail must not
    count as rally time (mid-interval coasting is unaffected).
    """
    raw: List[List[float]] = []       # [start, raw_end, anchored_end]
    for fr in replay:
        if not fr.genuine:
            continue
        anchored = fr.t - min(fr.tsd, fr.t)
        if raw and fr.t - raw[-1][1] <= merge_gap_s:
            raw[-1][1] = fr.t
            raw[-1][2] = max(raw[-1][2], anchored)
        else:
            raw.append([fr.t, fr.t, max(fr.t - fr.tsd, 0.0)])
    return [(start, max(start, anchor)) for start, _, anchor in raw]


def confirm_serve_trace(replay: List[ReplayFrame], cfg: SegmenterConfig) -> bool:
    """
    True when the replay window contains a serve-like trace: net downward
    (gravity after contact) AND net horizontal motion over any ~0.3 s stretch
    of genuine trace points.  Thresholds are perspective-scaled so far-side
    serves (fewer px/s) are judged fairly.
    """
    persp = make_image_row_perspective(cfg.frame_height_px)
    pts = [(fr.t, fr.position[0], fr.position[1])
           for fr in replay if fr.genuine and fr.position is not None]
    for i in range(1, len(pts)):
        t1, x1, y1 = pts[i]
        j = i - 1
        while j > 0 and t1 - pts[j][0] < 0.3:
            j -= 1
        t0, x0, y0 = pts[j]
        dt = t1 - t0
        if dt < 0.15:
            continue
        scale = persp((y0 + y1) / 2.0)
        if ((y1 - y0) / dt >= cfg.trace_downward_px_s * scale and
                abs(x1 - x0) / dt >= cfg.trace_horizontal_px_s * scale):
            return True
    return False


# =====================================================================
# Player kinematics — rally cues from world-space telemetry
# =====================================================================
class PlayerKinematics:
    """
    Precomputes, from the players' world positions:
      • per-record smoothed speed and x-velocity for each side,
      • direction-reversal times (a significant vx sign flip — the signature
        of rally footwork, absent from ball-retrieval walking),
      • "both players moving" times (server AND receiver active at once).
    Reversals + both-moving times together form the rally-cue timeline used
    to bridge trace gaps and to find point ends without a ball trace.
    """

    def __init__(self, match: MatchTelemetry, cfg: SegmenterConfig):
        self.cfg = cfg
        ts = match.ts
        n = len(ts)
        self.speed_near = [None] * n
        self.speed_far  = [None] * n
        reversal_times: List[float] = []

        for side, world_of, speed_arr in (
            ("near", lambda r: r.near_world, self.speed_near),
            ("far",  lambda r: r.far_world,  self.speed_far),
        ):
            valid: List[Tuple[float, float, float, int]] = []   # (t, wx, wy, idx)
            for i, rec in enumerate(match.records):
                w = world_of(rec)
                if w is not None:
                    valid.append((rec.t, w[0], w[1], i))

            j = 0
            last_sign = 0
            for k in range(len(valid)):
                t1, x1, y1, idx = valid[k]
                while j < k and t1 - valid[j][0] > cfg.speed_window_s:
                    j += 1
                # earliest sample still inside the window (or just before it)
                jj = max(0, j - 1) if j > 0 and t1 - valid[j][0] < cfg.speed_min_dt_s else j
                t0, x0, y0, _ = valid[jj]
                dt = t1 - t0
                if dt < cfg.speed_min_dt_s:
                    continue
                vx = (x1 - x0) / dt
                vy = (y1 - y0) / dt
                speed_arr[idx] = math.hypot(vx, vy)
                if abs(vx) >= cfg.reversal_speed_ft_s:
                    sign = 1 if vx > 0 else -1
                    if last_sign != 0 and sign != last_sign:
                        reversal_times.append(t1)
                    last_sign = sign

        # Forward-fill speeds onto the record grid (bounded staleness) and
        # collect "both players moving" cue times.
        both_times: List[float] = []
        last_near = last_far = None
        last_near_t = last_far_t = -1e9
        for i, t in enumerate(ts):
            if self.speed_near[i] is not None:
                last_near, last_near_t = self.speed_near[i], t
            if self.speed_far[i] is not None:
                last_far, last_far_t = self.speed_far[i], t
            near_ok = last_near is not None and t - last_near_t <= cfg.speed_stale_s
            far_ok  = last_far  is not None and t - last_far_t  <= cfg.speed_stale_s
            if (near_ok and far_ok and
                    last_near >= cfg.both_moving_ft_s and
                    last_far  >= cfg.both_moving_ft_s):
                both_times.append(t)

        self.rally_cues: List[float] = sorted(reversal_times + both_times)

    def rally_like(self, t0: float, t1: float) -> bool:
        """Any rally cue inside (t0, t1)?"""
        i = bisect.bisect_right(self.rally_cues, t0)
        return i < len(self.rally_cues) and self.rally_cues[i] < t1

    def chain_activity(self, seed_t: float, cap_t: float, gap_s: float) -> float:
        """Walk the rally-cue timeline forward from seed_t while consecutive
        cues are within gap_s; return the last chained cue time (or seed_t)."""
        last = seed_t
        i = bisect.bisect_right(self.rally_cues, seed_t)
        while i < len(self.rally_cues) and self.rally_cues[i] <= cap_t:
            if self.rally_cues[i] - last <= gap_s:
                last = self.rally_cues[i]
                i += 1
            else:
                break
        return last


# =====================================================================
# Point-end estimation
# =====================================================================
def find_point_end(match: MatchTelemetry, kin: PlayerKinematics,
                   serve_t: float, t_next: float,
                   cfg: SegmenterConfig,
                   replay: Optional[List[ReplayFrame]] = None
                   ) -> Tuple[float, str]:
    """
    Estimate when the point that started at serve_t ended, searching only
    inside (serve_t, t_next).  Returns (end_t, method).
    """
    cap = min(t_next - cfg.next_serve_guard_s, serve_t + cfg.max_point_s,
              match.duration)
    cap = max(cap, serve_t + cfg.min_point_s)

    if replay is None:
        replay = replay_ball_tracker(match, serve_t - 0.3, cap, cfg)
    intervals = alive_intervals(replay, cfg.alive_merge_gap_s)

    # ---- 1. Trace chain anchored at the serve ----
    chain_end: Optional[float] = None
    start_iv = next((iv for iv in intervals
                     if serve_t - 0.5 <= iv[0] <= serve_t + cfg.serve_chain_window_s),
                    None)
    if start_iv is not None:
        chain_end = max(start_iv[1], serve_t)
        for iv in intervals:
            if iv[0] <= chain_end:
                chain_end = max(chain_end, iv[1])
                continue
            gap = iv[0] - chain_end
            if gap <= cfg.chain_gap_s:
                chain_end = iv[1]
            elif (gap <= cfg.chain_gap_active_s and
                  kin.rally_like(chain_end, iv[0])):
                chain_end = iv[1]
            else:
                break

    # ---- 2. Player-activity chaining ----
    if chain_end is not None:
        # Extend a possibly-truncated trace chain (far-side dropouts) with
        # rally-like player activity, bounded so retrieval jogs can't run away.
        act_end = kin.chain_activity(chain_end, cap, cfg.activity_gap_s)
        act_end = min(act_end, chain_end + cfg.activity_extend_max_s)
        end = max(chain_end, act_end)
        method = "trace" if act_end <= chain_end else "trace+activity"
    else:
        # No usable serve trace (weak far-side ball tracking): player
        # kinematics are the sole authority.
        act_end = kin.chain_activity(serve_t, cap, cfg.activity_gap_s)
        if act_end > serve_t:
            end, method = act_end, "activity"
        else:
            end, method = serve_t + cfg.fallback_point_s, "fallback"

    end = min(max(end, serve_t + cfg.min_point_s), cap)
    return end, method


# =====================================================================
# Top-level segmentation
# =====================================================================
@dataclass
class PointSegment:
    point: int
    side: str
    serve_t: float
    end_t: float
    start: float             # serve_t - pre_roll (clamped)
    end: float               # end_t + end_pad (clamped)
    end_method: str
    trace_confirmed: bool
    score: float


def segment_match(match: MatchTelemetry,
                  cfg: Optional[SegmenterConfig] = None,
                  verbose: bool = True) -> List[PointSegment]:
    cfg = cfg or SegmenterConfig()
    if match.meta:
        cfg.court_length_ft = float(match.meta.get("court_length_ft",
                                                   cfg.court_length_ft))
        size = match.meta.get("analysis_size")
        if size:
            cfg.frame_height_px = float(size[1])

    near = detect_serve_events(match, "near", cfg)
    far  = detect_serve_events(match, "far",  cfg)
    if verbose:
        print(f"[SEG] Serve events: {len(near)} near, {len(far)} far")

    events = dedupe_serve_events(near + far, cfg)
    if verbose and len(events) != len(near) + len(far):
        print(f"[SEG] After dedupe: {len(events)} serve event(s)")

    # Serve-trace confirmation per event.
    for evt in events:
        rep = replay_ball_tracker(match, evt.t - 0.3,
                                  evt.t + cfg.confirm_window_s, cfg)
        evt.trace_confirmed = confirm_serve_trace(rep, cfg)

    events = hmm_filter_events(events, cfg, verbose=verbose)
    if not events:
        if verbose:
            print("[SEG] No serve events survived — no segments.")
        return []

    kin = PlayerKinematics(match, cfg)

    segments: List[PointSegment] = []
    for i, evt in enumerate(events):
        t_next = events[i + 1].t if i + 1 < len(events) else match.duration
        end_t, method = find_point_end(match, kin, evt.t, t_next, cfg)

        pre = cfg.far_pre_roll_s if evt.side == "far" else cfg.pre_roll_s
        seg = PointSegment(
            point=i + 1, side=evt.side, serve_t=evt.t, end_t=end_t,
            start=max(0.0, evt.t - pre),
            end=min(match.duration, end_t + cfg.end_pad_s),
            end_method=method,
            trace_confirmed=evt.trace_confirmed,
            score=evt.score,
        )
        segments.append(seg)
        if verbose:
            print(f"[SEG] Point {seg.point:3d}  {seg.side:4s}  "
                  f"serve {seg.serve_t:8.2f}s  end {seg.end_t:8.2f}s  "
                  f"({seg.end_t - seg.serve_t:5.1f}s, {method}"
                  f"{'' if seg.trace_confirmed else ', unconfirmed'})")

    if verbose and segments:
        kept = sum(s.end - s.start for s in segments)
        total = match.duration
        print(f"[SEG] {len(segments)} points; keeping {kept:.0f}s of {total:.0f}s "
              f"({100 * (1 - kept / max(total, 1e-9)):.0f}% dead time removed)")
    return segments


def write_segments_csv(segments: List[PointSegment], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["point", "side", "serve_t", "end_t", "start", "end",
                    "duration", "end_method", "trace_confirmed", "score"])
        for s in segments:
            w.writerow([s.point, s.side, f"{s.serve_t:.2f}", f"{s.end_t:.2f}",
                        f"{s.start:.2f}", f"{s.end:.2f}",
                        f"{s.end - s.start:.2f}", s.end_method,
                        int(s.trace_confirmed), f"{s.score:.3f}"])
    print(f"[SEG] Wrote segment report → {path}")


def write_segments_json(segments: List[PointSegment], path: str) -> None:
    data = [{
        "point": s.point, "side": s.side,
        "serve_t": round(s.serve_t, 2), "end_t": round(s.end_t, 2),
        "start": round(s.start, 2), "end": round(s.end, 2),
        "end_method": s.end_method, "trace_confirmed": s.trace_confirmed,
        "score": round(s.score, 3),
    } for s in segments]
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[SEG] Wrote segments JSON → {path}")


# =====================================================================
# Synthetic self-test — no models, no video (ball_tracker.py style).
#   python -m pipeline.point_segmenter --self-test
# =====================================================================
def _mk_rec(f, t, near_wy=None, far_wy=None, near_wx=13.5, far_wx=13.5,
            balls=(), toss=(), trophy=0.0, stgcn=0.0):
    near_box = (430, 350, 490, 500) if near_wy is not None else None
    far_box  = (450, 120, 480, 175) if far_wy  is not None else None
    return FrameRecord(
        f=f, t=t,
        near_box=near_box,
        near_world=(near_wx, near_wy) if near_wy is not None else None,
        far_box=far_box, far_held=False,
        far_world=(far_wx, far_wy) if far_wy is not None else None,
        balls=list(balls), toss=list(toss), trophy=trophy, stgcn=stgcn,
    )


def _run_self_test() -> int:
    fps = 30.0
    dt = 1.0 / fps
    failures: List[str] = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    def build_match(recs):
        return MatchTelemetry({"fps": fps, "stride": 1,
                               "court_length_ft": 78.0,
                               "analysis_size": [960, 540]}, recs)

    cfg = SegmenterConfig()

    # ---- Near serve detection: dwell + rising toss above the head ----
    recs = []
    f = 0
    t = 0.0
    for _ in range(45):                       # 1.5 s settled in the ready band
        recs.append(_mk_rec(f, t, near_wy=-1.5)); f += 1; t += dt
    toss_y = 345.0
    for _ in range(6):                        # toss climbing above head (box top 350)
        toss_y -= 9.0
        recs.append(_mk_rec(f, t, near_wy=-1.5,
                            toss=[(455.0, toss_y, 0.9)])); f += 1; t += dt
    for _ in range(60):
        recs.append(_mk_rec(f, t, near_wy=-1.5)); f += 1; t += dt
    m = build_match(recs)
    near_events = detect_serve_events(m, "near", cfg)
    check("near serve detected from toss", len(near_events) == 1)
    check("near serve time ≈ toss moment",
          near_events and 1.4 <= near_events[0].t <= 2.0)

    # Toss below the head must NOT fire.
    recs2 = []
    f = t = 0
    t = 0.0
    for _ in range(45):
        recs2.append(_mk_rec(f, t, near_wy=-1.5)); f += 1; t += dt
    low_y = 480.0
    for _ in range(6):
        low_y -= 5.0                          # rising but still inside/below the box
        recs2.append(_mk_rec(f, t, near_wy=-1.5,
                             toss=[(455.0, low_y, 0.9)])); f += 1; t += dt
    check("below-head toss never fires",
          len(detect_serve_events(build_match(recs2), "near", cfg)) == 0)

    # ---- Far serve detection: dwell + ST-GCN score ----
    recs3 = []
    f = 0
    t = 0.0
    for _ in range(45):
        recs3.append(_mk_rec(f, t, far_wy=80.0)); f += 1; t += dt
    for _ in range(10):
        recs3.append(_mk_rec(f, t, far_wy=80.0, stgcn=0.9)); f += 1; t += dt
    far_events = detect_serve_events(build_match(recs3), "far", cfg)
    check("far serve detected from ST-GCN score", len(far_events) == 1)

    # Out-of-band far player never fires even with a high score.
    recs4 = [_mk_rec(i, i * dt, far_wy=55.0, stgcn=0.95) for i in range(90)]
    check("far serve requires the ready band",
          len(detect_serve_events(build_match(recs4), "far", cfg)) == 0)

    # ---- Dedupe: near event wins a conflict with a far event ----
    ded = dedupe_serve_events(
        [ServeEvent(10.0, "far", 0.9), ServeEvent(13.0, "near", 0.8),
         ServeEvent(40.0, "far", 0.9)], cfg)
    check("dedupe keeps 2 of 3 conflicting events", len(ded) == 2)
    check("dedupe prefers the near event", ded[0].side == "near")

    # ---- Viterbi + HMM filter ----
    dec = _viterbi(["near", "near", "far", "near", "near"], 0.9355, 0.85)
    check("viterbi smooths a lone disagreeing side",
          dec == ["near"] * 5)
    evs = [ServeEvent(10.0 + 20 * i, s, 0.8) for i, s in
           enumerate(["near", "near", "far", "near", "near"])]
    for e in evs:
        e.trace_confirmed = e.side == "near"
    kept = hmm_filter_events(evs, cfg, verbose=False)
    check("HMM drops the weak disagreeing serve", len(kept) == 4)
    evs[2].trace_confirmed = True
    kept = hmm_filter_events(evs, cfg, verbose=False)
    check("HMM keeps a confirmed disagreeing serve", len(kept) == 5)

    # ---- Point end from a trace chain ----
    # Serve at t=1: ball flies for 3 s, 1 s gap (re-acquired), 2 s more, then
    # nothing until the next serve at t=30.
    recs5 = []
    f = 0
    t = 0.0
    bx, by = 300.0, 430.0
    for i in range(int(30.0 * fps)):
        balls = []
        if 1.0 <= t < 4.0:
            bx += 16.0 * math.cos(t * 2); by = 430.0 - 90.0 * abs(math.sin(t * 3))
            bx = max(80.0, min(880.0, bx))
            balls = [(bx, by, 0.8)]
        elif 5.0 <= t < 7.0:
            bx += 14.0 * math.cos(t * 2 + 1); by = 430.0 - 90.0 * abs(math.sin(t * 3))
            bx = max(80.0, min(880.0, bx))
            balls = [(bx, by, 0.8)]
        recs5.append(_mk_rec(f, t, near_wy=-1.5, far_wy=80.0, balls=balls))
        f += 1; t += dt
    m5 = build_match(recs5)
    kin5 = PlayerKinematics(m5, cfg)
    end, method = find_point_end(m5, kin5, 1.0, 30.0, cfg)
    check("trace-chain point end lands after the last rally motion",
          6.0 <= end <= 8.5)
    check("trace-chain method reported", method.startswith("trace"))

    # ---- Point end from player activity only (no ball trace) ----
    # Both players oscillate laterally (rally footwork) until t=8, then stand.
    recs6 = []
    f = 0
    t = 0.0
    for i in range(int(30.0 * fps)):
        if t < 8.0:
            nx = 13.5 + 6.0 * math.sin(t * 2.2)
            fx = 13.5 + 6.0 * math.sin(t * 1.8 + 1.0)
        else:
            nx = fx = 13.5
        recs6.append(_mk_rec(f, t, near_wy=-1.5, far_wy=80.0,
                             near_wx=nx, far_wx=fx))
        f += 1; t += dt
    m6 = build_match(recs6)
    kin6 = PlayerKinematics(m6, cfg)
    end6, method6 = find_point_end(m6, kin6, 1.0, 30.0, cfg)
    check("activity-only point end tracks the rally footwork",
          6.5 <= end6 <= 10.5)
    check("activity method reported", method6 == "activity")

    # ---- Fallback when there is no evidence at all ----
    recs7 = [_mk_rec(i, i * dt, near_wy=-1.5, far_wy=80.0)
             for i in range(int(20.0 * fps))]
    m7 = build_match(recs7)
    kin7 = PlayerKinematics(m7, cfg)
    end7, method7 = find_point_end(m7, kin7, 1.0, 18.0, cfg)
    check("no-evidence fallback uses the default point length",
          abs(end7 - (1.0 + cfg.fallback_point_s)) < 0.2)
    check("fallback method reported", method7 == "fallback")

    # ---- End-to-end segment_match on a 2-point synthetic mini-match ----
    recs8 = []
    f = 0
    t = 0.0
    toss_y = 345.0
    serve1, serve2 = 2.0, 30.0
    while t < 55.0:
        kw = dict(near_wy=-1.5, far_wy=80.0)
        # Point 1: near serve at 2 s (toss), rally balls 2–7 s
        if serve1 - 0.2 <= t < serve1:
            toss_y -= 9.0
            kw["toss"] = [(455.0, toss_y, 0.9)]
        if serve1 <= t < serve1 + 5.0:
            u = t - serve1
            kw["balls"] = [(max(80.0, min(880.0, 300.0 + 250.0 * math.sin(u * 1.5))),
                            430.0 - 100.0 * abs(math.sin(u * 3.0)) - 40.0 * u, 0.8)]
        # Point 2: far serve at 30 s (ST-GCN), rally balls 30–36 s
        if serve2 - 1.0 <= t < serve2 + 0.3:
            kw["stgcn"] = 0.9
        if serve2 <= t < serve2 + 6.0:
            u = t - serve2
            kw["balls"] = [(max(80.0, min(880.0, 300.0 + 250.0 * math.sin(u * 1.5))),
                            200.0 + 100.0 * abs(math.sin(u * 2.5)) + 20.0 * u, 0.8)]
        recs8.append(_mk_rec(f, t, **kw))
        f += 1; t += dt
    m8 = build_match(recs8)
    segs = segment_match(m8, SegmenterConfig(), verbose=False)
    check("mini-match yields 2 segments", len(segs) == 2)
    if len(segs) == 2:
        check("point 1 is a near serve at ~2 s",
              segs[0].side == "near" and abs(segs[0].serve_t - 2.0) < 0.6)
        check("point 2 is a far serve at ~30 s",
              segs[1].side == "far" and abs(segs[1].serve_t - 30.0) < 1.5)
        check("point 1 end covers the rally", 6.0 <= segs[0].end_t <= 10.0)
        check("point 2 end covers the rally", 34.5 <= segs[1].end_t <= 40.0)
        check("segments do not overlap", segs[0].end < segs[1].start)

    print()
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("SELF-TEST PASSED: all checks green.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2 of the dead-time cutter: telemetry → point segments")
    parser.add_argument("telemetry", nargs="?", default=None,
                        help="Match telemetry JSONL from match_telemetry.py")
    parser.add_argument("--csv",  default=None, help="Write a segment report CSV")
    parser.add_argument("--json", default=None, help="Write segments JSON")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the synthetic self-test (no files needed)")
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(_run_self_test())
    if not args.telemetry:
        parser.error("telemetry file required (or --self-test)")

    match = load_telemetry(args.telemetry)
    segments = segment_match(match)
    base = os.path.splitext(args.telemetry)[0]
    write_segments_csv(segments,  args.csv  or base + "_segments.csv")
    write_segments_json(segments, args.json or base + "_segments.json")
