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
     Near side: ready-band dwell behind the near baseline, then a
       toss+trophy weighted score.  The toss half is a Kalman-filtered
       tracker (_TossTracker) testing the physical signature directly — a
       ball moving up, predominantly vertically, above the player's head —
       rather than counting consecutive raw frame-to-frame rises.
     Far side:  BALL TRACE ONLY.  The pose/toss scoring path (ST-GCN blend)
       proved spurious on real footage, so far serves are detected directly
       from the trace signature no other tennis event reproduces: a fresh
       ball trace that (a) begins in the far region of the frame, (b) shows
       serve-like downward + horizontal motion (perspective-scaled) within
       its first far_trace_head_s, (c) follows far_trace_quiet_s with no
       ball activity (mid-rally far-side shots always have recent trace;
       serves follow dead time), and (d) has the far player tracked nearby
       (presence only — far world DISTANCE is too unreliable to band on).
       Recorded ST-GCN / ftoss scores are ignored.

2. SERVE VALIDATION
     Near candidates are checked for a serve-like ball trace (downward +
     horizontal motion shortly after the event, perspective-scaled) by
     replaying the recorded ball detections — including the native-res far
     crop (fballs) — through the IMM tracker.  Far candidates are born from
     the trace itself, so they are confirmed by construction.
     Candidates are then deduped: within min_serve_separation_s a
     trace-confirmed event beats an unconfirmed one — this is what recovers
     the real serve after an aborted toss (server catches the ball: rising
     toss fires a candidate, but the near-vertical drop of a caught ball
     never confirms, while the real serve seconds later does).  Ties fall
     back to near-beats-far, then earlier-wins.
     A Viterbi-decoded serving-side HMM (serves are sticky — the same player
     serves a whole game) then drops unconfirmed events whose side disagrees
     with the inferred pattern.

3. POINT ENDS
     Offline advantage: point i's end must lie in (serve_i, serve_{i+1}), a
     bounded search window.  Within it we fuse:
       • Ball-trace chain — maximal "genuinely moving" trace intervals from
         the tracker replay, chained across gaps; larger gaps are bridged
         only when player kinematics look rally-like.
       • Player kinematics — NEAR player only (far-side tracking is too
         unreliable to contribute cues).  Direction reversals are the rally
         signature; steady walking (ball retrieval) is not.  They extend a
         trace chain that died early (weak far-side ball tracking) and are
         the sole authority when no usable trace exists at all.
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
from filterpy.kalman import KalmanFilter

from .ball_tracker import BallTrackManager, make_image_row_perspective


# =====================================================================
# Configuration — every tunable knob for segmentation lives here.
# =====================================================================
@dataclass
class SegmenterConfig:
    # ---- Court constants (overridden from telemetry meta) ----
    court_length_ft: float = 78.0
    frame_height_px: float = 540.0

    # ---- Ready-band gating (near side only — the far detector gates on
    # far-player PRESENCE, not position: homography-projected far distances
    # proved too unreliable to band on, see far_presence_slack_s) ----
    near_band_ft: Tuple[float, float] = (-0.5, 3.5)
    ready_dwell_s:    float = 0.2       # lowered from 0.4: a brief in-band window
                                        # still needs to arm before the serve
                                        # itself, or the whole candidate is lost.
    band_window_s:    float = 2.0
    band_out_ratio:   float = 0.25

    # ---- Static-candidate suppression ----
    # A static ball-like object (light, sign, ball on the ground) can win the
    # best-confidence slot for thousands of frames and starve the toss
    # trackers.  Cells of the analysis frame where toss/ftoss/fballs
    # candidates appear in more than static_frac of ALL frames are noise: no
    # real ball hovers in one spot for minutes.
    static_cell_px: int   = 16
    static_frac:    float = 0.04

    # ---- Near serve scoring: Kalman-filtered toss (see _TossTracker) ----
    # The toss test is literally "a ball moving up, predominantly vertically,
    # above the player's head" — a constant-velocity KF over [x,y,vx,vy]
    # evaluates that directly on the FILTERED state, instead of the old
    # frame-to-frame pairwise y-comparison (one pixel of jitter read as
    # "stopped rising"; picking the frame's highest-confidence candidate let
    # an unrelated blob elsewhere in the toss ROI hijack the track).
    toss_conf_floor:      float = 0.5    # raw-candidate confidence floor,
                                         # applied before seeding/association
    toss_confirm_frames:  int   = 2      # consecutive REAL-DETECTION frames
                                         # (not coasting/predict-only ticks —
                                         # see had_detection in _score) that
                                         # must satisfy the rise test
    toss_min_rise_duration_s: float = 0.0  # optional extra wall-clock floor
                                         # on top of toss_confirm_frames.
                                         # Left at 0 (off): the had_detection
                                         # gate already does the real work —
                                         # a genuine dense toss can clear 2
                                         # consecutive real frames in ~17ms at
                                         # 60fps, just as fast as a spurious
                                         # rally motion can, so wall-clock
                                         # duration alone doesn't separate
                                         # them.  Requiring the streak to be
                                         # built from real detections (not
                                         # velocity coasted forward through
                                         # gaps) does: two genuine detections
                                         # 0.1s apart bridged by coasting
                                         # frames no longer inflates into a
                                         # false confirm (folder 21, t=266.7).
    toss_seed_gate_px:    float = 60.0   # two raw candidates within this and
    toss_seed_max_dt_s:   float = 0.25   # ... this much time seed a velocity
                                         # estimate (need 2 points to start)
    toss_assoc_gate_px:   float = 45.0   # radius around the filter's
                                         # PREDICTED position that a candidate
                                         # must fall inside to update it — a
                                         # confident detection outside the
                                         # gate can no longer hijack the track
    toss_coast_max_s:     float = 0.20   # predict-only grace period with no
                                         # associated detection before the
                                         # track is declared dead and must
                                         # reseed.  A constant-velocity
                                         # predict leaves vx/vy unchanged, so
                                         # a short gap doesn't erase the
                                         # rising trend the way the old
                                         # 1-frame tolerance did.
    toss_min_vy_px_s:     float = 80.0   # filtered upward speed required to
                                         # count as "rising" (not just wobble)
    toss_max_horiz_ratio: float = 1.2    # filtered |vx| may be at most this
                                         # multiple of |vy| — "predominantly
                                         # vertical", rejects lateral motion
    toss_kf_q_pos:        float = 4.0    # KF process noise: position
    toss_kf_q_vel:        float = 800.0  # KF process noise: velocity (a
                                         # toss decelerates under gravity; a
                                         # constant-velocity model needs live
                                         # slack to track that — same role as
                                         # ball_tracker.py's q_smooth)
    toss_kf_r_px:         float = 16.0   # KF measurement noise (detector jitter)
    toss_band_grace_s:    float = 0.35   # bridge a brief ready-band exit: toss
                                         # scoring (and the ratio-disarm) stay
                                         # live for this long past the moment
                                         # in_band goes False, clocked from
                                         # WALL TIME since the exit — not from
                                         # whether a Kalman track happens to
                                         # exist, which a real mid-rally motion
                                         # could otherwise exploit to inherit
                                         # an old track's momentum indefinitely
                                         # (see detect_serve_events near-loop)
    trophy_weight: float = 0.2
    toss_weight:   float = 0.8
    serve_score_threshold: float = 0.55
    serve_event_window_s:  float = 1.2

    # ---- Far serve detection (ball-trace only) ----
    # The ST-GCN/far-toss scoring path was spurious on real footage and was
    # removed: far serves fire only from the serve-signature ball trace
    # (see far_serve_trace_onsets).  The near-serve trace thresholds
    # (trace_downward_px_s / trace_horizontal_px_s, perspective-scaled)
    # define the motion part of the signature.
    # The far-region cutoff is derived from the OBSERVED far-player feet
    # (median feet-y + far_origin_pad_px) whenever enough far boxes exist —
    # a fixed frame fraction is not portable across cameras (folder 23's far
    # baseline sits at y≈0.5*h, folder 68's at y≈0.42*h, where a 0.6*h cutoff
    # swallowed near-serve ball flights as "far origin").
    far_origin_pad_px:    float = 45.0  # cutoff = far feet median + this;
                                        # folder-23 GT traces start up to
                                        # ~20 px below the feet median
    far_feet_min_samples: int   = 300   # below this, fall back to the frame
                                        # fraction
    far_trace_origin_frac: float = 0.6  # FALLBACK cutoff fraction of frame
                                        # height when far tracking is too
                                        # sparse to calibrate (tuned on
                                        # folder 23; 14/15 recall)
    far_trace_head_s:  float = 2.5      # signature must appear this early in
                                        # the trace (sparse far tracking can
                                        # take ~2 s to grow the stretch); a
                                        # near serve's ball needs longer than
                                        # this to come back down from the far
                                        # side, so rally returns don't qualify
    far_trace_quiet_s: float = 4.0      # no genuine trace this long before
                                        # the onset — mid-rally far shots
                                        # always have recent ball activity,
                                        # serves follow dead time
    far_trace_min_interval_s: float = 0.25  # intervals shorter than this are
                                        # tracker micro-blips: they don't
                                        # reset the quiet clock (a 0.07 s
                                        # flicker 0.7 s before a real
                                        # folder-23 serve was blocking it)
    far_presence_slack_s: float = 1.0   # far player must be tracked within
                                        # this of the onset.  Presence only:
                                        # the homography-projected far
                                        # DISTANCE is too unreliable to gate
                                        # on (folder 23: serve-time distance
                                        # flip-flopped between regimes ±20 ft
                                        # faster than any self-cal band could
                                        # track, rejecting 6 real serves)

    # ---- Serve-event bookkeeping ----
    min_serve_separation_s: float = 8.0  # dedupe window: two serves can't be this close
    serve_rearm_s:          float = 2.0  # detector re-arm after firing; short so an
                                         # aborted toss doesn't mask the real serve

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
    inbox_accept_px: float = 35.0   # accept detections inside a player box
                                    # when within this (perspective-scaled) of
                                    # the track's last position — keeps the
                                    # contact-moment samples the blanket
                                    # in-box exclusion deletes.  Neutral on
                                    # interval-level metrics (folder 23 A/B:
                                    # coverage/frags/lag unchanged, stray +1 s);
                                    # exists for contact anchoring in the
                                    # trace fitter / speed estimation.

    # ---- Carried-ball suppression (port of rally_detector coupling test) ----
    coupling_window_s:        float = 0.40
    coupling_min_player_speed: float = 25.0   # px/s
    coupling_ratio_max:        float = 0.50

    # ---- Point-end chaining ----
    serve_chain_window_s: float = 5.0   # first trace interval must start this soon after serve
    # chain_gap_s / chain_gap_active_s relaxed from 2.5/6.0: points were being
    # cut short — trace dropouts (especially far-side) opened gaps the old
    # thresholds refused to bridge.
    chain_gap_s:          float = 4.0   # always bridge trace gaps up to this
    chain_gap_active_s:   float = 8.0   # bridge up to this when players look rally-like
    activity_gap_s:       float = 2.5   # gap allowed between rally cues when chaining activity
    activity_extend_max_s: float = 12.0 # cap on extending a trace chain via activity alone
    fallback_point_s:     float = 6.0   # assumed length when no evidence at all (ace/short point)
    max_point_s:          float = 60.0
    min_point_s:          float = 1.5
    next_serve_guard_s:   float = 1.5

    # ---- Player-kinematics rally cues (NEAR player only — far tracking is
    # too unreliable to contribute; the old "both players moving" cue died
    # with it, since it needed a far speed) ----
    speed_window_s:       float = 0.4
    speed_min_dt_s:       float = 0.15
    reversal_speed_ft_s:  float = 3.0   # |vx| needed to count as a significant direction

    # ---- Output segments ----
    pre_roll_s:     float = 2.0    # before a near serve event (captures the motion start)
    far_pre_roll_s: float = 4.5    # trace onset = ball contact + tracker warmup,
                                   # up to ~4.2 s after the labeled serve-motion
                                   # start on folder-23 ground truth
    end_pad_s:      float = 2.0    # raised from 1.0: ends were landing ~0.5–2 s
                                   # before the labeled rally end (folder 23)


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
    ftoss:  List[Tuple[float, float, float]] = field(default_factory=list)
    fballs: List[Tuple[float, float, float]] = field(default_factory=list)
    rballs: List[Tuple[float, float, float]] = field(default_factory=list)
    # rballs: tracker-guided native-res re-detections added offline by
    # trace_enrich.py — optional channel, absent from raw stage-1 output.


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
                ftoss=[tuple(b) for b in obj.get("ftoss", [])],
                fballs=[tuple(b) for b in obj.get("fballs", [])],
                rballs=[tuple(b) for b in obj.get("rballs", [])],
            ))
    return MatchTelemetry(meta, records)


def suppress_static_candidates(match: MatchTelemetry,
                               cfg: SegmenterConfig) -> int:
    """
    Remove toss/ftoss/fballs candidates that sit in 'hot' cells — grid cells
    of the analysis frame where a candidate appears in more than static_frac
    of all frames.  Real balls move; a candidate that lives in one 16px cell
    for minutes is a static false positive that starves the toss trackers
    (it wins the best-confidence slot frame after frame).

    Mutates the records in place (idempotent per loaded telemetry) and
    returns the number of candidates dropped.
    """
    if getattr(match, "_static_suppressed", False):
        return 0
    n = len(match.records)
    if n == 0:
        return 0
    cell = max(4, int(cfg.static_cell_px))
    counts: dict = {}
    for rec in match.records:
        seen = set()
        for cand_list in (rec.toss, rec.ftoss, rec.fballs, rec.rballs):
            for x, y, _ in cand_list:
                seen.add((int(x) // cell, int(y) // cell))
        for key in seen:                      # count frames, not detections
            counts[key] = counts.get(key, 0) + 1
    hot = {k for k, c in counts.items() if c / n > cfg.static_frac}
    if not hot:
        match._static_suppressed = True
        return 0
    dropped = 0
    for rec in match.records:
        for attr in ("toss", "ftoss", "fballs", "rballs"):
            cands = getattr(rec, attr)
            kept = [c for c in cands
                    if (int(c[0]) // cell, int(c[1]) // cell) not in hot]
            dropped += len(cands) - len(kept)
            setattr(rec, attr, kept)
    match._static_suppressed = True
    return dropped


# =====================================================================
# Serve events
# =====================================================================
@dataclass
class ServeEvent:
    t: float
    side: str            # "near" | "far"
    score: float
    trace_confirmed: bool = False
    toss_seen: bool = False      # a toss track peaked above the server's head

    @property
    def supported(self) -> bool:
        """Independent evidence beyond the serve score: a serve-like ball
        trace.  Far events are trace-born, so they are always supported."""
        return self.trace_confirmed


class _TossTracker:
    """
    Kalman-filtered detector for "a ball moving up, predominantly
    vertically, above the player's head" — the near-serve toss signature.

    Replaces frame-to-frame pairwise y-comparison (fragile: one pixel of
    detection jitter reads as "stopped rising", and picking the frame's
    highest-confidence candidate lets an unrelated blob elsewhere in the
    toss ROI hijack the track) with state estimation over [x, y, vx, vy]:

      SEED       two raw candidates within toss_seed_gate_px / _max_dt_s of
                 each other start a constant-velocity filter (a velocity
                 estimate needs 2 points).
      ASSOCIATE  each frame, the candidate nearest the filter's PREDICTED
                 position — not the frame's highest raw confidence — updates
                 it.  A confident detection outside the gate is ignored, so
                 it can no longer hijack the track.
      COAST      a frame with no candidate inside the gate is predict-only.
                 A constant-velocity predict leaves vx/vy unchanged, so a
                 short detection gap doesn't erase the rising trend the way
                 the old 1-frame gap tolerance did.  The track only dies
                 after toss_coast_max_s with no association.
      CONFIRM    the physical test runs on the FILTERED velocity directly:
                 vy <= -toss_min_vy_px_s (rising at a meaningful rate) and
                 |vx| <= toss_max_horiz_ratio * |vy| (predominantly
                 vertical), while the filtered position sits above head_y —
                 sustained for toss_confirm_frames CONSECUTIVE REAL
                 detections.  Real detections specifically, not coasting
                 (predict-only) ticks: a coasting frame reuses the last
                 velocity estimate unchanged, so counting it toward the
                 streak lets 2 genuine detections a fraction of a second
                 apart, bridged by several coast frames, "confirm" for free
                 with no new evidence — that inflated a fast rally
                 racket-swing into a false near serve (folder 21, t=266.7).
    """

    def __init__(self, cfg: SegmenterConfig, conf_floor: float):
        self.cfg = cfg
        self.conf_floor = conf_floor
        self.reset()

    def reset(self) -> None:
        self.kf: Optional[KalmanFilter] = None
        self.last_t: float = 0.0
        self.last_update_t: float = -math.inf
        self.pending: Optional[Tuple[float, float, float]] = None  # (t, x, y)
        self.rising_consecutive: int = 0
        self.rising_since_t: Optional[float] = None
        self.toss_min_y: Optional[float] = None

    def _seed(self, x: float, y: float, vx: float, vy: float) -> None:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        kf.R = np.eye(2, dtype=float) * self.cfg.toss_kf_r_px
        kf.P = np.eye(4, dtype=float) * 100.0
        kf.x = np.array([[x], [y], [vx], [vy]], dtype=float)
        self.kf = kf

    def _predict(self, dt: float) -> None:
        dt = max(dt, 1e-6)
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1,  0],
                              [0, 0, 0,  1]], dtype=float)
        self.kf.Q = np.diag([self.cfg.toss_kf_q_pos, self.cfg.toss_kf_q_pos,
                             self.cfg.toss_kf_q_vel, self.cfg.toss_kf_q_vel]) * dt
        self.kf.predict()

    def update(self, candidates: List[Tuple[float, float, float]],
               head_y: float, now: float) -> float:
        candidates = [c for c in candidates if c[2] >= self.conf_floor]

        if self.kf is None:
            chosen = max(candidates, key=lambda c: c[2]) if candidates else None
            just_seeded = False
            if chosen is not None:
                if self.pending is not None:
                    pt, px, py = self.pending
                    dt = now - pt
                    if (0 < dt <= self.cfg.toss_seed_max_dt_s and
                            math.hypot(chosen[0] - px, chosen[1] - py)
                            <= self.cfg.toss_seed_gate_px):
                        self._seed(chosen[0], chosen[1],
                                  (chosen[0] - px) / dt, (chosen[1] - py) / dt)
                        self.last_t = now
                        self.last_update_t = now
                        self.pending = None
                        just_seeded = True
                    else:
                        self.pending = (now, chosen[0], chosen[1])
                else:
                    self.pending = (now, chosen[0], chosen[1])
            return self._score(head_y, now, had_detection=just_seeded)

        dt = now - self.last_t
        if dt > 0:
            self._predict(dt)
        self.last_t = now

        assoc = None
        if candidates:
            px, py = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
            gated = [c for c in candidates
                    if math.hypot(c[0] - px, c[1] - py)
                    <= self.cfg.toss_assoc_gate_px]
            if gated:
                assoc = max(gated, key=lambda c: c[2])

        if assoc is not None:
            self.kf.update(np.array([[assoc[0]], [assoc[1]]], dtype=float))
            self.last_update_t = now
        elif now - self.last_update_t > self.cfg.toss_coast_max_s:
            self.kf = None
            self.rising_consecutive = 0
            self.rising_since_t = None

        return self._score(head_y, now, had_detection=assoc is not None)

    def _score(self, head_y: float, now: float, had_detection: bool) -> float:
        if self.kf is None:
            self.rising_consecutive = 0
            self.rising_since_t = None
            return 0.0
        y, vx, vy = (float(self.kf.x[1, 0]), float(self.kf.x[2, 0]),
                    float(self.kf.x[3, 0]))
        above_head = y < head_y
        rising = (vy <= -self.cfg.toss_min_vy_px_s and
                 abs(vx) <= self.cfg.toss_max_horiz_ratio * abs(vy))
        if above_head and (self.toss_min_y is None or y < self.toss_min_y):
            self.toss_min_y = y
        # had_detection required: a coasting (predict-only) frame reuses the
        # last real velocity estimate unchanged, so it would otherwise keep
        # "confirming" a rise for free with no new evidence — that's exactly
        # how 2 genuine detections 0.1s apart, bridged by coast frames, once
        # inflated a fast rally racket-swing into a false near serve
        # (folder 21, t=266.7).  Requiring a fresh association every
        # qualifying frame means toss_confirm_frames now counts consecutive
        # REAL observations, not filter ticks.
        if above_head and rising and had_detection:
            if self.rising_consecutive == 0:
                self.rising_since_t = now
            self.rising_consecutive += 1
        else:
            self.rising_consecutive = 0
            self.rising_since_t = None
        duration_ok = (self.rising_since_t is not None and
                      now - self.rising_since_t >= self.cfg.toss_min_rise_duration_s)
        if self.rising_consecutive >= self.cfg.toss_confirm_frames and duration_ok:
            return 1.0
        if self.rising_consecutive >= 1:
            return 0.5
        return 0.0


class _NearServeScorer:
    """Toss (Kalman-filtered, see _TossTracker) + trophy-pose weighted
    scoring, replayed offline."""

    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.toss = _TossTracker(self.cfg, self.cfg.toss_conf_floor)
        self._trophy_scores: Deque[Tuple[float, float]] = deque()
        self._toss_scores:   Deque[Tuple[float, float]] = deque()

    def update(self, rec: FrameRecord, now: float) -> float:
        if rec.near_box is None:
            return 0.0
        if rec.trophy > 0:
            self._trophy_scores.append((rec.trophy, now))
        ts = self.toss.update(rec.toss, rec.near_box[1], now)
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
        return (self.toss.toss_min_y is not None and
                self.toss.toss_min_y < rec.near_box[1])


def far_region_cutoff_y(match: MatchTelemetry, cfg: SegmenterConfig) -> float:
    """Largest image-y a far-serve trace stretch may START at.

    Calibrated from the observed far-player feet (median feet-y +
    far_origin_pad_px) when enough far boxes exist; falls back to
    frame_height * far_trace_origin_frac on sparse far tracking.  A fixed
    frame fraction is not portable: camera framing moves the far baseline
    by ~0.1*frame_height between ground-truth folders.
    """
    feet = sorted(r.far_box[3] for r in match.records if r.far_box is not None)
    if len(feet) >= cfg.far_feet_min_samples:
        return feet[len(feet) // 2] + cfg.far_origin_pad_px
    return cfg.frame_height_px * cfg.far_trace_origin_frac


def _far_serve_stretch(pts: List[Tuple[float, float, float]],
                       cfg: SegmenterConfig, persp,
                       y_origin_max: float) -> bool:
    """confirm_serve_trace's motion test with the far-origin constraint:
    some ~0.3 s stretch of genuine trace points moves downward + horizontally
    (perspective-scaled) AND starts in the far region of the frame."""
    for i in range(1, len(pts)):
        t1, x1, y1 = pts[i]
        j = i - 1
        while j > 0 and t1 - pts[j][0] < 0.3:
            j -= 1
        t0, x0, y0 = pts[j]
        dt = t1 - t0
        if dt < 0.15 or y0 > y_origin_max:
            continue
        scale = persp((y0 + y1) / 2.0)
        if ((y1 - y0) / dt >= cfg.trace_downward_px_s * scale and
                abs(x1 - x0) / dt >= cfg.trace_horizontal_px_s * scale):
            return True
    return False


def far_serve_trace_onsets(match: MatchTelemetry,
                           cfg: SegmenterConfig) -> List[float]:
    """
    Scan a full-match ball-tracker replay for trace onsets bearing the
    far-serve signature — the one ball-trace shape no other tennis event
    reproduces:

      ORIGIN  the qualifying stretch starts in the far region of the frame
              (y <= far_region_cutoff_y — calibrated from observed far-player
              feet; near serves start at the bottom);
      MOTION  net downward + horizontal motion over ~0.3 s, perspective-
              scaled (the ball dropping toward the near court after far
              contact), within the first far_trace_head_s of the trace —
              a near serve's ball needs ~2 s+ to come back down from the
              far side, so rally returns never qualify this early;
      QUIET   no genuine trace for far_trace_quiet_s before the onset —
              mid-rally far-side shots always have recent ball activity,
              serves follow dead time.  Micro-blip intervals (shorter than
              far_trace_min_interval_s) don't reset the quiet clock: they
              are tracker flicker, not play.

    Returns onset times (the first genuine trace frame ≈ ball contact).
    """
    replay = replay_ball_tracker(match, 0.0, match.duration + 1.0, cfg)
    intervals = alive_intervals(replay, cfg.alive_merge_gap_s)
    persp = make_image_row_perspective(cfg.frame_height_px)
    cutoff_y = far_region_cutoff_y(match, cfg)

    onsets: List[float] = []
    prev_end = -math.inf
    for start, end in intervals:
        quiet_ok = start - prev_end >= cfg.far_trace_quiet_s
        if end - start >= cfg.far_trace_min_interval_s:
            prev_end = max(prev_end, end)
        if not quiet_ok:
            continue
        head_end = min(end, start + cfg.far_trace_head_s)
        pts = [(fr.t, fr.position[0], fr.position[1]) for fr in replay
               if fr.genuine and fr.position is not None
               and start <= fr.t <= head_end]
        if _far_serve_stretch(pts, cfg, persp, cutoff_y):
            onsets.append(start)
    return onsets


def _detect_far_serve_events(match: MatchTelemetry, cfg: SegmenterConfig,
                             far_misses: Optional[List[Tuple[float, float, str]]]
                             ) -> List[ServeEvent]:
    """Far serves from the ball trace alone (see far_serve_trace_onsets),
    gated on far-player PRESENCE — a tracked far box within
    far_presence_slack_s of the onset.  Presence only, not position: the
    homography-projected far distance flip-flops between ±20 ft regimes on
    real footage, and a positional band gate rejected 6 of 10 real serve
    onsets on folder-23 ground truth.  Onsets with no far player go to the
    far-miss report."""
    far_ts = [r.t for r in match.records if r.far_box is not None]
    events: List[ServeEvent] = []
    for onset in far_serve_trace_onsets(match, cfg):
        i = bisect.bisect_left(far_ts, onset - cfg.far_presence_slack_s)
        present = (i < len(far_ts) and
                   far_ts[i] <= onset + cfg.far_presence_slack_s)
        if present:
            events.append(ServeEvent(t=onset, side="far", score=1.0,
                                     trace_confirmed=True))
        elif far_misses is not None:
            far_misses.append((onset, 1.0, "no_far_player"))
    return events


def detect_serve_events(match: MatchTelemetry, side: str,
                        cfg: SegmenterConfig,
                        far_misses: Optional[List[Tuple[float, float, str]]] = None
                        ) -> List[ServeEvent]:
    """
    Serve events for one side.

    Far side: ball-trace onsets with the far-serve signature, gated on
    far-player presence (_detect_far_serve_events).

    Near side: ready-band dwell + toss/trophy score, replayed over the
    telemetry.  A serve fires when the player has settled behind the near
    baseline (ready_dwell_s inside the band), the recent out-of-band ratio
    stays low OR a toss is actively forming, and the toss+trophy score
    crosses its threshold.
    """
    if side == "far":
        return _detect_far_serve_events(match, cfg, far_misses)

    scorer = _NearServeScorer(cfg)
    events: List[ServeEvent] = []
    ready_start: Optional[float] = None
    armed = False
    band_hist: Deque[Tuple[float, bool]] = deque()
    cooldown_until = -math.inf
    band_exit_t: Optional[float] = None   # when in_band most recently went False

    for rec in match.records:
        now = rec.t
        if now < cooldown_until:
            continue

        in_band = False
        if rec.near_world is not None:
            dist = -rec.near_world[1]           # behind the near baseline (y=0)
            band = cfg.near_band_ft
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
        # Diagnosed on folder 68 (t=163.9): the near player's WORLD-POSITION
        # estimate (homography-projected feet, separate from the toss-ROI
        # detections that actually drive scoring) flickered out of
        # near_band_ft during a real toss — the ratio-disarm called
        # scorer.reset(), wiping an in-progress Kalman toss track.
        #
        # A brief flicker is bridged by toss_band_grace_s: scoring (and the
        # ratio-disarm) stay live for that long past the moment in_band went
        # False, timed from WALL CLOCK since the exit, not from whether a
        # Kalman track happens to exist.  That distinction matters: gating
        # on "track exists" instead let a seeded-but-unconfirmed track keep
        # itself alive indefinitely by re-associating with ANY trickling
        # candidate every frame (predict-only calls reset its own coast
        # timeout each time) — on folder 21 (t=266.7) that let a real
        # mid-rally racket/arm motion inherit an old, unrelated track's
        # momentum and fire a false near serve.  A bounded, exit-clocked
        # grace window can't be gamed that way: past toss_band_grace_s the
        # gate closes regardless of what the tracker is doing.
        if in_band:
            band_exit_t = None
        elif band_exit_t is None:
            band_exit_t = now
        in_grace = in_band or (now - band_exit_t <= cfg.toss_band_grace_s)

        band_hist.append((now, in_band))
        while band_hist and now - band_hist[0][0] > cfg.band_window_s:
            band_hist.popleft()
        ratio_exceeded = False
        if len(band_hist) > 1:
            total = band_hist[-1][0] - band_hist[0][0]
            if total > 1.0:
                t_out = sum(band_hist[i + 1][0] - band_hist[i][0]
                            for i in range(len(band_hist) - 1)
                            if not band_hist[i][1])
                ratio_exceeded = t_out / total > cfg.band_out_ratio

        if ratio_exceeded and not in_grace:
            armed = False
            ready_start = None
            continue
        if not in_grace:
            continue

        score = scorer.update(rec, now)
        if score >= cfg.serve_score_threshold and scorer.validate(rec):
            events.append(ServeEvent(t=now, side="near", score=score,
                                     toss_seen=True))   # validate() demands it
            armed = False
            ready_start = None
            # Short re-arm only: an aborted toss (caught ball) fires a false
            # candidate, and the REAL serve often follows within seconds —
            # it must also be captured.  The confirmation-aware dedupe picks
            # the right one of the resulting close pair.
            cooldown_until = now + cfg.serve_rearm_s

    return events


def dedupe_serve_events(events: List[ServeEvent],
                        cfg: SegmenterConfig) -> List[ServeEvent]:
    """
    Collapse events closer than min_serve_separation_s.  Two serves cannot
    happen that close together, so one of a conflicting pair is false.
    Priority within a pair:
      1. a trace-confirmed event beats an unconfirmed one — an aborted toss
         (ball caught, falls near-vertically, never confirms) loses to the
         real serve that follows it;
      2. near beats far (the toss-anchored near detector is better tuned);
      3. the earlier event wins (it marks the serve-motion onset).
    Run this AFTER trace confirmation so rule 1 has data.
    """
    events = sorted(events, key=lambda e: e.t)
    kept: List[ServeEvent] = []
    for evt in events:
        if kept and evt.t - kept[-1].t < cfg.min_serve_separation_s:
            prev = kept[-1]
            if evt.supported != prev.supported:
                if evt.supported:
                    kept[-1] = evt             # supported beats unsupported
                continue
            if prev.side != evt.side:
                if evt.side == "near":
                    kept[-1] = evt             # near beats far
                continue
            continue                           # same side/confirmation: earlier wins
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
    pattern AND that lack a confirming trace — weak side-anomalies only.

    This is a dead-time cutter: a trace-confirmed event marks a real point
    START wherever it fired, and the side LABEL on it is not part of the
    product.  So any confirmed event (near or far) is shielded — dropping a
    confirmed far event because the sticky pattern expected "near" would
    throw away a genuine point boundary to fix a label nobody consumes.
    (Earlier this shielded near-only, to suppress folder-21's near-descent
    "far" events — but those are harmless: they coincide with the near serve
    and lose to it in dedupe, or they merge back into the same kept segment.)
    The filter still earns its keep by removing UNCONFIRMED side-anomalies —
    isolated noise events with no trace behind them.
    """
    if len(events) < 2:
        return events
    decoded = _viterbi([e.side for e in events], cfg.hmm_p_stay, cfg.hmm_p_correct)
    kept = []
    for evt, dec in zip(events, decoded):
        if evt.side != dec and not evt.supported:
            if verbose:
                print(f"[HMM] Dropped serve @ {evt.t:.2f}s side={evt.side} "
                      f"(decoded={dec}, unconfirmed)")
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
    bounce_prob: float = 0.0           # IMM court-bounce model weight
    det: Optional[Tuple[float, float]] = None   # raw detection associated
                                       # this frame (None on coasting frames)
                                       # — the fitter fits these, not the
                                       # filtered states


def _replay_core(match: MatchTelemetry, t0: float, t1: float,
                 cfg: SegmenterConfig, collect: bool = False):
    """Shared replay loop; `collect=True` also returns per-record track
    predictions (x, y, time_since_detection) for the enrichment pass."""
    persp = make_image_row_perspective(cfg.frame_height_px)
    mgr = BallTrackManager(fps=match.fps, perspective_scale=persp)
    ball_vel = _SmoothedVelocity(cfg.coupling_window_s)
    player_vel = _SmoothedVelocity(cfg.coupling_window_s)

    out: List[ReplayFrame] = []
    preds: List[Optional[Tuple[float, float, float]]] = []
    for rec in match.slice(t0, t1):
        # Track position from the previous frame — gates the exception that
        # lets a detection INSIDE a player box through (the contact moment;
        # the blanket rule deletes the ball exactly when it is hit).
        tp = mgr.track.position if mgr.track is not None else None
        inbox_accept = 0.0
        if tp is not None and cfg.inbox_accept_px > 0:
            inbox_accept = cfg.inbox_accept_px * max(persp(tp[1]), 0.35)

        dets = []
        for bx, by, conf in (list(rec.balls) + list(rec.fballs)
                             + list(rec.rballs)):
            inside = False
            for box in (rec.near_box, rec.far_box):
                if box and box[0] <= bx <= box[2] and box[1] <= by <= box[3]:
                    inside = True
                    break
            if inside and not (
                    inbox_accept and
                    math.hypot(bx - tp[0], by - tp[1]) <= inbox_accept):
                continue        # racket/arm/body false positive
            # A ball both passes see shows up twice a few px apart — keep one.
            if any(abs(bx - dx) <= 6.0 and abs(by - dy) <= 6.0
                   for dx, dy, _ in dets):
                continue
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
        assoc = None
        if (dets and status.position is not None and
                status.time_since_detection < 0.75 / max(match.fps, 1e-6)):
            assoc = min(((bx, by) for bx, by, _ in dets),
                        key=lambda d: (d[0] - status.position[0]) ** 2 +
                                      (d[1] - status.position[1]) ** 2)
        out.append(ReplayFrame(rec.t, genuine, status.racket_prob, status.position,
                               status.time_since_detection,
                               getattr(status, "bounce_prob", 0.0), assoc))
        if collect:
            preds.append(None if status.position is None
                         else (status.position[0], status.position[1],
                               status.time_since_detection))
    if collect:
        return out, preds
    return out


def replay_ball_tracker(match: MatchTelemetry, t0: float, t1: float,
                        cfg: SegmenterConfig) -> List[ReplayFrame]:
    """
    Re-run the IMM ball tracker over the recorded detections in [t0, t1).

    Detections inside either player's box are excluded (racket/arm/body
    false positives) UNLESS the tracked ball was already predicted there —
    that's the contact moment, exactly where speed measurement starts
    (inbox_accept_px).  The native-res channels are merged in: fballs from
    the stage-1 far crop, rballs from the offline trace_enrich pass.
    """
    return _replay_core(match, t0, t1, cfg, collect=False)


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
    Precomputes, from the NEAR player's world positions (far-side tracking
    is too unreliable to contribute kinematic cues):
      • per-record smoothed speed,
      • direction-reversal times (a significant vx sign flip — the signature
        of rally footwork, absent from ball-retrieval walking).
    Reversals form the rally-cue timeline used to bridge trace gaps and to
    find point ends without a ball trace.
    """

    def __init__(self, match: MatchTelemetry, cfg: SegmenterConfig):
        self.cfg = cfg
        n = len(match.ts)
        self.speed_near = [None] * n
        reversal_times: List[float] = []

        valid: List[Tuple[float, float, float, int]] = []   # (t, wx, wy, idx)
        for i, rec in enumerate(match.records):
            if rec.near_world is not None:
                valid.append((rec.t, rec.near_world[0], rec.near_world[1], i))

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
            self.speed_near[idx] = math.hypot(vx, vy)
            if abs(vx) >= cfg.reversal_speed_ft_s:
                sign = 1 if vx > 0 else -1
                if last_sign != 0 and sign != last_sign:
                    reversal_times.append(t1)
                last_sign = sign

        self.rally_cues: List[float] = reversal_times   # built in time order

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
                  verbose: bool = True,
                  far_misses_out: Optional[List[Tuple[float, float, str]]] = None
                  ) -> List[PointSegment]:
    cfg = cfg or SegmenterConfig()
    if match.meta:
        cfg.court_length_ft = float(match.meta.get("court_length_ft",
                                                   cfg.court_length_ft))
        size = match.meta.get("analysis_size")
        if size:
            cfg.frame_height_px = float(size[1])

    n_static = suppress_static_candidates(match, cfg)
    if verbose and n_static:
        print(f"[SEG] Suppressed {n_static} static ball candidate(s)")

    far_misses: List[Tuple[float, float, str]] = (
        far_misses_out if far_misses_out is not None else [])
    near = detect_serve_events(match, "near", cfg)
    far  = detect_serve_events(match, "far",  cfg, far_misses=far_misses)
    if verbose:
        print(f"[SEG] Serve events: {len(near)} near, {len(far)} far")

    kin = PlayerKinematics(match, cfg)

    # Near-serve trace confirmation — BEFORE dedupe, so a confirmed real
    # serve can displace the unconfirmed aborted-toss candidate that fired
    # just before it.  Far events are trace-born (confirmed by construction).
    candidates = sorted(near + far, key=lambda e: e.t)
    for evt in candidates:
        if evt.side == "near":
            rep = replay_ball_tracker(match, evt.t - 0.3,
                                      evt.t + cfg.confirm_window_s, cfg)
            evt.trace_confirmed = confirm_serve_trace(rep, cfg)

    events = dedupe_serve_events(candidates, cfg)
    if verbose and len(events) != len(candidates):
        print(f"[SEG] After dedupe: {len(events)} serve event(s)")

    events = hmm_filter_events(events, cfg, verbose=verbose)
    if not events:
        if verbose:
            print("[SEG] No serve events survived — no segments.")
        return []

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
            balls=(), toss=(), trophy=0.0, stgcn=0.0, ftoss=(), fballs=()):
    near_box = (430, 350, 490, 500) if near_wy is not None else None
    far_box  = (450, 120, 480, 175) if far_wy  is not None else None
    return FrameRecord(
        f=f, t=t,
        near_box=near_box,
        near_world=(near_wx, near_wy) if near_wy is not None else None,
        far_box=far_box, far_held=False,
        far_world=(far_wx, far_wy) if far_wy is not None else None,
        balls=list(balls), toss=list(toss), trophy=trophy, stgcn=stgcn,
        ftoss=list(ftoss), fballs=list(fballs),
    )


def _run_self_test() -> int:
    fps = 30.0
    dt = 1.0 / fps
    failures: List[str] = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    def build_match(recs, **extra_meta):
        meta = {"fps": fps, "stride": 1,
                "court_length_ft": 78.0,
                "analysis_size": [960, 540]}
        meta.update(extra_meta)
        return MatchTelemetry(meta, recs)

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

    # ---- Toss survives a ready-band flicker mid-rise (folder-68 t=163.9) ----
    # The near player's WORLD-POSITION estimate can misread as out-of-band
    # for a stretch while the toss (a near_box-ROI signal, independent of
    # near_world) keeps rising — a real body shift during the service
    # motion, or plain position-tracking jitter.  The toss must still
    # confirm: scoring must not gate on instantaneous in_band, and the
    # ratio-disarm must not fire (and reset the Kalman toss track) while a
    # toss is actively forming.
    recs_flicker = []
    f, t = 0, 0.0
    for _ in range(45):                              # dwell in-band, arm
        recs_flicker.append(_mk_rec(f, t, near_wy=-1.5)); f += 1; t += dt
    toss_y = 345.0
    for _ in range(8):                                # rising toss, but
        toss_y -= 9.0                                 # world-position reads
        recs_flicker.append(_mk_rec(f, t, near_wy=10.0,   # OUT of band
                                    toss=[(455.0, toss_y, 0.9)])); f += 1; t += dt
    for _ in range(30):
        recs_flicker.append(_mk_rec(f, t, near_wy=-1.5)); f += 1; t += dt
    evs_flicker = detect_serve_events(build_match(recs_flicker), "near", cfg)
    check("toss confirms despite a concurrent ready-band flicker",
          len(evs_flicker) == 1)

    # ---- Far serve detection: ball-trace only ----
    # A far serve = fresh trace onset in the far region with downward +
    # horizontal motion, after a quiet spell, with the far player tracked.
    def far_trace_recs(far_wy=80.0, serve_t=3.0, dur=8.0, y0=140.0,
                       stgcn=0.0, with_trace=True):
        recs, ff, tt = [], 0, 0.0
        while tt < dur:
            kw = dict(near_wy=-1.5, far_wy=far_wy)
            if stgcn and serve_t - 0.3 <= tt < serve_t + 0.3:
                kw["stgcn"] = stgcn
            if with_trace and serve_t <= tt < serve_t + 0.8:
                u = tt - serve_t
                kw["fballs"] = [(480.0 + 90.0 * u, y0 + 120.0 * u, 0.8)]
            recs.append(_mk_rec(ff, tt, **kw)); ff += 1; tt += dt
        return build_match(recs)

    evs_far = detect_serve_events(far_trace_recs(), "far", cfg)
    check("far serve fires from a far-origin serve trace", len(evs_far) == 1)
    check("far event lands at the trace onset",
          bool(evs_far) and 2.8 <= evs_far[0].t <= 3.6)
    check("far event is trace-confirmed by construction",
          bool(evs_far) and evs_far[0].trace_confirmed)

    check("ST-GCN score alone no longer fires",
          len(detect_serve_events(far_trace_recs(with_trace=False, stgcn=0.95),
                                  "far", cfg)) == 0)

    misses_pres: list = []
    check("far serve requires a tracked far player",
          len(detect_serve_events(far_trace_recs(far_wy=None), "far", cfg,
                                  far_misses=misses_pres)) == 0)
    check("presence-rejected trace onset logged for review",
          len(misses_pres) == 1 and misses_pres[0][2] == "no_far_player")

    # Position must NOT matter — a far player tracked at a wildly offset
    # world distance (homography drift) still counts as present.
    check("far serve fires regardless of far world distance",
          len(detect_serve_events(far_trace_recs(far_wy=123.0), "far", cfg)) == 1)

    check("a near-origin trace never fires a far serve",
          len(detect_serve_events(far_trace_recs(y0=420.0), "far", cfg)) == 0)

    # Mid-rally suppression: ball activity shortly before the far-origin
    # trace (a rally in progress) blocks the onset via the quiet gate.
    recs_rally, ff, tt = [], 0, 0.0
    while tt < 8.0:
        kw = dict(near_wy=-1.5, far_wy=80.0)
        if 1.0 <= tt < 2.0:                       # near-region rally trace
            u = tt - 1.0
            kw["balls"] = [(300.0 + 200.0 * u,
                            420.0 - 60.0 * math.sin(u * 6.0), 0.8)]
        if 3.5 <= tt < 4.3:                       # far-origin trace 1.5 s later
            u = tt - 3.5
            kw["fballs"] = [(480.0 + 90.0 * u, 140.0 + 120.0 * u, 0.8)]
        recs_rally.append(_mk_rec(ff, tt, **kw)); ff += 1; tt += dt
    check("recent ball activity (mid-rally) blocks a far-trace onset",
          len(detect_serve_events(build_match(recs_rally), "far", cfg)) == 0)

    # ---- Static-candidate suppression ----
    # A static false ball above the near player's head wins the best-conf
    # slot every frame and starves the toss tracker; suppression removes it
    # so the real rising toss underneath can fire.
    recs_static = []
    f, t = 0, 0.0
    ty = 345.0
    for i in range(120):
        toss = [(700.0, 300.0, 0.95)]          # static object, all frames
        if 45 <= i < 51:
            ty -= 9.0
            toss.append((455.0, ty, 0.9))      # the real toss
        recs_static.append(_mk_rec(f, t, near_wy=-1.5, toss=toss))
        f += 1; t += dt
    m_static = build_match(recs_static)
    check("static toss noise blocks the serve without suppression",
          len(detect_serve_events(m_static, "near", cfg)) == 0)
    n_drop = suppress_static_candidates(m_static, cfg)
    check("suppression removes the static candidates", n_drop >= 100)
    check("serve fires once static noise is suppressed",
          len(detect_serve_events(m_static, "near", cfg)) == 1)

    # ---- Far detection survives mid-video far tracking regime changes ----
    # Folder-23 pattern: the homography-projected far distance steps between
    # regimes (+8 ft, then a dropout, then +45 ft).  Presence gating fires
    # both serves — a positional band gate could not straddle the regimes.
    recs_step = []
    f, t = 0, 0.0
    while t < 60.0:
        kw = dict(near_wy=-1.5, far_wy=86.0)          # dist = +8 ft
        if 20.0 <= t < 20.8:                          # serve trace, regime A
            u = t - 20.0
            kw["fballs"] = [(480.0 + 90.0 * u, 140.0 + 120.0 * u, 0.8)]
        recs_step.append(_mk_rec(f, t, **kw)); f += 1; t += dt
    while t < 65.0:                                    # gap: far player lost
        recs_step.append(_mk_rec(f, t, near_wy=-1.5)); f += 1; t += dt
    while t < 130.0:
        kw = dict(near_wy=-1.5, far_wy=123.0)         # dist = +45 ft
        if 100.0 <= t < 100.8:                        # serve trace, regime B
            u = t - 100.0
            kw["fballs"] = [(480.0 + 90.0 * u, 140.0 + 120.0 * u, 0.8)]
        recs_step.append(_mk_rec(f, t, **kw)); f += 1; t += dt
    m_step = build_match(recs_step)
    evs_step = detect_serve_events(m_step, "far", cfg)
    check("far serves fire across far-distance regime steps",
          len(evs_step) == 2)

    # ---- Far acceptance end-to-end: the trace is the only path ----
    def far_solo_match(with_fballs):
        recs, ff, t = [], 0, 0.0
        while t < 6.0:
            kw = dict(near_wy=-1.5, far_wy=80.0)
            if 1.5 <= t < 1.85:
                kw["stgcn"] = 0.9
            if with_fballs and 1.6 <= t < 2.3:
                u = t - 1.6                               # serve flight, native crop
                kw["fballs"] = [(485.0 + 85.0 * u, 130.0 + 85.0 * u, 0.8)]
            recs.append(_mk_rec(ff, t, **kw))
            ff += 1; t += dt
        return build_match(recs)

    segs_a = segment_match(far_solo_match(False), cfg, verbose=False)
    check("ST-GCN-only far candidate yields no segment", len(segs_a) == 0)

    segs_c = segment_match(far_solo_match(True), cfg, verbose=False)
    check("native-res far trace (fballs) fires the far serve",
          len(segs_c) == 1 and segs_c[0].side == "far")

    # ---- Dedupe: near event wins a conflict with a far event ----
    ded = dedupe_serve_events(
        [ServeEvent(10.0, "far", 0.9), ServeEvent(13.0, "near", 0.8),
         ServeEvent(40.0, "far", 0.9)], cfg)
    check("dedupe keeps 2 of 3 conflicting events", len(ded) == 2)
    check("dedupe prefers the near event", ded[0].side == "near")

    # ---- Dedupe: confirmed real serve displaces the aborted-toss event ----
    ded2 = dedupe_serve_events(
        [ServeEvent(10.0, "near", 0.8, trace_confirmed=False),
         ServeEvent(15.0, "near", 0.8, trace_confirmed=True)], cfg)
    check("dedupe keeps one of the aborted-toss pair", len(ded2) == 1)
    check("dedupe prefers the trace-confirmed serve",
          ded2 and ded2[0].t == 15.0)
    # ... but an early confirmed serve is never displaced by a later one.
    ded3 = dedupe_serve_events(
        [ServeEvent(10.0, "near", 0.8, trace_confirmed=True),
         ServeEvent(13.0, "near", 0.8, trace_confirmed=True)], cfg)
    check("dedupe keeps the earlier of two confirmed events",
          len(ded3) == 1 and ded3[0].t == 10.0)

    # ---- Re-arm: a second toss minutes-not-needed later still fires ----
    # Aborted toss at ~1.5 s, real toss at ~6.5 s: both must be candidates
    # (the old 8 s cooldown swallowed the second one).
    recs_rearm = []
    f, t = 0, 0.0
    for _ in range(45):
        recs_rearm.append(_mk_rec(f, t, near_wy=-1.5)); f += 1; t += dt
    ty = 345.0
    for _ in range(6):
        ty -= 9.0
        recs_rearm.append(_mk_rec(f, t, near_wy=-1.5,
                                  toss=[(455.0, ty, 0.9)])); f += 1; t += dt
    while t < 6.5:
        recs_rearm.append(_mk_rec(f, t, near_wy=-1.5)); f += 1; t += dt
    ty = 345.0
    for _ in range(6):
        ty -= 9.0
        recs_rearm.append(_mk_rec(f, t, near_wy=-1.5,
                                  toss=[(455.0, ty, 0.9)])); f += 1; t += dt
    for _ in range(30):
        recs_rearm.append(_mk_rec(f, t, near_wy=-1.5)); f += 1; t += dt
    evs_rearm = detect_serve_events(build_match(recs_rearm), "near", cfg)
    check("detector re-arms after an aborted toss", len(evs_rearm) == 2)

    # ---- Viterbi + HMM filter ----
    # Deadtime framing: any trace-CONFIRMED event marks a real point start
    # regardless of side, so the HMM only culls UNCONFIRMED side-anomalies.
    dec = _viterbi(["near", "near", "far", "near", "near"], 0.9355, 0.85)
    check("viterbi smooths a lone disagreeing side",
          dec == ["near"] * 5)
    evs = [ServeEvent(10.0 + 20 * i, s, 0.8) for i, s in
           enumerate(["near", "near", "far", "near", "near"])]
    for e in evs:
        e.trace_confirmed = True
    kept = hmm_filter_events(evs, cfg, verbose=False)
    check("HMM keeps a confirmed side-anomalous event (any side)",
          len(kept) == 5)
    evs[2].trace_confirmed = False
    kept = hmm_filter_events(evs, cfg, verbose=False)
    check("HMM drops an unconfirmed side-anomalous event", len(kept) == 4)
    # side pattern is respected: an unconfirmed event AGREEING with the
    # decoded pattern is never dropped.
    evs_ok = [ServeEvent(10.0 + 20 * i, "near", 0.8) for i in range(4)]
    kept_ok = hmm_filter_events(evs_ok, cfg, verbose=False)
    check("HMM keeps unconfirmed events that agree with the pattern",
          len(kept_ok) == 4)

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
    # Two far serves make a far BLOCK — a lone far event between near games
    # is exactly what the HMM side filter now removes (serves are sticky).
    serve1, serve2, serve3 = 2.0, 30.0, 55.0
    while t < 80.0:
        kw = dict(near_wy=-1.5, far_wy=80.0)
        # Point 1: near serve at 2 s (toss), rally balls 2–7 s
        if serve1 - 0.2 <= t < serve1:
            toss_y -= 9.0
            kw["toss"] = [(455.0, toss_y, 0.9)]
        if serve1 <= t < serve1 + 5.0:
            u = t - serve1
            kw["balls"] = [(max(80.0, min(880.0, 300.0 + 250.0 * math.sin(u * 1.5))),
                            430.0 - 100.0 * abs(math.sin(u * 3.0)) - 40.0 * u, 0.8)]
        # Points 2+3: far serves at 30/55 s (far-origin traces)
        for sv in (serve2, serve3):
            if sv <= t < sv + 6.0:
                u = t - sv
                kw["balls"] = [(max(80.0, min(880.0, 300.0 + 250.0 * math.sin(u * 1.5))),
                                150.0 + 100.0 * abs(math.sin(u * 2.5)) + 20.0 * u, 0.8)]
        recs8.append(_mk_rec(f, t, **kw))
        f += 1; t += dt
    m8 = build_match(recs8)
    segs = segment_match(m8, SegmenterConfig(), verbose=False)
    check("mini-match yields 3 segments", len(segs) == 3)
    if len(segs) == 3:
        check("point 1 is a near serve at ~2 s",
              segs[0].side == "near" and abs(segs[0].serve_t - 2.0) < 0.6)
        check("point 2 is a far serve at ~30 s",
              segs[1].side == "far" and abs(segs[1].serve_t - 30.0) < 1.5)
        check("point 3 is a far serve at ~55 s",
              segs[2].side == "far" and abs(segs[2].serve_t - 55.0) < 1.5)
        check("point 1 end covers the rally", 6.0 <= segs[0].end_t <= 10.0)
        check("point 2 end covers the rally", 34.5 <= segs[1].end_t <= 40.0)
        check("segments do not overlap",
              segs[0].end < segs[1].start and segs[1].end < segs[2].start)

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
