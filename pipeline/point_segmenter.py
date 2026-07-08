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
     Far side:  ready-band dwell behind the far baseline, then a weighted
       blend of the native-res far-toss track and the ST-GCN serve
       probability; a sustained-high ST-GCN score alone can also raise a
       candidate (far_stgcn_solo_threshold).  The toss is confirmatory, not
       mandatory.  v1 telemetry (no ftoss) falls back to the raw ST-GCN
       score alone.

2. SERVE VALIDATION
     Each candidate is checked for a serve-like ball trace (downward +
     horizontal motion shortly after the event, perspective-scaled) by
     replaying the recorded ball detections — including the native-res far
     crop (fballs), without which a far serve can never grow a confirmable
     trace — through the IMM tracker.  Far candidates additionally get
     receiver-side corroboration: the near player stands quasi-still, then
     bursts into motion right after a genuine far serve.  A far candidate
     with no support at all (no trace, no toss, no receiver reaction) is
     dropped and logged to the far-miss report.
     Candidates are then deduped: within min_serve_separation_s a supported
     event beats an unsupported one — this is what recovers the real serve
     after an aborted toss (server catches the ball: rising toss fires a
     candidate, but the near-vertical drop of a caught ball never confirms,
     while the real serve seconds later does).  Ties fall back to
     near-beats-far, then earlier-wins.
     A Viterbi-decoded serving-side HMM (serves are sticky — the same player
     serves a whole game) then drops unsupported events whose side disagrees
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
    far_band_ft:  Tuple[float, float] = (-6.0, 6.0)   # fallback when selfcal lacks data
    ready_dwell_s:    float = 0.2       # lowered from 0.4: a brief in-band window
                                        # (noisy far-court tracking) still needs to
                                        # arm before the serve itself, or the whole
                                        # candidate is lost — validated on folder 23
                                        # ground truth. Shared with the near-side
                                        # detector; watch near-side false-positive
                                        # rate if that ever regresses.
    band_window_s:    float = 2.0
    band_out_ratio:   float = 0.25

    # ---- Far-band self-calibration ----
    # far_world distances are homography-amplified and carry per-video offsets
    # of +15..+25 ft (measured on the ground-truth folders), so a fixed far
    # band misses real serves.  The far player spends most of the match near
    # their baseline: the local median recorded distance IS the calibration
    # offset.  WINDOWED, not whole-video: the offset itself can step mid-video
    # (a tracking dropout followed by a different steady-state — observed
    # jumping from ~10 ft to ~45 ft after a 40 s gap in one ground-truth
    # folder), so a single whole-video median band only covers whichever
    # regime dominates the sample count and blinds the segmenter to the rest.
    far_band_selfcal: bool = True
    far_band_halfwidth_ft:  float = 9.0
    far_selfcal_min_frames: int   = 500     # whole-video fallback threshold
    far_selfcal_window_s:   float = 60.0    # rolling recalibration window
    far_selfcal_min_window_frames: int = 300  # per-window fallback threshold:
                                              # high enough that a sparsely-
                                              # tracked window (noise) falls
                                              # back to the whole-video band
                                              # instead of recalibrating onto
                                              # a handful of stray samples

    # ---- Static-candidate suppression ----
    # A static ball-like object (light, sign, ball on the ground) can win the
    # best-confidence slot for thousands of frames and starve the toss
    # trackers.  Cells of the analysis frame where toss/ftoss/fballs
    # candidates appear in more than static_frac of ALL frames are noise: no
    # real ball hovers in one spot for minutes.
    static_cell_px: int   = 16
    static_frac:    float = 0.04

    # ---- Near serve scoring (toss + trophy) ----
    toss_conf_floor:     float = 0.5
    toss_gap_tolerance:  int   = 1
    toss_confirm_frames: int   = 2
    trophy_weight: float = 0.2
    toss_weight:   float = 0.8
    serve_score_threshold: float = 0.55
    serve_event_window_s:  float = 1.2

    # ---- Far serve scoring (toss-primary blend; see _FarServeScorer) ----
    # stgcn_threshold and far_stgcn_solo_threshold both lowered from 0.55/0.70:
    # on real ground truth (folder 23) the native-res far-toss detector rarely
    # fires, so the 65/35 blend chronically discounted decent ST-GCN-only
    # scores (e.g. 0.685 blended down to ~0.24, just under the old 0.70 solo
    # cutoff).  Recovered far_side_recall from 0/15 to 10/15 (precision 0.769)
    # combined with the windowed self-cal band below.
    stgcn_threshold: float = 0.30       # threshold on the blended far score
    far_toss_weight:  float = 0.65
    far_stgcn_weight: float = 0.35
    far_toss_conf_floor: float = 0.3    # native-res far balls score lower conf
    far_stgcn_solo_threshold: float = 0.30  # ST-GCN alone can fire above this
                                            # (toss confirms, it isn't mandatory)
    far_miss_floor: float = 0.30        # log far near-misses scoring above this

    # ---- Receiver-side corroboration of far serves ----
    # A far serve has a loud NEAR-side signature: the receiver stands quasi-
    # still, then bursts into motion within ~2 s of the serve.  Near-side
    # tracking is the strongest sensor in the pipeline, so this corroborates
    # far events that the weak far-side ball trace cannot confirm.
    receiver_still_window_s: float = 1.3
    receiver_still_max_ft_s: float = 2.5
    receiver_react_window_s: float = 2.0
    receiver_react_min_ft_s: float = 4.5

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
    ftoss:  List[Tuple[float, float, float]] = field(default_factory=list)
    fballs: List[Tuple[float, float, float]] = field(default_factory=list)


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
        for cand_list in (rec.toss, rec.ftoss, rec.fballs):
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
        for attr in ("toss", "ftoss", "fballs"):
            cands = getattr(rec, attr)
            kept = [c for c in cands
                    if (int(c[0]) // cell, int(c[1]) // cell) not in hot]
            dropped += len(cands) - len(kept)
            setattr(rec, attr, kept)
    match._static_suppressed = True
    return dropped


def selfcal_far_band(match: MatchTelemetry,
                     cfg: SegmenterConfig) -> Tuple[float, float]:
    """Whole-video far ready band (single median offset). Kept for callers
    that want one static band; detect_serve_events uses the windowed
    version (selfcal_far_bands) so a mid-video calibration step doesn't
    blind half the match."""
    if not cfg.far_band_selfcal:
        return cfg.far_band_ft
    L = cfg.court_length_ft
    dists = sorted(r.far_world[1] - L for r in match.records
                   if r.far_world is not None)
    if len(dists) < cfg.far_selfcal_min_frames:
        return cfg.far_band_ft
    center = dists[len(dists) // 2]
    return (center - cfg.far_band_halfwidth_ft,
            center + cfg.far_band_halfwidth_ft)


def selfcal_far_bands(match: MatchTelemetry,
                      cfg: SegmenterConfig) -> "_FarBandLookup":
    """
    Rolling far ready band: the video is chopped into far_selfcal_window_s
    chunks, each independently re-centered on its own median far-player
    distance.  A chunk with too few far-tracked frames borrows the
    whole-video band (selfcal_far_band) as fallback.  Returns a callable
    lookup: band_fn(t) -> (lo, hi).
    """
    fallback = selfcal_far_band(match, cfg)
    if not cfg.far_band_selfcal or not match.records:
        return _FarBandLookup([fallback], 0.0, match.duration or 1.0)

    L = cfg.court_length_ft
    win = max(1.0, cfg.far_selfcal_window_s)
    duration = match.duration
    n_chunks = max(1, int(duration // win) + 1)
    buckets: List[List[float]] = [[] for _ in range(n_chunks)]
    for r in match.records:
        if r.far_world is not None:
            idx = min(n_chunks - 1, int(r.t // win))
            buckets[idx].append(r.far_world[1] - L)

    bands = []
    for vals in buckets:
        if len(vals) >= cfg.far_selfcal_min_window_frames:
            vals.sort()
            center = vals[len(vals) // 2]
            bands.append((center - cfg.far_band_halfwidth_ft,
                         center + cfg.far_band_halfwidth_ft))
        else:
            bands.append(fallback)
    return _FarBandLookup(bands, 0.0, win)


class _FarBandLookup:
    def __init__(self, bands: List[Tuple[float, float]], t0: float, win: float):
        self.bands = bands
        self.win = win

    def __call__(self, t: float) -> Tuple[float, float]:
        idx = min(len(self.bands) - 1, max(0, int(t // self.win)))
        return self.bands[idx]


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
    corroborated: bool = False   # toss_seen or receiver-side reaction

    @property
    def supported(self) -> bool:
        """Any independent evidence beyond the serve score itself."""
        return self.trace_confirmed or self.corroborated


class _TossTracker:
    """Rising-ball-above-the-head evidence (TransitionEngine's ARMED toss
    logic), shared by both serve scorers — only the candidate list, the
    head line, and the confidence floor differ between sides."""

    def __init__(self, cfg: SegmenterConfig, conf_floor: float):
        self.cfg = cfg
        self.conf_floor = conf_floor
        self.reset()

    def reset(self) -> None:
        self.toss_consecutive = 0
        self.toss_gap = 0
        self.toss_above_head = False
        self.toss_min_y: Optional[float] = None
        self.last_toss: Optional[dict] = None

    def update(self, candidates: List[Tuple[float, float, float]],
               head_y: float, now: float) -> float:
        candidates = [c for c in candidates if c[2] >= self.conf_floor]
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
        above_head = cy < head_y

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


class _NearServeScorer:
    """Port of TransitionEngine's ARMED toss/trophy scoring, replayed offline."""

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


class _FarServeScorer:
    """Far serves blend the native-res far-toss track with the recorded
    ST-GCN probability: far_toss_weight * toss + far_stgcn_weight * stgcn.
    The toss is confirmatory, not mandatory: a confirmed toss alone clears
    the threshold, and a sustained-high ST-GCN score alone can also fire
    (>= far_stgcn_solo_threshold) — the segmenter then demands independent
    corroboration (toss / receiver reaction / serve trace) before such an
    event becomes a point.  v1 telemetry has no ftoss channel, so it falls
    back to the raw ST-GCN score (legacy behavior)."""

    def __init__(self, cfg: SegmenterConfig, has_far_toss: bool = True):
        self.cfg = cfg
        self.has_far_toss = has_far_toss
        self.reset()

    def reset(self) -> None:
        self.toss = _TossTracker(self.cfg, self.cfg.far_toss_conf_floor)
        self._toss_scores: Deque[Tuple[float, float]] = deque()

    def update(self, rec: FrameRecord, now: float) -> float:
        if rec.far_box is None:
            return 0.0
        if not self.has_far_toss:
            return rec.stgcn
        ts = self.toss.update(rec.ftoss, rec.far_box[1], now)
        if ts > 0:
            self._toss_scores.append((ts, now))
        while (self._toss_scores and
               now - self._toss_scores[0][1] > self.cfg.serve_event_window_s):
            self._toss_scores.popleft()
        max_toss = max((s for s, _ in self._toss_scores), default=0.0)
        blend = (self.cfg.far_toss_weight * max_toss +
                 self.cfg.far_stgcn_weight * rec.stgcn)
        if rec.stgcn >= self.cfg.far_stgcn_solo_threshold:
            return max(blend, rec.stgcn)     # solo path clears the threshold
        return blend

    def toss_confirmed(self, rec: FrameRecord) -> bool:
        """The toss track peaked above the far player's head."""
        return (rec.far_box is not None and
                self.toss.toss_min_y is not None and
                self.toss.toss_min_y < rec.far_box[1])

    def validate(self, rec: FrameRecord) -> bool:
        return rec.far_box is not None


def detect_serve_events(match: MatchTelemetry, side: str,
                        cfg: SegmenterConfig,
                        far_misses: Optional[List[Tuple[float, float, str]]] = None
                        ) -> List[ServeEvent]:
    """
    Ready-band dwell + score threshold, replayed over the telemetry.

    A serve fires when the player has settled behind their baseline
    (ready_dwell_s inside the band), the recent out-of-band ratio stays low,
    and the side's serve score crosses its threshold.

    For the far side, sub-threshold score peaks (>= far_miss_floor) are
    appended to far_misses as (t, score, "below_threshold") — a review log
    that doubles as a labeling aid for tuning far-serve detection.
    """
    if side == "near":
        baseline, direction = 0.0, -1.0
        band_fn = lambda t: cfg.near_band_ft
        scorer = _NearServeScorer(cfg)
        threshold = cfg.serve_score_threshold
        world_of = lambda r: r.near_world
    else:
        baseline, direction = cfg.court_length_ft, 1.0
        band_fn = selfcal_far_bands(match, cfg)
        scorer = _FarServeScorer(cfg, has_far_toss=bool(
            match.meta.get("has_far_toss", False)))
        threshold = cfg.stgcn_threshold
        world_of = lambda r: r.far_world

    events: List[ServeEvent] = []
    raw_misses: List[Tuple[float, float]] = []
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
            band = band_fn(now)
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
            evt = ServeEvent(t=now, side=side, score=score)
            if side == "near":
                evt.toss_seen = True            # near validate() demands it
            else:
                evt.toss_seen = scorer.toss_confirmed(rec)
            events.append(evt)
            armed = False
            ready_start = None
            # Short re-arm only: an aborted toss (caught ball) fires a false
            # candidate, and the REAL serve often follows within seconds —
            # it must also be captured.  The confirmation-aware dedupe picks
            # the right one of the resulting close pair.
            cooldown_until = now + cfg.serve_rearm_s
        elif side == "far" and far_misses is not None and score < threshold:
            # Review metric: the blend, or the raw ST-GCN when no toss backs
            # it up (blend alone caps at far_stgcn_weight without a toss, so
            # interesting pose-only moments would never clear the floor).
            miss_score = max(score, rec.stgcn)
            if miss_score >= cfg.far_miss_floor:
                raw_misses.append((now, miss_score))

    if far_misses is not None and raw_misses:
        # Coalesce into one peak per min_serve_separation_s, skipping peaks
        # that belong to a fired event.
        group_t, group_s = raw_misses[0]
        groups = []
        for t, s in raw_misses[1:]:
            if t - group_t < cfg.min_serve_separation_s:
                if s > group_s:
                    group_t, group_s = t, s
            else:
                groups.append((group_t, group_s))
                group_t, group_s = t, s
        groups.append((group_t, group_s))
        for t, s in groups:
            if not any(abs(t - e.t) < cfg.min_serve_separation_s for e in events):
                far_misses.append((t, s, "below_threshold"))

    return events


def dedupe_serve_events(events: List[ServeEvent],
                        cfg: SegmenterConfig) -> List[ServeEvent]:
    """
    Collapse events closer than min_serve_separation_s.  Two serves cannot
    happen that close together, so one of a conflicting pair is false.
    Priority within a pair:
      1. a supported event (serve trace, toss, or receiver corroboration)
         beats an unsupported one — an aborted toss (ball caught, falls
         near-vertically, never confirms) loses to the real serve that
         follows it;
      2. near beats far (the toss-anchored near detector is better tuned);
      3. the earlier event wins (it marks the serve-motion onset).
    Run this AFTER trace confirmation + corroboration so rule 1 has data.
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
    pattern AND that lack independent support (serve trace, toss, or receiver
    corroboration) — weak anomalies only."""
    if len(events) < 2:
        return events
    decoded = _viterbi([e.side for e in events], cfg.hmm_p_stay, cfg.hmm_p_correct)
    kept = []
    for evt, dec in zip(events, decoded):
        if evt.side != dec and not evt.supported:
            if verbose:
                print(f"[HMM] Dropped serve @ {evt.t:.2f}s side={evt.side} "
                      f"(decoded={dec}, unsupported)")
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
    The native-res far-crop detections (fballs) are merged in — the whole-
    court pass barely sees the ball near the far baseline, and without them
    far serves can never grow a confirmable trace.
    """
    persp = make_image_row_perspective(cfg.frame_height_px)
    mgr = BallTrackManager(fps=match.fps, perspective_scale=persp)
    ball_vel = _SmoothedVelocity(cfg.coupling_window_s)
    player_vel = _SmoothedVelocity(cfg.coupling_window_s)

    out: List[ReplayFrame] = []
    for rec in match.slice(t0, t1):
        dets = []
        for bx, by, conf in list(rec.balls) + list(rec.fballs):
            inside = False
            for box in (rec.near_box, rec.far_box):
                if box and box[0] <= bx <= box[2] and box[1] <= by <= box[3]:
                    inside = True
                    break
            if inside:
                continue
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


def receiver_reacts(match: MatchTelemetry, kin: PlayerKinematics,
                    serve_t: float, cfg: SegmenterConfig) -> bool:
    """
    Receiver-side corroboration of a FAR serve: the near player (the
    receiver) stands quasi-still while the far player serves, then bursts
    into motion within receiver_react_window_s.  Near-side tracking is the
    pipeline's strongest sensor, so this vouches for far events whose own
    ball trace is too weak to confirm.
    """
    i0, i1 = match.index_range(serve_t - cfg.receiver_still_window_s,
                               serve_t - 0.1)
    pre = [kin.speed_near[i] for i in range(i0, i1)
           if kin.speed_near[i] is not None]
    i0, i1 = match.index_range(serve_t + 0.1,
                               serve_t + cfg.receiver_react_window_s)
    post = [kin.speed_near[i] for i in range(i0, i1)
            if kin.speed_near[i] is not None]
    if not pre or not post:
        return False
    pre_median = sorted(pre)[len(pre) // 2]
    return (pre_median <= cfg.receiver_still_max_ft_s and
            max(post) >= cfg.receiver_react_min_ft_s)


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

    # Serve-trace confirmation + corroboration per candidate — BEFORE dedupe,
    # so a supported real serve can displace the unsupported aborted-toss
    # candidate that fired just before it.
    candidates = sorted(near + far, key=lambda e: e.t)
    for evt in candidates:
        rep = replay_ball_tracker(match, evt.t - 0.3,
                                  evt.t + cfg.confirm_window_s, cfg)
        evt.trace_confirmed = confirm_serve_trace(rep, cfg)
        if evt.side == "far":
            evt.corroborated = (evt.toss_seen or
                                receiver_reacts(match, kin, evt.t, cfg))

    # Far events must have at least one piece of independent evidence
    # (serve trace, toss, or receiver reaction) — the ST-GCN solo path is
    # a candidate generator, not a decision maker.
    accepted: List[ServeEvent] = []
    for evt in candidates:
        if evt.side == "far" and not evt.supported:
            far_misses.append((evt.t, evt.score, "uncorroborated"))
            if verbose:
                print(f"[SEG] Far candidate @ {evt.t:.2f}s "
                      f"(score {evt.score:.3f}) dropped: no corroboration")
            continue
        accepted.append(evt)

    events = dedupe_serve_events(accepted, cfg)
    if verbose and len(events) != len(accepted):
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

    # ---- Far serve detection, legacy v1 telemetry: dwell + raw ST-GCN ----
    recs3 = []
    f = 0
    t = 0.0
    for _ in range(45):
        recs3.append(_mk_rec(f, t, far_wy=80.0)); f += 1; t += dt
    for _ in range(10):
        recs3.append(_mk_rec(f, t, far_wy=80.0, stgcn=0.9)); f += 1; t += dt
    far_events = detect_serve_events(build_match(recs3), "far", cfg)
    check("far serve detected from ST-GCN score (legacy telemetry)",
          len(far_events) == 1)

    # Out-of-band far player never fires even with a high score.
    recs4 = [_mk_rec(i, i * dt, far_wy=55.0, stgcn=0.95) for i in range(90)]
    check("far serve requires the ready band",
          len(detect_serve_events(build_match(recs4), "far", cfg)) == 0)

    # ---- Far serve detection, v2 telemetry: toss-primary blend ----
    # Rising far toss above the far box top (120) + modest ST-GCN.
    def far_blend_recs(ftoss_on, stgcn_val):
        recs, f, t = [], 0, 0.0
        for _ in range(45):
            recs.append(_mk_rec(f, t, far_wy=80.0)); f += 1; t += dt
        fty = 119.0
        for _ in range(6):
            fty -= 4.0
            kw = {"stgcn": stgcn_val}
            if ftoss_on:
                kw["ftoss"] = [(465.0, fty, 0.8)]
            recs.append(_mk_rec(f, t, far_wy=80.0, **kw)); f += 1; t += dt
        for _ in range(30):
            recs.append(_mk_rec(f, t, far_wy=80.0)); f += 1; t += dt
        return recs

    evs_blend = detect_serve_events(
        build_match(far_blend_recs(True, 0.4), has_far_toss=True), "far", cfg)
    check("far toss + ST-GCN blend fires", len(evs_blend) == 1)
    check("far toss event records toss_seen",
          evs_blend and evs_blend[0].toss_seen)
    evs_toss_only = detect_serve_events(
        build_match(far_blend_recs(True, 0.0), has_far_toss=True), "far", cfg)
    check("far toss alone clears the threshold", len(evs_toss_only) == 1)
    evs_weak_stgcn = detect_serve_events(
        build_match(far_blend_recs(False, 0.15), has_far_toss=True), "far", cfg)
    check("weak ST-GCN alone cannot raise a candidate",
          len(evs_weak_stgcn) == 0)
    evs_solo = detect_serve_events(
        build_match(far_blend_recs(False, 0.9), has_far_toss=True), "far", cfg)
    check("strong ST-GCN alone raises a candidate (toss not mandatory)",
          len(evs_solo) == 1 and not evs_solo[0].toss_seen)

    # Sub-threshold far score peaks land in the miss log.  A dedicated,
    # higher stgcn_threshold carves out a below-threshold-but-above-
    # far_miss_floor gap to exercise (defaults have both at 0.30, folder-23
    # tuned, so there's no such gap to observe against the shared cfg).
    cfg_miss = SegmenterConfig(stgcn_threshold=0.6)
    misses = []
    detect_serve_events(build_match(far_blend_recs(False, 0.5),
                                    has_far_toss=True),
                        "far", cfg_miss, far_misses=misses)
    check("sub-threshold far peak logged as below_threshold",
          len(misses) == 1 and misses[0][2] == "below_threshold" and
          abs(misses[0][1] - 0.5) < 0.01)

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

    # ---- Self-calibrated far band ----
    # Folder-24 pattern: homography offset puts the serving far player at a
    # systematic +20 ft, outside the fixed (-6, 6) band.  With >=500 far
    # samples the band re-centers on the median distance and the serve fires.
    recs_cal = []
    f, t = 0, 0.0
    for i in range(600):
        kw = dict(near_wy=-1.5, far_wy=98.0)   # dist = +20 ft all match
        if 450 <= i < 462:
            kw["stgcn"] = 0.9
        recs_cal.append(_mk_rec(f, t, **kw))
        f += 1; t += dt
    m_cal = build_match(recs_cal, has_far_toss=True)
    check("self-calibrated band recovers an offset far serve",
          len(detect_serve_events(m_cal, "far", cfg)) == 1)
    cfg_nocal = SegmenterConfig(far_band_selfcal=False)
    check("fixed band misses the offset far serve (control)",
          len(detect_serve_events(m_cal, "far", cfg_nocal)) == 0)

    # ---- Windowed self-cal: the offset itself steps mid-video ----
    # Regime A (dist=+8ft, matches the static far_band_ft) for the first
    # minute, a tracking gap, then regime B (dist=+45ft, invisible to a
    # whole-video median dominated by regime A) for the second minute.
    recs_step = []
    f, t = 0, 0.0
    while t < 60.0:
        kw = dict(near_wy=-1.5, far_wy=86.0)          # dist = +8 ft
        if 20.0 <= t < 20.4:
            kw["stgcn"] = 0.9
        recs_step.append(_mk_rec(f, t, **kw)); f += 1; t += dt
    while t < 65.0:                                    # gap: far player lost
        recs_step.append(_mk_rec(f, t, near_wy=-1.5)); f += 1; t += dt
    while t < 130.0:
        kw = dict(near_wy=-1.5, far_wy=123.0)         # dist = +45 ft
        if 100.0 <= t < 100.4:
            kw["stgcn"] = 0.9
        recs_step.append(_mk_rec(f, t, **kw)); f += 1; t += dt
    cfg_step = SegmenterConfig(far_selfcal_window_s=30.0,
                               far_selfcal_min_window_frames=30)
    m_step = build_match(recs_step, has_far_toss=True)
    evs_step = detect_serve_events(m_step, "far", cfg_step)
    check("windowed self-cal recovers both regime-A and regime-B serves",
          len(evs_step) == 2)
    whole_band = selfcal_far_band(m_step, cfg_step)
    check("whole-video median band cannot straddle both regimes",
          not (whole_band[0] <= 45.0 <= whole_band[1] and
               whole_band[0] <= 8.0 <= whole_band[1]))

    # ---- Far acceptance: solo ST-GCN candidates need corroboration ----
    def far_solo_match(receiver_bursts, with_fballs=False):
        recs, ff, t = [], 0, 0.0
        while t < 6.0:
            kw = dict(near_wy=-1.5, far_wy=80.0)
            if 1.5 <= t < 1.85:
                kw["stgcn"] = 0.9
            if receiver_bursts and 1.7 <= t < 2.7:
                kw["near_wx"] = 13.5 + 8.0 * (t - 1.7)   # receiver sprints off
            elif receiver_bursts and t >= 2.7:
                kw["near_wx"] = 21.5
            if with_fballs and 1.6 <= t < 2.3:
                u = t - 1.6                               # serve flight, native crop
                kw["fballs"] = [(485.0 + 85.0 * u, 130.0 + 85.0 * u, 0.8)]
            recs.append(_mk_rec(ff, t, **kw))
            ff += 1; t += dt
        return build_match(recs, has_far_toss=True)

    misses_a: list = []
    segs_a = segment_match(far_solo_match(False), cfg, verbose=False,
                           far_misses_out=misses_a)
    check("uncorroborated solo far candidate is dropped", len(segs_a) == 0)
    check("dropped candidate logged as uncorroborated",
          any(r == "uncorroborated" for _, _, r in misses_a))

    segs_b = segment_match(far_solo_match(True), cfg, verbose=False)
    check("receiver reaction corroborates the far serve",
          len(segs_b) == 1 and segs_b[0].side == "far")

    segs_c = segment_match(far_solo_match(False, with_fballs=True), cfg,
                           verbose=False)
    check("native-res far trace (fballs) confirms the far serve",
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
