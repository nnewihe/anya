"""
anya_near_serve.py
==================
Near-side serve *probability* from the telemetry written by anya_telemetry.py.

This is the graded successor to anya_near_serve_archive.py.  The archive ran a
WAITING/ARMED/ACTIVE state machine over live video and emitted a binary event
when three hard gates fired in sequence (ready-zone dwell, ball toss, player
box aspect-ratio shift).  Here the same three cues are scored continuously in
[0, 1] and combined into a per-frame probability, so borderline serves are
visible instead of silently discarded, and the decision threshold moves to the
consumer.

No models are loaded and no video is decoded — this reads the telemetry JSONL
and nothing else.

Cues (all time-based, so a strided telemetry file scores the same):

  D  dwell      How long the near player has stood in the ready-zone band
                behind the baseline, times how still they stood.  Duration
                saturates (1 - exp(-dwell/tau)); stillness ramps down with the
                RMS scatter of their world feet position.  A long, motionless
                ready therefore scores far above a brief one, which is the main
                behavioural difference from the archive's boolean dwell gate.
                The score is latched with a short decay tail so it survives the
                serve motion that necessarily ends the dwell.

  T  toss       Upward ball motion inside the archive's toss ROI (a box of
                2/3 player-height width, spanning from 2/3 box-height above the
                player down to 1/3 into the box).  Graded on rise magnitude and
                on how monotonic the rise is, then held under a recency
                envelope so a toss only boosts the strike that follows it.
                `all_balls` is unfiltered, so meta's `exclusion_zones` are
                applied here.

  J  jerk       Player box aspect ratio (w/h) collapses as the server reaches
                up.  Scored on amplitude (max-min over the window, as the
                archive did) multiplied by abruptness: the peak second
                derivative of the ratio, in 1/s^2.  A sharp snap outscores the
                same amplitude spread over a slow lean.

Combination (gated product, weights in NearServeConfig):

    P = J * (toss_floor + (1-toss_floor)*T) * (dwell_floor + (1-dwell_floor)*D)

The strike motion is necessary — no ratio event, no serve — while toss and
dwell raise an otherwise ambiguous motion toward certainty.  These weights are
hand-set heuristics chosen to reproduce the archive's ordering, NOT a
calibrated probability fitted to labelled serves; treat P as a ranking score
until it has been tuned against ground truth.

Outputs (next to the telemetry file):
    <stem>_near_serve_prob.json    every frame: {f, t, p, dwell, toss, jerk}
    <stem>_near_serve_events.json  runs of p >= threshold, collapsed to one
                                   entry per peak, with hh:mm:ss.mmm stamps

Run:
    python -m pipeline.anya_near_serve match.mp4 [--threshold 0.5]
    python -m pipeline.anya_near_serve match_anya_telemetry.jsonl
"""

import argparse
import json
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

try:                                        # package import (python -m pipeline.x)
    from .anya_telemetry import telemetry_path_for
except ImportError:                         # script import (python pipeline/x.py)
    from anya_telemetry import telemetry_path_for

PROB_SUFFIX   = "_near_serve_prob.json"
EVENTS_SUFFIX = "_near_serve_events.json"


@dataclass
class NearServeConfig:
    # --- ready zone (world feet; y=0 is the near baseline, negative = behind)
    zone_y_min_ft: float = -3.5     # archive: near_baseline_y_min_ft
    zone_y_max_ft: float =  0.5     # archive: near_baseline_y_max_ft
    zone_x_pad_ft: float =  3.0     # tolerance outside the singles sidelines
    court_width_ft: float = 27.0

    # --- dwell scoring
    dwell_tau_s:      float = 1.5   # duration score = 1 - exp(-dwell/tau)
    dwell_max_s:      float = 6.0   # ignore dwell history older than this
    dwell_settle_s:   float = 0.4   # ignore the last N s (that's the strike)
    still_tight_ft:   float = 0.75  # RMS scatter at or below this = perfectly still
    still_loose_ft:   float = 5.0   # archive's hard reject radius = zero credit
    dwell_gap_s:      float = 0.3   # missing player box tolerated this long
    dwell_decay_s:    float = 0.5   # e-folding decay once the dwell breaks

    # --- toss scoring
    toss_window_s:    float = 0.5   # archive looked back 15 frames @30fps
    toss_min_samples: int   = 3
    toss_rise_lo_px:  float = 3.0   # analysis-space px (archive: 5 px native)
    toss_rise_hi_px:  float = 12.0  # full credit at this rise
    toss_hold_s:      float = 0.6   # full credit for this long after the toss
    toss_valid_s:     float = 1.5   # archive's toss timeout; zero credit beyond
    # Monotonicity counts sign flips between CONSECUTIVE samples, so without
    # this it silently measures frame rate as much as it measures the toss:
    # the same physical rise sampled twice as densely gets twice as many
    # chances for a pixel of jitter to flip a comparison.  Data/35 is 120 fps
    # and its one serve scored T=0.57 ungated — with an otherwise perfect
    # 246 px climb — against T=1.00 at any rate from 60 Hz down to 10 Hz.
    # Thinning to a fixed rate first makes the cue mean the same thing on 30,
    # 60 and 120 fps footage.  0 disables the thinning (the legacy default,
    # which keeps the original scorer bit-for-bit reproducible); for_low_rate
    # turns it on at 25 Hz — deliberately under 29.97 so the canonical rate is
    # reachable from every source rate in the corpus.
    toss_mono_hz:     float = 0.0

    # --- ratio-jerk scoring
    jerk_window_s:    float = 0.5   # archive: int(fps * 0.5)
    jerk_min_samples: int   = 5     # archive: len(ratios) < 5 -> no trigger
    ratio_smooth_n:   int   = 3     # taps of moving average before differencing
    amp_lo:           float = 0.06  # w/h spread giving zero credit
    amp_hi:           float = 0.20  # archive's trigger was 0.15
    jerk_lo:          float = 2.0   # |d2(w/h)/dt2| in 1/s^2 giving no boost
    jerk_hi:          float = 12.0  # full abruptness boost
    jerk_boost:       float = 0.5   # amp is multiplied by (1-boost + boost*jerkiness)

    # --- combination
    toss_floor:  float = 0.35       # P retained when no toss was seen
    dwell_floor: float = 0.40       # P retained when the player never settled
                                    # (shared with the legacy product form, so
                                    # it stays at 0.40 here; for_low_rate uses
                                    # the swept 0.45)

    # --- event extraction
    # 0.5 is the legacy default and is kept for the legacy scorer; the
    # low-rate profile overrides it to 0.85, which is where the additive P
    # sits best across the 13-clip corpus (57/59 recall, 57/76 precision,
    # zero events on every far-serve-only clip).
    threshold:      float = 0.5
    event_refract_s: float = 3.0    # archive held ACTIVE for fps*3 after a serve

    # ------------------------------------------------------------------
    # Low-rate profile.  Defaults reproduce the legacy behaviour exactly; the
    # "range"/"additive" modes are what anya_near_telemetry's 5 fps player
    # track is scored with.  See NearServeConfig.for_low_rate().
    # ------------------------------------------------------------------

    # J definition.  "legacy" = amplitude x abruptness (peak 2nd derivative of
    # w/h).  "range" = how much the near box aspect ratio CHANGED over a
    # sliding window, and nothing else — more change, higher J.  The second
    # derivative needs the strike resolved in time; at 5 fps the strike spans
    # ~2 samples and the derivative is dominated by sampling phase, whereas a
    # max-minus-min range over a 1 s window is unaffected by where in the
    # motion the samples happen to land.
    jerk_mode:            str   = "legacy"     # "legacy" | "range"
    jerk_range_window_s:  float = 1.0

    # P combination.  "legacy" = J * toss * dwell (strike necessary).
    # "additive" = (jerk_w*J + toss_w*T) * dwell — J and T now trade off
    # against each other instead of gating one another.
    # Weights swept over the 12-clip corpus (jerk_w x dwell_floor x threshold,
    # 1920 points, evaluated analytically from the cached cue values).  The
    # 0.3/0.7 this started with dated from when the toss cue was carrying the
    # decision; once the toss ROI was cropped tight and static clutter was
    # suppressed, J became the more reliable of the two and the optimum moved
    # past parity.  At full recall the best achievable is 29 false positives,
    # reached across a plateau spanning jerk_w 0.50-0.65 — 0.55 is its centre,
    # not its edge, so this is not a knife-edge fit.
    combine_mode: str   = "legacy"             # "legacy" | "additive"
    jerk_w:       float = 0.55
    toss_w:       float = 0.45

    # In additive mode, the J that enters P at a toss is the LARGEST J seen
    # within this many seconds of that toss, rather than J on the frame being
    # scored.  The aspect-ratio change and the toss are the same physical
    # event sampled by two different cues at two different rates; pairing them
    # by proximity stops a 5 fps J from missing a 30 fps toss by one sample.
    jerk_link_s: float = 1.5

    # ------------------------------------------------------------------
    # Sequence gate.  A serve is not three cues that happen to co-occur, it
    # is three cues in ORDER: settle in the ready zone, toss the ball up,
    # then snap the box aspect ratio as the racket arm extends.  The weighted
    # sum scores an unordered co-occurrence, and `jerk_link_mode="symmetric"`
    # even credits a ratio change that happened BEFORE the toss — which is
    # what a player bouncing the ball or shadow-swinging in dead time looks
    # like.  "after_toss" requires the strike to follow the toss inside
    # jerk_link_post_s, and require_sequence additionally kills P outright
    # when the dwell was too short or no strike followed at all.
    jerk_link_mode:   str   = "symmetric"   # "symmetric" | "after_toss"
    jerk_link_pre_s:  float = 0.2   # slop: the toss anchor is the latch time,
                                    # which can trail the true release slightly
    jerk_link_post_s: float = 1.0   # the strike must land within this
    require_sequence: bool  = False
    seq_dwell_min_s:  float = 1.0   # settled ready-zone time before the toss
    seq_jerk_min:     float = 0.30  # a strike this weak is not a strike

    # Adaptive static-clutter suppression for the toss ROI.
    #
    # The meta `exclusion_zones` come from a 50-frame scan at startup, so they
    # only catch clutter that is bright during those 50 frames.  Leaves and
    # signage come and go with the sun, and whatever is missed is fatal:
    # `_update_toss` keeps the single highest-confidence detection per frame,
    # so a static cell that fires thousands of times outvotes the real ball on
    # nearly every frame.  On Data/38 one cell fired 3350 times (335 per
    # active 10 s bucket) and 96% of all in-ROI detections were static.
    #
    # This scores each cell by hits-per-ACTIVE-bucket, mirroring
    # anya_far_serve.calibrate_static_blobs, which is what makes it robust to
    # flicker: a cell that only lights up when the sun catches it still scores
    # high, because the ratio is taken over the buckets where it was active
    # rather than over the whole clip.  Verified not to harm real tosses —
    # a toss cell sees a few hits spread over many buckets, so it scores ~1-3,
    # well under the threshold (clips 21/22/43 keep 12-31 points per serve).
    # Rate and cell size are swept values, not inherited ones.  4.0/12px came
    # from anya_far_serve, where the ball crosses any given cell only a few
    # times per pass; a near TOSS recurs in nearly the same place every serve,
    # so that rate suppressed the ball itself (Data/38 scored 1/8 at 4.0 and
    # 7/8 at 12.0).  Swept over 13 clips x 10 configs x 6 thresholds; 12.0 at
    # 16 px is the joint optimum and still fires zero events on all three
    # far-serve-only clips.
    toss_static_suppress: bool  = False   # on in for_low_rate()
    static_cell_px:       float = 16.0
    static_bucket_s:      float = 10.0
    static_min_hits:      int   = 8
    static_min_rate:      float = 12.0    # hits per active bucket

    @classmethod
    def for_low_rate(cls, **overrides) -> "NearServeConfig":
        """Config for a 5 fps player track (anya_near_telemetry output).

        dwell_gap_s is the one that bites: at 5 fps consecutive samples are
        0.2 s apart, so the legacy 0.3 s tolerance breaks a dwell run on a
        SINGLE missed detection (0.4 s gap).  ratio_smooth_n drops to 1
        because a 3-tap average at 5 fps smooths over 0.6 s — longer than the
        strike it is meant to preserve.
        """
        base = dict(
            jerk_mode="range", combine_mode="additive",
            dwell_gap_s=0.7, ratio_smooth_n=1, jerk_min_samples=3,
            toss_static_suppress=True, threshold=0.80, toss_mono_hz=25.0,
            dwell_floor=0.45,
        )
        base.update(overrides)
        return cls(**base)


# ----------------------------------------------------------------------------
def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at or below lo, 1 at or above hi, linear between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _hms(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _in_exclusion(cx: float, cy: float, zones: Sequence[Sequence[float]]) -> bool:
    for z in zones:
        x1, y1, x2, y2 = z[0], z[1], z[2], z[3]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    return False


def _toss_roi(box: Sequence[float]) -> Tuple[float, float, float, float]:
    """Archive's _get_near_toss_roi, in x1y1x2y2 terms.

    Width is 2/3 of the player height centred on the box; vertically the ROI
    runs from 2/3 of a box-height above the box top down to 1/3 of a
    box-height into the box — i.e. where a toss lives, not where the player is.
    """
    x1, y1, x2, y2 = box
    ph = y2 - y1
    roi_w = ph * (2.0 / 3.0)
    cx = (x1 + x2) / 2.0
    roi_bottom = y2 - ph * (2.0 / 3.0)          # == y1 + ph/3
    roi_top    = roi_bottom - ph                 # == y1 - 2*ph/3
    return (cx - roi_w / 2.0, roi_top, cx + roi_w / 2.0, roi_bottom)


def _record_balls(rec: Dict) -> Sequence:
    """Ball detections from either telemetry flavour.

    anya_telemetry writes whole-frame `all_balls`; anya_near_telemetry writes
    `balls` from the toss-ROI crop.  Both are [[cx, cy, conf], ...] in
    analysis coordinates.
    """
    balls = rec.get("all_balls")
    if balls is None:
        balls = rec.get("balls")
    return balls or []


def calibrate_static_cells(records: Sequence[Dict],
                           cfg: NearServeConfig) -> set:
    """Grid cells that behave like static false balls, over the whole stream.

    A real ball crosses any given cell a handful of times across a match; a
    static false positive fires on most frames it is visible for.  Scoring by
    hits-per-ACTIVE-bucket rather than raw hits is what makes this survive
    clutter that switches on and off with the light — a cell that is only
    bright for one stretch of the match is still judged on how densely it
    fired during that stretch.
    """
    hits: Dict = defaultdict(int)
    buckets: Dict = defaultdict(set)
    cell = cfg.static_cell_px
    for r in records:
        bucket = int(float(r["t"]) // cfg.static_bucket_s)
        for b in _record_balls(r):
            key = (int(float(b[0]) // cell), int(float(b[1]) // cell))
            hits[key] += 1
            buckets[key].add(bucket)
    return {k for k, n in hits.items()
            if n >= cfg.static_min_hits
            and n / len(buckets[k]) >= cfg.static_min_rate}


# ----------------------------------------------------------------------------
class NearServeScorer:
    """Streaming per-frame near-serve probability."""

    def __init__(self, cfg: Optional[NearServeConfig] = None,
                 exclusion_zones: Optional[Sequence[Sequence[float]]] = None,
                 static_cells: Optional[set] = None):
        self.cfg = cfg or NearServeConfig()
        self.exclusion_zones = list(exclusion_zones or [])
        # Calibrated by score_telemetry from the whole file; empty here means
        # "no suppression", which is what a caller streaming one record at a
        # time necessarily gets.
        self.static_cells = set(static_cells or ())

        self._dwell: deque = deque()        # (t, wx, wy) while in the ready zone
        self._last_seen_t: float = -1e9     # last frame with a near box
        self._last_in_zone_t: float = -1e9
        self._dwell_latch: float = 0.0      # score at the moment dwell broke
        self._dwell_latch_t: float = -1e9
        # Settled duration of the current (or just-ended) ready-zone run, in
        # seconds.  The score conflates duration with stillness, so the
        # sequence gate needs the raw seconds separately.
        self._dwell_dur: float = 0.0
        self._dwell_latch_dur: float = 0.0

        self._ball_y: deque = deque()       # (t, y) of balls inside the toss ROI
        self._toss_peak: float = 0.0
        self._toss_t: float = -1e9

        self._ratios: deque = deque()       # (t, w/h)

    # -- dwell ------------------------------------------------------------
    def _update_dwell(self, box, world, t: float, fresh: bool = True) -> float:
        cfg = self.cfg
        in_zone = False
        if fresh and box is not None and world is not None:
            wx, wy = world
            in_zone = (cfg.zone_y_min_ft <= wy <= cfg.zone_y_max_ft and
                       -cfg.zone_x_pad_ft <= wx <= cfg.court_width_ft + cfg.zone_x_pad_ft)

        if fresh and box is not None:
            self._last_seen_t = t

        if in_zone:
            self._last_in_zone_t = t
            self._dwell.append((t, world[0], world[1]))
        elif not fresh:
            # A held box carries no new position evidence — it must neither
            # extend the run nor end it.  Fall through to scoring the state
            # the last fresh sample left behind.
            pass
        elif t - self._last_in_zone_t > cfg.dwell_gap_s or box is not None:
            # Left the zone (or the box has been missing longer than the gap
            # tolerance): the run is over.  A brief detection dropout inside
            # the zone keeps the run alive.
            if self._dwell:
                self._dwell_latch = self._dwell_score(t)
                self._dwell_latch_t = t
                self._dwell_latch_dur = self._dwell_dur
                self._dwell.clear()

        while self._dwell and t - self._dwell[0][0] > cfg.dwell_max_s:
            self._dwell.popleft()

        if self._dwell:
            live = self._dwell_score(t)
            self._dwell_latch, self._dwell_latch_t = live, t
            self._dwell_latch_dur = self._dwell_dur
            return live
        # Run has ended: report the duration it reached, not zero — the gate
        # asks "did they settle before this toss", which stays true afterwards.
        self._dwell_dur = self._dwell_latch_dur

        # Decaying tail so the strike frames still see the ready that preceded
        # them, even though the serve itself ended the dwell.
        gap = t - self._dwell_latch_t
        if gap <= 0 or self._dwell_latch <= 0.0:
            return self._dwell_latch if gap <= 0 else 0.0
        return self._dwell_latch * math.exp(-gap / cfg.dwell_decay_s)

    def _dwell_score(self, now: float) -> float:
        """Duration x stillness over the settled part of the current run."""
        cfg = self.cfg
        pts = [p for p in self._dwell if now - p[0] >= cfg.dwell_settle_s]
        if len(pts) < 2:
            self._dwell_dur = 0.0
            return 0.0
        dur = pts[-1][0] - pts[0][0]
        self._dwell_dur = dur
        dur_score = 1.0 - math.exp(-dur / cfg.dwell_tau_s)

        n  = len(pts)
        mx = sum(p[1] for p in pts) / n
        my = sum(p[2] for p in pts) / n
        rms = math.sqrt(sum((p[1] - mx) ** 2 + (p[2] - my) ** 2 for p in pts) / n)
        still = 1.0 - _ramp(rms, cfg.still_tight_ft, cfg.still_loose_ft)
        return dur_score * still

    # -- toss -------------------------------------------------------------
    def _roi_candidates(self, box, balls):
        """Every in-ROI, non-excluded detection as (cy, conf), best first.

        Static-cell rejection happens BEFORE the caller takes its argmax,
        which is the whole point: a clutter cell that fires every frame would
        otherwise win the confidence contest against a genuine ball that is
        present on only a fraction of frames.
        """
        cfg = self.cfg
        rx1, ry1, rx2, ry2 = _toss_roi(box)
        cell = cfg.static_cell_px
        suppress = cfg.toss_static_suppress and self.static_cells
        out = []
        for b in balls:
            cx, cy, conf = float(b[0]), float(b[1]), float(b[2])
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue
            if _in_exclusion(cx, cy, self.exclusion_zones):
                continue
            if suppress and (int(cx // cell), int(cy // cell)) in self.static_cells:
                continue
            out.append((cy, conf))
        out.sort(key=lambda c: -c[1])
        return out

    def _update_toss(self, box, balls, t: float) -> float:
        cfg = self.cfg
        if box is not None and balls:
            cands = self._roi_candidates(box, balls)
            if cands:
                self._ball_y.append((t, cands[0][0]))

        while self._ball_y and t - self._ball_y[0][0] > cfg.toss_window_s:
            self._ball_y.popleft()

        raw = self._toss_raw()
        if raw > 0.0:
            # Latch the strongest rise seen; a later, weaker sample must not
            # erase a clean toss that is still inside its validity window.
            if raw >= self._toss_peak or t - self._toss_t > cfg.toss_valid_s:
                self._toss_peak, self._toss_t = raw, t

        age = t - self._toss_t
        if age > cfg.toss_valid_s:
            return 0.0
        if age <= cfg.toss_hold_s:
            return self._toss_peak
        decay = 1.0 - (age - cfg.toss_hold_s) / max(1e-6, cfg.toss_valid_s - cfg.toss_hold_s)
        return self._toss_peak * max(0.0, decay)

    def _toss_raw(self) -> float:
        """Rise magnitude x monotonicity for the balls currently in the ROI.

        The rise is measured over the full window; monotonicity is measured on
        a rate-thinned copy, so a 120 fps clip and a 30 fps clip score the same
        toss the same way (see cfg.toss_mono_hz).
        """
        cfg = self.cfg
        if len(self._ball_y) < cfg.toss_min_samples:
            return 0.0
        pts = list(self._ball_y)
        ys = [p[1] for p in pts]
        rise = ys[0] - ys[-1]                       # image y decreases upward
        if rise <= 0:
            return 0.0

        mono_ys = ys
        if cfg.toss_mono_hz and cfg.toss_mono_hz > 0:
            step = 1.0 / cfg.toss_mono_hz
            thinned, last = [], None
            for p in pts:
                if last is None or p[0] - last >= step - 1e-9:
                    thinned.append(p[1])
                    last = p[0]
            # Only thin when enough survives to still be a meaningful count.
            if len(thinned) >= cfg.toss_min_samples:
                mono_ys = thinned

        ups = sum(1 for a, b in zip(mono_ys, mono_ys[1:]) if b < a)
        mono = ups / max(1, len(mono_ys) - 1)
        return _ramp(rise, cfg.toss_rise_lo_px, cfg.toss_rise_hi_px) * mono

    # -- ratio jerk -------------------------------------------------------
    def _update_jerk(self, box, t: float, fresh: bool = True) -> float:
        cfg = self.cfg
        rng = cfg.jerk_mode == "range"
        window = cfg.jerk_range_window_s if rng else cfg.jerk_window_s

        # Only a fresh detection carries new shape information.  On a held box
        # the ratio is a repeat of the last sample, which would pad the window
        # with duplicates and (in legacy mode) inject dt=0 pairs.
        if fresh and box is not None:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                self._ratios.append((t, w / h))
        while self._ratios and t - self._ratios[0][0] > window:
            self._ratios.popleft()
        if len(self._ratios) < cfg.jerk_min_samples:
            return 0.0

        ts = [p[0] for p in self._ratios]
        rs = _moving_average([p[1] for p in self._ratios], cfg.ratio_smooth_n)

        amp = max(rs) - min(rs)
        amp_score = _ramp(amp, cfg.amp_lo, cfg.amp_hi)

        if rng:
            # J is the aspect-ratio CHANGE over the window, full stop.
            return amp_score
        if amp_score <= 0.0:
            return 0.0

        jerk = _peak_second_derivative(ts, rs)
        jerkiness = _ramp(jerk, cfg.jerk_lo, cfg.jerk_hi)
        return amp_score * (1.0 - cfg.jerk_boost + cfg.jerk_boost * jerkiness)

    # -- public -----------------------------------------------------------
    def update(self, rec: Dict) -> Dict:
        """Score one telemetry record. Returns {f, t, p, dwell, toss, jerk}.

        Accepts both telemetry flavours: anya_telemetry's `all_balls` (whole
        frame, 30 fps, every frame a fresh detection) and
        anya_near_telemetry's `balls` (toss-ROI crop) plus its `pn` flag
        marking which frames carry a fresh 5 fps player sample.

        In additive mode the `jerk` reported here is the instantaneous value
        and `p` is provisional — `_link_jerk_to_toss` rewrites both once the
        whole file is available, since the J that belongs to a toss can be
        sampled after it.
        """
        cfg = self.cfg
        t   = float(rec["t"])
        box = rec.get("np")
        world = rec.get("npw")
        balls = rec.get("all_balls")
        if balls is None:
            balls = rec.get("balls") or []
        fresh = bool(rec.get("pn", True))

        d = self._update_dwell(box, world, t, fresh=fresh)
        s = self._update_toss(box, balls, t)
        j = self._update_jerk(box, t, fresh=fresh)

        out = {"f": int(rec["f"]), "t": round(t, 4),
               "dwell": round(d, 4), "toss": round(s, 4), "jerk": round(j, 4)}
        if cfg.combine_mode == "additive":
            out["p"] = round(self.combine(j, s, d), 4)
            # Anchor for _link_jerk_to_toss: which toss, if any, this frame's
            # T is currently crediting.
            out["_toss_t"] = round(self._toss_t, 4) if s > 0.0 else None
            out["dwell_s"] = round(self._dwell_dur, 3)
        else:
            out["p"] = round(self.combine(j, s, d), 4)
        return out

    def combine(self, j: float, s: float, d: float) -> float:
        """P from the three cue scores."""
        cfg = self.cfg
        dwell_term = cfg.dwell_floor + (1.0 - cfg.dwell_floor) * d
        if cfg.combine_mode == "additive":
            return (cfg.jerk_w * j + cfg.toss_w * s) * dwell_term
        return j * (cfg.toss_floor + (1.0 - cfg.toss_floor) * s) * dwell_term


def _moving_average(vals: List[float], n: int) -> List[float]:
    if n <= 1 or len(vals) < n:
        return list(vals)
    half = n // 2
    out = []
    for i in range(len(vals)):
        lo, hi = max(0, i - half), min(len(vals), i + half + 1)
        out.append(sum(vals[lo:hi]) / (hi - lo))
    return out


def _peak_second_derivative(ts: List[float], vs: List[float]) -> float:
    """max |d2v/dt2| over the window, in units of v per second squared."""
    if len(vs) < 3:
        return 0.0
    d1 = []
    for i in range(len(vs) - 1):
        dt = ts[i + 1] - ts[i]
        if dt <= 0:
            continue
        d1.append(((ts[i] + ts[i + 1]) / 2.0, (vs[i + 1] - vs[i]) / dt))
    peak = 0.0
    for i in range(len(d1) - 1):
        dt = d1[i + 1][0] - d1[i][0]
        if dt <= 0:
            continue
        peak = max(peak, abs((d1[i + 1][1] - d1[i][1]) / dt))
    return peak


# ----------------------------------------------------------------------------
def _link_jerk_to_toss(frames: List[Dict], scorer: "NearServeScorer") -> None:
    """Replace each frame's J with the strongest J near the toss it credits.

    The aspect-ratio change and the ball toss are two views of one serve, but
    they are measured at different rates (5 fps and 30 fps here) and peak at
    different instants — the toss leads, the strike follows.  Scoring a frame
    with the J that happens to coincide with it therefore under-reads the
    strike whenever the 5 fps grid misses its peak.

    So for every frame whose toss score is live, J becomes max(J) over all
    frames within `jerk_link_s` of that toss's own timestamp.  Frames with no
    live toss keep their instantaneous J (they cannot clear any useful
    threshold in additive mode regardless: with T=0, P caps at jerk_w).

    Mutates `frames` in place, rewriting both "jerk" and "p".
    """
    import bisect

    cfg = scorer.cfg
    if cfg.combine_mode != "additive":
        for fr in frames:
            fr.pop("_toss_t", None)
        return

    ts = [fr["t"] for fr in frames]
    js = [fr["jerk"] for fr in frames]
    ds = [fr.get("dwell_s", 0.0) for fr in frames]

    causal = cfg.jerk_link_mode == "after_toss"
    pre  = cfg.jerk_link_pre_s if causal else cfg.jerk_link_s
    post = cfg.jerk_link_post_s if causal else cfg.jerk_link_s

    # Peak J in the strike window of each distinct toss, and the dwell the
    # player had accumulated when that toss happened.  Computed once per toss
    # rather than once per frame.
    cache: Dict[float, Tuple[float, float]] = {}

    def resolve(t_toss: float) -> Tuple[float, float]:
        hit = cache.get(t_toss)
        if hit is not None:
            return hit
        lo = bisect.bisect_left(ts, t_toss - pre)
        hi = bisect.bisect_right(ts, t_toss + post)
        j = max(js[lo:hi], default=0.0)
        # Dwell as of the toss: the last sample at or before the anchor.
        k = max(0, bisect.bisect_right(ts, t_toss) - 1)
        dwell_s = ds[k] if ds else 0.0
        cache[t_toss] = (j, dwell_s)
        return j, dwell_s

    for fr in frames:
        t_toss = fr.pop("_toss_t", None)
        if t_toss is None:
            if cfg.require_sequence:
                # No live toss crediting this frame -> no serve here at all.
                fr["p"] = 0.0
            continue
        j, dwell_s = resolve(t_toss)
        if cfg.require_sequence and (j < cfg.seq_jerk_min
                                     or dwell_s < cfg.seq_dwell_min_s):
            fr["jerk"] = round(j, 4)
            fr["p"] = 0.0
            continue
        fr["jerk"] = round(j, 4)
        fr["p"] = round(scorer.combine(j, fr["toss"], fr["dwell"]), 4)


def extract_events(frames: List[Dict], threshold: float,
                   refract_s: float) -> List[Dict]:
    """Collapse each contiguous run of p >= threshold to its peak frame.

    Runs whose peaks fall within refract_s of the previous accepted event are
    merged into it (the archive locked ACTIVE for 3 s after a serve, so two
    peaks that close are one serve seen twice)."""
    events: List[Dict] = []
    run: List[Dict] = []

    def _flush(run_frames: List[Dict]) -> None:
        if not run_frames:
            return
        peak = max(run_frames, key=lambda r: r["p"])
        ev = {
            "frame":     peak["f"],
            "t":         peak["t"],
            "timestamp": _hms(peak["t"]),
            "p":         peak["p"],
            "dwell":     peak["dwell"],
            "toss":      peak["toss"],
            "jerk":      peak["jerk"],
            "window": {
                "start_frame": run_frames[0]["f"], "end_frame": run_frames[-1]["f"],
                "start_t": run_frames[0]["t"],     "end_t": run_frames[-1]["t"],
            },
        }
        if events and ev["t"] - events[-1]["t"] <= refract_s:
            prev = events[-1]
            prev["window"]["end_frame"] = ev["window"]["end_frame"]
            prev["window"]["end_t"]     = ev["window"]["end_t"]
            if ev["p"] > prev["p"]:
                prev.update({k: ev[k] for k in
                             ("frame", "t", "timestamp", "p", "dwell", "toss", "jerk")})
            return
        events.append(ev)

    for fr in frames:
        if fr["p"] >= threshold:
            run.append(fr)
        else:
            _flush(run)
            run = []
    _flush(run)
    return events


def events_path_for(telemetry_path: str) -> str:
    """Where `score_telemetry` writes its events, without running it.

    Factored out so a caller that only wants the cached events — the energy
    tuner rebuilding the production start timeline — derives the path from the
    same expression score_telemetry does, instead of a second copy that drifts.
    """
    stem = telemetry_path[:-len(".jsonl")] if telemetry_path.endswith(".jsonl") \
        else os.path.splitext(telemetry_path)[0]
    return stem + EVENTS_SUFFIX


def score_telemetry(telemetry_path: str, cfg: Optional[NearServeConfig] = None,
                    prob_path: Optional[str] = None,
                    events_path: Optional[str] = None) -> Tuple[str, str]:
    """Score a telemetry JSONL. Returns (prob_path, events_path)."""
    cfg = cfg or NearServeConfig()

    meta: Dict = {}
    records: List[Dict] = []

    # Static-cell calibration needs the whole stream, so records are read
    # first and scored second.  (The scorer itself is still streaming — a
    # live caller can drive NearServeScorer.update() directly, it just gets
    # no suppression, since the statistics do not exist yet.)
    with open(telemetry_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj and not meta and not records:
                meta = obj["meta"]
                continue
            records.append(obj)

    if not records:
        raise RuntimeError(f"No telemetry records in {telemetry_path}")

    static_cells = (calibrate_static_cells(records, cfg)
                    if cfg.toss_static_suppress else set())
    if cfg.toss_static_suppress:
        n_det = sum(len(_record_balls(r)) for r in records)
        n_sup = sum(1 for r in records for b in _record_balls(r)
                    if (int(float(b[0]) // cfg.static_cell_px),
                        int(float(b[1]) // cfg.static_cell_px)) in static_cells)
        print(f"[NEAR-SERVE] static suppression: {len(static_cells)} cell(s), "
              f"{n_sup}/{n_det} detection(s) dropped "
              f"({(n_sup / n_det if n_det else 0):.0%})")

    scorer = NearServeScorer(cfg, meta.get("exclusion_zones", []), static_cells)
    frames = [scorer.update(obj) for obj in records]

    _link_jerk_to_toss(frames, scorer)
    events = extract_events(frames, cfg.threshold, cfg.event_refract_s)

    stem = telemetry_path[:-len(".jsonl")] if telemetry_path.endswith(".jsonl") \
        else os.path.splitext(telemetry_path)[0]
    prob_path   = prob_path   or (stem + PROB_SUFFIX)
    events_path = events_path or events_path_for(telemetry_path)

    out_meta = {
        "source_telemetry": os.path.basename(telemetry_path),
        "video":            meta.get("video"),
        "fps":              meta.get("fps"),
        "stride":           meta.get("stride", 1),
        "analysis_size":    meta.get("analysis_size"),
        "frames_scored":    len(frames),
        "threshold":        cfg.threshold,
        "config":           asdict(cfg),
    }

    with open(prob_path, "w") as fh:
        json.dump({"meta": out_meta, "frames": frames}, fh)
    with open(events_path, "w") as fh:
        json.dump({"meta": out_meta, "events": events}, fh, indent=2)

    above = sum(1 for f in frames if f["p"] >= cfg.threshold)
    peak  = max(f["p"] for f in frames)
    print(f"[NEAR-SERVE] {len(frames)} frames scored, {above} above "
          f"{cfg.threshold:.2f}, peak p={peak:.3f}")
    print(f"[NEAR-SERVE] {len(events)} serve event(s) → {events_path}")
    for ev in events:
        print(f"    {ev['timestamp']}  f={ev['frame']:<7} p={ev['p']:.3f}  "
              f"(dwell {ev['dwell']:.2f}  toss {ev['toss']:.2f}  jerk {ev['jerk']:.2f})")
    print(f"[NEAR-SERVE] per-frame probabilities → {prob_path}")
    return prob_path, events_path


def resolve_telemetry_path(path: str) -> str:
    """Accept a telemetry JSONL directly, or a video path to derive it from."""
    if path.endswith(".jsonl"):
        return path
    derived = telemetry_path_for(path)
    if not os.path.isfile(derived):
        raise FileNotFoundError(
            f"No telemetry for {path} — expected {derived}. "
            "Run: python -m pipeline.anya_telemetry <video>")
    return derived


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-frame near-serve probability from anya_telemetry.py output")
    parser.add_argument("input",
                        help="Telemetry JSONL, or the video it was extracted from")
    parser.add_argument("--threshold", type=float, default=NearServeConfig.threshold,
                        help="Probability at or above which a frame is a serve "
                             "event (default 0.5)")
    parser.add_argument("--refractory", type=float,
                        default=NearServeConfig.event_refract_s,
                        help="Merge peaks closer together than this many seconds")
    parser.add_argument("--prob-out", default=None, help="Override per-frame JSON path")
    parser.add_argument("--events-out", default=None, help="Override events JSON path")
    args = parser.parse_args()

    conf = NearServeConfig(threshold=args.threshold,
                           event_refract_s=args.refractory)
    score_telemetry(resolve_telemetry_path(args.input), conf,
                    prob_path=args.prob_out, events_path=args.events_out)
