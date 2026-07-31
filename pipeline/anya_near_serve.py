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
from collections import deque
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

    # --- event extraction
    threshold:      float = 0.5
    event_refract_s: float = 3.0    # archive held ACTIVE for fps*3 after a serve


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


# ----------------------------------------------------------------------------
class NearServeScorer:
    """Streaming per-frame near-serve probability."""

    def __init__(self, cfg: Optional[NearServeConfig] = None,
                 exclusion_zones: Optional[Sequence[Sequence[float]]] = None):
        self.cfg = cfg or NearServeConfig()
        self.exclusion_zones = list(exclusion_zones or [])

        self._dwell: deque = deque()        # (t, wx, wy) while in the ready zone
        self._last_seen_t: float = -1e9     # last frame with a near box
        self._last_in_zone_t: float = -1e9
        self._dwell_latch: float = 0.0      # score at the moment dwell broke
        self._dwell_latch_t: float = -1e9

        self._ball_y: deque = deque()       # (t, y) of balls inside the toss ROI
        self._toss_peak: float = 0.0
        self._toss_t: float = -1e9

        self._ratios: deque = deque()       # (t, w/h)

    # -- dwell ------------------------------------------------------------
    def _update_dwell(self, box, world, t: float) -> float:
        cfg = self.cfg
        in_zone = False
        if box is not None and world is not None:
            wx, wy = world
            in_zone = (cfg.zone_y_min_ft <= wy <= cfg.zone_y_max_ft and
                       -cfg.zone_x_pad_ft <= wx <= cfg.court_width_ft + cfg.zone_x_pad_ft)

        if box is not None:
            self._last_seen_t = t

        if in_zone:
            self._last_in_zone_t = t
            self._dwell.append((t, world[0], world[1]))
        elif t - self._last_in_zone_t > cfg.dwell_gap_s or box is not None:
            # Left the zone (or the box has been missing longer than the gap
            # tolerance): the run is over.  A brief detection dropout inside
            # the zone keeps the run alive.
            if self._dwell:
                self._dwell_latch = self._dwell_score(t)
                self._dwell_latch_t = t
                self._dwell.clear()

        while self._dwell and t - self._dwell[0][0] > cfg.dwell_max_s:
            self._dwell.popleft()

        if self._dwell:
            live = self._dwell_score(t)
            self._dwell_latch, self._dwell_latch_t = live, t
            return live

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
            return 0.0
        dur = pts[-1][0] - pts[0][0]
        dur_score = 1.0 - math.exp(-dur / cfg.dwell_tau_s)

        n  = len(pts)
        mx = sum(p[1] for p in pts) / n
        my = sum(p[2] for p in pts) / n
        rms = math.sqrt(sum((p[1] - mx) ** 2 + (p[2] - my) ** 2 for p in pts) / n)
        still = 1.0 - _ramp(rms, cfg.still_tight_ft, cfg.still_loose_ft)
        return dur_score * still

    # -- toss -------------------------------------------------------------
    def _update_toss(self, box, balls, t: float) -> float:
        cfg = self.cfg
        if box is not None and balls:
            rx1, ry1, rx2, ry2 = _toss_roi(box)
            best = None
            for b in balls:
                cx, cy, conf = float(b[0]), float(b[1]), float(b[2])
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    continue
                if _in_exclusion(cx, cy, self.exclusion_zones):
                    continue
                if best is None or conf > best[1]:
                    best = (cy, conf)
            if best is not None:
                self._ball_y.append((t, best[0]))

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
        """Rise magnitude x monotonicity for the balls currently in the ROI."""
        cfg = self.cfg
        if len(self._ball_y) < cfg.toss_min_samples:
            return 0.0
        ys = [y for _, y in self._ball_y]
        rise = ys[0] - ys[-1]                       # image y decreases upward
        if rise <= 0:
            return 0.0
        ups = sum(1 for a, b in zip(ys, ys[1:]) if b < a)
        mono = ups / max(1, len(ys) - 1)
        return _ramp(rise, cfg.toss_rise_lo_px, cfg.toss_rise_hi_px) * mono

    # -- ratio jerk -------------------------------------------------------
    def _update_jerk(self, box, t: float) -> float:
        cfg = self.cfg
        if box is not None:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                self._ratios.append((t, w / h))
        while self._ratios and t - self._ratios[0][0] > cfg.jerk_window_s:
            self._ratios.popleft()
        if len(self._ratios) < cfg.jerk_min_samples:
            return 0.0

        ts = [p[0] for p in self._ratios]
        rs = _moving_average([p[1] for p in self._ratios], cfg.ratio_smooth_n)

        amp = max(rs) - min(rs)
        amp_score = _ramp(amp, cfg.amp_lo, cfg.amp_hi)
        if amp_score <= 0.0:
            return 0.0

        jerk = _peak_second_derivative(ts, rs)
        jerkiness = _ramp(jerk, cfg.jerk_lo, cfg.jerk_hi)
        return amp_score * (1.0 - cfg.jerk_boost + cfg.jerk_boost * jerkiness)

    # -- public -----------------------------------------------------------
    def update(self, rec: Dict) -> Dict:
        """Score one telemetry record. Returns {f, t, p, dwell, toss, jerk}."""
        cfg = self.cfg
        t   = float(rec["t"])
        box = rec.get("np")
        world = rec.get("npw")
        balls = rec.get("all_balls") or []

        d = self._update_dwell(box, world, t)
        s = self._update_toss(box, balls, t)
        j = self._update_jerk(box, t)

        p = (j
             * (cfg.toss_floor  + (1.0 - cfg.toss_floor)  * s)
             * (cfg.dwell_floor + (1.0 - cfg.dwell_floor) * d))

        return {"f": int(rec["f"]), "t": round(t, 4), "p": round(p, 4),
                "dwell": round(d, 4), "toss": round(s, 4), "jerk": round(j, 4)}


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


def score_telemetry(telemetry_path: str, cfg: Optional[NearServeConfig] = None,
                    prob_path: Optional[str] = None,
                    events_path: Optional[str] = None) -> Tuple[str, str]:
    """Score a telemetry JSONL. Returns (prob_path, events_path)."""
    cfg = cfg or NearServeConfig()

    meta: Dict = {}
    frames: List[Dict] = []
    scorer: Optional[NearServeScorer] = None

    with open(telemetry_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "meta" in obj and scorer is None:
                meta = obj["meta"]
                scorer = NearServeScorer(cfg, meta.get("exclusion_zones", []))
                continue
            if scorer is None:                     # telemetry with no header
                scorer = NearServeScorer(cfg, [])
            frames.append(scorer.update(obj))

    if not frames:
        raise RuntimeError(f"No telemetry records in {telemetry_path}")

    events = extract_events(frames, cfg.threshold, cfg.event_refract_s)

    stem = telemetry_path[:-len(".jsonl")] if telemetry_path.endswith(".jsonl") \
        else os.path.splitext(telemetry_path)[0]
    prob_path   = prob_path   or (stem + PROB_SUFFIX)
    events_path = events_path or (stem + EVENTS_SUFFIX)

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
