"""
orchestrator.py
===============
Turns the three detectors' events into the list of reel segments.  It detects
nothing itself; it applies what tennis knows and the detectors don't, and
makes the result watchable.

THE REEL IS BUILT FROM SERVES; ENDS ONLY TRIM.  Serves are found ~80-90% of
the time, ends ~50%, so every serve becomes a segment, and a missing end is
replaced by an estimate.  A missed end costs a little dead time; a missed
start would lose a whole point.

The recipe (`build_reel`):

   1  merge near + far serves into one timeline        merge_starts
   2  drop serves while a point is already live (far)  suppress_in_rally
   3  collapse serves closer than a point can be       drop_rapid_repeats
   4  rule 1: far toss evidence adjusts confidence     apply_toss_rule
   5  rule 2: a far serve among near serves is suspect apply_neighbour_rule
   6  relabel sides so service games are runs          enforce_service_runs
   7  record which half of the court each serve was    annotate_serve_court
   8  one segment per serve, with an end               pair_ends
   9  flag where deuce/ad says a point is missing      flag_missing_points
  10  clamp, join near neighbours, drop flashes        smooth
  11  add live play that no serve explained            recover_missed
  12  smooth again                                     smooth

Steps 4-7 and 9 only change labels, confidences and notes -- never which
points are in the reel.

Inputs: the three `<stem>_anya2_*.json` event files, tracks.npz, and point
end's live score.  Output: `<stem>_anya2_reel.json` (written by run.py).

Evidence and history: docs/anya2/NOTES.md, "orchestrator.py".
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import workdir
from pipeline.anya2 import court
from pipeline.anya2 import point_end
from pipeline.anya2 import signals
from pipeline.anya2 import tracks
from pipeline.anya2.contract import (FAR_SERVE, NEAR_SERVE, POINT_END,
                                     Event, load_events)

SEGMENTS_SUFFIX = "_anya2_reel.json"


@dataclass
class ReelConfig:
    # ── 1, 3: merging the serve streams ──────────────────────────────────
    merge_window_s: float = 2.5      # near + far firing this close = one serve
    min_point_gap_s: float = 8.0     # two starts closer than this = one point

    # ── 6: service runs ──────────────────────────────────────────────────
    min_service_run: int = 3         # shortest service game (except first/last)
    enforce_runs: bool = True
    drop_side_conflicts: bool = False   # tested and rejected; keep False

    # ── 8: pairing serves with ends ──────────────────────────────────────
    min_point_s: float = 2.5         # an end sooner than this is the serve itself
    max_point_s: float = 40.0        # an end later than this belongs to something else
    default_point_s: float = 9.0     # assumed point length with < 3 detected ends
    est_duration_pct: float = 85.0   # else: this percentile of the clip's own
    est_duration_pad_s: float = 1.5  #   detected point lengths, plus this
    next_start_guard_s: float = 4.0  # a segment's end stays this far before the next serve

    # ── 10: smoothing (pre/post roll is the biggest lever on what survives) ──
    pre_roll_s: float = 1.0          # footage kept before the serve
    post_roll_s: float = 1.0         # ...and after the end
    merge_gap_s: float = 6.0         # segments closer than this are joined
    min_segment_s: float = 4.0       # shorter segments are dropped (or padded)

    # ── 1, 8: extra thresholds on incoming events (0 = take all) ─────────
    near_threshold: float = 0.0
    far_threshold: float = 0.0
    end_threshold: float = 0.0

    # Which event files to read at all (a disabled detector's file may be stale).
    use_near: bool = True
    use_far: bool = True
    use_end: bool = True

    # ── 2: in-rally suppression ──────────────────────────────────────────
    live_gate_far: float = 0.8       # drop a far serve if the median live score
    live_gate_near: float = 0.0      #   over the previous 3 s exceeds this (0 = off)
    live_lookback_s: float = 3.0

    # ── 11: recovering points with no serve ──────────────────────────────
    recover_live_thr: float = 0.75   # live score counted as "clearly playing"
    recover_min_s: float = 3.0       # ...held at least this long

    # ── 4: rule 1, far toss evidence ─────────────────────────────────────
    rule1_enabled: bool = True
    toss_pose_lo: float = 0.30       # toss score at/below this = no support
    toss_pose_hi: float = 0.60       # ...at/above this = corroborating
    rule1_max_boost: float = 0.15    # most a strong toss adds to confidence
    rule1_max_penalty: float = 0.35  # most a missing toss subtracts

    # ── 5: rule 2, far serve among near serves ───────────────────────────
    rule2_enabled: bool = True
    rule2_neighbours: int = 4        # look at this many surrounding starts
    rule2_min_near: int = 3          # ...penalise if at least this many are near
    rule2_penalty: float = 0.30


@dataclass
class PointStart:
    t: float                             # serve time, seconds
    side: str                            # "near" | "far" | "both"
    p: float = 0.0                       # detector confidence
    track: Optional[int] = None          # player slot
    detected_side: str = ""              # side before step 6 relabelled it
    side_conflict: bool = False          # step 6 changed the side
    court_x: Optional[float] = None      # metres from the centre line (step 7)
    toss_pose: Optional[float] = None    # far_serve's toss score
    toss_combined: Optional[float] = None
    conf_adj: float = 0.0                # rules 1 and 2 add up here
    notes: List[str] = field(default_factory=list)

    @property
    def adjusted_p(self) -> float:
        """Confidence after rules 1 and 2, clipped to [0, 1]."""
        return float(min(1.0, max(0.0, self.p + self.conf_adj)))


@dataclass
class Segment:
    start: float                    # cut in (includes pre-roll)
    stop: float                     # cut out (includes post-roll)
    serve_t: float
    end_t: float
    side: str
    end_source: str                 # "detected" | "estimated" | "recovered"
    p: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def duration(self):
        return self.stop - self.start


# ═════════════════════════════════════════════════════════════════════════
# The recipe
# ═════════════════════════════════════════════════════════════════════════

def build_reel(video: str, cfg: Optional[ReelConfig] = None,
               tracks_npz=None, verbose: bool = True) -> Dict:
    """Read the detectors' events for `video` and return the reel as a dict
    (the content of `<stem>_anya2_reel.json`; `segments` is the cut list)."""
    cfg = cfg or ReelConfig()
    near, far, ends = _load_detector_events(video, cfg)
    duration = _clip_duration(video, tracks_npz)
    live, live_fps = _live_score(video, tracks_npz)

    starts = merge_starts(near, far, cfg)                              # 1
    n_merged = len(starts)
    if live is not None:
        starts = suppress_in_rally(starts, live, live_fps, cfg)        # 2
    n_in_rally = n_merged - len(starts)
    starts = drop_rapid_repeats(starts, cfg)                           # 3
    n_rapid = n_merged - n_in_rally - len(starts)
    # Rules 1-2 run before step 6: rule 2 reads the sides AS DETECTED.
    starts = apply_toss_rule(starts, cfg)                              # 4
    starts = apply_neighbour_rule(starts, cfg)                         # 5
    starts = enforce_service_runs(starts, cfg)                         # 6
    starts = annotate_serve_court(starts, video, tracks_npz)           # 7
    segments = pair_ends(starts, ends, cfg, duration)                  # 8
    flag_missing_points(starts, segments, cfg)                         # 9
    n_before_smoothing = len(segments)
    segments = smooth(segments, cfg, duration)                         # 10
    segments = recover_missed(segments, live, live_fps, cfg, duration) # 11
    segments = smooth(segments, cfg, duration)                         # 12

    counts = dict(n_near=len(near), n_far=len(far), n_ends=len(ends),
                  n_merged=n_merged, n_rapid=n_rapid, n_in_rally=n_in_rally,
                  n_before_smoothing=n_before_smoothing)
    result = _summary(video, duration, starts, segments, counts)
    if verbose:
        _print_summary(video, result, counts)
    return result


def _load_detector_events(video, cfg):
    """(near serves, far serves, point ends) from the cached event files."""
    folder = workdir.artifact_dir(video)
    stem = os.path.splitext(os.path.basename(video))[0]

    def load(suffix, kind):
        path = os.path.join(folder, f"{stem}{suffix}")
        if not os.path.isfile(path):
            return []
        return [e for e in load_events(path) if e.kind == kind]

    near = load("_anya2_near_serve.json", NEAR_SERVE) if cfg.use_near else []
    far = load("_anya2_far_serve.json", FAR_SERVE) if cfg.use_far else []
    ends = load("_anya2_point_end.json", POINT_END) if cfg.use_end else []
    return near, far, ends


def _clip_duration(video, tracks_npz):
    """Seconds covered by the tracks, or None if they cannot be read."""
    try:
        track_data = tracks.load(video, tracks_npz)
        return len(track_data["kp"]) / float(track_data["fps"])
    except Exception:
        return None


def _live_score(video, tracks_npz):
    """(point_end's live score, its fps), or (None, None) if unavailable.

    Always uses point_end's DEFAULT smoothing, whatever the run's config says
    (QUIRKS.md #3).
    """
    live = None
    try:
        parts = point_end.end_signal(video, tracks_npz)
        live = point_end.live_score(parts, video, tracks_npz)
        live_fps = float(parts["fps"])
    except Exception:
        live_fps = None
    return live, live_fps


# ═════════════════════════════════════════════════════════════════════════
# Steps 1-7: one timeline of point starts
# ═════════════════════════════════════════════════════════════════════════

def merge_starts(near: Sequence[Event], far: Sequence[Event],
                 cfg: ReelConfig) -> List[PointStart]:
    """1. Union of the two serve streams, sorted.  Serves within merge_window_s
    of the previous kept one are folded into it (side becomes "both", the
    EARLIER time is kept: the two detectors anchor on different moments)."""
    candidates = [PointStart(t=float(e.t), side="near", p=float(e.p), track=e.track)
                  for e in near if e.p >= cfg.near_threshold]
    candidates += [PointStart(t=float(e.t), side="far", p=float(e.p), track=e.track,
                              toss_pose=e.detail.get("toss"))
                   for e in far if e.p >= cfg.far_threshold]
    candidates.sort(key=lambda c: c.t)

    merged: List[PointStart] = []
    for candidate in candidates:
        if merged and candidate.t - merged[-1].t <= cfg.merge_window_s:
            previous = merged[-1]
            if previous.side != candidate.side:
                previous.side = "both"
            previous.p = max(previous.p, candidate.p)
            continue
        merged.append(candidate)
    return merged


def suppress_in_rally(starts: List[PointStart], live, fps: float,
                      cfg: ReelConfig) -> List[PointStart]:
    """2. Drop a serve if the median live score over the preceding
    `live_lookback_s` is above its side's gate (a point was already on)."""
    if live is None:
        return starts
    n = len(live)
    kept = []
    for start in starts:
        gate = cfg.live_gate_far if start.side == "far" else cfg.live_gate_near
        if gate <= 0:
            kept.append(start)
            continue
        window_start = max(0, int((start.t - cfg.live_lookback_s) * fps))
        window_stop = max(window_start + 1, min(n, int(start.t * fps)))
        if window_stop <= window_start:
            kept.append(start)
            continue
        if float(np.median(live[window_start:window_stop])) > gate:
            continue
        kept.append(start)
    return kept


def drop_rapid_repeats(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:
    """3. Starts within min_point_gap_s of the previous kept one are folded into
    it.  The FIRST is kept (a later echo would cut into the serve)."""
    kept: List[PointStart] = []
    for start in starts:
        if kept and start.t - kept[-1].t < cfg.min_point_gap_s:
            kept[-1].p = max(kept[-1].p, start.p)
            if kept[-1].side != start.side:
                kept[-1].side = "both"
            continue
        kept.append(start)
    return kept


def apply_toss_rule(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:
    """4. Rule 1: a far (or "both") serve's toss score moves its confidence.

    toss <= toss_pose_lo -> -rule1_max_penalty ... toss >= toss_pose_hi ->
    +rule1_max_boost, linear in between.  Never removes a start.
    """
    if not cfg.rule1_enabled:
        return starts
    for start in starts:
        if start.side == "near" or start.toss_pose is None:
            continue
        toss = float(start.toss_pose)
        # -1 (no toss seen) .. +1 (clearly a toss)
        support = 2.0 * float(np.clip((toss - cfg.toss_pose_lo)
                                      / max(cfg.toss_pose_hi - cfg.toss_pose_lo, 1e-6),
                                      0.0, 1.0)) - 1.0
        adjustment = (cfg.rule1_max_boost * support if support >= 0
                      else cfg.rule1_max_penalty * support)
        start.toss_combined = round(toss, 4)
        start.conf_adj += adjustment
        start.notes.append(f"toss {toss:.2f} -> {adjustment:+.2f}")
    return starts


def apply_neighbour_rule(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:
    """5. Rule 2: penalise a far serve when most of its neighbouring starts
    (as detected, before step 6) are near serves."""
    if not cfg.rule2_enabled or len(starts) < 3:
        return starts
    half = max(1, cfg.rule2_neighbours // 2)
    for i, start in enumerate(starts):
        if start.side != "far":
            continue
        neighbour_sides = [starts[j].side
                           for j in range(max(0, i - half), min(len(starts), i + half + 1))
                           if j != i]
        if not neighbour_sides:
            continue
        n_near = sum(1 for side in neighbour_sides if side == "near")
        if n_near >= cfg.rule2_min_near:
            start.conf_adj -= cfg.rule2_penalty
            start.notes.append(f"far serve among {n_near}/{len(neighbour_sides)} near neighbours "
                               f"-> -{cfg.rule2_penalty:.2f}")
    return starts


def enforce_service_runs(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:
    """6. Relabel sides so every service game is at least `min_service_run` long.

    Exact dynamic programme (not greedy) over states
    (side_index, run_length capped at R, still_in_first_run).  Relabelling a
    start costs 1 + its confidence; "both" is free either way.  The first and
    last runs may be short.  Only labels change; no start is dropped.
    """
    if not cfg.enforce_runs or len(starts) < 2:
        return starts
    sides = ("near", "far")
    R = max(1, cfg.min_service_run)

    def relabel_cost(start: PointStart, side: str) -> float:
        if start.side in ("both", side):
            return 0.0
        return 1.0 + start.p

    # table[k][(side_index, run_length, in_first_run)] = (cost, previous state)
    table: List[Dict[tuple, tuple]] = []
    table.append({(side_index, 1, True): (relabel_cost(starts[0], side), None)
                  for side_index, side in enumerate(sides)})
    for k in range(1, len(starts)):
        column: Dict[tuple, tuple] = {}
        for state, (cost_so_far, _) in table[-1].items():
            side_index, run_length, in_first_run = state
            # Same server continues.
            stay = (side_index, min(run_length + 1, R), in_first_run)
            cost = cost_so_far + relabel_cost(starts[k], sides[side_index])
            if stay not in column or cost < column[stay][0]:
                column[stay] = (cost, state)
            # Server changes -- allowed once the run is long enough (or in the first run).
            if run_length >= R or in_first_run:
                other = 1 - side_index
                switch = (other, 1, False)
                cost = cost_so_far + relabel_cost(starts[k], sides[other])
                if switch not in column or cost < column[switch][0]:
                    column[switch] = (cost, state)
        table.append(column)

    state = min(table[-1].items(), key=lambda item: item[1][0])[0]
    labels: List[str] = []
    for k in range(len(starts) - 1, -1, -1):
        labels.append(sides[state[0]])
        state = table[k][state][1]
    labels.reverse()
    for start, side in zip(starts, labels):
        start.detected_side = start.side
        start.side_conflict = start.side not in (side, "both")
        start.side = side
    if cfg.drop_side_conflicts:
        starts = [s for s in starts if not s.side_conflict]
    return starts


def annotate_serve_court(starts: List[PointStart], video: str,
                         tracks_npz=None) -> List[PointStart]:
    """7. Record court_x (metres from the centre line) of the server over the
    3 s after each serve -- the deepest player on the serving side."""
    try:
        track_data = tracks.load(video, tracks_npz)
    except Exception:
        return starts
    fps = float(track_data["fps"])
    court_xy = track_data["court"]
    n = len(court_xy)
    for start in starts:
        slots = tracks.NEAR_SLOTS if start.side in ("near", "both") else tracks.FAR_SLOTS
        window_start = max(0, int(start.t * fps))
        window_stop = min(n, int((start.t + 3.0) * fps))
        if window_stop <= window_start:
            continue
        best = None                     # (depth key, median x)
        for slot in slots:
            xs = court_xy[window_start:window_stop, slot, 0]
            ys = court_xy[window_start:window_stop, slot, 1]
            ok = np.isfinite(xs) & np.isfinite(ys)
            if not ok.any():
                continue
            depth = float(np.median(ys[ok]))
            deeper_is_larger = -depth if slots is tracks.NEAR_SLOTS else depth
            if best is None or deeper_is_larger > best[0]:
                best = (deeper_is_larger, float(np.median(xs[ok])))
        if best is not None:
            start.court_x = best[1] - court.COURT_W / 2.0
    return starts


# ═════════════════════════════════════════════════════════════════════════
# Steps 8-9: segments
# ═════════════════════════════════════════════════════════════════════════

def pair_ends(starts: Sequence[PointStart], ends: Sequence[Event],
              cfg: ReelConfig, duration: Optional[float] = None) -> List[Segment]:
    """8. One segment per start, always.

    End = the first detected end in [serve + min_point_s, serve + max_point_s],
    and before (next serve - next_start_guard_s).  If none, the end is
    ESTIMATED as serve + a typical point length for this clip (see
    `estimate_point_s`), bounded the same way.  Two passes: the first only
    learns the typical length from the ends that were found.
    """
    end_times = sorted(float(e.t) for e in ends if e.p >= cfg.end_threshold)
    end_times = np.array(end_times) if end_times else np.zeros(0)
    first_pass = _pair_once(starts, end_times, cfg, duration, cfg.default_point_s)
    return _pair_once(starts, end_times, cfg, duration, estimate_point_s(first_pass, cfg))


def estimate_point_s(segments: Sequence[Segment], cfg: ReelConfig) -> float:
    """Typical point length for this clip: est_duration_pct of the detected
    durations + pad, or default_point_s with fewer than 3 detected."""
    detected = [s.end_t - s.serve_t for s in segments
                if s.end_source == "detected" and s.end_t > s.serve_t]
    if len(detected) < 3:
        return cfg.default_point_s
    return float(np.percentile(detected, cfg.est_duration_pct)) + cfg.est_duration_pad_s


def _pair_once(starts, end_times, cfg, duration, assumed_point_s):
    segments: List[Segment] = []
    for i, start in enumerate(starts):
        next_serve = starts[i + 1].t if i + 1 < len(starts) else None
        earliest = start.t + cfg.min_point_s
        latest = start.t + cfg.max_point_s
        if next_serve is not None:
            latest = min(latest, next_serve - cfg.next_start_guard_s)
        end_t, source = None, ""
        if end_times.size and latest > earliest:
            in_window = end_times[(end_times >= earliest) & (end_times <= latest)]
            if in_window.size:
                end_t, source = float(in_window[0]), "detected"
        if end_t is None:
            estimate = start.t + assumed_point_s
            if next_serve is not None:
                estimate = min(estimate, next_serve - cfg.next_start_guard_s)
            end_t, source = max(earliest, estimate), "estimated"
            if duration is not None:
                end_t = min(end_t, duration - 0.1)
        segments.append(Segment(start=start.t - cfg.pre_roll_s,
                                stop=end_t + cfg.post_roll_s,
                                serve_t=start.t, end_t=end_t, side=start.side,
                                end_source=source, p=start.p))
    return segments


def flag_missing_points(starts: List[PointStart], segments: List[Segment],
                        cfg: ReelConfig) -> None:
    """9. Note (only) where two consecutive same-side serves came from the same
    half of the court -- deuce/ad alternation suggests a point was missed."""
    for i in range(1, len(starts)):
        previous, current = starts[i - 1], starts[i]
        if previous.side != current.side or previous.court_x is None or current.court_x is None:
            continue
        if abs(previous.court_x) < 0.35 or abs(current.court_x) < 0.35:
            continue                       # too near the centre mark to call
        if np.sign(previous.court_x) == np.sign(current.court_x):
            segments[i].notes.append("same-court-as-previous: a point may be missing here")


# ═════════════════════════════════════════════════════════════════════════
# Steps 10-12: making it watchable
# ═════════════════════════════════════════════════════════════════════════

def smooth(segments: List[Segment], cfg: ReelConfig,
           duration: Optional[float] = None) -> List[Segment]:
    """10/12. Three passes, in order:

      a. clamp to the clip; split any overlap down the middle
      b. join segments whose gap is <= merge_gap_s (a cut costs attention)
      c. drop segments shorter than min_segment_s, unless they hold a
         detected end -- those are padded to the minimum instead
    """
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: s.start)
    for s in segments:
        s.start = max(0.0, s.start)
        if duration is not None:
            s.stop = min(s.stop, duration)
        if s.stop <= s.start:
            s.stop = s.start + cfg.min_segment_s

    for a, b in zip(segments, segments[1:]):
        if b.start < a.stop:
            middle = 0.5 * (a.stop + b.start)
            a.stop = b.start = middle

    joined: List[Segment] = [segments[0]]
    for s in segments[1:]:
        previous = joined[-1]
        if s.start - previous.stop <= cfg.merge_gap_s:
            previous.stop = max(previous.stop, s.stop)
            previous.end_t = s.end_t
            previous.notes.append(f"joined with the point at {s.serve_t:.0f}s")
            if previous.side != s.side:
                previous.side = "both"
            continue
        joined.append(s)

    kept: List[Segment] = []
    for s in joined:
        if s.duration < cfg.min_segment_s:
            if s.end_source == "detected":
                s.stop = s.start + cfg.min_segment_s
            else:
                continue
        kept.append(s)
    return kept


def recover_missed(segments: List[Segment], live, fps: float, cfg: ReelConfig,
                   duration: Optional[float] = None) -> List[Segment]:
    """11. Add a segment for every run of live score >= recover_live_thr lasting
    at least recover_min_s that no existing segment covers.  These carry no
    side (no serve was seen) and end_source "recovered"."""
    if live is None or not fps:
        return segments
    n = len(live)
    covered = np.zeros(n, dtype=bool)
    for s in segments:
        a, b = max(0, int(s.start * fps)), min(n, int(s.stop * fps))
        if b > a:
            covered[a:b] = True
    clearly_live = np.asarray(live) >= cfg.recover_live_thr
    min_samples = max(1, int(cfg.recover_min_s * fps))
    added: List[Segment] = []
    for run_start, run_stop in signals.true_runs(clearly_live & ~covered):
        if run_stop - run_start < min_samples:
            continue
        t0, t1 = run_start / fps, run_stop / fps
        segment = Segment(start=max(0.0, t0 - cfg.pre_roll_s),
                          stop=t1 + cfg.post_roll_s,
                          serve_t=t0, end_t=t1, side="unknown",
                          end_source="recovered", p=0.0,
                          notes=["recovered from live play; no serve was detected"])
        if duration is not None:
            segment.stop = min(segment.stop, duration)
        added.append(segment)
    return sorted(segments + added, key=lambda s: s.start)


# ═════════════════════════════════════════════════════════════════════════
# Output
# ═════════════════════════════════════════════════════════════════════════

def _summary(video, duration, starts, segments, counts) -> Dict:
    """The reel JSON: diagnostics, the final starts, and the segments."""
    # "next-serve" and "default" are always 0 -- see QUIRKS.md #4.
    end_sources = {"detected": 0, "next-serve": 0, "default": 0}
    for s in segments:
        end_sources[s.end_source] = end_sources.get(s.end_source, 0) + 1
    reel_s = sum(s.duration for s in segments)
    return {
        "video": video, "duration_s": duration,
        "n_rule1_toss_adjusted": sum(1 for p in starts if p.toss_combined is not None),
        "n_rule2_neighbour_penalised": sum(1 for p in starts
                                           if any("near neighbours" in note for note in p.notes)),
        "n_confidence_demoted": sum(1 for p in starts if p.conf_adj < -0.01),
        "starts": [{"t": round(p.t, 2), "side": p.side, "p": round(p.p, 3),
                    "adjusted_p": round(p.adjusted_p, 3),
                    "toss": p.toss_combined, "notes": p.notes}
                   for p in starts],
        "n_near": counts["n_near"], "n_far": counts["n_far"], "n_ends": counts["n_ends"],
        "n_starts_merged": counts["n_merged"],
        "n_rapid_repeats_dropped": counts["n_rapid"],
        "n_dropped_in_rally": counts["n_in_rally"],
        "n_side_relabelled": sum(1 for p in starts if p.side_conflict),
        "n_segments": len(segments), "n_before_smoothing": counts["n_before_smoothing"],
        "end_source": end_sources, "reel_s": reel_s,
        "compression": (reel_s / duration) if duration else None,
        "suspected_missing_points": sum(1 for s in segments
                                        if any("same-court" in note for note in s.notes)),
        "n_recovered_from_live": sum(1 for s in segments if s.end_source == "recovered"),
        "segments": [asdict(s) for s in segments],
    }


def _print_summary(video, result, counts):
    stem = os.path.splitext(os.path.basename(video))[0]
    sources = result["end_source"]
    duration, reel_s = result["duration_s"], result["reel_s"]
    print(f"[reel] {stem}: {counts['n_near']} near + {counts['n_far']} far serves -> "
          f"{counts['n_merged']} starts ({counts['n_rapid']} rapid repeats dropped, "
          f"{result['n_side_relabelled']} sides relabelled)")
    print(f"[reel] ends: {sources.get('detected', 0)} detected, "
          f"{sources.get('estimated', 0)} estimated)")
    print(f"[reel] {counts['n_before_smoothing']} points -> {result['n_segments']} segments "
          f"after smoothing, {reel_s:.0f}s of {duration:.0f}s "
          f"({100 * reel_s / duration:.0f}%)" if duration else "")
    if result["suspected_missing_points"]:
        print(f"[reel] {result['suspected_missing_points']} segment(s) flagged: "
              f"deuce/ad alternation suggests a missing point")


# ═════════════════════════════════════════════════════════════════════════
# Scoring a reel against labels (evaluation only)
# ═════════════════════════════════════════════════════════════════════════

def score_reel(res: Dict, clip_dir: str) -> Dict:
    """Is the tennis all there, how much waiting is left in, how often does it cut?"""
    from parse_ground_truth import load_rallies
    rallies = load_rallies(clip_dir)
    segments = [(s["start"], s["stop"]) for s in res["segments"]]
    span_s = rallies[-1]["end_s"] - rallies[0]["start_s"]

    whole = partial = 0
    live_kept = live_total = 0.0
    for rally in rallies:
        a, b = rally["start_s"], rally["end_s"]
        live_total += b - a
        kept = sum(max(0.0, min(b, q) - max(a, p)) for p, q in segments)
        live_kept += min(kept, b - a)
        if kept >= (b - a) - 0.25:
            whole += 1
        elif kept > 0:
            partial += 1

    reel_s = sum(q - p for p, q in segments)
    dead_kept = max(0.0, reel_s - live_kept)
    return {
        "clip": os.path.basename(clip_dir),
        "n_points": len(rallies), "n_segments": len(segments),
        "points_whole": whole, "points_partial": partial,
        "points_missing": len(rallies) - whole - partial,
        "live_retained": live_kept / live_total if live_total else float("nan"),
        "reel_s": reel_s, "span_s": span_s,
        "compression": reel_s / span_s if span_s else float("nan"),
        "dead_kept_s": dead_kept,
        "dead_per_point_s": dead_kept / max(len(rallies), 1),
        "cuts_per_min": len(segments) / (span_s / 60.0) if span_s else float("nan"),
    }


def _fmt_score(s: Dict) -> str:
    return (f"  {s['clip']:>4}  pts {s['n_points']:>3}  segs {s['n_segments']:>3} | "
            f"whole {s['points_whole']:>3} partial {s['points_partial']:>3} "
            f"missing {s['points_missing']:>3} | "
            f"live kept {100*s['live_retained']:5.1f}%  "
            f"reel {100*s['compression']:5.1f}% of span  "
            f"dead/pt {s['dead_per_point_s']:5.1f}s  cuts/min {s['cuts_per_min']:4.1f}")


def main():
    ap = argparse.ArgumentParser(description="Agent 4.  Takes the three detectors' "
                                 "event streams and produces the reel.")
    # The same one-or-many GoPro chapters build_reel takes; the joined path is
    # what the cached artifacts are keyed by.
    ap.add_argument("video", nargs="+")
    ap.add_argument("--json", default=None)
    ap.add_argument("--pre-roll", type=float, default=None)
    ap.add_argument("--post-roll", type=float, default=None)
    ap.add_argument("--merge-gap", type=float, default=None)
    a = ap.parse_args()
    cfg = ReelConfig()
    if a.pre_roll is not None:
        cfg.pre_roll_s = a.pre_roll
    if a.post_roll is not None:
        cfg.post_roll_s = a.post_roll
    if a.merge_gap is not None:
        cfg.merge_gap_s = a.merge_gap
    from pipeline import join
    video = join.resolve_input(a.video)
    result = build_reel(video, cfg)
    out = a.json or os.path.join(workdir.artifact_dir(video),
                                 os.path.splitext(os.path.basename(video))[0]
                                 + SEGMENTS_SUFFIX)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"[reel] wrote {out}")


if __name__ == "__main__":
    main()
