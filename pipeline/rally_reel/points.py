"""Turn serve events + walking intervals into rally segments.

Point starts come from the two serve detectors (near and far).  Point ends
come from the walking classifier, used as a dead-time proxy: the players
walk when the ball is not in play, so the first sustained walk after a serve
is where the rally stopped.

This is the only genuinely new logic in the package — every other stage is
an existing module called in order.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

from .config import ReelConfig


@dataclass
class PointStart:
    t: float
    side: str              # "near" | "far" | "both"
    confidence: str        # detector-reported, or "" when it has no notion
    score: Optional[float] = None
    detected_side: Optional[str] = None   # side before the service-run pass
    side_conflict: bool = False           # detectors disagreed with the run


@dataclass
class RallySegment:
    index: int
    side: str
    serve_t: float
    end_t: float
    start: float           # serve_t - pre_roll, clamped
    end: float             # end_t + post_roll, clamped
    end_method: str        # "walk" | "next-serve" | "cap"
    confidence: str

    def as_dict(self) -> Dict:
        return asdict(self)


def merge_serve_starts(far_serves: Sequence[Dict],
                       near_events: Sequence[Dict],
                       cfg: ReelConfig) -> List[PointStart]:
    """Union of the two detectors' events, deduped into one timeline.

    When both sides fire for the same serve the earlier timestamp wins: the
    far gate triggers on the toss and the near gate on the strike, so the
    earlier one is closer to the true start of the service motion.
    """
    cands: List[PointStart] = []
    if cfg.use_far:
        for s in far_serves:
            cands.append(PointStart(t=float(s["timestamp"]), side="far",
                                    confidence=s.get("confidence", "")))
    if cfg.use_near:
        for e in near_events:
            if float(e.get("p", 0.0)) < cfg.near_threshold:
                continue
            cands.append(PointStart(t=float(e["t"]), side="near",
                                    confidence="", score=float(e.get("p", 0.0))))

    cands.sort(key=lambda c: c.t)

    merged: List[PointStart] = []
    for c in cands:
        if merged and c.t - merged[-1].t <= cfg.merge_window_s:
            prev = merged[-1]
            if prev.side != c.side:
                prev.side = "both"
            # Keep the stronger confidence label when one side has one.
            if c.confidence == "HIGH":
                prev.confidence = "HIGH"
            if prev.score is None:
                prev.score = c.score
            continue
        merged.append(c)
    return merged


def enforce_service_runs(starts: List[PointStart],
                         cfg: ReelConfig) -> List[PointStart]:
    """Relabels point starts so each service block runs >= min_service_run.

    One player serves a whole game, so the side sequence should look like
    FFFFFFFF NNNNNNNN, not FFNFFFFNF.  This finds the labelling closest to
    what the detectors reported that still obeys that structure, which
    turns an isolated disagreeing detection into either a corrected label
    or (with drop_side_conflicts) a discard.

    Exact rather than greedy: a segmental DP over
    (side, run-so-far, still-in-first-run), where a switch is only legal
    once the current run has reached the minimum.  A greedy left-to-right
    pass would commit to an early wrong side and never recover.  The first
    and last runs are exempt from the minimum — a clip rarely starts or
    ends on a game boundary.
    """
    if not cfg.enforce_service_runs or len(starts) < 2:
        return starts

    sides = ("near", "far")
    R = max(1, cfg.min_service_run)
    B = max(1, min(cfg.min_boundary_run, R))

    def mismatch(ps: PointStart, side: str) -> float:
        # "both" means each detector saw it, so it supports either side.
        if ps.side == "both" or ps.side == side:
            return 0.0
        return (cfg.conflict_cost_high if ps.confidence == "HIGH"
                else cfg.conflict_cost_default)

    # One table per point start, so the backpointers stay resolvable.
    # state: (side_idx, min(run, R), in_first_run) -> (cost, prev_state)
    table: List[Dict[tuple, tuple]] = []
    col: Dict[tuple, tuple] = {}
    for si, s in enumerate(sides):
        col[(si, 1, True)] = (mismatch(starts[0], s), None)
    table.append(col)

    for i in range(1, len(starts)):
        nxt: Dict[tuple, tuple] = {}
        for state, (cost, _) in table[-1].items():
            si, run, first = state
            # stay on the same side
            key = (si, min(run + 1, R), first)
            c = cost + mismatch(starts[i], sides[si])
            if key not in nxt or c < nxt[key][0]:
                nxt[key] = (c, state)
            # switch sides — legal once the run is long enough, or while
            # still inside the opening run, which is truncated by the start
            # of the recording and so only owes min_boundary_run
            if run >= R or (first and run >= B):
                oi = 1 - si
                key = (oi, 1, False)
                c = cost + mismatch(starts[i], sides[oi])
                if key not in nxt or c < nxt[key][0]:
                    nxt[key] = (c, state)
        table.append(nxt)

    # The final run is truncated by the end of the recording, so it owes
    # min_boundary_run rather than the full minimum — but not less, or a
    # lone stray detection at the clip edge becomes its own "game".
    final = {s: v for s, v in table[-1].items() if s[1] >= min(B, R)}
    state = min((final or table[-1]).items(), key=lambda kv: kv[1][0])[0]
    labels: List[str] = []
    for i in range(len(starts) - 1, -1, -1):
        labels.append(sides[state[0]])
        state = table[i][state][1]
    labels.reverse()

    for ps, side in zip(starts, labels):
        ps.detected_side = ps.side
        ps.side_conflict = ps.side not in (side, "both")
        ps.side = side
    return starts


def usable_walk_intervals(intervals: Sequence[Dict],
                          cfg: ReelConfig) -> List[Dict]:
    """Drops walk runs too short or too poorly tracked to trust."""
    return [w for w in intervals
            if w.get("duration_s", 0.0) >= cfg.walk_min_duration_s
            and w.get("detection_coverage", 1.0) >= cfg.walk_min_coverage]


def walk_onsets(intervals: Sequence[Dict], cfg: ReelConfig) -> List[tuple]:
    """(t, "walk") for each usable walk interval start."""
    return [(float(w["start_second"]), "walk")
            for w in usable_walk_intervals(intervals, cfg)]


def find_point_end(serve_t: float, next_serve_t: Optional[float],
                   dead_onsets: Sequence[tuple], cfg: ReelConfig,
                   duration: float):
    """First dead-time onset after the serve, else a bounded fallback.

    `dead_onsets` is a sorted [(t, source)] list of every signal that says
    play has stopped — by default walking alone, which is the designated
    dead-time proxy.  Where walking is silent the point falls through to
    the next-serve guard or the cap, which is why those two methods show up
    in the output whenever the walking model is mistuned for the clip.

    Returns (end_t, method).
    """
    hard_cap = min(serve_t + cfg.point_max_s, duration)
    if next_serve_t is not None:
        hard_cap = min(hard_cap, next_serve_t - cfg.next_serve_guard_s)
    hard_cap = max(hard_cap, serve_t + cfg.point_min_s)

    earliest = serve_t + cfg.point_min_s
    for t, source in dead_onsets:
        if t < earliest:
            continue
        if t > hard_cap:
            break
        return min(t, hard_cap), source

    # Nothing said play stopped before the cap.
    if next_serve_t is not None and hard_cap < serve_t + cfg.point_max_s:
        return hard_cap, "next-serve"
    return hard_cap, "cap"


def build_segments(starts: Sequence[PointStart], dead_onsets: Sequence[tuple],
                   duration: float, cfg: ReelConfig) -> List[RallySegment]:
    dead_onsets = sorted(dead_onsets, key=lambda x: x[0])

    segments: List[RallySegment] = []
    for i, ps in enumerate(starts):
        next_t = starts[i + 1].t if i + 1 < len(starts) else None
        end_t, method = find_point_end(ps.t, next_t, dead_onsets, cfg, duration)

        start = max(0.0, ps.t - cfg.pre_roll_s)
        end = min(duration, end_t + cfg.post_roll_s)
        if end - start < cfg.min_segment_s:
            continue
        segments.append(RallySegment(
            index=len(segments), side=ps.side, serve_t=ps.t, end_t=end_t,
            start=start, end=end, end_method=method,
            confidence=ps.confidence,
        ))

    return _merge_overlaps(segments, cfg)


def _merge_overlaps(segments: List[RallySegment],
                    cfg: ReelConfig) -> List[RallySegment]:
    """Joins segments that overlap or nearly touch after roll padding."""
    if not segments:
        return []
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        if seg.start - prev.end <= cfg.merge_gap_s:
            # end_method has to follow end_t.  Keeping the first member's
            # method while taking the last member's end time describes an end
            # that was discarded — on Data/75 that reported 39 segments ending
            # on the next-serve guard when the true number was zero.
            if seg.end_t > prev.end_t:
                prev.end_method = seg.end_method
            prev.end = max(prev.end, seg.end)
            prev.end_t = max(prev.end_t, seg.end_t)
            if prev.side != seg.side:
                prev.side = "both"
            if seg.confidence == "HIGH":
                prev.confidence = "HIGH"
            continue
        out.append(seg)
    for i, seg in enumerate(out):
        seg.index = i
    return out
