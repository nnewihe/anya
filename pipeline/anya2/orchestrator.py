"""
orchestrator.py
===============
Agent 4.  Takes the three detectors' event streams and produces the reel.

Its job is NOT to detect anything.  It is to impose the structure that tennis
has and the detectors cannot see, and to make the result watchable.

THE ONE MEASUREMENT THAT DECIDES THE ARCHITECTURE
-------------------------------------------------
Over the 13-clip corpus the three detectors do not have comparable recall:

    near serve   90.7%      far serve  82.2%      point end  49.6%

Serves are found; ends are found half the time.  So THE REEL IS BUILT FROM
STARTS, and ends only trim.  A point whose end was never detected still becomes
a segment -- it runs to just before the next serve instead.  Building from ends,
or requiring a start/end pair, would silently drop half the points, and the
brief is that every point is in the reel.

That asymmetry is also why a missed end is cheap here and a missed START is not:
a missed end costs some dead time at the tail of one segment, while a missed
start loses the whole point.

WHAT TENNIS KNOWS THAT THE DETECTORS DO NOT
-------------------------------------------
  SERVICE RUNS.  One player serves a whole game, so the side sequence looks
      like NNNNN FFFFF NNNNN, never NNFNN.  An isolated side flip is a
      mislabel, and relabelling it is free -- it changes which detector gets
      credit, not whether a segment exists.

  DEUCE/AD ALTERNATION.  Within a game the server alternates courts every
      point, always.  In the video it is far weaker than that: re-measured over
      all 14 clip-sides with six or more labelled serves, the server's court-x
      flips sign across consecutive serves a MEAN OF 63% of the time, spanning
      20% (clip 43 near) to 91% (clip 58 near).  An earlier note here quoted the
      91% as representative; it is the best case, not the corpus.  Median-
      filtering the court track does not improve it at all (63% either way),
      because the per-serve reading is already a median over three seconds.  At
      63% this is barely above chance, so it FLAGS a suspected missing point and
      nothing more -- it must never delete a detection, and the flag should be
      read as a hint rather than a finding.

  A POINT IS NOT LIVE TWICE.  A serve detected while a point is already in
      progress is spurious, and this is where most of the detectors' false
      positives live: 23 of 24 near and 79% of far false positives fall inside
      a live point.  Both detectors declare `windows="between_points"` for
      exactly this reason and neither can enforce it alone.  The orchestrator
      can, because it knows where the previous point ended.

  RHYTHM.  Points are seconds long and separated by tens of seconds; a serve
      3 s after the last one is not a new point.

SMOOTHNESS IS A FIRST-CLASS OUTPUT, NOT A SIDE EFFECT
-----------------------------------------------------
The brief puts viewing experience above precision, and that changes two
choices a pure accuracy metric would make differently:

  * WHEN IN DOUBT, KEEP FOOTAGE.  An extra second of a player walking is
    barely noticeable; a cut that lands mid-rally is jarring and loses the
    point.  Pre-roll and post-roll are generous, and a synthesized end runs
    long rather than short.

  * A CUT COSTS SOMETHING.  Two segments separated by a couple of seconds of
    dead time are worse to watch than one continuous segment that includes
    those seconds -- the cut draws attention to itself for no gain.  So
    segments closer than `merge_gap_s` are joined, and very short segments are
    dropped or absorbed rather than flashed on screen.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.anya2 import court as C
from pipeline import workdir as WD
from pipeline.anya2.signals import runs as S_runs
from pipeline.anya2 import point_end as PE
from pipeline.anya2 import tracks as T
from pipeline.anya2.contract import (FAR_SERVE, NEAR_SERVE, POINT_END,
                                     Event, load_events)

SEGMENTS_SUFFIX = "_anya2_reel.json"


@dataclass
class ReelConfig:
    # ── merging the two serve streams ────────────────────────────────────
    merge_window_s: float = 2.5      # both detectors firing this close are one
                                     # serve.  Wider than it looks because the
                                     # two anchor on different events (hands
                                     # together vs trophy onset).
    min_point_gap_s: float = 8.0     # two starts closer than this are one point.
                                     # A real inter-point gap is 15-25 s (the
                                     # ATP shot clock is 25); 8 is deliberately
                                     # permissive so a genuinely quick point is
                                     # never merged away.

    # ── service structure ────────────────────────────────────────────────
    min_service_run: int = 3         # a game is 4+ points, but detections are
                                     # missed, so requiring 4 would force real
                                     # games to merge.  3 is the smallest run
                                     # that still rejects an isolated flip.
    enforce_runs: bool = True
    drop_side_conflicts: bool = False
    # Tested and rejected.  The idea was that on a near service game a lone far
    # detection is a phantom rather than a mislabel -- the far player is
    # RETURNING, and a return is a serve motion -- so it should be dropped, not
    # relabelled.  Measured over the corpus it changes almost nothing (whole
    # points 207 -> 206, reel 61.6% -> 60.9%) and it does not fix the case that
    # motivated it: on Data/21, where 9 of 21 merged starts are spurious far
    # detections, the reel is identical at 3 segments either way, because the
    # phantoms cluster into runs of 2-3 that any min_service_run of 3-5 accepts
    # as a legitimate game.  Left off, so the rule "structure relabels, it never
    # drops a point" holds without exception.

    # ── pairing starts with ends ─────────────────────────────────────────
    min_point_s: float = 2.5         # an end this soon after the serve is the
                                     # serve motion itself, not the point
    max_point_s: float = 40.0        # ...and beyond this the "end" is a missed
                                     # end and a later, unrelated quiet
    default_point_s: float = 9.0     # fallback when a clip has no detected ends
                                     # at all to estimate from
    est_duration_pct: float = 85.0   # when an end is missing, assume a point
                                     # this percentile of the clip's OWN
                                     # detected point durations.  Per clip
                                     # because rally length is a property of the
                                     # players, and high because overrunning
                                     # costs dead time while underrunning cuts
                                     # live tennis.
    est_duration_pad_s: float = 1.5
    next_start_guard_s: float = 4.0  # never run a segment closer than this to
                                     # the next serve
    end_pick_confident: bool = True  # choose the highest-confidence end in the
                                     # window rather than the earliest.  See
                                     # `_pair_once` for why the earliest was
                                     # the wrong rule and what `p` measures.
                                     # False restores the old behaviour, so the
                                     # two are A/B-comparable on the corpus.
    end_pick_tol: float = 0.05       # confidences within this are a tie, and a
                                     # tie goes to the earlier end

    # ── smoothness ───────────────────────────────────────────────────────
    # Roll is the single biggest lever on how much tennis survives, and it is
    # far more effective than any end-estimation tuning.  Measured over all 236
    # labelled points -- whole points / live retained / reel as a share of the
    # source:
    #
    #     pre 2.5 post 2.0    179    93.1%    54.9%
    #     pre 2.5 post 3.0    196    94.6%    57.3%
    #     pre 3.0 post 3.5    202    94.9%    59.5%
    #     pre 3.5 post 4.0    207    95.3%    61.6%     <-- here
    #     pre 3.5 post 5.0    212    95.8%    63.7%
    #
    # For comparison, moving the end-duration percentile from 85 to 97 buys 2
    # whole points.  The ends are close to right; they were simply being cut a
    # second or two early, and roll is what fixes that.
    #
    # 3.5/4.0 is chosen for the brief's stated priority -- viewing experience
    # first, dead time second.  Lower both to tighten the reel; the table above
    # is the exchange rate.
    # Set to 1.0/1.0 at the user's direction, tightening the reel.  The table
    # above is what that costs against the corpus; re-measured at 1.0/1.0 below.
    #
    # RE-MEASURED at post 1.0, over all 208 labelled ends on 11 clips, with the
    # three point-end fixes (far veto 0.8, turn hold 2.0, confident pairing) on.
    # `trunc` counts points losing more than 2 s off the TAIL -- the error an
    # early end makes, which `eval.py` cannot see and `score_reel` now can:
    #
    #     post   whole  partial  trunc  trunc_s  dead_s  live kept
    #     1.0     128      69      17      93     1213     86.9%
    #     2.0     138      60       9      59     1376     88.7%   <-- here
    #     3.0     146      52       5      32     1529     89.7%
    #
    # So post-roll at 1.0 WAS costing truncations, and the exchange rate is
    # good: 1.0 -> 2.0 halves the truncated points and buys 10 whole points for
    # 163 s across 11 clips -- about 0.8 s of dead time per point.  Set to 2.0
    # at the user's direction on that table.
    #
    # Note that roll and the point-end objective are NOT independent, though it
    # is natural to assume they are: PES reads `end_t`, so roll cannot shift a
    # scored value directly.  It moves the score anyway -- 1.0 -> 2.0 took PES
    # +0.051 -> +0.008 and the within-2 s count 75 -> 71 -- because roll changes
    # where segments END, `smooth` then merges a different set of them, and the
    # segment that covers a given rally (and therefore the `end_t` scored for
    # it) is not the same one.  The coupling runs through `merge_gap_s`, not
    # through the arithmetic.
    #
    # So the two still cannot be tuned in one pass, but neither can they be
    # tuned in strict isolation: re-check PES after moving roll.
    #
    # AND UNDER `point_end_score` -- the objective that now governs -- post-roll
    # is the single dominant parameter, worth several times any detector knob:
    #
    #     post    PES     0..5s   >2s early   median err
    #     1.0   -0.583      70        26        +1.9 s
    #     1.5   -0.509      73        24        +2.7 s
    #     2.0   -0.405      73        19        +3.5 s
    #     2.5   -0.313      76        16        +4.2 s
    #     3.0   -0.250      78        15        +4.9 s
    #     4.0   -0.101      82        11        +5.9 s   <-- optimum
    #     5.0   -0.175      60        13        +8.0 s
    #
    # That ranking is a direct consequence of the objective's shape: an
    # exponential penalty for cutting early against a merely linear decay for
    # running late will always buy insurance with post-roll.  Whether 4.0 is
    # the right PRODUCT answer is a separate question from whether it maximises
    # this number -- it adds ~3 s of dead time to every point in the reel.
    pre_roll_s: float = 1.0
    post_roll_s: float = 2.0
    merge_gap_s: float = 6.0         # segments closer than this are joined
                                     # rather than cut apart
    min_segment_s: float = 4.0       # anything shorter is a flash, not a point

    # ── thresholds on the incoming streams ───────────────────────────────
    near_threshold: float = 0.0      # the detectors already threshold; these
    far_threshold: float = 0.0       # exist so a caller can be stricter
    end_threshold: float = 0.0

    # Which streams to read AT ALL.  A disabled agent must not contribute, and
    # that cannot be left to simply not regenerating its file: the events are
    # cached beside the video, so a stale run would silently keep feeding the
    # reel after the agent was switched off.
    use_near: bool = True
    use_far: bool = True
    use_end: bool = True

    # ── in-rally suppression ─────────────────────────────────────────────
    # A serve struck while a point is already live is spurious, and agent 3
    # already computes the signal that says so.  Measured over the corpus, the
    # median live score in the 3 s BEFORE a detection separates true serves from
    # false ones far better on the far side than the near:
    #
    #     far   live > 0.8 drops 26% of false positives for 4% of true serves
    #     near  live > 0.8 drops 17% of false positives for 10% of true serves
    #
    # That asymmetry is the taxonomy showing through: 79% of far false
    # positives are the RETURNER mid-rally, so they sit on live play, while
    # near false positives are the server's own repeated motions in dead time
    # and look exactly like a serve from here.  So the gate is applied to the
    # far stream only -- on the near stream it would cost more than it buys.
    live_gate_far: float = 0.8
    live_gate_near: float = 0.0      # off: measured AUC 55%, not worth it
    live_lookback_s: float = 3.0

    # ── recovering points with no serve detection ────────────────────────
    recover_live_thr: float = 0.75   # live score above which play is clearly on
    recover_min_s: float = 3.0       # ...held this long to count as a point

    # ── rule 1: toss evidence adjusts a far serve's confidence ───────────
    # Two independent readings of the same event, neither of which the far
    # detector folds into its own score:
    #
    # The evidence is `far_serve.toss_score` -- the tossing arm's own motion,
    # AUC 75% against far false positives and correlated only +0.04 with the
    # serve score, so it is genuinely new evidence.
    #
    # A BALL toss detector was built and removed.  Re-aimed at the tossing wrist
    # with native-resolution SAHI tiling it detected the ball well (8 frames per
    # serve against 0-0.5 for a head-centred ROI), and its arc separated true
    # from false serves at AUC 75% -- but it correlated +0.49 with the pose toss
    # and the best blend of the two was no better than pose alone.  Two readings
    # of the same event, and the arm is far easier to see than a 3 px ball.  It
    # cost tiled inference over ~18 native frames per candidate for nothing.
    #
    # This ADJUSTS confidence rather than filtering.  A hard gate on either one
    # costs real serves (dropping pose-toss below 0.40 removes 54% of false
    # positives but 18% of true ones), and the brief is precision WITHOUT
    # giving up recall -- so a weak toss reading demotes a candidate and lets
    # the rest of the pipeline decide, instead of deleting it outright.
    rule1_enabled: bool = True
    toss_pose_lo: float = 0.30       # pose-toss at or below this is unsupportive
    toss_pose_hi: float = 0.60       # ...and at or above it, corroborating
    rule1_max_boost: float = 0.15    # most a strong toss may add
    rule1_max_penalty: float = 0.35  # ...and most a missing one may subtract.
                                     # Asymmetric on purpose: seeing a toss is
                                     # strong evidence FOR a serve, while not
                                     # seeing one is weak evidence against --
                                     # the ball is a few pixels at that range
                                     # and the arm may be occluded.

    # ── rule 2: a far serve among near serves is suspect ─────────────────
    # Measured over the corpus: requiring 3 of a far candidate's 4 nearest
    # neighbouring starts to be NEAR-side flags 18% of the remaining far false
    # positives and ZERO true far serves.  At 2 of 4 it reaches 27% of false
    # positives but starts costing 7% of true ones.
    #
    # So the 3-of-4 setting is free, and this too adjusts rather than deletes:
    # one player does serve a whole game, but detections are missed, and a real
    # far game whose first serves were not detected would look exactly like
    # this.  Demoting is recoverable; deleting is not.
    rule2_enabled: bool = True
    rule2_neighbours: int = 4        # how many surrounding starts to look at
    rule2_min_near: int = 3          # ...of which this many must be near-side
    rule2_penalty: float = 0.30


@dataclass
class PointStart:
    t: float
    side: str                       # "near" | "far" | "both"
    p: float = 0.0
    track: Optional[int] = None
    detected_side: str = ""
    side_conflict: bool = False
    court_x: Optional[float] = None
    toss_pose: Optional[float] = None    # far_serve's pose toss score
    toss_combined: Optional[float] = None
    conf_adj: float = 0.0                # rules 1 and 2 accumulate here
    notes: List[str] = field(default_factory=list)

    @property
    def adjusted_p(self) -> float:
        """Confidence after the orchestrator's rules, clipped to [0, 1]."""
        return float(min(1.0, max(0.0, self.p + self.conf_adj)))


@dataclass
class Segment:
    start: float                    # cut in (with pre-roll)
    stop: float                     # cut out (with post-roll)
    serve_t: float
    end_t: float
    side: str
    end_source: str                 # "detected" | "next-serve" | "default"
    p: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def duration(self):
        return self.stop - self.start


# ── 1. one timeline of serves ────────────────────────────────────────────

def merge_starts(near: Sequence[Event], far: Sequence[Event],
                 cfg: ReelConfig) -> List[PointStart]:
    """Union the two serve streams, deduped.

    When both sides fire for the same serve the EARLIER timestamp wins and the
    side becomes "both".  Earlier because the two detectors anchor on different
    moments of the same action -- the near one on the hands leaving the grip,
    the far one on the trophy -- so the earlier is closer to where the point
    actually began, and erring early only costs pre-roll.
    """
    cands = [PointStart(t=float(e.t), side="near", p=float(e.p), track=e.track)
             for e in near if e.p >= cfg.near_threshold]
    cands += [PointStart(t=float(e.t), side="far", p=float(e.p), track=e.track,
                         toss_pose=e.detail.get("toss"))
              for e in far if e.p >= cfg.far_threshold]
    cands.sort(key=lambda c: c.t)

    out: List[PointStart] = []
    for c in cands:
        if out and c.t - out[-1].t <= cfg.merge_window_s:
            prev = out[-1]
            if prev.side != c.side:
                prev.side = "both"
            prev.p = max(prev.p, c.p)
            continue
        out.append(c)
    return out


def drop_rapid_repeats(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:
    """Collapse starts closer together than a point can possibly be.

    Keeps the FIRST, not the strongest.  A cluster of serve detections around
    one point is usually the real serve plus an echo of the same motion, and the
    first is the one that opens the point -- keeping a later, higher-scoring
    echo would cut into the serve itself.
    """
    out: List[PointStart] = []
    for s in starts:
        if out and s.t - out[-1].t < cfg.min_point_gap_s:
            out[-1].p = max(out[-1].p, s.p)
            if out[-1].side != s.side:
                out[-1].side = "both"
            continue
        out.append(s)
    return out


def apply_toss_rule(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:
    """RULE 1 -- adjust a far serve's confidence by its toss evidence.

    Pose evidence only -- `far_serve`'s toss score.  The result shifts confidence
    within [-rule1_max_penalty, +rule1_max_boost] and never removes a start; see
    the config for why the bound is asymmetric, and for why the ball half of
    this was measured and removed.
    """
    if not cfg.rule1_enabled:
        return starts
    for ps in starts:
        if ps.side == "near" or ps.toss_pose is None:
            continue
        t = float(ps.toss_pose)
        # -1 (no toss seen) .. +1 (clearly a toss)
        u = 2.0 * float(np.clip((t - cfg.toss_pose_lo)
                                / max(cfg.toss_pose_hi - cfg.toss_pose_lo, 1e-6),
                                0.0, 1.0)) - 1.0
        adj = (cfg.rule1_max_boost * u if u >= 0
               else cfg.rule1_max_penalty * u)
        ps.toss_combined = round(t, 4)
        ps.conf_adj += adj
        ps.notes.append(f"toss {t:.2f} -> {adj:+.2f}")
    return starts


def apply_neighbour_rule(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:
    """RULE 2 -- a far serve surrounded by near serves is suspect.

    Uses the sides AS DETECTED, before the service-run DP relabels anything:
    the DP's own output would make this circular, since it has already decided
    the run each start belongs to.
    """
    if not cfg.rule2_enabled or len(starts) < 3:
        return starts
    k = max(1, cfg.rule2_neighbours // 2)
    for i, ps in enumerate(starts):
        if ps.side != "far":
            continue
        nb = [starts[j].side for j in range(max(0, i - k), min(len(starts), i + k + 1))
              if j != i]
        if not nb:
            continue
        n_near = sum(1 for x in nb if x == "near")
        if n_near >= cfg.rule2_min_near:
            ps.conf_adj -= cfg.rule2_penalty
            ps.notes.append(f"far serve among {n_near}/{len(nb)} near neighbours "
                            f"-> -{cfg.rule2_penalty:.2f}")
    return starts


# ── 2. service runs ──────────────────────────────────────────────────────

def enforce_service_runs(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:
    """Relabel sides so every service block runs at least `min_service_run`.

    A segmental DP over (side, run-so-far, still-in-the-first-run), exact rather
    than greedy: a greedy left-to-right pass commits to an early wrong side and
    can never recover.  The first and last runs are exempt from the minimum,
    because a clip rarely starts or ends on a game boundary.

    This only ever CHANGES A LABEL, never drops a start.  Which side served is
    metadata; that a point happened is not, and the brief is that every point is
    in the reel.
    """
    if not cfg.enforce_runs or len(starts) < 2:
        return starts
    sides = ("near", "far")
    R = max(1, cfg.min_service_run)

    def cost(ps: PointStart, side: str) -> float:
        if ps.side in ("both", side):
            return 0.0
        return 1.0 + ps.p          # a confident disagreement costs more to overrule

    table: List[Dict[tuple, tuple]] = []
    col = {(i, 1, True): (cost(starts[0], s), None) for i, s in enumerate(sides)}
    table.append(col)
    for k in range(1, len(starts)):
        nxt: Dict[tuple, tuple] = {}
        for st, (c0, _) in table[-1].items():
            si, run, first = st
            key = (si, min(run + 1, R), first)
            c = c0 + cost(starts[k], sides[si])
            if key not in nxt or c < nxt[key][0]:
                nxt[key] = (c, st)
            if run >= R or first:
                oi = 1 - si
                key = (oi, 1, False)
                c = c0 + cost(starts[k], sides[oi])
                if key not in nxt or c < nxt[key][0]:
                    nxt[key] = (c, st)
        table.append(nxt)
    state = min(table[-1].items(), key=lambda kv: kv[1][0])[0]
    labels: List[str] = []
    for k in range(len(starts) - 1, -1, -1):
        labels.append(sides[state[0]])
        state = table[k][state][1]
    labels.reverse()
    for ps, side in zip(starts, labels):
        ps.detected_side = ps.side
        ps.side_conflict = ps.side not in (side, "both")
        ps.side = side
    if cfg.drop_side_conflicts:
        starts = [p for p in starts if not p.side_conflict]
    return starts


def annotate_serve_court(starts: List[PointStart], video: str,
                         tracks_npz=None) -> List[PointStart]:
    """Record which half of the court each serve was struck from.

    Within a game the server alternates deuce/ad EVERY point, without
    exception, so consecutive serves in one service run should flip sign.  In
    the video they flip a mean of 63% of the time across 14 clip-sides, ranging
    20% to 91% -- weak enough that this is a hint and not a finding.

    So this is recorded and used to FLAG, never to delete: a pair of
    consecutive same-court serves inside one service run suggests a point was
    missed between them, which is a recall question, not a reason to throw away
    a detection that is probably real.
    """
    try:
        z = T.load(video, tracks_npz)
    except Exception:
        return starts
    fps = float(z["fps"])
    ct, bb = z["court"], z["bbox"]
    n = len(ct)
    for ps in starts:
        slots = T.NEAR_SLOTS if ps.side in ("near", "both") else T.FAR_SLOTS
        a, b = max(0, int(ps.t * fps)), min(n, int((ps.t + 3.0) * fps))
        if b <= a:
            continue
        best = None
        for s in slots:
            xs = ct[a:b, s, 0]
            ys = ct[a:b, s, 1]
            ok = np.isfinite(xs) & np.isfinite(ys)
            if not ok.any():
                continue
            depth = float(np.median(ys[ok]))
            key = -depth if slots is T.NEAR_SLOTS else depth
            if best is None or key > best[0]:
                best = (key, float(np.median(xs[ok])))
        if best is not None:
            ps.court_x = best[1] - C.COURT_W / 2.0
    return starts


# ── 3. pair starts with ends ─────────────────────────────────────────────

def suppress_in_rally(starts: List[PointStart], live, fps: float,
                      cfg: ReelConfig) -> List[PointStart]:
    """Drop serves struck while a point was already live. See ReelConfig."""
    if live is None:
        return starts
    n = len(live)
    out = []
    for ps in starts:
        thr = cfg.live_gate_far if ps.side == "far" else cfg.live_gate_near
        if thr <= 0:
            out.append(ps)
            continue
        a = max(0, int((ps.t - cfg.live_lookback_s) * fps))
        b = max(a + 1, min(n, int(ps.t * fps)))
        if b <= a:
            out.append(ps)
            continue
        if float(np.median(live[a:b])) > thr:
            continue                      # a point was already in progress
        out.append(ps)
    return out


def estimate_point_s(segs: Sequence[Segment], cfg: ReelConfig) -> float:
    """How long to assume a point ran when its end was never detected.

    Taken from THIS clip's detected point durations, because rally length is a
    property of the players rather than of tennis, and at a high percentile
    because the two errors are not symmetric: overrunning leaves dead time in
    the reel, underrunning cuts live tennis out of it.
    """
    got = [s.end_t - s.serve_t for s in segs
           if s.end_source == "detected" and s.end_t > s.serve_t]
    if len(got) < 3:
        return cfg.default_point_s
    return float(np.percentile(got, cfg.est_duration_pct)) + cfg.est_duration_pad_s


def pair_ends(starts: Sequence[PointStart], ends: Sequence[Event],
              cfg: ReelConfig, duration: Optional[float] = None) -> List[Segment]:
    """One segment per start.  Every start becomes a segment, always.

    The end is chosen in this order:

      DETECTED   the MOST CONFIDENT end event in the plausible window for this
                 point -- later than `min_point_s` (before that it is the serve
                 motion itself), earlier than `max_point_s` (after that it is a
                 missed end and some later, unrelated quiet), and before the
                 next serve.  Most confident rather than earliest because at
                 40.2% precision most candidates in a window are artefacts, and
                 picking the earliest chose them by construction; see
                 `_pair_once`.

      ESTIMATED  no end was detected in the window, so the point is assumed to
                 have run for `estimate_point_s` -- a high percentile of THIS
                 clip's own detected point durations -- bounded by the next
                 serve.  Running to the next serve instead would keep every
                 inter-point gap on the ~50% of points whose end is missed.

    Note what this does NOT do: it never drops a start for want of an end.
    Point-end recall is 49.6% against 90.7%/82.2% for the serves, so requiring a
    pair would discard half the points.
    """
    kept = sorted(((float(e.t), float(e.p)) for e in ends
                   if e.p >= cfg.end_threshold), key=lambda x: x[0])
    et_arr = np.array([t for t, _ in kept]) if kept else np.zeros(0)
    ep_arr = np.array([p for _, p in kept]) if kept else np.zeros(0)

    # Two passes: the first only to learn this clip's typical point length from
    # the ends that WERE detected, the second to use it for the ones that were
    # not.  One pass cannot do it -- the estimate is derived from the same
    # pairing it feeds.
    segs = _pair_once(starts, et_arr, ep_arr, cfg, duration, cfg.default_point_s)
    return _pair_once(starts, et_arr, ep_arr, cfg, duration,
                      estimate_point_s(segs, cfg))


def _pair_once(starts, et_arr, ep_arr, cfg, duration, est_s):
    segs: List[Segment] = []
    for i, ps in enumerate(starts):
        nxt = starts[i + 1].t if i + 1 < len(starts) else None
        lo = ps.t + cfg.min_point_s
        hi = ps.t + cfg.max_point_s
        if nxt is not None:
            hi = min(hi, nxt - cfg.next_start_guard_s)
        end_t, src = None, ""
        if et_arr.size and hi > lo:
            sel = (et_arr >= lo) & (et_arr <= hi)
            cand = et_arr[sel]
            if cand.size:
                # THE MOST CONFIDENT END IN THE WINDOW, NOT THE FIRST ONE.
                # Taking `cand[0]` handed the point to the earliest candidate
                # unconditionally, and with point-end precision at 40.2% most
                # candidates are false positives -- so the rule was
                # systematically choosing a mid-rally artefact over the real
                # end whenever both fell in the window.  That is the truncation
                # mechanism, and the eval never saw it: `eval.py` counts a
                # truncation only for a MATCHED end landing early, so the 60%
                # of ends that match no label were invisible to the one column
                # that was supposed to catch this.
                #
                # `detect_ends` already computes the discriminator and the
                # orchestrator was throwing it away.  Its `p` is how far the
                # live score falls AND STAYS fallen over the next 4 s: a real
                # end drops to the floor and scores near 1, while a mid-rally
                # dip is followed by more play and scores low.  Picking the
                # argmax uses evidence that was already paid for.
                cp = ep_arr[sel]
                if cfg.end_pick_confident:
                    # Near-equal confidences resolve to the EARLIER end, which
                    # is why this is a tolerance and not a bare argmax: `p` is a
                    # mean over a 4 s window and separates a real end from an
                    # artefact by tenths, not by hundredths.  Inside the
                    # tolerance the two candidates are not distinguishable and
                    # the earlier one costs less dead time.
                    ok = np.flatnonzero(cp >= cp.max() - cfg.end_pick_tol)
                    j = int(ok[0])
                else:
                    j = 0
                end_t, src = float(cand[j]), "detected"
        if end_t is None:
            # An UNDETECTED end must not mean "run to the next serve": with
            # point-end recall at 49.6% that would keep every inter-point gap on
            # half the points, which is the opposite of the brief.  Assume a
            # point of the clip's own typical length instead, bounded by the
            # next serve.
            est = ps.t + est_s
            if nxt is not None:
                est = min(est, nxt - cfg.next_start_guard_s)
            end_t, src = max(lo, est), "estimated"
            if duration is not None:
                end_t = min(end_t, duration - 0.1)
        segs.append(Segment(start=ps.t - cfg.pre_roll_s,
                            stop=end_t + cfg.post_roll_s,
                            serve_t=ps.t, end_t=end_t, side=ps.side,
                            end_source=src, p=ps.p))
    return segs


def flag_missing_points(starts: List[PointStart], segs: List[Segment],
                        cfg: ReelConfig) -> None:
    """Note where the deuce/ad alternation says a point is missing.

    Advisory only -- it adds a note, changes no timing.  Two consecutive serves
    from the same court inside one service run means an odd number of points
    happened between them, and the simplest explanation is one we did not
    detect.  Surfacing it is how a recall gap becomes visible in the output
    instead of silently absent.
    """
    for i in range(1, len(starts)):
        a, b = starts[i - 1], starts[i]
        if a.side != b.side or a.court_x is None or b.court_x is None:
            continue
        if abs(a.court_x) < 0.35 or abs(b.court_x) < 0.35:
            continue                       # too near the centre mark to call
        if np.sign(a.court_x) == np.sign(b.court_x):
            segs[i].notes.append("same-court-as-previous: a point may be missing here")


# ── 4. smoothing ─────────────────────────────────────────────────────────

def smooth(segs: List[Segment], cfg: ReelConfig,
           duration: Optional[float] = None) -> List[Segment]:
    """Make the sequence watchable: no overlaps, no flashes, no needless cuts.

    Three passes, in this order, because each can create work for the next:

      1. CLAMP AND ORDER -- a segment may not start before the clip or overlap
         its neighbour.  Overlap is resolved by splitting the difference, which
         keeps both points rather than truncating one.
      2. JOIN NEAR NEIGHBOURS -- if the dead gap between two segments is under
         `merge_gap_s`, the cut costs more attention than the gap costs time.
         Keep it continuous.
      3. DROP FLASHES -- a segment under `min_segment_s` is a blink.  It is
         dropped only if it holds no detected end; one that does is a real short
         point and is PADDED to the minimum instead.
    """
    if not segs:
        return []
    segs = sorted(segs, key=lambda s: s.start)
    for s in segs:
        s.start = max(0.0, s.start)
        if duration is not None:
            s.stop = min(s.stop, duration)
        if s.stop <= s.start:
            s.stop = s.start + cfg.min_segment_s

    for a, b in zip(segs, segs[1:]):
        if b.start < a.stop:
            mid = 0.5 * (a.stop + b.start)
            a.stop = b.start = mid

    joined: List[Segment] = [segs[0]]
    for s in segs[1:]:
        prev = joined[-1]
        if s.start - prev.stop <= cfg.merge_gap_s:
            prev.stop = max(prev.stop, s.stop)
            prev.end_t = s.end_t
            prev.notes.append(f"joined with the point at {s.serve_t:.0f}s")
            if prev.side != s.side:
                prev.side = "both"
            continue
        joined.append(s)

    out: List[Segment] = []
    for s in joined:
        if s.duration < cfg.min_segment_s:
            if s.end_source == "detected":
                s.stop = s.start + cfg.min_segment_s
            else:
                continue
        out.append(s)
    return out


# ── 5. the whole thing ───────────────────────────────────────────────────

def build_reel(video: str, cfg: Optional[ReelConfig] = None,
               tracks_npz=None, verbose: bool = True) -> Dict:
    cfg = cfg or ReelConfig()
    d = WD.artifact_dir(video)
    stem = os.path.splitext(os.path.basename(video))[0]

    def _load(suffix, kind):
        p = os.path.join(d, f"{stem}{suffix}")
        if not os.path.isfile(p):
            return []
        return [e for e in load_events(p) if e.kind == kind]

    near = _load("_anya2_near_serve.json", NEAR_SERVE) if cfg.use_near else []
    far = _load("_anya2_far_serve.json", FAR_SERVE) if cfg.use_far else []
    ends = _load("_anya2_point_end.json", POINT_END) if cfg.use_end else []

    duration = None
    try:
        z = T.load(video, tracks_npz)
        duration = len(z["kp"]) / float(z["fps"])
    except Exception:
        pass

    live = None
    try:
        parts = PE.end_signal(video, tracks_npz)
        live = PE.live_score(parts, video, tracks_npz)
        live_fps = float(parts["fps"])
    except Exception:
        live_fps = None

    starts = merge_starts(near, far, cfg)
    n_merged = len(starts)
    if live is not None:
        starts = suppress_in_rally(starts, live, live_fps, cfg)
    n_inrally = n_merged - len(starts)
    starts = drop_rapid_repeats(starts, cfg)
    n_rapid = n_merged - n_inrally - len(starts)
    # Rules 1 and 2 run BEFORE the service-run DP: rule 2 reads the sides as
    # detected, and the DP would otherwise have already overwritten them.
    starts = apply_toss_rule(starts, cfg)
    starts = apply_neighbour_rule(starts, cfg)
    starts = enforce_service_runs(starts, cfg)
    starts = annotate_serve_court(starts, video, tracks_npz)
    segs = pair_ends(starts, ends, cfg, duration)
    flag_missing_points(starts, segs, cfg)
    n_raw = len(segs)
    segs = smooth(segs, cfg, duration)
    n_before_recover = len(segs)
    segs = recover_missed(segs, live, live_fps, cfg, duration)
    segs = smooth(segs, cfg, duration)
    n_recovered = sum(1 for s in segs if s.end_source == "recovered")

    src = {"detected": 0, "next-serve": 0, "default": 0}
    for s in segs:
        src[s.end_source] = src.get(s.end_source, 0) + 1
    total = sum(s.duration for s in segs)
    flips = sum(1 for s in segs if any("same-court" in n for n in s.notes))
    conflicts = sum(1 for p in starts if p.side_conflict)
    rule1 = sum(1 for p_ in starts if p_.toss_combined is not None)
    rule2 = sum(1 for p_ in starts
                if any("near neighbours" in n for n in p_.notes))
    demoted = sum(1 for p_ in starts if p_.conf_adj < -0.01)
    res = {
        "video": video, "duration_s": duration,
        "n_rule1_toss_adjusted": rule1,
        "n_rule2_neighbour_penalised": rule2,
        "n_confidence_demoted": demoted,
        "starts": [{"t": round(p_.t, 2), "side": p_.side, "p": round(p_.p, 3),
                    "adjusted_p": round(p_.adjusted_p, 3),
                    "toss": p_.toss_combined, "notes": p_.notes}
                   for p_ in starts],
        "n_near": len(near), "n_far": len(far), "n_ends": len(ends),
        "n_starts_merged": n_merged, "n_rapid_repeats_dropped": n_rapid,
        "n_dropped_in_rally": n_inrally,
        "n_side_relabelled": conflicts,
        "n_segments": len(segs), "n_before_smoothing": n_raw,
        "end_source": src, "reel_s": total,
        "compression": (total / duration) if duration else None,
        "suspected_missing_points": flips,
        "n_recovered_from_live": n_recovered,
        "segments": [asdict(s) for s in segs],
    }
    if verbose:
        print(f"[reel] {stem}: {len(near)} near + {len(far)} far serves -> "
              f"{n_merged} starts ({n_rapid} rapid repeats dropped, "
              f"{conflicts} sides relabelled)")
        print(f"[reel] ends: {src.get('detected',0)} detected, "
              f"{src.get('estimated',0)} estimated)")
        print(f"[reel] {n_raw} points -> {len(segs)} segments after smoothing, "
              f"{total:.0f}s of {duration:.0f}s "
              f"({100*total/duration:.0f}%)" if duration else "")
        if flips:
            print(f"[reel] {flips} segment(s) flagged: deuce/ad alternation "
                  f"suggests a missing point")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("video")
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
    res = build_reel(a.video, cfg)
    out = a.json or os.path.join(WD.artifact_dir(a.video),
                                 os.path.splitext(os.path.basename(a.video))[0]
                                 + SEGMENTS_SUFFIX)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[reel] wrote {out}")




# ── 5b. recovering points nobody detected ────────────────────────────────

def recover_missed(segs: List[Segment], live, fps: float, cfg: ReelConfig,
                   duration: Optional[float] = None) -> List[Segment]:
    """Add segments for sustained live play that no serve detector caught.

    The brief is that every point is in the reel, and the serve detectors miss
    some -- 15 of 236 labelled points had no detection at all before this pass.
    Nothing in the serve streams can recover those, but agent 3's LIVE SCORE is
    a different measurement entirely: it separates live play from dead time at
    AUC 86.7%, and it does not care whether a serve was seen.

    So any run of sustained live play that no segment already covers is treated
    as a point that was missed, and kept.  It is marked `recovered` and carries
    no side, because there is no serve detection to say who served -- the reel
    gains the tennis without inventing a fact about it.

    This is deliberately the LAST pass: it fills gaps rather than competing with
    the detectors, so a point that WAS detected keeps its proper serve-anchored
    boundaries and only genuinely unexplained play is added.
    """
    if live is None or not fps:
        return segs
    n = len(live)
    covered = np.zeros(n, dtype=bool)
    for s in segs:
        a, b = max(0, int(s.start * fps)), min(n, int(s.stop * fps))
        if b > a:
            covered[a:b] = True
    hot = np.asarray(live) >= cfg.recover_live_thr
    wmin = max(1, int(cfg.recover_min_s * fps))
    added: List[Segment] = []
    for a, b in S_runs(hot & ~covered):
        if b - a < wmin:
            continue
        t0, t1 = a / fps, b / fps
        seg = Segment(start=max(0.0, t0 - cfg.pre_roll_s),
                      stop=t1 + cfg.post_roll_s,
                      serve_t=t0, end_t=t1, side="unknown",
                      end_source="recovered", p=0.0,
                      notes=["recovered from live play; no serve was detected"])
        if duration is not None:
            seg.stop = min(seg.stop, duration)
        added.append(seg)
    return sorted(segs + added, key=lambda s: s.start)


# ── 6. scoring a reel ────────────────────────────────────────────────────
# Serve and end detectors are scored on event timing; a REEL is not. What
# matters to a viewer is whether the tennis is all there, how much waiting was
# left in, and how often the picture cuts. Those are the numbers below.

# ── the point-end objective ──────────────────────────────────────────────
# THE metric, at the user's direction, and the one to tune against.  Everything
# else in `score_reel` is diagnostic: it explains why this number moved.
#
# It scores THE POINT END THE SYSTEM CHOSE -- `Segment.end_t`, before pre- or
# post-roll is applied -- against the labelled end.  Deliberately not the cut
# point: roll is a separate, later decision about how much air to leave around
# a correct answer, and folding it in here lets a tuning run paper over a bad
# end by padding it.  Optimising this number optimises point-end detection and
# nothing else.
#
# The shape, with e = end_t - gt_end:
#
#   |e| <= PLATEAU_S            score 1.  A point end is not a timestamp, it is
#                               a moment a couple of seconds wide, and inside
#                               that band there is nothing to choose between
#                               two answers.
#
#   e > PLATEAU_S               decays LINEARLY from 1 to 0 at the next
#                               labelled point start.  Late is waste, not
#                               damage, and it never goes negative.
#
#   e < -PLATEAU_S              decays EXPONENTIALLY from 1 and CROSSES INTO
#                               PENALTY: 2 - exp((|e| - PLATEAU_S) / EARLY_TAU).
#                               Zero at 3.39 s early, -5.4 at 6 s, and falling
#                               fast after that.  Cutting early deletes tennis
#                               and the damage compounds.
PLATEAU_S = 2.0
EARLY_TAU = 2.0
# ...and this is where the penalty SATURATES, which the corpus forced rather
# than the design asking for it.  Uncapped, the exponential stops measuring the
# reel and starts measuring its single worst point: on the previous objective
# one point of 208, cut 21.4 s early, was 88.2% of the whole number and the
# worst three were 98.9%.
#
# The cap is also the honest shape.  The exponential encodes "losing the last
# seconds of a rally loses the winner", and that argument runs out once the
# point is gone: an end 15 s early and one 21 s early have destroyed the same
# rally, and calling the second 11x worse states something about tennis that is
# not true.
#
# Set to None for the pure exponential; `pes_errors` is reported either way, so
# the uncapped number is always recoverable.
EARLY_MAX_PENALTY = 10.0
# A rally with no segment at all is scored as if its end landed at its own
# start -- early by the full rally duration.  Scoring it zero instead would let
# a tuning run quietly DISCARD the points it finds hard rather than risk
# mistiming them, and buy a better score by showing less tennis.
MISS_IS_FULL_TRUNCATION = True


def _pe_score(e: float, gap: float) -> float:
    """Score one point end. `e` is signed error, `gap` is end -> next start."""
    if abs(e) <= PLATEAU_S:
        return 1.0
    if e > PLATEAU_S:
        # Linear from 1 at +PLATEAU_S to 0 at the next point's start.  A gap at
        # or inside the plateau leaves no room to decay across, so anything
        # beyond it is simply worthless rather than dividing by ~zero.
        span = gap - PLATEAU_S
        if span <= 1e-6:
            return 0.0
        return max(0.0, 1.0 - (e - PLATEAU_S) / span)
    v = 2.0 - float(np.exp((-e - PLATEAU_S) / EARLY_TAU))
    if EARLY_MAX_PENALTY is not None:
        v = max(v, -EARLY_MAX_PENALTY)
    return v


def point_end_score(segs, rallies) -> Dict:
    """Per-point closeness of the CHOSEN point end to the labelled one.

    `segs` is the orchestrator's segment dicts (needs `end_t`, `start`, `stop`
    and `end_source`); `rallies` is `load_rallies`' output.  Returns the mean
    score plus the signed per-point errors, so a caller can see the
    distribution rather than only its mean.
    """
    if not rallies:
        return {"score": float("nan"), "errors": [], "n": 0}
    gaps = [rallies[i + 1]["start_s"] - rallies[i]["end_s"]
            for i in range(len(rallies) - 1)]
    # The last rally has no next start to decay toward, so it borrows the
    # clip's own typical gap.  Median rather than mean: one changeover is
    # minutes long and would make the last point almost impossible to score
    # badly.
    fallback = float(np.median(gaps)) if gaps else 20.0

    scores, errors, sources = [], [], []
    n_early = n_missed = 0
    for i, x in enumerate(rallies):
        a, b = x["start_s"], x["end_s"]
        nxt = rallies[i + 1]["start_s"] if i + 1 < len(rallies) else b + fallback
        gap = max(nxt - b, 1e-6)
        # The segment that COVERS this rally supplies the end.  Several can
        # touch one point once merging has run, and it is the last of them that
        # carries the end the system settled on.
        over = [s for s in segs if s["stop"] > a and s["start"] < b]
        if not over:
            n_missed += 1
            sources.append("missed")
            if not MISS_IS_FULL_TRUNCATION:
                scores.append(0.0)
                errors.append(float("nan"))
                continue
            e = -(b - a)
        else:
            top = max(over, key=lambda s: s["end_t"])
            e = top["end_t"] - b
            sources.append(top["end_source"])
        errors.append(e)
        if e < -PLATEAU_S:
            n_early += 1
        scores.append(_pe_score(e, gap))
    return {
        "score": float(np.mean(scores)),
        "errors": errors,
        "sources": sources,
        "n": len(scores),
        "n_early": n_early,
        "n_missed": n_missed,
        "n_plateau": sum(1 for e in errors
                         if np.isfinite(e) and abs(e) <= PLATEAU_S),
        "median_err": float(np.median([e for e in errors if np.isfinite(e)]))
                      if errors else float("nan"),
    }


def score_reel(res: Dict, clip_dir: str) -> Dict:
    """Measure a reel against the labelled rallies."""
    from parse_ground_truth import load_rallies
    r = load_rallies(clip_dir)
    segs = [(s["start"], s["stop"]) for s in res["segments"]]
    span = (r[0]["start_s"], r[-1]["end_s"])
    span_s = span[1] - span[0]

    full = partial = 0
    live_kept = live_total = 0.0
    # TRUNCATION IS MEASURED HERE OR IT IS NOT MEASURED AT ALL.  `eval.py`'s
    # truncation column only ever sees ends that MATCHED a label, so at 40.2%
    # point-end precision the majority of emitted ends -- and every one that
    # cuts a rally short by landing mid-point -- were invisible to it.  That is
    # why the corpus read "zero truncations" while the reel truncated.
    #
    # The reel-level question is different and blunt: for a rally the reel got
    # SOME of, how many seconds of its tail are missing.  Tail specifically,
    # because that is the error an early end makes; a late start is a separate
    # error that `points_partial` already folds in.
    trunc_s = 0.0
    trunc_n = 0
    for x in r:
        a, b = x["start_s"], x["end_s"]
        live_total += b - a
        kept = sum(max(0.0, min(b, q) - max(a, p)) for p, q in segs)
        live_kept += min(kept, b - a)
        if kept > 0:
            last = max((min(b, q) for p, q in segs if min(b, q) > max(a, p)),
                       default=a)
            lost = max(0.0, b - last)
            trunc_s += lost
            if lost > 2.0:
                trunc_n += 1
        if kept >= (b - a) - 0.25:
            full += 1
        elif kept > 0:
            partial += 1

    reel_s = sum(q - p for p, q in segs)
    dead_kept = max(0.0, reel_s - live_kept)
    # The objective reads `end_t` and `end_source`, so it gets the segment
    # dicts rather than the (start, stop) pairs the coverage maths above uses.
    pes = point_end_score(res["segments"], r)
    return {
        "clip": os.path.basename(clip_dir),
        "point_end_score": pes["score"],
        "pes_early": pes["n_early"], "pes_missed": pes["n_missed"],
        "pes_median_err": pes["median_err"],
        "pes_plateau": pes["n_plateau"],
        "pes_errors": pes["errors"],
        "pes_sources": pes["sources"],
        "n_points": len(r), "n_segments": len(segs),
        "points_whole": full, "points_partial": partial,
        "points_missing": len(r) - full - partial,
        "live_retained": live_kept / live_total if live_total else float("nan"),
        "reel_s": reel_s, "span_s": span_s,
        "compression": reel_s / span_s if span_s else float("nan"),
        "dead_kept_s": dead_kept,
        "dead_per_point_s": dead_kept / max(len(r), 1),
        "trunc_s": trunc_s,
        "trunc_per_point_s": trunc_s / max(len(r), 1),
        "points_truncated": trunc_n,
        "cuts_per_min": len(segs) / (span_s / 60.0) if span_s else float("nan"),
    }


def _fmt_score(s: Dict) -> str:
    # The objective leads; everything after the first bar is why it moved.
    return (f"  {s['clip']:>4}  PES {s['point_end_score']:+6.3f} "
            f"(early {s['pes_early']:>3} miss {s['pes_missed']:>2} "
            f"med {s['pes_median_err']:+5.1f}s) | "
            f"pts {s['n_points']:>3}  segs {s['n_segments']:>3} | "
            f"whole {s['points_whole']:>3} partial {s['points_partial']:>3} "
            f"missing {s['points_missing']:>3} | "
            f"live kept {100*s['live_retained']:5.1f}%  "
            f"reel {100*s['compression']:5.1f}% of span  "
            f"dead/pt {s['dead_per_point_s']:5.1f}s  "
            f"trunc {s['points_truncated']:>3} ({s['trunc_per_point_s']:4.1f}s/pt)  "
            f"cuts/min {s['cuts_per_min']:4.1f}")


if __name__ == "__main__":
    main()
