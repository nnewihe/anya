"""
arbitrate.py
============
Decide point starts and point ends from RAW detector candidates plus what is
known about how tennis is played.

Why this module exists
----------------------
The three detectors each applied a threshold and a refractory window and wrote
only the survivors.  Measured over the 13 trusted clips, the candidates they
discarded contain **13.6 points of point-start recall**: the raw pools reach
90.2% against 76.6% for the thresholded streams (far serve alone: 87.5% against
64.8%).  Every one of those was a decision made with no knowledge of the rest of
the match, by a detector looking at one player through one window.

The evidence that recovers them is not in the pose.  It is in the structure:

  * A GAME IS A RUN.  One player serves a whole game.  A lone far detection
    inside eleven near ones is a returner, not a server.
  * THE COURT ALTERNATES.  Within a game the server changes court EVERY point,
    without exception.  Two consecutive serves from the same court mean either a
    fault or a missed point -- and which one is decidable from the gap.
  * A FAULT IS NOT A POINT.  A second serve follows its fault by 10-25 s FROM
    THE SAME COURT.  The shipped orchestrator merges starts closer than 8 s, so
    it currently emits a fault and its second serve as TWO points.
  * POINTS HAVE A PERIOD.  Serve to serve is 15-40 s.  A 90 s hole with no
    accepted start is a claim that nothing happened, and that claim should have
    to beat the alternative that a weak candidate inside it was real.

None of that is visible to a detector and all of it is cheap here.

What it does NOT do
-------------------
It does not invent points.  Every accepted start is a candidate some detector
produced; the prior decides which candidates to believe and how to group them,
never where a serve would have been convenient.  A point nobody detected stays
missing, and `orchestrator.recover_missed` is still the only thing that can find
those -- see `suspected_missing` in the result, which reports where the prior
believes a point was skipped without asserting one.

The solver
----------
A Viterbi over serve attempts in time order.  State is
`(anchor attempt, server, court, run length)` and each transition is one of:

    NEW           this attempt opens a point
    SECOND_SERVE  this attempt is the second serve of the point already open
    (skip)        implicit -- an attempt no transition lands on is rejected

Exact rather than greedy for the same reason `enforce_service_runs` is: a greedy
pass commits to an early wrong server and can never recover, and the whole value
of the structural evidence is that it is retrospective.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import workdir as WD
from pipeline.anya2 import court as C
from pipeline.anya2 import far_serve as FS
from pipeline.anya2 import near_serve as NS
from pipeline.anya2 import point_end as PE
from pipeline.anya2.contract import (FAR_SERVE, NEAR_SERVE, POINT_END, Event,
                                     load_events)

SUFFIX = "_anya2_points.json"


# ── configuration ────────────────────────────────────────────────────────

@dataclass
class ArbConfig:
    # ── clustering raw candidates into serve attempts ────────────────────
    cluster_s: float = 2.5        # candidates this close are one attempt.  The
                                  # near and far detectors anchor on different
                                  # moments of the same action (hands-together
                                  # vs trophy onset), so the window has to cover
                                  # that offset, not just detector jitter.
    min_p: float = 0.15           # ignore candidates weaker than this entirely.
                                  # Not a threshold on the answer -- a floor on
                                  # what is worth carrying through an O(n^2)
                                  # solver.

    # ── evidence ─────────────────────────────────────────────────────────
    w_evidence: float = 4.0       # weight on detector confidence
    p_neutral: float = 0.45       # accepting a candidate weaker than this is a
                                  # net cost, stronger a net gain.  This is the
                                  # ONLY place a detector's own confidence turns
                                  # into a decision, and it sits well below both
                                  # detectors' shipped thresholds (0.70 near,
                                  # 0.75 far) -- the structure is expected to
                                  # carry candidates the detector would refuse.

    # ── the court alternates ─────────────────────────────────────────────
    # EVERY STRUCTURAL TERM IS A COST, NEVER A REWARD.  A reward per accepted
    # point makes a longer path score higher for being longer, which is a bias
    # toward over-acceptance dressed up as evidence -- and it cost ~8 points of
    # precision before it was noticed.  Detector confidence is the only thing
    # that can add to a path; structure can only subtract.
    #
    # Measured over 140 consecutive same-server LABELLED pairs, court read at
    # the trophy:
    #
    #     a real next point (gap > 28 s)     n=60    flips  91.7%
    #     fault-shaped      (gap < 28 s)     n=80    flips  57.5%
    #
    # Both numbers are the tennis: consecutive points alternate, and a fault
    # and its second serve do not.  The 91.7% holds across every margin band
    # (88.9 / 95.2 / 91.7% at 0.35-1 / 1-2 / 2+ m from the centre line), so the
    # signal is not an artefact of easy geometry.
    w_alternation: float = 1.6    # charged when consecutive points of one game
                                  # do NOT change court.  Flipping costs
                                  # nothing: it is the expected case, not a
                                  # bonus to be collected.
    court_margin_m: float = 0.35  # how far from the centre line a serve has to
                                  # be measured for its court to count as known.
                                  # Inside this band the court reads "unknown"
                                  # and scores zero -- neutral, because the
                                  # measurement failed, not the tennis.

    # ── a fault is a serve from the same court ───────────────────────────
    # MEASURED, and it settles a design question rather than a constant: the
    # corpus labels a fault and its second serve as TWO point starts, not one.
    # On clip 22 all five missed starts were attempts this solver had absorbed
    # as second serves, each with a strong detection at the labelled time (e.g.
    # 64.1 s and 72.4 s, both labelled, 8.3 s apart, same court).
    #
    # So absorption is OFF by default.  What survives is the half that is pure
    # gain: a same-court repeat at a fault-shaped gap must NOT be charged the
    # alternation penalty, because a fault is exactly when tennis serves twice
    # from one court.  Without that exemption the prior punishes the most
    # ordinary event in the sport.
    #
    # `merge_second_serves` keeps the absorbing behaviour available, because for
    # a REEL it is arguably the right product answer -- a fault and its second
    # serve are one thing to watch -- and it is the labels, not the tennis, that
    # say otherwise.  Turning it on costs recall against this corpus by
    # construction; see the README.
    merge_second_serves: bool = False
    fault_max_s: float = 28.0     # a second serve follows its fault by less
                                  # than this, from the same court
    fault_min_s: float = 6.0      # ...and by more than this; closer than that
                                  # it is one motion detected twice
    w_fault: float = 0.9          # score for absorbing an attempt as a second
                                  # serve, used only when `merge_second_serves`
                                  # is on.  With it off a fault-shaped repeat is
                                  # simply EXEMPT from the alternation cost --
                                  # exemption, not reward, for the reason above.
    fault_ev_scale: float = 0.5   # a second serve's own evidence counts for
                                  # less: it is being explained by the point
                                  # already open, not proposing one

    # ── a game is a run ──────────────────────────────────────────────────
    min_game_points: int = 4
    w_switch: float = 2.2         # cost of changing server
    w_short_game: float = 1.1     # ...charged again per point short of
                                  # `min_game_points` when a run is cut early
    w_newgame_deuce: float = 0.0  # the first point of a game is served from the
                                  # deuce court.  True in tennis, but it needs
                                  # the deuce/ad SIGN fixed per camera, which is
                                  # not measured yet.  Off until it is.

    # ── points have a period ─────────────────────────────────────────────
    gap_lo_s: float = 12.0        # serve-to-serve shorter than this is suspect
    gap_hi_s: float = 45.0        # ...and longer than this implies a hole
    w_gap: float = 0.8            # cost per unit of implausibility
    period_s: float = 30.0        # assumed serve-to-serve period, used only to
                                  # count how many points a long hole implies
    w_missing: float = 0.7        # cost per implied missing point.  This is what
                                  # makes the solver prefer a weak candidate
                                  # inside a 90 s hole over an empty stretch.
    max_missing: float = 4.0      # ...capped, so one long changeover does not
                                  # dominate the whole path

    # ── ends ─────────────────────────────────────────────────────────────
    min_point_s: float = 2.5      # an end sooner than this after the serve is
                                  # the serve motion itself
    max_point_s: float = 40.0
    end_guard_s: float = 4.0      # never place an end closer than this to the
                                  # next accepted start
    w_end_evidence: float = 1.0
    w_end_duration: float = 0.35  # prefer ends that give a plausible rally
                                  # length over ends that merely fall hard
    typical_point_s: float = 9.0


# ── candidate assembly ───────────────────────────────────────────────────

@dataclass
class Attempt:
    """One serve attempt: every candidate that refers to the same action."""
    t: float                       # point-start time (earliest member)
    p_near: float = 0.0
    p_far: float = 0.0
    court_x: Optional[float] = None
    track: Optional[int] = None
    n_cands: int = 0
    flags: List[str] = field(default_factory=list)

    @property
    def side(self) -> str:
        return "near" if self.p_near >= self.p_far else "far"

    @property
    def p(self) -> float:
        return max(self.p_near, self.p_far)

    def court(self, cfg: "ArbConfig") -> int:
        """-1 / +1 for the two halves, 0 for "not measured well enough".

        The sign is relative to the centre line and is NOT claimed to be deuce
        vs ad: alternation only needs the two halves to be distinguishable, and
        which one is the deuce court depends on where the camera stands.
        """
        if self.court_x is None or not np.isfinite(self.court_x):
            return 0
        d = float(self.court_x) - C.COURT_W / 2.0
        if abs(d) < cfg.court_margin_m:
            return 0
        return 1 if d > 0 else -1


def _candidate_events(video, tracks_npz=None, prefer_raw=True):
    """Load the three candidate streams, falling back to the shipped events.

    A missing candidate file is not an error: the shipped stream is a valid, if
    impoverished, input, and this keeps the arbitrator runnable on a clip whose
    detectors were run before raw emission existed.
    """
    d = WD.artifact_dir(video)
    stem = os.path.splitext(os.path.basename(video))[0]

    def pick(raw_suffix, shipped_suffix, kind):
        order = (raw_suffix, shipped_suffix) if prefer_raw else (shipped_suffix,
                                                                raw_suffix)
        for s in order:
            p = os.path.join(d, f"{stem}{s}")
            if os.path.isfile(p):
                return [e for e in load_events(p) if e.kind == kind]
        return []

    return (pick(NS.CANDIDATES_SUFFIX, NS.EVENTS_SUFFIX, NEAR_SERVE),
            pick(FS.CANDIDATES_SUFFIX, FS.EVENTS_SUFFIX, FAR_SERVE),
            pick(PE.CANDIDATES_SUFFIX, PE.EVENTS_SUFFIX, POINT_END))


def cluster(near: Sequence[Event], far: Sequence[Event],
            cfg: ArbConfig) -> List[Attempt]:
    """Collapse both candidate streams into one timeline of serve attempts.

    Both detectors firing for one action, and one detector firing twice for it,
    are the same situation: several views of a single serve.  They are pooled
    rather than deduped, so an attempt only the weaker detector saw still
    arrives with that detector's evidence attached.
    """
    rows = []
    for e in near:
        if e.p >= cfg.min_p:
            rows.append((float(e.t), "near", float(e.p),
                         e.detail.get("court_x"), e.track,
                         e.detail.get("flags") or []))
    for e in far:
        if e.p >= cfg.min_p:
            rows.append((float(e.t), "far", float(e.p),
                         e.detail.get("court_x"), e.track,
                         e.detail.get("flags") or []))
    rows.sort(key=lambda r: r[0])

    out: List[Attempt] = []
    for t, side, p, cx, track, flags in rows:
        if out and t - out[-1].t <= cfg.cluster_s:
            a = out[-1]
        else:
            a = Attempt(t=t)
            out.append(a)
        strongest_so_far = a.p
        a.n_cands += 1
        a.flags = sorted(set(a.flags) | set(flags))
        if side == "near":
            a.p_near = max(a.p_near, p)
        else:
            a.p_far = max(a.p_far, p)
        # Court and slot come from the strongest member: a weak echo of the same
        # action is measured at a worse moment of it.
        if p >= strongest_so_far:
            if cx is not None:
                a.court_x = float(cx)
            a.track = track
    return out


# ── the solver ───────────────────────────────────────────────────────────

NEW, SECOND = "new", "second_serve"


def _gap_cost(dt: float, cfg: ArbConfig) -> Tuple[float, float]:
    """(implausibility, implied missing points) for a serve-to-serve gap."""
    if dt < cfg.gap_lo_s:
        return (cfg.gap_lo_s - dt) / max(cfg.gap_lo_s, 1e-6), 0.0
    if dt <= cfg.gap_hi_s:
        return 0.0, 0.0
    missing = min(cfg.max_missing, (dt - cfg.gap_hi_s) / max(cfg.period_s, 1e-6))
    return 0.0, missing


def solve(attempts: List[Attempt], cfg: ArbConfig) -> List[Dict]:
    """Viterbi over attempts.  Returns the accepted points, in time order.

    One pass in anchor order.  Every transition moves the anchor forward, so a
    state is final the moment its anchor index comes up and no state needs
    revisiting -- which is what keeps an O(n^2) relaxation honest.
    """
    if not attempts:
        return []
    n = len(attempts)
    sides = ("near", "far")
    R = max(1, cfg.min_game_points)

    # state -> (score, predecessor state, how this attempt was explained)
    # state = (anchor index, server index, court sign, run length capped at R)
    best: Dict[tuple, tuple] = {}
    by_anchor: Dict[int, set] = {}

    def offer(st, score, back, kind):
        cur = best.get(st)
        if cur is None or score > cur[0]:
            best[st] = (score, back, kind)
            by_anchor.setdefault(st[0], set()).add(st)

    # A path may begin at any attempt: rejection is free by construction (see
    # p_neutral), so there is nothing to charge for the attempts before it.  The
    # first point is exempt from the run minimum -- a clip rarely starts on a
    # game boundary.
    for i, a in enumerate(attempts):
        offer((i, sides.index(a.side), a.court(cfg), 1),
              cfg.w_evidence * (a.p - cfg.p_neutral), None, NEW)

    horizon = cfg.gap_hi_s + cfg.period_s * cfg.max_missing
    for i in range(n):
        a_i = attempts[i]
        for st in sorted(by_anchor.get(i, ())):
            score = best[st][0]
            _, si, crt, run = st
            for j in range(i + 1, n):
                a_j = attempts[j]
                dt = a_j.t - a_i.t
                if dt > horizon:
                    break                     # nothing further can be cheaper
                sj = sides.index(a_j.side)
                cj = a_j.court(cfg)
                ev = cfg.w_evidence * (a_j.p - cfg.p_neutral)

                # A fault and its second serve come from the SAME court at a
                # gap in this band -- the one case where not flipping is
                # correct tennis rather than a missed point.
                fault_shaped = (sj == si and cj != 0 and cj == crt
                                and cfg.fault_min_s <= dt <= cfg.fault_max_s)

                # ── second serve, absorbed into the open point ───────────
                if fault_shaped and cfg.merge_second_serves:
                    offer((j, si, crt, run),
                          score + cfg.w_fault + ev * cfg.fault_ev_scale,
                          st, SECOND)

                # ── a new point ──────────────────────────────────────────
                gap_imp, missing = _gap_cost(dt, cfg)
                s1 = score + ev - cfg.w_gap * gap_imp - cfg.w_missing * missing
                if sj == si:
                    if not fault_shaped and cj != 0 and crt != 0 and cj == crt:
                        s1 -= cfg.w_alternation    # same court, not a fault
                    new_run = min(run + 1, R)
                else:
                    s1 -= cfg.w_switch
                    if run < R:
                        s1 -= cfg.w_short_game * (R - run)
                    if cfg.w_newgame_deuce and cj != 0:
                        s1 += cfg.w_newgame_deuce * (1 if cj > 0 else -1)
                    new_run = 1
                offer((j, sj, cj if cj != 0 else crt, new_run), s1, st, NEW)

    end_state = max(best.items(), key=lambda kv: kv[1][0])[0]
    chain: List[Tuple[int, str]] = []
    st = end_state
    while st is not None:
        _, back, kind = best[st]
        chain.append((st[0], kind))
        st = back
    chain.reverse()

    points: List[Dict] = []
    for idx, kind in chain:
        a = attempts[idx]
        if kind == SECOND and points:
            points[-1]["second_serve_t"] = a.t
            points[-1]["n_serves"] = points[-1].get("n_serves", 1) + 1
            points[-1]["p"] = max(points[-1]["p"], a.p)
            continue
        points.append({
            "start_s": a.t, "side": a.side, "p": a.p,
            "court": a.court(cfg), "court_x": a.court_x,
            "track": a.track, "n_cands": a.n_cands, "n_serves": 1,
            "flags": list(a.flags),
        })
    return points


# ── ends ─────────────────────────────────────────────────────────────────

def assign_ends(points: List[Dict], ends: Sequence[Event], cfg: ArbConfig,
                duration: Optional[float] = None) -> List[Dict]:
    """Give every point exactly one end, inside its own window.

    The window is what the structure already fixes: an end cannot precede its
    own serve by less than the serve motion, and cannot outlive the next point's
    start.  Inside it the choice trades the candidate's own confidence against
    giving the point a plausible length -- a hard fall 35 s in is worse evidence
    for THIS point than a softer one at 9 s.
    """
    et = np.array([float(e.t) for e in ends], dtype=float)
    ep = np.array([float(e.p) for e in ends], dtype=float)
    for i, pt in enumerate(points):
        lo = pt["start_s"] + cfg.min_point_s
        nxt = (points[i + 1]["start_s"] if i + 1 < len(points)
               else (duration if duration else pt["start_s"] + cfg.max_point_s))
        hi = min(pt["start_s"] + cfg.max_point_s, nxt - cfg.end_guard_s)
        if hi <= lo or et.size == 0:
            pt["end_s"] = pt["start_s"] + min(cfg.typical_point_s,
                                              max(cfg.min_point_s, hi - pt["start_s"]))
            pt["end_source"] = "estimated"
            continue
        m = (et >= lo) & (et <= hi)
        if not m.any():
            pt["end_s"] = min(pt["start_s"] + cfg.typical_point_s, hi)
            pt["end_source"] = "estimated"
            continue
        cand_t, cand_p = et[m], ep[m]
        dur = cand_t - pt["start_s"]
        # Duration prior: flat through a normal rally, falling away past it.
        plaus = np.exp(-np.abs(dur - cfg.typical_point_s)
                       / (2.0 * cfg.typical_point_s))
        sc = cfg.w_end_evidence * cand_p + cfg.w_end_duration * plaus
        k = int(np.argmax(sc))
        pt["end_s"] = float(cand_t[k])
        pt["end_source"] = "detected"
    return points


# ── top level ────────────────────────────────────────────────────────────

def decide(video, tracks_npz=None, cfg: Optional[ArbConfig] = None,
           verbose: bool = True) -> Dict:
    cfg = cfg or ArbConfig()
    near, far, ends = _candidate_events(video, tracks_npz)
    attempts = cluster(near, far, cfg)
    points = solve(attempts, cfg)

    duration = None
    try:
        from pipeline.anya2 import tracks as T
        z = T.load(video, tracks_npz)
        duration = len(z["kp"]) / float(z["fps"])
    except Exception:
        pass
    points = assign_ends(points, ends, cfg, duration)

    # What the prior believes it could not explain: holes wider than a point
    # period with nothing accepted inside them.
    missing = 0
    for a, b in zip(points, points[1:]):
        dt = b["start_s"] - a["start_s"]
        if dt > cfg.gap_hi_s:
            missing += int(min(cfg.max_missing,
                               (dt - cfg.gap_hi_s) / cfg.period_s))

    res = {
        "video": os.path.abspath(video),
        "n_candidates": {"near": len(near), "far": len(far), "end": len(ends)},
        "n_attempts": len(attempts),
        "n_points": len(points),
        "n_second_serves": sum(p.get("n_serves", 1) - 1 for p in points),
        "n_rejected": len(attempts) - sum(p.get("n_serves", 1) for p in points),
        "suspected_missing": missing,
        "config": asdict(cfg),
        "points": points,
    }
    if verbose:
        print(f"[arbitrate] {len(near)}+{len(far)} serve candidates -> "
              f"{len(attempts)} attempts -> {len(points)} points "
              f"({res['n_second_serves']} second serves, "
              f"{res['n_rejected']} rejected, {missing} holes)")
    return res


def points_path(video, suffix=SUFFIX):
    d = WD.artifact_dir(video)
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(d, f"{stem}{suffix}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("video")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = decide(a.video, a.tracks)
    out = a.json or points_path(a.video)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"[arbitrate] wrote {out}")


if __name__ == "__main__":
    main()
