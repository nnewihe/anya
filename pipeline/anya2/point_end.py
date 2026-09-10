"""
point_end.py
============
When did the point stop?  Pose only -- the ball is not read anywhere in this
module.

Why no ball
-----------
The shipped policy makes the ball trace the primary point-end signal, and it
works on hard courts.  It is not dependable on clay, where the ball is low
contrast against the surface for much of its flight, and a point-end policy
whose primary evidence disappears on one surface is not a policy.  So this
module is built from player pose, and the ball is left for a later, MINOR
corroborating role if one is ever shown to earn its cost.

Two measured facts shape everything here
----------------------------------------
FACT 1: INSTANTANEOUS ACTIVITY DOES NOT SEPARATE LIVE FROM DEAD.  Measured over
four clips, per-sample player motion and limb energy separate a live point from
the three seconds after it at an AUC of 38-75% -- at or BELOW chance on the two
hardest clips.  The reason is not subtle: a rally contains long quiet beats
while the opponent plays the ball, and dead time contains a player walking to
retrieve it.  Motion does not stop at the end of a point; it CHANGES CHARACTER.
Any construction that thresholds "how much are they moving" is doomed, however
the terms are weighted.

FACT 2: SUSTAINED QUIET IS ALMOST PERFECTLY SPECIFIC AND USELESS FOR TIMING.
A window in which both players stay quiet for 1.5 s covers 0.0-1.2% of live
play and 7-21% of dead time -- so it is nearly proof that the point is over.
But the FIRST such window after a labelled end arrives a median of +78 SECONDS
later, because players do not stand still after a point: they walk, bounce a
ball, and reposition.  Quiet marks changeovers, not point ends.  It can confirm
that a point ended; it cannot say when.

What is left is the character of the motion, and that is exactly what the
walking classifier already answers.

The construction
----------------
The point ends when the near player stops playing and starts TRAVELLING --
walking to the ball, to position, or to the towel.  `walking/predict.py` is a
HistGradientBoosting model over 373 gait-window features, trained on hand
labels from an indoor hard clip and an outdoor clay clip, and validated
LEAVE-ONE-CLIP-OUT at frame F1 0.82-0.84 across surface, camera and players.
It is the only learned component in anya2 and the only one that has ever been
shown to transfer across surfaces, which is precisely the property the ball
trace lacks.

Timing comes from the ONSET of a walking interval; corroborators only score it.
Measured on Data/21, walk onsets sit at a median +0.6 s from the labelled end
(p25 -0.9, p75 +6.4) and every one of the 12 labelled ends has one.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import workdir as WD

from pipeline.anya2 import signals as S
from pipeline.anya2 import tracks as T
from pipeline.anya2.contract import (POINT_END, ROI_BOTH, W_AFTER_SERVE, Event,
                                     Requirement, dump_events)

EVENTS_SUFFIX = "_anya2_point_end.json"
WALK_SUFFIX = "_anya2_walk.npz"
MAX_GAP_S = 0.5

# ── activity, for the quiet corroborator ─────────────────────────────────
# Scale-free and homography-free: box travel and limb speed are both divided by
# box height, giving body heights per second.  Court speed is deliberately NOT
# used -- a far player's ground point carries metres of projection error (see
# tracks.FAR_BACK_M), so any court-metre speed for them is noise, and a signal
# that means different things on the two sides of the net cannot be combined.
QUIET_BH_S = 0.45         # activity below this counts as quiet
QUIET_WIN_S = 1.5         # ...held this long to count as a quiet window

# ── walking onsets ───────────────────────────────────────────────────────
WALK_MIN_S = 1.0          # ignore walking intervals shorter than this
REFRACT_S = 6.0           # one end per point; ends are seconds apart at worst

REQUIREMENT = Requirement(roi=ROI_BOTH, pose_fps=15.0, needs_ball=False,
                          windows=W_AFTER_SERVE)


def events_path(video, suffix=EVENTS_SUFFIX):
    d = WD.artifact_dir(video)
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(d, f"{stem}{suffix}")


def walk_path(video, suffix=WALK_SUFFIX):
    return events_path(video, suffix)


def player_activity(z, fps) -> np.ndarray:
    """Per-slot activity in body heights/second. [4, N], NaN where untracked."""
    kp, bb = z["kp"], z["bbox"]
    n = len(kp)
    gap = int(MAX_GAP_S * fps)
    out = []
    for s in range(T.N_SLOTS):
        k = S.interp_gaps(kp[:, s], gap)
        b = S.interp_gaps(bb[:, s], gap)
        h = b[:, 3] - b[:, 1]
        h = np.where(h > 8.0, h, np.nan)
        cx, cy = 0.5 * (b[:, 0] + b[:, 2]), b[:, 3]
        mv = np.full(n, np.nan)
        mv[1:] = np.hypot(np.diff(cx), np.diff(cy)) / h[1:] * fps
        hip = S.mid(S.kp_xy(k, S.L_HIP, 0.2), S.kp_xy(k, S.R_HIP, 0.2))
        limb = np.full(n, np.nan)
        for j in (S.L_WRI, S.R_WRI, S.L_ANK, S.R_ANK):
            rel = (S.kp_xy(k, j, 0.2) - hip) / h[:, None]
            d = np.full(n, np.nan)
            d[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps
            limb = np.fmax(limb, d)
        out.append(np.fmax(mv, limb))
    return np.array(out)


def quiet_mask(act: np.ndarray, fps: float) -> np.ndarray:
    """True where EVERY tracked player has been quiet for QUIET_WIN_S.

    An untracked player contributes nothing rather than counting as quiet:
    absence of a player is not evidence that the point ended -- the same rule
    near_end applies to the energy bar, and for the same reason.
    """
    with np.errstate(invalid="ignore"):
        loud = np.nanmax(np.where(np.isfinite(act), act, np.nan), axis=0)
    tracked = np.isfinite(act).any(axis=0)
    q = tracked & (np.nan_to_num(loud, nan=np.inf) < QUIET_BH_S)
    w = max(1, int(round(QUIET_WIN_S * fps)))
    out = np.zeros(len(q), dtype=bool)
    for lo, hi in S.runs(q):
        if hi - lo >= w:
            out[lo:hi] = True
    return out


# ── the non-rally union ──────────────────────────────────────────────────
# The point end is the onset of the first sustained NON-RALLY state.  Walking
# alone is not that state: measured over all 216 labelled ends, a walk onset
# follows 212 of them -- essentially every point end leads to the near player
# travelling -- but at a median of +5.1 s and a p75 of +15.9, because the walk
# is a CONSEQUENCE of the end rather than the end itself.  Between the last ball
# and the first step the player stands, turns, or celebrates.
#
# `near_end`'s four pose signals cover exactly that gap, and the union of all
# five is much better than any of them:
#
#     signal          p25     median   p75      (clip 58, 81 ends)
#     walk           +2.1     +6.8    +24.8
#     settle         +5.6    +10.7    +19.1
#     turn_away      +5.3    +14.8    +36.9
#     stance_drop   +26.2    +68.9   +118.1
#     UNION          +0.3     +3.0     +6.7
#
# Every corroborator is individually WORSE than walking, some grossly so, and
# the union still beats walking by a factor of four at the p75.  That is not a
# paradox: they are worse on average but they fire EARLIER on the points where
# walking is late, which is the only thing a max() is asking of them.  It is the
# same "individually noisy, collectively decisive" construction near_end was
# written for, and the reason the combination is a max and not a weighted sum.
UNION_NAMES = ("settle", "turn_away", "stance_drop", "idle_hands")

# Swept over all 216 labelled ends.  Both parameters want to be LOW: at 0.40 /
# 0.6 s every end has an onset, median +0.5 s and p75 +3.7 s, and raising either
# only pushes the tail out (0.70 / 1.5 s gives median +3.5, p75 +14.0).  A
# lower bar finds the moment play stopped; a higher one waits for proof.
UNION_THR = 0.40
UNION_WIN_S = 0.6


def end_signal(video, tracks_npz=None) -> Dict[str, np.ndarray]:
    """The non-rally union and its parts, off the cached pose passes."""
    stem = os.path.splitext(os.path.basename(video))[0]
    d = WD.artifact_dir(video)
    w = np.load(os.path.join(d, f"{stem}_anya2_walk.npz"))
    sg = np.load(os.path.join(d, f"{stem}_anya2_endsig.npz"))
    fps = float(w["fps"])
    n = min(len(w["prob"]), len(sg[UNION_NAMES[0]]))
    parts = {k: np.nan_to_num(sg[k][:n], nan=0.0) for k in UNION_NAMES}
    parts["walk"] = np.asarray(w["prob"][:n], dtype=np.float64)
    union = np.max(np.stack([parts[k] for k in list(UNION_NAMES) + ["walk"]]), axis=0)

    z = T.load(video, tracks_npz)
    act = player_activity(z, fps)[:, :n]
    parts["quiet"] = quiet_mask(act, fps).astype(np.float64)
    parts["union"] = union
    parts["fps"] = np.float64(fps)
    return parts


# ── the live score ───────────────────────────────────────────────────────
# The union above is a DEAD-time indicator and, taken alone, a weak one: at the
# frame level it separates live from dead at AUC 60%, and its parts are worse
# still (settle 48% -- below chance, turn_away 53%, stance_drop 54%, walk 59%).
# Those numbers are not a failure of the signals; they are a statement that the
# question they were built for is not this one.  The walking classifier answers
# "is this person travelling" at F1 0.82, and near_end's four answer "does this
# posture look between-points" -- neither is "is the ball in play".
#
# What does carry live/dead is PLAYER ACTIVITY INTEGRATED OVER SECONDS.  Per
# sample it is useless (AUC 33%, and inverted against the three seconds after an
# end, because that window is full of walking).  Smoothed over 8 s it separates
# at 79.5% for the near player and 75.1% for the far one -- and the far number is
# only available at all because anya2 tracks the far player, which no previous
# point-end work here could do.
#
# The two combine multiplicatively, and that is the whole construction:
#
#     near activity, 8 s               79.5%
#     far activity, 8 s                75.1%
#     max(near, far), 8 s              78.7%
#     max(near, far) - union, 8 s      82.6%
#     max(near, far) * (1 - union)     86.7%     <-- this
#
# A product, not a sum, because the union's job is to VETO activity rather than
# to be traded off against it: a player walking to the ball is active and is
# emphatically not playing, and only a multiplicative term can say so.  It is
# the same arbitration shape the near serve detector's swing term has, for the
# same reason.
LIVE_SMOOTH_S = 4.0       # 8 s separates live from dead slightly better but
                          # blurs the EDGE, and the edge is what is being timed:
                          # at 8 s the best F1 is 40.3% against 42.8% at 4 s.
LIVE_SCALE_PCT = 90       # per-clip normaliser.  Activity is in body heights per
                          # second, so its absolute level depends on how large
                          # the players are in frame -- a fixed threshold would
                          # mean a different thing on every camera.  The clip's
                          # own 90th percentile is what makes the hysteresis
                          # levels below portable.
# Swept against `orchestrator.point_end_score` -- closeness of the chosen point
# end to the labelled one, flat inside +/-2 s, linear-decaying when late and
# exponentially penalised when early -- over all 208 labelled ends on 11 clips:
#
#     hi / lo        PES     within +/-2 s   >2 s early   >2 s late   median err
#     0.50 / 0.35   +0.065        64             48           96        +1.1 s
#     0.45 / 0.30   +0.065        66             44           98        +1.5 s
#     0.40 / 0.25   +0.093        67             40          101        +1.8 s  <--
#     0.35 / 0.25   +0.096        65             37          106        +2.5 s
#     0.35 / 0.20   +0.085        55             31          122        +4.3 s
#     0.30 / 0.20   +0.086        53             31          124        +4.5 s
#     0.30 / 0.15   +0.088        54             21          133        +7.1 s
#
# 0.40/0.25 is at once the top of the PES column (0.35/0.25 is +0.003 above it,
# inside the noise of this corpus) and the best row on the column that says
# "the end was right" -- 67 ends inside +/-2 s, more than any other setting.
# When those two columns agree there is nothing to trade off, which is the
# happy case and not the usual one; an earlier pass of this table, run against
# STALE far-serve artifacts, had them disagreeing and the choice was made on
# the plateau column at the user's direction.  It survived the correction.
#
# The general shape is still worth knowing, because it will reappear: pushing
# the bar DOWN does not put more ends on the labelled time, it holds the live
# state open longer so every falling edge lands later.  Below 0.35/0.25 the
# plateau column collapses (65 -> 55) while the early count keeps falling, and
# an objective that charges exponentially for early and only linearly for late
# will happily buy that trade.  The plateau column is what stops it.
#
# The histogram is what settles it.  Binned by how early, the mistimed ends are
# one mode against the plateau edge and nothing beyond it:
#
#     s early   0.50/0.35   0.40/0.25   0.35/0.20
#      2-3          23          21          14
#      3-4          11           5           5
#      4-5           2           2           1
#      5-6           2           2           1
#      6+            0           0           0      <- all mistimed ends
#
# EVERY genuinely mistimed end in the corpus is under 6 s early, at every
# setting, with a median of ~2.7 s and p90 ~4.0 s.  There is no catastrophic
# mid-rally tail for the exponential to punish -- at 2.7 s early the score is
# still +0.35, positive -- so the penalty term is barely engaging and the PES
# ranking is driven mostly by the eleven points with NO segment at all, which
# are scored as full-duration truncations and which no hysteresis setting
# changes (11, 11, 10 across the three rows).  Optimising PES here is largely
# optimising a serve-recall problem through the wrong knob.
#
# It is not one clip: with clip 58 excluded the ranking is unchanged.  Every
# other parameter was re-checked at this hysteresis and none of them moved
# (FAR_VETO_W 0.8, TURN_HOLD_S 0, confident pairing, LIVE_SMOOTH_S 4.0,
# est_duration_pct 85).
LIVE_HI = 0.40            # enter "live" above this fraction of that scale
LIVE_LO = 0.25            # ...and leave it below this
LIVE_MIN_S = 2.0          # ignore live runs shorter than this

# How much of the union's veto the FAR activity term is subject to.  1.0 is the
# original shape (the near-only union multiplied the max of both players);
# 0.0 exempts the far player from it entirely.  Neither end is right, and the
# corpus says so plainly -- swept over all 208 labelled ends on 11 clips, with
# the turn_away hold gate and confidence-aware pairing both on:
#
#     w      recall  precision | whole  trunc_s  dead_s
#     1.0     48.1%     37.6%  |   123      100    1179
#     0.9     49.5%     38.4%  |   126      100    1143
#     0.8     50.5%     39.3%  |   128       93    1213   <-- here
#     0.7     49.5%     38.1%  |   127       90    1272
#     0.6     48.6%     37.5%  |   129       83    1304
#     0.5     47.1%     36.3%  |   131       78    1333
#     0.4     42.3%     32.4%  |   134       71    1367
#     0.2     37.0%     28.2%  |   135       76    1463
#     0.0     35.6%     26.7%  |   138       78    1553
#
# Recall and precision BOTH peak at 0.8 and fall away in both directions, which
# is the shape a real optimum has.  Truncation keeps improving down to 0.4, but
# only by buying it with recall, precision and 154 s of extra dead time -- the
# detector stops finding ends at all and the reel runs on estimates instead.
FAR_VETO_W = 0.8


def live_score(parts: Dict[str, np.ndarray], video, tracks_npz=None,
               smooth_s: Optional[float] = None) -> np.ndarray:
    """Normalised [0, ~1] score for "a point is in play"."""
    fps = float(parts["fps"])
    n = len(parts["union"])
    z = T.load(video, tracks_npz)
    act = player_activity(z, fps)[:, :n]
    with np.errstate(invalid="ignore"):
        near = np.nan_to_num(np.nanmax(act[list(T.NEAR_SLOTS)], axis=0), nan=0.0)
        far = np.nan_to_num(np.nanmax(act[list(T.FAR_SLOTS)], axis=0), nan=0.0)
    # THE UNION VETOES ONLY THE NEAR PLAYER, because the near player is all it
    # is evidence about.  Every one of its five members -- walk and near_end's
    # four -- is computed from the near pose shim that `run._end_signals`
    # builds, so `union` says "the near player is not playing" and nothing
    # whatever about the far one.  Multiplying the MAX of both by (1 - union)
    # let a near-player posture erase the far player's activity, which is
    # exactly backwards on the commonest truncating case: the near player hits
    # an approach and stands watching (settle saturates) while the far player
    # sprints to run it down.  The point is emphatically live and the score went
    # to zero.
    #
    # Vetoing inside the max instead keeps the arbitration the module was built
    # on -- a near player travelling is active and is not playing -- while
    # leaving far-side evidence able to carry a rally on its own.  No new
    # parameter; the far term is simply no longer answerable to near-player
    # posture.
    # ...but exempting the far player OUTRIGHT is the opposite error, and the
    # corpus charges for it: at FAR_VETO_W = 0 the detected segments run 24.3 s
    # on average against 15.7 s at 1.0, because after the point ends nothing
    # vetoes the far player walking to the ball, so the falling edge arrives
    # late and the reel gains 54% more dead time.  Truncation is fixed and
    # replaced by overrun.
    #
    # The honest reading is that the union is WEAK evidence about the far
    # player rather than none: the two players' states are strongly correlated,
    # because a point ends for both of them at the same instant.  A partial
    # veto says exactly that, and the corpus picks the weight.
    raw = np.fmax(near * (1.0 - parts["union"]),
                  far * (1.0 - FAR_VETO_W * parts["union"]))
    w = max(1, int(round((LIVE_SMOOTH_S if smooth_s is None else float(smooth_s)) * fps)))
    sm = np.convolve(raw, np.ones(w) / w, mode="same")
    return sm / max(np.percentile(sm, LIVE_SCALE_PCT), 1e-6)


def _hysteresis(x, hi, lo):
    out = np.zeros(len(x), dtype=bool)
    on = False
    for i, v in enumerate(x):
        on = v >= hi if not on else v >= lo
        out[i] = on
    return out


def detect_ends(parts: Dict[str, np.ndarray], live: np.ndarray,
                hi: float = LIVE_HI, lo: float = LIVE_LO,
                min_s: float = LIVE_MIN_S) -> List[Dict]:
    """A point end is the FALLING EDGE of the live score.

    Not the onset of a dead state -- the falling edge.  Onsets of "looks dead"
    fire all over dead time, because dead time is noisy and flickers: scored
    that way the detector emitted 504 candidates for 216 labelled ends, and no
    local feature separated the good ones (run duration AUC 53%, mean depth 64%,
    quiet-window overlap 1%).  What DOES separate them is that a real end is
    PRECEDED BY PLAY, which is exactly what a falling edge encodes and what an
    onset does not.
    """
    fps = float(parts["fps"])
    m = _hysteresis(live, hi, lo)
    w = max(1, int(round(min_s * fps)))
    out: List[Dict] = []
    look = int(round(4.0 * fps))
    for a, b in S.runs(m):
        if b - a < w:
            continue
        after = live[b:min(len(live), b + look)]
        out.append({
            "t": b / fps,
            # Confidence is how far the live score falls and stays fallen: a
            # real end drops to the floor, a lull between shots does not.
            "p": round(float(np.clip(1.0 - (after.mean() if after.size else 1.0), 0, 1)), 4),
            "rally_s": round((b - a) / fps, 2),
        })
    return S.refractory(out, REFRACT_S)


def detect_video(video, tracks_npz=None, hi: float = LIVE_HI,
                 verbose: bool = True, lo: Optional[float] = None,
                 smooth_s: Optional[float] = None,
                 min_live_s: Optional[float] = None) -> List[Event]:
    parts = end_signal(video, tracks_npz)
    live = live_score(parts, video, tracks_npz, smooth_s=smooth_s)
    ev = detect_ends(parts, live, hi,
                     LIVE_LO if lo is None else float(lo),
                     LIVE_MIN_S if min_live_s is None else float(min_live_s))
    if verbose:
        print(f"[point-end] {len(ev)} ends (falling edges of the live score)")
    return [Event(t=float(e["t"]), p=float(e["p"]), kind=POINT_END, track=None,
                  detail={"rally_s": e["rally_s"]}) for e in ev]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("video")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--hi", type=float, default=LIVE_HI)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ev = detect_video(a.video, a.tracks, a.hi)
    for e in ev[:25]:
        print(f"  {e.t:8.2f}s  p={e.p:.3f}  rally={e.detail['rally_s']:5.1f}s")
    out = a.json or events_path(a.video)
    dump_events(ev, out, hi=a.hi, requirement=REQUIREMENT.__dict__)
    print(f"[point-end] wrote {out}")


if __name__ == "__main__":
    main()
