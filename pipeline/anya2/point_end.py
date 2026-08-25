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
    d = os.path.dirname(os.path.abspath(video))
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
    d = os.path.dirname(os.path.abspath(video))
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
LIVE_HI = 0.50            # enter "live" above this fraction of that scale
LIVE_LO = 0.35            # ...and leave it below this
LIVE_MIN_S = 2.0          # ignore live runs shorter than this


def live_score(parts: Dict[str, np.ndarray], video, tracks_npz=None) -> np.ndarray:
    """Normalised [0, ~1] score for "a point is in play"."""
    fps = float(parts["fps"])
    n = len(parts["union"])
    z = T.load(video, tracks_npz)
    act = player_activity(z, fps)[:, :n]
    with np.errstate(invalid="ignore"):
        near = np.nan_to_num(np.nanmax(act[list(T.NEAR_SLOTS)], axis=0), nan=0.0)
        far = np.nan_to_num(np.nanmax(act[list(T.FAR_SLOTS)], axis=0), nan=0.0)
    raw = np.fmax(near, far) * (1.0 - parts["union"])
    w = max(1, int(round(LIVE_SMOOTH_S * fps)))
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
                 verbose: bool = True) -> List[Event]:
    parts = end_signal(video, tracks_npz)
    live = live_score(parts, video, tracks_npz)
    ev = detect_ends(parts, live, hi)
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
