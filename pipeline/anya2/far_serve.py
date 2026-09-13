"""
far_serve.py
============
A far-side serve detector: the same phase-sequence idea as `near_serve`, with
the geometry re-derived for a player 30 pixels tall on the other side of the
net.

Why not just run the near detector on the far slots
---------------------------------------------------
Because two of its three trophy terms do not survive the change of viewpoint
and scale, and one of them is actively harmful.  Measured on Data/23's 15
labelled far serves, per pose sample, against every non-serve sample in the
clip:

    hi_head > 0.05      serve 13%   non-serve  0%   (non-serve p99 = +0.002)
    hi_elev > 0.10      serve 14%   non-serve  1%
    lo_elev > -0.05     serve 11%   non-serve  2%
    gap     > 0.18      serve 35%   non-serve 58%   <-- INVERTED

The elevation terms are close to perfectly discriminative: a far wrist above
the head essentially never happens outside a serve.  THE HAND SPLIT IS WORSE
THAN USELESS.  On the near player -- filmed from behind, 200 px tall -- the
hands coming apart off the grip is the sharpest transition in the action.  On
the far player it is not measurable: the arms are extended VERTICALLY at the
trophy, so their horizontal separation is small exactly when the near view
would have it large, while ordinary walking swings the arms apart.  So the far
trophy is elevation-only, and `gap` is not read anywhere in this module.

That asymmetry is the reason these are two detectors and not one with a flag.

What is kept from the near construction
---------------------------------------
  * THE ORDERING.  ready -> trophy -> swing, each within a bounded delay.  A
    lone thresholded shape is what the shipped `anya_far_serve` hand-raise gate
    is, and its measured cost is 39 false positives over 14 clips.

  * THE PHASE DILATION.  Terms are dilated before the conjunction, so the
    product describes a phase and not a single 67 ms sample.  This mattered
    more on the near side than anything else and the sampling argument is
    identical here.

  * THE SWING MULTIPLIES WITH A FLOOR, the shape terms add.  Same arbitration,
    same reason: a raise that never becomes a strike must be vetoable, but a
    serve caught mid-stroke must not be thrown away.

What is dropped, and why
------------------------
  * THE HAND SPLIT -- measured inverted, see above.

  * THE SERVE-ZONE BAND ON court_y.  The near detector gates on the server
    standing within a 1.4 m band about the near baseline, which is what
    separates a serve from a mid-court smash.  The same gate cannot be built
    here: a far player's ground point is 22-32 px tall in the analysis frame,
    and at that depth two pixels of box-bottom error is metres of court.
    Measured on Data/23 the far player's court_y spans 19.7-28.6 m (p5-p95)
    while really moving about three, and sits a median 4 m BEHIND the baseline
    it is standing on.  A 1.4 m band on that quantity is noise.  The far
    player is still required to be on the far side and inside the lateral
    doubles+3ft gate (`tracks.eligible`), because those are coarse enough to
    survive the noise -- 100% of far detections satisfy both.

BOX SANITY, which the near side never needed
--------------------------------------------
Every quantity here is normalised by box height, and at 30 px a bad box makes
those ratios meaningless rather than merely noisy -- a merged two-person box or
a fragment produced apparent wrist separations of 1.4-1.9 body heights, which
is anatomically impossible.  `_sane` rejects boxes by height and aspect before
anything is divided by them.  This is not a detection threshold; it is refusing
to divide by a number that is wrong.

What this detector structurally cannot do
-----------------------------------------
The same thing the near one cannot: know whether a point is already in
progress.  The shipped far detector's 39 false positives over 14 clips broke
down as 17 idle raises in dead time, 14 in-rally overheads and returns, and 8
the far player REACTING to a near serve -- 22 of 39 being "play was live and
the detector did not know it".  None of that is visible from inside a far-player
pose crop, and every far-side pose formulation aimed at it has failed.  This
module therefore does not try: it declares `windows="between_points"` and
leaves the arbitration to the composition layer, which will have the near
serves and the point ends that actually settle it.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import workdir as WD

from pipeline.anya2 import court as C
from pipeline.anya2 import signals as S
from pipeline.anya2 import tracks as T
from pipeline.anya2.contract import (FAR_SERVE, ROI_FAR, W_BETWEEN, Event,
                                     Requirement, dump_events)

EVENTS_SUFFIX = "_anya2_far_serve.json"
MAX_GAP_S = 0.5
KP_CONF = 0.20            # keypoint confidence floor.  Higher than the near
                          # side's implicit 0: at this scale a low-confidence
                          # keypoint is a guess, and a guessed wrist reads as a
                          # posture the player never held.

# ── box sanity ───────────────────────────────────────────────────────────
BOX_H_MIN_PX, BOX_H_MAX_PX = 12.0, 120.0
BOX_ASPECT_MAX = 1.6      # wider than this is two people merged or a fragment

# ── head reference ───────────────────────────────────────────────────────
# The far player FACES this camera, so nose/eyes/ears are the keypoints the
# model is most confident about -- the exact opposite of the near player, who
# has their back turned.  The geometric fallback is therefore rarely used here.
HEAD_ABOVE_SHO_BH = 0.14

# ── ready ────────────────────────────────────────────────────────────────
READY_HI_MAX_BH = 0.02    # both wrists carried at or below the shoulder line.
                          # Tighter than the near side's 0.10 because the far
                          # ready position is measured over a 30 px body and a
                          # loose bound admits half the clip.
READY_STILL_BH_S = 1.2    # limb speed relative to the hip, body heights/s.
                          # Looser than near's 0.9: at 30 px the same real
                          # motion is a smaller pixel motion but a NOISIER
                          # normalised one, so the quiet bar has to allow for
                          # keypoint jitter that the near view does not have.
READY_WIN_S = 0.7

# ── how the ready phase is READ ──────────────────────────────────────────
# The original reading -- the MAX of `ready` anywhere in the six seconds before
# the trophy -- is nearly useless, and measurably so: over 105 true far serves
# and 101 false positives its median is 1.00 on BOTH, separating them at AUC
# 62%.  Any six-second window contains some quiet moment, so a returner who
# stood still once five seconds ago scores a perfect ready.
#
# Read instead as the MEAN over a short window ENDING AT THE TROPHY, the
# separation is AUC 79% (true 0.90, false 0.62).  That is the difference
# between "was quiet at some point recently" and "was quiet right up until the
# racket went up", and only the second is what a service stance is.
READY_MEAN_FROM_S = 2.0   # window start, before the trophy
READY_MEAN_TO_S = 0.3     # ...and end, stopping short of the toss itself

# ── pre-serve stillness ──────────────────────────────────────────────────
# A SERVER IS STATIONARY BEFORE SERVING; A RETURNER HAS JUST BEEN RUNNING.
# That is the single largest thing separating a real far serve from the far
# player's return, and it is local to this detector -- no other agent needed.
#
# Measured as the median of the player's own translation (box centre, in body
# heights per second) over the window below: true far serves 0.15, false
# positives 0.31, separating at AUC 78%.  It is the mirror of the ready term
# and independent of it -- ready is about the ARMS, this is about the FEET.
#
# It enters multiplicatively but softened, so it can veto a candidate that is
# clearly mid-rally without deleting one whose track was noisy.
# ── the toss ─────────────────────────────────────────────────────────────
# A SEPARATE SCORE, deliberately never folded into `p`.  It answers a different
# question -- "did the tossing arm do what a toss does" -- and the orchestrator
# arbitrates the two, so keeping them apart is what lets it trade precision
# against recall without this module choosing on its behalf.
#
# Read from the TOSSING ARM ONLY, which is the arm carrying the higher wrist at
# the trophy (`hi_side`).  Four components, each measured against the trophy
# peak k, each ramped to [0, 1] and then AVERAGED:
#
#   low    the wrist STARTED low, below the shoulder, before the rise
#   lat    at the top the wrist is offset LATERALLY from its own shoulder
#   hold   the wrist stays above head height for a sustained beat
#   peak   how far above the head it gets
#
# Every one is weak alone -- over 103 true far serves and 56 false positives
# their individual AUCs are 66%, 66%, 64% and 65% -- and the average of the four
# reaches AUC 74%, true serves at median 0.72 against 0.33 for false positives.
#
# Two findings worth keeping:
#
#   THE TOSS ARM IS NOT VERTICAL.  The obvious formulation -- the wrist travels
#   straight up over its own shoulder -- is BACKWARDS.  True serves have a
#   LARGER lateral offset than false positives (0.148 vs 0.090 body heights),
#   because a real toss is released up and forward across the body while a
#   return's arm stays nearer the shoulder line.  `lat` is scored in the
#   direction the data gives, not the one anatomy-by-description suggests.
#
#   AVERAGE, DO NOT MULTIPLY.  A product of the same four scores the same AUC
#   (74%) but is degenerate -- most candidates land at exactly zero, so as a
#   filter it costs 53% of true serves to remove 88% of false positives.  The
#   mean keeps it graded, which is what a separate signal handed to an
#   arbitrator has to be.
TOSS_PRE_FROM_S, TOSS_PRE_TO_S = -1.3, -0.2   # window for the pre-rise low point
TOSS_TOP_FROM_S, TOSS_TOP_TO_S = -0.3, 0.6    # window for the extended top
TOSS_LOW_NONE_BH, TOSS_LOW_FULL_BH = -0.22, -0.42
TOSS_LAT_MIN_BH, TOSS_LAT_FULL_BH = 0.07, 0.17
TOSS_HOLD_MIN_S, TOSS_HOLD_FULL_S = 0.13, 0.42
TOSS_PEAK_MIN_BH, TOSS_PEAK_FULL_BH = 0.08, 0.17

STILL_FROM_S, STILL_TO_S = 5.0, 1.0
STILL_FULL_BH_S = 0.20    # at or below this the player was stationary
STILL_NONE_BH_S = 0.55    # ...and at or above it they were travelling
W_STILL = 0.30            # weight of the stillness veto; see THRESHOLD

# ── trophy (elevation only -- see the module docstring) ──────────────────
TROPHY_HEAD_MIN_BH = -0.02   # higher wrist below head height = no credit
TROPHY_HEAD_FULL_BH = 0.06   # ...and this far above it = full credit.  Lower
                             # than the near side's 0.10: a 0.10 bh margin is
                             # 3 px on a 30 px body and is not resolvable.
TROPHY_LO_MIN_BH = -0.18     # lower wrist this far below the shoulder = none
TROPHY_LO_FULL_BH = -0.02    # ...and level with it = full credit
TROPHY_DILATE_S = 0.20
TROPHY_MIN = 0.30

# ── swing ────────────────────────────────────────────────────────────────
SWING_MIN_S, SWING_MAX_S = 0.06, 0.90
PEAK_ABOVE_HEAD_BH = 0.04

# ── ready lookback ───────────────────────────────────────────────────────
READY_BACK_MIN_S, READY_BACK_MAX_S = 0.20, 6.00

# ── combination ──────────────────────────────────────────────────────────
W_TROPHY, W_READY = 0.45, 0.20

# SWING_EITHER_WRIST asks "does SOME wrist rise above the head after the
# trophy" instead of picking the racket arm off the hand split.  It is OFF, and
# the reason is worth keeping because the experiment that motivated it looked
# convincing at every stage but the last.
#
# On Data/21's missed far serve at imgsz 960 the two wrists differ by 0.016
# body heights -- the model is not separating them -- so `hi_side` is noise and
# whichever arm the swing term picks carries the raised wrist anyway.  Re-run
# at imgsz 1280 the wrists differ by 0.322, the split starts meaning something,
# and it points at the arm that does NOT go up (max head_r -0.028 against
# max head_l +0.087).  Taking either wrist fixes that serve outright: p 0.525
# -> 0.848, detected.
#
# Corpus-wide it costs more than it buys, at BOTH resolutions.  RE-TESTED
# after the baseline x gate below was added -- that gate did not exist when
# imgsz 1280 was first tried, and the hope was that it would reject the
# out-of-court people whose detections cost 1280 its precision.  It does help
# 1280 a little (85.1% -> 86.0%) and it does not change the verdict; every row
# below has the gate ON:
#
#     dets    swing    thr     recall   precision   clip 21's far serve
#     960     split    0.75     92.3%     88.4%     missed      <-- shipped
#     1280    split    0.75     87.9%     86.0%     missed
#     1280    either   0.75     92.3%     80.0%     FOUND
#     1280    either   0.80     89.0%     82.7%     FOUND
#     1280    either   0.85     79.1%     83.7%     missed
#
# 960 DOMINATES THE WHOLE 1280 FRONTIER: equal-or-better recall and strictly
# better precision at every operating point.  imgsz 1280 buys clip 21's one
# far serve and pays more than that for it everywhere else, which is the same
# answer the first test gave for the same reason -- the extra detections are
# mostly people who are not playing on this court.
#
# With the arms unresolved, "either wrist" is simply a looser test, and the
# looseness lands on false positives.  Left in as a flag rather than deleted
# because it is the right construction the moment the far pose pass can resolve
# two arms reliably, which at 960 it cannot.
SWING_EITHER_WRIST = False

# ── the far baseline's own x extent ──────────────────────────────────────
# `court.in_bounds` -- the gate `tracks.eligible` stores -- is the DOUBLES court
# plus a 3 ft margin, x in [-2.284, 10.514].  That is the user's
# court-membership rule and it is deliberately generous: it decides whether
# someone is on the court at all, and a player chasing a wide ball is.
#
# A SERVER is a stricter case.  A serve is struck from behind the baseline and
# between its ends, so a box whose x-centre projects outside the baseline is
# not a server whatever its pose looks like.  This is a separate gate rather
# than a tightening of `in_bounds`, because the two answer different questions
# and only this one is entitled to be strict.
#
# WHICH BASELINE, measured rather than argued.  The doubles baseline is the
# rules-correct bound -- in doubles a server may legally stand out to the
# doubles sideline -- and on this corpus it is worth exactly nothing: every
# detection already inside `in_bounds` is inside it too, and recall, precision
# and the per-clip table are identical to no gate at all.  The SINGLES baseline
# removes two false positives for no lost serve:
#
#     x gate                  range            recall   precision
#     none / in_bounds        [-2.28, 10.51]    92.3%     86.6%
#     doubles baseline        [-1.37,  9.60]    92.3%     86.6%     (no change)
#     SINGLES baseline        [ 0.00,  8.23]    92.3%     88.4%     <-- here
#
# All 90 labelled far serves sit inside the singles baseline, INCLUDING all 23
# on the two doubles clips: the extremes are x = 0.74 and x = 7.59, leaving
# 0.74 m and 0.64 m of margin.  Doubles servers in this corpus stand between
# the centre mark and the singles sideline, as most players do.
#
# THE RISK IS NAMED RATHER THAN HIDDEN: a doubles server standing wide in the
# alley is legal and would be gated out here, and 0.64 m is not a large margin.
# Nothing in the corpus exercises it. Widen these two constants to
# -C.ALLEY_W / C.COURT_W + C.ALLEY_W the first time a real serve is lost to it;
# that costs 1.8 points of precision and nothing else.
FAR_X_LO = 0.0                           # singles sideline
FAR_X_HI = C.COURT_W                     # 8.23


def _on_baseline(court_x):
    """Box x-centre inside the far baseline's own x extent. NaN -> False."""
    x = np.asarray(court_x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(x) & (x >= FAR_X_LO) & (x <= FAR_X_HI)
SWING_FLOOR = 0.45
# Swept on the nine clips carrying a far serve.  Over the six FAR-DOMINANT
# clips (70 of the 77 labelled far serves) the curve reads:
#
#     thr    recall   precision
#     0.55   100.0%     65.4%
#     0.90   100.0%     71.4%      <-- here
#     0.95    98.6%     73.4%
#     0.999   94.3%     76.7%
#
# RE-SWEPT after the ready and stillness terms were added.  Those two multiply
# the score down, so the old 0.90 knee no longer sits in the same place; over
# all 129 labelled far serves on 13 clips:
#
#     ready  w_still  thr     recall  precision   F1
#     max      0.0    0.90     81.4%     51.0%   62.7   <- the previous default
#     max      0.3    0.85     80.6%     63.0%   70.7
#     mean     0.3    0.75     79.8%     65.2%   71.8   <- here
#     mean     0.3    0.85     70.5%     72.8%   71.7
#     mean     0.5    0.90     56.6%     72.3%   63.5
#
# 0.75 with both terms holds recall (-1.6 points) and buys FOURTEEN points of
# precision.  Pushing to 0.85 buys another 7.6 of precision for 9.3 of recall,
# which is the wrong direction for a point START -- a missed serve loses a
# whole point from the reel, an extra one is something the orchestrator can
# arbitrate.
COURT_DILATE_S = 0.4       # see `detect_serves`: the serve-zone gate is
                           # asked of the trophy's NEIGHBOURHOOD, not of each
                           # sample, because the box shape moves the projected
                           # ground point exactly at the trophy.

THRESHOLD = 0.75
REFRACT_S = 3.0

# The label lead, as on the near side: `ground_truth.json`'s `start` is a point
# boundary marked before the server moves, so it is corrected as a constant.
# The far anchor is the TROPHY ONSET, not the hands-together instant the near
# detector uses -- the hand split is not measurable on a 30 px body (see the
# module docstring), so there is no earlier anchor to walk back to.  Swept over
# the far corpus, the lead is almost inert: 0.9, 1.2 and 1.6 all produce
# identical matches at a +/-2.0 s tolerance, and only below ~0.6 does it start
# costing hits.  0.90 sits in the middle of that flat region.
SERVE_LEAD_S = 0.90

REQUIREMENT = Requirement(roi=ROI_FAR, pose_fps=15.0, needs_ball=False,
                          windows=W_BETWEEN)


def events_path(video, suffix=EVENTS_SUFFIX):
    d = WD.artifact_dir(video)
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(d, f"{stem}{suffix}")


def _sane(bbox):
    """Box height, NaN where the box cannot be trusted to normalise by."""
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    with np.errstate(invalid="ignore"):
        ok = ((h > BOX_H_MIN_PX) & (h < BOX_H_MAX_PX)
              & (w / np.maximum(h, 1e-6) < BOX_ASPECT_MAX))
    return np.where(ok, h, np.nan)


def serve_primitives(kp, bbox, fps: float,
                     eligible: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per-sample phase primitives for one far-player track.

    `kp` [N, 17, 3] and `bbox` [N, 4] are one far slot of `anya2.tracks`.
    `eligible` [N] is that module's lateral gate.  There is deliberately no
    `court_y` parameter -- see the module docstring on why a serve-zone band
    cannot be built on the far side.
    """
    kp = np.asarray(kp, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64)
    n = len(kp)
    gap = int(MAX_GAP_S * fps)
    kp = S.interp_gaps(kp, gap)
    bbox = S.interp_gaps(bbox, gap)

    h = _sane(bbox)
    valid = np.isfinite(h)

    l_sho, r_sho = S.kp_xy(kp, S.L_SHO, KP_CONF), S.kp_xy(kp, S.R_SHO, KP_CONF)
    l_wri, r_wri = S.kp_xy(kp, S.L_WRI, KP_CONF), S.kp_xy(kp, S.R_WRI, KP_CONF)
    l_hip, r_hip = S.kp_xy(kp, S.L_HIP, KP_CONF), S.kp_xy(kp, S.R_HIP, KP_CONF)
    nose = S.kp_xy(kp, S.NOSE, KP_CONF)
    l_ear, r_ear = S.kp_xy(kp, S.L_EAR, KP_CONF), S.kp_xy(kp, S.R_EAR, KP_CONF)

    sho, hip = S.mid(l_sho, r_sho), S.mid(l_hip, r_hip)
    head = S.mid(S.mid(l_ear, r_ear), nose)
    head_y = np.where(np.isfinite(head[:, 1]),
                      head[:, 1], sho[:, 1] - HEAD_ABOVE_SHO_BH * h)

    elev_l = (sho[:, 1] - l_wri[:, 1]) / h
    elev_r = (sho[:, 1] - r_wri[:, 1]) / h
    head_l = (head_y - l_wri[:, 1]) / h
    head_r = (head_y - r_wri[:, 1]) / h

    both = np.isfinite(elev_l) & np.isfinite(elev_r)
    with np.errstate(invalid="ignore"):
        hi_elev = np.where(both, np.fmax(elev_l, elev_r), np.nan)
        lo_elev = np.where(both, np.fmin(elev_l, elev_r), np.nan)
        hi_head = np.where(both, np.fmax(head_l, head_r), np.nan)
    hi_side = np.where(both, np.where(elev_l >= elev_r, 1.0, -1.0), 0.0)

    # Lateral offset of each wrist from ITS OWN shoulder (not the midpoint), in
    # body heights.  The toss score reads this; see the TOSS block for why it
    # points the opposite way to the obvious guess.
    lat_l = np.abs(l_wri[:, 0] - l_sho[:, 0]) / h
    lat_r = np.abs(r_wri[:, 0] - r_sho[:, 0]) / h

    def rel_speed(pt):
        rel = (pt - hip) / h[:, None]
        d = np.full(n, np.nan)
        d[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps
        return d

    # The player's own translation across the ground, in body heights per
    # second.  Box centre rather than a keypoint: at 30 px an individual joint
    # is noisy, while the box as a whole is what moves when the player runs.
    cx = 0.5 * (bbox[:, 0] + bbox[:, 2])
    cy_px = bbox[:, 3]
    self_move = np.full(n, np.nan)
    self_move[1:] = np.hypot(np.diff(cx), np.diff(cy_px)) / h[1:] * fps

    limb = np.fmax(rel_speed(l_wri), rel_speed(r_wri))
    still = 1.0 - S.ramp(limb, READY_STILL_BH_S * 0.5, READY_STILL_BH_S * 1.6)
    carried = 1.0 - S.ramp(hi_elev, READY_HI_MAX_BH, READY_HI_MAX_BH + 0.14)
    ready_raw = carried * still
    w = max(1, int(round(READY_WIN_S * fps)))
    ready = ready_raw if w <= 1 else np.convolve(
        np.nan_to_num(ready_raw, nan=0.0), np.ones(w) / w, mode="same")

    dil = max(1, int(round(TROPHY_DILATE_S * fps)) * 2 + 1)
    trophy = (S.movmax(S.ramp(hi_head, TROPHY_HEAD_MIN_BH, TROPHY_HEAD_FULL_BH), dil)
              * S.movmax(S.ramp(lo_elev, TROPHY_LO_MIN_BH, TROPHY_LO_FULL_BH), dil))

    on_court = (np.asarray(eligible, dtype=bool) if eligible is not None
                else np.ones(n, dtype=bool))

    return {"valid": valid & both, "ready": ready, "trophy": trophy,
            "hi_elev": hi_elev, "lo_elev": lo_elev, "hi_head": hi_head,
            "elev_l": elev_l, "elev_r": elev_r,
            "head_l": head_l, "head_r": head_r, "hi_side": hi_side,
            "lat_l": lat_l, "lat_r": lat_r,
            "still": still, "self_move": self_move,
            "on_court": on_court, "fps": np.float64(fps)}


def toss_score(prim: Dict[str, np.ndarray], k: int, toss_left: bool) -> Dict:
    """Did the tossing arm perform a toss?  Independent of the serve score.

    Returns the mean of four ramped components plus the components themselves,
    so a low score can be attributed rather than merely observed.  Missing
    keypoints yield None, which a caller must NOT read as zero -- an untracked
    arm is not evidence of a bad toss.
    """
    fps = float(prim["fps"])
    elev = prim["elev_l"] if toss_left else prim["elev_r"]
    head = prim["head_l"] if toss_left else prim["head_r"]
    lat = prim["lat_l"] if toss_left else prim["lat_r"]

    def win(x, a, b):
        lo = max(0, k + int(round(a * fps)))
        hi = min(len(x), k + int(round(b * fps)))
        q = x[lo:hi]
        return q[np.isfinite(q)]

    pre = win(elev, TOSS_PRE_FROM_S, TOSS_PRE_TO_S)
    lt = win(lat, TOSS_TOP_FROM_S, TOSS_TOP_TO_S)
    ab = win(head, TOSS_TOP_FROM_S, TOSS_TOP_TO_S)
    if not (pre.size and lt.size and ab.size):
        return {"toss": None, "toss_parts": None}

    parts = {
        "low": 1.0 - float(S.ramp(float(np.min(pre)),
                                  TOSS_LOW_FULL_BH, TOSS_LOW_NONE_BH)),
        "lat": float(S.ramp(float(np.median(lt)),
                            TOSS_LAT_MIN_BH, TOSS_LAT_FULL_BH)),
        "hold": float(S.ramp(float(np.sum(ab > 0.0) / fps),
                             TOSS_HOLD_MIN_S, TOSS_HOLD_FULL_S)),
        "peak": float(S.ramp(float(np.max(ab)),
                             TOSS_PEAK_MIN_BH, TOSS_PEAK_FULL_BH)),
    }
    return {"toss": round(float(np.mean(list(parts.values()))), 4),
            "toss_parts": {k2: round(v, 3) for k2, v in parts.items()}}


def detect_serves(prim, threshold: float = THRESHOLD,
                  require_court: bool = True,
                  track: Optional[int] = None,
                  lead_s: Optional[float] = None,
                  refract_s: Optional[float] = None,
                  w_still: Optional[float] = None) -> List[Dict]:
    """Sequence-match ready -> trophy -> swing, as `near_serve.detect_serves`."""
    fps = float(prim["fps"])
    trophy, ready = prim["trophy"], prim["ready"]
    n = len(trophy)
    tro = np.nan_to_num(trophy, nan=0.0)
    cand = tro >= TROPHY_MIN
    if require_court:
        # DILATED, because the gate flickers exactly where it hurts most.  The
        # gate is a serve-zone band on the box's projected ground point, and at
        # the trophy the box changes shape -- arms and racket go up, the box
        # grows, and the ground point moves.  On Data/21's one far serve the
        # gate is True through the wind-up and goes False ON THE TROPHY PEAK
        # FRAME, cutting a five-sample trophy run down to two and capping the
        # trophy score below the value the peak would have given: 0.671 against
        # a run that reaches 1.00 two samples later.
        #
        # A serve is one action, so the question "was this struck from the
        # serve zone" belongs to the action and not to each sample of it.
        # Dilating by COURT_DILATE_S asks it of the neighbourhood instead, which
        # is the same fix near_serve.py applies to its three shape terms and for
        # the same reason -- the phases of a serve do not line up sample for
        # sample.  The gate still rejects a player standing anywhere else; it
        # just stops rejecting the one frame the whole detection hangs on.
        w = max(1, int(round(2 * COURT_DILATE_S * fps)) | 1)
        cand = cand & (S.movmax(prim["on_court"].astype(float), w) > 0.5)

    back_lo = int(round(READY_BACK_MAX_S * fps))
    back_hi = int(round(READY_BACK_MIN_S * fps))
    fwd_lo = max(1, int(round(SWING_MIN_S * fps)))
    fwd_hi = int(round(SWING_MAX_S * fps))

    out: List[Dict] = []
    for lo, hi in S.runs(cand):
        k = lo + int(np.argmax(tro[lo:hi]))
        s_trophy = float(tro[k])

        # Ready as a MEAN over a short window ending at the trophy, not a max
        # over six seconds -- see READY_MEAN_FROM_S for why the max is inert.
        a = max(0, k - int(round(READY_MEAN_FROM_S * fps)))
        b = max(1, k - int(round(READY_MEAN_TO_S * fps)))
        rd = ready[a:b]
        rd = rd[np.isfinite(rd)]
        s_ready = float(rd.mean()) if rd.size else 0.0

        # Was this player standing still before the racket went up?
        sa = max(0, k - int(round(STILL_FROM_S * fps)))
        sb = max(1, k - int(round(STILL_TO_S * fps)))
        mv = prim["self_move"][sa:sb]
        mv = mv[np.isfinite(mv)]
        # An untracked stretch scores 0.5 rather than 0 or 1: not knowing where
        # the player was is not evidence either way, and this term must not
        # delete a serve just because the track had a hole before it.
        s_still = (1.0 - float(S.ramp(float(np.median(mv)),
                                      STILL_FULL_BH_S, STILL_NONE_BH_S))
                   if mv.size else 0.5)

        toss_left = prim["hi_side"][k] > 0
        # EITHER WRIST (SWING_EITHER_WRIST), not the one the hand split calls the racket arm.  The
        # trophy term above is already elevation-only because the split is not
        # reliable at far scale; the swing term trusting it was an
        # inconsistency, and it only became visible when the far pose pass got
        # good enough to resolve the two arms at all.
        #
        # At imgsz 960 the two wrists on Data/21's far server differ by 0.016
        # body heights -- the model is not separating them, so `hi_side` is
        # noise and whichever arm the swing term picked happened to carry the
        # raised wrist.  At imgsz 1280 they differ by 0.322, the split starts
        # meaning something, and it points at the arm that does NOT go up:
        # max(head_r) = -0.028 against max(head_l) = +0.087 over the same
        # window.  A better measurement turned a term that worked by accident
        # into one that fails, which is the sign of a latent bug rather than a
        # regression.
        #
        # "Some wrist rises above the head shortly after the trophy" is what a
        # serve looks like and needs no attribution, so that is what is asked.
        c, d = min(n, k + fwd_lo), min(n, k + fwd_hi + 1)
        s_swing, t_contact = 0.0, None
        if d > c:
            win = (np.fmax(prim["head_l"][c:d], prim["head_r"][c:d])
                   if SWING_EITHER_WRIST
                   else (prim["head_r"] if toss_left else prim["head_l"])[c:d])
            ok = np.isfinite(win)
            if ok.any():
                j = int(np.nanargmax(np.where(ok, win, -np.inf)))
                s_swing = float(S.ramp(float(win[j]), 0.0, PEAK_ABOVE_HEAD_BH))
                t_contact = (c + j) / fps

        shape = (W_TROPHY * s_trophy + W_READY * s_ready) / (W_TROPHY + W_READY)
        p = shape * (SWING_FLOOR + (1.0 - SWING_FLOOR) * s_swing)
        ws = W_STILL if w_still is None else float(w_still)
        p *= (1.0 - ws) + ws * s_still
        if p < threshold:
            continue
        out.append({
            "t": lo / fps - (SERVE_LEAD_S if lead_s is None else float(lead_s)),
            "p": round(p, 4),
            "trophy": round(s_trophy, 4), "swing": round(s_swing, 4),
            "ready": round(s_ready, 4), "still": round(s_still, 4),
            **toss_score(prim, k, bool(toss_left)),
            "t_trophy": round(k / fps, 3),
            "t_contact": round(t_contact, 3) if t_contact is not None else None,
            "track": track,
        })
    return S.refractory(out, REFRACT_S if refract_s is None else float(refract_s))


# ── one far player, two slots ────────────────────────────────────────────
# The tracker gives the far side two slots and ONE player oscillates between
# them.  Measured over the corpus, the two far slots are simultaneously tracked
# on 0.9-18% of frames, while taking whichever is available at each frame gains
# 22-38 POINTS of coverage over the better single slot:
#
#     clip   slot2   slot3   either   both at once   gain over best slot
#      21    43.8%   39.8%   65.8%       17.9%            +22.0
#      24    42.0%   39.2%   80.4%        0.9%            +38.4
#      35    41.4%   55.0%   90.1%        6.3%            +35.1
#      43    35.7%   25.7%   60.5%        0.9%            +24.8
#
# That matters because this detector scores each slot INDEPENDENTLY, and two of
# its four terms are window statistics: `ready` is a mean over a window before
# the trophy and `still` a median over one.  A window half full of NaN does not
# fail loudly, it scores LOW -- so a serve by a player whose track keeps jumping
# slots is scored as a player who was never ready and never still.
#
# DOUBLES WAS THE REASON TO EXPECT THIS TO BE GATED, AND IT MEASURED THE OTHER
# WAY.  Clips 25 and 40 have two real far players, co-tracking 36.0% and 22.7%
# of frames, so merging them should fuse two people into one timeline and lose
# serves.  It does not -- both clips keep 100% recall and gain precision
# (25: 90.9% -> 100%, 40: 86.7% -> 92.9%).  Only one player serves, the
# continuity preference below follows whoever the tracker is holding, and the
# partner standing at the baseline does not produce a trophy.  So the merge is
# unconditional, and every attempt to gate it was worse than not gating it.
# Two gates were tried and neither separates the corpus: the co-track rate puts
# singles clip 21 (27% of tracked frames) above doubles clip 40 (30%), and the
# spatial gap between co-tracked slots puts singles clip 21 at 2.6 body heights
# and doubles clip 40 at 0.03.
#
# Corpus effect, at the shipped threshold:
#
#     per slot   recall 89.0%   precision 81.0%
#     merged     recall 92.3%   precision 86.6%


def merge_far_slots(kp, bbox, eligible, court=None):
    """One timeline from the two far slots. Returns (kp, bbox, eligible[, court]).

    Where only one slot is tracked, take it.  Where both are, prefer the slot
    used for the PREVIOUS frame -- continuity is the whole point, and switching
    on a per-frame tie-break would reintroduce the oscillation this removes.
    With no previous choice, take the taller box: at far-court scale the better
    detection is the bigger one.
    """
    n = len(bbox)
    ok = [np.isfinite(bbox[:, s, 0]) for s in range(bbox.shape[1])]
    h = [bbox[:, s, 3] - bbox[:, s, 1] for s in range(bbox.shape[1])]
    out_kp = np.full(kp.shape[:1] + kp.shape[2:], np.nan, dtype=kp.dtype)
    out_bb = np.full((n, 4), np.nan, dtype=bbox.dtype)
    out_el = np.zeros(n, dtype=bool)
    out_ct = np.full((n, 2), np.nan) if court is not None else None
    prev = None
    for i in range(n):
        live = [s for s in (0, 1) if ok[s][i]]
        if not live:
            continue
        if len(live) == 1:
            pick = live[0]
        elif prev in live:
            pick = prev
        else:
            pick = 0 if h[0][i] >= h[1][i] else 1
        out_kp[i] = kp[i, pick]
        out_bb[i] = bbox[i, pick]
        out_el[i] = eligible[i, pick]
        if out_ct is not None:
            out_ct[i] = court[i, pick]
        prev = pick
    if out_ct is not None:
        return out_kp, out_bb, out_el, out_ct
    return out_kp, out_bb, out_el


def detect_video(video, tracks_npz=None, threshold: float = THRESHOLD,
                 require_court: bool = True, verbose: bool = True,
                 lead_s: Optional[float] = None,
                 refract_s: Optional[float] = None,
                 w_still: Optional[float] = None,
                 merge_slots: Optional[bool] = None):
    """Score every far slot; the server is whichever produced the candidate."""
    z = T.load(video, tracks_npz)
    fps = float(z["fps"])
    kp, bbox, el = z["kp"], z["bbox"], z["eligible"]

    # One player across two slots?  Then score one merged timeline instead of
    # two fragments.  See FAR_MERGE_MAX_COTRACK.
    fa = np.isfinite(bbox[:, T.FAR_SLOTS[0], 0])
    fb = np.isfinite(bbox[:, T.FAR_SLOTS[1], 0])
    either = float((fa | fb).mean())
    if merge_slots is None:
        merge_slots = True
    union_mode = (merge_slots == "both")
    if (merge_slots or union_mode) and either > 0:
        mk, mb, me, mc = merge_far_slots(kp[:, list(T.FAR_SLOTS)],
                                         bbox[:, list(T.FAR_SLOTS)],
                                         el[:, list(T.FAR_SLOTS)],
                                         court=z["court"][:, list(T.FAR_SLOTS)])
        prim = serve_primitives(mk, mb, fps,
                                eligible=me & _on_baseline(mc[:, 0]))
        ev = detect_serves(prim, threshold, require_court,
                           track=int(T.FAR_SLOTS[0]), lead_s=lead_s,
                           refract_s=refract_s, w_still=w_still)
        if verbose:
            print(f"[far-serve] merged far slots: coverage "
                  f"{100*either:.1f}% -> {len(ev)} candidates")
        if not union_mode:
            return [Event(t=float(e["t"]), p=float(e["p"]), kind=FAR_SERVE,
                          track=e["track"],
                          detail={k: e[k] for k in ("trophy", "swing", "ready",
                                                    "still", "toss",
                                                    "toss_parts", "t_trophy",
                                                    "t_contact")})
                    for e in S.refractory(ev, REFRACT_S if refract_s is None
                                          else float(refract_s))]
        merged_ev = ev
    else:
        merged_ev = []

    raw: List[Dict] = list(merged_ev)
    for slot in T.FAR_SLOTS:
        seen = np.isfinite(bbox[:, slot, 0])
        if not seen.any():
            if verbose:
                print(f"[far-serve] slot {slot}: never tracked, skipped")
            continue
        prim = serve_primitives(kp[:, slot], bbox[:, slot], fps,
                                eligible=(el[:, slot]
                                          & _on_baseline(z["court"][:, slot, 0])))
        ev = detect_serves(prim, threshold, require_court, track=int(slot),
                           lead_s=lead_s, refract_s=refract_s, w_still=w_still)
        if verbose:
            print(f"[far-serve] slot {slot}: tracked {100 * seen.mean():5.1f}%"
                  f"  sane boxes {100 * np.mean(prim['valid']):5.1f}%"
                  f"  -> {len(ev)} candidates")
        raw.extend(ev)

    kept = S.refractory(raw, REFRACT_S if refract_s is None else float(refract_s))
    return [Event(t=float(e["t"]), p=float(e["p"]), kind=FAR_SERVE,
                  track=e["track"],
                  detail={k: e[k] for k in ("trophy", "swing", "ready",
                                            "still", "toss", "toss_parts",
                                            "t_trophy", "t_contact")})
            for e in kept]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("video")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-court", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ev = detect_video(a.video, a.tracks, a.threshold, not a.no_court)
    print(f"[far-serve] {len(ev)} serves at p >= {a.threshold}")
    for e in ev[:40]:
        d = e.detail
        print(f"  {e.t:8.2f}s  p={e.p:.3f}  slot={e.track}  "
              f"trophy={d['trophy']:.2f} swing={d['swing']:.2f} ready={d['ready']:.2f}")
    out = a.json or events_path(a.video)
    dump_events(ev, out, threshold=a.threshold, requirement=REQUIREMENT.__dict__)
    print(f"[far-serve] wrote {out}")


if __name__ == "__main__":
    main()
