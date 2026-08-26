"""
near_serve.py
=============
A near-side serve detector built from POSE ALONE — no ball evidence anywhere.

Why a second near detector at all
---------------------------------
`anya_near_serve` scores three cues (dwell, toss, ratio-jerk) and two of them
saturate, so in practice its P is dwell-only: "the near player stood near the
baseline and held still".  That is a description of a player about to serve AND
of a player who has just picked up a ball, is arguing a call, or is waiting for
their opponent.  Measured against Data/77's tags, 33 of 44 false positives fell
in tagged DEAD TIME rather than mid-rally, which is exactly what a dwell-shaped
cue should be expected to do.

The fix is not a higher threshold on the same cue.  It is to require the thing
that actually distinguishes a serve from standing still: THE SERVE MOTION.

The motion, and what survives 15 Hz
-----------------------------------
The service action has three phases with distinct, pose-visible shapes:

  1. READY     hands together on the racket, both wrists carried between hip and
               shoulder, body quiet.  Lasts 1-4 s, so at the 15 Hz the pose npz
               is written at this is 15-60 samples.  Abundant, and cheap.

  2. TROPHY    the hands SPLIT and both go up.  The tossing wrist rises clear
               above the head; the racket wrist is above the shoulder with the
               elbow flexed.  Held ~0.2-0.5 s, so 3-7 samples.  Thin, but this
               is the discriminative core: "both wrists above the shoulder line,
               one of them above the head" is a shape that between-point
               behaviour essentially never produces.

  3. SWING     the racket wrist accelerates to a peak above the head while the
               tossing wrist DROPS.  The two wrists trade vertical order.  This
               takes 0.10-0.15 s — 1.5-2 samples at 15 Hz.

That last number sets a hard design rule: **the swing is read as a position
extreme, never as a velocity or an acceleration.**  At 15 Hz the contact spans
about two samples, so any derivative across it is dominated by sampling phase
rather than by the stroke — the same trap that made the ratio-jerk cue
meaningless at 5 fps (see NearServeConfig.jerk_mode).  What IS robust across two
samples is "the racket wrist got higher than it had been", because an extremum
does not care where in the motion the samples landed.

Three ideas carry the detector, and only the first is in the phase list above:

  * THE HAND SPLIT.  Ready has the hands together on the grip; trophy has them
    as far apart as they get all point.  The transition together -> apart is
    sharper and better sampled than anything inside the trophy itself.

  * THE WRIST-ORDER CROSSOVER.  At trophy the tossing wrist is the higher one;
    a fifth of a second later the racket wrist is.  Nothing in the between-point
    repertoire swaps which arm is on top while both are above the shoulders.

  * THE ORDERING ITSELF.  ready -> trophy -> peak, each within a bounded delay.
    A single thresholded shape is what `anya_far_serve`'s hand-raise gate is,
    and its measured cost is 29-37 false positives per corpus.  Requiring the
    three to arrive in sequence is what a lone raise cannot fake.

What pose alone genuinely cannot do
-----------------------------------
An OVERHEAD SMASH is the same motion.  Trophy shape, wrist crossover, peak above
the head — a smash has all of it, and no amount of keypoint work separates the
two, because they are not different at the joints.  The separators are
contextual, and one of them is still ball-free: a serve is struck at or behind
the baseline, a smash almost never is.  So `court_y` gates the detector when a
homography is available (`require_court=True`, the default when it is).  This is
pose + court geometry, not ball evidence.  Without the gate, expect smashes.

Handedness is never assumed.  Every measurement is over "the higher wrist" and
"the lower wrist", or over the two sides symmetrically, so a lefty and a righty
score identically and no per-clip configuration is needed.

Reference frames
----------------
Image y grows DOWNWARD, so "above" is a SMALLER y.  Every elevation below is
written as (reference_y - point_y) / body_height, i.e. positive = higher up, in
body heights — the same normalisation `near_end` uses, and for the same reason:
the near player's box ranges over tens of pixels of height as they move, so a
pixel threshold means different postures at different depths.

The head reference deserves its own note.  The near player has their BACK to
this camera, so NOSE/EYE/EAR keypoints are exactly the ones a pose model is
least sure of here.  `_head_y` therefore uses the detected head when it is
confidently seen and falls back to a fixed offset above the shoulder line when
it is not — never NaN, because "we could not see the face" must not be allowed
to read as "the wrist was not above the head".
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# COCO indices, as in near_end / walking.features.
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12

MAX_GAP_S = 0.5          # as near_end: fill short occlusions, leave real holes

# ── head reference ───────────────────────────────────────────────────────
HEAD_ABOVE_SHO_BH = 0.14  # fallback head height above the shoulder line, body
                          # heights.  A head is ~0.13 of standing height and the
                          # shoulder line sits just below it; erring LOW makes
                          # "above the head" easier to satisfy, so the fallback
                          # is the permissive direction and the gate leans on
                          # the trophy shape rather than on this constant.

# ── ready phase ──────────────────────────────────────────────────────────
TOGETHER_BH = 0.10        # wrist separation at or under this = hands on the grip
SPLIT_BH    = 0.26        # ...and at or over this they are definitely apart
READY_HI_MAX_BH = 0.10    # in ready, even the higher wrist stays under this far
                          # above the shoulder line.  Not below the shoulder:
                          # plenty of players carry the racket head-high while
                          # waiting, and rejecting that costs real serves.
READY_STILL_BH_S = 0.9    # limb speed relative to the hip, body heights/s,
                          # under which the player counts as quiet.  Looser than
                          # near_end.SETTLE_LIMB_BH_S (0.75) on purpose: a
                          # server bouncing a ball is NOT still by that standard
                          # but is unambiguously in the ready phase.
READY_WIN_S = 0.7         # sustain window; a serve stance is held

# ── trophy phase ─────────────────────────────────────────────────────────
# ── how strict is the trophy? ────────────────────────────────────────────
# These three full-credit points were loosened by 35% of their range toward
# their own no-credit floors, because SOME SERVERS SIMPLY DO NOT MAKE A TEXTBOOK
# TROPHY -- a compact or abbreviated service motion carries the racket lower and
# splits the hands less, and the original bands scored those at nearly zero
# however clean the rest of the sequence was.
#
# Swept over all 107 labelled near serves.  Loosening trades a little precision
# for recall and the trade is worth taking for a point START, where a miss loses
# a whole point from the reel and an extra start is something the composition
# layer can arbitrate:
#
#     loosen   recall   precision   F1
#      0.00     86.9%     78.2%    82.3      (the original bands)
#      0.20     88.8%     77.9%    83.0
#      0.35     90.7%     76.4%    82.9      <-- here
#      0.50     89.7%     74.4%    81.4
#
# 0.35 is where recall peaks; 0.50 gives it back, so this is a maximum and not
# a slope being ridden. F1 is flat across 0.20-0.35, so the choice between them
# is the recall/precision preference above, not a measurement.
#
# TROPHY_MIN is deliberately NOT part of this. Swept from 0.35 down to 0.16 it
# changes nothing at all -- the run threshold is not binding, because the final
# probability threshold already dominates it. Lowering it would look like
# loosening while doing nothing.
TROPHY_ABOVE_HEAD_BH = 0.058  # higher wrist this far over the head = full credit
TROPHY_HEAD_MIN_BH   = -0.02  # ...and no credit at all below head height
# The lower wrist is measured against the SHOULDER LINE, and the band below is
# set from data rather than from anatomy-by-description.  The original 0.00/0.14
# encoded "the other wrist is also up", read literally as above the shoulder.
# That is wrong: at trophy the racket ARM is up but the racket WRIST hangs just
# under the shoulder with the elbow flexed — across Data/21's 11 tagged near
# serves the lower wrist sits at -0.096..+0.079, median -0.028.  The original
# band scored every one of those at or near zero.  Away from serves the same
# quantity has p50 -0.276 and p95 -0.157, so the discriminative band is narrow
# and sits just below the shoulder, not above it.
TROPHY_LO_MIN_BH = -0.13      # lower wrist this far BELOW the shoulder = no credit
TROPHY_LO_FULL_BH = -0.052    # ...and near the shoulder = full credit.
                              # This is the term that rejects the whole one-armed
                              # family (hand to cap, hand to face, a wave, a
                              # raised finger): those leave the other wrist down.
TROPHY_SPLIT_MIN_BH = 0.149   # hands must have come apart to score at all
TROPHY_MIN = 0.35             # run threshold for calling a sample "trophy"

# THE TROPHY IS A PHASE, NOT AN INSTANT.
#
# The three terms above are multiplied, so as written they must all hold ON THE
# SAME SAMPLE.  Measured on Data/38's five missed serves, they do not: at each
# one every term comfortably clears its own threshold somewhere in the window
# (hi_head peaks at +0.19..+0.27 against a 0.10 full-credit line, lo_elev at
# +0.16..+0.29 against -0.01, gap at 0.33..0.61 against 0.18) while the PRODUCT
# never exceeds 0.30 and mostly sits at 0.00.  The terms are satisfied; they are
# just not satisfied simultaneously.
#
# That is a sampling artefact, not a shape failure.  The trophy is held for
# 0.2-0.5 s, which is 3-7 samples at 15 Hz, and within it the tossing arm
# reaches full extension before the racket arm settles -- so the two elevation
# terms peak a sample or two apart and the conjunction falls between them.  It
# is the same class of problem as reading contact off a derivative, and it has
# the same answer: measure the phase over a window rather than at a point.
#
# So each term is dilated (running max) before the product.  The conjunction
# then asks "did this hold ANYWHERE NEAR now", which is what a phase means,
# instead of "did all three hold in this one 67 ms sample".
TROPHY_DILATE_S = 0.20        # +/- this around each sample; ~3 samples at 15 Hz,
                              # comfortably inside the 0.2-0.5 s the trophy is
                              # held, so it cannot merge two separate actions

# ── swing phase ──────────────────────────────────────────────────────────
SWING_MIN_S = 0.06        # contact cannot precede the trophy by less than one
                          # sample, so the search opens just past it
SWING_MAX_S = 0.90        # ...and a real trophy-to-contact is 0.15-0.45 s; the
                          # slack is for a slow recreational action, not for
                          # finding an unrelated arm raise a second later
PEAK_ABOVE_HEAD_BH = 0.06 # the racket wrist must clear the head at contact
# The tossing arm's descent is DELIBERATELY NOT REQUIRED.  The original design
# read contact as a wrist-order crossover — racket wrist up AND toss wrist
# falling away — but measured on Data/22 that second half is simply not true at
# these sample rates: across 7 missed serves the racket wrist peaked above the
# head on time (0.20-0.23 bh at +0.20..+0.53 s) while the tossing arm had not
# yet come down, so the crossover term zeroed a swing that had plainly happened.
# The toss arm drops AFTER contact, not by it.  Contact is therefore read as the
# racket wrist clearing the head, full stop, and the tossing arm is free to stay
# up through the whole swing.

# ── ready lookback ───────────────────────────────────────────────────────
READY_BACK_MIN_S = 0.20   # the ready phase ends as the hands split, so the
                          # window closes just before the trophy
READY_BACK_MAX_S = 6.00   # and opens far enough back to cover a long ritual

# ── court gate ───────────────────────────────────────────────────────────
# `walking.court.load_homography` maps the clicked corners so that the NEAR
# baseline is court_y = 0 and court_y grows toward the far end — verified on
# Data/21, where the near player's court_y sits at -0.07..+0.10 across all 11
# tagged near serves and ranges to +12.8 during rallies.  A gate written the
# other way round (">= 21.3", i.e. assuming the near baseline was at COURT_L)
# rejects 100% of samples, which is how this was found.
# The gate is a BAND on the server's FEET, not a one-sided test on the body.
# `walking.features.frame_signals` already projects the ankle midpoint (falling
# back to the bbox bottom-centre when both ankles are lost) through the court
# homography, so `court_y` is the ground-contact point — which is what a
# standing-position rule has to be measured on.
#
# The band is the same idea as `anya_near_serve.NearServeConfig`'s ready zone
# (zone_y_min_ft -3.5, zone_y_max_ft 0.5), sharing its lower edge so the two
# near detectors agree about how far back a server may stand.  The court-side
# edge is 1.0 ft rather than 0.5: the ready zone is scored on a player who is
# still settling, whereas this gate is evaluated at the TROPHY, by which point
# the front foot has rocked forward into the swing and a half-foot ceiling
# clips real serves.
FT_TO_M = 0.3048
SERVE_ZONE_MIN_FT = -3.5  # behind the baseline; a server stands back from it
SERVE_ZONE_MAX_FT =  1.0  # ...and no further INTO the court than this, which is
                          # what excludes mid-court overheads and smashes

# ── combination / events ─────────────────────────────────────────────────
# The shape terms are additive; the SWING is a multiplicative gate with a floor.
#
# That asymmetry is the whole arbitration, and it was chosen against a measured
# failure rather than picked for elegance.  Under a flat additive form
# (0.45/0.35/0.20) two behaviours scored 0.600 and fired: a player who stands
# quiet with the hands together and THEN raises both arms to stretch or adjust
# strings, and a server who tosses and catches an abandoned toss.  Both have a
# real ready phase and a textbook trophy shape.  Neither has a racket wrist that
# peaks above the head while the tossing wrist falls away, because in neither
# case was a ball ever struck.
#
# Additive weighting cannot fix that: trophy and ready are genuinely present, so
# any threshold that rejects those two also rejects real serves scoring the same
# 0.65 on the same two terms.  The swing has to be able to VETO, which means it
# multiplies.  The floor keeps the veto soft — at 15 Hz contact spans about two
# samples, so a real serve can be caught mid-stroke with a partial swing score,
# and a hard gate would throw those away.
W_TROPHY, W_READY = 0.45, 0.20
SWING_FLOOR = 0.45        # what a candidate retains with NO swing evidence at
                          # all.  Chosen so the two adversarial cases above land
                          # at 0.42 (rejected) while a serve seen with only half
                          # its swing lands at 0.67 (kept).
# Swept over the nine labelled clips with a near serve on them.  The curve is
# flat from 0.60 to 0.70 (recall 100%, 4 false positives) and only starts
# costing recall at 0.75, so this is the top of a plateau rather than a point
# fitted to the corpus -- which is the only reason a threshold chosen on the
# same clips it is scored on is worth anything.  Clip 58's 44 near serves are
# the holdout.
THRESHOLD = 0.70
REFRACT_S = 3.0           # as anya_near_serve.event_refract_s

# ── when is the point start? ─────────────────────────────────────────────
# The seed reported the TROPHY ONSET as the serve time, and measured against
# `ground_truth.json` that is systematically LATE: over 46 matched near serves
# on nine clips the trophy onset lands +1.63 s after the label, sd 0.66.
#
# The first fix is physical.  The seed's own docstring names the hand split as
# the sharpest, best-sampled transition in the whole action, but only ever uses
# it as a SHAPE term inside the trophy product, never as a TIME anchor.  Walking
# back from the trophy peak to the last sample at which the hands were still
# together on the grip -- the last instant of the ready stance -- moves the
# residual to +1.13 s and, more usefully, cuts the share of detections already
# inside +/-2.0 s of the label from 63% to 84%.
#
# The remaining +1.13 s is NOT physical and is not treated as though it were.
# `ground_truth.json`'s `start` is a POINT boundary, not a stroke event: it is
# marked before the server's hands ever move, so no definition taken from the
# serve motion can reach it.  Every candidate onset tried (trophy onset, hands
# together, ready>=0.5, wrists-still-low) left a residual of +1.10 to +1.63 s,
# which is what a convention offset looks like and not what a mis-detection
# looks like.  So it is corrected as one explicitly-fitted constant.
#
# It is ONE parameter over 45 observations, and it was validated the only way
# that means anything -- LEAVE-ONE-CLIP-OUT, refitting on the other eight clips
# and scoring the held-out one: 44/45 (98%) land within +/-2.0 s.  Per-clip
# medians span 0.93-2.16 s, so the constant is not equally right everywhere;
# it is right within the tolerance the corpus is scored at.  A detector asked
# to hit a tighter tolerance than 2 s would need a per-clip estimate, and this
# constant should be refitted, not trusted, if the labelling convention changes.
# 1.63 is the MEDIAN OVER ALL TEN labelled clips carrying a near serve.  Fitted
# on the nine short clips alone it was 1.13, and clip 58 -- a full 55-minute
# match, held out until its perception pass finished -- wanted 2.4.  That is
# the honest headline of this constant: the per-clip lead genuinely spans
# 0.9-2.4 s, and no single value satisfies both ends at a +/-2.0 s tolerance.
# Measured directly: at 1.13 clip 21 scores 100% and clip 58 57%; at 2.00 clip
# 21 drops to 91% and clip 58 reaches 68%.  Clips 22-50 do not move at all.
#
# So this is a compromise, not a fit, and it is a small effect: the lead is
# worth about nine points of recall on clip 58 and the remaining third of that
# clip's misses are a different problem (see the README).  A detector required
# to hold a tighter tolerance would need a PER-CLIP lead, estimated from the
# clip's own detections rather than from this constant.
SERVE_LEAD_S = 1.63       # subtracted from the hands-together time
TROPHY_LEAD_S = 2.13      # ...and from the trophy onset, when the hands were
                          # never seen together (occluded grip).  Fitted on the
                          # same 46 detections, so both paths land on the same
                          # convention rather than one silently drifting.
TOGETHER_BACK_S = 6.00    # how far back from the trophy peak to look for the
                          # ready stance; matches READY_BACK_MAX_S


def _kin():
    """`walking.features`' pose helpers, imported lazily — see near_end._kin."""
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from walking.features import _interp_gaps, _kp_xy, _mid, _movmean
    return _interp_gaps, _kp_xy, _mid, _movmean


def _ramp(x, lo, hi):
    """Soft 0->1 ramp from lo to hi (either order); NaN in, NaN out."""
    with np.errstate(invalid="ignore"):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _movmean_nan(x, w, movmean):
    return x if w <= 1 else movmean(x, w)


def _movmax(x, w):
    """Centred running max over `w` samples, NaN-tolerant (NaN acts as -inf).

    Written with shifts rather than a stride trick so the edges degrade to a
    shorter window instead of to NaN -- a serve in the first second of a clip
    is still a serve.
    """
    if w <= 1:
        return x
    k = w // 2
    out = np.array(x, dtype=np.float64, copy=True)
    for d in range(1, k + 1):
        out[:-d] = np.fmax(out[:-d], x[d:])
        out[d:] = np.fmax(out[d:], x[:-d])
    return out


def serve_primitives(kp, bbox, fps: float,
                     court_y: Optional[Sequence[float]] = None,
                     eligible: Optional[Sequence[bool]] = None) -> Dict[str, np.ndarray]:
    """Per-sample phase primitives for one near-player pose track.

    `kp` [N, 17, 3] and `bbox` [N, 4] are one near slot of `anya2.tracks`,
    the same layout `near_end.near_signals` consumes.  `court_y` [N] in metres is
    that slot's court position and enables the baseline gate.  `eligible` [N]
    is `anya2.tracks`' strict lateral gate -- the user's "inside the doubles
    court plus 3 ft" rule -- and is ANDed into it.

    The two gates are separate on purpose and are not interchangeable.
    `eligible` asks WHO IS A PLAYER ON THIS COURT; the serve-zone band below
    asks WHO IS STANDING WHERE A SERVER STANDS.  A receiver passes the first
    and fails the second, which is why both are kept.

    Returns arrays of length N.  Scores are in [0, 1] with NaN where the player
    was not tracked; geometry terms are in body heights and may be NaN freely.
    """
    _interp_gaps, _kp_xy, _mid, _movmean = _kin()
    kp = np.asarray(kp, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64)
    n = len(kp)
    gap = int(MAX_GAP_S * fps)
    kp = _interp_gaps(kp, gap)
    bbox = _interp_gaps(bbox, gap)

    h = bbox[:, 3] - bbox[:, 1]
    h = np.where(h > 1.0, h, np.nan)
    valid = np.isfinite(h)

    l_sho, r_sho = _kp_xy(kp, L_SHO), _kp_xy(kp, R_SHO)
    l_wri, r_wri = _kp_xy(kp, L_WRI), _kp_xy(kp, R_WRI)
    l_hip, r_hip = _kp_xy(kp, L_HIP), _kp_xy(kp, R_HIP)
    nose = _kp_xy(kp, NOSE)
    l_ear, r_ear = _kp_xy(kp, L_EAR), _kp_xy(kp, R_EAR)

    sho = _mid(l_sho, r_sho)
    hip = _mid(l_hip, r_hip)
    head = _mid(_mid(l_ear, r_ear), nose)

    # Head reference: measured when seen, geometric when not.  Never NaN — see
    # the module docstring on why a missing face must not read as a low wrist.
    head_y = np.where(np.isfinite(head[:, 1]),
                      head[:, 1], sho[:, 1] - HEAD_ABOVE_SHO_BH * h)

    # Elevations, positive = higher in the frame, in body heights.
    elev_l = (sho[:, 1] - l_wri[:, 1]) / h        # above the shoulder line
    elev_r = (sho[:, 1] - r_wri[:, 1]) / h
    head_l = (head_y - l_wri[:, 1]) / h           # above the head
    head_r = (head_y - r_wri[:, 1]) / h

    # Both wrists must actually be SEEN for a two-armed shape to mean anything:
    # np.fmax ignores NaN, so without this an occluded second arm would be
    # silently read as whatever the visible arm is doing.
    both = np.isfinite(elev_l) & np.isfinite(elev_r)
    with np.errstate(invalid="ignore"):
        hi_elev = np.where(both, np.fmax(elev_l, elev_r), np.nan)
        lo_elev = np.where(both, np.fmin(elev_l, elev_r), np.nan)
        hi_head = np.where(both, np.fmax(head_l, head_r), np.nan)
    # Which side is carrying the high wrist: +1 left, -1 right, 0 unknown.
    hi_side = np.where(both, np.where(elev_l >= elev_r, 1.0, -1.0), 0.0)

    gap_bh = np.linalg.norm(l_wri - r_wri, axis=1) / h

    # ── limb quiet, for the ready phase ──────────────────────────────────
    # Relative to the hip, so a player walking into position does not read as
    # limb activity and a racket take-back while standing still does — the same
    # construction as near_end's settle term.
    def rel_speed(pt):
        rel = (pt - hip) / h[:, None]
        d = np.full(n, np.nan)
        d[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps
        return d

    limb = np.fmax(rel_speed(l_wri), rel_speed(r_wri))
    still = 1.0 - _ramp(limb, READY_STILL_BH_S * 0.5, READY_STILL_BH_S * 1.6)

    # ── ready ────────────────────────────────────────────────────────────
    together = 1.0 - _ramp(gap_bh, TOGETHER_BH, SPLIT_BH)
    carried = 1.0 - _ramp(hi_elev, READY_HI_MAX_BH, READY_HI_MAX_BH + 0.14)
    ready_raw = together * carried * still
    ready = _movmean_nan(ready_raw, max(1, int(round(READY_WIN_S * fps))), _movmean)

    # ── trophy ───────────────────────────────────────────────────────────
    # A product, not a sum: all three have to hold at once.  Both wrists up AND
    # one over the head AND the hands apart is the conjunction that between-point
    # behaviour does not produce; any two of the three regularly happen alone.
    # Each term is dilated over TROPHY_DILATE_S before the product -- see the
    # constant.  Dilating the RAMPS rather than the raw geometry keeps every
    # term bounded in [0, 1], so the product is still a score and not a
    # quantity that a noisy keypoint can drive arbitrarily high.
    dil = max(1, int(round(TROPHY_DILATE_S * fps)) * 2 + 1)
    trophy = (_movmax(_ramp(hi_head, TROPHY_HEAD_MIN_BH, TROPHY_ABOVE_HEAD_BH), dil)
              * _movmax(_ramp(lo_elev, TROPHY_LO_MIN_BH, TROPHY_LO_FULL_BH), dil)
              * _movmax(_ramp(gap_bh, TROPHY_SPLIT_MIN_BH,
                              TROPHY_SPLIT_MIN_BH + 0.12), dil))

    if court_y is not None:
        cy = np.asarray(court_y, dtype=np.float64)
        on_court = ((cy >= SERVE_ZONE_MIN_FT * FT_TO_M)
                    & (cy <= SERVE_ZONE_MAX_FT * FT_TO_M))
        # A lost foot must not silently pass the gate: court_y is NaN there and
        # both comparisons are False, so an untracked player is excluded rather
        # than admitted.  That is the conservative direction for a position rule.
    else:
        on_court = np.ones(n, dtype=bool)

    if eligible is not None:
        on_court = on_court & np.asarray(eligible, dtype=bool)

    return {
        "valid": valid & both,
        "ready": ready,
        "trophy": trophy,
        "elev_l": elev_l, "elev_r": elev_r,
        "head_l": head_l, "head_r": head_r,
        "hi_elev": hi_elev, "lo_elev": lo_elev, "hi_head": hi_head,
        "hi_side": hi_side,
        "gap_bh": gap_bh,
        "still": still,
        "on_court": on_court,
        "fps": np.float64(fps),
    }


def serve_onset(prim: Dict[str, np.ndarray], k: int,
                lead_s: Optional[float] = None) -> tuple:
    """Point-start time for a trophy peaking at sample `k`, and how it was found.

    Returns (t_seconds, basis) with basis "together" or "trophy" -- the basis is
    carried into the event so a timing complaint can be traced to which of the
    two calibrations produced it, rather than to an unlabelled number.
    """
    fps = float(prim["fps"])
    gap = np.nan_to_num(prim["gap_bh"], nan=np.inf)
    a = max(0, k - int(round(TOGETHER_BACK_S * fps)))
    idx = np.flatnonzero(gap[a:k] <= TOGETHER_BH)
    lead = SERVE_LEAD_S if lead_s is None else float(lead_s)
    # The trophy fallback keeps its offset RELATIVE to the hands-together lead,
    # so overriding one moves both onto the same convention.
    tro_lead = lead + (TROPHY_LEAD_S - SERVE_LEAD_S)
    if idx.size:
        return (a + int(idx[-1])) / fps - lead, "together"
    return k / fps - tro_lead, "trophy"


def _runs(mask: np.ndarray) -> List[tuple]:
    """[(lo, hi)) index runs of True."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(m.astype(np.int8))
    lo = list(np.flatnonzero(d == 1) + 1)
    hi = list(np.flatnonzero(d == -1) + 1)
    if m[0]:
        lo.insert(0, 0)
    if m[-1]:
        hi.append(len(m))
    return list(zip(lo, hi))


def detect_serves(prim: Dict[str, np.ndarray],
                  threshold: float = THRESHOLD,
                  require_court: bool = True,
                  track: Optional[int] = None,
                  lead_s: Optional[float] = None,
                  refract_s: Optional[float] = None) -> List[Dict]:
    """Sequence-match ready -> trophy -> swing over the primitives.

    Each trophy run is one candidate.  The run's peak sample fixes the trophy
    time and which arm is tossing; the ready term is searched BACKWARD and the
    swing term FORWARD, both in bounded windows.  A candidate that cannot find
    its swing scores zero on that term rather than being discarded outright, so
    the arbitration stays graded — a strong trophy over a long, clean ready is
    still allowed to carry a serve whose contact fell between two samples.

    Returns dicts with `t` (trophy onset — see `serve_t` below), `p`, the three
    component scores, and the phase timestamps, sorted by time.
    """
    fps = float(prim["fps"])
    trophy, ready = prim["trophy"], prim["ready"]
    n = len(trophy)

    tro = np.nan_to_num(trophy, nan=0.0)
    cand = tro >= TROPHY_MIN
    if require_court:
        cand = cand & prim["on_court"]

    back_lo, back_hi = int(round(READY_BACK_MAX_S * fps)), int(round(READY_BACK_MIN_S * fps))
    fwd_lo, fwd_hi = max(1, int(round(SWING_MIN_S * fps))), int(round(SWING_MAX_S * fps))

    out: List[Dict] = []
    for lo, hi in _runs(cand):
        k = lo + int(np.argmax(tro[lo:hi]))
        s_trophy = float(tro[k])

        # Ready: the best sustained ready score in the lookback window.
        a, b = max(0, k - back_lo), max(0, k - back_hi)
        rd = ready[a:b]
        rd = rd[np.isfinite(rd)]
        s_ready = float(rd.max()) if rd.size else 0.0

        # Swing: the tossing arm is whichever carried the high wrist at trophy;
        # the racket arm is the other one.  Contact wants the racket wrist over
        # the head AND the tossing wrist fallen away from its own trophy height.
        toss_left = prim["hi_side"][k] > 0
        rack_head = prim["head_r"] if toss_left else prim["head_l"]

        c, d = min(n, k + fwd_lo), min(n, k + fwd_hi + 1)
        s_swing, t_contact = 0.0, None
        if d > c:
            win_head = rack_head[c:d]
            ok = np.isfinite(win_head)
            if ok.any():
                j = int(np.nanargmax(np.where(ok, win_head, -np.inf)))
                peak = float(win_head[j])
                s_swing = float(_ramp(peak, 0.0, PEAK_ABOVE_HEAD_BH))
                t_contact = (c + j) / fps

        # Shape first, normalised so a perfect trophy over a perfect ready is
        # 1.0, then gated by the swing — see the SWING_FLOOR comment.
        shape = (W_TROPHY * s_trophy + W_READY * s_ready) / (W_TROPHY + W_READY)
        p = shape * (SWING_FLOOR + (1.0 - SWING_FLOOR) * s_swing)
        if p < threshold:
            continue
        t_start, basis = serve_onset(prim, k, lead_s=lead_s)
        out.append({
            "t": t_start,                     # POINT START: see SERVE_LEAD_S
            "t_basis": basis,
            "p": round(p, 4),
            "trophy": round(s_trophy, 4),
            "swing": round(s_swing, 4),
            "ready": round(s_ready, 4),
            "t_trophy": round(k / fps, 3),
            "t_contact": round(t_contact, 3) if t_contact is not None else None,
            "toss_arm": "left" if toss_left else "right",
            "track": track,
        })

    # Refractory: keep the strongest candidate in each window.  A serve produces
    # one trophy run, but a double fault produces two serves 15-25 s apart and
    # both are real point starts, so this only suppresses the immediate
    # re-detection of a single action, never a second serve.
    gap = REFRACT_S if refract_s is None else float(refract_s)
    out.sort(key=lambda e: (-e["p"], e["t"]))
    kept: List[Dict] = []
    for e in out:
        if all(abs(e["t"] - k["t"]) >= gap for k in kept):
            kept.append(e)
    kept.sort(key=lambda e: e["t"])
    return kept




# =========================================================================
# anya2 plumbing
# =========================================================================
# Everything above this line is the validated seed, changed only to accept the
# substrate's eligibility mask and to carry a slot id.  Everything below is the
# redesign: it replaces the seed's single-player `signals_for_video` with a
# per-slot pass over `anya2.tracks`.

from pipeline.anya2 import court as C          # noqa: E402
from pipeline.anya2 import tracks as T         # noqa: E402
from pipeline.anya2.contract import (          # noqa: E402
    NEAR_SERVE, ROI_NEAR, W_BETWEEN, Event, Requirement, dump_events)

EVENTS_SUFFIX = "_anya2_near_serve.json"

# What this detector needs from perception.  Pose-only is not an accident: the
# seed establishes that the serve MOTION is what separates a server from a
# player standing still, and the motion is entirely in the joints.  Carrying
# that into Phase E means the near path never has to run a ball model.
REQUIREMENT = Requirement(roi=ROI_NEAR, pose_fps=15.0, needs_ball=False,
                          windows=W_BETWEEN)


def events_path(video, suffix=EVENTS_SUFFIX):
    d = os.path.dirname(os.path.abspath(video))
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(d, f"{stem}{suffix}")


def detect_video(video, tracks_npz=None, threshold: float = THRESHOLD,
                 require_court: bool = True, verbose: bool = True,
                 lead_s: Optional[float] = None,
                 refract_s: Optional[float] = None):
    """Score every near slot and return the serves, as contract `Event`s.

    DOUBLES IS WHY THIS IS A LOOP.  Each near slot is scored independently and
    on its own merits; the server is whichever slot produced the candidate, and
    two slots are free to produce different serves at different times (they
    alternate service games).  Singles is the case where slot 1 is always NaN
    and the loop runs once for real -- there is no separate code path, and so no
    second path to keep correct.

    What the loop must NOT do is average or fuse the slots.  A serve is one
    player's action; blending the partner's quiet pose into it would dilute
    exactly the trophy shape the detector exists to find.
    """
    z = T.load(video, tracks_npz)
    fps = float(z["fps"])
    kp, bbox, ct, el = z["kp"], z["bbox"], z["court"], z["eligible"]

    raw: List[Dict] = []
    for slot in T.NEAR_SLOTS:
        seen = np.isfinite(bbox[:, slot, 0])
        if not seen.any():
            if verbose:
                print(f"[near-serve] slot {slot}: never tracked, skipped")
            continue
        prim = serve_primitives(kp[:, slot], bbox[:, slot], fps,
                                court_y=ct[:, slot, 1],
                                eligible=el[:, slot])
        ev = detect_serves(prim, threshold=threshold,
                           require_court=require_court, track=int(slot),
                           lead_s=lead_s, refract_s=refract_s)
        if verbose:
            print(f"[near-serve] slot {slot}: tracked {100 * seen.mean():5.1f}%"
                  f"  eligible {100 * el[:, slot][seen].mean():5.1f}%"
                  f"  -> {len(ev)} candidates")
        raw.extend(ev)

    # Cross-slot refractory.  Within a slot `detect_serves` has already applied
    # REFRACT_S; across slots it has not, and it must be applied here rather
    # than left out: the two near players stand close together at the baseline
    # and a partner's raised arm can produce a weak candidate alongside the
    # server's real one.  Keeping the STRONGER is the same arbitration used
    # within a slot, and it is what decides WHO SERVED in doubles.
    gap = REFRACT_S if refract_s is None else float(refract_s)
    raw.sort(key=lambda e: (-e["p"], e["t"]))
    kept: List[Dict] = []
    for e in raw:
        if all(abs(e["t"] - k["t"]) >= gap for k in kept):
            kept.append(e)
    kept.sort(key=lambda e: e["t"])

    return [Event(t=float(e["t"]), p=float(e["p"]), kind=NEAR_SERVE,
                  track=e["track"],
                  detail={k: e[k] for k in
                          ("trophy", "swing", "ready", "t_trophy",
                           "t_contact", "toss_arm", "t_basis")})
            for e in kept]


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("video")
    ap.add_argument("--tracks", default=None,
                    help="anya2 tracks npz (default: <stem>_anya2_tracks.npz)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-court", action="store_true",
                    help="Disable the baseline gate (expect smashes)")
    ap.add_argument("--json", default=None,
                    help=f"Write events here (default: <stem>{EVENTS_SUFFIX})")
    a = ap.parse_args()

    ev = detect_video(a.video, a.tracks, threshold=a.threshold,
                      require_court=not a.no_court)
    print(f"[near-serve] {len(ev)} serves at p >= {a.threshold}")
    for e in ev:
        d = e.detail
        print(f"  {e.t:8.2f}s  p={e.p:.3f}  slot={e.track}  "
              f"trophy={d['trophy']:.2f} swing={d['swing']:.2f} "
              f"ready={d['ready']:.2f}  toss={d['toss_arm']:5s}")
    out = a.json or events_path(a.video)
    dump_events(ev, out, fps=None, threshold=a.threshold,
                requirement=REQUIREMENT.__dict__)
    print(f"[near-serve] wrote {out}")


if __name__ == "__main__":
    main()
