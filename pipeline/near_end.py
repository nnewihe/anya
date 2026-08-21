"""
near_end.py
===========
Four near-player point-end signals, alongside the walking classifier that
already ships.

The walking classifier answers one question — "is this person going somewhere"
— and it is the only near-player evidence the energy policy reads.  It misses
the whole family of end-of-point behaviour that happens WITHOUT translation: the
player who wins the point standing still, turns to the back fence, drops the
racket hand, reaches into a pocket for the second ball, or stands with hands on
hips getting their breath back.  None of that moves the feet, so `walk_prob`
stays at zero through all of it.

This module computes four such signals, each per pose sample, each in [0, 1]:

    settle        court speed and limb energy both collapse and STAY collapsed.
                  Rally motion is bursty and never quiet for long; the settle
                  after the last ball is.

    turn_away     the shoulder line rotates so the player faces the CAMERA
                  rather than the net.  The near player plays with their back to
                  this camera, so turning toward it is turning away from the
                  point.

    stance_drop   loss of the ready stance: the wrists fall to or below the hip
                  line, the two-handed grip opens, and the knees straighten out
                  of the athletic crouch.  A player who has stopped expecting a
                  ball stands up.

    idle_hands    the between-point rituals, as alternatives under a max():
                  hand to the pocket for a second ball, hand to face / cap /
                  hair, or both hands on the hips with the elbows flared.

The design rule behind all four is the one the user's framing states outright:
INDIVIDUALLY NOISY, COLLECTIVELY NEAR-CERTAIN.  Each of these fires during live
tennis too — a player is briefly still between shots, turns their back chasing a
lob, straightens up after a short ball, wipes their face during a long point.
So none of them is a rule here.  Each is a graded score, and the arbitration
between them is `rally_reel/energy.py`'s: they enter the bar as weighted terms
on the ball-silence drain, which means that WITH THE BALL VISIBLY IN PLAY THEY
DRAIN NOTHING, however confident they are.  That is the same construction (and
the same reason) as `energy_walk_boost` — see that config comment for the
mid-rally-walking failure an additive term reintroduces.

Two things the user's list asked for are deliberately NOT here:

  * BALL BOUNCING (the player bouncing a ball before serving) needs ball
    evidence, not pose, and the energy bar already reads a ball stream —
    `ball_trace` in-court intervals.  A bounce is in-court ball motion, so it
    would read as rally activity and CHARGE the bar.  Separating a serve bounce
    from a rally requires the ball's position relative to the near player, which
    is a change to `ball_trace`, not a fifth pose signal.

  * SERVE-PREP RITUAL AS RETROACTIVE CONFIRMATION ("the next serve proves the
    last point ended") is already structural in the pipeline and does not want
    to be a signal: `points.find_point_end` carries `next_serve_guard_s`, and
    the serve detectors are what produce the next start.  Adding a backwards
    confirmation here would be a second, weaker copy of that guard.

Everything is MODEL-FREE and measured in body-height units or court metres —
no classifier, no weights fit off this corpus.  That is deliberate: stage 2 of
the dead-time cutter has to port to Dart on-device, and a trained sklearn model
would not.  The thresholds below are anatomical constants, not tuned
parameters; the tuning happens once, downstream, on the four weights in
ReelConfig.

Input is the near-player pose the walking classifier already selected
(`walking.select_near`, cached as `<stem>_end_walk_pose.npz`), so this costs one
pass over an array and no perception at all.

Run standalone to inspect the signals on a clip:
    python -m pipeline.near_end /Volumes/Anya/Data/21/snippet.mp4
    python -m pipeline.near_end /Volumes/Anya/Data/21/snippet.mp4 --profile
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

SIGNAL_NAMES = ("settle", "turn_away", "stance_drop", "idle_hands")

# COCO indices, mirroring walking/features.py.  The geometry helpers are shared
# with that module (see `_kin`) but the indices are named here so that importing
# SIGNAL_NAMES — which `rally_reel/energy.py` does, at module scope — costs
# nothing.  The shared helpers arrive lazily instead: `walking` is a sibling
# top-level package that is not on the path until the repo root is, and paying
# for that at import time would put a second package's import graph behind
# every `from .energy import ...` in the reel.
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

MAX_GAP_S = 0.5        # matches walking.features: fill single-frame occlusions
                       # only.  A longer hole stays NaN and surfaces as `valid`
                       # False, which the energy bar HOLDS on rather than
                       # draining — absence of a player is not evidence that the
                       # point ended.

# ── settle ───────────────────────────────────────────────────────────────
# A rally is bursty: even the quiet beat while the opponent swings carries a
# split-step.  These are the levels below which the near player is doing
# nothing, not merely between things.
SETTLE_SPEED_MPS = 0.55     # court speed that scores 0; a walk sits at 1.0-1.5
                            # and walking.features' own "slow" band is < 0.4
SETTLE_LIMB_BH_S = 0.75     # wrist/ankle speed RELATIVE TO THE HIP, in body
                            # heights per second.  Relative, so a player being
                            # carried across court by a run does not read as
                            # limb activity, and a racket take-back while
                            # standing still does.
SETTLE_WIN_S = 1.2          # the sustain window.  Shorter than this and every
                            # between-shot pause scores; much longer and the
                            # signal arrives after the reel has already cut.

# ── turn_away ────────────────────────────────────────────────────────────
# The camera sits behind the near baseline, so the near player faces AWAY from
# it while the point is live.  Facing is read from the image-x order of the two
# shoulders: with the back to the camera the anatomical right shoulder falls to
# the image right of the left one, and the sign flips when they turn round.
TURN_MIN_SHO_BH = 0.055     # shoulder separation below this is a player seen
                            # edge-on; the facing sign is then pure noise and
                            # the sample contributes nothing
TURN_FULL_BH = 0.14         # separation at which the facing read is at full
                            # confidence
TURN_WIN_S = 0.8            # a rally turn (chasing a lob) is a fast pivot back;
                            # an end-of-point turn is held

# ── stance_drop ──────────────────────────────────────────────────────────
# Ready: wrists carried at or above the hip line, hands together on the racket,
# knees bent.  Dropped: hands by the sides, legs straight.
READY_WRIST_BH = 0.06       # highest wrist ABOVE the hip line, body heights.
                            # Above this is a ready carry, below DROP is a hand
                            # hanging by the side.
DROP_WRIST_BH = -0.06
READY_KNEE_DEG = 152.0      # mean knee interior angle: an athletic ready
                            # position, against ~172 for standing upright
STRAIGHT_KNEE_DEG = 172.0
STANCE_WIN_S = 0.8

# ── idle_hands ───────────────────────────────────────────────────────────
# Radii are body-height fractions of the distance from the wrist to the target
# landmark, so they hold at any depth in frame.
POCKET_R_BH = 0.13          # wrist within this of the same-side hip, and below
                            # the hip line: reaching for the second ball
HEAD_R_BH = 0.22            # wrist within this of the head centre: cap, hair,
                            # wiping the face
HIP_R_BH = 0.11             # both wrists this close to their hips …
FLARE_BH = 0.10             # … with both elbows this far LATERAL of the
                            # shoulders is the hands-on-hips akimbo shape, which
                            # the flare is what distinguishes from arms hanging
IDLE_WIN_S = 0.6            # rituals are held for a beat; a single frame of a
                            # wrist passing a hip is not one


def _kin():
    """The pose helpers `walking.features` already defines, imported lazily.

    Shared rather than re-derived on purpose: gap interpolation, the
    confidence-gated keypoint read, the midpoint fallback and the NaN-tolerant
    moving mean all have to behave here EXACTLY as they do under the walking
    classifier, or the two near-player signals disagree about what a tracked
    frame even is.  A second copy would drift on the first bug fix.
    """
    _REPO_ROOT = str(Path(__file__).resolve().parents[1])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from walking.features import (_angle, _interp_gaps, _kp_xy, _mid, _movmean)
    return _angle, _interp_gaps, _kp_xy, _mid, _movmean


def _ramp(x, lo, hi):
    """Soft 0->1 ramp from lo to hi (either order); NaN in, NaN out."""
    with np.errstate(invalid="ignore"):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _dist_bh(a, b, h):
    """Distance between two [N,2] keypoint tracks, in body heights."""
    return np.linalg.norm(a - b, axis=1) / h


def _smooth(x, fps, win_s, movmean):
    """Centred moving mean over `win_s`, NaN-tolerant.

    Centred rather than causal because this is an offline cutter: the reel is
    built after the clip exists, and a causal filter would lag every signal by
    half a window into the dead time it is trying to time.  A future on-device
    port that must run live gets a causal variant here and re-measures — the
    lag is a real change to the timing, not an implementation detail.
    """
    return movmean(x, max(1, int(round(win_s * fps))))


def _nan_to_zero(x):
    """Score arrays leave here as evidence, and missing evidence is 0."""
    return np.where(np.isfinite(x), x, 0.0)


def near_signals(kp, bbox, fps: float,
                 speed: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
    """The four signals for one near-player pose track.

    `kp` [N, 17, 3] and `bbox` [N, 4] are `walking.select_near`'s output — the
    near player only, NaN where they were not tracked.  `speed` is the court
    speed in m/s that `walking.features.frame_signals` already computed off the
    same track; passing it keeps this module from re-deriving a homography and
    guarantees the settle signal and the walking classifier disagree about
    nothing.  Without it, settle falls back to limb energy alone.

    Returns each signal as [N] in [0, 1], plus `valid` [N] bool.
    """
    _angle, _interp_gaps, _kp_xy, _mid, _movmean = _kin()
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
    l_elb, r_elb = _kp_xy(kp, L_ELB), _kp_xy(kp, R_ELB)
    l_wri, r_wri = _kp_xy(kp, L_WRI), _kp_xy(kp, R_WRI)
    l_hip, r_hip = _kp_xy(kp, L_HIP), _kp_xy(kp, R_HIP)
    l_kne, r_kne = _kp_xy(kp, L_KNE), _kp_xy(kp, R_KNE)
    l_ank, r_ank = _kp_xy(kp, L_ANK), _kp_xy(kp, R_ANK)
    nose = _kp_xy(kp, NOSE)
    l_ear, r_ear = _kp_xy(kp, L_EAR), _kp_xy(kp, R_EAR)

    hip = _mid(l_hip, r_hip)
    sho = _mid(l_sho, r_sho)
    head = _mid(_mid(l_ear, r_ear), nose)

    # ── settle ───────────────────────────────────────────────────────────
    def rel_speed(pt):
        rel = (pt - hip) / h[:, None]
        d = np.full(n, np.nan)
        d[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps
        return d

    limb = np.fmax(np.fmax(rel_speed(l_wri), rel_speed(r_wri)),
                   np.fmax(rel_speed(l_ank), rel_speed(r_ank)))
    quiet_limb = _ramp(limb, SETTLE_LIMB_BH_S, 0.0)
    if speed is not None:
        sp = np.asarray(speed, dtype=np.float64)
        quiet_foot = _ramp(sp, SETTLE_SPEED_MPS, 0.0)
        # MEAN, not min — measured, against the opposite prior.  The argument
        # for min() is that a settle ought to be BOTH feet and hands quiet, and
        # it scores worse on every clip: post/live separation over
        # Data/21,22,23,24,43 is
        #
        #     min   1.27  1.38  1.82  0.94  1.56     (negative on 24)
        #     mean  2.16  1.40  1.51  1.24  1.63     (positive on all five)
        #
        # because at 15 Hz off a 540p proxy the two channels fail
        # independently: a lost ankle keypoint zeroes the foot term while the
        # hands are perfectly still, and min() then throws away the half of the
        # evidence that survived.  Averaging degrades toward whichever channel
        # is readable, which is the behaviour this input actually needs.
        quiet = np.nanmean(np.stack([quiet_foot, quiet_limb], axis=1), axis=1)
    else:
        quiet = quiet_limb
    settle = _smooth(quiet, fps, SETTLE_WIN_S, _movmean)

    # ── turn_away ────────────────────────────────────────────────────────
    # Signed shoulder order.  Positive = anatomical right shoulder to the image
    # right = back to the camera = facing the net = playing.  Negative = turned
    # round.  Normalised by body height so it is a shape, not a pixel count.
    sho_dx = (r_sho[:, 0] - l_sho[:, 0]) / h
    sep = np.abs(sho_dx)
    # Confidence in the SIGN, which is the only part being read.  Edge-on, the
    # two shoulders project on top of each other and the order is a coin flip.
    conf = _ramp(sep, TURN_MIN_SHO_BH, TURN_FULL_BH)
    facing_cam = _ramp(-sho_dx, 0.0, TURN_FULL_BH)
    # SHOULDER ORDER ALONE.  Nose visibility looks like free corroboration — the
    # model finds a face far more readily from the front than from behind — and
    # it is the single most damaging term measured in this module.  Blending it
    # in at weight w costs separation monotonically over Data/21,22,23,24,43:
    #
    #     w = 0.00   11.83  4.62  1.48  3.49  88.67
    #     w = 0.25    4.56  2.61  1.49  2.57   4.73
    #     w = 0.50    3.11  2.00  1.50  2.17   2.79
    #
    # The nose is found on a large share of frames from BEHIND as well (the
    # model hallucinates a plausible face onto the back of a head at this
    # resolution), so the term is close to a constant, and a constant added to a
    # discriminative signal can only dilute it.  Left out rather than
    # down-weighted: there is no weight at which it pays.
    turn_raw = np.clip(conf * facing_cam, 0.0, 1.0)
    turn_away = _smooth(turn_raw, fps, TURN_WIN_S, _movmean)

    # ── stance_drop ──────────────────────────────────────────────────────
    # Image y grows downward, so hip_y - wrist_y is positive above the hip.
    wri_above_hip = (hip[:, 1] - np.fmin(l_wri[:, 1], r_wri[:, 1])) / h
    hands_low = _ramp(wri_above_hip, READY_WRIST_BH, DROP_WRIST_BH)
    knee = np.nanmean(np.stack([_angle(l_hip, l_kne, l_ank),
                                _angle(r_hip, r_kne, r_ank)], axis=1), axis=1)
    legs_straight = _ramp(knee, READY_KNEE_DEG, STRAIGHT_KNEE_DEG)
    # MIN, not mean — the opposite of the choice `settle` makes ten lines up,
    # and for the opposite reason.  Straight legs on their own are not evidence
    # of anything (a player stands upright constantly while the ball is at the
    # far end), and it shows: as a lone cue the knee term is negative on one
    # clip and flat on the rest.  Conjunction is what makes it mean something.
    # Over Data/21,22,23,24,43:
    #
    #     knees only  1.56  1.16  0.93  1.21  1.38
    #     hands only  1.96  1.34  2.42  6.44  2.36
    #     mean        1.65  1.20  1.27  1.69  1.59
    #     min         2.80  1.53  2.24  7.08  3.32
    #
    # min() beats hands-alone on four clips of five, so the knee term is real —
    # it just has to VETO rather than vote.  Averaging it in makes the pair
    # worse than the better half of it, which is the signature of a weak cue
    # being given independent authority it has not earned.
    stance_drop = _smooth(np.fmin(hands_low, legs_straight),
                          fps, STANCE_WIN_S, _movmean)

    # ── idle_hands ───────────────────────────────────────────────────────
    l_hip_d, r_hip_d = _dist_bh(l_wri, l_hip, h), _dist_bh(r_wri, r_hip, h)
    l_below = (l_wri[:, 1] - l_hip[:, 1]) / h      # positive = wrist below hip
    r_below = (r_wri[:, 1] - r_hip[:, 1]) / h

    # (a) hand to pocket: at the hip AND dropped to it, which is what separates
    # a reach into a pocket from a hand resting on the hip bone.
    pocket = np.fmax(
        _ramp(l_hip_d, POCKET_R_BH, 0.0) * _ramp(l_below, 0.0, 0.05),
        _ramp(r_hip_d, POCKET_R_BH, 0.0) * _ramp(r_below, 0.0, 0.05))

    # (b) hand to face / cap / hair: wrist at the head and raised above the
    # shoulder line, so a racket held up in front of the chest does not count.
    l_head_d, r_head_d = _dist_bh(l_wri, head, h), _dist_bh(r_wri, head, h)
    l_up = _ramp((sho[:, 1] - l_wri[:, 1]) / h, 0.0, 0.06)
    r_up = _ramp((sho[:, 1] - r_wri[:, 1]) / h, 0.0, 0.06)
    to_head = np.fmax(_ramp(l_head_d, HEAD_R_BH, 0.0) * l_up,
                      _ramp(r_head_d, HEAD_R_BH, 0.0) * r_up)

    # (c) hands on hips, elbows flared.  Both sides, or it is one hand on a hip
    # — which is (a) — and the flare is measured outward from each shoulder, so
    # arms hanging straight down score zero however close the wrists sit.
    l_flare = _ramp((l_sho[:, 0] - l_elb[:, 0]) / h, 0.0, FLARE_BH)
    r_flare = _ramp((r_elb[:, 0] - r_sho[:, 0]) / h, 0.0, FLARE_BH)
    # The shoulder order flips with the player, so take the flare either way
    # round rather than trusting the sign of a single frame's left/right call.
    l_flare = np.fmax(l_flare, _ramp((l_elb[:, 0] - l_sho[:, 0]) / h, 0.0, FLARE_BH))
    r_flare = np.fmax(r_flare, _ramp((r_sho[:, 0] - r_elb[:, 0]) / h, 0.0, FLARE_BH))
    akimbo = (np.fmin(_ramp(l_hip_d, HIP_R_BH, 0.0), _ramp(r_hip_d, HIP_R_BH, 0.0))
              * np.fmin(l_flare, r_flare))

    # POCKET ONLY.  The three rituals were built as alternatives under a max(),
    # on the reasoning that a player does one of them and summing would reward
    # only the rare player who does two at once.  The max() is right; the
    # membership was not.  Measured separately over Data/21,22,23,24,43
    # (post/live over tracked samples):
    #
    #     pocket    3.09  3.16  2.29  10.05   7.71     every clip, strongly
    #     akimbo    1.76  0.77  2.83   5.74   1.00     negative on 22, flat on 43
    #     to_head   0.26  0.00  0.07   0.00   0.00     ANTI-correlated everywhere
    #
    # to_head is not weak, it is backwards, and no radius fixes it (0.22 and
    # 0.32 body heights score the same): at 540p the wrist that comes near the
    # head is the racket arm at the top of a swing or a serve toss, which is
    # live play, and a genuine cap adjust between points is a handful of frames
    # per clip.  akimbo is closer to real but its flare term is unreachable at
    # this resolution — loosening FLARE_BH to 0.05 raises its level 4x and moves
    # separation the wrong way on two clips, which is a threshold buying noise.
    #
    # Both are still COMPUTED and returned under `cue_` keys.  They cost two
    # array ops, the measurement above is five clips of one camera height, and a
    # deleted cue cannot be re-measured on better footage.  They just do not
    # enter the signal the energy bar reads.
    idle_hands = _smooth(pocket, fps, IDLE_WIN_S, _movmean)

    out = {"settle": settle, "turn_away": turn_away,
           "stance_drop": stance_drop, "idle_hands": idle_hands,
           # Measured and excluded from idle_hands; kept for re-measurement.
           # `SIGNAL_NAMES` is what the energy bar reads, so nothing downstream
           # picks these up by accident.
           "cue_pocket": pocket, "cue_to_head": to_head, "cue_akimbo": akimbo}
    out = {k: np.clip(_nan_to_zero(v), 0.0, 1.0) for k, v in out.items()}
    # Zeroed wherever there is no player at all: a signal computed across a hole
    # is a guess, and `valid` is what lets the energy bar hold instead.
    for k in out:
        out[k] = np.where(valid, out[k], 0.0)
    out["valid"] = valid
    out["fps"] = float(fps)
    return out


def signals_for_video(video: str, pose_npz: Optional[str] = None,
                      sig: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """`near_signals` off the cached near-player pose for `video`.

    `sig` is `walking.features.frame_signals`' output when the caller already
    has it (the reel and the tuner both do, inside `walk_result`); otherwise the
    court speed is recomputed here from the same npz.
    """
    from walking.court import load_homography
    from walking.features import frame_signals
    from walking.select_near import pose_path

    pose_npz = pose_npz or pose_path(video)
    z = np.load(pose_npz)
    kp, bbox, fps = z["kp"], z["bbox"], float(z["fps"])
    if sig is None:
        sig = frame_signals(kp, bbox, load_homography(video), fps,
                            on_court=z.get("on_court"))
    return near_signals(kp, bbox, fps, speed=sig["speed"])


# ── CLI ──────────────────────────────────────────────────────────────────

def _profile(video: str, sigs: Dict[str, np.ndarray]) -> None:
    """Mean of each signal in bands around every labelled rally end.

    The question this answers is the only one that matters before any weight is
    fitted: does the signal RISE after the point ends, and by how much against
    its own level during play.  A signal that reads the same either side of the
    boundary cannot help the bar whatever weight it is given.

    Means are taken over TRACKED SAMPLES ONLY, with the coverage printed
    alongside.  Pooling the untracked ones — which the energy bar never sees,
    because `near_held` makes it hold rather than drain — would deflate every
    band by however much of it the player spent off camera, and the dead bands
    are exactly where they leave.  On Data/23 that is 58% of the clip, so the
    difference between the two readings is not a rounding detail.
    """
    from pipeline.parse_ground_truth import load_rallies

    fps = float(sigs["fps"])
    rallies = load_rallies(os.path.dirname(os.path.abspath(video)))
    if not rallies:
        print("[NEAR-END] no ground truth in this directory; skipping profile")
        return

    bands = ((-4.0, -1.0, "live"), (-1.0, 1.0, "boundary"),
             (1.0, 4.0, "post"), (4.0, 8.0, "dead"))
    valid = sigs["valid"]
    n = len(valid)

    def slices(lo, hi):
        for r in rallies:
            a = max(0, int(round((r["end_s"] + lo) * fps)))
            b = min(n, int(round((r["end_s"] + hi) * fps)))
            if b > a:
                yield a, b

    def band(name, lo, hi):
        vals = [sigs[name][a:b][valid[a:b]] for a, b in slices(lo, hi)]
        vals = [v for v in vals if v.size]
        return float(np.mean(np.concatenate(vals))) if vals else float("nan")

    def band_cov(lo, hi):
        vals = [valid[a:b] for a, b in slices(lo, hi)]
        return float(np.mean(np.concatenate(vals))) if vals else float("nan")

    print(f"[NEAR-END] {len(rallies)} labelled rally end(s), signal mean over "
          f"TRACKED samples by band relative to the end:")
    head = "  ".join(f"{lbl:>9}" for _, _, lbl in bands)
    print(f"  {'signal':>12}  {head}   post/live")
    for name in SIGNAL_NAMES:
        m = [band(name, lo, hi) for lo, hi, _ in bands]
        ratio = m[2] / m[0] if m[0] > 1e-6 else float("nan")
        row = "  ".join(f"{v:9.3f}" for v in m)
        print(f"  {name:>12}  {row}   {ratio:8.2f}x")
    cov = "  ".join(f"{band_cov(lo, hi):9.1%}" for lo, hi, _ in bands)
    print(f"  {'coverage':>12}  {cov}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Near-player point-end signals off the cached pose track")
    ap.add_argument("video")
    ap.add_argument("--pose-npz", default=None,
                    help="Near-player pose npz (default: <stem>_walk_pose.npz; "
                         "the reel's fast path writes <stem>_end_walk_pose.npz)")
    ap.add_argument("--out", default=None, help="Write per-second JSONL here")
    ap.add_argument("--profile", action="store_true",
                    help="Summarise each signal around labelled rally ends")
    a = ap.parse_args()

    sigs = signals_for_video(a.video, pose_npz=a.pose_npz)
    fps, n = float(sigs["fps"]), len(sigs["valid"])
    print(f"[NEAR-END] {n} pose samples @{fps:.2f} fps, "
          f"{float(np.mean(sigs['valid'])):.1%} with a tracked player")
    for name in SIGNAL_NAMES:
        v = sigs[name][sigs["valid"]]
        if not v.size:
            continue
        print(f"[NEAR-END]   {name:>12}: mean {v.mean():.3f}  "
              f"p90 {np.percentile(v, 90):.3f}  "
              f"frac>0.5 {float(np.mean(v > 0.5)):.1%}")

    if a.profile:
        _profile(a.video, sigs)

    if a.out:
        with open(a.out, "w") as fh:
            for s in range(int(np.ceil(n / fps))):
                lo, hi = int(round(s * fps)), min(n, int(round((s + 1) * fps)))
                if hi <= lo:
                    break
                rec = {"second": s,
                       "coverage": round(float(np.mean(sigs["valid"][lo:hi])), 3)}
                for name in SIGNAL_NAMES:
                    rec[name] = round(float(np.mean(sigs[name][lo:hi])), 4)
                fh.write(json.dumps(rec) + "\n")
        print(f"[NEAR-END] per-second signals → {a.out}")


if __name__ == "__main__":
    main()
