"""
near_end.py
===========
Four "the point is over" signals for the NEAR player, from pose alone.  Each is
per pose sample, in [0, 1]:

    settle        feet and limbs go quiet and STAY quiet
    turn_away     the player turns round to face the camera (they play with
                  their back to it)
    stance_drop   hands drop below the hips AND the knees straighten
    idle_hands    a hand goes to the pocket (reaching for the second ball)

Each one also fires sometimes during live play, so none is a rule on its own.
Consumers combine them:
  * anya2's point_end.py takes the max of these and the walking probability.
  * rally_reel's energy bar weights them against the ball trace.

Model-free and measured in body heights / court metres, so it can be ported
to the phone.

Input: one near-player pose track, the same one the walking classifier reads
(`walking.select_near` output, or anya2's shim npz).  The pose helpers are
the walking classifier's own (see anya2/signals.py, "two families").

Run standalone to inspect the signals on a clip:
    python -m pipeline.near_end /Volumes/Anya/Data/21/snippet.mp4
    python -m pipeline.near_end /Volumes/Anya/Data/21/snippet.mp4 --profile

Evidence and history: docs/anya2/NOTES.md, "near_end.py".
"""

import argparse
import json
import os
from typing import Dict, Optional, Sequence

import numpy as np

# What consumers read.  Defined first and cheaply: rally_reel imports it at
# module scope.
SIGNAL_NAMES = ("settle", "turn_away", "stance_drop", "idle_hands")

# COCO keypoint indices.
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

MAX_GAP_S = 0.5            # fill occlusions up to this long; longer holes stay NaN

# ── settle ───────────────────────────────────────────────────────────────
SETTLE_SPEED_MPS = 0.55    # court speed scoring 0 (a walk is 1.0-1.5 m/s)
SETTLE_LIMB_BH_S = 0.75    # wrist/ankle speed relative to the hip, body heights/s,
                           # scoring 0
SETTLE_WIN_S = 1.2         # smoothing window

# ── turn_away ────────────────────────────────────────────────────────────
TURN_MIN_SHO_BH = 0.055    # shoulders closer than this (edge-on): no read
TURN_FULL_BH = 0.14        # shoulder separation for a fully confident read
TURN_WIN_S = 0.8

# ── stance_drop ──────────────────────────────────────────────────────────
READY_WRIST_BH = 0.06      # highest wrist this far ABOVE the hip line = ready
DROP_WRIST_BH = -0.06      # ...this far below = hands dropped
READY_KNEE_DEG = 152.0     # mean knee angle of an athletic crouch
STRAIGHT_KNEE_DEG = 172.0  # ...and of standing upright
STANCE_WIN_S = 0.8

# ── idle_hands (radii in body heights) ───────────────────────────────────
POCKET_R_BH = 0.13         # wrist this close to the same-side hip, and below it
HEAD_R_BH = 0.22           # wrist this close to the head (cue_to_head only)
HIP_R_BH = 0.11            # both wrists this close to the hips (cue_akimbo only)
FLARE_BH = 0.10            # ...with elbows this far out (cue_akimbo only)
IDLE_WIN_S = 0.6


def _ramp(x, lo, hi):
    """Linear 0 -> 1 from lo to hi (either order), clipped; NaN stays NaN."""
    with np.errstate(invalid="ignore"):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _distance_bh(a, b, body_height):
    """Distance between two [N, 2] tracks, in body heights."""
    return np.linalg.norm(a - b, axis=1) / body_height


def near_signals(kp, bbox, fps: float,
                 speed: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
    """The four signals for one near-player pose track.

    kp [N, 17, 3], bbox [N, 4]: the near player only, NaN where untracked.
    speed [N]: court speed in m/s from `walking.features.frame_signals` (the
    same number the walking classifier uses).  Without it, `settle` uses limb
    motion alone.

    Returns each SIGNAL_NAMES entry and the three `cue_*` sub-signals as [N]
    in [0, 1] (0 wherever the player is untracked), plus `valid` [N] bool and
    `fps`.
    """
    from pipeline.anya2.signals import walking_pose_helpers
    walk = walking_pose_helpers()

    gap = int(MAX_GAP_S * fps)
    kp = walk.fill_gaps_between_seen(np.asarray(kp, dtype=np.float64), gap)
    bbox = walk.fill_gaps_between_seen(np.asarray(bbox, dtype=np.float64), gap)

    body_height = bbox[:, 3] - bbox[:, 1]
    body_height = np.where(body_height > 1.0, body_height, np.nan)
    valid = np.isfinite(body_height)

    joints = _Joints(kp, walk)

    def smooth(x, window_s):
        """Centred NaN-aware moving mean (offline, so no lag)."""
        return walk.nan_moving_mean(x, max(1, int(round(window_s * fps))))

    pocket, to_head, akimbo = _idle_hand_cues(joints, body_height)
    signals = {
        "settle": smooth(_settle_raw(joints, body_height, fps, speed), SETTLE_WIN_S),
        "turn_away": smooth(_turn_away_raw(joints, body_height), TURN_WIN_S),
        "stance_drop": smooth(_stance_drop_raw(joints, body_height, walk), STANCE_WIN_S),
        "idle_hands": smooth(pocket, IDLE_WIN_S),
        # Measured and NOT part of idle_hands (to_head is anti-correlated,
        # akimbo is unreachable at 540p); returned for re-measurement only.
        "cue_pocket": pocket, "cue_to_head": to_head, "cue_akimbo": akimbo,
    }
    # Missing evidence is 0, and nothing is claimed where there is no player.
    signals = {name: np.clip(np.where(np.isfinite(x), x, 0.0), 0.0, 1.0)
               for name, x in signals.items()}
    for name in signals:
        signals[name] = np.where(valid, signals[name], 0.0)
    signals["valid"] = valid
    signals["fps"] = float(fps)
    return signals


class _Joints:
    """The keypoints this module reads, as [N, 2] image positions."""

    def __init__(self, kp, walk):
        xy = walk.keypoint_xy_walking
        mid = walk.midpoint_with_fallback
        self.l_sho, self.r_sho = xy(kp, L_SHO), xy(kp, R_SHO)
        self.l_elb, self.r_elb = xy(kp, L_ELB), xy(kp, R_ELB)
        self.l_wri, self.r_wri = xy(kp, L_WRI), xy(kp, R_WRI)
        self.l_hip, self.r_hip = xy(kp, L_HIP), xy(kp, R_HIP)
        self.l_kne, self.r_kne = xy(kp, L_KNE), xy(kp, R_KNE)
        self.l_ank, self.r_ank = xy(kp, L_ANK), xy(kp, R_ANK)
        nose = xy(kp, NOSE)
        l_ear, r_ear = xy(kp, L_EAR), xy(kp, R_EAR)
        self.hip = mid(self.l_hip, self.r_hip)
        self.sho = mid(self.l_sho, self.r_sho)
        self.head = mid(mid(l_ear, r_ear), nose)


def _settle_raw(j, body_height, fps, speed):
    """How quiet the player is right now (before smoothing).

    Mean of "feet quiet" (court speed) and "limbs quiet" (wrist/ankle speed
    relative to the hip).  Mean, not min: the two channels drop out
    independently at 15 Hz, and the mean keeps whichever is readable.
    """
    def limb_speed(point):
        relative = (point - j.hip) / body_height[:, None]
        speed_bh_s = np.full(len(point), np.nan)
        speed_bh_s[1:] = np.linalg.norm(np.diff(relative, axis=0), axis=1) * fps
        return speed_bh_s

    fastest_limb = np.fmax(np.fmax(limb_speed(j.l_wri), limb_speed(j.r_wri)),
                           np.fmax(limb_speed(j.l_ank), limb_speed(j.r_ank)))
    limbs_quiet = _ramp(fastest_limb, SETTLE_LIMB_BH_S, 0.0)
    if speed is None:
        return limbs_quiet
    feet_quiet = _ramp(np.asarray(speed, dtype=np.float64), SETTLE_SPEED_MPS, 0.0)
    return np.nanmean(np.stack([feet_quiet, limbs_quiet], axis=1), axis=1)


def _turn_away_raw(j, body_height):
    """How confidently the player faces the camera, from shoulder order alone.

    Back to the camera (playing): right shoulder is image-right of the left.
    Turned round: the order flips.  Edge-on shoulders give no read.  (Nose
    visibility was tried as extra evidence and only hurt -- see NOTES.)
    """
    shoulder_dx = (j.r_sho[:, 0] - j.l_sho[:, 0]) / body_height
    read_confidence = _ramp(np.abs(shoulder_dx), TURN_MIN_SHO_BH, TURN_FULL_BH)
    facing_camera = _ramp(-shoulder_dx, 0.0, TURN_FULL_BH)
    return np.clip(read_confidence * facing_camera, 0.0, 1.0)


def _stance_drop_raw(j, body_height, walk):
    """Hands dropped AND legs straight (min: straight legs alone mean nothing)."""
    # Image y grows downward, so hip_y - wrist_y > 0 means above the hip.
    highest_wrist_above_hip = (j.hip[:, 1] - np.fmin(j.l_wri[:, 1], j.r_wri[:, 1])) / body_height
    hands_low = _ramp(highest_wrist_above_hip, READY_WRIST_BH, DROP_WRIST_BH)
    knee_angle = np.nanmean(np.stack([walk.joint_angle_deg(j.l_hip, j.l_kne, j.l_ank),
                                      walk.joint_angle_deg(j.r_hip, j.r_kne, j.r_ank)],
                                     axis=1), axis=1)
    legs_straight = _ramp(knee_angle, READY_KNEE_DEG, STRAIGHT_KNEE_DEG)
    return np.fmin(hands_low, legs_straight)


def _idle_hand_cues(j, body_height):
    """(pocket, to_head, akimbo), each [N] before smoothing.  Only pocket is used."""
    l_hip_dist = _distance_bh(j.l_wri, j.l_hip, body_height)
    r_hip_dist = _distance_bh(j.r_wri, j.r_hip, body_height)
    l_below_hip = (j.l_wri[:, 1] - j.l_hip[:, 1]) / body_height   # > 0: wrist below hip
    r_below_hip = (j.r_wri[:, 1] - j.r_hip[:, 1]) / body_height

    # Hand to pocket: at the hip AND dropped below it.
    pocket = np.fmax(
        _ramp(l_hip_dist, POCKET_R_BH, 0.0) * _ramp(l_below_hip, 0.0, 0.05),
        _ramp(r_hip_dist, POCKET_R_BH, 0.0) * _ramp(r_below_hip, 0.0, 0.05))

    # Hand to face/cap: wrist at the head and above the shoulder line.
    l_head_dist = _distance_bh(j.l_wri, j.head, body_height)
    r_head_dist = _distance_bh(j.r_wri, j.head, body_height)
    l_up = _ramp((j.sho[:, 1] - j.l_wri[:, 1]) / body_height, 0.0, 0.06)
    r_up = _ramp((j.sho[:, 1] - j.r_wri[:, 1]) / body_height, 0.0, 0.06)
    to_head = np.fmax(_ramp(l_head_dist, HEAD_R_BH, 0.0) * l_up,
                      _ramp(r_head_dist, HEAD_R_BH, 0.0) * r_up)

    # Hands on hips with elbows flared out (either direction, since the
    # left/right call can flip).
    l_flare = _ramp((j.l_sho[:, 0] - j.l_elb[:, 0]) / body_height, 0.0, FLARE_BH)
    r_flare = _ramp((j.r_elb[:, 0] - j.r_sho[:, 0]) / body_height, 0.0, FLARE_BH)
    l_flare = np.fmax(l_flare, _ramp((j.l_elb[:, 0] - j.l_sho[:, 0]) / body_height, 0.0, FLARE_BH))
    r_flare = np.fmax(r_flare, _ramp((j.r_sho[:, 0] - j.r_elb[:, 0]) / body_height, 0.0, FLARE_BH))
    akimbo = (np.fmin(_ramp(l_hip_dist, HIP_R_BH, 0.0), _ramp(r_hip_dist, HIP_R_BH, 0.0))
              * np.fmin(l_flare, r_flare))
    return pocket, to_head, akimbo


def signals_for_video(video: str, pose_npz: Optional[str] = None,
                      sig: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """`near_signals` off a cached near-player pose npz for `video`.

    `sig` is `walking.features.frame_signals` output if the caller already has
    it; otherwise court speed is recomputed from the same npz.
    """
    from pipeline.anya2.signals import walking_pose_helpers
    walking_pose_helpers()          # puts the repo root on sys.path for `walking`
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
    """Mean of each signal in bands around every labelled rally end, over
    TRACKED samples only.  A useful signal rises after the end."""
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
