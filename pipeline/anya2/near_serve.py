"""
near_serve.py
=============
Did a point start with a NEAR-side serve?  Pose only, no ball.

The near player is filmed from behind and is large (72-275 px), so the full
serve motion is readable:

    READY    hands together on the racket, wrists low-ish, body quiet
    TROPHY   hands SPLIT, tossing wrist above the head, racket wrist up near
             the shoulder
    SWING    the racket wrist peaks above the head within ~0.9 s

Score per candidate (see serve_common.py for the shared skeleton):

    p = (0.45 * trophy + 0.20 * ready) / 0.65  *  (0.45 + 0.55 * swing)

and a serve is reported when p >= THRESHOLD (0.70).

Two gates decide who may be serving:
  * `eligible`  (tracks.py)  inside the doubles court + 3 ft laterally
  * serve zone  feet between 3.5 ft behind and 1 ft inside the near
                baseline -- this is what rejects mid-court smashes, which
                look identical to a serve in pose

Handedness is never assumed: everything is "the higher wrist" / "the lower
wrist".

Reported time
-------------
`t` is the POINT START: the trophy peak minus SERVE_LEAD_S (1.75 s).  The lead
is a fitted labelling convention, not physics (see NOTES, and the 2026-09-05
note "Anchor the near serve on the trophy peak").

Pose helpers: the walking classifier's (signals.walking_pose_helpers).
Evidence and history: docs/anya2/NOTES.md, "near_serve.py".
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import workdir
from pipeline.anya2 import serve_common
from pipeline.anya2 import signals
from pipeline.anya2 import tracks
from pipeline.anya2.contract import (NEAR_SERVE, ROI_NEAR, W_BETWEEN, Event,
                                     Requirement, dump_events)

EVENTS_SUFFIX = "_anya2_near_serve.json"

# COCO keypoint indices used here.
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHO, R_SHO = 5, 6
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12

MAX_GAP_S = 0.5            # fill pose occlusions up to this long

# ── head reference ───────────────────────────────────────────────────────
# The near player's face is turned away, so head keypoints are often missing.
# Then the head is assumed this far above the shoulder line (never NaN).
HEAD_ABOVE_SHO_BH = 0.14

# ── ready ────────────────────────────────────────────────────────────────
TOGETHER_BH = 0.10         # wrists this close = hands together on the grip
SPLIT_BH = 0.26            # ...this far apart = definitely split
READY_HI_MAX_BH = 0.10     # higher wrist no more than this above the shoulders
READY_STILL_BH_S = 0.9     # limb speed (relative to hip) counted as quiet
READY_WIN_S = 0.7          # smoothing window for the ready score

# ── trophy: all three must hold (product), each dilated by TROPHY_DILATE_S ──
TROPHY_ABOVE_HEAD_BH = 0.058   # higher wrist this far over the head = full credit
TROPHY_HEAD_MIN_BH = -0.02     # ...at/below head height = no credit
TROPHY_LO_MIN_BH = -0.13       # lower wrist this far below the shoulders = no credit
TROPHY_LO_FULL_BH = -0.052     # ...near the shoulders = full credit
TROPHY_SPLIT_MIN_BH = 0.149    # hands must be at least this far apart to score
TROPHY_SPLIT_RANGE_BH = 0.12   # ...and get full credit this much further apart
TROPHY_DILATE_S = 0.20         # +/- window each term is max-dilated over
TROPHY_MIN = 0.35              # trophy score that makes a sample a candidate

# ── swing ────────────────────────────────────────────────────────────────
SWING_MIN_S = 0.06         # contact search starts this long after the trophy
SWING_MAX_S = 0.90         # ...and ends this long after
PEAK_ABOVE_HEAD_BH = 0.06  # racket wrist this far over the head = full swing

# ── ready lookback window, before the trophy peak ────────────────────────
READY_BACK_MIN_S = 0.20
READY_BACK_MAX_S = 6.00

# ── serve zone (feet position, court y; near baseline is y = 0) ──────────
FT_TO_M = 0.3048
SERVE_ZONE_MIN_FT = -3.5   # up to 3.5 ft behind the baseline
SERVE_ZONE_MAX_FT = 1.0    # ...and up to 1 ft inside the court

# ── combination ──────────────────────────────────────────────────────────
W_TROPHY, W_READY = 0.45, 0.20
SWING_FLOOR = 0.45         # score multiplier kept with NO swing evidence
THRESHOLD = 0.70           # report serves with p >= this
REFRACT_S = 3.0            # at most one serve per this many seconds

# ── point-start time ─────────────────────────────────────────────────────
SERVE_LEAD_S = 1.75        # point start = trophy peak minus this.  A labelling
                           # convention, fitted (1.40-1.80 score identically).

# What this detector needs from perception (informational).
REQUIREMENT = Requirement(roi=ROI_NEAR, pose_fps=15.0, needs_ball=False,
                          windows=W_BETWEEN)


def events_path(video, suffix=EVENTS_SUFFIX):
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(workdir.artifact_dir(video), f"{stem}{suffix}")


# ═════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════

def detect_video(video, tracks_npz=None, threshold: float = THRESHOLD,
                 require_court: bool = True, verbose: bool = True,
                 lead_s: Optional[float] = None,
                 refract_s: Optional[float] = None):
    """Near serves for a video, as contract Events.

    Each near slot is scored on its own (never averaged: a serve is one
    player's action).  Then one serve is kept per refractory window across
    both slots -- the stronger one, which in doubles decides who served.
    """
    track_data = tracks.load(video, tracks_npz)
    fps = float(track_data["fps"])
    keypoints, boxes = track_data["kp"], track_data["bbox"]
    court_xy, eligible = track_data["court"], track_data["eligible"]

    candidates: List[Dict] = []
    for slot in tracks.NEAR_SLOTS:
        seen = np.isfinite(boxes[:, slot, 0])
        if not seen.any():
            if verbose:
                print(f"[near-serve] slot {slot}: never tracked, skipped")
            continue
        features = serve_features(keypoints[:, slot], boxes[:, slot], fps,
                                  court_y=court_xy[:, slot, 1],
                                  eligible=eligible[:, slot])
        serves = detect_serves(features, threshold=threshold,
                               require_court=require_court, track=int(slot),
                               lead_s=lead_s, refract_s=refract_s)
        if verbose:
            print(f"[near-serve] slot {slot}: tracked {100 * seen.mean():5.1f}%"
                  f"  eligible {100 * eligible[:, slot][seen].mean():5.1f}%"
                  f"  -> {len(serves)} candidates")
        candidates.extend(serves)

    kept = signals.keep_one_per_window(candidates, _refract(refract_s))
    return [Event(t=float(e["t"]), p=float(e["p"]), kind=NEAR_SERVE,
                  track=e["track"],
                  detail={key: e[key] for key in
                          ("trophy", "swing", "ready", "t_trophy",
                           "t_contact", "toss_arm", "t_basis")})
            for e in kept]


def _refract(refract_s):
    return REFRACT_S if refract_s is None else float(refract_s)


# ═════════════════════════════════════════════════════════════════════════
# Per-sample features for one player
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class NearServeFeatures:
    """Per pose sample, for one near player.  Elevations in body heights, + = higher."""
    fps: float
    valid: np.ndarray       # bool: box usable and both wrists seen
    ready: np.ndarray       # 0..1 ready-stance score (smoothed)
    trophy: np.ndarray      # 0..1 trophy-shape score
    elev_l: np.ndarray      # left wrist above the shoulder line
    elev_r: np.ndarray
    head_l: np.ndarray      # left wrist above the head
    head_r: np.ndarray
    hi_elev: np.ndarray     # higher / lower wrist above the shoulder line
    lo_elev: np.ndarray
    hi_head: np.ndarray     # higher wrist above the head
    hi_side: np.ndarray     # +1 left wrist higher, -1 right, 0 unknown
    gap_bh: np.ndarray      # distance between the wrists
    still: np.ndarray       # 0..1 limbs quiet
    on_court: np.ndarray    # bool: in the serve zone AND eligible


def serve_features(kp, bbox, fps: float,
                   court_y: Optional[Sequence[float]] = None,
                   eligible: Optional[Sequence[bool]] = None) -> NearServeFeatures:
    """Compute NearServeFeatures for one near slot of tracks.npz.

    kp [N,17,3], bbox [N,4]; court_y [N] (feet position, metres) enables the
    serve-zone gate; eligible [N] is ANDed into it.
    """
    walk = signals.walking_pose_helpers()
    kp = np.asarray(kp, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64)
    n = len(kp)
    gap = int(MAX_GAP_S * fps)
    kp = walk.fill_gaps_between_seen(kp, gap)
    bbox = walk.fill_gaps_between_seen(bbox, gap)

    body_height = bbox[:, 3] - bbox[:, 1]
    body_height = np.where(body_height > 1.0, body_height, np.nan)
    box_ok = np.isfinite(body_height)

    xy = walk.keypoint_xy_walking
    mid = walk.midpoint_with_fallback
    l_sho, r_sho = xy(kp, L_SHO), xy(kp, R_SHO)
    l_wri, r_wri = xy(kp, L_WRI), xy(kp, R_WRI)
    l_hip, r_hip = xy(kp, L_HIP), xy(kp, R_HIP)
    nose = xy(kp, NOSE)
    l_ear, r_ear = xy(kp, L_EAR), xy(kp, R_EAR)
    shoulders = mid(l_sho, r_sho)
    hips = mid(l_hip, r_hip)
    head = mid(mid(l_ear, r_ear), nose)

    # Head height: measured when seen, else a fixed offset above the shoulders.
    head_y = np.where(np.isfinite(head[:, 1]),
                      head[:, 1], shoulders[:, 1] - HEAD_ABOVE_SHO_BH * body_height)

    elev_l = (shoulders[:, 1] - l_wri[:, 1]) / body_height
    elev_r = (shoulders[:, 1] - r_wri[:, 1]) / body_height
    head_l = (head_y - l_wri[:, 1]) / body_height
    head_r = (head_y - r_wri[:, 1]) / body_height

    # Two-armed shapes need BOTH wrists seen (fmax would silently use one).
    both_wrists = np.isfinite(elev_l) & np.isfinite(elev_r)
    with np.errstate(invalid="ignore"):
        hi_elev = np.where(both_wrists, np.fmax(elev_l, elev_r), np.nan)
        lo_elev = np.where(both_wrists, np.fmin(elev_l, elev_r), np.nan)
        hi_head = np.where(both_wrists, np.fmax(head_l, head_r), np.nan)
    hi_side = np.where(both_wrists, np.where(elev_l >= elev_r, 1.0, -1.0), 0.0)
    gap_bh = np.linalg.norm(l_wri - r_wri, axis=1) / body_height

    # ── ready: hands together * wrists carried low * limbs quiet ──
    fastest_wrist = np.fmax(
        signals.limb_speed_relative_to_hip(l_wri, hips, body_height, fps),
        signals.limb_speed_relative_to_hip(r_wri, hips, body_height, fps))
    still = 1.0 - signals.ramp(fastest_wrist, READY_STILL_BH_S * 0.5, READY_STILL_BH_S * 1.6)
    together = 1.0 - signals.ramp(gap_bh, TOGETHER_BH, SPLIT_BH)
    carried_low = 1.0 - signals.ramp(hi_elev, READY_HI_MAX_BH, READY_HI_MAX_BH + 0.14)
    ready = walk.nan_moving_mean(together * carried_low * still,
                                 max(1, int(round(READY_WIN_S * fps))))

    # ── trophy: (higher wrist over head) * (lower wrist near shoulders) * (hands apart) ──
    dilate = max(1, int(round(TROPHY_DILATE_S * fps)) * 2 + 1)
    trophy = (signals.running_max(signals.ramp(hi_head, TROPHY_HEAD_MIN_BH, TROPHY_ABOVE_HEAD_BH), dilate)
              * signals.running_max(signals.ramp(lo_elev, TROPHY_LO_MIN_BH, TROPHY_LO_FULL_BH), dilate)
              * signals.running_max(signals.ramp(gap_bh, TROPHY_SPLIT_MIN_BH,
                                                 TROPHY_SPLIT_MIN_BH + TROPHY_SPLIT_RANGE_BH), dilate))

    # ── who may serve: feet in the serve zone (NaN fails), AND eligible ──
    if court_y is not None:
        feet_y = np.asarray(court_y, dtype=np.float64)
        on_court = ((feet_y >= SERVE_ZONE_MIN_FT * FT_TO_M)
                    & (feet_y <= SERVE_ZONE_MAX_FT * FT_TO_M))
    else:
        on_court = np.ones(n, dtype=bool)
    if eligible is not None:
        on_court = on_court & np.asarray(eligible, dtype=bool)

    return NearServeFeatures(
        fps=float(fps), valid=box_ok & both_wrists, ready=ready, trophy=trophy,
        elev_l=elev_l, elev_r=elev_r, head_l=head_l, head_r=head_r,
        hi_elev=hi_elev, lo_elev=lo_elev, hi_head=hi_head, hi_side=hi_side,
        gap_bh=gap_bh, still=still, on_court=on_court)


# ═════════════════════════════════════════════════════════════════════════
# Candidates -> serves
# ═════════════════════════════════════════════════════════════════════════

def detect_serves(features: NearServeFeatures,
                  threshold: float = THRESHOLD,
                  require_court: bool = True,
                  track: Optional[int] = None,
                  lead_s: Optional[float] = None,
                  refract_s: Optional[float] = None) -> List[Dict]:
    """Score every trophy candidate; return serves with p >= threshold.

    Each returned dict: t (point start), t_basis, p, trophy, swing, ready,
    t_trophy, t_contact, toss_arm, track.
    """
    fps = features.fps
    ready_from = serve_common.samples(READY_BACK_MAX_S, fps)
    ready_to = serve_common.samples(READY_BACK_MIN_S, fps)

    serves: List[Dict] = []
    for cand in serve_common.trophy_candidates(features.trophy, features.on_court,
                                               TROPHY_MIN, require_court):
        # Ready: the best smoothed ready score in the lookback window.
        lookback = features.ready[max(0, cand.peak - ready_from):max(0, cand.peak - ready_to)]
        lookback = lookback[np.isfinite(lookback)]
        ready = float(lookback.max()) if lookback.size else 0.0

        swing = serve_common.swing(features, cand.peak, fps,
                                   SWING_MIN_S, SWING_MAX_S, PEAK_ABOVE_HEAD_BH)
        p = serve_common.combine(cand.trophy, ready, swing.score,
                                 W_TROPHY, W_READY, SWING_FLOOR)
        if p < threshold:
            continue
        t_start, basis = serve_onset(features, cand.peak, lead_s=lead_s)
        serves.append({
            "t": t_start,
            "t_basis": basis,
            "p": round(p, 4),
            "trophy": round(cand.trophy, 4),
            "swing": round(swing.score, 4),
            "ready": round(ready, 4),
            "t_trophy": round(cand.peak / fps, 3),
            "t_contact": round(swing.contact_s, 3) if swing.contact_s is not None else None,
            "toss_arm": "left" if swing.toss_is_left else "right",
            "track": track,
        })
    # Within this slot: one serve per refractory window, the strongest.  (A
    # double fault's two serves are 15-25 s apart, so both survive.)
    return signals.keep_one_per_window(serves, _refract(refract_s))


def serve_onset(features: NearServeFeatures, trophy_peak: int,
                lead_s: Optional[float] = None) -> tuple:
    """(point-start seconds, "trophy"): the trophy peak minus the lead.

    The trophy peak, not the hand split: the split's distance before the
    trophy varies 0.07-4.07 s (servers pause after splitting), while the
    peak is where the shape is maximal.  `t_basis` is always "trophy" but
    stays in the event so a future per-clip lead can label itself.
    """
    lead = SERVE_LEAD_S if lead_s is None else float(lead_s)
    return trophy_peak / features.fps - lead, "trophy"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="A near-side serve detector built "
                                 "from POSE ALONE — no ball evidence anywhere.")
    ap.add_argument("video")
    ap.add_argument("--tracks", default=None,
                    help="anya2 tracks npz (default: <stem>_anya2_tracks.npz)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-court", action="store_true",
                    help="Disable the baseline gate (expect smashes)")
    ap.add_argument("--json", default=None,
                    help=f"Write events here (default: <stem>{EVENTS_SUFFIX})")
    a = ap.parse_args()

    serves = detect_video(a.video, a.tracks, threshold=a.threshold,
                          require_court=not a.no_court)
    print(f"[near-serve] {len(serves)} serves at p >= {a.threshold}")
    for e in serves:
        d = e.detail
        print(f"  {e.t:8.2f}s  p={e.p:.3f}  slot={e.track}  "
              f"trophy={d['trophy']:.2f} swing={d['swing']:.2f} "
              f"ready={d['ready']:.2f}  toss={d['toss_arm']:5s}")
    out = a.json or events_path(a.video)
    dump_events(serves, out, fps=None, threshold=a.threshold,
                requirement=REQUIREMENT.__dict__)
    print(f"[near-serve] wrote {out}")


if __name__ == "__main__":
    main()
