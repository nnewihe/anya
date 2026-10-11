"""
far_serve.py
============
Did a point start with a FAR-side serve?  Pose only, no ball.

Same READY -> TROPHY -> SWING idea as near_serve.py (shared skeleton in
serve_common.py), re-derived for a player ~30 px tall who FACES the camera:

  * TROPHY is elevation only: higher wrist over the head * lower wrist near
    the shoulders.  The hand split used on the near side is not measurable
    here (it is even inverted), so it is not read at all.
  * READY is the MEAN over the 2 s before the trophy (arms carried low and
    quiet), not the max over 6 s.
  * STILL: a server stands still before serving; a returner has just been
    running.  Median box travel 5 s .. 1 s before the trophy.
  * No serve-zone band: far court positions carry metres of error.  Only the
    lateral `eligible` gate (doubles court + 3 ft) applies.
  * Boxes with an implausible height or aspect are not used at all (`_sane`).

Score per candidate:

    p = (0.45 * trophy + 0.20 * ready) / 0.65 * (0.45 + 0.55 * swing)
        * (0.70 + 0.30 * still)

reported when p >= THRESHOLD (0.75).

A separate TOSS score (the tossing arm's own motion) is attached to each serve
but never folded into p; the orchestrator uses it (rule 1).

Reported time: trophy-run START minus SERVE_LEAD_S (0.90 s).  Note this is the
run start, not the peak (QUIRKS.md #5).

Pose helpers: anya2's own (fill_interior_gaps / keypoint_xy / midpoint_both_required).
Evidence and history: docs/anya2/NOTES.md, "far_serve.py".
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import workdir
from pipeline.anya2 import serve_common
from pipeline.anya2 import signals
from pipeline.anya2 import tracks
from pipeline.anya2.contract import (FAR_SERVE, ROI_FAR, W_BETWEEN, Event,
                                     Requirement, dump_events)

EVENTS_SUFFIX = "_anya2_far_serve.json"

MAX_GAP_S = 0.5            # fill pose occlusions up to this long
KP_CONF = 0.20             # keypoints at or below this confidence are ignored

# ── box sanity (refuse to normalise by a bad box) ────────────────────────
BOX_H_MIN_PX, BOX_H_MAX_PX = 12.0, 120.0
BOX_ASPECT_MAX = 1.6       # wider than this (w/h) is a merged pair or a fragment

# ── head reference (rarely needed: the far player faces the camera) ──────
HEAD_ABOVE_SHO_BH = 0.14

# ── ready ────────────────────────────────────────────────────────────────
READY_HI_MAX_BH = 0.02     # both wrists at or below the shoulder line
READY_STILL_BH_S = 1.2     # limb speed (relative to hip) counted as quiet
READY_WIN_S = 0.7          # smoothing window for the ready score
READY_MEAN_FROM_S = 2.0    # ready = mean over [trophy - 2.0 s, trophy - 0.3 s)
READY_MEAN_TO_S = 0.3

# ── pre-serve stillness (the feet; ready is the arms) ────────────────────
STILL_FROM_S, STILL_TO_S = 5.0, 1.0   # window before the trophy
STILL_FULL_BH_S = 0.20     # median travel at or below this = stationary
STILL_NONE_BH_S = 0.55     # ...at or above this = travelling
W_STILL = 0.30             # weight of the stillness multiplier
STILL_UNKNOWN = 0.5        # score when the player was untracked in the window

# ── trophy (elevation only), each term dilated by TROPHY_DILATE_S ────────
TROPHY_HEAD_MIN_BH = -0.02     # higher wrist at/below head height = no credit
TROPHY_HEAD_FULL_BH = 0.06     # ...this far above = full credit
TROPHY_LO_MIN_BH = -0.18       # lower wrist this far below the shoulders = none
TROPHY_LO_FULL_BH = -0.02      # ...level with them = full credit
TROPHY_DILATE_S = 0.20
TROPHY_MIN = 0.30

# ── swing ────────────────────────────────────────────────────────────────
SWING_MIN_S, SWING_MAX_S = 0.06, 0.90
PEAK_ABOVE_HEAD_BH = 0.04

# ── toss score: mean of four ramped parts, read from the tossing arm ─────
TOSS_PRE_FROM_S, TOSS_PRE_TO_S = -1.3, -0.2   # window (rel. trophy) for "started low"
TOSS_TOP_FROM_S, TOSS_TOP_TO_S = -0.3, 0.6    # window for the top of the toss
TOSS_LOW_NONE_BH, TOSS_LOW_FULL_BH = -0.22, -0.42   # lowest wrist vs shoulders
TOSS_LAT_MIN_BH, TOSS_LAT_FULL_BH = 0.07, 0.17      # wrist sideways from its shoulder
TOSS_HOLD_MIN_S, TOSS_HOLD_FULL_S = 0.13, 0.42      # time spent above the head
TOSS_PEAK_MIN_BH, TOSS_PEAK_FULL_BH = 0.08, 0.17    # highest point above the head

# ── combination ──────────────────────────────────────────────────────────
W_TROPHY, W_READY = 0.45, 0.20
SWING_FLOOR = 0.45
THRESHOLD = 0.75
REFRACT_S = 3.0
SERVE_LEAD_S = 0.90        # subtracted from the trophy-run start

REQUIREMENT = Requirement(roi=ROI_FAR, pose_fps=15.0, needs_ball=False,
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
                 refract_s: Optional[float] = None,
                 w_still: Optional[float] = None):
    """Far serves for a video, as contract Events.  Each far slot is scored on
    its own; then one serve per refractory window across both slots."""
    track_data = tracks.load(video, tracks_npz)
    fps = float(track_data["fps"])
    keypoints, boxes, eligible = track_data["kp"], track_data["bbox"], track_data["eligible"]

    candidates: List[Dict] = []
    for slot in tracks.FAR_SLOTS:
        seen = np.isfinite(boxes[:, slot, 0])
        if not seen.any():
            if verbose:
                print(f"[far-serve] slot {slot}: never tracked, skipped")
            continue
        features = serve_features(keypoints[:, slot], boxes[:, slot], fps,
                                  eligible=eligible[:, slot])
        serves = detect_serves(features, threshold, require_court, track=int(slot),
                               lead_s=lead_s, refract_s=refract_s, w_still=w_still)
        if verbose:
            print(f"[far-serve] slot {slot}: tracked {100 * seen.mean():5.1f}%"
                  f"  sane boxes {100 * np.mean(features.valid):5.1f}%"
                  f"  -> {len(serves)} candidates")
        candidates.extend(serves)

    kept = signals.keep_one_per_window(candidates, _refract(refract_s))
    return [Event(t=float(e["t"]), p=float(e["p"]), kind=FAR_SERVE,
                  track=e["track"],
                  detail={key: e[key] for key in ("trophy", "swing", "ready",
                                                  "still", "toss", "toss_parts",
                                                  "t_trophy", "t_contact")})
            for e in kept]


def _refract(refract_s):
    return REFRACT_S if refract_s is None else float(refract_s)


# ═════════════════════════════════════════════════════════════════════════
# Per-sample features for one player
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class FarServeFeatures:
    """Per pose sample, for one far player.  Elevations in body heights, + = higher."""
    fps: float
    valid: np.ndarray       # bool: sane box and both wrists seen
    ready: np.ndarray       # 0..1 arms-low-and-quiet score (smoothed)
    trophy: np.ndarray      # 0..1 trophy-shape score
    hi_elev: np.ndarray     # higher / lower wrist above the shoulder line
    lo_elev: np.ndarray
    hi_head: np.ndarray     # higher wrist above the head
    elev_l: np.ndarray      # each wrist above the shoulder line
    elev_r: np.ndarray
    head_l: np.ndarray      # each wrist above the head
    head_r: np.ndarray
    hi_side: np.ndarray     # +1 left wrist higher, -1 right, 0 unknown
    lat_l: np.ndarray       # each wrist's sideways offset from its own shoulder
    lat_r: np.ndarray
    still: np.ndarray       # 0..1 limbs quiet
    self_move: np.ndarray   # box travel, body heights/s
    on_court: np.ndarray    # bool: eligible


def serve_features(kp, bbox, fps: float,
                   eligible: Optional[np.ndarray] = None) -> FarServeFeatures:
    """Compute FarServeFeatures for one far slot of tracks.npz."""
    kp = np.asarray(kp, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64)
    n = len(kp)
    gap = int(MAX_GAP_S * fps)
    kp = signals.fill_interior_gaps(kp, gap)
    bbox = signals.fill_interior_gaps(bbox, gap)

    body_height = _sane_box_height(bbox)
    box_ok = np.isfinite(body_height)

    def xy(index):
        return signals.keypoint_xy(kp, index, KP_CONF)

    l_sho, r_sho = xy(signals.L_SHO), xy(signals.R_SHO)
    l_wri, r_wri = xy(signals.L_WRI), xy(signals.R_WRI)
    l_hip, r_hip = xy(signals.L_HIP), xy(signals.R_HIP)
    nose = xy(signals.NOSE)
    l_ear, r_ear = xy(signals.L_EAR), xy(signals.R_EAR)
    mid = signals.midpoint_both_required
    shoulders, hips = mid(l_sho, r_sho), mid(l_hip, r_hip)
    head = mid(mid(l_ear, r_ear), nose)
    head_y = np.where(np.isfinite(head[:, 1]),
                      head[:, 1], shoulders[:, 1] - HEAD_ABOVE_SHO_BH * body_height)

    elev_l = (shoulders[:, 1] - l_wri[:, 1]) / body_height
    elev_r = (shoulders[:, 1] - r_wri[:, 1]) / body_height
    head_l = (head_y - l_wri[:, 1]) / body_height
    head_r = (head_y - r_wri[:, 1]) / body_height

    both_wrists = np.isfinite(elev_l) & np.isfinite(elev_r)
    with np.errstate(invalid="ignore"):
        hi_elev = np.where(both_wrists, np.fmax(elev_l, elev_r), np.nan)
        lo_elev = np.where(both_wrists, np.fmin(elev_l, elev_r), np.nan)
        hi_head = np.where(both_wrists, np.fmax(head_l, head_r), np.nan)
    hi_side = np.where(both_wrists, np.where(elev_l >= elev_r, 1.0, -1.0), 0.0)

    lat_l = np.abs(l_wri[:, 0] - l_sho[:, 0]) / body_height
    lat_r = np.abs(r_wri[:, 0] - r_sho[:, 0]) / body_height

    self_move = signals.box_travel_speed(bbox, body_height, fps)

    # ── ready: wrists carried low * limbs quiet, smoothed ──
    fastest_wrist = np.fmax(
        signals.limb_speed_relative_to_hip(l_wri, hips, body_height, fps),
        signals.limb_speed_relative_to_hip(r_wri, hips, body_height, fps))
    still = 1.0 - signals.ramp(fastest_wrist, READY_STILL_BH_S * 0.5, READY_STILL_BH_S * 1.6)
    carried_low = 1.0 - signals.ramp(hi_elev, READY_HI_MAX_BH, READY_HI_MAX_BH + 0.14)
    ready_raw = carried_low * still
    window = max(1, int(round(READY_WIN_S * fps)))
    # Plain moving mean with NaN counted as 0, unlike the near side (QUIRKS.md #9).
    ready = ready_raw if window <= 1 else np.convolve(
        np.nan_to_num(ready_raw, nan=0.0), np.ones(window) / window, mode="same")

    # ── trophy: (higher wrist over head) * (lower wrist near shoulders) ──
    dilate = max(1, int(round(TROPHY_DILATE_S * fps)) * 2 + 1)
    trophy = (signals.running_max(signals.ramp(hi_head, TROPHY_HEAD_MIN_BH, TROPHY_HEAD_FULL_BH), dilate)
              * signals.running_max(signals.ramp(lo_elev, TROPHY_LO_MIN_BH, TROPHY_LO_FULL_BH), dilate))

    on_court = (np.asarray(eligible, dtype=bool) if eligible is not None
                else np.ones(n, dtype=bool))

    return FarServeFeatures(
        fps=float(fps), valid=box_ok & both_wrists, ready=ready, trophy=trophy,
        hi_elev=hi_elev, lo_elev=lo_elev, hi_head=hi_head,
        elev_l=elev_l, elev_r=elev_r, head_l=head_l, head_r=head_r, hi_side=hi_side,
        lat_l=lat_l, lat_r=lat_r, still=still, self_move=self_move, on_court=on_court)


def _sane_box_height(bbox):
    """Box height in px, NaN where the box is too small, too big or too wide."""
    height = bbox[:, 3] - bbox[:, 1]
    width = bbox[:, 2] - bbox[:, 0]
    with np.errstate(invalid="ignore"):
        ok = ((height > BOX_H_MIN_PX) & (height < BOX_H_MAX_PX)
              & (width / np.maximum(height, 1e-6) < BOX_ASPECT_MAX))
    return np.where(ok, height, np.nan)


# ═════════════════════════════════════════════════════════════════════════
# Candidates -> serves
# ═════════════════════════════════════════════════════════════════════════

def detect_serves(features: FarServeFeatures, threshold: float = THRESHOLD,
                  require_court: bool = True,
                  track: Optional[int] = None,
                  lead_s: Optional[float] = None,
                  refract_s: Optional[float] = None,
                  w_still: Optional[float] = None) -> List[Dict]:
    """Score every trophy candidate; return serves with p >= threshold.

    Each returned dict: t, p, trophy, swing, ready, still, toss, toss_parts,
    t_trophy, t_contact, track.
    """
    fps = features.fps
    still_weight = W_STILL if w_still is None else float(w_still)
    lead = SERVE_LEAD_S if lead_s is None else float(lead_s)

    serves: List[Dict] = []
    for cand in serve_common.trophy_candidates(features.trophy, features.on_court,
                                               TROPHY_MIN, require_court):
        ready = _ready_before(features, cand.peak)
        still = _still_before(features, cand.peak)
        swing = serve_common.swing(features, cand.peak, fps,
                                   SWING_MIN_S, SWING_MAX_S, PEAK_ABOVE_HEAD_BH)
        p = serve_common.combine(cand.trophy, ready, swing.score,
                                 W_TROPHY, W_READY, SWING_FLOOR)
        p *= (1.0 - still_weight) + still_weight * still
        if p < threshold:
            continue
        serves.append({
            "t": cand.run_start / fps - lead,
            "p": round(p, 4),
            "trophy": round(cand.trophy, 4), "swing": round(swing.score, 4),
            "ready": round(ready, 4), "still": round(still, 4),
            **toss_score(features, cand.peak, bool(swing.toss_is_left)),
            "t_trophy": round(cand.peak / fps, 3),
            "t_contact": round(swing.contact_s, 3) if swing.contact_s is not None else None,
            "track": track,
        })
    return signals.keep_one_per_window(serves, _refract(refract_s))


def _ready_before(features, peak):
    """Mean ready score over [peak - READY_MEAN_FROM_S, peak - READY_MEAN_TO_S)."""
    fps = features.fps
    start = max(0, peak - serve_common.samples(READY_MEAN_FROM_S, fps))
    stop = max(1, peak - serve_common.samples(READY_MEAN_TO_S, fps))
    window = features.ready[start:stop]
    window = window[np.isfinite(window)]
    return float(window.mean()) if window.size else 0.0


def _still_before(features, peak):
    """1 = stood still before the trophy, 0 = was travelling; STILL_UNKNOWN if untracked."""
    fps = features.fps
    start = max(0, peak - serve_common.samples(STILL_FROM_S, fps))
    stop = max(1, peak - serve_common.samples(STILL_TO_S, fps))
    travel = features.self_move[start:stop]
    travel = travel[np.isfinite(travel)]
    if not travel.size:
        return STILL_UNKNOWN
    return 1.0 - float(signals.ramp(float(np.median(travel)), STILL_FULL_BH_S, STILL_NONE_BH_S))


def toss_score(features: FarServeFeatures, peak: int, toss_is_left: bool) -> Dict:
    """Did the tossing arm do what a toss does?  Separate from p.

    Mean of four 0..1 parts, all relative to the trophy peak:
      low   the wrist started well below the shoulders
      lat   at the top, the wrist is offset sideways from its shoulder
      hold  the wrist stayed above the head for a while
      peak  how far above the head it got
    Returns {"toss": mean, "toss_parts": {...}}, or Nones if the arm was not
    tracked (which must not be read as a bad toss).
    """
    fps = features.fps
    elevation = features.elev_l if toss_is_left else features.elev_r
    above_head = features.head_l if toss_is_left else features.head_r
    lateral = features.lat_l if toss_is_left else features.lat_r

    def window(x, from_s, to_s):
        start = max(0, peak + int(round(from_s * fps)))
        stop = min(len(x), peak + int(round(to_s * fps)))
        values = x[start:stop]
        return values[np.isfinite(values)]

    before_rise = window(elevation, TOSS_PRE_FROM_S, TOSS_PRE_TO_S)
    top_lateral = window(lateral, TOSS_TOP_FROM_S, TOSS_TOP_TO_S)
    top_above_head = window(above_head, TOSS_TOP_FROM_S, TOSS_TOP_TO_S)
    if not (before_rise.size and top_lateral.size and top_above_head.size):
        return {"toss": None, "toss_parts": None}

    parts = {
        "low": 1.0 - float(signals.ramp(float(np.min(before_rise)),
                                        TOSS_LOW_FULL_BH, TOSS_LOW_NONE_BH)),
        "lat": float(signals.ramp(float(np.median(top_lateral)),
                                  TOSS_LAT_MIN_BH, TOSS_LAT_FULL_BH)),
        "hold": float(signals.ramp(float(np.sum(top_above_head > 0.0) / fps),
                                   TOSS_HOLD_MIN_S, TOSS_HOLD_FULL_S)),
        "peak": float(signals.ramp(float(np.max(top_above_head)),
                                   TOSS_PEAK_MIN_BH, TOSS_PEAK_FULL_BH)),
    }
    return {"toss": round(float(np.mean(list(parts.values()))), 4),
            "toss_parts": {name: round(value, 3) for name, value in parts.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description="A far-side serve detector: the same "
                                 "phase-sequence idea as `near_serve`, with")
    ap.add_argument("video")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-court", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    serves = detect_video(a.video, a.tracks, a.threshold, not a.no_court)
    print(f"[far-serve] {len(serves)} serves at p >= {a.threshold}")
    for e in serves[:40]:
        d = e.detail
        print(f"  {e.t:8.2f}s  p={e.p:.3f}  slot={e.track}  "
              f"trophy={d['trophy']:.2f} swing={d['swing']:.2f} ready={d['ready']:.2f}")
    out = a.json or events_path(a.video)
    dump_events(serves, out, threshold=a.threshold, requirement=REQUIREMENT.__dict__)
    print(f"[far-serve] wrote {out}")


if __name__ == "__main__":
    main()
