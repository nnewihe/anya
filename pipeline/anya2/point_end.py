"""
point_end.py
============
When did the point stop?  Pose only; the ball is never read.

The live score
--------------
Per pose sample:

    activity  = max over all tracked players of how fast they are moving
                (box travel or limb speed, body heights/s)          [player_activity]
    not_rally = max(walking prob., settle, turn_away, stance_drop, idle_hands)
                -- "this looks like between points"                  [end_signal]
    raw_live  = max(near activity, far activity) * (1 - not_rally)
    live      = raw_live smoothed over 4 s, divided by the clip's own
                90th percentile (so thresholds mean the same on any camera)
                                                                     [live_score]

The non-rally term multiplies (vetoes) rather than adds: a player walking to
the ball is active but definitely not playing.

The point end
-------------
"Live" is hysteresis on the live score (enter at 0.50, leave at 0.35).  A point
end is the FALLING EDGE of a live run at least 2 s long.  Its confidence is how
far the live score stays down over the next 4 s.               [detect_ends]

Inputs (cached by run.py, stage 6):
    <stem>_anya2_walk.npz     walking classifier probability per pose sample
    <stem>_anya2_endsig.npz   near_end.py's four signals
    <stem>_anya2_tracks.npz   for player activity

Evidence and history: docs/anya2/NOTES.md, "point_end.py".
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import workdir
from pipeline.anya2 import signals
from pipeline.anya2 import tracks
from pipeline.anya2.contract import (POINT_END, ROI_BOTH, W_AFTER_SERVE, Event,
                                     Requirement, dump_events)

EVENTS_SUFFIX = "_anya2_point_end.json"
WALK_SUFFIX = "_anya2_walk.npz"
ENDSIG_SUFFIX = "_anya2_endsig.npz"

MAX_GAP_S = 0.5            # fill pose occlusions up to this long

# ── player activity ──────────────────────────────────────────────────────
ACTIVITY_MIN_BOX_PX = 8.0  # boxes shorter than this are not measured
ACTIVITY_KP_CONF = 0.2     # keypoint confidence floor for the limb term

# ── the non-rally union: max of these and the walking probability ────────
UNION_NAMES = ("settle", "turn_away", "stance_drop", "idle_hands")

# ── the live score ───────────────────────────────────────────────────────
LIVE_SMOOTH_S = 4.0        # smoothing window; longer separates better but blurs the edge
LIVE_SCALE_PCT = 90        # normalise by this percentile of the clip's own score
LIVE_HI = 0.50             # enter "live" at or above this
LIVE_LO = 0.35             # ...stay live while at or above this
LIVE_MIN_S = 2.0           # ignore live runs shorter than this

# ── events ───────────────────────────────────────────────────────────────
CONFIDENCE_LOOKAHEAD_S = 4.0   # confidence = 1 - mean live score over this window
REFRACT_S = 6.0                # at most one end per this many seconds

REQUIREMENT = Requirement(roi=ROI_BOTH, pose_fps=15.0, needs_ball=False,
                          windows=W_AFTER_SERVE)


def events_path(video, suffix=EVENTS_SUFFIX):
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(workdir.artifact_dir(video), f"{stem}{suffix}")


def walk_path(video, suffix=WALK_SUFFIX):
    return events_path(video, suffix)


def detect_video(video, tracks_npz=None, hi: float = LIVE_HI,
                 verbose: bool = True, lo: Optional[float] = None,
                 smooth_s: Optional[float] = None,
                 min_live_s: Optional[float] = None) -> List[Event]:
    """Point ends for a video, as contract Events."""
    parts = end_signal(video, tracks_npz)
    live = live_score(parts, video, tracks_npz, smooth_s=smooth_s)
    ends = detect_ends(parts, live, hi,
                       LIVE_LO if lo is None else float(lo),
                       LIVE_MIN_S if min_live_s is None else float(min_live_s))
    if verbose:
        print(f"[point-end] {len(ends)} ends (falling edges of the live score)")
    return [Event(t=float(e["t"]), p=float(e["p"]), kind=POINT_END, track=None,
                  detail={"rally_s": e["rally_s"]}) for e in ends]


def end_signal(video, tracks_npz=None) -> Dict[str, np.ndarray]:
    """The non-rally union and its parts, read from the cached files.

    Returns {"settle", "turn_away", "stance_drop", "idle_hands", "walk",
    "union", "fps"}, all trimmed to the shorter of the two files.
    (`tracks_npz` is accepted for interface compatibility and not used.)
    """
    stem = os.path.splitext(os.path.basename(video))[0]
    folder = workdir.artifact_dir(video)
    walk = np.load(os.path.join(folder, f"{stem}{WALK_SUFFIX}"))
    near_signals = np.load(os.path.join(folder, f"{stem}{ENDSIG_SUFFIX}"))
    fps = float(walk["fps"])
    n = min(len(walk["prob"]), len(near_signals[UNION_NAMES[0]]))
    parts = {name: np.nan_to_num(near_signals[name][:n], nan=0.0) for name in UNION_NAMES}
    parts["walk"] = np.asarray(walk["prob"][:n], dtype=np.float64)
    parts["union"] = np.max(np.stack([parts[name] for name in list(UNION_NAMES) + ["walk"]]),
                            axis=0)
    parts["fps"] = np.float64(fps)
    return parts


def live_score(parts: Dict[str, np.ndarray], video, tracks_npz=None,
               smooth_s: Optional[float] = None) -> np.ndarray:
    """[N] score for "a point is in play", ~0..1 (1 = the clip's 90th percentile)."""
    fps = float(parts["fps"])
    n = len(parts["union"])
    activity = player_activity(tracks.load(video, tracks_npz), fps)[:, :n]
    with np.errstate(invalid="ignore"):
        near = np.nan_to_num(np.nanmax(activity[list(tracks.NEAR_SLOTS)], axis=0), nan=0.0)
        far = np.nan_to_num(np.nanmax(activity[list(tracks.FAR_SLOTS)], axis=0), nan=0.0)
    raw_live = np.fmax(near, far) * (1.0 - parts["union"])
    window = max(1, int(round((LIVE_SMOOTH_S if smooth_s is None else float(smooth_s)) * fps)))
    smoothed = np.convolve(raw_live, np.ones(window) / window, mode="same")
    return smoothed / max(np.percentile(smoothed, LIVE_SCALE_PCT), 1e-6)


def player_activity(track_data, fps) -> np.ndarray:
    """[4 slots, N] how fast each player moves, body heights/s; NaN if untracked.

    max(box travel across the image, fastest wrist/ankle relative to the hip).
    Deliberately not court speed: far-player court positions carry metres of
    projection error.
    """
    all_kp, all_boxes = track_data["kp"], track_data["bbox"]
    gap = int(MAX_GAP_S * fps)
    per_slot = []
    for slot in range(tracks.N_SLOTS):
        kp = signals.fill_interior_gaps(all_kp[:, slot], gap)
        boxes = signals.fill_interior_gaps(all_boxes[:, slot], gap)
        body_height = boxes[:, 3] - boxes[:, 1]
        body_height = np.where(body_height > ACTIVITY_MIN_BOX_PX, body_height, np.nan)
        travel = signals.box_travel_speed(boxes, body_height, fps)
        hip = signals.midpoint_both_required(
            signals.keypoint_xy(kp, signals.L_HIP, ACTIVITY_KP_CONF),
            signals.keypoint_xy(kp, signals.R_HIP, ACTIVITY_KP_CONF))
        fastest_limb = np.full(len(kp), np.nan)
        for joint in (signals.L_WRI, signals.R_WRI, signals.L_ANK, signals.R_ANK):
            limb = signals.keypoint_xy(kp, joint, ACTIVITY_KP_CONF)
            fastest_limb = np.fmax(fastest_limb, signals.limb_speed_relative_to_hip(
                limb, hip, body_height, fps))
        per_slot.append(np.fmax(travel, fastest_limb))
    return np.array(per_slot)


def detect_ends(parts: Dict[str, np.ndarray], live: np.ndarray,
                hi: float = LIVE_HI, lo: float = LIVE_LO,
                min_s: float = LIVE_MIN_S) -> List[Dict]:
    """A point end is the FALLING EDGE of a live run (not the onset of "looks dead":
    dead time flickers, while a real end is preceded by play)."""
    fps = float(parts["fps"])
    is_live = signals.hysteresis(live, hi, lo)
    min_run = max(1, int(round(min_s * fps)))
    lookahead = int(round(CONFIDENCE_LOOKAHEAD_S * fps))
    ends: List[Dict] = []
    for run_start, run_stop in signals.true_runs(is_live):
        if run_stop - run_start < min_run:
            continue
        after = live[run_stop:min(len(live), run_stop + lookahead)]
        mean_after = after.mean() if after.size else 1.0
        ends.append({
            "t": run_stop / fps,
            "p": round(float(np.clip(1.0 - mean_after, 0, 1)), 4),
            "rally_s": round((run_stop - run_start) / fps, 2),
        })
    return signals.keep_one_per_window(ends, REFRACT_S)


def main() -> None:
    ap = argparse.ArgumentParser(description="When did the point stop?  Pose only -- "
                                 "the ball is not read anywhere in this")
    ap.add_argument("video")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--hi", type=float, default=LIVE_HI)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ends = detect_video(a.video, a.tracks, a.hi)
    for e in ends[:25]:
        print(f"  {e.t:8.2f}s  p={e.p:.3f}  rally={e.detail['rally_s']:5.1f}s")
    out = a.json or events_path(a.video)
    dump_events(ends, out, hi=a.hi, requirement=REQUIREMENT.__dict__)
    print(f"[point-end] wrote {out}")


if __name__ == "__main__":
    main()
