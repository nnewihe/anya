"""
signals.py
==========
Small, parameter-free maths shared by the detectors: ramps, running maxima,
run-length decomposition, gap filling, keypoint reads.

Thresholds never live here.  Every number that decides what counts as a serve
or a point end belongs to the detector that owns it, so one detector can be
retuned without touching another.

TWO FAMILIES OF POSE HELPERS -- know which one you are using
------------------------------------------------------------
There are two sets of keypoint helpers in this codebase, and they look alike
but do NOT behave alike.  Swapping one for the other changes results.

  anya2's own (defined below)           used by far_serve, point_end
    fill_interior_gaps      fills each column on its own; never fills a gap
                            touching the start or end of the clip
    keypoint_xy             NaN where confidence <= conf_floor
    midpoint_both_required  NaN unless BOTH points are present

  the walking classifier's (`walking_pose_helpers()`)
                                        used by near_serve, near_end
    fill_gaps_between_seen  fills between samples where column 0 was seen
    keypoint_xy_walking     NaN where confidence < walking.features.KP_CONF
    midpoint_with_fallback  falls back to whichever point exists
    nan_moving_mean         centred moving mean that skips NaN
    joint_angle_deg         interior angle at the middle joint

near_serve and near_end use the walking set on purpose: they read the same
near-player track as the walking classifier and must agree with it about what
a tracked frame is.
"""

from typing import List

import numpy as np

# ── COCO keypoint indices ────────────────────────────────────────────────
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16


# ── scores and windows ───────────────────────────────────────────────────

def ramp(x, lo, hi):
    """Linear 0 -> 1 as x goes from `lo` to `hi` (either order), clipped.

    NaN in, NaN out.  Pass hi < lo for a falling ramp (1 at lo, 0 at hi).
    """
    with np.errstate(invalid="ignore"):
        return np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)


def running_max(x, width):
    """Centred running max over `width` samples.  NaN is ignored (acts as -inf).

    Near the clip edges the window just gets shorter, so an event in the first
    second still counts.  Used to "dilate" a score so that a product of several
    scores asks "did each of these hold somewhere near now" rather than "on
    exactly this sample" (see NOTES: "The trophy is a phase, not an instant").
    """
    x = np.asarray(x, dtype=np.float64)
    if width <= 1:
        return x
    out = x.copy()
    for shift in range(1, width // 2 + 1):
        out[:-shift] = np.fmax(out[:-shift], x[shift:])
        out[shift:] = np.fmax(out[shift:], x[:-shift])
    return out


def true_runs(mask) -> List[tuple]:
    """The [start, stop) index ranges where `mask` is True."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    edges = np.diff(m.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    stops = list(np.flatnonzero(edges == -1) + 1)
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        stops.append(len(m))
    return list(zip(starts, stops))


def keep_one_per_window(events, gap_s, key="t", strongest=True):
    """Thin a list of event dicts so no two kept events are within `gap_s`.

    strongest=True  keep the highest "p" in each cluster (ties: earliest).
                    Right when a cluster is one action detected several times.
    strongest=False keep the EARLIEST of each cluster.
                    Right when the first member is the one being timed.
    """
    if strongest:
        by_strength = sorted(events, key=lambda e: (-e.get("p", 0.0), e[key]))
        kept = []
        for event in by_strength:
            if all(abs(event[key] - other[key]) >= gap_s for other in kept):
                kept.append(event)
        return sorted(kept, key=lambda e: e[key])
    kept = []
    for event in sorted(events, key=lambda e: e[key]):
        if not kept or event[key] - kept[-1][key] >= gap_s:
            kept.append(event)
    return kept


def nan_skipping_moving_mean(x, width):
    """Centred moving mean over `width` samples that ignores NaN.

    NaN only where the whole window is NaN.  Same formula as the walking
    classifier's `_movmean`.
    """
    x = np.asarray(x, dtype=np.float64)
    if width <= 1:
        return x
    seen = np.isfinite(x)
    kernel = np.ones(width)
    total = np.convolve(np.where(seen, x, 0.0), kernel, mode="same")
    count = np.convolve(seen.astype(float), kernel, mode="same")
    return np.where(count > 0, total / np.maximum(count, 1e-9), np.nan)


def hysteresis(x, enter_at, leave_at):
    """Boolean state that turns on at x >= enter_at and off at x < leave_at."""
    out = np.zeros(len(x), dtype=bool)
    on = False
    for i, value in enumerate(x):
        on = value >= enter_at if not on else value >= leave_at
        out[i] = on
    return out


def frame_to_frame_speed(points, fps):
    """Speed of an [N, 2] track in units/s; element 0 is NaN (no previous frame)."""
    speed = np.full(len(points), np.nan)
    speed[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) * fps
    return speed


def limb_speed_relative_to_hip(limb_xy, hip_xy, body_height_px, fps):
    """How fast a joint moves RELATIVE TO THE HIP, in body heights per second.

    Relative to the hip so that a player running across court does not read as
    arm activity, while a racket take-back while standing still does.
    """
    relative = (limb_xy - hip_xy) / body_height_px[:, None]
    return frame_to_frame_speed(relative, fps)


def box_travel_speed(boxes, body_height_px, fps):
    """How fast a player's box moves across the image, in body heights per second.

    Measured on the box's bottom-centre x and bottom y.  The box is used rather
    than a keypoint because for a small (far) player single joints are noisy,
    while the box as a whole is what moves when the player runs.
    """
    centre_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    bottom_y = boxes[:, 3]
    speed = np.full(len(boxes), np.nan)
    speed[1:] = np.hypot(np.diff(centre_x), np.diff(bottom_y)) / body_height_px[1:] * fps
    return speed


# ── anya2's own pose helpers (see the module docstring) ──────────────────

def fill_interior_gaps(a, max_gap):
    """Linearly fill runs of NaN no longer than `max_gap` samples.

    Each column is filled independently.  Gaps touching the first or last
    sample are left alone, and so are gaps longer than `max_gap`: a filled long
    hole would be a made-up posture that every later score reads as evidence.
    """
    a = np.array(a, dtype=np.float64, copy=True)
    columns = a.reshape(len(a), -1)
    index = np.arange(len(a))
    for c in range(columns.shape[1]):
        column = columns[:, c]
        seen = np.isfinite(column)
        if seen.sum() < 2:
            continue
        interpolated = np.interp(index, index[seen], column[seen])
        for start, stop in true_runs(~seen):
            if stop - start <= max_gap and start > 0 and stop < len(a):
                column[start:stop] = interpolated[start:stop]
    return columns.reshape(a.shape)


def keypoint_xy(keypoints, index, conf_floor=0.0):
    """[N, 2] image position of keypoint `index`; NaN where conf <= conf_floor."""
    k = np.asarray(keypoints, dtype=np.float64)
    xy = k[:, index, :2].copy()
    xy[k[:, index, 2] <= conf_floor] = np.nan
    return xy


def midpoint_both_required(a, b):
    """Midpoint of two [N, 2] tracks; NaN wherever either one is missing."""
    return 0.5 * (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64))


# ── the walking classifier's pose helpers ────────────────────────────────

def walking_pose_helpers():
    """`walking.features`' own helpers, under descriptive names.

    Returned as a simple namespace so callers read e.g.
    `walk.midpoint_with_fallback(l_hip, r_hip)`.  Imported lazily because
    `walking` is a sibling top-level package that may not be on sys.path yet,
    and importing it costs time callers of this module may not need to pay.

    These must stay the walking classifier's functions, not copies: the near
    serve and near-end signals read the same track as that classifier and have
    to agree with it exactly.
    """
    import os
    import sys
    from types import SimpleNamespace
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from walking import features
    return SimpleNamespace(
        fill_gaps_between_seen=features._interp_gaps,
        keypoint_xy_walking=features._kp_xy,
        midpoint_with_fallback=features._mid,
        nan_moving_mean=features._movmean,
        joint_angle_deg=features._angle,
    )
