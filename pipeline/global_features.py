"""
global_features.py
==================
Stream B of the dead/live feature set: the near player's GLOBAL trajectory, in
full-frame coordinates.

Stream A (extract_pose.py) normalizes keypoints *inside* the player box, which
by construction discards where the player is on court and how fast they are
covering it — the strongest dead/live cue there is. Live play is explosive
court coverage; dead time is drifting near the fence. This module recovers it.

Per frame, 8 dims:
    0 cx, 1 cy      bbox center, normalized to frame size
    2 w,  3 h       bbox size, normalized to frame size  (h is a depth proxy)
    4 dcx, 5 dcy    center velocity, per second, in frame widths/heights
    6 speed         |(dcx, dcy)|
    7 disp          displacement over a short horizon (DISP_SEC), a
                    smoothing-robust "did they actually go anywhere" term

THE STAIR-STEP TRAP
-------------------
`energy_telemetry_cache.json` runs the player detector every PLAYER_STRIDE (10)
frames and *holds* the last bbox in between, so the cached center track is a
staircase, not a trajectory. Differencing it frame-to-frame — which is what the
spec's "frame-to-frame delta of center" literally asks for — yields nine zeros
and one spike per cycle: the velocity channel becomes stride aliasing rather
than motion, and it aliases identically whether the player is sprinting or
standing still.

So we de-step first: consecutive identical bboxes are collapsed to knots, the
track is linearly interpolated between knots, and only then differenced. Real
detection gaps (None) stay NaN and are never interpolated across — they are
masked downstream like missing pose.

Used by make_windows.py (training windows) and extract_timeline.py (full-match
inference), so both see identical feature semantics.
"""

import numpy as np

N_GLOBAL = 8
DISP_SEC = 0.5          # horizon for the displacement channel
ANALYSIS_SIZE = (960, 540)


def _destep(track: np.ndarray) -> np.ndarray:
    """Undo sample-and-hold: keep the first frame of each repeated run as a knot
    and linearly interpolate between knots. NaN runs are left as NaN.

    track: [T, D] float array with NaN rows where no detection.
    """
    T = track.shape[0]
    out = np.full_like(track, np.nan)
    valid = ~np.isnan(track[:, 0])
    if valid.sum() == 0:
        return out
    idx = np.flatnonzero(valid)

    # Knots = first index of each run of identical values (plus the last sample,
    # which anchors the tail of the final run).
    knots = [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if not np.allclose(track[a], track[b], equal_nan=True):
            knots.append(b)
    if idx[-1] != knots[-1]:
        knots.append(idx[-1])
    knots = np.array(sorted(set(knots)))

    if len(knots) == 1:
        out[idx] = track[knots[0]]
        return out

    # Interpolate each dim over knots, then re-mask frames with no detection so
    # genuine dropouts do not get silently invented.
    grid = np.arange(T)
    for d in range(track.shape[1]):
        out[:, d] = np.interp(grid, knots, track[knots, d])
    out[~valid] = np.nan
    return out


def bbox_stream(bboxes, fps, frame_size=ANALYSIS_SIZE, destep=True):
    """Build the [T, 8] global-motion stream from a per-frame bbox list.

    bboxes: length-T sequence of (x, y, w, h) in frame_size pixel coords, or None.
    Returns float32 [T, N_GLOBAL] with NaN on frames that had no detection.
    """
    T = len(bboxes)
    FW, FH = float(frame_size[0]), float(frame_size[1])
    raw = np.full((T, 4), np.nan, dtype=np.float64)
    for t, b in enumerate(bboxes):
        if b is None:
            continue
        x, y, w, h = b
        raw[t] = (x + w / 2.0) / FW, (y + h / 2.0) / FH, w / FW, h / FH

    track = _destep(raw) if destep else raw
    out = np.full((T, N_GLOBAL), np.nan, dtype=np.float32)
    out[:, 0:4] = track

    # Velocity in units per second (fps-invariant, so 30/60/120 fps clips align).
    d = np.full((T, 2), np.nan)
    d[1:] = (track[1:, 0:2] - track[:-1, 0:2]) * float(fps)
    valid = ~np.isnan(track[:, 0])
    if valid.sum() > 1:
        first = int(np.flatnonzero(valid)[0])
        d[first] = 0.0                    # no velocity defined on the first sample
    out[:, 4:6] = d
    out[:, 6] = np.linalg.norm(d, axis=1)

    # Short-horizon displacement: |center(t) - center(t-k)|, a coarse motion term
    # that survives detector jitter better than the per-frame derivative.
    k = max(1, int(round(DISP_SEC * fps)))
    disp = np.full(T, np.nan)
    if T > k:
        disp[k:] = np.linalg.norm(track[k:, 0:2] - track[:-k, 0:2], axis=1)
        disp[:k] = disp[k] if not np.isnan(disp[k:k + 1]).all() else np.nan
    out[:, 7] = disp
    out[~valid] = np.nan
    return out


def bboxes_from_telemetry_rally(rally, start, span_end):
    """Per-frame bbox list for [start, span_end] from one telemetry-cache rally."""
    frames = rally["frames"]
    return [(frames.get(str(f)) or {}).get("near_bbox") for f in range(start, span_end + 1)]


# Channel names, for manifests and ablation reports.
NAMES = ["cx", "cy", "bw", "bh", "dcx", "dcy", "speed", "disp"]
