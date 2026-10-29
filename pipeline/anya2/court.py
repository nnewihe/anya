"""
court.py
========
Court geometry: the image -> court-metres map, and the two player gates.

Court frame (metres)
--------------------
    origin           near-left SINGLES corner
    +x               along the near baseline
    +y               away from the camera
    near baseline    y = 0          (a server stands at y < 0)
    far baseline     y = COURT_L = 23.77
    mid court        y = MID_Y   = 11.885
    singles lines    x = 0 and x = COURT_W = 8.23

The clicked corners are the SINGLES corners; the doubles alleys are a fixed
1.37 m outside them, so doubles needs no extra calibration.

The two player gates (the user's rules)
---------------------------------------
Both are evaluated on the box's BOTTOM-CENTRE projected onto the ground:

    is_near()    the box bottom is closer to the near baseline than the far one
    in_bounds()  the box x-centre is inside the doubles court plus 3 ft

NaN always answers False: a player we cannot place is excluded, not admitted.

The map can change over time (camera bumps) -- see `Geometry` and camera.py.
Evidence and history: docs/anya2/NOTES.md, "court.py".
"""

import os

import numpy as np

# The homography itself comes from walking.court so the two never drift apart.
from walking.court import (ANALYSIS_SIZE, COURT_L, COURT_W, HALF_L,  # noqa: F401
                           load_homography, to_court)
from pipeline import workdir


def court_cache_path(video_path):
    """Where the clicked court corners are cached (`<stem>_court_cache.json`).

    Same filename as `walking.court.court_cache_path`, but honours the desktop
    app's work-dir override, so anya2 finds the file `utilities.init_court`
    wrote.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(workdir.artifact_dir(video_path), f"{stem}_court_cache.json")


# ── doubles court and the 3 ft margin ────────────────────────────────────
ALLEY_W = 1.37                            # doubles alley, each side, metres
DOUBLES_W = COURT_W + 2 * ALLEY_W         # 10.97

FT_TO_M = 0.3048
MARGIN_FT = 3.0
MARGIN_M = MARGIN_FT * FT_TO_M            # 0.9144

# Lateral eligibility band, court-frame x.
X_LO = -(ALLEY_W + MARGIN_M)              # -2.2844
X_HI = COURT_W + ALLEY_W + MARGIN_M       # 10.5144

MID_Y = COURT_L / 2.0                     # 11.885

NEAR, FAR = "near", "far"


def ground_point(boxes):
    """Bottom-centre (x, y) image px of xyxy box(es).  [..., 4] -> [..., 2]."""
    b = np.asarray(boxes, dtype=np.float64)
    return np.stack([0.5 * (b[..., 0] + b[..., 2]), b[..., 3]], axis=-1)


def project(H, boxes):
    """Court metres of each box's ground point.  [..., 4] -> [..., 2]."""
    image_points = ground_point(boxes)
    flat = image_points.reshape(-1, 2)
    court_points = np.full_like(flat, np.nan)
    finite = np.isfinite(flat).all(axis=1)
    if finite.any():
        court_points[finite] = to_court(H, flat[finite])
    return court_points.reshape(image_points.shape)


def is_near(court_y):
    """True where the point is closer to the near baseline than the far. NaN -> False."""
    y = np.asarray(court_y, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(y) & (np.abs(y) < np.abs(y - COURT_L))


def in_bounds(court_x):
    """True where x is inside the doubles court plus 3 ft.  NaN -> False."""
    x = np.asarray(court_x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(x) & (x >= X_LO) & (x <= X_HI)


# ── kept for callers outside the shipped pipeline (not used by it) ───────

def side(court_y):
    """NEAR / FAR / None per point (None where y is NaN)."""
    y = np.asarray(court_y, dtype=np.float64)
    scalar = y.ndim == 0
    y = np.atleast_1d(y)
    out = np.full(y.shape, None, dtype=object)
    ok = np.isfinite(y)
    out[ok] = np.where(np.abs(y[ok]) < np.abs(y[ok] - COURT_L), NEAR, FAR)
    return out[0] if scalar else out


def is_far(court_y):
    y = np.asarray(court_y, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(y) & (np.abs(y) >= np.abs(y - COURT_L))


def eligible(court_xy):
    """`in_bounds` over a [..., 2] court array."""
    c = np.asarray(court_xy, dtype=np.float64)
    return in_bounds(c[..., 0])


# ── the map at a given frame ─────────────────────────────────────────────

class Geometry:
    """Image -> court metres AT A GIVEN SOURCE FRAME.

        H_at(frame) = H_ref @ W[frame]

    H_ref is the homography from the clicked corners; W is the camera track's
    warp from that frame's image back onto the calibration frame's image
    (camera.py).  With no camera track W is the identity, so this is exactly
    the fixed homography -- through the same code path.

    Frame numbers are SOURCE frames: the one clock every decimated pass shares.
    """

    def __init__(self, video, track=None, enabled=True):
        self.H_ref = load_homography(video)
        if enabled:
            from pipeline.anya2 import camera
            self.track = track if track is not None else camera.load(video)
        else:
            from pipeline.anya2.camera import CameraTrack
            self.track = CameraTrack.identity()
        self._H_by_sample = {}
        self._H_inv_by_sample = {}

    @property
    def is_static(self):
        """True when the camera track is the identity (camera never moved)."""
        return self.track.is_identity

    def H_at(self, src_frame):
        """3x3 homography: image at `src_frame` -> court metres."""
        sample = self.track.index_at(src_frame)
        H = self._H_by_sample.get(sample)
        if H is None:
            H = self._H_by_sample[sample] = self.H_ref @ self.track.W[sample]
        return H

    def H_inv_at(self, src_frame):
        """3x3 homography: court metres -> image at `src_frame`."""
        sample = self.track.index_at(src_frame)
        H_inv = self._H_inv_by_sample.get(sample)
        if H_inv is None:
            H_inv = self._H_inv_by_sample[sample] = np.linalg.inv(self.H_at(src_frame))
        return H_inv

    def project_at(self, src_frame, boxes):
        return project(self.H_at(src_frame), boxes)

    def to_court_at(self, src_frame, points):
        return to_court(self.H_at(src_frame), points)
