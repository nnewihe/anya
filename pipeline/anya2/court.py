"""
court.py
========
Court geometry for anya2: the image -> court-metres map, and the two player
gates that every detector in this package shares.

The homography itself is NOT redefined here.  `walking.court.load_homography`
already maps the four cached corners onto a singles rectangle, and duplicating
that would let the two drift; this module imports it and adds only what anya2
needs on top.

Reference frame
---------------
Origin at the near-left SINGLES corner, +x along the near baseline, +y running
away from the camera, metres.  So:

    near baseline   y = 0
    far  baseline   y = COURT_L = 23.77
    mid court       y = MID_Y   = 11.885
    singles sidelines  x = 0 and x = COURT_W = 8.23

y goes NEGATIVE behind the near baseline, which is where a server stands.

Doubles without re-calibration
------------------------------
The cached corners are the SINGLES corners and the homography is a full plane
map, so the doubles sidelines need no clicking and no new cache format: the
alley is a fixed 1.37 m outside each singles line, in the same court frame.
That is why `ALLEY_W` is a constant here rather than a calibration input.

The two gates
-------------
Both are the user's rules, and both are evaluated on ONE projected point --
`ground_point`, the bounding box's bottom-centre:

  side()       near vs far by which baseline the box BOTTOM is closer to in y.
  in_bounds()  the box X-CENTRE inside the doubles court plus a 3 ft margin.

Bottom-centre is the only point that carries both quantities (it is the box
x-centre, and it is the box bottom) while still being a legitimate ground-plane
projection.  Projecting a mid-body point through a GROUND-plane homography puts
the player metres up the court, so the bottom is not a detail.

NaN is False, never True.  A player whose projection is unavailable is excluded
rather than admitted -- the conservative direction for a rule whose job is to
say who is allowed to be serving.
"""

import numpy as np

from walking.court import (ANALYSIS_SIZE, COURT_L, COURT_W, HALF_L,  # noqa: F401
                           load_homography, to_court)
from pipeline import workdir as WD


def court_cache_path(video_path):
    """Override-aware court cache path.

    `walking.court.court_cache_path` computes the SAME filename convention
    (`{stem}_court_cache.json`) beside the video, unconditionally -- it has no
    knowledge of the app's work-dir override, because `walking/` is used by
    CLI/eval callers that must never be redirected.  anya2 wraps it so the
    court cache anya2 looks for is the same one `pipeline.utilities.init_court`
    writes when a work dir IS set.
    """
    import os
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(WD.artifact_dir(video_path), f"{stem}_court_cache.json")

# ── doubles geometry ─────────────────────────────────────────────────────
ALLEY_W = 1.37          # doubles alley width, metres, each side of the singles
                        # line.  Doubles court is 10.97 m wide against singles
                        # 8.23, so the alley is (10.97 - 8.23) / 2.
DOUBLES_W = COURT_W + 2 * ALLEY_W        # 10.97

# ── the 3 ft margin ──────────────────────────────────────────────────────
FT_TO_M = 0.3048
MARGIN_FT = 3.0
MARGIN_M = MARGIN_FT * FT_TO_M           # 0.9144

# Lateral eligibility band, in the singles-origin court frame.
X_LO = -(ALLEY_W + MARGIN_M)             # -2.2844
X_HI = COURT_W + ALLEY_W + MARGIN_M      # 10.5144

MID_Y = COURT_L / 2.0                    # 11.885

NEAR, FAR = "near", "far"


def ground_point(bbox):
    """Bottom-centre of an xyxy box (or [..., 4] array of them), in image px.

    Returns [..., 2].  NaN in, NaN out.
    """
    b = np.asarray(bbox, dtype=np.float64)
    return np.stack([0.5 * (b[..., 0] + b[..., 2]), b[..., 3]], axis=-1)


def project(H, bbox):
    """Court metres for the ground point of each box. [..., 4] -> [..., 2]."""
    pts = ground_point(bbox)
    flat = pts.reshape(-1, 2)
    out = np.full_like(flat, np.nan)
    ok = np.isfinite(flat).all(axis=1)
    if ok.any():
        out[ok] = to_court(H, flat[ok])
    return out.reshape(pts.shape)


def side(court_y):
    """NEAR / FAR by proximity of the ground point to each baseline.

    The user's rule stated literally: |y - 0| < |y - COURT_L|.  That reduces to
    y < MID_Y, but it is written out because the literal form is the one that
    stays correct if the court frame is ever re-origined.
    Returns an object array of NEAR/FAR/None (None where y is NaN).
    """
    y = np.asarray(court_y, dtype=np.float64)
    scalar = y.ndim == 0
    y = np.atleast_1d(y)
    out = np.full(y.shape, None, dtype=object)
    ok = np.isfinite(y)
    out[ok] = np.where(np.abs(y[ok]) < np.abs(y[ok] - COURT_L), NEAR, FAR)
    return out[0] if scalar else out


def is_near(court_y):
    """Boolean form of `side`. NaN -> False."""
    y = np.asarray(court_y, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(y) & (np.abs(y) < np.abs(y - COURT_L))


def is_far(court_y):
    y = np.asarray(court_y, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(y) & (np.abs(y) >= np.abs(y - COURT_L))


def in_bounds(court_x):
    """Box x-centre inside the doubles court plus the 3 ft margin. NaN -> False."""
    x = np.asarray(court_x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.isfinite(x) & (x >= X_LO) & (x <= X_HI)


def eligible(court_xy):
    """The lateral gate over a [..., 2] court array. NaN -> False.

    Deliberately NOT combined with a y gate: how far up or down the court a
    player may be is a per-detector question (a server stands behind the
    baseline, a receiver does not), whereas the lateral band is the user's
    court-membership rule and is the same for everyone.
    """
    c = np.asarray(court_xy, dtype=np.float64)
    return in_bounds(c[..., 0])


# ── the map is a function of time, not a constant ────────────────────────
class Geometry:
    """Image -> court metres AT A GIVEN FRAME.

    `load_homography` returns the one map implied by the four clicked corners,
    and every caller used to apply it to every frame of the video.  That is only
    right while the camera does not move.  This wraps it with the camera track
    (`pipeline.anya2.camera`), which says where frame t's image sits relative to
    the frame those corners were clicked on:

        H_at(t)  =  H_ref @ W_t

    With no cached track the camera track is the identity and `H_at` returns
    `H_ref` for every frame -- byte for byte the old behaviour, through the same
    code path rather than through a branch beside it.

    FRAME INDICES ARE SOURCE FRAMES.  Every pass in anya2 is decimated by its
    own stride, and a sample index means nothing without knowing which; the
    source frame number is the one clock they all share.
    """

    def __init__(self, video, track=None, enabled=True):
        self.H_ref = load_homography(video)
        if enabled:
            from pipeline.anya2 import camera as CAM
            self.track = track if track is not None else CAM.load(video)
        else:
            from pipeline.anya2.camera import CameraTrack
            self.track = CameraTrack.identity()
        self._H = {}
        self._Hi = {}

    @property
    def is_static(self):
        """True when this is the plain fixed homography after all."""
        return self.track.is_identity

    def H_at(self, src_frame):
        """3x3 image(src_frame) -> court metres."""
        i = self.track.index_at(src_frame)
        H = self._H.get(i)
        if H is None:
            # Cached per SAMPLE, not per frame: at 15 Hz pose over a 30 minute
            # match this is 9000 lookups against ~9000 distinct warps, and the
            # matrix product is the only work that would otherwise repeat.
            H = self._H[i] = self.H_ref @ self.track.W[i]
        return H

    def H_inv_at(self, src_frame):
        i = self.track.index_at(src_frame)
        Hi = self._Hi.get(i)
        if Hi is None:
            Hi = self._Hi[i] = np.linalg.inv(self.H_at(src_frame))
        return Hi

    def project_at(self, src_frame, bbox):
        """`project`, at a frame. [..., 4] -> [..., 2] court metres."""
        return project(self.H_at(src_frame), bbox)

    def to_court_at(self, src_frame, pts):
        """`to_court`, at a frame. [N,2] image px -> [N,2] court metres."""
        return to_court(self.H_at(src_frame), pts)
