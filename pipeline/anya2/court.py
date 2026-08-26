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
                           court_cache_path, load_homography, to_court)

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
