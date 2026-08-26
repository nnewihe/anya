"""
court.py
========
Image -> court-metres mapping for the 960x540 analysis frame.

The four cached court points (``<clip>/<video>_court_cache.json``) are the
singles corners in 960x540 order [near-left, near-right, far-right, far-left].
They map onto a singles rectangle 8.23 m wide by 23.77 m long with the origin
at the near-left corner, +y running away from the camera. Court y therefore
splits the halves at 11.885 m and goes negative behind the near baseline, which
is exactly the discriminant used to pick the near player.
"""

import json
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from pipeline import workdir as _workdir

import cv2
import numpy as np

COURT_W = 8.23      # singles width, metres
COURT_L = 23.77     # baseline to baseline, metres
HALF_L = COURT_L / 2.0
ANALYSIS_SIZE = (960, 540)


def court_cache_path(video_path):
    d = _workdir.artifact_dir(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}_court_cache.json")


def load_homography(video_path):
    """3x3 image(960x540) -> court metres homography from the cached corners."""
    data = json.load(open(court_cache_path(video_path)))
    src = np.array(data["points"], dtype=np.float32)
    dst = np.array([[0.0, 0.0], [COURT_W, 0.0], [COURT_W, COURT_L], [0.0, COURT_L]],
                   dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_court(H, pts):
    """Project [N,2] image points to [N,2] court metres."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)
