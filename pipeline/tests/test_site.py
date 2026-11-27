"""pipeline/anya2/site.py: carrying a fixed camera's corners to a new recording."""
import cv2
import numpy as np
import pytest

from pipeline.anya2 import camera as CAM
from pipeline.anya2 import site as S


def _shifted(img, dx, dy):
    A = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
    return cv2.warpAffine(img, A, (img.shape[1], img.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


def test_small_nudge_moves_the_corners_with_it():
    ref, quad = CAM._synthetic_court()
    cur = _shifted(ref, 6.0, -4.0)
    corners, shift, n = S.register_corners(ref, quad, cur)
    err = np.hypot(*(corners - (quad + [6.0, -4.0])).T).max()
    assert err < 1.0, err
    assert 6.0 < shift < 8.0


def test_unrelated_view_needs_calibration():
    ref, quad = CAM._synthetic_court(seed=0)
    other = np.random.default_rng(5).integers(0, 255, ref.shape).astype(np.uint8)
    with pytest.raises(S.NeedsCalibration):
        S.register_corners(ref, quad, other)
