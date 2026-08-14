from types import SimpleNamespace
import numpy as np
import pytest
from tennis_walking.detector import WalkingDetector, _select_near_pose, project_point


def pose(hip_y, left_x, right_x):
    points = [SimpleNamespace(x=.5, y=.7, visibility=1.) for _ in range(33)]
    points[23] = SimpleNamespace(x=.49, y=hip_y, visibility=1.)
    points[24] = SimpleNamespace(x=.51, y=hip_y, visibility=1.)
    points[27] = SimpleNamespace(x=left_x, y=.86, visibility=1.)
    points[28] = SimpleNamespace(x=right_x, y=.86, visibility=1.)
    return points


def test_project_point_identity():
    assert project_point((12, 4), np.eye(3)) == (12.0, 4.0)


def test_far_side_pose_is_ignored():
    detector = WalkingDetector(np.eye(3))
    metrics, confidence, *_ = detector.update(0, pose(.3, .4, .6), 100, 100)
    assert metrics is None and confidence == 0


def test_near_pose_produces_court_metrics():
    detector = WalkingDetector(np.eye(3), near_side_min_image_y=.5)
    detector.update(0, pose(.7, .4, .6), 10, 10)
    metrics, *_ = detector.update(.5, pose(.72, .4, .6), 10, 10)
    assert metrics is not None
    assert metrics["hip"] == pytest.approx((5.0, 7.2))
    assert metrics["speed"] is not None and metrics["speed"] > 0


def test_multi_pose_selects_lower_reliable_player():
    far, near = pose(.35, .4, .6), pose(.72, .4, .6)
    assert _select_near_pose([far, near], 100, .4, .45) is near
