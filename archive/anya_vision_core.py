"""
anya_vision_core.py
===================
Unified vision engine for near-side tennis serve detection and rally extraction.

Stage 1 of the two-stage Anya pipeline.

Merges the telemetry extraction capabilities of extract_telemetry.py directly
into near_anya_v2.py to create a single-pass video processing engine.

New capabilities vs. near_anya_v2.py
--------------------------------------
* BoxSmoother  — ported from extract_telemetry.py and applied to every
                  near_box reading in WAITING / ARMED / ACTIVE states.
* telemetry.csv — smoothed (frame_id, x, y, w, h) written every frame a
                  near player is detected.
* serve_events.json — deterministic serve events written whenever the system
                      transitions ARMED → ACTIVE.

Usage
-----
  python anya_vision_core.py path/to/match.mp4
  python anya_vision_core.py path/to/match.mp4 -o highlights.mp4 --headless
"""

import time
import random
from enum import Enum
from collections import deque
from dataclasses import dataclass, field
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Optional, Tuple
import math
from sklearn.cluster import DBSCAN
from moviepy import VideoFileClip, concatenate_videoclips
from sort import Sort
import os
import json
import csv
import re
import subprocess
import argparse
import tempfile
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from gait_detector import GaitDetector, GaitSignal
    _GAIT_DETECTOR_AVAILABLE = True
except ImportError:
    _GAIT_DETECTOR_AVAILABLE = False
    print("[WARN] gait_detector.py not importable — walking gait signal disabled.")


# =============================================================
# BoxSmoother  (ported from extract_telemetry.py)
# =============================================================

@dataclass
class BoxSmoother:
    """
    Exponentially-weighted moving average for a bounding box (cx, cy, w, h).

    Uses separate alphas for position and size, and suppresses size updates
    when the player appears stationary — the single most important guard
    against false delta_area spikes in the downstream RAI.

    Parameters
    ----------
    alpha_pos   : EWA weight for centre-point updates  (higher = more responsive)
    alpha_size  : EWA weight for width/height updates  (lower  = more stable)
    still_thresh: pixel/frame velocity below which size smoothing is further
                  suppressed (multiplied by 0.3).
    """
    alpha_pos:   float = 0.35
    alpha_size:  float = 0.12
    still_thresh: float = 4.0

    _cx: Optional[float] = field(default=None, init=False, repr=False)
    _cy: Optional[float] = field(default=None, init=False, repr=False)
    _w:  Optional[float] = field(default=None, init=False, repr=False)
    _h:  Optional[float] = field(default=None, init=False, repr=False)

    def update(self, cx: float, cy: float, w: float, h: float
               ) -> Tuple[float, float, float, float]:
        """Feed a raw (cx, cy, w, h) observation and get the smoothed version."""
        if self._cx is None:
            self._cx, self._cy, self._w, self._h = cx, cy, w, h
            return cx, cy, w, h

        vel = math.hypot(cx - self._cx, cy - self._cy)

        self._cx = (1 - self.alpha_pos) * self._cx + self.alpha_pos * cx
        self._cy = (1 - self.alpha_pos) * self._cy + self.alpha_pos * cy

        eff_alpha = self.alpha_size * 0.3 if vel < self.still_thresh else self.alpha_size
        self._w = (1 - eff_alpha) * self._w + eff_alpha * w
        self._h = (1 - eff_alpha) * self._h + eff_alpha * h

        return self._cx, self._cy, self._w, self._h

    def reset(self):
        """Clear the smoother's state (e.g. when tracking is interrupted)."""
        self._cx = self._cy = self._w = self._h = None

    def smooth_box_xyxy(self, x1: int, y1: int, x2: int, y2: int
                        ) -> Tuple[int, int, int, int]:
        """
        Convenience wrapper: accept raw (x1,y1,x2,y2), smooth internally,
        and return a smoothed (x1,y1,x2,y2) ready to draw on frame.
        """
        raw_cx = (x1 + x2) / 2.0
        raw_cy = (y1 + y2) / 2.0
        raw_w  = float(x2 - x1)
        raw_h  = float(y2 - y1)

        scx, scy, sw, sh = self.update(raw_cx, raw_cy, raw_w, raw_h)

        sx1 = int(scx - sw / 2.0)
        sy1 = int(scy - sh / 2.0)
        sx2 = int(scx + sw / 2.0)
        sy2 = int(scy + sh / 2.0)
        return sx1, sy1, sx2, sy2


# =============================================================
# BallPosition3D — 3D ball trajectory data structure
# =============================================================

@dataclass
class BallPosition3D:
    frame_id: int
    timestamp: float
    world_x: float    # ft, along baseline (0 = center mark)
    world_y: float    # ft, vertical (0 = court surface)
    world_z: float    # ft, into court (0 = baseline)
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0
    speed_mph: float = 0.0


# =============================================================
# Trajectory3DEngine — 3D ball tracking via solvePnP + optical scaling
# =============================================================

class Trajectory3DEngine:
    """
    Encapsulates camera calibration (solvePnP), 3D reconstruction via
    optical scaling of the ball's apparent size, and velocity computation.
    """

    def __init__(self, image_points_2d: np.ndarray,
                 frame_shape: Tuple[int, int], fps: float):
        """
        Parameters
        ----------
        image_points_2d : (4, 2) array of clicked pixel coordinates
            Order: BL, BR, TR, TL of the court.
        frame_shape : (height, width) of the analysis frame.
        fps : video frame rate.
        """
        self.fps = fps
        self.frame_h, self.frame_w = frame_shape

        # Camera intrinsics — estimate focal length from frame width
        self.fx = self.frame_w * 0.85
        self.fy = self.fx
        self.cx = self.frame_w / 2.0
        self.cy = self.frame_h / 2.0
        self.K = np.array([
            [self.fx,    0, self.cx],
            [   0, self.fy, self.cy],
            [   0,       0,       1],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(4)

        # 3D world points (Y=0 plane, coplanar)
        self.world_pts_3d = np.array(Config.WORLD_POINTS_3D, dtype=np.float64)

        # Calibrate
        self._calibrate(image_points_2d)

        # Position history for velocity computation
        self._positions: deque = deque(maxlen=Config.VELOCITY_3D_BUFFER_SIZE)

        # Full telemetry history (all observations across all serves)
        self._all_telemetry: List[BallPosition3D] = []

    def _calibrate(self, image_points_2d: np.ndarray):
        """Run solvePnP with IPPE solver (designed for coplanar points)."""
        img_pts = np.array(image_points_2d, dtype=np.float64).reshape(-1, 1, 2)

        # IPPE returns two solutions for coplanar points
        success, rvec, tvec = cv2.solvePnP(
            self.world_pts_3d, img_pts, self.K, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if not success:
            print("[3D ENGINE] WARNING: solvePnP failed, falling back to ITERATIVE")
            success, rvec, tvec = cv2.solvePnP(
                self.world_pts_3d, img_pts, self.K, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                raise RuntimeError("solvePnP failed with both IPPE and ITERATIVE solvers")

        self.rvec = rvec
        self.tvec = tvec
        self.R, _ = cv2.Rodrigues(rvec)
        self.R_inv = self.R.T

        # Camera position in world frame
        cam_pos = -self.R_inv @ self.tvec

        # Validate: camera should be BEHIND the baseline (Z < 0)
        # If Z > 0, we picked the wrong solvePnP solution. Flip rvec and retry.
        if cam_pos[2, 0] > 0:
            print(f"[3D ENGINE] WARNING: Camera Z={cam_pos[2,0]:.1f} > 0 (wrong solution)")
            print("[3D ENGINE] Flipping rotation vector and recalibrating...")
            rvec_flipped = -rvec  # Negate rotation vector to get the other solution
            self.rvec = rvec_flipped
            self.R, _ = cv2.Rodrigues(rvec_flipped)
            self.R_inv = self.R.T
            cam_pos = -self.R_inv @ self.tvec
            print(f"[3D ENGINE] Retrying with flipped rvec...")

        print(f"[3D ENGINE] Camera position (world): "
              f"X={cam_pos[0,0]:.1f}, Y={cam_pos[1,0]:.1f}, Z={cam_pos[2,0]:.1f} ft")
        print(f"[3D ENGINE] Expected: ~(0, {Config.CAMERA_HEIGHT_FT}, "
              f"-{Config.CAMERA_BEHIND_BASELINE_FT}) ft")

    def pixel_to_world_3d(self, px_cx: float, px_cy: float,
                          pixel_diameter: float) -> Optional[Tuple[float, float, float]]:
        """
        Reconstruct 3D world position from pixel coordinates and apparent size.

        Uses optical scaling: Z_c = (focal_length * ball_real_diam) / pixel_diam
        Then pinhole inversion + world frame transform.
        """
        if pixel_diameter < Config.MIN_PIXEL_DIAMETER:
            return None

        # Depth in camera frame
        Z_c = (self.fx * Config.BALL_REAL_DIAMETER_FT) / pixel_diameter

        # Camera frame coordinates via pinhole inversion
        X_c = (px_cx - self.cx) * Z_c / self.fx
        Y_c = (px_cy - self.cy) * Z_c / self.fy

        # Transform to world frame: P_world = R^T * (P_camera - tvec)
        P_cam = np.array([[X_c], [Y_c], [Z_c]], dtype=np.float64)
        P_world = self.R_inv @ (P_cam - self.tvec)

        return float(P_world[0, 0]), float(P_world[1, 0]), float(P_world[2, 0])

    def add_observation(self, frame_id: int, timestamp: float,
                        px_cx: float, px_cy: float,
                        px_w: float, px_h: float) -> Optional[BallPosition3D]:
        """
        Add a ball detection. Uses max(w,h) for motion blur correction.
        Returns BallPosition3D if successful, None if rejected.
        """
        pixel_diameter = max(px_w, px_h)
        result = self.pixel_to_world_3d(px_cx, px_cy, pixel_diameter)
        if result is None:
            return None

        wx, wy, wz = result
        pos = BallPosition3D(
            frame_id=frame_id, timestamp=timestamp,
            world_x=wx, world_y=wy, world_z=wz,
        )

        # Compute velocity if we have history
        vx, vy, vz, speed = self.get_velocity_3d()
        pos.vel_x = vx
        pos.vel_y = vy
        pos.vel_z = vz
        pos.speed_mph = speed

        self._positions.append(pos)
        self._all_telemetry.append(pos)
        return pos

    def get_velocity_3d(self) -> Tuple[float, float, float, float]:
        """
        Compute velocity vector from recent position history.
        Returns (vx, vy, vz, speed_mph) in ft/s (speed in mph).
        Uses median-filtered finite differences for robustness.
        """
        n = len(self._positions)
        if n < 2:
            return 0.0, 0.0, 0.0, 0.0

        window = min(Config.VELOCITY_MEDIAN_WINDOW, n - 1)
        vxs, vys, vzs = [], [], []

        for i in range(n - window, n):
            prev = self._positions[i - 1]
            curr = self._positions[i]
            dt = curr.timestamp - prev.timestamp
            if dt <= 0:
                continue
            # Check for timestamp gaps (>1 frames)
            frame_gap = curr.frame_id - prev.frame_id
            if frame_gap > 1:
                continue
            vxs.append((curr.world_x - prev.world_x) / dt)
            vys.append((curr.world_y - prev.world_y) / dt)
            vzs.append((curr.world_z - prev.world_z) / dt)

        if not vxs:
            return 0.0, 0.0, 0.0, 0.0

        def _median(vals):
            s = sorted(vals)
            mid = len(s) // 2
            if len(s) % 2 == 0:
                return (s[mid - 1] + s[mid]) / 2.0
            return s[mid]

        vx = _median(vxs)
        vy = _median(vys)
        vz = _median(vzs)
        speed_fts = math.sqrt(vx*vx + vy*vy + vz*vz)
        speed_mph = speed_fts * 3600.0 / 5280.0  # ft/s -> mph

        return vx, vy, vz, speed_mph

    def classify_serve_phase(self, player_box_top_y_px: float) -> str:
        """
        Classify current ball motion as TOSS, SERVE, or NONE.

        SERVE:  High |Vy| (upward), low |Vz|, ball above player box.
        NONE:  Insufficient data or no clear pattern.
        """
        if len(self._positions) < 2:
            return "NONE"

        vx, vy, vz, speed_mph = self.get_velocity_3d()
        latest = self._positions[-1]

        # Check if ball is above player (in pixel space, lower y = higher)
        ball_px_cy = None
        # Project latest world position back to pixel to check vs player box
        P_world = np.array([[latest.world_x], [latest.world_y], [latest.world_z]],
                           dtype=np.float64)
        P_cam = self.R @ P_world + self.tvec
        if P_cam[2, 0] > 0:
            px_x = self.fx * P_cam[0, 0] / P_cam[2, 0] + self.cx
            px_y = self.fy * P_cam[1, 0] / P_cam[2, 0] + self.cy
            ball_px_cy = px_y

        ball_above_player = (ball_px_cy is not None and
                             ball_px_cy < player_box_top_y_px)

        # SERVE: significant upward velocity, minimal depth change, ball above player
        # In our coordinate system, Y is vertical with 0 = court surface,
        # so upward toss means positive Vy (increasing Y)
        if (abs(vy) > Config.TOSS_VY_MIN_FT_SEC and
                abs(vz) < Config.TOSS_VZ_MAX_FT_SEC and
                ball_above_player):
            return "SERVE"

        return "NONE"

    def get_world_pos_2d(self, px_x: float, px_y: float) -> Tuple[float, float]:
        """
        Backward-compatible 2D projection: ray-plane intersection with Y=0 plane.
        Returns (world_x, world_z) on the court surface.
        """
        # Ray from camera through pixel in camera frame
        ray_cam = np.array([
            [(px_x - self.cx) / self.fx],
            [(px_y - self.cy) / self.fy],
            [1.0],
        ], dtype=np.float64)

        # Transform ray to world frame
        ray_world = self.R_inv @ ray_cam

        # Camera position in world frame
        cam_pos = -self.R_inv @ self.tvec

        # Intersect with Y=0 plane: cam_pos.y + t * ray_world.y = 0
        if abs(ray_world[1, 0]) < 1e-10:
            return 0.0, 0.0
        t = -cam_pos[1, 0] / ray_world[1, 0]

        world_x = cam_pos[0, 0] + t * ray_world[0, 0]
        world_z = cam_pos[2, 0] + t * ray_world[2, 0]

        # Map from spec coordinate system (X: baseline, center=0) to
        # old homography system (X: 0..27, Z: 0..78)
        # Spec: X = -13.5..13.5, Z = 0..60 (service line)
        # Old: X = 0..27, Y = 0..78
        court_x = world_x + Config.COURT_WIDTH_FT / 2.0  # shift to 0..27
        court_y = world_z  # Z in spec = Y (depth) in old system

        return court_x, court_y

    def get_telemetry_history(self) -> List[BallPosition3D]:
        return self._all_telemetry

    def reset(self):
        """Clear position buffer (e.g. between serves)."""
        self._positions.clear()


# =============================================================
# Exclusion Zone Helpers
# =============================================================

def create_auto_exclusion_zones(
    video_path: str,
    ball_model,
    num_frames: int = 20,
    conf: float = 0.05,
    eps: int = 30,
    min_samples: int = 3,
    padding: int = 5,
    ball_class_index: int = 0,
    analysis_size: tuple = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Analyse a video to find static clusters of objects that look like balls
    (e.g. ball-baskets) and return exclusion zones (rectangles) for them.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        return []

    frame_indices = random.sample(range(total_frames), num_frames)

    all_detections = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        if analysis_size is not None:
            frame = cv2.resize(frame, analysis_size, interpolation=cv2.INTER_AREA)

        res = ball_model(frame, verbose=False, conf=conf, imgsz=Config.BALL_IMGSZ)
        if res and res[0].boxes:
            for b in res[0].boxes:
                if int(b.cls[0]) != ball_class_index:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                all_detections.append((cx, cy))

    cap.release()

    if len(all_detections) < min_samples:
        return []

    X = np.array(all_detections)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    zones = []
    for k in set(labels):
        if k == -1:
            continue
        pts = X[labels == k]
        if len(pts) > 0:
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            zones.append((
                int(x_min - padding), int(y_min - padding),
                int(x_max + padding), int(y_max + padding),
            ))

    return zones


def _is_in_exclusion_zone(x, y, exclusion_zones):
    for (x1, y1, x2, y2) in exclusion_zones:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def _court_cache_path(video_path: str) -> str:
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_court_cache.json")


def init_court(video_path: str, target_idx: int = 300, analysis_size: tuple = None):
    """Interactive court corner selection with JSON caching."""
    cache_path = _court_cache_path(video_path)

    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            cached_size = tuple(cached.get("analysis_size", [None, None]))
            if cached_size == (analysis_size if analysis_size else (None, None)):
                pts   = [tuple(p) for p in cached["points"]]
                shape = tuple(cached["frame_shape"])
                print(f"[COURT] Loaded cached corners from: {os.path.basename(cache_path)}")
                return pts, shape
            else:
                print("[COURT] Analysis size changed — re-selecting corners.")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[COURT] Cache corrupt ({e}), re-selecting.")

    num_points = 4
    win = "Click 4 court corners (any order). Press r=reset, q=quit"

    base = get_reference_frame(video_path, target_idx=target_idx)
    if analysis_size is not None:
        base = cv2.resize(base, analysis_size, interpolation=cv2.INTER_AREA)
    img = base.copy()

    state = {"img": img, "clicked_pts": [], "done": False, "win": win, "num_points": num_points}

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, state["img"])
    cv2.setMouseCallback(win, select_points, state)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if state["done"]:
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            pts   = [(float(x), float(y)) for x, y in state["clicked_pts"]]
            shape = base.shape
            try:
                cache_data = {
                    "points": pts,
                    "frame_shape": list(shape),
                    "analysis_size": list(analysis_size) if analysis_size else [None, None],
                    "video": os.path.basename(video_path),
                }
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f, indent=2)
                print(f"[COURT] Saved corners to: {os.path.basename(cache_path)}")
            except Exception as e:
                print(f"[COURT] WARN: Could not save cache: {e}")
            return pts, shape

        if key == ord("r"):
            state["clicked_pts"].clear()
            state["done"] = False
            state["img"] = base.copy()
            cv2.imshow(win, state["img"])

        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            raise RuntimeError("Court polygon selection aborted by user.")


def point_line_distance_px(P, A, B):
    Px, Py = P
    Ax, Ay = A
    Bx, By = B
    ABx, ABy = Bx - Ax, By - Ay
    APx, APy = Px - Ax, Py - Ay
    cross = abs(ABx * APy - ABy * APx)
    denom = math.hypot(ABx, ABy)
    return 0.0 if denom == 0 else cross / denom


def get_reference_frame(video_path: str, target_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Could not read any frame from video.")
        return frame
    ref_idx = min(target_idx, total_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read reference frame (idx={ref_idx}).")
    return frame


def select_points(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    state = param
    state["clicked_pts"].append((x, y))
    cv2.circle(state["img"], (x, y), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    if len(state["clicked_pts"]) == state["num_points"]:
        state["done"] = True
    cv2.imshow(state["win"], state["img"])


def build_mask(frame_shape, poly):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def point_in_mask(mask, x, y):
    h, w = mask.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return mask[int(y), int(x)] != 0


def probe_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or fps > 300:
        print(f"[WARN] Video reported FPS={fps}, falling back to 30.0")
        fps = 30.0

    duration_sec = frame_count / fps if fps > 0 else 0.0
    info = {
        "fps": fps, "frame_count": frame_count,
        "width": width, "height": height, "duration_sec": duration_sec,
    }

    print(f"\n{'='*50}")
    print(f"  VIDEO PROBE: {os.path.basename(video_path)}")
    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {frame_count}")
    print(f"  Duration   : {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
    print(f"{'='*50}\n")
    return info


def resize_for_analysis(frame):
    return cv2.resize(frame, (Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT),
                      interpolation=cv2.INTER_AREA)


# =============================================================
# 1. CONFIGURATION
# =============================================================

class Config:
    # Player velocity thresholds (used for ACTIVE state exit logic)
    PLAYER_WALK_VELOCITY_THRESHOLD   = 2.0

    # Time windows
    EVENT_WINDOW_SECONDS    = 1.2
    BALL_LOST_TIMEOUT_SECONDS = 2.0

    # Thresholds
    MIN_BALL_VELOCITY_FT_SEC   = 15.0
    COURT_X_PADDING_FT         = 15.0

    ABSOLUTE_BALL_LOST_TIMEOUT_ACTIVE = 20.0
    ABSOLUTE_BALL_LOST_TIMEOUT_IDLE   = 6.0

    # Model paths
    DEFAULT_NEAR_TROPHY_MODEL_PATH = "weights/trophy_pose_cls2/weights/best.pt"
    DEFAULT_NEAR_TROPHY_CLASS_INDEX = 1
    DEFAULT_TROPHY_PAD              = 0.30
    DEFAULT_BALL_MODEL_PATH         = "weights/ball/weights/best.pt"
    DEFAULT_BALL_CLASS_INDEX        = 0
    DEFAULT_BALL_CONF_MIN           = 0.10

    # Court geometry
    COURT_WIDTH_FT  = 27.0
    COURT_LENGTH_FT = 78.0
    FT_TO_M         = 0.3048
    COURT_WIDTH_M   = COURT_WIDTH_FT  * FT_TO_M
    COURT_LENGTH_M  = COURT_LENGTH_FT * FT_TO_M

    DEFAULT_PLAYER_MODEL_PATH = "yolo26n.pt"
    VELOCITY_WINDOW_SIZE      = 20

    # Ready / Armed thresholds
    READY_MIN_DIST_FT        = -0.5
    READY_MAX_DIST_FT        = 3.5
    READY_WAIT_TIME_SEC      = 0.4
    ARMED_BAND_WINDOW_SEC    = 2.0
    ARMED_OUT_RATIO_THRESHOLD = 0.25

    MAX_BALL_SIZE_PX = 20

    # Analysis resolution
    ANALYSIS_HEIGHT = 540
    ANALYSIS_WIDTH  = 960

    BALL_IMGSZ      = 1280
    PLAYER_IMGSZ    = 480
    TROPHY_IMGSZ    = 320
    CROP_UPSCALE_FACTOR = 2.0

    # Racquet tracking (ARMED state)
    DEFAULT_RACQUET_MODEL_PATH = "yolo26n.pt"
    DEFAULT_RACQUET_CLASS_INDEX = 38
    DEFAULT_RACQUET_CONF_MIN    = 0.25
    RACQUET_IMGSZ               = 320
    RACQUET_CROP_PAD            = 0.5   # padding factor around player box for crop


    # Velocity stabilisation
    BALL_POSITION_BUFFER_SIZE = 15
    MAX_BALL_SPEED_FT_SEC     = 180.0
    VELOCITY_MEDIAN_WINDOW    = 10

    END_TRIM_BUFFER_SEC = 2.0

    # BoxSmoother parameters (matching extract_telemetry.py defaults)
    SMOOTHER_ALPHA_POS    = 0.35
    SMOOTHER_ALPHA_SIZE   = 0.12
    SMOOTHER_STILL_THRESH = 4.0

    # 3D Velocity Engine
    BALL_REAL_DIAMETER_FT       = 0.22       # tennis ball ~6.7 cm
    CAMERA_HEIGHT_FT            = 6.0
    CAMERA_BEHIND_BASELINE_FT   = 15.0
    SERVE_VELOCITY_THRESHOLD_MPH = 37.5      # midpoint of 35-40 mph range
    TOSS_VY_MIN_FT_SEC          = 5.0        # minimum upward velocity for toss
    TOSS_VZ_MAX_FT_SEC          = 10.0       # maximum depth velocity during toss
    VELOCITY_3D_BUFFER_SIZE     = 8          # frames of 3D position history
    POST_CONTACT_RECORD_SEC     = 1.0        # record 1s after serve contact
    MIN_PIXEL_DIAMETER          = 2.0        # reject ball detections smaller than this
    WORLD_POINTS_3D = [
        [-13.5, 0,  0],   # Left Baseline Corner
        [ 13.5, 0,  0],   # Right Baseline Corner
        [-13.5, 0, 60],   # Left Service Corner
        [ 13.5, 0, 60],   # Right Service Corner
    ]


# =============================================================
# 2. DATA STRUCTURES
# =============================================================

class SystemState(Enum):
    WAITING = "WAITING"
    ARMED   = "ARMED"
    ACTIVE  = "ACTIVE"


# =============================================================
# 3. MAIN SYSTEM LOGIC
# =============================================================

class AnyaSystem:
    def __init__(self, video_path: str):
        # ------------------------------------------------------------------
        # Video parameters
        # ------------------------------------------------------------------
        self.video_path  = video_path
        self.video_info  = probe_video(video_path)
        self.fps         = self.video_info["fps"]
        self.frame_width  = self.video_info["width"]
        self.frame_height = self.video_info["height"]
        self.total_frames = self.video_info["frame_count"]

        # ------------------------------------------------------------------
        # Core state
        # ------------------------------------------------------------------
        self.frame_counter     = 0
        self.state             = SystemState.WAITING
        self.last_ball_seen_time = 0.0

        # ------------------------------------------------------------------
        # Court selection
        # ------------------------------------------------------------------
        self.court_vertices = None
        self.TL = self.TR = self.BR = self.BL = None
        self.exclusion_zones = []

        analysis_size = (Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT)
        self.court_vertices, frame_shape = init_court(
            self.video_path, analysis_size=analysis_size
        )
        self.BL, self.BR, self.TR, self.TL = self.court_vertices

        # 3D calibration via solvePnP (replaces 2D homography)
        src_pts = np.array([self.BL, self.BR, self.TR, self.TL], dtype=np.float64)
        self.engine_3d = Trajectory3DEngine(
            image_points_2d=src_pts,
            frame_shape=(Config.ANALYSIS_HEIGHT, Config.ANALYSIS_WIDTH),
            fps=self.fps,
        )

        # ------------------------------------------------------------------
        # YOLO models
        # ------------------------------------------------------------------
        self.last_ball_coord = None
        self.ball_model    = YOLO(Config.DEFAULT_BALL_MODEL_PATH)
        self.player_model  = YOLO(Config.DEFAULT_PLAYER_MODEL_PATH)
        self.trophy_model  = YOLO(Config.DEFAULT_NEAR_TROPHY_MODEL_PATH)

        # Racquet model — optional; gracefully disabled if weights are missing
        self.racquet_model = None
        try:
            self.racquet_model = YOLO(Config.DEFAULT_RACQUET_MODEL_PATH)
            print(f"[INFO] Racquet model loaded: {Config.DEFAULT_RACQUET_MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Racquet model not found ({Config.DEFAULT_RACQUET_MODEL_PATH}): {e}"
                  " — racquet overlay disabled.")

        # ------------------------------------------------------------------
        # State trackers
        # ------------------------------------------------------------------
        self.near_ready_start_time = None
        self.armed_band_history    = deque()

        # 3D serve detection sub-state within ARMED
        self._serve_phase = "IDLE"   # IDLE -> TOSS -> SERVE
        self._toss_detected_time = None

        self.near_player_positions = deque(maxlen=Config.VELOCITY_WINDOW_SIZE)
        self.near_player_boxes     = deque(maxlen=5)
        self.active_ball_positions = deque(maxlen=Config.BALL_POSITION_BUFFER_SIZE)
        self.active_start_time     = 0.0

        self.active_segments      = []
        self.current_segment_start = None
        self.dead_ball_refs        = []

        # ------------------------------------------------------------------
        # BoxSmoother  — one instance for the near player box (shared across
        #                all states so smoothing is continuous over the clip)
        # ------------------------------------------------------------------
        self.near_box_smoother = BoxSmoother(
            alpha_pos   =Config.SMOOTHER_ALPHA_POS,
            alpha_size  =Config.SMOOTHER_ALPHA_SIZE,
            still_thresh=Config.SMOOTHER_STILL_THRESH,
        )

        # Separate smoother for the racquet box (ARMED state only)
        self.racquet_box_smoother = BoxSmoother(
            alpha_pos   =Config.SMOOTHER_ALPHA_POS,
            alpha_size  =Config.SMOOTHER_ALPHA_SIZE,
            still_thresh=Config.SMOOTHER_STILL_THRESH,
        )

        # ------------------------------------------------------------------
        # Telemetry logging
        #   telemetry.csv  — 3D ball trajectory per frame
        #   serve_events   — collected in memory, flushed to JSON on finalize()
        # ------------------------------------------------------------------
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir   = os.path.dirname(os.path.abspath(video_path))
        self._base_name = base_name
        self._out_dir   = out_dir

        self._telemetry_path     = os.path.join(out_dir, f"{base_name}_telemetry.csv")
        self._serve_events_path  = os.path.join(out_dir, f"{base_name}_serve_events.json")

        self._telemetry_file     = open(self._telemetry_path, "w", newline="")
        self._telemetry_writer   = csv.writer(self._telemetry_file)
        self._telemetry_writer.writerow([
            "frame_id", "world_x", "world_y", "world_z",
            "vel_x", "vel_y", "vel_z", "speed_mph",
        ])

        self._serve_events: List[dict] = []   # accumulated; written on finalize()

        print(f"[TELEMETRY] CSV  → {self._telemetry_path}")
        print(f"[TELEMETRY] JSON → {self._serve_events_path}")

        # ------------------------------------------------------------------
        # Exclusion zone analysis
        # ------------------------------------------------------------------
        print("\n[INFO] Analysing video for static objects to exclude...")
        try:
            self.exclusion_zones = create_auto_exclusion_zones(
                self.video_path, self.ball_model,
                ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                analysis_size=(Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT),
            )
            if self.exclusion_zones:
                print(f"[INFO] Found {len(self.exclusion_zones)} static zone(s) to exclude.")
        except Exception as e:
            print(f"[WARN] Could not run auto-exclusion analysis: {e}")
            self.exclusion_zones = []

        # ------------------------------------------------------------------
        # Gait detector (walking dead-ball signal)
        # ------------------------------------------------------------------
        self.gait_detector: Optional["GaitDetector"] = None
        if _GAIT_DETECTOR_AVAILABLE:
            self.gait_detector = GaitDetector(fps=self.fps)
            print("[INFO] GaitDetector initialised.")

    # ------------------------------------------------------------------
    # Finalize — flush telemetry files to disk
    # ------------------------------------------------------------------

    def finalize(self):
        """
        Close the telemetry CSV, write serve events JSON, and generate
        velocity plots from accumulated 3D ball telemetry.
        """
        try:
            self._telemetry_file.flush()
            self._telemetry_file.close()
            print(f"\n[TELEMETRY] Closed telemetry CSV: {self._telemetry_path}")
        except Exception as e:
            print(f"[WARN] Could not close telemetry CSV: {e}")

        try:
            with open(self._serve_events_path, "w") as f:
                json.dump(self._serve_events, f, indent=2)
            print(f"[TELEMETRY] Wrote {len(self._serve_events)} serve event(s)"
                  f" to {self._serve_events_path}")
        except Exception as e:
            print(f"[WARN] Could not write serve events JSON: {e}")

        self._generate_velocity_plot()

    def _generate_velocity_plot(self):
        """Generate a 3-subplot velocity plot (Vx, Vy, Vz) from serve telemetry."""
        history = self.engine_3d.get_telemetry_history()
        if not history:
            print("[PLOT] No 3D ball telemetry to plot.")
            return

        timestamps = [p.timestamp for p in history]
        vxs = [p.vel_x for p in history]
        vys = [p.vel_y for p in history]
        vzs = [p.vel_z for p in history]
        speeds = [p.speed_mph for p in history]

        # Normalize timestamps relative to start
        t0 = timestamps[0]
        ts = [t - t0 for t in timestamps]

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("3D Ball Velocity — Serve Trajectory", fontsize=14)

        axes[0].plot(ts, vxs, "b-", linewidth=1.5, label="Vx (lateral)")
        axes[0].set_ylabel("Vx (ft/s)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, vys, "g-", linewidth=1.5, label="Vy (vertical)")
        axes[1].set_ylabel("Vy (ft/s)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(ts, vzs, "r-", linewidth=1.5, label="Vz (depth)")
        axes[2].set_ylabel("Vz (ft/s)")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        # Add speed annotation on the top subplot
        ax_speed = axes[0].twinx()
        ax_speed.plot(ts, speeds, "k--", linewidth=1, alpha=0.5, label="Speed (mph)")
        ax_speed.set_ylabel("Speed (mph)")
        ax_speed.legend(loc="upper left")

        plt.tight_layout()
        plot_path = os.path.join(self._out_dir, f"{self._base_name}_velocity_plot.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[PLOT] Velocity plot saved to: {plot_path}")

    # ------------------------------------------------------------------
    # Internal: log one telemetry row for the current frame
    # ------------------------------------------------------------------

    def _log_telemetry(self, near_box: Optional[Tuple]):
        """
        Apply the BoxSmoother to the raw near_box (player tracking continuity).
        Does NOT write to CSV — ball telemetry is written by _log_ball_telemetry.
        """
        if near_box is None:
            return None

        x1, y1, x2, y2 = near_box
        scx, scy, sw, sh = self.near_box_smoother.update(
            (x1 + x2) / 2.0, (y1 + y2) / 2.0,
            float(x2 - x1), float(y2 - y1),
        )
        return scx, scy, sw, sh

    def _log_ball_telemetry(self, pos: BallPosition3D):
        """Write a 3D ball position row to telemetry.csv."""
        self._telemetry_writer.writerow([
            pos.frame_id,
            f"{pos.world_x:.3f}", f"{pos.world_y:.3f}", f"{pos.world_z:.3f}",
            f"{pos.vel_x:.2f}",   f"{pos.vel_y:.2f}",   f"{pos.vel_z:.2f}",
            f"{pos.speed_mph:.1f}",
        ])

    # ------------------------------------------------------------------
    # Internal: log a serve event (ARMED → ACTIVE transition)
    # ------------------------------------------------------------------

    def _log_serve_event(self):
        current_time = self.frame_counter / self.fps
        event = {
            "frame_id":  self.frame_counter,
            "timestamp": round(current_time, 4),
        }
        self._serve_events.append(event)
        print(f"[SERVE EVENT] Logged serve at frame {self.frame_counter} "
              f"({current_time:.3f}s)")

    # ------------------------------------------------------------------
    # Helper: apply smoother to raw box and return smoothed xyxy
    # ------------------------------------------------------------------

    def _smooth_near_box(self, near_box: Optional[Tuple]) -> Optional[Tuple]:
        """
        Given a raw (x1,y1,x2,y2) near_box, return the smoother-filtered
        version as (sx1,sy1,sx2,sy2).  Returns None if input is None.
        """
        if near_box is None:
            return None
        x1, y1, x2, y2 = near_box
        return self.near_box_smoother.smooth_box_xyxy(x1, y1, x2, y2)

    # ------------------------------------------------------------------
    # Export highlights
    # ------------------------------------------------------------------

    def export_highlights(self, output_path="clean_highlights.mp4"):
        if not self.active_segments:
            print("[INFO] No active segments to export.")
            return

        merged_segments = []
        for start, end in sorted(self.active_segments):
            if merged_segments and merged_segments[-1][1] >= start:
                merged_segments[-1] = (merged_segments[-1][0],
                                       max(merged_segments[-1][1], end))
            else:
                merged_segments.append((start, end))

        print(f"\n[INFO] Found {len(merged_segments)} active rallies.")
        fps = self.fps

        txt_path = output_path.rsplit('.', 1)[0] + "_timestamps.txt"
        try:
            with open(txt_path, "w") as f:
                f.write("🎾 Anya System — Active Rally Timestamps 🎾\n")
                f.write(f"Source Video: {self.video_path}\n")
                f.write(f"FPS: {fps}\n")
                f.write(f"Resolution: {self.frame_width}x{self.frame_height}\n")
                f.write("-" * 45 + "\n")
                for i, (sf, ef) in enumerate(merged_segments):
                    st, et = sf / fps, ef / fps
                    f.write(f"Rally {i+1:02d} | Time: {st:>7.2f}s to {et:>7.2f}s"
                            f" | Duration: {et-st:>5.2f}s | Frames: {sf}-{ef}\n")
            print(f"[INFO] Timestamps → {txt_path}")
        except Exception as e:
            print(f"[WARN] Could not write timestamps: {e}")

        n = len(merged_segments)
        filter_parts, concat_inputs = [], []

        for i, (sf, ef) in enumerate(merged_segments):
            st, et = sf / fps, ef / fps
            filter_parts.append(
                f"[0:v]trim=start={st:.4f}:end={et:.4f},setpts=PTS-STARTPTS[v{i}]"
            )
            filter_parts.append(
                f"[0:a]atrim=start={st:.4f}:end={et:.4f},asetpts=PTS-STARTPTS[a{i}]"
            )
            concat_inputs.append(f"[v{i}][a{i}]")
            print(f"  {i+1:02d}/{n} | {st:.2f}s → {et:.2f}s  ({et-st:.2f}s)")

        concat_str = "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]"
        filter_parts.append(concat_str)
        filtergraph = ";\n".join(filter_parts)

        cmd = [
            "ffmpeg", "-y", "-i", self.video_path,
            "-filter_complex", filtergraph,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

        print("\nEncoding highlight video (single pass)...")
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True
            )
            if result.returncode != 0:
                print(f"[WARN] Single-pass failed; falling back to sequential...")
                self._export_highlights_sequential(merged_segments, fps, output_path)
                return
            print(f"\n✅ Highlight video saved to:\n{output_path}")
        except Exception as e:
            print(f"\n❌ Error during export: {e}")

    def _export_highlights_sequential(self, merged_segments, fps, output_path):
        temp_dir       = tempfile.mkdtemp(prefix="rally_clips_")
        list_file_path = os.path.join(temp_dir, "concat_list.txt")
        try:
            clip_files = []
            n = len(merged_segments)
            for i, (sf, ef) in enumerate(merged_segments, 1):
                st  = sf / fps
                dur = (ef - sf) / fps
                clip_path = os.path.join(temp_dir, f"clip_{i:03d}.mp4")
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(st), "-i", self.video_path,
                     "-t", str(dur), "-c", "copy", "-avoid_negative_ts", "make_zero",
                     clip_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )
                clip_files.append(clip_path)
                print(f"  ✓ {i}/{n}")

            with open(list_file_path, "w", encoding="utf-8") as f:
                for cp in clip_files:
                    f.write(f"file '{cp.replace(chr(92), '/')}'\n")

            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", list_file_path,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-c:a", "aac", "-b:a", "192k", "-async", "1",
                 output_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            print(f"\n✅ Highlight video saved to:\n{output_path}")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ FFmpeg error: {e}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def get_world_pos(self, px_x, px_y):
        """Project pixel to court surface (Y=0 plane) via 3D calibration."""
        return self.engine_3d.get_world_pos_2d(px_x, px_y)

    def _overlay_exclusion_zones(self, frame, alpha=0.25):
        if not self.exclusion_zones:
            return frame
        overlay = frame.copy()
        for (x1, y1, x2, y2) in self.exclusion_zones:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.rectangle(frame,   (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "EXCLUDE", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def _overlay_armed_zone(self, frame, alpha=0.15):
        """
        Draw the ready position zone (armed area) on the frame.
        This is the baseline area where the player must stand to transition
        from WAITING to ARMED state: -0.5 ft to 3.5 ft behind the baseline.
        """
        if self.engine_3d is None:
            return frame

        # Get the frame dimensions
        h, w = frame.shape[:2]

        # The armed zone spans the full court width and the ready distance band
        # World coordinates: X from -13.5 to 13.5 (full width), Z from -0.5 to 3.5 ft
        zone_points_3d = [
            [-13.5, 0, Config.READY_MIN_DIST_FT],      # left, bottom of zone
            [ 13.5, 0, Config.READY_MIN_DIST_FT],      # right, bottom of zone
            [ 13.5, 0, Config.READY_MAX_DIST_FT],      # right, top of zone
            [-13.5, 0, Config.READY_MAX_DIST_FT],      # left, top of zone
        ]

        # Project to pixel coordinates
        px_points = []
        for wx, wy, wz in zone_points_3d:
            P_world = np.array([[wx], [wy], [wz]], dtype=np.float64)
            P_cam = self.engine_3d.R @ P_world + self.engine_3d.tvec
            if P_cam[2, 0] > 0:  # Only project if in front of camera
                px_x = self.engine_3d.fx * P_cam[0, 0] / P_cam[2, 0] + self.engine_3d.cx
                px_y = self.engine_3d.fy * P_cam[1, 0] / P_cam[2, 0] + self.engine_3d.cy
                px_x = int(np.clip(px_x, 0, w - 1))
                px_y = int(np.clip(px_y, 0, h - 1))
                px_points.append((px_x, px_y))

        if len(px_points) == 4:
            # Draw semi-transparent green rectangle for armed zone
            overlay = frame.copy()
            pts = np.array(px_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            cv2.putText(frame, "ARMED ZONE", (px_points[0][0] + 10, px_points[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame

    # ------------------------------------------------------------------
    # Frame processing entry point
    # ------------------------------------------------------------------

    def process_frame(self, frame_data):
        """
        Main entry point called every frame.
        Downscales to analysis resolution before running any detectors.
        """
        self.frame_counter += 1
        current_time = self.frame_counter / self.fps

        out = resize_for_analysis(frame_data)

        if self.state == SystemState.WAITING:
            self._run_waiting_state(out, current_time)
        elif self.state == SystemState.ARMED:
            self._run_armed_state(out, current_time)
        elif self.state == SystemState.ACTIVE:
            self._run_active_state(out, current_time)

        out = self._overlay_exclusion_zones(out, alpha=0.25)
        out = self._overlay_armed_zone(out, alpha=0.15)

        cv2.putText(out, f"GLOBAL STATE: {self.state.name}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
        cv2.putText(out,
                    f"VIDEO TIME: {current_time:.2f}s  |  FPS: {self.fps:.1f}",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return out

    # ------------------------------------------------------------------
    # Player tracking
    # ------------------------------------------------------------------

    def _track_players(self, frame):
        results = self.player_model(frame, verbose=False, conf=0.5,
                                    imgsz=Config.PLAYER_IMGSZ)
        near_serve_candidates = []
        all_player_boxes      = []

        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                if int(b.cls[0]) == 0:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cx     = 0.5 * (x1 + x2)
                    y_feet = y2
                    P      = (cx, y_feet)
                    box    = (int(x1), int(y1), int(x2), int(y2))
                    all_player_boxes.append(box)

                    if self.engine_3d is not None:
                        world_x, world_y = self.get_world_pos(cx, y_feet)
                        if (world_x < -Config.COURT_X_PADDING_FT or
                                world_x > Config.COURT_WIDTH_FT + Config.COURT_X_PADDING_FT):
                            continue
                        dist_bottom = abs(world_y)
                        dist_top    = abs(world_y - Config.COURT_LENGTH_FT)
                        if dist_bottom < dist_top:
                            near_serve_candidates.append((P, dist_bottom, box))
                    else:
                        dist_top    = point_line_distance_px(P, self.TL, self.TR)
                        dist_bottom = point_line_distance_px(P, self.BL, self.BR)
                        if dist_bottom < dist_top:
                            near_serve_candidates.append((P, dist_bottom, box))

        curr_near_pos = curr_near_box = None
        if near_serve_candidates:
            best = min(near_serve_candidates, key=lambda x: x[1])
            curr_near_pos = best[0]
            curr_near_box = best[2]

        return curr_near_pos, curr_near_box, all_player_boxes

    # ------------------------------------------------------------------
    # Racquet tracking (ARMED state)
    # ------------------------------------------------------------------

    def _track_racquet_in_armed(self, frame,
                                 near_box: Optional[Tuple]) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the near player's racquet during the ARMED state using the
        dedicated racquet YOLO model.

        Searches within a padded crop centred on the player bounding box, then
        maps the best detection back to full-frame coordinates and smooths it
        with racquet_box_smoother.

        Returns
        -------
        (x1, y1, x2, y2) in full-frame pixels, or None if not detected.
        """
        if self.racquet_model is None or near_box is None:
            return None

        nx1, ny1, nx2, ny2 = near_box
        pw, ph = nx2 - nx1, ny2 - ny1
        fh, fw = frame.shape[:2]

        pad_x = int(pw * Config.RACQUET_CROP_PAD)
        pad_y = int(ph * Config.RACQUET_CROP_PAD)
        cx1 = max(0, nx1 - pad_x)
        cy1 = max(0, ny1 - pad_y)
        cx2 = min(fw, nx2 + pad_x)
        cy2 = min(fh, ny2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            self.racquet_box_smoother.reset()
            return None

        results = self.racquet_model(crop, verbose=False,
                                      conf=Config.DEFAULT_RACQUET_CONF_MIN,
                                      imgsz=Config.RACQUET_IMGSZ)

        best: Optional[dict] = None
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                if int(b.cls[0]) != Config.DEFAULT_RACQUET_CLASS_INDEX:
                    continue
                conf = float(b.conf[0])
                if best is None or conf > best["conf"]:
                    rx1, ry1, rx2, ry2 = map(int, b.xyxy[0].tolist())
                    best = {
                        "conf": conf,
                        "box": (cx1 + rx1, cy1 + ry1, cx1 + rx2, cy1 + ry2),
                    }

        if best is None:
            self.racquet_box_smoother.reset()
            return None

        return self.racquet_box_smoother.smooth_box_xyxy(*best["box"])

    # ------------------------------------------------------------------
    # Ball detection pipeline
    # ------------------------------------------------------------------

    def _detect_all_balls(self, frame, all_player_boxes):
        ball_res   = self.ball_model(frame, verbose=False, conf=0.05,
                                     imgsz=Config.BALL_IMGSZ)
        candidates = []

        if ball_res and ball_res[0].boxes is not None:
            for b in ball_res[0].boxes:
                if int(b.cls[0]) != Config.DEFAULT_BALL_CLASS_INDEX:
                    continue
                bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                conf   = float(b.conf[0])

                bw, bh = bx2 - bx1, by2 - by1
                if bw > Config.MAX_BALL_SIZE_PX or bh > Config.MAX_BALL_SIZE_PX:
                    continue

                PLAYER_PAD_X       = 10
                PLAYER_PAD_Y_TOP    = 5
                PLAYER_PAD_Y_BOTTOM = 25
                in_player = False
                for (px1, py1, px2, py2) in all_player_boxes:
                    if ((px1 - PLAYER_PAD_X) <= cx <= (px2 + PLAYER_PAD_X) and
                            (py1 - PLAYER_PAD_Y_TOP) <= cy <= (py2 + PLAYER_PAD_Y_BOTTOM)):
                        in_player = True
                        break
                if in_player:
                    continue

                if _is_in_exclusion_zone(cx, cy, self.exclusion_zones):
                    continue

                world_x, world_y = self.get_world_pos(cx, cy)
                if (world_x < -Config.COURT_X_PADDING_FT or
                        world_x > Config.COURT_WIDTH_FT + Config.COURT_X_PADDING_FT):
                    continue

                candidates.append({
                    "world_x":      world_x,
                    "world_y":      world_y,
                    "pixel_box":    (bx1, by1, bx2, by2),
                    "pixel_center": (cx, cy),
                    "pixel_w":      bw,
                    "pixel_h":      bh,
                    "conf":         conf,
                })

        return candidates

    def _select_best_ball(self, candidates, now):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        scored = []
        for c in candidates:
            wx, wy = c["world_x"], c["world_y"]
            score  = 0.0

            if len(self.active_ball_positions) >= 1:
                last = self.active_ball_positions[-1]
                dist = math.hypot(wx - last[0], wy - last[1])
                score += max(0, 50.0 - dist) * 2.0
            else:
                if 0 <= wx <= Config.COURT_WIDTH_FT and 0 <= wy <= Config.COURT_LENGTH_FT:
                    score += 20.0

            score += c["conf"] * 30.0

            if self.dead_ball_refs:
                min_dead = min(math.hypot(wx - dx, wy - dy)
                               for dx, dy in self.dead_ball_refs)
                score += min(min_dead, 30.0) * 1.5
            else:
                score += 30.0

            scored.append((score, c))

        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1]

    def _snapshot_dead_balls(self, frame, all_player_boxes):
        candidates = self._detect_all_balls(frame, all_player_boxes)
        positions  = [(c["world_x"], c["world_y"]) for c in candidates]
        if positions:
            print(f"[DEAD BALL SNAPSHOT] {len(positions)} ball(s) at serve time:")
            for i, (wx, wy) in enumerate(positions):
                print(f"  #{i}: ({wx:.1f}, {wy:.1f}) ft")
        return positions

    def _validate_ball_jump(self, new_pos, now):
        if not self.active_ball_positions:
            return True
        last  = self.active_ball_positions[-1]
        dist  = math.hypot(new_pos[0] - last[0], new_pos[1] - last[1])
        dt    = now - last[2]
        if dt <= 0:
            return dist < 5.0
        return (dist / dt) <= Config.MAX_BALL_SPEED_FT_SEC

    def _compute_stable_velocity(self):
        n = len(self.active_ball_positions)
        if n < 2:
            return 0.0
        window    = min(Config.VELOCITY_MEDIAN_WINDOW, n - 1)
        pairwise  = []
        for i in range(n - window, n):
            prev = self.active_ball_positions[i - 1]
            curr = self.active_ball_positions[i]
            dist = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
            dt   = curr[2] - prev[2]
            if dt > 0:
                pairwise.append(dist / dt)
        if not pairwise:
            return 0.0
        pairwise.sort()
        mid = len(pairwise) // 2
        if len(pairwise) % 2 == 0:
            return (pairwise[mid - 1] + pairwise[mid]) / 2.0
        return pairwise[mid]

    # ------------------------------------------------------------------
    # ACTIVE STATE
    # ------------------------------------------------------------------

    def _run_active_state(self, frame, now):
        near_pos, raw_near_box, all_player_boxes = self._track_players(frame)

        # ---- Apply BoxSmoother to the raw near_box ----
        near_box = self._smooth_near_box(raw_near_box)
        self._log_telemetry(raw_near_box)

        # ---- Gait detector update ----
        gait_signal: Optional["GaitSignal"] = None
        if self.gait_detector is not None:
            gait_signal = self.gait_detector.update(
                frame=frame,
                near_player_box=near_box,
                frame_idx=self.frame_counter,
            )

        player_velocity = 0.0

        if near_pos:
            self.near_player_positions.append(near_pos)
            if near_box:
                self.near_player_boxes.append(near_box)
                nx1, ny1, nx2, ny2 = near_box
                cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)

            if len(self.near_player_positions) >= 5:
                old_p = self.near_player_positions[0]
                new_p = self.near_player_positions[-1]
                dist  = math.hypot(new_p[0] - old_p[0], new_p[1] - old_p[1])
                player_velocity = dist / len(self.near_player_positions)

        # ---- Ball detection and 3D tracking ----
        candidates = self._detect_all_balls(frame, all_player_boxes)
        best_ball  = self._select_best_ball(candidates, now)

        for c in candidates:
            bx1, by1, bx2, by2 = c["pixel_box"]
            if c is best_ball:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, f"LIVE {c['conf']:.2f}", (bx1, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 140, 255), 2)
                cv2.putText(frame, f"ALT {c['conf']:.2f}", (bx1, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 140, 255), 1)

        ball_pos_3d = None
        if best_ball:
            proposed = (best_ball["world_x"], best_ball["world_y"], now)
            if self._validate_ball_jump(proposed, now):
                self.last_ball_seen_time = now
                self.active_ball_positions.append(proposed)

                # Feed into 3D engine
                px_cx, px_cy = best_ball["pixel_center"]
                ball_pos_3d = self.engine_3d.add_observation(
                    frame_id=self.frame_counter,
                    timestamp=now,
                    px_cx=px_cx, px_cy=px_cy,
                    px_w=best_ball["pixel_w"],
                    px_h=best_ball["pixel_h"],
                )
                if ball_pos_3d:
                    self._log_ball_telemetry(ball_pos_3d)
            else:
                best_ball = None

        # ---- 3D velocity display (convert ft/s to mph for display) ----
        vx, vy, vz, speed_mph = self.engine_3d.get_velocity_3d()
        vx_mph = vx * 0.681818
        vy_mph = vy * 0.681818
        vz_mph = vz * 0.681818
        time_since_last_ball = now - self.last_ball_seen_time

        cv2.putText(frame, "STATUS: ACTIVE", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(frame, f"Vx:{vx_mph:.1f} Vy:{vy_mph:.1f} Vz:{vz_mph:.1f} mph", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"SPEED: {speed_mph:.1f} mph", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame,
                    f"DETS: {len(candidates)}  DEAD REFS: {len(self.dead_ball_refs)}",
                    (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f"BALL LOST: {time_since_last_ball:.1f}s", (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ---- Gait HUD overlay ----
        if gait_signal is not None:
            gait_color  = (0, 255, 128) if gait_signal.is_walking else (180, 180, 180)
            gait_label  = (f"GAIT: WALKING {gait_signal.walk_duration_s:.1f}s"
                           if gait_signal.is_walking else
                           f"GAIT: conf={gait_signal.confidence:.2f}")
            cv2.putText(frame, gait_label, (20, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, gait_color, 1)
            if not math.isnan(gait_signal.stride_freq_hz):
                cv2.putText(frame,
                            f"  stride {gait_signal.stride_freq_hz:.2f} Hz",
                            (20, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (180, 180, 180), 1)

        # ---- Transition: ACTIVE -> END (ball lost timeout) ----
        player_is_active = (near_pos is not None and
                            player_velocity >= Config.PLAYER_WALK_VELOCITY_THRESHOLD)
        timeout = (Config.ABSOLUTE_BALL_LOST_TIMEOUT_ACTIVE
                   if player_is_active
                   else Config.ABSOLUTE_BALL_LOST_TIMEOUT_IDLE)

        if time_since_last_ball > timeout:
            print(f"\n[TRANSITION] ACTIVE -> END. "
                  f"Ball missing > {timeout:.1f}s.")

            if self.current_segment_start is not None:
                self.active_segments.append(
                    (self.current_segment_start, self.frame_counter))
                self.current_segment_start = None

            next_state = SystemState.WAITING
            if near_pos and self.engine_3d is not None:
                px_ft, py_ft = self.get_world_pos(near_pos[0], near_pos[1])
                if py_ft < 0 and Config.READY_MIN_DIST_FT <= abs(py_ft) <= Config.READY_MAX_DIST_FT:
                    next_state = SystemState.ARMED
                    print("[BYPASS] Player already at baseline. Jumping to ARMED.")

            self.state = next_state
            self.near_player_positions.clear()
            self.active_ball_positions.clear()
            self.near_ready_start_time = None
            self.dead_ball_refs.clear()
            self.engine_3d.reset()
            if self.gait_detector is not None:
                self.gait_detector.reset()

        # ---- Gait override: sustained walking → force dead-ball splice ----
        # The player is walking for ≥ gait_override_duration_s even though the
        # ball may still be visible/bouncing.  This catches the case where the
        # ball-energy signal is slow to decay after a winner.
        elif (gait_signal is not None and
              gait_signal.is_walking and
              gait_signal.walk_duration_s >= 2.0):
            splice_frame = (gait_signal.walk_start_frame +
                            int(self.fps * 0.20)
                            if gait_signal.walk_start_frame is not None
                            else self.frame_counter)
            print(f"\n[GAIT OVERRIDE] ACTIVE -> END."
                  f" Player walking {gait_signal.walk_duration_s:.1f}s."
                  f" Splice @ frame {splice_frame}.")

            if self.current_segment_start is not None:
                self.active_segments.append(
                    (self.current_segment_start, splice_frame))
                self.current_segment_start = None

            next_state = SystemState.WAITING
            if near_pos and self.engine_3d is not None:
                px_ft, py_ft = self.get_world_pos(near_pos[0], near_pos[1])
                if py_ft < 0 and Config.READY_MIN_DIST_FT <= abs(py_ft) <= Config.READY_MAX_DIST_FT:
                    next_state = SystemState.ARMED
                    print("[BYPASS] Player already at baseline. Jumping to ARMED.")

            self.state = next_state
            self.near_player_positions.clear()
            self.active_ball_positions.clear()
            self.near_ready_start_time = None
            self.dead_ball_refs.clear()
            self.engine_3d.reset()
            if self.gait_detector is not None:
                self.gait_detector.reset()

        # ---- Emergency override: ACTIVE -> ARMED (trophy check) ----
        elif (time_since_last_ball > 2.0 and near_pos and
              self.engine_3d is not None):
            px_ft, py_ft = self.get_world_pos(near_pos[0], near_pos[1])
            dist_ft = abs(py_ft)
            in_band = (py_ft < 0 and
                       Config.READY_MIN_DIST_FT <= dist_ft <= Config.READY_MAX_DIST_FT)

            if (in_band and
                    player_velocity < Config.PLAYER_WALK_VELOCITY_THRESHOLD and
                    near_box):
                nx1, ny1, nx2, ny2 = near_box
                pw, ph = nx2 - nx1, ny2 - ny1
                fh, fw = frame.shape[:2]
                pad_x = int(pw * Config.DEFAULT_TROPHY_PAD)
                pad_y = int(ph * Config.DEFAULT_TROPHY_PAD)
                tx1, ty1 = max(0, nx1 - pad_x), max(0, ny1 - pad_y)
                tx2, ty2 = min(fw, nx2 + pad_x), min(fh, ny2 + pad_y)
                trophy_crop = frame[ty1:ty2, tx1:tx2]
                best_trophy_score = 0.0

                if trophy_crop.size > 0:
                    tr = self.trophy_model(trophy_crop, verbose=False,
                                           imgsz=Config.TROPHY_IMGSZ)
                    if tr and hasattr(tr[0], "probs") and tr[0].probs is not None:
                        idx = Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX
                        if idx < len(tr[0].probs.data):
                            best_trophy_score = float(tr[0].probs.data[idx])

                if best_trophy_score > 0.6:
                    print(f"\n[EMERGENCY OVERRIDE] ACTIVE -> ARMED."
                          f" Pose: {best_trophy_score:.2f}")
                    self.state = SystemState.ARMED
                    self.near_player_positions.clear()
                    self.active_ball_positions.clear()
                    self.dead_ball_refs.clear()
                    self.engine_3d.reset()
                    if self.gait_detector is not None:
                        self.gait_detector.reset()

    # ------------------------------------------------------------------
    # WAITING STATE
    # ------------------------------------------------------------------

    def _run_waiting_state(self, frame, now):
        near_pos, raw_near_box, _ = self._track_players(frame)

        # ---- Apply BoxSmoother ----
        near_box = self._smooth_near_box(raw_near_box)
        self._log_telemetry(raw_near_box)

        cv2.putText(frame, "STATUS: WAITING", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        if near_box and self.engine_3d is not None:
            nx1, ny1, nx2, ny2 = near_box
            cx = (nx1 + nx2) / 2.0
            _, player_y_ft = self.get_world_pos(cx, ny2)

            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (150, 150, 150), 2)

            is_behind = player_y_ft < 0
            dist_ft   = abs(player_y_ft)

            if is_behind and Config.READY_MIN_DIST_FT <= dist_ft <= Config.READY_MAX_DIST_FT:
                if self.near_ready_start_time is None:
                    self.near_ready_start_time = now
                elapsed = now - self.near_ready_start_time
                cv2.putText(frame, f"IN ZONE: {elapsed:.1f}s", (nx1, ny1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if elapsed > Config.READY_WAIT_TIME_SEC:
                    print(f"[TRANSITION] WAITING -> ARMED. "
                          f"Player held ready for {elapsed:.1f}s.")
                    self.state             = SystemState.ARMED
                    self.near_ready_start_time = None
            else:
                self.near_ready_start_time = None
                cv2.putText(frame, "NEAR WAITING", (nx1, ny1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        else:
            self.near_ready_start_time = None

    # ------------------------------------------------------------------
    # ARMED STATE
    # ------------------------------------------------------------------

    def _run_armed_state(self, frame, now):
        near_pos, raw_near_box, all_player_boxes = self._track_players(frame)

        # ---- Apply BoxSmoother ----
        near_box = self._smooth_near_box(raw_near_box)
        self._log_telemetry(raw_near_box)

        cv2.putText(frame, "STATUS: ARMED", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        in_band = False
        ny1 = 0
        if near_box and self.engine_3d is not None:
            nx1, ny1, nx2, ny2 = near_box
            cx = (nx1 + nx2) / 2.0
            _, player_y_ft = self.get_world_pos(cx, ny2)

            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (150, 150, 150), 2)
            is_behind = player_y_ft < 0
            dist_ft   = abs(player_y_ft)
            in_band   = (is_behind and
                         Config.READY_MIN_DIST_FT <= dist_ft <= Config.READY_MAX_DIST_FT)

            # ---- Racquet tracking overlay ----
            racquet_box = self._track_racquet_in_armed(frame, near_box)
            if racquet_box is not None:
                rx1, ry1, rx2, ry2 = racquet_box
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 128), 2)
                cv2.putText(frame, "RACQUET", (rx1, ry1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 2)

        # ---- Band-check: player must stay near baseline ----
        self.armed_band_history.append((now, in_band))
        while (self.armed_band_history and
               (now - self.armed_band_history[0][0]) > Config.ARMED_BAND_WINDOW_SEC):
            self.armed_band_history.popleft()

        if len(self.armed_band_history) > 1:
            time_out = 0.0
            for i in range(len(self.armed_band_history) - 1):
                t1, b1 = self.armed_band_history[i]
                t2, _  = self.armed_band_history[i + 1]
                if not b1:
                    time_out += (t2 - t1)

            total_time = (self.armed_band_history[-1][0] -
                          self.armed_band_history[0][0])

            if total_time > 1.0:
                out_ratio = time_out / total_time
                cv2.putText(frame, f"OUT BAND: {out_ratio:.0%}", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if out_ratio > Config.ARMED_OUT_RATIO_THRESHOLD:
                    print(f"[TRANSITION] ARMED -> WAITING. "
                          f"Out of band {out_ratio:.0%} over {total_time:.1f}s.")
                    self.state = SystemState.WAITING
                    self.armed_band_history.clear()
                    self.near_ready_start_time = None
                    self._serve_phase = "IDLE"
                    self.engine_3d.reset()
                    self.racquet_box_smoother.reset()
                    return

        # ---- 3D ball detection and velocity-based serve detection ----
        candidates = self._detect_all_balls(frame, all_player_boxes)
        best_ball  = self._select_best_ball(candidates, now)

        # Draw ball candidates
        for c in candidates:
            bx1, by1, bx2, by2 = c["pixel_box"]
            if c is best_ball:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, f"BALL {c['conf']:.2f}", (bx1, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 140, 255), 2)

        # Feed best ball into 3D engine
        ball_pos_3d = None
        if best_ball:
            px_cx, px_cy = best_ball["pixel_center"]
            ball_pos_3d = self.engine_3d.add_observation(
                frame_id=self.frame_counter,
                timestamp=now,
                px_cx=px_cx, px_cy=px_cy,
                px_w=best_ball["pixel_w"],
                px_h=best_ball["pixel_h"],
            )
            if ball_pos_3d:
                self._log_ball_telemetry(ball_pos_3d)

        # Get velocity and classify serve phase
        vx, vy, vz, speed_mph = self.engine_3d.get_velocity_3d()
        phase = self.engine_3d.classify_serve_phase(ny1)

        # Display 3D velocity telemetry (convert ft/s to mph for display)
        vx_mph = vx * 0.681818
        vy_mph = vy * 0.681818
        vz_mph = vz * 0.681818
        cv2.putText(frame, f"PHASE: {self._serve_phase}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Vx:{vx_mph:.1f} Vy:{vy_mph:.1f} Vz:{vz_mph:.1f} mph", (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"SPEED: {speed_mph:.1f} mph", (20, 235),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # T velocity transition: IDLE -> TOSS/SERVE
        if self._serve_phase == "IDLE" and phase == "SERVE":
            self._toss_detected_time = now
            print(f"[3D] TOSS detected at {now:.2f}s. "
                  f"Vy={vy:.1f} ft/s, Vz={vz:.1f} ft/s")

            self._serve_phase = "SERVE"
            print(f"[TRANSITION] ARMED -> ACTIVE. ")

            self._log_serve_event()

            self.state = SystemState.ACTIVE
            buffer_frames = int(self.fps * 1.0)
            self.current_segment_start = max(0, self.frame_counter - buffer_frames)

            _, _, snap_players = self._track_players(frame)
            self.dead_ball_refs = self._snapshot_dead_balls(frame, snap_players)

            self._serve_phase = "IDLE"
            self.racquet_box_smoother.reset()

            self.near_player_positions.clear()
            self.active_ball_positions.clear()
            self.last_ball_seen_time = now
            self.active_start_time   = now



# =============================================================
# 4. ENTRY POINT
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Anya Vision Core — serve detection + telemetry logging."
    )
    parser.add_argument(
        "video",
        help="Path to the input video file",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output highlight video path (default: <input>_highlights.mp4)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without live preview (faster).",
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    if args.output:
        output_path = args.output
    else:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_highlights.mp4"

    system = AnyaSystem(video_path)
    cap    = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    print(f"\n[INFO] Processing {system.fps:.1f} FPS "
          f"({system.frame_width}x{system.frame_height})...")
    print("[INFO] Press 'q' to stop early.\n")

    if not args.headless:
        cv2.namedWindow("Anya Vision Core", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = system.process_frame(frame)
            if not args.headless:
                cv2.imshow("Anya Vision Core", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] Stopped early by user.")
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"\n[INFO] Processed {system.frame_counter} frames.")

    # ---- Write telemetry outputs ----
    system.finalize()

    # ---- Export highlight video ----
    system.export_highlights(output_path)


if __name__ == "__main__":
    main()
