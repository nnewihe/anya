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


# =============================================================
# BoxSmoother  
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
    # Energy thresholds
    ENERGY_BOOST_BALL_FAST        = 1000
    ENERGY_DECAY_BALL_ROLLING     = 0.3
    SHAPE_CHANGE_THRESHOLD_PX     = 15.0
    ENERGY_BOOST_PLAYER_ACTION    = 1000
    ENERGY_DECAY_BALL_OCCLUDED    = 0.15
    ENERGY_DECAY_BALL_DEAD        = 0.15
    ENERGY_DECAY_BALL_ACTION_ZONE = 0.03
    ACTION_ZONE_MAX_Y_FT          = 0.0
    ENERGY_BOOST_PLAYER_SPRINT    = 1000.0
    ENERGY_DECAY_PLAYER_WALK      = 0.2
    ENERGY_DECAY_PLAYER_MISSING   = 0.5
    PLAYER_WALK_VELOCITY_THRESHOLD   = 2.0
    PLAYER_SPRINT_VELOCITY_THRESHOLD = 6.0

    # Net-proximity energy scaling
    # As the near player moves from baseline (0 ft) toward net (~39 ft),
    # player energy deltas are amplified and ball energy deltas are attenuated.
    NET_PROXIMITY_COURT_DEPTH_FT = 39.0   # baseline-to-net distance, singles court
    NET_PROXIMITY_PLAYER_SCALE   = 3.0    # player_delta multiplier at net (1.0 = no boost)
    NET_PROXIMITY_BALL_SCALE     = 0   # ball_delta multiplier at net   (1.0 = no attenuation)

    # Gait detection
    GAIT_BUFFER_FRAMES        = 45
    GAIT_MIN_REVERSALS        = 2
    GAIT_MAX_REVERSALS        = 8
    GAIT_MIN_DRIFT_PX         = 10.0
    ENERGY_DECAY_PLAYER_WALKING_GAIT = 0.4

    # Time windows
    EVENT_WINDOW_SECONDS    = 1.2
    BALL_LOST_TIMEOUT_SECONDS = 2.0

    # Thresholds
    TRANSITION_SCORE_THRESHOLD = 0.55
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
    DEFAULT_DRAW_TOSS_ROI           = True
    MIN_FAR_TROPHY_CONF             = 0.5
    MIN_FAR_TOSS_CONF               = 0.5

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
    TOSS_BALL_IMGSZ = 320
    CROP_UPSCALE_FACTOR = 2.0

    # Racquet tracking (ARMED state)
    DEFAULT_RACQUET_MODEL_PATH = "yolo26n.pt"
    DEFAULT_RACQUET_CLASS_INDEX = 38
    DEFAULT_RACQUET_CONF_MIN    = 0.25
    RACQUET_IMGSZ               = 320
    RACQUET_CROP_PAD            = 0.5   # padding factor around player box for crop

    # Toss ball — frame differencing filter
    # Rejects static background detections (trees, sun) by requiring the YOLO
    # candidate region to have changed meaningfully from the background captured
    # at ARMED entry.
    TOSS_DIFF_MIN_MEAN    = 10.0  # min mean absolute pixel diff (0-255) inside ball box
    TOSS_DIFF_BLUR_KERNEL = 5     # Gaussian blur kernel size applied before differencing

    # Velocity stabilisation
    BALL_POSITION_BUFFER_SIZE = 15
    MAX_BALL_SPEED_FT_SEC     = 180.0
    VELOCITY_MEDIAN_WINDOW    = 5

    END_TRIM_BUFFER_SEC = 2.0

    # BoxSmoother parameters (matching extract_telemetry.py defaults)
    SMOOTHER_ALPHA_POS    = 0.35
    SMOOTHER_ALPHA_SIZE   = 0.12
    SMOOTHER_STILL_THRESH = 4.0


# =============================================================
# 2. DATA STRUCTURES
# =============================================================

class SystemState(Enum):
    WAITING = "WAITING"
    ARMED   = "ARMED"
    ACTIVE  = "ACTIVE"


@dataclass
class Detection:
    score: float
    timestamp: float


class SideBuffer:
    def __init__(self, name):
        self.name = name
        self.trophy_scores = deque()
        self.toss_scores   = deque()

    def add_trophy_score(self, score, timestamp):
        self.trophy_scores.append(Detection(score, timestamp))
        self._cleanup_old_data(self.trophy_scores, timestamp)

    def add_toss_score(self, score, timestamp):
        self.toss_scores.append(Detection(score, timestamp))
        self._cleanup_old_data(self.toss_scores, timestamp)

    def _cleanup_old_data(self, buffer, current_time):
        while buffer and (current_time - buffer[0].timestamp) > Config.EVENT_WINDOW_SECONDS:
            buffer.popleft()

    def get_max_combined_score(self):
        if not self.trophy_scores or not self.toss_scores:
            return 0.0
        return max(d.score for d in self.trophy_scores) + \
               max(d.score for d in self.toss_scores)


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

        dst_pts = np.array([
            [0,                    0],
            [Config.COURT_WIDTH_FT, 0],
            [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT],
            [0,                    Config.COURT_LENGTH_FT],
        ], dtype=np.float32)
        src_pts = np.array([self.BL, self.BR, self.TR, self.TL], dtype=np.float32)
        self.H, _ = cv2.findHomography(src_pts, dst_pts)

        # ------------------------------------------------------------------
        # Side buffers
        # ------------------------------------------------------------------
        self.near_side = SideBuffer("Near")

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
        self.last_toss_ball        = None
        self.toss_min_y_px         = None  # ← ADD THIS: tracks max height (min y-pixel) during toss

        # Toss detection: consecutive frame tracking with hysteresis
        self.toss_consecutive_frames = 0
        self.toss_gap_frames = 0
        self.toss_ball_above_head_detected = False

        # Frame differencing background — grayscale snapshot taken on ARMED entry;
        # used to reject static background detections in the toss ROI.
        self._armed_bg_gray: Optional[np.ndarray] = None

        self.near_player_positions = deque(maxlen=Config.VELOCITY_WINDOW_SIZE)
        self.near_player_boxes     = deque(maxlen=5)
        self.active_ball_positions = deque(maxlen=Config.BALL_POSITION_BUFFER_SIZE)
        self.point_energy          = 1.0
        self.active_start_time     = 0.0
        self.gait_y_buffer         = deque(maxlen=Config.GAIT_BUFFER_FRAMES)

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
        #   telemetry.csv  — smoothed box per frame
        #   serve_events   — collected in memory, flushed to JSON on finalize()
        # ------------------------------------------------------------------
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir   = os.path.dirname(os.path.abspath(video_path))

        self._telemetry_path     = os.path.join(out_dir, f"{base_name}_telemetry.csv")
        self._serve_events_path  = os.path.join(out_dir, f"{base_name}_serve_events.json")

        self._telemetry_file     = open(self._telemetry_path, "w", newline="")
        self._telemetry_writer   = csv.writer(self._telemetry_file)
        self._telemetry_writer.writerow(["frame_id", "x", "y", "w", "h"])

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
    # Finalize — flush telemetry files to disk
    # ------------------------------------------------------------------

    def finalize(self):
        """
        Close the telemetry CSV and write the serve events JSON.
        Must be called after the main processing loop.
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

    # ------------------------------------------------------------------
    # Internal: log one telemetry row for the current frame
    # ------------------------------------------------------------------

    def _log_telemetry(self, near_box: Optional[Tuple]):
        """
        Apply the BoxSmoother to near_box and write a row to telemetry.csv.
        If near_box is None (player not detected) the smoother is NOT updated
        and no row is written — downstream gaps are handled by the state machine.
        """
        if near_box is None:
            # No detection this frame — let the smoother hold its last value
            # but do not write a row (consistent with extract_telemetry.py behaviour)
            return None

        x1, y1, x2, y2 = near_box
        scx, scy, sw, sh = self.near_box_smoother.update(
            (x1 + x2) / 2.0, (y1 + y2) / 2.0,
            float(x2 - x1), float(y2 - y1),
        )

        self._telemetry_writer.writerow([
            self.frame_counter,
            f"{scx:.2f}", f"{scy:.2f}",
            f"{sw:.2f}",  f"{sh:.2f}",
        ])
        return scx, scy, sw, sh

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
        if self.H is None:
            return 0.0, 0.0
        pt_px    = np.array([[[px_x, px_y]]], dtype=np.float32)
        pt_world = cv2.perspectiveTransform(pt_px, self.H)
        return pt_world[0][0][0], pt_world[0][0][1]

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

                    if self.H is not None:
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

                world_x = world_y = None
                if self.H is not None:
                    world_x, world_y = self.get_world_pos(cx, cy)
                    if (world_x < -Config.COURT_X_PADDING_FT or
                            world_x > Config.COURT_WIDTH_FT + Config.COURT_X_PADDING_FT):
                        continue

                candidates.append({
                    "world_x":      world_x if world_x is not None else cx,
                    "world_y":      world_y if world_y is not None else cy,
                    "pixel_box":    (bx1, by1, bx2, by2),
                    "pixel_center": (cx, cy),
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

    def _detect_walking_gait(self, near_box):
        if near_box is None:
            self.gait_y_buffer.clear()
            return False
        feet_y = near_box[3]
        self.gait_y_buffer.append(feet_y)
        n = len(self.gait_y_buffer)
        if n < Config.GAIT_BUFFER_FRAMES * 0.6:
            return False
        ys    = list(self.gait_y_buffer)
        drift = abs(ys[-1] - ys[0])
        if drift < Config.GAIT_MIN_DRIFT_PX:
            return False
        residuals = [y - (ys[0] + (ys[-1] - ys[0]) * (i / (n - 1)))
                     for i, y in enumerate(ys)]
        reversals  = 0
        prev_dir   = 0
        for i in range(1, len(residuals)):
            delta = residuals[i] - residuals[i - 1]
            if abs(delta) < 0.5:
                continue
            direction = 1 if delta > 0 else -1
            if prev_dir != 0 and direction != prev_dir:
                reversals += 1
            prev_dir = direction
        return Config.GAIT_MIN_REVERSALS <= reversals <= Config.GAIT_MAX_REVERSALS

    # ------------------------------------------------------------------
    # ACTIVE STATE
    # ------------------------------------------------------------------

    def _run_active_state(self, frame, now):
        dt = 1.0 / self.fps

        near_pos, raw_near_box, all_player_boxes = self._track_players(frame)

        # ---- Apply BoxSmoother to the raw near_box ----
        near_box = self._smooth_near_box(raw_near_box)
        self._log_telemetry(raw_near_box)   # log raw → smoother → CSV

        player_velocity = 0.0
        shape_change    = 0.0

        if near_pos:
            self.near_player_positions.append(near_pos)
            if near_box:
                self.near_player_boxes.append(near_box)
                nx1, ny1, nx2, ny2 = near_box
                cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)

            if len(self.near_player_positions) >= 5:
                old_p           = self.near_player_positions[0]
                new_p           = self.near_player_positions[-1]
                dist            = math.hypot(new_p[0] - old_p[0], new_p[1] - old_p[1])
                player_velocity = dist / len(self.near_player_positions)

            if len(self.near_player_boxes) >= 5:
                old_b   = self.near_player_boxes[0]
                new_b   = self.near_player_boxes[-1]
                old_w, old_h = old_b[2] - old_b[0], old_b[3] - old_b[1]
                new_w, new_h = new_b[2] - new_b[0], new_b[3] - new_b[1]
                shape_change = abs(new_w - old_w) + abs(new_h - old_h)

        candidates    = self._detect_all_balls(frame, all_player_boxes)
        best_ball     = self._select_best_ball(candidates, now)
        current_ball_pos = None

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

        if best_ball:
            proposed = (best_ball["world_x"], best_ball["world_y"], now)
            if self._validate_ball_jump(proposed, now):
                current_ball_pos         = proposed
                self.last_ball_seen_time  = now
                self.active_ball_positions.append(current_ball_pos)
            else:
                best_ball = None

        time_since_last_ball = now - self.last_ball_seen_time
        ball_velocity_fts    = self._compute_stable_velocity()

        cv2.putText(frame, f"BALL VEL: {ball_velocity_fts:.1f} ft/s", (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame,
                    f"DETS: {len(candidates)}  DEAD REFS: {len(self.dead_ball_refs)}",
                    (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        ball_delta   = 0.0
        player_delta = 0.0
        status_notes = []

        player_is_active = (near_pos is not None and
                            player_velocity >= Config.PLAYER_WALK_VELOCITY_THRESHOLD)

        player_in_action_zone = False
        net_proximity_factor  = 0.0
        if near_pos and self.H is not None:
            _, pwy = self.get_world_pos(near_pos[0], near_pos[1])
            player_in_action_zone = pwy > Config.ACTION_ZONE_MAX_Y_FT
            net_proximity_factor  = max(0.0, min(1.0, pwy / Config.NET_PROXIMITY_COURT_DEPTH_FT))

        if current_ball_pos and ball_velocity_fts > Config.MIN_BALL_VELOCITY_FT_SEC:
            ball_delta += Config.ENERGY_BOOST_BALL_FAST * dt
            status_notes.append("BALL: FLYING")
        elif current_ball_pos:
            ball_delta -= Config.ENERGY_DECAY_BALL_ROLLING * dt
            status_notes.append("BALL: ROLLING")
        elif time_since_last_ball > 0.25:
            if player_in_action_zone:
                ball_delta -= Config.ENERGY_DECAY_BALL_ACTION_ZONE * dt
                status_notes.append("BALL: OCCLUDED (player in court)")
            elif player_is_active:
                ball_delta -= Config.ENERGY_DECAY_BALL_OCCLUDED * dt
                status_notes.append("BALL: OCCLUDED (player at baseline)")
            else:
                ball_delta -= Config.ENERGY_DECAY_BALL_DEAD * dt
                status_notes.append("BALL: LIKELY DEAD")

        walking_gait = self._detect_walking_gait(near_box)

        if not near_pos:
            player_delta -= Config.ENERGY_DECAY_PLAYER_MISSING * dt
            status_notes.append("PLAYER: OFF SCREEN")
        elif player_velocity > Config.PLAYER_SPRINT_VELOCITY_THRESHOLD:
            player_delta += Config.ENERGY_BOOST_PLAYER_SPRINT * dt
            status_notes.append("PLAYER: SPRINTING")
        elif shape_change > Config.SHAPE_CHANGE_THRESHOLD_PX:
            player_delta += Config.ENERGY_BOOST_PLAYER_ACTION * dt
            status_notes.append("PLAYER: ACTIVE (SWING/STEP)")
        elif walking_gait:
            player_delta -= Config.ENERGY_DECAY_PLAYER_WALKING_GAIT * dt
            status_notes.append("PLAYER: WALKING (GAIT)")
        elif player_velocity < Config.PLAYER_WALK_VELOCITY_THRESHOLD:
            player_delta -= Config.ENERGY_DECAY_PLAYER_WALK * dt
            status_notes.append("PLAYER: SLOW/STOPPED")

        # Scale player energy UP and ball energy DOWN as player approaches net
        player_scale  = 1.0 + net_proximity_factor * (Config.NET_PROXIMITY_PLAYER_SCALE - 1.0)
        ball_scale    = 1.0 - net_proximity_factor * (1.0 - Config.NET_PROXIMITY_BALL_SCALE)
        player_delta *= player_scale
        ball_delta   *= ball_scale

        self.point_energy = max(0.0, min(1.0, self.point_energy + ball_delta + player_delta))

        cv2.putText(frame, "STATUS: ACTIVE", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        bar_w   = 200
        cur_bar = int(bar_w * self.point_energy)
        bar_col = (0, 255, 0) if self.point_energy > 0.4 else (0, 165, 255)
        cv2.rectangle(frame, (20, 130), (20 + bar_w, 150), (100, 100, 100), -1)
        cv2.rectangle(frame, (20, 130), (20 + cur_bar, 150), bar_col, -1)
        cv2.putText(frame, f"ENERGY: {self.point_energy:.2f}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_off = 180
        for note in status_notes:
            cv2.putText(frame, note, (20, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_off += 25

        timeout   = (Config.ABSOLUTE_BALL_LOST_TIMEOUT_ACTIVE
                     if (player_in_action_zone or player_is_active)
                     else Config.ABSOLUTE_BALL_LOST_TIMEOUT_IDLE)
        force_kill = time_since_last_ball > timeout

        if self.point_energy <= 0.0 or force_kill:
            zone   = ("in_court" if player_in_action_zone
                      else ("active" if player_is_active else "idle"))
            reason = ("Energy Depleted" if not force_kill
                      else f"Ball Missing > {timeout:.1f}s (player {zone})")
            print(f"\n[TRANSITION] ACTIVE -> END. Point dead ({reason}).")

            if self.current_segment_start is not None:
                self.active_segments.append(
                    (self.current_segment_start, self.frame_counter))
                self.current_segment_start = None

            next_state = SystemState.WAITING
            if near_pos and self.H is not None:
                px_ft, py_ft = self.get_world_pos(near_pos[0], near_pos[1])
                if py_ft < 0 and Config.READY_MIN_DIST_FT <= abs(py_ft) <= Config.READY_MAX_DIST_FT:
                    next_state = SystemState.ARMED
                    print("[BYPASS] Player already at baseline. Jumping to ARMED.")

            self.state = next_state
            self.near_player_positions.clear()
            self.active_ball_positions.clear()
            self.near_ready_start_time = None
            self.dead_ball_refs.clear()
            self.gait_y_buffer.clear()

        # Ghost-state check (emergency override to ARMED)
        if self.point_energy < 0.5 and near_pos and self.H is not None:
            px_ft, py_ft = self.get_world_pos(near_pos[0], near_pos[1])
            dist_ft = abs(py_ft)
            in_band = py_ft < 0 and Config.READY_MIN_DIST_FT <= dist_ft <= Config.READY_MAX_DIST_FT

            if in_band and player_velocity < Config.PLAYER_WALK_VELOCITY_THRESHOLD and near_box:
                nx1, ny1, nx2, ny2 = near_box
                pw, ph = nx2 - nx1, ny2 - ny1
                fh, fw = frame.shape[:2]
                pad_x  = int(pw * Config.DEFAULT_TROPHY_PAD)
                pad_y  = int(ph * Config.DEFAULT_TROPHY_PAD)
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
                    self.point_energy = 1.0
                    self.dead_ball_refs.clear()
                    self.gait_y_buffer.clear()
                    self.near_side.trophy_scores.clear()
                    self.near_side.toss_scores.clear()
                    self.near_side.add_trophy_score(best_trophy_score, now)

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

        if near_box and self.H is not None:
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
        # ---- Background capture (first frame of each ARMED entry) ----
        k = Config.TOSS_DIFF_BLUR_KERNEL
        curr_gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (k, k), 0
        )
        if self._armed_bg_gray is None:
            self._armed_bg_gray = curr_gray.copy()
            print(f"[DIFF] Background captured at {now:.3f}s for toss differencing.")

        diff_frame = cv2.absdiff(curr_gray, self._armed_bg_gray)

        near_pos, raw_near_box, _ = self._track_players(frame)

        # ---- Apply BoxSmoother ----
        near_box = self._smooth_near_box(raw_near_box)
        self._log_telemetry(raw_near_box)

        cv2.putText(frame, "STATUS: ARMED", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        in_band = False
        if near_box and self.H is not None:
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
                    self.toss_min_y_px = None
                    self.toss_consecutive_frames = 0
                    self.toss_gap_frames = 0
                    self.toss_ball_above_head_detected = False
                    self._armed_bg_gray = None
                    self.racquet_box_smoother.reset()

                    return

        if near_box and in_band:
            nx1, ny1, nx2, ny2 = near_box
            pw, ph = nx2 - nx1, ny2 - ny1
            fh, fw = frame.shape[:2]

            # Trophy detection
            pad_x  = int(pw * Config.DEFAULT_TROPHY_PAD)
            pad_y  = int(ph * Config.DEFAULT_TROPHY_PAD)
            tx1, ty1 = max(0, nx1 - pad_x), max(0, ny1 - pad_y)
            tx2, ty2 = min(fw, nx2 + pad_x), min(fh, ny2 + pad_y)

            trophy_crop = frame[ty1:ty2, tx1:tx2]
            if trophy_crop.size > 0:
                trophy_res = self.trophy_model(trophy_crop, verbose=False,
                                               imgsz=Config.TROPHY_IMGSZ)
                best_trophy_score = 0.0

                if (trophy_res and hasattr(trophy_res[0], "probs") and
                        trophy_res[0].probs is not None):
                    probs = trophy_res[0].probs
                    idx   = Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX
                    if idx < len(probs.data):
                        conf = float(probs.data[idx])
                        best_trophy_score = conf
                        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 255), 2)
                        cv2.putText(frame, f"POSE CLS: {conf:.2f}",
                                    (tx1, ty1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                elif (trophy_res and hasattr(trophy_res[0], "boxes") and
                      trophy_res[0].boxes is not None):
                    for b in trophy_res[0].boxes:
                        dcls = int(b.cls[0])
                        conf = float(b.conf[0])
                        cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
                        mx1, my1 = int(tx1 + cx1), int(ty1 + cy1)
                        mx2, my2 = int(tx1 + cx2), int(ty1 + cy2)
                        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 0, 255), 2)
                        if dcls == Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX:
                            if conf > best_trophy_score:
                                best_trophy_score = conf

                if best_trophy_score > 0:
                    self.near_side.add_trophy_score(best_trophy_score, now)

            # Ball toss detection
            cx_box = nx1 + pw / 2.0
            toss_w = pw * 2
            rx1 = max(0, int(cx_box - toss_w / 2.0))
            rx2 = min(fw, int(cx_box + toss_w / 2.0))
            ry1 = max(0, int(ny1 - ph))
            ry2 = min(fh, int(ny1 + ph / 2))

            if Config.DEFAULT_DRAW_TOSS_ROI:
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "TOSS ROI", (rx1, ry1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            toss_crop = frame[ry1:ry2, rx1:rx2]
            current_toss_score = 0.0

            if toss_crop.size > 0:
                ball_res   = self.ball_model(toss_crop, verbose=False,
                                              conf=Config.DEFAULT_BALL_CONF_MIN,
                                              imgsz=Config.TOSS_BALL_IMGSZ)
                best_ball  = None
                fh, fw = frame.shape[:2]
                if ball_res and ball_res[0].boxes is not None:
                    for b in ball_res[0].boxes:
                        if int(b.cls[0]) != Config.DEFAULT_BALL_CLASS_INDEX:
                            continue
                        cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
                        mx1 = int(rx1 + cx1); my1 = int(ry1 + cy1)
                        mx2 = int(rx1 + cx2); my2 = int(ry1 + cy2)
                        cy_full = (my1 + my2) / 2.0
                        conf    = float(b.conf[0])

                        # ---- Frame differencing gate ----
                        # Reject if the candidate region hasn't changed meaningfully
                        # from the background captured at ARMED entry — filters out
                        # static trees, sun glare, and other fixed outdoor elements.
                        rx1c = max(0, mx1); ry1c = max(0, my1)
                        rx2c = min(fw, mx2); ry2c = min(fh, my2)
                        diff_region = diff_frame[ry1c:ry2c, rx1c:rx2c]
                        if diff_region.size > 0:
                            mean_diff = float(diff_region.mean())
                            if mean_diff < Config.TOSS_DIFF_MIN_MEAN:
                                cv2.rectangle(frame, (mx1, my1), (mx2, my2),
                                              (0, 0, 200), 1)
                                cv2.putText(frame, f"REJ d={mean_diff:.0f}",
                                            (mx1, my1 - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.35, (0, 0, 200), 1)
                                continue

                        if best_ball is None or conf > best_ball["conf"]:
                            best_ball = {
                                "y": cy_full, "conf": conf,
                                "box": (mx1, my1, mx2, my2)
                            }

                # NEW TOSS SCORING LOGIC: Consecutive frame tracking with hysteresis
                is_moving_upward = False
                is_ball_above_head = False

                if best_ball:
                    # Check for upward motion
                    if self.last_toss_ball:
                        dy = best_ball["y"] - self.last_toss_ball["y"]
                        dt_t = now - self.last_toss_ball["time"]
                        if dy < 0 and dt_t > 0:
                            is_moving_upward = True

                    # Ball must be strictly above the player YOLO box top (ny1).
                    # The toss ROI extends below ny1, so this check is required —
                    # without it a ball held at shoulder/chest height falsely
                    # triggers the toss counter.
                    is_ball_above_head = best_ball["y"] < ny1

                    # Only track the peak height when the ball is genuinely above head
                    if is_ball_above_head:
                        if self.toss_min_y_px is None or best_ball["y"] < self.toss_min_y_px:
                            self.toss_min_y_px = best_ball["y"]

                    # Visual feedback: colour-code by above/below player box top
                    box_colour = (0, 255, 128) if is_ball_above_head else (0, 100, 255)
                    bx1, by1, bx2, by2 = best_ball["box"]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_colour, 2)
                    cv2.putText(frame,
                                f"{'ABOVE' if is_ball_above_head else 'BELOW'} "
                                f"{best_ball['conf']:.2f}",
                                (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, box_colour, 1)

                    self.last_toss_ball = {"y": best_ball["y"], "time": now}
                else:
                    self.last_toss_ball = None

                # Update consecutive frame counter with hysteresis.
                # Both conditions must hold: ball is moving upward AND above the
                # player box top — a rising ball still inside the box doesn't count.
                if is_moving_upward and is_ball_above_head:
                    # Reset gap counter and increment consecutive frames
                    self.toss_gap_frames = 0
                    self.toss_consecutive_frames += 1
                    self.toss_ball_above_head_detected = True
                else:
                    # No upward motion detected
                    self.toss_gap_frames += 1
                    # If gap exceeds 3 frames, reset consecutive counter
                    if self.toss_gap_frames > 3:
                        self.toss_consecutive_frames = 0
                        self.toss_ball_above_head_detected = False

                # Assign toss_score based on consecutive frames and height
                if self.toss_ball_above_head_detected:
                    if self.toss_consecutive_frames >= 3:
                        current_toss_score = 1.0
                    elif self.toss_consecutive_frames >= 2:
                        current_toss_score = 0.7
                    else:
                        current_toss_score = 0.0
                else:
                    current_toss_score = 0.0

            if current_toss_score > 0:
                self.near_side.add_toss_score(current_toss_score, now)

            max_trophy = max([d.score for d in self.near_side.trophy_scores] + [0.0])
            max_toss   = max([d.score for d in self.near_side.toss_scores]   + [0.0])
            serve_score = 0.2 * max_trophy + 0.8 * max_toss
            """
            if max_trophy > 0.5 or max_toss > 0.5:
                serve_score = 1.0
            """

            cv2.putText(frame, f"SERVE SCORE: {serve_score:.2f}", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAX TROPHY:  {max_trophy:.2f}", (20, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
            cv2.putText(frame, f"MAX TOSS:    {max_toss:.2f}", (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if serve_score >= Config.TRANSITION_SCORE_THRESHOLD:
                # Validate toss goes above player box top
                toss_height_valid = (self.toss_min_y_px is None or
                                     self.toss_min_y_px < ny1)

                if not toss_height_valid:
                    print(f"[DEBUG] Toss height invalid: min_y={self.toss_min_y_px:.1f} "
                          f"must be < player_top={ny1}")
                    self.toss_min_y_px = None
                    return

                toss_height_str = (f"{self.toss_min_y_px:.1f}px (above {ny1})"
                                   if self.toss_min_y_px is not None else "N/A")
                print(f"[TRANSITION] ARMED -> ACTIVE. "
                      f"Serve detected! Score: {serve_score:.2f} "
                      f"Toss height: {toss_height_str}")

                self._log_serve_event()

                self.state = SystemState.ACTIVE
                buffer_frames = int(self.fps * 1.0)
                self.current_segment_start = max(0, self.frame_counter - buffer_frames)

                _, _, snap_players = self._track_players(frame)
                self.dead_ball_refs = self._snapshot_dead_balls(frame, snap_players)

                self.near_side.trophy_scores.clear()
                self.near_side.toss_scores.clear()
                self.last_toss_ball = None
                self.toss_min_y_px = None
                self.toss_consecutive_frames = 0
                self.toss_gap_frames = 0
                self.toss_ball_above_head_detected = False
                self._armed_bg_gray = None
                self.racquet_box_smoother.reset()

                self.near_player_positions.clear()
                self.active_ball_positions.clear()
                self.last_ball_seen_time = now
                self.active_start_time   = now
                self.point_energy        = 1.0
                self.gait_y_buffer.clear()



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
