"""
full_anya.py
============
Unified near + far serve detector.

Both players run independent WAITING/ARMED state machines simultaneously.
When either triggers a serve, the system enters a shared ACTIVE state tracked
by a single ActiveEngine.  After the point ends both players reset to WAITING.

A ServingSideFilter requires ≥ 8 consecutive serves on a side before confirming
a side switch; detections before confirmation are emitted (Option A bootstrap).

Usage
-----
  python -m src.ai.full_anya video.mp4
  python -m src.ai.full_anya video.mp4 --output highlights.mp4 --headless
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.ai.utilities import (
    BoxSmoother, Config, _is_in_exclusion_zone,
    init_court, init_far_player_roi,
    create_auto_exclusion_zones, get_exclusion_zones_from_frames,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Near-side
NEAR_ACTIVE_BALL_CONF      = 0.10
NEAR_TOSS_BALL_CONF        = 0.05
NEAR_TOSS_BALL_IMGSZ       = 320
NEAR_ACTIVE_BALL_IMGSZ     = 960
NEAR_PLAYER_MISSING_GRACE  = 5

# Far-side
FAR_ACTIVE_BALL_CONF       = 0.10
FAR_TOSS_BALL_CONF         = 0.05
FAR_TOSS_BALL_IMGSZ        = 640
FAR_ACTIVE_BALL_IMGSZ      = 960
FAR_PLAYER_PERSIST_FRAMES  = 20
FAR_PLAYER_MISSING_GRACE   = 15
NET_OCCLUDE_TOLERANCE_PX   = 25

# Shared
MIN_SERVES_TO_CONFIRM      = 8
GAP_THRESHOLD_SEC          = 240.0
HIGHLIGHT_END_PAD_SEC      = 1.0
HIGHLIGHT_START_PAD_SEC    = 0.5
RACQUET_CLASS_ID           = 38
RACQUET_CONF               = 0.15  # A) lower threshold catches partial/blurred racquet views
RACQUET_CROP_PAD           = 40    # C) wider crop so extended-arm racquets are still in frame
YELLOW_HSV_LOWER           = (20, 100, 100)  # D) HSV yellow range for racquet color filter
YELLOW_HSV_UPPER           = (35, 255, 255)
YELLOW_PIXEL_THRESHOLD     = 0.25  # D) reject ball if ≥25% of its box is yellow


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry frame
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FullTelemetryFrame:
    frame_id:      int
    timestamp:     float
    system_state:  str          # "IDLE" | "ACTIVE"
    active_player: str          # "near" | "far" | "joint" | ""

    # Near player
    near_player_box:           Optional[Tuple[int, int, int, int]] = None
    near_player_world:         Optional[Tuple[float, float]]       = None
    near_toss_ball_candidates: List[dict] = field(default_factory=list)
    near_z_box:                Optional[Tuple[int, int, int, int]] = None
    near_trophy_score:         float = 0.0

    # Far player
    far_player_box:            Optional[Tuple[int, int, int, int]] = None
    far_player_world:          Optional[Tuple[float, float]]       = None
    far_toss_ball_candidates:  List[dict] = field(default_factory=list)
    far_z_box:                 Optional[Tuple[int, int, int, int]] = None
    far_trophy_score:          float = 0.0
    far_mhi_toss_score:        float = 0.0

    # Racquet boxes
    near_racquet_box:           Optional[Tuple[int, int, int, int]] = None
    far_racquet_box:            Optional[Tuple[int, int, int, int]] = None

    # Walking classifier
    near_walking_confirmed:     bool = False

    # Serve classifiers
    near_serve_confirmed:       bool = False
    far_serve_confirmed:        bool = False

    # Shared ACTIVE
    active_ball_candidates:    List[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Provider
# ─────────────────────────────────────────────────────────────────────────────

class FullTelemetryProvider:
    """
    Sensor layer for both near and far players on the same frame.

    Tracks system_state / near_state / far_state internally so that each call
    to process_frame() runs only the detectors relevant to the current states.
    Call update_state() after the engine evaluates transitions.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._init_video_props()

        # ── Models ─────────────────────────────────────────────────────────
        self.player_model  = YOLO("yolo26n.pt")
        self.ball_model    = YOLO("weights/ball/weights/best.pt")
        self.near_trophy   = YOLO(Config.DEFAULT_NEAR_TROPHY_MODEL_PATH)
        self.far_trophy    = YOLO(Config.DEFAULT_FAR_TROPHY_MODEL_PATH)

        # ── Court geometry ─────────────────────────────────────────────────
        mid_frame = self.total_frames // 2 if self.total_frames > 0 else 300
        self.court_vertices, self.frame_shape = init_court(
            self.video_path, target_idx=mid_frame, analysis_size=(960, 540)
        )
        self.far_player_roi = init_far_player_roi(
            self.video_path, analysis_size=(960, 540)
        )
        self.H     = self._compute_homography()
        self.H_inv = np.linalg.inv(self.H)
        self.net_y_px = self._compute_net_y_px()
        print(f"[FULL] Net pixel-y: {self.net_y_px:.1f}px")

        # ── Active-zone polygon (shared by both sides) ────────────────────
        _vdir  = os.path.dirname(os.path.abspath(self.video_path))
        _vstem = os.path.splitext(os.path.basename(self.video_path))[0]
        _zone_cache = os.path.join(_vdir, f"{_vstem}_active_zone_config.json")
        self.near_active_zone = self._get_or_define_zone(_zone_cache, "active zone")
        self.far_active_zone  = self.near_active_zone

        # ── Static exclusion zones ─────────────────────────────────────────
        print("\n[FULL] Scanning for static exclusion zones...")
        try:
            self.static_exclusion_zones = create_auto_exclusion_zones(
                self.video_path, self.ball_model,
                num_frames=50, conf=0.04, eps=12, padding=5,
                ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                analysis_size=(960, 540),
            )
            print(f"[FULL] {len(self.static_exclusion_zones)} static zone(s)")
        except Exception as e:
            print(f"[FULL] WARN: static exclusion zones failed: {e}")
            self.static_exclusion_zones = []

        self.near_dynamic_exclusion_zones: List = []
        self.far_dynamic_exclusion_zones:  List = []

        # ── ARMED dynamic zone buffering (per side) ────────────────────────
        self._near_armed_buf:        List            = []
        self._near_armed_entry_time: Optional[float] = None
        self._near_armed_done:       bool            = False
        self._far_armed_buf:         List            = []
        self._far_armed_entry_time:  Optional[float] = None
        self._far_armed_done:        bool            = False
        self.ARMED_DYNAMIC_SEC       = 0.5
        self.ARMED_DYNAMIC_SAMPLES   = 5

        # ── State ──────────────────────────────────────────────────────────
        self.system_state  = "IDLE"
        self.near_state    = "WAITING"
        self.far_state     = "WAITING"
        self.active_player = ""
        self.frame_counter = 0
        buffer_size = int(self.fps * Config.TELEMETRY_BUFFER_SECONDS)
        self.telemetry_history: deque = deque(maxlen=buffer_size)

        # ── Far-player persistence ─────────────────────────────────────────
        self._last_far_box:   Optional[Tuple[int, int, int, int]] = None
        self._last_far_world: Optional[Tuple[float, float]]       = None
        self._far_persist_ctr: int = 0
        self._far_box_heights: deque = deque(maxlen=30)
        self._far_smoother = BoxSmoother(alpha_pos=0.50, alpha_size=0.12)

        # ── Near-player cache ──────────────────────────────────────────────
        self._last_near_box:   Optional[Tuple[int, int, int, int]] = None
        self._last_near_world: Optional[Tuple[float, float]]       = None

        # ── ACTIVE player striding ─────────────────────────────────────────
        self.ACTIVE_PLAYER_STRIDE = 4
        self._cached_players: Tuple = (None, None, None, None)  # near_box,near_world,far_box,far_world

        # ── Trophy stride ──────────────────────────────────────────────────
        self.ARMED_TROPHY_STRIDE  = 2
        self._last_near_trophy:   float = 0.0
        self._last_far_trophy:    float = 0.0

        # ── MHI (far player toss fallback) ─────────────────────────────────
        self.MHI_BUFFER_FRAMES = 15
        self._mhi_roi_buf:     deque = deque(maxlen=self.MHI_BUFFER_FRAMES)
        self._mhi_last_score:  float = 0.0

        # ── Walking detector ───────────────────────────────────────────────
        _ai_dir = os.path.dirname(os.path.abspath(__file__))
        self.walking_detector = WalkingDetector(
            os.path.join(_ai_dir, "walking_lstm.pt"))

        # ── Serve detectors (near and far share same model weights) ───────
        self.serve_detector     = ServeDetector(os.path.join(_ai_dir, "serve_lstm.pt"))
        self.far_serve_detector = ServeDetector(os.path.join(_ai_dir, "serve_lstm.pt"))

        # ── Ball-trace timestamp (pushed from main loop via update_state) ──
        # Tracks when ball trace was last seen; used to skip player YOLO
        # while the ball is actively being tracked.
        self._last_active_trace_time: float = 0.0
        self.NO_TRACE_PLAYER_RESUME_SEC = 0.5  # re-run YOLO this many s after trace lost

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def far_baseline_strip(self) -> Tuple[float, float, float, float]:
        BL, BR, TR, TL = self.court_vertices
        x1 = float(min(TL[0], TR[0]))
        x2 = float(max(TL[0], TR[0]))
        y_baseline = (TL[1] + TR[1]) / 2.0
        return (x1, y_baseline - 50.0, x2, y_baseline)

    @property
    def exclusion_zones(self) -> List:
        return (self.static_exclusion_zones
                + self.near_dynamic_exclusion_zones
                + self.far_dynamic_exclusion_zones)

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _init_video_props(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width  = 960
        self.height = 540
        cap.release()

    def _compute_homography(self):
        BL, BR, TR, TL = self.court_vertices
        dst = np.array([
            [0, 0],
            [Config.COURT_WIDTH_FT, 0],
            [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT],
            [0,                     Config.COURT_LENGTH_FT],
        ], dtype=np.float32)
        src = np.array([BL, BR, TR, TL], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H

    def _compute_net_y_px(self) -> float:
        net_world = np.array(
            [[[Config.COURT_WIDTH_FT / 2.0, Config.COURT_LENGTH_FT / 2.0]]],
            dtype=np.float32,
        )
        net_px = cv2.perspectiveTransform(net_world, self.H_inv)
        return float(net_px[0][0][1])

    def get_world_pos(self, px_x: float, px_y: float) -> Tuple[float, float]:
        pt = np.array([[[px_x, px_y]]], dtype=np.float32)
        w  = cv2.perspectiveTransform(pt, self.H)
        return float(w[0][0][0]), float(w[0][0][1])

    def _get_or_define_zone(self, cache_path: str, label: str) -> np.ndarray:
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    pts = json.load(f)
                print(f"[FULL] Loaded {label} zone from {cache_path}")
                return np.array(pts, dtype=np.int32)
            except Exception as e:
                print(f"[FULL] WARN: could not load {cache_path}: {e}")
        print(f"[FULL] Define {label} active zone. Click 8 points. Press q to confirm.")
        pts = self._interactive_polygon(label)
        with open(cache_path, "w") as f:
            json.dump(pts.tolist(), f)
        return pts

    def _interactive_polygon(self, label: str) -> np.ndarray:
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read frame for polygon.")
        frame   = cv2.resize(frame, (960, 540))
        display = frame.copy()
        pts: List[Tuple[int, int]] = []

        def cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 8:
                pts.append((x, y))
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                if len(pts) > 1:
                    cv2.line(display, pts[-2], pts[-1], (0, 255, 0), 2)
                if len(pts) == 8:
                    cv2.line(display, pts[-1], pts[0], (0, 255, 0), 2)
                cv2.imshow(label, display)

        cv2.namedWindow(label)
        cv2.setMouseCallback(label, cb)
        while True:
            cv2.imshow(label, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27) and len(pts) == 8:
                break
        cv2.destroyWindow(label)
        return np.array(pts, dtype=np.int32)

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _is_in_near_active_zone(self, cx: float, cy: float) -> bool:
        return cv2.pointPolygonTest(self.near_active_zone, (cx, cy), False) >= 0

    def _is_in_far_active_zone(self, cx: float, cy: float) -> bool:
        return cv2.pointPolygonTest(self.far_active_zone, (cx, cy), False) >= 0

    def _is_in_player_box(self, bx, by, box, padding: int = 10) -> bool:
        if box is None:
            return False
        x1, y1, x2, y2 = box
        return (x1 - padding <= bx <= x2 + padding and
                y1 - padding <= by <= y2 + padding)

    def _estimate_far_feet_y(self, box: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        bh = y2 - y1
        if bh > 0:
            self._far_box_heights.append(bh)
        occluded = abs(y2 - self.net_y_px) < NET_OCCLUDE_TOLERANCE_PX or y2 > self.net_y_px
        if occluded and self._far_box_heights:
            med = sorted(self._far_box_heights)[len(self._far_box_heights) // 2]
            return float(y1 + med)
        return float(y2)

    def _create_z_box(self, player_box, far: bool = False):
        if player_box is None:
            return None
        x1, y1, x2, y2 = player_box
        pw, ph   = x2 - x1, y2 - y1
        pcx, pcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        z_w = pw * 2.0
        z_h = ph * (2.5 if far else 1.5)
        zx1 = int(pcx - z_w / 2.0)
        zx2 = int(pcx + z_w / 2.0)
        zy2 = int(pcy)
        zy1 = max(0, int(zy2 - z_h))
        return (zx1, zy1, zx2, zy2)

    def _is_in_z_box(self, bx, by, z_box) -> bool:
        if z_box is None:
            return False
        x1, y1, x2, y2 = z_box
        return x1 <= bx <= x2 and y1 <= by <= y2

    def _detect_racquet(self, frame, player_box) -> Optional[Tuple[int, int, int, int]]:
        """Detect racquet in a crop around player_box. Only accepts detections
        whose center falls within player_box + RACQUET_CROP_PAD."""
        if player_box is None:
            return None
        x1, y1, x2, y2 = player_box
        fh, fw = frame.shape[:2]
        rx1 = max(0, x1 - RACQUET_CROP_PAD)
        ry1 = max(0, y1 - RACQUET_CROP_PAD)
        rx2 = min(fw, x2 + RACQUET_CROP_PAD)
        ry2 = min(fh, y2 + RACQUET_CROP_PAD)
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None
        results = self.player_model(roi, verbose=False, conf=RACQUET_CONF,
                                    imgsz=Config.PLAYER_IMGSZ)
        if not (results and results[0].boxes):
            return None
        best_conf, best_box = -1.0, None
        for b in results[0].boxes:
            if int(b.cls[0]) != RACQUET_CLASS_ID:
                continue
            conf = float(b.conf[0])
            lx1, ly1, lx2, ly2 = map(int, b.xyxy[0].tolist())
            gx1, gy1 = rx1 + lx1, ry1 + ly1
            gx2, gy2 = rx1 + lx2, ry1 + ly2
            gcx = (gx1 + gx2) / 2.0
            gcy = (gy1 + gy2) / 2.0
            if self._is_in_player_box(gcx, gcy, player_box, RACQUET_CROP_PAD):
                if conf > best_conf:
                    best_conf = conf
                    best_box  = (gx1, gy1, gx2, gy2)
        return best_box

    def _is_yellow_region(self, frame, box) -> bool:
        """D) Returns True if ≥ YELLOW_PIXEL_THRESHOLD of pixels in box are yellow."""
        bx1, by1, bx2, by2 = map(int, box)
        fh, fw = frame.shape[:2]
        bx1 = max(0, bx1); by1 = max(0, by1)
        bx2 = min(fw, bx2); by2 = min(fh, by2)
        if bx2 <= bx1 or by2 <= by1:
            return False
        roi = frame[by1:by2, bx1:bx2]
        if roi.size == 0:
            return False
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array(YELLOW_HSV_LOWER, dtype=np.uint8),
                           np.array(YELLOW_HSV_UPPER, dtype=np.uint8))
        return float(np.count_nonzero(mask)) / mask.size >= YELLOW_PIXEL_THRESHOLD

    # ── Player tracking ───────────────────────────────────────────────────────

    def _track_far_player_strip(self, frame) -> Optional[Tuple[int, int, int, int]]:
        fh, fw = frame.shape[:2]
        bsx1, bsy1, bsx2, bsy2 = self.far_baseline_strip
        rx1 = int(max(0, bsx1))
        ry1 = int(max(0, bsy1))
        rx2 = int(min(fw, bsx2))
        ry2 = int(min(fh, bsy2 + 10))
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None
        results = self.player_model(roi, verbose=False, conf=0.5,
                                    imgsz=Config.FAR_PLAYER_IMGSZ)
        if not (results and results[0].boxes):
            return None
        best_conf, best_box = -1.0, None
        for b in results[0].boxes:
            if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                continue
            lx1, ly1, lx2, ly2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box  = (rx1 + lx1, ry1 + ly1, rx1 + lx2, ry1 + ly2)
        return best_box

    def _track_players(self, frame):
        """Returns (near_box, near_world, far_box, far_world)."""
        results = self.player_model(frame, verbose=False, conf=0.5,
                                    imgsz=Config.PLAYER_IMGSZ)
        candidates = []
        if results and results[0].boxes:
            for b in results[0].boxes:
                if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cx = (x1 + x2) / 2.0
                wx, wy = self.get_world_pos(cx, float(y2))
                candidates.append((x1, y1, x2, y2, wx, wy))

        pad = Config.NEAR_PLAYER_X_PAD_FT
        near_cands = [
            c for c in candidates
            if (abs(c[5]) < abs(c[5] - Config.COURT_LENGTH_FT) and
                -pad <= c[4] <= Config.COURT_WIDTH_FT + pad)
        ]
        near_box = near_world = None
        if near_cands:
            nc = min(near_cands, key=lambda c: abs(c[5]))
            near_box   = nc[:4]
            near_world = (nc[4], nc[5])

        far_box = far_world = None
        far_roi = self._track_far_player_strip(frame)
        if far_roi is not None:
            est_fy = self._estimate_far_feet_y(far_roi)
            fx1, fy1, fx2, fy2 = far_roi
            fcx = (fx1 + fx2) / 2.0
            wx, wy = self.get_world_pos(fcx, est_fy)
            sx1, sy1, sx2, sy2 = self._far_smoother.smooth_box_xyxy(fx1, fy1, fx2, fy2)
            far_box   = (sx1, sy1, sx2, sy2)
            far_world = (wx, wy)

        return near_box, near_world, far_box, far_world

    # ── MHI (far toss fallback) ───────────────────────────────────────────────

    def _compute_mhi_toss_score(self, frame, far_box) -> float:
        if far_box is None:
            self._mhi_roi_buf.clear()
            return 0.0
        x1, y1, x2, y2 = far_box
        fh, fw = frame.shape[:2]
        ph = y2 - y1
        rx1 = max(0, x1);       rx2 = min(fw, x2)
        ry1 = max(0, y1 - ph);  ry2 = max(0, y1)
        if rx2 <= rx1 or ry2 <= ry1:
            self._mhi_roi_buf.clear()
            return 0.0
        roi_g = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
        roi_g = cv2.GaussianBlur(roi_g, (5, 5), 0)
        self._mhi_roi_buf.append(roi_g)
        if len(self._mhi_roi_buf) < 3:
            return self._mhi_last_score
        ref  = self._mhi_roi_buf[0]
        curr = self._mhi_roi_buf[-1]
        if ref.shape != curr.shape:
            self._mhi_roi_buf.clear()
            return 0.0
        diff  = cv2.absdiff(curr, ref)
        score = float(np.mean(diff)) / 255.0
        MHI_LOW, MHI_HIGH = 0.02, 0.10
        normalized = max(0.0, min(1.0, (score - MHI_LOW) / (MHI_HIGH - MHI_LOW)))
        self._mhi_last_score = normalized
        return normalized

    # ── Main frame processor ──────────────────────────────────────────────────

    def process_frame(self, frame) -> FullTelemetryFrame:
        self.frame_counter += 1
        ts = self.frame_counter / self.fps

        tel = FullTelemetryFrame(
            frame_id=self.frame_counter,
            timestamp=ts,
            system_state=self.system_state,
            active_player=self.active_player,
        )

        # ── 1. Player tracking ────────────────────────────────────────────
        # During ACTIVE state, skip YOLO entirely while the ball trace is
        # fresh.  Once the trace has been absent for NO_TRACE_PLAYER_RESUME_SEC
        # the cached box may be stale, so resume strided detection so it is
        # up-to-date before the 1.5 s walking-exit check can fire.
        time_since_trace = ts - self._last_active_trace_time
        trace_fresh = (self.system_state == "ACTIVE" and
                       time_since_trace < self.NO_TRACE_PLAYER_RESUME_SEC)
        use_cache = (
            (trace_fresh or
             (self.system_state == "ACTIVE"
              and self.frame_counter % self.ACTIVE_PLAYER_STRIDE != 0))
            and self._cached_players[0] is not None
        )
        if use_cache:
            near_box, near_world, far_box, far_world = self._cached_players
        else:
            near_box, near_world, far_box, far_world = self._track_players(frame)

            # Far player persistence
            if far_box is not None:
                self._last_far_box    = far_box
                self._last_far_world  = far_world
                self._far_persist_ctr = 0
            else:
                self._far_persist_ctr += 1
                if self._far_persist_ctr <= FAR_PLAYER_PERSIST_FRAMES:
                    far_box   = self._last_far_box
                    far_world = self._last_far_world

            # Near player cache
            if near_box is not None:
                self._last_near_box   = near_box
                self._last_near_world = near_world
            self._cached_players = (near_box, near_world, far_box, far_world)

        tel.near_player_box   = near_box   if near_box   is not None else self._last_near_box
        tel.near_player_world = near_world if near_world is not None else self._last_near_world
        tel.far_player_box    = far_box
        tel.far_player_world  = far_world

        fh, fw = frame.shape[:2]
        """
        # ── 1.5 Racquet detection (ARMED and ACTIVE) ──────────────────────
        if self.system_state == "ACTIVE":
            tel.near_racquet_box = self._detect_racquet(frame, tel.near_player_box)
            tel.far_racquet_box  = self._detect_racquet(frame, tel.far_player_box)
        elif self.system_state == "IDLE":
            if self.near_state == "ARMED":
                tel.near_racquet_box = self._detect_racquet(frame, tel.near_player_box)
            if self.far_state == "ARMED":
                tel.far_racquet_box  = self._detect_racquet(frame, tel.far_player_box)
        """
        # ── 2. IDLE: armed-side detectors ────────────────────────────────
        if self.system_state == "IDLE":
            now_t = ts

            # ── Near ARMED detectors ──────────────────────────────────────
            if self.near_state == "ARMED":
                self._tick_near_armed_buffer(frame, now_t)
                if tel.near_player_box is not None:
                    nx1, ny1, nx2, ny2 = tel.near_player_box
                    pw, ph = nx2 - nx1, ny2 - ny1
                    z_box = self._create_z_box(tel.near_player_box, far=False)
                    tel.near_z_box = z_box
                    # Trophy
                    if self.frame_counter % self.ARMED_TROPHY_STRIDE == 0:
                        pad_x = int(pw * Config.DEFAULT_TROPHY_PAD)
                        pad_y = int(ph * Config.DEFAULT_TROPHY_PAD)
                        tx1 = max(0, nx1 - pad_x); ty1 = max(0, ny1 - pad_y)
                        tx2 = min(fw, nx2 + pad_x); ty2 = min(fh, ny2 + pad_y)
                        crop = frame[ty1:ty2, tx1:tx2]
                        if crop.size > 0:
                            tr = self.near_trophy(crop, verbose=False,
                                                  imgsz=Config.TROPHY_IMGSZ)
                            if tr and hasattr(tr[0], "probs") and tr[0].probs is not None:
                                idx = Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX
                                if idx < len(tr[0].probs.data):
                                    self._last_near_trophy = float(tr[0].probs.data[idx])
                    tel.near_trophy_score = self._last_near_trophy
                    # Toss ball ROI (1× height above head)
                    rx1 = max(0, int(nx1 - pw / 2))
                    ry1 = max(0, int(ny1 - ph))
                    rx2 = min(fw, int(nx2 + pw / 2))
                    ry2 = min(fh, int(ny1 + ph / 2))
                    roi = frame[ry1:ry2, rx1:rx2]
                    if roi.size > 0:
                        br = self.ball_model(roi, verbose=False,
                                             conf=NEAR_TOSS_BALL_CONF,
                                             imgsz=NEAR_TOSS_BALL_IMGSZ)
                        if br and br[0].boxes:
                            for b in br[0].boxes:
                                cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
                                bcx = rx1 + (cx1 + cx2) / 2.0
                                bcy = ry1 + (cy1 + cy2) / 2.0
                                full_box = (rx1 + cx1, ry1 + cy1, rx1 + cx2, ry1 + cy2)
                                if (self._is_in_z_box(bcx, bcy, z_box) and
                                        not _is_in_exclusion_zone(bcx, bcy, self.exclusion_zones) and
                                        not self._is_in_player_box(bcx, bcy, tel.near_player_box, 15) and
                                        not self._is_in_player_box(bcx, bcy, tel.near_racquet_box, 25) and  # B)
                                        not self._is_yellow_region(frame, full_box)):                        # D)
                                    tel.near_toss_ball_candidates.append({
                                        "box":  full_box,
                                        "conf": float(b.conf[0]),
                                    })

            # ── Far ARMED detectors ───────────────────────────────────────
            if self.far_state == "ARMED":
                self._tick_far_armed_buffer(frame, now_t)
                far_box_det = tel.far_player_box
                if far_box_det is not None:
                    fx1, fy1, fx2, fy2 = far_box_det
                    pw, ph = fx2 - fx1, fy2 - fy1
                    z_box = self._create_z_box(far_box_det, far=True)
                    tel.far_z_box = z_box
                    # Trophy
                    if self.frame_counter % self.ARMED_TROPHY_STRIDE == 0:
                        pad_x = int(pw * Config.DEFAULT_TROPHY_PAD)
                        pad_y = int(ph * Config.DEFAULT_TROPHY_PAD)
                        tx1 = max(0, fx1 - pad_x); ty1 = max(0, fy1 - pad_y)
                        tx2 = min(fw, fx2 + pad_x); ty2 = min(fh, fy2 + pad_y)
                        crop = frame[ty1:ty2, tx1:tx2]
                        if crop.size > 0:
                            tr = self.far_trophy(crop, verbose=False,
                                                 imgsz=Config.TROPHY_IMGSZ)
                            if tr and hasattr(tr[0], "probs") and tr[0].probs is not None:
                                idx = Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX
                                if idx < len(tr[0].probs.data):
                                    self._last_far_trophy = float(tr[0].probs.data[idx])
                    tel.far_trophy_score = self._last_far_trophy
                    # MHI
                    tel.far_mhi_toss_score = self._compute_mhi_toss_score(frame, far_box_det)
                    # Toss ball ROI (1.5× height above head)
                    rx1 = max(0,  int(fx1 - pw / 2))
                    ry1 = max(0,  int(fy1 - 1.5 * ph))
                    rx2 = min(fw, int(fx2 + pw / 2))
                    ry2 = min(fh, int(fy1 + ph / 2))
                    roi = frame[ry1:ry2, rx1:rx2]
                    if roi.size > 0:
                        br = self.ball_model(roi, verbose=False,
                                             conf=FAR_TOSS_BALL_CONF,
                                             imgsz=FAR_TOSS_BALL_IMGSZ)
                        if br and br[0].boxes:
                            for b in br[0].boxes:
                                cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
                                bcx = rx1 + (cx1 + cx2) / 2.0
                                bcy = ry1 + (cy1 + cy2) / 2.0
                                full_box = (rx1 + cx1, ry1 + cy1, rx1 + cx2, ry1 + cy2)
                                if (self._is_in_z_box(bcx, bcy, z_box) and
                                        not _is_in_exclusion_zone(bcx, bcy, self.exclusion_zones) and
                                        not self._is_in_player_box(bcx, bcy, far_box_det, 15) and
                                        not self._is_in_player_box(bcx, bcy, tel.far_racquet_box, 15) and  # B)
                                        not self._is_yellow_region(frame, full_box)):                       # D)
                                    tel.far_toss_ball_candidates.append({
                                        "box":  full_box,
                                        "conf": float(b.conf[0]),
                                    })
                else:
                    tel.far_mhi_toss_score = self._compute_mhi_toss_score(frame, None)

        # ── 3. ACTIVE: full-frame ball detection ──────────────────────────
        if self.system_state == "ACTIVE":
            ap = self.active_player
            conf   = FAR_ACTIVE_BALL_CONF  if ap == "far"  else NEAR_ACTIVE_BALL_CONF
            imgsz  = FAR_ACTIVE_BALL_IMGSZ if ap == "far"  else NEAR_ACTIVE_BALL_IMGSZ
            br = self.ball_model(frame, verbose=False, conf=conf, imgsz=imgsz)
            if br and br[0].boxes:
                nb = tel.near_player_box
                fb = tel.far_player_box
                for b in br[0].boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    if ap == "far":
                        in_zone = self._is_in_far_active_zone(bcx, bcy)
                    elif ap == "near":
                        in_zone = self._is_in_near_active_zone(bcx, bcy)
                    else:  # joint
                        in_zone = (self._is_in_near_active_zone(bcx, bcy) or
                                   self._is_in_far_active_zone(bcx, bcy))
                    if (in_zone and not _is_in_exclusion_zone(bcx, bcy, self.exclusion_zones) and not self._is_yellow_region(frame, (bx1, by1, bx2, by2))):           # D)
                        """                    
                            not self._is_in_player_box(bcx, bcy, nb, 10) and
                            not self._is_in_player_box(bcx, bcy, fb, 10) and
                            not self._is_in_player_box(bcx, bcy, tel.near_racquet_box, 25) and  # B)
                            not self._is_in_player_box(bcx, bcy, tel.far_racquet_box, 15) and   # B)
                        """    
                        tel.active_ball_candidates.append({
                            "box":          (bx1, by1, bx2, by2),
                            "conf":         float(b.conf[0]),
                            "pixel_center": (bcx, bcy),
                        })

        # ── 4. ACTIVE: walking detection ─────────────────────────────────
        if self.system_state == "ACTIVE":
            tel.near_walking_confirmed = self.walking_detector.update(
                frame, tel.near_player_box
            )

        # ── 5. IDLE/ARMED: serve detection ───────────────────────────────
        if self.system_state == "IDLE" and self.near_state == "ARMED":
            if self.serve_detector._ready:
                tel.near_serve_confirmed = self.serve_detector.update(
                    frame, tel.near_player_box
                )
            else:
                tel.near_serve_confirmed = True
        else:
            if self.near_state != "ARMED":
                self.serve_detector.reset()

        if self.system_state == "IDLE" and self.far_state == "ARMED":
            if self.far_serve_detector._ready:
                tel.far_serve_confirmed = self.far_serve_detector.update(
                    frame, tel.far_player_box
                )
            else:
                tel.far_serve_confirmed = True
        else:
            if self.far_state != "ARMED":
                self.far_serve_detector.reset()

        self.telemetry_history.append(tel)
        return tel

    # ── Dynamic exclusion zone helpers ────────────────────────────────────────

    def _tick_near_armed_buffer(self, frame, now_t: float):
        if self._near_armed_done or self._near_armed_entry_time is None:
            return
        elapsed = now_t - self._near_armed_entry_time
        if elapsed <= self.ARMED_DYNAMIC_SEC:
            self._near_armed_buf.append(frame.copy())
        elif self._near_armed_buf:
            self.near_dynamic_exclusion_zones = get_exclusion_zones_from_frames(
                self._near_armed_buf, self.ball_model,
                sample_size=self.ARMED_DYNAMIC_SAMPLES,
                conf=0.05, eps=5, min_samples=15, padding=5,
                ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
            )
            self._near_armed_done = True
            self._near_armed_buf  = []
            print(f"[FULL] Near dynamic exclusion: {len(self.near_dynamic_exclusion_zones)} zone(s)")

    def _tick_far_armed_buffer(self, frame, now_t: float):
        if self._far_armed_done or self._far_armed_entry_time is None:
            return
        elapsed = now_t - self._far_armed_entry_time
        if elapsed <= self.ARMED_DYNAMIC_SEC:
            self._far_armed_buf.append(frame.copy())
        elif self._far_armed_buf:
            self.far_dynamic_exclusion_zones = get_exclusion_zones_from_frames(
                self._far_armed_buf, self.ball_model,
                sample_size=self.ARMED_DYNAMIC_SAMPLES,
                conf=0.05, eps=5, min_samples=15, padding=5,
                ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
            )
            self._far_armed_done = True
            self._far_armed_buf  = []
            print(f"[FULL] Far dynamic exclusion: {len(self.far_dynamic_exclusion_zones)} zone(s)")

    # ── State update (called by main loop after engine evaluation) ────────────

    def update_state(self, system_state: str, near_state: str, far_state: str,
                     active_player: str, last_active_trace_time: float = 0.0):
        old_sys  = self.system_state
        old_near = self.near_state
        old_far  = self.far_state
        self._last_active_trace_time = last_active_trace_time

        self.system_state  = system_state
        self.near_state    = near_state
        self.far_state     = far_state
        self.active_player = active_player

        now_t = self.frame_counter / self.fps

        # Near ARMED entry
        if near_state == "ARMED" and old_near != "ARMED":
            self.near_dynamic_exclusion_zones = []
            self._near_armed_buf   = []
            self._near_armed_entry_time = now_t
            self._near_armed_done  = False
            self._last_near_trophy = 0.0
            print("[FULL] Near ARMED — starting dynamic exclusion collection")

        # Far ARMED entry
        if far_state == "ARMED" and old_far != "ARMED":
            self.far_dynamic_exclusion_zones = []
            self._far_armed_buf    = []
            self._far_armed_entry_time = now_t
            self._far_armed_done   = False
            self._last_far_trophy  = 0.0
            self._mhi_roi_buf.clear()
            self._mhi_last_score   = 0.0
            print("[FULL] Far ARMED — starting dynamic exclusion collection")

        # ACTIVE exit
        if old_sys == "ACTIVE" and system_state == "IDLE":
            self._last_far_box    = None
            self._last_far_world  = None
            self._far_persist_ctr = 0
            self._far_smoother.reset()
            self._cached_players  = (None, None, None, None)
            self.walking_detector.reset()


# ─────────────────────────────────────────────────────────────────────────────
# NearSubEngine  (WAITING / ARMED only — returns TRIGGER_ACTIVE)
# ─────────────────────────────────────────────────────────────────────────────

class NearSubEngine:
    """
    Manages WAITING ↔ ARMED for the near player.
    Returns "TRIGGER_ACTIVE" when a serve is detected (instead of "ACTIVE"),
    so the FullTransitionEngine decides the system-level transition.
    Logic is verbatim from anya_transitions.TransitionEngine.
    """

    def __init__(self, fps: float):
        self.fps = fps

        # WAITING
        self.READY_MIN_DIST_FT   = -0.5
        self.READY_MAX_DIST_FT   =  3.5
        self.READY_WAIT_TIME_SEC =  0.4

        # ARMED band monitor
        self.ARMED_BAND_WINDOW_SEC     = 2.0
        self.ARMED_OUT_RATIO_THRESHOLD = 0.25
        self.armed_band_history: deque = deque()

        # ARMED serve scoring
        self.TRANSITION_SCORE_THRESHOLD = 0.55
        self.EVENT_WINDOW_SECONDS       = 1.2
        self._trophy_scores: deque = deque()
        self._toss_scores:   deque = deque()

        # Toss state
        self.toss_consecutive_frames:       int             = 0
        self.toss_gap_frames:               int             = 0
        self.toss_ball_above_head_detected: bool            = False
        self.toss_min_y_px:                 Optional[float] = None
        self.last_toss_ball:                Optional[dict]  = None

        self.near_ready_start_time: Optional[float] = None

        self.last_serve_scores = {
            "trophy_score": 0.0,
            "toss_score":   0.0,
            "serve_score":  0.0,
        }
        self.state = "WAITING"

    def tick(self, history: deque) -> str:
        if not history:
            return self.state
        if self.state == "WAITING":
            self.state = self._check_waiting(history)
        elif self.state == "ARMED":
            self.state = self._check_armed(history)
        return self.state

    def _check_waiting(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        if frame.near_player_world is None:
            self.near_ready_start_time = None
            return "WAITING"

        _, wy   = frame.near_player_world
        dist_ft = abs(wy)
        in_zone = wy < 0 and self.READY_MIN_DIST_FT <= dist_ft <= self.READY_MAX_DIST_FT

        if in_zone:
            if self.near_ready_start_time is None:
                self.near_ready_start_time = now
            if now - self.near_ready_start_time > self.READY_WAIT_TIME_SEC:
                print(f"[NEAR] WAITING -> ARMED. Held {now - self.near_ready_start_time:.1f}s.")
                self.near_ready_start_time = None
                return "ARMED"
        else:
            self.near_ready_start_time = None
        return "WAITING"

    def _check_armed(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        in_band = False
        if frame.near_player_world is not None:
            _, wy   = frame.near_player_world
            dist_ft = abs(wy)
            in_band = wy < 0 and self.READY_MIN_DIST_FT <= dist_ft <= self.READY_MAX_DIST_FT

        self.armed_band_history.append((now, in_band))
        while (self.armed_band_history and
               now - self.armed_band_history[0][0] > self.ARMED_BAND_WINDOW_SEC):
            self.armed_band_history.popleft()

        if len(self.armed_band_history) > 1:
            total_time = self.armed_band_history[-1][0] - self.armed_band_history[0][0]
            if total_time > 1.0:
                time_out = sum(
                    self.armed_band_history[i + 1][0] - self.armed_band_history[i][0]
                    for i in range(len(self.armed_band_history) - 1)
                    if not self.armed_band_history[i][1]
                )
                if time_out / total_time > self.ARMED_OUT_RATIO_THRESHOLD:
                    print(f"[NEAR] ARMED -> WAITING. Out of band {time_out/total_time:.0%}.")
                    self._reset_armed_state()
                    return "WAITING"

        if not in_band or frame.near_player_box is None:
            return "ARMED"

        nx1, ny1, nx2, ny2 = frame.near_player_box
        trophy = getattr(frame, "near_trophy_score", 0.0) or 0.0
        if trophy > 0:
            self._trophy_scores.append((trophy, now))
        toss = self._update_toss_detection(frame, ny1, now)
        if toss > 0:
            self._toss_scores.append((toss, now))

        for buf in (self._trophy_scores, self._toss_scores):
            while buf and now - buf[0][1] > self.EVENT_WINDOW_SECONDS:
                buf.popleft()

        max_trophy  = max((s for s, _ in self._trophy_scores), default=0.0)
        max_toss    = max((s for s, _ in self._toss_scores),   default=0.0)
        serve_score = 0.2 * max_trophy + 0.8 * max_toss

        self.last_serve_scores = {
            "trophy_score": max_trophy,
            "toss_score":   max_toss,
            "serve_score":  serve_score,
        }

        if serve_score >= self.TRANSITION_SCORE_THRESHOLD:
            if self.toss_min_y_px is not None and self.toss_min_y_px >= ny1:
                self.toss_min_y_px = None
                return "ARMED"
            # Gate: require LSTM confirmation when the model is available.
            # near_serve_confirmed is True when model is absent (transparent).
            if not frame.near_serve_confirmed:
                return "ARMED"
            print(f"[NEAR] ARMED -> TRIGGER_ACTIVE. Score={serve_score:.2f}")
            self._reset_armed_state()
            return "TRIGGER_ACTIVE"

        return "ARMED"

    def _update_toss_detection(self, frame, ny1: float, now: float) -> float:
        if not frame.near_toss_ball_candidates:
            self.last_toss_ball   = None
            self.toss_gap_frames += 1
            if self.toss_gap_frames > 3:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False
            return 0.0

        best = max(frame.near_toss_ball_candidates, key=lambda x: x["conf"])
        bx1, by1, bx2, by2 = best["box"]
        cy = (by1 + by2) / 2.0

        is_moving_upward   = False
        is_ball_above_head = cy < ny1
        if self.last_toss_ball is not None:
            if cy - self.last_toss_ball["y"] < 0:
                is_moving_upward = True
        if is_ball_above_head and (self.toss_min_y_px is None or cy < self.toss_min_y_px):
            self.toss_min_y_px = cy
        self.last_toss_ball = {"y": cy, "time": now}

        if is_moving_upward and is_ball_above_head:
            self.toss_gap_frames              = 0
            self.toss_consecutive_frames     += 1
            self.toss_ball_above_head_detected = True
        else:
            self.toss_gap_frames += 1
            if self.toss_gap_frames > 3:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False

        if not self.toss_ball_above_head_detected:
            return 0.0
        if self.toss_consecutive_frames >= 3:
            return 1.0
        if self.toss_consecutive_frames >= 2:
            return 0.7
        return 0.0

    def _reset_armed_state(self):
        self.armed_band_history.clear()
        self.toss_consecutive_frames       = 0
        self.toss_gap_frames               = 0
        self.toss_ball_above_head_detected = False
        self.toss_min_y_px                 = None
        self.last_toss_ball                = None
        self._trophy_scores.clear()
        self._toss_scores.clear()
        self.last_serve_scores = {"trophy_score": 0.0, "toss_score": 0.0, "serve_score": 0.0}

    def reset_to_waiting(self):
        self._reset_armed_state()
        self.near_ready_start_time = None
        self.state = "WAITING"


# ─────────────────────────────────────────────────────────────────────────────
# FarSubEngine  (WAITING / ARMED only — returns TRIGGER_ACTIVE)
# ─────────────────────────────────────────────────────────────────────────────

class FarSubEngine:
    """
    Manages WAITING ↔ ARMED for the far player.
    Returns "TRIGGER_ACTIVE" on serve detection.
    Logic is verbatim from far_anya.FarTransitionEngine.
    """

    def __init__(self, fps: float, far_baseline_strip: Tuple[float, float, float, float]):
        self.fps              = fps
        self.far_baseline_strip = far_baseline_strip

        # WAITING / ARMED movement monitor
        self.MOVEMENT_WINDOW_SEC  = 2.0
        self.READY_MAX_LATERAL_FT = 3.0
        self.ARMED_MAX_TOTAL_FT   = 3.0
        self._movement_history: deque = deque()

        # ARMED serve scoring
        self.TRANSITION_SCORE_THRESHOLD = 0.55
        self.EVENT_WINDOW_SECONDS       = 1.2
        self.TROPHY_WEIGHT  = 0.05
        self.TOSS_WEIGHT    = 0.95
        self.MHI_THRESHOLD        = 0.30
        self.MHI_MAX_CONTRIBUTION = 0.50

        self._trophy_scores: deque = deque()
        self._toss_scores:   deque = deque()

        # Toss state
        self.toss_consecutive_frames:       int             = 0
        self.toss_gap_frames:               int             = 0
        self.toss_ball_above_head_detected: bool            = False
        self.toss_min_y_px:                 Optional[float] = None
        self.last_toss_ball:                Optional[dict]  = None
        self.TOSS_ARC_WINDOW_SEC:           float           = 1.5
        self.TOSS_ARC_MIN_POINTS:           int             = 3
        self.TOSS_ARC_R2_THRESHOLD:         float           = 0.80
        self._toss_arc_buffer:              deque           = deque()

        self.last_serve_scores = {
            "trophy_score": 0.0,
            "toss_score":   0.0,
            "mhi_score":    0.0,
            "serve_score":  0.0,
        }
        self.state = "WAITING"

    def tick(self, history: deque) -> str:
        if not history:
            return self.state
        if self.state == "WAITING":
            self.state = self._check_waiting(history)
        elif self.state == "ARMED":
            self.state = self._check_armed(history)
        return self.state

    def _check_waiting(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        if frame.far_player_world is not None:
            wx, wy = frame.far_player_world
            self._movement_history.append((now, wx, wy))
        while (self._movement_history and
               now - self._movement_history[0][0] > self.MOVEMENT_WINDOW_SEC):
            self._movement_history.popleft()

        centre_in_strip = False
        if frame.far_player_box is not None:
            fx1, fy1, fx2, fy2 = frame.far_player_box
            fcx = (fx1 + fx2) / 2.0
            fcy = (fy1 + fy2) / 2.0
            sx1, sy1, sx2, sy2 = self.far_baseline_strip
            centre_in_strip = sx1 <= fcx <= sx2 and sy1 <= fcy <= sy2

        lateral_travel = sum(
            abs(self._movement_history[i][1] - self._movement_history[i - 1][1])
            for i in range(1, len(self._movement_history))
        )

        if centre_in_strip and lateral_travel < self.READY_MAX_LATERAL_FT:
            print(f"[FAR] WAITING -> ARMED. lateral={lateral_travel:.2f}ft.")
            self._movement_history.clear()
            return "ARMED"
        return "WAITING"

    def _check_armed(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        if frame.far_player_world is not None:
            wx, wy = frame.far_player_world
            self._movement_history.append((now, wx, wy))
        while (self._movement_history and
               now - self._movement_history[0][0] > self.MOVEMENT_WINDOW_SEC):
            self._movement_history.popleft()

        total_travel = sum(
            math.hypot(
                self._movement_history[i][1] - self._movement_history[i - 1][1],
                self._movement_history[i][2] - self._movement_history[i - 1][2],
            )
            for i in range(1, len(self._movement_history))
        )
        if total_travel > self.ARMED_MAX_TOTAL_FT:
            print(f"[FAR] ARMED -> WAITING. total_travel={total_travel:.2f}ft.")
            self._reset_armed_state()
            return "WAITING"

        if frame.far_player_box is None:
            return "ARMED"

        fx1, fy1, fx2, fy2 = frame.far_player_box

        trophy = getattr(frame, "far_trophy_score", 0.0) or 0.0
        if trophy > 0:
            self._trophy_scores.append((trophy, now))

        yolo_toss = self._update_toss_detection(frame, fy1, now)
        if yolo_toss > 0:
            self._toss_scores.append((yolo_toss, now))

        mhi = getattr(frame, "far_mhi_toss_score", 0.0)
        if mhi > self.MHI_THRESHOLD:
            self._toss_scores.append((mhi * self.MHI_MAX_CONTRIBUTION, now))

        for buf in (self._trophy_scores, self._toss_scores):
            while buf and now - buf[0][1] > self.EVENT_WINDOW_SECONDS:
                buf.popleft()

        max_trophy  = max((s for s, _ in self._trophy_scores), default=0.0)
        max_toss    = max((s for s, _ in self._toss_scores),   default=0.0)
        serve_score = self.TROPHY_WEIGHT * max_trophy + self.TOSS_WEIGHT * max_toss

        self.last_serve_scores = {
            "trophy_score": max_trophy,
            "toss_score":   max_toss,
            "mhi_score":    mhi,
            "serve_score":  serve_score,
        }

        if serve_score >= self.TRANSITION_SCORE_THRESHOLD:
            if self.toss_min_y_px is not None and self.toss_min_y_px >= fy1:
                self.toss_min_y_px = None
                return "ARMED"
            # Gate: require LSTM confirmation when the model is available.
            # far_serve_confirmed is True when model is absent (transparent).
            if not frame.far_serve_confirmed:
                return "ARMED"
            print(f"[FAR] ARMED -> TRIGGER_ACTIVE. Score={serve_score:.2f}")
            self._reset_armed_state()
            return "TRIGGER_ACTIVE"

        return "ARMED"

    def _update_toss_detection(self, frame, fy1: float, now: float) -> float:
        if not frame.far_toss_ball_candidates:
            self.last_toss_ball   = None
            self.toss_gap_frames += 1
            if self.toss_gap_frames > 3:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False
            return 0.0

        best = max(frame.far_toss_ball_candidates, key=lambda x: x["conf"])
        bx1, by1, bx2, by2 = best["box"]
        cy = (by1 + by2) / 2.0

        is_moving_upward   = False
        is_ball_above_head = cy < fy1
        if self.last_toss_ball is not None:
            if cy - self.last_toss_ball["y"] < 0:
                is_moving_upward = True
        if is_ball_above_head:
            if self.toss_min_y_px is None or cy < self.toss_min_y_px:
                self.toss_min_y_px = cy
            self._toss_arc_buffer.append((now, cy))

        while (self._toss_arc_buffer and
               now - self._toss_arc_buffer[0][0] > self.TOSS_ARC_WINDOW_SEC):
            self._toss_arc_buffer.popleft()

        self.last_toss_ball = {"y": cy, "time": now}

        if is_moving_upward and is_ball_above_head:
            self.toss_gap_frames              = 0
            self.toss_consecutive_frames     += 1
            self.toss_ball_above_head_detected = True
        else:
            self.toss_gap_frames += 1
            if self.toss_gap_frames > 3:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False

        consecutive_score = 0.0
        if self.toss_ball_above_head_detected:
            if self.toss_consecutive_frames >= 2:
                consecutive_score = 1.0
            elif self.toss_consecutive_frames >= 1:
                consecutive_score = 0.7

        arc_score = 0.0
        if len(self._toss_arc_buffer) >= self.TOSS_ARC_MIN_POINTS:
            arc_score = self._score_toss_arc()

        return max(consecutive_score, arc_score)

    def _score_toss_arc(self) -> float:
        pts    = list(self._toss_arc_buffer)
        ts     = np.array([p[0] for p in pts], dtype=np.float64)
        cys    = np.array([p[1] for p in pts], dtype=np.float64)
        t0, t1 = ts[0], ts[-1]
        if t1 - t0 < 1e-6:
            return 0.0
        ts_n   = (ts - t0) / (t1 - t0)
        coeffs = np.polyfit(ts_n, cys, 2)
        if coeffs[0] >= 0:
            return 0.0
        pred   = np.polyval(coeffs, ts_n)
        ss_res = np.sum((cys - pred) ** 2)
        ss_tot = np.sum((cys - cys.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        if r2 >= self.TOSS_ARC_R2_THRESHOLD:
            return 1.0
        if r2 >= 0.60:
            return 0.7
        return 0.0

    def _reset_armed_state(self):
        self._movement_history.clear()
        self.toss_consecutive_frames       = 0
        self.toss_gap_frames               = 0
        self.toss_ball_above_head_detected = False
        self.toss_min_y_px                 = None
        self.last_toss_ball                = None
        self._toss_arc_buffer.clear()
        self._trophy_scores.clear()
        self._toss_scores.clear()
        self.last_serve_scores = {
            "trophy_score": 0.0, "toss_score": 0.0,
            "mhi_score": 0.0,    "serve_score": 0.0,
        }

    def reset_to_waiting(self):
        self._reset_armed_state()
        self.state = "WAITING"


# ─────────────────────────────────────────────────────────────────────────────
# WalkingDetector  (LSTM-based near-player walking classifier)
# ─────────────────────────────────────────────────────────────────────────────

class WalkingDetector:
    """
    Runs a pre-trained 2-layer LSTM to classify whether the near-side player
    is walking.  Called once per frame during ACTIVE state with the cropped
    near-player region.

    Features match the training pipeline in train_walking_rnn.py:
      - 33 MediaPipe pose landmarks × (x, y) = 66 position features
      - 66 frame-to-frame velocity features (Δx, Δy per landmark)
      = 132 features per frame, over a 45-frame (≈1.5 s) rolling window.

    Returns False silently when the model file is absent or MediaPipe fails.
    """

    _NUM_LANDMARKS  = 33
    _POS_FEAT       = _NUM_LANDMARKS * 2   # 66
    _VEL_FEAT       = _NUM_LANDMARKS * 2   # 66
    _TOTAL_FEAT     = _POS_FEAT + _VEL_FEAT  # 132
    _WALK_THRESHOLD = 0.5

    def __init__(self, model_path: str):
        self._ready = False
        self._pose  = None
        self._model = None
        self._device = None

        # ── MediaPipe Tasks pose ───────────────────────────────────────────
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as _mpt
            from mediapipe.tasks.python import vision as _mpv
            opts = _mpv.PoseLandmarkerOptions(
                base_options=_mpt.BaseOptions(model_asset_path=model_path.replace(
                    "walking_lstm.pt", "pose_landmarker_full.task"
                )),
                running_mode=_mpv.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._pose   = _mpv.PoseLandmarker.create_from_options(opts)
            self._mp_img = mp.Image   # keep class reference
            self._mp_fmt = mp.ImageFormat.SRGB
        except Exception as e:
            print(f"[WALK] MediaPipe init failed — walking detector disabled: {e}")
            return

        # ── LSTM model ─────────────────────────────────────────────────────
        try:
            import torch
            import torch.nn as nn

            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

            class _LSTM(nn.Module):
                def __init__(self, inp, hid, layers, drop):
                    super().__init__()
                    self.lstm = nn.LSTM(inp, hid, layers, batch_first=True,
                                        dropout=drop if layers > 1 else 0.0)
                    self.drop = nn.Dropout(drop)
                    self.head = nn.Linear(hid, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.head(self.drop(out[:, -1, :])).squeeze(-1)

            m = _LSTM(ckpt["input_size"], ckpt["hidden_size"],
                      ckpt["num_layers"],  ckpt["dropout"])
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            self._model   = m
            self._seq_len = ckpt["seq_len"]
            self._device  = torch.device("cpu")
            self._torch   = torch
        except Exception as e:
            print(f"[WALK] LSTM load failed — walking detector disabled: {e}")
            return

        self._pos_buf: deque = deque(maxlen=self._seq_len)
        self._ready = True
        print(f"[WALK] Walking detector ready  (seq_len={self._seq_len}  "
              f"feat={self._TOTAL_FEAT}  model={model_path})")

    # ── per-frame inference ────────────────────────────────────────────────

    def update(self, frame_bgr, near_box) -> bool:
        """
        Feed one frame.  Returns True if the player is confirmed walking.
        Appends a zero vector when detection fails so the buffer stays full.
        """
        if not self._ready:
            return False

        pos = self._extract_pos(frame_bgr, near_box)
        self._pos_buf.append(pos)

        if len(self._pos_buf) < self._seq_len:
            return False   # window not filled yet

        return self._classify()

    def reset(self):
        if self._ready:
            self._pos_buf.clear()

    # ── internals ─────────────────────────────────────────────────────────

    def _extract_pos(self, frame_bgr, box) -> np.ndarray:
        zeros = np.zeros(self._POS_FEAT, dtype=np.float32)
        if box is None or self._pose is None:
            return zeros
        fh, fw = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            return zeros
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return zeros
        try:
            import cv2 as _cv2
            rgb = _cv2.cvtColor(crop, _cv2.COLOR_BGR2RGB)
            mp_img = self._mp_img(image_format=self._mp_fmt, data=rgb)
            res = self._pose.detect(mp_img)
            if not res.pose_landmarks:
                return zeros
            feats = []
            for lm in res.pose_landmarks[0]:
                feats.extend([lm.x, lm.y])
            return np.array(feats, dtype=np.float32)
        except Exception:
            return zeros

    def _classify(self) -> bool:
        pos_arr = np.stack(self._pos_buf)                  # (T, 66)
        vel_arr = np.zeros_like(pos_arr)
        vel_arr[1:] = pos_arr[1:] - pos_arr[:-1]
        no_det = (pos_arr == 0).all(axis=1)
        vel_arr[no_det] = 0.0
        seq = np.concatenate([pos_arr, vel_arr], axis=1)   # (T, 132)
        x   = self._torch.tensor(seq[np.newaxis], dtype=self._torch.float32)
        with self._torch.no_grad():
            prob = self._model(x).sigmoid().item()
        return prob >= self._WALK_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# ServeDetector  (LSTM-based near-player serve classifier)
# ─────────────────────────────────────────────────────────────────────────────

class ServeDetector:
    """
    Runs a pre-trained 2-layer LSTM to confirm whether the near-side player
    is executing a serve.  Fed one frame at a time during IDLE/ARMED state.

    Features match train_serve_rnn.py:
      66 position + 66 velocity = 132 features/frame over 60 frames (≈2 s).

    Returns False silently when the model file is absent.
    """

    _NUM_LANDMARKS  = 33
    _POS_FEAT       = _NUM_LANDMARKS * 2
    _VEL_FEAT       = _NUM_LANDMARKS * 2
    _TOTAL_FEAT     = _POS_FEAT + _VEL_FEAT   # 132
    _SERVE_THRESHOLD = 0.5

    def __init__(self, model_path: str):
        self._ready = False
        self._pose  = None
        self._model = None

        # ── MediaPipe ─────────────────────────────────────────────────────
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as _mpt
            from mediapipe.tasks.python import vision as _mpv
            pose_model = model_path.replace("serve_lstm.pt",
                                            "pose_landmarker_full.task")
            opts = _mpv.PoseLandmarkerOptions(
                base_options=_mpt.BaseOptions(model_asset_path=pose_model),
                running_mode=_mpv.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._pose   = _mpv.PoseLandmarker.create_from_options(opts)
            self._mp_img = mp.Image
            self._mp_fmt = mp.ImageFormat.SRGB
        except Exception as e:
            print(f"[SERVE] MediaPipe init failed — serve detector disabled: {e}")
            return

        # ── LSTM ──────────────────────────────────────────────────────────
        try:
            import torch
            import torch.nn as nn

            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

            class _LSTM(nn.Module):
                def __init__(self, inp, hid, layers, drop):
                    super().__init__()
                    self.lstm = nn.LSTM(inp, hid, layers, batch_first=True,
                                        dropout=drop if layers > 1 else 0.0)
                    self.drop = nn.Dropout(drop)
                    self.head = nn.Linear(hid, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.head(self.drop(out[:, -1, :])).squeeze(-1)

            m = _LSTM(ckpt["input_size"], ckpt["hidden_size"],
                      ckpt["num_layers"],  ckpt["dropout"])
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            self._model   = m
            self._seq_len = ckpt["seq_len"]
            self._torch   = torch
        except Exception as e:
            print(f"[SERVE] LSTM load failed — serve detector disabled: {e}")
            return

        self._pos_buf: deque = deque(maxlen=self._seq_len)
        self._ready = True
        print(f"[SERVE] Serve detector ready  (seq_len={self._seq_len}  "
              f"feat={self._TOTAL_FEAT}  model={model_path})")

    # ── per-frame update ───────────────────────────────────────────────────

    def update(self, frame_bgr, near_box) -> bool:
        """Feed one frame; returns True if a serve is confirmed."""
        if not self._ready:
            return False
        self._pos_buf.append(self._extract_pos(frame_bgr, near_box))
        if len(self._pos_buf) < self._seq_len:
            return False
        return self._classify()

    def reset(self):
        if self._ready:
            self._pos_buf.clear()

    # ── internals ─────────────────────────────────────────────────────────

    def _extract_pos(self, frame_bgr, box) -> np.ndarray:
        zeros = np.zeros(self._POS_FEAT, dtype=np.float32)
        if box is None or self._pose is None:
            return zeros
        fh, fw = frame_bgr.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            return zeros
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return zeros
        try:
            import cv2 as _cv2
            rgb    = _cv2.cvtColor(crop, _cv2.COLOR_BGR2RGB)
            mp_img = self._mp_img(image_format=self._mp_fmt, data=rgb)
            res    = self._pose.detect(mp_img)
            if not res.pose_landmarks:
                return zeros
            feats = []
            for lm in res.pose_landmarks[0]:
                feats.extend([lm.x, lm.y])
            return np.array(feats, dtype=np.float32)
        except Exception:
            return zeros

    def predict_proba(self, frame_bgr, near_box) -> float:
        """Feed one frame; returns raw sigmoid probability (0.0–1.0)."""
        if not self._ready:
            return 0.0
        self._pos_buf.append(self._extract_pos(frame_bgr, near_box))
        if len(self._pos_buf) < self._seq_len:
            return 0.0
        return self._proba()

    def _classify(self) -> bool:
        return self._proba() >= self._SERVE_THRESHOLD

    def _proba(self) -> float:
        pos_arr = np.stack(self._pos_buf)
        vel_arr = np.zeros_like(pos_arr)
        vel_arr[1:] = pos_arr[1:] - pos_arr[:-1]
        vel_arr[(pos_arr == 0).all(axis=1)] = 0.0
        seq = np.concatenate([pos_arr, vel_arr], axis=1)
        x   = self._torch.tensor(seq[np.newaxis], dtype=self._torch.float32)
        with self._torch.no_grad():
            return self._model(x).sigmoid().item()


# ─────────────────────────────────────────────────────────────────────────────
# ActiveEngine  (shared ACTIVE → IDLE logic)
# ─────────────────────────────────────────────────────────────────────────────

class ActiveEngine:
    """
    Shared ACTIVE state: ball-trace gate + energy-bar fallback.
    Always tracks the NEAR player for the energy bar.
    Returns "END_ACTIVE" when the point is over.
    """

    def __init__(self, fps: float):
        self.fps = fps

        self.BALL_HISTORY_SEC       = 1.5
        self.TRACE_NEARBY_PX        = 40.0
        self.TRACE_NEARBY_MIN_COUNT = 5

        self.ENERGY_BOOST_SPRINT        = 4.0
        self.ENERGY_BOOST_SWING         = 4.0
        self.ENERGY_DECAY_WALKING       = 0.5
        self.ENERGY_DECAY_MISSING       = 0.5
        self.ENERGY_DECAY_STILL         = 0.5
        self.PLAYER_SPRINT_VELOCITY_FTS = 7.0
        self.PLAYER_STILL_VELOCITY_FTS  = 2.0
        self.VELOCITY_WINDOW_SIZE       = 20
        self.ACTIVE_PLAYER_STRIDE       = 4

        self.NEAR_PLAYER_MISSING_GRACE  = NEAR_PLAYER_MISSING_GRACE
        self.FAR_PLAYER_MISSING_GRACE   = FAR_PLAYER_MISSING_GRACE
        self.PLAYER_EMA_ALPHA           = 0.25

        self.GAIT_BUFFER_FRAMES  = 45
        self.GAIT_MIN_REVERSALS  = 2
        self.GAIT_MAX_REVERSALS  = 8
        self.GAIT_MIN_DRIFT_PX   = 10.0
        self.SCREEN_HEIGHT_PX           = 540
        self.BOTTOM_SCREEN_TOLERANCE_PX = 8

        self._reset_active_state()
        self.last_transition_time: Optional[float] = None
        self.last_active_debug = {
            "time_since_trace": 0.0,
            "has_active_trace": False,
            "energy_bar_mode":  False,
            "point_energy":     1.0,
            "ball_count":       0,
        }

    def init_active(self, now: float, active_player: str):
        self._reset_active_state()
        self.active_start_time      = now
        self.last_active_trace_time = now
        self.last_transition_time   = None
        self._active_player         = active_player

    def tick(self, history: deque) -> str:
        return self._check_active(history)

    def _check_active(self, history: deque) -> str:
        frame      = history[-1]
        now        = frame.timestamp
        candidates = frame.active_ball_candidates or []
        cutoff     = now - self.BALL_HISTORY_SEC

        self._update_player_tracking(frame)

        for c in candidates:
            px, py = c["pixel_center"]
            nearby = sum(1 for _, hx, hy in self._all_ball_history
                         if math.hypot(px - hx, py - hy) < self.TRACE_NEARBY_PX)
            if nearby < self.TRACE_NEARBY_MIN_COUNT:
                self._trace_ball_history.append((now, px, py))
        for c in candidates:
            px, py = c["pixel_center"]
            self._all_ball_history.append((now, px, py))

        while self._all_ball_history   and self._all_ball_history[0][0]   < cutoff:
            self._all_ball_history.popleft()
        while self._trace_ball_history and self._trace_ball_history[0][0] < cutoff:
            self._trace_ball_history.popleft()

        has_active_trace = bool(self._trace_ball_history)

        self.last_active_debug = {
            "time_since_trace": now - self.last_active_trace_time,
            "has_active_trace": has_active_trace,
            "energy_bar_mode":  self.energy_bar_mode,
            "point_energy":     self.point_energy,
            "ball_count":       len(candidates),
        }

        if has_active_trace:
            self.last_active_trace_time = now
            if self.energy_bar_mode:
                print(f"[ACTIVE] Ball trace restored at t={now:.2f}s.")
                self.energy_bar_mode = False
                self.point_energy    = 1.0
                self._energy_player_positions.clear()
                self._energy_player_boxes.clear()
                self._energy_gait_y_buffer.clear()
            return "ACTIVE"

        # ── Walking-exit check ────────────────────────────────────────────
        # Once ball trace has been absent for ≥ 1.5 s, a confirmed walking
        # player, off-screen player, or player at the bottom edge of the
        # frame all indicate the point is over — end immediately.
        time_since_trace = now - self.last_active_trace_time
        if time_since_trace >= self.BALL_HISTORY_SEC:
            near_box = frame.near_player_box
            off_screen    = near_box is None                              # (b)
            at_bottom     = (near_box is not None and
                             near_box[3] >= self.SCREEN_HEIGHT_PX
                             - self.BOTTOM_SCREEN_TOLERANCE_PX)          # (c)
            ml_walking    = getattr(frame, "near_walking_confirmed", False)  # (a)

            if ml_walking or off_screen or at_bottom:
                reason = ("ML_WALKING"   if ml_walking  else
                          "OFF_SCREEN"   if off_screen  else
                          "AT_BOTTOM")
                self.last_transition_time = self.last_active_trace_time
                elapsed = now - self.active_start_time
                print(f"\n[ACTIVE] -> END_ACTIVE ({reason}, "
                      f"no-trace {time_since_trace:.1f}s). "
                      f"Lasted {elapsed:.1f}s. "
                      f"Rewind to t={self.last_active_trace_time:.2f}s.")
                self._reset_active_state()
                return "END_ACTIVE"

        if not self.energy_bar_mode:
            print(f"[ACTIVE] No trace. Energy bar starts (anchor={self.last_active_trace_time:.2f}s).")
            self.energy_bar_mode       = True
            self.energy_bar_start_time = self.last_active_trace_time
            self.point_energy          = 1.0

        dt = 1.0 / self.fps
        delta, status = self._compute_energy_delta(frame, dt)
        self.point_energy = max(0.0, min(1.0, self.point_energy + delta))
        self.last_active_debug.update({
            "energy_bar_mode": self.energy_bar_mode,
            "point_energy":    self.point_energy,
            "energy_status":   status,
        })

        if self.point_energy <= 0.0:
            self.last_transition_time = self.energy_bar_start_time
            elapsed = now - self.active_start_time
            print(f"\n[ACTIVE] -> END_ACTIVE (energy depleted [{status}]). "
                  f"Lasted {elapsed:.1f}s. Rewind to t={self.energy_bar_start_time:.2f}s.")
            self._reset_active_state()
            return "END_ACTIVE"

        return "ACTIVE"

    def _update_player_tracking(self, frame) -> None:
        """Always tracks the near player for the energy bar."""
        near_box   = frame.near_player_box
        near_world = frame.near_player_world
        if near_box is None or near_world is None:
            self._player_missing_frames += 1
            self._energy_gait_y_buffer.clear()
            return
        self._player_missing_frames = 0
        wx, wy = near_world
        if self._smoothed_player_world is None:
            self._smoothed_player_world = (wx, wy)
        else:
            α = self.PLAYER_EMA_ALPHA
            self._smoothed_player_world = (
                α * wx + (1 - α) * self._smoothed_player_world[0],
                α * wy + (1 - α) * self._smoothed_player_world[1],
            )
        self._energy_player_positions.append(self._smoothed_player_world)
        self._energy_player_boxes.append(near_box)
        self._energy_gait_y_buffer.append(float(near_box[3]))

    def _compute_energy_delta(self, frame, dt: float):
        grace = (FAR_PLAYER_MISSING_GRACE
                 if getattr(self, "_active_player", "") == "far"
                 else NEAR_PLAYER_MISSING_GRACE)
        if self._player_missing_frames > grace:
            return -(self.ENERGY_DECAY_MISSING * dt), "MISSING"

        near_box = frame.near_player_box
        if (near_box is not None and
                near_box[3] >= self.SCREEN_HEIGHT_PX - self.BOTTOM_SCREEN_TOLERANCE_PX):
            return -(self.ENERGY_DECAY_WALKING * dt), "WALKING_OFFSCREEN"

        if self._detect_walking_gait():
            return -(self.ENERGY_DECAY_WALKING * dt), "WALKING"

        vel = 0.0
        if len(self._energy_player_positions) >= 5:
            old_p = self._energy_player_positions[0]
            new_p = self._energy_player_positions[-1]
            dist  = math.hypot(new_p[0] - old_p[0], new_p[1] - old_p[1])
            elapsed = len(self._energy_player_positions) * self.ACTIVE_PLAYER_STRIDE / self.fps
            vel = dist / elapsed if elapsed > 0 else 0.0

        if vel > self.PLAYER_SPRINT_VELOCITY_FTS:
            return (self.ENERGY_BOOST_SPRINT * dt), f"SPRINTING {vel:.1f}ft/s"

        if len(self._energy_player_boxes) >= 5:
            ob, nb = self._energy_player_boxes[0], self._energy_player_boxes[-1]
            bh = ob[3] - ob[1]
            if bh > 0:
                dw = abs((nb[2] - nb[0]) - (ob[2] - ob[0]))
                dh = abs((nb[3] - nb[1]) - (ob[3] - ob[1]))
                if (dw + dh) / bh > 0.25:
                    return (self.ENERGY_BOOST_SWING * dt), "SWING"

        if vel < self.PLAYER_STILL_VELOCITY_FTS:
            return -(self.ENERGY_DECAY_STILL * dt), f"STILL {vel:.1f}ft/s"
        return (0.1 * dt), f"MOVING {vel:.1f}ft/s"

    def _detect_walking_gait(self) -> bool:
        ys = list(self._energy_gait_y_buffer)
        n  = len(ys)
        if n < self.GAIT_BUFFER_FRAMES * 0.6:
            return False
        if abs(ys[-1] - ys[0]) < self.GAIT_MIN_DRIFT_PX:
            return False
        residuals = [ys[i] - (ys[0] + (ys[-1] - ys[0]) * (i / (n - 1))) for i in range(n)]
        reversals = prev_dir = 0
        for i in range(1, len(residuals)):
            d = residuals[i] - residuals[i - 1]
            if abs(d) < 0.5:
                continue
            direction = 1 if d > 0 else -1
            if prev_dir != 0 and direction != prev_dir:
                reversals += 1
            prev_dir = direction
        return self.GAIT_MIN_REVERSALS <= reversals <= self.GAIT_MAX_REVERSALS

    def _reset_active_state(self):
        self._all_ball_history:      deque = deque()
        self._trace_ball_history:    deque = deque()
        self.active_start_time       = 0.0
        self.last_active_trace_time  = 0.0
        self._player_missing_frames  = 0
        self.energy_bar_mode         = False
        self.energy_bar_start_time   = 0.0
        self.point_energy            = 1.0
        self._energy_player_positions: deque = deque(maxlen=self.VELOCITY_WINDOW_SIZE)
        self._energy_player_boxes:     deque = deque(maxlen=5)
        self._energy_gait_y_buffer:    deque = deque(maxlen=self.GAIT_BUFFER_FRAMES)
        self._smoothed_player_world: Optional[Tuple[float, float]] = None
        self._active_player          = ""


# ─────────────────────────────────────────────────────────────────────────────
# FullTransitionEngine
# ─────────────────────────────────────────────────────────────────────────────

class FullTransitionEngine:
    """
    Orchestrates NearSubEngine, FarSubEngine, and ActiveEngine.

    Returns (system_state, near_state, far_state, active_player) each tick.
    """

    def __init__(self, fps: float, far_baseline_strip: Tuple[float, float, float, float]):
        self.near_engine   = NearSubEngine(fps)
        self.far_engine    = FarSubEngine(fps, far_baseline_strip)
        self.active_engine = ActiveEngine(fps)

        self.system_state  = "IDLE"
        self.near_state    = "WAITING"
        self.far_state     = "WAITING"
        self.active_player = ""

    def evaluate_transitions(self, history: deque):
        if not history:
            return self.system_state, self.near_state, self.far_state, self.active_player

        if self.system_state == "IDLE":
            near_result = self.near_engine.tick(history)
            far_result  = self.far_engine.tick(history)

            near_fires  = near_result == "TRIGGER_ACTIVE"
            far_fires   = far_result  == "TRIGGER_ACTIVE"

            if near_fires or far_fires:
                if near_fires and far_fires:
                    ap = "joint"
                elif near_fires:
                    ap = "near"
                else:
                    ap = "far"
                now = history[-1].timestamp
                self.system_state  = "ACTIVE"
                self.active_player = ap
                # Freeze sub-engine states at WAITING (they reset on trigger)
                self.near_state    = "WAITING"
                self.far_state     = "WAITING"
                self.active_engine.init_active(now, ap)
                print(f"[FULL] System -> ACTIVE. active_player={ap}")
            else:
                self.near_state = self.near_engine.state
                self.far_state  = self.far_engine.state

        elif self.system_state == "ACTIVE":
            result = self.active_engine.tick(history)
            if result == "END_ACTIVE":
                self.system_state  = "IDLE"
                self.active_player = ""
                self.near_engine.reset_to_waiting()
                self.far_engine.reset_to_waiting()
                self.near_state = "WAITING"
                self.far_state  = "WAITING"
                print("[FULL] System -> IDLE.")

        return self.system_state, self.near_state, self.far_state, self.active_player

    @property
    def last_transition_time(self) -> Optional[float]:
        return self.active_engine.last_transition_time

    @property
    def last_active_debug(self) -> dict:
        return self.active_engine.last_active_debug


# ─────────────────────────────────────────────────────────────────────────────
# ServingSideFilter
# ─────────────────────────────────────────────────────────────────────────────

class ServingSideFilter:
    """
    Confirms the serving side after MIN_SERVES_TO_CONFIRM consecutive detections.
    Before confirmation all detections are emitted (Option A bootstrap).
    After confirmation, cross-side detections are suppressed until the new side
    accumulates MIN_SERVES_TO_CONFIRM consecutive detections.
    """

    def __init__(self, min_serves: int = MIN_SERVES_TO_CONFIRM):
        self.min_serves     = min_serves
        self.confirmed_side: Optional[str] = None
        self.streak:         List[str]     = []
        self.pending_side:   Optional[str] = None
        self.pending_count:  int           = 0

    def record(self, side: str) -> Tuple[bool, str]:
        """
        Record a detected serve on `side`. Returns (keep, resolved_side).
        resolved_side may differ from `side` when "joint" is resolved.
        """
        resolved = self._resolve(side)

        if self.confirmed_side is None:
            self.streak.append(resolved)
            if (len(self.streak) >= self.min_serves and
                    len(set(self.streak[-self.min_serves:])) == 1):
                self.confirmed_side = self.streak[-1]
                print(f"[FILTER] Confirmed side: {self.confirmed_side}")
            return True, resolved

        if resolved == self.confirmed_side:
            self.streak.append(resolved)
            self.pending_side  = None
            self.pending_count = 0
            return True, resolved

        # Cross-side detection
        if resolved == self.pending_side:
            self.pending_count += 1
        else:
            self.pending_side  = resolved
            self.pending_count = 1

        if self.pending_count >= self.min_serves:
            print(f"[FILTER] Side switch: {self.confirmed_side} -> {resolved}")
            self.confirmed_side = resolved
            self.streak         = [resolved] * self.pending_count
            self.pending_side   = None
            self.pending_count  = 0
            return True, resolved

        print(f"[FILTER] Suppressed {resolved} serve (pending={self.pending_count}/{self.min_serves})")
        return False, resolved

    def _resolve(self, side: str) -> str:
        if side != "joint":
            return side
        return self.confirmed_side if self.confirmed_side is not None else "near"


# ─────────────────────────────────────────────────────────────────────────────
# Debug rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(frame, tel: FullTelemetryFrame, engine: FullTransitionEngine,
                 provider: FullTelemetryProvider):
    sys_state  = tel.system_state
    near_state = engine.near_state
    far_state  = engine.far_state

    # Active zone fill
    if sys_state == "ACTIVE":
        ap = tel.active_player
        zone = (provider.far_active_zone  if ap == "far"  else
                provider.near_active_zone if ap == "near" else None)
        if zone is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], (144, 238, 144))
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
            cv2.polylines(frame, [zone], True, (0, 200, 0), 1)

    # System state badge
    color = (0, 255, 0) if sys_state == "ACTIVE" else (0, 255, 255)
    ap_label = f" [{tel.active_player}]" if sys_state == "ACTIVE" else ""
    cv2.putText(frame, f"STATE: {sys_state}{ap_label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Near player — blue box
    if tel.near_player_box:
        x1, y1, x2, y2 = tel.near_player_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"NEAR [{near_state}]", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 80), 1, cv2.LINE_AA)

    # Far player — orange box
    if tel.far_player_box:
        x1, y1, x2, y2 = tel.far_player_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(frame, f"FAR [{far_state}]", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1, cv2.LINE_AA)

    # Baseline strip
    sx1, sy1, sx2, sy2 = provider.far_baseline_strip
    cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (255, 255, 0), 1)

    # Exclusion zones
    for ex1, ey1, ex2, ey2 in provider.exclusion_zones:
        cv2.rectangle(frame, (int(ex1), int(ey1)), (int(ex2), int(ey2)), (0, 0, 255), 1)

    # Racquet boxes — magenta (near) / cyan-magenta (far)
    if tel.near_racquet_box:
        rx1, ry1, rx2, ry2 = tel.near_racquet_box
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
    if tel.far_racquet_box:
        rx1, ry1, rx2, ry2 = tel.far_racquet_box
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (200, 0, 200), 2)

    # ARMED: z_box + toss candidates (near)
    if near_state == "ARMED":
        if tel.near_z_box:
            x1, y1, x2, y2 = tel.near_z_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for ball in tel.near_toss_ball_candidates:
            bx1, by1, bx2, by2 = ball["box"]
            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)

    # ARMED: z_box + toss candidates (far)
    if far_state == "ARMED":
        if tel.far_z_box:
            x1, y1, x2, y2 = tel.far_z_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        for ball in tel.far_toss_ball_candidates:
            bx1, by1, bx2, by2 = ball["box"]
            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 200, 0), 2)

    # ACTIVE: ball trace
    if sys_state == "ACTIVE":
        trace = [(px, py) for _, px, py in engine.active_engine._trace_ball_history]
        n = len(trace)
        if n >= 2:
            for i in range(1, n):
                age       = i / (n - 1)
                col       = (0, int(120 * age), int(255 * age))
                thickness = max(1, int(3 * age))
                cv2.line(frame,
                         (int(trace[i-1][0]), int(trace[i-1][1])),
                         (int(trace[i][0]),   int(trace[i][1])),
                         col, thickness, cv2.LINE_AA)
        if n >= 1:
            cv2.circle(frame, (int(trace[-1][0]), int(trace[-1][1])),
                       5, (0, 200, 255), -1, cv2.LINE_AA)
        for ball in tel.active_ball_candidates:
            bx1, by1, bx2, by2 = ball["box"]
            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)),
                          (0, 255, 255), 2)


def render_debug_panel(engine: FullTransitionEngine) -> np.ndarray:
    panel = np.ones((300, 520, 3), dtype=np.uint8) * 240
    sys_state = engine.system_state

    if sys_state == "ACTIVE":
        _render_active_panel(panel, engine)
    else:
        _render_idle_panel(panel, engine)
    return panel


def _render_idle_panel(panel, engine: FullTransitionEngine):
    x0, y, lh, fs = 12, 28, 26, 0.48
    cv2.putText(panel, "IDLE — Near / Far Sub-Engines", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2, cv2.LINE_AA)
    y += 8
    bar_w, bar_h, label_w = 160, 12, 110

    for side, state, scores in [
        ("NEAR", engine.near_state, engine.near_engine.last_serve_scores),
        ("FAR",  engine.far_state,  engine.far_engine.last_serve_scores),
    ]:
        y += lh
        cv2.putText(panel, f"{side} [{state}]", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
        if state == "ARMED":
            for label, key, color in [
                ("Toss",  "toss_score",   (0, 200, 200)),
                ("Serve", "serve_score",  None),
            ]:
                y += lh
                val = scores.get(key, 0.0)
                if color is None:
                    color = (0, 220, 0) if val >= 0.55 else (0, 140, 255)
                cv2.putText(panel, f"  {label}:", (x0, y),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 1, cv2.LINE_AA)
                bx = x0 + label_w
                cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + bar_w, y + 2),
                              (190, 190, 190), -1)
                cv2.rectangle(panel, (bx, y - bar_h + 2),
                              (bx + int(val * bar_w), y + 2), color, -1)
                cv2.putText(panel, f"{val:.3f}", (bx + bar_w + 6, y),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
        y += 4


def _render_active_panel(panel, engine: FullTransitionEngine):
    x0, y, lh, fs = 15, 25, 28, 0.48
    ap = engine.active_player
    cv2.putText(panel, f"ACTIVE [{ap}] — Ball Trace / Energy Bar", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2, cv2.LINE_AA)
    d = engine.last_active_debug

    y += lh
    has_trace   = d.get("has_active_trace", False)
    tst         = d.get("time_since_trace", 0.0)
    trace_col   = (0, 180, 0) if has_trace else (0, 0, 200)
    trace_label = "YES" if has_trace else f"NO ({tst:.1f}s ago)"
    cv2.putText(panel, f"Trace: {trace_label}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, trace_col, 1, cv2.LINE_AA)

    y += lh
    em     = d.get("energy_bar_mode", False)
    energy = d.get("point_energy", 1.0)
    status = d.get("energy_status", "--")
    elabel = "ENERGY BAR" if em else "Energy (dormant)"
    ecol   = (0, 180, 0) if not em else (
        (0, 0, 220) if energy < 0.3 else (0, 165, 255) if energy < 0.6 else (0, 200, 0))
    cv2.putText(panel, f"{elabel}: {energy:.2f} [{status}]", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, ecol, 2 if em else 1, cv2.LINE_AA)
    y += 6
    bar_w, bg = 200, (180, 180, 180) if not em else (100, 100, 100)
    cv2.rectangle(panel, (x0, y), (x0 + bar_w, y + 14), bg, -1)
    if em and energy > 0:
        fc = (0, 0, 220) if energy < 0.3 else (0, 165, 255) if energy < 0.6 else (0, 200, 0)
        cv2.rectangle(panel, (x0, y), (x0 + int(energy * bar_w), y + 14), fc, -1)
    y += 24
    cv2.putText(panel, f"Balls: {d.get('ball_count', 0)}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 80, 80), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# CSV helper
# ─────────────────────────────────────────────────────────────────────────────

_CSV_COLS = [
    "serve", "frame", "timestamp", "side", "system_state",
    "near_state", "far_state",
    "time_since_trace", "has_active_trace",
    "energy_bar_mode", "point_energy", "energy_status",
    "ball_count",
]


def _write_csv_row(csv_writer, engine: FullTransitionEngine, tel: FullTelemetryFrame,
                   serve_number: int, frame_in_serve: int, side: str):
    d = engine.last_active_debug
    csv_writer.writerow({
        "serve":            serve_number,
        "frame":            frame_in_serve,
        "timestamp":        round(tel.timestamp, 4),
        "side":             side,
        "system_state":     tel.system_state,
        "near_state":       engine.near_state,
        "far_state":        engine.far_state,
        "time_since_trace": round(d.get("time_since_trace", 0.0), 3),
        "has_active_trace": d.get("has_active_trace", False),
        "energy_bar_mode":  d.get("energy_bar_mode", False),
        "point_energy":     round(d.get("point_energy", 1.0), 3),
        "energy_status":    d.get("energy_status", ""),
        "ball_count":       d.get("ball_count", 0),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Core segment-collection loop
# ─────────────────────────────────────────────────────────────────────────────

def _collect_full_segments(video_path: str, headless: bool = False,
                            start_frame: int = 0, csv_path: Optional[str] = None):
    """
    Run the full (near + far) pipeline on a single video.

    Returns
    -------
    active_segments : list of (start_sec, end_sec, side)
    serve_number    : total serves emitted (after filter)
    csv_path        : path written
    timestamps      : list of (timestamp_sec, side)
    """
    if csv_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        csv_path   = os.path.join(video_dir, f"{video_stem}_full_telemetry.csv")

    _probe   = cv2.VideoCapture(video_path)
    orig_fps = _probe.get(cv2.CAP_PROP_FPS)
    _total   = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    _probe.release()
    if orig_fps <= 0 or orig_fps > 300:
        orig_fps = 30.0
    video_duration_sec = _total / orig_fps if _total > 0 else float("inf")

    provider = FullTelemetryProvider(video_path)
    engine   = FullTransitionEngine(
        fps=provider.fps,
        far_baseline_strip=provider.far_baseline_strip,
    )
    side_filter = ServingSideFilter()

    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLS)
    csv_writer.writeheader()

    video_time_offset       = start_frame / orig_fps
    active_segments:        List[Tuple[float, float, str]] = []
    timestamps:             List[Tuple[float, str]]        = []
    current_segment_start:  float = 0.0
    current_active_player:  str   = ""
    last_telemetry_ts:      float = 0.0
    serve_number:           int   = 0
    _raw_serve_number:      int   = 0   # before filter
    frame_in_serve:         int   = 0

    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[FULL] Seeking to frame {start_frame}")

    WAITING_STRIDE = 3
    interrupted    = False

    try:
        while cap.isOpened():
            success, orig_frame = cap.read()
            if not success:
                break

            frame = cv2.resize(orig_frame, (960, 540), interpolation=cv2.INTER_LINEAR)

            both_waiting = (engine.near_state == "WAITING" and
                            engine.far_state  == "WAITING" and
                            engine.system_state == "IDLE")
            skip = (both_waiting
                    and provider.frame_counter % WAITING_STRIDE != 0
                    and bool(provider.telemetry_history))

            if skip:
                provider.frame_counter += 1
                last = provider.telemetry_history[-1]
                tel = FullTelemetryFrame(
                    frame_id=provider.frame_counter,
                    timestamp=provider.frame_counter / provider.fps,
                    system_state="IDLE",
                    active_player="",
                    near_player_box=last.near_player_box,
                    near_player_world=last.near_player_world,
                    far_player_box=last.far_player_box,
                    far_player_world=last.far_player_world,
                )
                provider.telemetry_history.append(tel)
            else:
                tel = provider.process_frame(frame)

            last_telemetry_ts = tel.timestamp

            new_sys, new_near, new_far, new_ap = engine.evaluate_transitions(
                provider.telemetry_history
            )

            old_sys = provider.system_state

            if old_sys == "IDLE" and new_sys == "ACTIVE":
                _raw_serve_number += 1
                frame_in_serve = 0
                serve_ts = video_time_offset + tel.timestamp
                current_segment_start = serve_ts
                current_active_player = new_ap
                print(f"[FULL] Serve #{_raw_serve_number} detected at {serve_ts:.2f}s "
                      f"(side={new_ap})")

            elif old_sys == "ACTIVE" and new_sys == "IDLE":
                end_t = (engine.last_transition_time
                         if engine.last_transition_time is not None
                         else tel.timestamp)
                padded_end = min(video_time_offset + end_t + HIGHLIGHT_END_PAD_SEC,
                                 video_duration_sec)
                keep, resolved_side = side_filter.record(current_active_player)
                if keep:
                    serve_number += 1
                    active_segments.append((current_segment_start, padded_end, resolved_side))
                    timestamps.append((current_segment_start, resolved_side))

            provider.update_state(new_sys, new_near, new_far, new_ap,
                                  last_active_trace_time=engine.active_engine.last_active_trace_time)

            if provider.system_state == "ACTIVE":
                frame_in_serve += 1
                _write_csv_row(csv_writer, engine, tel, _raw_serve_number,
                               frame_in_serve, current_active_player)

            if not headless:
                render_frame(frame, tel, engine, provider)
                debug_panel = render_debug_panel(engine)
                cv2.imshow("Full Anya Pipeline", frame)
                cv2.imshow("Full Debug Panel",   debug_panel)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        interrupted = True
        print("\n[FULL] Ctrl-C — saving completed segments...")

    finally:
        if provider.system_state == "ACTIVE":
            padded_end = min(video_time_offset + last_telemetry_ts + HIGHLIGHT_END_PAD_SEC,
                             video_duration_sec)
            keep, resolved_side = side_filter.record(current_active_player)
            if keep:
                serve_number += 1
                active_segments.append((current_segment_start, padded_end, resolved_side))
                timestamps.append((current_segment_start, resolved_side))
        cap.release()
        csv_file.close()
        if not headless:
            cv2.destroyAllWindows()

    print(f"[FULL] {os.path.basename(video_path)}: {_raw_serve_number} raw detections, "
          f"{serve_number} kept after filter.")
    if interrupted:
        print("[FULL] (interrupted — results cover completed detections only)")

    return active_segments, serve_number, csv_path, timestamps


# ─────────────────────────────────────────────────────────────────────────────
# Serve-run filter helpers (reused from run_anya logic)
# ─────────────────────────────────────────────────────────────────────────────

def _group_segments_into_runs(segments, gap_threshold_sec=GAP_THRESHOLD_SEC):
    if not segments:
        return []
    runs, current_run = [], [segments[0]]
    for seg in segments[1:]:
        if seg[0] - current_run[-1][1] <= gap_threshold_sec:
            current_run.append(seg)
        else:
            runs.append(current_run)
            current_run = [seg]
    runs.append(current_run)
    return runs


def _filter_by_serve_run(segments, min_run=MIN_SERVES_TO_CONFIRM,
                          gap_threshold_sec=GAP_THRESHOLD_SEC):
    # segments are (start, end, side); pass just (start, end) to the grouper
    seg_pairs = [(s, e) for s, e, _ in segments]
    side_map  = {(s, e): side for s, e, side in segments}
    runs      = _group_segments_into_runs(seg_pairs, gap_threshold_sec)
    valid     = []
    for i, run in enumerate(runs):
        if len(run) >= min_run:
            for pair in run:
                valid.append((pair[0], pair[1], side_map[pair]))
            print(f"[FILTER] Run {i+1}: {len(run)} serves — VALID")
        else:
            print(f"[FILTER] Run {i+1}: {len(run)} serves — DISCARDED "
                  f"(< {min_run} required)")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Highlight assembly (mirrors highlights_from_csv.py)
# ─────────────────────────────────────────────────────────────────────────────

def _video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return float("inf")


def _resolve_overlaps(raw_segments, start_buf: float, end_buf: float,
                      video_duration: float):
    if not raw_segments:
        return []

    starts = [max(0.0, rs - start_buf) for rs, _, _ in raw_segments]
    ends   = [min(video_duration, re + end_buf) for _, re, _ in raw_segments]
    sides  = [side for _, _, side in raw_segments]
    raw_s  = [rs for rs, _, _ in raw_segments]
    raw_e  = [re for _, re, _ in raw_segments]

    sb = [raw_s[i] - starts[i] for i in range(len(raw_segments))]
    eb = [ends[i]  - raw_e[i]  for i in range(len(raw_segments))]

    for i in range(len(raw_segments) - 1):
        overlap = ends[i] - starts[i + 1]
        if overlap <= 0:
            continue

        trim_end   = min(eb[i],     overlap / 2)
        trim_start = min(sb[i + 1], overlap - trim_end)
        remaining  = overlap - trim_end - trim_start
        if remaining > 0:
            extra = min(eb[i] - trim_end, remaining)
            trim_end += extra

        eb[i]        -= trim_end
        sb[i + 1]    -= trim_start
        ends[i]       = raw_e[i]  + eb[i]
        starts[i + 1] = raw_s[i + 1] - sb[i + 1]

        if ends[i] > starts[i + 1] + 1e-6:
            mid = (raw_e[i] + raw_s[i + 1]) / 2
            ends[i]       = mid
            starts[i + 1] = mid

    segments = []
    for i in range(len(raw_segments)):
        s, e = starts[i], ends[i]
        if e > s:
            segments.append((s, e, sides[i]))
    return segments


def _create_highlights(video_path: str, segments, output_path: str):
    if not segments:
        print("[HIGHLIGHT] No segments found — nothing to export.")
        return

    print(f"\n[HIGHLIGHT] {len(segments)} segment(s) → {output_path}")
    tmpdir = tempfile.mkdtemp(prefix="anya_highlights_")
    try:
        seg_files = []
        for i, (start, end, side) in enumerate(segments):
            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-i", video_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                "-vsync", "cfr",
                seg_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"[HIGHLIGHT] Warning: segment {i+1} failed.")
                print(result.stderr.decode(errors="replace"))
                continue
            seg_files.append(seg_path)
            mins, secs = int(start // 60), start % 60
            print(f"[HIGHLIGHT]   Segment {i+1}/{len(segments)}: "
                  f"{mins}:{secs:05.2f} – {end:.2f}s  [{side}]")

        if not seg_files:
            print("[HIGHLIGHT] No segments were successfully cut.")
            return

        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for sf in seg_files:
                f.write(f"file '{sf}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print("[HIGHLIGHT] Concat failed:")
            print(result.stderr.decode(errors="replace"))
        else:
            print(f"[HIGHLIGHT] Done → {output_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_full_anya_pipeline(video_path: str, output_path: Optional[str] = None,
                            headless: bool = False, start_frame: int = 0):
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_full_highlights.mp4")

    csv_path = os.path.splitext(output_path)[0] + "_telemetry.csv"

    segments, serve_number, _, timestamps = _collect_full_segments(
        video_path, headless, start_frame, csv_path=csv_path
    )

    print("\n" + "=" * 55)
    print(f"  SERVES DETECTED: {serve_number}")
    print("=" * 55)
    for i, (ts, side) in enumerate(timestamps, 1):
        mins, secs = int(ts // 60), ts % 60
        print(f"  Serve #{i:>3}: {mins}:{secs:05.2f}  ({ts:.2f}s)  [{side}]")
    print("=" * 55)

    if segments:
        duration = _video_duration(video_path)
        resolved = _resolve_overlaps(segments, HIGHLIGHT_START_PAD_SEC, 0.0, duration)
        _create_highlights(video_path, resolved, output_path)
        print(f"\n[FULL] Output video  : {output_path}")
    else:
        print("\n[FULL] No segments to export.")
    print(f"[FULL] Telemetry CSV : {csv_path}")
    return timestamps


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full (near + far) Anya serve detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ai.full_anya video.mp4
  python -m src.ai.full_anya video.mp4 --output highlights.mp4 --headless
  python -m src.ai.full_anya video.mp4 --start-frame 900
""",
    )
    parser.add_argument("input", metavar="VIDEO", help="Input video file")
    parser.add_argument("--output",      default=None,  help="Output MP4 path")
    parser.add_argument("--headless",    action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()
    run_full_anya_pipeline(args.input, args.output, args.headless, args.start_frame)
