"""
far_anya.py
============
Far-side serve detector.  Mirrors the near-side pipeline (anya_base /
anya_transitions / run_anya) but adapted for the player at the far end of the
court:

  • Far player is the primary tracked player (near player tracked only for ball
    exclusion).
  • Ready zone: dist from far baseline (78 ft) ∈ [−0.5, 3.5] ft.
  • Extended player persistence: FAR_PLAYER_PERSIST_FRAMES = 20 frames.
  • Net-occlusion correction: foot y estimated when box bottom is near net line.
  • Lower ball-detection confidence thresholds (ball is smaller at far end).
  • Serve-score weights: 0.05 × trophy + 0.95 × toss  (trophy unreliable from
    opposite angle).
  • MHI frame-diff fallback: secondary toss signal when YOLO misses small ball.
  • Far-side toss: 1 consecutive YOLO frame above head → score 0.7 (relaxed
    from near-side's 2-frame requirement).

Usage
-----
  python -m src.ai.far_anya video.mp4
  python -m src.ai.far_anya video.mp4 --output far_highlights.mp4 --headless
"""

import argparse
import csv
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from src.ai.utilities import (
    BoxSmoother, Config, _is_in_exclusion_zone,
    init_court, init_far_player_roi,
    create_auto_exclusion_zones, get_exclusion_zones_from_frames,
    create_highlights_ffmpeg,
)


# ─────────────────────────────────────────────────────────────────────────────
# Far-side constants
# ─────────────────────────────────────────────────────────────────────────────

FAR_BASELINE_FT            = Config.COURT_LENGTH_FT   # 78.0 ft
FAR_PLAYER_PERSIST_FRAMES  = 20    # frames to hold last-known box when detection drops
FAR_ACTIVE_BALL_CONF       = 0.10  # lower than near-side 0.15 (small ball at distance)
FAR_TOSS_BALL_CONF         = 0.05  # lower than near-side 0.10
FAR_TOSS_BALL_IMGSZ        = 480
FAR_ACTIVE_BALL_IMGSZ      = 960
FAR_ACTIVE_ZONE_CACHE      = "far_active_zone_config.json"
NET_OCCLUDE_TOLERANCE_PX   = 25    # px: box bottom within this of net_y → assume occlusion


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FarTelemetryFrame:
    frame_id:              int
    timestamp:             float
    state:                 str
    far_player_box:        Optional[Tuple[int, int, int, int]] = None  # serving player
    far_player_world:      Optional[Tuple[float, float]]       = None  # (wx, wy) ft
    near_player_box:       Optional[Tuple[int, int, int, int]] = None  # opponent (exclusion only)
    near_player_world:     Optional[Tuple[float, float]]       = None  # (wx, wy) ft
    toss_ball_candidates:  List[dict]                          = None
    active_ball_candidates: List[dict]                         = None
    trophy_score:          float                               = 0.0
    z_box:                 Optional[Tuple[int, int, int, int]] = None
    mhi_toss_score:        float                               = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry Provider
# ─────────────────────────────────────────────────────────────────────────────

class FarTelemetryProvider:
    """
    Per-frame sensor layer for far-side serve detection.

    Key adaptations vs AnyaTelemetryProvider:
      - Far player is the primary tracked player (the server).
      - Extended persistence (FAR_PLAYER_PERSIST_FRAMES) for intermittent detection.
      - Net-occlusion correction: when the player box bottom is at/near net pixel-y,
        the foot position is estimated from box top + rolling-median box height.
      - Ball detection uses lower confidence thresholds.
      - MHI frame-differencing over the head ROI provides a secondary toss signal.
    """

    def __init__(self, video_path: str, court_frame_idx: int = 300,
                 active_zone_polygon: Optional[np.ndarray] = None):
        self.video_path = video_path
        self._init_video_props()

        # ── Models ─────────────────────────────────────────────────────────
        self.player_model = YOLO("yolo26n.pt")
        self.ball_model   = YOLO("weights/ball/weights/best.pt")
        self.trophy_model = YOLO(Config.DEFAULT_NEAR_TROPHY_MODEL_PATH)

        # ── Court geometry ─────────────────────────────────────────────────
        self.court_vertices, self.frame_shape = init_court(
            self.video_path, target_idx=court_frame_idx, analysis_size=(960, 540)
        )
        self.far_player_roi = init_far_player_roi(
            self.video_path, analysis_size=(960, 540)
        )
        self.H     = self._compute_homography()
        self.H_inv = np.linalg.inv(self.H)   # world → pixel (for net-y projection)

        # Net pixel-y for occlusion detection
        self.net_y_px = self._compute_net_y_px()
        print(f"[FAR] Net pixel-y estimate: {self.net_y_px:.1f}px")

        # Far-side active zone polygon (upper region of frame where far player operates)
        if active_zone_polygon is not None:
            self.active_zone_polygon = active_zone_polygon
        else:
            self.active_zone_polygon = self._get_or_define_active_zone()

        # ── Static exclusion zones ─────────────────────────────────────────
        print("\n[FAR] Scanning video for static exclusion zones...")
        try:
            self.static_exclusion_zones = create_auto_exclusion_zones(
                self.video_path, self.ball_model,
                num_frames=50, conf=0.04, eps=12, padding=5,
                ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                analysis_size=(960, 540),
            )
            print(f"[FAR] {len(self.static_exclusion_zones)} static exclusion zone(s)")
        except Exception as e:
            print(f"[FAR] WARN: Could not compute static exclusion zones: {e}")
            self.static_exclusion_zones = []
        self.dynamic_exclusion_zones: List = []

        # ── Dynamic exclusion zone buffering (0.5s after ARMED entry) ─────
        self._armed_frame_buffer:    List            = []
        self._armed_entry_time:      Optional[float] = None
        self._armed_collection_done: bool            = False
        self.ARMED_DYNAMIC_COLLECTION_SEC  = 0.5
        self.ARMED_DYNAMIC_SAMPLE_FRAMES   = 5

        # ── State & telemetry buffer ───────────────────────────────────────
        self.current_state = "WAITING"
        self.frame_counter = 0
        buffer_size = int(self.fps * Config.TELEMETRY_BUFFER_SECONDS)
        self.telemetry_history = deque(maxlen=buffer_size)

        # ── Far-player persistence ─────────────────────────────────────────
        self._last_known_far_box:   Optional[Tuple[int, int, int, int]] = None
        self._last_known_far_world: Optional[Tuple[float, float]]       = None
        self._far_persist_counter:  int                                  = 0
        # Rolling box heights for net-occlusion foot estimation
        self._far_box_heights: deque = deque(maxlen=30)
        # Aggressive alpha_pos (0.5) to adapt quickly after detection gaps
        self._far_box_smoother = BoxSmoother(alpha_pos=0.50, alpha_size=0.12)

        # ── Near player cache (for ball exclusion) ─────────────────────────
        self._last_near_box: Optional[Tuple[int, int, int, int]] = None

        # ── ACTIVE player-detection striding ──────────────────────────────
        self.ACTIVE_PLAYER_STRIDE  = 4
        # (far_box, far_world, near_box)
        self._cached_player_boxes: Tuple = (None, None, None)

        # ── Trophy classification stride ───────────────────────────────────
        self.ARMED_TROPHY_STRIDE = 2
        self._last_trophy_score: float = 0.0

        # ── MHI (Motion History Image) for toss fallback ───────────────────
        self.MHI_BUFFER_FRAMES = 15
        self._mhi_roi_buffer: deque = deque(maxlen=self.MHI_BUFFER_FRAMES)
        self._mhi_last_score:  float = 0.0

    # ── Video properties ──────────────────────────────────────────────────────

    @property
    def far_baseline_strip(self) -> Tuple[float, float, float, float]:
        """
        Pixel-space strip where the far player is considered 'at the baseline'.

        Bottom edge = average pixel-y of the far baseline corners (TL, TR).
        Top edge    = bottom edge − 50 px (toward the back of the court).
        Width       = horizontal span of TL → TR (court width in pixels).

        Returns (x1, y1, x2, y2) with y1 < y2 (standard OpenCV rect convention).
        """
        BL, BR, TR, TL = self.court_vertices
        x1 = float(min(TL[0], TR[0]))
        x2 = float(max(TL[0], TR[0]))
        y_baseline = (TL[1] + TR[1]) / 2.0   # average y of far baseline
        return (x1, y_baseline - 50.0, x2, y_baseline)

    def _init_video_props(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width  = 960
        self.height = 540
        cap.release()

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _compute_homography(self):
        BL, BR, TR, TL = self.court_vertices
        dst_pts = np.array([
            [0, 0],
            [Config.COURT_WIDTH_FT, 0],
            [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT],
            [0,                     Config.COURT_LENGTH_FT],
        ], dtype=np.float32)
        src_pts = np.array([BL, BR, TR, TL], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H

    def _compute_net_y_px(self) -> float:
        """Map net centre (world: 13.5 ft, 39 ft) to pixel-y using H_inv."""
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

    def _estimate_far_feet_y(self, box: Tuple[int, int, int, int]) -> float:
        """
        Return the best estimate of the far player's foot pixel-y.

        The far player stands behind the far baseline (top of frame, small y).
        The net occupies a band around net_y_px (mid-frame, larger y).
        When the box bottom is within NET_OCCLUDE_TOLERANCE pixels of the net
        line — or below it — the bottom is likely clipped by the net's visual
        presence.  In that case, extrapolate the foot position from the box top
        using the rolling-median box height seen in prior un-occluded frames.
        """
        x1, y1, x2, y2 = box
        box_h = y2 - y1
        if box_h > 0:
            self._far_box_heights.append(box_h)

        occluded = abs(y2 - self.net_y_px) < NET_OCCLUDE_TOLERANCE_PX or y2 > self.net_y_px
        if occluded and self._far_box_heights:
            median_h = sorted(self._far_box_heights)[len(self._far_box_heights) // 2]
            return float(y1 + median_h)
        return float(y2)

    # ── Active zone polygon ───────────────────────────────────────────────────

    def _get_or_define_active_zone(self) -> np.ndarray:
        if os.path.exists(FAR_ACTIVE_ZONE_CACHE):
            try:
                with open(FAR_ACTIVE_ZONE_CACHE, "r") as f:
                    points = json.load(f)
                print(f"[FAR] Loaded active zone from {FAR_ACTIVE_ZONE_CACHE}")
                return np.array(points, dtype=np.int32)
            except Exception as e:
                print(f"[FAR] WARN: Could not load polygon cache: {e}")

        print("[FAR] Define the far-side active zone. Click 8 points (clockwise). Press q to confirm.")
        points = self._interactive_polygon_selector()
        with open(FAR_ACTIVE_ZONE_CACHE, "w") as f:
            json.dump(points.tolist(), f)
        return points

    def _interactive_polygon_selector(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read frame for polygon definition.")
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
                cv2.imshow("Far Active Zone", display)

        cv2.namedWindow("Far Active Zone")
        cv2.setMouseCallback("Far Active Zone", cb)
        while True:
            cv2.imshow("Far Active Zone", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27) and len(pts) == 8:
                break
        cv2.destroyWindow("Far Active Zone")
        return np.array(pts, dtype=np.int32)

    def _is_in_active_zone(self, cx: float, cy: float) -> bool:
        return cv2.pointPolygonTest(
            self.active_zone_polygon, (float(cx), float(cy)), False
        ) >= 0

    # ── Exclusion zones ───────────────────────────────────────────────────────

    @property
    def exclusion_zones(self) -> List:
        return self.static_exclusion_zones + self.dynamic_exclusion_zones

    def _is_in_player_box(self, bx, by, player_box, padding: int = 10) -> bool:
        if player_box is None:
            return False
        x1, y1, x2, y2 = player_box
        return (x1 - padding <= bx <= x2 + padding and
                y1 - padding <= by <= y2 + padding)

    # ── Player tracking ───────────────────────────────────────────────────────

    def _track_far_player_roi(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect far player within the user-defined ROI.
        Same logic as AnyaTelemetryProvider._track_far_player_roi.
        """
        if self.far_player_roi is None:
            return None
        (rx1, ry1), (rx2, ry2) = self.far_player_roi
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None

        results = self.player_model(roi, verbose=False, conf=0.5, imgsz=Config.FAR_PLAYER_IMGSZ)
        if not (results and results[0].boxes):
            return None

        roi_h     = ry2 - ry1
        best_conf = -1.0
        best_box  = None
        for b in results[0].boxes:
            if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                continue
            lx1, ly1, lx2, ly2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            if (roi_h - ly2) <= Config.FAR_ROI_BOTTOM_TOLERANCE:
                continue
            if conf > best_conf:
                best_conf = conf
                best_box  = (rx1 + lx1, ry1 + ly1, rx1 + lx2, ry1 + ly2)
        return best_box

    def _track_players(self, frame) -> Tuple[
        Optional[Tuple[int, int, int, int]],
        Optional[Tuple[float, float]],
        Optional[Tuple[int, int, int, int]],
    ]:
        """
        Detect all players and classify far vs near using world-space geometry.

        Strategy:
          Far player  — ROI detection is tried first (more focused); falls back
                        to full-frame candidate closest to far baseline.
          Near player — full-frame detection, closest to near baseline (y=0).

        Returns (far_box, far_world, near_box).
        far_world uses net-occlusion corrected foot position.
        """
        # Far player via ROI (primary path)
        far_box_roi = self._track_far_player_roi(frame)

        # Full-frame detection for near player (and far player fallback)
        results = self.player_model(frame, verbose=False, conf=0.5, imgsz=Config.PLAYER_IMGSZ)
        candidates = []
        if results and results[0].boxes:
            for b in results[0].boxes:
                if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cx = (x1 + x2) / 2.0
                wx, wy = self.get_world_pos(cx, float(y2))
                candidates.append((x1, y1, x2, y2, wx, wy))

        # Near player: closest to near baseline (world_y = 0), in near half
        pad = Config.NEAR_PLAYER_X_PAD_FT
        near_candidates = [
            c for c in candidates
            if (abs(c[5]) < abs(c[5] - Config.COURT_LENGTH_FT) and
                -pad <= c[4] <= Config.COURT_WIDTH_FT + pad)
        ]
        near_box = None
        if near_candidates:
            nc = min(near_candidates, key=lambda c: abs(c[5]))
            near_box = nc[:4]

        # Far player: prefer ROI detection; fall back to full-frame
        if far_box_roi is not None:
            est_foot_y = self._estimate_far_feet_y(far_box_roi)
            fx1, fy1, fx2, fy2 = far_box_roi
            fcx = (fx1 + fx2) / 2.0
            wx, wy = self.get_world_pos(fcx, est_foot_y)
            sx1, sy1, sx2, sy2 = self._far_box_smoother.smooth_box_xyxy(fx1, fy1, fx2, fy2)
            far_box   = (sx1, sy1, sx2, sy2)
            far_world = (wx, wy)
        else:
            near_box_coords = near_box  # may be None
            far_candidates  = [
                c for c in candidates
                if (near_box_coords is None or c[:4] != near_box_coords) and
                   abs(c[5] - Config.COURT_LENGTH_FT) < abs(c[5])
            ]
            if far_candidates:
                fc = min(far_candidates, key=lambda c: abs(c[5] - Config.COURT_LENGTH_FT))
                est_foot_y = self._estimate_far_feet_y(fc[:4])
                fcx = (fc[0] + fc[2]) / 2.0
                wx, wy = self.get_world_pos(fcx, est_foot_y)
                sx1, sy1, sx2, sy2 = self._far_box_smoother.smooth_box_xyxy(*fc[:4])
                far_box   = (sx1, sy1, sx2, sy2)
                far_world = (wx, wy)
            else:
                far_box = far_world = None

        return far_box, far_world, near_box

    # ── z_box (toss zone above player head) ──────────────────────────────────

    def _create_z_box(self, player_box) -> Optional[Tuple[int, int, int, int]]:
        if player_box is None:
            return None
        x1, y1, x2, y2 = player_box
        pw, ph   = x2 - x1, y2 - y1
        pcx, pcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        z_w  = pw * 2.0
        z_h  = ph * 2.5
        zx1  = int(pcx - z_w / 2.0)
        zx2  = int(pcx + z_w / 2.0)
        zy2  = int(pcy)
        zy1  = max(0, int(zy2 - z_h))
        return (zx1, zy1, zx2, zy2)

    def _is_in_z_box(self, bx: float, by: float, z_box) -> bool:
        if z_box is None:
            return False
        x1, y1, x2, y2 = z_box
        return x1 <= bx <= x2 and y1 <= by <= y2

    # ── MHI toss fallback ─────────────────────────────────────────────────────

    def _compute_mhi_toss_score(self, frame, player_box) -> float:
        """
        Motion History Image score for the region immediately above the player's head.

        Compares the current ROI frame against the oldest frame in a 15-frame
        rolling buffer.  The mean absolute pixel difference — normalised to [0,1]
        via a soft threshold band — represents motion intensity.  Values > 0.3
        indicate meaningful movement (toss candidate).

        Returns 0.0 when the buffer is too short or the player box is unknown.
        """
        if player_box is None:
            self._mhi_roi_buffer.clear()
            return 0.0

        x1, y1, x2, y2 = player_box
        fh, fw = frame.shape[:2]
        ph     = y2 - y1

        rx1 = max(0, x1)
        rx2 = min(fw, x2)
        ry1 = max(0, y1 - ph)   # 1× player height above box top
        ry2 = max(0, y1)        # top of player box

        if rx2 <= rx1 or ry2 <= ry1:
            self._mhi_roi_buffer.clear()
            return 0.0

        roi_gray = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        self._mhi_roi_buffer.append(roi_gray)

        if len(self._mhi_roi_buffer) < 3:
            return self._mhi_last_score

        ref  = self._mhi_roi_buffer[0]
        curr = self._mhi_roi_buffer[-1]
        if ref.shape != curr.shape:
            self._mhi_roi_buffer.clear()
            return 0.0

        diff  = cv2.absdiff(curr, ref)
        score = float(np.mean(diff)) / 255.0

        # Soft threshold: 0.02 → 0.0,  0.10 → 1.0
        MHI_LOW, MHI_HIGH = 0.02, 0.10
        normalized = max(0.0, min(1.0, (score - MHI_LOW) / (MHI_HIGH - MHI_LOW)))
        self._mhi_last_score = normalized
        return normalized

    # ── Main frame processing ─────────────────────────────────────────────────

    def process_frame(self, frame) -> FarTelemetryFrame:
        self.frame_counter += 1
        timestamp = self.frame_counter / self.fps

        tel = FarTelemetryFrame(
            frame_id=self.frame_counter,
            timestamp=timestamp,
            state=self.current_state,
            toss_ball_candidates=[],
            active_ball_candidates=[],
        )

        # ── 1. Player tracking ────────────────────────────────────────────
        if (self.current_state == "ACTIVE"
                and self.frame_counter % self.ACTIVE_PLAYER_STRIDE != 0
                and self._cached_player_boxes[0] is not None):
            far_box, far_world, near_box = self._cached_player_boxes
        else:
            far_box, far_world, near_box = self._track_players(frame)

            # Persist far player across short detection gaps
            if far_box is not None:
                self._last_known_far_box   = far_box
                self._last_known_far_world = far_world
                self._far_persist_counter  = 0
            else:
                self._far_persist_counter += 1
                if self._far_persist_counter <= FAR_PLAYER_PERSIST_FRAMES:
                    far_box   = self._last_known_far_box
                    far_world = self._last_known_far_world  # stale but stable over short gaps

            if near_box is not None:
                self._last_near_box = near_box

            self._cached_player_boxes = (far_box, far_world, near_box)

        tel.far_player_box   = far_box
        tel.far_player_world = far_world
        resolved_near_box    = near_box if near_box is not None else self._last_near_box
        tel.near_player_box  = resolved_near_box
        if resolved_near_box is not None:
            nx1, ny1, nx2, ny2 = resolved_near_box
            tel.near_player_world = self.get_world_pos(
                (nx1 + nx2) / 2.0, float(ny2)
            )

        # ── 2. ARMED: dynamic exclusion zone buffering (first 0.5s) ──────
        if self.current_state == "ARMED":
            now_t = self.frame_counter / self.fps
            if not self._armed_collection_done and self._armed_entry_time is not None:
                elapsed = now_t - self._armed_entry_time
                if elapsed <= self.ARMED_DYNAMIC_COLLECTION_SEC:
                    self._armed_frame_buffer.append(frame.copy())
                elif len(self._armed_frame_buffer) >= 1:
                    self.dynamic_exclusion_zones = get_exclusion_zones_from_frames(
                        self._armed_frame_buffer, self.ball_model,
                        sample_size=self.ARMED_DYNAMIC_SAMPLE_FRAMES,
                        conf=0.05, eps=5, min_samples=15, padding=5,
                        ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                    )
                    self._armed_collection_done = True
                    self._armed_frame_buffer    = []
                    print(f"[FAR] Dynamic exclusion: {len(self.dynamic_exclusion_zones)} zone(s)")

        # ── 2b. ARMED state detectors ─────────────────────────────────────
        if self.current_state == "ARMED" and far_box is not None:
            fx1, fy1, fx2, fy2 = far_box
            pw, ph = fx2 - fx1, fy2 - fy1
            fh, fw = frame.shape[:2]

            z_box = self._create_z_box(far_box)
            tel.z_box = z_box

            # Trophy pose — run every ARMED_TROPHY_STRIDE frames
            if self.frame_counter % self.ARMED_TROPHY_STRIDE == 0:
                pad_x = int(pw * Config.DEFAULT_TROPHY_PAD)
                pad_y = int(ph * Config.DEFAULT_TROPHY_PAD)
                tx1 = max(0, fx1 - pad_x);  ty1 = max(0, fy1 - pad_y)
                tx2 = min(fw, fx2 + pad_x); ty2 = min(fh, fy2 + pad_y)
                trophy_crop = frame[ty1:ty2, tx1:tx2]
                if trophy_crop.size > 0:
                    tr = self.trophy_model(trophy_crop, verbose=False, imgsz=Config.TROPHY_IMGSZ)
                    if tr and hasattr(tr[0], "probs") and tr[0].probs is not None:
                        idx = Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX
                        if idx < len(tr[0].probs.data):
                            self._last_trophy_score = float(tr[0].probs.data[idx])
            tel.trophy_score = self._last_trophy_score

            # MHI toss fallback — computed every ARMED frame
            tel.mhi_toss_score = self._compute_mhi_toss_score(frame, far_box)

            # YOLO toss ball — ROI above player head, lower conf than near-side
            rx1 = max(0,  int(fx1 - pw / 2))
            ry1 = max(0,  int(fy1 - 1.5 * ph))  # 1.5× player height above box top
            rx2 = min(fw, int(fx2 + pw / 2))
            ry2 = min(fh, int(fy1 + ph / 2))
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                ball_res = self.ball_model(
                    roi, verbose=False,
                    conf=FAR_TOSS_BALL_CONF, imgsz=FAR_TOSS_BALL_IMGSZ,
                )
                if ball_res and ball_res[0].boxes:
                    for b in ball_res[0].boxes:
                        cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
                        ball_cx = rx1 + (cx1 + cx2) / 2.0
                        ball_cy = ry1 + (cy1 + cy2) / 2.0
                        if (self._is_in_z_box(ball_cx, ball_cy, z_box) and
                                not _is_in_exclusion_zone(ball_cx, ball_cy, self.exclusion_zones) and
                                not self._is_in_player_box(ball_cx, ball_cy, far_box, padding=15)):
                            tel.toss_ball_candidates.append({
                                "box":  (rx1 + cx1, ry1 + cy1, rx1 + cx2, ry1 + cy2),
                                "conf": float(b.conf[0]),
                            })

        elif self.current_state == "ARMED":
            # far_box is None — still compute MHI (uses last ROI gracefully)
            tel.mhi_toss_score = self._compute_mhi_toss_score(frame, None)

        # ── 3. ACTIVE: full-frame ball detection ──────────────────────────
        if self.current_state == "ACTIVE":
            ball_res = self.ball_model(
                frame, verbose=False,
                conf=FAR_ACTIVE_BALL_CONF, imgsz=FAR_ACTIVE_BALL_IMGSZ,
            )
            if ball_res and ball_res[0].boxes:
                for b in ball_res[0].boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    if (self._is_in_active_zone(bcx, bcy) and
                            not _is_in_exclusion_zone(bcx, bcy, self.exclusion_zones) and
                            not self._is_in_player_box(bcx, bcy, far_box,          padding=10) and
                            not self._is_in_player_box(bcx, bcy, tel.near_player_box, padding=10)):
                        tel.active_ball_candidates.append({
                            "box":          (bx1, by1, bx2, by2),
                            "conf":         float(b.conf[0]),
                            "pixel_center": (bcx, bcy),
                        })

        self.telemetry_history.append(tel)
        return tel

    def update_state(self, new_state: str):
        old_state = self.current_state
        self.current_state = new_state

        if new_state == "WAITING" and old_state == "ACTIVE":
            # Reset persistence so next point starts fresh
            self._last_known_far_box   = None
            self._last_known_far_world = None
            self._far_persist_counter  = 0
            self._far_box_smoother.reset()

        if new_state == "ARMED" and old_state != "ARMED":
            now = self.frame_counter / self.fps
            self.dynamic_exclusion_zones = []
            self._armed_frame_buffer     = []
            self._armed_entry_time       = now
            self._armed_collection_done  = False
            self._last_trophy_score      = 0.0
            self._mhi_roi_buffer.clear()
            self._mhi_last_score         = 0.0
            print("[FAR] ARMED entered — starting dynamic exclusion zone collection (0–0.5s)")


# ─────────────────────────────────────────────────────────────────────────────
# State-machine (far-side adapted)
# ─────────────────────────────────────────────────────────────────────────────

class FarTransitionEngine:
    """
    Three-state machine (WAITING / ARMED / ACTIVE) adapted for the far-side player.

    Key differences vs TransitionEngine:
      WAITING → ARMED : far player bounding-box centre falls inside the
                        pixel-space baseline strip (50 px tall, court width wide,
                        bottom = far baseline y) AND lateral world-space travel
                        over the last 2 s is < 3 ft.
      ARMED → WAITING : total world-space distance traveled in the last 2 s
                        exceeds 3 ft.
      ARMED → ACTIVE  : serve_score = 0.05 × trophy + 0.95 × toss.
                        Toss score blends YOLO and MHI signals.
                        1 consecutive YOLO frame above head → toss score 0.7
                        (relaxed from near-side's 2-frame requirement because the
                        ball is small and harder to catch on consecutive frames).
      ACTIVE → WAITING: identical ball-trace / energy-bar hybrid as near-side,
                        but tracking the far player and using an extended
                        PLAYER_MISSING_GRACE_FRAMES of 15 (vs 5 near-side).
    """

    def __init__(self, fps: float, far_baseline_strip: Tuple[float, float, float, float]):
        self.fps = fps

        # ── WAITING / ARMED — movement-based zone ─────────────────────────
        # Pixel-space strip: (x1, y1, x2, y2) auto-computed from court vertices
        self.far_baseline_strip     = far_baseline_strip
        self.MOVEMENT_WINDOW_SEC    = 2.0   # rolling window for movement tracking
        self.READY_MAX_LATERAL_FT   = 4.0   # max lateral displacement in 2 s → allow ARMED
        self.ARMED_MAX_TOTAL_FT     = 4.0   # net displacement in 2 s → drop back to WAITING
        self.READY_WAIT_TIME_SEC    = 2.0   # sustained baseline presence required before ARMED
        self.READY_MISS_GRACE_FRAMES = 10   # missed-detection frames tolerated before timer resets

        # ── ARMED ─────────────────────────────────────────────────────────
        self.TRANSITION_SCORE_THRESHOLD = 0.55
        self.EVENT_WINDOW_SECONDS       = 1.2

        # Serve-score weights: trophy unreliable from far-side camera angle
        self.TROPHY_WEIGHT = 0.05
        self.TOSS_WEIGHT   = 0.95

        # MHI max contribution to toss score (avoids MHI alone firing the serve)
        self.MHI_THRESHOLD       = 0.30   # MHI score must exceed this to contribute
        self.MHI_MAX_CONTRIBUTION = 0.50  # MHI at 1.0 contributes this many toss points

        # ── ACTIVE — ball history ─────────────────────────────────────────
        self.BALL_HISTORY_SEC       = 1.5
        self.TRACE_NEARBY_PX        = 40.0
        self.TRACE_NEARBY_MIN_COUNT = 5

        # ── ACTIVE — energy bar ────────────────────────────────────────────
        self.ENERGY_BOOST_SPRINT        = 4.0
        self.ENERGY_BOOST_SWING         = 4.0
        self.ENERGY_DECAY_WALKING       = 0.3
        self.ENERGY_DECAY_MISSING       = 0.4
        self.ENERGY_DECAY_STILL         = 0.2
        self.PLAYER_SPRINT_VELOCITY_FTS = 7.0
        self.PLAYER_STILL_VELOCITY_FTS  = 2.0
        self.VELOCITY_WINDOW_SIZE       = 20
        self.ACTIVE_PLAYER_STRIDE       = 4
        # Extended grace: far player detected intermittently
        self.PLAYER_MISSING_GRACE_FRAMES = 15
        self.PLAYER_EMA_ALPHA            = 0.25

        # ── ACTIVE — walking gait ─────────────────────────────────────────
        self.GAIT_BUFFER_FRAMES = 45
        self.GAIT_MIN_REVERSALS = 2
        self.GAIT_MAX_REVERSALS = 8
        self.GAIT_MIN_DRIFT_PX  = 10.0

        # ── Persistent state — WAITING / ARMED (shared movement history) ──
        # Rolling window of (timestamp, wx, wy) used by both states.
        # WAITING reads lateral (x) cumulative travel; ARMED reads total travel.
        self._movement_history: deque = deque()

        # ── Persistent state — WAITING hold timer ─────────────────────────
        self.far_ready_start_time:  Optional[float] = None
        self._far_ready_miss_frames: int            = 0

        # ── Persistent state — ARMED ──────────────────────────────────────

        self.toss_consecutive_frames:       int             = 0
        self.toss_gap_frames:               int             = 0
        self.toss_ball_above_head_detected: bool            = False
        self.toss_min_y_px:                 Optional[float] = None
        self.last_toss_ball:                Optional[dict]  = None
        # Parabolic arc buffer: (timestamp, cy) over 1.5 s window
        self.TOSS_ARC_WINDOW_SEC:           float           = 1.5
        self.TOSS_ARC_MIN_POINTS:           int             = 3
        self.TOSS_ARC_R2_THRESHOLD:         float           = 0.80
        self._toss_arc_buffer:              deque           = deque()

        self._trophy_scores: deque = deque()
        self._toss_scores:   deque = deque()

        self.last_serve_scores = {
            "trophy_score": 0.0,
            "toss_score":   0.0,
            "mhi_score":    0.0,
            "serve_score":  0.0,
        }

        # ── Persistent state — ACTIVE ─────────────────────────────────────
        self.active_start_time:      float = 0.0
        self._all_ball_history:      deque = deque()
        self._trace_ball_history:    deque = deque()
        self.last_active_trace_time: float = 0.0

        self.energy_bar_mode:       bool  = False
        self.energy_bar_start_time: float = 0.0
        self.point_energy:          float = 1.0

        self._energy_player_positions: deque = deque(maxlen=self.VELOCITY_WINDOW_SIZE)
        self._energy_player_boxes:     deque = deque(maxlen=5)
        self._energy_gait_y_buffer:    deque = deque(maxlen=self.GAIT_BUFFER_FRAMES)
        self._player_missing_frames:   int   = 0
        self._smoothed_player_world:   Optional[Tuple[float, float]] = None

        # ── Output signal ─────────────────────────────────────────────────
        self.last_transition_time: Optional[float] = None

        self.last_active_debug = {
            "time_since_trace": 0.0,
            "has_active_trace": False,
            "energy_bar_mode":  False,
            "point_energy":     1.0,
            "ball_count":       0,
        }

    # ── Public entry point ────────────────────────────────────────────────────

    def evaluate_transitions(self, history: deque, current_state: str) -> str:
        if not history:
            return current_state
        if current_state == "WAITING":
            return self._check_waiting(history)
        if current_state == "ARMED":
            return self._check_armed(history)
        if current_state == "ACTIVE":
            return self._check_active(history)
        return current_state

    # ── WAITING → ARMED ───────────────────────────────────────────────────────

    def _check_waiting(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        # ── Update rolling movement history ──────────────────────────────
        if frame.far_player_world is not None:
            wx, wy = frame.far_player_world
            self._movement_history.append((now, wx, wy))
        while (self._movement_history and
               now - self._movement_history[0][0] > self.MOVEMENT_WINDOW_SEC):
            self._movement_history.popleft()

        # ── Evaluate zone conditions ──────────────────────────────────────
        # Condition 1: box centre inside the baseline strip
        centre_in_strip = False
        if frame.far_player_box is not None:
            fx1, fy1, fx2, fy2 = frame.far_player_box
            fcx = (fx1 + fx2) / 2.0
            fcy = (fy1 + fy2) / 2.0
            sx1, sy1, sx2, sy2 = self.far_baseline_strip
            centre_in_strip = sx1 <= fcx <= sx2 and sy1 <= fcy <= sy2

        # Condition 2: net lateral displacement in last 2 s < 4 ft
        lateral_disp = 0.0
        if len(self._movement_history) >= 2:
            lateral_disp = abs(
                self._movement_history[-1][1] - self._movement_history[0][1]
            )
        still_enough = lateral_disp < self.READY_MAX_LATERAL_FT

        # ── Hold timer with missed-detection grace ────────────────────────
        # Three cases per frame:
        #   A. Box detected AND in zone   → continue / start timer
        #   B. Box not detected           → grace buffer; only reset after
        #                                   READY_MISS_GRACE_FRAMES consecutive misses
        #   C. Box detected but out of zone → genuine movement; reset immediately
        box_missing = frame.far_player_box is None

        if centre_in_strip and still_enough:
            # Case A: valid in-zone detection
            self._far_ready_miss_frames = 0
            if self.far_ready_start_time is None:
                self.far_ready_start_time = now
            elapsed = now - self.far_ready_start_time
            if elapsed >= self.READY_WAIT_TIME_SEC:
                print(f"[FAR TRANSITION] WAITING -> ARMED. "
                      f"Held baseline {elapsed:.1f}s, lateral_disp={lateral_disp:.2f} ft.")
                self.far_ready_start_time   = None
                self._far_ready_miss_frames = 0
                self._movement_history.clear()
                return "ARMED"

        elif box_missing:
            # Case B: missed detection — apply grace before resetting
            self._far_ready_miss_frames += 1
            if self._far_ready_miss_frames > self.READY_MISS_GRACE_FRAMES:
                self.far_ready_start_time   = None
                self._far_ready_miss_frames = 0

        else:
            # Case C: player detected but outside zone — reset immediately
            self.far_ready_start_time   = None
            self._far_ready_miss_frames = 0

        return "WAITING"

    # ── ARMED → ACTIVE or ARMED → WAITING ─────────────────────────────────────

    def _check_armed(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        # ── Update rolling movement history ──────────────────────────────
        if frame.far_player_world is not None:
            wx, wy = frame.far_player_world
            self._movement_history.append((now, wx, wy))
        while (self._movement_history and
               now - self._movement_history[0][0] > self.MOVEMENT_WINDOW_SEC):
            self._movement_history.popleft()

        # ── Movement monitor: net displacement in last 2 s > 4 ft → WAITING ─
        # Use start-to-end straight-line distance rather than cumulative travel
        # so that box jitter does not accumulate into a spurious dropout.
        net_disp = 0.0
        if len(self._movement_history) >= 2:
            dx = self._movement_history[-1][1] - self._movement_history[0][1]
            dy = self._movement_history[-1][2] - self._movement_history[0][2]
            net_disp = math.hypot(dx, dy)

        if net_disp > self.ARMED_MAX_TOTAL_FT:
            print(f"[FAR TRANSITION] ARMED -> WAITING. "
                  f"Net displacement={net_disp:.2f} ft in last {self.MOVEMENT_WINDOW_SEC:.0f}s.")
            self._reset_armed_state()
            return "WAITING"

        if frame.far_player_box is None:
            return "ARMED"

        fx1, fy1, fx2, fy2 = frame.far_player_box

        # Trophy score
        trophy_score = getattr(frame, "trophy_score", 0.0) or 0.0
        if trophy_score > 0:
            self._trophy_scores.append((trophy_score, now))

        # YOLO toss score (relaxed: 1 consecutive frame → 0.7, 2+ → 1.0)
        yolo_toss = self._update_toss_detection(frame, fy1, now)
        if yolo_toss > 0:
            self._toss_scores.append((yolo_toss, now))

        # MHI secondary toss signal
        mhi_score = getattr(frame, "mhi_toss_score", 0.0)
        if mhi_score > self.MHI_THRESHOLD:
            scaled_mhi = mhi_score * self.MHI_MAX_CONTRIBUTION
            self._toss_scores.append((scaled_mhi, now))

        # Prune deques to event window
        for buf in (self._trophy_scores, self._toss_scores):
            while buf and now - buf[0][1] > self.EVENT_WINDOW_SECONDS:
                buf.popleft()

        max_trophy = max((s for s, _ in self._trophy_scores), default=0.0)
        max_toss   = max((s for s, _ in self._toss_scores),   default=0.0)
        serve_score = self.TROPHY_WEIGHT * max_trophy + self.TOSS_WEIGHT * max_toss

        self.last_serve_scores = {
            "trophy_score": max_trophy,
            "toss_score":   max_toss,
            "mhi_score":    mhi_score,
            "serve_score":  serve_score,
        }

        if serve_score >= self.TRANSITION_SCORE_THRESHOLD:
            # Validate toss height: ball must have appeared above player's head
            if self.toss_min_y_px is not None and self.toss_min_y_px >= fy1:
                print(f"[FAR DEBUG] Toss height invalid: min_y={self.toss_min_y_px:.1f} "
                      f"must be < player_top={fy1}")
                self.toss_min_y_px = None
                return "ARMED"

            toss_h_str = (f"{self.toss_min_y_px:.1f}px (above {fy1})"
                          if self.toss_min_y_px is not None else "MHI only")
            print(f"[FAR TRANSITION] ARMED -> ACTIVE. "
                  f"Serve! Score={serve_score:.2f}  Toss height: {toss_h_str}")
            self._reset_armed_state()
            self._init_active(now)
            return "ACTIVE"

        return "ARMED"

    def _update_toss_detection(self, frame, fy1: float, now: float) -> float:
        """
        Toss detection sub-machine.

        Two independent signals, take the max:

        1. Consecutive-frame signal (existing):
           - 1 frame above head with upward motion → 0.7
           - 2+ consecutive → 1.0
           - Gap > 3 frames resets counter

        2. Parabolic-arc signal (new):
           - Maintains a 1.5 s rolling buffer of (timestamp, cy) for detections
             above the player's head.
           - When ≥ 3 points exist, fits a 1-D parabola (cy = a·t² + b·t + c).
           - A valid toss arc is concave-down (a < 0) with R² ≥ 0.80.
           - R² < 0.80 but ≥ 0.60 → 0.7 bonus; R² ≥ 0.80 → 1.0.
           - Tolerant of gaps: sparse detections that collectively trace a good arc
             still score high.
        """
        if not frame.toss_ball_candidates:
            self.last_toss_ball   = None
            self.toss_gap_frames += 1
            if self.toss_gap_frames > 3:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False
            return 0.0

        best = max(frame.toss_ball_candidates, key=lambda x: x["conf"])
        bx1, by1, bx2, by2 = best["box"]
        cy = (by1 + by2) / 2.0

        is_moving_upward   = False
        is_ball_above_head = cy < fy1

        if self.last_toss_ball is not None:
            dy  = cy - self.last_toss_ball["y"]
            dtt = now - self.last_toss_ball["time"]
            if dy < 0 and dtt > 0:
                is_moving_upward = True

        if is_ball_above_head:
            if self.toss_min_y_px is None or cy < self.toss_min_y_px:
                self.toss_min_y_px = cy
            self._toss_arc_buffer.append((now, cy))

        # Prune arc buffer to 1.5 s window
        while self._toss_arc_buffer and now - self._toss_arc_buffer[0][0] > self.TOSS_ARC_WINDOW_SEC:
            self._toss_arc_buffer.popleft()

        self.last_toss_ball = {"y": cy, "time": now}

        # ── Signal 1: consecutive-frame score ────────────────────────────
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

        # ── Signal 2: parabolic arc score ────────────────────────────────
        arc_score = 0.0
        if len(self._toss_arc_buffer) >= self.TOSS_ARC_MIN_POINTS:
            arc_score = self._score_toss_arc()

        return max(consecutive_score, arc_score)

    def _score_toss_arc(self) -> float:
        """
        Fit a parabola to the buffered (timestamp, cy) points and return a
        confidence score based on concavity and goodness-of-fit.

        Returns 1.0 if concave-down and R² ≥ 0.80,
                0.7 if concave-down and R² ≥ 0.60,
                0.0 otherwise.
        """
        pts  = list(self._toss_arc_buffer)
        ts   = np.array([p[0] for p in pts], dtype=np.float64)
        cys  = np.array([p[1] for p in pts], dtype=np.float64)
        # Normalise time to [0, 1] for numerical stability
        t0, t1 = ts[0], ts[-1]
        if t1 - t0 < 1e-6:
            return 0.0
        ts_norm = (ts - t0) / (t1 - t0)

        coeffs  = np.polyfit(ts_norm, cys, 2)
        a       = coeffs[0]   # positive a → concave-up (U-shape); negative → concave-down (∩)

        if a >= 0:
            return 0.0   # not a toss arc

        cys_pred = np.polyval(coeffs, ts_norm)
        ss_res   = np.sum((cys - cys_pred) ** 2)
        ss_tot   = np.sum((cys - cys.mean()) ** 2)
        r2       = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        if r2 >= self.TOSS_ARC_R2_THRESHOLD:
            return 1.0
        if r2 >= 0.60:
            return 0.7
        return 0.0

    # ── ACTIVE → WAITING ──────────────────────────────────────────────────────

    def _check_active(self, history: deque) -> str:
        frame      = history[-1]
        now        = frame.timestamp
        candidates = frame.active_ball_candidates or []
        cutoff     = now - self.BALL_HISTORY_SEC

        # ── 1. Update far-player tracking buffers for energy bar ──────────
        self._update_player_tracking(frame)

        # ── 2. Ball history update ────────────────────────────────────────
        for c in candidates:
            px, py = c["pixel_center"]
            nearby = sum(
                1 for _, hx, hy in self._all_ball_history
                if math.hypot(px - hx, py - hy) < self.TRACE_NEARBY_PX
            )
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

        # ── 3. Active trace present → point alive ─────────────────────────
        if has_active_trace:
            self.last_active_trace_time = now
            if self.energy_bar_mode:
                print(f"[FAR ACTIVE] Ball trace restored at t={now:.2f}s. "
                      f"Discarding energy bar (was {self.point_energy:.2f}).")
                self.energy_bar_mode = False
                self.point_energy    = 1.0
                self._energy_player_positions.clear()
                self._energy_player_boxes.clear()
                self._energy_gait_y_buffer.clear()
            return "ACTIVE"

        # ── 4. No trace → enter energy bar mode ───────────────────────────
        if not self.energy_bar_mode:
            print(f"[FAR ACTIVE] No ball trace. Entering energy bar mode "
                  f"(anchor={self.last_active_trace_time:.2f}s, now={now:.2f}s).")
            self.energy_bar_mode       = True
            self.energy_bar_start_time = self.last_active_trace_time
            self.point_energy          = 1.0

        dt = 1.0 / self.fps
        energy_delta, status = self._compute_energy_delta(frame, dt)
        self.point_energy = max(0.0, min(1.0, self.point_energy + energy_delta))

        self.last_active_debug.update({
            "energy_bar_mode": self.energy_bar_mode,
            "point_energy":    self.point_energy,
            "energy_status":   status,
        })

        if self.point_energy <= 0.0:
            self.last_transition_time = self.energy_bar_start_time
            elapsed = now - self.active_start_time
            print(f"\n[FAR TRANSITION] ACTIVE -> WAITING (Energy Depleted [{status}]). "
                  f"Lasted {elapsed:.1f}s. Rewind to t={self.energy_bar_start_time:.2f}s.")
            self._reset_active_state()
            return "WAITING"

        return "ACTIVE"

    # ── Player tracking for energy bar ────────────────────────────────────────

    def _update_player_tracking(self, frame) -> None:
        """Append near-player position (world) and box (pixel) to rolling buffers for energy bar."""
        near_box   = frame.near_player_box
        near_world = getattr(frame, "near_player_world", None)
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
        # y2 (box bottom) as gait signal — oscillation pattern same as near-side
        self._energy_gait_y_buffer.append(float(near_box[3]))

    def _compute_energy_delta(self, frame, dt: float):
        """
        Return (energy_delta, status_label) for one frame.

        Priority order mirrors near-side (TransitionEngine._compute_energy_delta).
        Key difference: larger PLAYER_MISSING_GRACE_FRAMES (15 vs 5) tolerates
        the intermittent far-player detections.
        """
        if self._player_missing_frames > self.PLAYER_MISSING_GRACE_FRAMES:
            return -(self.ENERGY_DECAY_MISSING * dt), "MISSING"

        if self._detect_walking_gait():
            return -(self.ENERGY_DECAY_WALKING * dt), "WALKING"

        player_velocity_fts = 0.0
        if len(self._energy_player_positions) >= 5:
            old_p   = self._energy_player_positions[0]
            new_p   = self._energy_player_positions[-1]
            dist_ft = math.hypot(new_p[0] - old_p[0], new_p[1] - old_p[1])
            elapsed = len(self._energy_player_positions) * self.ACTIVE_PLAYER_STRIDE / self.fps
            player_velocity_fts = dist_ft / elapsed if elapsed > 0 else 0.0

        if player_velocity_fts > self.PLAYER_SPRINT_VELOCITY_FTS:
            return (self.ENERGY_BOOST_SPRINT * dt), f"SPRINTING {player_velocity_fts:.1f}ft/s"

        if len(self._energy_player_boxes) >= 5:
            old_b      = self._energy_player_boxes[0]
            new_b      = self._energy_player_boxes[-1]
            box_height = old_b[3] - old_b[1]
            if box_height > 0:
                dw = abs((new_b[2] - new_b[0]) - (old_b[2] - old_b[0]))
                dh = abs((new_b[3] - new_b[1]) - (old_b[3] - old_b[1]))
                if (dw + dh) / box_height > 0.25:
                    return (self.ENERGY_BOOST_SWING * dt), "SWING"

        if player_velocity_fts < self.PLAYER_STILL_VELOCITY_FTS:
            return -(self.ENERGY_DECAY_STILL * dt), f"STILL {player_velocity_fts:.1f}ft/s"
        return (0.1 * dt), f"MOVING {player_velocity_fts:.1f}ft/s"

    def _detect_walking_gait(self) -> bool:
        """Detect walking gait from oscillatory y-movement (identical to near-side)."""
        ys = list(self._energy_gait_y_buffer)
        n  = len(ys)
        if n < self.GAIT_BUFFER_FRAMES * 0.6:
            return False
        if abs(ys[-1] - ys[0]) < self.GAIT_MIN_DRIFT_PX:
            return False
        residuals = [
            ys[i] - (ys[0] + (ys[-1] - ys[0]) * (i / (n - 1)))
            for i in range(n)
        ]
        reversals      = 0
        prev_direction = 0
        for i in range(1, len(residuals)):
            delta = residuals[i] - residuals[i - 1]
            if abs(delta) < 0.5:
                continue
            direction = 1 if delta > 0 else -1
            if prev_direction != 0 and direction != prev_direction:
                reversals += 1
            prev_direction = direction
        return self.GAIT_MIN_REVERSALS <= reversals <= self.GAIT_MAX_REVERSALS

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def _reset_armed_state(self) -> None:
        self._movement_history.clear()
        self.far_ready_start_time   = None
        self._far_ready_miss_frames = 0
        self.toss_consecutive_frames       = 0
        self.toss_gap_frames               = 0
        self.toss_ball_above_head_detected = False
        self.toss_min_y_px                 = None
        self.last_toss_ball                = None
        self._toss_arc_buffer.clear()
        self._trophy_scores.clear()
        self._toss_scores.clear()
        self.last_serve_scores = {
            "trophy_score": 0.0,
            "toss_score":   0.0,
            "mhi_score":    0.0,
            "serve_score":  0.0,
        }

    def _reset_active_state(self) -> None:
        self._movement_history.clear()
        self.far_ready_start_time   = None
        self._far_ready_miss_frames = 0
        self._all_ball_history.clear()
        self._trace_ball_history.clear()
        self.active_start_time      = 0.0
        self.last_active_trace_time = 0.0
        self._player_missing_frames = 0
        self.energy_bar_mode        = False
        self.energy_bar_start_time  = 0.0
        self.point_energy           = 1.0
        self._energy_player_positions.clear()
        self._energy_player_boxes.clear()
        self._energy_gait_y_buffer.clear()
        self._smoothed_player_world = None

    def _init_active(self, now: float) -> None:
        self._reset_active_state()
        self.active_start_time      = now
        self.last_active_trace_time = now
        self.last_transition_time   = None


# ─────────────────────────────────────────────────────────────────────────────
# Debug rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(frame, tel: FarTelemetryFrame, state: str,
                 engine: Optional[FarTransitionEngine] = None,
                 exclusion_zones: Optional[list] = None,
                 active_zone_polygon: Optional[np.ndarray] = None,
                 far_baseline_strip: Optional[Tuple[float, float, float, float]] = None):
    """Debug overlay: state badge, player boxes, balls, exclusion zones, ball trace."""

    # Baseline strip (cyan dashed rectangle) — always visible for alignment
    if far_baseline_strip is not None:
        sx1, sy1, sx2, sy2 = far_baseline_strip
        cv2.rectangle(frame, (int(sx1), int(sy1)), (int(sx2), int(sy2)), (255, 255, 0), 1)

    if state == "ACTIVE" and active_zone_polygon is not None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [active_zone_polygon], (144, 238, 144))
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        cv2.polylines(frame, [active_zone_polygon], True, (0, 200, 0), 1)

    hud_color = (0, 255, 0) if state == "ACTIVE" else (0, 255, 255)
    cv2.putText(frame, f"FAR STATE: {state}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, hud_color, 2)

    # Far player (serving player) — orange box
    if tel.far_player_box:
        x1, y1, x2, y2 = tel.far_player_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)   # orange
        cv2.putText(frame, "FAR-SRV", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

    # Near player (opponent) — blue box
    if tel.near_player_box:
        x1, y1, x2, y2 = tel.near_player_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # Exclusion zones
    if exclusion_zones:
        for x1, y1, x2, y2 in exclusion_zones:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # ARMED: z_box and toss detections
    if state == "ARMED" and tel.z_box:
        x1, y1, x2, y2 = tel.z_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if state == "ARMED" and tel.toss_ball_candidates:
        for ball in tel.toss_ball_candidates:
            bx1, by1, bx2, by2 = ball["box"]
            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)

    # ACTIVE: ball trace
    if state == "ACTIVE" and engine is not None:
        trace = [(px, py) for _, px, py in engine._trace_ball_history]
        n = len(trace)
        if n >= 2:
            for i in range(1, n):
                age       = i / (n - 1)
                color     = (0, int(120 * age), int(255 * age))
                thickness = max(1, int(3 * age))
                cv2.line(frame,
                         (int(trace[i - 1][0]), int(trace[i - 1][1])),
                         (int(trace[i][0]),     int(trace[i][1])),
                         color, thickness, cv2.LINE_AA)
        if n >= 1:
            cv2.circle(frame, (int(trace[-1][0]), int(trace[-1][1])),
                       5, (0, 200, 255), -1, cv2.LINE_AA)

        if tel.active_ball_candidates:
            for ball in tel.active_ball_candidates:
                bx1, by1, bx2, by2 = ball["box"]
                cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)),
                              (0, 255, 255), 2)


def render_debug_panel(state: str, engine: FarTransitionEngine) -> np.ndarray:
    panel = np.ones((300, 500, 3), dtype=np.uint8) * 240

    if state == "ACTIVE":
        _render_active_panel(panel, engine)
    elif state == "ARMED":
        _render_armed_panel(panel, engine)
    else:
        cv2.putText(panel, "WAITING FOR FAR PLAYER", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 1)
    return panel


def _render_active_panel(panel, engine: FarTransitionEngine):
    x0, y, lh, fs = 15, 35, 30, 0.5
    cv2.putText(panel, "FAR ACTIVE — BALL TRACE / ENERGY BAR", (x0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
    d = engine.last_active_debug

    has_trace   = d.get("has_active_trace", False)
    tst         = d.get("time_since_trace", 0.0)
    trace_color = (0, 180, 0) if has_trace else (0, 0, 200)
    trace_label = "YES" if has_trace else f"NO  ({tst:.1f}s ago)"
    cv2.putText(panel, f"Active Trace: {trace_label}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, trace_color, 1)
    y += lh

    energy_mode = d.get("energy_bar_mode", False)
    energy      = d.get("point_energy", 1.0)
    status      = d.get("energy_status", "--")
    bar_label   = "ENERGY BAR" if energy_mode else "Energy (dormant)"
    bar_color   = (0, 180, 0) if not energy_mode else (
        (0, 0, 220) if energy < 0.3 else (0, 165, 255) if energy < 0.6 else (0, 200, 0)
    )
    cv2.putText(panel, f"{bar_label}: {energy:.2f}  [{status}]", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, bar_color, 2 if energy_mode else 1)
    y += 6
    bar_w  = 200
    bg_col = (180, 180, 180) if not energy_mode else (100, 100, 100)
    cv2.rectangle(panel, (x0, y), (x0 + bar_w, y + 14), bg_col, -1)
    if energy_mode and energy > 0:
        fill_col = (0, 0, 220) if energy < 0.3 else (0, 165, 255) if energy < 0.6 else (0, 200, 0)
        cv2.rectangle(panel, (x0, y), (x0 + int(energy * bar_w), y + 14), fill_col, -1)
    y += 24
    cv2.putText(panel, f"Balls detected: {d.get('ball_count', 0)}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 80, 80), 1)


def _render_armed_panel(panel, engine: FarTransitionEngine):
    x0, bar_w, bar_h, lh, label_w = 12, 200, 14, 30, 120
    cv2.putText(panel, "FAR ARMED — Serve Detection", (x0, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    scores = engine.last_serve_scores
    rows = [
        ("Trophy (0.05)",  scores.get("trophy_score", 0.0), (0, 120, 255)),
        ("Toss (0.95)",    scores.get("toss_score",   0.0), (0, 200, 200)),
        ("MHI",            scores.get("mhi_score",    0.0), (180, 180, 0)),
        ("Serve Score",    scores.get("serve_score",  0.0), None),
    ]
    y = 65
    for label, value, color in rows:
        if color is None:
            color = (0, 220, 0) if value >= 0.55 else (0, 140, 255)
        cv2.putText(panel, f"{label}:", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        bx = x0 + label_w
        cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + bar_w, y + 2), (190, 190, 190), -1)
        cv2.rectangle(panel, (bx, y - bar_h + 2),
                      (bx + int(value * bar_w), y + 2), color, -1)
        cv2.putText(panel, f"{value:.3f}", (bx + bar_w + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if label == "Serve Score":
            thresh_x = bx + int(0.55 * bar_w)
            cv2.line(panel, (thresh_x, y - bar_h + 2), (thresh_x, y + 2), (0, 0, 0), 2)
        y += lh


# ─────────────────────────────────────────────────────────────────────────────
# Core segment-collection loop
# ─────────────────────────────────────────────────────────────────────────────

def _collect_far_segments(video_path: str, headless: bool = False,
                           start_frame: int = 0, csv_path: Optional[str] = None):
    """
    Run the far-side pipeline on a single video.

    Returns
    -------
    active_segments : list of (start_sec, end_sec) in source-video time
    point_number    : total serves detected
    csv_path        : path written to
    timestamps      : list of serve start timestamps (seconds)
    """
    if csv_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        csv_path   = os.path.join(video_dir, f"{video_stem}_far_telemetry.csv")

    _probe   = cv2.VideoCapture(video_path)
    orig_fps = _probe.get(cv2.CAP_PROP_FPS)
    _total   = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    _probe.release()
    if orig_fps <= 0 or orig_fps > 300:
        orig_fps = 30.0
    video_duration_sec = _total / orig_fps if _total > 0 else float("inf")

    provider = FarTelemetryProvider(video_path)
    engine   = FarTransitionEngine(fps=provider.fps,
                                   far_baseline_strip=provider.far_baseline_strip)

    _CSV_COLS = [
        "serve", "frame", "timestamp", "state",
        "time_since_trace", "has_active_trace",
        "energy_bar_mode", "point_energy", "energy_status",
        "ball_count",
    ]
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLS)
    csv_writer.writeheader()

    video_time_offset     = start_frame / orig_fps
    active_segments:  List[Tuple[float, float]] = []
    timestamps:       List[float]               = []
    current_segment_start: float = 0.0
    last_telemetry_ts:     float = 0.0
    HIGHLIGHT_END_PAD_SEC = 1.0

    cap           = cv2.VideoCapture(video_path)
    point_number  = 0
    frame_in_point = 0

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[FAR] Seeking to frame {start_frame}")

    WAITING_STRIDE = 3
    interrupted    = False

    try:
        while cap.isOpened():
            success, orig_frame = cap.read()
            if not success:
                break

            frame = cv2.resize(orig_frame, (960, 540), interpolation=cv2.INTER_LINEAR)

            # WAITING stride optimisation: skip inference on 2 of every 3 frames
            skip_inference = (
                provider.current_state == "WAITING"
                and provider.frame_counter % WAITING_STRIDE != 0
                and bool(provider.telemetry_history)
            )

            if skip_inference:
                provider.frame_counter += 1
                last = provider.telemetry_history[-1]
                tel = FarTelemetryFrame(
                    frame_id=provider.frame_counter,
                    timestamp=provider.frame_counter / provider.fps,
                    state="WAITING",
                    far_player_box=last.far_player_box,
                    far_player_world=last.far_player_world,
                    near_player_box=last.near_player_box,
                    toss_ball_candidates=[],
                    active_ball_candidates=[],
                )
                provider.telemetry_history.append(tel)
            else:
                tel = provider.process_frame(frame)

            last_telemetry_ts = tel.timestamp

            new_state = engine.evaluate_transitions(
                provider.telemetry_history,
                provider.current_state,
            )

            old_state = provider.current_state
            if new_state != old_state:
                if new_state == "ACTIVE":
                    point_number    += 1
                    frame_in_point   = 0
                    serve_ts = video_time_offset + tel.timestamp
                    current_segment_start = serve_ts
                    timestamps.append(serve_ts)
                    print(f"[FAR] Serve #{point_number} detected at {serve_ts:.2f}s")
                elif old_state == "ACTIVE":
                    end_t = (engine.last_transition_time
                             if engine.last_transition_time is not None
                             else tel.timestamp)
                    padded_end = min(video_time_offset + end_t + HIGHLIGHT_END_PAD_SEC,
                                     video_duration_sec)
                    active_segments.append((current_segment_start, padded_end))
                provider.update_state(new_state)

            if provider.current_state == "ACTIVE":
                frame_in_point += 1
                _write_csv_row(csv_writer, engine, tel, point_number, frame_in_point)

            if not headless:
                render_frame(frame, tel, provider.current_state, engine,
                             provider.exclusion_zones,
                             provider.active_zone_polygon,
                             provider.far_baseline_strip)
                debug_panel = render_debug_panel(provider.current_state, engine)
                cv2.imshow("Far Anya Pipeline", frame)
                cv2.imshow("Far Debug Panel", debug_panel)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        interrupted = True
        print("\n[FAR] Ctrl-C — creating highlights from completed segments...")

    finally:
        if provider.current_state == "ACTIVE":
            padded_end = min(video_time_offset + last_telemetry_ts + HIGHLIGHT_END_PAD_SEC,
                             video_duration_sec)
            active_segments.append((current_segment_start, padded_end))
        cap.release()
        csv_file.close()
        if not headless:
            cv2.destroyAllWindows()

    print(f"[FAR] {os.path.basename(video_path)}: {point_number} far-side serves, "
          f"{len(active_segments)} segments")
    if interrupted:
        print("[FAR] (interrupted — results cover completed detections only)")

    return active_segments, point_number, csv_path, timestamps


def _write_csv_row(csv_writer, engine: FarTransitionEngine,
                   tel: FarTelemetryFrame, point_number: int, frame_in_point: int):
    d = engine.last_active_debug
    csv_writer.writerow({
        "serve":            point_number,
        "frame":            frame_in_point,
        "timestamp":        round(tel.timestamp, 4),
        "state":            tel.state,
        "time_since_trace": round(d.get("time_since_trace", 0.0), 3),
        "has_active_trace": d.get("has_active_trace", False),
        "energy_bar_mode":  d.get("energy_bar_mode", False),
        "point_energy":     round(d.get("point_energy", 1.0), 3),
        "energy_status":    d.get("energy_status", ""),
        "ball_count":       d.get("ball_count", 0),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_far_anya_pipeline(video_path: str, output_path: Optional[str] = None,
                           headless: bool = False, start_frame: int = 0):
    """
    Run the far-side serve detector on a single video.

    Prints detected serve timestamps to stdout.
    Writes a telemetry CSV and (optionally) a highlights MP4.
    """
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_far_highlights.mp4")

    csv_path = os.path.splitext(output_path)[0] + "_telemetry.csv"

    segments, point_number, _, timestamps = _collect_far_segments(
        video_path, headless, start_frame, csv_path=csv_path
    )

    print("\n" + "=" * 50)
    print(f"  FAR-SIDE SERVES DETECTED: {point_number}")
    print("=" * 50)
    for i, ts in enumerate(timestamps, 1):
        mins  = int(ts // 60)
        secs  = ts % 60
        print(f"  Serve #{i:>3}: {mins}:{secs:05.2f}  ({ts:.2f}s)")
    print("=" * 50)

    if segments:
        create_highlights_ffmpeg(video_path, segments, output_path)
        print(f"\n[FAR] Output video  : {output_path}")
    else:
        print("\n[FAR] No segments to export.")

    print(f"[FAR] Telemetry CSV : {csv_path}")
    return timestamps


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Far-side serve detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ai.far_anya video.mp4
  python -m src.ai.far_anya video.mp4 --output far_out.mp4 --headless
  python -m src.ai.far_anya video.mp4 --headless --start-frame 1800
""",
    )
    parser.add_argument("video", help="Input video file.")
    parser.add_argument("--output",      default=None,
                        help="Output highlights MP4 (default: <video>_far_highlights.mp4).")
    parser.add_argument("--headless",    action="store_true",
                        help="Run without display windows.")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start processing from this frame number (default: 0).")
    args = parser.parse_args()

    run_far_anya_pipeline(
        args.video,
        output_path=args.output,
        headless=args.headless,
        start_frame=args.start_frame,
    )
