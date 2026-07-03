"""
far_anya_base.py
================
Core telemetry provider for far-side serve detection.

Key differences vs anya_base.py:
  - Far player is the primary tracked player (server); near player is tracked
    only for ball exclusion and ACTIVE energy-bar driving.
  - WAITING: far-player box-top position is monitored against the far-baseline
    pixel y for the WAITING → ARMED steadiness check.
  - ARMED: toss ball detection with relaxed confidence thresholds (ball is
    smaller at distance); trophy pose as minor signal; MHI frame-diff fallback.
  - ACTIVE: whole-court ball detection + near-player tracking for energy bar.
  - Extended far-player persistence (FAR_PLAYER_PERSIST_FRAMES) to bridge
    intermittent detections.
  - Net-occlusion correction: foot y estimated from rolling-median box height
    when the box bottom drifts near the net pixel line.
"""

import json
import os

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ultralytics import YOLO

from utilities import (
    BoxSmoother, Config, _is_in_exclusion_zone,
    init_court, init_far_player_roi,
    create_auto_exclusion_zones, get_exclusion_zones_from_frames,
    Point3D, Box,
)


# ── Far-side constants ────────────────────────────────────────────────────────
FAR_PLAYER_PERSIST_FRAMES = 20     # frames to hold last-known box on detection drop
FAR_ACTIVE_BALL_CONF      = 0.10   # lower than near-side 0.15 (small ball at distance)
FAR_TOSS_BALL_CONF        = 0.05   # lower than near-side 0.10
FAR_TOSS_BALL_IMGSZ       = 480
FAR_ACTIVE_BALL_IMGSZ     = 960
FAR_ACTIVE_ZONE_CACHE     = "far_active_zone_config.json"
FAR_BASELINE_BAND_CACHE   = "far_baseline_band_config.json"
NET_OCCLUDE_TOLERANCE_PX  = 25     # px: box bottom within this of net_y → assume occlusion


# ── Telemetry data container ──────────────────────────────────────────────────

@dataclass
class FarTelemetryFrame:
    frame_id:               int
    timestamp:              float
    state:                  str
    far_player_box:         Optional[Tuple[int, int, int, int]] = None  # server
    far_player_world:       Optional[Tuple[float, float]]       = None  # (wx, wy) ft
    near_player_box:        Optional[Tuple[int, int, int, int]] = None  # receiver (excl + energy)
    near_player_world:      Optional[Tuple[float, float]]       = None  # (wx, wy) ft
    toss_ball_candidates:   List[dict]                          = None
    active_ball_candidates: List[dict]                          = None
    far_player_foot_y_px:   Optional[float]                     = None  # corrected foot pixel-y
    trophy_score:           float                               = 0.0
    z_box:                  Optional[Tuple[int, int, int, int]] = None
    mhi_toss_score:         float                               = 0.0


# ── Telemetry Provider ────────────────────────────────────────────────────────

class FarTelemetryProvider:
    """
    Per-frame sensor layer for far-side serve detection.

    Tracked player roles
    --------------------
    far_player : the server at the far baseline — primary detection target.
                 Uses a user-defined ROI for focused YOLO inference.
    near_player: the receiver at the near baseline — secondary; used only for
                 ball-box exclusion (ARMED/ACTIVE) and energy-bar driving (ACTIVE).

    Player persistence
    ------------------
    The far player is often partially occluded by the net or blends into the
    background.  Up to FAR_PLAYER_PERSIST_FRAMES consecutive missed detections
    are tolerated by carrying the last-known box forward.

    Net-occlusion correction
    ------------------------
    When the far-player box bottom drifts within NET_OCCLUDE_TOLERANCE_PX of the
    net pixel line (or below it), the visible box bottom is not the feet — the
    feet are estimated from the box top plus the rolling-median box height.
    """

    def __init__(self, video_path: str, court_frame_idx: int = 300,
                 active_zone_polygon: Optional[np.ndarray] = None):
        self.video_path = video_path
        self._init_video_props()

        # ── Models ─────────────────────────────────────────────────────────
        self.player_model = YOLO("yolo26n.pt")
        self.ball_model   = YOLO("/Users/tennis/Documents/Code/Laptop/weights/ball/weights/best.pt")

        # ── Court geometry ─────────────────────────────────────────────────
        self.court_vertices, self.frame_shape = init_court(
            self.video_path, target_idx=court_frame_idx, analysis_size=(960, 540)
        )
        self.far_player_roi = init_far_player_roi(
            self.video_path, analysis_size=(960, 540)
        )
        self.H     = self._compute_homography()
        self.H_inv = np.linalg.inv(self.H)

        self.net_y_px = self._compute_net_y_px()
        print(f"[FAR] Net pixel-y estimate: {self.net_y_px:.1f}px")

        self.baseline_front_px, self.baseline_behind_px = self._get_or_define_baseline_band()
        print(f"[FAR] Baseline pixel band: top={self.baseline_behind_px:.1f}px  "
              f"bottom={self.baseline_front_px:.1f}px")

        # ── Active zone polygon ────────────────────────────────────────────
        if active_zone_polygon is not None:
            self.active_zone_polygon = active_zone_polygon
        else:
            self.active_zone_polygon = self._get_or_define_active_zone()

        # ── Static exclusion zones ─────────────────────────────────────────
        print("\n[FAR] Scanning video for static exclusion zones...")
        try:
            self.static_exclusion_zones = create_auto_exclusion_zones(
                self.video_path, self.ball_model,
                num_frames=50, conf=0.04, eps=8, padding=1,
                ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                analysis_size=(960, 540),
            )
            print(f"[FAR] {len(self.static_exclusion_zones)} static exclusion zone(s)")
        except Exception as e:
            print(f"[FAR] WARN: Could not compute static exclusion zones: {e}")
            self.static_exclusion_zones = []
        self.dynamic_exclusion_zones: List = []

        # ── Dynamic exclusion zone buffering (0.5 s after ARMED entry) ────
        self._armed_frame_buffer:    List            = []
        self._armed_entry_time:      Optional[float] = None
        self._armed_collection_done: bool            = False
        self.ARMED_DYNAMIC_COLLECTION_SEC = 0.5
        self.ARMED_DYNAMIC_SAMPLE_FRAMES  = 5

        # ── State & telemetry buffer ───────────────────────────────────────
        self.current_state = "WAITING"
        self.frame_counter = 0
        buffer_size = int(self.fps * Config.TELEMETRY_BUFFER_SECONDS)
        self.telemetry_history = deque(maxlen=buffer_size)

        # ── Far-player persistence ─────────────────────────────────────────
        self._last_known_far_box:   Optional[Tuple[int, int, int, int]] = None
        self._last_known_far_world: Optional[Tuple[float, float]]       = None
        self._far_persist_counter:  int                                  = 0
        self._far_box_heights:      deque                                = deque(maxlen=30)
        self._far_box_smoother      = BoxSmoother(alpha_pos=0.50, alpha_size=0.12)

        # ── Near-player cache (for ball exclusion + energy bar) ────────────
        self._last_near_box: Optional[Tuple[int, int, int, int]] = None

        # ── ACTIVE player-detection striding ──────────────────────────────
        self.ACTIVE_PLAYER_STRIDE  = 2   # every other frame during ACTIVE
        self._cached_player_boxes: Tuple = (None, None, None)  # (far, far_world, near)
        self.energy_bar_active: bool = False  # set by far_anya.py; retained for energy bar logic

        # ── Toss-box persistence ───────────────────────────────────────────
        self._last_z_box: Optional[Tuple[int, int, int, int]] = None

        # ── MHI (Motion History Image) for secondary toss signal ───────────
        self.MHI_BUFFER_FRAMES = 15
        self._mhi_roi_buffer: deque = deque(maxlen=self.MHI_BUFFER_FRAMES)
        self._mhi_last_score:  float = 0.0

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def far_baseline_y(self) -> float:
        """Average pixel-y of the far baseline (TL and TR court corners)."""
        BL, BR, TR, TL = self.court_vertices
        return (TL[1] + TR[1]) / 2.0

    @property
    def exclusion_zones(self) -> List:
        return self.static_exclusion_zones + self.dynamic_exclusion_zones

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _init_video_props(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width  = 960
        self.height = 540
        cap.release()

    # ── Geometry ──────────────────────────────────────────────────────────────

    def _compute_homography(self):
        BL, BR, TR, TL = self.court_vertices
        dst_pts = np.array([
            [0,                     0                        ],
            [Config.COURT_WIDTH_FT, 0                        ],
            [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT   ],
            [0,                     Config.COURT_LENGTH_FT   ],
        ], dtype=np.float32)
        src_pts = np.array([BL, BR, TR, TL], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H

    def _compute_net_y_px(self) -> float:
        """Map net centre world position to pixel-y via inverse homography."""
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

    def _get_or_define_baseline_band(self) -> Tuple[float, float]:
        """Load baseline pixel band from cache, or launch interactive selector."""
        if os.path.exists(FAR_BASELINE_BAND_CACHE):
            try:
                with open(FAR_BASELINE_BAND_CACHE, "r") as f:
                    data = json.load(f)
                front_px, behind_px = float(data["front_px"]), float(data["behind_px"])
                print(f"[FAR] Loaded baseline band from {FAR_BASELINE_BAND_CACHE}")
                return front_px, behind_px
            except Exception as e:
                print(f"[FAR] WARN: Could not load baseline band cache: {e}")

        print("[FAR] Define the baseline band. "
              "Click the BOTTOM line (inside court), then the TOP line (behind baseline). "
              "Press Q to confirm.")
        front_px, behind_px = self._interactive_baseline_band_selector()
        with open(FAR_BASELINE_BAND_CACHE, "w") as f:
            json.dump({"front_px": front_px, "behind_px": behind_px}, f)
        return front_px, behind_px

    def _interactive_baseline_band_selector(self) -> Tuple[float, float]:
        """
        Show a video frame and let the user click two horizontal lines:
          Click 1 — bottom of band (larger pixel-y, inside court side)
          Click 2 — top of band   (smaller pixel-y, behind baseline)
        Draws lines and a translucent fill between them. Press Q to confirm.
        """
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            cap2 = cv2.VideoCapture(self.video_path)
            ret, frame = cap2.read()
            cap2.release()
        if not ret:
            raise RuntimeError("Could not read frame for baseline band selection.")

        frame   = cv2.resize(frame, (960, 540))
        display = frame.copy()
        clicks: List[int] = []   # stores pixel-y values

        LABEL    = ["BOTTOM (inside court)", "TOP (behind baseline)"]
        COLOR    = [(0, 200, 255), (255, 200, 0)]   # cyan-ish, gold

        def _redraw():
            nonlocal display
            display = frame.copy()
            # Draw instruction
            idx = min(len(clicks), 1)
            msg = (f"Click {LABEL[idx]}  ({len(clicks)}/2)"
                   if len(clicks) < 2 else "Press Q to confirm")
            cv2.putText(display, msg, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, msg, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
            # Fill band if both clicked
            if len(clicks) == 2:
                y_top = min(clicks)
                y_bot = max(clicks)
                overlay = display.copy()
                cv2.rectangle(overlay, (0, y_top), (display.shape[1], y_bot), (0, 200, 100), -1)
                cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            # Draw clicked lines
            for i, y in enumerate(clicks):
                cv2.line(display, (0, y), (display.shape[1], y), COLOR[i], 2, cv2.LINE_AA)
                cv2.putText(display, LABEL[i], (10, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR[i], 2, cv2.LINE_AA)
            cv2.imshow("Far Baseline Band", display)

        def cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
                clicks.append(y)
                _redraw()

        cv2.namedWindow("Far Baseline Band")
        cv2.setMouseCallback("Far Baseline Band", cb)
        _redraw()
        while True:
            if cv2.waitKey(20) & 0xFF in (ord("q"), 27) and len(clicks) == 2:
                break
        cv2.destroyWindow("Far Baseline Band")

        front_px  = float(max(clicks))   # larger y = bottom = inside court
        behind_px = float(min(clicks))   # smaller y = top   = behind baseline
        return front_px, behind_px

    def _estimate_far_feet_y(self, box: Tuple[int, int, int, int]) -> float:
        """
        Return the best estimate of the far player's foot pixel-y.

        When the box bottom is within NET_OCCLUDE_TOLERANCE_PX of the net line
        (or below it), the bottom is likely clipped.  Extrapolate from box top
        using the rolling-median box height seen in un-occluded prior frames.
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

    # ── Exclusion / player-box helpers ────────────────────────────────────────

    def _is_in_player_box(self, bx, by, player_box, padding: int = 10) -> bool:
        if player_box is None:
            return False
        x1, y1, x2, y2 = player_box
        return (x1 - padding <= bx <= x2 + padding and
                y1 - padding <= by <= y2 + padding)

    # ── Player tracking ───────────────────────────────────────────────────────

    def _track_far_player_roi(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """Detect the far player inside the user-defined ROI (focused inference)."""
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
        Detect all players and classify as far vs near via world-space geometry.

        Far player  — ROI detection tried first; falls back to the full-frame
                      candidate closest to the far baseline in world space.
        Near player — full-frame candidate closest to the near baseline (wy ≈ 0).

        Returns (far_box, far_world, near_box).
        far_world uses net-occlusion corrected foot position.
        """
        far_box_roi = self._track_far_player_roi(frame)

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

        # Near player: closest to near baseline (wy ≈ 0), in near half
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
            far_candidates = [
                c for c in candidates
                if (near_box is None or c[:4] != near_box) and
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

    # ── Toss detection helpers ────────────────────────────────────────────────

    def _create_z_box(self, player_box) -> Optional[Tuple[int, int, int, int]]:
        """Toss detection zone: 2× player width, 2.5× player height, bottom at player centre."""
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

    def _compute_mhi_toss_score(self, frame, player_box) -> float:
        """
        Motion History Image score for the region immediately above the player's head.

        Compares the current ROI against the oldest frame in a 15-frame rolling
        buffer.  Mean absolute pixel difference (normalised to [0,1]) represents
        motion intensity.  Returned values > 0.3 indicate a toss candidate.
        """
        if player_box is None:
            self._mhi_roi_buffer.clear()
            return 0.0

        x1, y1, x2, y2 = player_box
        fh, fw = frame.shape[:2]
        ph     = y2 - y1

        rx1 = max(0, x1)
        rx2 = min(fw, x2)
        ry1 = max(0, y1 - ph)
        ry2 = max(0, y1)

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

        MHI_LOW, MHI_HIGH = 0.02, 0.10
        normalized = max(0.0, min(1.0, (score - MHI_LOW) / (MHI_HIGH - MHI_LOW)))
        self._mhi_last_score = normalized
        return normalized

    # ── Main frame-processing entry point ────────────────────────────────────

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

            if far_box is not None:
                self._last_known_far_box   = far_box
                self._last_known_far_world = far_world
                self._far_persist_counter  = 0
            else:
                self._far_persist_counter += 1
                if self._far_persist_counter <= FAR_PLAYER_PERSIST_FRAMES:
                    far_box   = self._last_known_far_box
                    far_world = self._last_known_far_world

            if near_box is not None:
                self._last_near_box = near_box

            self._cached_player_boxes = (far_box, far_world, near_box)

        tel.far_player_box   = far_box
        tel.far_player_world = far_world
        if far_world is not None:
            pt = np.array([[[far_world[0], far_world[1]]]], dtype=np.float32)
            tel.far_player_foot_y_px = float(cv2.perspectiveTransform(pt, self.H_inv)[0][0][1])

        resolved_near_box   = near_box if near_box is not None else self._last_near_box
        tel.near_player_box = resolved_near_box
        if resolved_near_box is not None:
            nx1, ny1, nx2, ny2 = resolved_near_box
            tel.near_player_world = self.get_world_pos(
                (nx1 + nx2) / 2.0, float(ny2)
            )

        # ── 2. ARMED: dynamic exclusion zone buffering (first 0.5 s) ─────
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
                        conf=0.05, eps=5, min_samples=15, padding=1,
                        ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                    )
                    self._armed_collection_done = True
                    self._armed_frame_buffer    = []
                    print(f"[FAR] Dynamic exclusion: {len(self.dynamic_exclusion_zones)} zone(s)")

        # ── 2b. ARMED state detectors ─────────────────────────────────────
        if self.current_state == "ARMED":
            tel.mhi_toss_score = self._compute_mhi_toss_score(frame, far_box)

            if far_box is not None:
                fx1, fy1, fx2, fy2 = far_box
                fh, fw = frame.shape[:2]
                fcx = (fx1 + fx2) // 2

                # YOLO toss ball — static 200×100 ROI centred on player, bottom at head
                rx1 = max(0,  fcx - 100)
                rx2 = min(fw, fcx + 100)
                ry1 = max(0,  fy1 - 100)
                ry2 = min(fh, fy1)
                self._last_z_box = (rx1, ry1, rx2, ry2)
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
                            if (not _is_in_exclusion_zone(ball_cx, ball_cy, self.exclusion_zones) and
                                    not self._is_in_player_box(ball_cx, ball_cy, far_box, padding=15)):
                                tel.toss_ball_candidates.append({
                                    "box":  (rx1 + cx1, ry1 + cy1, rx1 + cx2, ry1 + cy2),
                                    "conf": float(b.conf[0]),
                                })

            # Always show the last known toss box (persists when far_box drops out)
            tel.z_box = self._last_z_box

        # ── 3. ACTIVE: whole-court ball detection ─────────────────────────
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

    # ── State update ──────────────────────────────────────────────────────────

    def update_state(self, new_state: str):
        old_state = self.current_state
        self.current_state = new_state

        if new_state == "WAITING" and old_state == "ACTIVE":
            self._last_known_far_box   = None
            self._last_known_far_world = None
            self._far_persist_counter  = 0
            self._far_box_smoother.reset()

        if new_state != "ARMED" and old_state == "ARMED":
            self._last_z_box = None

        if new_state == "ARMED" and old_state != "ARMED":
            now = self.frame_counter / self.fps
            self.dynamic_exclusion_zones = []
            self._armed_frame_buffer     = []
            self._armed_entry_time       = now
            self._armed_collection_done  = False
            self._mhi_roi_buffer.clear()
            self._mhi_last_score         = 0.0
            self._last_z_box             = None
            print("[FAR] ARMED entered — starting dynamic exclusion zone collection (0–0.5s)")
