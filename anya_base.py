"""
anya_base.py
=============
Core telemetry provider. Handles homography, exclusion zones, and runs the 
necessary detectors (YOLO / MediaPipe) based on the current state.
Maintains a 5-second rolling buffer of telemetry data.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
import os
import random
import math

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
from sklearn.cluster import DBSCAN
from utilities import (Config, _is_in_exclusion_zone, init_court,
                               create_auto_exclusion_zones,
                               get_exclusion_zones_from_frames, Point3D, Box)
from collections import deque

@dataclass
class TelemetryFrame:
    frame_id: int
    timestamp: float
    state: str
    near_player_box: Optional[Tuple[int, int, int, int]] = None
    near_player_world: Optional[Tuple[float, float]] = None
    far_player_box: Optional[Tuple[int, int, int, int]] = None   # far-side player (ACTIVE)
    toss_ball_candidates: List[dict] = None
    active_ball_candidates: List[dict] = None
    trophy_score: float = 0.0          # Probability of trophy/serve pose (ARMED state)
    pose_landmarks: Any = None         # MediaPipe results (future use)
    player_crop: Any = None            # BGR crop of near player (ACTIVE state, for GaitAnalyzer)
    player_crop_rect: Any = None       # (cx1, cy1, cx2, cy2) frame coords of player_crop
    z_box: Optional[Tuple[int, int, int, int]] = None  # Zone box for ARMED toss detection (x1, y1, x2, y2)


class AnyaTelemetryProvider:
    def __init__(
        self,
        video_path: str,
        court_vertices=None,
        active_zone_polygon=None,
        ball_model_path: str = None,
        player_model_path: str = None,
        trophy_model_path: str = None,
        skip_exclusion_zones: bool = False,
    ):
        self.video_path = video_path
        self._init_video_props()

        _ball_path   = ball_model_path   or "/Users/tennis/Documents/Code/Laptop/weights/ball/weights/best.pt"
        _player_path = player_model_path or "yolo26n.pt"
        # trophy_model_path=None  → explicitly disabled (caller set it to None)
        # trophy_model_path unset → falls back to Config default
        _trophy_path = (None if trophy_model_path is None
                        else (trophy_model_path or Config.DEFAULT_NEAR_TROPHY_MODEL_PATH))

        # Models
        self.player_model = YOLO(_player_path)
        self.ball_model   = YOLO(_ball_path)
        # Trophy model is only used in ARMED state; skip loading if path is None or unavailable.
        if _trophy_path is None:
            self.trophy_model = None
            print("[INFO] Trophy model skipped (trophy_model_path=None).")
        else:
            try:
                self.trophy_model = YOLO(_trophy_path)
            except Exception as _e:
                print(f"[WARN] Could not load trophy model ({_e}); ARMED-state trophy scoring disabled.")
                self.trophy_model = None

        # Define the cache path
        self.active_zone_cache_path = "active_zone_config.json"

        # 1. Initialize Court Geometry (at 960x540 resolution)
        if court_vertices is not None:
            self.court_vertices = [tuple(p) for p in court_vertices]
            self.frame_shape    = (540, 960, 3)
            print("[COURT] Using provided court vertices (skipping interactive selection).")
        else:
            self.court_vertices, self.frame_shape = init_court(
                self.video_path,
                analysis_size=(960, 540)
            )

        # 2. Compute Homography (image→world)
        self.H = self._compute_homography()

        # 3. Active-zone polygon — use provided value, or load/select interactively.
        if active_zone_polygon is not None:
            self.active_zone_polygon = np.array(active_zone_polygon, dtype=np.int32)
            print("[ZONE] Using provided active-zone polygon (skipping interactive selection).")
        else:
            self.active_zone_polygon = self._get_or_define_active_zone()

        # 4. Compute static exclusion zones from full video scan (one-time at startup)
        if skip_exclusion_zones:
            print("[INFO] Skipping static exclusion zone scan (skip_exclusion_zones=True).")
            self.static_exclusion_zones = []
        else:
            print("\n[INFO] Scanning video for static exclusion zones...")
            try:
                self.static_exclusion_zones = create_auto_exclusion_zones(
                    self.video_path, self.ball_model,
                    num_frames=50,
                    conf=0.04,
                    eps=12,
                    padding=5,
                    ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                    analysis_size=(960, 540),
                )
                print(f"[INFO] Found {len(self.static_exclusion_zones)} static exclusion zone(s)")
            except Exception as e:
                print(f"[WARN] Could not compute static exclusion zones: {e}")
                self.static_exclusion_zones = []

        # Dynamic exclusion zones — recomputed on each ARMED entry
        self.dynamic_exclusion_zones: List = []

        # ------------------------------------------------------------------
        # Dynamic exclusion zone state
        # ------------------------------------------------------------------
        self._armed_frame_buffer: List = []
        self._armed_entry_time: Optional[float] = None
        self._armed_collection_done: bool = False

        self.ARMED_DYNAMIC_COLLECTION_SEC = 0.5  # collect for 0.5s after ARMED entry
        self.ARMED_DYNAMIC_SAMPLE_FRAMES = 5     # sample this many frames from buffer

        # State & Buffer
        self.current_state = "WAITING"
        self.frame_counter = 0
        buffer_size = int(self.fps * Config.TELEMETRY_BUFFER_SECONDS)
        self.telemetry_history = deque(maxlen=buffer_size)

        # Cached player boxes for ACTIVE-state striding (player tracked every N frames)
        self.ACTIVE_PLAYER_STRIDE = 4
        self._cached_player_boxes: Tuple = (None, None, None)  # (near_box, near_world, far_box)

        # Last known far player box — persists across frames where ROI detection returns None
        self._last_known_far_box: Optional[Tuple[int, int, int, int]] = None

        # Trophy model stride (run every N frames in ARMED state)
        self.ARMED_TROPHY_STRIDE = 2
        self._last_trophy_score: float = 0.0


        # MediaPipe Pose for ACTIVE state
        """
        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1, # 0 might be needed for Pi 5 if dropping frames
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        """
    

    def _get_or_define_active_zone(self) -> np.ndarray:
        """Loads cached polygon or triggers interactive UI to define 8 points."""
        if os.path.exists(self.active_zone_cache_path):
            try:
                with open(self.active_zone_cache_path, 'r') as f:
                    points = json.load(f)
                print(f"[INFO] Loaded 8-sided active zone from {self.active_zone_cache_path}")
                return np.array(points, dtype=np.int32)
            except Exception as e:
                print(f"[WARN] Failed to load cached polygon: {e}")

        # If no cache exists, run the interactive selector
        print("[INFO] Defining new 8-sided active zone. Click 8 points on the frame.")
        points = self._interactive_polygon_selector()
        
        # Cache the points
        with open(self.active_zone_cache_path, 'w') as f:
            json.dump(points.tolist(), f)
        
        return points

    def _interactive_polygon_selector(self) -> np.ndarray:
        """OpenCV window to collect exactly 8 points from the user."""
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Could not read frame for polygon definition.")

        # Resample to analysis size
        frame = cv2.resize(frame, (960, 540))
        display_frame = frame.copy()
        selected_points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 8:
                selected_points.append((x, y))
                # Draw point and line to previous point
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                if len(selected_points) > 1:
                    cv2.line(display_frame, selected_points[-2], selected_points[-1], (0, 255, 0), 2)
                if len(selected_points) == 8:
                    cv2.line(display_frame, selected_points[-1], selected_points[0], (0, 255, 0), 2)
                cv2.imshow("Define 8-Sided Active Zone", display_frame)

        cv2.namedWindow("Define 8-Sided Active Zone")
        cv2.setMouseCallback("Define 8-Sided Active Zone", mouse_callback)

        print("Instructions: Click 8 points to define the zone. Press 'q' to confirm once finished.")
        
        while True:
            cv2.imshow("Define 8-Sided Active Zone", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if (key == ord('q') or key == 27) and len(selected_points) == 8:
                break
        
        cv2.destroyWindow("Define 8-Sided Active Zone")
        return np.array(selected_points, dtype=np.int32)

    def _init_video_props(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Frames are resampled to 960x540 in run_anya.py
        self.width = 960
        self.height = 540
        cap.release()

    @property
    def exclusion_zones(self) -> List:
        """Combined static + dynamic exclusion zones for filtering."""
        return self.static_exclusion_zones + self.dynamic_exclusion_zones

    def _compute_active_zone_polygon(self) -> np.ndarray:
        """
        Build a 6-vertex pixel-space polygon that defines where ball detections
        are accepted in the ACTIVE phase.

        Vertex order (clockwise from near-left):
          BL  →  BR  →  BR-150px  →  TR-100px  →  TL-100px  →  BL-150px

        Where:
          BL, BR = near baseline / doubles-alley corners  (first two court vertices)
          TR, TL = far  baseline / doubles-alley corners  (last two court vertices)
          -Npx   = shifted N pixels upward in image space (lower y value)
        """
        BL, BR, TR, TL = self.court_vertices
        pts = np.array([
            [BL[0],       BL[1]      ],   # near-left  baseline
            [BR[0],       BR[1]      ],   # near-right baseline
            [BR[0],       BR[1] - 150],   # near-right +150px up
            [TR[0],       TR[1] - 200],   # far-right  +200px up
            [TL[0],       TL[1] - 200],   # far-left   +200px up
            [BL[0],       BL[1] - 150],   # near-left  +150px up
        ], dtype=np.int32)
        return pts

    def _is_in_active_zone(self, cx: float, cy: float) -> bool:
        """Return True if (cx, cy) lies inside or on the active-zone polygon."""
        return cv2.pointPolygonTest(
            self.active_zone_polygon, (float(cx), float(cy)), False
        ) >= 0

    def _stub_init_court(self):
        # Stub for the interactive init_court function
        return [(0,0), (100,0), (100,100), (0,100)], (1080, 1920, 3)

    def _compute_homography(self):
        BL, BR, TR, TL = self.court_vertices
        dst_pts = np.array([
            [0, 0], [Config.COURT_WIDTH_FT, 0],
            [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT], [0, Config.COURT_LENGTH_FT],
        ], dtype=np.float32)
        src_pts = np.array([BL, BR, TR, TL], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H

    def get_world_pos(self, px_x, px_y):
        if self.H is None: return 0.0, 0.0
        pt_px = np.array([[[px_x, px_y]]], dtype=np.float32)
        pt_world = cv2.perspectiveTransform(pt_px, self.H)
        return pt_world[0][0][0], pt_world[0][0][1]

    def _is_in_player_box(self, ball_cx, ball_cy, player_box, padding=15):
        """Check if ball center is within player bounding box + padding."""
        if player_box is None:
            return False
        x1, y1, x2, y2 = player_box
        return (x1 - padding <= ball_cx <= x2 + padding and
                y1 - padding <= ball_cy <= y2 + padding)

    def _create_z_box(self, player_box):
        """
        Create zone box for ARMED phase toss detection.
        Bottom line bisects player box vertically (at player center Y).
        Width 2x player width, height 1.5x player height.
        """
        if player_box is None:
            return None
        x1, y1, x2, y2 = player_box
        player_width = x2 - x1
        player_height = y2 - y1
        player_cx = (x1 + x2) / 2.0
        player_cy = (y1 + y2) / 2.0

        z_width = player_width * 2.0
        z_height = player_height * 1.5

        # Bottom of z_box at player center Y (bisects vertically)
        z_x1 = player_cx - z_width / 2.0
        z_x2 = player_cx + z_width / 2.0
        z_y2 = player_cy
        z_y1 = z_y2 - z_height

        # Cap at top of frame
        z_y1 = max(0, z_y1)

        return (int(z_x1), int(z_y1), int(z_x2), int(z_y2))

    def _is_in_z_box(self, ball_cx, ball_cy, z_box):
        """Check if ball center is within z_box."""
        if z_box is None:
            return False
        x1, y1, x2, y2 = z_box
        return x1 <= ball_cx <= x2 and y1 <= ball_cy <= y2

    def _track_near_player(self, frame):
        """
        Detect all players and classify using world-space distance from each baseline.

        Near player: detection whose feet (world_y) are closest to the near baseline (y=0).
        Far player:  detection (excluding near) whose feet are closest to the far baseline (y=78 ft).

        Near-player candidates are pre-filtered to:
          1. Feet closer to the near baseline than the far baseline (world_y < COURT_LENGTH/2).
          2. Feet x within the lateral span of the near baseline (0..COURT_WIDTH_FT) plus a
             small homography-tolerance padding (NEAR_PLAYER_X_PAD_FT).

        Returns (near_box, near_world, far_box).
        """
        results = self.player_model(frame, verbose=False, conf=0.5, imgsz=Config.PLAYER_IMGSZ)

        if not (results and results[0].boxes):
            return None, None, None

        candidates = []
        for b in results[0].boxes:
            if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cx = (x1 + x2) / 2.0
            wx, wy = self.get_world_pos(cx, y2)
            candidates.append((x1, y1, x2, y2, wx, wy))

        if not candidates:
            return None, None, None

        # ── Near-player candidate filter ──────────────────────────────────
        # Criterion 1: feet closer to near baseline (y=0) than far baseline (y=78 ft)
        # Criterion 2: feet x within the near-baseline width (+/- padding)
        pad = Config.NEAR_PLAYER_X_PAD_FT
        near_candidates = [
            c for c in candidates
            if (abs(c[5]) < abs(c[5] - Config.COURT_LENGTH_FT) and          # criterion 1
                -pad <= c[4] <= Config.COURT_WIDTH_FT + pad)                 # criterion 2
        ]

        if not near_candidates:
            return None, None, None

        # Near player: smallest |world_y| (closest to near baseline at y=0)
        near = min(near_candidates, key=lambda c: abs(c[5]))
        near_box   = near[:4]
        near_world = (near[4], near[5])

        # Far player: anyone who isn't the near player.
        rest = [c for c in candidates if c[:4] != near_box]
        far_box = None
        if rest:
            far = min(rest, key=lambda c: abs(c[5] - Config.COURT_LENGTH_FT))
            far_box = far[:4]

        return near_box, near_world, far_box

    def process_frame(self, frame) -> TelemetryFrame:
        self.frame_counter += 1
        timestamp = self.frame_counter / self.fps

        telemetry = TelemetryFrame(
            frame_id=self.frame_counter,
            timestamp=timestamp,
            state=self.current_state,
            toss_ball_candidates=[],
            active_ball_candidates=[]
        )

        # 1. Track near/far player.
        # In ACTIVE state, run the player models every ACTIVE_PLAYER_STRIDE frames and
        # hold the cached results in between — the player position changes slowly and
        # the boxes are only used for ball-detection filtering and the near-player timer.
        # Far player is detected via the user-defined ROI (Config.FAR_PLAYER_IMGSZ) in
        # ACTIVE state; full-frame detection is used in WAITING/ARMED states.
        if (self.current_state == "ACTIVE"
                and self.frame_counter % self.ACTIVE_PLAYER_STRIDE != 0
                and self._cached_player_boxes[0] is not None):
            p_box, p_world, far_box = self._cached_player_boxes
        else:
            p_box, p_world, far_box = self._track_near_player(frame)
            if self.current_state == "ACTIVE":
                if far_box is not None:
                    self._last_known_far_box = far_box
                else:
                    far_box = self._last_known_far_box
            else:
                self._last_known_far_box = None

            self._cached_player_boxes = (p_box, p_world, far_box)
        telemetry.near_player_box   = p_box
        telemetry.near_player_world = p_world
        telemetry.far_player_box    = far_box

        # 2. ARMED State — buffer frames for dynamic exclusion zone computation (0-0.5s window)
        if self.current_state == "ARMED":
            now_t = self.frame_counter / self.fps
            if (not self._armed_collection_done
                    and self._armed_entry_time is not None):
                elapsed = now_t - self._armed_entry_time
                if elapsed <= self.ARMED_DYNAMIC_COLLECTION_SEC:
                    # Still inside the 0.5-second collection window — store frame
                    self._armed_frame_buffer.append(frame.copy())
                elif len(self._armed_frame_buffer) >= 1:
                    # Collection window closed — compute dynamic zones from sampled frames
                    # using same DBSCAN logic as static zones
                    self.dynamic_exclusion_zones = get_exclusion_zones_from_frames(
                        self._armed_frame_buffer,
                        self.ball_model,
                        sample_size=self.ARMED_DYNAMIC_SAMPLE_FRAMES,
                        conf=0.05,
                        eps=5,
                        min_samples=15,
                        padding=5,
                        ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                    )
                    self._armed_collection_done = True
                    self._armed_frame_buffer    = []  # free memory
                    print(f"[INFO] Dynamic exclusion zones: {len(self.dynamic_exclusion_zones)} zone(s)")

        # 2b. ARMED State Detectors
        if self.current_state == "ARMED" and p_box:
            # Create zone box for toss detection
            z_box = self._create_z_box(p_box)
            telemetry.z_box = z_box

            nx1, ny1, nx2, ny2 = p_box
            pw, ph = nx2 - nx1, ny2 - ny1
            fh, fw = frame.shape[:2]

            # Trophy pose classification — run every ARMED_TROPHY_STRIDE frames,
            # carry forward the last score in between (pose changes slowly).
            if self.trophy_model is not None and self.frame_counter % self.ARMED_TROPHY_STRIDE == 0:
                pad_x = int(pw * Config.DEFAULT_TROPHY_PAD)
                pad_y = int(ph * Config.DEFAULT_TROPHY_PAD)
                tx1 = max(0, nx1 - pad_x); ty1 = max(0, ny1 - pad_y)
                tx2 = min(fw, nx2 + pad_x); ty2 = min(fh, ny2 + pad_y)
                trophy_crop = frame[ty1:ty2, tx1:tx2]
                if trophy_crop.size > 0:
                    tr = self.trophy_model(trophy_crop, verbose=False, imgsz=Config.TROPHY_IMGSZ)
                    if tr and hasattr(tr[0], "probs") and tr[0].probs is not None:
                        idx = Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX
                        if idx < len(tr[0].probs.data):
                            self._last_trophy_score = float(tr[0].probs.data[idx])
            telemetry.trophy_score = self._last_trophy_score

            # Toss ball detection — ROI above player box
            rx1 = max(0,  int(nx1 - pw / 2))
            ry1 = max(0,  int(ny1 - ph))
            rx2 = min(fw, int(nx2 + pw / 2))
            ry2 = min(fh, int(ny1 + ph / 2))
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                ball_res = self.ball_model(roi, verbose=False, conf=Config.TOSS_BALL_CONF,
                                           imgsz=Config.TOSS_BALL_IMGSZ)
                if ball_res and ball_res[0].boxes:
                    for b in ball_res[0].boxes:
                        cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
                        ball_x = rx1 + cx1
                        ball_y = ry1 + cy1
                        ball_cx = (ball_x + rx1 + cx2) / 2.0
                        ball_cy = (ball_y + ry1 + cy2) / 2.0

                        # Filter: must be in z_box, not in exclusion zones, and not in player box
                        if (self._is_in_z_box(ball_cx, ball_cy, z_box) and
                            not _is_in_exclusion_zone(ball_cx, ball_cy, self.exclusion_zones) and
                            not self._is_in_player_box(ball_cx, ball_cy, p_box, padding=15)):
                            telemetry.toss_ball_candidates.append({
                                "box":  (ball_x, ball_y, rx1 + cx2, ry1 + cy2),
                                "conf": float(b.conf[0]),
                            })

        # 3. ACTIVE State Detectors
        if self.current_state == "ACTIVE":
            # MediaPipe on Cropped Player
            if p_box:
                nx1, ny1, nx2, ny2 = p_box
                # Add padding for pose tracking
                pad = 30
                cx1, cy1 = max(0, nx1 - pad), max(0, ny1 - pad)
                cx2, cy2 = min(frame.shape[1], nx2 + pad), min(frame.shape[0], ny2 + pad)

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    telemetry.player_crop = crop
                    telemetry.player_crop_rect = (cx1, cy1, cx2, cy2)
                    """
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pose_results = self.pose.process(crop_rgb)
                    # Translate landmarks back to full frame coordinates if needed here
                    telemetry.pose_landmarks = pose_results.pose_landmarks
                    """

        # Whole-court ball detection (plain YOLO — no internal tracker).  Track
        # identity/coherence is assigned downstream by the trajectory-coherent tracker
        # in TransitionEngine.  Only runs in ACTIVE (point-end authority).
        if self.current_state == "ACTIVE":
            ball_res = self.ball_model(
                frame, verbose=False, conf=Config.ACTIVE_BALL_CONF, imgsz=Config.BALL_IMGSZ,
            )

            if ball_res and ball_res[0].boxes:
                for b in ball_res[0].boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

                    # Filter: must be inside active-zone polygon and outside exclusion zones.
                    # Player-box exclusion is intentionally NOT applied here: during a rally
                    # the ball spends a lot of time overlapping a player's body box, and
                    # dropping those detections starves the tracker.  The trajectory-coherent
                    # tracker's gating + coherence checks reject body-borne false positives.
                    if (self._is_in_active_zone(bcx, bcy) and
                            not _is_in_exclusion_zone(bcx, bcy, self.exclusion_zones)):
                        telemetry.active_ball_candidates.append({
                            "box":          (bx1, by1, bx2, by2),
                            "conf":         float(b.conf[0]),
                            "pixel_center": (bcx, bcy),
                        })

        # Append to buffer
        self.telemetry_history.append(telemetry)
        return telemetry

    def update_state(self, new_state: str):
        old_state = self.current_state
        self.current_state = new_state

        if old_state == "ACTIVE" and new_state != "ACTIVE":
            self._last_known_far_box = None

        if new_state == "ARMED" and old_state != "ARMED":
            now = self.frame_counter / self.fps
            # Clear dynamic zones and start fresh collection
            self.dynamic_exclusion_zones = []
            self._armed_frame_buffer     = []
            self._armed_entry_time       = now
            self._armed_collection_done  = False
            self._last_trophy_score      = 0.0   # don't carry score from previous ARMED entry
            print("[INFO] ARMED entered — starting dynamic exclusion zone collection (0-0.5s)")
