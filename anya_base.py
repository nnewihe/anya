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
from serve_stgcn import ServeSTGCNDetector

@dataclass
class TelemetryFrame:
    frame_id: int
    timestamp: float
    state: str
    near_player_box: Optional[Tuple[int, int, int, int]] = None
    near_player_world: Optional[Tuple[float, float]] = None
    far_player_box: Optional[Tuple[int, int, int, int]] = None    # far-side player
    far_player_world: Optional[Tuple[float, float]] = None        # far-side player, world coords
    toss_ball_candidates: List[dict] = None
    active_ball_candidates: List[dict] = None
    trophy_score: float = 0.0          # Probability of trophy/serve pose (ARMED state)
    far_serve_score: float = 0.0       # Far-side serve probability from serve_stgcn.pt (ARMED state)
    pose_landmarks: Any = None         # MediaPipe results (future use)
    player_crop: Any = None            # BGR crop of near player (ACTIVE state, for GaitAnalyzer)
    player_crop_rect: Any = None       # (cx1, cy1, cx2, cy2) frame coords of player_crop
    z_box: Optional[Tuple[int, int, int, int]] = None  # Zone box for ARMED toss detection (x1, y1, x2, y2)


class AnyaTelemetryProvider:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._init_video_props()

        # Models
        self.player_model = YOLO("yolo26n.pt")
        self.ball_model   = YOLO("/Users/tennis/Documents/Code/Laptop/weights/ball/weights/best.pt")
        self.trophy_model = YOLO(Config.DEFAULT_NEAR_TROPHY_MODEL_PATH)

        # Define the cache path — alongside the input video, not the CWD.
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        self.active_zone_cache_path = os.path.join(video_dir, "active_zone_config.json")

        # 1. Initialize Court Geometry (at 960x540 resolution)
        self.court_vertices, self.frame_shape = init_court(
            self.video_path,
            analysis_size=(960, 540)
        )

        # 2. Compute Homography (image→world)
        self.H = self._compute_homography()

        # 3. Compute the active-zone polygon from court vertices (used in ACTIVE state)
        self.active_zone_polygon = self._get_or_define_active_zone()

        # 4. Compute static exclusion zones from full video scan (one-time at startup)
        print("\n[INFO] Scanning video for static exclusion zones...")
        try:
            self.static_exclusion_zones = create_auto_exclusion_zones(
                self.video_path, self.ball_model,
                num_frames=50,
                conf=0.04,
                eps=12,
                padding=0,
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

    def process_frame(self, frame, orig_frame=None) -> TelemetryFrame:
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
        # Far player here is just the incidental "other player" output of
        # _track_near_player (full-frame detection); FarSideTelemetryProvider
        # runs its own dedicated far-baseline tracking for the far-side pass.
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
                        padding=0,
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
            if self.frame_counter % self.ARMED_TROPHY_STRIDE == 0:
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


class FarSideTelemetryProvider:
    """
    Telemetry provider for the separate far-side serve-detection pass.

    Mirrors AnyaTelemetryProvider's WAITING/ACTIVE mechanism, pointed at the
    far baseline (Config.COURT_LENGTH_FT) instead of the near one (y=0):
      - Far player is detected full-frame and selected by world-distance
        proximity to the far baseline plus a singles-sideline gate (the
        mirror of _track_near_player's near-baseline selection) — no static/
        hand-clicked search ROI.
      - ARMED entry is gated by the far player settling into the ready band
        behind the far baseline (WAITING -> ARMED), exactly like the near side.

    Unlike the near side, ARMED has no trophy-pose or ball-toss signal: the
    far-side serve decision is made solely by serve_stgcn.pt classifying a
    rolling window of upper-body joint-graph kinematics from the far player's
    padded box (see ServeSTGCNDetector, FarSideTransitionEngine._check_armed).

    Reuses the cached court/homography/active-zone artifacts from the
    near-side pass on this same video (run the near-side pass at least once
    first so the active-zone polygon cache exists).
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._init_video_props()

        self.player_model = YOLO("yolo26n.pt")
        self.ball_model   = YOLO("/Users/tennis/Documents/Code/Laptop/weights/ball/weights/best.pt")
        self.far_serve_detector = ServeSTGCNDetector()

        video_dir = os.path.dirname(os.path.abspath(video_path))
        self.active_zone_cache_path = os.path.join(video_dir, "active_zone_config.json")

        self.court_vertices, self.frame_shape = init_court(
            self.video_path, analysis_size=(960, 540)
        )
        self.H = self._compute_homography()
        self.active_zone_polygon = self._load_polygon(
            self.active_zone_cache_path,
            "Run the near-side pass on this video at least once first to define it.",
        )

        print("\n[INFO] Scanning video for static exclusion zones (far-side pass)...")
        try:
            self.static_exclusion_zones = create_auto_exclusion_zones(
                self.video_path, self.ball_model,
                num_frames=50,
                conf=0.04,
                eps=12,
                padding=0,
                ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                analysis_size=(960, 540),
            )
            print(f"[INFO] Found {len(self.static_exclusion_zones)} static exclusion zone(s)")
        except Exception as e:
            print(f"[WARN] Could not compute static exclusion zones: {e}")
            self.static_exclusion_zones = []

        # Dynamic exclusion zones — recomputed on each ARMED entry (mirrors near side)
        self.dynamic_exclusion_zones: List = []
        self._armed_frame_buffer: List = []
        self._armed_entry_time: Optional[float] = None
        self._armed_collection_done: bool = False
        self.ARMED_DYNAMIC_COLLECTION_SEC = 0.5
        self.ARMED_DYNAMIC_SAMPLE_FRAMES  = 5

        self.current_state = "WAITING"
        self.frame_counter = 0
        buffer_size = int(self.fps * Config.TELEMETRY_BUFFER_SECONDS)
        self.telemetry_history = deque(maxlen=buffer_size)

        # Cached far-player box/world for ACTIVE-state striding
        self.ACTIVE_PLAYER_STRIDE = 4
        self._cached_player_boxes: Tuple = (None, None, [])   # (far_box, far_world, all_boxes)

        # Far-player world-position smoothing: at far-court distance, a few
        # pixels of bounding-box jitter on the feet gets amplified by the
        # homography into several feet of world-coordinate noise — enough to
        # make the ready-band test flicker frame-to-frame even when the player
        # is standing still. A short rolling average absorbs that without
        # changing the near side's algorithm (which doesn't need it at close
        # range).
        self.FAR_WORLD_SMOOTH_WINDOW_SEC = 0.3
        self._far_world_history: deque = deque()   # (t, wx, wy)

    @staticmethod
    def _load_polygon(cache_path: str, hint: str) -> np.ndarray:
        """Load an 8-point polygon (list of [x, y] pairs) cached at cache_path."""
        if not os.path.exists(cache_path):
            raise RuntimeError(f"No cached polygon at {cache_path}. {hint}")
        with open(cache_path, "r") as f:
            points = json.load(f)
        return np.array(points, dtype=np.int32)

    def _init_video_props(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = 960
        self.height = 540
        cap.release()

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

    @property
    def exclusion_zones(self) -> List:
        return self.static_exclusion_zones + self.dynamic_exclusion_zones

    def _is_in_active_zone(self, cx: float, cy: float) -> bool:
        return cv2.pointPolygonTest(
            self.active_zone_polygon, (float(cx), float(cy)), False
        ) >= 0

    def _is_in_player_box(self, ball_cx, ball_cy, player_box, padding=15):
        """Check if ball center is within player bounding box + padding."""
        if player_box is None:
            return False
        x1, y1, x2, y2 = player_box
        return (x1 - padding <= ball_cx <= x2 + padding and
                y1 - padding <= ball_cy <= y2 + padding)

    def _is_in_any_player_box(self, ball_cx, ball_cy, player_boxes, padding=0):
        """Check if ball center is within ANY of the given player boxes + padding."""
        return any(self._is_in_player_box(ball_cx, ball_cy, box, padding=padding)
                   for box in (player_boxes or []))

    def _track_far_player(self, frame):
        """
        Detect all players full-frame and select the far player, mirroring
        AnyaTelemetryProvider._track_near_player's near-baseline selection:

          1. Feet closer (world distance) to the far baseline
             (Config.COURT_LENGTH_FT) than to the near baseline (y=0).
          2. Feet x within the singles sidelines (0..COURT_WIDTH_FT), extended
             indefinitely beyond the baseline, plus a small homography-
             tolerance padding (FAR_PLAYER_X_PAD_FT) — this is what actually
             rejects spectators/ball kids standing laterally outside the
             court, regardless of how far behind the baseline they are.

        Returns (far_box, far_world, all_boxes) — all_boxes is every detected
        player box this frame (near players, doubles partners, etc.), used to
        exclude ball detections from any player's body, not just the gating
        far player's.
        """
        results = self.player_model(frame, verbose=False, conf=0.2, imgsz=Config.PLAYER_IMGSZ)

        if not (results and results[0].boxes):
            return None, None, []

        candidates = []
        for b in results[0].boxes:
            if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cx = (x1 + x2) / 2.0
            wx, wy = self.get_world_pos(cx, y2)
            candidates.append((x1, y1, x2, y2, wx, wy))

        if not candidates:
            return None, None, []

        pad = Config.FAR_PLAYER_X_PAD_FT
        far_candidates = [
            c for c in candidates
            if (abs(c[5] - Config.COURT_LENGTH_FT) < abs(c[5]) and       # criterion 1
                -pad <= c[4] <= Config.COURT_WIDTH_FT + pad)             # criterion 2
        ]

        if not far_candidates:
            return None, None, []

        all_boxes = [c[:4] for c in candidates]
        far = min(far_candidates, key=lambda c: abs(c[5] - Config.COURT_LENGTH_FT))
        far_box   = far[:4]
        far_world = (far[4], far[5])
        return far_box, far_world, all_boxes

    def _smoothed_far_world(self, world, now: float) -> Optional[Tuple[float, float]]:
        """
        Rolling-average smoothing for far_player_world over
        FAR_WORLD_SMOOTH_WINDOW_SEC. A missed detection this frame doesn't
        reset the window — it just isn't added — so a single dropped frame
        decays out naturally rather than abruptly invalidating the estimate.
        Returns None only once the window is empty (no recent detection at all).
        """
        if world is not None:
            self._far_world_history.append((now, world[0], world[1]))
        while (self._far_world_history and
               now - self._far_world_history[0][0] > self.FAR_WORLD_SMOOTH_WINDOW_SEC):
            self._far_world_history.popleft()

        if not self._far_world_history:
            return None

        n = len(self._far_world_history)
        avg_wx = sum(s[1] for s in self._far_world_history) / n
        avg_wy = sum(s[2] for s in self._far_world_history) / n
        return (avg_wx, avg_wy)

    def process_frame(self, frame, orig_frame=None) -> TelemetryFrame:
        self.frame_counter += 1
        timestamp = self.frame_counter / self.fps

        telemetry = TelemetryFrame(
            frame_id=self.frame_counter,
            timestamp=timestamp,
            state=self.current_state,
            toss_ball_candidates=[],
            active_ball_candidates=[],
        )

        # 1. Track far player (mirrors AnyaTelemetryProvider's near-player
        #    tracking/striding, selecting by far-baseline proximity instead).
        #    The world position is smoothed (see _smoothed_far_world) before
        #    being cached/used — raw per-frame homography noise at far-court
        #    distance is large enough to break the ready-band dwell check.
        if (self.current_state == "ACTIVE"
                and self.frame_counter % self.ACTIVE_PLAYER_STRIDE != 0
                and self._cached_player_boxes[0] is not None):
            f_box, f_world, all_player_boxes = self._cached_player_boxes
        else:
            f_box, raw_world, all_player_boxes = self._track_far_player(frame)
            f_world = self._smoothed_far_world(raw_world, timestamp)
            self._cached_player_boxes = (f_box, f_world, all_player_boxes)
        telemetry.far_player_box   = f_box
        telemetry.far_player_world = f_world

        # 2. ARMED — buffer frames for dynamic exclusion zone computation (0-0.5s window)
        if self.current_state == "ARMED":
            now_t = timestamp
            if (not self._armed_collection_done
                    and self._armed_entry_time is not None):
                elapsed = now_t - self._armed_entry_time
                if elapsed <= self.ARMED_DYNAMIC_COLLECTION_SEC:
                    self._armed_frame_buffer.append(frame.copy())
                elif len(self._armed_frame_buffer) >= 1:
                    self.dynamic_exclusion_zones = get_exclusion_zones_from_frames(
                        self._armed_frame_buffer,
                        self.ball_model,
                        sample_size=self.ARMED_DYNAMIC_SAMPLE_FRAMES,
                        conf=0.05,
                        eps=5,
                        min_samples=15,
                        padding=0,
                        ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                    )
                    self._armed_collection_done = True
                    self._armed_frame_buffer    = []
                    print(f"[INFO] Dynamic exclusion zones (far side): "
                          f"{len(self.dynamic_exclusion_zones)} zone(s)")

        # 2b. ARMED detector — serve_stgcn.pt classification of the far player's
        # padded box. This is the sole far-side serve signal (no toss/trophy):
        # the crop is taken at native video resolution (orig_frame), scaled up
        # from the box's 960x540 coordinates rather than resized/interpolated,
        # since the far player is small enough at this distance that resize
        # interpolation would blur exactly the pose detail the model needs.
        if self.current_state == "ARMED" and f_box:
            fx1, fy1, fx2, fy2 = f_box
            pw, ph = fx2 - fx1, fy2 - fy1

            if orig_frame is not None:
                oh, ow = orig_frame.shape[:2]
                sx, sy = ow / float(self.width), oh / float(self.height)
                nx1, ny1, nx2, ny2 = fx1 * sx, fy1 * sy, fx2 * sx, fy2 * sy
                npw, nph = nx2 - nx1, ny2 - ny1
                pad_x, pad_y = npw * Config.FAR_SERVE_LSTM_PAD, nph * Config.FAR_SERVE_LSTM_PAD
                cx1 = max(0, int(nx1 - pad_x)); cy1 = max(0, int(ny1 - pad_y))
                cx2 = min(ow, int(nx2 + pad_x)); cy2 = min(oh, int(ny2 + pad_y))
                crop = orig_frame[cy1:cy2, cx1:cx2]
            else:
                fh, fw = frame.shape[:2]
                pad_x, pad_y = pw * Config.FAR_SERVE_LSTM_PAD, ph * Config.FAR_SERVE_LSTM_PAD
                cx1 = max(0, int(fx1 - pad_x)); cy1 = max(0, int(fy1 - pad_y))
                cx2 = min(fw, int(fx2 + pad_x)); cy2 = min(fh, int(fy2 + pad_y))
                crop = frame[cy1:cy2, cx1:cx2]

            self.far_serve_detector.update(crop)
            telemetry.far_serve_score = self.far_serve_detector.score()

        # 3. ACTIVE — whole-court ball detection. Unlike the near side, ALL
        #    detected player boxes are excluded here (every phase, whenever
        #    boxes are available) — racket/arm motion is otherwise picked up
        #    as a ball candidate too readily at this scale.
        if self.current_state == "ACTIVE":
            ball_res = self.ball_model(
                frame, verbose=False, conf=Config.ACTIVE_BALL_CONF, imgsz=Config.BALL_IMGSZ,
            )
            if ball_res and ball_res[0].boxes:
                for b in ball_res[0].boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

                    if (self._is_in_active_zone(bcx, bcy) and
                            not _is_in_exclusion_zone(bcx, bcy, self.exclusion_zones) and
                            not self._is_in_any_player_box(bcx, bcy, all_player_boxes, padding=0)):
                        telemetry.active_ball_candidates.append({
                            "box":          (bx1, by1, bx2, by2),
                            "conf":         float(b.conf[0]),
                            "pixel_center": (bcx, bcy),
                        })

        self.telemetry_history.append(telemetry)
        return telemetry

    def update_state(self, new_state: str):
        old_state = self.current_state
        self.current_state = new_state

        if new_state == "ARMED" and old_state != "ARMED":
            now = self.frame_counter / self.fps
            self.dynamic_exclusion_zones = []
            self._armed_frame_buffer     = []
            self._armed_entry_time       = now
            self._armed_collection_done  = False
            self.far_serve_detector.reset()
            print("[INFO] ARMED entered (far side) — starting dynamic exclusion zone collection (0-0.5s)")
