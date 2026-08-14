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
                               get_exclusion_zones_from_frames, Point3D, Box,
                               BoxSmoother)
from collections import deque

# ── Far-side tracking constants ───────────────────────────────────────────────
# The far player stands at the top of the frame (small y); the net crosses the
# mid-frame band.  When the far box bottom approaches/passes the net line it is
# likely clipped by the net, so the foot position is extrapolated instead.
FAR_PLAYER_PERSIST_FRAMES = 20    # hold last-known far box across this many dropped frames
NET_OCCLUDE_TOLERANCE_PX  = 25    # box bottom within this of net_y (or below) → assume occlusion
FAR_TOSS_BALL_CONF        = 0.05  # lower than near-side TOSS_BALL_CONF (ball small at distance)
FAR_TOSS_BALL_IMGSZ       = 480
FAR_ACTIVE_BALL_CONF      = 0.10  # lower than near-side ACTIVE_BALL_CONF


@dataclass
class TelemetryFrame:
    frame_id: int
    timestamp: float
    state: str
    near_player_box: Optional[Tuple[int, int, int, int]] = None
    near_player_world: Optional[Tuple[float, float]] = None
    far_player_box: Optional[Tuple[int, int, int, int]] = None   # far-side player (ACTIVE)
    far_player_world: Optional[Tuple[float, float]] = None        # far-side player world (ft)
    near_player_boxes: List[Tuple[int, int, int, int]] = None    # all near-side players
    far_player_boxes: List[Tuple[int, int, int, int]] = None     # all far-side players
    serve_player_box: Optional[Tuple[int, int, int, int]] = None    # serving-side primary player
    serve_player_world: Optional[Tuple[float, float]] = None        # serving-side primary world (ft)
    toss_ball_candidates: List[dict] = None
    active_ball_candidates: List[dict] = None
    trophy_score: float = 0.0          # Probability of trophy/serve pose (ARMED state)
    mhi_toss_score: float = 0.0        # motion-history toss score (far-side fallback)
    pose_landmarks: Any = None         # MediaPipe results (future use)
    player_crop: Any = None            # BGR crop of near player (ACTIVE state, for GaitAnalyzer)
    player_crop_rect: Any = None       # (cx1, cy1, cx2, cy2) frame coords of player_crop
    z_box: Optional[Tuple[int, int, int, int]] = None  # Zone box for ARMED toss detection (x1, y1, x2, y2)


class AnyaTelemetryProvider:
    def __init__(self, video_path: str, serve_side: str = "near",
                 define_far_region: bool = False):
        if serve_side not in ("near", "far"):
            raise ValueError(f"serve_side must be 'near' or 'far', got {serve_side!r}")
        self.serve_side = serve_side
        self.video_path = video_path
        self.define_far_region = define_far_region
        self._init_video_props()

        # Models
        self.player_model = YOLO("yolo26n.pt")
        self.ball_model   = YOLO("/Users/tennis/Documents/Code/Laptop/weights/ball/weights/best.pt")
        self.trophy_model = YOLO(Config.DEFAULT_NEAR_TROPHY_MODEL_PATH)

        # Define the cache path — alongside the input video, not the CWD.
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        self.active_zone_cache_path = os.path.join(video_dir, "active_zone_config.json")
        self.far_region_cache_path  = os.path.join(video_dir, "far_region_config.json")

        # 1. Initialize Court Geometry (at 960x540 resolution)
        self.court_vertices, self.frame_shape = init_court(
            self.video_path,
            analysis_size=(960, 540)
        )

        # 2. Compute Homography (image→world) and its inverse (world→image)
        self.H = self._compute_homography()
        self.H_inv = np.linalg.inv(self.H) if self.H is not None else None
        # Net line pixel-y (used for far-side net-occlusion foot correction)
        self.net_y_px = self._compute_net_y_px()

        # 3. Compute the active-zone polygon from court vertices (used in ACTIVE state)
        self.active_zone_polygon = self._get_or_define_active_zone()

        # 3b. Optional far-side player detection region — restricts which boxes
        # qualify as far-side candidates (e.g. to exclude an adjacent court or a
        # bench visible behind the far baseline).  Only prompted interactively
        # during a far-side pass; a near-side pass just loads the cache if present.
        self.far_region_polygon = self._get_or_define_far_region()

        # 4. Compute static exclusion zones from full video scan (one-time at startup)
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
        # (near_boxes, near_worlds, far_boxes, far_worlds)
        self._cached_player_boxes: Tuple = ([], [], [], [])

        # Last known far player boxes — persists across frames where detection returns none
        self._last_known_far_boxes: List[Tuple[int, int, int, int]] = []

        # ── Far-side serve tracking state (serve_side == "far") ───────────────
        # Box smoother + rolling height buffer for net-occlusion foot correction.
        self._far_box_smoother = BoxSmoother(alpha_pos=0.35, alpha_size=0.12)
        self._far_box_heights: deque = deque(maxlen=30)
        # Persistence of the serving far box across intermittent YOLO misses so a
        # dropped detection does not reset the ARMED ready timer.
        self._far_serve_box:     Optional[Tuple[int, int, int, int]] = None
        self._far_serve_world:   Optional[Tuple[float, float]]       = None
        self._far_persist_count: int = 0
        # Motion-history toss fallback (rolling grayscale ROI above the head).
        self._mhi_roi_buffer: deque = deque(maxlen=15)
        self._mhi_last_score: float = 0.0

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
        points = self._interactive_polygon_selector(
            num_points=8, window_title="Define 8-Sided Active Zone"
        )

        # Cache the points
        with open(self.active_zone_cache_path, 'w') as f:
            json.dump(points.tolist(), f)

        return points

    def _get_or_define_far_region(self) -> Optional[np.ndarray]:
        """
        Loads the cached far-side player detection region, if any.

        This region is independent of (and typically smaller/looser than) the
        active zone — it lets the user fence off where far-side player boxes are
        allowed to come from, e.g. to exclude an adjacent court, a bench, or
        spectators that sit just behind the far baseline and would otherwise be
        mis-tracked as the far server.  A region is optional: if no cache exists
        and we're not in the far-side pass, far-side detection is left unfiltered
        (every far candidate that already passes the active-zone test qualifies).
        """
        if not self.define_far_region and os.path.exists(self.far_region_cache_path):
            try:
                with open(self.far_region_cache_path, 'r') as f:
                    points = json.load(f)
                print(f"[INFO] Loaded far-side detection region from {self.far_region_cache_path}")
                return np.array(points, dtype=np.int32)
            except Exception as e:
                print(f"[WARN] Failed to load far-region cache: {e}")

        # Only prompt interactively during a far-side pass — a near-side pass
        # has no use for this region and shouldn't block on user input for it.
        if self.serve_side != "far":
            return None

        print("[INFO] Defining far-side player detection region. Click 4 points on the frame.")
        points = self._interactive_polygon_selector(
            num_points=4, window_title="Define Far-Side Detection Region"
        )

        with open(self.far_region_cache_path, 'w') as f:
            json.dump(points.tolist(), f)

        return points

    def _interactive_polygon_selector(self, num_points: int = 8,
                                       window_title: str = "Define 8-Sided Active Zone") -> np.ndarray:
        """OpenCV window to collect exactly `num_points` points from the user."""
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
            if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < num_points:
                selected_points.append((x, y))
                # Draw point and line to previous point
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                if len(selected_points) > 1:
                    cv2.line(display_frame, selected_points[-2], selected_points[-1], (0, 255, 0), 2)
                if len(selected_points) == num_points:
                    cv2.line(display_frame, selected_points[-1], selected_points[0], (0, 255, 0), 2)
                cv2.imshow(window_title, display_frame)

        cv2.namedWindow(window_title)
        cv2.setMouseCallback(window_title, mouse_callback)

        print(f"Instructions: Click {num_points} points to define the zone. Press 'q' to confirm once finished.")

        while True:
            cv2.imshow(window_title, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if (key == ord('q') or key == 27) and len(selected_points) == num_points:
                break

        cv2.destroyWindow(window_title)
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

    def _is_in_far_region(self, cx: float, cy: float) -> bool:
        """
        Return True if (cx, cy) lies inside the user-defined far-side detection
        region.  If no region was configured, every point qualifies (unfiltered).
        """
        if self.far_region_polygon is None:
            return True
        return cv2.pointPolygonTest(
            self.far_region_polygon, (float(cx), float(cy)), False
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

    def _compute_net_y_px(self) -> float:
        """Map the net centre (world x=COURT_WIDTH/2, y=COURT_LENGTH/2) to pixel-y."""
        if self.H_inv is None:
            return float(self.height) / 2.0
        net_world = np.array(
            [[[Config.COURT_WIDTH_FT / 2.0, Config.COURT_LENGTH_FT / 2.0]]],
            dtype=np.float32,
        )
        net_px = cv2.perspectiveTransform(net_world, self.H_inv)
        return float(net_px[0][0][1])

    def _estimate_far_feet_y(self, box: Tuple[int, int, int, int]) -> float:
        """
        Best estimate of the far player's foot pixel-y, robust to net occlusion.

        The far player stands behind the far baseline (small y); the net crosses
        the mid-frame band around net_y_px.  When the box bottom is within
        NET_OCCLUDE_TOLERANCE_PX of the net line — or below it — the bottom is
        likely clipped by the net, so the foot y is extrapolated from the (stable)
        box top plus the rolling-median box height from prior un-occluded frames.
        This removes the bottom-edge jitter that otherwise corrupts world_y.
        """
        x1, y1, x2, y2 = box
        box_h = y2 - y1
        occluded = abs(y2 - self.net_y_px) < NET_OCCLUDE_TOLERANCE_PX or y2 > self.net_y_px
        if not occluded and box_h > 0:
            # Only learn heights from clean (un-occluded) detections.
            self._far_box_heights.append(box_h)
        if occluded and self._far_box_heights:
            median_h = sorted(self._far_box_heights)[len(self._far_box_heights) // 2]
            return float(y1 + median_h)
        return float(y2)

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

    def _create_z_box(self, player_box, height_mult: float = 1.5):
        """
        Create zone box for ARMED phase toss detection.
        Bottom line bisects player box vertically (at player center Y).
        Width 2x player width, height height_mult x player height.
        """
        if player_box is None:
            return None
        x1, y1, x2, y2 = player_box
        player_width = x2 - x1
        player_height = y2 - y1
        player_cx = (x1 + x2) / 2.0
        player_cy = (y1 + y2) / 2.0

        z_width = player_width * 2.0
        z_height = player_height * height_mult

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

    def _compute_mhi_toss_score(self, frame, player_box) -> float:
        """
        Motion-History-Image toss score for the region just above the player's
        head — a forgiving secondary toss cue for the far side, where YOLO often
        misses the small ball.

        Compares the current grayscale ROI against the oldest in a rolling buffer;
        the mean absolute difference (soft-thresholded to [0,1]) measures motion
        intensity.  Returns 0.0 when the buffer is short or the box is unknown.
        """
        if player_box is None:
            self._mhi_roi_buffer.clear()
            return 0.0

        x1, y1, x2, y2 = player_box
        fh, fw = frame.shape[:2]
        ph     = y2 - y1

        rx1 = max(0, x1)
        rx2 = min(fw, x2)
        ry1 = max(0, y1 - ph)   # 1x player height above box top
        ry2 = max(0, y1)
        if rx2 <= rx1 or ry2 <= ry1:
            self._mhi_roi_buffer.clear()
            return 0.0

        roi_gray = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        self._mhi_roi_buffer.append(roi_gray)

        if len(self._mhi_roi_buffer) < 3:
            return self._mhi_last_score

        ref, curr = self._mhi_roi_buffer[0], self._mhi_roi_buffer[-1]
        if ref.shape != curr.shape:
            self._mhi_roi_buffer.clear()
            return 0.0

        score = float(np.mean(cv2.absdiff(curr, ref))) / 255.0
        MHI_LOW, MHI_HIGH = 0.02, 0.10   # soft threshold band
        normalized = max(0.0, min(1.0, (score - MHI_LOW) / (MHI_HIGH - MHI_LOW)))
        self._mhi_last_score = normalized
        return normalized

    def _track_players(self, frame):
        """
        Detect all players in-frame and split them into near-side / far-side groups.

        A detection only qualifies as a tracked player if its feet (bottom-center
        of its bounding box) fall inside the active-zone polygon — this rejects
        spectators, ballkids, and players on adjacent courts.

        Among qualifying detections, a player is near-side if their feet are
        closer (in homography world-feet) to the near baseline (world_y = 0) than
        to the far baseline (world_y = COURT_LENGTH_FT), and far-side otherwise.
        This supports doubles, where two players can occupy the same side.

        Far-side candidates are additionally gated by `far_region_polygon`, an
        optional user-defined region (see `_get_or_define_far_region`) that fences
        off where far-side players are allowed to be detected from.

        Returns (near_boxes, near_worlds, far_boxes, far_worlds):
          near_boxes  : list of (x1, y1, x2, y2), ordered by increasing distance
                        to the near baseline.
          near_worlds : list of (wx, wy), parallel to near_boxes.
          far_boxes   : list of (x1, y1, x2, y2), ordered by increasing distance
                        to the far baseline.
          far_worlds  : list of (wx, wy), parallel to far_boxes, using the
                        net-occlusion-corrected foot position.

        For far-side serving, the primary (closest-to-far-baseline) box is passed
        through a BoxSmoother to suppress net-induced bottom-edge jitter.
        """
        results = self.player_model(frame, verbose=False, conf=0.5, imgsz=Config.PLAYER_IMGSZ)

        if not (results and results[0].boxes):
            return [], [], [], []

        near_candidates = []
        far_candidates  = []
        for b in results[0].boxes:
            if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            box = (x1, y1, x2, y2)
            cx  = (x1 + x2) / 2.0

            # Classify side using the net-occlusion-corrected foot y, so a box
            # clipped by the net is not mis-binned.
            feet_y = self._estimate_far_feet_y(box)
            wx, wy = self.get_world_pos(cx, feet_y)

            # Feet (bottom-center) must lie within the active zone.
            if not self._is_in_active_zone(cx, feet_y):
                continue

            dist_near = abs(wy)
            dist_far  = abs(wy - Config.COURT_LENGTH_FT)
            if dist_near <= dist_far:
                near_candidates.append((dist_near, box, (wx, wy)))
            else:
                # User-defined far-side detection region (if configured) is an
                # additional gate beyond the active zone — rejects boxes from an
                # adjacent court or bench that would otherwise pass as far-side.
                if self._is_in_far_region(cx, feet_y):
                    far_candidates.append((dist_far, box, (wx, wy)))

        near_candidates.sort(key=lambda c: c[0])
        far_candidates.sort(key=lambda c: c[0])

        near_boxes  = [c[1] for c in near_candidates]
        near_worlds = [c[2] for c in near_candidates]
        far_boxes   = [c[1] for c in far_candidates]
        far_worlds  = [c[2] for c in far_candidates]

        # Smooth the primary far box (the serving player) to kill net jitter.
        if self.serve_side == "far" and far_boxes:
            far_boxes[0] = self._far_box_smoother.smooth_box_xyxy(*far_boxes[0])

        return near_boxes, near_worlds, far_boxes, far_worlds

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

        # 1. Track near/far players (supports multiple players per side, e.g. doubles).
        # In ACTIVE state, run the player models every ACTIVE_PLAYER_STRIDE frames and
        # hold the cached results in between — player positions change slowly and
        # the boxes are only used for ball-detection filtering and the near-player timer.
        if (self.current_state == "ACTIVE"
                and self.frame_counter % self.ACTIVE_PLAYER_STRIDE != 0
                and self._cached_player_boxes[0]):
            near_boxes, near_worlds, far_boxes, far_worlds = self._cached_player_boxes
        else:
            near_boxes, near_worlds, far_boxes, far_worlds = self._track_players(frame)
            if self.current_state == "ACTIVE":
                if far_boxes:
                    self._last_known_far_boxes = far_boxes
                else:
                    far_boxes = self._last_known_far_boxes
            else:
                self._last_known_far_boxes = []

            self._cached_player_boxes = (near_boxes, near_worlds, far_boxes, far_worlds)

        telemetry.near_player_boxes = near_boxes
        telemetry.far_player_boxes  = far_boxes

        # Primary (closest-to-baseline) player on each side, kept for callers that
        # only care about a single near/far player (serve tracking, gait analysis).
        p_box    = near_boxes[0] if near_boxes else None
        p_world  = near_worlds[0] if near_worlds else None
        far_box  = far_boxes[0] if far_boxes else None
        far_world = far_worlds[0] if far_worlds else None
        telemetry.near_player_box   = p_box
        telemetry.near_player_world = p_world
        telemetry.far_player_box    = far_box
        telemetry.far_player_world  = far_world

        # ── Serving-side primary player ───────────────────────────────────────
        # For far-side serving, hold the last-known box across intermittent YOLO
        # misses (FAR_PLAYER_PERSIST_FRAMES) so a dropped detection does not reset
        # the ARMED ready timer.  Persistence is refreshed only when the player
        # model actually ran this frame (i.e. not on a strided cache hit).
        if self.serve_side == "near":
            serve_box   = p_box
            serve_world = p_world
        else:
            serve_box   = far_box
            serve_world = far_world
            if serve_box is not None:
                self._far_serve_box     = serve_box
                self._far_serve_world   = serve_world
                self._far_persist_count = 0
            elif self._far_serve_box is not None:
                self._far_persist_count += 1
                if self._far_persist_count <= FAR_PLAYER_PERSIST_FRAMES:
                    serve_box   = self._far_serve_box
                    serve_world = self._far_serve_world   # stale but stable over short gaps
                    telemetry.far_player_box   = serve_box
                    telemetry.far_player_world = serve_world

        telemetry.serve_player_box   = serve_box
        telemetry.serve_player_world = serve_world

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

        # 2b. ARMED State Detectors — operate on the serving-side player box.
        is_far = self.serve_side == "far"
        if self.current_state == "ARMED" and serve_box:
            # Create zone box for toss detection (taller on the far side — the
            # toss arc spans more of the frame relative to the small far box).
            z_box = self._create_z_box(serve_box, height_mult=2.5 if is_far else 1.5)
            telemetry.z_box = z_box

            nx1, ny1, nx2, ny2 = serve_box
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

            # Far side: motion-history toss fallback (the small ball is often
            # missed by YOLO; head-region motion is a forgiving secondary signal).
            if is_far:
                telemetry.mhi_toss_score = self._compute_mhi_toss_score(frame, serve_box)

            # Toss ball detection — ROI above player box.  Far side searches a
            # taller ROI at lower confidence (small, faint ball at distance).
            toss_conf  = FAR_TOSS_BALL_CONF if is_far else Config.TOSS_BALL_CONF
            toss_imgsz = FAR_TOSS_BALL_IMGSZ if is_far else Config.TOSS_BALL_IMGSZ
            up_mult    = 1.5 if is_far else 1.0
            rx1 = max(0,  int(nx1 - pw / 2))
            ry1 = max(0,  int(ny1 - up_mult * ph))
            rx2 = min(fw, int(nx2 + pw / 2))
            ry2 = min(fh, int(ny1 + ph / 2))
            roi = frame[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                ball_res = self.ball_model(roi, verbose=False, conf=toss_conf,
                                           imgsz=toss_imgsz)
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
                            not self._is_in_player_box(ball_cx, ball_cy, serve_box, padding=15)):
                            telemetry.toss_ball_candidates.append({
                                "box":  (ball_x, ball_y, rx1 + cx2, ry1 + cy2),
                                "conf": float(b.conf[0]),
                            })
        elif self.current_state == "ARMED" and is_far:
            # No serving box this frame — keep the MHI buffer coherent.
            telemetry.mhi_toss_score = self._compute_mhi_toss_score(frame, None)

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
            active_conf = FAR_ACTIVE_BALL_CONF if self.serve_side == "far" else Config.ACTIVE_BALL_CONF
            ball_res = self.ball_model(
                frame, verbose=False, conf=active_conf, imgsz=Config.BALL_IMGSZ,
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
            self._last_known_far_boxes = []

        if new_state == "WAITING" and old_state == "ACTIVE":
            # Reset far-side persistence/smoothing so the next point starts clean.
            self._far_serve_box     = None
            self._far_serve_world   = None
            self._far_persist_count = 0
            self._far_box_smoother.reset()

        if new_state == "ARMED" and old_state != "ARMED":
            now = self.frame_counter / self.fps
            # Clear dynamic zones and start fresh collection
            self.dynamic_exclusion_zones = []
            self._armed_frame_buffer     = []
            self._armed_entry_time       = now
            self._armed_collection_done  = False
            self._last_trophy_score      = 0.0   # don't carry score from previous ARMED entry
            self._mhi_roi_buffer.clear()
            self._mhi_last_score         = 0.0
            print("[INFO] ARMED entered — starting dynamic exclusion zone collection (0-0.5s)")
