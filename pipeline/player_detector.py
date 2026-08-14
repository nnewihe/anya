import cv2
import numpy as np
import math
import os
import json
from collections import deque
from ultralytics import YOLO

# Real-world tennis court dimensions (singles), in feet.
COURT_WIDTH_FT = 27.0
COURT_LENGTH_FT = 78.0

# Assumed far-player box height in pixels, used in place of the net-occluded
# raw detection bottom edge.
FAR_BOX_HEIGHT_PX = 100

# Far-player track smoothing / gap-holding.
FAR_TRACK_ALPHA = 0.15      # EMA weight applied to each new detection (lower = more conservative)
FAR_TRACK_MAX_MISSES = 30   # consecutive missed frames before the track goes inactive


class AnyaTwoStateSystem:
    def __init__(self, video_path):
        self.video_path = video_path
        self.model = YOLO('yolov8n.pt')
        self.court_points = []
        self.active_zone_points = []

        self.far_baseline_midpoint = None

        # Exclusion zones: rectangles (x1, y1, x2, y2) in full-frame coordinates.
        # Any player whose center falls inside a zone is discarded. Mirrors the
        # ball exclusion-zone filtering in the run_anya pipeline (see
        # utilities._is_in_exclusion_zone) and is used to reject court-side
        # false positives such as spectators, coaches, or ball kids.
        self.exclusion_points = []
        self.exclusion_zones = []

        # Ground-plane homography (image px -> world ft), computed once the
        # court corners are known. Used to classify far- vs near-side
        # detections in world space.
        self.H = None

        # Smoothed far-player track: bridges missing detections and
        # disambiguates multiple far-side candidates via motion + size.
        self.far_track = None

        # Frames held back during an active gap (missed far detection), so
        # the box can be interpolated retroactively once the gap resolves
        # instead of coasting/jumping forward in real time.
        self._far_gap_buffer = []

        # Tracking buffers for the vector
        self.history_length = 15 
        self.mag_history = deque(maxlen=self.history_length)
        
        # State tracking
        self.state = "WAITING" 
        self.tracking_mode = None # "DUAL" or "SINGLE"
        
        # Thresholds
        self.steady_variance_thresh = 5.0  
        self.serve_change_thresh = 15.0    


    def select_court(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.court_points) < 4:
            self.court_points.append((x, y))

    def select_active_zone(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.active_zone_points) < 2:
            self.active_zone_points.append((x, y))

    def get_active_zone(self, frame):
        window_name = "Select Active Zone (Click 2 corners)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.select_active_zone)

        while True:
            display_frame = frame.copy()
            for pt in self.active_zone_points:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            if len(self.active_zone_points) == 2:
                pts = np.array(self.active_zone_points, np.int32)
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)

            cv2.imshow(window_name, display_frame)
            if len(self.active_zone_points) == 2:
                cv2.waitKey(1500)
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyWindow(window_name)


    def get_court_polygon(self, frame):
        window_name = "Select Court Boundaries (Click 4 corners)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.select_court)
        
        while True:
            display_frame = frame.copy()
            for pt in self.court_points:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            if len(self.court_points) == 4:
                pts = np.array(self.court_points, np.int32)
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_frame)
            if len(self.court_points) == 4:
                cv2.waitKey(1500) 
                break
            if cv2.waitKey(1) & 0xFF == 27: 
                break
        cv2.destroyWindow(window_name)


    def select_exclusion(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.exclusion_points.append((x, y))
            # Every pair of clicks defines one exclusion rectangle.
            if len(self.exclusion_points) % 2 == 0:
                (x1, y1), (x2, y2) = self.exclusion_points[-2:]
                self.exclusion_zones.append((
                    min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                ))

    def get_exclusion_zones(self, frame):
        """Interactively define rectangular exclusion zones.

        Click two opposite corners per zone (repeat for multiple zones).
        Press ENTER to finish, or ESC to skip.
        """
        window_name = "Select Exclusion Zones (2 clicks per zone, ENTER to finish)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.select_exclusion)

        while True:
            display_frame = frame.copy()
            for pt in self.exclusion_points:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            for (x1, y1, x2, y2) in self.exclusion_zones:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10) or key == 27:  # ENTER or ESC
                break
        cv2.destroyWindow(window_name)

    def _is_in_exclusion_zone(self, x, y):
        for (x1, y1, x2, y2) in self.exclusion_zones:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False

    def _config_path(self):
        """Path to the geometry config saved alongside the input video."""
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        return os.path.join(video_dir, f"{video_name}_player_config.json")

    def save_config(self):
        """Persist court poly, active zone, and exclusion zones next to the video."""
        config = {
            "court_points": [list(p) for p in self.court_points],
            "active_zone_points": [list(p) for p in self.active_zone_points],
            "exclusion_zones": [list(z) for z in self.exclusion_zones],
        }
        try:
            with open(self._config_path(), "w") as f:
                json.dump(config, f, indent=2)
            print(f"[INFO] Saved player config to {self._config_path()}")
        except Exception as e:
            print(f"[WARN] Could not save player config: {e}")

    def load_config(self):
        """Load geometry from the cached config. Returns True on success."""
        path = self._config_path()
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "r") as f:
                config = json.load(f)
            self.court_points = [tuple(p) for p in config["court_points"]]
            self.active_zone_points = [tuple(p) for p in config["active_zone_points"]]
            self.exclusion_zones = [tuple(z) for z in config["exclusion_zones"]]
            self._compute_far_baseline_midpoint()
            print(f"[INFO] Loaded player config from {path}")
            return True
        except Exception as e:
            print(f"[WARN] Player config unreadable ({e}), re-prompting")
            return False

    def _compute_far_baseline_midpoint(self):
        """Midpoint of the two active-zone points with the lowest Y (far baseline).

        Mirrors the prompt flow, where get_active_zone runs last and derives the
        far baseline from the active-zone corners.
        """
        sorted_pts = sorted(self.active_zone_points, key=lambda p: p[1])
        top_pts = sorted_pts[:2]
        self.far_baseline_midpoint = (
            int((top_pts[0][0] + top_pts[1][0]) / 2),
            int((top_pts[0][1] + top_pts[1][1]) / 2)
        )

    def _order_court_corners(self):
        """Derive (BL, BR, TR, TL) from the 4 clicked court corners regardless
        of click order: image-y picks near (larger y) vs far (smaller y), and
        image-x picks left vs right within each pair. Mirrors the same
        near/far-by-y logic already used in _compute_far_baseline_midpoint.
        """
        pts = sorted(self.court_points, key=lambda p: p[1])
        far_pair, near_pair = pts[:2], pts[2:]
        TL, TR = sorted(far_pair, key=lambda p: p[0])
        BL, BR = sorted(near_pair, key=lambda p: p[0])
        return BL, BR, TR, TL

    def _compute_homography(self):
        """Ground-plane homography mapping image pixels to world feet,
        anchored on the singles court rectangle (0,0)-(27,78)."""
        BL, BR, TR, TL = self._order_court_corners()
        src_pts = np.array([BL, BR, TR, TL], dtype=np.float32)
        dst_pts = np.array([
            [0, 0], [COURT_WIDTH_FT, 0],
            [COURT_WIDTH_FT, COURT_LENGTH_FT], [0, COURT_LENGTH_FT],
        ], dtype=np.float32)
        H, _ = cv2.findHomography(src_pts, dst_pts)
        return H

    def _pixel_to_world(self, x, y):
        pt = np.array([[[x, y]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self.H)
        return float(world[0, 0, 0]), float(world[0, 0, 1])

    def _select_far_player(self, valid_players):
        """Pick this frame's single best far-side candidate, or None.

        Classifies far-side membership in world space (feet closer to the far
        baseline than the near one). When multiple far-side candidates exist
        (e.g. court-line false positives), disambiguates using motion
        continuity and size consistency against the smoothed far-player track.
        """
        far_candidates = []
        for (cx, cy, x1, y1, x2, y2) in valid_players:
            wx, wy = self._pixel_to_world(cx, y2)
            if abs(wy - COURT_LENGTH_FT) < abs(wy):
                far_candidates.append((cx, cy, x1, y1, x2, y2))

        if not far_candidates:
            return None
        if len(far_candidates) == 1:
            return far_candidates[0]

        track = self.far_track
        if track is None or not track["active"]:
            # No prior yet: prefer the largest (most person-like) box.
            return max(far_candidates, key=lambda c: (c[4] - c[2]) * (c[5] - c[3]))

        # Conservative prior: assume the player is still near their last
        # confirmed position rather than extrapolating motion.
        pred_cx = track["cx"]
        pred_top_y = track["top_y"]

        def cost(c):
            cx, cy, x1, y1, x2, y2 = c
            motion_cost = math.hypot(cx - pred_cx, y1 - pred_top_y)
            size_cost = abs((x2 - x1) - track["w"])
            return motion_cost + size_cost

        return min(far_candidates, key=cost)

    def _update_far_track(self, detection):
        """Fold a confirmed far-side hit into the smoothed track (EMA on
        position/width). Call only when detection is not None -- misses are
        handled by the caller (process_video), which holds the box in place
        and buffers frames for retroactive interpolation once the gap ends.
        """
        cx, cy, x1, y1, x2, y2 = detection
        top_y = y1
        w = x2 - x1
        track = self.far_track
        if track is None or not track["active"]:
            self.far_track = {
                "cx": cx, "top_y": top_y, "w": w,
                "misses": 0, "active": True,
            }
        else:
            track["cx"] += FAR_TRACK_ALPHA * (cx - track["cx"])
            track["top_y"] += FAR_TRACK_ALPHA * (top_y - track["top_y"])
            track["w"] += FAR_TRACK_ALPHA * (w - track["w"])
            track["misses"] = 0

    def _far_box_from(self, cx, top_y, w):
        bottom_y = top_y + FAR_BOX_HEIGHT_PX
        return (int(cx - w / 2), int(top_y), int(cx + w / 2), int(bottom_y))

    def _select_near_player(self, valid_players):
        """Return the single best near-side candidate (cx, cy, x1, y1, x2, y2),
        or None. Classifies near-side membership in world space (feet closer
        to the near baseline than the far one); when multiple qualify, picks
        the one closest to the near baseline.
        """
        near_candidates = []
        for (cx, cy, x1, y1, x2, y2) in valid_players:
            wx, wy = self._pixel_to_world(cx, y2)
            if abs(wy) <= abs(wy - COURT_LENGTH_FT):
                near_candidates.append((cx, cy, x1, y1, x2, y2, wy))

        if not near_candidates:
            return None

        best = min(near_candidates, key=lambda c: abs(c[6]))
        return best[:6]

    def get_far_player_box(self):
        """Current smoothed far-player box (x1, y1, x2, y2), or None if inactive."""
        track = self.far_track
        if track is None or not track["active"]:
            return None
        return self._far_box_from(track["cx"], track["top_y"], track["w"])

    def calculate_vector(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        magnitude = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return magnitude

    def _draw_and_show(self, frame, near_detection, far_box):
        """Draw all overlays on frame and display it.

        Returns True if the user pressed 'q' (caller should stop the video).
        """
        if near_detection:
            ncx, ncy, nx1, ny1, nx2, ny2 = near_detection
            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)
            cv2.circle(frame, (ncx, ncy), 4, (0, 255, 0), -1)

        if far_box:
            fx1, fy1, fx2, fy2 = far_box
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
            cv2.putText(frame, "FAR", (fx1, fy1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.polylines(frame, [self._court_poly], True, (0, 255, 0), 2)
        cv2.polylines(frame, [self._active_zone], True, (255, 255, 0), 2)

        for (ex1, ey1, ex2, ey2) in self.exclusion_zones:
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 255), 2)

        crop_x, crop_y, crop_w, crop_h = self._crop_rect
        cv2.rectangle(frame, (crop_x, crop_y), (crop_x + crop_w, crop_y + crop_h), (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"State: {self.state}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if self.state == "WAITING" else (0, 255, 0), 3)
        cv2.putText(frame, f"Mode: {self.tracking_mode}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Tennis Serve Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            self.state = "WAITING"
            self.mag_history.clear()
        elif key == ord('q'):
            return True
        return False

    def _flush_gap_buffer_interpolated(self, resolved_detection):
        """Gap resolved: retroactively interpolate the far box across the
        buffered frames from the last confirmed track position to the newly
        resolved detection, then display them in order.

        Returns True if the user quit mid-flush.
        """
        track = self.far_track
        start_cx, start_top_y, start_w = track["cx"], track["top_y"], track["w"]
        end_cx = resolved_detection[0]
        end_top_y = resolved_detection[3]
        end_w = resolved_detection[4] - resolved_detection[2]

        n = len(self._far_gap_buffer)
        quit_requested = False
        for i, (buf_frame, buf_near) in enumerate(self._far_gap_buffer, start=1):
            t = i / (n + 1)
            cx = start_cx + t * (end_cx - start_cx)
            top_y = start_top_y + t * (end_top_y - start_top_y)
            w = start_w + t * (end_w - start_w)
            far_box = self._far_box_from(cx, top_y, w)
            if self._draw_and_show(buf_frame, buf_near, far_box):
                quit_requested = True
                break

        self._far_gap_buffer = []
        return quit_requested

    def _flush_gap_buffer_held(self, held_box):
        """Gap timed out with no resolving detection: flush the buffered
        frames using the frozen last-known far box.

        Returns True if the user quit mid-flush.
        """
        quit_requested = False
        for buf_frame, buf_near in self._far_gap_buffer:
            if self._draw_and_show(buf_frame, buf_near, held_box):
                quit_requested = True
                break

        self._far_gap_buffer = []
        return quit_requested

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read video")
            return

        # 1. Load cached geometry if available, otherwise prompt the user and save.
        if not self.load_config():
            # 1a. Initialize court boundaries and anchor point
            self.get_court_polygon(first_frame)

            # 1b. Initialize active zone
            self.get_active_zone(first_frame)

            # 1c. Define exclusion zones to reject court-side false positives.
            self.get_exclusion_zones(first_frame)

            # 1d. Persist geometry next to the video for future runs.
            self.save_config()

        self._court_poly = np.array(self.court_points, np.int32)

        # Ground-plane homography from the 4 court corners, used for near/far
        # side classification.
        self.H = self._compute_homography()

        # active_zone_points holds only the 2 clicked corners; expand to the
        # 4 corners of that axis-aligned rectangle so cv2.pointPolygonTest
        # gets an actual closed polygon (a 2-point "contour" always tests
        # as outside, which silently dropped every detection).
        (az_x1, az_y1), (az_x2, az_y2) = self.active_zone_points
        self._active_zone = np.array([
            [min(az_x1, az_x2), min(az_y1, az_y2)],
            [max(az_x1, az_x2), min(az_y1, az_y2)],
            [max(az_x1, az_x2), max(az_y1, az_y2)],
            [min(az_x1, az_x2), max(az_y1, az_y2)],
        ], np.int32)
        active_zone = self._active_zone

        # Calculate the bounding box of the polygon to use as our inference crop
        crop_x, crop_y, crop_w, crop_h = cv2.boundingRect(active_zone)
        self._crop_rect = (crop_x, crop_y, crop_w, crop_h)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 2. Crop the frame to the court area for YOLO
            crop_frame = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

            # 3. Run YOLO *only* on the cropped region
            # (Set imgsz relative to the crop size for massive speedups)
            results = self.model(crop_frame, conf = 0.2 , classes=[0], imgsz=480, verbose=False) 
            
            
            valid_players = []
            
            # 4. Map coordinates back to the full frame space
            for box in results[0].boxes:
                # Bounding box coordinates relative to the cropped image
                x1_c, y1_c, x2_c, y2_c = map(int, box.xyxy[0])
                cx_c, cy_c = (x1_c + x2_c) // 2, (y1_c + y2_c) // 2
                
                # Global coordinates (remapped back to the original full frame)
                x1 = x1_c + crop_x
                y1 = y1_c + crop_y
                x2 = x2_c + crop_x
                y2 = y2_c + crop_y
                cx = cx_c + crop_x
                cy = cy_c + crop_y
                
                # Drop detections whose center falls inside an exclusion zone
                # (spectators, coaches, ball kids court-side).
                if self._is_in_exclusion_zone(cx, cy):
                    continue

                # Precise polygon check to filter out the corner triangles of the bounding rect
                if cv2.pointPolygonTest(active_zone, (cx, cy), False) >= 0:
                    valid_players.append((cx, cy, x1, y1, x2, y2))

            # 5. Far-side selection (handles multiple court-line false positives).
            far_detection = self._select_far_player(valid_players)

            # 6. Near-side selection (raw detection; no smoothing/correction needed).
            near_detection = self._select_near_player(valid_players)

            quit_requested = False

            if far_detection is not None:
                # A confirmed detection: if we were mid-gap, retroactively
                # interpolate the box across the held frames first, now that
                # both ends of the gap are known.
                if self._far_gap_buffer:
                    quit_requested = self._flush_gap_buffer_interpolated(far_detection)

                if not quit_requested:
                    self._update_far_track(far_detection)
                    far_box = self.get_far_player_box()
                    quit_requested = self._draw_and_show(frame, near_detection, far_box)
            else:
                track = self.far_track
                if track is not None and track["active"]:
                    track["misses"] += 1
                    if track["misses"] > FAR_TRACK_MAX_MISSES:
                        # Gap timed out with no resolution: flush the held
                        # frames on the frozen last-known box, then go cold.
                        held_box = self.get_far_player_box()
                        quit_requested = self._flush_gap_buffer_held(held_box)
                        track["active"] = False
                        if not quit_requested:
                            quit_requested = self._draw_and_show(frame, near_detection, None)
                    else:
                        # Still within budget: hold this frame back instead of
                        # showing/coasting it now.
                        self._far_gap_buffer.append((frame, near_detection))
                        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                            quit_requested = True
                else:
                    quit_requested = self._draw_and_show(frame, near_detection, None)

            if quit_requested:
                break

        cap.release()
        cv2.destroyAllWindows()
# Usage:
if __name__ == "__main__":
    # Replace with the path to your actual video file
    system = AnyaTwoStateSystem("/Volumes/Anya/Data/22/snippet.mp4")
    system.process_video()