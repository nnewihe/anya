import cv2
import numpy as np
import math
from collections import deque
from ultralytics import YOLO


# === CHANGE: New helper class =============================================
# WHY: The old code decided DUAL vs SINGLE from len(valid_players) on a SINGLE
# frame. yolov8n on a court crop downscaled to 480px leaves the far player only
# ~15px tall, so it flickers in and out. One missed frame -> mode flips ->
# mag_history.clear() + state="WAITING" -> the 15-frame history can never fill,
# so the state machine never reaches PRE_SERVE/POINT_STARTED. Detection was
# effectively disabled by the flicker.
#
# A PlayerRole holds one logical slot (FAR or NEAR). When YOLO misses it for a
# frame, we don't drop the slot -- we "coast" on its last known position for up
# to coast_limit frames. This keeps DUAL mode stable across short detection
# gaps, which is the root cause of the oscillation.
class PlayerRole:
    def __init__(self, name, coast_limit=30):
        self.name = name              # "FAR" or "NEAR"
        self.track_id = None          # ByteTrack ID currently owning this role
        self.box = None               # (cx, cy, w, h) in FULL-frame coords
        self.centroid = None          # (cx, cy) in FULL-frame coords
        self.frames_missing = 0       # consecutive frames with no fresh detection
        self.coast_limit = coast_limit

    @property
    def active(self):
        # Active while we've ever seen it AND haven't been coasting too long.
        return self.centroid is not None and self.frames_missing <= self.coast_limit

    @property
    def coasting(self):
        return self.active and self.frames_missing > 0

    def update(self, centroid, box, track_id):
        self.centroid = centroid
        self.box = box
        self.track_id = track_id
        self.frames_missing = 0

    def mark_missing(self):
        # Age the track but keep its last position (coasting). Only drop the
        # slot once we've been blind for longer than coast_limit.
        self.frames_missing += 1
        if self.frames_missing > self.coast_limit:
            self.centroid = None
            self.box = None
            self.track_id = None


class AnyaTwoStateSystem:
    def __init__(self, video_path):
        self.video_path = video_path
        self.model = YOLO('yolov8n.pt')
        self.court_points = []
        self.far_baseline_midpoint = None

        # === CHANGE: added near-baseline midpoint =========================
        # WHY: role assignment now uses proximity to BOTH baselines instead of
        # raw y-extremes, so an umpire/ball-kid/spectator inside the polygon no
        # longer automatically wins the FAR or NEAR slot.
        self.near_baseline_midpoint = None

        # Tracking buffers for the vector
        self.history_length = 15
        self.mag_history = deque(maxlen=self.history_length)

        # State tracking
        self.state = "WAITING"
        self.tracking_mode = None  # "DUAL" or "SINGLE"

        # === CHANGE: persistent role slots + coast window =================
        # WHY: coast_limit ~= 30 frames (~1s at 30fps). Long enough to bridge
        # YOLO misses on the tiny far player; short enough that a real exit
        # (changeover, player leaves court) still collapses to SINGLE/None.
        self.coast_limit = 30
        self.far_role = PlayerRole("FAR", coast_limit=self.coast_limit)
        self.near_role = PlayerRole("NEAR", coast_limit=self.coast_limit)

        # Thresholds (kept in PIXELS -- see note in the refactor summary about
        # why normalization was intentionally NOT applied).
        self.steady_variance_thresh = 5.0
        self.serve_change_thresh = 15.0

    def select_court(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.court_points) < 6:
            self.court_points.append((x, y))

    def get_court_polygon(self, frame):
        window_name = "Select Court Boundaries (Click 6 corners)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.select_court)

        while True:
            display_frame = frame.copy()
            for pt in self.court_points:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
            if len(self.court_points) == 6:
                pts = np.array(self.court_points, np.int32)
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)

            cv2.imshow(window_name, display_frame)
            if len(self.court_points) == 6:
                cv2.waitKey(1500)
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyWindow(window_name)

        # Far baseline = the two points with the lowest Y (top of image).
        sorted_pts = sorted(self.court_points, key=lambda p: p[1])
        top_pts = sorted_pts[:2]
        self.far_baseline_midpoint = (
            int((top_pts[0][0] + top_pts[1][0]) / 2),
            int((top_pts[0][1] + top_pts[1][1]) / 2)
        )

        # === CHANGE: also compute the near baseline midpoint ==============
        # WHY: needed as the "target baseline" for the NEAR role during
        # proximity-based (re)assignment.
        bottom_pts = sorted_pts[-2:]
        self.near_baseline_midpoint = (
            int((bottom_pts[0][0] + bottom_pts[1][0]) / 2),
            int((bottom_pts[0][1] + bottom_pts[1][1]) / 2)
        )

    def calculate_vector(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        magnitude = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        return magnitude

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _draw_tracker(self, frame, role, color):
        """Draw a role's (possibly coasted) bounding box.

        === CHANGE: this previously-orphaned helper is now actually used ===
        WHY: it referenced tracker.active / tracker.box but nothing ever
        supplied such an object. PlayerRole now exposes exactly those fields,
        so predicted (coasted) boxes get drawn here.
        """
        if role.active and role.box:
            cx, cy, w, h = role.box
            x1 = int(cx - (w / 2))
            y1 = int(cy - (h / 2))
            x2 = int(cx + (w / 2))
            y2 = int(cy + (h / 2))
            # Dim/orange the box while coasting so it's visually obvious the
            # position is predicted, not freshly detected.
            draw_color = (0, 165, 255) if role.coasting else color
            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)

    # === CHANGE: new method -- all the frame-by-frame role bookkeeping ====
    # WHY: pulled out of the main loop to keep process_video readable. This is
    # where the flicker fix lives.
    def _assign_roles(self, valid_players, court_poly):
        # Index detections by their ByteTrack ID (None-id detections excluded).
        detections_by_id = {d['id']: d for d in valid_players if d['id'] is not None}
        updated = set()   # role names refreshed with a real detection this frame

        # --- Step 1: ID continuity -------------------------------------------------
        # WHY: if a role's track_id is still present, keep it -- this is the
        # cheap, reliable path that makes IDs "sticky" and prevents role swaps.
        for role in (self.far_role, self.near_role):
            if role.track_id in detections_by_id:
                d = detections_by_id[role.track_id]
                role.update(d['centroid'], d['box'], d['id'])
                updated.add(role.name)

        # --- Step 2: (re)acquire missing roles by baseline proximity ---------------
        # WHY: handles first acquisition AND re-acquisition after ByteTrack
        # assigns a NEW id to a re-appearing player. We match leftover
        # detections to whichever baseline they're closest to, instead of the
        # old "sort by y, take first/last" which breaks with 3+ people.
        used_ids = {self.far_role.track_id, self.near_role.track_id}
        leftovers = [d for d in valid_players if d['id'] not in used_ids or d['id'] is None]
        pending_roles = [r for r in (self.far_role, self.near_role) if r.name not in updated]

        if leftovers and pending_roles:
            targets = {
                "FAR": self.far_baseline_midpoint,
                "NEAR": self.near_baseline_midpoint,
            }
            # Score every (detection, pending-role) pair, then greedily take the
            # globally closest pairs 1:1.
            pairs = []
            for d in leftovers:
                for r in pending_roles:
                    pairs.append((self._dist(d['centroid'], targets[r.name]), d, r))
            pairs.sort(key=lambda t: t[0])

            claimed_dets, claimed_roles = set(), set()
            for _, d, r in pairs:
                if id(d) in claimed_dets or r.name in claimed_roles:
                    continue
                r.update(d['centroid'], d['box'], d['id'])
                claimed_dets.add(id(d))
                claimed_roles.add(r.name)
                updated.add(r.name)

        # --- Step 3: coast whatever wasn't refreshed -------------------------------
        for role in (self.far_role, self.near_role):
            if role.name not in updated:
                role.mark_missing()

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read video")
            return

        # 1. Initialize court boundaries and anchor point
        self.get_court_polygon(first_frame)
        court_poly = np.array(self.court_points, np.int32)

        # Bounding box of the polygon = the inference crop.
        crop_x, crop_y, crop_w, crop_h = cv2.boundingRect(court_poly)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 2. Crop the frame to the court area for YOLO
            crop_frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

            # === CHANGE: model.track(persist=True) instead of model() =========
            # WHY: this is the core fix. track() assigns stable IDs across frames
            # (ByteTrack) and keeps lost tracks in a buffer for re-association.
            # Combined with our own coasting, a missed far-player frame no longer
            # collapses the mode. imgsz raised 480->640 and conf lowered to 0.2
            # to make the tiny far player easier to detect in the first place
            # (the polygon test below rejects the extra false positives).
            results = self.model.track(
                crop_frame,
                persist=True,
                classes=[0],
                imgsz=640,
                conf=0.2,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            valid_players = []

            # 4. Map coordinates back to the full frame space
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    # Center-form box (cx, cy, w, h) relative to the crop.
                    cx_c, cy_c, w_c, h_c = map(float, box.xywh[0])
                    cx = int(cx_c + crop_x)
                    cy = int(cy_c + crop_y)
                    w = int(w_c)
                    h = int(h_c)

                    # Precise polygon check to reject the corner triangles of the
                    # bounding-rect crop (and stray people outside the court).
                    if cv2.pointPolygonTest(court_poly, (cx, cy), False) >= 0:
                        # === CHANGE: carry the ByteTrack ID with each detection =
                        tid = int(box.id[0]) if box.id is not None else None
                        valid_players.append({
                            "centroid": (cx, cy),
                            "box": (cx, cy, w, h),
                            "id": tid,
                        })

            # === CHANGE: role assignment replaces the old inline mode logic ===
            self._assign_roles(valid_players, court_poly)

            # 5. Derive mode from role activity (not from a per-frame count).
            current_mode = None
            p_anchor = None
            p_target = None

            if self.far_role.active and self.near_role.active:
                current_mode = "DUAL"
                p_target = self.far_role.centroid
                p_anchor = self.near_role.centroid
                # Blue = far, green = near (unchanged from original coloring).
                self._draw_tracker(frame, self.far_role, (255, 0, 0))
                self._draw_tracker(frame, self.near_role, (0, 255, 0))

            elif self.far_role.active or self.near_role.active:
                current_mode = "SINGLE"
                active_role = self.far_role if self.far_role.active else self.near_role
                p_target = active_role.centroid
                p_anchor = self.far_baseline_midpoint
                self._draw_tracker(frame, active_role, (255, 0, 0))
                cv2.circle(frame, p_anchor, 6, (0, 165, 255), -1)

            # 6. Vector Calculation and State Logic
            if current_mode:
                # === CHANGE (behavioral note, not code) ========================
                # We STILL reset history + state on a mode change. That reset was
                # never the bug -- it's correct, because DUAL measures
                # near->far while SINGLE measures player->baseline, so each mode
                # needs its own baseline. The bug was that flicker fired this
                # reset every few frames. With coasting, mode changes are now
                # rare and real, so per-mode reset is exactly what we want and
                # NO vector normalization is needed.
                if current_mode != self.tracking_mode:
                    self.tracking_mode = current_mode
                    self.mag_history.clear()
                    self.state = "WAITING"

                cv2.line(frame, p_anchor, p_target, (0, 255, 255), 2)
                mag = self.calculate_vector(p_anchor, p_target)
                self.mag_history.append(mag)

                if len(self.mag_history) == self.history_length:
                    mag_variance = np.var(self.mag_history)
                    avg_mag = np.mean(self.mag_history)

                    if self.state == "WAITING" and mag_variance < self.steady_variance_thresh:
                        self.state = "PRE_SERVE"
                        self.baseline_mag = avg_mag

                    elif self.state == "PRE_SERVE":
                        current_diff = abs(mag - self.baseline_mag)
                        if current_diff > self.serve_change_thresh:
                            self.state = "POINT_STARTED"

            # Draw UI
            cv2.polylines(frame, [court_poly], True, (0, 255, 0), 2)
            cv2.rectangle(frame, (crop_x, crop_y),
                          (crop_x + crop_w, crop_y + crop_h),
                          (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, f"State: {self.state}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if self.state == "WAITING" else (0, 255, 0), 3)
            cv2.putText(frame, f"Mode: {self.tracking_mode}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # === CHANGE: small diagnostic -- shows when a role is coasting =====
            # WHY: makes it obvious at a glance that the mode is being held
            # through a detection gap rather than genuinely dropping.
            coast_flags = []
            if self.far_role.coasting:
                coast_flags.append("FAR")
            if self.near_role.coasting:
                coast_flags.append("NEAR")
            if coast_flags:
                cv2.putText(frame, "COASTING: " + ",".join(coast_flags), (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow("Tennis Serve Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                self.state = "WAITING"
                self.mag_history.clear()
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Usage:
if __name__ == "__main__":
    system = AnyaTwoStateSystem("/Volumes/Anya/Data/21/snippet.mp4")
    system.process_video()