import cv2
import numpy as np
import json
import math
import torch
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from ultralytics import YOLO

class MatchState(Enum):
    INACTIVE = 0  
    FAR_SERVE_IN_FLIGHT = 1       
    NEAR_SERVE_IN_FLIGHT = 2
    ACTIVE = 3           

@dataclass
class TrackData:
    frame_idx: int
    near_player_bbox: Optional[Tuple[float, float, float, float]] 
    # --- MODIFIED: Added near player keypoints field ---
    # --------------------------------------------------
    far_player_bbox: Optional[Tuple[float, float, float, float]]  
    ball_pos: Optional[Tuple[float, float]] 
    net_y_coord: float
    near_player_keypoints: Optional[np.ndarray] = None


class PointStartSystem:
    def __init__(self, court_corners_pixels: np.ndarray, video_width: int, video_height: int, fps: float = 30.0):
        self.state = MatchState.INACTIVE
        self.fps = float(fps)
        self.video_width = video_width
        self.video_height = video_height
        self.frame_buffer: List[TrackData] = []
        
        self.active_frame_counter = 0

        # --- MODIFIED: Pose Debounce Counters ---
        self.dead_pose_frame_counter = 0
        self.dead_pose_required_frames = int(self.fps * 1.5)  # Must hold dead pose for 1.5s
        # ----------------------------------------

        # --- MODIFIED: End-detection guard rails ---
        # A point cannot end until it has been ACTIVE for this long (blocks the
        # mid-rally false-end -> re-trigger seen at frame 7636).
        self.min_active_frames = int(self.fps * 1.5)
        # After a point ends, no new serve can be armed for this long (blocks
        # immediate re-triggers; also damps dead-time juggling).
        self.refractory_frames = int(self.fps * 1.5)
        self.last_point_end_frame = -100000
        # -------------------------------------------

        self.serve_events: List[Dict] = []
        self.current_toss_roi: Optional[Tuple[float, float, float, float]] = None

        self.near_dwell_radius_ft = 2.0
        self.near_baseline_y_min_ft = -3.5
        self.near_baseline_y_max_ft = 0.5
        
        self.far_dwell_radius_ft = 2.0
        self.far_baseline_y_min_ft = 72.0 
        self.far_baseline_y_max_ft = 85.0 

        self.toss_detected_frame_idx = 0
        self.dwell_frames_required = int(self.fps * 1.5)
        # Far serve is confirmed by a net-cross (not a toss); give the ball time
        # to travel the full court and be re-detected in the corridor.
        self.toss_to_net_timeout = int(self.fps * 3.0)
        
        self.H = self._compute_homography(court_corners_pixels)
        self.H_inv = np.linalg.inv(self.H) 
        _, self.net_pixel_y = self._get_pixel_coords(13.5, 39.0)
    
    def _compute_homography(self, pixel_corners: np.ndarray) -> np.ndarray:
        world_corners = np.array([
            [0, 0], [27, 0], [27, 78], [0, 78]
        ], dtype=np.float32)
        H, _ = cv2.findHomography(pixel_corners, world_corners)
        return H

    def _get_world_coords(self, px: float, py: float) -> Tuple[float, float]:
        pts = np.array([[[px, py]]], dtype=np.float32)
        world_pts = cv2.perspectiveTransform(pts, self.H)
        return world_pts[0][0][0], world_pts[0][0][1]
    
    def _get_pixel_coords(self, wx: float, wy: float) -> Tuple[float, float]:
        pts = np.array([[[wx, wy]]], dtype=np.float32)
        px_pts = cv2.perspectiveTransform(pts, self.H_inv)
        return px_pts[0][0][0], px_pts[0][0][1]
    
    def get_ready_zone_polygon(self) -> np.ndarray:
        world_zone = np.array([
            [0, self.near_baseline_y_min_ft],
            [27, self.near_baseline_y_min_ft],
            [27, self.near_baseline_y_max_ft],
            [0, self.near_baseline_y_max_ft]
        ], dtype=np.float32).reshape(-1, 1, 2)
        pixel_zone = cv2.perspectiveTransform(world_zone, self.H_inv)
        return np.int32(pixel_zone)

    def _get_near_toss_roi(self, px, py, pw, ph) -> Tuple[float, float, float, float]:
        roi_h, roi_w = ph, ph * (2.0 / 3.0)
        player_center_x, player_bottom_y = px + (pw / 2.0), py + ph
        roi_left_x = player_center_x - (roi_w / 2.0)
        roi_bottom_y = player_bottom_y - (ph * (2.0 / 3.0)) 
        return (roi_left_x, roi_bottom_y - roi_h, roi_left_x + roi_w, roi_bottom_y)

    def _check_near_dwell(self) -> bool:
        if len(self.frame_buffer) < self.dwell_frames_required: return False
        recent = self.frame_buffer[-self.dwell_frames_required:]
        if any(f.near_player_bbox is None for f in recent): return False 
            
        sx, sy, sw, sh = recent[0].near_player_bbox
        start_wx, start_wy = self._get_world_coords(sx + (sw / 2.0), sy + sh)
        
        for frame in recent:
            px, py, pw, ph = frame.near_player_bbox
            wx, wy = self._get_world_coords(px + (pw / 2.0), py + ph)
            if not (self.near_baseline_y_min_ft <= wy <= self.near_baseline_y_max_ft): return False 
            if math.sqrt((wx - start_wx)**2 + (wy - start_wy)**2) > self.near_dwell_radius_ft: return False 
        return True
    
    def _detect_near_toss(self) -> bool:
        if len(self.frame_buffer) < 15: return False
        toss_y_history = []
        last_ph = 0.0
        for frame in self.frame_buffer[-15:]:
            if frame.ball_pos is None or frame.near_player_bbox is None: continue
            bx, by = frame.ball_pos
            px, py, pw, ph = frame.near_player_bbox
            last_ph = ph
            rl, rt, rr, rb = self._get_near_toss_roi(px, py, pw, ph)
            if (rl <= bx <= rr) and (rt <= by <= rb): toss_y_history.append(by)

        # --- MODIFIED: scale the required toss rise to player height ---
        # A real serve toss travels a large fraction of the player's height; a
        # fixed 5px gate at 4K also fired on dead-time ball juggling (false
        # starts at frames 9427 / 9544). 10% of bbox height is resolution-robust.
        min_rise = max(5.0, 0.10 * last_ph)
        if len(toss_y_history) >= 3:
            y_oldest, y_mid, y_newest = toss_y_history[-3:]
            if (y_newest < y_mid < y_oldest) and (y_oldest - y_newest) > min_rise: return True
        return False
    
    def _detect_near_strike(self) -> bool:
        window = int(self.fps * 1.0)
        if len(self.frame_buffer) < window: return False
        balls = [(f.frame_idx, f.ball_pos[0], f.ball_pos[1]) for f in self.frame_buffer[-window:] if f.ball_pos]
        if len(balls) < 4: return False

        min_y, apex_idx = float('inf'), 0
        for i in range(len(balls) - 2): 
            if balls[i][2] < min_y: min_y, apex_idx = balls[i][2], i

        if apex_idx >= len(balls) - 2: return False

        for i in range(apex_idx + 1, len(balls) - 1):
            f1, x1, y1 = balls[i]
            f2, x2, y2 = balls[i + 1]
            dt = f2 - f1 
            if dt <= 0: continue
            if (y2 - y1)/dt > 8.0 and math.sqrt(((x2-x1)/dt)**2 + ((y2-y1)/dt)**2) > 15.0 and dt <= 3: return True
        return False

    def _get_far_toss_roi(self, px, py, pw, ph) -> Tuple[float, float, float, float]:
        roi_h, roi_w = ph * 1.5, pw * 1.5
        roi_left_x = (px + (pw / 2.0)) - (roi_w / 2.0)
        roi_bottom_y = py + (ph * 0.2)
        return (roi_left_x, roi_bottom_y - roi_h, roi_left_x + roi_w, roi_bottom_y)

    # --- MODIFIED: corridor spanning the far court down through the net ---
    # Far-serve start is confirmed by a net-cross, so we must keep detecting the
    # ball from the far baseline all the way to the net (the far *toss* itself is
    # not detectable at 4K). Returns a pixel bbox covering that corridor.
    def _get_far_corridor_roi(self) -> Tuple[float, float, float, float]:
        world_corners = [(-1.0, 33.0), (28.0, 33.0), (28.0, 82.0), (-1.0, 82.0)]
        xs, ys = [], []
        for wx, wy in world_corners:
            px, py = self._get_pixel_coords(wx, wy)
            xs.append(px); ys.append(py)
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        return (left, top, right, bottom)

    def _check_far_dwell(self) -> bool:
        if len(self.frame_buffer) < self.dwell_frames_required: return False
        recent = self.frame_buffer[-self.dwell_frames_required:]
        if any(f.far_player_bbox is None for f in recent): return False 
            
        sx, sy, sw, sh = recent[0].far_player_bbox
        start_wx, start_wy = self._get_world_coords(sx + (sw / 2.0), sy + sh)
        
        for frame in recent:
            px, py, pw, ph = frame.far_player_bbox
            wx, wy = self._get_world_coords(px + (pw / 2.0), py + ph)
            if not (self.far_baseline_y_min_ft <= wy <= self.far_baseline_y_max_ft): return False 
            if math.sqrt((wx - start_wx)**2 + (wy - start_wy)**2) > self.far_dwell_radius_ft: return False 
        return True

    def _detect_far_toss(self) -> bool:
        if len(self.frame_buffer) < 15: return False
        toss_y_history = []
        for frame in self.frame_buffer[-15:]:
            if frame.ball_pos is None or frame.far_player_bbox is None: continue
            bx, by = frame.ball_pos
            px, py, pw, ph = frame.far_player_bbox
            rl, rt, rr, rb = self._get_far_toss_roi(px, py, pw, ph)
            if (rl <= bx <= rr) and (rt <= by <= rb): toss_y_history.append(by)

        if len(toss_y_history) >= 3:
            y_oldest, y_mid, y_newest = toss_y_history[-3:]
            if (y_newest < y_mid < y_oldest) and (y_oldest - y_newest) > 2.0: return True
        return False

    def _detect_net_cross(self) -> bool:
        window = int(self.fps * 0.75) 
        if len(self.frame_buffer) < window: return False
        balls = [(f.frame_idx, f.ball_pos[0], f.ball_pos[1]) for f in self.frame_buffer[-window:] if f.ball_pos]
        if len(balls) < 2: return False
            
        for i in range(len(balls) - 1):
            f1, _, y1 = balls[i]
            for j in range(i + 1, len(balls)):
                f2, _, y2 = balls[j]
                if y1 < self.net_pixel_y and y2 > self.net_pixel_y:
                    if (f2 - f1) > 0 and ((y2 - y1) / (f2 - f1)) > 4.0: return True
        return False

    # --- MODIFIED: NEW METHOD - Pose Evaluation Heuristics ---
    def _is_dead_pose(self, kpts: Optional[np.ndarray], near_bbox: Optional[Tuple[float, float, float, float]]) -> bool:
        """
        Evaluates near player pose for a between-points ("dead") posture.

        NOTE: the previous "Backwards" (shoulder inversion) and "Sideways"
        (torso compression) heuristics were removed -- both are *normal in-rally
        postures* (players hit groundstrokes sideways and turn their backs
        chasing wide balls), so they fired mid-point and caused false early ends
        (e.g. the re-trigger at frame 7636 inside rally 4). Only the upright /
        stationary signal, which does not occur during an active rally, is kept.
        """
        if kpts is None or len(kpts) < 17:
            return False

        conf_thresh = 0.35
        l_hip, r_hip = kpts[11], kpts[12]
        l_knee, r_knee = kpts[13], kpts[14]
        l_ank, r_ank = kpts[15], kpts[16]

        # Upright / Stationary (Leg Extension & Narrow Stance)
        is_upright = False
        if (l_hip[2] > conf_thresh and l_knee[2] > conf_thresh and l_ank[2] > conf_thresh and
            r_hip[2] > conf_thresh and r_knee[2] > conf_thresh and r_ank[2] > conf_thresh):

            def calc_angle(p1, p2, p3):
                v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 == 0 or n2 == 0: return 180.0
                return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))

            l_knee_angle = calc_angle(l_hip, l_knee, l_ank)
            r_knee_angle = calc_angle(r_hip, r_knee, r_ank)

            # Both legs fully extended (> 165 deg) indicates upright posture vs athletic crouch
            if l_knee_angle > 165.0 and r_knee_angle > 165.0:
                if near_bbox:
                    bbox_w = near_bbox[2]
                    stance_w = abs(l_ank[0] - r_ank[0])
                    # Stance is narrow relative to body width
                    if stance_w / max(1.0, bbox_w) < 0.30:
                        is_upright = True

        return is_upright
    # --------------------------------------------------------

    def process_frame(self, current_data: TrackData) -> MatchState:
        self.frame_buffer.append(current_data)
        if len(self.frame_buffer) > self.fps * 4: self.frame_buffer.pop(0)
        self.current_toss_roi = None 
        
        # --- MODIFIED: block arming a new serve during the post-point refractory ---
        in_refractory = (current_data.frame_idx - self.last_point_end_frame) < self.refractory_frames

        if self.state == MatchState.INACTIVE:
            if in_refractory:
                pass  # too soon after the last point ended; do not arm a serve
            elif current_data.near_player_bbox and self._check_near_dwell():
                px, py, pw, ph = current_data.near_player_bbox
                self.current_toss_roi = self._get_near_toss_roi(px, py, pw, ph)
                if self._detect_near_toss():
                    self.state = MatchState.NEAR_SERVE_IN_FLIGHT
                    self.toss_detected_frame_idx = current_data.frame_idx
                    print(f"[State] Near toss detected at frame {current_data.frame_idx}")

            elif current_data.far_player_bbox and self._check_far_dwell():
                # --- MODIFIED: far serve no longer requires a (undetectable at 4K)
                # toss. Dwell alone arms the flight; the start is confirmed by a
                # net-cross. Keep the ball corridor active so the descending ball
                # is detected all the way to the net.
                self.current_toss_roi = self._get_far_corridor_roi()
                self.state = MatchState.FAR_SERVE_IN_FLIGHT
                self.toss_detected_frame_idx = current_data.frame_idx
                print(f"[State] Far dwell satisfied; awaiting net-cross at frame {current_data.frame_idx}")

        elif self.state == MatchState.NEAR_SERVE_IN_FLIGHT:
            if self._detect_near_strike():
                # --- MODIFIED: report the point start at the toss onset, not the
                # racket strike, to remove the systematic ~1.5s late bias. ---
                self._trigger_active(self.toss_detected_frame_idx, "Near Serve Strike")
            elif (current_data.frame_idx - self.toss_detected_frame_idx) > int(self.fps * 2.5):
                self.state = MatchState.INACTIVE
                print(f"[State] Aborted Toss/Timeout: Reset to INACTIVE")

        elif self.state == MatchState.FAR_SERVE_IN_FLIGHT:
            # Keep detecting the ball across the far court -> net corridor.
            self.current_toss_roi = self._get_far_corridor_roi()
            if self._detect_net_cross():
                self._trigger_active(current_data.frame_idx, "Far Serve Net Cross")
            elif (current_data.frame_idx - self.toss_detected_frame_idx) > self.toss_to_net_timeout:
                self.state = MatchState.INACTIVE
                print(f"[State] Aborted Far Serve/Timeout: Reset to INACTIVE")

        # --- MODIFIED: Point End Trigger Logic ---
        elif self.state == MatchState.ACTIVE:
            self.active_frame_counter += 1

            # A point must run for a minimum duration before it can end. This
            # blocks the mid-rally false-end -> re-trigger observed at frame 7636.
            can_end = self.active_frame_counter >= self.min_active_frames

            if self._is_dead_pose(current_data.near_player_keypoints, current_data.near_player_bbox):
                self.dead_pose_frame_counter += 1
                if can_end and self.dead_pose_frame_counter >= self.dead_pose_required_frames:
                    self.state = MatchState.INACTIVE
                    self.active_frame_counter = 0
                    self.dead_pose_frame_counter = 0
                    self.last_point_end_frame = current_data.frame_idx
                    self.serve_events.append({"frame": current_data.frame_idx, "event": "Point End"})
                    print(f"[State] Point ENDED at frame {current_data.frame_idx} via Dead Pose Trigger")
            else:
                # Slowly decay counter if pose turns active briefly (grace buffer)
                self.dead_pose_frame_counter = max(0, self.dead_pose_frame_counter - 1)
        # ----------------------------------------

        return self.state

    def _trigger_active(self, frame_idx: int, trigger_source: str):
        self.state = MatchState.ACTIVE
        self.active_frame_counter = 0
        self.dead_pose_frame_counter = 0
        self.serve_events.append({"frame": frame_idx, "event": trigger_source})
        print(f"[State] Point STARTED at frame {frame_idx} via {trigger_source}")

def get_court_corners_interactive(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Could not open video {video_path}")
    mid_frame_idx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret: raise ValueError("Could not read frame.")
    points = []
    window_name = "Select Court Corners"
    display_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1: cv2.line(display_frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(display_frame, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
                cv2.putText(display_frame, "Press any key to start.", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, display_frame)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, display_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            points.clear()
            display_frame = frame.copy()
            cv2.imshow(window_name, display_frame)
        elif len(points) == 4 and key != 255: break
            
    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)

def run_point_detector(video_path: str, output_path: str, ball_model_path: str, stride: int = 10):

    print("Opening UI for court calibration...")
    court_corners = get_court_corners_interactive(video_path)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading Models on device: {device}...")
    
    # --- MODIFIED: Swapped to Pose estimation model ---
    yolo_player_model = YOLO("yolov8n-pose.pt") 
    # --------------------------------------------------
    yolo_ball_model = YOLO(ball_model_path)
    
    cap = cv2.VideoCapture(video_path)
    # --- MODIFIED: keep fps as float; int(29.97) -> 29 biased every time-based
    # threshold ~3% short. ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    system = PointStartSystem(court_corners, width, height, fps=fps)
    ready_zone_poly = system.get_ready_zone_polygon()

    img_pts, wld_pts = [], []
    for py in range(0, height, height // 10):
        for px in range(0, width, width // 10):
            wx, wy = system._get_world_coords(px, py)
            img_pts.append([px, py])
            wld_pts.append([wx, wy])
            
    H_wld_to_img, _ = cv2.findHomography(np.array(wld_pts, dtype=np.float32), np.array(img_pts, dtype=np.float32))
    base_wld = np.array([[[0, 78]], [[27, 78]]], dtype=np.float32)
    base_img = cv2.perspectiveTransform(base_wld, H_wld_to_img)
    px_width_at_baseline = np.linalg.norm(base_img[0][0] - base_img[1][0])
    
    px_per_ft_far = px_width_at_baseline / 27.0
    calc_far_player_height = 6.0 * px_per_ft_far

    last_far_player_bbox = None
    far_missing_frame_count = 0
    max_far_missing_frames = int(fps * 3.0) 

    frame_idx = 0
    print("Processing Video...")
    cv2.namedWindow("Anya Tennis - Processing", cv2.WINDOW_NORMAL)
    
    cached_near_player_bbox = None
    cached_near_player_kpts = None
    cached_raw_far_player_bbox = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        # --- A. Player Inference ---
        if (frame_idx - 1) % stride == 0:
            # --- MODIFIED: Extracting keypoints from pose model ---
            player_results = yolo_player_model.predict(frame, imgsz=1280, device=device, verbose=False)[0]
            
            near_player_candidates = []
            far_player_candidates = []

            boxes = player_results.boxes
            keypoints = player_results.keypoints

            for i, box in enumerate(boxes):
                if int(box.cls[0]) != 0: continue
                
                x, y, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x, y2 - y
                px_center = x + (w / 2.0)
                
                kpts = keypoints.data[i].cpu().numpy() if keypoints is not None and len(keypoints.data) > i else None
                
                wx, wy = system._get_world_coords(px_center, y + h)
                if -2.0 <= wx <= 29.0 and wy <= 38.0:
                    near_player_candidates.append(((x, y, w, h), kpts, wx, wy))
                    
                new_h = calc_far_player_height
                fwx, fwy = system._get_world_coords(px_center, y + new_h)
                if -2.0 <= fwx <= 29.0 and 74.0 <= fwy <= 85.0:
                    far_player_candidates.append(((x, y, w, new_h), fwx, fwy))

            if near_player_candidates:
                best_near = min(near_player_candidates, key=lambda p: abs(p[2] - 0.0))
                cached_near_player_bbox = best_near[0]
                cached_near_player_kpts = best_near[1]
            else:
                cached_near_player_bbox = None
                cached_near_player_kpts = None

            cached_raw_far_player_bbox = min(far_player_candidates, key=lambda p: abs(p[2] - 78.0))[0] if far_player_candidates else None
            # -----------------------------------------------------

        near_player_bbox = cached_near_player_bbox
        near_player_kpts = cached_near_player_kpts
        raw_far_player_bbox = cached_raw_far_player_bbox

        is_far_persisted = False
        if raw_far_player_bbox is not None:
            far_player_bbox = raw_far_player_bbox
            last_far_player_bbox = raw_far_player_bbox
            far_missing_frame_count = 0
        else:
            far_missing_frame_count += 1
            if last_far_player_bbox is not None and far_missing_frame_count <= max_far_missing_frames:
                far_player_bbox = last_far_player_bbox
                is_far_persisted = True
            else:
                far_player_bbox = None
                last_far_player_bbox = None

        # --- B. Gated Ball Inference ---
        ball_pos = None
        if system.current_toss_roi is not None:
            r_left, r_top, r_right, r_bottom = system.current_toss_roi
            c_left, c_top = max(0, int(r_left)), max(0, int(r_top))
            c_right, c_bottom = min(width, int(r_right)), min(height, int(r_bottom))
            
            if c_right > c_left and c_bottom > c_top:
                roi_crop = frame[c_top:c_bottom, c_left:c_right]
                # --- MODIFIED: adapt inference resolution to ROI size. The far
                # corridor is large and its ball is tiny, so it needs a higher
                # imgsz; the small toss ROIs stay cheap at 256. ---
                roi_max_dim = max(c_right - c_left, c_bottom - c_top)
                ball_imgsz = 960 if roi_max_dim > 400 else 256
                ball_results = yolo_ball_model.predict(roi_crop, imgsz=ball_imgsz, device=device, verbose=False)[0]

                if len(ball_results.boxes) > 0:
                    bx1, by1, bx2, by2 = ball_results.boxes[0].xyxy[0].cpu().numpy()
                    ball_pos = (bx1 + c_left + (bx2 - bx1)/2.0, by1 + c_top + (by2 - by1)/2.0)

        # --- C. System Update ---
        # --- MODIFIED: Pass near player keypoints into TrackData ---
        track_data = TrackData(
            frame_idx=frame_idx, near_player_bbox=near_player_bbox,
            far_player_bbox=far_player_bbox, ball_pos=ball_pos, net_y_coord=height / 2.0,
            near_player_keypoints=near_player_kpts if near_player_bbox else None
        )
        # -----------------------------------------------------------
        current_state = system.process_frame(track_data)
        
        # --- D. Visualizations ---
        cv2.putText(frame, f"State: {current_state.name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if current_state == MatchState.ACTIVE:
            cv2.putText(frame, f"Dead Pose Buffer: {system.dead_pose_frame_counter}/{system.dead_pose_required_frames}", 
                        (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if system.dead_pose_frame_counter > 0 else (0, 255, 0), 2)

        cv2.polylines(frame, [ready_zone_poly], isClosed=True, color=(0, 255, 0), thickness=2)
        
        if near_player_bbox:
            nx, ny, nw, nh = near_player_bbox
            cv2.rectangle(frame, (int(nx), int(ny)), (int(nx+nw), int(ny+nh)), (255, 0, 0), 2)
        
        if far_player_bbox:
            fx, fy, fw, fh = far_player_bbox
            cv2.rectangle(frame, (int(fx), int(fy)), (int(fx+fw), int(fy+fh)), (0, 165, 255), 2)
            
        if ball_pos:
            cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 5, (0, 255, 255), -1)
            
        if system.current_toss_roi:
            r_left, r_top, r_right, r_bottom = system.current_toss_roi
            cv2.rectangle(frame, (int(r_left), int(r_top)), (int(r_right), int(r_bottom)), (0, 0, 255), 2)
        
        cv2.imshow("Anya Tennis - Processing", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    json_path = "/Volumes/Anya/Data/21/serve_events.json"
    with open(json_path, 'w') as f:
        json.dump(system.serve_events, f, indent=4)
        
if __name__ == "__main__":
    VIDEO_IN = "/Volumes/Anya/Data/21/snippet.mp4"
    VIDEO_OUT = "/Volumes/Anya/Data/21/output_match_annotated.mp4"
    BALL_MODEL = "/Users/tennis/Documents/Code/Laptop/src/anya/pipeline/models/ball_best.pt"
    
    run_point_detector(VIDEO_IN, VIDEO_OUT, BALL_MODEL)