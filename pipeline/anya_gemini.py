import cv2
import numpy as np
import json
import math
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
    near_player_bbox: Optional[Tuple[float, float, float, float]] # [x, y, w, h]
    far_player_bbox: Optional[Tuple[float, float, float, float]]  # [x, y, w, h]
    ball_pos: Optional[Tuple[float, float]] 
    net_y_coord: float 

class PointStartSystem:
    def __init__(self, court_corners_pixels: np.ndarray, video_width: int, video_height: int, fps: int = 30):
        self.state = MatchState.INACTIVE
        self.fps = fps
        self.video_width = video_width
        self.video_height = video_height
        self.frame_buffer: List[TrackData] = []
        
        # Logging & Debugging
        self.active_frame_counter = 0
        self.serve_events: List[Dict] = []
        self.current_toss_roi: Optional[Tuple[float, float, float, float]] = None 

        # Near Serve World Thresholds
        self.near_dwell_radius_ft = 2.0
        self.near_baseline_y_min_ft = -3.5
        self.near_baseline_y_max_ft = 0.5
        self.toss_detected_frame_idx = 0

        # Far Serve Parameters (Stubs for now)
        self.dwell_frames_required = int(self.fps * 1.5)
        self.armed_frame_idx = 0
        self.toss_to_net_timeout = int(self.fps * 1.5)
        self.dwell_radius_pixels = 30 
        
        # 1. Initialize Homography Matrix
        self.H = self._compute_homography(court_corners_pixels)
        self.H_inv = np.linalg.inv(self.H) # Used for drawing the ready zone on the video
    
    def _compute_homography(self, pixel_corners: np.ndarray) -> np.ndarray:
        world_corners = np.array([
            [0, 0],       # Near left corner
            [27, 0],      # Near right corner
            [27, 78],     # Far right corner
            [0, 78]       # Far left corner
        ], dtype=np.float32)
        H, _ = cv2.findHomography(pixel_corners, world_corners)
        return H

    def _get_world_coords(self, px: float, py: float) -> Tuple[float, float]:
        pts = np.array([[[px, py]]], dtype=np.float32)
        world_pts = cv2.perspectiveTransform(pts, self.H)
        return world_pts[0][0][0], world_pts[0][0][1]
    
    def get_ready_zone_polygon(self) -> np.ndarray:
        """Returns the pixel coordinates of the Near Player Ready Zone for visualization."""
        world_zone = np.array([
            [0, self.near_baseline_y_min_ft],
            [27, self.near_baseline_y_min_ft],
            [27, self.near_baseline_y_max_ft],
            [0, self.near_baseline_y_max_ft]
        ], dtype=np.float32).reshape(-1, 1, 2)
        pixel_zone = cv2.perspectiveTransform(world_zone, self.H_inv)
        return np.int32(pixel_zone)

    def _get_near_toss_roi(self, px, py, pw, ph) -> Tuple[float, float, float, float]:
        """Helper to calculate ROI, shared by detection and visualization."""
        roi_h = ph
        roi_w = ph * (2.0 / 3.0)
        player_center_x = px + (pw / 2.0)
        player_bottom_y = py + ph
        
        roi_left_x = player_center_x - (roi_w / 2.0)
        roi_right_x = roi_left_x + roi_w
        roi_bottom_y = player_bottom_y - (ph * (2.0 / 3.0)) 
        roi_top_y = roi_bottom_y - roi_h
        return (roi_left_x, roi_top_y, roi_right_x, roi_bottom_y)

    def _check_near_dwell(self) -> bool:
        if len(self.frame_buffer) < self.dwell_frames_required:
            return False
            
        recent_frames = self.frame_buffer[-self.dwell_frames_required:]
        if any(f.near_player_bbox is None for f in recent_frames):
            return False # Player lost track
            
        start_frame = recent_frames[0]
        sx, sy, sw, sh = start_frame.near_player_bbox
        start_px, start_py = sx + (sw / 2.0), sy + sh
        start_wx, start_wy = self._get_world_coords(start_px, start_py)
        
        for frame in recent_frames:
            px, py, pw, ph = frame.near_player_bbox
            wx, wy = self._get_world_coords(px + (pw / 2.0), py + ph)
            
            if not (self.near_baseline_y_min_ft <= wy <= self.near_baseline_y_max_ft):
                return False 
                
            if math.sqrt((wx - start_wx)**2 + (wy - start_wy)**2) > self.near_dwell_radius_ft:
                return False 
                
        return True
    
    def _detect_near_toss(self) -> bool:
        if len(self.frame_buffer) < 15:
            return False
            
        recent_frames = self.frame_buffer[-15:]
        toss_y_history = []
        
        for frame in recent_frames:
            if frame.ball_pos is None or frame.near_player_bbox is None:
                continue
                
            bx, by = frame.ball_pos
            px, py, pw, ph = frame.near_player_bbox
            roi_left_x, roi_top_y, roi_right_x, roi_bottom_y = self._get_near_toss_roi(px, py, pw, ph)
            
            if (roi_left_x <= bx <= roi_right_x) and (roi_top_y <= by <= roi_bottom_y):
                toss_y_history.append(by)

        if len(toss_y_history) >= 3:
            y_oldest, y_mid, y_newest = toss_y_history[-3:]
            is_moving_up = (y_newest < y_mid) and (y_mid < y_oldest)
            if is_moving_up and (y_oldest - y_newest) > 5.0:
                return True
        return False
    
    def _detect_near_strike(self) -> bool:
        window_size = int(self.fps * 1.0)
        if len(self.frame_buffer) < window_size:
            return False

        recent_frames = self.frame_buffer[-window_size:]
        ball_history = [(f.frame_idx, f.ball_pos[0], f.ball_pos[1]) for f in recent_frames if f.ball_pos]

        if len(ball_history) < 4:
            return False

        min_y = float('inf')
        apex_idx = 0
        for i in range(len(ball_history) - 2): 
            _, _, by = ball_history[i]
            if by < min_y:
                min_y, apex_idx = by, i

        if apex_idx >= len(ball_history) - 2:
            return False

        post_apex_balls = ball_history[apex_idx + 1 :]
        for i in range(len(post_apex_balls) - 1):
            f_idx1, x1, y1 = post_apex_balls[i]
            f_idx2, x2, y2 = post_apex_balls[i + 1]
            dt_frames = f_idx2 - f_idx1 
            
            if dt_frames <= 0: continue
            dy_per_frame = (y2 - y1) / dt_frames
            dx_per_frame = (x2 - x1) / dt_frames
            velocity_per_frame = math.sqrt(dx_per_frame**2 + dy_per_frame**2)

            if dy_per_frame > 8.0 and velocity_per_frame > 15.0 and dt_frames <= 3:
                return True
        return False

    def _check_far_dwell(self) -> bool:
        # Stubbed out for now as requested
        return False

    def process_frame(self, current_data: TrackData) -> MatchState:
        self.frame_buffer.append(current_data)
        if len(self.frame_buffer) > self.fps * 4: 
            self.frame_buffer.pop(0)
            
        self.current_toss_roi = None # Reset ROI vis every frame unless dwelling
        
        # ---------------------------------------------------------
        # STATE: INACTIVE
        # ---------------------------------------------------------
        if self.state == MatchState.INACTIVE:
            if current_data.near_player_bbox and self._check_near_dwell():
                px, py, pw, ph = current_data.near_player_bbox
                self.current_toss_roi = self._get_near_toss_roi(px, py, pw, ph)

                if self._detect_near_toss():
                    self.state = MatchState.NEAR_SERVE_IN_FLIGHT
                    self.toss_detected_frame_idx = current_data.frame_idx
                    print(f"[State] Near toss detected at frame {current_data.frame_idx}")
            
            # Far Serve logic stubbed here ...

        # ---------------------------------------------------------
        # STATE: NEAR SERVE IN FLIGHT
        # ---------------------------------------------------------
        elif self.state == MatchState.NEAR_SERVE_IN_FLIGHT:
            if self._detect_near_strike():
                self._trigger_active(current_data.frame_idx, "Near Serve Strike")
            elif (current_data.frame_idx - self.toss_detected_frame_idx) > int(self.fps * 2.5):
                self.state = MatchState.INACTIVE
                print(f"[State] Aborted Toss/Timeout: Reset to INACTIVE at frame {current_data.frame_idx}")

        # ---------------------------------------------------------
        # STATE: ACTIVE (Hardcoded 3 second reset for debugging)
        # ---------------------------------------------------------
        elif self.state == MatchState.ACTIVE:
            self.active_frame_counter += 1
            if self.active_frame_counter >= (self.fps * 3):
                self.state = MatchState.INACTIVE
                self.active_frame_counter = 0
                print(f"[State] 3-sec debug period ended. Resetting to INACTIVE at frame {current_data.frame_idx}")

        return self.state

    def _trigger_active(self, frame_idx: int, trigger_source: str):
        self.state = MatchState.ACTIVE
        self.active_frame_counter = 0
        self.serve_events.append({
            "frame": frame_idx,
            "event": trigger_source
        })
        print(f"[State] Point STARTED at frame {frame_idx} via {trigger_source}")

# ==========================================
# INTERACTIVE COURT SELECTION
# ==========================================
def get_court_corners_interactive(video_path: str) -> np.ndarray:
    """
    Extracts the middle frame of the video and opens an interactive OpenCV 
    window for the user to select the 4 corners of the court.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
        
    # Get the middle frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Could not read the middle frame of the video.")

    points = []
    window_name = "Select Court Corners"
    
    # Text overlays
    instructions = [
        "Click the 4 corners of the SINGLES court in order:",
        "1. NEAR LEFT corner",
        "2. NEAR RIGHT corner",
        "3. FAR RIGHT corner",
        "4. FAR LEFT corner",
        "",
        "Press 'r' to reset points."
    ]

    display_frame = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            
            # Draw lines between points as they are clicked
            if len(points) > 1:
                cv2.line(display_frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
                
            # Close the polygon on the 4th click
            if len(points) == 4:
                cv2.line(display_frame, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
                cv2.putText(display_frame, "Done! Press any key to start processing.", (50, 280), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display_frame)

    def draw_instructions(img):
        for i, inst in enumerate(instructions):
            color = (255, 255, 255) if i == 0 else (0, 255, 255)
            cv2.putText(img, inst, (50, 50 + (i*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    draw_instructions(display_frame)
    cv2.imshow(window_name, display_frame)
    
    # Event loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # Reset points
            points.clear()
            display_frame = frame.copy()
            draw_instructions(display_frame)
            cv2.imshow(window_name, display_frame)
        elif len(points) == 4 and key != 255:  # Any key pressed after 4 points
            break
            
    cv2.destroyAllWindows()
    return np.array(points, dtype=np.float32)

# ==========================================
# MAIN EXECUTION SCRIPT
# ==========================================
def run_point_detector(video_path: str, output_path: str, ball_model_path: str):
    # 1. Get court corners interactively BEFORE loading heavy models
    print("Opening UI for court calibration...")
    court_corners = get_court_corners_interactive(video_path)
    
    # 2. Load YOLO Models
    print("Loading Models...")
    yolo_player_model = YOLO("yolov8n.pt") 
    yolo_ball_model = YOLO(ball_model_path)
    
    # 3. Open Video for Processing
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 4. Initialize System
    system = PointStartSystem(court_corners, width, height, fps=fps)
    ready_zone_poly = system.get_ready_zone_polygon()

    frame_idx = 0
    print("Processing Video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # --- A. Inference ---
        player_results = yolo_player_model.predict(frame, classes=[0], verbose=False)[0]
        player_bboxes = []
        for box in player_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            player_bboxes.append((x1, y1, x2 - x1, y2 - y1))
            
        ball_results = yolo_ball_model.predict(frame, verbose=False)[0]
        ball_pos = None
        if len(ball_results.boxes) > 0:
            bx1, by1, bx2, by2 = ball_results.boxes[0].xyxy[0].cpu().numpy()
            ball_pos = (bx1 + (bx2 - bx1)/2.0, by1 + (by2 - by1)/2.0)
            
        # --- B. Identify Near Player ---
        near_player_bbox = None
        if player_bboxes:
            near_player_bbox = max(player_bboxes, key=lambda b: b[1] + b[3])
            
        # --- C. System Update ---
        track_data = TrackData(
            frame_idx=frame_idx,
            near_player_bbox=near_player_bbox,
            far_player_bbox=None, 
            ball_pos=ball_pos,
            net_y_coord=height / 2.0 
        )
        current_state = system.process_frame(track_data)
        
        # --- D. Visualizations ---
        cv2.putText(frame, f"State: {current_state.name}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.polylines(frame, [ready_zone_poly], isClosed=True, color=(0, 255, 0), thickness=2)
        
        if near_player_bbox:
            nx, ny, nw, nh = near_player_bbox
            cv2.rectangle(frame, (int(nx), int(ny)), (int(nx+nw), int(ny+nh)), (255, 0, 0), 2)
            cv2.putText(frame, "Near Player", (int(nx), int(ny)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        if ball_pos:
            cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 5, (0, 255, 255), -1)
            
        if system.current_toss_roi:
            r_left, r_top, r_right, r_bottom = system.current_toss_roi
            cv2.rectangle(frame, (int(r_left), int(r_top)), (int(r_right), int(r_bottom)), (0, 0, 255), 2)
            cv2.putText(frame, "TOSS ROI", (int(r_left), int(r_top)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        out.write(frame)

    cap.release()
    out.release()
    
    # 5. Save JSON Results
    json_path = "/Volumes/Anya/Data/21/serve_events.json"
    with open(json_path, 'w') as f:
        json.dump(system.serve_events, f, indent=4)
        
    print(f"Done! Video saved to {output_path}")
    print(f"Events saved to {json_path}")

# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    VIDEO_IN = "/Volumes/Anya/Data/21/snippet.mp4"
    VIDEO_OUT = "/Volumes/Anya/Data/21/output_match_annotated.mp4"
    BALL_MODEL = "/Users/tennis/Documents/Code/Laptop/src/anya/pipeline/models/ball_best.pt"
    
    run_point_detector(VIDEO_IN, VIDEO_OUT, BALL_MODEL)