import argparse
import cv2
from ultralytics import YOLO

import time
import numpy as np
import cv2
from enum import Enum
from collections import deque
from pathlib import Path


class GameState(Enum):
    OUT_OF_PLAY = 0
    PRE_SERVE = 1
    IN_PLAY = 2

class CourtHomography:
    """
    Handles mapping of 2D screen pixels to real-world 2D court coordinates (meters).
    Standard Singles Court Dimensions: 8.23m x 23.77m
    """
    def __init__(self, clicked_corners=None):
        self.SINGLES_W = 8.23
        self.SINGLES_L = 23.77
        self.NET_Y = self.SINGLES_L / 2.0 
        
        self.real_world_corners = np.array([
            [0, 0],                  
            [self.SINGLES_W, 0],     
            [self.SINGLES_W, self.SINGLES_L], 
            [0, self.SINGLES_L]      
        ], dtype=np.float32)
        
        self.H = None
        if clicked_corners is not None:
            self.set_matrix(clicked_corners)

    def set_matrix(self, clicked_corners):
        src_pts = np.array(clicked_corners, dtype=np.float32).reshape(-1, 1, 2)
        dst_pts = self.real_world_corners.reshape(-1, 1, 2)
        self.H, _ = cv2.findHomography(src_pts, dst_pts)

    def to_real_world(self, pixel_coord):
        if self.H is None:
            raise ValueError("Homography matrix not initialized.")
        
        px, py = pixel_coord
        point = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.H)
        return transformed[0][0][0], transformed[0][0][1]


class PlayerTracker:
    def __init__(self):
        self.history = deque(maxlen=30) 
        self.velocity = 0.0
        self.is_stationary = False
        self.is_upright = True

    def update(self, bbox, court_position):
        if bbox is None or court_position is None:
            return
        self.history.append((time.time(), bbox, court_position))
        self._calculate_kinematics()

    def _calculate_kinematics(self):
        if len(self.history) < 5:
            return
            
        t1, _, p1 = self.history[-5]
        t2, _, p2 = self.history[-1]
        
        dt = t2 - t1
        if dt > 0:
            distance = np.linalg.norm(np.array(p2) - np.array(p1))
            self.velocity = distance / dt
        
        self.is_stationary = self.velocity < 0.4
        
        _, bbox, _ = self.history[-1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        aspect_ratio = h / w if w > 0 else 0
        
        # Lowered to 1.4 to accommodate the distant far-player resolution and shallow angle
        self.is_upright = aspect_ratio > 1.4


class BallTracker:
    def __init__(self):
        self.history = deque(maxlen=30)  
        self.bounces = []                
        self.velocity_2d = 0.0
        self.trajectory_broken = False

    def update(self, pixel_coord, net_y_threshold=11.885, homography=None):
        # NOTE: pixel_coord is processed regardless of whether it falls inside 
        # a player bounding box. This prevents tracking failures on down-the-middle shots.
        if pixel_coord is None:
            return
        self.history.append((time.time(), pixel_coord))
        self._analyze_motion(net_y_threshold, homography)

    def _analyze_motion(self, net_y, homography):
        if len(self.history) < 4:
            return
            
        t1, p1 = self.history[-3]
        t2, p2 = self.history[-2]
        t3, p3 = self.history[-1]
        
        v1 = np.linalg.norm(np.array(p2) - np.array(p1)) / (t2 - t1)
        v2 = np.linalg.norm(np.array(p3) - np.array(p2)) / (t3 - t2)
        
        self.velocity_2d = v2
        
        # Percentage-based velocity drop (e.g., losing 80% of speed instantly)
        if v1 > 50.0 and (v2 / v1) < 0.2:
            self.trajectory_broken = True

        vec1 = np.array(p2) - np.array(p1)
        vec2 = np.array(p3) - np.array(p2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            cos_theta = np.dot(vec1, vec2) / (norm1 * norm2)
            if cos_theta < -0.2 and v2 < 50.0:
                self.trajectory_broken = True

        # Shallow-angle Bounce Detection
        p0_y = self.history[-4][1][1]
        p1_y = self.history[-3][1][1]
        p2_y = self.history[-2][1][1]
        p3_y = self.history[-1][1][1]
        
        # Using a small > 1.0 pixel buffer instead of strict equality to catch subtle far-court bounces
        if (p1_y < p2_y and (p2_y - p3_y) > 1.0) or (p1_y > p2_y and (p3_y - p2_y) > 1.0):
            if homography is not None:
                rx, ry = homography.to_real_world(p2)
                side = "near" if ry < net_y else "far"
                
                if not self.bounces or (time.time() - self.bounces[-1][0] > 0.2):
                    self.bounces.append((time.time(), side))


class HybridTwoStateSystem:
    def __init__(self, clicked_corners):
        self.court = CourtHomography(clicked_corners)
        self.state = GameState.OUT_OF_PLAY
        
        self.player_near = PlayerTracker()
        self.player_far = PlayerTracker()
        self.ball = BallTracker()
        
        self.pre_serve_start_time = None
        self.stationary_threshold = 1.5

    def process_frame(self, yolo_player_boxes, yolo_ball_center):
        near_box, far_box = self._assign_players(yolo_player_boxes)
        
        near_pos = self.court.to_real_world(((near_box[0]+near_box[2])/2, near_box[3])) if near_box is not None else None
        far_pos = self.court.to_real_world(((far_box[0]+far_box[2])/2, far_box[3])) if far_box is not None else None
        
        self.player_near.update(near_box, near_pos)
        self.player_far.update(far_box, far_pos)
        
        if yolo_ball_center is not None:
            self.ball.update(yolo_ball_center, net_y_threshold=self.court.NET_Y, homography=self.court)

        if self.state == GameState.OUT_OF_PLAY:
            self._check_for_pre_serve(near_pos, far_pos)
            
        elif self.state == GameState.PRE_SERVE:
            self._check_for_serve_trigger()
            
        elif self.state == GameState.IN_PLAY:
            self._check_for_point_end()

        return self.state

    def _assign_players(self, boxes):
        if not boxes:
            return None, None
        sorted_boxes = sorted(boxes, key=lambda b: b[3])
        if len(sorted_boxes) >= 2:
            return sorted_boxes[-1], sorted_boxes[0]  
        elif len(sorted_boxes) == 1:
            return (sorted_boxes[0], None) if sorted_boxes[0][3] > 540 else (None, sorted_boxes[0])
        return None, None

    def _check_for_pre_serve(self, near_pos, far_pos):
        if near_pos is None or far_pos is None:
            return

        near_serving = self._is_behind_baseline(near_pos, side="near") and self._is_in_return_position(far_pos, side="far")
        far_serving = self._is_behind_baseline(far_pos, side="far") and self._is_in_return_position(near_pos, side="near")

        if (near_serving or far_serving) and self.player_near.is_stationary and self.player_far.is_stationary:
            if self.pre_serve_start_time is None:
                self.pre_serve_start_time = time.time()
            elif (time.time() - self.pre_serve_start_time) >= self.stationary_threshold:
                self.state = GameState.PRE_SERVE
        else:
            self.pre_serve_start_time = None

    def _check_for_serve_trigger(self):
        if not (self.player_near.is_stationary or self.player_far.is_stationary):
            self.state = GameState.OUT_OF_PLAY
            self.pre_serve_start_time = None
            return

        if self.ball.velocity_2d > 800.0:  
            self.state = GameState.IN_PLAY
            self.ball.trajectory_broken = False
            self.ball.bounces.clear()

    def _check_for_point_end(self):
        if len(self.ball.bounces) >= 2:
            if self.ball.bounces[-1][1] == self.ball.bounces[-2][1]:
                self.state = GameState.OUT_OF_PLAY
                return

        if self.ball.trajectory_broken:
            self.state = GameState.OUT_OF_PLAY
            return

        if self.player_near.velocity < 1.2 and self.player_near.is_upright:
            if self.player_far.velocity < 1.2 and self.player_far.is_upright:
                self.state = GameState.OUT_OF_PLAY

    def _is_behind_baseline(self, pos, side):
        x, y = pos
        if side == "near":
            return y < 0.0 and (-2.0 <= x <= self.court.SINGLES_W + 2.0)
        else:
            return y > self.court.SINGLES_L and (-2.0 <= x <= self.court.SINGLES_W + 2.0)

    def _is_in_return_position(self, pos, side):
        x, y = pos
        if side == "near":
            return 0.0 <= y <= 6.0 and (-1.0 <= x <= self.court.SINGLES_W + 1.0)
        else:
            return (self.court.SINGLES_L - 6.0) <= y <= self.court.SINGLES_L and (-1.0 <= x <= self.court.SINGLES_W + 1.0)

# Global variables for the click-to-map UI
clicked_points = []
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        # Draw a visible dot where the user clicked
        frame_copy = params['frame']
        cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click 4 Corners (Near-L, Near-R, Far-R, Far-L)", frame_copy)

def main():
    parser = argparse.ArgumentParser(description="Anya Tennis - Two-State Tracking Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to the input tennis video")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save processed video")
    args = parser.parse_args()

    # 1. Load Custom Models
    print("Loading YOLO models...")
    _MODELS_DIR = Path(__file__).parent / "models"
    player_model = YOLO(str(_MODELS_DIR / "yolo26n.pt"))
    ball_model   = YOLO(str(_MODELS_DIR / "ball_best.pt"))
   

    # 2. Open Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    # 3. Interactive Court Mapping (First Frame)
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    print("UI OPENED: Click the 4 corners of the singles court in this order:")
    print("1. Near-Left  2. Near-Right  3. Far-Right  4. Far-Left")
    
    cv2.namedWindow("Click 4 Corners (Near-L, Near-R, Far-R, Far-L)")
    params = {'frame': first_frame.copy()}
    cv2.setMouseCallback("Click 4 Corners (Near-L, Near-R, Far-R, Far-L)", click_event, params)

    while len(clicked_points) < 4:
        cv2.imshow("Click 4 Corners (Near-L, Near-R, Far-R, Far-L)", params['frame'])
        if cv2.waitKey(1) & 0xFF == 27: # Press ESC to exit early
            break

    cv2.destroyAllWindows()

    if len(clicked_points) < 4:
        print("Error: 4 corners were not selected. Exiting.")
        return

    print(f"Corners mapped at: {clicked_points}")

    # 4. Initialize the Tracking System
    tracker = HybridTwoStateSystem(clicked_corners=clicked_points)

    # 5. Setup Video Writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 6. Main Processing Loop
    print("Starting video processing...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        player_results = player_model(frame, verbose=False)[0]
        ball_results = ball_model(frame, verbose=False)[0]

        # Extract Player Boxes [xmin, ymin, xmax, ymax]
        player_boxes = []
        for box in player_results.boxes.xyxy.cpu().numpy():
            player_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])

        # Extract Ball Center (assuming ball model outputs bounding boxes)
        ball_center = None
        if len(ball_results.boxes) > 0:
            # Take the highest confidence ball detection
            best_ball = ball_results.boxes.xyxy[0].cpu().numpy()
            ball_center = ((best_ball[0] + best_ball[2]) / 2.0, (best_ball[1] + best_ball[3]) / 2.0)

        # Process the state machine
        current_state = tracker.process_frame(player_boxes, ball_center)

        # --- Visualization ---
        # Draw State
        state_text = f"STATE: {current_state.name}"
        color = (0, 255, 0) if current_state == GameState.IN_PLAY else (0, 0, 255)
        cv2.putText(frame, state_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Draw Players
        for box in player_boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)

        # Draw Ball
        if ball_center:
            cv2.circle(frame, (int(ball_center[0]), int(ball_center[1])), 5, (0, 255, 255), -1)

        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Processing complete. Saved to {args.output}")

if __name__ == "__main__":
    main()