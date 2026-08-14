import cv2
import numpy as np
import json
import math
import torch
import argparse
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from ultralytics import YOLO

class MatchState(Enum):
    WAITING = 0
    ARMED = 1
    ACTIVE = 2

@dataclass
class TrackData:
    frame_idx: int
    near_player_bbox: Optional[Tuple[float, float, float, float]]
    ball_pos: Optional[Tuple[float, float]]

class PointStartSystem:
    def __init__(self, court_corners_pixels: np.ndarray, video_width: int, video_height: int, fps: int = 30):
        self.state = MatchState.WAITING
        self.fps = fps
        self.video_width = video_width
        self.video_height = video_height
        self.frame_buffer: List[TrackData] = []

        self.active_frame_counter = 0
        self.serve_events: List[Dict] = []
        self.current_toss_roi: Optional[Tuple[float, float, float, float]] = None

        self.near_dwell_radius_ft = 5.0
        self.near_baseline_y_min_ft = -3.5
        self.near_baseline_y_max_ft = 0.5

        self.toss_detected_frame_idx = 0
        self.dwell_frames_required = int(self.fps * 1.5)

        self.H = self._compute_homography(court_corners_pixels)
        self.H_inv = np.linalg.inv(self.H)

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
        for frame in self.frame_buffer[-15:]:
            if frame.ball_pos is None or frame.near_player_bbox is None: continue
            bx, by = frame.ball_pos
            px, py, pw, ph = frame.near_player_bbox
            rl, rt, rr, rb = self._get_near_toss_roi(px, py, pw, ph)
            if (rl <= bx <= rr) and (rt <= by <= rb): toss_y_history.append(by)

        if len(toss_y_history) >= 3:
            y_oldest, y_mid, y_newest = toss_y_history[-3:]
            # Ball is moving upward inside ROI
            if (y_newest < y_mid < y_oldest) and (y_oldest - y_newest) > 5.0: return True
        return False

    def _detect_ratio_shift(self) -> bool:
        """Detects a rapid shift in player box aspect ratio indicating a serve execution (e.g., reaching up)"""
        window = int(self.fps * 0.5)
        if len(self.frame_buffer) < window: return False
        ratios = []
        for frame in self.frame_buffer[-window:]:
            if frame.near_player_bbox:
                _, _, w, h = frame.near_player_bbox
                ratios.append(w / h)
        if len(ratios) < 5: return False

        # If difference between min and max ratio over half a second is > 0.15, ratio shift triggered
        return (max(ratios) - min(ratios)) > 0.15

    def process_frame(self, current_data: TrackData) -> MatchState:
        self.frame_buffer.append(current_data)
        if len(self.frame_buffer) > self.fps * 4: self.frame_buffer.pop(0)
        self.current_toss_roi = None

        if self.state == MatchState.WAITING:
            if current_data.near_player_bbox and self._check_near_dwell():
                self.state = MatchState.ARMED
                print(f"[State] Player ready. Transition to ARMED at frame {current_data.frame_idx}")

        elif self.state == MatchState.ARMED:
            if not self._check_near_dwell():
                self.state = MatchState.WAITING
                self.toss_detected_frame_idx = 0
                print(f"[State] Player broke dwell. Reset to WAITING")
                return self.state

            px, py, pw, ph = current_data.near_player_bbox
            self.current_toss_roi = self._get_near_toss_roi(px, py, pw, ph)

            if self._detect_near_toss():
                self.toss_detected_frame_idx = current_data.frame_idx
                print(f"[Event] Toss detected at frame {current_data.frame_idx}")

            # If toss happened within the last 1 second, look for rapid ratio shift
            if self.toss_detected_frame_idx > 0:
                frames_since_toss = current_data.frame_idx - self.toss_detected_frame_idx

                if frames_since_toss <= int(self.fps * 1.5):
                    if self._detect_ratio_shift():
                        self._trigger_active(current_data.frame_idx, "Serve Executed (Toss + Ratio Shift)")
                        self.toss_detected_frame_idx = 0
                else:
                    print(f"[Event] Toss timed out. Continuing ARMED watch.")
                    self.toss_detected_frame_idx = 0

        elif self.state == MatchState.ACTIVE:
            self.active_frame_counter += 1
            if self.active_frame_counter >= (self.fps * 3):
                self.state = MatchState.WAITING
                self.active_frame_counter = 0
                print(f"[State] Resetting to WAITING")

        return self.state

    def _trigger_active(self, frame_idx: int, trigger_source: str):
        self.state = MatchState.ACTIVE
        self.active_frame_counter = 0
        self.serve_events.append({"frame": frame_idx, "event": trigger_source})
        print(f"[State] Point STARTED at frame {frame_idx} via {trigger_source}")


def get_court_corners_interactive(video_path: str, headless: bool) -> np.ndarray:
    if headless:
        print("[Headless] Skipping UI court calibration. Using fallback/default corners.")
        return np.array([[300, 800], [1620, 800], [1300, 200], [600, 200]], dtype=np.float32)

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

def run_point_detector(video_path: str, output_path: str, ball_model_path: str, stride: int = 10, headless: bool = False):

    court_corners = get_court_corners_interactive(video_path, headless)

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading Models on device: {device}...")

    yolo_player_model = YOLO("yolov8n.pt")
    yolo_ball_model = YOLO(ball_model_path)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    system = PointStartSystem(court_corners, width, height, fps=fps)
    ready_zone_poly = system.get_ready_zone_polygon()

    frame_idx = 0
    print("Processing Video...")
    if not headless:
        cv2.namedWindow("Anya Tennis - Processing", cv2.WINDOW_NORMAL)

    cached_near_player_bbox = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # --- A. Player Inference ---
        if (frame_idx - 1) % stride == 0:
            player_results = yolo_player_model.predict(frame, classes=[0], imgsz=640, device=device, verbose=False)[0]

            near_player_candidates = []
            for box in player_results.boxes:
                x, y, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x, y2 - y
                px_center = x + (w / 2.0)

                wx, wy = system._get_world_coords(px_center, y + h)
                if -2.0 <= wx <= 29.0 and wy <= 38.0:
                    near_player_candidates.append(((x, y, w, h), wx, wy))

            cached_near_player_bbox = min(near_player_candidates, key=lambda p: abs(p[2] - 0.0))[0] if near_player_candidates else None

        near_player_bbox = cached_near_player_bbox

        # --- B. Gated Ball Inference ---
        ball_pos = None
        if system.current_toss_roi is not None:
            r_left, r_top, r_right, r_bottom = system.current_toss_roi
            c_left, c_top = max(0, int(r_left)), max(0, int(r_top))
            c_right, c_bottom = min(width, int(r_right)), min(height, int(r_bottom))

            if c_right > c_left and c_bottom > c_top:
                roi_crop = frame[c_top:c_bottom, c_left:c_right]
                ball_results = yolo_ball_model.predict(roi_crop, imgsz=256, device=device, verbose=False)[0]

                if len(ball_results.boxes) > 0:
                    bx1, by1, bx2, by2 = ball_results.boxes[0].xyxy[0].cpu().numpy()
                    ball_pos = (bx1 + c_left + (bx2 - bx1)/2.0, by1 + c_top + (by2 - by1)/2.0)

        # --- C. System Update ---
        track_data = TrackData(
            frame_idx=frame_idx, near_player_bbox=near_player_bbox, ball_pos=ball_pos
        )
        current_state = system.process_frame(track_data)

        # --- D. Visualizations ---
        cv2.putText(frame, f"State: {current_state.name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.polylines(frame, [ready_zone_poly], isClosed=True, color=(0, 255, 0), thickness=2)

        if near_player_bbox:
            nx, ny, nw, nh = near_player_bbox
            cv2.rectangle(frame, (int(nx), int(ny)), (int(nx+nw), int(ny+nh)), (255, 0, 0), 2)

        if ball_pos:
            cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 5, (0, 255, 255), -1)

        if system.current_toss_roi:
            r_left, r_top, r_right, r_bottom = system.current_toss_roi
            cv2.rectangle(frame, (int(r_left), int(r_top)), (int(r_right), int(r_bottom)), (0, 0, 255), 2)

        out.write(frame)

        if not headless:
            cv2.imshow("Anya Tennis - Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    if not headless:
        cv2.destroyAllWindows()

    json_path = "serve_events.json"
    with open(json_path, 'w') as f:
        json.dump(system.serve_events, f, indent=4)
    print(f"Finished processing. Events saved to {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anya Tennis Near-Serve Detector")
    parser.add_argument("--video_in", type=str, default="/Volumes/Anya/Data/21/snippet.mp4", help="Input video path")
    parser.add_argument("--video_out", type=str, default="/Volumes/Anya/Data/21/output_match_annotated.mp4", help="Output video path")
    parser.add_argument("--ball_model", type=str, default="/Users/tennis/Documents/Code/Laptop/src/anya/pipeline/models/ball_best.pt", help="Ball YOLO model path")
    parser.add_argument("--headless", action="store_true", help="Disable real-time visualizations")

    args = parser.parse_args()

    run_point_detector(args.video_in, args.video_out, args.ball_model, headless=args.headless)
