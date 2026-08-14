import cv2
import numpy as np
import torch
import argparse
import csv
from typing import List, Tuple, Optional
from ultralytics import YOLO

class AntiVisionTwoStateSystem:
    def __init__(self, court_corners_pixels: np.ndarray, fps: int = 30):
        self.fps = fps
        self.pose_buffer: List[np.ndarray] = []
        
        # Precompute homography for identifying the near player
        self.H = self._compute_homography(court_corners_pixels)
        
        # Assuming YOLOv8 pose outputs 17 keypoints (x, y, conf)
        # We will strip conf and flatten to 34 features per frame
        self.feature_dim = 17 * 2 

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

    def process_frame_pose(self, keypoints: Optional[np.ndarray]):
        """
        Takes a [17, 2] or [17, 3] keypoint array for the near player.
        Pads with zeros if the player is lost in a frame to maintain sequence length.
        """
        if keypoints is not None:
            # Extract just X and Y, flatten to 1D array of length 34
            flat_pose = keypoints[:, :2].flatten()
        else:
            flat_pose = np.zeros(self.feature_dim, dtype=np.float32)
            
        self.pose_buffer.append(flat_pose)

    def is_buffer_full(self) -> bool:
        return len(self.pose_buffer) >= self.fps

    def get_sequence_tensor(self) -> torch.Tensor:
        """Returns the buffer as a tensor of shape (1, seq_len, feature_dim) and clears the buffer."""
        seq_array = np.array(self.pose_buffer, dtype=np.float32)
        self.pose_buffer.clear()
        
        # Add batch dimension
        return torch.tensor(seq_array).unsqueeze(0)


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


def run_state_estimator(video_path: str, output_csv: str, gru_model_path: str, headless: bool = False):
    court_corners = get_court_corners_interactive(video_path, headless)

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading Models on device: {device}...")

    # Load Pose Estimator
    yolo_pose_model = YOLO("yolov8n-pose.pt")
    
    # Load your trained GRU model
    # Note: Adjust loading mechanism if it's a state_dict rather than TorchScript
    gru_model = torch.jit.load(gru_model_path, map_location=device) if gru_model_path.endswith('.pt') else torch.load(gru_model_path, map_location=device)
    gru_model.eval()

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    system = AntiVisionTwoStateSystem(court_corners, fps=fps)

    frame_idx = 0
    current_second = 0

    print(f"Processing Video... Saving output to {output_csv}")
    
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Timestamp (s)", "P_Dead"])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1

            # --- A. Pose Inference ---
            pose_results = yolo_pose_model.predict(frame, classes=[0], imgsz=640, device=device, verbose=False)[0]

            near_player_keypoints = None
            near_player_candidates = []
            
            # Find the near player using homography filtering
            if pose_results.keypoints is not None and len(pose_results.keypoints) > 0:
                boxes = pose_results.boxes.xyxy.cpu().numpy()
                kpts = pose_results.keypoints.data.cpu().numpy() # Shape: (N, 17, 3)

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    px_center = x1 + ((x2 - x1) / 2.0)
                    
                    wx, wy = system._get_world_coords(px_center, y2)
                    if -2.0 <= wx <= 29.0 and wy <= 38.0:
                        near_player_candidates.append((kpts[i], wx, wy))

                if near_player_candidates:
                    # Select the player closest to the baseline (wy = 0)
                    near_player_keypoints = min(near_player_candidates, key=lambda p: abs(p[2] - 0.0))[0]

            # --- B. System Update ---
            system.process_frame_pose(near_player_keypoints)

            # --- C. GRU Inference (Every Second) ---
            if system.is_buffer_full():
                current_second += 1
                
                seq_tensor = system.get_sequence_tensor().to(device)
                
                with torch.no_grad():
                    # Assuming model outputs a raw logit or probability of ACTIVE
                    output = gru_model(seq_tensor)
                    
                    # Apply sigmoid if the model outputs raw logits, otherwise use output directly
                    p_active = torch.sigmoid(output).item() if not isinstance(output, float) else output.item()
                    
                    p_dead = 1.0 - p_active
                
                csv_writer.writerow([current_second, p_dead])
                print(f"[Time: {current_second}s] P_Dead: {p_dead:.4f}")

    cap.release()
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anti-Vision Two-State System")
    parser.add_argument("--video_in", type=str, required=True, help="Input video path")
    parser.add_argument("--output_csv", type=str, default="dead_time_probabilities.csv", help="Output CSV path")
    parser.add_argument("--gru_model", type=str, default="/Users/tennis/Documents/Code/Laptop/src/anya/spikes/models/active_model.pt", help="Path to trained GRU pose estimator")
    parser.add_argument("--headless", action="store_true", help="Disable UI for court calibration")

    args = parser.parse_args()

    run_state_estimator(args.video_in, args.output_csv, args.gru_model, headless=args.headless)