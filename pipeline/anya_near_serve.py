import cv2
import numpy as np
import json
import math
import torch
import argparse
from enum import Enum
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from ultralytics import YOLO

try:
    from utilities import (create_auto_exclusion_zones, _is_in_exclusion_zone,
                           load_cached_exclusion_zones, save_cached_exclusion_zones)
except ImportError:
    from .utilities import (create_auto_exclusion_zones, _is_in_exclusion_zone,
                            load_cached_exclusion_zones, save_cached_exclusion_zones)

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

        # ── Energy bar (point-end detection) ──────────────────────────────
        # The point stays alive while the system has "energy". A live ball
        # trace or the near player moving rapidly pumps energy in; the near
        # player still/paused with no ball trace drains it hard. When the
        # energy is exhausted the rally is declared dead. (The far player is
        # deliberately excluded — its bounding box is too noisy to trust.)
        self.energy = 0.0
        self.energy_status = "-"
        self.ENERGY_MAX            = 1.0
        self.ENERGY_START          = 0.6    # energy seeded on entering ACTIVE (survives serve gap)
        self.ENERGY_DEAD           = 0.02   # point ends at/below this
        self.ENERGY_BOOST_BALL     = 3.0    # /s while a live ball trace exists
        self.ENERGY_BOOST_MOTION   = 2.5    # /s while either player sprints
        self.ENERGY_DECAY_DEAD     = 1.5    # /s when the near player is still AND no ball (substantial)
        self.ENERGY_DECAY_BASE     = 0.4    # /s mild idle drain otherwise
        self.ENERGY_DECAY_MISSING  = 2.5    # /s rapid drain once the near player has been missing past the grace

        self.PLAYER_FAST_FTS       = 6.0    # world-space ft/s → "moving rapidly"
        self.PLAYER_STILL_FTS      = 2.0    # world-space ft/s → "very low velocity"
        self.STILL_PROLONGED_SEC   = 0.6    # the near player must be still this long before the hard drain
        self.PLAYER_MISSING_GRACE_SEC = 3.0 # near player may be missing this long (energy holds) before rapid drain
        self.BALL_TRACE_SEC        = 0.7    # a ball seen within this window counts as a live trace
        self.TRACE_MOVE_PX         = 10.0   # min spread across the trace window to count as "live" (not stationary)
        self.MAX_POINT_SEC         = 40.0   # hard safety cap on rally length

        vel_window = max(3, int(self.fps * 0.5))
        self.near_pos_buffer: deque = deque(maxlen=vel_window)
        self.ball_trace:      deque = deque()
        self.still_frames = 0
        self.missing_frames = 0

        self.points: List[Dict] = []          # completed serve→point-end records
        self.current_point_start: Optional[int] = None

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

    # ── Energy bar helpers ────────────────────────────────────────────────
    def _player_velocity_fts(self, pos_buffer: deque) -> float:
        """World-space speed (ft/s) from oldest→newest foot position in the buffer."""
        if len(pos_buffer) < 3:
            return 0.0
        (ox, oy), (nx, ny) = pos_buffer[0], pos_buffer[-1]
        elapsed = len(pos_buffer) / self.fps
        if elapsed <= 0:
            return 0.0
        return math.hypot(nx - ox, ny - oy) / elapsed

    def _has_live_ball_trace(self, frame_idx: int) -> bool:
        """True if the ball has been seen recently and is actually moving (not sitting still)."""
        window = int(self.fps * self.BALL_TRACE_SEC)
        recent = [p for p in self.ball_trace if frame_idx - p[0] <= window]
        if not recent:
            return False
        if len(recent) == 1:
            return True
        xs = [p[1] for p in recent]
        ys = [p[2] for p in recent]
        spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return spread > self.TRACE_MOVE_PX

    def _update_active_buffers(self, d: TrackData):
        """Push per-frame telemetry (foot positions + ball trace) used by the energy bar."""
        if d.near_player_bbox:
            self.missing_frames = 0
            x, y, w, h = d.near_player_bbox
            self.near_pos_buffer.append(self._get_world_coords(x + w / 2.0, y + h))
        else:
            self.missing_frames += 1
            self.near_pos_buffer.clear()  # avoid a teleport-velocity spike when the player reappears
        if d.ball_pos:
            self.ball_trace.append((d.frame_idx, d.ball_pos[0], d.ball_pos[1]))
        window = int(self.fps * self.BALL_TRACE_SEC)
        while self.ball_trace and d.frame_idx - self.ball_trace[0][0] > window:
            self.ball_trace.popleft()

    def _update_energy(self, frame_idx: int) -> float:
        """Advance the energy bar one frame based on the ball trace + near-player motion."""
        dt = 1.0 / self.fps
        has_ball  = self._has_live_ball_trace(frame_idx)
        missing   = self.missing_frames > 0
        near_v    = self._player_velocity_fts(self.near_pos_buffer)
        near_fast = (not missing) and near_v > self.PLAYER_FAST_FTS
        near_slow = (not missing) and near_v < self.PLAYER_STILL_FTS

        # A prolonged pause with no ball is the trigger for the hard drain.
        if near_slow and not has_ball:
            self.still_frames += 1
        else:
            self.still_frames = 0

        delta = 0.0
        labels: List[str] = []
        if has_ball:
            delta += self.ENERGY_BOOST_BALL * dt
            labels.append("BALL")
        if near_fast:
            delta += self.ENERGY_BOOST_MOTION * dt
            labels.append(f"MOTION {near_v:.1f}ft/s")

        if missing and not has_ball:
            # Grace: energy holds while the near player is briefly lost. Past the
            # grace window a continued absence drains rapidly.
            missing_sec = self.missing_frames / self.fps
            if self.missing_frames > int(self.fps * self.PLAYER_MISSING_GRACE_SEC):
                delta -= self.ENERGY_DECAY_MISSING * dt
                labels.append(f"MISSING {missing_sec:.1f}s (rapid drain)")
            else:
                labels.append(f"MISSING {missing_sec:.1f}s (grace)")
        elif not has_ball and not near_fast:
            prolonged = self.still_frames >= int(self.fps * self.STILL_PROLONGED_SEC)
            if near_slow and prolonged:
                delta -= self.ENERGY_DECAY_DEAD * dt
                labels.append("DEAD (near still, no ball)")
            else:
                delta -= self.ENERGY_DECAY_BASE * dt
                labels.append("draining")

        self.energy = max(0.0, min(self.ENERGY_MAX, self.energy + delta))
        self.energy_status = " + ".join(labels) if labels else "-"
        return self.energy

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
            self._update_active_buffers(current_data)
            self._update_energy(current_data.frame_idx)

            depleted = self.energy <= self.ENERGY_DEAD
            timed_out = self.active_frame_counter >= int(self.fps * self.MAX_POINT_SEC)
            if depleted or timed_out:
                reason = "energy depleted" if depleted else "max duration"
                self.points.append({
                    "start_frame": self.current_point_start,
                    "end_frame": current_data.frame_idx,
                    "reason": reason,
                })
                print(f"[State] Point ENDED at frame {current_data.frame_idx} ({reason}). Reset to WAITING")
                self.state = MatchState.WAITING
                self.active_frame_counter = 0
                self.current_point_start = None

        return self.state

    def _trigger_active(self, frame_idx: int, trigger_source: str):
        self.state = MatchState.ACTIVE
        self.active_frame_counter = 0
        self.serve_events.append({"frame": frame_idx, "event": trigger_source})

        # Seed the energy bar so the serve/return gap doesn't kill the fresh point,
        # and clear the per-point telemetry buffers.
        self.energy = self.ENERGY_START
        self.energy_status = "SERVE"
        self.still_frames = 0
        self.missing_frames = 0
        self.near_pos_buffer.clear()
        self.ball_trace.clear()
        self.current_point_start = frame_idx

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

def run_point_detector(video_path: str, output_path: str, ball_model_path: str, stride: int = 10, headless: bool = False, energy_debug: bool = False):

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

    # Bounding rect of the court (with margin) — the ball-search region while a point is ACTIVE.
    margin = int(0.10 * height)
    xs, ys = court_corners[:, 0], court_corners[:, 1]
    court_rect = (
        max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + margin), min(height, int(ys.max()) + margin),
    )

    # Static exclusion zones — auto-detected clusters of stationary ball-like
    # objects (e.g. ball baskets, court logos) whose detections should be ignored.
    # Reused from utilities.create_auto_exclusion_zones; scanned at full resolution
    # so the rects share ball_pos' full-frame coords, and cached beside the video.
    exclusion_zones = load_cached_exclusion_zones(video_path) or []
    if not exclusion_zones:
        print("[INFO] Scanning video for static exclusion zones...")
        try:
            exclusion_zones = create_auto_exclusion_zones(video_path, yolo_ball_model, analysis_size=None)
            save_cached_exclusion_zones(video_path, exclusion_zones)
        except Exception as e:
            print(f"[WARN] Could not compute exclusion zones: {e}")
            exclusion_zones = []
    print(f"[INFO] {len(exclusion_zones)} exclusion zone(s) active")

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
        # ARMED: tight toss ROI (serve detection). ACTIVE (or energy-debug): full
        # court region so the energy bar sees a live ball trace during the rally.
        ball_pos = None
        ball_roi = None
        ball_imgsz = 256
        if system.current_toss_roi is not None:
            ball_roi = system.current_toss_roi
            ball_imgsz = 256
        elif system.state == MatchState.ACTIVE or energy_debug:
            ball_roi = court_rect
            ball_imgsz = 640

        if ball_roi is not None:
            r_left, r_top, r_right, r_bottom = ball_roi
            c_left, c_top = max(0, int(r_left)), max(0, int(r_top))
            c_right, c_bottom = min(width, int(r_right)), min(height, int(r_bottom))

            if c_right > c_left and c_bottom > c_top:
                roi_crop = frame[c_top:c_bottom, c_left:c_right]
                ball_results = yolo_ball_model.predict(roi_crop, imgsz=ball_imgsz, device=device, verbose=False)[0]

                for box in ball_results.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    cand = (bx1 + c_left + (bx2 - bx1)/2.0, by1 + c_top + (by2 - by1)/2.0)
                    if _is_in_exclusion_zone(cand[0], cand[1], exclusion_zones):
                        continue  # stationary ball-like object (basket, logo) — ignore
                    ball_pos = cand
                    break

        # --- C. System Update ---
        track_data = TrackData(
            frame_idx=frame_idx, near_player_bbox=near_player_bbox, ball_pos=ball_pos
        )
        current_state = system.process_frame(track_data)

        # In energy-debug mode keep the bar alive outside ACTIVE so its response to
        # real ball/player motion is visible across the whole clip. ACTIVE frames are
        # already advanced inside process_frame; here we drive the other states.
        if energy_debug and current_state != MatchState.ACTIVE:
            system._update_active_buffers(track_data)
            system._update_energy(frame_idx)

        # --- D. Visualizations ---
        cv2.putText(frame, f"State: {current_state.name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.polylines(frame, [ready_zone_poly], isClosed=True, color=(0, 255, 0), thickness=2)

        for (zx1, zy1, zx2, zy2) in exclusion_zones:
            cv2.rectangle(frame, (int(zx1), int(zy1)), (int(zx2), int(zy2)), (255, 0, 255), 2)
            cv2.putText(frame, "EXCL", (int(zx1), int(zy1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        
        if near_player_bbox:
            nx, ny, nw, nh = near_player_bbox
            cv2.rectangle(frame, (int(nx), int(ny)), (int(nx+nw), int(ny+nh)), (255, 0, 0), 2)

        if ball_pos:
            cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 5, (0, 255, 255), -1)

        if system.current_toss_roi:
            r_left, r_top, r_right, r_bottom = system.current_toss_roi
            cv2.rectangle(frame, (int(r_left), int(r_top)), (int(r_right), int(r_bottom)), (0, 0, 255), 2)

        # Energy bar — always visible in energy-debug mode, otherwise only while a point is live.
        if energy_debug or current_state == MatchState.ACTIVE:
            bar_h, pad = 26, 20
            bar_w_max = width - 2 * pad
            by0 = height - bar_h - pad
            cv2.rectangle(frame, (pad, by0), (pad + bar_w_max, by0 + bar_h), (40, 40, 40), -1)
            e = system.energy / system.ENERGY_MAX
            fill = int(bar_w_max * e)
            colour = (0, int(255 * e), int(255 * (1.0 - e)))  # green (full) → red (empty)
            cv2.rectangle(frame, (pad, by0), (pad + fill, by0 + bar_h), colour, -1)
            cv2.putText(frame, f"ENERGY {system.energy:.2f} [{system.energy_status}]",
                        (pad + 6, by0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

            if energy_debug:
                near_v = system._player_velocity_fts(system.near_pos_buffer)
                has_ball = system._has_live_ball_trace(frame_idx)
                readout = f"near {near_v:.1f} ft/s   ball_trace: {'LIVE' if has_ball else '--'}"
                cv2.putText(frame, readout, (pad, by0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if has_ball else (200, 200, 200), 2, cv2.LINE_AA)

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
        json.dump({"serves": system.serve_events, "points": system.points}, f, indent=4)
    print(f"Finished processing. {len(system.serve_events)} serve(s), {len(system.points)} point(s) saved to {json_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anya Tennis Near-Serve Detector")
    parser.add_argument("--video_in", type=str, default="/Volumes/Anya/Data/21/snippet.mp4", help="Input video path")
    parser.add_argument("--video_out", type=str, default="/Volumes/Anya/Data/21/output_match_annotated.mp4", help="Output video path")
    parser.add_argument("--ball_model", type=str, default="/Users/tennis/Documents/Code/Laptop/src/anya/pipeline/models/ball_best.pt", help="Ball YOLO model path")
    parser.add_argument("--headless", action="store_true", help="Disable real-time visualizations")
    parser.add_argument("--energy_debug", action="store_true",
                        help="Run the energy bar every frame (all states) with a near-player velocity + ball-trace readout, for tuning")

    args = parser.parse_args()

    run_point_detector(args.video_in, args.video_out, args.ball_model,
                       headless=args.headless, energy_debug=args.energy_debug)