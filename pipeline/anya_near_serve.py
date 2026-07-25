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
import os
import random
from sklearn.cluster import DBSCAN

try:
    from utilities import _is_in_exclusion_zone, create_highlights_ffmpeg
except ImportError:
    from .utilities import _is_in_exclusion_zone, create_highlights_ffmpeg

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
    def __init__(self, court_corners_pixels: np.ndarray, video_width: int, video_height: int, fps: int = 30,
                 params: Optional[Dict] = None, verbose: bool = True):
        self.state = MatchState.WAITING
        self.fps = fps
        self.verbose = verbose   # False silences state-transition prints (used by the optimizer)
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
        self.POINT_START_GRACE_SEC = 2.5    # energy cannot drain until this long after the serve registers
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
        self.use_fusion = False                # True: point-end driven by the pose+ball fusion

        # Optional overrides (used by the parameter optimizer) — set after the
        # defaults above so a params dict can tune any energy constant in place.
        if params:
            for k, v in params.items():
                if not hasattr(self, k):
                    raise KeyError(f"Unknown PointStartSystem param: {k}")
                setattr(self, k, v)

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

        # Start-of-point grace: for the first POINT_START_GRACE_SEC after the serve
        # registers, energy may still rise but is never allowed to drain.
        in_start_grace = (self.current_point_start is not None and
                          frame_idx - self.current_point_start < int(self.fps * self.POINT_START_GRACE_SEC))
        if in_start_grace and delta < 0:
            delta = 0.0
            labels = [f"START-GRACE {(frame_idx - self.current_point_start) / self.fps:.1f}s"]

        self.energy = max(0.0, min(self.ENERGY_MAX, self.energy + delta))
        self.energy_status = " + ".join(labels) if labels else "-"
        return self.energy

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def process_frame(self, current_data: TrackData) -> MatchState:
        self.frame_buffer.append(current_data)
        if len(self.frame_buffer) > self.fps * 4: self.frame_buffer.pop(0)
        self.current_toss_roi = None 
        
        if self.state == MatchState.WAITING:
            if current_data.near_player_bbox and self._check_near_dwell():
                self.state = MatchState.ARMED
                self._log(f"[State] Player ready. Transition to ARMED at frame {current_data.frame_idx}")
            
        elif self.state == MatchState.ARMED:
            if not self._check_near_dwell():
                self.state = MatchState.WAITING
                self.toss_detected_frame_idx = 0
                self._log(f"[State] Player broke dwell. Reset to WAITING")
                return self.state

            px, py, pw, ph = current_data.near_player_bbox
            self.current_toss_roi = self._get_near_toss_roi(px, py, pw, ph)
            
            if self._detect_near_toss():
                self.toss_detected_frame_idx = current_data.frame_idx
                self._log(f"[Event] Toss detected at frame {current_data.frame_idx}")
                
            # If toss happened within the last 1 second, look for rapid ratio shift
            if self.toss_detected_frame_idx > 0:
                frames_since_toss = current_data.frame_idx - self.toss_detected_frame_idx
                
                if frames_since_toss <= int(self.fps * 1.5):
                    if self._detect_ratio_shift():
                        self._trigger_active(current_data.frame_idx, "Serve Executed (Toss + Ratio Shift)")
                        self.toss_detected_frame_idx = 0
                else:
                    self._log(f"[Event] Toss timed out. Continuing ARMED watch.")
                    self.toss_detected_frame_idx = 0

        elif self.state == MatchState.ACTIVE:
            self.active_frame_counter += 1
            self._update_active_buffers(current_data)
            self._update_energy(current_data.frame_idx)

            timed_out = self.active_frame_counter >= int(self.fps * self.MAX_POINT_SEC)
            if self.use_fusion:
                # Point-end is driven externally by the pose+ball fusion in the
                # frame loop (via end_active_point); here we only enforce the cap.
                if timed_out:
                    self.end_active_point(current_data.frame_idx, "max duration")
            else:
                depleted = self.energy <= self.ENERGY_DEAD
                if depleted or timed_out:
                    reason = "energy depleted" if depleted else "max duration"
                    self.end_active_point(current_data.frame_idx, reason)

        return self.state

    def end_active_point(self, frame_idx: int, reason: str, end_frame: int = None):
        """Record the current point and reset to WAITING. end_frame lets the
        caller apply an offset (e.g. align a player-transition end to the ball)."""
        self.points.append({
            "start_frame": self.current_point_start,
            "end_frame": end_frame if end_frame is not None else frame_idx,
            "reason": reason,
        })
        self._log(f"[State] Point ENDED at frame {frame_idx} ({reason}). Reset to WAITING")
        self.state = MatchState.WAITING
        self.active_frame_counter = 0
        self.current_point_start = None

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

        self._log(f"[State] Point STARTED at frame {frame_idx} via {trigger_source}")


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

ANALYSIS_SIZE = (960, 540)   # pose is evaluated here to match the trained model
_N_KP = 17


def _iou_xywh_xyxy(nb, xyxy):
    ax1, ay1, aw, ah = nb
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bx2, by2 = xyxy
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    ua = aw * ah + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _select_near_pose(result, near_bbox, iou_min=0.2):
    if near_bbox is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    kpts = result.keypoints.data.cpu().numpy()
    best_i, best = -1, iou_min
    for i, b in enumerate(boxes):
        j = _iou_xywh_xyxy(near_bbox, b)
        if j > best:
            best, best_i = j, i
    return kpts[best_i] if best_i >= 0 else None


def _normalize_kp(kp, near_bbox):
    bx, by, bw, bh = near_bbox
    bw = bw or 1.0; bh = bh or 1.0
    out = np.empty(_N_KP * 3, dtype=np.float32)
    for k in range(_N_KP):
        x, y, c = kp[k]
        out[3*k], out[3*k+1], out[3*k+2] = (x - bx) / bw, (y - by) / bh, c
    return out


class ActiveDeadFusion:
    """Live pose+ball fusion point-end. Holds the trained GRU and a rolling 2s
    pose buffer; each frame returns the fused activity and whether the point
    should end (sustained-dead run after a full-window start grace)."""
    def __init__(self, model_path, fps, device="cpu", w_player=0.8, w_ball=0.2,
                 thr=0.45, sustain_sec=1.0, smooth_sec=0.4, start_grace_sec=2.0,
                 win_sec=2.0, offset_sec=0.0):
        import torch
        from train_active import make_model, featurize
        self._torch = torch
        self._featurize = featurize
        ck = torch.load(model_path, map_location=device)
        self.model = make_model(); self.model.load_state_dict(ck["state"]); self.model.eval()
        self.fps = fps
        self.win = max(2, round(fps * win_sec))
        self.pose_buf: deque = deque(maxlen=self.win)
        self.a_buf: deque = deque(maxlen=max(1, round(fps * smooth_sec)))
        self.w_player, self.w_ball, self.thr = w_player, w_ball, thr
        self.sustain = max(1, round(fps * sustain_sec))
        self.grace = round(fps * start_grace_sec)
        self.offset = round(offset_sec * fps)
        self.run = 0
        self.P = 0.0
        self.A = 0.0

    def reset(self):
        self.pose_buf.clear(); self.a_buf.clear(); self.run = 0; self.P = 0.0; self.A = 0.0

    def push_pose(self, kp):
        self.pose_buf.append(kp)   # normalized [51] or None

    def _p_active(self):
        buf = list(self.pose_buf)
        W = np.full((self.win, _N_KP * 3), np.nan, dtype=np.float32)
        base = self.win - len(buf)
        for i, kp in enumerate(buf):
            if kp is not None:
                W[base + i] = kp
        idx = np.linspace(0, self.win - 1, 60).round().astype(int)
        Xf = self._featurize(W[idx][None, ...])
        with self._torch.no_grad():
            return float(self._torch.sigmoid(self.model(self._torch.tensor(Xf))).item())

    def step(self, ball_live, frames_since_start):
        self.P = self._p_active()
        a = self.w_player * self.P + self.w_ball * (1.0 if ball_live else 0.0)
        self.a_buf.append(a)
        self.A = sum(self.a_buf) / len(self.a_buf)
        ended = False
        if frames_since_start >= self.grace:
            self.run = self.run + 1 if self.A < self.thr else 0
            if self.run >= self.sustain:
                ended = True
        return self.A, ended


def _excl_cache_path(video_path: str) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    s = os.path.splitext(os.path.basename(video_path))[0]
    # Distinct from the shared pipeline cache so a 960x540-space cache can't be
    # loaded here and drawn on a full-resolution frame at the wrong scale.
    return os.path.join(d, f"{s}_near_exclusion_cache.json")


def _merge_overlapping_zones(zones, gap: int = 0):
    """Union any rectangles that overlap (or sit within `gap` px), repeatedly,
    so a fragmented cluster reads as one clean zone."""
    zones = [list(z) for z in zones]
    changed = True
    while changed:
        changed = False
        out = []
        while zones:
            a = zones.pop()
            i = 0
            while i < len(zones):
                b = zones[i]
                overlap = not (a[2] + gap < b[0] or b[2] + gap < a[0] or
                               a[3] + gap < b[1] or b[3] + gap < a[1])
                if overlap:
                    z = zones.pop(i)
                    a = [min(a[0], z[0]), min(a[1], z[1]), max(a[2], z[2]), max(a[3], z[3])]
                    changed = True
                else:
                    i += 1
            out.append(a)
        zones = out
    return [tuple(z) for z in zones]


def build_auto_exclusion_zones(video_path: str, ball_model, device, num_frames: int = 60,
                               conf: float = 0.05, eps: int = None, min_samples: int = 8,
                               padding: int = 8) -> List[Tuple[int, int, int, int]]:
    """
    Full-resolution variant of utilities.create_auto_exclusion_zones: sample
    random frames, DBSCAN-cluster stationary ball detections by center, then
    bound each cluster by the UNION of its member detection boxes (not just
    their centers) so the zone covers the actual object footprint.

    For recall on high-res footage: more frames give each stationary object
    more chances to be seen, eps scales with resolution so a cluster doesn't
    fragment, min_samples is modest so intermittently-detected objects still
    survive, and overlapping boxes are merged into one zone.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    num_frames = min(num_frames, total)
    if total < min_samples:
        cap.release()
        return []

    if eps is None:
        eps = max(5, int(round(6.0 * width / 960.0)))  # ~24px at 4K, ~5px at 960

    centers, boxes = [], []
    for fi in random.sample(range(total), num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        res = ball_model.predict(frame, conf=conf, imgsz=1920, device=device, verbose=False)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            centers.append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))
            boxes.append((x1, y1, x2, y2))
    cap.release()

    print(f"[INFO]   scanned {num_frames} frames, {len(centers)} ball detection(s), "
          f"eps={eps}, min_samples={min_samples}")
    if len(centers) < min_samples:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(centers)).labels_
    boxes = np.array(boxes)
    zones = []
    for k in set(labels):
        if k == -1:
            continue
        member = boxes[labels == k]
        x1, y1 = member[:, 0].min(), member[:, 1].min()
        x2, y2 = member[:, 2].max(), member[:, 3].max()
        zones.append((int(x1 - padding), int(y1 - padding),
                      int(x2 + padding), int(y2 + padding)))
    return _merge_overlapping_zones(zones, gap=padding)


def load_or_build_exclusion_zones(video_path, ball_model, device, padding=8, rescan=False,
                                  num_frames=60, min_samples=8):
    path = _excl_cache_path(video_path)
    if not rescan and os.path.isfile(path):
        try:
            with open(path) as f:
                zones = [tuple(z) for z in json.load(f)]
            print(f"[INFO] Loaded {len(zones)} exclusion zone(s) from cache")
            return zones
        except Exception as e:
            print(f"[WARN] Exclusion cache unreadable ({e}), recomputing")

    print("[INFO] Scanning video for static exclusion zones...")
    try:
        zones = build_auto_exclusion_zones(video_path, ball_model, device, padding=padding,
                                           num_frames=num_frames, min_samples=min_samples)
    except Exception as e:
        print(f"[WARN] Could not compute exclusion zones: {e}")
        return []
    try:
        with open(path, "w") as f:
            json.dump([list(z) for z in zones], f)
    except Exception as e:
        print(f"[WARN] Could not save exclusion cache: {e}")
    return zones


def create_highlight_reel(video_path, points, fps, output_path,
                          pre_roll: float = 1.0, post_roll: float = 1.0):
    """
    Splice each detected active point (start_frame..end_frame) into a single
    highlight reel with pre_roll seconds before the serve and post_roll seconds
    after the point end. Delegates to utilities.create_highlights_ffmpeg
    (audio-preserving): the post-roll is baked into each segment end, and
    pre_roll plus overlap-merging (merge_gap_sec=0) handle the starts so
    back-to-back points never produce duplicated footage.
    """
    segments = []
    for p in points:
        sf, ef = p.get("start_frame"), p.get("end_frame")
        if sf is None or ef is None:
            continue
        segments.append((sf / fps, ef / fps + post_roll))

    if not segments:
        print("[HIGHLIGHT] No active points to export.")
        return

    create_highlights_ffmpeg(video_path, segments, output_path,
                             pre_roll=pre_roll, merge_gap_sec=0.0)


def run_point_detector(video_path: str, output_path: str, ball_model_path: str, stride: int = 10, headless: bool = False, energy_debug: bool = False, exclusion_padding: int = 8, rescan_exclusion: bool = False, exclusion_frames: int = 60, exclusion_min_samples: int = 8, highlights: bool = False, highlight_out: str = None, pre_roll: float = 1.0, post_roll: float = 1.0, energy_params: str = None, active_model: str = None, active_offset_sec: float = 0.0):

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
    
    tuned = None
    if energy_params:
        with open(energy_params) as f:
            data = json.load(f)
        tuned = data.get("params", data)   # accept either {"params": {...}} or a bare dict
        print(f"[INFO] Loaded {len(tuned)} tuned energy param(s) from {energy_params}")
    system = PointStartSystem(court_corners, width, height, fps=fps, params=tuned)

    # Optional learned point-end: pose GRU + ball fusion replaces the energy bar.
    fusion = None
    pose_model = None
    if active_model:
        pose_model = YOLO("yolov8n-pose.pt")
        fusion = ActiveDeadFusion(active_model, fps, device=device, offset_sec=active_offset_sec)
        system.use_fusion = True
        print(f"[INFO] Pose+ball fusion point-end enabled ({active_model}, offset {active_offset_sec}s)")
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
    # Zones bound the union of the member detection boxes (full-resolution) so
    # they cover the real object footprint, and are cached beside the video.
    exclusion_zones = load_or_build_exclusion_zones(
        video_path, yolo_ball_model, device, padding=exclusion_padding, rescan=rescan_exclusion,
        num_frames=exclusion_frames, min_samples=exclusion_min_samples)
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

        # --- C2. Pose + ball fusion point-end (learned) ---
        if fusion is not None and current_state == MatchState.ACTIVE:
            if system.active_frame_counter == 0:      # serve trigger frame → new point
                fusion.reset()
            norm = None
            if near_player_bbox is not None:
                sx, sy = ANALYSIS_SIZE[0] / width, ANALYSIS_SIZE[1] / height
                nb960 = (near_player_bbox[0]*sx, near_player_bbox[1]*sy,
                         near_player_bbox[2]*sx, near_player_bbox[3]*sy)
                frame960 = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
                pres = pose_model.predict(frame960, imgsz=640, device=device, verbose=False)[0]
                kp = _select_near_pose(pres, nb960)
                norm = _normalize_kp(kp, nb960) if kp is not None else None
            fusion.push_pose(norm)
            ball_live = system._has_live_ball_trace(frame_idx)
            _, ended = fusion.step(ball_live, system.active_frame_counter)
            if ended:
                system.end_active_point(frame_idx, "pose fusion", end_frame=frame_idx + fusion.offset)
                current_state = system.state

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

        # Activity bar — fusion (A) when the learned point-end is on, else energy.
        if fusion is not None and current_state == MatchState.ACTIVE:
            bar_h, pad = 26, 20
            bar_w_max = width - 2 * pad
            by0 = height - bar_h - pad
            cv2.rectangle(frame, (pad, by0), (pad + bar_w_max, by0 + bar_h), (40, 40, 40), -1)
            a = max(0.0, min(1.0, fusion.A))
            cv2.rectangle(frame, (pad, by0), (pad + int(bar_w_max * a), by0 + bar_h),
                          (0, int(255 * a), int(255 * (1.0 - a))), -1)
            cv2.rectangle(frame, (pad + int(bar_w_max * fusion.thr), by0),
                          (pad + int(bar_w_max * fusion.thr), by0 + bar_h), (255, 255, 255), 1)
            cv2.putText(frame, f"ACTIVE P={fusion.P:.2f} A={fusion.A:.2f} (thr {fusion.thr})",
                        (pad + 6, by0 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        elif energy_debug or current_state == MatchState.ACTIVE:
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

    # Flush a point still in progress at end-of-video so it makes the reel.
    if system.state == MatchState.ACTIVE and system.current_point_start is not None:
        system.points.append({
            "start_frame": system.current_point_start,
            "end_frame": frame_idx,
            "reason": "video end",
        })

    json_path = "serve_events.json"
    with open(json_path, 'w') as f:
        json.dump({"serves": system.serve_events, "points": system.points}, f, indent=4)
    print(f"Finished processing. {len(system.serve_events)} serve(s), {len(system.points)} point(s) saved to {json_path}")

    if highlights:
        if highlight_out is None:
            d = os.path.dirname(os.path.abspath(video_path))
            s = os.path.splitext(os.path.basename(video_path))[0]
            highlight_out = os.path.join(d, f"{s}_highlights.mp4")
        create_highlight_reel(video_path, system.points, fps, highlight_out,
                              pre_roll=pre_roll, post_roll=post_roll)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anya Tennis Near-Serve Detector")
    parser.add_argument("--video_in", type=str, default="/Volumes/Anya/Data/21/snippet.mp4", help="Input video path")
    parser.add_argument("--video_out", type=str, default="/Volumes/Anya/Data/21/output_match_annotated.mp4", help="Output video path")
    parser.add_argument("--ball_model", type=str, default="/Users/tennis/Documents/Code/Laptop/src/anya/pipeline/models/ball_best.pt", help="Ball YOLO model path")
    parser.add_argument("--headless", action="store_true", help="Disable real-time visualizations")
    parser.add_argument("--energy_debug", action="store_true",
                        help="Run the energy bar every frame (all states) with a near-player velocity + ball-trace readout, for tuning")
    parser.add_argument("--exclusion_padding", type=int, default=8,
                        help="Pixels to expand each auto exclusion zone beyond the detected object footprint")
    parser.add_argument("--rescan_exclusion", action="store_true",
                        help="Ignore the cached exclusion zones and recompute them")
    parser.add_argument("--exclusion_frames", type=int, default=60,
                        help="Number of random frames to scan for exclusion zones (more = better recall, slower)")
    parser.add_argument("--exclusion_min_samples", type=int, default=8,
                        help="Min clustered detections for a zone (lower = catches more, intermittently-seen objects)")
    parser.add_argument("--highlights", action="store_true",
                        help="Splice the active points into a highlight reel (on by default when --active_model is set)")
    parser.add_argument("--no_highlights", action="store_true",
                        help="Disable the highlight reel even when --active_model is set")
    parser.add_argument("--highlight_out", type=str, default=None,
                        help="Highlight reel path (default: <video>_highlights.mp4 beside the input)")
    parser.add_argument("--pre_roll", type=float, default=1.0,
                        help="Seconds of footage before each serve in the highlight reel")
    parser.add_argument("--post_roll", type=float, default=1.0,
                        help="Seconds of footage after each point end in the highlight reel")
    parser.add_argument("--energy_params", type=str, default=None,
                        help="Path to energy_params.json (from optimize_energy.py) to load tuned energy constants")
    parser.add_argument("--active_model", type=str, default=None,
                        help="Path to active_model.pt (pose GRU) — enables the learned pose+ball fusion point-end")
    parser.add_argument("--active_offset_sec", type=float, default=0.0,
                        help="Seconds added to the fusion point-end (e.g. ~1.7 to align player-transition to ball end)")

    args = parser.parse_args()

    # Highlights default on whenever the learned point-end is used, unless disabled.
    make_highlights = (args.highlights or bool(args.active_model)) and not args.no_highlights

    run_point_detector(args.video_in, args.video_out, args.ball_model,
                       headless=args.headless, energy_debug=args.energy_debug,
                       exclusion_padding=args.exclusion_padding, rescan_exclusion=args.rescan_exclusion,
                       exclusion_frames=args.exclusion_frames, exclusion_min_samples=args.exclusion_min_samples,
                       highlights=make_highlights, highlight_out=args.highlight_out,
                       pre_roll=args.pre_roll, post_roll=args.post_roll, energy_params=args.energy_params,
                       active_model=args.active_model, active_offset_sec=args.active_offset_sec)