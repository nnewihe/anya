import time
import random
from enum import Enum
from collections import deque
from dataclasses import dataclass
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple
import math
from sklearn.cluster import DBSCAN
from moviepy import VideoFileClip, concatenate_videoclips
import os
import json
import re
import subprocess
import argparse
import tempfile
import shutil

# ---------------------------------------------------------
# Exclusion Zone Helpers
# ---------------------------------------------------------
def create_auto_exclusion_zones(
    video_path: str,
    ball_model,
    num_frames: int = 20,
    conf: float = 0.05,
    eps: int = 30,
    min_samples: int = 3,
    padding: int = 5,
    ball_class_index: int = 0,
    analysis_size: tuple = None,
) -> List[Tuple[int, int, int, int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        return []

    frame_indices = random.sample(range(total_frames), num_frames)
    
    all_detections = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        
        if analysis_size is not None:
            frame = cv2.resize(frame, analysis_size, interpolation=cv2.INTER_AREA)

        res = ball_model(frame, verbose=False, conf=conf, imgsz=Config.BALL_IMGSZ)
        if res and res[0].boxes:
            for b in res[0].boxes:
                if int(b.cls[0]) != ball_class_index:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                all_detections.append((cx, cy))
    
    cap.release()

    if len(all_detections) < min_samples:
        return []

    X = np.array(all_detections)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    
    unique_labels = set(labels)
    zones = []
    
    for k in unique_labels:
        if k == -1:
            continue
        
        class_member_mask = (labels == k)
        cluster_points = X[class_member_mask]
        
        if len(cluster_points) > 0:
            x_min, y_min = np.min(cluster_points, axis=0)
            x_max, y_max = np.max(cluster_points, axis=0)
            
            zones.append((
                int(x_min - padding),
                int(y_min - padding),
                int(x_max + padding),
                int(y_max + padding),
            ))
            
    return zones

def _is_in_exclusion_zone(x, y, exclusion_zones):
    for (x1, y1, x2, y2) in exclusion_zones:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def _court_cache_path(video_path: str) -> str:
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_court_cache.json")


def init_court(video_path: str, target_idx: int = 300, analysis_size: tuple = None):
    cache_path = _court_cache_path(video_path)

    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)

            cached_size = tuple(cached.get("analysis_size", [None, None]))
            if cached_size == (analysis_size if analysis_size else (None, None)):
                pts = [tuple(p) for p in cached["points"]]
                shape = tuple(cached["frame_shape"])
                print(f"[COURT] Loaded cached court corners from: {os.path.basename(cache_path)}")
                for i, (x, y) in enumerate(pts):
                    print(f"  Corner {i+1}: ({x:.1f}, {y:.1f})")
                return pts, shape
            else:
                print(f"[COURT] Analysis size changed — re-selecting corners.")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[COURT] Cache file corrupt ({e}), re-selecting corners.")

    num_points = 4
    win = "Click 4 court corners (any order). Press r=reset, q=quit"

    base = get_reference_frame(video_path, target_idx=target_idx)
    if analysis_size is not None:
        base = cv2.resize(base, analysis_size, interpolation=cv2.INTER_AREA)
    img = base.copy()

    state = {"img": img, "clicked_pts": [], "done": False, "win": win, "num_points": num_points}

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, state["img"])
    cv2.setMouseCallback(win, select_points, state)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if state["done"]:
            cv2.destroyWindow(win)
            cv2.waitKey(1)

            pts = [(float(x), float(y)) for x, y in state["clicked_pts"]]
            shape = base.shape

            try:
                cache_data = {
                    "points": pts,
                    "frame_shape": list(shape),
                    "analysis_size": list(analysis_size) if analysis_size else [None, None],
                    "video": os.path.basename(video_path),
                }
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f, indent=2)
                print(f"[COURT] Saved court corners to: {os.path.basename(cache_path)}")
            except Exception as e:
                print(f"[COURT] WARN: Could not save cache: {e}")

            return pts, shape

        if key == ord("r"):
            state["clicked_pts"].clear()
            state["done"] = False
            state["img"] = base.copy()
            cv2.imshow(win, state["img"])

        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            raise RuntimeError("Court polygon selection aborted by user.")
    
def point_line_distance_px(P, A, B):
    Px, Py = P
    Ax, Ay = A
    Bx, By = B

    ABx = Bx - Ax
    ABy = By - Ay

    APx = Px - Ax
    APy = Py - Ay

    cross = abs(ABx * APy - ABy * APx)

    denom = math.hypot(ABx, ABy)
    if denom == 0:
        return 0.0

    return cross / denom

def get_reference_frame(video_path: str, target_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Could not read any frame from video.")
        return frame

    mid_idx = total_frames // 2
    ref_idx = min(target_idx, mid_idx)

    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    ok, frame = cap.read()

    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()

    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to read reference frame (idx={ref_idx}).")

    return frame


def select_points(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    state = param
    state["clicked_pts"].append((x, y))

    cv2.circle(state["img"], (x, y), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    
    if len(state["clicked_pts"]) == state["num_points"]:
        state["done"] = True

    cv2.imshow(state["win"], state["img"])

def build_mask(frame_shape, poly):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask

def point_in_mask(mask, x, y):
    h, w = mask.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return mask[int(y), int(x)] != 0


# ---------------------------------------------------------
# Video Probe
# ---------------------------------------------------------
def probe_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or fps > 300:
        print(f"[WARN] Video reported FPS={fps}, falling back to 30.0")
        fps = 30.0

    duration_sec = frame_count / fps if fps > 0 else 0.0

    info = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
    }

    print(f"\n{'='*50}")
    print(f"  VIDEO PROBE: {os.path.basename(video_path)}")
    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {frame_count}")
    print(f"  Duration   : {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
    print(f"{'='*50}\n")

    return info


def resize_for_analysis(frame):
    return cv2.resize(frame, (Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT),
                      interpolation=cv2.INTER_AREA)


# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
class Config:
    """
    Central place for all tuning parameters.
    
    V3: ACTIVE state uses player-only energy model.
    No ball tracking in ACTIVE — energy is driven entirely by
    player bounding box behavior.
    """
    
    # ===================================================
    # ACTIVE STATE: Player-Only Energy Model
    # ===================================================
    # 
    # INCREASING forces (strongest → weakest):
    #   1. Sprinting/running      → big boost
    #   2. Swinging (shape change) → moderate boost
    #
    # DECREASING forces (strongest → weakest):
    #   1. Walking (gait detected) → fastest drain
    #   2. Player missing/off-frame → medium-fast drain
    #   3. Standing still          → moderate drain
    # ===================================================

    SERVE_TO_ENERGY_DELAY = 1.5  # 1.5 second pause before energy starts draining
    
    # --- Boost rates (energy/second) ---
    ENERGY_BOOST_SPRINT = 4.0       # Strongest boost: player sprinting
    ENERGY_BOOST_SWING = 4.0        # Player box changing shape (swing/split-step)
    
    # --- Decay rates (energy/second) ---
    ENERGY_DECAY_WALKING = 0.25      # Strongest drain: walking gait detected
    ENERGY_DECAY_MISSING = 0.2      # Medium drain: player out of frame
    ENERGY_DECAY_STILL = 0.15       # Moderate drain: player standing still
    
    # --- Player velocity thresholds (pixels/frame over window) ---
    PLAYER_SPRINT_VELOCITY_THRESHOLD = 6.0   # Above = sprinting
    PLAYER_STILL_VELOCITY_THRESHOLD = 1.5    # Below = standing still
    
    # --- Shape change threshold ---
    SHAPE_CHANGE_THRESHOLD_PX = 80.0  # Min change in width+height for swing detection

    # --- Walking Gait Detection ---
    GAIT_BUFFER_FRAMES = 45           # ~0.75s at 60fps
    GAIT_MIN_REVERSALS = 2            
    GAIT_MAX_REVERSALS = 8            
    GAIT_MIN_DRIFT_PX = 10.0          
    
    # --- Absolute timeout (seconds since ACTIVE started, no ball tracking) ---
    # Safety valve: if energy never drains, force-end after this many seconds
    ABSOLUTE_ACTIVE_TIMEOUT = 45.0

    # --- Time Windows ---
    EVENT_WINDOW_SECONDS = 2.0 

    # --- Thresholds ---
    TRANSITION_SCORE_THRESHOLD = 0.5
    
    # Tiered serve detection:
    #   Path 1: Strong toss alone fires (no trophy needed)
    #   Path 2: Weaker toss + trophy confirmation fires
    #   Path 3: Neither path clears → no serve
    TOSS_MIN_RISE_FT = 4.0          # Min upward travel (feet) to register as a toss
    PLAYER_HEIGHT_FT = 5.75         # Assumed player height for px-to-ft conversion
    TOSS_SOLO_THRESHOLD = 0.5      # Toss confident enough to fire alone
    TOSS_MIN_THRESHOLD = 0.15       # Toss floor for trophy-assisted path
    TROPHY_MIN_THRESHOLD = 0.15     # Trophy floor to rescue a weaker toss
    
    COURT_X_PADDING_FT = 15.0

    # --- Model Paths ---
    DEFAULT_NEAR_TROPHY_MODEL_PATH = "weights/trophy_pose_cls2/weights/best.pt"
    DEFAULT_NEAR_TROPHY_CLASS_INDEX = 1

    DEFAULT_TROPHY_PAD = 0.30

    DEFAULT_BALL_MODEL_PATH = "weights/ball/weights/best.pt"
    DEFAULT_BALL_CLASS_INDEX = 0

    DEFAULT_BALL_CONF_MIN = 0.10
    DEFAULT_DRAW_TOSS_ROI = True

    MIN_FAR_TROPHY_CONF = 0.5
    MIN_FAR_TOSS_CONF = 0.5

    COURT_WIDTH_FT = 27.0
    COURT_LENGTH_FT = 78.0
    FT_TO_M = 0.3048
    COURT_WIDTH_M = COURT_WIDTH_FT * FT_TO_M
    COURT_LENGTH_M = COURT_LENGTH_FT * FT_TO_M

    DEFAULT_PLAYER_MODEL_PATH = "yolo26n.pt"
    VELOCITY_WINDOW_SIZE = 20

    # --- Ready State Thresholds ---
    READY_MIN_DIST_FT = -0.5
    READY_MAX_DIST_FT = 3.5
    READY_WAIT_TIME_SEC = 0.4

    # --- Armed State Thresholds ---
    ARMED_BAND_WINDOW_SEC = 2.0
    ARMED_OUT_RATIO_THRESHOLD = 0.25

    MAX_BALL_SIZE_PX = 20

    # --- Analysis Resolution ---
    ANALYSIS_HEIGHT = 540
    ANALYSIS_WIDTH = 960

    BALL_IMGSZ = 960
    PLAYER_IMGSZ = 480
    TROPHY_IMGSZ = 320
    TOSS_BALL_IMGSZ = 320

    CROP_UPSCALE_FACTOR = 2.0

    END_TRIM_BUFFER_SEC = 2.0


# ==========================================
# 2. DATA STRUCTURES
# ==========================================
class SystemState(Enum):
    WAITING = "WAITING"
    ARMED = "ARMED"
    ACTIVE = "ACTIVE"

@dataclass
class Detection:
    score: float
    timestamp: float

class SideBuffer:
    def __init__(self, name):
        self.name = name
        self.trophy_scores = deque()
        self.toss_scores = deque()

    def add_trophy_score(self, score, timestamp):
        self.trophy_scores.append(Detection(score, timestamp))
        self._cleanup_old_data(self.trophy_scores, timestamp)

    def add_toss_score(self, score, timestamp):
        self.toss_scores.append(Detection(score, timestamp))
        self._cleanup_old_data(self.toss_scores, timestamp)

    def _cleanup_old_data(self, buffer, current_time):
        while len(buffer) > 0:
            age = current_time - buffer[0].timestamp
            if age > Config.EVENT_WINDOW_SECONDS:
                buffer.popleft()
            else:
                break

    def get_max_combined_score(self):
        if not self.trophy_scores or not self.toss_scores:
            return 0.0
        max_trophy = max(d.score for d in self.trophy_scores)
        max_toss = max(d.score for d in self.toss_scores)
        return max_trophy + max_toss


# ==========================================
# 3. MAIN SYSTEM LOGIC
# ==========================================
class AnyaSystem:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_info = probe_video(video_path)

        self.fps = self.video_info["fps"]
        self.frame_width = self.video_info["width"]
        self.frame_height = self.video_info["height"]
        self.total_frames = self.video_info["frame_count"]

        self.frame_counter = 0
        self.state = SystemState.WAITING

        self.court_vertices = None
        self.TL = self.TR = self.BR = self.BL = None
        self.exclusion_zones = []

        analysis_size = (Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT)
        self.court_vertices, frame_shape = init_court(self.video_path, analysis_size=analysis_size)
        self.BL, self.BR, self.TR, self.TL = self.court_vertices  

        dst_pts = np.array([
            [0, 0],
            [Config.COURT_WIDTH_FT, 0],
            [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT],
            [0, Config.COURT_LENGTH_FT]
        ], dtype=np.float32)

        src_pts = np.array([self.BL, self.BR, self.TR, self.TL], dtype=np.float32)
        self.H, _ = cv2.findHomography(src_pts, dst_pts)

        self.near_side = SideBuffer("Near")

        self.last_ball_coord = None
        self.ball_model = YOLO(Config.DEFAULT_BALL_MODEL_PATH)  
        self.player_model = YOLO(Config.DEFAULT_PLAYER_MODEL_PATH)
        self.trophy_model = YOLO(Config.DEFAULT_NEAR_TROPHY_MODEL_PATH)

        self.near_ready_start_time = None
        self.armed_band_history = deque()
        self.last_toss_ball = None
        self.toss_upward_frames = 0  # consecutive frames with upward ball motion in toss ROI
        self.toss_start_y = None     # pixel y of ball when upward motion began
        self.toss_above_box_detections = 0  # times ball detected above player box while rising

        self.near_player_positions = deque(maxlen=Config.VELOCITY_WINDOW_SIZE)
        self.near_player_boxes = deque(maxlen=5)
        self.point_energy = 1.0
        self.active_start_time = 0.0

        # Rolling trophy score buffer for emergency override (last N frames in ACTIVE)
        self.active_trophy_buffer = deque(maxlen=4)

        # Gait detection
        self.gait_y_buffer = deque(maxlen=Config.GAIT_BUFFER_FRAMES)

        self.active_segments = []
        self.current_segment_start = None

        # Telemetry logging for offline evaluation
        self.telemetry_log = []

        # Exclusion zone analysis (still needed for ARMED ball toss detection)
        print("\n[INFO] Analyzing video for static objects to exclude...")
        try:
            self.exclusion_zones = create_auto_exclusion_zones(
                self.video_path,
                self.ball_model,
                ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                analysis_size=(Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT),
            )
            if self.exclusion_zones:
                print(f"[INFO] Found {len(self.exclusion_zones)} static zone(s) to exclude.")
        except Exception as e:
            print(f"[WARN] Could not run auto-exclusion analysis: {e}")
            self.exclusion_zones = []

    # ---------------------------------------------------------
    # Telemetry helpers
    # ---------------------------------------------------------
    TELEMETRY_COLUMNS = [
        "frame", "time_s", "state",
        "near_pos_x", "near_pos_y",
        "near_box_x1", "near_box_y1", "near_box_x2", "near_box_y2",
        "player_world_x_ft", "player_world_y_ft",
        "in_band",
        "trophy_conf", "toss_score",
        "max_trophy_window", "max_toss_window", "serve_score",
        "player_velocity", "shape_change", "walking_gait",
        "point_energy", "energy_delta",
    ]

    def _log_telemetry(self, frame, time_s, state, **kwargs):
        row = {"frame": frame, "time_s": time_s, "state": state}
        row.update(kwargs)
        self.telemetry_log.append(row)

    def save_telemetry(self, path):
        import csv as _csv
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=self.TELEMETRY_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for row in self.telemetry_log:
                w.writerow(row)
        print(f"[INFO] Telemetry saved to {path} ({len(self.telemetry_log)} rows)")

    # ---------------------------------------------------------
    # Export highlights
    # ---------------------------------------------------------
    def export_highlights(self, output_path="clean_highlights.mp4"):
        if not self.active_segments:
            print("[INFO] No active segments to export.")
            return

        merged_segments = []
        for start, end in sorted(self.active_segments):
            if merged_segments and merged_segments[-1][1] >= start:
                merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], end))
            else:
                merged_segments.append((start, end))

        print(f"\n[INFO] Found {len(merged_segments)} active rallies.")

        fps = self.fps

        txt_path = output_path.rsplit('.', 1)[0] + "_timestamps.txt"
        
        try:
            with open(txt_path, "w") as f:
                f.write("🎾 Anya System V3 - Active Rally Timestamps 🎾\n")
                f.write(f"Source Video: {self.video_path}\n")
                f.write(f"FPS: {fps}\n")
                f.write(f"Resolution: {self.frame_width}x{self.frame_height}\n")
                f.write(f"Energy Model: Player-Only (no ball tracking in ACTIVE)\n")
                f.write("-" * 45 + "\n")
                
                for i, (start_f, end_f) in enumerate(merged_segments):
                    start_time = start_f / fps
                    end_time = end_f / fps
                    duration = end_time - start_time
                    
                    line = f"Rally {i+1:02d} | Time: {start_time:>7.2f}s to {end_time:>7.2f}s | Duration: {duration:>5.2f}s | Frames: {start_f}-{end_f}\n"
                    f.write(line)
                    
            print(f"[INFO] Timestamps successfully backed up to: {txt_path}")
        except Exception as e:
            print(f"[WARN] Failed to write timestamps to {txt_path}: {e}")

        n = len(merged_segments)
        print(f"\nBuilding single-pass FFmpeg command for {n} rallies...")

        filter_parts = []
        concat_inputs = []

        for i, (start_f, end_f) in enumerate(merged_segments):
            start_time = start_f / fps
            end_time = end_f / fps
            duration = end_time - start_time

            filter_parts.append(
                f"[0:v]trim=start={start_time:.4f}:end={end_time:.4f},setpts=PTS-STARTPTS[v{i}]"
            )
            filter_parts.append(
                f"[0:a]atrim=start={start_time:.4f}:end={end_time:.4f},asetpts=PTS-STARTPTS[a{i}]"
            )
            concat_inputs.append(f"[v{i}][a{i}]")
            print(f"  {i+1:02d}/{n} | {start_time:.2f}s → {end_time:.2f}s  ({duration:.2f}s)")

        concat_str = "".join(concat_inputs) + f"concat=n={n}:v=1:a=1[outv][outa]"
        filter_parts.append(concat_str)
        filtergraph = ";\n".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            "-i", self.video_path,
            "-filter_complex", filtergraph,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ]

        print(f"\nEncoding highlight video (single pass)...")

        try:
            result = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
            if result.returncode != 0:
                print(f"[WARN] Single-pass failed, falling back to sequential extraction...")
                print(f"       FFmpeg stderr (last 500 chars): ...{result.stderr[-500:]}")
                self._export_highlights_sequential(merged_segments, fps, output_path)
                return

            print(f"\n✅ Success! Highlight video saved to:\n{output_path}")

        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")

    def _export_highlights_sequential(self, merged_segments, fps, output_path):
        temp_dir = tempfile.mkdtemp(prefix="rally_clips_")
        list_file_path = os.path.join(temp_dir, "concat_list.txt")

        try:
            clip_files = []
            n = len(merged_segments)

            print(f"\nExtracting {n} clips sequentially (fallback)...")
            for i, (start_f, end_f) in enumerate(merged_segments, 1):
                start_time = start_f / fps
                duration = (end_f - start_f) / fps

                clip_path = os.path.join(temp_dir, f"clip_{i:03d}.mp4")

                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-i', self.video_path,
                    '-t', str(duration),
                    '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    clip_path,
                ]

                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                clip_files.append(clip_path)
                print(f"  ✓ {i}/{n} (Start: {start_time:.2f}s, Duration: {duration:.2f}s)")

            with open(list_file_path, 'w', encoding='utf-8') as f:
                for clip in clip_files:
                    safe_path = clip.replace('\\', '/')
                    f.write(f"file '{safe_path}'\n")

            print("\nMerging and fixing audio sync (single re-encode pass)...")
            concat_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file_path,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-c:a', 'aac', '-b:a', '192k',
                '-async', '1',
                output_path,
            ]

            subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"\n✅ Success! Highlight video saved to:\n{output_path}")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ FFmpeg error: {e}")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")
        finally:
            print("Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ---------------------------------------------------------
    # Coordinate helpers
    # ---------------------------------------------------------
    def get_world_pos(self, px_x, px_y):
        if self.H is None:
            return 0.0, 0.0
            
        pt_px = np.array([[[px_x, px_y]]], dtype=np.float32)
        pt_world = cv2.perspectiveTransform(pt_px, self.H)
        return pt_world[0][0][0], pt_world[0][0][1]

    def _overlay_exclusion_zones(self, frame, alpha=0.25):
        if not self.exclusion_zones:
            return frame

        overlay = frame.copy()

        for (x1, y1, x2, y2) in self.exclusion_zones:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "EXCLUDE",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    # ---------------------------------------------------------
    # Frame processing entry point
    # ---------------------------------------------------------
    def process_frame(self, frame_data):
        self.frame_counter += 1
        current_time = self.frame_counter / self.fps

        out = resize_for_analysis(frame_data)

        if self.state == SystemState.WAITING:
            self._run_waiting_state(out, current_time)
        elif self.state == SystemState.ARMED:
            self._run_armed_state(out, current_time)
        elif self.state == SystemState.ACTIVE:
            self._run_active_state(out, current_time)
        
        out = self._overlay_exclusion_zones(out, alpha=0.25)

        cv2.putText(out, f"GLOBAL STATE: {self.state.name}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
        
        cv2.putText(out, f"VIDEO TIME: {current_time:.2f}s  |  FPS: {self.fps:.1f}", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return out

    # ---------------------------------------------------------
    # Player tracking
    # ---------------------------------------------------------
    def _track_players(self, frame):
        results = self.player_model(frame, verbose=False, conf=0.5, imgsz=Config.PLAYER_IMGSZ)
        
        near_serve_candidates = []
        all_player_boxes = []
        
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                if int(b.cls[0]) == 0:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    cx = 0.5 * (x1 + x2)
                    y_feet = y2
                    P = (cx, y_feet) 
                    box = (int(x1), int(y1), int(x2), int(y2))
                    
                    all_player_boxes.append(box)
                    
                    if self.H is not None:
                        world_x, world_y = self.get_world_pos(cx, y_feet)
                        
                        if world_x < -Config.COURT_X_PADDING_FT or world_x > Config.COURT_WIDTH_FT + Config.COURT_X_PADDING_FT:
                            continue 
                        
                        dist_bottom_ft = abs(world_y)
                        dist_top_ft = abs(world_y - Config.COURT_LENGTH_FT)
                        
                        if dist_bottom_ft < dist_top_ft:
                            near_serve_candidates.append((P, dist_bottom_ft, box))
                    else:
                        dist_top = point_line_distance_px(P, self.TL, self.TR)
                        dist_bottom = point_line_distance_px(P, self.BL, self.BR)
                        if dist_bottom < dist_top:
                            near_serve_candidates.append((P, dist_bottom, box))

        curr_near_pos, curr_near_box = None, None
        if near_serve_candidates:
            best_near = min(near_serve_candidates, key=lambda x: x[1])
            curr_near_pos = best_near[0]  
            curr_near_box = best_near[2]
        
        return curr_near_pos, curr_near_box, all_player_boxes

    # ---------------------------------------------------------
    # Walking gait detection
    # ---------------------------------------------------------
    def _detect_walking_gait(self, near_box):
        if near_box is None:
            self.gait_y_buffer.clear()
            return False

        feet_y = near_box[3]
        self.gait_y_buffer.append(feet_y)

        n = len(self.gait_y_buffer)
        if n < Config.GAIT_BUFFER_FRAMES * 0.6:
            return False

        ys = list(self.gait_y_buffer)

        drift = abs(ys[-1] - ys[0])
        if drift < Config.GAIT_MIN_DRIFT_PX:
            return False

        residuals = []
        for i, y in enumerate(ys):
            trend = ys[0] + (ys[-1] - ys[0]) * (i / (n - 1))
            residuals.append(y - trend)

        reversals = 0
        prev_direction = 0

        for i in range(1, len(residuals)):
            delta = residuals[i] - residuals[i - 1]
            if abs(delta) < 0.5:
                continue

            direction = 1 if delta > 0 else -1

            if prev_direction != 0 and direction != prev_direction:
                reversals += 1

            prev_direction = direction

        is_gait = Config.GAIT_MIN_REVERSALS <= reversals <= Config.GAIT_MAX_REVERSALS
        return is_gait

    # ---------------------------------------------------------
    # ACTIVE STATE: Player-Only Energy Model (V3)
    # ---------------------------------------------------------
    def _run_active_state(self, frame, now):
        """
        V3 Active State: Energy is driven SOLELY by player bounding box.
        
        No ball detection or tracking at all.
        
        INCREASING forces (strongest → weakest):
          1. Sprinting (high velocity)   → ENERGY_BOOST_SPRINT
          2. Swinging (shape change)     → ENERGY_BOOST_SWING
        
        DECREASING forces (strongest → weakest):
          1. Walking (gait oscillation)  → ENERGY_DECAY_WALKING
          2. Missing (out of frame)      → ENERGY_DECAY_MISSING
          3. Standing still              → ENERGY_DECAY_STILL
        """
        dt = 1.0 / self.fps
        
        # ==========================================
        # 1. GATHER PLAYER METRICS
        # ==========================================
        near_pos, near_box, all_player_boxes = self._track_players(frame)
        
        player_velocity = 0.0
        shape_change = 0.0

        if near_pos:
            self.near_player_positions.append(near_pos)
            if near_box:
                self.near_player_boxes.append(near_box)
                nx1, ny1, nx2, ny2 = near_box
                cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)
            
            if len(self.near_player_positions) >= 5:
                old_p = self.near_player_positions[0]
                new_p = self.near_player_positions[-1]
                dist = math.hypot(new_p[0] - old_p[0], new_p[1] - old_p[1])
                player_velocity = dist / len(self.near_player_positions)

            # Shape change: detect swings/split-steps
            if len(self.near_player_boxes) >= 5:
                old_b = self.near_player_boxes[0]
                new_b = self.near_player_boxes[-1]
                
                old_w, old_h = old_b[2] - old_b[0], old_b[3] - old_b[1]
                new_w, new_h = new_b[2] - new_b[0], new_b[3] - new_b[1]
                
                dw = abs(new_w - old_w)
                dh = abs(new_h - old_h)
                shape_change = dw + dh

        # Detect walking gait
        walking_gait = self._detect_walking_gait(near_box)

        # ==========================================
        # 2. COMPUTE ENERGY DELTA (PLAYER-ONLY)
        # ==========================================
        active_duration = now - self.active_start_time
        energy_delta = 0.0
        status_notes = []

        if active_duration < Config.SERVE_TO_ENERGY_DELAY:
            # Post-serve buffer window — hold energy at max before decay begins
            remaining = Config.SERVE_TO_ENERGY_DELAY - active_duration
            status_notes.append(f"SERVE BUFFER: {remaining:.1f}s")
        elif near_pos is None:
            # --- Player NOT detected (out of frame) ---
            energy_delta -= (Config.ENERGY_DECAY_MISSING * dt)
            status_notes.append("PLAYER: OFF SCREEN")
        elif walking_gait:
            # --- Walking gait detected (strongest drain) ---
            energy_delta -= (Config.ENERGY_DECAY_WALKING * dt)
            status_notes.append("PLAYER: WALKING (GAIT)")
        elif player_velocity > Config.PLAYER_SPRINT_VELOCITY_THRESHOLD:
            # --- Sprinting (strongest boost) ---
            energy_delta += (Config.ENERGY_BOOST_SPRINT * dt)
            status_notes.append("PLAYER: SPRINTING")
        elif shape_change > Config.SHAPE_CHANGE_THRESHOLD_PX:
            # --- Swinging / split-step (moderate boost) ---
            energy_delta += (Config.ENERGY_BOOST_SWING * dt)
            status_notes.append("PLAYER: SWING/SPLIT-STEP")
        elif player_velocity < Config.PLAYER_STILL_VELOCITY_THRESHOLD:
            # --- Standing still (moderate drain) ---
            energy_delta -= (Config.ENERGY_DECAY_STILL * dt)
            status_notes.append("PLAYER: STILL")
        else:
            # --- In-between: moving but not sprinting, no gait, no shape change ---
            # Neutral — small boost for general movement
            energy_delta += 0.1 * dt
            status_notes.append("PLAYER: MOVING")

        self.point_energy = max(0.0, min(1.0, self.point_energy + energy_delta))

        # Telemetry
        w_x_ft, w_y_ft = "", ""
        if near_pos and self.H is not None:
            w_x_ft, w_y_ft = self.get_world_pos(near_pos[0], near_pos[1])
        self._log_telemetry(
            self.frame_counter, now, "ACTIVE",
            near_pos_x=near_pos[0] if near_pos else "",
            near_pos_y=near_pos[1] if near_pos else "",
            near_box_x1=near_box[0] if near_box else "",
            near_box_y1=near_box[1] if near_box else "",
            near_box_x2=near_box[2] if near_box else "",
            near_box_y2=near_box[3] if near_box else "",
            player_world_x_ft=w_x_ft, player_world_y_ft=w_y_ft,
            player_velocity=player_velocity,
            shape_change=shape_change,
            walking_gait=walking_gait,
            point_energy=self.point_energy,
            energy_delta=energy_delta,
        )

        # ==========================================
        # 3. HUD
        # ==========================================
        cv2.putText(frame, "STATUS: ACTIVE (V3 Player-Only)", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        bar_width = 200
        current_bar_width = int(bar_width * self.point_energy)
        bar_color = (0, 255, 0) if self.point_energy > 0.4 else (0, 165, 255)
        cv2.rectangle(frame, (20, 130), (20 + bar_width, 150), (100, 100, 100), -1) 
        cv2.rectangle(frame, (20, 130), (20 + current_bar_width, 150), bar_color, -1) 
        cv2.putText(frame, f"ENERGY: {self.point_energy:.2f}", (20, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"VEL: {player_velocity:.1f}  SHAPE: {shape_change:.0f}  GAIT: {'Y' if walking_gait else 'N'}", 
                    (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        y_offset = 195
        for note in status_notes:
            cv2.putText(frame, note, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        # ==========================================
        # 4. TRANSITION LOGIC
        # ==========================================
        force_kill = active_duration > Config.ABSOLUTE_ACTIVE_TIMEOUT

        if self.point_energy <= 0.0 or force_kill:
            reason = "Energy Depleted" if not force_kill else \
                     f"Active > {Config.ABSOLUTE_ACTIVE_TIMEOUT:.0f}s timeout"
            print(f"\n[TRANSITION] ACTIVE -> END. Point dead ({reason}). Duration: {active_duration:.1f}s")
            
            if self.current_segment_start is not None:
                self.active_segments.append((self.current_segment_start, self.frame_counter))
                self.current_segment_start = None

            next_state = SystemState.WAITING
            
            # Check if player is already at baseline → skip to ARMED
            if near_pos and self.H is not None:
                player_x_ft, player_y_ft = self.get_world_pos(near_pos[0], near_pos[1])
                dist_ft = abs(player_y_ft)
                is_behind = player_y_ft < 0
                
                if is_behind and Config.READY_MIN_DIST_FT <= dist_ft <= Config.READY_MAX_DIST_FT:
                    next_state = SystemState.ARMED
                    print(f"[BYPASS] Player already at baseline. Jumping straight to ARMED.")
            
            self.state = next_state

            self.near_player_positions.clear()
            self.near_player_boxes.clear()
            self.near_ready_start_time = None
            self.gait_y_buffer.clear()
            self.active_trophy_buffer.clear()
        
        # ==========================================
        # 5. GHOST STATE: Detect next serve prep mid-rally
        # ==========================================
        # Check early (energy < 0.85) so we catch the player returning to baseline
        # before the toss has started.  No velocity gate — player may still be
        # settling into position.  Use a 4-frame rolling max on trophy score so
        # early/partial trophy frames (0.4-0.5) can accumulate to a trigger.
        if self.point_energy < 0.85 and near_pos and near_box and self.H is not None:
            player_x_ft, player_y_ft = self.get_world_pos(near_pos[0], near_pos[1])
            is_behind = player_y_ft < 0
            dist_ft = abs(player_y_ft)
            in_band = is_behind and Config.READY_MIN_DIST_FT <= dist_ft <= Config.READY_MAX_DIST_FT

            if in_band:
                nx1, ny1, nx2, ny2 = near_box
                pw = nx2 - nx1
                ph = ny2 - ny1
                frame_h, frame_w = frame.shape[:2]

                pad_x = int(pw * Config.DEFAULT_TROPHY_PAD)
                pad_y = int(ph * Config.DEFAULT_TROPHY_PAD)
                tx1 = max(0, nx1 - pad_x)
                ty1 = max(0, ny1 - pad_y)
                tx2 = min(frame_w, nx2 + pad_x)
                ty2 = min(frame_h, ny2 + pad_y)

                trophy_crop = frame[ty1:ty2, tx1:tx2]
                best_trophy_score = 0.0
                if trophy_crop.size > 0:
                    trophy_res = self.trophy_model(trophy_crop, verbose=False, imgsz=Config.TROPHY_IMGSZ)
                    if trophy_res and hasattr(trophy_res[0], 'probs') and trophy_res[0].probs is not None:
                        if Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX < len(trophy_res[0].probs.data):
                            best_trophy_score = float(trophy_res[0].probs.data[Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX])

                self.active_trophy_buffer.append(best_trophy_score)
                rolling_trophy_max = max(self.active_trophy_buffer)

                if rolling_trophy_max > 0.45:
                    print(f"\n[EMERGENCY OVERRIDE] ACTIVE -> ARMED. Next serve prep "
                          f"(Pose rolling max: {rolling_trophy_max:.2f}, energy: {self.point_energy:.2f})!")

                    # Save current segment
                    if self.current_segment_start is not None:
                        self.active_segments.append((self.current_segment_start, self.frame_counter))
                        self.current_segment_start = None

                    self.state = SystemState.ARMED

                    self.near_player_positions.clear()
                    self.near_player_boxes.clear()
                    self.point_energy = 1.0
                    self.gait_y_buffer.clear()
                    self.active_trophy_buffer.clear()

                    self.near_side.trophy_scores.clear()
                    self.near_side.toss_scores.clear()
                    self.near_side.add_trophy_score(rolling_trophy_max, now)
            else:
                # Player left the baseline band — reset the buffer so stale scores
                # don't carry over when they return
                self.active_trophy_buffer.clear()

    # ---------------------------------------------------------
    # WAITING STATE (unchanged from V2)
    # ---------------------------------------------------------
    def _run_waiting_state(self, frame, now):
        near_pos, near_box, _ = self._track_players(frame)
        
        cv2.putText(frame, "STATUS: WAITING", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        if near_box and self.H is not None:
            nx1, ny1, nx2, ny2 = near_box
            cx = (nx1 + nx2) / 2.0
            
            player_x_ft, player_y_ft = self.get_world_pos(cx, ny2)

            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (150, 150, 150), 2)
            
            is_behind = player_y_ft < 0
            dist_ft = abs(player_y_ft)

            if is_behind and Config.READY_MIN_DIST_FT <= dist_ft <= Config.READY_MAX_DIST_FT:
                if self.near_ready_start_time is None:
                    self.near_ready_start_time = now
                
                elapsed = now - self.near_ready_start_time
                cv2.putText(frame, f"IN ZONE: {elapsed:.1f}s", (nx1, ny1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if elapsed > Config.READY_WAIT_TIME_SEC:
                    print(f"[TRANSITION] WAITING -> ARMED. Player held ready position for {elapsed:.1f}s.")
                    self.state = SystemState.ARMED
                    self.near_ready_start_time = None
            else:
                self.near_ready_start_time = None
                cv2.putText(frame, "NEAR WAITING", (nx1, ny1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        else:
            self.near_ready_start_time = None

        # Telemetry
        w_x_ft, w_y_ft = "", ""
        in_band = False
        if near_box and self.H is not None:
            nx1, ny1, nx2, ny2 = near_box
            w_x_ft, w_y_ft = self.get_world_pos((nx1 + nx2) / 2.0, ny2)
            is_behind = w_y_ft < 0
            dist_ft = abs(w_y_ft)
            in_band = is_behind and Config.READY_MIN_DIST_FT <= dist_ft <= Config.READY_MAX_DIST_FT
        self._log_telemetry(
            self.frame_counter, now, "WAITING",
            near_pos_x=near_pos[0] if near_pos else "",
            near_pos_y=near_pos[1] if near_pos else "",
            near_box_x1=near_box[0] if near_box else "",
            near_box_y1=near_box[1] if near_box else "",
            near_box_x2=near_box[2] if near_box else "",
            near_box_y2=near_box[3] if near_box else "",
            player_world_x_ft=w_x_ft, player_world_y_ft=w_y_ft,
            in_band=in_band,
        )

    # ---------------------------------------------------------
    # ARMED STATE (unchanged from V2)
    # ---------------------------------------------------------
    def _run_armed_state(self, frame, now):
        near_pos, near_box, _ = self._track_players(frame)
        cv2.putText(frame, "STATUS: ARMED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        in_band = False
        if near_box and self.H is not None:
            nx1, ny1, nx2, ny2 = near_box
            cx = (nx1 + nx2) / 2.0
            
            player_x_ft, player_y_ft = self.get_world_pos(cx, ny2)

            cv2.rectangle(frame, (nx1, ny1), (nx2, ny2), (150, 150, 150), 2)
            
            is_behind = player_y_ft < 0
            dist_ft = abs(player_y_ft)

            in_band = is_behind and Config.READY_MIN_DIST_FT <= dist_ft <= Config.READY_MAX_DIST_FT

        self.armed_band_history.append((now, in_band))

        while self.armed_band_history and (now - self.armed_band_history[0][0]) > Config.ARMED_BAND_WINDOW_SEC:
            self.armed_band_history.popleft()

        if len(self.armed_band_history) > 1:
            time_out_of_band = 0.0
            for i in range(len(self.armed_band_history) - 1):
                t1, b1 = self.armed_band_history[i]
                t2, b2 = self.armed_band_history[i+1]
                
                if not b1:
                    time_out_of_band += (t2 - t1)
            
            total_history_time = self.armed_band_history[-1][0] - self.armed_band_history[0][0]

            if total_history_time > 1.0:
                out_ratio = time_out_of_band / total_history_time
                
                cv2.putText(frame, f"OUT BAND: {out_ratio:.0%}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if out_ratio > Config.ARMED_OUT_RATIO_THRESHOLD:
                    print(f"[TRANSITION] ARMED -> WAITING. Player out of band for {out_ratio:.0%} of the last {total_history_time:.1f}s.")
                    # Telemetry before early return
                    w_x_ft, w_y_ft = "", ""
                    if near_box and self.H is not None:
                        nx1, ny1, nx2, ny2 = near_box
                        w_x_ft, w_y_ft = self.get_world_pos((nx1 + nx2) / 2.0, ny2)
                    self._log_telemetry(
                        self.frame_counter, now, "ARMED",
                        near_pos_x=near_pos[0] if near_pos else "",
                        near_pos_y=near_pos[1] if near_pos else "",
                        near_box_x1=near_box[0] if near_box else "",
                        near_box_y1=near_box[1] if near_box else "",
                        near_box_x2=near_box[2] if near_box else "",
                        near_box_y2=near_box[3] if near_box else "",
                        player_world_x_ft=w_x_ft, player_world_y_ft=w_y_ft,
                        in_band=in_band,
                    )
                    self.state = SystemState.WAITING
                    self.armed_band_history.clear()
                    self.near_ready_start_time = None
                    self.toss_upward_frames = 0
                    self.toss_start_y = None
                    self.last_toss_ball = None
                    self.toss_above_box_detections = 0
                    return

        if near_box and in_band:
            nx1, ny1, nx2, ny2 = near_box
            pw = nx2 - nx1
            ph = ny2 - ny1
            frame_h, frame_w = frame.shape[:2]

            # --- A. Trophy Detection ---
            pad_x = int(pw * Config.DEFAULT_TROPHY_PAD)
            pad_y = int(ph * Config.DEFAULT_TROPHY_PAD)
            tx1 = max(0, nx1 - pad_x)
            ty1 = max(0, ny1 - pad_y)
            tx2 = min(frame_w, nx2 + pad_x)
            ty2 = min(frame_h, ny2 + pad_y)

            trophy_crop = frame[ty1:ty2, tx1:tx2]
            if trophy_crop.size > 0:
                trophy_res = self.trophy_model(trophy_crop, verbose=False, imgsz=Config.TROPHY_IMGSZ)
                best_trophy_score = 0.0
                
                if trophy_res and hasattr(trophy_res[0], 'probs') and trophy_res[0].probs is not None:
                    probs = trophy_res[0].probs
                    
                    if Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX < len(probs.data):
                        conf = float(probs.data[Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX])
                        best_trophy_score = conf

                        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 255), 2)
                        cv2.putText(frame, f"POSE CLS: {conf:.2f}", 
                                    (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                elif trophy_res and hasattr(trophy_res[0], 'boxes') and trophy_res[0].boxes is not None:
                    for b in trophy_res[0].boxes:
                        detected_class = int(b.cls[0])
                        conf = float(b.conf[0])
                        
                        crop_x1, crop_y1, crop_x2, crop_y2 = b.xyxy[0].tolist()
                        mx1, my1 = int(tx1 + crop_x1), int(ty1 + crop_y1)
                        mx2, my2 = int(tx1 + crop_x2), int(ty1 + crop_y2)
                        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 0, 255), 2)
                        cv2.putText(frame, f"CLS:{detected_class} ({conf:.2f})", 
                                    (mx1, my1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                        if detected_class == Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX:
                            if conf > best_trophy_score:
                                best_trophy_score = conf
                
                if best_trophy_score > 0:
                    self.near_side.add_trophy_score(best_trophy_score, now)
            
            # --- B. Ball Toss Detection ---
            cx = nx1 + pw / 2.0
            toss_w = pw * 2
            rx1 = max(0, int(cx - toss_w / 2.0))
            rx2 = min(frame_w, int(cx + toss_w / 2.0))
            ry1 = max(0, int(ny1 - ph))           
            ry2 = min(frame_h, int(ny1 + ph / 2)) 

            if Config.DEFAULT_DRAW_TOSS_ROI:
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "TOSS ROI", (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            toss_crop = frame[ry1:ry2, rx1:rx2]
            current_toss_score = 0.0
            
            if toss_crop.size > 0:
                ball_res = self.ball_model(toss_crop, verbose=False, conf=Config.DEFAULT_BALL_CONF_MIN, imgsz=Config.TOSS_BALL_IMGSZ)
                best_ball = None
                
                if ball_res and ball_res[0].boxes is not None:
                    for b in ball_res[0].boxes:
                        if int(b.cls[0]) == Config.DEFAULT_BALL_CLASS_INDEX:
                            crop_x1, crop_y1, crop_x2, crop_y2 = b.xyxy[0].tolist()
                            main_x1 = int(rx1 + crop_x1)
                            main_y1 = int(ry1 + crop_y1)
                            main_x2 = int(rx1 + crop_x2)
                            main_y2 = int(ry1 + crop_y2)
                            
                            cy_full = (main_y1 + main_y2) / 2.0 
                            conf = float(b.conf[0])
                            
                            if best_ball is None or conf > best_ball['conf']:
                                best_ball = {
                                    'y': cy_full, 
                                    'conf': conf, 
                                    'box': (main_x1, main_y1, main_x2, main_y2)
                                }

                if best_ball:
                    # Reset missing frames since we found the ball
                    self.toss_missing_frames = 0
                    
                    bx1, by1, bx2, by2 = best_ball['box']
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
                    cv2.putText(frame, f"{best_ball['conf']:.2f}", (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    
                    if self.last_toss_ball:
                        dy = best_ball['y'] - self.last_toss_ball['y']
                        dt = now - self.last_toss_ball['time']

                        if abs(dy) < 1.5:
                            # Ball is static — ignore this detection entirely
                            pass

                        # 1. UPWARD PHASE: ball must genuinely move upward (negative dy)
                        elif dy < -1.5 and dt > 0:
                            self.toss_upward_frames += 1
                            if self.toss_start_y is None:
                                self.toss_start_y = self.last_toss_ball['y']

                            # Track detections above the player box top while rising
                            if cy_full < ny1:
                                self.toss_above_box_detections += 1

                            rise_px = self.toss_start_y - best_ball['y']
                            player_height_px = ph
                            rise_pct = rise_px / player_height_px if player_height_px > 0 else 0.0

                            cv2.putText(frame, f"RISE: {rise_pct:.1f} UP:{self.toss_upward_frames} ABV:{self.toss_above_box_detections}",
                                        (bx1, by2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                            # Require sustained upward motion (>=4 frames) AND ball seen
                            # above the player box at least twice before banking a score
                            if (rise_pct >= 0.2
                                    and self.toss_upward_frames >= 4
                                    and self.toss_above_box_detections >= 2):
                                current_toss_score = min(1.0, self.toss_upward_frames * 0.2)
                                self.near_side.add_toss_score(current_toss_score, now)

                        # 2. PEAK/DESCENT PHASE: Do NOT reset.
                        # If the ball is dropping significantly, reset the streak.
                        elif dy > (ph * 0.5):
                            self.toss_upward_frames = 0
                            self.toss_start_y = None
                            self.toss_above_box_detections = 0

                    self.last_toss_ball = {'y': best_ball['y'], 'time': now}                
                
                else:
                    # 3. OUT-OF-FRAME / PERSISTENCE LOGIC
                    self.toss_missing_frames = getattr(self, 'toss_missing_frames', 0) + 1
                    
                    # If ball vanishes, check if it "Topped Out"
                    if self.last_toss_ball and self.toss_start_y is not None:
                        last_y = self.last_toss_ball['y']
                        
                        # If it vanished near the top (top 30% of ROI) after a valid toss
                        if (last_y < (ry1 + (ph * 0.5))
                                and self.toss_upward_frames >= 4
                                and self.toss_above_box_detections >= 2):
                            # Bank a high score because it successfully cleared the frame
                            forced_score = min(1.0, (self.toss_upward_frames * 0.2) + 0.2)
                            self.near_side.add_toss_score(forced_score, now)
                            print(f"[DEBUG] Toss Topped-Out at y={last_y:.1f}. Forced Score: {forced_score:.2f}")

                    # Only fully wipe the memory after 10 frames (~0.16s) of the ball being gone
                    # This gives the Trophy Pose logic time to combine with the "Latched" toss score
                    if self.toss_missing_frames > 10:
                        self.last_toss_ball = None
                        self.toss_upward_frames = 0
                        self.toss_start_y = None
                        self.toss_above_box_detections = 0
            if current_toss_score > 0:
                self.near_side.add_toss_score(current_toss_score, now)

            # --- C. Tiered Serve Score ---
            # Path 1: Strong toss fires solo (trophy is unreliable)
            # Path 2: Weaker toss + trophy confirmation
            max_trophy = max([d.score for d in self.near_side.trophy_scores] + [0.0])
            max_toss = max([d.score for d in self.near_side.toss_scores] + [0.0])
            
            serve_path = "--"
            """
            if max_toss >= Config.TOSS_SOLO_THRESHOLD:
                # Path 1: toss alone is convincing
                serve_score = max_toss
                serve_path = "TOSS SOLO"
            elif max_toss >= Config.TOSS_MIN_THRESHOLD and max_trophy >= Config.TROPHY_MIN_THRESHOLD:
                # Path 2: trophy rescues a weaker toss
                serve_score = 0.5 * max_toss + 0.5 * max_trophy
                serve_path = "TOSS+TROPHY"
            else:
                serve_score = 0.0
            """

            serve_score = 0.7 * max_toss + 0.3 * max_trophy

            cv2.putText(frame, f"SERVE SCORE: {serve_score:.2f} [{serve_path}]", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAX TROPHY:  {max_trophy:.2f}", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
            cv2.putText(frame, f"MAX TOSS:    {max_toss:.2f}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Telemetry (in-band with scores computed)
            self._log_telemetry(
                self.frame_counter, now, "ARMED",
                near_pos_x=near_pos[0] if near_pos else "",
                near_pos_y=near_pos[1] if near_pos else "",
                near_box_x1=near_box[0] if near_box else "",
                near_box_y1=near_box[1] if near_box else "",
                near_box_x2=near_box[2] if near_box else "",
                near_box_y2=near_box[3] if near_box else "",
                player_world_x_ft=player_x_ft if self.H is not None and near_box else "",
                player_world_y_ft=player_y_ft if self.H is not None and near_box else "",
                in_band=in_band,
                trophy_conf=best_trophy_score,
                toss_score=current_toss_score,
                max_trophy_window=max_trophy,
                max_toss_window=max_toss,
                serve_score=serve_score,
            )

            if serve_score >= Config.TRANSITION_SCORE_THRESHOLD:
                print(f"[TRANSITION] ARMED -> ACTIVE. Serve detected! Path={serve_path} Trophy={max_trophy:.2f} Toss={max_toss:.2f} Score={serve_score:.2f}")
                self.state = SystemState.ACTIVE

                buffer_frames = int(self.fps * 1.0)
                self.current_segment_start = max(0, self.frame_counter - buffer_frames)

                self.near_side.trophy_scores.clear()
                self.near_side.toss_scores.clear()
                self.last_toss_ball = None
                self.toss_upward_frames = 0
                self.toss_start_y = None
                self.toss_above_box_detections = 0

                self.near_player_positions.clear()
                self.near_player_boxes.clear()
                self.active_start_time = now
                self.point_energy = 1.0
                self.gait_y_buffer.clear()
            return

        # Telemetry (not in-band or no near_box)
        w_x_ft, w_y_ft = "", ""
        if near_box and self.H is not None:
            nx1, ny1, nx2, ny2 = near_box
            w_x_ft, w_y_ft = self.get_world_pos((nx1 + nx2) / 2.0, ny2)
        self._log_telemetry(
            self.frame_counter, now, "ARMED",
            near_pos_x=near_pos[0] if near_pos else "",
            near_pos_y=near_pos[1] if near_pos else "",
            near_box_x1=near_box[0] if near_box else "",
            near_box_y1=near_box[1] if near_box else "",
            near_box_x2=near_box[2] if near_box else "",
            near_box_y2=near_box[3] if near_box else "",
            player_world_x_ft=w_x_ft, player_world_y_ft=w_y_ft,
            in_band=in_band,
        )


# ==========================================
# 4. RUN
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Anya System V3 — Player-only energy model for rally detection."
    )
    parser.add_argument(
        "video",
        help="Path to the input video file",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output highlight video path (default: <input>_highlights_v3.mp4)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the live preview window (faster processing).",
    )
    args = parser.parse_args()

    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_highlights_v3.mp4"

    system = AnyaSystem(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video for processing: {video_path}")
        return

    print(f"\n[INFO] Processing video at {system.fps:.1f} FPS ({system.frame_width}x{system.frame_height})...")
    print(f"[INFO] Energy Model: Player-Only (V3 — no ball tracking in ACTIVE)")
    print(f"[INFO] Press 'q' to stop early.\n")

    if not args.headless:
        cv2.namedWindow("Anya System V3", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            out = system.process_frame(frame)

            if not args.headless:
                cv2.imshow("Anya System V3", out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] Stopped early by user.")
                    break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"\n[INFO] Processing complete. Processed {system.frame_counter} frames.")

    # Save telemetry for offline evaluation
    telemetry_path = os.path.join(os.path.dirname(os.path.abspath(video_path)), "telemetry.csv")
    system.save_telemetry(telemetry_path)

    system.export_highlights(output_path)


if __name__ == "__main__":
    main()
