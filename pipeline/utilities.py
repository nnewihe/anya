import time
import random
from enum import Enum
from collections import deque
from dataclasses import dataclass, field
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Optional, Tuple
import math
from sklearn.cluster import DBSCAN
import os
import json
import csv
import re
import argparse
import tempfile
import shutil
from pathlib import Path

try:                                        # package import (python -m pipeline.x)
    from .subproc import run as _run
except ImportError:                         # script import (python pipeline/x.py)
    from subproc import run as _run


    
# =============================================================
# Config
# =============================================================
class Config:
    ANALYSIS_WIDTH = 960
    ANALYSIS_HEIGHT = 540
    PLAYER_IMGSZ = 960
    BALL_IMGSZ = 1920         # exclusion-zone scans (stationary objects; keep high)
    ACTIVE_BALL_IMGSZ = 1920  # per-frame ball detection in ACTIVE state (input is 960px wide)
    TOSS_BALL_IMGSZ = 320
    FAR_PLAYER_X_PAD_FT  = 3.0      # homography-tolerance padding beyond the singles sidelines
                                     # for far-player feet (mirrors NEAR_PLAYER_X_PAD_FT)
    FAR_SERVE_LSTM_PAD   = 0.30     # padding fraction (of box w/h) around the far player's box
                                     # before cropping for the serve_lstm.pt classifier
    ACTIVE_BALL_CONF = 0.10   # confidence threshold for whole-court ball detection (ACTIVE)
    TOSS_BALL_CONF   = 0.10   # confidence threshold for toss ROI ball detection (ARMED)
    TROPHY_IMGSZ = 320
    COURT_WIDTH_FT = 27.0
    COURT_LENGTH_FT = 78.0
    COURT_X_PADDING_FT = 15.0
    NEAR_PLAYER_X_PAD_FT = 3.0   # homography-tolerance padding beyond near-baseline width
    DEFAULT_BALL_CLASS_INDEX = 0
    DEFAULT_PLAYER_CLASS_INDEX = 0
    DEFAULT_NEAR_TROPHY_MODEL_PATH  = str(Path(__file__).parent / "models" / "trophy_best.pt")
    DEFAULT_NEAR_TROPHY_CLASS_INDEX = 1
    DEFAULT_TROPHY_PAD              = 0.30
    TELEMETRY_BUFFER_SECONDS = 5.0
    HORIZON_Y_PX             = 200

# =============================================================
# BoxSmoother  
# =============================================================
# Add this to src/ai/utilities.py
@dataclass
class Point3D:
    x: float
    y: float
    z: float = 0.0

@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def w(self): return self.x2 - self.x1
    @property
    def h(self): return self.y2 - self.y1

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

@dataclass
class BoxSmoother:
    """
    Exponentially-weighted moving average for a bounding box (cx, cy, w, h).

    Uses separate alphas for position and size, and suppresses size updates
    when the player appears stationary.

    Parameters
    ----------
    alpha_pos   : EWA weight for centre-point updates  (higher = more responsive)
    alpha_size  : EWA weight for width/height updates  (lower  = more stable)
    still_thresh: pixel/frame velocity below which size smoothing is further
                  suppressed (multiplied by 0.3).
    """
    alpha_pos:   float = 0.35
    alpha_size:  float = 0.12
    still_thresh: float = 4.0

    _cx: Optional[float] = field(default=None, init=False, repr=False)
    _cy: Optional[float] = field(default=None, init=False, repr=False)
    _w:  Optional[float] = field(default=None, init=False, repr=False)
    _h:  Optional[float] = field(default=None, init=False, repr=False)

    def update(self, cx: float, cy: float, w: float, h: float
               ) -> Tuple[float, float, float, float]:
        """Feed a raw (cx, cy, w, h) observation and get the smoothed version."""
        if self._cx is None:
            self._cx, self._cy, self._w, self._h = cx, cy, w, h
            return cx, cy, w, h

        vel = math.hypot(cx - self._cx, cy - self._cy)

        self._cx = (1 - self.alpha_pos) * self._cx + self.alpha_pos * cx
        self._cy = (1 - self.alpha_pos) * self._cy + self.alpha_pos * cy

        eff_alpha = self.alpha_size * 0.3 if vel < self.still_thresh else self.alpha_size
        self._w = (1 - eff_alpha) * self._w + eff_alpha * w
        self._h = (1 - eff_alpha) * self._h + eff_alpha * h

        return self._cx, self._cy, self._w, self._h

    def reset(self):
        """Clear the smoother's state (e.g. when tracking is interrupted)."""
        self._cx = self._cy = self._w = self._h = None

    def smooth_box_xyxy(self, x1: int, y1: int, x2: int, y2: int
                        ) -> Tuple[int, int, int, int]:
        """
        Convenience wrapper: accept raw (x1,y1,x2,y2), smooth internally,
        and return a smoothed (x1,y1,x2,y2) ready to draw on frame.
        """
        raw_cx = (x1 + x2) / 2.0
        raw_cy = (y1 + y2) / 2.0
        raw_w  = float(x2 - x1)
        raw_h  = float(y2 - y1)

        scx, scy, sw, sh = self.update(raw_cx, raw_cy, raw_w, raw_h)

        sx1 = int(scx - sw / 2.0)
        sy1 = int(scy - sh / 2.0)
        sx2 = int(scx + sw / 2.0)
        sy2 = int(scy + sh / 2.0)
        return sx1, sy1, sx2, sy2


# =============================================================
# Exclusion Zone Helpers
# =============================================================

def create_auto_exclusion_zones(
    video_path: str,
    ball_model,
    num_frames: int = 20,
    conf: float = 0.05,
    eps: int = 5,
    min_samples: int = 15,
    padding: int = 5,
    ball_class_index: int = 0,
    analysis_size: tuple = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Scan random frames across the full video to find static clusters of objects
    that look like balls (e.g. ball-baskets). Uses DBSCAN clustering of detection
    centers. Returns exclusion zone rectangles with padding.
    """
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

    zones = []
    for k in set(labels):
        if k == -1:
            continue
        pts = X[labels == k]
        if len(pts) > 0:
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            zones.append((
                int(x_min - padding), int(y_min - padding),
                int(x_max + padding), int(y_max + padding),
            ))

    return zones


def get_exclusion_zones_from_frames(
    frames: List[np.ndarray],
    ball_model,
    sample_size: int = 5,
    conf: float = 0.10,
    eps: int = 5,
    min_samples: int = 15,
    padding: int = 5,
    ball_class_index: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """
    Given a list of frames, detect balls and return exclusion zones using the
    same DBSCAN clustering logic as create_auto_exclusion_zones().
    Randomly sample frames, cluster detection centers, and return bounding boxes
    for valid clusters.
    """
    if not frames:
        return []

    sample = random.sample(frames, min(len(frames), sample_size))

    # Collect detection centers (same logic as static exclusion zones)
    all_detections = []
    for frm in sample:
        res = ball_model(frm, verbose=False, conf=conf, imgsz=Config.BALL_IMGSZ)
        if res and res[0].boxes:
            for b in res[0].boxes:
                if int(b.cls[0]) != ball_class_index:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                all_detections.append((cx, cy))

    if len(all_detections) < min_samples:
        return []

    # Cluster detection centers using DBSCAN (same params as static zones)
    X = np.array(all_detections)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_

    zones = []
    for k in set(labels):
        if k == -1:
            continue
        pts = X[labels == k]
        if len(pts) > 0:
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            zones.append((
                int(x_min - padding), int(y_min - padding),
                int(x_max + padding), int(y_max + padding),
            ))

    return zones


def _is_in_exclusion_zone(x, y, exclusion_zones):
    for (x1, y1, x2, y2) in exclusion_zones:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def _exclusion_zone_cache_path(video_path: str) -> str:
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_exclusion_cache.json")


def load_cached_exclusion_zones(video_path: str):
    """Return cached static exclusion zones for video_path, or None if not cached."""
    path = _exclusion_zone_cache_path(video_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        zones = [tuple(z) for z in data]
        print(f"[INFO] Loaded {len(zones)} static exclusion zone(s) from cache")
        return zones
    except Exception as e:
        print(f"[WARN] Exclusion zone cache unreadable ({e}), recomputing")
        return None


def save_cached_exclusion_zones(video_path: str, zones) -> None:
    """Persist static exclusion zones alongside the video for future runs."""
    try:
        with open(_exclusion_zone_cache_path(video_path), "w") as f:
            json.dump([list(z) for z in zones], f)
    except Exception as e:
        print(f"[WARN] Could not save exclusion zone cache: {e}")


def _court_cache_path(video_path: str) -> str:
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_court_cache.json")


def _far_player_roi_cache_path(video_path: str) -> str:
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_far_player_roi.json")


# Homography (_compute_homography in anya_base.py / match_telemetry.py) unpacks
# court_vertices positionally as `BL, BR, TR, TL = court_vertices` — the click
# order below is not cosmetic, it is the contract those callers rely on.
COURT_CORNER_ORDER = ["bottom-left", "bottom-right", "top-right", "top-left"]
COURT_CORNER_TAGS  = ["BL", "BR", "TR", "TL"]


def init_court(video_path: str, target_idx: int = 300, analysis_size: tuple = None):
    """One-time interactive court corner calibration, cached to disk.

    The user clicks the four court corners in a fixed order — bottom-left,
    bottom-right, top-right, top-left — matching the vertex order the
    homography (image -> world) computation expects. Once clicked, the
    points are cached alongside the video and reused on every future run.
    """
    cache_path = _court_cache_path(video_path)

    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            cached_size = tuple(cached.get("analysis_size", [None, None]))
            if cached_size == (analysis_size if analysis_size else (None, None)):
                pts   = [tuple(p) for p in cached["points"]]
                shape = tuple(cached["frame_shape"])
                print(f"[COURT] Loaded cached corners from: {os.path.basename(cache_path)}")
                return pts, shape
            else:
                print("[COURT] Analysis size changed — re-selecting corners.")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[COURT] Cache corrupt ({e}), re-selecting.")

    num_points = len(COURT_CORNER_ORDER)
    win = "Court calibration click corners IN ORDER: bottom-left, bottom-right, top-right, top-left  (r=reset, q=quit)"

    base = get_reference_frame(video_path, target_idx=target_idx)
    if analysis_size is not None:
        base = cv2.resize(base, analysis_size, interpolation=cv2.INTER_AREA)
    img = base.copy()

    print("[COURT] One-time court calibration click these 4 points in order:")
    for i, label in enumerate(COURT_CORNER_ORDER, start=1):
        print(f"[COURT]   {i}. {label}")

    state = {"img": img, "clicked_pts": [], "done": False, "win": win,
             "num_points": num_points, "labels": COURT_CORNER_TAGS}

    # WINDOW_AUTOSIZE, not WINDOW_NORMAL: a resizable window hits a real
    # OpenCV bug (opencv/opencv#23870) where the mouse callback only reports
    # correct coordinates in the top-left quadrant once the window has been
    # resized — which Windows triggers far more readily than macOS (DPI
    # scaling, auto-fit-to-screen), making clicks silently land nowhere.
    # AUTOSIZE locks the window to the image's native size (960x540 here,
    # small enough to always fit) so it never resizes and never hits it.
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, state["img"])
    cv2.setMouseCallback(win, select_points, state)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if state["done"]:
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            pts   = [(float(x), float(y)) for x, y in state["clicked_pts"]]
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
                print(f"[COURT] Saved corners to: {os.path.basename(cache_path)}")
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


def init_far_player_roi(
    video_path: str,
    target_idx: int = 300,
    analysis_size: tuple = None,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Interactive ROI selection for the far-player YOLO search area.

    The user clicks two diagonal corners of the desired rectangle.  The result
    is normalised to (top-left, bottom-right) and cached next to the video as
    ``{name}_far_player_roi.json``.

    Returns
    -------
    ((x1, y1), (x2, y2))  — axis-aligned ROI in analysis-frame pixel coordinates.

    Keys
    ----
    Left-click : place a corner (auto-confirms after second click)
    r          : reset and re-click
    q / Esc    : abort (raises RuntimeError)
    """
    cache_path = _far_player_roi_cache_path(video_path)
    target_size = tuple(list(analysis_size) if analysis_size else [None, None])

    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            if tuple(cached.get("analysis_size", [None, None])) == target_size:
                pt1 = tuple(cached["pt1"])
                pt2 = tuple(cached["pt2"])
                print(f"[FAR ROI] Loaded cached ROI from: {os.path.basename(cache_path)}")
                return pt1, pt2
            else:
                print("[FAR ROI] Analysis size changed — re-selecting ROI.")
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            print(f"[FAR ROI] Cache corrupt or wrong format ({e}), re-selecting.")

    base = get_reference_frame(video_path, target_idx=target_idx)
    if analysis_size is not None:
        base = cv2.resize(base, analysis_size, interpolation=cv2.INTER_AREA)

    win   = "Click 2 diagonal corners of far-player search area  [r=reset  q=quit]"
    state = {"base": base.copy(), "img": base.copy(), "pts": [], "done": False, "win": win}

    def _cb(event, x, y, flags, param):
        s = param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        s["pts"].append((x, y))
        if len(s["pts"]) == 1:
            cv2.circle(s["img"], (x, y), 6, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        elif len(s["pts"]) == 2:
            s["img"] = s["base"].copy()
            p1 = (min(s["pts"][0][0], x), min(s["pts"][0][1], y))
            p2 = (max(s["pts"][0][0], x), max(s["pts"][0][1], y))
            cv2.rectangle(s["img"], p1, p2, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(s["img"], s["pts"][0], 6, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(s["img"], (x, y),       6, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            s["done"] = True
        cv2.imshow(s["win"], s["img"])

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)  # see init_court for why not WINDOW_NORMAL
    cv2.imshow(win, state["img"])
    cv2.setMouseCallback(win, _cb, state)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if state["done"]:
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            pts = state["pts"]
            x1  = min(pts[0][0], pts[1][0])
            y1  = min(pts[0][1], pts[1][1])
            x2  = max(pts[0][0], pts[1][0])
            y2  = max(pts[0][1], pts[1][1])
            pt1, pt2 = (x1, y1), (x2, y2)
            try:
                cache_data = {
                    "pt1":           list(pt1),
                    "pt2":           list(pt2),
                    "analysis_size": list(analysis_size) if analysis_size else [None, None],
                    "video":         os.path.basename(video_path),
                }
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f, indent=2)
                print(f"[FAR ROI] Saved ROI to: {os.path.basename(cache_path)}")
            except Exception as e:
                print(f"[FAR ROI] WARN: Could not save cache: {e}")
            return pt1, pt2

        if key == ord("r"):
            state["pts"].clear()
            state["done"] = False
            state["img"] = state["base"].copy()
            cv2.imshow(win, state["img"])

        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            raise RuntimeError("Far-player ROI selection aborted by user.")


def point_line_distance_px(P, A, B):
    Px, Py = P
    Ax, Ay = A
    Bx, By = B
    ABx, ABy = Bx - Ax, By - Ay
    APx, APy = Px - Ax, Py - Ay
    cross = abs(ABx * APy - ABy * APx)
    denom = math.hypot(ABx, ABy)
    return 0.0 if denom == 0 else cross / denom


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
    ref_idx = min(target_idx, total_frames // 2)
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
    idx = len(state["clicked_pts"])
    state["clicked_pts"].append((x, y))
    cv2.circle(state["img"], (x, y), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    labels = state.get("labels")
    if labels and idx < len(labels):
        cv2.putText(state["img"], labels[idx], (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        print(f"[COURT] {idx + 1}/{state['num_points']} placed: {labels[idx]}")
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


def probe_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or fps > 300:
        print(f"[WARN] Video reported FPS={fps}, falling back to 30.0")
        fps = 30.0

    duration_sec = frame_count / fps if fps > 0 else 0.0
    info = {
        "fps": fps, "frame_count": frame_count,
        "width": width, "height": height, "duration_sec": duration_sec,
    }

    print(f"\n{'='*50}")
    print(f"  VIDEO PROBE: {os.path.basename(video_path)}")
    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {frame_count}")
    print(f"  Duration   : {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
    print(f"{'='*50}\n")
    return info


class TruncatedDecodeError(RuntimeError):
    """A decode loop ended before the frame it was asked to reach.

    Raised rather than returning short because every extraction pass in this
    package is written as ``while True: if not cap.grab(): break``, and a
    mid-file decode failure is byte-for-byte indistinguishable from a clean
    EOF.  Without this check the pass finishes "successfully" on whatever it
    managed to read, every stage downstream runs on that fragment, and the
    run produces a plausible-looking empty result.
    """


# A container's frame count comes from stream metadata and can disagree with
# the decoder by a frame or two (and by more on VFR sources), so an exact
# match is too strict to enforce.  These bound the disagreement we accept:
# generous enough that ordinary drift passes, tight enough that real
# truncation cannot hide — the failure this exists to catch decoded 53 of
# 31,860 frames, short by 99.8%.
DECODE_SLACK_FRAC = 0.005
DECODE_SLACK_MIN  = 2

# Escape hatch for a tester sitting on a file we mis-measure. Deliberately an
# env var rather than a config knob: it has to be *chosen*, and the warning it
# prints keeps the degradation on the record instead of silent again.
_ALLOW_TRUNCATED = "ANYA_ALLOW_TRUNCATED_DECODE"


def assert_decode_complete(label: str, video_path: str, reached_idx: int,
                           expected_idx: int, fps: float = 0.0) -> None:
    """Raise unless a decode loop reached (near enough) `expected_idx`.

    Call it after the loop, with the last index successfully grabbed and the
    last index the pass needed — `total_frames - 1` for a whole-video walk, or
    the last frame of the last window for a window-bounded one.
    """
    if expected_idx < 0:
        return
    short = expected_idx - reached_idx
    if short <= max(DECODE_SLACK_MIN, int(expected_idx * DECODE_SLACK_FRAC)):
        return

    got, want = reached_idx + 1, expected_idx + 1
    detail = (f"{got} of {want} frames" if not fps else
              f"{got} of {want} frames ({got / fps:.1f}s of {want / fps:.1f}s)")
    msg = (
        f"[{label}] decoding {os.path.basename(video_path)} stopped early: "
        f"{detail}. The file's metadata says it is longer than what the "
        f"decoder would return, so the rest of the video was never seen and "
        f"any result from this run covers only the opening fragment. This is "
        f"usually a missing or wrong video backend (on Windows, OpenCV "
        f"falling back to MSMF when its FFmpeg DLL is absent) or a codec the "
        f"decoder cannot read past the first GOP. "
        f"Set {_ALLOW_TRUNCATED}=1 to continue on the partial decode anyway."
    )
    if os.environ.get(_ALLOW_TRUNCATED) == "1":
        print(f"[WARN] {msg}")
        return
    raise TruncatedDecodeError(msg)


def resize_for_analysis(frame):
    return cv2.resize(frame, (Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT),
                      interpolation=cv2.INTER_AREA)


def create_highlights_ffmpeg_multisource(
    segments: List[Tuple[str, float, float]],
    output_path: str,
) -> None:
    """
    Cut segments from multiple source videos and concatenate into output_path.

    Parameters
    ----------
    segments    : list of (source_video_path, start_sec, end_sec)
    output_path : destination path for the highlight reel
    """
    if not segments:
        print("[HIGHLIGHT] No segments to export.")
        return

    valid = [(src, s, e) for src, s, e in segments if e > s]
    if not valid:
        print("[HIGHLIGHT] All segments are zero-length — nothing to export.")
        return

    print(f"\n[HIGHLIGHT] Creating multi-source highlight reel: {len(valid)} segment(s) → {output_path}")

    tmpdir = tempfile.mkdtemp(prefix="anya_highlights_")
    try:
        seg_files = []
        for i, (src, start, end) in enumerate(valid):
            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-i", src,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                "-vsync", "cfr",
                seg_path,
            ]
            result = _run(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"[HIGHLIGHT] Warning: segment {i} ({start:.2f}s–{end:.2f}s) from "
                      f"{os.path.basename(src)} failed.")
                print(result.stderr.decode(errors="replace"))
                continue
            seg_files.append(seg_path)
            print(f"[HIGHLIGHT]   Segment {i+1}/{len(valid)}: {os.path.basename(src)} "
                  f"{start:.2f}s – {end:.2f}s")

        if not seg_files:
            print("[HIGHLIGHT] No segments were successfully extracted.")
            return

        concat_list = os.path.join(tmpdir, "concat.txt")
        # Explicit utf-8: Python's default text encoding on Windows is the
        # legacy ANSI code page, but the concat demuxer reads this file as
        # utf-8. The temp path runs through the user's profile directory, so
        # any account whose name isn't representable in the local code page —
        # Cyrillic or CJK on a cp1252 box — either raises UnicodeEncodeError
        # here or writes bytes ffmpeg then fails to resolve. Every segment
        # encodes fine and only the final concat dies, which makes it a
        # miserable thing to debug from a bug report.
        #
        # Forward slashes are belt-and-braces. ffmpeg accepts them on Windows,
        # and normalizing keeps the entry from depending on the demuxer's
        # quoting rules (backslash is literal inside single quotes today, but
        # nothing in the format guarantees that).
        with open(concat_list, "w", encoding="utf-8") as f:
            for sf in seg_files:
                f.write("file '{}'\n".format(sf.replace("\\", "/")))

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_path,
        ]
        result = _run(cmd)
        if result.returncode == 0:
            print(f"[HIGHLIGHT] Saved: {output_path}")
        else:
            print("[HIGHLIGHT] Concatenation step failed — check FFMPEG output above.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def create_highlights_ffmpeg(
    video_path: str,
    segments: List[Tuple[float, float]],
    output_path: str,
    pre_roll: float = 0.0,
    merge_gap_sec: float = 3.0,
) -> None:
    """
    Cut active segments from video_path using FFMPEG (preserves audio) and
    concatenate into output_path.

    Parameters
    ----------
    video_path    : path to the original source video
    segments      : list of (start_sec, end_sec) in source video time
    output_path   : destination path for the highlight reel
    pre_roll      : seconds to extend each segment's start backward (clamped to 0)
    merge_gap_sec : merge consecutive segments whose gap is <= this value (default 3.0s)
    """
    if not segments:
        print("[HIGHLIGHT] No active segments to export.")
        return

    # Apply pre_roll first, then merge — so extended starts can close gaps too.
    extended = [(max(0.0, s - pre_roll), e) for s, e in segments if e > s]
    if not extended:
        print("[HIGHLIGHT] All segments are zero-length — nothing to export.")
        return

    # Merge segments whose gap is within merge_gap_sec.
    extended.sort()
    valid = [extended[0]]
    for start, end in extended[1:]:
        if start - valid[-1][1] <= merge_gap_sec:
            valid[-1] = (valid[-1][0], max(valid[-1][1], end))
        else:
            valid.append((start, end))

    n_raw = len(extended)
    n_merged = len(valid)
    merged_note = f" ({n_raw - n_merged} merged)" if n_merged < n_raw else ""
    print(f"\n[HIGHLIGHT] Creating highlight reel: {n_merged} segment(s){merged_note} → {output_path}")

    tmpdir = tempfile.mkdtemp(prefix="anya_highlights_")
    try:
        seg_files = []
        for i, (start, end) in enumerate(valid):
            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-i", video_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                "-vsync", "cfr",
                seg_path,
            ]
            result = _run(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"[HIGHLIGHT] Warning: segment {i} ({start:.2f}s–{end:.2f}s) failed.")
                print(result.stderr.decode(errors="replace"))
                continue
            seg_files.append(seg_path)
            print(f"[HIGHLIGHT]   Segment {i+1}/{len(valid)}: {start:.2f}s – {end:.2f}s")

        if not seg_files:
            print("[HIGHLIGHT] No segments were successfully extracted.")
            return

        concat_list = os.path.join(tmpdir, "concat.txt")
        # Explicit utf-8: Python's default text encoding on Windows is the
        # legacy ANSI code page, but the concat demuxer reads this file as
        # utf-8. The temp path runs through the user's profile directory, so
        # any account whose name isn't representable in the local code page —
        # Cyrillic or CJK on a cp1252 box — either raises UnicodeEncodeError
        # here or writes bytes ffmpeg then fails to resolve. Every segment
        # encodes fine and only the final concat dies, which makes it a
        # miserable thing to debug from a bug report.
        #
        # Forward slashes are belt-and-braces. ffmpeg accepts them on Windows,
        # and normalizing keeps the entry from depending on the demuxer's
        # quoting rules (backslash is literal inside single quotes today, but
        # nothing in the format guarantees that).
        with open(concat_list, "w", encoding="utf-8") as f:
            for sf in seg_files:
                f.write("file '{}'\n".format(sf.replace("\\", "/")))

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_path,
        ]
        result = _run(cmd)
        if result.returncode == 0:
            print(f"[HIGHLIGHT] Saved: {output_path}")
        else:
            print("[HIGHLIGHT] Concatenation step failed — check FFMPEG output above.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
