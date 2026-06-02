"""
extract_telemetry.py
Extract near-player bounding-box telemetry from a tennis match video.

Workflow
--------
1. On first run, launch an interactive window so the user can click the 4
   singles-court corners on a reference frame.  Pixel coordinates are saved
   to a JSON cache so subsequent runs skip this step.
2. Compute a homography that maps image pixels -> real-world court feet.
3. For every frame, run YOLO person detection, project each detection's
   foot-point into court coordinates, discard out-of-bounds detections, and
   select the player closest to the near baseline (y ≈ 0 ft).
4. Smooth the selected bounding box to eliminate jitter during idle periods
   (critical: delta_area spikes break the downstream state machine).
5. Write frame_id, x, y, w, h to a CSV file.

Usage
-----
  python extract_telemetry.py video.mp4
  python extract_telemetry.py video.mp4 --output telemetry.csv --court-json court.json
  python extract_telemetry.py video.mp4 --fps 60 --conf 0.35
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ================================================================
# Configuration
# ================================================================

CONFIG = {
    # YOLO
    "YOLO_MODEL": "yolov8n.pt",
    "YOLO_CONF": 0.40,

    # Court geometry (real-world feet)
    "COURT_WIDTH_FT": 27.0,      # singles width
    "COURT_LENGTH_FT": 78.0,     # baseline to baseline
    "LATERAL_PADDING_FT": 15.0,  # allow detections this far outside singles sidelines

    # Bounding-box smoothing (EWA)
    # Separate alphas for position (x,y) and size (w,h).
    # Lower value → heavier smoothing → more stable (less jitter).
    # Critical: aggressive size smoothing prevents false delta_area spikes
    # in the downstream RAI calculation when the player is standing still.
    "SMOOTH_ALPHA_POS": 0.35,    # positional responsiveness
    "SMOOTH_ALPHA_SIZE": 0.12,   # size stability (lower = very stable)

    # When raw velocity (px/frame) drops below this threshold the player is
    # considered stationary; size smoothing is further suppressed.
    "VELOCITY_STILL_THRESH": 4.0,

    # Video
    "FPS": 30,
}

# ================================================================
# Court setup — interactive corner picking
# ================================================================

CORNER_LABELS = [
    "near-left  (bottom-left of court)",
    "near-right (bottom-right of court)",
    "far-right  (top-right of court)",
    "far-left   (top-left of court)",
]

# Corresponding real-world court coordinates (x_ft, y_ft).
# Near baseline: y = 0; far baseline: y = COURT_LENGTH_FT.
# Singles sidelines: x = ±COURT_WIDTH_FT/2.
def _court_world_corners(cfg: dict) -> np.ndarray:
    hw = cfg["COURT_WIDTH_FT"] / 2.0
    L  = cfg["COURT_LENGTH_FT"]
    return np.array([
        [-hw,  0.0],   # near-left
        [ hw,  0.0],   # near-right
        [ hw,   L ],   # far-right
        [-hw,   L ],   # far-left
    ], dtype=np.float32)


def init_court(frame: np.ndarray, json_path: str) -> np.ndarray:
    """
    If json_path exists, load and return the 4 pixel corner coordinates.
    Otherwise open an interactive window for the user to click them.

    Returns
    -------
    corners : (4, 2) float32 array in the order
              [near-left, near-right, far-right, far-left]
    """
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r") as f:
            data = json.load(f)
        corners = np.array(data, dtype=np.float32).reshape(4, 2)
        print(f"[court] Loaded {len(corners)} corners from {json_path}")
        return corners

    # Interactive picking
    corners: List[Tuple[int, int]] = []
    vis = frame.copy()
    label_idx = [0]  # mutable for closure

    WINDOW = "Court Setup — click corners in order shown in title"

    def _on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(corners) >= 4:
            return
        corners.append((x, y))
        cv2.circle(vis, (x, y), 7, (0, 255, 0), -1)
        cv2.putText(vis, str(len(corners)), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, vis)

        if len(corners) < 4:
            label_idx[0] += 1
            cv2.setWindowTitle(WINDOW,
                f"Click corner {len(corners)+1}/4 — {CORNER_LABELS[len(corners)]}")
        else:
            cv2.setWindowTitle(WINDOW, "All 4 corners marked — press ENTER to confirm, R to redo")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, min(1280, frame.shape[1]), min(720, frame.shape[0]))
    cv2.setMouseCallback(WINDOW, _on_click)
    cv2.setWindowTitle(WINDOW, f"Click corner 1/4 — {CORNER_LABELS[0]}")
    cv2.imshow(WINDOW, vis)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(corners) == 4:      # ENTER — confirm
            break
        elif key == ord('r') or key == ord('R'):  # R — redo
            corners.clear()
            vis[:] = frame
            cv2.imshow(WINDOW, vis)
            cv2.setWindowTitle(WINDOW, f"Click corner 1/4 — {CORNER_LABELS[0]}")
        elif key == 27:                            # ESC — abort
            cv2.destroyAllWindows()
            raise RuntimeError("Court setup aborted by user.")

    cv2.destroyAllWindows()

    arr = np.array(corners, dtype=np.float32)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(arr.tolist(), f, indent=2)
    print(f"[court] Saved corners to {json_path}")
    return arr


# ================================================================
# Homography helpers
# ================================================================

def build_homography(img_corners: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Compute H such that:  world_pt (homogeneous) = H @ img_pt (homogeneous)
    i.e. maps image pixels -> real-world court feet.
    """
    world_corners = _court_world_corners(cfg)
    H, _ = cv2.findHomography(img_corners, world_corners, method=0)
    if H is None:
        raise RuntimeError("Could not compute court homography — check corner points.")
    return H


def img_to_world(H: np.ndarray, px: float, py: float) -> Tuple[float, float]:
    """Project a single image point to world (court) coordinates."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)
    return float(world[0, 0, 0]), float(world[0, 0, 1])


# ================================================================
# Bounding-box smoother
# ================================================================

@dataclass
class BoxSmoother:
    """
    Exponentially-weighted moving average for a bounding box (cx, cy, w, h).
    Uses separate alphas for position and size, and suppresses size updates
    when the player appears stationary — the single most important guard
    against false delta_area spikes in the downstream RAI.
    """
    alpha_pos: float
    alpha_size: float
    still_thresh: float   # px/frame velocity below which size is frozen

    _cx: Optional[float] = field(default=None, init=False)
    _cy: Optional[float] = field(default=None, init=False)
    _w:  Optional[float] = field(default=None, init=False)
    _h:  Optional[float] = field(default=None, init=False)

    def update(self, cx: float, cy: float, w: float, h: float
               ) -> Tuple[float, float, float, float]:
        if self._cx is None:
            # First observation — initialise
            self._cx, self._cy, self._w, self._h = cx, cy, w, h
            return cx, cy, w, h

        # Pixel velocity from the last position
        vel = ((cx - self._cx) ** 2 + (cy - self._cy) ** 2) ** 0.5

        # Position: always update at alpha_pos
        self._cx = (1 - self.alpha_pos) * self._cx + self.alpha_pos * cx
        self._cy = (1 - self.alpha_pos) * self._cy + self.alpha_pos * cy

        # Size: use a smaller effective alpha when near-still
        if vel < self.still_thresh:
            # Player barely moved → hold box size very stable
            eff_alpha = self.alpha_size * 0.3
        else:
            eff_alpha = self.alpha_size

        self._w = (1 - eff_alpha) * self._w + eff_alpha * w
        self._h = (1 - eff_alpha) * self._h + eff_alpha * h

        return self._cx, self._cy, self._w, self._h

    def reset(self):
        self._cx = self._cy = self._w = self._h = None


# ================================================================
# Player selection
# ================================================================

def select_near_player(
    boxes_xyxy: List[Tuple[float, float, float, float]],
    H: np.ndarray,
    cfg: dict,
) -> Optional[Tuple[float, float, float, float]]:
    """
    From a list of person bounding boxes (x1, y1, x2, y2), choose the
    near-side player using real-world court coordinates.

    Steps
    -----
    1. Use the foot-point (bottom-centre of each box) for homography projection.
    2. Discard any detection outside the court + lateral padding.
    3. Return the box whose foot-point is closest to the near baseline (y = 0 ft).
    """
    half_w    = cfg["COURT_WIDTH_FT"] / 2.0
    court_len = cfg["COURT_LENGTH_FT"]
    x_pad     = cfg["LATERAL_PADDING_FT"]
    x_limit   = half_w + x_pad                  # ±28.5 ft by default
    y_limit   = court_len + 10.0                 # small margin past far baseline

    best_box:    Optional[Tuple[float, float, float, float]] = None
    best_near_y: float = float("inf")            # smallest = nearest to near baseline

    for (x1, y1, x2, y2) in boxes_xyxy:
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)
        wx, wy = img_to_world(H, foot_x, foot_y)

        # Filter: must be within expanded court boundaries
        if abs(wx) > x_limit:
            continue
        if wy < -10.0 or wy > y_limit:
            continue

        # Select closest to near baseline (y = 0)
        dist_near = abs(wy)
        if dist_near < best_near_y:
            best_near_y = dist_near
            best_box = (x1, y1, x2, y2)

    return best_box


# ================================================================
# Main extraction loop
# ================================================================

def extract_telemetry(
    video_path: str,
    output_csv: str = "telemetry.csv",
    court_json: str = "court_corners.json",
    cfg: dict = CONFIG,
) -> None:
    """
    End-to-end telemetry extraction pipeline.

    Parameters
    ----------
    video_path  : path to the input video
    output_csv  : path for the output CSV (frame_id, x, y, w, h)
    court_json  : path to cache file for court corner pixels
    cfg         : configuration dictionary
    """
    # ---- Video setup ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or cfg["FPS"]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[telemetry] Video: {video_path}  |  FPS={fps:.1f}  |  Frames={total}")

    # ---- Read reference frame for court setup ----
    ok, ref_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame from video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

    # ---- Court corners & homography ----
    img_corners = init_court(ref_frame, court_json)
    H = build_homography(img_corners, cfg)

    # ---- YOLO model ----
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("[error] ultralytics not installed — run: pip install ultralytics")

    print(f"[telemetry] Loading YOLO model: {cfg['YOLO_MODEL']}")
    model = YOLO(cfg["YOLO_MODEL"])

    # ---- Smoother ----
    smoother = BoxSmoother(
        alpha_pos=cfg["SMOOTH_ALPHA_POS"],
        alpha_size=cfg["SMOOTH_ALPHA_SIZE"],
        still_thresh=cfg["VELOCITY_STILL_THRESH"],
    )

    # ---- Output CSV ----
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_f = open(output_csv, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow(["frame_id", "x", "y", "w", "h"])

    frame_id   = 0
    rows_written = 0
    missing_frames = 0   # frames where no valid near player was found

    print("[telemetry] Processing frames …")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, classes=[0], conf=cfg["YOLO_CONF"], verbose=False)
        r = results[0] if results else None

        boxes_xyxy: List[Tuple[float, float, float, float]] = []
        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes_xyxy.append((x1, y1, x2, y2))

        best = select_near_player(boxes_xyxy, H, cfg)

        if best is not None:
            x1, y1, x2, y2 = best
            raw_cx = (x1 + x2) / 2.0
            raw_cy = (y1 + y2) / 2.0
            raw_w  = x2 - x1
            raw_h  = y2 - y1

            cx, cy, w, h = smoother.update(raw_cx, raw_cy, raw_w, raw_h)
            writer.writerow([frame_id, f"{cx:.2f}", f"{cy:.2f}",
                             f"{w:.2f}", f"{h:.2f}"])
            rows_written += 1
        else:
            missing_frames += 1
            # Do not write a row for this frame; downstream handles gaps.

        if frame_id % 300 == 0:
            pct = 100.0 * frame_id / max(1, total)
            print(f"  frame {frame_id}/{total}  ({pct:.1f}%)  written={rows_written}  missing={missing_frames}")

        frame_id += 1

    cap.release()
    out_f.close()

    print(f"\n[telemetry] Done — {rows_written} rows written to {output_csv}")
    print(f"[telemetry] {missing_frames} frames had no valid near-player detection "
          f"({100.0*missing_frames/max(1,frame_id):.1f}%)")


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract near-player bounding-box telemetry from a tennis video."
    )
    parser.add_argument("video", help="Input video file path")
    parser.add_argument(
        "--output", "-o",
        default="telemetry.csv",
        help="Output CSV path (default: telemetry.csv)",
    )
    parser.add_argument(
        "--court-json",
        default="court_corners.json",
        help="Path for court corner cache JSON (default: court_corners.json)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"YOLO model name or path (default: {CONFIG['YOLO_MODEL']})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"YOLO confidence threshold (default: {CONFIG['YOLO_CONF']})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override video FPS (used for reporting only)",
    )
    parser.add_argument(
        "--alpha-pos",
        type=float,
        default=None,
        help=f"EWA alpha for position smoothing (default: {CONFIG['SMOOTH_ALPHA_POS']})",
    )
    parser.add_argument(
        "--alpha-size",
        type=float,
        default=None,
        help=f"EWA alpha for size smoothing (default: {CONFIG['SMOOTH_ALPHA_SIZE']})",
    )
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.model:
        cfg["YOLO_MODEL"] = args.model
    if args.conf is not None:
        cfg["YOLO_CONF"] = args.conf
    if args.fps is not None:
        cfg["FPS"] = args.fps
    if args.alpha_pos is not None:
        cfg["SMOOTH_ALPHA_POS"] = args.alpha_pos
    if args.alpha_size is not None:
        cfg["SMOOTH_ALPHA_SIZE"] = args.alpha_size

    extract_telemetry(
        video_path=args.video,
        output_csv=args.output,
        court_json=args.court_json,
        cfg=cfg,
    )
