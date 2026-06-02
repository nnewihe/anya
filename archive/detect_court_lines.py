"""
detect_court_lines.py

Derives the nearside baseline and singles sidelines from the 4-vertex
singles-court polygon produced by court_mask.py.  No additional YOLO or
edge detection is needed — the homography is recomputed from the stored
polygon corners.

Canonical court coordinate system (feet):
    x ∈ [-13.5, 13.5]   singles width = 27 ft
    y = 0               near baseline  (closest to camera)
    y = 78              far baseline

Polygon vertex order (matches _build_singles_polygon_from_6kps):
    poly[0] = near-left   (-13.5,  0)
    poly[1] = near-right  ( 13.5,  0)
    poly[2] = far-right   ( 13.5, 78)
    poly[3] = far-left    (-13.5, 78)
"""

from __future__ import annotations
import os, sys
from dataclasses import dataclass
from typing import Tuple, Optional
import cv2
import numpy as np

_HALF_W  = 13.5   # singles half-width (ft)
_Y_NEAR  = 0.0    # near baseline y (ft)
_Y_FAR   = 78.0   # far  baseline y (ft)

# Canonical corners in the same order as the polygon produced by court_mask.py
_CANONICAL_CORNERS = np.array([
    [-_HALF_W, _Y_NEAR],   # near-left
    [ _HALF_W, _Y_NEAR],   # near-right
    [ _HALF_W, _Y_FAR ],   # far-right
    [-_HALF_W, _Y_FAR ],   # far-left
], dtype=np.float32)


@dataclass
class CourtLines:
    """All detected court lines in image-pixel coordinates."""
    near_baseline:          Tuple[float, float, float, float]  # (x1,y1,x2,y2)
    singles_left_sideline:  Tuple[float, float, float, float]
    singles_right_sideline: Tuple[float, float, float, float]
    near_left_vertex:       Tuple[float, float]   # baseline ∩ left sideline
    near_right_vertex:      Tuple[float, float]   # baseline ∩ right sideline
    H:                      np.ndarray             # (3,3) court-ft → image-px


def homography_from_poly(poly: np.ndarray) -> np.ndarray:
    """
    Compute a 3×3 homography that maps canonical court coordinates (feet)
    to image pixel coordinates, using the 4 stored polygon corners.

    poly: (4,2) float32 in order [near-left, near-right, far-right, far-left]
    """
    poly = np.asarray(poly, dtype=np.float32).reshape(4, 2)
    H = cv2.getPerspectiveTransform(_CANONICAL_CORNERS, poly)
    return H


def build_court_lines(H: np.ndarray) -> CourtLines:
    """
    Project the near baseline and singles sidelines through H.

    H: (3,3) homography from court-ft → image-px
    """
    def _proj(pts_ft: np.ndarray) -> np.ndarray:
        """Project (N,2) court-ft points → (N,2) image-px via H."""
        src = pts_ft.reshape(-1, 1, 2).astype(np.float32)
        dst = cv2.perspectiveTransform(src, H)
        return dst.reshape(-1, 2)

    corners_ft = np.array([
        [-_HALF_W, _Y_NEAR],   # near-left  (= near_left_vertex)
        [ _HALF_W, _Y_NEAR],   # near-right (= near_right_vertex)
        [ _HALF_W, _Y_FAR ],   # far-right
        [-_HALF_W, _Y_FAR ],   # far-left
    ], dtype=np.float32)

    px = _proj(corners_ft)   # shape (4,2)

    near_left  = (float(px[0, 0]), float(px[0, 1]))
    near_right = (float(px[1, 0]), float(px[1, 1]))
    far_right  = (float(px[2, 0]), float(px[2, 1]))
    far_left   = (float(px[3, 0]), float(px[3, 1]))

    near_baseline          = (*near_left,  *near_right)
    singles_left_sideline  = (*near_left,  *far_left)
    singles_right_sideline = (*near_right, *far_right)

    return CourtLines(
        near_baseline          = near_baseline,
        singles_left_sideline  = singles_left_sideline,
        singles_right_sideline = singles_right_sideline,
        near_left_vertex       = near_left,
        near_right_vertex      = near_right,
        H                      = H,
    )


def draw_court_lines(
    frame: np.ndarray,
    lines: CourtLines,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    vertex_radius: int = 6,
) -> np.ndarray:
    """
    Draw the near baseline, singles sidelines, and vertex dots on a copy of frame.
    Returns the annotated frame.
    """
    out = frame.copy()

    def _seg(seg):
        return (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3]))

    cv2.line(out, *_seg(lines.near_baseline),          color, thickness, cv2.LINE_AA)
    cv2.line(out, *_seg(lines.singles_left_sideline),  color, thickness, cv2.LINE_AA)
    cv2.line(out, *_seg(lines.singles_right_sideline), color, thickness, cv2.LINE_AA)

    vertex_color = (0, 0, 255)
    cv2.circle(out, (int(lines.near_left_vertex[0]),  int(lines.near_left_vertex[1])),
               vertex_radius, vertex_color, -1, cv2.LINE_AA)
    cv2.circle(out, (int(lines.near_right_vertex[0]), int(lines.near_right_vertex[1])),
               vertex_radius, vertex_color, -1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# CLI helper — run directly to verify on a video
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_court_lines.py <video_path> [poly_json_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    default_json = os.path.join(
        os.path.dirname(video_path),
        os.path.splitext(os.path.basename(video_path))[0] + "_court_poly.json",
    )
    poly_json = sys.argv[2] if len(sys.argv) >= 3 else default_json

    # Import here to avoid circular dependency when used as a library
    sys.path.insert(0, _this_dir)
    from court_mask import load_or_calibrate

    _cap = cv2.VideoCapture(video_path)
    _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    _cap.release()
    result = load_or_calibrate(video_path, poly_json, frame_start=0, frame_end=_total)

    if result.court_lines is None:
        print("[ERROR] court_lines not populated — check court_mask.py integration.")
        sys.exit(1)

    lines = result.court_lines
    print(f"\nNear-left  vertex : {lines.near_left_vertex[0]:.1f}, {lines.near_left_vertex[1]:.1f} px")
    print(f"Near-right vertex : {lines.near_right_vertex[0]:.1f}, {lines.near_right_vertex[1]:.1f} px")
    print(f"Near baseline     : {lines.near_baseline}")
    print(f"Left sideline     : {lines.singles_left_sideline}")
    print(f"Right sideline    : {lines.singles_right_sideline}")

    # Draw on middle frame and save
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, _total // 2)
    ok, frame = cap.read()
    cap.release()

    if ok:
        vis = draw_court_lines(frame, lines)
        out_path = os.path.splitext(video_path)[0] + "_court_lines_debug.png"
        cv2.imwrite(out_path, vis)
        print(f"\nDebug image saved: {out_path}")
    else:
        print("[WARN] Could not read middle frame for debug image.")
