"""
cache_detections.py
===================
Pass-1-only driver for the ball tuner.  Runs the exact same detection setup as
``ball_detector.py`` (court corners -> homography, auto exclusion zones, YOLO
over every frame) but at a *low confidence floor* so the confidence threshold
itself can be tuned later by filtering the cache — no re-detection needed.

The result is a single JSON next to the video:

    <stem>_dets.json = {
        "fps", "width", "height",
        "court_points": [[x,y] x4] | [],
        "exclusion_zones": [[x1,y1,x2,y2], ...],
        "conf_floor": 0.02,
        "dets": [ [[x,y,conf], ...],   # frame 0
                  [[x,y,conf], ...],   # frame 1
                  ... ]                # one list per video frame (stride=1)
    }

Detections are stored in full-resolution pixels, exactly as the tracker consumes
them.  This is the only slow step; run it once.

    python pipeline/ball_tuning/cache_detections.py /Volumes/Anya/Data/69/snippet_1min.mp4
"""

from __future__ import annotations

import json
import os
import sys

import cv2

# Make the pipeline dir importable whether run from repo root or elsewhere.
_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from ball_detector import AnyaBallDetector  # noqa: E402


def cache_path(video_path: str) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}_dets.json")


def main(video_path: str, conf_floor: float = 0.02,
         imgsz: int = 1280, batch_size: int = 16) -> None:
    # Detect at a low floor so `ball_conf` becomes a tunable filter on the cache.
    det = AnyaBallDetector(video_path, imgsz=imgsz, stride=1,
                           batch_size=batch_size, ball_conf=conf_floor)

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det._set_infer_size(width, height)

    # Court corners (cached/click-selected once) + exclusion zones — identical
    # setup to ball_detector.process_video so the tuner sees production gating.
    if not det.load_config():
        det.get_court_polygon(first_frame)
        det.save_config()
    det._build_homography()
    det._init_exclusion_zones(width, height)

    # Detection pass over EVERY frame (stride=1), batched on the GPU.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    dets_per_frame = []
    batch_frames, batch_slots = [], []

    def _flush():
        if not batch_frames:
            return
        for slot, dets in zip(batch_slots, det._detect_batch(batch_frames)):
            dets_per_frame[slot] = [[round(float(x), 2), round(float(y), 2),
                                     round(float(c), 4)] for (x, y, c) in dets]
        batch_frames.clear()
        batch_slots.clear()

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        dets_per_frame.append([])
        batch_slots.append(idx)
        batch_frames.append(frame)
        if len(batch_frames) >= batch_size:
            _flush()
        idx += 1
        if idx % 200 == 0:
            print(f"[INFO] Detected {idx} frames...")
    _flush()
    cap.release()

    payload = {
        "fps": float(fps),
        "width": width,
        "height": height,
        "court_points": [list(p) for p in det.court_points],
        "exclusion_zones": [list(z) for z in det.exclusion_zones],
        "conf_floor": conf_floor,
        "dets": dets_per_frame,
    }
    out = cache_path(video_path)
    with open(out, "w") as f:
        json.dump(payload, f)
    total = sum(len(d) for d in dets_per_frame)
    print(f"[INFO] Cached {total} detections over {idx} frames -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python cache_detections.py <video.mp4> [conf_floor]")
        sys.exit(1)
    cf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    main(sys.argv[1], conf_floor=cf)
