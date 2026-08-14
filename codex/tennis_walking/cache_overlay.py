"""Render the walking overlay from the project's cached near-player poses.

This is a fallback for environments where MediaPipe cannot initialize (for
example, a headless macOS session). The cache must have been produced for the
same clip by the pipeline's near-player pose extractor.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import json

import cv2
import numpy as np

from .detector import WalkingDetector, _summarize_second


COCO_EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
              (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def _cached_landmarks(keypoints: np.ndarray, bbox: list[float]) -> list[object] | None:
    points = keypoints.reshape(17, 3)
    if not np.isfinite(points).all() or min(points[11, 2], points[12, 2], points[15, 2], points[16, 2]) < .55:
        return None
    bx, by, bw, bh = bbox
    result = [SimpleNamespace(x=0., y=0., visibility=0.) for _ in range(33)]
    # COCO 11/12/15/16 map to BlazePose's left/right hips and ankles.
    for coco, blaze in ((11, 23), (12, 24), (15, 27), (16, 28)):
        x, y, confidence = points[coco]
        result[blaze] = SimpleNamespace(x=float((bx + x * bw) / 960), y=float((by + y * bh) / 540), visibility=float(confidence))
    return result


def render(video: Path, homography: np.ndarray, pose_cache: Path, telemetry: Path, output_video: Path, output_jsonl: Path) -> None:
    meta = json.loads(telemetry.read_text())
    cached = np.load(pose_cache)
    frames: dict[int, tuple[np.ndarray, list[float]]] = {}
    for index, rally in enumerate(meta['rallies']):
        poses = cached[f'r{index}']
        for offset, keypoints in enumerate(poses):
            frame_number = rally['start'] + offset
            cell = rally['frames'].get(str(frame_number))
            if cell and cell.get('near_bbox'):
                frames[frame_number] = (keypoints, cell['near_bbox'])
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (960, 540))
    detector = WalkingDetector(homography, near_side_min_image_y=.5)
    results, bucket, second, frame_number = [], [], 0, 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        timestamp = frame_number / fps
        while int(timestamp) > second:
            results.append(_summarize_second(second, bucket))
            bucket, second = [], second + 1
        entry = frames.get(frame_number)
        metrics, confidence, stride, separation = (None, 0., None, None)
        if entry:
            points, bbox = entry
            landmarks = _cached_landmarks(points, bbox)
            metrics, confidence, stride, separation = detector.update(timestamp, landmarks, 960, 540)
            if metrics:
                bucket.append((metrics, confidence, stride, separation))
            reshaped = points.reshape(17, 3)
            for a, b in COCO_EDGES:
                if reshaped[a, 2] >= .4 and reshaped[b, 2] >= .4:
                    ax, ay = bbox[0] + reshaped[a, 0] * bbox[2], bbox[1] + reshaped[a, 1] * bbox[3]
                    bx, by = bbox[0] + reshaped[b, 0] * bbox[2], bbox[1] + reshaped[b, 1] * bbox[3]
                    cv2.line(frame, (round(ax), round(ay)), (round(bx), round(by)), (0, 220, 255), 2)
        walking = confidence >= .55
        speed_text = f"speed: {metrics['speed']:.2f} m/s" if metrics and metrics['speed'] is not None else 'speed: unavailable'
        cv2.putText(frame, speed_text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .7, (30, 220, 30), 2)
        cv2.putText(frame, f'walking: {walking} ({confidence:.2f})', (18, 64), cv2.FONT_HERSHEY_SIMPLEX, .7, (30, 220, 30) if walking else (20, 40, 220), 2)
        cv2.putText(frame, 'cached-pose fallback', (18, 94), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 180, 20), 1)
        writer.write(frame)
        frame_number += 1
    if frame_number:
        results.append(_summarize_second(second, bucket))
    capture.release(); writer.release()
    with output_jsonl.open('w') as stream:
        for item in results:
            stream.write(json.dumps(item.__dict__) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render walking overlay using an existing pose cache.')
    parser.add_argument('video', type=Path); parser.add_argument('--homography', type=Path, required=True)
    parser.add_argument('--pose-cache', type=Path, required=True); parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--jsonl', type=Path, required=True)
    args = parser.parse_args()
    render(args.video, np.load(args.homography), args.pose_cache, args.telemetry, args.output, args.jsonl)
