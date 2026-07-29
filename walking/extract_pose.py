"""
extract_pose.py
===============
Person-pose detection pass. Runs yolov8n-pose on every frame of a clip resized
to the 960x540 analysis frame and caches EVERY person detection, unfiltered.

Selecting which person is the near player is deliberately not done here — see
``walking/select_near.py``. Detection costs ten minutes of GPU per clip while
selection is a rule that wants iterating, so the two are separate passes with a
cache in between.

Output ``<clip>/<stem>_walk_dets.npz``:
    kp    [N, K, 17, 3]  COCO keypoints in 960x540 image pixels (NaN = no slot)
    box   [N, K, 4]      person boxes xyxy in 960x540 pixels
    conf  [N, K]         detection confidence
    fps   scalar
K is ``MAX_PERSONS``; slots are ordered by confidence and padded with NaN.

Usage:
    python -m walking.extract_pose /Volumes/Anya/Data/21/snippet.mp4
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from walking.court import ANALYSIS_SIZE

N_KP = 17
MAX_PERSONS = 8
POSE_CONF = 0.20


def dets_path(video_path):
    d = os.path.dirname(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}_walk_dets.npz")


def extract(video_path, model_path="yolov8n-pose.pt", device="mps",
            limit=None, rescan=False):
    out_p = dets_path(video_path)
    if os.path.isfile(out_p) and not rescan:
        print(f"[walk-dets] cached: {out_p}")
        return out_p

    from ultralytics import YOLO
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit:
        total = min(total, limit)

    kp = np.full((total, MAX_PERSONS, N_KP, 3), np.nan, dtype=np.float32)
    bx = np.full((total, MAX_PERSONS, 4), np.nan, dtype=np.float32)
    cf = np.full((total, MAX_PERSONS), np.nan, dtype=np.float32)

    t0, empty = time.time(), 0
    for f in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        res = model.predict(frame, imgsz=960, conf=POSE_CONF, device=device,
                            classes=[0], verbose=False)[0]
        if res.keypoints is None or len(res.boxes) == 0:
            empty += 1
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kpts = res.keypoints.data.cpu().numpy()
        order = np.argsort(confs)[::-1][:MAX_PERSONS]
        k = len(order)
        kp[f, :k] = kpts[order]
        bx[f, :k] = boxes[order]
        cf[f, :k] = confs[order]
        if f and f % 1000 == 0:
            el = time.time() - t0
            print(f"  {f}/{total}  {f / el:.1f} fps  empty {empty / (f + 1):.1%}",
                  flush=True)
    cap.release()

    np.savez_compressed(out_p, kp=kp, box=bx, conf=cf, fps=np.float64(fps))
    per = np.mean(np.sum(np.isfinite(cf), axis=1))
    print(f"[walk-dets] {out_p}: {total} frames, {per:.2f} persons/frame, "
          f"{empty / max(total, 1):.1%} empty, {time.time() - t0:.0f}s")
    return out_p


def rescue(video_path, model_path="yolov8n-pose.pt", device="mps",
           imgsz=1920, work_size=(1920, 1080)):
    """Second pass at high resolution over the frames that came back empty.

    A player walking to the ball carts is ~60 px tall in the 960x540 analysis
    frame, which yolov8n-pose misses entirely; at 1920 px input it is found
    reliably. Those frames are 25% of this clip and they are not random — they
    are the mid-clip break, which carries hand-labelled walking — so leaving
    them empty biases both training and evaluation. Coordinates are rescaled
    back to the 960x540 frame so downstream code sees one coordinate system.
    """
    from ultralytics import YOLO
    out_p = dets_path(video_path)
    z = np.load(out_p)
    kp, bx, cf, fps = z["kp"].copy(), z["box"].copy(), z["conf"].copy(), z["fps"]
    todo = np.flatnonzero(~np.isfinite(cf).any(axis=1))
    if len(todo) == 0:
        print("[rescue] nothing to do")
        return out_p
    print(f"[rescue] {len(todo)} empty frames at {imgsz}px")

    model = YOLO(model_path)
    sx = ANALYSIS_SIZE[0] / work_size[0]
    sy = ANALYSIS_SIZE[1] / work_size[1]
    cap = cv2.VideoCapture(video_path)
    todo_set = set(int(t) for t in todo)
    found, t0 = 0, time.time()
    for f in range(len(cf)):
        ok, frame = cap.read()
        if not ok:
            break
        if f not in todo_set:
            continue
        frame = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
        res = model.predict(frame, imgsz=imgsz, conf=POSE_CONF, device=device,
                            classes=[0], verbose=False)[0]
        if res.keypoints is None or len(res.boxes) == 0:
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kpts = res.keypoints.data.cpu().numpy()
        order = np.argsort(confs)[::-1][:MAX_PERSONS]
        k = len(order)
        b = boxes[order] * np.array([sx, sy, sx, sy])
        p = kpts[order].copy()
        p[..., 0] *= sx
        p[..., 1] *= sy
        kp[f, :k], bx[f, :k], cf[f, :k] = p, b, confs[order]
        found += 1
        if found and found % 500 == 0:
            print(f"  rescued {found}/{len(todo)}  "
                  f"{found / (time.time() - t0):.1f} fps", flush=True)
    cap.release()

    np.savez_compressed(out_p, kp=kp, box=bx, conf=cf, fps=fps)
    empty = float(np.mean(~np.isfinite(cf).any(axis=1)))
    print(f"[rescue] recovered {found}/{len(todo)} frames, empty now {empty:.1%}, "
          f"{time.time() - t0:.0f}s")
    return out_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="yolov8n-pose.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--rescue", action="store_true",
                    help="high-resolution second pass over empty frames")
    ap.add_argument("--imgsz", type=int, default=1920)
    a = ap.parse_args()
    if a.rescue:
        rescue(a.video, a.model, a.device, a.imgsz)
    else:
        extract(a.video, a.model, a.device, a.limit, a.rescan)


if __name__ == "__main__":
    main()
