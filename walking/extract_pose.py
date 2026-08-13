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
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from walking.court import ANALYSIS_SIZE

N_KP = 17
MAX_PERSONS = 8
POSE_CONF = 0.20

# Absolute path to the pose weights, NOT the bare "yolov8n-pose.pt".
#
# Handed a bare name, ultralytics looks in the current working directory and
# then downloads the weights over the internet. In the packaged app that meant
# ignoring the bundled copy and fetching one at the start of every reel: it
# worked silently for anyone online, and failed outright for a tester who was
# not. It also quietly made a "runs 100% on this computer" app depend on a
# network call.
#
# walking/ is a sibling of pipeline/, and that relative layout is preserved
# inside the PyInstaller bundle (rally_app.spec ships the weights to
# pipeline/models), so parents[1] resolves in both the frozen app and a source
# run. Falls back to the bare name if the file is somehow absent, which keeps
# a bare checkout of walking/ working the way it always did.
_POSE_WEIGHTS = Path(__file__).resolve().parents[1] / "pipeline" / "models" / "yolov8n-pose.pt"
DEFAULT_POSE_MODEL = str(_POSE_WEIGHTS) if _POSE_WEIGHTS.is_file() else "yolov8n-pose.pt"

BATCH = 16
# One model call per frame pays a fixed per-call cost (Python preprocess, the
# MPS dispatch, postprocess) that batching amortises: measured on an M4 over
# Data/21, 14.69 ms/frame at B=1 -> 12.31 at B=8 -> 12.02 at B=16, flat after.
# Every frame here is the same ANALYSIS_SIZE, and ultralytics only changes its
# letterbox mode for mixed-shape batches (`pre_transform`: `auto=same_shapes
# and ...`), so results are unchanged — verified identical on Data/21.
# Set to 1 for the old one-frame-at-a-time path.


def dets_path(video_path):
    d = os.path.dirname(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}_walk_dets.npz")


def _read_frames(cap, total, batch):
    """Yields lists of (frame_index, analysis_frame), at most `batch` long."""
    buf = []
    for f in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        buf.append((f, cv2.resize(frame, ANALYSIS_SIZE,
                                  interpolation=cv2.INTER_AREA)))
        if len(buf) >= batch:
            yield buf
            buf = []
    if buf:
        yield buf


def _prefetched(cap, total, batch, depth=2):
    """`_read_frames` on a reader thread, so decode overlaps inference.

    Decode here is 6.4 ms/frame (4K read + INTER_AREA downscale to the
    analysis frame) against 12.6 ms/frame of model time, and the two were
    running strictly one after the other.  Both are real work, but one is CPU
    and one is GPU, so overlapping them is free.  Order is preserved (one
    reader, one queue) and exceptions cross the queue rather than vanishing
    into the thread.
    """
    q = queue.Queue(maxsize=depth)
    SENTINEL = object()

    def _read():
        try:
            for chunk in _read_frames(cap, total, batch):
                q.put(chunk)
        except BaseException as ex:          # surfaced on the consumer side
            q.put(ex)
        finally:
            q.put(SENTINEL)

    t = threading.Thread(target=_read, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        t.join(timeout=5.0)


def extract(video_path, model_path=DEFAULT_POSE_MODEL, device="mps",
            limit=None, rescan=False, batch=BATCH, prefetch=True):
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
    batch = max(1, int(batch))
    source = (_prefetched(cap, total, batch) if prefetch
              else _read_frames(cap, total, batch))
    for chunk in source:
        frames = [c[1] for c in chunk]
        results = model.predict(frames if batch > 1 else frames[0], imgsz=960,
                                conf=POSE_CONF, device=device, classes=[0],
                                verbose=False)
        for (f, _), res in zip(chunk, results):
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


def rescue(video_path, model_path=DEFAULT_POSE_MODEL, device="mps",
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
    ap.add_argument("--model", default=DEFAULT_POSE_MODEL)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--rescue", action="store_true",
                    help="high-resolution second pass over empty frames")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--batch", type=int, default=BATCH,
                    help=f"frames per model call (default {BATCH}; 1 = the old "
                         "one-frame-at-a-time path)")
    a = ap.parse_args()
    if a.rescue:
        rescue(a.video, a.model, a.device, a.imgsz)
    else:
        extract(a.video, a.model, a.device, a.limit, a.rescan, batch=a.batch)


if __name__ == "__main__":
    main()
