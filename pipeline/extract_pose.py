"""
extract_pose.py
==============
Stage 1 of the active/dead near-player classifier: extract a normalized pose
sequence for each near-side rally, cached so training/windowing never re-runs
detection.

For every frame in a rally's cached span [start, span_end] we run yolov8n-pose
on the 960x540 frame, pick the near player by IoU against the cached near_bbox
(from energy_telemetry_cache.json — keeps selection consistent with the energy
bar), and store the 17 COCO keypoints normalized to that box:
    nx = (x - box_x) / box_w,  ny = (y - box_y) / box_h,  conf
Frames with no near player (or no pose match) are stored as NaN and masked
downstream.

Output per clip:
    <clip>/pose_cache.npz    arrays "r0","r1",... each [T, 51] (T frames, 17x3)
    <clip>/pose_meta.json    per-rally {start,end,span_end}, plus fps

Usage:
    python pipeline/extract_pose.py --clips 36        # smoke test
    python pipeline/extract_pose.py                   # all cached near clips
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_energy import _near_rallies, _video_path, ANALYSIS_SIZE, TELEMETRY_CACHE

POSE_NPZ  = "pose_cache.npz"
POSE_META = "pose_meta.json"
N_KP      = 17
IOU_MIN   = 0.2


def _iou_xywh_xyxy(nb, xyxy):
    ax1, ay1, aw, ah = nb
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bx2, by2 = xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = aw * ah + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _select_near_pose(result, near_bbox):
    """Return the [17,3] keypoints of the pose detection best matching near_bbox."""
    if near_bbox is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    kpts = result.keypoints.data.cpu().numpy()   # [N,17,3]
    best_i, best_iou = -1, IOU_MIN
    for i, b in enumerate(boxes):
        j = _iou_xywh_xyxy(near_bbox, b)
        if j > best_iou:
            best_iou, best_i = j, i
    return kpts[best_i] if best_i >= 0 else None


def _normalize(kp, near_bbox):
    bx, by, bw, bh = near_bbox
    bw = bw or 1.0
    bh = bh or 1.0
    out = np.empty(N_KP * 3, dtype=np.float32)
    for k in range(N_KP):
        x, y, c = kp[k]
        out[3 * k + 0] = (x - bx) / bw
        out[3 * k + 1] = (y - by) / bh
        out[3 * k + 2] = c
    return out


def extract_clip(clip_dir, pose_model, device, rescan=False):
    name = os.path.basename(clip_dir)
    npz_path = os.path.join(clip_dir, POSE_NPZ)
    if os.path.isfile(npz_path) and not rescan:
        print(f"[pose] {name}: cached")
        return

    tel_path = os.path.join(clip_dir, TELEMETRY_CACHE)
    if not os.path.isfile(tel_path):
        print(f"[pose] {name}: no telemetry cache — skip")
        return
    tel = json.load(open(tel_path))
    vid = _video_path(clip_dir)
    cap = cv2.VideoCapture(vid)
    fps = tel["fps"]

    arrays, meta = {}, {"fps": fps, "rallies": []}
    for ri, r in enumerate(tel["rallies"]):
        start, end, span_end = r["start"], r["end"], r["span_end"]
        frames = r["frames"]
        T = span_end - start + 1
        arr = np.full((T, N_KP * 3), np.nan, dtype=np.float32)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for t in range(T):
            ok, frame = cap.read()
            if not ok:
                break
            f = start + t
            cell = frames.get(str(f))
            nb = cell["near_bbox"] if cell else None
            if nb is None:
                continue
            frame = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
            res = pose_model.predict(frame, imgsz=640, device=device, verbose=False)[0]
            kp = _select_near_pose(res, nb)
            if kp is not None:
                arr[t] = _normalize(kp, nb)

        arrays[f"r{ri}"] = arr
        present = int(np.sum(~np.isnan(arr[:, 0])))
        meta["rallies"].append({"start": start, "end": end, "span_end": span_end})
        print(f"    {name} rally {ri+1}/{len(tel['rallies'])} [{start}-{end}]  pose {present}/{T}")

    cap.release()
    np.savez_compressed(npz_path, **arrays)
    json.dump(meta, open(os.path.join(clip_dir, POSE_META), "w"))
    print(f"[pose] {name}: cached -> {POSE_NPZ}")


def discover(data_root):
    clips = []
    for name in sorted(os.listdir(data_root)):
        d = os.path.join(data_root, name)
        if os.path.isfile(os.path.join(d, TELEMETRY_CACHE)) and _near_rallies(d):
            clips.append(d)
    return clips


def main():
    ap = argparse.ArgumentParser(description="Extract normalized near-player pose sequences per rally")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--pose_model", default="yolov8n-pose.pt")
    ap.add_argument("--rescan", action="store_true")
    args = ap.parse_args()

    clip_dirs = ([os.path.join(args.data_root, c) for c in args.clips]
                 if args.clips else discover(args.data_root))
    print(f"[init] {len(clip_dirs)} clips: {[os.path.basename(c) for c in clip_dirs]}")

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[init] loading {args.pose_model} on {device}")
    pose_model = YOLO(args.pose_model)

    for c in clip_dirs:
        try:
            extract_clip(c, pose_model, device, rescan=args.rescan)
        except Exception as e:
            print(f"[WARN] pose extraction failed for {os.path.basename(c)}: {e}")


if __name__ == "__main__":
    main()
