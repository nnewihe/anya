"""
extract_serve_pose.py
====================
Stage 1 of the learned serve detector: extract near-player pose over the serve
run-up for each GT near serve. The serve motion (ready -> toss -> trophy ->
contact) happens BEFORE the GT `start`, which the point-end pose cache (starting
at `start`) misses — so we scan [start - PRE, start + POST] here.

Near player is selected by court zone (no cached box exists before `start`):
the pose detection whose foot maps closest to the near baseline. Keypoints are
normalized to that detection's box, matching the point-end features.

Output per clip:
    <clip>/serve_pose_cache.npz   arrays "s0","s1",... each [T,51]
    <clip>/serve_pose_meta.json   {fps, serves:[{start, f0}]}

Usage:
    python pipeline/extract_serve_pose.py --clips 36
    python pipeline/extract_serve_pose.py
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
from extract_pose import _normalize, N_KP
from anya_near_serve import PointStartSystem
from optimize_energy import _near_rallies, _video_path, _court_cache_path, ANALYSIS_SIZE

PRE_SEC, POST_SEC = 2.5, 1.5
SERVE_NPZ  = "serve_pose_cache.npz"
SERVE_META = "serve_pose_meta.json"


def _select_near_by_zone(result, geom):
    """Pose detection whose foot maps into the near half, closest to the baseline."""
    if result.keypoints is None or len(result.boxes) == 0:
        return None, None
    boxes = result.boxes.xyxy.cpu().numpy()
    kpts = result.keypoints.data.cpu().numpy()
    best_i, best_key, best_box = -1, None, None
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        wx, wy = geom._get_world_coords((x1 + x2) / 2, y2)
        if -2.0 <= wx <= 29.0 and wy <= 38.0:
            if best_key is None or abs(wy) < best_key:
                best_key, best_i = abs(wy), i
                best_box = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
    if best_i < 0:
        return None, None
    return kpts[best_i], best_box


def extract_clip(clip_dir, pose_model, device, rescan=False):
    name = os.path.basename(clip_dir)
    npz_path = os.path.join(clip_dir, SERVE_NPZ)
    if os.path.isfile(npz_path) and not rescan:
        print(f"[serve-pose] {name}: cached"); return

    vid = _video_path(clip_dir)
    corners = np.array(json.load(open(_court_cache_path(vid)))["points"], dtype=np.float32)
    serves = sorted(r["start"] for r in _near_rallies(clip_dir))
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    geom = PointStartSystem(corners, ANALYSIS_SIZE[0], ANALYSIS_SIZE[1], fps=int(round(fps)), verbose=False)
    pre, post = round(fps * PRE_SEC), round(fps * POST_SEC)

    arrays, meta = {}, {"fps": fps, "serves": []}
    for si, s in enumerate(serves):
        f0, f1 = max(0, s - pre), min(total - 1, s + post)
        T = f1 - f0 + 1
        arr = np.full((T, N_KP * 3), np.nan, dtype=np.float32)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for t in range(T):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
            res = pose_model.predict(frame, imgsz=640, device=device, verbose=False)[0]
            kp, nb = _select_near_by_zone(res, geom)
            if kp is not None:
                arr[t] = _normalize(kp, nb)
        arrays[f"s{si}"] = arr
        meta["serves"].append({"start": int(s), "f0": int(f0)})
        present = int(np.sum(~np.isnan(arr[:, 0])))
        print(f"    {name} serve {si+1}/{len(serves)} @ {s}  pose {present}/{T}")

    cap.release()
    np.savez_compressed(npz_path, **arrays)
    json.dump(meta, open(os.path.join(clip_dir, SERVE_META), "w"))
    print(f"[serve-pose] {name}: cached -> {SERVE_NPZ}")


def main():
    ap = argparse.ArgumentParser(description="Extract near-player pose over serve run-ups")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--pose_model", default="yolov8n-pose.pt")
    ap.add_argument("--rescan", action="store_true")
    args = ap.parse_args()

    if args.clips:
        clip_dirs = [os.path.join(args.data_root, c) for c in args.clips]
    else:
        clip_dirs = [os.path.join(args.data_root, d) for d in sorted(os.listdir(args.data_root))
                     if _near_rallies(os.path.join(args.data_root, d))
                     and os.path.isfile(_court_cache_path(_video_path(os.path.join(args.data_root, d)) or ""))]

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[init] {len(clip_dirs)} clips on {device}")
    pose_model = YOLO(args.pose_model)
    for c in clip_dirs:
        try:
            extract_clip(c, pose_model, device, rescan=args.rescan)
        except Exception as e:
            print(f"[WARN] {os.path.basename(c)}: {e}")


if __name__ == "__main__":
    main()
