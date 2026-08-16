"""
extract_far_pose.py
====================
Offline pose pass over the far-player ROI, keyed by frame index.  Sibling
cache to an anya_telemetry v2 JSONL: reads that file's `fpr` box (the
native-resolution far-baseline crop) per frame, runs yolov8n-pose on the
padded crop, and writes the shoulder/wrist relationship anya_far_serve.py
needs for its hand-above-shoulder gate.

Feasibility was spot-checked before building this (see conversation record):
over 2s windows around 11 confirmed far serves, max wrist-above-shoulder
margin was 5.8-27px with the arm raised across many sampled frames; 10
negative-control windows (>=4s from any far serve) mostly stayed at or below
a ~5px noise floor. The separation is clean enough to gate on, provided the
gate looks for a RISE across a short window rather than a single-frame
threshold — a player can rest with a raised arm for reasons unrelated to
serving, but a sustained transition from low to high within ~1.5s is
specific to the service motion.

Per-frame record (compact JSONL keys, one per source telemetry frame):
    f     frame index (matches the source telemetry's `f`)
    t     timestamp seconds
    k     flat COCO-17 keypoints [x,y,conf] * 17 in CROP pixels (the fpr box
          padded by pad_px, at native resolution — so y-differences are in
          native pixels).  Absent when there is no `fpr` box or no pose
          detection on that frame.
    bh    fpr box height in native pixels, for scale normalisation.

Raw keypoints are stored rather than a pre-collapsed margin so the metric
can be redefined — normalisation, which joints, smoothing — without paying
for another full extraction pass.

First line is a meta header: {"meta": {...}}, carrying the pad/conf values
used so a consumer can tell how a cache was built.

A cheaper route to the same consumer exists: `anya_far_telemetry.py` produces
this cache and the far telemetry together in one pass over a band proxy, at
roughly a fifth of the cost of stage 1 + this module.  Its keypoints are not
identical to these (its crops are canonicalised and re-encoded), so
anya_far_serve picks its thresholds from the telemetry's provenance.  This
module remains the reference and is what the full-telemetry path uses.

Run:
    python -m pipeline.extract_far_pose match_anya_telemetry.jsonl [--force]
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional

import cv2
from ultralytics import YOLO

try:                                        # package import (python -m pipeline.x)
    from .videoio import open_video
    from .anya_telemetry import _DEVICE, TELEMETRY_SUFFIX
except ImportError:                         # script import (python pipeline/x.py)
    from videoio import open_video
    from anya_telemetry import _DEVICE, TELEMETRY_SUFFIX

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FAR_POSE_SUFFIX = "_far_pose.jsonl"
FAR_POSE_VERSION = 2   # v2 stores all 17 keypoints; v1 stored a single
                       # pre-collapsed pixel margin, which pinned the metric
                       # definition to extraction time and forced a full
                       # re-run to change it.

N_KP = 17
# COCO-17 keypoint indices.
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


@dataclass
class FarPoseConfig:
    pad_px:        int   = 25     # padding added around the fpr box before crop
    pose_conf:     float = 0.05   # detection floor — crop is tiny, keep permissive


def _load_telemetry(telemetry_path: str):
    """Minimal reader: meta header + records. Duplicated from anya_far_serve
    (not imported) to avoid a circular import — that module imports the pose
    cache helpers below."""
    with open(telemetry_path, "r") as fh:
        first_line = fh.readline()
        meta = {}
        records = []
        try:
            meta = json.loads(first_line).get("meta", {})
        except json.JSONDecodeError:
            pass
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "f" in rec:
                records.append(rec)
    if meta.get("video"):
        meta["_video_path"] = os.path.join(
            os.path.dirname(os.path.abspath(telemetry_path)), meta["video"])
    return meta, records


def far_pose_path_for(telemetry_path: str) -> str:
    if telemetry_path.endswith(TELEMETRY_SUFFIX):
        return telemetry_path[: -len(TELEMETRY_SUFFIX)] + FAR_POSE_SUFFIX
    base, _ = os.path.splitext(telemetry_path)
    return base + FAR_POSE_SUFFIX


def load_far_pose(path: str) -> dict:
    """Returns {frame_idx: record}, each record carrying `k` (flat 17x3
    keypoints in crop pixels) and `bh` (fpr box height), or None if no pose
    was detected on that frame."""
    out = {}
    with open(path, "r") as fh:
        fh.readline()  # meta header
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["f"]] = rec if rec.get("k") else None
    return out


def _keypoints(result) -> Optional[list]:
    """Flat [x,y,conf] * 17 for the highest-confidence detection."""
    if result.keypoints is None or len(result.boxes) == 0:
        return None
    k = result.keypoints.data.cpu().numpy()[0]  # [17,3]
    out = []
    for i in range(N_KP):
        out.extend((round(float(k[i][0]), 1),
                    round(float(k[i][1]), 1),
                    round(float(k[i][2]), 3)))
    return out


def extract_far_pose(telemetry_path: str, force: bool = False,
                     cfg: FarPoseConfig = None, progress_cb=None) -> str:
    cfg = cfg or FarPoseConfig()
    out_path = far_pose_path_for(telemetry_path)
    if not force and os.path.isfile(out_path):
        try:
            with open(out_path) as fh:
                ver = json.loads(fh.readline()).get("meta", {}).get("version", 0)
            if ver == FAR_POSE_VERSION:
                print(f"[FAR-POSE] Using cached: {out_path}  (--force to re-extract)")
                return out_path
        except Exception:
            pass

    meta, records = _load_telemetry(telemetry_path)
    video_path = meta.get("_video_path")
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"far-player video not found next to telemetry: {video_path}")

    pose_model = YOLO(str(os.path.join(_MODELS_DIR, "yolov8n-pose.pt")))
    cap = open_video(video_path, "FAR-POSE")

    by_frame = {r["f"]: r for r in records}
    max_f = max(by_frame) if by_frame else -1

    tmp_path = out_path + ".part"
    n_written = 0
    with open(tmp_path, "w") as fh:
        header = {
            "meta": {
                "version": FAR_POSE_VERSION,
                "source_telemetry": os.path.basename(telemetry_path),
                "fps": meta.get("fps"),
                "pad_px": cfg.pad_px,
                "pose_conf": cfg.pose_conf,
                "n_kp": N_KP,
                "coords": "crop pixels (native resolution, fpr box + pad_px)",
            }
        }
        fh.write(json.dumps(header) + "\n")

        frame_idx = -1
        while cap.isOpened() and frame_idx < max_f:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            rec = by_frame.get(frame_idx)
            if rec is None:
                continue  # source telemetry skipped this frame (stride > 1)

            box = rec.get("fpr")
            kpts, box_h = None, None
            if box:
                x1, y1, x2, y2 = box
                box_h = y2 - y1
                h, w = frame.shape[:2]
                cx1 = max(0, x1 - cfg.pad_px); cy1 = max(0, y1 - cfg.pad_px)
                cx2 = min(w, x2 + cfg.pad_px); cy2 = min(h, y2 + cfg.pad_px)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    result = pose_model(crop, verbose=False, conf=cfg.pose_conf,
                                        device=_DEVICE)[0]
                    kpts = _keypoints(result)

            out = {"f": frame_idx, "t": rec["t"]}
            if kpts:
                out["k"] = kpts
                out["bh"] = box_h
            fh.write(json.dumps(out) + "\n")
            n_written += 1
            if n_written % 1000 == 0:
                print(f"[FAR-POSE] frame {frame_idx}/{max_f} ({n_written} written)")
            if progress_cb is not None and n_written % 30 == 0:
                progress_cb(frame_idx, max_f)

        cap.release()

    os.replace(tmp_path, out_path)
    print(f"[FAR-POSE] Wrote {n_written} records → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline far-player wrist/shoulder pose pass, keyed by frame index.")
    parser.add_argument("telemetry_file", help="Path to _anya_telemetry.jsonl (v2, needs `fpr`)")
    parser.add_argument("--force", action="store_true", help="Re-extract even if a cache exists")
    args = parser.parse_args()
    extract_far_pose(args.telemetry_file, force=args.force)
