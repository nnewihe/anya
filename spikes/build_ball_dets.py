"""
build_ball_dets.py — headless ball-detection cache for the Phase 0 spike.
========================================================================

Folders 21 and 23 have no telemetry JSONL (only folder 68 does), so to measure
ball-onset lag there we need a detection pass.  This runs AnyaBallDetector's
real detection (whole-frame YOLO + exclusion zones) headlessly and caches the
per-frame detections to `<stem>_ball_dets.jsonl` next to the video, so the onset
analysis (serve_onset_lag.py --dets) can re-run for free.

Detections are stored in FULL-RESOLUTION pixels (as _detect_batch emits them);
the analysis applies px_scale = width/960 exactly like ball_detector.process_video.

Run:
    python spikes/build_ball_dets.py /Volumes/Anya/Data/23/snippet.mp4
    python spikes/build_ball_dets.py /Volumes/Anya/Data/21/snippet.mp4 --stride 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "pipeline"))

from ball_detector import AnyaBallDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    stem = os.path.splitext(args.video)[0]
    out_path = f"{stem}_ball_dets.jsonl"

    det = AnyaBallDetector(args.video, imgsz=args.imgsz, stride=args.stride,
                           batch_size=args.batch)

    cap = cv2.VideoCapture(args.video)
    ret, _ = cap.read()
    if not ret:
        print("[ERR] cannot read video")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    det._set_infer_size(width, height)

    # Reuse cached exclusion zones (both folders have them); court corners only
    # if a ball_config is present (folder 23 has one -> homography for gating).
    det._init_exclusion_zones(width, height)
    has_court = det.load_config()
    if has_court:
        det._build_homography()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    with open(out_path, "w") as f:
        f.write(json.dumps({"meta": {"video": os.path.basename(args.video),
                                     "fps": fps, "width": width, "height": height,
                                     "stride": args.stride,
                                     "court_points": det.court_points}}) + "\n")
        batch, slots, idx = [], [], 0
        pending_lines = {}

        def flush():
            if not batch:
                return
            for slot, dets in zip(slots, det._detect_batch(batch)):
                pending_lines[slot] = dets
            batch.clear(); slots.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % args.stride == 0:
                batch.append(frame); slots.append(idx)
                if len(batch) >= args.batch:
                    flush()
            idx += 1
            if idx % 500 == 0:
                # Drain pending in order up to the last flushed frame to keep memory flat.
                print(f"[INFO] {idx} frames read...", flush=True)
        flush()

        for i in range(idx):
            dets = pending_lines.get(i, [])
            f.write(json.dumps([[round(x, 1), round(y, 1), round(c, 3)]
                                for (x, y, c) in dets]) + "\n")

    cap.release()
    print(f"[INFO] wrote {out_path} ({idx} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
