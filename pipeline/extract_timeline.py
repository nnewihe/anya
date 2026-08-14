"""
extract_timeline.py
===================
Near-player pose + bbox extraction over arbitrary frame spans of a clip.

Two modes:

  --mode rallies  (default)  Spans covering every rally in ground truth — NEAR
      AND FAR — expanded by --margin_sec on each side and merged. This is the
      training-data pass. It exists because `energy_telemetry_cache.json` was
      built from `_near_rallies`, so no bbox exists for far-serve rallies at
      all; they cannot be trained on until this runs.

  --mode full                Every frame. Needed for event-level evaluation,
      where false-fires-per-live-minute requires continuous coverage.

Why this pass is needed at all, beyond far rallies: the telemetry cache stops
6s after each rally end, so every dead sample in the original training set is
the 6 seconds following a point — the most live-looking dead there is. Measured
on those spans, mean bbox speed 1s AFTER the point ends (0.123) is HIGHER than
1s before it (0.091); separation only appears ~3s out. Genuine deep-dead
behaviour is simply absent from the cache.

Frame-rate normalization: clips run 29.97 / 59.94 / 119.88 fps, and 30fps is the
target rate, so --target_fps sets pose_stride = round(fps / target_fps) and the
skipped frames are linearly interpolated back within each span. A 119.88fps clip
costs a quarter as much with no information lost at the target rate.

Output <clip>/timeline_cache.npz:
    pose      [N, 51]  17 COCO keypoints normalized to the near bbox (NaN = none)
    bbox      [N,  4]  near-player box in 960x540 analysis coords (NaN = none)
    frame_idx [N]      ABSOLUTE frame number of each row — coverage is not
                       contiguous in rallies mode, so consumers must use this
                       rather than assuming row i is frame start+i.

Usage:
    python pipeline/extract_timeline.py --clips all --mode rallies --target_fps 30
    python pipeline/extract_timeline.py --clips 21 22 --mode full
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
from optimize_energy import (_video_path, _load_corners, _detect_near_player,
                             ANALYSIS_SIZE, PLAYER_STRIDE)
from anya_near_serve_archive import PointStartSystem
from extract_pose import _select_near_pose, _normalize, N_KP
from parse_ground_truth import load_rallies, gt_path, _fps_for

TIMELINE_NPZ = "timeline_cache.npz"


def merge_spans(spans, lo, hi):
    """Clip to [lo, hi], drop empties, merge overlapping/adjacent spans."""
    cl = sorted((max(lo, a), min(hi, b)) for a, b in spans)
    cl = [(a, b) for a, b in cl if b >= a]
    out = []
    for a, b in cl:
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def rally_spans(clip_dir, fps, total, margin_sec):
    """Spans around EVERY rally (near and far), expanded and merged.

    Far rallies are included deliberately: the near player is the receiver, and
    is genuinely active, so far rallies are live time the model must learn.
    """
    m = int(round(margin_sec * fps))
    r = load_rallies(clip_dir, fps)
    return merge_spans([(x["start"] - m, x["end"] + m) for x in r], 0, total - 1)


def _interp_gaps(arr, max_gap):
    """Linearly interpolate NaN runs no longer than max_gap, in place-ish.

    Used to refill frames skipped by pose_stride. Longer runs are genuine
    detection failures and stay NaN so they are masked downstream.
    """
    if max_gap < 1:
        return arr
    valid = ~np.isnan(arr[:, 0])
    if valid.sum() < 2:
        return arr
    idx = np.flatnonzero(valid)
    out = arr.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for d in range(arr.shape[1]):
                out[a + 1:b, d] = np.interp(np.arange(a + 1, b), [a, b], [arr[a, d], arr[b, d]])
    return out


def extract_clip(clip_dir, pose_model, player_model, device, args):
    name = os.path.basename(clip_dir)
    out_p = os.path.join(clip_dir, TIMELINE_NPZ)
    if os.path.isfile(out_p) and not args.rescan:
        print(f"[timeline] {name}: cached")
        return

    vid = _video_path(clip_dir)
    if vid is None:
        print(f"[timeline] {name}: no video — skip")
        return
    if args.mode == "rallies" and gt_path(clip_dir) is None:
        print(f"[timeline] {name}: no ground truth — skip")
        return
    corners = _load_corners(vid)

    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.mode == "rallies":
        spans = rally_spans(clip_dir, fps, total, args.margin_sec)
    else:
        a = max(0, args.start)
        b = min(total - 1, args.end if args.end >= 0 else total - 1)
        if args.max_frames > 0:
            b = min(b, a + args.max_frames - 1)
        spans = merge_spans([(a, b)], 0, total - 1)
    if not spans:
        print(f"[timeline] {name}: no spans — skip")
        return

    pose_stride = args.pose_stride
    if args.target_fps > 0:
        pose_stride = max(1, int(round(fps / args.target_fps)))
    n_cov = sum(b - a + 1 for a, b in spans)

    geom = PointStartSystem(corners, ANALYSIS_SIZE[0], ANALYSIS_SIZE[1],
                            fps=int(round(fps)))
    print(f"[timeline] {name}: {len(spans)} spans, {n_cov}/{total} frames "
          f"({n_cov/total:.0%}) fps={fps:.2f} pose_stride={pose_stride}")

    pose_parts, bbox_parts, idx_parts = [], [], []
    done = 0
    for a, b in spans:
        T = b - a + 1
        p = np.full((T, N_KP * 3), np.nan, dtype=np.float32)
        bb = np.full((T, 4), np.nan, dtype=np.float32)
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        held = None                      # reset per span: no carry across a seek
        for t in range(T):
            ok, frame = cap.read()
            if not ok:
                p, bb, T = p[:t], bb[:t], t
                break
            frame = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
            # Same detect-and-hold cadence as the production telemetry pass, so
            # the bbox track has the identical stair-step global_features undoes.
            if t % args.player_stride == 0:
                held = _detect_near_player(frame, player_model, device, geom)
            if held is not None:
                bb[t] = held
                if t % pose_stride == 0:
                    res = pose_model.predict(frame, imgsz=640, device=device, verbose=False)[0]
                    kp = _select_near_pose(res, held)
                    if kp is not None:
                        p[t] = _normalize(kp, held)
        if T > 0:
            # Refill only the frames pose_stride skipped, not real dropouts.
            pose_parts.append(_interp_gaps(p, pose_stride - 1) if pose_stride > 1 else p)
            bbox_parts.append(bb)
            idx_parts.append(np.arange(a, a + T, dtype=np.int64))
        done += T
        print(f"    span {a}-{b}  {done}/{n_cov}")

    cap.release()
    if not pose_parts:
        print(f"[timeline] {name}: nothing decoded — skip")
        return
    pose = np.concatenate(pose_parts); bbox = np.concatenate(bbox_parts)
    frame_idx = np.concatenate(idx_parts)
    np.savez_compressed(out_p, pose=pose, bbox=bbox, frame_idx=frame_idx,
                        fps=np.array(fps), start=np.array(int(frame_idx[0])),
                        n_frames=np.array(len(frame_idx)),
                        total_frames=np.array(total),
                        mode=np.array(args.mode),
                        pose_stride=np.array(pose_stride),
                        player_stride=np.array(args.player_stride))
    N = len(frame_idx)
    got = int(np.sum(~np.isnan(pose[:, 0]))); bgot = int(np.sum(~np.isnan(bbox[:, 0])))
    print(f"[timeline] {name}: pose {got}/{N} ({got/N:.0%})  "
          f"bbox {bgot}/{N} ({bgot/N:.0%})  -> {TIMELINE_NPZ}")


THROUGHPUT_FPS = 12.0      # measured pose forwards/sec on this box (MPS, yolov8n-pose)


def plan(clips, args):
    """Dry-run the frame budget so compute is a decision, not a surprise."""
    import cv2
    print(f"{'clip':>5} {'fps':>7} {'frames':>8} {'spans':>6} {'covered':>9} "
          f"{'cov%':>6} {'stride':>7} {'forwards':>9} {'est':>7}")
    tot_fwd = 0
    for c in clips:
        d = os.path.join(args.data_root, c)
        vid = _video_path(d)
        if vid is None:
            continue
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        spans = (rally_spans(d, fps, total, args.margin_sec) if args.mode == "rallies"
                 else [(0, total - 1)])
        cov = sum(b - a + 1 for a, b in spans)
        stride = (max(1, int(round(fps / args.target_fps))) if args.target_fps > 0
                  else args.pose_stride)
        fwd = cov // stride
        tot_fwd += fwd
        print(f"{c:>5} {fps:>7.2f} {total:>8} {len(spans):>6} {cov:>9} "
              f"{cov/max(total,1)*100:>5.0f}% {stride:>7} {fwd:>9} "
              f"{fwd/THROUGHPUT_FPS/60:>6.0f}m")
    print(f"\n[plan] {tot_fwd} pose forwards total ~= "
          f"{tot_fwd/THROUGHPUT_FPS/3600:.1f} hours at {THROUGHPUT_FPS:.0f}/s")
    print("[plan] tune with --margin_sec (rallies mode) and --target_fps")


def main():
    ap = argparse.ArgumentParser(description="Near-player pose+bbox over rally spans or full video")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--clips", nargs="+", required=True,
                    help="clip names, or 'all' for every clip with ground truth")
    ap.add_argument("--mode", choices=("rallies", "full"), default="rallies")
    ap.add_argument("--margin_sec", type=float, default=6.0,
                    help="rallies mode: dead context kept on each side of a rally. "
                         "Gaps between points run ~20-25s, so 2*margin approaching "
                         "that merges every span and the pass degenerates into a "
                         "full-video decode — 6s keeps deep-dead samples (band 1.5s "
                         "+ guard 2s) while leaving real gaps uncovered.")
    ap.add_argument("--target_fps", type=float, default=30.0,
                    help="derive pose_stride from clip fps (0 disables)")
    ap.add_argument("--pose_model", default="yolov8n-pose.pt")
    ap.add_argument("--player_model", default="yolov8n.pt")
    ap.add_argument("--player_stride", type=int, default=PLAYER_STRIDE)
    ap.add_argument("--pose_stride", type=int, default=1,
                    help="used only when --target_fps 0")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--rescan", action="store_true")
    ap.add_argument("--plan", action="store_true",
                    help="report the frame/time budget per clip and exit — no detection")
    ap.add_argument("--allow_derived", action="store_true",
                    help="also accept derived_ground_truth.json (model-inferred labels)")
    args = ap.parse_args()

    import parse_ground_truth as _pgt
    _pgt.ALLOW_DERIVED = args.allow_derived

    if len(args.clips) == 1 and args.clips[0] == "all":
        clips = [d for d in sorted(os.listdir(args.data_root))
                 if os.path.isdir(os.path.join(args.data_root, d))
                 and gt_path(os.path.join(args.data_root, d))]
    else:
        clips = args.clips

    if args.plan:
        plan(clips, args)
        return

    device = ('mps' if torch.backends.mps.is_available()
              else 'cuda' if torch.cuda.is_available() else 'cpu')
    here = os.path.dirname(os.path.abspath(__file__))
    def _m(p):
        return p if os.path.isfile(p) else os.path.join(here, p)
    print(f"[init] device={device}  {len(clips)} clips  mode={args.mode}")
    pose_model = YOLO(_m(args.pose_model))
    player_model = YOLO(_m(args.player_model))

    for c in clips:
        d = os.path.join(args.data_root, c)
        try:
            extract_clip(d, pose_model, player_model, device, args)
        except Exception as e:
            print(f"[WARN] {c}: {e}")


if __name__ == "__main__":
    main()
