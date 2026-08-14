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

The pass is GATED (v3): pose runs only inside the windows where anya_far_serve
is armed, widened by a lead-in so the trailing median has a settled baseline.
A serve is reported only on a frame that is both armed and carries a raise
edge, so pose on an unarmed frame was always computed and then discarded.
See `_candidate_frames`, which drives the detector's own arming code rather
than reimplementing the condition.

First line is a meta header: {"meta": {...}}, carrying the pad/conf values
used so a consumer can tell how a cache was built.

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
    from .anya_telemetry import _DEVICE, TELEMETRY_SUFFIX
except ImportError:                         # script import (python pipeline/x.py)
    from anya_telemetry import _DEVICE, TELEMETRY_SUFFIX

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

FAR_POSE_SUFFIX = "_far_pose.jsonl"
FAR_POSE_VERSION = 3   # v2 stores all 17 keypoints; v1 stored a single
                       # pre-collapsed pixel margin, which pinned the metric
                       # definition to extraction time and forced a full
                       # re-run to change it.  v3 runs the model only on
                       # armed windows (see _candidate_frames); the record
                       # schema is unchanged, so a consumer cannot tell a
                       # skipped frame from an undetected one — which is
                       # exactly right, since neither can trigger a serve.
                       # Verified serve-list-identical to a dense v2 pass.

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
    pose_imgsz:    int   = 640    # Explicitly the ultralytics default this pass
                                  # used to get implicitly.  DO NOT lower it
                                  # without re-sweeping the gate.
                                  #
                                  # It looks like free compute — the crop is an
                                  # fpr box a few hundred px tall, so 640
                                  # letterboxes it up and 256 costs half as much
                                  # (measured 18.3 -> 8.8 ms/call).  It is not
                                  # free.  Scored end-to-end on Data/21, 256
                                  # keeps recall (all 7 serves still cross
                                  # RAISE_RATIO, with HIGHER peaks — less
                                  # interpolation blur) but moves the operating
                                  # point: 72 vs 58 frames above threshold, and
                                  # the reported serve list went from
                                  # [29.8, 74.6, 238.0, 241.2, 250.5, 322.3, 375.5]
                                  # to only 4 of those 7 plus 3 new ones.
                                  #
                                  # RAISE_RATIO=0.35 is the F1 peak of a 72-point
                                  # sweep taken against 640-upscaled keypoints,
                                  # so the input scale is effectively a tuned
                                  # hyperparameter of the gate, not a knob that
                                  # can be turned on its own.  Lowering it is a
                                  # real speedup available for the taking, but
                                  # only jointly with a fresh RAISE_RATIO sweep.

    # Armed-window gating.  The hand-raise gate can only produce a serve on a
    # frame where anya_far_serve is ARMED (see FarServeDetector.process_frame:
    # `if armed and raised`), and arming is decided entirely by the far
    # player's world-x track, which telemetry already holds.  Pose on any other
    # frame is computed, stored, and then provably discarded.
    gate_armed:    bool  = True
    gate_lead_s:   float = 1.0    # pose this long BEFORE each armed window, so
                                  # the trailing median (SMOOTH_WINDOW_S) and
                                  # the rising-edge test both have a settled
                                  # resting baseline before the window opens
    gate_tail_s:   float = 0.5


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


def _candidate_frames(meta, records, cfg: FarPoseConfig) -> Optional[set]:
    """Frame indices where a pose sample could still change the outcome.

    anya_far_serve reports a serve only on a frame that is both ARMED and
    carries a hand-raise rising edge.  Arming depends only on the far player's
    world-x track (`fprw`), which is already in telemetry, so the armed set is
    computable here with no model at all — and pose outside it cannot trigger
    anything.

    The detector's own `_update_arming` is driven directly rather than
    reimplemented, so the gate cannot drift away from the thing it is gating.
    Each armed run is then widened by gate_lead_s / gate_tail_s: the raise test
    is a rising edge on a trailing median, so it needs a settled below-threshold
    baseline in hand before the window opens or a serve at the very start of a
    window could be missed.

    Returns None to mean "no gate — do every frame", which is what a telemetry
    file with no usable far-player track gets.
    """
    if not cfg.gate_armed or not records:
        return None

    # Imported inside the function: anya_far_serve imports THIS module at
    # module level, so a top-level import here would be a cycle.
    from .anya_far_serve import (FarServeDetector, FarServeDetectorConfig,
                                 pick_far_player_source)

    fcfg = FarServeDetectorConfig()
    det = FarServeDetector(fcfg)
    det.fp_key = pick_far_player_source(records)
    if not any(r.get(det.fp_key) for r in records):
        print("[FAR-POSE] no far-player track in telemetry — gating disabled, "
              "running every frame")
        return None

    armed_idx = []
    for i, r in enumerate(records):
        t = r["t"]
        det._update_arming(r, t)
        if (t - det.last_armed_t) <= fcfg.ARM_TO_TRACE_S:
            armed_idx.append(i)
    if not armed_idx:
        print("[FAR-POSE] far player never armed — gating disabled, "
              "running every frame")
        return None

    fps = float(meta.get("fps") or 30.0)
    lead = max(1, int(round(cfg.gate_lead_s * fps)))
    tail = max(1, int(round(cfg.gate_tail_s * fps)))

    # Expand each contiguous armed run, then merge the overlaps.
    runs, start, prev = [], armed_idx[0], armed_idx[0]
    for i in armed_idx[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))

    keep = set()
    n = len(records)
    for a, b in runs:
        for i in range(max(0, a - lead), min(n, b + tail + 1)):
            keep.add(records[i]["f"])
    return keep


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
    cap = cv2.VideoCapture(video_path)

    by_frame = {r["f"]: r for r in records}
    max_f = max(by_frame) if by_frame else -1

    candidates = _candidate_frames(meta, records, cfg)
    if candidates is not None:
        print(f"[FAR-POSE] armed-window gate: {len(candidates)}/{len(records)} "
              f"frame(s) ({len(candidates) / max(1, len(records)):.1%}) need pose")

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
                "pose_imgsz": cfg.pose_imgsz,
                "gated": candidates is not None,
                "n_gated_frames": len(candidates) if candidates is not None else None,
            }
        }
        fh.write(json.dumps(header) + "\n")

        frame_idx = -1
        n_inferred = 0
        while cap.isOpened() and frame_idx < max_f:
            # grab() advances the decoder without converting the frame into a
            # numpy array; retrieve() pays that cost only for frames we keep.
            # On a gated run that is most of the decode saved outright.
            if not cap.grab():
                break
            frame_idx += 1
            rec = by_frame.get(frame_idx)
            if rec is None:
                continue  # source telemetry skipped this frame (stride > 1)

            box = rec.get("fpr")
            wanted = box and (candidates is None or frame_idx in candidates)

            kpts, box_h = None, None
            if wanted:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                x1, y1, x2, y2 = box
                box_h = y2 - y1
                h, w = frame.shape[:2]
                cx1 = max(0, x1 - cfg.pad_px); cy1 = max(0, y1 - cfg.pad_px)
                cx2 = min(w, x2 + cfg.pad_px); cy2 = min(h, y2 + cfg.pad_px)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    result = pose_model(crop, verbose=False, conf=cfg.pose_conf,
                                        imgsz=cfg.pose_imgsz, device=_DEVICE)[0]
                    kpts = _keypoints(result)
                    n_inferred += 1

            out = {"f": frame_idx, "t": rec["t"]}
            if kpts:
                out["k"] = kpts
                out["bh"] = box_h
            fh.write(json.dumps(out) + "\n")
            n_written += 1
            if n_written % 1000 == 0:
                print(f"[FAR-POSE] frame {frame_idx}/{max_f} "
                      f"({n_written} written, {n_inferred} inferred)")
            if progress_cb is not None and n_written % 30 == 0:
                progress_cb(frame_idx, max_f)

        cap.release()

    os.replace(tmp_path, out_path)
    print(f"[FAR-POSE] Wrote {n_written} records ({n_inferred} pose calls) "
          f"→ {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline far-player wrist/shoulder pose pass, keyed by frame index.")
    parser.add_argument("telemetry_file", help="Path to _anya_telemetry.jsonl (v2, needs `fpr`)")
    parser.add_argument("--force", action="store_true", help="Re-extract even if a cache exists")
    args = parser.parse_args()
    extract_far_pose(args.telemetry_file, force=args.force)
