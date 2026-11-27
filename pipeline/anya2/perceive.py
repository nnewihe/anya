"""
perceive.py
===========
The two pose passes every detector reads.  This is most of a run's cost.

One pass cannot see both ends of the court: at 540p a far-baseline player is
20-40 px tall and the pose model does not find them.  So:

    near()   540p whole-frame proxy          all persons, 15 Hz
    far()    native-resolution crop of a     all persons, 15 Hz
             band around the far baseline

Both passes MUST run at the same rate.  tracks.py merges them sample for
sample and refuses if they disagree.

Frames that are not sampled are dropped, not repeated.  Each npz records the
effective `fps` and the `stride` back to source frames.

Output `<stem>_anya2_near_dets.npz` / `<stem>_anya2_far_dets.npz`:
    kp    [N, MAX_PERSONS, 17, 3]  keypoints (x, y, conf), ANALYSIS_SIZE px
    box   [N, MAX_PERSONS, 4]      xyxy, ANALYSIS_SIZE px
    conf  [N, MAX_PERSONS]         person confidence; NaN = empty slot
    fps, src_fps, stride, n_src_frames, crop (far pass only; else zeros)

Evidence and history: docs/anya2/NOTES.md, "perceive.py".
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline import cancel
from pipeline import proxy
from pipeline import workdir
from pipeline.anya2 import court
from pipeline.utilities import assert_decode_complete
from pipeline.videoio import open_video

N_KEYPOINTS = 17
MAX_PERSONS = 8            # people kept per frame, highest confidence first
POSE_CONF = 0.20           # pose model's detection threshold
BATCH = 16                 # frames per model call
POSE_FPS = 15.0            # default sample rate for both passes

_bundled_weights = Path(__file__).resolve().parents[1] / "models" / "yolov8n-pose.pt"
POSE_MODEL = str(_bundled_weights) if _bundled_weights.is_file() else "yolov8n-pose.pt"

# Model input sizes.  Do NOT tune these on a throughput benchmark: only an
# end-to-end recall number can price them (see NOTES).
FAR_IMGSZ = 960            # the far player is at the model's limit; 768 lost recall
NEAR_IMGSZ = 640           # near players are large; 640 is as accurate and cheaper

# The far band, in court metres around the far baseline.
FAR_BAND_UP_M = 2.8        # how high above the ground: a raised racket at the trophy
FAR_BAND_PAD_M = 4.0       # along the court, each side of the far baseline
FAR_BAND_EDGE_PX = 2.0     # a box this close to the crop's bottom edge is cut off

NEAR_SUFFIX = "_anya2_near_dets.npz"
FAR_SUFFIX = "_anya2_far_dets.npz"


def dets_path(video, suffix=NEAR_SUFFIX):
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(workdir.artifact_dir(video), f"{stem}{suffix}")


# ═════════════════════════════════════════════════════════════════════════
# The two passes
# ═════════════════════════════════════════════════════════════════════════

def near(video, device="mps", pose_fps=POSE_FPS, force=False, limit=None, imgsz=None):
    """All-persons pose over the 540p whole-frame proxy.  Returns the npz path.

    `imgsz` overrides NEAR_IMGSZ.  The cache does not record it: pass
    force=True after changing it.
    """
    out = dets_path(video, NEAR_SUFFIX)
    if os.path.isfile(out) and not force:
        print(f"[near-dets] cached: {out}")
        return out
    # crf 14 (not 20): the proxy is shared and must keep small details intact.
    proxy_path = proxy.ensure_proxy(video, size=court.ANALYSIS_SIZE, crf=14, label="PROXY540")
    proxy_fps = _video_fps(proxy_path)
    stride, effective_fps = _stride_for(proxy_fps, pose_fps)
    print(f"[near-dets] {proxy_fps:.2f} fps source, stride {stride} -> {effective_fps:.2f} fps")
    return _pose_pass(proxy_path, out, stride, imgsz=NEAR_IMGSZ if imgsz is None else imgsz, device=device,
                      label="NEAR-DETS", limit=limit)


def far(video, device="mps", pose_fps=POSE_FPS, force=False, limit=None, imgsz=None):
    """All-persons pose over a native-resolution crop of the far baseline band.

    `imgsz` overrides FAR_IMGSZ.  The cache does not record it: pass
    force=True after changing it.
    """
    out = dets_path(video, FAR_SUFFIX)
    crop, (scale_x, scale_y) = far_band(video)
    if os.path.isfile(out) and not force and _cached_far_pass_still_valid(video, out, crop):
        print(f"[far-dets] cached: {out}")
        return out
    print(f"[far-dets] band {crop} in source pixels")
    band_path = proxy.ensure_crop_proxy(video, crop, crf=14, label="FARBAND")
    band_fps = _video_fps(band_path)
    stride, effective_fps = _stride_for(band_fps, pose_fps)
    print(f"[far-dets] {band_fps:.2f} fps source, stride {stride} -> {effective_fps:.2f} fps")
    # The crop is in SOURCE pixels; detections are mapped back to the analysis frame.
    return _pose_pass(band_path, out, stride, imgsz=FAR_IMGSZ if imgsz is None else imgsz, device=device,
                      offset=(crop[0] / scale_x, crop[1] / scale_y),
                      scale=(1.0 / scale_x, 1.0 / scale_y),
                      label="FAR-DETS", limit=limit,
                      drop_bottom_edge=True, crop=crop)


def _cached_far_pass_still_valid(video, dets_npz, crop):
    """Was the cached far pass run over the band we would use now?

    The band widens when the camera track finds a jostle, and a pass over the
    old band silently misses the far player.  A file too old to record its
    crop is trusted only when the camera never moved (then the band cannot
    have changed).
    """
    from pipeline.anya2 import camera
    cached_crop = _crop_of(dets_npz)
    if cached_crop == tuple(crop) or (cached_crop is None and camera.load(video).is_identity):
        return True
    print(f"[far-dets] cached pass covers {cached_crop}, band is now "
          f"{tuple(crop)} -- re-running")
    return False


def _crop_of(dets_npz):
    """The crop a cached pose pass recorded, or None for an older file."""
    try:
        with np.load(dets_npz, allow_pickle=False) as z:
            if "crop" not in z:
                return None
            return tuple(int(v) for v in z["crop"])
    except Exception:
        return None


def _video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def _stride_for(src_fps, pose_fps):
    """(integer stride >= 1, the effective rate it gives)."""
    stride = max(1, int(round(src_fps / float(pose_fps))))
    return stride, src_fps / stride


# ═════════════════════════════════════════════════════════════════════════
# The far band
# ═════════════════════════════════════════════════════════════════════════

def far_band(video, pad_m=FAR_BAND_PAD_M, up_m=FAR_BAND_UP_M):
    """Crop rectangle (x1, y1, x2, y2) around the far baseline, in SOURCE px,
    plus (scale_x, scale_y) = source px per analysis px.

      1. Take a grid of court points along the far baseline +/- `pad_m`, and
         project them into the image with the clicked homography.
      2. Add where those points land under every camera-track warp, so the
         fixed crop covers everywhere the camera went (`_band_union`).
      3. Grow the top edge by `up_m` metres, using px-per-metre measured across
         the band's width (the ground homography carries no height).
    """
    H = court.load_homography(video)
    H_inv = np.linalg.inv(H)
    xs = np.linspace(court.X_LO, court.X_HI, 9)
    ys = [court.COURT_L - pad_m, court.COURT_L + pad_m]
    court_points = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    ref_image_points = cv2.perspectiveTransform(court_points.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    all_image_points = _band_union(video, ref_image_points)

    probe = proxy.probe_video(video)
    src_w, src_h = int(probe["width"]), int(probe["height"])
    analysis_w, analysis_h = court.ANALYSIS_SIZE
    scale_x, scale_y = src_w / analysis_w, src_h / analysis_h
    x1 = max(0, int(np.floor(all_image_points[:, 0].min() * scale_x)))
    x2 = min(src_w, int(np.ceil(all_image_points[:, 0].max() * scale_x)))
    y1 = max(0, int(np.floor(all_image_points[:, 1].min() * scale_y)))
    y2 = min(src_h, int(np.ceil(all_image_points[:, 1].max() * scale_y)))

    # From the REFERENCE band's width, not the union's (which camera motion inflates).
    px_per_m = (((ref_image_points[:, 0].max() - ref_image_points[:, 0].min()) * scale_x)
                / (court.X_HI - court.X_LO))
    y1 = max(0, int(y1 - round(up_m * px_per_m)))
    return (x1, y1, x2, y2), (scale_x, scale_y)


def _band_union(video, ref_points):
    """`ref_points` plus where they land under every camera-track warp. [M, 2]."""
    from pipeline.anya2 import camera
    track = camera.load(video)
    if track.is_identity:
        return ref_points
    W_inv = np.linalg.inv(track.W)
    stacked = [ref_points]
    as_cv = ref_points.reshape(-1, 1, 2)
    for i in range(len(W_inv)):
        stacked.append(cv2.perspectiveTransform(as_cv, W_inv[i]).reshape(-1, 2))
    return np.concatenate(stacked, axis=0)


# ═════════════════════════════════════════════════════════════════════════
# One pose pass
# ═════════════════════════════════════════════════════════════════════════

def _pose_pass(video, out_path, stride, imgsz, device, to_analysis=None,
               offset=(0.0, 0.0), scale=(1.0, 1.0), label="POSE", limit=None,
               drop_bottom_edge=False, edge_px=FAR_BAND_EDGE_PX, crop=None):
    """All-persons pose on every `stride`-th frame of `video`, written decimated.

    offset/scale  map detections from this video's pixels into the 960x540
                  analysis frame (needed for the far crop).
    drop_bottom_edge  discard people whose box bottom touches the crop's lower
                  edge: they are cut off by the crop, so the box bottom is not
                  their feet and would project somewhere they never stood.
    """
    from pipeline.anya2 import pose_backend as PB

    cap = open_video(video, label)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_hw = ((to_analysis[1], to_analysis[0]) if to_analysis is not None else
                (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    # torch (default), or an NCNN/ONNX export -- see pose_backend.py for why an
    # NCNN export must match this frame's letterbox scale and runs batch-1.
    model = PB.load(frame_hw, imgsz)
    print(f"[{label}] pose model: {model}")
    if limit:
        total = min(total, limit)
    sampled_frames = list(range(0, total, stride))
    n_samples = len(sampled_frames)
    row_of_frame = {f: row for row, f in enumerate(sampled_frames)}

    keypoints = np.full((n_samples, MAX_PERSONS, N_KEYPOINTS, 3), np.nan, dtype=np.float32)
    boxes = np.full((n_samples, MAX_PERSONS, 4), np.nan, dtype=np.float32)
    confidences = np.full((n_samples, MAX_PERSONS), np.nan, dtype=np.float32)
    crop_height = (crop[3] - crop[1]) if crop is not None else None

    started = time.time()
    progress = {"done": 0, "empty": 0}
    batch_frames, batch_rows = [], []

    def run_batch():
        if not batch_frames:
            return
        results = model.predict(list(batch_frames), conf=POSE_CONF, device=device,
                                classes=[0], verbose=False)
        if len(results) != len(batch_rows):
            # zip() below would otherwise truncate silently and leave the
            # remaining frames looking like frames with nobody in them.
            raise RuntimeError(f"[{label}] {model} returned {len(results)} results "
                               f"for {len(batch_rows)} frames")
        for row, result in zip(batch_rows, results):
            progress["done"] += 1
            people = _people_in_result(result, drop_bottom_edge, crop_height, edge_px)
            if people is None:
                progress["empty"] += 1
                continue
            frame_boxes, frame_kps, frame_confs = _to_analysis_frame(*people, offset, scale)
            m = min(len(frame_boxes), MAX_PERSONS)
            keypoints[row, :m] = frame_kps[:m]
            boxes[row, :m] = frame_boxes[:m]
            confidences[row, :m] = frame_confs[:m]
        batch_frames.clear()
        batch_rows.clear()

    wanted = set(sampled_frames)
    frame = 0
    while frame < total:
        # grab() for every frame, retrieve() only for the sampled ones: a
        # skipped frame still has to be decoded (it is a reference for the
        # next), but not colour-converted and copied out.  Half the frames at
        # 30 fps / stride 2, and it matters on a CPU that is also running pose.
        if not cap.grab():
            break
        if frame in wanted:
            ok, image = cap.retrieve()
            if not ok:
                break
            if to_analysis is not None:
                image = cv2.resize(image, to_analysis, interpolation=cv2.INTER_AREA)
            batch_frames.append(image)
            batch_rows.append(row_of_frame[frame])
            if len(batch_frames) >= BATCH:
                cancel.check()      # where a Cancel lands during the long passes
                run_batch()
                done = progress["done"]
                if done % 2000 < BATCH:
                    elapsed = time.time() - started
                    print(f"  [{label}] {done}/{n_samples}  {done / max(elapsed, 1e-9):.1f} sps"
                          f"  empty {progress['empty'] / max(done, 1):.1%}", flush=True)
        frame += 1
    run_batch()
    cap.release()

    # A decode failure mid-file looks like a clean end of file to the loop
    # above; without this check the rest of the match would silently read as
    # "nobody there".
    assert_decode_complete(label, video, frame - 1, total - 1, src_fps)

    np.savez_compressed(out_path, kp=keypoints, box=boxes, conf=confidences,
                        fps=np.float64(src_fps / stride),
                        src_fps=np.float64(src_fps),
                        stride=np.float64(stride),
                        n_src_frames=np.float64(total),
                        pose_backend=np.str_(model.kind),
                        # The crop travels with the detections so far() can tell
                        # whether a cached pass still covers the right band.
                        crop=np.asarray(crop if crop is not None else (0, 0, 0, 0),
                                        dtype=np.int32))
    persons_per_sample = np.mean(np.sum(np.isfinite(confidences), axis=1))
    print(f"[{label}] {out_path}: {n_samples} samples @ {src_fps / stride:.2f} fps, "
          f"{persons_per_sample:.2f} persons/sample, "
          f"{progress['empty'] / max(n_samples, 1):.1%} empty, "
          f"{time.time() - started:.0f}s")
    return out_path


def _people_in_result(result, drop_bottom_edge, crop_height, edge_px):
    """(boxes, keypoints, confidences) of one frame's people, most confident
    first, or None if there is nobody (left)."""
    if result.keypoints is None or len(result.boxes) == 0:
        return None
    frame_boxes = result.boxes.xyxy.cpu().numpy()
    frame_confs = result.boxes.conf.cpu().numpy()
    frame_kps = result.keypoints.data.cpu().numpy()
    order = np.argsort(frame_confs)[::-1][:MAX_PERSONS]
    frame_boxes, frame_kps = frame_boxes[order].copy(), frame_kps[order].copy()
    frame_confs = frame_confs[order]
    if drop_bottom_edge and crop_height is not None:
        keep = frame_boxes[:, 3] < crop_height - edge_px
        frame_boxes, frame_kps, frame_confs = frame_boxes[keep], frame_kps[keep], frame_confs[keep]
        if len(frame_boxes) == 0:
            return None
    return frame_boxes, frame_kps, frame_confs


def _to_analysis_frame(frame_boxes, frame_kps, frame_confs, offset, scale):
    """Map one frame's detections from this video's pixels to analysis pixels."""
    offset_x, offset_y = offset
    scale_x, scale_y = scale
    frame_boxes[:, [0, 2]] = frame_boxes[:, [0, 2]] * scale_x + offset_x
    frame_boxes[:, [1, 3]] = frame_boxes[:, [1, 3]] * scale_y + offset_y
    frame_kps[:, :, 0] = frame_kps[:, :, 0] * scale_x + offset_x
    frame_kps[:, :, 1] = frame_kps[:, :, 1] * scale_y + offset_y
    return frame_boxes, frame_kps, frame_confs


def main():
    ap = argparse.ArgumentParser(description="The perception passes anya2's "
                                 "detectors read, cached beside the video.")
    ap.add_argument("video")
    ap.add_argument("--roi", choices=["near", "far", "both"], default="near")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--pose-fps", type=float, default=POSE_FPS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.roi in ("near", "both"):
        near(a.video, a.device, a.pose_fps, a.force, a.limit)
    if a.roi in ("far", "both"):
        far(a.video, a.device, a.pose_fps, a.force, a.limit)


if __name__ == "__main__":
    main()
