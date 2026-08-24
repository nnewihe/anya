"""
perceive.py
===========
The perception passes anya2's detectors read, cached beside the video.

TWO ROIs, BECAUSE ONE DOES NOT WORK
-----------------------------------
The obvious design is one all-persons pose pass over a downscaled proxy, and it
is the design the current pipeline nominally has: `anya_end_telemetry` runs pose
full-frame over a 540p proxy at 15 Hz, with no crop.  Measured on Data/21,
ZERO of 192 sampled detections from that pass project past mid-court.  The far
player is simply not there.

The cause is scale, not the projection: at 540p a player standing on the far
baseline is 20-40 px tall in the 960x540 analysis frame, and yolov8n-pose at
imgsz 960 does not find them.  `walking/extract_pose.py` already documents the
same effect for the near player walking to the ball carts, and works around it
with a `--rescue` pass at 1920 px.

So the far side gets its own ROI: a NATIVE-RESOLUTION CROP of the band around
the far baseline, which keeps source pixels on the subject while paying to
decode only the part of the frame it occupies.  This is the same conclusion
`anya_far_telemetry` reached independently, and it is recorded here as a
property of the perception layer rather than rediscovered by each detector.

    near   540p whole-frame proxy   15 Hz pose   near players, point ends
    far    native-res band crop     15 Hz pose   far players

RATES ARE INPUTS, NOT DECISIONS
-------------------------------
15 Hz is where `anya_end_telemetry` landed and the reasoning holds here: the
walking classifier scores at 15 Hz, and 15 Hz is above Nyquist for the 0.7-4.0
Hz gait cadence band, while 7.5 Hz is not.  But nothing in anya2 depends on it
being 15 -- each detector DECLARES its rate in its `Requirement`, and Phase E
is where those declarations are unioned into whatever passes actually run.
Until then this module just makes the two caches.

Unsampled frames are DROPPED, never held.  A zero-order hold turns a
differentiated feature into spurious events; the npz is written decimated with
`fps` set to the effective rate and `stride` alongside, so every consumer sees
one consistent, slower clip and only the mapping back to source frames needs
the stride.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.videoio import open_video
from pipeline import proxy as P
from pipeline.anya2 import court as C

N_KP = 17
MAX_PERSONS = 8
POSE_CONF = 0.20
BATCH = 16
POSE_FPS = 15.0

_W = Path(__file__).resolve().parents[1] / "models" / "yolov8n-pose.pt"
POSE_MODEL = str(_W) if _W.is_file() else "yolov8n-pose.pt"

NEAR_SUFFIX = "_anya2_near_dets.npz"
FAR_SUFFIX = "_anya2_far_dets.npz"


def dets_path(video, suffix=NEAR_SUFFIX):
    d = os.path.dirname(os.path.abspath(video))
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(d, f"{stem}{suffix}")


def _stride_for(src_fps, pose_fps):
    """Integer stride, never below 1. Returns (stride, effective_fps)."""
    s = max(1, int(round(src_fps / float(pose_fps))))
    return s, src_fps / s


def _pose_pass(video, out_path, stride, imgsz, device, to_analysis=None,
               offset=(0.0, 0.0), scale=(1.0, 1.0), label="POSE", limit=None):
    """All-persons pose over every `stride`-th frame, written decimated.

    `offset`/`scale` map detections back into the 960x540 ANALYSIS frame, which
    is the one frame every consumer and the homography agree on.  A crop proxy
    produces coordinates in the crop; without this they would silently be a
    different coordinate system wearing the same field names.
    """
    from ultralytics import YOLO
    model = YOLO(POSE_MODEL)

    cap = open_video(video, label)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit:
        total = min(total, limit)
    idx = list(range(0, total, stride))
    n = len(idx)

    kp = np.full((n, MAX_PERSONS, N_KP, 3), np.nan, dtype=np.float32)
    bx = np.full((n, MAX_PERSONS, 4), np.nan, dtype=np.float32)
    cf = np.full((n, MAX_PERSONS), np.nan, dtype=np.float32)

    ox, oy = offset
    sx, sy = scale
    t0, empty, done = time.time(), 0, 0
    buf, slots = [], []

    def flush():
        nonlocal empty, done
        if not buf:
            return
        res = model.predict(buf if len(buf) > 1 else buf[0], imgsz=imgsz,
                            conf=POSE_CONF, device=device, classes=[0],
                            verbose=False)
        for j, r in zip(slots, res):
            done += 1
            if r.keypoints is None or len(r.boxes) == 0:
                empty += 1
                continue
            b = r.boxes.xyxy.cpu().numpy()
            c = r.boxes.conf.cpu().numpy()
            k = r.keypoints.data.cpu().numpy()
            o = np.argsort(c)[::-1][:MAX_PERSONS]
            m = len(o)
            b, k = b[o].copy(), k[o].copy()
            b[:, [0, 2]] = b[:, [0, 2]] * sx + ox
            b[:, [1, 3]] = b[:, [1, 3]] * sy + oy
            k[:, :, 0] = k[:, :, 0] * sx + ox
            k[:, :, 1] = k[:, :, 1] * sy + oy
            kp[j, :m], bx[j, :m], cf[j, :m] = k, b, c[o]
        buf.clear()
        slots.clear()

    want = set(idx)
    pos = {f: j for j, f in enumerate(idx)}
    f = 0
    while f < total:
        ok, frame = cap.read()
        if not ok:
            break
        if f in want:
            if to_analysis is not None:
                frame = cv2.resize(frame, to_analysis, interpolation=cv2.INTER_AREA)
            buf.append(frame)
            slots.append(pos[f])
            if len(buf) >= BATCH:
                flush()
                if done % 2000 < BATCH:
                    el = time.time() - t0
                    print(f"  [{label}] {done}/{n}  {done / max(el, 1e-9):.1f} sps"
                          f"  empty {empty / max(done, 1):.1%}", flush=True)
        f += 1
    flush()
    cap.release()

    stride_src = stride * (1 if to_analysis is None else 1)
    np.savez_compressed(out_path, kp=kp, box=bx, conf=cf,
                        fps=np.float64(src_fps / stride),
                        src_fps=np.float64(src_fps),
                        stride=np.float64(stride_src),
                        n_src_frames=np.float64(total))
    per = np.mean(np.sum(np.isfinite(cf), axis=1))
    print(f"[{label}] {out_path}: {n} samples @ {src_fps / stride:.2f} fps, "
          f"{per:.2f} persons/sample, {empty / max(n, 1):.1%} empty, "
          f"{time.time() - t0:.0f}s")
    return out_path


def near(video, device="mps", pose_fps=POSE_FPS, force=False, limit=None):
    """540p whole-frame proxy, all-persons pose at `pose_fps`."""
    out = dets_path(video, NEAR_SUFFIX)
    if os.path.isfile(out) and not force:
        print(f"[near-dets] cached: {out}")
        return out
    # CRF 14, not the default 20: the proxy is shared, and a detector that
    # wants the ball off it needs the ball to have survived the re-encode.
    prox = P.ensure_proxy(video, size=C.ANALYSIS_SIZE, crf=14, label="PROXY540")
    src = cv2.VideoCapture(prox)
    fps = src.get(cv2.CAP_PROP_FPS)
    src.release()
    stride, eff = _stride_for(fps, pose_fps)
    print(f"[near-dets] {fps:.2f} fps source, stride {stride} -> {eff:.2f} fps")
    return _pose_pass(prox, out, stride, imgsz=960, device=device,
                      label="NEAR-DETS", limit=limit)


# How far above the ground the far band must reach, in metres.  A standing
# player is ~1.8 m; a server at the trophy has the racket tip near 2.6 m, and
# THAT is the far-serve signal -- a band that clips the raised arm deletes the
# thing it exists to capture.
FAR_BAND_UP_M = 2.8
FAR_BAND_PAD_M = 4.0      # along the court, either side of the far baseline


def far_band(video, pad_m=FAR_BAND_PAD_M, up_m=FAR_BAND_UP_M):
    """Native-res crop rectangle around the far baseline, in SOURCE pixels.

    Derived from the court homography rather than clicked: the far baseline is
    a known line in court metres, so its image position is already known, and
    asking the user for a second rectangle would be asking for something the
    calibration already contains.

    HEIGHT IS NOT FORESHORTENING.  The homography is a GROUND-plane map, so it
    can place the baseline but cannot project a standing player's head -- there
    is no height in it.  Taking the band from the ground strip alone gives ~55
    px on a 4K clip, which is a third of a far player and none of a raised arm.
    So the vertical extent is derived instead from the image SCALE at that
    depth: the band's own width spans a known number of court metres, giving
    px-per-metre there, and the band grows upward by `up_m` of them.  That
    scales with the camera instead of being a pixel constant tuned on one clip.
    """
    H = C.load_homography(video)
    Hi = np.linalg.inv(H)
    xs = np.linspace(C.X_LO, C.X_HI, 9)
    ys = [C.COURT_L - pad_m, C.COURT_L + pad_m]
    pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    img = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hi).reshape(-1, 2)

    probe = P.probe_video(video)
    W, Hh = int(probe["width"]), int(probe["height"])
    ax, ay = C.ANALYSIS_SIZE
    sx, sy = W / ax, Hh / ay
    x1 = max(0, int(np.floor(img[:, 0].min() * sx)))
    x2 = min(W, int(np.ceil(img[:, 0].max() * sx)))
    y1 = max(0, int(np.floor(img[:, 1].min() * sy)))
    y2 = min(Hh, int(np.ceil(img[:, 1].max() * sy)))

    px_per_m = (x2 - x1) / (C.X_HI - C.X_LO)
    y1 = max(0, int(y1 - round(up_m * px_per_m)))
    return (x1, y1, x2, y2), (sx, sy)


def far(video, device="mps", pose_fps=POSE_FPS, force=False, limit=None):
    """Native-resolution far-baseline band, all-persons pose at `pose_fps`."""
    out = dets_path(video, FAR_SUFFIX)
    if os.path.isfile(out) and not force:
        print(f"[far-dets] cached: {out}")
        return out
    crop, (sx, sy) = far_band(video)
    print(f"[far-dets] band {crop} in source pixels")
    prox = P.ensure_crop_proxy(video, crop, crf=14, label="FARBAND")
    src = cv2.VideoCapture(prox)
    fps = src.get(cv2.CAP_PROP_FPS)
    src.release()
    stride, eff = _stride_for(fps, pose_fps)
    print(f"[far-dets] {fps:.2f} fps source, stride {stride} -> {eff:.2f} fps")
    # Crop pixels are SOURCE pixels; map them back to the analysis frame.
    return _pose_pass(prox, out, stride, imgsz=960, device=device,
                      offset=(crop[0] / sx, crop[1] / sy),
                      scale=(1.0 / sx, 1.0 / sy),
                      label="FAR-DETS", limit=limit)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
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
