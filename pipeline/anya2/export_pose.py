"""
export_pose.py
==============
Export the pose model for `pose_backend` (NCNN at the exact rectangular shapes
anya2 infers at, or one dynamic ONNX), and prove the export sees what PyTorch
sees before anything is allowed to run on it.

    python -m pipeline.anya2.export_pose export --backend ncnn --video V.mp4
    python -m pipeline.anya2.export_pose export --backend ncnn --shape 384x640
    python -m pipeline.anya2.export_pose export --backend onnx
    python -m pipeline.anya2.export_pose parity --backend ncnn V.mp4

`--video` needs the video's court calibration: the far shape comes from its
far band.  The same happens automatically at `site save --export ncnn`.

WHY A PARITY GATE AND NOT A BENCHMARK
-------------------------------------
Both passes feed thresholds tuned on PyTorch detections, and `perceive.py`
records how a change that looked free on throughput (imgsz 960 -> 768) cost
13 points of far recall.  An fp16 NCNN graph is a different numerical path;
"it runs and finds people" is not evidence it finds the SAME people.  So
`parity` runs both over sampled frames of the real proxies and compares them
detection by detection.  It is necessary, not sufficient: the end-to-end check
is `anya2/eval.py` over the corpus with `ANYA_POSE_BACKEND` set.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from pipeline.anya2 import pose_backend as PB

# Parity thresholds.  PyTorch CPU and MPS agree BIT-EXACTLY on these frames,
# so there is no runtime noise floor to hide behind: every difference here is
# the export's.  Measured, NCNN at the exact shape (Langmead clip, 100 frames):
#     near  iou p5 0.976   conf p95 0.005   kp err p95 1.0% / median 0.14% of height
#     far   iou p5 0.981   conf p95 0.014   kp err p95 0.6% / median 0.20% of height
# fp32 NCNN is no closer than fp16, so the residual is the runtime, not the
# precision.  Keypoint error is scored RELATIVE TO BOX HEIGHT: an absolute pixel
# limit fails a large near player for the same relative wobble a far player
# passes with.
IOU_MIN = 0.95
KP_REL_MAX = 0.015        # p95 of per-detection median keypoint error / height
CONF_MAX = 0.02
COUNT_AGREE_MIN = 0.97     # frames where both runtimes find the same count;
                           # a detection sitting on POSE_CONF can flip either way


def near_shape():
    from pipeline.anya2 import perceive as PC
    from pipeline.anya2 import court as C
    w, h = C.ANALYSIS_SIZE
    return PB.rect_shape((h, w), PC.NEAR_IMGSZ)


def band_hw(crop):
    """The far band proxy's (h, w) -- rounded exactly as ensure_crop_proxy does."""
    x1, y1, x2, y2 = (int(v) for v in crop)
    return ((y2 - y1) & ~1, (x2 - x1) & ~1)


def far_shape(crop):
    from pipeline.anya2 import perceive as PC
    return PB.rect_shape(band_hw(crop), PC.FAR_IMGSZ)


def export(kind, shape=None, out_root=None):
    """Export and install one model. Returns the installed path."""
    from ultralytics import YOLO
    out_root = Path(out_root or PB.models_dir())
    out_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="anya_pose_export_") as tmp:
        # Exported next to a COPY: Ultralytics writes its output beside the
        # weights, and pipeline/models is not the place for scratch files.
        pt = Path(tmp) / "yolov8n-pose.pt"
        shutil.copy2(PB.POSE_PT, pt)
        m = YOLO(str(pt))
        if kind == "ncnn":
            if shape is None:
                raise ValueError("an NCNN export needs a shape")
            res = Path(m.export(format="ncnn", imgsz=list(shape), half=True,
                                batch=1, verbose=False))
            dst = out_root / PB.export_dir_name("ncnn", shape)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(res), dst)
        elif kind == "onnx":
            res = Path(m.export(format="onnx", imgsz=640, dynamic=True,
                                simplify=True, verbose=False))
            dst = out_root / PB.ONNX_NAME
            shutil.move(str(res), dst)
        else:
            raise ValueError(f"cannot export {kind!r}")
    print(f"[EXPORT] {kind} {shape or 'dynamic'} -> {dst}")
    return dst


def shapes_for_video(video):
    from pipeline.anya2 import perceive as PC
    crop, _ = PC.far_band(video)
    return [near_shape(), far_shape(crop)]


def export_for_site(prof, kind):
    if kind == "onnx":
        return [export("onnx")]
    return [export("ncnn", s) for s in (near_shape(), far_shape(prof["far_band"]))]


# ── parity ───────────────────────────────────────────────────────────────
def _frames(path, n, skip_s=10.0):
    """`n` frames spread evenly over the file, skipping the first seconds."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lo = min(int(skip_s * fps), max(total - 1, 0))
    want = set(np.linspace(lo, max(total - 1, lo), n).astype(int).tolist())
    out, f = [], 0
    while f < total and len(out) < len(want):
        if not cap.grab():
            break
        if f in want:
            ok, img = cap.retrieve()
            if ok:
                out.append(img)
        f += 1
    cap.release()
    return out


def _dets(r):
    if r.keypoints is None or len(r.boxes) == 0:
        return (np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3)))
    return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
            r.keypoints.data.cpu().numpy())


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def compare(ref_results, test_results):
    """Detection-by-detection agreement between two runtimes' results."""
    ious, kps, confs = [], [], []
    same_count, unmatched = 0, 0
    for ra, rb in zip(ref_results, test_results):
        ba, ca, ka = _dets(ra)
        bb, cb, kb = _dets(rb)
        same_count += int(len(ba) == len(bb))
        used = set()
        for i in np.argsort(-ca):
            best, bj = 0.0, -1
            for j in range(len(bb)):
                if j in used:
                    continue
                v = _iou(ba[i], bb[j])
                if v > best:
                    best, bj = v, j
            if bj < 0 or best < 0.5:
                unmatched += 1
                continue
            used.add(bj)
            ious.append(best)
            confs.append(abs(float(ca[i]) - float(cb[bj])))
            vis = (ka[i][:, 2] > 0.5) & (kb[bj][:, 2] > 0.5)
            h = float(ba[i][3] - ba[i][1])
            if vis.any() and h > 0:
                err = np.hypot(*(ka[i][vis, :2] - kb[bj][vis, :2]).T)
                kps.append(float(np.median(err)) / h)
    n = max(len(ref_results), 1)
    return {"frames": len(ref_results),
            "count_agree": same_count / n,
            "matched": len(ious), "unmatched_ref": unmatched,
            "iou_p5": float(np.percentile(ious, 5)) if ious else None,
            "kp_rel_p95": float(np.percentile(kps, 95)) if kps else None,
            "conf_p95": float(np.percentile(confs, 95)) if confs else None}


def passes(s):
    return (s["matched"] > 0
            and s["count_agree"] >= COUNT_AGREE_MIN
            and s["iou_p5"] >= IOU_MIN
            and (s["kp_rel_p95"] is None or s["kp_rel_p95"] <= KP_REL_MAX)
            and s["conf_p95"] <= CONF_MAX)


def parity(video, kind, n=200, device="cpu"):
    """Compare `kind` against torch on both of the video's pose proxies."""
    from pipeline import proxy as P
    from pipeline.anya2 import court as C
    from pipeline.anya2 import perceive as PC
    roles = [("near", P.ensure_proxy(video, size=C.ANALYSIS_SIZE, crf=14,
                                     label="PROXY540"), PC.NEAR_IMGSZ)]
    crop, _ = PC.far_band(video)
    roles.append(("far", P.ensure_crop_proxy(video, crop, crf=14, label="FARBAND"),
                  PC.FAR_IMGSZ))
    report, ok = {}, True
    for role, path, imgsz in roles:
        frames = _frames(path, n)
        hw = frames[0].shape[:2]
        ref = PB.load(hw, imgsz, kind="torch")
        test = PB.load(hw, imgsz, kind=kind)
        kw = dict(conf=PC.POSE_CONF, device=device, classes=[0], verbose=False)
        ra = [ref.predict([f], **kw)[0] for f in frames]
        rb = [test.predict([f], **kw)[0] for f in frames]
        s = compare(ra, rb)
        s["model"] = repr(test)
        s["pass"] = passes(s)
        ok &= s["pass"]
        report[role] = s
        print(f"[PARITY] {role}: {json.dumps(s)}")
    return ok, report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pose model export + parity.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--backend", choices=["ncnn", "onnx"], required=True)
    e.add_argument("--shape", action="append", default=[],
                   help="HxW, e.g. 384x640 (repeatable; NCNN only)")
    e.add_argument("--video", help="derive the near and far shapes from this "
                                   "calibrated video")
    p = sub.add_parser("parity")
    p.add_argument("video")
    p.add_argument("--backend", choices=["ncnn", "onnx"], required=True)
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--json", help="write the report here")
    a = ap.parse_args(argv)

    if a.cmd == "export":
        if a.backend == "onnx":
            export("onnx")
            return 0
        shapes = [tuple(int(v) for v in s.lower().split("x")) for s in a.shape]
        if a.video:
            shapes += shapes_for_video(a.video)
        if not shapes:
            shapes = [near_shape()]
            print("[EXPORT] no --video: exporting the near shape only; the far "
                  "shape depends on the court and needs --video or --shape")
        for s in dict.fromkeys(shapes):
            export("ncnn", s)
        return 0

    ok, rep = parity(a.video, a.backend, a.n)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    print("[PARITY] PASS" if ok else "[PARITY] FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
