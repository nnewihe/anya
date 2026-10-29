"""
pose_backend.py
===============
Which runtime runs the pose model: PyTorch (the default, what every number in
this package was measured on), NCNN, or ONNX Runtime.  Chosen with
`ANYA_POSE_BACKEND=torch|ncnn|onnx`.

Why this exists: on a Raspberry Pi 5 the PyTorch CPU path makes a one-hour
match take most of a day, and NCNN is the runtime ARM CPUs are fast on.  Both
exports load through the same Ultralytics `YOLO(...)` API, so the pose passes
do not change -- only what `load` hands them.

THE SHAPE TRAP
--------------
PyTorch letterboxes RECTANGULARLY: the 960x540 near proxy at imgsz 640 is
inferred at 640x384, and the far band (~730 px wide, a few hundred tall) at
imgsz 960 is inferred at 960 x (its height, scaled, rounded up to 32).  An NCNN
export has a FIXED input shape, and Ultralytics feeds a fixed-shape model a
SQUARE letterbox -- 640x640 and 960x960, 40-80% of it grey padding, paid for on
every frame.  So each NCNN export is made at EXACTLY the rectangle PyTorch
would have inferred at (`rect_shape`), and `load` accepts nothing else.

Not "the same scale with a little spare padding", which was tried: one far
export 32 px taller than the band needed, so it could serve a band that grows
between recordings.  Same resize, same content, only more grey below -- and
conf drifted 0.14 at p95 against PyTorch, 4 of 145 far detections vanished,
and box IoU fell to 0.91, against 0.014 / 0 / 0.98 for the exact shape (Langmead
clip, 100 frames).  The network is not padding-invariant at this image height.
So a far band of a new size gets its own export, made on demand (`load`, about
two seconds) and cached by shape; the fixed camera means that is once per site
in practice.

ONNX is exported with dynamic axes, so Ultralytics letterboxes it
rectangularly exactly like PyTorch and one file serves every shape.

NCNN IS BATCH-1
---------------
Ultralytics' NCNN backend runs `im[0]` and returns one result whatever it was
given.  `_pose_pass` zips results onto frame slots, so a batch of 16 would
silently fill 1 frame and leave 15 empty -- a pass that finishes, fast, with
no players in it.  `Pose.predict` splits the batch for any backend that is not
batch-capable.
"""

import math
import os
from pathlib import Path

BACKENDS = ("torch", "ncnn", "onnx")
STRIDE = 32

_MODELS = Path(__file__).resolve().parents[1] / "models"
_PT = _MODELS / "yolov8n-pose.pt"
POSE_PT = str(_PT) if _PT.is_file() else "yolov8n-pose.pt"
ONNX_NAME = "onnx_dynamic.onnx"


def backend():
    b = os.environ.get("ANYA_POSE_BACKEND", "torch").strip().lower() or "torch"
    if b not in BACKENDS:
        raise ValueError(f"ANYA_POSE_BACKEND={b!r}; expected one of {BACKENDS}")
    return b


def models_dir():
    return Path(os.environ.get("ANYA_POSE_MODELS", str(_MODELS / "pose")))


# ── letterbox geometry (Ultralytics' LetterBox, reproduced) ──────────────
def scale_for(src_hw, imgsz):
    """The resize ratio PyTorch uses for a frame of `src_hw` at int `imgsz`."""
    h, w = src_hw
    return min(imgsz / h, imgsz / w)


def rect_shape(src_hw, imgsz):
    """(H, W) PyTorch actually infers at: resized frame, rounded up to STRIDE."""
    h, w = src_hw
    r = scale_for(src_hw, imgsz)
    uh, uw = int(round(h * r)), int(round(w * r))
    return (int(math.ceil(uh / STRIDE) * STRIDE),
            int(math.ceil(uw / STRIDE) * STRIDE))


def fits(export_hw, src_hw, imgsz):
    """True when a fixed `export_hw` is exactly what PyTorch infers `src_hw` at."""
    return tuple(export_hw) == rect_shape(src_hw, imgsz)


def auto_export():
    return os.environ.get("ANYA_POSE_AUTO_EXPORT", "1").strip().lower() not in (
        "0", "false", "no", "off")


# Ultralytics recognises an NCNN model by the directory SUFFIX "_ncnn_model";
# any other name is refused as an unknown format.
NCNN_SUFFIX = "_ncnn_model"


def export_dir_name(kind, hw):
    return f"pose_{hw[0]}x{hw[1]}_{kind}_model"


def _ncnn_exports():
    d = models_dir()
    out = []
    if not d.is_dir():
        return out
    for p in d.iterdir():
        if (p.is_dir() and p.name.startswith("pose_")
                and p.name.endswith(NCNN_SUFFIX) and any(p.glob("*.param"))):
            try:
                h, w = (int(v) for v in
                        p.name[len("pose_"):-len(NCNN_SUFFIX)].split("x"))
            except ValueError:
                continue
            out.append(((h, w), p))
    return out


def find_ncnn(src_hw, imgsz):
    """The NCNN export that `fits`, or None."""
    ok = [(hw, p) for hw, p in _ncnn_exports() if fits(hw, src_hw, imgsz)]
    return ok[0] if ok else None


class MissingExport(FileNotFoundError):
    pass


class Pose:
    """One loaded pose model plus the `imgsz` and batching it must be run with."""

    def __init__(self, model, imgsz, batch_ok, kind, where):
        self.model = model
        self.imgsz = imgsz
        self.batch_ok = batch_ok
        self.kind = kind
        self.where = where

    def predict(self, frames, **kw):
        """Ultralytics results, one per frame, whatever the backend."""
        if not isinstance(frames, list):
            frames = [frames]
        if self.batch_ok:
            return list(self.model.predict(frames if len(frames) > 1 else frames[0],
                                           imgsz=self.imgsz, **kw))
        out = []
        for f in frames:
            out.extend(self.model.predict(f, imgsz=self.imgsz, **kw))
        return out

    def __repr__(self):
        return f"Pose({self.kind} @ {self.imgsz}, {self.where})"


def load(src_hw, imgsz, kind=None):
    """The pose model for frames of `src_hw` at PyTorch-equivalent `imgsz`."""
    from ultralytics import YOLO
    kind = kind or backend()
    if kind == "torch":
        return Pose(YOLO(POSE_PT), imgsz, True, kind, POSE_PT)
    if kind == "onnx":
        p = models_dir() / ONNX_NAME
        if not p.is_file():
            raise MissingExport(
                f"ANYA_POSE_BACKEND=onnx but {p} does not exist; run "
                f"`python -m pipeline.anya2.export_pose export --backend onnx`")
        return Pose(YOLO(str(p), task="pose"), imgsz, True, kind, str(p))
    hit = find_ncnn(src_hw, imgsz)
    if hit is None and auto_export():
        from pipeline.anya2 import export_pose as EP
        print(f"[POSE] no NCNN export for {rect_shape(src_hw, imgsz)} yet -- "
              f"exporting one (cached in {models_dir()})")
        EP.export("ncnn", rect_shape(src_hw, imgsz))
        hit = find_ncnn(src_hw, imgsz)
    if hit is None:
        need = rect_shape(src_hw, imgsz)
        have = [f"{hw[0]}x{hw[1]}" for hw, _ in _ncnn_exports()] or ["none"]
        raise MissingExport(
            f"no NCNN pose export fits a {src_hw[1]}x{src_hw[0]} frame at imgsz "
            f"{imgsz} (needs {need[0]}x{need[1]} at the same scale; have "
            f"{', '.join(have)} in {models_dir()}). Run `python -m "
            f"pipeline.anya2.export_pose export --backend ncnn --shape "
            f"{need[0]}x{need[1]}`, or `... --video <this video>`.")
    hw, p = hit
    return Pose(YOLO(str(p), task="pose"), hw, False, kind, str(p))
