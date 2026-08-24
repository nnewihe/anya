"""
balls.py
========
Ball detection for anya2, with the exclusion-zone filter applied AT SOURCE.

Why the filter lives here and not in each detector
--------------------------------------------------
A tennis court is full of things that look like a tennis ball to a small-object
detector and never move: the ball hopper, a basket by the fence, a stack of
spare balls on the bench, a yellow bag.  Every consumer that reads a ball stream
has to reject them, and in the current pipeline each one rejects them slightly
differently -- which means a ball filtered out of one detector's evidence is
still charging another's.

Here the zones are applied inside `detect`, so a detector CANNOT accidentally
read an unfiltered stream.  There is no "raw" accessor.

The zones themselves are reused, not reinvented: `pipeline.utilities`'
`create_auto_exclusion_zones` samples frames across the whole video, runs the
ball model at high imgsz, and DBSCAN-clusters the detection centres.  A cluster
means "a ball-shaped thing was found in the same few pixels across frames
minutes apart", which a real ball in play can never be.  The cache format is
unchanged, so a clip already calibrated for the shipping pipeline needs no
rescan and the two agree about what is excluded.

Tracing is offered, not imposed
-------------------------------
`trace` wraps the IMM Kalman tracker, but nothing calls it by default.  Each
detector decides whether it wants ball evidence at all -- the near-serve
detector, for one, is pose-only and never touches this module.  That is a real
compute win and it should stay a decision rather than becoming a default.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.utilities import (Config, _is_in_exclusion_zone,
                                create_auto_exclusion_zones,
                                load_cached_exclusion_zones,
                                save_cached_exclusion_zones)
from pipeline.videoio import open_video
from pipeline.anya2 import court as C

_W = Path(__file__).resolve().parents[1] / "models" / "ball_best.pt"
BALL_MODEL = str(_W) if _W.is_file() else "ball_best.pt"

BALL_CONF = 0.10
BALL_IMGSZ = 1920          # the ball is a handful of pixels at 960x540; this is
                           # the shipping pipeline's ACTIVE_BALL_IMGSZ and the
                           # reason it is not lower is measured there, not here.


def _model(path=None):
    from ultralytics import YOLO
    return YOLO(path or BALL_MODEL)


def zones_for(video, model=None, analysis_size=C.ANALYSIS_SIZE, force=False):
    """Static exclusion zones in ANALYSIS-frame pixels, cached beside the video.

    Cache misses are the expensive path (a scan over the whole clip), which is
    why the shipping cache is reused rather than a new one being written under
    a different name.
    """
    cached = None if force else load_cached_exclusion_zones(video)
    if cached is not None:
        return _to_analysis(cached, video, analysis_size)
    print("[balls] scanning for static exclusion zones...")
    zones = create_auto_exclusion_zones(
        video, model or _model(), analysis_size=analysis_size,
        ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX)
    save_cached_exclusion_zones(video, zones)
    print(f"[balls] {len(zones)} zone(s)")
    return _to_analysis(zones, video, analysis_size)


def _to_analysis(zones, video, analysis_size):
    """Zones may have been cached in SOURCE pixels by an older run; rescale.

    Detected by extent rather than by a flag: a zone whose corner exceeds the
    analysis frame cannot be in analysis coordinates.  Mirrors
    `anya_end_telemetry._to_analysis_coords` so the two caches stay compatible.
    """
    if not zones:
        return []
    aw, ah = analysis_size
    if max(z[2] for z in zones) <= aw and max(z[3] for z in zones) <= ah:
        return [tuple(z) for z in zones]
    cap = open_video(video, "BALLS")
    sw = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or aw
    sh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or ah
    cap.release()
    fx, fy = aw / sw, ah / sh
    return [(int(z[0] * fx), int(z[1] * fy), int(z[2] * fx), int(z[3] * fy))
            for z in zones]


def detect(frames, model, zones=(), conf=BALL_CONF, imgsz=BALL_IMGSZ,
           offset=(0.0, 0.0), scale=(1.0, 1.0)):
    """Ball centres per frame, exclusion zones already applied.

    Returns a list (one entry per frame) of [(cx, cy, conf), ...] in ANALYSIS
    coordinates.  There is deliberately no way to ask for the unfiltered list.
    """
    res = model.predict(frames if len(frames) > 1 else frames[0], imgsz=imgsz,
                        conf=conf, verbose=False)
    ox, oy = offset
    sx, sy = scale
    out = []
    for r in res:
        hits = []
        if r.boxes is not None and len(r.boxes):
            xy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xy, cf, cl):
                if k != Config.DEFAULT_BALL_CLASS_INDEX:
                    continue
                cx = (0.5 * (x1 + x2)) * sx + ox
                cy = (0.5 * (y1 + y2)) * sy + oy
                if _is_in_exclusion_zone(cx, cy, zones):
                    continue
                hits.append((float(cx), float(cy), float(c)))
        out.append(hits)
    return out


def in_court(cx, cy, H, pad_m=1.5):
    """Is this ball centre over the court (plus a margin), in court metres?

    A SECOND filter, and a different one: exclusion zones remove things that are
    always there, this removes a real detection that is somewhere a rally ball
    cannot be (the fence, the stands, the next court over).  Uses the same
    lateral band as the player gate so the two agree about where the court is.
    """
    x, y = C.to_court(H, [[cx, cy]])[0]
    return (C.X_LO - pad_m <= x <= C.X_HI + pad_m
            and -pad_m <= y <= C.COURT_L + pad_m)


def trace(detections, fps, frame_height, **kw):
    """IMM Kalman single-ball trace over per-frame detections.

    Offered, never run by default -- see the module docstring.  Returns the
    tracker so the caller can read whatever it needs rather than a shape chosen
    here on its behalf.
    """
    from pipeline.ball_tracker import (BallTrackManager,
                                       make_image_row_perspective)
    mgr = BallTrackManager(fps=fps,
                           perspective=make_image_row_perspective(frame_height),
                           **kw)
    for i, hits in enumerate(detections):
        mgr.update(i / fps, [(x, y) for x, y, _ in hits])
    return mgr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("video")
    ap.add_argument("--force", action="store_true", help="Rescan zones")
    a = ap.parse_args()
    z = zones_for(a.video, force=a.force)
    print(f"[balls] {len(z)} exclusion zone(s) in {C.ANALYSIS_SIZE} coords:")
    for r in z:
        print(f"   {r}")


if __name__ == "__main__":
    main()
