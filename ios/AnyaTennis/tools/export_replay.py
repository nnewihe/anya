"""
export_replay.py
================
The reference the Swift engine is checked against: main's own anya2 stages
5-8, run on the corpus clips' cached pose passes, with every intermediate
written in a form Swift reads directly.

For each clip, into OUT/<clip>/:

    meta.json         corners, fps, stride, the camera track (frames + warps),
                      and the byte layout of the two .bin files below
    inputs.bin        the near and far pose passes, float32 little-endian
    reference.bin     main's outputs, little-endian: tracks (float32), walking
                      probability (float64), near_end signals (float32), the
                      orchestrator's live score (float64)
    reference.json    the three detectors' events and the orchestrator's reel

Every stage runs in a temporary work dir holding LOCAL copies of the clip's
court cache, camera track and pose passes, so nothing on the data drive is
read stale mid-run or overwritten.

    ~/.pyenv/versions/anya-build/bin/python ios/AnyaTennis/tools/export_replay.py OUT [--clips 21 22 ...]
"""

import argparse
import dataclasses
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from pipeline import workdir  # noqa: E402
from pipeline.anya2 import eval as anya2_eval  # noqa: E402

DATA = os.environ.get("ANYA_DATA", "/Volumes/Anya/Data")
# The trusted corpus: ground truth plus cached pose passes. 37 and 63 carry
# labels but are untrusted (see memory "Excluded clips 37 & 63").
TRUSTED_CLIPS = ["21", "22", "23", "24", "25", "26", "35", "36", "38", "40", "43", "50", "58"]


class BinaryWriter:
    """Appends little-endian arrays to one file and records where each one is."""

    def __init__(self, path):
        self.file = open(path, "wb")
        self.layout = {}
        self.offset = 0

    def add(self, name, array, dtype):
        a = np.ascontiguousarray(array, dtype=dtype)
        self.layout[name] = {"offset": self.offset, "count": int(a.size),
                             "shape": list(a.shape), "dtype": np.dtype(dtype).name}
        self.file.write(a.tobytes())
        self.offset += a.nbytes

    def close(self):
        self.file.close()
        return self.layout


def _event_rows(path):
    with open(path) as fh:
        return json.load(fh)["events"]


def export_clip(clip, out_dir):
    from pipeline.anya2 import orchestrator, run, tracks
    from pipeline.anya2.config import Anya2Config

    clip_dir = os.path.join(DATA, clip)
    video = anya2_eval.clip_video(clip_dir)
    stem = os.path.splitext(os.path.basename(video))[0]
    work = tempfile.mkdtemp(prefix=f"anya2_replay_{clip}_")
    try:
        for suffix in ("_court_cache.json", "_anya2_camera.npz",
                       "_anya2_near_dets.npz", "_anya2_far_dets.npz"):
            source = os.path.join(clip_dir, stem + suffix)
            if os.path.isfile(source):
                shutil.copy(source, work)
        workdir.set_work_dir(work)
        started = time.time()

        near_npz = os.path.join(work, f"{stem}_anya2_near_dets.npz")
        far_npz = os.path.join(work, f"{stem}_anya2_far_dets.npz")
        cfg = Anya2Config()
        tracks.build(video, near_npz, far_npz, verbose=False)
        run._compute_end_signals(video, force=True)
        run._run_detectors(video, cfg, work, stem)
        reel_cfg = run._reel_config(cfg)
        reel = orchestrator.build_reel(video, cfg=reel_cfg, verbose=False)
        live, _ = orchestrator._live_score(video, None, smooth_s=reel_cfg.live_smooth_s)

        os.makedirs(out_dir, exist_ok=True)
        near, far = np.load(near_npz), np.load(far_npz)
        inputs = BinaryWriter(os.path.join(out_dir, "inputs.bin"))
        for name, z in (("near", near), ("far", far)):
            inputs.add(f"{name}.keypoints", z["kp"], "<f4")
            inputs.add(f"{name}.boxes", z["box"], "<f4")
            inputs.add(f"{name}.confidences", z["conf"], "<f4")
        input_layout = inputs.close()

        t = tracks.load(video)
        walk = np.load(os.path.join(work, f"{stem}_anya2_walk.npz"))
        endsig = np.load(os.path.join(work, f"{stem}_anya2_endsig.npz"))
        reference = BinaryWriter(os.path.join(out_dir, "reference.bin"))
        reference.add("tracks.keypoints", t["kp"], "<f4")
        reference.add("tracks.boxes", t["bbox"], "<f4")
        reference.add("tracks.court", t["court"], "<f4")
        reference.add("tracks.eligible", t["eligible"], "u1")
        reference.add("walking.probability", walk["prob"], "<f8")
        for name in endsig.files:
            reference.add(f"near_end.{name}", endsig[name], "<f4")
        reference.add("orchestrator.live", live if live is not None else np.zeros(0), "<f8")
        reference_layout = reference.close()

        corners = json.load(open(os.path.join(work, f"{stem}_court_cache.json")))["points"]
        camera_path = os.path.join(work, f"{stem}_anya2_camera.npz")
        camera = None
        if os.path.isfile(camera_path):
            from pipeline.anya2.camera import load as load_camera
            track = load_camera(video, path=camera_path)
            camera = {"frames": [int(f) for f in track.frames],
                      "warps": [[float(v) for v in w.ravel()] for w in track.W]}
        from walking.court import load_homography
        meta = {
            "clip": clip, "stem": stem, "corners": corners,
            # The homography Python actually used (cv2.getPerspectiveTransform),
            # so Swift's own solve can be checked against it entry by entry.
            "homography": [float(v) for v in load_homography(video).ravel()],
            "fps": float(near["fps"]),
            "stride": int(near["stride"]) if "stride" in near else 1,
            "src_fps": float(near["src_fps"]) if "src_fps" in near else None,
            "camera": camera,
            "inputs": input_layout, "reference": reference_layout,
            "reel_config": dataclasses.asdict(reel_cfg),
        }
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"))
        json.dump({
            "near_serve": _event_rows(os.path.join(work, f"{stem}_anya2_near_serve.json")),
            "far_serve": _event_rows(os.path.join(work, f"{stem}_anya2_far_serve.json")),
            "point_end": _event_rows(os.path.join(work, f"{stem}_anya2_point_end.json")),
            "reel": reel,
        }, open(os.path.join(out_dir, "reference.json"), "w"), indent=1, default=float)
        print(f"[{clip}] {len(reel['segments'])} segments, "
              f"camera {'tracked' if camera else 'static'}, {time.time() - started:.0f}s",
              flush=True)
    finally:
        workdir.clear_work_dir()
        shutil.rmtree(work, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("out")
    parser.add_argument("--clips", nargs="*", default=TRUSTED_CLIPS)
    args = parser.parse_args()
    for clip in args.clips:
        export_clip(clip, os.path.join(args.out, clip))


if __name__ == "__main__":
    main()
