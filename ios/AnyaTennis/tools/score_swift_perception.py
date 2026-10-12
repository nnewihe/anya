"""
score_swift_perception.py
=========================
Is the app's perception good enough? Scored against ground truth, the only
question that matters once the pixels differ at all (fp16 on the Neural
Engine is not Ultralytics' fp32 on the GPU).

For each corpus clip: run the Swift perception (`anya2-perceive`) on the
clip's own proxies, store its people as perceive.py's npz files, run main's
stages 5-8 on them, and score detectors and reel exactly as for main's own
detections -- side by side.

    python ios/AnyaTennis/tools/score_swift_perception.py OUT.jsonl [CLIP ...] [--cpu] [--model M.mlpackage]

Needs `swift build -c release` in Packages/Anya2Engine first, and the data drive.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

from pipeline import proxy, workdir                                      # noqa: E402
from pipeline.anya2 import eval as E, orchestrator, perceive, run, tracks  # noqa: E402
from pipeline.anya2.config import Anya2Config                           # noqa: E402
from pipeline.anya2.contract import FAR_SERVE, NEAR_SERVE, POINT_END, load_events  # noqa: E402

DATA = os.environ.get("ANYA_DATA", "/Volumes/Anya/Data")
CLIPS = "21 22 23 24 25 26 35 36 38 40 43 50 58".split()
PACKAGE = os.path.join(HERE, "..", "Packages", "Anya2Engine")
BINARY = os.path.join(PACKAGE, ".build", "release", "anya2-perceive")
MODEL = os.path.join(HERE, "..", "AnyaTennis", "Resources", "PoseModel.mlpackage")


def swift_detections(video, work, stem, compute, model=MODEL, extra=()):
    """Run anya2-perceive and write its people as the two dets npz files."""
    with np.load(os.path.join(work, stem + perceive.FAR_SUFFIX)) as z:
        crop = [int(v) for v in z["crop"]]
    with np.load(os.path.join(work, stem + perceive.NEAR_SUFFIX)) as z:
        near_meta = {k: z[k] for k in ("fps", "src_fps", "stride", "n_src_frames")}
    probe = proxy.probe_video(video)
    near_video = os.path.join(work, stem + "_proxy540.mp4")
    far_video = os.path.join(work, stem + "_farband.mp4")
    out = os.path.join(work, "swift")
    subprocess.run([BINARY, model, near_video, far_video, ",".join(map(str, crop)),
                    f"{probe['width']},{probe['height']}", out, compute, *extra], check=True,
                   stdout=subprocess.DEVNULL)
    for name, suffix, extra in (("near", perceive.NEAR_SUFFIX, (0, 0, 0, 0)), ("far", perceive.FAR_SUFFIX, crop)):
        raw = open(os.path.join(out, name + ".bin"), "rb").read()
        n = int(np.frombuffer(raw[:4], np.int32)[0])
        floats = np.frombuffer(raw[4:], np.float32)
        kp = floats[:n * 8 * 51].reshape(n, 8, 17, 3)
        box = floats[n * 8 * 51:n * 8 * 55].reshape(n, 8, 4)
        conf = floats[n * 8 * 55:].reshape(n, 8)
        np.savez_compressed(os.path.join(work, stem + suffix), kp=kp, box=box, conf=conf,
                            crop=np.asarray(extra, np.int32), **near_meta)


def stages_5_to_8(video, work, stem):
    tracks.build(video, os.path.join(work, stem + perceive.NEAR_SUFFIX),
                 os.path.join(work, stem + perceive.FAR_SUFFIX), verbose=False)
    run._compute_end_signals(video, force=True)
    cfg = Anya2Config()
    run._run_detectors(video, cfg, work, stem)
    return orchestrator.build_reel(video, cfg=run._reel_config(cfg), verbose=False)


def score(work, stem, reel, clip_dir):
    span = E.labelled_span(clip_dir)
    out = {}
    for mode, suffix in ((NEAR_SERVE, "_anya2_near_serve.json"), (FAR_SERVE, "_anya2_far_serve.json"),
                         (POINT_END, "_anya2_point_end.json")):
        t = sorted(e.t for e in load_events(os.path.join(work, stem + suffix))
                   if span is None or span[0] <= e.t <= span[1])
        r = E.score(t, E.gt_times(clip_dir, mode), mode)
        out[mode] = [r["hit"], r["n_gt"], r["n_det"]]
    s = orchestrator.score_reel(reel, clip_dir)
    out["reel"] = [s["points_whole"], s["n_points"], round(s["live_retained"], 3), round(s["reel_s"], 1)]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("out")
    parser.add_argument("clips", nargs="*", default=CLIPS)
    parser.add_argument("--cpu", action="store_true", help="Core ML on the CPU only (default: Neural Engine allowed)")
    parser.add_argument("--bt601", action="store_true",
                        help="decode with the BT.601 matrix, as ffmpeg/OpenCV do for perceive.py")
    parser.add_argument("--model", default=MODEL, help="an .mlpackage (default: the shipped fp16 model)")
    args = parser.parse_args()
    with open(args.out, "a") as results:
        for clip in args.clips:
            clip_dir = os.path.join(DATA, clip)
            video = E.clip_video(clip_dir)
            stem = os.path.splitext(os.path.basename(video))[0]
            work = tempfile.mkdtemp(prefix=f"swiftpose_{clip}_")
            for suffix in ("_court_cache.json", "_anya2_camera.npz", perceive.NEAR_SUFFIX, perceive.FAR_SUFFIX,
                           "_proxy540.mp4", "_proxy540.mp4.build.json", "_farband.mp4", "_farband.mp4.build.json"):
                if os.path.isfile(os.path.join(clip_dir, stem + suffix)):
                    shutil.copy(os.path.join(clip_dir, stem + suffix), work)
            workdir.set_work_dir(work)
            row = {"clip": clip, "python": score(work, stem, stages_5_to_8(video, work, stem), clip_dir)}
            swift_detections(video, work, stem, "cpu" if args.cpu else "all", args.model,
                             ("bt601",) if args.bt601 else ())
            row["swift"] = score(work, stem, stages_5_to_8(video, work, stem), clip_dir)
            print(json.dumps(row), flush=True)
            results.write(json.dumps(row) + "\n")
            results.flush()
            workdir.clear_work_dir()
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
