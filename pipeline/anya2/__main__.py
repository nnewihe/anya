"""
python -m pipeline.anya2 VIDEO [VIDEO ...] [options]

Video in, dead-time-removed reel out, with no prompts -- see headless.py.
Several VIDEOs are chapters of ONE recording and are joined first.
"""

import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pipeline.anya2",
                                 description="Cut the dead time out of a tennis video.")
    ap.add_argument("videos", nargs="+", help="one video, or the chapters of one recording")
    ap.add_argument("-o", "--output", help="reel path (default: <first>_anya2_reel.mp4)")
    ap.add_argument("--site", help="fixed-camera site profile dir (pipeline.anya2.site)")
    ap.add_argument("--device", help="torch device: cpu / mps / cuda (default: best available)")
    ap.add_argument("--backend", choices=["torch", "ncnn", "onnx"],
                    help="pose runtime (sets ANYA_POSE_BACKEND; default torch)")
    ap.add_argument("--copy", action="store_true",
                    help="no re-encode (video or audio): original resolution and quality, cuts "
                         "start on the keyframe at or before each point "
                         "(fastest; ignores --scale-height)")
    ap.add_argument("--scale-height", type=int, default=1080,
                    help="re-encoded output height; 0 keeps native (slow on a Pi)")
    ap.add_argument("--work-dir", help="put every interim file here instead of beside the video")
    ap.add_argument("--dry-run", action="store_true", help="detect only, write no video")
    ap.add_argument("--segments-json", help="also write the kept segments here")
    a = ap.parse_args(argv)

    if a.backend:
        os.environ["ANYA_POSE_BACKEND"] = a.backend
    from pipeline import workdir as WD
    from pipeline.anya2 import headless as H
    from pipeline.anya2 import site as S

    if a.work_dir:
        WD.set_work_dir(os.path.abspath(a.work_dir))
    cfg = H.make_config(a.device, a.scale_height or None, copy_video=a.copy)
    try:
        segs, out = H.build(a.videos, a.output, cfg, site=a.site,
                            on_progress=H.ProgressPrinter(), dry_run=a.dry_run)
    except (H.NotCalibrated, S.NeedsCalibration) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    kept = sum(s["stop"] - s["start"] for s in segs)
    print(f"{len(segs)} segments, {kept / 60:.1f} min kept" +
          (f" -> {out}" if out else ""))
    if a.segments_json:
        with open(a.segments_json, "w") as fh:
            json.dump(segs, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
