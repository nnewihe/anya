"""
calibrate_court.py
==================
Interactive one-time court-corner calibration for a clip, cached in the same
960x540 analysis space the optimizer and pipeline use. Opens a window on a
reference frame; click the four court corners IN ORDER:

    bottom-left, bottom-right, top-right, top-left   (near baseline first)

Keys: r = reset, q/Esc = abort. Writes <video>_court_cache.json beside the
video, which optimize_energy.py then discovers automatically.

Usage:
    python pipeline/calibrate_court.py /Volumes/Anya/Data/21/snippet.mp4
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utilities import init_court


def main():
    ap = argparse.ArgumentParser(description="Interactive court-corner calibration (960x540 cache)")
    ap.add_argument("video", help="Path to the clip video (e.g. /Volumes/Anya/Data/21/snippet.mp4)")
    ap.add_argument("--width", type=int, default=960, help="Analysis width (default 960 — matches the pipeline)")
    ap.add_argument("--height", type=int, default=540, help="Analysis height (default 540)")
    ap.add_argument("--frame", type=int, default=300, help="Reference frame index to calibrate on")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    print(f"Calibrating {args.video} at {args.width}x{args.height} on frame {args.frame}")
    print("Click IN ORDER: bottom-left, bottom-right, top-right, top-left  (r=reset, q=quit)")
    pts, shape = init_court(args.video, target_idx=args.frame, analysis_size=(args.width, args.height))
    print(f"Saved corners: {pts}")


if __name__ == "__main__":
    main()
