"""
deadtime_cutter.py
==================
End-to-end dead-time removal for a tennis match video.

Dead time = everything from the end of one point to the start of the next
service motion.  The cutter keeps [service motion .. point end] for every
point and drops the rest:

    Stage 1  match_telemetry.py   one slow perception pass  (cached JSONL)
    Stage 2  point_segmenter.py   serve starts (near + far) + fused point ends
    Stage 3  ffmpeg               cut & concatenate the kept segments

Because stage 1 is cached, re-running after tuning SegmenterConfig only costs
seconds — the video is only decoded again for the final cut.

Run (both work):
    python -m pipeline.deadtime_cutter match.mp4
    python pipeline/deadtime_cutter.py match.mp4

Useful flags:
    --dry-run            segment + report only, no output video
    --force-telemetry    re-run the perception pass even if cached
    --stride N           perception on every Nth frame (quick experiments)
"""

import argparse
import os
import sys
from pathlib import Path

# Allow `python pipeline/deadtime_cutter.py ...` in addition to `-m` form.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pipeline"

from .match_telemetry import extract_match_telemetry
from .point_segmenter import (SegmenterConfig, load_telemetry, segment_match,
                              write_segments_csv, write_segments_json)
from .utilities import Config, create_highlights_ffmpeg, init_court


def cut_dead_time(video_path: str, output_path: str = None,
                  cfg: SegmenterConfig = None,
                  force_telemetry: bool = False, stride: int = 1,
                  enable_far_serve: bool = True,
                  dry_run: bool = False, progress_cb=None):
    """Full pipeline. Returns (segments, output_path or None)."""
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    if output_path is None:
        output_path = os.path.join(video_dir, f"{video_stem}_no_deadtime.mp4")

    # ── Stage 0: one-time court calibration (cached to disk after first run) ──
    # Runs up front, before the YOLO models load, so the corner-click prompt
    # (bottom-left, bottom-right, top-right, top-left — same order the
    # run_anya.py pipeline uses) doesn't appear to hang mid-load.
    init_court(video_path, analysis_size=(Config.ANALYSIS_WIDTH, Config.ANALYSIS_HEIGHT))

    # ── Stage 1: perception (cached) ─────────────────────────────────────
    telemetry_path = extract_match_telemetry(
        video_path, force=force_telemetry, stride=stride,
        enable_far_serve=enable_far_serve, progress_cb=progress_cb)

    # ── Stage 2: segmentation ────────────────────────────────────────────
    match = load_telemetry(telemetry_path)
    far_misses = []
    segments = segment_match(match, cfg, far_misses_out=far_misses)

    base = os.path.join(video_dir, video_stem)
    write_segments_csv(segments,  base + "_points.csv")
    write_segments_json(segments, base + "_points.json")

    # Far-serve near-misses: serve-like trace onsets with no tracked far
    # player nearby, for review and for labeling far-serve tuning data.
    if far_misses:
        miss_path = base + "_far_misses.csv"
        import csv as _csv
        with open(miss_path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["t", "score", "reason"])
            for t, s, reason in sorted(far_misses):
                w.writerow([f"{t:.2f}", f"{s:.3f}", reason])
        print(f"[CUT] {len(far_misses)} far-serve near-miss(es) → {miss_path}")

    if not segments:
        print("[CUT] No points detected — nothing to cut.")
        return segments, None

    # ── Stage 3: cut ─────────────────────────────────────────────────────
    if dry_run:
        print("[CUT] --dry-run: skipping video export.")
        return segments, None

    create_highlights_ffmpeg(
        video_path,
        [(s.start, s.end) for s in segments],
        output_path,
        pre_roll=0.0,          # pre-roll is already baked into each segment
        merge_gap_sec=1.0,     # fault → second serve stays one continuous cut
    )
    print(f"\n[DONE] Output video : {output_path}")
    print(f"[DONE] Point report : {base}_points.csv")
    return segments, output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut dead time (between-point time) out of a tennis match video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pipeline.deadtime_cutter match.mp4
  python -m pipeline.deadtime_cutter match.mp4 --dry-run
  python -m pipeline.deadtime_cutter match.mp4 --force-telemetry --output tight.mp4
""",
    )
    parser.add_argument("video", help="Input tennis match video")
    parser.add_argument("--output", default=None,
                        help="Output MP4 (default: <input>_no_deadtime.mp4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect and report points but skip the video export")
    parser.add_argument("--force-telemetry", action="store_true",
                        help="Re-run the perception pass even if a cache exists")
    parser.add_argument("--stride", type=int, default=1,
                        help="Perception pass on every Nth frame (default 1)")
    parser.add_argument("--no-far-serve", action="store_true",
                        help="Skip the far-side ST-GCN model during telemetry "
                             "extraction (stage 2 detects far serves from the "
                             "ball trace and no longer reads its scores)")
    args = parser.parse_args()

    cut_dead_time(args.video, args.output,
                  force_telemetry=args.force_telemetry,
                  stride=args.stride,
                  enable_far_serve=not args.no_far_serve,
                  dry_run=args.dry_run)
