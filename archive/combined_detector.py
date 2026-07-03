"""
combined_detector.py
====================
Side-aware highlight detector for a single fixed camera (Approach A — merge layer).

The two existing detectors each excel on one serve side:

  • run_anya.py        — serve-gated state machine (WAITING→ARMED→ACTIVE).  The
                         ARMED→ACTIVE transition requires a toss above the NEAR
                         player's head, which is only observable for NEAR serves;
                         the toss gate also rejects "walking with a ball".  Great
                         on near serves, blind to far serves.
  • rally_detector.py  — pure moving-ball-trace detector.  Needs no toss, so it
                         nails FAR serves, but is ungated near-side.

This orchestrator runs both over the same video and unions their outputs in each
detector's sweet spot:

  • NEAR points  ← run_anya's segments (all near-serve by construction).
  • FAR  points  ← rally_detector's segments whose serve ORIGIN is the far player
                   (classified by the nearest player box at segment open).

The two sets are largely disjoint (near vs. far serves alternate by game on a
single camera).  Overlapping/adjacent segments are merged before cutting.
"""

import argparse
import os

from rally_detector import collect_rally_segments
from run_anya import _collect_segments
from utilities import create_highlights_ffmpeg


# Segments within this gap (seconds) are fused when unioning the two detectors,
# absorbing small boundary differences where both happen to fire on one point.
COMBINED_MERGE_GAP_SEC = 1.0


def _merge_overlaps(segments, gap_sec=COMBINED_MERGE_GAP_SEC):
    """Sort (start, end) segments and fuse those that overlap or are within gap_sec."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(s) for s in merged]


def detect_combined(video_path, output_path=None, headless=False, start_frame=0):
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_combined.mp4")

    # ── Near side: run_anya's serve-gated state machine ───────────────────
    print(f"\n{'='*60}\n  COMBINED DETECTOR\n  Video : {os.path.basename(video_path)}\n{'='*60}")
    print("\n[COMBINED] Pass 1/2 — run_anya (near-side serve-gated) …")
    near_segs, near_points, _ = _collect_segments(video_path, headless, start_frame)
    near_segs = [(s, e) for s, e in near_segs]
    print(f"[COMBINED] run_anya: {len(near_segs)} near-side segment(s)")

    # ── Far side: rally_detector, keep only far-origin segments ───────────
    print("\n[COMBINED] Pass 2/2 — rally_detector (far-side trace) …")
    rally_segs = collect_rally_segments(video_path, headless, start_frame)
    far_segs = [(s, e) for s, e, origin in rally_segs if origin == "far"]
    print(f"[COMBINED] rally_detector: {len(far_segs)} far-origin segment(s) "
          f"(of {len(rally_segs)} total)")

    # ── Union + merge overlaps ────────────────────────────────────────────
    combined = _merge_overlaps(near_segs + far_segs, COMBINED_MERGE_GAP_SEC)
    print(f"\n[COMBINED] Union after overlap-merge: {len(combined)} segment(s)")

    if not combined:
        print("[COMBINED] No segments detected — no output produced.")
        return

    create_highlights_ffmpeg(video_path, combined, output_path)
    print(f"\n[DONE] Output   : {output_path}")
    print(f"[DONE] Segments : {len(combined)}  "
          f"({len(near_segs)} near + {len(far_segs)} far, merged)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined side-aware tennis highlight detector "
                    "(run_anya near + rally_detector far)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combined_detector.py match.mp4
  python combined_detector.py match.mp4 --output combined.mp4 --headless
  python combined_detector.py match.mp4 --start-frame 9000 --headless
""",
    )
    parser.add_argument("video",        metavar="VIDEO", help="Input tennis video")
    parser.add_argument("--output",     default=None,    help="Output MP4 path")
    parser.add_argument("--headless",   action="store_true")
    parser.add_argument("--start-frame", type=int, default=0, metavar="N",
                        help="Start from frame N (default: 0)")
    args = parser.parse_args()

    detect_combined(args.video, args.output, args.headless, args.start_frame)
