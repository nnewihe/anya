"""
combined_anya.py
================
Single-camera combined near + far serve detector.

Two-pass design
---------------
Pass 1 — NEAR detection (run_anya).  Full video.  Produces near-end serve
         segments (each an ACTIVE rally following a near serve).
Pass 2 — FAR detection (far_anya).  Processes the video but SKIPS the time
         spans already claimed by robust near-serving runs.
Merge  — Near + far segments are tagged by side, merged chronologically into a
         single highlight reel, and written to a combined side-tagged CSV.

Why skipping is valid — tennis serve-side structure
--------------------------------------------------
A near-end serve and a far-end serve can never occur at the same instant.
With a fixed camera, serving alternates ends in pairs of games:

    Game     1  2  3  4  5  6  ...
    Side     A  A  B  B  A  A  ...     (period-4 AABB pattern)

So serves arrive in *runs* of ~2 games from one end, then switch.  The long
changeover (after odd games) falls *inside* a run, so a run-grouping gap
threshold that bridges changeovers captures the whole ~2-game block.  During a
near-serving run no far serve is possible, so Pass 2 can skip those spans
entirely — bypassing the expensive per-frame ball inference (and frame decode)
for roughly half the match, and avoiding spurious far ARMs while the far player
is merely receiving.

Robustness
----------
Only near runs with >= MIN_SERVES_FOR_MASK detected serves build the skip mask,
so a few false near detections inside a far-serving block cannot mask out a real
far run.  Masks are padded slightly at the start (to cover near serve prep) and
end.

Usage
-----
  python combined_anya.py video.mp4
  python combined_anya.py video.mp4 --output combined.mp4 --headless
  python combined_anya.py video.mp4 --headless --start-frame 1800
"""

import argparse
import csv
import os
from typing import List, Tuple

from run_anya import _collect_segments, _group_segments_into_runs
from far_anya import _collect_far_segments
from utilities import create_highlights_ffmpeg


# ── Skip-mask construction constants ──────────────────────────────────────────
# Gap (s) bridged when grouping near serves into a run.  Must span the longest
# in-run changeover (~90-120s) without bridging the break between serving ends.
RUN_GROUP_GAP_SEC   = 240.0
# Minimum detected near serves for a run to be trusted as a real near-serving
# block (and thus mask the far detector there).  Guards against false positives.
MIN_SERVES_FOR_MASK = 3
# Mask padding: extend backward to cover near serve prep (far receiver stands
# still and could otherwise false-ARM); keep the forward pad small so the first
# far serve after the run is not skipped.
MASK_PAD_START_SEC  = 8.0
MASK_PAD_END_SEC    = 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_skip_intervals(
    near_segments: List[Tuple[float, float]],
    gap_sec: float = RUN_GROUP_GAP_SEC,
    min_serves: int = MIN_SERVES_FOR_MASK,
    pad_start: float = MASK_PAD_START_SEC,
    pad_end: float = MASK_PAD_END_SEC,
) -> List[Tuple[float, float]]:
    """
    Group near segments into serving runs and return skip intervals for the far
    pass — one padded (start, end) span per run that has >= min_serves serves.
    """
    runs = _group_segments_into_runs(near_segments, gap_sec)
    intervals: List[Tuple[float, float]] = []
    for i, run in enumerate(runs):
        if len(run) < min_serves:
            print(f"[COMBINED] Near run {i + 1}: {len(run)} serve(s) "
                  f"(< {min_serves}) — NOT masked")
            continue
        run_start = run[0][0]
        run_end   = run[-1][1]
        span = (max(0.0, run_start - pad_start), run_end + pad_end)
        intervals.append(span)
        print(f"[COMBINED] Near run {i + 1}: {len(run)} serves — "
              f"masking far pass over {span[0]:.1f}s–{span[1]:.1f}s")
    return intervals


def _merge_overlaps(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort by start and merge any overlapping/touching (start, end) spans."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s[0])
    merged  = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _fmt_tc(t: float) -> str:
    """Seconds → H:MM:SS.cs timecode."""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _write_combined_csv(
    csv_path: str,
    tagged_segments: List[Tuple[str, float, float]],
) -> None:
    """Write a segment-level, side-tagged CSV sorted chronologically."""
    cols = ["index", "side", "start_sec", "end_sec", "start_tc", "duration_sec"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, (side, s, e) in enumerate(tagged_segments, 1):
            w.writerow({
                "index":        i,
                "side":         side,
                "start_sec":    round(s, 3),
                "end_sec":      round(e, 3),
                "start_tc":     _fmt_tc(s),
                "duration_sec": round(e - s, 3),
            })


# ─────────────────────────────────────────────────────────────────────────────
# Combined pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_combined_pipeline(
    video_path: str,
    output_path: str = None,
    headless: bool = False,
    start_frame: int = 0,
):
    """
    Run near detection, then far detection (skipping near-run spans), and merge.

    Returns the chronologically-sorted, side-tagged segment list:
        [(side, start_sec, end_sec), ...]  with side in {"NEAR", "FAR"}.
    """
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_combined_highlights.mp4")

    base_no_ext  = os.path.splitext(output_path)[0]
    near_csv     = base_no_ext + "_near_telemetry.csv"
    far_csv      = base_no_ext + "_far_telemetry.csv"
    combined_csv = base_no_ext + "_segments.csv"

    # ── Pass 1: NEAR (full video) ─────────────────────────────────────────
    print(f"\n{'=' * 60}\n  COMBINED PIPELINE — PASS 1 / 2: NEAR\n{'=' * 60}")
    near_segments, near_points, _ = _collect_segments(
        video_path, headless=headless, start_frame=start_frame, csv_path=near_csv
    )
    print(f"[COMBINED] Near pass: {near_points} points, {len(near_segments)} segments")

    # ── Build skip mask from robust near runs ─────────────────────────────
    print(f"\n{'=' * 60}\n  COMBINED PIPELINE — BUILDING FAR SKIP MASK\n{'=' * 60}")
    skip_intervals = _build_skip_intervals(near_segments)
    if not skip_intervals:
        print("[COMBINED] No near runs qualified — far pass will scan the full video.")

    # ── Pass 2: FAR (skips near-run spans) ────────────────────────────────
    print(f"\n{'=' * 60}\n  COMBINED PIPELINE — PASS 2 / 2: FAR\n{'=' * 60}")
    far_segments, far_points, _, _ = _collect_far_segments(
        video_path, headless=headless, start_frame=start_frame,
        csv_path=far_csv, skip_intervals=skip_intervals,
    )
    print(f"[COMBINED] Far pass: {far_points} serves, {len(far_segments)} segments")

    # ── Merge + tag ───────────────────────────────────────────────────────
    tagged = ([("NEAR", s, e) for s, e in near_segments] +
              [("FAR",  s, e) for s, e in far_segments])
    tagged.sort(key=lambda x: x[1])

    _write_combined_csv(combined_csv, tagged)

    # Reel from overlap-merged time ranges (avoids duplicate footage if a near
    # and far segment ever overlap at a boundary).
    reel_segments = _merge_overlaps([(s, e) for _, s, e in tagged])

    # ── Report ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}\n  COMBINED RESULTS\n{'=' * 60}")
    print(f"  Near serves : {near_points:>3}  ({len(near_segments)} segments)")
    print(f"  Far serves  : {far_points:>3}  ({len(far_segments)} segments)")
    print(f"  Total       : {near_points + far_points:>3}  "
          f"({len(reel_segments)} reel segments after overlap-merge)")
    print(f"{'=' * 60}")
    for i, (side, s, e) in enumerate(tagged, 1):
        print(f"  {i:>3}. [{side:<4}] {_fmt_tc(s)}  ({s:.2f}s – {e:.2f}s)")
    print(f"{'=' * 60}")

    if reel_segments:
        create_highlights_ffmpeg(video_path, reel_segments, output_path)
        print(f"\n[COMBINED] Output video  : {output_path}")
    else:
        print("\n[COMBINED] No segments to export.")

    print(f"[COMBINED] Combined CSV  : {combined_csv}")
    print(f"[COMBINED] Near telemetry: {near_csv}")
    print(f"[COMBINED] Far telemetry : {far_csv}")
    return tagged


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined near + far single-camera serve detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combined_anya.py video.mp4
  python combined_anya.py video.mp4 --output combined.mp4 --headless
  python combined_anya.py video.mp4 --headless --start-frame 1800
""",
    )
    parser.add_argument("video", help="Input video file.")
    parser.add_argument("--output",      default=None,
                        help="Output highlights MP4 (default: <video>_combined_highlights.mp4).")
    parser.add_argument("--headless",    action="store_true",
                        help="Run without display windows.")
    parser.add_argument("--start-frame", type=int, default=0, metavar="N",
                        help="Start processing from this frame number (default: 0).")
    args = parser.parse_args()

    run_combined_pipeline(
        args.video,
        output_path=args.output,
        headless=args.headless,
        start_frame=args.start_frame,
    )
