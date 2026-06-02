"""
highlights_from_csv.py
======================
Build a highlights video from a full_anya telemetry CSV without re-running
the pipeline.

Each row in the CSV is an ACTIVE frame.  Rows are grouped by the ``serve``
column; the earliest timestamp in a group is the raw segment start and the
latest is the raw segment end.  Separate start/end buffers are prepended and
appended to each segment.  If adjacent buffered segments would overlap, both
buffers are trimmed equally until the segments just touch.

Usage
-----
  python highlights_from_csv.py telemetry.csv video.mp4
  python highlights_from_csv.py telemetry.csv video.mp4 --output highlights.mp4
  python highlights_from_csv.py telemetry.csv video.mp4 --start-buffer 1.0 --end-buffer 0.5
"""

import argparse
import csv
import os
import subprocess
import tempfile
from collections import defaultdict


def _video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return float("inf")


def _resolve_overlaps(raw_segments, start_buf: float, end_buf: float, video_duration: float):
    """
    Apply start/end buffers to each (raw_start, raw_end, side) tuple, then
    resolve any overlaps between adjacent segments by trimming buffers.

    For each overlapping pair (i, i+1):
      overlap = seg[i].end - seg[i+1].start
      Reduce the end buffer of i and the start buffer of i+1 each by overlap/2,
      clamping both to zero.  If one buffer is exhausted first the remainder
      comes from the other buffer.
    """
    if not raw_segments:
        return []

    # Build per-segment mutable buffers
    starts = [max(0.0, rs - start_buf) for rs, _, _ in raw_segments]
    ends   = [min(video_duration, re + end_buf) for _, re, _ in raw_segments]
    sides  = [side for _, _, side in raw_segments]
    raw_s  = [rs for rs, _, _ in raw_segments]
    raw_e  = [re for _, re, _ in raw_segments]

    # Compute effective buffers applied (may already be clamped by video edges)
    sb = [raw_s[i] - starts[i] for i in range(len(raw_segments))]
    eb = [ends[i]  - raw_e[i]  for i in range(len(raw_segments))]

    for i in range(len(raw_segments) - 1):
        overlap = ends[i] - starts[i + 1]
        if overlap <= 0:
            continue

        # Split the trim equally; if one side runs out the other absorbs the rest
        trim_end   = min(eb[i],      overlap / 2)
        trim_start = min(sb[i + 1],  overlap - trim_end)
        # If start buffer couldn't absorb its share, take more from end buffer
        remaining = overlap - trim_end - trim_start
        if remaining > 0:
            extra = min(eb[i] - trim_end, remaining)
            trim_end += extra

        eb[i]       -= trim_end
        sb[i + 1]   -= trim_start
        ends[i]      = raw_e[i]  + eb[i]
        starts[i + 1] = raw_s[i + 1] - sb[i + 1]

        if ends[i] > starts[i + 1] + 1e-6:
            # Buffers are both zero; hard-clamp so segments touch
            mid = (raw_e[i] + raw_s[i + 1]) / 2
            ends[i]       = mid
            starts[i + 1] = mid

    segments = []
    for i in range(len(raw_segments)):
        s, e = starts[i], ends[i]
        if e > s:
            segments.append((s, e, sides[i]))
    return segments


def segments_from_csv(csv_path: str, start_buf: float, end_buf: float,
                      video_duration: float):
    """
    Returns list of (start_sec, end_sec, side) sorted by start time with
    overlap-resolved buffers applied.
    """
    groups: dict[int, list] = defaultdict(list)
    sides:  dict[int, str]  = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            serve = int(row["serve"])
            ts    = float(row["timestamp"])
            groups[serve].append(ts)
            if serve not in sides:
                sides[serve] = row.get("side", "")

    raw_segments = []
    for serve, timestamps in sorted(groups.items()):
        raw_segments.append((min(timestamps), max(timestamps), sides[serve]))

    return _resolve_overlaps(raw_segments, start_buf, end_buf, video_duration)


def create_highlights(video_path: str, segments, output_path: str):
    if not segments:
        print("[HIGHLIGHT] No segments found — nothing to export.")
        return

    print(f"\n[HIGHLIGHT] {len(segments)} segment(s) → {output_path}")
    tmpdir = tempfile.mkdtemp(prefix="anya_csv_highlights_")
    try:
        seg_files = []
        for i, (start, end, side) in enumerate(segments):
            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-i", video_path,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k",
                "-vsync", "cfr",
                seg_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"[HIGHLIGHT] Warning: segment {i+1} failed.")
                print(result.stderr.decode(errors="replace"))
                continue
            seg_files.append(seg_path)
            mins, secs = int(start // 60), start % 60
            print(f"[HIGHLIGHT]   Segment {i+1}/{len(segments)}: "
                  f"{mins}:{secs:05.2f} – {end:.2f}s  [{side}]")

        if not seg_files:
            print("[HIGHLIGHT] No segments were successfully cut.")
            return

        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for sf in seg_files:
                f.write(f"file '{sf}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print("[HIGHLIGHT] Concat failed:")
            print(result.stderr.decode(errors="replace"))
        else:
            print(f"[HIGHLIGHT] Done → {output_path}")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build highlights video from full_anya telemetry CSV"
    )
    parser.add_argument("csv_path",   help="Path to the _telemetry.csv file")
    parser.add_argument("video_path", help="Path to the original source video")
    parser.add_argument("--output",   default=None,
                        help="Output path (default: <csv_stem>_highlights.mp4)")
    parser.add_argument("--start-buffer", type=float, default=0.5,
                        dest="start_buffer",
                        help="Seconds to pad before each segment (default: 0.5)")
    parser.add_argument("--end-buffer",   type=float, default=0.5,
                        dest="end_buffer",
                        help="Seconds to pad after each segment (default: 0.5)")
    args = parser.parse_args()

    if args.output is None:
        stem = os.path.splitext(args.csv_path)[0]
        args.output = stem + "_highlights.mp4"

    duration = _video_duration(args.video_path)
    segments = segments_from_csv(args.csv_path, args.start_buffer,
                                 args.end_buffer, duration)

    print(f"[CSV] {len(segments)} serve segment(s) found in {os.path.basename(args.csv_path)}")
    for i, (s, e, side) in enumerate(segments, 1):
        mins, secs = int(s // 60), s % 60
        print(f"  Serve #{i:>3}: {mins}:{secs:05.2f} – {e:.2f}s  [{side}]  "
              f"({e - s:.1f}s)")

    create_highlights(args.video_path, segments, args.output)


if __name__ == "__main__":
    main()
