#!/usr/bin/env python3
"""Stitch a highlights reel from the tracker's verified ball trace.

    python3 make_highlights.py /path/to/match.mp4
    python3 make_highlights.py match.mp4 --min-len 1.5 --merge-gap 2.0
    python3 make_highlights.py match.mp4 --csv trace.csv   # reuse an existing trace

Runs the real Swift tracker over the video (via run_video_check.sh), keeps the
stretches where the ball is actually tracked, and cuts them into one reel —
the same way pipeline/utilities.py:create_highlights_ffmpeg builds its
_highlights files (ffmpeg -ss/-to per segment, concat demuxer, crf 18, audio
preserved).

Segment logic, matching the request:
  * a "live span" is a contiguous run of moving/coasting frames — the same
    definition the harness reports as live-trace coverage
  * keep spans at least --min-len seconds long (verified trace)
  * merge kept spans whose gap is <= --merge-gap seconds into one continuous
    cut (a brief loss of track mid-rally doesn't chop the reel in two)
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "run_video_check.sh")
LIVE_STATES = {"moving", "coasting"}


def run_tracker(video, csv_path):
    env = dict(os.environ)
    env["DUMP_CSV"] = csv_path
    print(f"[TRACE] running tracker over {os.path.basename(video)} …")
    p = subprocess.run([CHECK, video], env=env, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        sys.exit("tracker failed")
    # Surface the harness's own summary lines (coverage, states, timing).
    for line in p.stdout.splitlines():
        if line.startswith(("video", "frames", "live trace", "states")):
            print("       " + line)


def live_spans(csv_path):
    """Contiguous (start, end) runs of live frames, in source-video seconds."""
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append((float(r["t"]), r["state"] in LIVE_STATES))
    spans = []
    run_start = None
    prev_t = None
    for t, live in rows:
        if live and run_start is None:
            run_start = t
        elif not live and run_start is not None:
            spans.append((run_start, prev_t))
            run_start = None
        prev_t = t
    if run_start is not None:
        spans.append((run_start, prev_t))
    return spans


def build_segments(spans, min_len, merge_gap, pad, duration):
    kept = [(s, e) for s, e in spans if (e - s) >= min_len]
    if not kept:
        return []
    kept.sort()
    merged = [list(kept[0])]
    for s, e in kept[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)   # bridge the gap
        else:
            merged.append([s, e])
    # Pad for watchability, then clamp and re-merge any overlaps the pad created.
    out = []
    for s, e in merged:
        s = max(0.0, s - pad)
        e = e + pad if duration is None else min(duration, e + pad)
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def probe_duration(video):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", video],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def make_reel(video, segments, out_path):
    tmp = tempfile.mkdtemp(prefix="trace_highlights_")
    seg_files = []
    for i, (s, e) in enumerate(segments):
        seg = os.path.join(tmp, f"seg_{i:04d}.mp4")
        cmd = ["ffmpeg", "-y", "-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", video,
               "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-c:a", "aac", "-b:a", "192k", "-vsync", "cfr",
               "-loglevel", "error", seg]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[HIGHLIGHT] segment {i} ({s:.2f}-{e:.2f}s) failed: {r.stderr.strip()[:160]}")
            continue
        seg_files.append(seg)
        print(f"[HIGHLIGHT]   {i+1}/{len(segments)}: {s:6.2f}s - {e:6.2f}s  ({e-s:.2f}s)")
    if not seg_files:
        sys.exit("no segments were extracted")
    concat = os.path.join(tmp, "concat.txt")
    with open(concat, "w") as f:
        for sf in seg_files:
            f.write(f"file '{sf}'\n")
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat,
                        "-c", "copy", "-loglevel", "error", out_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("concat failed: " + r.stderr.strip()[:300])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", help="output path (default <name>_trace_highlights.mp4)")
    ap.add_argument("--csv", help="reuse an existing trace CSV instead of re-running")
    ap.add_argument("--min-len", type=float, default=1.5,
                    help="minimum verified-trace span to keep, seconds (default 1.5)")
    ap.add_argument("--merge-gap", type=float, default=2.0,
                    help="merge kept spans within this gap, seconds (default 2.0)")
    ap.add_argument("--pad", type=float, default=0.3,
                    help="seconds added either side of each cut for watchability (default 0.3)")
    ap.add_argument("--dry-run", action="store_true", help="report segments, no video")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        sys.exit(f"no such video: {a.video}")
    out_path = a.out or (os.path.splitext(a.video)[0] + "_trace_highlights.mp4")

    if a.csv:
        csv_path = a.csv
        print(f"[TRACE] using existing trace {csv_path}")
    else:
        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        run_tracker(a.video, csv_path)

    spans = live_spans(csv_path)
    duration = probe_duration(a.video)
    segments = build_segments(spans, a.min_len, a.merge_gap, a.pad, duration)

    total = sum(e - s for s, e in segments)
    print(f"\n[SEGMENTS] {len(spans)} live span(s) -> "
          f"{len(segments)} segment(s) >= {a.min_len}s, merged within {a.merge_gap}s")
    print(f"[SEGMENTS] {total:.1f}s of highlights"
          + (f" from {duration:.0f}s source" if duration else ""))
    if not segments:
        sys.exit("nothing met the threshold — try a lower --min-len")
    if a.dry_run:
        for s, e in segments:
            print(f"   {s:7.2f}s - {e:7.2f}s  ({e-s:.2f}s)")
        return

    print(f"\n[HIGHLIGHT] {len(segments)} segment(s) -> {out_path}")
    make_reel(a.video, segments, out_path)
    print(f"\ndone: {out_path}")


if __name__ == "__main__":
    main()
