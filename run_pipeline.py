#!/usr/bin/env python3
"""
run_pipeline.py
---------------
1. Finds all video files matching the GoPro naming pattern GH0#0897.MP4
   (where # is a single digit clip-counter) inside a given folder.
2. Sorts them by the clip-counter digit so they are concatenated in order.
3. Uses ffmpeg to concatenate them into match.mp4 in the same folder.
4. Runs:  python3 -m combined_detector  <folder>/match.mp4 --headless

Usage:
    python3 run_pipeline.py /Volumes/Anya/Data/54
    python3 run_pipeline.py /Volumes/Anya/Data/54 --skip-concat   # if match.mp4 already exists
    python3 run_pipeline.py /Volumes/Anya/Data/54 --dry-run        # preview without executing
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile


# ── Pattern ────────────────────────────────────────────────────────────────────
# Matches:  GH0<digit>0897.MP4   (case-insensitive extension)
# Group 1 = the clip-counter digit
FILENAME_RE = re.compile(r'^GH0(\d)0897\.MP4$', re.IGNORECASE)


def find_clips(folder: str) -> list[str]:
    """Return absolute paths of matching clips, sorted by clip-counter digit."""
    entries = os.listdir(folder)
    clips = []
    for name in entries:
        m = FILENAME_RE.match(name)
        if m:
            clips.append((int(m.group(1)), os.path.join(folder, name)))
    clips.sort(key=lambda t: t[0])
    return [path for _, path in clips]


def write_concat_list(clips: list[str], tmp_file) -> None:
    """Write an ffmpeg concat-demuxer list file."""
    for path in clips:
        # ffmpeg requires forward slashes and escaped apostrophes
        safe = path.replace("'", r"'\''")
        tmp_file.write(f"file '{safe}'\n")
    tmp_file.flush()


def concatenate(clips: list[str], output: str, dry_run: bool = False) -> None:
    """Concatenate clips into output using ffmpeg stream-copy (lossless, fast)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                    delete=False, encoding='utf-8') as tmp:
        write_concat_list(clips, tmp)
        list_path = tmp.name

    cmd = [
        'ffmpeg',
        '-y',                      # overwrite output without prompting
        '-f', 'concat',
        '-safe', '0',              # allow absolute paths in the list
        '-i', list_path,
        '-c', 'copy',              # stream-copy: no re-encode
        output,
    ]

    print("\n── ffmpeg concat command ──────────────────────────────────")
    print(' '.join(cmd))
    print()

    if dry_run:
        print("[dry-run] Skipping execution.")
    else:
        result = subprocess.run(cmd, check=False)
        os.unlink(list_path)
        if result.returncode != 0:
            sys.exit(f"ffmpeg failed with exit code {result.returncode}.")
        print(f"\n✓ Concatenated video saved to: {output}\n")


def run_detector(video_path: str, dry_run: bool = False) -> None:
    """Run the combined_detector module against the concatenated video."""
    cmd = [
        sys.executable, '-m', 'combined_detector',
        video_path,
        '--headless',
    ]

    print("── detector command ───────────────────────────────────────")
    print(' '.join(cmd))
    print()

    if dry_run:
        print("[dry-run] Skipping execution.")
    else:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            sys.exit(f"combined_detector exited with code {result.returncode}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concatenate GH0#0897.MP4 clips and run combined_detector."
    )
    parser.add_argument(
        'folder',
        help="Directory containing the GH0#0897.MP4 files (e.g. /Volumes/Anya/Data/54)",
    )
    parser.add_argument(
        '--output-name', default='match.mp4',
        help="Name of the concatenated output file (default: match.mp4)",
    )
    parser.add_argument(
        '--skip-concat', action='store_true',
        help="Skip concatenation and go straight to the detector (match.mp4 must exist)",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Print commands that would be run without executing them",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        sys.exit(f"Error: '{folder}' is not a directory.")

    output = os.path.join(folder, args.output_name)

    # ── Step 1: find & report clips ───────────────────────────────────────────
    clips = find_clips(folder)

    if not args.skip_concat:
        if not clips:
            sys.exit(
                f"No files matching GH0#0897.MP4 were found in:\n  {folder}"
            )

        print(f"Found {len(clips)} clip(s) in order:")
        for p in clips:
            print(f"  {os.path.basename(p)}")

        # ── Step 2: concatenate ───────────────────────────────────────────────
        concatenate(clips, output, dry_run=args.dry_run)

    else:
        if not os.path.isfile(output):
            sys.exit(f"--skip-concat requested but '{output}' does not exist.")
        print(f"Skipping concatenation; using existing: {output}\n")

    # ── Step 3: run detector ──────────────────────────────────────────────────
    run_detector(output, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
