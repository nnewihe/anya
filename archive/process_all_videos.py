"""
process_all_videos.py
=====================
Batch pipeline runner: serve detection → point-end detection for every
snippet.mp4 found under /Volumes/Anya/Data/##/.

Usage:
    python -m src.ai.process_all_videos
    python -m src.ai.process_all_videos --highlights
    python -m src.ai.process_all_videos --matches 21 22 35
"""

import argparse
import glob
import os
import time
import traceback
from pathlib import Path

DATA_ROOT = Path("/Volumes/Anya/Data")


def find_videos(match_ids=None):
    """
    Return sorted list of snippet.mp4 paths under DATA_ROOT/##/.
    Optionally restricted to the given match_ids (list of ints or strings).
    """
    pattern = str(DATA_ROOT / "??" / "snippet.mp4")
    all_videos = sorted(glob.glob(pattern))

    if match_ids:
        wanted = {f"{int(m):02d}" for m in match_ids}
        all_videos = [
            p for p in all_videos
            if Path(p).parent.name in wanted
        ]

    return all_videos


def process_video(video_path: str, make_highlights: bool) -> dict:
    """
    Run the full pipeline for one video.
    Returns a result dict with timing and status.
    """
    from src.ai.serve_detector import run_pipeline as run_serve
    from src.ai.point_end_detector import run_pipeline as run_point_end
    from src.ai.point_end_detector import create_point_highlights

    match_id = Path(video_path).parent.name
    result   = {"match": match_id, "video": video_path, "status": "ok",
                "serves": 0, "points": 0, "error": None}

    t0 = time.time()

    serve_candidates = run_serve(video_path)
    result["serves"] = len(serve_candidates)

    point_candidates = run_point_end(video_path, serve_candidates=serve_candidates)
    result["points"] = len(point_candidates)

    if make_highlights and point_candidates:
        create_point_highlights(video_path, point_candidates)

    result["elapsed_sec"] = round(time.time() - t0, 1)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch serve + point-end detection for all snippet videos"
    )
    parser.add_argument(
        "--matches", nargs="+", metavar="ID",
        help="Process only these match IDs (e.g. --matches 21 22 35)"
    )
    parser.add_argument(
        "--highlights", action="store_true",
        help="Also create a highlights video for each match"
    )
    args = parser.parse_args()

    if not DATA_ROOT.exists():
        print(f"[ERROR] Data root not found: {DATA_ROOT}")
        print("        Is the drive mounted?")
        return

    videos = find_videos(args.matches)
    if not videos:
        print(f"[ERROR] No snippet.mp4 files found under {DATA_ROOT}")
        return

    print(f"Found {len(videos)} video(s):")
    for v in videos:
        print(f"  {v}")
    print()

    results = []
    for i, video_path in enumerate(videos, 1):
        match_id = Path(video_path).parent.name
        print(f"\n{'='*60}")
        print(f"[{i}/{len(videos)}]  Match {match_id}  —  {video_path}")
        print('='*60)

        try:
            r = process_video(video_path, args.highlights)
            results.append(r)
            print(f"\n[MATCH {match_id}] Done — "
                  f"{r['serves']} serve(s), {r['points']} point(s)  "
                  f"({r['elapsed_sec']}s)")
        except Exception as e:
            results.append({
                "match":   match_id,
                "video":   video_path,
                "status":  "error",
                "error":   str(e),
                "serves":  0,
                "points":  0,
            })
            print(f"\n[MATCH {match_id}] FAILED: {e}")
            traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    ok      = [r for r in results if r["status"] == "ok"]
    failed  = [r for r in results if r["status"] == "error"]

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE  —  {len(ok)}/{len(results)} succeeded")
    print('='*60)

    if ok:
        print("\nSucceeded:")
        for r in ok:
            print(f"  Match {r['match']:>2}  {r['serves']:2d} serve(s)  "
                  f"{r['points']:2d} point(s)  ({r.get('elapsed_sec', '?')}s)")

    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  Match {r['match']:>2}  {r['error']}")


if __name__ == "__main__":
    main()
