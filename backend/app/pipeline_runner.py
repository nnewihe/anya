"""
pipeline_runner.py
==================
Thin adapter between the job system and the existing rally_detector.py pipeline.

The pipeline (rally_detector.py + anya_base.py + ball_tracker.py + utilities.py
+ models/) lives at the repo root.  We add PIPELINE_DIR to sys.path and import
its two public entry points:

  • collect_rally_segments(video_path, headless, start_frame, progress_cb)
        → list[(start, end, origin)]
  • create_highlights_ffmpeg(video_path, [(start, end), ...], output_path)

Imports are done lazily inside run_rally_job so that importing this module
(e.g. from the API process) does not pull in OpenCV / Ultralytics / Torch.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

from .config import get_settings


def _ensure_pipeline_importable() -> None:
    pdir = str(get_settings().PIPELINE_DIR)
    if pdir not in sys.path:
        sys.path.insert(0, pdir)


# Progress callback: (fraction 0..1, human message) -> None
ProgressFn = Callable[[float, str], None]


def run_rally_job(
    input_path: Path,
    output_path: Path,
    progress: Optional[ProgressFn] = None,
) -> list[dict]:
    """
    Run the rally detector on input_path, write the rally reel to output_path,
    and return the detected segments as a list of dicts:
        {"start": float, "end": float, "origin": "near"|"far"}

    `progress` is invoked periodically with (fraction, message).
    """
    _ensure_pipeline_importable()

    # Imported here (not at module top) so the heavy CV stack only loads inside
    # the worker process.
    from rally_detector import collect_rally_segments  # type: ignore
    from utilities import create_highlights_ffmpeg      # type: ignore

    if progress:
        progress(0.0, "Loading models and probing video…")

    def _cb(current_frame: int, total_frames: int) -> None:
        if progress and total_frames > 0:
            # Reserve the last 5% for the ffmpeg cut step.
            frac = 0.95 * (current_frame / total_frames)
            progress(frac, f"Analyzing frame {current_frame}/{total_frames}")

    segments = collect_rally_segments(
        str(input_path),
        headless=True,
        start_frame=0,
        progress_cb=_cb,
    )

    if progress:
        progress(0.95, f"Detected {len(segments)} rallies — cutting reel…")

    create_highlights_ffmpeg(
        str(input_path),
        [(s, e) for s, e, _ in segments],
        str(output_path),
    )

    if progress:
        progress(1.0, "Done")

    return [
        {"start": float(s), "end": float(e), "origin": str(o)}
        for s, e, o in segments
    ]
