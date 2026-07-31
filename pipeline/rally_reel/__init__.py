"""rally_reel — video in, highlight reel of the active rallies out.

Composes the four existing stages (anya_telemetry, anya_far_serve,
anya_near_serve, walking) into one cached pipeline.  See reel.py for the
stage order.

    python -m pipeline.rally_reel match.mp4
"""

from .config import ReelConfig
from .points import (PointStart, RallySegment, build_segments,
                     enforce_service_runs, find_point_end,
                     merge_serve_starts, usable_walk_intervals, walk_onsets)
from .reel import build_reel

__all__ = [
    "ReelConfig", "PointStart", "RallySegment",
    "build_reel", "build_segments", "enforce_service_runs",
    "find_point_end", "merge_serve_starts", "usable_walk_intervals",
    "walk_onsets",
]
