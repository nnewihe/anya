"""scoreboard_reel — tennis scoring engine + scoreboard-burn-in renderer.

Ports src/scoreboard/ (a separate, standalone browser app + Node/ffmpeg
render script) into Python so the desktop app's Scoreboard tab can tag
points against a raw video (or import pipeline.rally_reel's already-detected
segments), track the score, and render a scored highlight video — all
without leaving the desktop app.

    from pipeline.scoreboard_reel import (
        MatchFormat, PointTag, ProjectState,
        replay_match, display_columns, describe_score,
        import_segments_as_points, segments_path_for, tags_path_for,
        load_tags, save_tags,
        render_scoreboard_video,
    )
"""

from .config import MatchFormat, default_format
from .scoring import ReplayResult, describe_score, display_columns, replay_match
from .tags import (PointTag, ProjectState, import_segments_as_points,
                   load_tags, save_tags, segments_path_for, tags_path_for)
from .render import render_scoreboard_video, find_font

__all__ = [
    "MatchFormat", "default_format",
    "ReplayResult", "replay_match", "display_columns", "describe_score",
    "PointTag", "ProjectState", "import_segments_as_points",
    "load_tags", "save_tags", "segments_path_for", "tags_path_for",
    "render_scoreboard_video", "find_font",
]
