"""Point-tag project state + persistence — port of the project/tags.json
handling in src/scoreboard/src/App.tsx (a separate, standalone reference
project), plus the bridge into pipeline.rally_reel's segment output.

tags.json schema (unchanged from the reference app, for interop):

    {
      "players": {"a": "...", "b": "..."},
      "format": {...},            # MatchFormat.as_dict(), camelCase accepted too
      "videoName": "...",
      "points": [{"start": 12.4, "end": 31.9, "winner": "A"}, ...]
    }

A `winner` of null/None marks a point whose start/end are known (e.g.
imported from a rally_reel segment) but that hasn't been assigned to a
player yet — a desktop-app-only "pending" state. Export before every point
has a winner and a re-import elsewhere will treat pending points as
incomplete; that's expected, not a bug.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import MatchFormat

TAGS_SUFFIX = "_scoreboard_tags.json"


@dataclass
class PointTag:
    start: float
    end: float
    winner: Optional[str] = None  # "A" | "B" | None (pending)

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "winner": self.winner}

    @classmethod
    def from_dict(cls, d: dict) -> "PointTag":
        return cls(start=float(d["start"]), end=float(d["end"]), winner=d.get("winner"))


@dataclass
class ProjectState:
    names: Dict[str, str] = field(default_factory=lambda: {"a": "Player A", "b": "Player B"})
    format: MatchFormat = field(default_factory=MatchFormat)
    points: List[PointTag] = field(default_factory=list)
    video_name: str = ""

    def as_dict(self) -> dict:
        return {
            "players": self.names,
            "format": self.format.as_dict(),
            "videoName": self.video_name,
            "points": [p.as_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectState":
        return cls(
            names=d.get("players") or {"a": "Player A", "b": "Player B"},
            format=MatchFormat.from_dict(d.get("format") or {}),
            points=[PointTag.from_dict(p) for p in d.get("points") or []],
            video_name=d.get("videoName", ""),
        )


def tags_path_for(video_path: str) -> str:
    """`<video-stem>_scoreboard_tags.json` next to the video — mirrors
    rally_reel's `<stem>_rally_segments.json` naming convention.
    """
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{TAGS_SUFFIX}")


def save_tags(path: str, project: ProjectState) -> None:
    with open(path, "w") as fh:
        json.dump(project.as_dict(), fh, indent=2)


def load_tags(path: str) -> ProjectState:
    with open(path) as fh:
        return ProjectState.from_dict(json.load(fh))


def import_segments_as_points(segments_path: str) -> List[PointTag]:
    """Read a `<stem>_rally_segments.json` written by
    pipeline.rally_reel.reel.build_reel and turn its segments into pending
    (winner=None) PointTags, sorted by start.

    Only `start`/`end` are carried over — a RallySegment's other fields
    (`side`, `serve_t`, `end_method`, `confidence`, ...) are pipeline
    provenance, not scoring input.
    """
    with open(segments_path) as fh:
        payload = json.load(fh)
    segments = payload.get("segments") or []
    points = [PointTag(start=float(s["start"]), end=float(s["end"])) for s in segments]
    points.sort(key=lambda p: p.start)
    return points


def segments_path_for(video_path: str) -> str:
    """Where rally_reel would have written `<stem>_rally_segments.json` for
    this video — used to auto-detect an importable segments file next to a
    freshly loaded video, matching rally_reel.reel's own `_stem_path` naming.
    """
    from pipeline.rally_reel.reel import SEGMENTS_SUFFIX
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{SEGMENTS_SUFFIX}")
