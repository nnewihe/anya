"""
contract.py
===========
What the detectors hand to the rest of the pipeline.

  Event        one detected point boundary (a serve, or a point end).  The
               ONLY thing a detector emits.
  Requirement  what perception a detector needs.  Declarative; nothing reads
               it at run time.  It is written into each detector's CLI output
               so a future scheduler can work out what must run where.

Neither carries thresholds or weights: those stay inside each detector, so one
detector can be retuned without the others knowing.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

# Event kinds.  Also eval.py's `--mode` values.  Constants so a typo is an
# ImportError rather than a silently empty list.
NEAR_SERVE = "near_serve"
FAR_SERVE = "far_serve"
POINT_END = "point_end"
KINDS = (NEAR_SERVE, FAR_SERVE, POINT_END)

# Requirement.roi -- which pose pass a detector reads (see perceive.py).
ROI_NEAR = "near"
ROI_FAR = "far"
ROI_BOTH = "both"

# Requirement.windows -- when a detector is able to fire at all.
W_ALWAYS = "always"                  # anywhere in the clip
W_BETWEEN = "between_points"         # only while no point is live (serves)
W_AFTER_SERVE = "after_serve"        # only once a point has started (ends)


@dataclass
class Event:
    """One detected point boundary.

    `t` is always SECONDS ON THE SOURCE VIDEO's timeline -- never a pose-sample
    index or a proxy frame number.
    """
    t: float
    p: float                         # confidence, 0..1
    kind: str                        # one of KINDS
    track: Optional[int] = None      # player slot (0,1 near; 2,3 far), if any
    detail: Dict = field(default_factory=dict)   # diagnostics for humans only

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}, expected one of {KINDS}")


@dataclass
class Requirement:
    """What a detector needs from perception.  Validated, never acted on."""
    roi: str
    pose_fps: float
    needs_ball: bool = False
    ball_fps: Optional[float] = None
    windows: str = W_ALWAYS

    def __post_init__(self):
        if self.roi not in (ROI_NEAR, ROI_FAR, ROI_BOTH):
            raise ValueError(f"unknown roi {self.roi!r}")
        if self.windows not in (W_ALWAYS, W_BETWEEN, W_AFTER_SERVE):
            raise ValueError(f"unknown windows {self.windows!r}")
        if self.needs_ball and not self.ball_fps:
            raise ValueError("needs_ball=True requires a ball_fps")
        if not self.needs_ball and self.ball_fps:
            raise ValueError("ball_fps set but needs_ball=False -- say which")


def dump_events(events: List[Event], path: str, **meta) -> str:
    """Write events as `{"events": [...sorted by t...], **meta}` JSON."""
    rows = sorted((asdict(e) for e in events), key=lambda r: r["t"])
    with open(path, "w") as fh:
        json.dump({"events": rows, **meta}, fh, indent=1)
    return path


def load_events(path: str) -> List[Event]:
    with open(path) as fh:
        data = json.load(fh)
    return [Event(**{k: v for k, v in row.items() if k in Event.__dataclass_fields__})
            for row in data.get("events", [])]
