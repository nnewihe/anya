"""
contract.py
===========
The two dataclasses that let three detectors be built independently and still
compose.

`Event` is the only thing a detector emits.  `Requirement` is the only thing it
says about what perception it needs.  Neither carries a threshold, a weight, or
anything else a detector might want to tune -- that is the point: an agent
working on point ends can change every constant it owns without a near-serve
agent having to know.

Why `Requirement` exists at all
-------------------------------
Compute optimization is deliberately the LAST phase of this redesign, not a
constraint during it.  But an optimization pass can only collapse perception if
it knows what each consumer actually reads, and asking three finished detectors
after the fact means reverse-engineering it from their code.  So each declares
it up front, and the collapse becomes a mechanical union over the declarations
rather than an archaeology exercise.

`windows` is the field that matters most there.  A near-serve detector can only
fire between points; a point-end detector can only fire after one has started.
Neither detector knows the other exists -- they just say when they are capable
of firing, and the scheduler intersects that with what the other detectors have
already found.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# Detector kinds.  Also the `--mode` values `eval.py` accepts, and the keys the
# eventual fusion pass will read; kept as constants so a typo is an ImportError
# rather than a silently empty event list.
NEAR_SERVE = "near_serve"
FAR_SERVE = "far_serve"
POINT_END = "point_end"
KINDS = (NEAR_SERVE, FAR_SERVE, POINT_END)

# `Requirement.roi` values -- see perceive.py for why there are two and not one.
ROI_NEAR = "near"
ROI_FAR = "far"
ROI_BOTH = "both"

# `Requirement.windows` values.
W_ALWAYS = "always"                  # must run over the whole clip
W_BETWEEN = "between_points"         # can only fire while no point is live
W_AFTER_SERVE = "after_serve"        # can only fire once a point has started


@dataclass
class Event:
    """One detected point boundary.

    `t` is SECONDS ON THE SOURCE TIMELINE, always -- not proxy frames, not
    decimated pose samples.  Every detector runs on a decimated and/or rescaled
    stream, and having each consumer undo a different stride is exactly how the
    current pipeline accumulated three incompatible notions of "when".
    """
    t: float
    p: float
    kind: str
    track: Optional[int] = None      # which player slot; doubles: who served
    detail: Dict = field(default_factory=dict)   # diagnostics; never read by
                                                 # anything but a human

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}, expected one of {KINDS}")


@dataclass
class Requirement:
    """What a detector needs from the perception layer.

    Declarative and inert: nothing reads it during development.  Phase E's
    scheduler unions these to decide what actually has to run where.
    """
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
    """Write events as JSON, sorted by time.

    The schema is `{"events": [...], **meta}` with each event a flat dict, which
    is what `eval.py` reads.  `meta` is for whatever the detector wants on the
    record -- its Requirement, its thresholds, the git sha -- so a scored run
    can be traced back to the code that produced it.
    """
    import json
    rows = sorted((asdict(e) for e in events), key=lambda r: r["t"])
    with open(path, "w") as fh:
        json.dump({"events": rows, **meta}, fh, indent=1)
    return path


def load_events(path: str) -> List[Event]:
    import json
    with open(path) as fh:
        data = json.load(fh)
    return [Event(**{k: v for k, v in r.items() if k in Event.__dataclass_fields__})
            for r in data.get("events", [])]
