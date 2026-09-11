"""
config.py
=========
One knob-set per agent, so each role can be adjusted independently.

WHY THIS EXISTS SEPARATELY FROM THE DETECTORS
---------------------------------------------
Each detector keeps its constants at module level, and that is deliberate: the
constant sits next to the paragraph explaining what was measured to choose it,
which is the only reason those numbers are trustworthy.  Moving them wholesale
into a config object would separate every value from its evidence.

So this module does not replace them.  It exposes the handful that a CALLER
legitimately wants to vary per run -- thresholds, timing, and the few weights
that trade recall against precision -- and leaves the measured geometry where it
is documented.  A field of `None` means "use the module's own value", so a
default config reproduces the shipped behaviour exactly.

Anything not here is not meant to be tuned per run; it is meant to be re-derived
against the corpus and re-committed with its evidence.
"""

from dataclasses import dataclass, field
from typing import Optional

from pipeline.anya2.orchestrator import ReelConfig


@dataclass
class PerceiveConfig:
    """The two pose passes. See perceive.py for why the two ROIs differ."""
    pose_fps: float = 15.0        # locked to ONE rate for both ROIs -- tracks
                                  # interleaves them sample for sample
    near_imgsz: Optional[int] = None   # default 640; near players have headroom
    far_imgsz: Optional[int] = None    # default 960; the far player has NONE --
                                       # 768 costs 13 points of far recall
    device: str = "mps"
    force: bool = False
    camera_sample_fps: Optional[float] = None
                                       # default 5.0 -- how often the camera
                                       # track registers a frame against the
                                       # calibration frame.  Raise it only to
                                       # place a jostle more precisely in time;
                                       # a jostle is a step, and the geometry
                                       # either side of it is already right.
                                       # Turn tracking OFF entirely with
                                       # ANYA_CAMERA_TRACK=0, not from here --
                                       # see pipeline/anya2/camera.py.


@dataclass
class NearServeConfig:
    """Agent 1. Pose-only ready -> trophy -> swing."""
    threshold: Optional[float] = None      # default 0.70, top of a flat plateau
    lead_s: Optional[float] = None         # default 1.63, the label convention
    refract_s: Optional[float] = None      # default 3.0
    require_court: bool = True             # the serve-zone band; off = smashes
    enabled: bool = True


@dataclass
class FarServeConfig:
    """Agent 2. Elevation-only trophy, plus stillness and a separate toss score."""
    threshold: Optional[float] = None      # default 0.75
    lead_s: Optional[float] = None         # default 0.90
    refract_s: Optional[float] = None      # default 3.0
    w_still: Optional[float] = None        # default 0.30 -- weight of the
                                           # pre-serve stillness veto, the term
                                           # that took precision 51% -> 65%
    require_court: bool = True
    enabled: bool = True


@dataclass
class RallyConfig:
    """Agent 3. Pose-only rally confidence -- a curve, not events.

    It has no `enabled`, unlike the two serve agents, and that is deliberate:
    since the point-end event stream was removed the orchestrator ends every
    point off this curve, so switching it off would not disable a feature, it
    would leave nothing able to end a point.  The knobs that decide WHERE an
    end lands live on `ReelConfig` (`end_mode`, `end_rel`, `end_dwell_s`),
    because that is a reel decision rather than a perception one.
    """
    smooth_s: Optional[float] = None       # default 5.0; see rally.SMOOTH_S


@dataclass
class Anya2Config:
    """Everything, per agent, plus the orchestrator.

    `reel` is the orchestrator's own config and already carries its rules,
    rolls and recovery settings -- see orchestrator.ReelConfig.
    """
    perceive: PerceiveConfig = field(default_factory=PerceiveConfig)
    near: NearServeConfig = field(default_factory=NearServeConfig)
    far: FarServeConfig = field(default_factory=FarServeConfig)
    rally: RallyConfig = field(default_factory=RallyConfig)
    reel: ReelConfig = field(default_factory=ReelConfig)

    # Cutting
    crf: int = 20
    preset: str = "veryfast"
    scale_height: Optional[int] = None     # None = native resolution
    keep_audio: bool = True
