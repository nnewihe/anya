"""
config.py
=========
The knobs a CALLER may change per run, one group per stage.

How overrides resolve
---------------------
A field set to `None` means "use the detector module's own constant", so the
default `Anya2Config()` reproduces the shipped behaviour exactly.  The
module constant is the real default and lives next to its explanation (for
example near_serve.THRESHOLD).

Careful: run.py resolves `threshold` (near, far) and `live_hi` (end) with
`value or MODULE_DEFAULT`, so setting them to 0.0 also falls back to the
default.  See docs/anya2/QUIRKS.md #1.

Anything not listed here is not meant to be tuned per run: change it in its
module and re-verify against the corpus.
"""

from dataclasses import dataclass, field
from typing import Optional

from pipeline.anya2.orchestrator import ReelConfig


@dataclass
class PerceiveConfig:
    """The camera track and the two pose passes (perceive.py, camera.py)."""
    pose_fps: float = 15.0        # BOTH pose passes run at this one rate;
                                  # tracks.py interleaves them sample for sample
    near_imgsz: Optional[int] = None   # NOT READ (QUIRKS.md #7); perceive.NEAR_IMGSZ applies
    far_imgsz: Optional[int] = None    # NOT READ (QUIRKS.md #7); perceive.FAR_IMGSZ applies
    device: str = "mps"           # torch device for the pose model
    force: bool = False           # recompute cached camera/pose/end-signal files
    camera_sample_fps: Optional[float] = None
                                  # default camera.SAMPLE_FPS (5.0).  To turn
                                  # camera tracking off use the environment
                                  # variable ANYA_CAMERA_TRACK=0.


@dataclass
class NearServeConfig:
    """Near-side serve detector (near_serve.py)."""
    threshold: Optional[float] = None      # default 0.70
    lead_s: Optional[float] = None         # default 1.63, time from serve
                                           # motion back to the labelled start
    refract_s: Optional[float] = None      # default 3.0, min gap between serves
    require_court: bool = True             # server must stand in the serve
                                           # zone; off lets smashes through
    enabled: bool = True


@dataclass
class FarServeConfig:
    """Far-side serve detector (far_serve.py)."""
    threshold: Optional[float] = None      # default 0.75
    lead_s: Optional[float] = None         # default 0.90
    refract_s: Optional[float] = None      # default 3.0
    w_still: Optional[float] = None        # default 0.30, weight of the
                                           # "stood still before serving" term
    require_court: bool = True             # server must be inside the doubles
                                           # court + 3 ft laterally
    enabled: bool = True


@dataclass
class PointEndConfig:
    """Point-end detector (point_end.py)."""
    live_hi: Optional[float] = None        # default 0.50, enter "live" above
    live_lo: Optional[float] = None        # default 0.35, leave "live" below
    smooth_s: Optional[float] = None       # default 4.0 (detector only; the
                                           # orchestrator always uses 4.0 --
                                           # see QUIRKS.md #3)
    min_live_s: Optional[float] = None     # default 2.0, shortest live run
    enabled: bool = True


@dataclass
class Anya2Config:
    """Everything for one run.  `reel` is the orchestrator's ReelConfig."""
    perceive: PerceiveConfig = field(default_factory=PerceiveConfig)
    near: NearServeConfig = field(default_factory=NearServeConfig)
    far: FarServeConfig = field(default_factory=FarServeConfig)
    end: PointEndConfig = field(default_factory=PointEndConfig)
    reel: ReelConfig = field(default_factory=ReelConfig)

    # Cutting (ffmpeg)
    crf: int = 20                          # x264 quality; lower = better/bigger
    preset: str = "veryfast"
    scale_height: Optional[int] = None     # None = native resolution
    keep_audio: bool = True
