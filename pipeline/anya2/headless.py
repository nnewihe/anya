"""
headless.py
===========
`run.build_reel` for a machine with no one at it: the CLI
(`python -m pipeline.anya2`) and the Raspberry Pi service (`pi/anya_pi`).

The one thing `build_reel` cannot do unattended is court calibration:
`run.ensure_court` opens an OpenCV window when the corners are not cached, and
on a headless box that either crashes (no display) or hangs forever.  So this
seeds the cache first -- from a fixed-camera site profile when one is given
(`site.apply`) -- and refuses, with a message saying how to fix it, when there
is neither a cache nor a profile.  `build_reel` then finds the corners already
cached and never reaches the prompt.
"""

import os
import time

from pipeline.anya2 import court as C
from pipeline.anya2 import run as R
from pipeline.anya2.config import Anya2Config


class NotCalibrated(RuntimeError):
    pass


def default_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def make_config(device=None, scale_height=1080, crf=None, preset=None,
                copy_video=False):
    cfg = Anya2Config()
    cfg.perceive.device = device or default_device()
    cfg.scale_height = scale_height
    cfg.copy_video = bool(copy_video)
    if crf is not None:
        cfg.crf = crf
    if preset:
        cfg.preset = preset
    return cfg


def prepare(videos, site=None, on_progress=None):
    """Join, then make sure the joined video is calibrated. Returns its path."""
    joined = R._join(videos, on_progress)
    if os.path.isfile(C.court_cache_path(joined)):
        return joined
    if site:
        from pipeline.anya2 import site as S
        S.apply(site, joined)          # raises S.NeedsCalibration on a moved mount
        return joined
    raise NotCalibrated(
        f"{os.path.basename(joined)} has no court calibration and no site "
        f"profile was given. Calibrate once on a machine with a screen:\n"
        f"    python -m pipeline.anya2.site save <a video from this camera> <profile dir>\n"
        f"then pass --site <profile dir>.")


def build(videos, output=None, cfg=None, site=None, on_progress=None,
          dry_run=False):
    """Calibrate-if-possible, then `run.build_reel`. Returns (segments, output)."""
    cfg = cfg or make_config()
    prepare(videos, site, on_progress)
    return R.build_reel(videos, output, cfg, on_progress=on_progress,
                        dry_run=dry_run)


class ProgressPrinter:
    """An `on_progress` that prints one line per stage change / 5% step, with
    wall time, so a log (journalctl) doubles as the per-stage benchmark."""

    def __init__(self, sink=print):
        self.sink = sink
        self.t0 = time.time()
        self.stage_t0 = self.t0
        self.last = None

    def __call__(self, i, n, label, frac=None):
        now = time.time()
        key = (i, label)
        if self.last is None or self.last[0] != key:
            if self.last is not None:
                self.sink(f"[stage {self.last[0][0]}/{n}] done: {self.last[0][1]} "
                          f"({now - self.stage_t0:.0f}s)")
            self.stage_t0 = now
            self.last = (key, -1)
            self.sink(f"[stage {i}/{n}] {label}")
        if frac is not None:
            step = int(frac * 20)
            if step > self.last[1]:
                self.last = (key, step)
                self.sink(f"[stage {i}/{n}] {label} {frac:.0%} "
                          f"(+{now - self.stage_t0:.0f}s, total {now - self.t0:.0f}s)")
