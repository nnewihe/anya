"""
workdir.py
==========
A process-wide override for where per-video artifact files are written and
read: court/exclusion caches, proxies, pose detections, tracks, every
detector's event JSON, the reel JSON, and the segment files a cut passes
through on its way to the final video.

Why one override instead of a parameter on every function
-----------------------------------------------------------
Every artifact-path function in this codebase -- `pipeline.utilities`'s court
and exclusion caches, `pipeline.proxy`'s transcodes, every `pipeline.anya2`
module's `*_path` helper, `pipeline.rally_reel`'s telemetry paths, `walking`'s
pose and detection caches -- independently computes
`os.path.dirname(os.path.abspath(video))` and writes there. That is correct
and load-bearing for every CLI and scoring use of this codebase: the corpus
under /Volumes/Anya/Data depends on artifacts sitting beside the clip they
describe, and every eval script in this repo assumes it.

The desktop app is the one caller that wants those files somewhere else -- a
`tmp_anya` folder beside the input video, so a run's footprint can be found
and cleaned up as a unit. Adding a `work_dir` PARAMETER to every one of those
functions, and to every function that calls them transitively across
`pipeline/`, `pipeline/anya2/` and `walking/`, would touch dozens of call
sites for a feature only one caller uses -- and a single missed call site
would silently split one run's files across two directories, which is worse
than not having the feature.

So instead: one override, set once by the app before a run starts and cleared
when it ends. It is safe as a plain global rather than thread-local because
the app never has two renders live at once -- court calibration runs
synchronously on the MAIN thread before the worker thread that does detection
even starts, and only one worker runs at a time -- so there is never a moment
where two different work dirs need to be visible in the same process. CLI and
eval scripts never call `set_work_dir`, so `artifact_dir` returns the video's
own directory for them, exactly as before this module existed.
"""

import os
import threading

_lock = threading.Lock()
_override = None


def set_work_dir(path):
    """Route every artifact path into `path` instead of beside the video.

    Creates the directory (and any missing parents) if it does not exist yet
    -- this IS the "create tmp_anya if it is not there" step; a caller does
    not need a separate os.makedirs.
    """
    global _override
    with _lock:
        _override = path
        if path:
            os.makedirs(path, exist_ok=True)


def clear_work_dir():
    """Return to the default: every artifact beside its video."""
    set_work_dir(None)


def get_work_dir():
    return _override


def artifact_dir(video_path):
    """Where a per-video artifact belongs: the override if one is set, else
    the video's own directory."""
    return _override or os.path.dirname(os.path.abspath(video_path))
