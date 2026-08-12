"""
subproc.py — subprocess wrapper that stays invisible in a windowed build.

The desktop app is packaged with ``console=False`` (see desktop/rally_app.spec)
because a GUI app must not drag a terminal along behind it.  On Windows that
has a consequence macOS does not have: the frozen process owns no console, so
every child process the pipeline spawns *allocates its own*.  A reel with 40
segments shells out to ffmpeg 41 times, and a tester watching a 10-minute job
sees 41 black console windows blink open and shut over whatever else they were
doing.  It looks broken, and on some machines the stolen focus makes the app
genuinely unusable while a job runs.

``CREATE_NO_WINDOW`` suppresses that allocation.  It exists only on Windows,
so the flag is computed once at import and is simply 0 elsewhere, which makes
``run()`` a transparent pass-through to ``subprocess.run`` on macOS and Linux.

Use this for anything the pipeline shells out to.  A caller that has already
set ``creationflags`` itself wins — the flag is OR-ed into whatever was passed
rather than replacing it.
"""

import subprocess
import sys

# subprocess.CREATE_NO_WINDOW is defined on Windows only; 0 is the documented
# "no special creation flags" value everywhere else, so OR-ing it is a no-op.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def run(cmd, **kwargs):
    """``subprocess.run(cmd, **kwargs)`` with no console window on Windows."""
    if _NO_WINDOW:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | _NO_WINDOW
    return subprocess.run(cmd, **kwargs)


def popen(cmd, **kwargs):
    """``subprocess.Popen(cmd, **kwargs)`` with no console window on Windows."""
    if _NO_WINDOW:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | _NO_WINDOW
    return subprocess.Popen(cmd, **kwargs)
