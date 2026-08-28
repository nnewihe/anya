"""
cancel.py
=========
A process-wide "stop what you are doing" flag, checked cooperatively by the
long-running loops in the pipeline so the desktop app's Cancel button can end a
run that has already started.

Why a flag and not a killed thread
----------------------------------
Python cannot interrupt a thread from outside, and the app runs detection on a
QThread precisely so the window stays alive. Killing the worker is therefore
not on the menu; the alternatives are to let a cancelled job run to completion
in the background (which is what the old `_Worker.stop` did -- it set a flag
that only suppressed the RESULT, so the machine stayed pinned for another ten
minutes and the tester could not start a second run) or to have the work check
in periodically and raise. This is the second.

Why a module-level flag rather than a token passed down the call chain
----------------------------------------------------------------------
The same reasoning as `pipeline.workdir`, and for the same shape of problem:
threading a `cancel` parameter down through `anya2.run` -> `perceive` ->
`proxy`, and through `walking` and every detector, would touch dozens of call
sites for one caller's feature, and a call site that was missed would be a
stage that silently ignores Cancel. One flag, set by the app and cleared when
the next run starts, is visible everywhere without any of that.

It is safe as a plain global for the same reason the work dir is: the app never
has two renders live at once. CLI and eval scripts never call `request()`, so
`check()` is a predictable-branch no-op for them.

Where the checks are
--------------------
Cancel is only as responsive as the coarsest loop between check-ins, so the
checks sit in the places that actually hold the clock:

  * `pipeline.proxy._transcode`   -- the one-time 540p/crop transcodes, a
                                     single multi-minute ffmpeg call each, so
                                     these are cancelled by terminating the
                                     child rather than by waiting for it.
  * `pipeline.anya2.perceive`     -- the two pose passes, which are most of the
                                     run; checked once per inference batch.
  * `pipeline.anya2.run`          -- between stages, and per segment in `cut`.

Everything else is short enough that the next check-in is seconds away.
"""

import threading

try:                                        # package import (python -m pipeline.x)
    from .subproc import popen
except ImportError:                         # script import (python pipeline/x.py)
    from subproc import popen

_flag = threading.Event()


class Cancelled(Exception):
    """Raised by `check()` when a cancel has been requested.

    Callers should let this propagate. It is deliberately NOT an
    ``Exception`` subclass that any of the pipeline's broad ``except
    Exception`` handlers should swallow into a warning -- if you add such a
    handler around code that calls `check()`, re-raise this.
    """


def request() -> None:
    """Ask the running job to stop at its next check-in."""
    _flag.set()


def clear() -> None:
    """Arm for a new run. The app calls this when a job STARTS, not when one
    ends: a job that was cancelled may still be unwinding when the tester
    starts the next one, and clearing at the end would race that unwind."""
    _flag.clear()


def requested() -> bool:
    return _flag.is_set()


def check() -> None:
    """Raise `Cancelled` if a cancel is pending. Cheap enough for inner loops."""
    if _flag.is_set():
        raise Cancelled("cancelled by the user")


def run(cmd, poll: float = 0.25, **kwargs):
    """`subproc.run`, but killable.

    A single ffmpeg call can run for minutes, and `subprocess.run` gives no
    way in. This waits on the child in `poll`-second slices and terminates it
    if a cancel arrives, then raises `Cancelled`. The return value is shaped
    like `subprocess.run`'s so callers do not need a second code path.

    `capture_output=True` and `check=True` are honoured; nothing else in this
    codebase passes anything more exotic to a cancellable call.
    """
    import subprocess

    capture = kwargs.pop("capture_output", False)
    check_rc = kwargs.pop("check", False)
    if capture:
        kwargs.setdefault("stdout", subprocess.PIPE)
        kwargs.setdefault("stderr", subprocess.PIPE)

    proc = popen(cmd, **kwargs)
    while True:
        try:
            out, err = proc.communicate(timeout=poll)
            break
        except subprocess.TimeoutExpired:
            if _flag.is_set():
                proc.terminate()
                try:
                    proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                raise Cancelled("cancelled by the user")

    result = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    if check_rc and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=out, stderr=err)
    return result
