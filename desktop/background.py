"""
background.py — keep desktop analysis running unattended.

The GUI already runs analysis on a background QThread, so it continues while the
window is unfocused or minimized. The remaining gap is the OS putting the
machine to sleep (or macOS "app nap" throttling a backgrounded app) mid-job, and
the user not knowing when a long job finishes. This module covers both, using
only the standard library (no extra dependencies):

  * ``SleepBlocker`` — prevents idle/system sleep while a job runs.
  * ``notify``       — posts a native "done" notification on completion.

Both are best-effort and degrade to a no-op on unsupported platforms.
"""

import os
import subprocess
import sys


class SleepBlocker:
    """Prevents the system from sleeping while held. Start on job begin, stop on
    finish/error. Safe to start/stop repeatedly and to call on any platform."""

    def __init__(self):
        self._caffeinate = None       # macOS: the caffeinate subprocess
        self._win_state_set = False   # Windows: whether we changed exec state

    def start(self):
        if self._active():
            return
        try:
            if sys.platform == "darwin":
                # -i: prevent idle sleep, -m: prevent disk sleep, -s: prevent
                # system sleep on AC. -w: auto-exit if our process dies.
                self._caffeinate = subprocess.Popen(
                    ["caffeinate", "-ims", "-w", str(os.getpid())]
                )
            elif sys.platform == "win32":
                import ctypes

                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                )
                self._win_state_set = True
            # Linux: left as a no-op (would need systemd-inhibit / D-Bus).
        except Exception:
            # Never let a power-management failure break analysis.
            pass

    def stop(self):
        try:
            if self._caffeinate is not None:
                self._caffeinate.terminate()
                self._caffeinate = None
            if self._win_state_set:
                import ctypes

                ES_CONTINUOUS = 0x80000000
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                self._win_state_set = False
        except Exception:
            pass

    def _active(self):
        return self._caffeinate is not None or self._win_state_set


def notify(title, message):
    """Post a native desktop notification. Best-effort; silent on failure."""
    try:
        if sys.platform == "darwin":
            safe_msg = message.replace('"', "'")
            safe_title = title.replace('"', "'")
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{safe_msg}" with title "{safe_title}"'],
                check=False,
            )
        elif sys.platform.startswith("linux"):
            subprocess.run(["notify-send", title, message], check=False)
        # Windows: no reliable stdlib toast; the in-app result panel suffices.
    except Exception:
        pass
