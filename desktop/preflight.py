"""
preflight.py — checks to run before kicking off a (possibly 10+ minute) job.

ffmpeg is not bundled (see rally_app.spec) and both rally_reel and
scoreboard_reel shell out to it only at their very last stage. Without this
check, a tester missing ffmpeg would sit through the entire pipeline before
hitting a raw ``FileNotFoundError`` traceback. Checking up front turns that
into an immediate, actionable message.
"""

import shutil
import sys

from PyQt6.QtWidgets import QMessageBox

_INSTALL_HINT = {
    "darwin": "Open Terminal and run:\n\n    brew install ffmpeg",
    "win32": "Open a terminal and run:\n\n    winget install Gyan.FFmpeg",
}.get(sys.platform, "Open a terminal and run:\n\n    sudo apt install ffmpeg")


def ensure_ffmpeg(parent) -> bool:
    """Return True if ffmpeg is on PATH; otherwise show a dialog and return False."""
    if shutil.which("ffmpeg") is not None:
        return True

    QMessageBox.warning(
        parent,
        "ffmpeg required",
        "Anya Tennis needs ffmpeg installed to render the final video, and it "
        "wasn't found on this machine.\n\n"
        f"{_INSTALL_HINT}\n\n"
        "Then relaunch Anya Tennis and try again.",
    )
    return False
