"""
preflight.py — checks to run before kicking off a (possibly 10+ minute) job.

Both rally_reel and scoreboard_reel shell out to ffmpeg only at their very
last stage. Without a check up front, a tester missing ffmpeg would sit
through the entire pipeline before hitting a raw ``FileNotFoundError``
traceback.

The packaged macOS app ships its own static ffmpeg (rally_app.spec /
fetch_ffmpeg.sh), so `ensure_ffmpeg` should never fire there — it remains the
path for source runs, and for Windows/Linux builds where nothing is bundled.

Either way the resolution happens in `repair_path()`, which must run before
anything shells out — see below.
"""

import os
import shutil
import sys
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

_INSTALL_HINT = {
    "darwin": "Open Terminal and run:\n\n    brew install ffmpeg",
    "win32": "Open a terminal and run:\n\n    winget install Gyan.FFmpeg",
}.get(sys.platform, "Open a terminal and run:\n\n    sudo apt install ffmpeg")

# Where package managers put ffmpeg, in the order we'd rather find it.
# Homebrew is /opt/homebrew on Apple silicon and /usr/local on Intel;
# MacPorts is /opt/local; Linux desktop launchers can miss /usr/local/bin
# and a Nix profile the same way.
_CANDIDATE_BINDIRS = {
    "darwin": ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"),
    "win32": (),
}.get(sys.platform, ("/usr/local/bin", os.path.expanduser("~/.nix-profile/bin")))


def bundled_ffmpeg() -> "Path | None":
    """The static ffmpeg shipped inside the app bundle, if there is one.

    Only exists in a PyInstaller build: `sys._MEIPASS` is Contents/Frameworks
    in the macOS .app, which is where rally_app.spec's `binaries` entry puts
    it. Returns None when running from source, or on a platform whose build
    doesn't vendor one.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is None:
        return None
    p = Path(meipass) / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    return p if p.is_file() else None


def repair_path() -> None:
    """Put the bundled ffmpeg, then the standard package-manager bindirs, on PATH.

    A GUI app launched from Finder/Dock is spawned by launchd, not by a shell,
    so it inherits launchd's PATH — `/usr/bin:/bin:/usr/sbin:/sbin` unless
    someone has run `launchctl setenv PATH`. Homebrew's `/opt/homebrew/bin` is
    not on it. Nothing in the app's own environment reflects the PATH the
    tester sees in Terminal, so a machine with a perfectly good
    `brew install ffmpeg` reports ffmpeg missing, and the tester is told to
    install what they already have. It works when run from source purely
    because a terminal-launched process inherits the shell's PATH.

    Mutating os.environ here (rather than resolving ffmpeg to an absolute path)
    is deliberate: every `subprocess.run(["ffmpeg", ...])` in `pipeline/` —
    the final cut, and `proxy.py`'s one-time transcodes — inherits this
    process's environment, so one repair at startup fixes all of them without
    threading a path through the pipeline's call signatures.

    The package-manager bindirs are *appended*: if a tester has deliberately
    put an ffmpeg earlier on PATH, theirs still wins. The bundled one is
    *prepended*, which is the deliberate inversion — it is the exact build the
    release was tested against, and a tester's own ffmpeg is as likely to be a
    four-year-old one missing an encoder as it is to be newer. Shipping a known
    ffmpeg is most of the point of shipping one at all.
    """
    parts = os.environ.get("PATH", "").split(os.pathsep)

    bundled = bundled_ffmpeg()
    prefix = [str(bundled.parent)] if bundled and str(bundled.parent) not in parts else []

    missing = [d for d in _CANDIDATE_BINDIRS if d and d not in parts and os.path.isdir(d)]

    if prefix or missing:
        os.environ["PATH"] = os.pathsep.join(prefix + parts + missing)


def ensure_ffmpeg(parent) -> bool:
    """Return True if ffmpeg is on PATH; otherwise show a dialog and return False."""
    repair_path()
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
