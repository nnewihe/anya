"""
preflight.py — checks to run before kicking off a (possibly 10+ minute) job.

ffmpeg is not bundled (see rally_app.spec) and both rally_reel and
scoreboard_reel shell out to it only at their very last stage. Without this
check, a tester missing ffmpeg would sit through the entire pipeline before
hitting a raw ``FileNotFoundError`` traceback. Checking up front turns that
into an immediate, actionable message.

The check itself needs `repair_path()` first, or it reports ffmpeg missing on
machines that plainly have it — see below.
"""

import os
import shutil
import sys

from PyQt6.QtWidgets import QMessageBox

_INSTALL_HINT = {
    "darwin": "Open Terminal and run:\n\n    brew install ffmpeg",
    "win32": "Open a terminal and run:\n\n    winget install Gyan.FFmpeg",
}.get(sys.platform, "Open a terminal and run:\n\n    sudo apt install ffmpeg")


def _windows_bindirs():
    """Where Windows package managers drop ffmpeg (or its shim).

    Unlike macOS, an app launched from Explorer *does* inherit the user's PATH,
    so this is a second line of defence rather than the primary fix. It matters
    because of when PATH is read: winget/choco/scoop edit the persistent user
    PATH, but already-running processes — including the Explorer that launches
    Anya Tennis — keep the environment they started with. A tester who installs
    ffmpeg and immediately relaunches the app is told it is still missing, and
    the only advice that works is "sign out and back in". Probing the install
    locations directly skips that.

    winget's Links directory holds a shim rather than ffmpeg.exe itself, which
    is why it is listed instead of the versioned Packages\\... path the shim
    points at (that path carries the release number and changes on upgrade).
    """
    local = os.environ.get("LOCALAPPDATA", "")
    home = os.path.expanduser("~")
    program_data = os.environ.get("ProgramData", r"C:\ProgramData")
    return tuple(d for d in (
        os.path.join(local, "Microsoft", "WinGet", "Links") if local else "",
        os.path.join(program_data, "chocolatey", "bin"),
        os.path.join(home, "scoop", "shims"),
        # The two spots a manual unzip conventionally lands in.
        r"C:\ffmpeg\bin",
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "ffmpeg", "bin"),
    ) if d)


# Where package managers put ffmpeg, in the order we'd rather find it.
# Homebrew is /opt/homebrew on Apple silicon and /usr/local on Intel;
# MacPorts is /opt/local; Linux desktop launchers can miss /usr/local/bin
# and a Nix profile the same way.
_CANDIDATE_BINDIRS = {
    "darwin": ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin"),
    "win32": _windows_bindirs() if sys.platform == "win32" else (),
}.get(sys.platform, ("/usr/local/bin", os.path.expanduser("~/.nix-profile/bin")))


def repair_path() -> None:
    """Add the standard package-manager bindirs to PATH if they're missing.

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

    Appends rather than prepends: if a tester has deliberately put an ffmpeg
    earlier on PATH, theirs still wins.

    On Windows the inheritance problem is different but the fix is the same —
    see _windows_bindirs().
    """
    parts = os.environ.get("PATH", "").split(os.pathsep)
    missing = [d for d in _CANDIDATE_BINDIRS if d and d not in parts and os.path.isdir(d)]
    if missing:
        os.environ["PATH"] = os.pathsep.join(parts + missing)


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
