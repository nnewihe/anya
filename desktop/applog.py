"""
applog.py — diagnostic logging for beta testers.

The packaged app runs with ``console=False`` (rally_app.spec), so on a
tester's machine there is no terminal to see tracebacks in — an unhandled
error today just silently vanishes, or at best surfaces as a one-line message
with no stack trace. This module gives every run a log file on disk and
routes both uncaught exceptions and caught-but-reported pipeline errors into
it, so a beta report can come with something more useful than "it didn't
work".
"""

import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from version import APP_VERSION

_LOGGER_NAME = "anya_tennis"


def _ensure(d: Path) -> Path:
    """Best-effort mkdir shared by both directory helpers. Callers (e.g.
    error-message text that just wants the path to display) must get a path
    back even if the directory couldn't be created — actually writing to it
    is the caller's job, which handles that failure separately."""
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return d


def app_data_dir() -> Path:
    """Per-platform directory for state the app keeps between runs.

    Deliberately NOT log_dir(). The signed-in session file lives here, and it
    carries a Firebase refresh token — while docs/index.html tells testers to
    open the log directory and email us app.log when something breaks. A
    credential inside a directory we actively ask people to mail out is a
    credential we have handed away.

    On Windows the two happen to nest (logs are a subdirectory of this), which
    is exactly the layout that shipped before this function existed. On macOS
    they are deliberately far apart: Application Support vs Library/Logs.
    """
    if sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / "Anya Tennis"
    elif sys.platform == "win32":
        import os
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        d = Path(base) / "Anya Tennis"
    else:
        import os
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        d = Path(base) / "anya-tennis"
    return _ensure(d)


def log_dir() -> Path:
    """Per-platform user log directory.

    The macOS path is ``~/Library/Logs/Anya Tennis`` rather than a subdirectory
    of app_data_dir(): it is the platform convention, it is what Console.app
    shows, and it is the path printed in the crash dialog and quoted in
    docs/index.html. Moving it would strand every existing tester's
    instructions, so the two helpers stay separate rather than one being
    written in terms of the other.
    """
    if sys.platform == "darwin":
        d = Path.home() / "Library" / "Logs" / "Anya Tennis"
    else:
        d = app_data_dir() / "logs"
    return _ensure(d)


def log_path() -> Path:
    return log_dir() / "app.log"


def setup_logging():
    """Configure the root logger: rotating file handler + stderr (dev only).

    Call once, as early as possible in main(). Safe to call more than once
    (handlers aren't duplicated). Must never raise: this runs before anything
    else in main(), and on a machine where the log directory can't be created
    (locked-down permissions, read-only home, unmounted volume) a failure
    here would crash the app before sys.excepthook is even installed — in
    the packaged build (console=False) that's a silent no-op launch with
    zero diagnostic info, the exact failure mode this module exists to
    prevent. So the file handler is best-effort; the excepthook installs
    regardless of whether it succeeded.
    """
    logger = logging.getLogger()
    if any(isinstance(h, (RotatingFileHandler, logging.StreamHandler)) for h in logger.handlers):
        sys.excepthook = _excepthook
        return  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    try:
        # 5MB x 3 backups is generous for text logs and bounds disk use over
        # a long-running beta without needing the tester to ever clean it up.
        file_handler = RotatingFileHandler(
            log_path(), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        pass  # no writable log location — still run, just without a log file

    if sys.stderr is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

    logging.getLogger(_LOGGER_NAME).info(
        "==== Anya Tennis %s starting (platform=%s) ====", APP_VERSION, sys.platform
    )
    sys.excepthook = _excepthook


def _excepthook(exc_type, exc_value, exc_tb):
    logging.getLogger(_LOGGER_NAME).error(
        "Unhandled exception:\n%s",
        "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
    )
    _show_crash_dialog(exc_value)


def _show_crash_dialog(exc_value):
    # Imported lazily: if PyQt6 itself is what's broken, this must not be
    # the thing that raises a second exception out of the hook.
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is None:
            return
        QMessageBox.critical(
            None,
            "Anya Tennis — unexpected error",
            f"Something went wrong: {exc_value}\n\n"
            f"Details were saved to:\n{log_path()}\n\n"
            "Please send that file along when you report this.",
        )
    except Exception:
        pass


def logger():
    return logging.getLogger(_LOGGER_NAME)
