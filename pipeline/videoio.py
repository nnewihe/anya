"""
videoio.py — one way to open a video file, for every pass in every package.

Deliberately dependency-free (cv2 and the standard library only) so that
`walking/`, a sibling top-level package, can import it without dragging in
`utilities`' ultralytics/sklearn import cost. `utilities` re-exports
`open_video`, so pipeline modules that already import from there need no new
import line.

Why not just call `cv2.VideoCapture(path)`
-----------------------------------------
Because which decoder that gets you is a property of the machine, not of the
code. OpenCV picks a backend from an internal priority list, and on Windows
the FFmpeg backend lives in a separate DLL (`opencv_videoio_ffmpeg*.dll`) that
is loaded by name at runtime. If that DLL cannot be found — a real hazard in a
PyInstaller bundle, where nothing about the layout resembles a pip install —
OpenCV does not complain. It silently drops to Media Foundation, which opens a
GoPro file, reports a correct frame count from the container, and then fails
partway through the first sequential read.

That is not hypothetical: it shipped. A tester's 531-second match decoded 53
frames, every pass read the result as "the player was never in the serve
zone", and the run produced an empty reel with no error
(`utilities.assert_decode_complete` is the guard that now catches the
consequence; this is the half that prevents the cause).

Asking for `CAP_FFMPEG` by name turns "which decoder did I get?" from an
accident into a question with an answer, and the fallback below keeps a
machine where FFmpeg genuinely is not available working exactly as it did.
"""

import logging
import os

import cv2

_log = logging.getLogger("anya_tennis.videoio")

# Backends already reported for a path, so seven passes over one video do not
# write the same line seven times. Keyed by (path, backend) so a *change* of
# backend mid-run — which would be worth knowing about — still gets logged.
_reported = set()

# Set by the frozen build's runtime hook (desktop/rthook_cv2.py) on Windows.
# Recorded here only so the log line can say whether it was in play.
_DLL_DIR_VAR = "OPENCV_FFMPEG_DLL_DIR"


def _report(video_path: str, backend: str, label: str) -> None:
    key = (video_path, backend)
    if key in _reported:
        return
    _reported.add(key)
    _log.info("[%s] %s opened with the %s backend",
              label, os.path.basename(video_path), backend)


def open_video(video_path: str, label: str = "VIDEO") -> "cv2.VideoCapture":
    """`cv2.VideoCapture` that asks for FFmpeg first and says what it got.

    Returns an unopened capture (rather than raising) when neither backend can
    open the file, because every caller already checks `isOpened()` or handles
    a failed first `grab()`.
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if cap.isOpened():
        _report(video_path, cap.getBackendName() or "FFMPEG", label)
        return cap
    cap.release()

    # No FFmpeg. Whatever OpenCV picks instead may well be fine — AVFoundation
    # on macOS is — so this is a warning and not a failure. It is loud because
    # on Windows it is the signature of the bundled DLL having gone missing,
    # and because the pass that comes next may now truncate.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _log.warning("[%s] could not open %s with any backend",
                     label, os.path.basename(video_path))
        return cap

    backend = cap.getBackendName() or "unknown"
    msg = (f"[{label}] WARN: the FFmpeg backend could not open "
           f"{os.path.basename(video_path)} — falling back to {backend}. "
           f"On Windows this usually means opencv_videoio_ffmpeg*.dll is "
           f"missing from the bundle ({_DLL_DIR_VAR}="
           f"{os.environ.get(_DLL_DIR_VAR, 'unset')}); the fallback decoder "
           f"may stop partway through the file.")
    print(msg)
    _log.warning(msg)
    _report(video_path, backend, label)
    return cap
