"""Single source of truth for the desktop app's version string.

Bump this for every build handed to beta testers — it's shown in the app
header and embedded in the packaged bundle (rally_app.spec reads it directly),
so a tester's bug report can always be tied to the exact build they ran.
"""

APP_VERSION = "0.1.0-beta.5"
# beta.5 — first build distributed through GitHub Releases rather than by
# hand, and the first that installs with nothing else installed: ffmpeg is
# bundled (fetch_ffmpeg.sh), so no Homebrew step.  Also ships two weight files
# the spec had been omitting — trophy_best.pt, which the near-serve stage
# loads on its first ARMED window, and walking_model_15hz.joblib, without
# which the fast point-end path silently ran the 30 Hz model.  Adds a
# launch-time check for newer releases (update_check.py).
# beta.4 — fixes the SIGABRT right as a reel finished: the completion slot
# dropped the last reference to the still-running QThread.  The reel itself
# was already written before the crash.  See HighlightReelTab._release_worker.
# beta.3 — found ffmpeg when launched from Finder (preflight.repair_path):
# a launchd-spawned app inherits `/usr/bin:/bin:/usr/sbin:/sbin`, so Homebrew's
# ffmpeg was invisible and the app told testers to install what they had.
# DO NOT SHIP beta.2 (no ffmpeg) or beta.3 (crashes on completion).

# beta.2 — partial telemetry.  Every stage now acquires only the telemetry it
# reads (decimated sampling off a shared 540p proxy) instead of sharing one
# full-resolution pass over every frame, so that pass is skipped entirely.
# First-run time went 3x -> 1.6x the clip length (7.0-min 4K clip: ~20 -> 11
# minutes, M4).  Reels differ slightly from beta.1: see ReelConfig.fast_end.
