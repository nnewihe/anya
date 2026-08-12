"""Single source of truth for the desktop app's version string.

Bump this for every build handed to beta testers — it's shown in the app
header and embedded in the packaged bundle (rally_app.spec reads it directly),
so a tester's bug report can always be tied to the exact build they ran.
"""

APP_VERSION = "0.1.0-beta.3"
# beta.3 — fixes a bug that made beta.2 useless when launched from Finder:
# a launchd-spawned app inherits `/usr/bin:/bin:/usr/sbin:/sbin`, so ffmpeg
# in /opt/homebrew/bin was invisible and the app told testers to install what
# they already had.  See preflight.repair_path.  DO NOT SHIP beta.2.

# beta.2 — partial telemetry.  Every stage now acquires only the telemetry it
# reads (decimated sampling off a shared 540p proxy) instead of sharing one
# full-resolution pass over every frame, so that pass is skipped entirely.
# First-run time went 3x -> 1.6x the clip length (7.0-min 4K clip: ~20 -> 11
# minutes, M4).  Reels differ slightly from beta.1: see ReelConfig.fast_end.
