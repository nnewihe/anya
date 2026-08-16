"""Single source of truth for the desktop app's version string.

Bump this for every build handed to beta testers — it's shown in the app
header and embedded in the packaged bundle (rally_app.spec reads it directly),
so a tester's bug report can always be tied to the exact build they ran.
"""

APP_VERSION = "0.1.0-beta.10"
# beta.10 — a Windows run could analyse the first second of a match, report
# success, and hand back an empty reel.  Two independent causes, both fixed.
#   (1) The Windows build bundled no ffmpeg, so proxy.py fell back to the
#       SOURCE path whenever the tester's own ffmpeg could not transcode it
#       (proxy.py returns the source on ANY failure, by design).  Every pass
#       then decoded a 2.7K GoPro file directly through whatever video backend
#       OpenCV had, and that decode died ~53 frames in.  fetch_ffmpeg.ps1 now
#       vendors a pinned gyan.dev build the way fetch_ffmpeg.sh does on macOS,
#       and rally_app.spec hard-fails the build without it.
#   (2) Every extraction pass is `while True: if not cap.grab(): break`, which
#       cannot tell a dead decoder from EOF.  Pass A returned 5 samples out of
#       an expected 2,655, the near extractor read that as "the player was
#       never in the serve zone", and the run completed with 0 rallies.
#       pipeline/utilities.assert_decode_complete now raises when a pass ends
#       short of the frame it was asked to reach; ANYA_ALLOW_TRUNCATED_DECODE=1
#       is the escape hatch.  proxy.py's fallback warnings also go to the log
#       now — a windowed build owns no console, so its prints went nowhere.
# beta.9 — strips the em dashes out of the court-calibration window title and
# its console prompt.  Windows gives an OpenCV window title an ANSI code page
# (cp1252 here), not UTF-8, so the em dash arrived as mojibake in the title
# bar; the same character in the print() can raise UnicodeEncodeError on a
# cp1252 stdout, which on Windows kills calibration before the window is even
# shown.  Neither is reachable on macOS, where both surfaces are UTF-8.  This
# is the second Windows-only court-calibration fix in a row — see beta.8's
# WINDOW_AUTOSIZE note in pipeline/utilities.py for the first.
# beta.8 — widens the walk-ball veto (see beta.7) from 4.0s to 5.0s, matching
# no_walk_quiet_s. Measured on Data/75: 4.0 vs 5.0 differed on only 16 of 180
# serve starts, and on every one of those 16 the wider window was the correct
# call — 4.0 cut a live point short, 5.0 correctly held on. No case went the
# other way. Also evaluated raising the ball detector's inference resolution
# (ball_imgsz 960 -> 1920) to improve the 12.1% raw recall behind this whole
# feature; rejected after measurement showed the recall gain was 98%
# background noise (tree-canopy gaps misread as a ball at the higher
# resolution) at 3.75x the per-frame cost. ball_imgsz stays 960.
# beta.7 — point ends now use the "walk-ball" policy instead of the old union
# of a walking classifier and a ball-quiet fallback. Walking is primary; a
# ball seen while the player walks vetoes the end (the point continues) and
# only 4.0s of continuous ball silence during a walk lets it end. Where no
# walk is detected, 5.0s of ball silence alone ends the point. Both
# thresholds were widened from an initial 1.0s/1.5s after measuring how often
# the ball tracker drops out mid-rally (12.1% per-frame recall on a test
# clip) — the short window was ending live points whenever the tracker
# blinked, e.g. a player walking into a backhand. See ReelConfig.end_policy.
# beta.6 — the packaged app was DOWNLOADING yolov8n-pose.pt at the start of
# every reel instead of using its own bundled copy: anya_end_telemetry's
# EndExtractorConfig.pose_model and walking/extract_pose.py's defaults were the
# bare filename, and ultralytics resolves a bare name by checking the CWD and
# then fetching from the internet.  Invisible in testing because the download
# just worked when online; an Intel tester without a reachable network got a
# download failure mid-run.  Affected every packaged build, both architectures.
# desktop/check_model_paths.py now fails the build on a non-absolute default.
# beta.5 — first build distributed through GitHub Releases rather than by
# hand, and the first that installs with nothing else installed: ffmpeg is
# bundled (fetch_ffmpeg.sh), so no Homebrew step.  Also ships two weight files
# the spec had been omitting — trophy_best.pt, which the near-serve stage
# loads on its first ARMED window, and walking_model_15hz.joblib, without
# which the fast point-end path silently ran the 30 Hz model.  Adds a
# launch-time check for newer releases (update_check.py).  First release to
# ship TWO DMGs: Apple silicon and Intel.  The Intel build is pinned to
# torch 2.2.2 (its last macOS x86_64 wheel) and has no MPS, so it is roughly
# 5-6x slower — see desktop/README.md for the measurements.
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
