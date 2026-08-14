# Anya Tennis — changelog

Written for beta testers, not for the commit log: this is what `release.sh`
puts on the GitHub release and what people read when the in-app update banner
sends them to the download page. Keep entries plain — what changed for the
person using it. The engineering detail belongs in `version.py`'s comments.

Newest first. Add a section here *before* running `release.sh`; it refuses to
release a version it can't find a section for.

## 0.1.0-beta.8

- **Point endings are a bit more patient.** Following up on beta.7's fix,
  testing showed the app could still occasionally end a point a little too
  early right after a player started walking. It now waits slightly longer
  before deciding the ball is really gone, which should mean even less live
  tennis gets trimmed from the reel.

## 0.1.0-beta.7

- **Better point endings.** The app used to end a point as soon as it saw a
  player walking, even if the ball was still clearly in play — most often
  when a player walked into a shot rather than running for it, which could
  cut the reel off mid-rally. Walking near the ball no longer ends the
  point; it now takes a longer, more deliberate stretch of the ball being
  out of view before a point is called over. Reels should include less
  accidentally-cut live tennis.

## 0.1.0-beta.6

- **Fixed: the app tried to download part of itself on every run.** It was
  fetching one of its own detection models from the internet at the start of
  each reel instead of using the copy already installed. If you were online you
  never noticed; if you weren't, it failed partway through with a download
  error. It now uses the bundled copy and needs no network at all.

## 0.1.0-beta.5

- **Intel Macs are supported.** There are now two downloads — one for Apple
  silicon (M1 and newer) and one for Intel Macs. The download page picks the
  right one for you. Note that Intel Macs have no graphics acceleration for
  this work, so processing takes roughly an hour for a 7-minute clip instead
  of about ten minutes.

- **No more installing ffmpeg.** Anya Tennis now includes everything it needs.
  If you previously ran `brew install ffmpeg` you can leave it — the app uses
  its own copy either way.
- **Better point endings.** Two model files were missing from the installed
  app that were present when running from source, so the installed version had
  been ending points slightly early. Fixed.
- **Fewer failures on serves.** The near-side serve detector could fail on the
  first serve it looked at in the installed app. Fixed.
- **Update notices.** The app now checks on launch whether a newer version has
  been released and shows a banner if so. It sends nothing about you or your
  video — see the FAQ on the download page.

## 0.1.0-beta.4

- Fixed a crash that happened right as a highlight reel finished. The reel
  itself was fine and already saved, but the app quit instead of telling you.

## 0.1.0-beta.3

- Fixed the app not finding ffmpeg when opened from the Dock or Finder, which
  made it ask you to install something you already had.

## 0.1.0-beta.2

- Roughly twice as fast. A 7-minute 4K clip now takes about 11 minutes on an
  M4, down from about 20.
