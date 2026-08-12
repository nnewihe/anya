# Anya Tennis — changelog

Written for beta testers, not for the commit log: this is what `release.sh`
puts on the GitHub release and what people read when the in-app update banner
sends them to the download page. Keep entries plain — what changed for the
person using it. The engineering detail belongs in `version.py`'s comments.

Newest first. Add a section here *before* running `release.sh`; it refuses to
release a version it can't find a section for.

## 0.1.0-beta.5

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
