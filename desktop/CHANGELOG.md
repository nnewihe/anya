# Anya Tennis — changelog

Written for beta testers, not for the commit log: this is what `release.sh`
puts on the GitHub release and what people read when the in-app update banner
sends them to the download page. Keep entries plain — what changed for the
person using it. The engineering detail belongs in `version.py`'s comments.

Newest first. Add a section here *before* running `release.sh`; it refuses to
release a version it can't find a section for.

## 0.2.0

- **Anya Tennis is now a paid app.** $40 a year, or $5 a month. You create an
  account the first time you open it, and everything works the same way once
  you're in.

- **If you tested the beta, you get a free year.** Create your account with
  the same email address you used during the beta and it's applied
  automatically — you won't be asked for a card. Thank you; thirteen builds of
  finding things that were broken is what got it here.

- **Not for you? Press one button and get your money back.** For the first 14
  days there's a **Cancel & refund** button in **Account**, top-right. Your
  subscription ends straight away and the full amount goes back to your card,
  usually within a few business days. No email, no form, no explaining
  yourself. It can be used once per account.

- **It still runs entirely on your Mac, and your video still never leaves it.**
  That hasn't changed and won't. Signing in and paying talk to Google and to
  Stripe — nothing about your matches does. Your card details go to Stripe's
  own page in your browser and never touch the app. There's a full
  [privacy policy](https://nnewihe.github.io/anya/privacy.html) now.

- **You don't need to be online to use it.** The app checks your subscription
  when it opens; if it can't reach the internet it keeps working for up to 14
  days on what it already knows. A reel that's already running never stops
  because your connection dropped.

- **It opens faster when you're signed out**, because it no longer loads the
  match-analysis machinery just to show you the sign-in screen.

- **A match split across several GoPro files can go in as one.** A GoPro
  starts a new file every few gigabytes, so a long match arrives as
  GX010123.MP4, GX020123.MP4, and so on. Select all of them at once and the
  app joins them before it starts — you no longer have to stitch them together
  yourself first. It works out the recording order for you and shows it, so
  you can see it got it right before a long run starts.

## 0.1.0-beta.14

Nothing in this build changes how Anya Tennis works. It exists to tell you
something before it happens.

- **The next version will be a paid app**, and **you get a free year.** You've
  been testing this through thirteen builds and telling me what was broken.
  When the next version arrives, make an account with the same email address
  you use for feedback and the year is applied automatically — you won't be
  asked for a card.

- **After that, and for everyone else:** $40 a year, or $5 a month. If it
  turns out not to be for you, there's a button in the app that cancels and
  refunds you in full, any time within 14 days of your first payment. No
  email, no form, no explaining yourself.

- **Your video is still never uploaded.** That isn't changing and won't.
  Finding the rallies still happens entirely on your own Mac. Signing in
  checks your subscription and nothing else; paying happens on Stripe's own
  page in your browser, so your card details never touch the app.

- **This build is unaffected.** Keep using it exactly as you are for as long
  as you like. Nothing starts until you choose to update.

There's a strip at the top of the window saying the same thing, with the
details behind **What's changing?**. Close it and it stays closed.

## 0.1.0-beta.13

- **A bumped camera no longer wrecks the rest of the video.** The app
  learns where the court is from the four corners you click once, at
  the start. If the camera then got knocked — a nudged tripod, someone
  leaning on the fence — everything after that was measured against a
  court that had moved, and the app quietly began missing points. On a
  49-minute match where the camera was knocked at 10:53, it found 66%
  of the points after the bump against 88% before it. It now finds
  95%, and the reel gained about four minutes of rally that had been
  dropped. Nothing changes for video where the camera never moved, and
  you don't need to re-click anything.

## 0.1.0-beta.12

- **Sound now stays with the picture.** In longer reels the audio slowly
  slid out of sync with the video — barely noticeable at the start and
  clearly wrong by the end. On 60 fps footage it drifted about four
  seconds an hour. Fixed: the drift is now under a single frame no
  matter how long the reel is, on both 30 and 60 fps video. This
  applies to scoreboard videos too.
- **You can cancel a run.** A Cancel button sits next to Build Rally
  Reel while a job is going. Pressing it actually stops the work within
  a few seconds and gives you your machine back — previously the only
  way out was to quit the app, and even then the run kept going until
  it finished. Nothing is saved from a cancelled run, but the court
  setup you clicked for that video is kept, so starting again doesn't
  ask you to do it twice.

## 0.1.0-beta.11

- **A new detection engine.** The app now uses anya2, a rebuilt detector
  that finds near-side serves, far-side serves, and point ends
  independently and then assembles them into the reel, rather than one
  combined pass. In testing it matches or beats the previous engine on
  every one of those, most clearly on point ends.
- **Nothing extra left behind.** Every working file a run creates —
  court setup, detection data, and everything in between — now goes into
  a `tmp_anya` folder next to your video instead of scattering files
  alongside it. A new checkbox on the Highlight Reel page, unchecked by
  default, lets you keep that folder if you want to inspect it or reuse
  it on a rerun; otherwise it's cleaned up automatically once the video
  is done, whether the run succeeds or fails.

## 0.1.0-beta.10

- **Fixes a Windows run that quietly analysed only the first second of a
  match.** It looked like it worked — the job ran to the end and reported no
  errors — but the reel came back empty, because the app had only ever seen
  the opening moment of the video. Windows installs now come with everything
  needed to read your footage, instead of relying on other video software
  being on the machine.
- **The app now tells you when it can't read a whole video** rather than
  carrying on with the part it managed to open. If this happens you get a
  clear error naming the file, instead of an empty reel and no explanation.
- **Windows reads video the same way a Mac does.** The app now insists on the
  same video decoder on both, so footage that works on one works on the other.
  Nothing changes on a Mac.

## 0.1.0-beta.9

- **Court setup works properly on Windows.** The one-time "click the four
  corners" step showed a garbled title bar and, on some machines, stopped
  before the picture appeared at all. Both are fixed. Nothing changes on a
  Mac.

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
