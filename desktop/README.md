# Anya Tennis — desktop app

> Watch your matches in minutes, not hours.

PyQt6 desktop app around `pipeline.rally_reel`. Runs entirely locally — no
server, no upload. Pick a match video, click the four court corners once, get
back a reel containing only the rallies.

Runs on **macOS, Linux and Windows**.

## Run

```bash
pip install -r desktop/requirements.txt
python desktop/app.py
```

`ffmpeg` must be on PATH — the final cut shells out to it:

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
winget install Gyan.FFmpeg   # Windows
```

## What it does

One pipeline, seven stages, all reported live in the progress bar:

| Stage | Work | Cached as |
|---|---|---|
| 0 | Court corners (interactive, first run only) | `<video>_court_cache.json` |
| 1 | Telemetry — players + raw ball detections | *skipped by default* |
| 2 | Far telemetry + pose (own partial pass) | `<video>_anya_far_telemetry.jsonl` |
| 3 | Far-side serve starts | — |
| 4 | Near-side serve starts (own partial pass) | `<video>_..._near_serve_events.json` |
| 5 | Walking + ball quiet (own partial pass) | `<video>_end_walk_pose.npz` |
| 6 | Segment assembly | `<video>_rally_segments.json` |
| 7 | Cut + concatenate | `<video>_rally_reel.mp4` |

**Nothing runs a full-resolution pass over every frame any more.** Stages 2, 4
and 5 each acquire only the telemetry they actually read — decimated pose and
ball sampling off a shared 540p proxy (plus a native-resolution band proxy for
the far baseline), gated to the windows where their signal can occur. Stage 1
existed to feed all three at full rate; with each of them extracting its own,
it has no consumer left and is skipped outright. That skip is where most of the
speedup lands, not in any one stage.

Every stage caches next to the input video, so a second run on the same match
skips straight to whatever changed. Expect roughly **1.6x the clip length** on a
first run — a 7-minute 4K clip measured 11 minutes end to end on an M4, against
~20 minutes (3x) before the partial passes. Later runs are seconds plus the
ffmpeg cut, and the two proxies are built once per video.

The tradeoff is recorded in `ReelConfig.fast_end` and DESIGN.md 8.6: pooled
point-end recall, precision and timing all improve, but the cheap path truncates
a live point slightly more often (13 vs 11 over 135 labelled ends). Run with
`--no-fast-end` to get the full-rate point-end path back — it also brings
stages 1-2 back, so it costs far more than the point-end difference alone.

The court click happens on the main thread before analysis starts — `init_court`
opens an OpenCV window, which is unsafe from a worker thread.

## How it connects to the pipeline

`app.py` puts the **repo root** on `sys.path` and imports the pipeline as a
package (its modules use relative imports, so bare top-level imports don't
work):

```python
from pipeline.rally_reel import ReelConfig, build_reel
```

**Editing anything under `pipeline/` takes effect on the next run — no rebuild.**
That extends to the stage list itself: stage count, labels and progress all come
from `rally_reel.reel`, and the GUI renders whatever it is handed, so adding or
reordering a stage there needs no change in `app.py`.

Tuning lives in `pipeline/rally_reel/config.py` (`ReelConfig`) — thresholds, roll
padding, point-end policy. The app instantiates a default `ReelConfig`; edit that
file to change behaviour globally, or use the CLI for one-off runs:

```bash
python -m pipeline.rally_reel match.mp4 --dry-run --ball-quiet-mode off
```

## Build a distributable

> **Build with Python >= 3.12.1.** Python 3.12.0 has a CPython bug
> ([cpython#110543](https://github.com/python/cpython/issues/110543)): `code.replace()`
> drops the `CO_FAST_HIDDEN` flag that PEP 709 inlined comprehensions depend on.
> PyInstaller calls `code.replace()` on every collected module, so on 3.12.0 the
> build *succeeds* and the packaged app then dies at startup with
> `NameError: name 'name' is not defined` from `torch/_numpy/_ufuncs.py` (scipy
> breaks identically). Running from source is unaffected — nothing calls
> `code.replace()` — which is why this only ever shows up in the bundle.
> `build_macos.sh` refuses to run on 3.12.0. A ready-made env:
> `pyenv virtualenv 3.12.6 anya-build && pyenv activate anya-build`.

```bash
pip install pyinstaller
cd desktop && pyinstaller rally_app.spec
```

- macOS → `dist/Anya Tennis.app`
- Windows / Linux → `dist/AnyaTennis/`

Weights are pulled from the repo automatically (`pipeline/models/`,
`walking/outputs/`); nothing to copy by hand. Expect a **large bundle (~2 GB)** —
torch and ultralytics are required by the telemetry and pose stages and cannot be
excluded. `ffmpeg` is not bundled and must be present on the target machine.

PyInstaller cannot cross-compile: **each target must be built on its own OS.**
macOS is built locally with `build_macos.sh`; Windows is built in CI (below).
Linux has no build script and remains untested.

### macOS: signing + notarization

A plain `pyinstaller` build (above) is unsigned — Gatekeeper blocks it on any
Mac that isn't the one that built it, showing a tester "Apple could not
verify ... is malware". `build_macos.sh` builds, deep-signs, notarizes and
staples the app in one step, so testers get a clean double-click install.

One-time account setup (already done for this project's Apple ID):

1. A **Developer ID Application** certificate in the login keychain — Xcode
   → Settings → Accounts → select the paid team → Manage Certificates → **+**
   → Developer ID Application. (Requires the Program License Agreement to be
   current — if Xcode reports "PLA Update available", accept it at
   [developer.apple.com/account](https://developer.apple.com/account) first.)
2. Notarization credentials stored under a keychain profile:
   ```bash
   xcrun notarytool store-credentials "anya-notary" \
     --apple-id "<email>" --team-id "<TEAMID>"
   ```
   (interactive — prompts for an app-specific password from
   [appleid.apple.com](https://appleid.apple.com) → Sign-In and Security →
   App-Specific Passwords)

With both in place:

```bash
cd desktop && ./build_macos.sh
```

This cleans `build/`/`dist/`, runs PyInstaller, signs every nested binary
inside-out with the hardened runtime (`entitlements.plist`) plus the app
bundle itself, submits it to Apple for notarization and waits, then staples
the ticket so Gatekeeper can verify it offline. Takes several minutes longer
than a plain build, most of it waiting on Apple's notarization service.

### macOS: DMG for distribution

`build_macos.sh` produces `dist/Anya Tennis.app` — the thing that actually
runs, useful for local testing, but not what you hand to a beta tester.
`make_dmg.sh` wraps it into a disk image with the standard drag-to-Applications
layout, then **separately** signs, notarizes and staples the DMG itself:

```bash
cd desktop && ./make_dmg.sh
```

The DMG needs its own notarization pass because Apple's notarization checks
whatever file actually carries the browser/Finder quarantine flag — that's
the DMG a tester downloads, not the `.app` buried inside it. Stapling the DMG
means the *first* Gatekeeper check (opening the downloaded file) resolves
offline too, instead of requiring a live call to Apple at exactly the moment
a tester double-clicks it. Output: `dist/Anya Tennis <version>.dmg`, named
from `version.py` — that's the file to actually distribute.

### Windows: installer

The Windows equivalent of the DMG is a single
`dist/AnyaTennis-Setup-<version>.exe` produced by
[`installer.iss`](installer.iss) (Inno Setup) on top of the one-folder
PyInstaller output. Testers get one file to double-click, a Start Menu entry
and a working uninstaller.

**Normal path — build it in CI.** There is no Windows machine in this project,
and PyInstaller cannot cross-compile from macOS. Run the
[Build Windows installer](../.github/workflows/build-windows.yml) workflow from
the repo's Actions tab (or push a `desktop-v*` tag), then download the
`AnyaTennis-Setup-<version>` artifact. Takes roughly 20–30 minutes, most of it
pip-installing torch and compressing ~2 GB.

**If you do have a Windows box**, the same script CI runs works locally:

```powershell
cd desktop
.\build_windows.ps1
```

It guards the Python version, checks the app imports before spending 20 minutes
on a bundle that can't launch, runs PyInstaller, sanity-checks the output size,
and compiles the installer. Prerequisites: Python ≥ 3.12.1,
`pip install -r requirements.txt`, and Inno Setup 6.3+
(`winget install JRSoftware.InnoSetup`). Pass `-SkipInstaller` to stop at
`dist\AnyaTennis\`.

Windows-specific notes:

- **The installer is unsigned.** There is no Windows code-signing certificate
  for this project, so SmartScreen shows *"Windows protected your PC"* on first
  run. Testers need to click **More info → Run anyway**. Tell them this up
  front — it looks alarming and it is the most likely reason a beta tester
  quietly gives up. (An OV/EV certificate is the only real fix; EV clears
  SmartScreen immediately, OV only after the build accumulates reputation.)
- **It installs per-user**, into `%LOCALAPPDATA%\Programs\Anya Tennis`, so
  there is no UAC elevation prompt on top of the SmartScreen one.
- **ffmpeg is still required** and still not bundled:
  `winget install Gyan.FFmpeg`. The app probes winget/Chocolatey/Scoop install
  locations directly, so a fresh install is picked up without signing out
  first (Windows only hands a running Explorer the PATH it started with).
- **UPX is disabled on Windows** (`rally_app.spec`) — it corrupts torch and Qt
  DLLs, producing a build that dies with `DLL load failed while importing _C`.
- **Logs** land in `%LOCALAPPDATA%\Anya Tennis\logs\app.log` — ask for this
  file with any bug report.

## Design

Colours, tagline and logo follow [`DESIGN.md`](../DESIGN.md) and are shared with
the mobile app — black ground, `#E8FF3D` yellow as the single focal accent, and
the logo loaded from `mobile/assets/images/anya_logo_black.svg` (the
white-on-dark variant, named for its target background).
