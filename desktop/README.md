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

**Running from source**, `ffmpeg` must be on PATH — the final cut shells out
to it:

```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Debian/Ubuntu
winget install Gyan.FFmpeg   # Windows
```

The **packaged macOS app bundles its own** (see *Build a distributable*), so
testers install nothing.

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
cd desktop && ./fetch_ffmpeg.sh && pyinstaller rally_app.spec
```

- macOS → `dist/Anya Tennis.app`
- Windows / Linux → `dist/AnyaTennis/`

Weights are pulled from the repo automatically (`pipeline/models/`,
`walking/outputs/`); nothing to copy by hand. Expect a **large bundle (~2 GB)** —
torch and ultralytics are required by the telemetry and pose stages and cannot be
excluded.

### The bundled ffmpeg

`fetch_ffmpeg.sh` puts a static arm64 ffmpeg in `desktop/vendor/` (gitignored,
reproducible from a pinned SHA-256), and the spec ships it inside the bundle.
`preflight.repair_path()` then puts that directory **first** on PATH, so every
`subprocess.run(["ffmpeg", ...])` under `pipeline/` finds it with no change to
any of them. `build_macos.sh` runs the fetch itself, so a plain
`./build_macos.sh` needs no extra step.

Bundling it is what makes the install single-click: telling a coach to install
Homebrew ends the trial. The binary has to be both **arm64** (Rosetta is not
installed by default on Apple silicon) and **redistributable** — most prebuilt
macOS ffmpeg binaries fail one of those, including the popular
eugeneware/ffmpeg-static arm64 build, which is `--enable-nonfree`. The script's
header records why this particular source was chosen. It is GPL, so
`assets/FFMPEG-LICENSE.txt` and `assets/COPYING.GPLv2` ship in the bundle and
in the DMG's `Licenses/` folder.

Windows/Linux builds still expect ffmpeg on PATH; the spec skips the vendored
Mach-O there and `preflight.ensure_ffmpeg` keeps showing the install hint.

**The two architectures use different ffmpeg builds under different licences,
and this is not tidyable.** A bundled ffmpeg must be both statically linked
and redistributable, and no single project supplies both architectures that
way. imageio-ffmpeg's arm64 wheel is `--enable-gpl` (fine); its **x86_64 wheel
is also `--enable-nonfree`**, which cannot be redistributed by anyone at any
price. eugeneware/ffmpeg-static's arm64 build is nonfree too. evermeet.cx is
clean but x86_64-only. So arm64 comes from a PyPI wheel under **GPLv2** and
Intel from evermeet.cx under **GPLv3**, and `assets/` carries both licence
texts plus a per-architecture notice. `fetch_ffmpeg.sh` re-checks
`ffmpeg -buildconf` for `--enable-nonfree` on every fetch and refuses the
binary if it appears — do not "simplify" it to one source.

## The Intel build

`requirements-intel.txt` pins the whole stack back, and every pin follows from
one fact: **torch's last macOS x86_64 wheel is 2.2.2** (March 2024). Everything
newer is arm64-only on macOS. That forces `numpy<2` (2.2.2 predates the numpy 2
C ABI), which forces `opencv-python<4.12`; `PyQt6==6.5.3` is separate, chosen
because Qt 6.5 is the last LTS supporting macOS 11.

`setup_intel_env.sh` builds the environment. It downloads a **prebuilt
standalone x86_64 CPython** rather than compiling one, because both obvious
routes fail: `arch -x86_64 /usr/bin/python3` is Python 3.9 and unwritable, and
`pyenv install` under Rosetta links the arm64 Homebrew openssl, then — with
Homebrew skipped — produces a Python with no `_lzma`, which **torchvision
imports at module load**, breaking the entire stack. Supplying lzma means
building xz from source first. The standalone build ships a complete stdlib
and its own OpenSSL, and lives entirely under `desktop/`.

PyInstaller freezes the interpreter it runs in, so the Intel build runs it
under that x86_64 interpreter via Rosetta; no flag on an arm64 interpreter can
produce an Intel app. `build_macos.sh` verifies the interpreter's actual
architecture before starting, and checks the finished bundle's executable and
ffmpeg are both the requested architecture — a bundle carrying the other
architecture's ffmpeg signs, notarizes and installs perfectly, then demands
Rosetta at the final render.

**Intel is much slower, and testers must be told.** Measured on an M4 across
all four models at 1080p: arm64+MPS 58.5 ms/frame, arm64 CPU 149 ms/frame,
x86_64 CPU 330 ms/frame. Losing the GPU costs ~2.5x and the x86 path another
~2.2x, so expect roughly **an hour for a 7-minute clip on a 2018–2019 Intel
Mac** against ~10 minutes on Apple silicon, and worse on older machines. The
landing page and the release notes both say so up front.

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

## Shipping it to testers

Two DMGs ship per release — Apple silicon and Intel. Build both:

```bash
cd desktop
# 1. bump APP_VERSION in version.py, add a "## <version>" section to CHANGELOG.md
./setup_intel_env.sh              # once, and after requirements-intel.txt changes
./build_macos.sh arm64  && ./make_dmg.sh arm64
./build_macos.sh x86_64 && ./make_dmg.sh x86_64
./release.sh                      # tags, creates the release, uploads both
```

Output lands in `dist/<arch>/`, kept apart because the two bundles are
otherwise indistinguishable by filename and handing a tester the wrong one
gives them an app that won't launch.

`release.sh` refuses to publish rather than shipping something subtly wrong:
it checks the DMG is stapled (an unstapled one gives every tester a Gatekeeper
warning), that `CHANGELOG.md` has a section for this version, that the tag
doesn't already exist, and that **every package the app imports is tracked in
git** — `pipeline/scoreboard_reel/` was untracked through the whole
beta.2–beta.4 run, so the app built here and failed at import from a clean
clone.

The asset is uploaded as `AnyaTennis.dmg` with no version in the name, which
makes this a permanent URL:

```
https://github.com/nnewihe/anya/releases/latest/download/AnyaTennis.dmg
```

That's what the landing page's Download button and the in-app update banner
both point at, so neither needs touching per release.

**Testers get one URL:** <https://nnewihe.github.io/anya/> — served by GitHub
Pages from `docs/` on `main` (Settings → Pages → Deploy from branch → `main`
/ `docs`). It carries the install steps, the Apple-silicon-only caveat and the
privacy FAQ, which a raw releases page doesn't.

### Update notices

`update_check.py` asks the GitHub Releases API on launch whether a newer
`desktop-v*` tag exists and, if so, `app.py` shows a dismissible banner with a
Download button. Deliberately not a real auto-updater — Sparkle in a
PyInstaller bundle means an embedded framework, a second signing key and a
320 MB background download, for roughly the same time-to-new-build as a banner
and a drag.

It filters on the `desktop-v` prefix because `mobile/` versions independently
in this same repo; `/releases/latest` would eventually hand the desktop app a
Flutter version number.

The check fails silently on any error, runs off the GUI thread, and sends
nothing but an HTTP GET — no identifier, no payload. That matters because the
app header promises *"Runs 100% on this computer — your video is never
uploaded"*, which stays true. `ANYA_NO_UPDATE_CHECK=1` disables it.

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
- **Pin the dependencies.** `constraints-windows.txt` holds CI to the versions
  the macOS build is validated against, because `requirements.txt` only sets
  floors and a runner resolves the newest of everything. Skipping this is not
  cosmetic: the first Windows build picked up ultralytics 8.4.118 against a Mac
  on 8.4.36 and died at launch on `No module named 'matplotlib'`, with an
  opencv 4→5 major bump and an sklearn bump under the pickled
  `walking_model.joblib` queued up behind it. Update it *after* the macOS side
  has been bumped and proven, never before.

**Not yet wired into the release flow.** `release.sh` builds and uploads the
two DMGs only, and <https://nnewihe.github.io/anya/> offers a Mac download; the
Windows installer is still a CI artifact you fetch and attach by hand. Adding
it to `release.sh` means uploading `AnyaTennis-Setup.exe` next to
`AnyaTennis.dmg` on the same `desktop-v*` tag, which is what
`update_check.py` already watches.

## Design

Colours, tagline and logo follow [`DESIGN.md`](../DESIGN.md) and are shared with
the mobile app — black ground, `#E8FF3D` yellow as the single focal accent, and
the logo loaded from `mobile/assets/images/anya_logo_black.svg` (the
white-on-dark variant, named for its target background).
