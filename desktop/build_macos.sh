#!/usr/bin/env bash
# build_macos.sh — build, sign, notarize and staple "Anya Tennis.app" for
# direct (non-App-Store) distribution to beta testers.
#
# One-time account setup this depends on (see desktop/README.md):
#   - a "Developer ID Application" certificate in the login keychain
#   - notarization credentials stored under a keychain profile:
#       xcrun notarytool store-credentials "anya-notary" \
#         --apple-id "<email>" --team-id "<TEAMID>"
#
# Usage:
#   cd desktop && ./build_macos.sh            # host architecture (arm64)
#   cd desktop && ./build_macos.sh x86_64     # Intel, needs ./setup_intel_env.sh first
#
# Output for both is dist/<arch>/Anya Tennis.app — kept in separate directories
# because the two builds are otherwise indistinguishable by filename, and
# handing a tester the wrong one produces an app that either won't launch at
# all (Intel binary, no Rosetta) or is silently slower.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${1:-$(uname -m)}"
case "$ARCH" in
arm64|x86_64) ;;
*) echo "error: unsupported architecture '$ARCH' (expected arm64 or x86_64)" >&2; exit 1 ;;
esac

IDENTITY="Developer ID Application: Anderson Nnewihe (696S9GCN96)"
NOTARY_PROFILE="anya-notary"
DIST="dist/$ARCH"
APP="$DIST/Anya Tennis.app"
ZIP="$DIST/AnyaTennis-notarize.zip"

# Which interpreter builds this. The Intel build MUST run PyInstaller under an
# x86_64 interpreter — PyInstaller freezes the environment it is running in, so
# an arm64 interpreter cannot produce an Intel app no matter what flags it is
# given. setup_intel_env.sh creates that interpreter.
if [ "$ARCH" = "x86_64" ] && [ "$(uname -m)" != "x86_64" ]; then
    RUN=(arch -x86_64)
    PY=".venv-intel/bin/python"
    if [ ! -x "$PY" ]; then
        echo "error: $PY not found — run ./setup_intel_env.sh first" >&2
        exit 1
    fi
elif [ "$ARCH" = "x86_64" ]; then
    RUN=()
    PY=".venv-intel/bin/python"
    [ -x "$PY" ] || { echo "error: $PY not found — run ./setup_intel_env.sh first" >&2; exit 1; }
else
    RUN=()
    PY="$(command -v python3)"
fi

# Python 3.12.0 ships a CPython bug (cpython#110543) where code.replace()
# drops the CO_FAST_HIDDEN flag used by PEP 709 inlined comprehensions.
# PyInstaller calls code.replace() on every module, so on 3.12.0 the build
# SUCCEEDS and then the app dies at startup with a NameError from
# torch/_numpy/_ufuncs.py. Fail loudly here instead of shipping that.
if ! "${RUN[@]}" "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:3] != (3, 12, 0) else 1)'; then
    echo "error: Python 3.12.0 cannot build this app (cpython#110543 breaks torch/scipy" >&2
    echo "       in the frozen bundle). Use Python >= 3.12.1, e.g.:" >&2
    echo "         pyenv activate anya-build" >&2
    exit 1
fi

# The interpreter's architecture is what actually determines the output's, so
# check it rather than trusting the argument.
BUILD_ARCH="$("${RUN[@]}" "$PY" -c 'import platform; print(platform.machine())')"
if [ "$BUILD_ARCH" != "$ARCH" ]; then
    echo "error: asked for $ARCH but $PY runs as $BUILD_ARCH" >&2
    exit 1
fi
echo "==> Building for $ARCH using $PY"

if ! security find-identity -v -p codesigning | grep -qF "$IDENTITY"; then
    echo "error: signing identity not found in keychain: $IDENTITY" >&2
    exit 1
fi
if ! xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
    echo "error: notarytool keychain profile '$NOTARY_PROFILE' not set up or invalid" >&2
    exit 1
fi

echo "==> Checking model defaults won't auto-download"
# beta.5 shipped with the reel's pose-model default set to a bare filename,
# which makes ultralytics ignore the bundled weights and download its own at
# the start of every job. It passed every test here because the download
# succeeded; a tester whose machine couldn't reach the internet got a download
# failure mid-run instead.
"${RUN[@]}" "$PY" check_model_paths.py

echo "==> Ensuring the vendored static ffmpeg is present"
# Idempotent: a no-op once vendor/<arch>/ffmpeg is in place and verifies. Run
# here rather than left as a manual step because forgetting it produces an app
# that builds, signs, notarizes and installs cleanly and then fails at the last
# stage of a tester's first 10-minute job.
./fetch_ffmpeg.sh "$ARCH"

echo "==> Cleaning previous build"
# A DMG from a previous run is very likely still mounted — make_dmg.sh's own
# verify step opens one, and testers double-click them. Its backing file lives
# in dist/, so `rm -rf dist` deletes the image out from under a live mount and
# leaves a zombie volume; worse, Finder recreates dist/.DS_Store mid-delete and
# rm then fails with "Directory not empty", aborting the build under `set -e`
# AFTER it has already destroyed the previous artifacts. Detach first.
for vol in /Volumes/"Anya Tennis"*; do
    [ -d "$vol" ] && hdiutil detach "$vol" -force >/dev/null 2>&1 || true
done
# Only this architecture's output — the other one is a valid artifact that a
# release needs, and wiping it would silently make release.sh publish a stale
# DMG for the arch you didn't just build.
rm -rf "build/$ARCH" "$DIST"

echo "==> Running PyInstaller"
# --distpath/--workpath keep the two architectures' outputs apart; the spec
# itself is architecture-agnostic and reads platform.machine() to pick the
# matching vendored ffmpeg and licence.
"${RUN[@]}" "$PY" -m PyInstaller --distpath "$DIST" --workpath "build/$ARCH" rally_app.spec

echo "==> Signing nested binaries (inside-out)"
# codesign --deep on a bundle this large (torch/opencv/ultralytics — hundreds
# of dylibs) is known to sign in the wrong order or skip binaries, which
# passes locally but notarization then rejects with "invalid nested code" or
# "the signature does not include a secure timestamp". Signing every nested
# Mach-O explicitly first, leaves before the app itself, avoids that class of
# failure. `|| true` skips non-Mach-O files codesign refuses to touch (shell
# scripts, data files that happen to carry the executable bit).
find "$APP" \( -name "*.so" -o -name "*.dylib" -o -type f -perm +111 \) -print0 |
    while IFS= read -r -d '' bin; do
        codesign --force --options runtime --timestamp \
            --entitlements entitlements.plist --sign "$IDENTITY" "$bin" 2>/dev/null || true
    done

echo "==> Signing $APP"
codesign --force --options runtime --timestamp \
    --entitlements entitlements.plist --sign "$IDENTITY" "$APP"

echo "==> Verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP"

# The signing sweep above catches ffmpeg via `-type f -perm +111`, but silently
# (`|| true`), so confirm the one nested binary a tester's whole first run
# depends on actually came out signed and runnable.
FFMPEG_IN_APP="$APP/Contents/Frameworks/ffmpeg"
if [ ! -x "$FFMPEG_IN_APP" ]; then
    echo "error: bundled ffmpeg missing from $FFMPEG_IN_APP" >&2
    exit 1
fi
codesign --verify --strict "$FFMPEG_IN_APP"

# The whole point of two builds is that each one runs without Rosetta on its
# target Mac, and nothing else in the pipeline checks it: a bundle carrying the
# other architecture's ffmpeg signs, notarizes and installs perfectly, then
# demands Rosetta at the final render.
FFMPEG_ARCH="$(lipo -archs "$FFMPEG_IN_APP")"
APP_ARCH="$(lipo -archs "$APP/Contents/MacOS/AnyaTennis")"
echo "==> Architectures: app=$APP_ARCH  ffmpeg=$FFMPEG_ARCH  (expected $ARCH)"
if [ "$FFMPEG_ARCH" != "$ARCH" ] || [ "$APP_ARCH" != "$ARCH" ]; then
    echo "error: architecture mismatch — refusing to ship this bundle" >&2
    exit 1
fi
arch -"$ARCH" "$FFMPEG_IN_APP" -hide_banner -version | head -1

echo "==> Zipping for notarization"
ditto -c -k --keepParent "$APP" "$ZIP"

echo "==> Submitting for notarization (can take several minutes)"
xcrun notarytool submit "$ZIP" --keychain-profile "$NOTARY_PROFILE" --wait

echo "==> Stapling notarization ticket"
xcrun stapler staple "$APP"

echo "==> Confirming Gatekeeper accepts it"
spctl --assess --type execute -vv "$APP"

rm -f "$ZIP"
echo "==> Done: $APP is signed, notarized and stapled."
