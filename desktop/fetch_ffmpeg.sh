#!/usr/bin/env bash
# fetch_ffmpeg.sh — put the static arm64 ffmpeg that gets bundled into
# "Anya Tennis.app" at desktop/vendor/ffmpeg.
#
# Why bundle at all: beta testers are coaches and players, not developers.
# Telling them to install Homebrew and run `brew install ffmpeg` ends the trial
# for most of them — preflight.ensure_ffmpeg's dialog is a dead end, not a fix.
#
# Why THIS binary rather than Homebrew's: a Homebrew ffmpeg is dynamically
# linked against dozens of dylibs under /opt/homebrew, so copying it into the
# bundle produces a binary that only runs on machines that already have
# Homebrew — exactly the machines we don't need to support. This one is fully
# static, so it has no dependency beyond libSystem.
#
# Why a PyPI wheel as the source: it needs to be an arm64 build (Apple's
# Rosetta is NOT installed by default on Apple silicon, so shipping an x86_64
# ffmpeg would make the first render pop a system install prompt or just fail)
# AND it needs to be redistributable. Most prebuilt macOS ffmpeg binaries fail
# one or the other — evermeet.cx is x86_64-only, and the popular
# eugeneware/ffmpeg-static arm64 build is configured --enable-nonfree, which
# makes it undistributable at any price. imageio-ffmpeg's wheel carries a plain
# --enable-gpl arm64 build, and PyPI URLs are immutable and publish a SHA-256,
# which gives us a checksum to pin against.
#
# vendor/ is gitignored: a 47 MB binary has no business in git history, and
# this script plus the pinned hashes make it reproducible.
#
# Usage (idempotent — a no-op once the binary is in place):
#   cd desktop && ./fetch_ffmpeg.sh
set -euo pipefail
cd "$(dirname "$0")"

WHEEL_URL="https://files.pythonhosted.org/packages/40/5c/f3d8a657d362cc93b81aab8feda487317da5b5d31c0e1fdfd5e986e55d17/imageio_ffmpeg-0.6.0-py3-none-macosx_11_0_arm64.whl"
WHEEL_SHA256="b1ae3173414b5fc5f538a726c4e48ea97edc0d2cdc11f103afee655c463fa742"
MEMBER="imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1"
BIN_SHA256="6d175a4743ca50256e89a8cdd731100f9cee33bd79aeea46894d209410dc6617"
DEST="vendor/ffmpeg"

verify() {
    # Both the arch and the hash, because a wrong-arch ffmpeg fails in the
    # least obvious way possible: the build succeeds, the app starts, and the
    # tester only finds out at the very last stage of a 10-minute job.
    [ -f "$DEST" ] || return 1
    [ "$(shasum -a 256 "$DEST" | cut -d' ' -f1)" = "$BIN_SHA256" ] || return 1
    [ "$(lipo -archs "$DEST" 2>/dev/null)" = "arm64" ] || return 1
    return 0
}

if verify; then
    echo "==> $DEST already present and verified (arm64, sha256 ok)"
    exit 0
fi

mkdir -p vendor
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Downloading ffmpeg (21 MB wheel)"
curl -fL --retry 3 --max-time 300 -o "$TMP/ffmpeg.whl" "$WHEEL_URL"

echo "==> Verifying wheel checksum"
got="$(shasum -a 256 "$TMP/ffmpeg.whl" | cut -d' ' -f1)"
if [ "$got" != "$WHEEL_SHA256" ]; then
    echo "error: wheel checksum mismatch" >&2
    echo "  expected $WHEEL_SHA256" >&2
    echo "  got      $got" >&2
    exit 1
fi

echo "==> Extracting $MEMBER"
unzip -o -q -j "$TMP/ffmpeg.whl" "$MEMBER" -d "$TMP"
mv "$TMP/$(basename "$MEMBER")" "$DEST"
chmod +x "$DEST"

if ! verify; then
    echo "error: extracted binary failed verification (arch or checksum)" >&2
    rm -f "$DEST"
    exit 1
fi

# The GPL obliges us to ship the licence alongside the binary, and the DMG is
# where a tester can actually see it.  assets/ holds the tracked copy; this
# puts it next to the binary so rally_app.spec has one place to collect from.
cp assets/FFMPEG-LICENSE.txt vendor/FFMPEG-LICENSE.txt
cp assets/COPYING.GPLv2 vendor/COPYING.GPLv2

echo "==> Done: $DEST"
"$DEST" -hide_banner -version | head -1
