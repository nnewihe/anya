#!/usr/bin/env bash
# fetch_ffmpeg.sh — put the static ffmpeg that gets bundled into
# "Anya Tennis.app" at desktop/vendor/<arch>/ffmpeg.
#
#   ./fetch_ffmpeg.sh            # host architecture
#   ./fetch_ffmpeg.sh arm64      # Apple silicon
#   ./fetch_ffmpeg.sh x86_64     # Intel
#
# Why bundle at all: beta testers are coaches and players, not developers.
# Telling them to install Homebrew and run `brew install ffmpeg` ends the trial
# for most of them — preflight.ensure_ffmpeg's dialog is a dead end, not a fix.
#
# Why a different source per architecture, rather than one project's two
# builds: a bundled ffmpeg has to be BOTH statically linked (a Homebrew binary
# is linked against dozens of dylibs under /opt/homebrew, so it only runs on
# machines that already have Homebrew — exactly the machines we don't need to
# support) AND redistributable. Most prebuilt macOS ffmpeg fails one of those,
# and the failure is not consistent within a project:
#
#   * imageio-ffmpeg's arm64 wheel  -> --enable-gpl only.        USABLE
#   * imageio-ffmpeg's x86_64 wheel -> ALSO --enable-nonfree.    UNUSABLE.
#     A nonfree build cannot be redistributed at any price, by anyone. This
#     was found by reading `ffmpeg -buildconf`, not by assuming the two wheels
#     of one project matched. Do not "simplify" this script by pointing both
#     architectures at imageio-ffmpeg.
#   * eugeneware/ffmpeg-static arm64 -> also --enable-nonfree.   UNUSABLE.
#   * evermeet.cx                    -> --enable-gpl --enable-version3,
#     x86_64 only.                                              USABLE (GPLv3)
#
# So arm64 comes from a PyPI wheel and x86_64 from evermeet.cx, and the two
# carry DIFFERENT licences — GPLv2 and GPLv3 respectively. That is why
# assets/ has a per-architecture FFMPEG-LICENSE-*.txt and both COPYING.GPLv2
# and COPYING.GPLv3; make_dmg.sh ships whichever pair matches the build.
#
# vendor/ is gitignored: a ~47 MB binary has no business in git history, and
# this script plus the pinned hashes make it reproducible.
#
# Idempotent — a no-op once the binary is in place and verifies.
set -euo pipefail
cd "$(dirname "$0")"

ARCH="${1:-$(uname -m)}"

case "$ARCH" in
arm64)
    # imageio-ffmpeg 0.6.0, macosx_11_0_arm64 wheel. PyPI URLs are immutable
    # and publish a SHA-256, which is what makes this pinnable at all.
    URL="https://files.pythonhosted.org/packages/40/5c/f3d8a657d362cc93b81aab8feda487317da5b5d31c0e1fdfd5e986e55d17/imageio_ffmpeg-0.6.0-py3-none-macosx_11_0_arm64.whl"
    ARCHIVE_SHA256="b1ae3173414b5fc5f538a726c4e48ea97edc0d2cdc11f103afee655c463fa742"
    MEMBER="imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1"
    BIN_SHA256="6d175a4743ca50256e89a8cdd731100f9cee33bd79aeea46894d209410dc6617"
    ;;
x86_64)
    # evermeet.cx pins by version in the URL. If this 404s, upstream has
    # rotated builds: pick the current version, re-record BOTH hashes, and
    # re-check `ffmpeg -buildconf` for --enable-nonfree before trusting it.
    URL="https://evermeet.cx/ffmpeg/ffmpeg-9.0.1.zip"
    ARCHIVE_SHA256="8a8c9e549983409fe6604b9aa665648b7a5def9407fe814c39c8b2ea7f64a48f"
    MEMBER="ffmpeg"
    BIN_SHA256="e27de05e3a9f9c758f9766d15d1a069fddeed5f725e35d9ab28683be4740dad7"
    ;;
*)
    echo "error: unsupported architecture '$ARCH' (expected arm64 or x86_64)" >&2
    exit 1
    ;;
esac

DEST="vendor/$ARCH/ffmpeg"

verify() {
    # Architecture as well as hash. A wrong-arch ffmpeg fails in the least
    # obvious way possible: the build succeeds, the app starts, and the tester
    # only finds out at the very last stage of a long job. On Apple silicon an
    # x86_64 binary would additionally demand Rosetta, which is not installed
    # by default — the exact prompt bundling exists to avoid.
    [ -f "$DEST" ] || return 1
    [ "$(shasum -a 256 "$DEST" | cut -d' ' -f1)" = "$BIN_SHA256" ] || return 1
    [ "$(lipo -archs "$DEST" 2>/dev/null)" = "$ARCH" ] || return 1
    return 0
}

if verify; then
    echo "==> $DEST already present and verified ($ARCH, sha256 ok)"
    exit 0
fi

mkdir -p "vendor/$ARCH"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "==> Downloading ffmpeg for $ARCH"
curl -fL --retry 3 --max-time 300 -o "$TMP/ffmpeg.archive" "$URL"

echo "==> Verifying archive checksum"
got="$(shasum -a 256 "$TMP/ffmpeg.archive" | cut -d' ' -f1)"
if [ "$got" != "$ARCHIVE_SHA256" ]; then
    echo "error: archive checksum mismatch for $ARCH" >&2
    echo "  expected $ARCHIVE_SHA256" >&2
    echo "  got      $got" >&2
    exit 1
fi

echo "==> Extracting $MEMBER"
unzip -o -q -j "$TMP/ffmpeg.archive" "$MEMBER" -d "$TMP"
mv "$TMP/$(basename "$MEMBER")" "$DEST"
chmod +x "$DEST"

if ! verify; then
    echo "error: extracted binary failed verification (arch or checksum)" >&2
    rm -f "$DEST"
    exit 1
fi

# Belt and braces on the thing that makes this binary shippable at all. The
# x86_64 wheel of the project that supplies our arm64 binary IS nonfree, so
# this is a real hazard, not a theoretical one.
#
# `arch -<arch>` runs the foreign binary through Rosetta when it's installed,
# so an Apple silicon build machine can still check the Intel binary. If
# neither native execution nor Rosetta is available the check is skipped with
# a warning rather than silently passing — a skipped licence check should be
# visible in the build log.
if arch -"$ARCH" /usr/bin/true >/dev/null 2>&1; then
    if arch -"$ARCH" "$DEST" -hide_banner -buildconf 2>/dev/null | grep -q -- "--enable-nonfree"; then
        echo "error: $DEST is built --enable-nonfree and CANNOT be redistributed." >&2
        rm -f "$DEST"
        exit 1
    fi
    echo "==> Licence check: no --enable-nonfree"
else
    echo "warning: cannot execute $ARCH binaries here — skipped the nonfree check" >&2
fi

# The GPL obliges us to ship the licence alongside the binary, and the DMG is
# where a tester can actually see it. assets/ holds the tracked copies; this
# puts the matching pair next to the binary so make_dmg.sh has one place to
# collect from.
cp "assets/FFMPEG-LICENSE-$ARCH.txt" "vendor/$ARCH/FFMPEG-LICENSE.txt"
if [ "$ARCH" = "arm64" ]; then
    cp assets/COPYING.GPLv2 "vendor/$ARCH/COPYING.txt"
else
    cp assets/COPYING.GPLv3 "vendor/$ARCH/COPYING.txt"
fi

echo "==> Done: $DEST"
[ "$ARCH" = "$(uname -m)" ] && "$DEST" -hide_banner -version | head -1 || true
