#!/usr/bin/env bash
# release.sh — publish the built DMG as a GitHub Release.
#
# Run last, after build_macos.sh and make_dmg.sh:
#   cd desktop && ./build_macos.sh && ./make_dmg.sh && ./release.sh
#
# What this exists to prevent: handing testers a build that is subtly wrong.
# Every check below is something that has either already shipped broken or
# would be invisible until a tester hit it — an unstapled DMG (Gatekeeper
# warning), a version that was never tagged (bug reports that can't be tied to
# a build), source the app imports that only exists on the release machine.
#
# The uploaded asset is named AnyaTennis.dmg, with no version in it, on
# purpose: that makes
#   https://github.com/nnewihe/anya/releases/latest/download/AnyaTennis.dmg
# a permanent URL, which is what the landing page's Download button and the
# in-app update banner both point at. The version is still in the release
# title, the app header and app.log, so a bug report can name its build.
set -euo pipefail
cd "$(dirname "$0")"

REPO="nnewihe/anya"
VERSION="$(python3 -c 'from version import APP_VERSION; print(APP_VERSION)')"
TAG="desktop-v${VERSION}"

# Both architectures ship in one release. The asset names are stable and
# version-free so the landing page and the in-app update banner can link to
# /releases/latest/download/<name> forever; the architecture IS in the name
# because a tester picking wrong gets an app that won't launch.
# Plain functions rather than associative arrays: `declare -A` needs bash 4,
# and macOS ships bash 3.2, so an associative array aborts this script on the
# very machine that builds the release.
ARCHES="arm64 x86_64"

dmg_for() {
    case "$1" in
    arm64)  echo "dist/arm64/Anya Tennis ${VERSION} (Apple Silicon).dmg" ;;
    x86_64) echo "dist/x86_64/Anya Tennis ${VERSION} (Intel).dmg" ;;
    esac
}

asset_for() {
    case "$1" in
    arm64)  echo "dist/AnyaTennis.dmg" ;;
    x86_64) echo "dist/AnyaTennis-Intel.dmg" ;;
    esac
}

echo "==> Releasing ${VERSION} as ${TAG}"

if ! command -v gh >/dev/null; then
    echo "error: gh CLI not installed (brew install gh)" >&2
    exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
    echo "error: gh is not authenticated — run: gh auth login" >&2
    exit 1
fi

# ── Both artifacts are real, Gatekeeper-clean, and the right architecture ──
# Checked before anything is tagged or uploaded: a release carrying only one
# architecture, or an Intel DMG that is secretly an arm64 build, is worse than
# no release — the landing page offers both regardless.
for a in $ARCHES; do
    dmg="$(dmg_for "$a")"
    if [ ! -f "$dmg" ]; then
        echo "error: $dmg not found — run ./build_macos.sh $a && ./make_dmg.sh $a" >&2
        exit 1
    fi
    if ! xcrun stapler validate "$dmg" >/dev/null 2>&1; then
        echo "error: $dmg has no stapled notarization ticket." >&2
        echo "       Publishing it means every tester gets a Gatekeeper warning." >&2
        exit 1
    fi
    # Verify the architecture of the app INSIDE the image rather than trusting
    # the filename, which is just a string this script wrote earlier.
    mnt="$(mktemp -d)"
    hdiutil attach "$dmg" -nobrowse -quiet -mountpoint "$mnt"
    got="$(lipo -archs "$mnt/Anya Tennis.app/Contents/MacOS/AnyaTennis" 2>/dev/null || echo unknown)"
    ff="$(lipo -archs "$mnt/Anya Tennis.app/Contents/Frameworks/ffmpeg" 2>/dev/null || echo unknown)"
    hdiutil detach "$mnt" -quiet -force || true
    rmdir "$mnt" 2>/dev/null || true
    if [ "$got" != "$a" ] || [ "$ff" != "$a" ]; then
        echo "error: $dmg contains app=$got ffmpeg=$ff, expected $a" >&2
        exit 1
    fi
    echo "==> $a DMG verified (stapled, app and ffmpeg both $a)"
done

# ── The source that built it is actually in the repo ───────────────────────
# pipeline/scoreboard_reel/ was untracked for the whole beta.2-beta.4 run: the
# Scoreboard tab imports it, so the app builds on this machine and fails at
# import from a fresh clone. Check the packages the app imports, not the whole
# tree — a dirty tree of research scratch files is normal here and blocking on
# it would just get this script bypassed.
#
# Paths are repo-relative and run against the repo root: `git ls-files` does
# not resolve `../` pathspecs — from desktop/ it reports every one of these as
# untracked, which would make the check pass or fail for the wrong reason.
ROOT="$(git rev-parse --show-toplevel)"
for path in \
    pipeline/rally_reel pipeline/scoreboard_reel pipeline/utilities.py \
    walking desktop \
    pipeline/models/ball_best.pt pipeline/models/yolo26n.pt \
    pipeline/models/yolov8n-pose.pt pipeline/models/trophy_best.pt \
    walking/outputs/walking_model.joblib walking/outputs/walking_model_15hz.joblib \
    mobile/assets/images/anya_logo.png
do
    if ! git -C "$ROOT" ls-files --error-unmatch "$path" >/dev/null 2>&1; then
        echo "error: $path is not tracked in git." >&2
        echo "       The app imports it, so a clean clone of this tag could not" >&2
        echo "       rebuild the release. Commit it (or drop the dependency) first." >&2
        exit 1
    fi
done

# ── Release notes exist and are for THIS version ───────────────────────────
NOTES="$(mktemp)"
trap 'rm -f "$NOTES"' EXIT
awk -v v="## ${VERSION}" '
    $0 == v { on = 1; next }
    on && /^## / { exit }
    on { print }
' CHANGELOG.md > "$NOTES"
# awk exits 0 whether or not it matched, so the emptiness of the file is the
# only signal that the section is missing.
if [ ! -s "$NOTES" ]; then
    echo "error: CHANGELOG.md has no '## ${VERSION}' section." >&2
    echo "       Those notes are what testers read in the release and after" >&2
    echo "       the in-app update banner — don't ship a version without them." >&2
    exit 1
fi
cat >> "$NOTES" <<EOF

---
**Which download?** Apple menu → About This Mac.
- **Apple M1/M2/M3/M4** → \`AnyaTennis.dmg\`
- **Intel Core i5/i7/i9** → \`AnyaTennis-Intel.dmg\`

Requires macOS 11 (Big Sur) or later. Install: open the DMG and drag Anya
Tennis to Applications.

Intel Macs have no GPU acceleration for this work, so processing takes
substantially longer than on Apple silicon — expect roughly an hour for a
7-minute clip on a 2018–2019 Mac, and longer on older ones.
EOF

# ── Tag ────────────────────────────────────────────────────────────────────
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "error: tag $TAG already exists — bump APP_VERSION in version.py" >&2
    exit 1
fi
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
    echo "error: release $TAG already exists on $REPO" >&2
    exit 1
fi

echo "==> Tagging $TAG at $(git rev-parse --short HEAD)"
git tag -a "$TAG" -m "Anya Tennis $VERSION"
git push origin "$TAG"

# ── Publish ────────────────────────────────────────────────────────────────
UPLOADS=()
for a in $ARCHES; do
    src="$(dmg_for "$a")"
    dst="$(asset_for "$a")"
    echo "==> Staging $dst ($(du -h "$src" | cut -f1))"
    cp "$src" "$dst"
    UPLOADS[${#UPLOADS[@]}]="$dst"
done

echo "==> Creating release and uploading both architectures"
gh release create "$TAG" "${UPLOADS[@]}" \
    --repo "$REPO" \
    --title "Anya Tennis $VERSION" \
    --notes-file "$NOTES"

rm -f "${UPLOADS[@]}"

echo "==> Done."
echo "    Release:       https://github.com/${REPO}/releases/tag/${TAG}"
echo "    Apple silicon: https://github.com/${REPO}/releases/latest/download/AnyaTennis.dmg"
echo "    Intel:         https://github.com/${REPO}/releases/latest/download/AnyaTennis-Intel.dmg"
echo "    Send testers:  https://nnewihe.github.io/anya/"
