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
DMG="dist/Anya Tennis ${VERSION}.dmg"
ASSET="dist/AnyaTennis.dmg"

echo "==> Releasing ${VERSION} as ${TAG}"

if ! command -v gh >/dev/null; then
    echo "error: gh CLI not installed (brew install gh)" >&2
    exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
    echo "error: gh is not authenticated — run: gh auth login" >&2
    exit 1
fi

# ── The artifact is real and Gatekeeper-clean ──────────────────────────────
if [ ! -f "$DMG" ]; then
    echo "error: $DMG not found — run ./make_dmg.sh first" >&2
    exit 1
fi
if ! xcrun stapler validate "$DMG" >/dev/null 2>&1; then
    echo "error: $DMG has no stapled notarization ticket." >&2
    echo "       Publishing it means every tester gets a Gatekeeper warning." >&2
    exit 1
fi

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
Requires an Apple silicon Mac (M1 or newer), macOS 12 or later.
Install: open the DMG and drag Anya Tennis to Applications.
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
echo "==> Staging asset as $ASSET"
cp "$DMG" "$ASSET"

echo "==> Creating release (uploading $(du -h "$ASSET" | cut -f1))"
gh release create "$TAG" "$ASSET" \
    --repo "$REPO" \
    --title "Anya Tennis $VERSION" \
    --notes-file "$NOTES"

rm -f "$ASSET"

echo "==> Done."
echo "    Release:  https://github.com/${REPO}/releases/tag/${TAG}"
echo "    Download: https://github.com/${REPO}/releases/latest/download/AnyaTennis.dmg"
echo "    Send testers: https://nnewihe.github.io/anya/"
