#!/usr/bin/env bash
# make_dmg.sh — wrap the already-signed, notarized, stapled "Anya Tennis.app"
# (produced by build_macos.sh) into a distributable DMG: drag-to-Applications
# layout, then sign, notarize and staple the DMG itself.
#
# Why notarize the DMG too, separately from the .app inside it: Apple's own
# guidance is to notarize whatever file actually carries the browser/Finder
# quarantine flag — that's the DMG a tester downloads, not the .app buried
# inside it. The .app's own staple (done by build_macos.sh) still matters —
# it's what Gatekeeper checks at launch — but stapling the DMG as well means
# the *first* Gatekeeper check, when the tester opens the downloaded file,
# also resolves offline instead of depending on a live call to Apple.
#
# Usage:
#   cd desktop && ./make_dmg.sh
set -euo pipefail
cd "$(dirname "$0")"

IDENTITY="Developer ID Application: Anderson Nnewihe (696S9GCN96)"
NOTARY_PROFILE="anya-notary"
APP="dist/Anya Tennis.app"
VERSION="$(python3 -c 'from version import APP_VERSION; print(APP_VERSION)')"
DMG_PATH="dist/Anya Tennis ${VERSION}.dmg"

if [ ! -d "$APP" ]; then
    echo "error: $APP not found — run build_macos.sh first" >&2
    exit 1
fi
if ! codesign --verify --deep --strict "$APP" 2>/dev/null; then
    echo "error: $APP is not validly signed — run build_macos.sh first" >&2
    exit 1
fi
if ! xcrun stapler validate "$APP" >/dev/null 2>&1; then
    echo "error: $APP is not notarization-stapled — run build_macos.sh first" >&2
    exit 1
fi

echo "==> Staging DMG contents"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

echo "==> Building $DMG_PATH"
rm -f "$DMG_PATH"
hdiutil create -volname "Anya Tennis ${VERSION}" -srcfolder "$STAGE" -ov -format UDZO "$DMG_PATH"

echo "==> Signing DMG"
codesign --force --sign "$IDENTITY" "$DMG_PATH"

echo "==> Submitting DMG for notarization (can take several minutes)"
xcrun notarytool submit "$DMG_PATH" --keychain-profile "$NOTARY_PROFILE" --wait

echo "==> Stapling DMG"
xcrun stapler staple "$DMG_PATH"

echo "==> Verifying"
spctl --assess --type open --context context:primary-signature -v "$DMG_PATH"

echo "==> Done: $DMG_PATH"
ls -lh "$DMG_PATH"
