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
#   cd desktop && ./build_macos.sh
set -euo pipefail
cd "$(dirname "$0")"

IDENTITY="Developer ID Application: Anderson Nnewihe (696S9GCN96)"
NOTARY_PROFILE="anya-notary"
APP="dist/Anya Tennis.app"
ZIP="dist/AnyaTennis-notarize.zip"

# Python 3.12.0 ships a CPython bug (cpython#110543) where code.replace()
# drops the CO_FAST_HIDDEN flag used by PEP 709 inlined comprehensions.
# PyInstaller calls code.replace() on every module, so on 3.12.0 the build
# SUCCEEDS and then the app dies at startup with a NameError from
# torch/_numpy/_ufuncs.py. Fail loudly here instead of shipping that.
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info[:3] != (3, 12, 0) else 1)'; then
    echo "error: Python 3.12.0 cannot build this app (cpython#110543 breaks torch/scipy" >&2
    echo "       in the frozen bundle). Use Python >= 3.12.1, e.g.:" >&2
    echo "         pyenv activate anya-build" >&2
    exit 1
fi

if ! security find-identity -v -p codesigning | grep -qF "$IDENTITY"; then
    echo "error: signing identity not found in keychain: $IDENTITY" >&2
    exit 1
fi
if ! xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
    echo "error: notarytool keychain profile '$NOTARY_PROFILE' not set up or invalid" >&2
    exit 1
fi

echo "==> Cleaning previous build"
rm -rf build dist

echo "==> Running PyInstaller"
pyinstaller rally_app.spec

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
