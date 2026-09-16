"""
check_oauth_client.py — fail the build if Google sign-in would be missing.

Run by build_macos.sh and build_windows.ps1 before PyInstaller, for the same
reason as check_model_paths.py: the failure is silent, and it ships.

The bug this exists to catch: `desktop/oauth_client.py` holds the Google OAuth
client and is GITIGNORED on purpose (see oauth_client.example.py — publishing
an installed-app client id and secret in a PUBLIC repo lets a bot stand up a
phishing app showing our own consent screen). `firebase_config` imports it in
a try/except and falls back to empty strings, and `gate_screen` then HIDES the
"Continue with Google" button rather than showing one that always fails.

Every one of those decisions is right on its own. Together they mean a build
made anywhere the file is absent produces a working, signed, notarized app
with one sign-in method quietly missing, and nothing anywhere says so.

That is exactly what happened to 0.2.0. The macOS DMGs were built on a machine
that had the file, so PyInstaller picked the module up through the import and
Google sign-in worked. The Windows installer was built by GitHub Actions from
a clean clone, where the file does not exist — so it shipped without it, and
the first Windows user to look simply found the button gone.

The fix is that CI now writes the file from repository secrets. This check is
what makes sure a future build cannot forget again.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import firebase_config as cfg  # noqa: E402


def main() -> int:
    if cfg.google_configured():
        # Enough to confirm WHICH client without putting it in a build log.
        print(f"[oauth] Google sign-in configured "
              f"({cfg.GOOGLE_CLIENT_ID.split('-')[0]}…)")
        return 0

    print(
        "error: Google sign-in is not configured, so this build would ship\n"
        "       without a 'Continue with Google' button and say nothing.\n"
        "\n"
        "       Locally: create desktop/oauth_client.py — see\n"
        "       desktop/oauth_client.example.py for the shape and the reason\n"
        "       it is gitignored.\n"
        "\n"
        "       In CI: set the ANYA_GOOGLE_CLIENT_ID and\n"
        "       ANYA_GOOGLE_CLIENT_SECRET repository secrets; the workflow\n"
        "       writes the file from them before building.\n"
        "\n"
        "       To build deliberately without it, set ANYA_ALLOW_NO_GOOGLE=1.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    if os.environ.get("ANYA_ALLOW_NO_GOOGLE") == "1":
        print("[oauth] WARNING: ANYA_ALLOW_NO_GOOGLE=1 — building with no "
              "Google sign-in.")
        sys.exit(0)
    sys.exit(main())
