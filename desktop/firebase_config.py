"""
firebase_config.py — the public identifiers the desktop client is built with.

Everything in this file is meant to be readable by anyone who downloads the
app. That is not an oversight, and it is not something to "fix" later:

  * The Firebase Web API key identifies the project to Google's Identity
    Toolkit. It authorizes nothing on its own — account security comes from
    the ID token and the Firestore rules, both enforced server-side. Google
    ships this key in the <script> tag of every Firebase web app.
  * The Google OAuth client here is of type "Desktop app". RFC 8252 §8.5 is
    explicit that such a client's "secret" is not confidential: it cannot be,
    because the client is installed on the user's machine. Google issues one
    anyway because the token endpoint's shape requires the field. Security for
    this flow comes from PKCE (see oauth_loopback.py), not from the secret.

What must NEVER appear in this file, or anywhere else in the bundle: the
Stripe secret key, the Stripe webhook signing secret, a Firebase
service-account JSON, or the Sign in with Apple .p8. Those live in Google
Secret Manager and are read only by the Cloud Functions (see functions/).

Note there is no Stripe key here at all — not even the publishable one. The
desktop app never talks to Stripe: it asks a Cloud Function for a hosted
Checkout URL and opens that in the browser.
"""

import os

# ── Firebase project ───────────────────────────────────────────────────────
# Set for real when the project is created (Phase 0/S1). Left as obvious
# placeholders rather than empty strings so a half-configured build fails with
# a recognisable 400 from Google instead of a confusing "API key not valid".
PROJECT_ID = os.environ.get("ANYA_FIREBASE_PROJECT", "anya-tennis")
WEB_API_KEY = os.environ.get("ANYA_FIREBASE_API_KEY", "REPLACE_ME_WEB_API_KEY")

# Issuer/audience the ID token must carry. Derived rather than hardcoded so a
# staging project only needs PROJECT_ID overridden.
TOKEN_ISSUER = f"https://securetoken.google.com/{PROJECT_ID}"
TOKEN_AUDIENCE = PROJECT_ID

# ── Google sign-in (OAuth client of type "Desktop app") ────────────────────
GOOGLE_CLIENT_ID = os.environ.get("ANYA_GOOGLE_CLIENT_ID", "REPLACE_ME.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.environ.get("ANYA_GOOGLE_CLIENT_SECRET", "REPLACE_ME_NOT_A_SECRET")

# ── Cloud Functions ────────────────────────────────────────────────────────
# Callable functions are POSTed directly rather than through a client SDK; the
# callable protocol is just {"data": {...}} in and {"result": {...}} out.
FUNCTIONS_REGION = "us-central1"
FUNCTIONS_BASE = os.environ.get(
    "ANYA_FUNCTIONS_BASE",
    f"https://{FUNCTIONS_REGION}-{PROJECT_ID}.cloudfunctions.net",
)

# ── Marketing surfaces ─────────────────────────────────────────────────────
# Streamed, not bundled: the installed app is already ~2 GB and a video that
# ships inside the DMG cannot be changed without a signed, notarized release.
# Must be a progressive-download MP4 with +faststart — see gate_screen.py.
PREVIEW_VIDEO_URL = os.environ.get(
    "ANYA_PREVIEW_VIDEO_URL", f"https://{PROJECT_ID}.web.app/preview.mp4"
)
LANDING_URL = "https://nnewihe.github.io/anya/"
TERMS_URL = "https://nnewihe.github.io/anya/terms.html"
PRIVACY_URL = "https://nnewihe.github.io/anya/privacy.html"

# ── Pricing, for display only ──────────────────────────────────────────────
# The prices that actually charge anyone live in Stripe and are selected by
# price ID inside createCheckoutSession. These strings exist so the gate
# screen has something to render; they are never used in a calculation.
PRICE_ANNUAL_DISPLAY = "$40 / year"
PRICE_MONTHLY_DISPLAY = "$5 / month"
REFUND_WINDOW_DAYS = 14


def is_configured() -> bool:
    """False in a build where the placeholders were never replaced.

    Lets the gate screen say "this build isn't configured for sign-in" rather
    than surfacing a raw Google API error to someone who cannot act on it.
    """
    return "REPLACE_ME" not in WEB_API_KEY
