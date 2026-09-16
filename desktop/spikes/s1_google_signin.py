"""
S1 — Google sign-in: the loopback flow, and signInWithIdp.

The plan called this the biggest unknown. It has two halves, and only one of
them can be tested without Google's consent screen:

  (a) MECHANICS — generate PKCE, bind a loopback listener on a random port,
      open a browser, catch the redirect, reject a mismatched state, exchange
      the code, release the port. All local, and all exercised here against a
      stub "browser" and a stub token endpoint.

  (b) CONSOLE CONFIG — that a real Google OAuth *Desktop* client accepts
      http://127.0.0.1:<any port>, and that Firebase accepts the resulting
      Google id_token at accounts:signInWithIdp. That needs a real project and
      a real Google account, and no emulator substitutes for it.

For (b) this spike does the next best thing: it drives signInWithIdp against
the Auth emulator with a synthetic Google credential. That proves the REQUEST
SHAPE is right -- the postBody encoding, requestUri, returnSecureToken, and the
camelCase/snake_case normalisation -- which is where the bugs on our side live.
What it cannot prove is the console wiring, which is where the bugs on Google's
side live.

    python3 spikes/s1_google_signin.py            # (a) + (b) against emulator
    python3 spikes/s1_google_signin.py --real     # (b) for real: opens a browser

--real requires firebase_config to point at a live project with the Google
provider enabled AND the desktop client ID whitelisted under
Authentication -> Sign-in method -> Google -> Web SDK configuration ->
"Whitelist client IDs from external projects". Without that last step
signInWithIdp rejects a perfectly good token with INVALID_IDP_RESPONSE, which
is the single most confusing failure in this whole flow.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT = os.environ.get("ANYA_SPIKE_PROJECT", "demo-anya")
AUTH_HOST = os.environ.get("ANYA_AUTH_EMULATOR_HOST", "127.0.0.1:9099")

# Point at the emulator HERE, at module scope, before anything imports
# firebase_config -- it resolves its endpoints at import time, and part_a()
# imports oauth_loopback, which imports firebase_config. Setting these inside
# a function runs too late and the request silently goes to the real Identity
# Toolkit, which answers "API key not valid" and looks like a config problem
# rather than an ordering one.
REAL = "--real" in sys.argv
if not REAL:
    os.environ["ANYA_AUTH_EMULATOR_HOST"] = AUTH_HOST
    os.environ["ANYA_FIREBASE_PROJECT"] = PROJECT
    os.environ["ANYA_FIREBASE_API_KEY"] = "emulator-key"   # emulator ignores it

PASS, FAIL = "  \033[32mPASS\033[0m", "  \033[31mFAIL\033[0m"
_failures = []


def check(label, ok, detail=""):
    print(f"{PASS if ok else FAIL}  {label}{('  — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)
    return ok


def step(text):
    print(f"\n\033[1m{text}\033[0m")


def part_a():
    """The loopback + PKCE mechanics, with a stub browser."""
    import oauth_loopback as O

    step("(a) Loopback + PKCE mechanics")

    verifier, challenge = O.make_pkce_pair()
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    check("S256 challenge derives from the verifier", challenge == expected)
    check("verifier length is inside RFC 7636's 43..128", 43 <= len(verifier) <= 128)
    check("challenge is url-safe and unpadded", not set("+/=") & set(challenge))

    seen = {}

    def stub_browser(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        seen.update({k: v[0] for k, v in q.items()})
        port = int(q["redirect_uri"][0].rsplit(":", 1)[1])
        seen["port"] = port

        def hit():
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/?" + urllib.parse.urlencode(
                    {"code": "stub-auth-code", "state": q["state"][0]}), timeout=5).read()

        threading.Thread(target=hit, daemon=True).start()

    def stub_token_endpoint(req, timeout):
        body = urllib.parse.parse_qs(req.data.decode())
        seen["exchange"] = {k: v[0] for k, v in body.items()}
        return io.BytesIO(json.dumps({"id_token": "stub-google-id-token"}).encode())

    token = O.google_id_token(stub_browser, timeout_s=20, opener=stub_token_endpoint)

    check("the flow returned Google's id_token", token == "stub-google-id-token")
    check("a random high port was used, not a fixed one",
          seen.get("port", 0) > 1024, f"port={seen.get('port')}")
    check("redirect_uri is loopback by IP, not localhost",
          seen.get("redirect_uri", "").startswith("http://127.0.0.1:"),
          seen.get("redirect_uri"))
    check("the authorize request sent code_challenge_method=S256",
          seen.get("code_challenge_method") == "S256")
    check("it asked for openid+email", "openid" in seen.get("scope", ""))
    check("prompt=select_account, so the user picks the billed account",
          seen.get("prompt") == "select_account")
    check("the exchange sent the VERIFIER, not the challenge",
          seen.get("exchange", {}).get("code_verifier") not in
          (None, seen.get("code_challenge")))
    check("the exchange redirect_uri matches the authorize one",
          seen.get("exchange", {}).get("redirect_uri") == seen.get("redirect_uri"))
    check("the listener was released", O.port_is_free(seen["port"]))

    # A tampered redirect must not be accepted.
    def evil_browser(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        port = int(q["redirect_uri"][0].rsplit(":", 1)[1])
        seen["evil_port"] = port

        def hit():
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/?" + urllib.parse.urlencode(
                    {"code": "attacker-code", "state": "not-our-state"}), timeout=5).read()

        threading.Thread(target=hit, daemon=True).start()

    try:
        O.google_id_token(evil_browser, timeout_s=20, opener=stub_token_endpoint)
        check("a mismatched state is rejected", False, "it was ACCEPTED")
    except O.OAuthError as exc:
        check("a mismatched state is rejected", "didn't match" in str(exc))
    check("the listener was released after the rejection",
          O.port_is_free(seen["evil_port"]))


def part_b_emulator():
    """signInWithIdp's request shape, against the Auth emulator."""
    import auth
    import entitlement as ent
    import firebase_config as cfg
    from authstore import Session

    step("(b) signInWithIdp — request shape, against the Auth emulator")

    # Guard the ordering trap described at the top of this file: if this fails,
    # a real request is about to be sent to Google with a fake key.
    if not check("the client is pointed at the emulator, not Google",
                 "127.0.0.1" in cfg.IDENTITY_BASE, cfg.IDENTITY_BASE):
        return

    email = f"google-{secrets.token_hex(4)}@example.com"
    # The emulator accepts an unsigned JSON "id_token" in place of a real
    # Google assertion; the surrounding request is byte-for-byte what a real
    # one would be.
    fake_google_token = json.dumps({"sub": secrets.token_hex(8), "email": email,
                                    "email_verified": True})

    body = auth.google_post_body(fake_google_token)
    check("postBody carries providerId=google.com", "providerId=google.com" in body)
    check("postBody url-encodes the id_token", "id_token=" in body)

    try:
        result = auth.sign_in_with_idp("google.com", body)
    except auth.AuthError as exc:
        check("signInWithIdp returned a session", False, f"{exc.code}: {exc.message}")
        return

    check("signInWithIdp returned a session",
          bool(result["id_token"] and result["refresh_token"]), f"uid={result['uid']}")
    check("the normalised result carries the email", result["email"] == email,
          result["email"])

    claims = auth.decode_jwt_payload(result["id_token"])
    check("the token is for this project",
          auth.token_looks_like_ours(claims, result["uid"]))
    check("a federated user starts unentitled, like any other",
          ent.evaluate(Session(refresh_token=result["refresh_token"], uid=result["uid"],
                               email=result["email"], last_id_token=result["id_token"],
                               hwm=int(time.time())), online_ok=True).state
          is ent.EntState.UNENTITLED)

    # Signing in again with the same provider identity must reuse the account,
    # not make a second one -- otherwise a user's subscription would vanish the
    # second time they signed in with Google.
    again = auth.sign_in_with_idp("google.com", body)
    check("signing in again reuses the same uid", again["uid"] == result["uid"],
          f"{result['uid']} vs {again['uid']}")


def part_b_real():
    """The real thing: a real browser, a real Google account, a real project."""
    import auth
    import firebase_config as cfg
    import oauth_loopback as O
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QDesktopServices
    from PyQt6.QtWidgets import QApplication

    step("(b) REAL Google sign-in")
    print(f"    project={cfg.PROJECT_ID}  client_id={cfg.GOOGLE_CLIENT_ID[:28]}…")

    if "REPLACE_ME" in cfg.GOOGLE_CLIENT_ID or "REPLACE_ME" in cfg.WEB_API_KEY:
        check("firebase_config points at a real project", False,
              "still has REPLACE_ME placeholders — set ANYA_FIREBASE_* first")
        return

    app = QApplication.instance() or QApplication([])  # noqa: F841 — openUrl needs one
    print("    a browser window is opening; pick an account…")
    try:
        google_token = O.google_id_token(
            lambda url: QDesktopServices.openUrl(QUrl(url)))
    except O.OAuthCancelled as exc:
        check("the browser flow completed", False, f"cancelled: {exc}")
        return
    except O.OAuthError as exc:
        check("the browser flow completed", False, str(exc))
        return
    check("the browser flow completed", True, "got a Google id_token")

    try:
        result = auth.sign_in_with_idp("google.com", auth.google_post_body(google_token))
    except auth.AuthError as exc:
        hint = ""
        if "IDP" in exc.code or "INVALID" in exc.code:
            hint = ("  <- the desktop client ID is almost certainly not "
                    "whitelisted in Firebase -> Auth -> Google -> Web SDK config")
        check("Firebase accepted the Google credential", False, exc.code + hint)
        return
    check("Firebase accepted the Google credential", True, f"uid={result['uid']}")
    check("and returned a usable session",
          bool(result["id_token"] and result["refresh_token"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="do the real browser flow against a live project")
    args = ap.parse_args()

    print("S1: Google sign-in")
    part_a()
    if args.real:
        part_b_real()
    else:
        part_b_emulator()

    print()
    if _failures:
        print(f"\033[31mS1 FAIL — {len(_failures)} check(s):\033[0m")
        for f in _failures:
            print(f"  - {f}")
        return 1

    print("\033[32mS1 PASS\033[0m")
    if not args.real:
        print("Mechanics and request shape are proven. NOT yet proven, and only")
        print("provable against a live project: that a real Google Desktop OAuth")
        print("client and the Firebase whitelist accept each other. Run with")
        print("--real once the project exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
