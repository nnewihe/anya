"""
oauth_loopback.py — "Sign in with Google" for an installed desktop app.

The Firebase REST API has no browser flow of its own, so the dance is: run
OAuth against Google ourselves, then hand the resulting Google id_token to
accounts:signInWithIdp (auth.sign_in_with_idp). This is the flow RFC 8252
prescribes for native apps — the system browser, a redirect to a loopback
address on a port the OS picks, and PKCE.

PKCE is what makes it safe. The OAuth client here is of type "Desktop app",
and its "client secret" ships inside the bundle where anyone can read it; RFC
8252 §8.5 says as much. Security comes from the code_verifier, which is
generated fresh per attempt, never leaves the process, and is required to
redeem the authorization code.

Console setup this depends on, recorded here because it is invisible from the
code and takes an afternoon to rediscover:
  1. Google Cloud console -> Credentials -> OAuth client ID -> type "Desktop
     app". Desktop clients accept http://127.0.0.1 on ANY port, which is why
     binding port 0 works without registering each one.
  2. Firebase console -> Authentication -> Sign-in method -> Google -> Web SDK
     configuration -> "Whitelist client IDs from external projects" -> add the
     desktop client ID. Without this step signInWithIdp rejects a perfectly
     good Google token with INVALID_IDP_RESPONSE.

Deliberately provider-agnostic where it can be, because Apple is next. Apple
will not fit this module as it stands: it rejects http and IP-literal redirect
URIs outright, so it needs a hosted https callback that accepts a form_post and
bounces back to loopback. That is a new strategy alongside this one, not a
rewrite of auth.sign_in_with_idp.

No Qt import: the caller passes `open_url`, which keeps this testable without a
QApplication and lets authworker supply QDesktopServices.openUrl.
"""

import base64
import hashlib
import json
import secrets
import socket
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from applog import logger
from firebase_config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_SCOPES = "openid email profile"

# Long enough to find a password manager and pick an account; short enough
# that an abandoned attempt cannot leave a listener up for the session.
DEFAULT_TIMEOUT_S = 180


class OAuthError(Exception):
    pass


class OAuthCancelled(OAuthError):
    """The user closed the tab, denied consent, or the wait timed out."""


# ── PKCE ───────────────────────────────────────────────────────────────────

def make_pkce_pair():
    """(verifier, challenge) per RFC 7636, S256.

    The challenge is base64url of the SHA-256 of the *ASCII of the verifier* —
    not of its raw bytes decoded from base64 — and the padding is stripped.
    Both are easy to get subtly wrong, and the failure is an opaque
    invalid_grant at redemption time, so test_oauth_loopback checks this
    against the vector published in RFC 7636 appendix B.
    """
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


# ── The loopback receiver ──────────────────────────────────────────────────

_PAGE = """<!doctype html><meta charset="utf-8">
<title>Anya Tennis</title>
<style>
  body {{ background:#000; color:#fff; font:16px -apple-system,Segoe UI,sans-serif;
         display:flex; align-items:center; justify-content:center; height:100vh; margin:0 }}
  .c {{ text-align:center }} .y {{ color:#E8FF3D; font-weight:700; letter-spacing:.06em }}
  p {{ color:#A0A099; font-size:14px }}
</style>
<div class="c"><p class="y">{heading}</p><p>{body}</p></div>
"""


class _Handler(BaseHTTPRequestHandler):
    """Captures exactly one redirect and hands it to the waiting thread."""

    result = None  # set on the server instance, not here

    def do_GET(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        self.server.result = {k: v[0] for k, v in query.items()}

        if "code" in self.server.result:
            page = _PAGE.format(
                heading="SIGNED IN", body="You can close this tab and return to Anya Tennis.")
        else:
            page = _PAGE.format(
                heading="SIGN-IN CANCELLED", body="You can close this tab and try again.")
        body = page.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.server.done.set()

    def log_message(self, fmt, *args):
        # BaseHTTPRequestHandler logs to stderr by default, which in the
        # packaged app (console=False) goes nowhere useful and in dev prints
        # the full redirect — including the authorization code — to the
        # terminal. Route it to the app log at debug, without the query.
        logger().debug("oauth loopback: %s", fmt % args if args else fmt)


def _serve_one_redirect(timeout_s):
    """Bind 127.0.0.1 on an OS-chosen port; return (port, wait_fn, close_fn).

    Port 0 asks the kernel for a free port, which is what makes this safe to
    run twice and what avoids hardcoding a port some other app may hold.
    """
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.result = None
    server.done = threading.Event()
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True,
                              name="anya-oauth-loopback")
    thread.start()

    def wait():
        if not server.done.wait(timeout_s):
            raise OAuthCancelled("Timed out waiting for the browser sign-in to finish.")
        return server.result or {}

    def close():
        server.shutdown()
        server.server_close()
        # Join so a cancelled sign-in cannot leave the port held; the thread
        # is out of serve_forever the moment shutdown() returns.
        thread.join(timeout=5)

    return port, wait, close


# ── The flow ───────────────────────────────────────────────────────────────

def authorize_url(port, challenge, state):
    return _AUTH_ENDPOINT + "?" + urllib.parse.urlencode({
        "response_type": "code",
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": f"http://127.0.0.1:{port}",
        "scope": _SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        # Someone signing into a paid product should get to choose which of
        # their Google accounts it is billed to, rather than being silently
        # handed whichever one the browser happens to be holding.
        "prompt": "select_account",
    })


def _exchange_code(code, verifier, port, opener=None):
    body = urllib.parse.urlencode({
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "code_verifier": verifier,
        "redirect_uri": f"http://127.0.0.1:{port}",
        "grant_type": "authorization_code",
    }).encode("utf-8")

    req = urllib.request.Request(
        _TOKEN_ENDPOINT, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    open_fn = opener or (lambda r, timeout: urllib.request.urlopen(r, timeout=timeout))
    try:
        with open_fn(req, 15) as resp:
            payload = json.load(resp)
    except Exception as exc:
        raise OAuthError("Couldn't complete sign-in with Google. Please try again.") from exc

    token = payload.get("id_token")
    if not token:
        raise OAuthError("Google didn't return an identity token. Please try again.")
    return token


def google_id_token(open_url, timeout_s=DEFAULT_TIMEOUT_S, opener=None):
    """Run the whole flow and return Google's id_token.

    `open_url` is called with the authorize URL; in the app that is
    QDesktopServices.openUrl, and in tests it is a stub. Raises OAuthCancelled
    if the user backs out, OAuthError for anything else.
    """
    verifier, challenge = make_pkce_pair()
    state = secrets.token_urlsafe(32)

    try:
        port, wait, close = _serve_one_redirect(timeout_s)
    except OSError as exc:
        raise OAuthError(
            "Couldn't open a local port to complete sign-in. "
            "A firewall or security tool may be blocking it."
        ) from exc

    try:
        open_url(authorize_url(port, challenge, state))
        params = wait()
    finally:
        close()

    if params.get("error"):
        raise OAuthCancelled(params.get("error_description") or params["error"])

    # Constant-time and before anything else is trusted: a mismatch means the
    # redirect we caught was not the one we sent.
    if not secrets.compare_digest(params.get("state", ""), state):
        raise OAuthError("Sign-in response didn't match this request. Please try again.")

    code = params.get("code")
    if not code:
        raise OAuthCancelled("Sign-in was cancelled.")

    return _exchange_code(code, verifier, port, opener=opener)


def port_is_free(port):
    """Small helper used by the tests to assert the listener was released."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False
