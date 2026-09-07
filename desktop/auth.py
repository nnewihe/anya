"""
auth.py — Firebase Authentication from a Python desktop client.

Firebase publishes no Python *client* SDK. `firebase-admin` is a server SDK:
it authenticates with a service-account key, which is a root credential for
the whole project — shipping one inside a PyInstaller bundle would hand every
user the ability to read every other user's data. So this module talks to the
Identity Toolkit REST API directly, with the public Web API key, which is
exactly what the official JavaScript SDK does under the covers.

Stdlib only, on purpose. update_check.py already makes HTTPS requests with
`urllib` from inside the signed, notarized bundle on testers' machines, so TLS
trust, proxy handling and certificate verification in the frozen build are
proven for this exact code path. Adding `requests` would mean a certifi CA
bundle in rally_app.spec's datas, four more pinned versions in
constraints-windows.txt, and four more licences to carry — to save a thin
wrapper around urlopen.

No Qt in here. Everything is synchronous and blocking; authworker.py is what
puts it on a QThread. That split is what lets the whole module be tested
without a QApplication.
"""

import base64
import json
import urllib.error
import urllib.parse
import urllib.request

from firebase_config import (
    GOOGLE_CLIENT_ID,
    TOKEN_AUDIENCE,
    TOKEN_ISSUER,
    WEB_API_KEY,
)

_IDENTITY_BASE = "https://identitytoolkit.googleapis.com/v1/accounts"
_SECURETOKEN_URL = "https://securetoken.googleapis.com/v1/token"

# Interactive: a person has pressed a button and is watching a spinner, so a
# slow connection is worth waiting out. update_check.py uses 5s because
# nobody is waiting on it.
_TIMEOUT_S = 15


class AuthError(Exception):
    """A failure the user might be able to do something about.

    `code` is Firebase's machine-readable string (EMAIL_EXISTS,
    INVALID_LOGIN_CREDENTIALS, ...). `message` is already user-facing — see
    friendly_message() for the mapping. Network failures surface as
    AuthError(code="NETWORK") so callers have exactly one exception type to
    catch around a sign-in attempt.
    """

    def __init__(self, code, message=None, http_status=None):
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.http_status = http_status


_FRIENDLY = {
    # Email-enumeration protection is on by default in new Firebase projects,
    # so a wrong password and an unknown address return the SAME code. Do not
    # turn that off to give a nicer message: the nicer message is precisely
    # the account-enumeration oracle the protection exists to close.
    "INVALID_LOGIN_CREDENTIALS": "That email and password don't match an account.",
    "EMAIL_NOT_FOUND": "That email and password don't match an account.",
    "INVALID_PASSWORD": "That email and password don't match an account.",
    "EMAIL_EXISTS": "There's already an account with that email. Try signing in.",
    # Firebase appends detail to this one ("WEAK_PASSWORD : Password should
    # be at least 6 characters"), which is why friendly_message() also tries
    # the leading token.
    "WEAK_PASSWORD": "Please choose a password of at least 6 characters.",
    "INVALID_EMAIL": "That doesn't look like an email address.",
    "USER_DISABLED": "This account has been disabled. Please get in touch.",
    "TOO_MANY_ATTEMPTS_TRY_LATER":
        "Too many attempts from this device. Please wait a few minutes and try again.",
    "TOKEN_EXPIRED": "Your session expired. Please sign in again.",
    "USER_NOT_FOUND": "Your session is no longer valid. Please sign in again.",
    "NETWORK": "Couldn't reach the sign-in service. Check your connection and try again.",
}


def friendly_message(code):
    """User-facing text for a Firebase error code.

    Firebase sometimes appends detail after the code (`WEAK_PASSWORD : ...`),
    so fall back to matching on the leading token before giving up.
    """
    if code in _FRIENDLY:
        return _FRIENDLY[code]
    head = code.split(" ")[0].split(":")[0].strip()
    return _FRIENDLY.get(head, "Sign-in failed. Please try again.")


# ── HTTP ───────────────────────────────────────────────────────────────────

def _request(url, data, content_type, opener=None):
    """POST and return parsed JSON, converting every failure into AuthError.

    `opener` exists so tests can inject a fake without a network or a
    monkeypatched module global; it takes (request, timeout) like
    urlopen does.
    """
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"},
    )
    open_fn = opener or (lambda r, timeout: urllib.request.urlopen(r, timeout=timeout))
    try:
        with open_fn(req, _TIMEOUT_S) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        # HTTPError is itself a file-like object holding the response body,
        # and for a 400 that body is the only place Firebase says WHY. Reading
        # it is the difference between "sign-in failed" and "that email is
        # already registered".
        code, message = "HTTP_%s" % exc.code, None
        try:
            payload = json.load(exc)
            err = payload.get("error") or {}
            code = err.get("message") or code
            message = friendly_message(code)
        except Exception:
            pass
        raise AuthError(code, message, http_status=exc.code) from exc
    except Exception as exc:
        # URLError, socket.timeout, ssl errors, a proxy returning garbage.
        # None of them are distinguishable to a user, and all of them mean
        # the same thing: try again when the network is better.
        raise AuthError("NETWORK", friendly_message("NETWORK")) from exc


def _post_json(path, payload, opener=None):
    url = f"{_IDENTITY_BASE}:{path}?key={WEB_API_KEY}"
    body = json.dumps(payload).encode("utf-8")
    return _request(url, body, "application/json", opener=opener)


# ── JWT ────────────────────────────────────────────────────────────────────

def decode_jwt_payload(token):
    """The claims of a JWT, with NO signature verification.

    Verifying would mean fetching and caching Google's rotating x509 certs and
    doing RSA in Python — which means `cryptography`, a compiled extension with
    per-architecture wheels, a new entry in the inside-out signing loop in
    build_macos.sh, and a new notarization risk. It would also buy nothing that
    matters here: a forged token cannot make the *server* grant anything, and
    the local entitlement cache is protected against casual editing by
    authstore's HMAC. See entitlement.py for the structural checks that stand
    in for verification, and its docstring for why none of this is DRM.

    Raises ValueError on anything that isn't a decodable three-part JWT.
    """
    if not token or not isinstance(token, str):
        raise ValueError("not a token")
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("not a three-part JWT")
    seg = parts[1]
    # Firebase strips base64 padding; urlsafe_b64decode raises without it.
    seg += "=" * (-len(seg) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(seg.encode("ascii")))
    except Exception as exc:
        raise ValueError("undecodable JWT payload") from exc


# ── Normalised result ──────────────────────────────────────────────────────

def _normalise(raw):
    """One shape out of two differently-spelled APIs.

    accounts:* return camelCase (idToken, refreshToken, expiresIn, localId).
    securetoken:token returns snake_case (id_token, refresh_token,
    expires_in, user_id) — same values, different spelling, and mixing them up
    silently yields None where a token should be.
    """
    id_token = raw.get("idToken") or raw.get("id_token")
    refresh_token = raw.get("refreshToken") or raw.get("refresh_token")
    uid = raw.get("localId") or raw.get("user_id")
    email = raw.get("email") or ""
    if not id_token or not refresh_token:
        raise AuthError("MISSING_TOKENS", "Sign-in didn't return a session. Please try again.")
    if not uid or not email:
        # localId is always present on accounts:*; the refresh endpoint gives
        # user_id but no email, so fill both from the token itself.
        try:
            claims = decode_jwt_payload(id_token)
            uid = uid or claims.get("user_id") or claims.get("sub") or ""
            email = email or claims.get("email") or ""
        except ValueError:
            pass
    return {
        "id_token": id_token,
        "refresh_token": refresh_token,
        "uid": uid,
        "email": email,
    }


# ── The API ────────────────────────────────────────────────────────────────

def sign_up(email, password, opener=None):
    return _normalise(_post_json(
        "signUp",
        {"email": email, "password": password, "returnSecureToken": True},
        opener=opener,
    ))


def sign_in(email, password, opener=None):
    return _normalise(_post_json(
        "signInWithPassword",
        {"email": email, "password": password, "returnSecureToken": True},
        opener=opener,
    ))


def sign_in_with_idp(provider_id, post_body, opener=None):
    """Exchange a federated provider's id_token for a Firebase session.

    Provider-agnostic by design: Google today, Apple later. `post_body` is the
    form-encoded blob the provider dictates —
    "id_token=<...>&providerId=google.com" for Google, plus a `nonce` for
    Apple. requestUri must be present but is not meaningfully checked for
    installed apps.
    """
    return _normalise(_post_json(
        "signInWithIdp",
        {
            "postBody": post_body,
            "requestUri": "http://127.0.0.1",
            "returnIdpCredential": True,
            "returnSecureToken": True,
        },
        opener=opener,
    ))


def refresh(refresh_token, opener=None):
    """Mint a fresh ID token from a refresh token.

    Also the entitlement-polling mechanism: custom claims are baked into a
    token when it is minted, so a token issued before Stripe's webhook ran
    will not carry `ent` no matter how long you wait — only a refresh will
    show it. See authworker.CheckoutPollWorker.
    """
    body = urllib.parse.urlencode(
        {"grant_type": "refresh_token", "refresh_token": refresh_token}
    ).encode("utf-8")
    raw = _request(
        f"{_SECURETOKEN_URL}?key={WEB_API_KEY}",
        body, "application/x-www-form-urlencoded", opener=opener,
    )
    return _normalise(raw)


def send_password_reset(email, opener=None):
    return _post_json(
        "sendOobCode", {"requestType": "PASSWORD_RESET", "email": email}, opener=opener
    )


def send_email_verification(id_token, opener=None):
    return _post_json(
        "sendOobCode", {"requestType": "VERIFY_EMAIL", "idToken": id_token}, opener=opener
    )


def lookup(id_token, opener=None):
    """The server's view of the account (emailVerified, linked providers)."""
    raw = _post_json("lookup", {"idToken": id_token}, opener=opener)
    users = raw.get("users") or []
    return users[0] if users else {}


def google_post_body(google_id_token):
    """The `postBody` signInWithIdp wants for a Google credential."""
    return urllib.parse.urlencode(
        {"id_token": google_id_token, "providerId": "google.com"}
    )


def token_looks_like_ours(claims, uid=None):
    """Structural check on a decoded ID token: is this our project's token?

    Not a signature check (see decode_jwt_payload). This catches a truncated
    file, a token from a staging project, or a token pasted in from somewhere
    else — the mistakes that actually happen — and nothing more.
    """
    if not isinstance(claims, dict):
        return False
    if claims.get("iss") != TOKEN_ISSUER:
        return False
    if claims.get("aud") != TOKEN_AUDIENCE:
        return False
    if not claims.get("sub"):
        return False
    if uid and claims.get("sub") != uid:
        return False
    return True


__all__ = [
    "AuthError", "friendly_message", "decode_jwt_payload", "sign_up", "sign_in",
    "sign_in_with_idp", "refresh", "send_password_reset", "send_email_verification",
    "lookup", "google_post_body", "token_looks_like_ours", "GOOGLE_CLIENT_ID",
]
