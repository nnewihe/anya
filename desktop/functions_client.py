"""
functions_client.py — calling the Cloud Functions callables from Python.

There is no Firebase client SDK for Python, but the callable protocol is
simple enough to speak by hand and is a documented wire format rather than an
internal detail: POST {"data": {...}} with the ID token as a bearer token, get
back {"result": {...}} or {"error": {"status", "message", "details"}}.

Speaking it directly is also what keeps this module stdlib-only, for the same
reasons auth.py is — see its docstring.

No Qt: authworker.py runs these on a QThread.
"""

import json
import urllib.error
import urllib.request

from firebase_config import FUNCTIONS_BASE

_TIMEOUT_S = 30  # Stripe round-trips happen inside some of these


class FunctionError(Exception):
    """A callable returned an error.

    `status` is the canonical gRPC-ish code Firebase uses
    ("failed-precondition", "unauthenticated", ...). `details` carries whatever
    the function attached — cancelAndRefund uses it to say WHY a refund was
    refused, so the dialog can explain rather than just failing.
    """

    def __init__(self, status, message, details=None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details or {}


def call(name, id_token, data=None, opener=None):
    """Invoke a callable. Returns its `result`.

    Raises FunctionError("unavailable", ...) for any network failure, so
    callers have one exception type and one obvious branch for "try again when
    you're online".
    """
    body = json.dumps({"data": data or {}}).encode("utf-8")
    req = urllib.request.Request(
        f"{FUNCTIONS_BASE}/{name}",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {id_token}",
        },
    )
    open_fn = opener or (lambda r, timeout: urllib.request.urlopen(r, timeout=timeout))

    try:
        with open_fn(req, _TIMEOUT_S) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        # As with auth.py: the body of a 4xx is where the reason is, and
        # HTTPError is file-like, so it can be read directly.
        try:
            err = (json.load(exc) or {}).get("error") or {}
        except Exception:
            err = {}
        raise FunctionError(
            err.get("status") or f"http_{exc.code}",
            err.get("message") or "That didn't work. Please try again.",
            err.get("details"),
        ) from exc
    except Exception as exc:
        raise FunctionError(
            "unavailable",
            "Couldn't reach the server. Check your connection and try again.",
        ) from exc

    if "error" in payload:
        err = payload["error"]
        raise FunctionError(
            err.get("status", "unknown"),
            err.get("message", "That didn't work."),
            err.get("details"),
        )
    return payload.get("result", {})


# ── The five calls, named ──────────────────────────────────────────────────

def create_checkout_session(id_token, plan="annual", opener=None):
    """Returns a hosted Stripe Checkout URL to open in the system browser."""
    return call("createCheckoutSession", id_token, {"plan": plan}, opener=opener).get("url")


def create_portal_session(id_token, opener=None):
    """Returns a Stripe billing-portal URL: card changes, invoices, cancel."""
    return call("createPortalSession", id_token, opener=opener).get("url")


def get_entitlement(id_token, opener=None):
    """Subscription detail for the account screen.

    Distinct from entitlement.evaluate(): the claim in the token says WHETHER
    the app may run — the only question that matters offline — while this says
    when it renews and whether the refund offer is still open. The gate never
    waits on this call.
    """
    return call("getEntitlement", id_token, opener=opener)


def cancel_and_refund(id_token, opener=None):
    return call("cancelAndRefund", id_token, opener=opener)


def revoke_sessions(id_token, opener=None):
    """Sign out everywhere: invalidates every refresh token for the account."""
    return call("revokeSessions", id_token, opener=opener)
