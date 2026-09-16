"""What each callable hands back to its caller.

One of these functions is not like the others: `create_portal_session` unwraps
the response and returns the URL STRING, while every other callable returns the
whole result dict. That asymmetry is deliberate -- the caller wants a URL, not
an envelope -- but it is invisible at the call site, and it shipped a crash:
account_dialog called `.get("url")` on the already-unwrapped string, so the
first person to press Manage Subscription against a real subscription got
"'str' object has no attribute 'get'" instead of a billing portal.

These tests exist to make the shapes explicit, so changing one of them fails
here rather than in front of a paying customer. No PyQt6, in keeping with the
rest of the suite.
"""

import pytest

import functions_client


@pytest.fixture
def fake_call(monkeypatch):
    """Stand in for the HTTP round trip; record what was asked for."""
    seen = {}

    def _call(name, id_token, data=None, opener=None):
        seen["name"] = name
        seen["id_token"] = id_token
        return {"url": "https://billing.stripe.com/session/test", "ok": True}

    monkeypatch.setattr(functions_client, "call", _call)
    return seen


def test_create_portal_session_returns_a_bare_url_string(fake_call):
    result = functions_client.create_portal_session("tok")
    assert isinstance(result, str), (
        "create_portal_session unwraps the response; account_dialog._open takes "
        "the URL directly and must not call .get() on it")
    assert result == "https://billing.stripe.com/session/test"
    assert fake_call["name"] == "createPortalSession"


@pytest.mark.parametrize("fn,expected_name", [
    ("get_entitlement", "getEntitlement"),
    ("cancel_and_refund", "cancelAndRefund"),
    ("revoke_sessions", "revokeSessions"),
])
def test_the_other_callables_return_the_whole_dict(fake_call, fn, expected_name):
    result = getattr(functions_client, fn)("tok")
    assert isinstance(result, dict)
    assert fake_call["name"] == expected_name


def test_the_id_token_is_passed_through(fake_call):
    functions_client.create_portal_session("the-token")
    assert fake_call["id_token"] == "the-token"
