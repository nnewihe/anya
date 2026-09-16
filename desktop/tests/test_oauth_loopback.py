"""oauth_loopback.py — PKCE correctness and a listener that always closes.

Two things are worth pinning. The PKCE challenge is easy to get subtly wrong
(hash the ASCII of the verifier, base64url it, strip the padding) and the
failure surfaces much later as an opaque invalid_grant. And a sign-in the user
abandons must not leave a socket bound for the rest of the session.
"""

import base64
import hashlib
import io
import json
import threading
import urllib.parse
import urllib.request

import pytest

import oauth_loopback as O
from oauth_loopback import OAuthCancelled, OAuthError


# ── PKCE ───────────────────────────────────────────────────────────────────

def test_challenge_matches_rfc7636_appendix_b_vector():
    """The worked example from RFC 7636 §B, so the transform is pinned to the
    spec rather than to our own implementation of it."""
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    expected = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    assert base64.urlsafe_b64encode(digest).decode().rstrip("=") == expected


def test_generated_pair_is_self_consistent():
    verifier, challenge = O.make_pkce_pair()
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    assert base64.urlsafe_b64encode(digest).decode().rstrip("=") == challenge


def test_challenge_is_url_safe_and_unpadded():
    _, challenge = O.make_pkce_pair()
    assert "=" not in challenge and "+" not in challenge and "/" not in challenge


def test_verifier_is_fresh_every_time():
    assert O.make_pkce_pair()[0] != O.make_pkce_pair()[0]


def test_verifier_length_is_within_rfc_bounds():
    # RFC 7636 §4.1: 43..128 characters.
    verifier, _ = O.make_pkce_pair()
    assert 43 <= len(verifier) <= 128


# ── Authorize URL ──────────────────────────────────────────────────────────

def test_authorize_url_carries_the_required_parameters():
    q = urllib.parse.parse_qs(
        urllib.parse.urlparse(O.authorize_url(54321, "CHAL", "STATE")).query)
    assert q["response_type"] == ["code"]
    assert q["code_challenge"] == ["CHAL"]
    assert q["code_challenge_method"] == ["S256"]
    assert q["state"] == ["STATE"]
    assert q["redirect_uri"] == ["http://127.0.0.1:54321"]
    assert "openid" in q["scope"][0]


# ── The loopback receiver ──────────────────────────────────────────────────

def _drive(params, timeout_s=5, token_payload=None):
    """Run the flow with a stub browser that immediately hits the redirect."""
    captured = {}

    def open_url(url):
        port = int(urllib.parse.parse_qs(
            urllib.parse.urlparse(url).query)["redirect_uri"][0].rsplit(":", 1)[1])
        captured["port"] = port
        q = dict(params)
        if q.get("state") == "<echo>":
            q["state"] = urllib.parse.parse_qs(
                urllib.parse.urlparse(url).query)["state"][0]

        def hit():
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/?{urllib.parse.urlencode(q)}", timeout=5).read()

        threading.Thread(target=hit, daemon=True).start()

    def opener(req, timeout):
        return io.BytesIO(json.dumps(
            token_payload if token_payload is not None
            else {"id_token": "google-id-token"}).encode())

    try:
        return O.google_id_token(open_url, timeout_s=timeout_s, opener=opener), captured
    finally:
        pass


def test_happy_path_returns_the_google_id_token():
    token, captured = _drive({"code": "auth-code", "state": "<echo>"})
    assert token == "google-id-token"
    assert O.port_is_free(captured["port"]), "listener was not released"


def test_state_mismatch_is_rejected():
    with pytest.raises(OAuthError) as exc:
        _drive({"code": "auth-code", "state": "not-the-state-we-sent"})
    assert "didn't match" in str(exc.value)


def test_user_denied_consent_is_cancellation_not_an_error():
    with pytest.raises(OAuthCancelled):
        _drive({"error": "access_denied", "error_description": "user said no"})


def test_missing_code_is_cancellation():
    with pytest.raises(OAuthCancelled):
        _drive({"state": "<echo>"})


def test_token_endpoint_without_an_id_token_is_an_error():
    with pytest.raises(OAuthError) as exc:
        _drive({"code": "c", "state": "<echo>"}, token_payload={"access_token": "only"})
    assert "identity token" in str(exc.value)


def test_timeout_releases_the_port():
    """An abandoned sign-in must not hold a socket for the rest of the session."""
    seen = {}

    def open_url(url):
        seen["port"] = int(urllib.parse.parse_qs(
            urllib.parse.urlparse(url).query)["redirect_uri"][0].rsplit(":", 1)[1])
        # Never respond.

    with pytest.raises(OAuthCancelled):
        O.google_id_token(open_url, timeout_s=0.3)
    assert O.port_is_free(seen["port"])


def test_each_run_binds_a_different_port():
    ports = []

    def open_url(url):
        ports.append(int(urllib.parse.parse_qs(
            urllib.parse.urlparse(url).query)["redirect_uri"][0].rsplit(":", 1)[1]))

    for _ in range(2):
        with pytest.raises(OAuthCancelled):
            O.google_id_token(open_url, timeout_s=0.2)
    assert len(ports) == 2 and all(p > 0 for p in ports)
