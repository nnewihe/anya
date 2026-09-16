"""auth.py — JWT decoding, error mapping, and the two-spellings problem.

No network: every test injects a fake `opener`, which is the reason auth.py
takes one.
"""

import io
import json
import urllib.error

import pytest

from conftest import make_claims, make_jwt

import auth
from auth import AuthError, decode_jwt_payload


# ── Fake transport ─────────────────────────────────────────────────────────

def ok(payload):
    """An opener that returns `payload` as a JSON response.

    BytesIO is already a context manager and already file-like, which is all
    _request() asks of what urlopen hands back.
    """
    def _open(req, timeout):
        return io.BytesIO(json.dumps(payload).encode())
    return _open


def fails(status, firebase_code):
    def _open(req, timeout):
        body = json.dumps({"error": {"message": firebase_code}}).encode()
        raise urllib.error.HTTPError(
            req.full_url, status, "Bad Request", {}, io.BytesIO(body))
    return _open


def offline():
    def _open(req, timeout):
        raise urllib.error.URLError("no route to host")
    return _open


# ── decode_jwt_payload ─────────────────────────────────────────────────────

def test_decodes_a_normal_token():
    claims = make_claims()
    assert decode_jwt_payload(make_jwt(claims))["sub"] == claims["sub"]


@pytest.mark.parametrize("email", [
    "a@b.co",        # payload length ≡ 0 mod 4 after encoding
    "ab@b.co",
    "abc@b.co",
    "abcd@b.co",
])
def test_handles_every_stripped_padding_length(email):
    """Firebase strips base64 '=' padding; urlsafe_b64decode raises without it.

    Varying the payload length walks all four residues mod 4, so this covers
    the cases the padding restoration exists for rather than whichever one a
    single sample happened to hit.
    """
    claims = make_claims()
    claims["email"] = email
    token = make_jwt(claims)
    assert "=" not in token.split(".")[1]
    assert decode_jwt_payload(token)["email"] == email


@pytest.mark.parametrize("bad", ["", None, "one-part", "two.parts", "a.b.c.d", 42])
def test_rejects_non_tokens(bad):
    with pytest.raises(ValueError):
        decode_jwt_payload(bad)


def test_rejects_undecodable_payload():
    with pytest.raises(ValueError):
        decode_jwt_payload("header.!!!not-base64!!!.sig")


# ── token_looks_like_ours ──────────────────────────────────────────────────

def test_accepts_our_own_token():
    assert auth.token_looks_like_ours(make_claims(uid="u1"), uid="u1")


@pytest.mark.parametrize("kwargs", [
    {"iss": "https://securetoken.google.com/other-project"},
    {"aud": "other-project"},
])
def test_rejects_foreign_project(kwargs):
    assert not auth.token_looks_like_ours(make_claims(**kwargs))


def test_rejects_mismatched_uid():
    assert not auth.token_looks_like_ours(make_claims(uid="u1"), uid="u2")


def test_rejects_non_dict():
    assert not auth.token_looks_like_ours("nope")


# ── Error mapping ──────────────────────────────────────────────────────────

def test_wrong_password_and_unknown_email_are_indistinguishable():
    """Email-enumeration protection means one message for both. Keep it that
    way: a friendlier message here is an account-enumeration oracle."""
    assert auth.friendly_message("INVALID_LOGIN_CREDENTIALS") == \
           auth.friendly_message("EMAIL_NOT_FOUND")


def test_weak_password_detail_suffix_still_maps():
    assert "6 characters" in auth.friendly_message(
        "WEAK_PASSWORD : Password should be at least 6 characters")


def test_unknown_code_gets_a_generic_message():
    assert auth.friendly_message("SOMETHING_WE_NEVER_SAW") == \
           "Sign-in failed. Please try again."


def test_http_error_body_is_read_for_the_code():
    with pytest.raises(AuthError) as exc:
        auth.sign_in("a@b.co", "pw", opener=fails(400, "EMAIL_EXISTS"))
    assert exc.value.code == "EMAIL_EXISTS"
    assert exc.value.http_status == 400
    assert "already an account" in exc.value.message


def test_rate_limit_maps_to_a_wait_message():
    with pytest.raises(AuthError) as exc:
        auth.sign_in("a@b.co", "pw", opener=fails(400, "TOO_MANY_ATTEMPTS_TRY_LATER"))
    assert "wait" in exc.value.message.lower()


def test_network_failure_becomes_a_single_recognisable_code():
    with pytest.raises(AuthError) as exc:
        auth.sign_in("a@b.co", "pw", opener=offline())
    assert exc.value.code == "NETWORK"


# ── camelCase vs snake_case ────────────────────────────────────────────────

def test_accounts_endpoints_return_camel_case():
    out = auth.sign_in("a@b.co", "pw", opener=ok({
        "idToken": make_jwt(make_claims()), "refreshToken": "r1",
        "localId": "uid-123", "email": "coach@example.com",
    }))
    assert out == {
        "id_token": out["id_token"], "refresh_token": "r1",
        "uid": "uid-123", "email": "coach@example.com",
    }


def test_refresh_endpoint_returns_snake_case_and_is_normalised():
    """The one API that spells it differently. Mixing the two silently yields
    None where a token should be, which is why this is normalised in one place
    and asserted here."""
    token = make_jwt(make_claims(uid="uid-123"))
    out = auth.refresh("r0", opener=ok({
        "id_token": token, "refresh_token": "r2",
        "user_id": "uid-123", "expires_in": "3600",
    }))
    assert out["refresh_token"] == "r2"
    assert out["uid"] == "uid-123"
    # The refresh response carries no email; it is recovered from the token.
    assert out["email"] == "coach@example.com"


def test_missing_tokens_raise_rather_than_returning_none():
    with pytest.raises(AuthError) as exc:
        auth.sign_in("a@b.co", "pw", opener=ok({"localId": "uid-123"}))
    assert exc.value.code == "MISSING_TOKENS"


def test_google_post_body_shape():
    body = auth.google_post_body("goog-id-token")
    assert "id_token=goog-id-token" in body
    assert "providerId=google.com" in body
