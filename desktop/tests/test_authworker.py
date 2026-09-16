"""fresh_id_token — the one piece of authworker that is a pure function.

Everything else in that module is QThread plumbing, which the release
checklist covers by hand. This is here because the bug it fixes is invisible
for the first hour of any session and then breaks every account action:
Firebase ID tokens last an hour, and the app is routinely open far longer.

Importing authworker pulls in PyQt6, but only class definitions — no
QApplication is created and no widget is touched.
"""

import io
import json
import time
import urllib.error

import pytest

from conftest import NOW, make_claims, make_jwt, make_session

import authstore


def _refresher(payload):
    def _open(req, timeout):
        return io.BytesIO(json.dumps(payload).encode())
    return _open


@pytest.fixture
def worker(monkeypatch):
    import auth
    import authworker
    # auth.refresh() takes no opener at authworker's call site, so the fake
    # goes in at the module boundary rather than through a parameter.
    calls = []

    def fake_refresh(refresh_token):
        calls.append(refresh_token)
        return {
            "id_token": make_jwt(make_claims(iat=NOW)),
            "refresh_token": "r-rotated",
            "uid": "uid-123",
            "email": "coach@example.com",
        }

    monkeypatch.setattr(authworker.auth, "refresh", fake_refresh)
    authworker._calls = calls
    return authworker


def _token_expiring_at(exp):
    claims = make_claims()
    claims["exp"] = exp
    return make_jwt(claims)


def test_a_still_valid_token_is_reused_without_a_network_call(worker):
    s = make_session()
    s.last_id_token = _token_expiring_at(int(time.time()) + 3600)
    assert worker.fresh_id_token(s) == s.last_id_token
    assert worker._calls == []


def test_an_expired_token_is_refreshed(worker):
    s = make_session()
    old = _token_expiring_at(int(time.time()) - 10)
    s.last_id_token = old
    assert worker.fresh_id_token(s) != old
    assert worker._calls == ["refresh-abc"]


def test_a_token_expiring_within_the_headroom_is_refreshed(worker):
    """It has to still be valid when it ARRIVES, not merely when it is sent."""
    s = make_session()
    s.last_id_token = _token_expiring_at(int(time.time()) + 30)
    worker.fresh_id_token(s)
    assert worker._calls == ["refresh-abc"]


def test_a_token_just_outside_the_headroom_is_kept(worker):
    s = make_session()
    s.last_id_token = _token_expiring_at(int(time.time()) + 300)
    worker.fresh_id_token(s)
    assert worker._calls == []


def test_an_undecodable_token_is_refreshed_rather_than_trusted(worker):
    s = make_session()
    s.last_id_token = "not.a.jwt"
    worker.fresh_id_token(s)
    assert worker._calls == ["refresh-abc"]


def test_refresh_mutates_the_session_in_place(worker):
    """app.py and the dialogs hold this object; a new one would leave them
    pointing at a token that has already been rotated away."""
    s = make_session()
    s.last_id_token = _token_expiring_at(int(time.time()) - 10)
    before = id(s)
    worker.fresh_id_token(s)
    assert id(s) == before
    assert s.refresh_token == "r-rotated"


def test_the_rotated_token_is_persisted(worker):
    s = make_session()
    s.last_id_token = _token_expiring_at(int(time.time()) - 10)
    worker.fresh_id_token(s)
    stored = authstore.load()
    assert stored is not None and stored.refresh_token == "r-rotated"
