"""The grace table.

This is the reason the suite exists. Getting these branches wrong either locks
a paying customer out mid-trip or gives the app away, and none of it is
observable by running the app for five minutes — the interesting cases need a
clock twenty days from now.
"""

import io
import json
import urllib.error

import pytest

from conftest import DAY, NOW, make_claims, make_jwt, make_session

import entitlement as E
from entitlement import EntState, evaluate


def state(session, now=NOW, online_ok=False):
    return evaluate(session, now=now, online_ok=online_ok).state


# ── The basics ─────────────────────────────────────────────────────────────

def test_no_session_is_signed_out():
    assert state(None) is EntState.SIGNED_OUT


def test_session_without_refresh_token_is_signed_out():
    s = make_session()
    s.refresh_token = ""
    assert state(s) is EntState.SIGNED_OUT


def test_session_without_a_token_is_unentitled():
    s = make_session()
    s.last_id_token = ""
    assert state(s) is EntState.UNENTITLED


def test_token_without_entitlement_claim_is_unentitled():
    s = make_session(claims=make_claims(ent=None))
    assert state(s) is EntState.UNENTITLED


def test_verified_online_is_entitled():
    assert state(make_session(), online_ok=True) is EntState.ENTITLED


def test_fresh_token_offline_is_entitled_via_grace():
    # Even a token minted one second ago is "grace", not "entitled": entitled
    # means we actually reached the server this launch.
    assert state(make_session(iat=NOW - 1)) is EntState.GRACE


# ── The grace window ───────────────────────────────────────────────────────

@pytest.mark.parametrize("age_days,expected", [
    (0,  EntState.GRACE),
    (1,  EntState.GRACE),
    (13, EntState.GRACE),
    (13.9, EntState.GRACE),
    (14, EntState.EXPIRED),   # the boundary is exclusive
    (15, EntState.EXPIRED),
    (60, EntState.EXPIRED),
])
def test_grace_window_boundaries(age_days, expected):
    iat = NOW - int(age_days * DAY)
    s = make_session(iat=iat, hwm=iat)
    assert state(s, now=NOW) is expected


def test_grace_is_measured_from_token_iat_not_from_hwm():
    """The window starts at a server-stamped time, not one we wrote down.

    hwm is deliberately stale here: if grace were measured from anything the
    app records locally, moving that number would extend access. It cannot,
    because iat lives inside the token.
    """
    old_iat = NOW - 20 * DAY
    s = make_session(iat=old_iat, hwm=old_iat)
    assert state(s, now=NOW) is EntState.EXPIRED


# ── Clock attacks ──────────────────────────────────────────────────────────

def test_clock_rolled_back_below_hwm_fails_closed():
    s = make_session(iat=NOW, hwm=NOW + 10 * DAY)
    assert state(s, now=NOW) is EntState.EXPIRED


def test_small_backward_skew_is_tolerated():
    # An NTP correction or a DST oddity must not sign anyone out.
    s = make_session(iat=NOW, hwm=NOW + E.CLOCK_SKEW_TOLERANCE_S - 1)
    assert state(s, now=NOW) is EntState.GRACE


def test_clock_rolled_forward_only_shortens_grace():
    iat = NOW
    s = make_session(iat=iat, hwm=iat)
    assert state(s, now=NOW + 20 * DAY) is EntState.EXPIRED


def test_rollback_reason_is_distinguishable():
    s = make_session(iat=NOW, hwm=NOW + 10 * DAY)
    assert "clock" in evaluate(s, now=NOW).reason


# ── Usage bound ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("launches,expected", [
    (0, EntState.GRACE),
    (29, EntState.GRACE),
    (30, EntState.EXPIRED),
    (31, EntState.EXPIRED),
])
def test_offline_launch_ceiling(launches, expected):
    s = make_session(iat=NOW, offline_launches=launches)
    assert state(s, now=NOW) is expected


# ── Subscription vs check ──────────────────────────────────────────────────

def test_lapsed_subscription_with_a_fresh_token_is_expired():
    """Grace extends a missed CHECK, never a lapsed SUBSCRIPTION."""
    s = make_session(iat=NOW, ent_exp=NOW - 1)
    assert state(s, now=NOW) is EntState.EXPIRED


def test_lapsed_subscription_is_expired_even_when_online():
    s = make_session(iat=NOW, ent_exp=NOW - 1)
    assert state(s, now=NOW, online_ok=True) is EntState.EXPIRED


def test_live_subscription_with_a_stale_token_is_expired():
    s = make_session(iat=NOW - 20 * DAY, ent_exp=NOW + 300 * DAY, hwm=NOW - 20 * DAY)
    assert state(s, now=NOW) is EntState.EXPIRED


def test_token_without_iat_is_expired_offline():
    claims = make_claims()
    del claims["iat"]
    assert state(make_session(claims=claims)) is EntState.EXPIRED


# ── Foreign / malformed tokens ─────────────────────────────────────────────

def test_token_from_another_project_is_unentitled():
    s = make_session(claims=make_claims(iss="https://securetoken.google.com/someone-else"))
    assert state(s) is EntState.UNENTITLED


def test_token_for_another_uid_is_unentitled():
    s = make_session(uid="uid-123", claims=make_claims(uid="uid-999"))
    assert state(s) is EntState.UNENTITLED


def test_garbage_token_is_unentitled_not_a_crash():
    s = make_session()
    s.last_id_token = "not.a.jwt"
    assert state(s) is EntState.UNENTITLED


# ── The property the app actually asks ─────────────────────────────────────

@pytest.mark.parametrize("st,allowed", [
    (EntState.ENTITLED, True),
    (EntState.GRACE, True),
    (EntState.SIGNED_OUT, False),
    (EntState.UNENTITLED, False),
    (EntState.EXPIRED, False),
])
def test_allows_app(st, allowed):
    assert st.allows_app is allowed


def test_plan_and_expiry_survive_into_the_result():
    ent = evaluate(make_session(plan="m", ent_exp=NOW + 30 * DAY), now=NOW, online_ok=True)
    assert (ent.plan, ent.expires_at) == ("m", NOW + 30 * DAY)


# ── verify_online: what a refresh does to the session ──────────────────────

def _ok(payload):
    def _open(req, timeout):
        return io.BytesIO(json.dumps(payload).encode())
    return _open


def _offline():
    def _open(req, timeout):
        raise urllib.error.URLError("no route to host")
    return _open


def _rejects(code):
    def _open(req, timeout):
        body = json.dumps({"error": {"message": code}}).encode()
        raise urllib.error.HTTPError(req.full_url, 400, "Bad", {}, io.BytesIO(body))
    return _open


def _refreshed(**claim_kwargs):
    return _ok({
        "id_token": make_jwt(make_claims(**claim_kwargs)),
        "refresh_token": "r-new",
        "user_id": claim_kwargs.get("uid", "uid-123"),
    })


def test_successful_refresh_stores_new_tokens_and_resets_offline_count():
    s = make_session(offline_launches=7, hwm=NOW - DAY)
    ent, out = E.verify_online(s, opener=_refreshed(iat=NOW), now=NOW)
    assert ent.state is EntState.ENTITLED
    assert out.refresh_token == "r-new"
    assert out.offline_launches == 0
    assert out.hwm == NOW


def test_network_failure_falls_back_to_grace_and_counts_the_launch():
    s = make_session(iat=NOW - DAY, offline_launches=2)
    ent, out = E.verify_online(s, opener=_offline(), now=NOW)
    assert ent.state is EntState.GRACE
    assert out.offline_launches == 3
    # The cached token is what grace is computed from — it must survive.
    assert out.last_id_token


def test_repeated_offline_launches_eventually_expire():
    s = make_session(iat=NOW, offline_launches=E.MAX_OFFLINE_LAUNCHES - 1)
    ent, _ = E.verify_online(s, opener=_offline(), now=NOW)
    assert ent.state is EntState.EXPIRED


def test_a_revoked_session_is_signed_out_not_granted_grace():
    """Grace covers an unreachable server, never a server that said no.

    Being able to revoke a leaked refresh token is the whole point of "sign
    out everywhere"; honouring grace here would defeat it.
    """
    s = make_session(iat=NOW)
    ent, out = E.verify_online(s, opener=_rejects("TOKEN_EXPIRED"), now=NOW)
    assert ent.state is EntState.SIGNED_OUT
    assert out is None


def test_disabled_account_is_signed_out_immediately():
    ent, out = E.verify_online(make_session(), opener=_rejects("USER_DISABLED"), now=NOW)
    assert ent.state is EntState.SIGNED_OUT and out is None


def test_verify_online_with_no_session_is_signed_out():
    ent, out = E.verify_online(None, now=NOW)
    assert ent.state is EntState.SIGNED_OUT and out is None


def test_refresh_that_reports_no_entitlement_is_unentitled():
    s = make_session()
    ent, _ = E.verify_online(s, opener=_refreshed(ent=None), now=NOW)
    assert ent.state is EntState.UNENTITLED
