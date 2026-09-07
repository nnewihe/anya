"""Shared fixtures for the desktop test suite.

Two jobs, both about isolation:

  * put desktop/ on sys.path so the modules import under the same bare names
    they use at runtime (`import auth`, not `import desktop.auth`) — app.py
    does the same thing at app.py:41-42, and the modules' own imports assume it;
  * redirect app_data_dir() at a tmp_path so a test run never reads or writes
    the developer's real signed-in session.

Nothing here imports PyQt6. The suite covers the pure modules only — auth,
entitlement, authstore, oauth_loopback — which is exactly the split that lets
it run in a second with no display and no QApplication.
"""

import sys
import time
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parent.parent
if str(DESKTOP) not in sys.path:
    sys.path.insert(0, str(DESKTOP))

import applog          # noqa: E402
import authstore       # noqa: E402
import firebase_config # noqa: E402


@pytest.fixture(autouse=True)
def isolated_app_data(tmp_path, monkeypatch):
    """Every test gets its own empty app-data directory."""
    monkeypatch.setattr(applog, "app_data_dir", lambda: tmp_path)
    monkeypatch.setattr(authstore, "app_data_dir", lambda: tmp_path)
    return tmp_path


NOW = 1_760_000_000  # a fixed "now" so no test depends on the wall clock
DAY = 86_400


def make_claims(*, iat=None, ent=1, ent_exp=None, plan="a", uid="uid-123",
                iss=None, aud=None):
    return {
        "iss": iss if iss is not None else firebase_config.TOKEN_ISSUER,
        "aud": aud if aud is not None else firebase_config.TOKEN_AUDIENCE,
        "sub": uid,
        "user_id": uid,
        "email": "coach@example.com",
        "iat": NOW if iat is None else iat,
        "exp": (NOW if iat is None else iat) + 3600,
        **({"ent": ent} if ent is not None else {}),
        **({"entExp": (NOW + 365 * DAY) if ent_exp is None else ent_exp} if ent is not None else {}),
        **({"pl": plan} if ent is not None else {}),
    }


def make_jwt(claims):
    """An unsigned-but-well-formed JWT.

    The signature segment is junk on purpose: decode_jwt_payload does not
    verify signatures (and its docstring explains why), so a test that fed it a
    genuinely signed token would be testing Google, not us.
    """
    import base64, json

    def seg(obj):
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{seg({'alg': 'RS256', 'typ': 'JWT'})}.{seg(claims)}.not-a-real-signature"


def make_session(*, claims=None, hwm=None, offline_launches=0,
                 refresh_token="refresh-abc", uid="uid-123", **claim_kwargs):
    claims = claims if claims is not None else make_claims(uid=uid, **claim_kwargs)
    return authstore.Session(
        refresh_token=refresh_token,
        uid=uid,
        email="coach@example.com",
        last_id_token=make_jwt(claims),
        hwm=NOW if hwm is None else hwm,
        offline_launches=offline_launches,
    )


@pytest.fixture
def now():
    return NOW


@pytest.fixture
def day():
    return DAY
