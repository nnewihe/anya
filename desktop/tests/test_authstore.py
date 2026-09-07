"""authstore.py — the session file must fail closed, never fail loudly.

Two properties matter. The file has to be unreadable by other users on the
machine, and every way of it being wrong — truncated, edited, half-written,
from an older version — has to come back as "signed out" rather than as an
exception out of launch or, worse, as a granted entitlement.
"""

import json
import os
import stat
import sys

import pytest

from conftest import NOW, make_session

import authstore
from authstore import Session, clear, load, save, session_path


def test_round_trip():
    s = make_session()
    assert save(s)
    got = load()
    assert got is not None
    assert (got.uid, got.email, got.refresh_token) == (s.uid, s.email, s.refresh_token)
    assert got.last_id_token == s.last_id_token


def test_missing_file_is_none_not_an_error():
    assert load() is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_written_0600():
    """The file holds a refresh token; other users on the machine must not be
    able to read it. The mode is set on the temp file before any bytes are
    written, so there is no window where it is world-readable."""
    save(make_session())
    mode = stat.S_IMODE(os.stat(session_path()).st_mode)
    assert mode == 0o600


def test_tamper_with_a_value_is_detected():
    """The case this actually defends against: someone opens session.json and
    edits a number."""
    save(make_session(offline_launches=5))
    raw = json.loads(session_path().read_text())
    raw["offline_launches"] = 0
    session_path().write_text(json.dumps(raw))
    assert load() is None


def test_tamper_with_the_tag_is_detected():
    save(make_session())
    raw = json.loads(session_path().read_text())
    raw["tag"] = "0" * 64
    session_path().write_text(json.dumps(raw))
    assert load() is None


def test_removing_the_tag_is_detected():
    save(make_session())
    raw = json.loads(session_path().read_text())
    del raw["tag"]
    session_path().write_text(json.dumps(raw))
    assert load() is None


def test_body_and_salt_from_different_files_do_not_validate():
    """The tag covers the salt as well as the body, so they cannot be mixed.

    Note what this does NOT assert, because the module docstring is explicit
    about it: a session.json copied WHOLE to another machine still validates,
    salt and tag together. Account sharing is revoked server-side, not
    prevented here.
    """
    save(make_session())
    theirs = json.loads(session_path().read_text())
    clear()

    save(make_session(uid="me"))
    mine = json.loads(session_path().read_text())

    forged = dict(theirs)
    forged["salt"] = mine["salt"]      # their body, my install's salt
    session_path().write_text(json.dumps(forged))
    assert load() is None


def test_a_whole_file_copy_does_still_validate():
    """Pins the honest limit of the tag, so nobody later mistakes it for DRM."""
    save(make_session(uid="them"))
    theirs = session_path().read_text()
    clear()
    session_path().write_text(theirs)
    got = load()
    assert got is not None and got.uid == "them"


def test_truncated_file_is_none():
    save(make_session())
    blob = session_path().read_text()
    session_path().write_text(blob[: len(blob) // 2])
    assert load() is None


def test_not_json_is_none():
    session_path().write_text("this is not json at all")
    assert load() is None


def test_wrong_version_is_none():
    save(make_session())
    raw = json.loads(session_path().read_text())
    raw["v"] = 99
    session_path().write_text(json.dumps(raw))
    assert load() is None


def test_unknown_future_fields_do_not_crash_the_constructor():
    """A file written by a NEWER build must sign the user out, not raise."""
    s = make_session()
    save(s)
    raw = json.loads(session_path().read_text())
    raw["some_future_field"] = "hello"
    raw["tag"] = authstore._tag(raw)   # re-tag so integrity passes
    session_path().write_text(json.dumps(raw))
    got = load()
    assert got is not None and got.uid == s.uid


def test_save_is_atomic_and_leaves_no_temp_files(isolated_app_data):
    for _ in range(3):
        save(make_session())
    leftovers = [p.name for p in isolated_app_data.iterdir() if p.name.startswith(".session-")]
    assert leftovers == []


def test_clear_removes_it_and_is_idempotent():
    save(make_session())
    clear()
    assert not session_path().exists()
    clear()  # must not raise


def test_save_returns_false_rather_than_raising_when_unwritable(monkeypatch):
    """A read-only home must cost the user their "stay signed in", not their
    launch."""
    def boom(*a, **k):
        raise OSError("read-only file system")
    monkeypatch.setattr(authstore.tempfile, "mkstemp", boom)
    assert save(make_session()) is False


def test_touch_clock_only_moves_forward():
    s = make_session(hwm=NOW)
    authstore.touch_clock(s, now=NOW - 5000)
    assert s.hwm == NOW
    authstore.touch_clock(s, now=NOW + 5000)
    assert s.hwm == NOW + 5000


def test_redacted_carries_no_credential():
    s = make_session()
    blob = repr(s.redacted())
    assert s.refresh_token not in blob
    assert s.last_id_token not in blob
    assert s.uid in blob


def test_each_session_gets_its_own_salt():
    assert Session(refresh_token="a", uid="u").salt != Session(refresh_token="a", uid="u").salt
