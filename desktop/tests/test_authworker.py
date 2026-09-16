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


# ── The GUI-thread hop for the sign-in URL ─────────────────────────────────
#
# These do create a QApplication, unlike everything above. It is the only way
# to observe the property that matters — that the callback runs on a DIFFERENT
# thread from the one that asked for it — because the hop IS Qt's event loop.

@pytest.fixture
def qapp():
    """An offscreen QApplication, or a skip on a machine that can't make one."""
    import os as _os
    _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def test_the_url_is_opened_on_the_gui_thread_not_the_worker(qapp):
    """The Windows hazard: QDesktopServices.openUrl off the GUI thread.

    On Windows openUrl goes through ShellExecute, which needs COM initialised
    on the calling thread — Qt does that for the GUI thread only. Calling it
    from the worker returned False with no exception: no browser, no error,
    and a three-minute wait ending in a cancellation nobody asked for.

    So the assertion is about threads, not about the URL: whoever emits, the
    callback must land on the thread that owns the opener.
    """
    import threading

    from PyQt6.QtCore import QThread

    import authworker

    gui_thread = threading.get_ident()
    seen = {}
    done = threading.Event()

    def fake_open_url(url):
        seen["url"] = url
        seen["thread"] = threading.get_ident()
        done.set()

    opener = authworker._GuiThreadUrlOpener(None, fake_open_url)

    class _Emitter(QThread):
        def run(self):
            seen["emitted_from"] = threading.get_ident()
            opener.request.emit("https://accounts.google.com/o/oauth2/v2/auth?x=1")

    emitter = _Emitter()
    emitter.start()
    emitter.wait(5000)

    # The queued call is sitting in the GUI thread's event queue until pumped.
    assert "thread" not in seen, "delivered synchronously — the connection is not queued"
    for _ in range(100):
        qapp.processEvents()
        if done.is_set():
            break

    assert done.is_set(), "the queued call never reached the GUI thread"
    assert seen["url"].startswith("https://accounts.google.com/")
    assert seen["emitted_from"] != gui_thread, "the emitter never left the GUI thread"
    assert seen["thread"] == gui_thread


def test_a_browser_that_refuses_to_launch_does_not_raise(qapp):
    """openUrl failing must not take the GUI thread down with it — the sign-in
    just times out, exactly as it does when the user closes the tab."""
    import authworker

    def exploding_open_url(url):
        raise RuntimeError("no browser here")

    opener = authworker._GuiThreadUrlOpener(None, exploding_open_url)
    opener.request.emit("https://example.invalid/")
    qapp.processEvents()  # would propagate out of here if it were not caught
