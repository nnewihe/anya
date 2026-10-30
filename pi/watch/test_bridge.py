"""
Tests for bridge.py's recorder side (no Bluetooth): the status bytes, the
commands, and the real recorder's web API with its fake camera.

    python -m pytest pi/watch
"""

import asyncio
import json
import shutil
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "recorder"))
import bridge as B  # noqa: E402

FAKE = str(HERE.parent / "recorder" / "fake_rpicam_vid.py")


def status(state="idle", elapsed=None, error=None, saving=0):
    return {"state": state, "elapsed_s": elapsed, "error": error, "max_s": 5400,
            "saving": saving, "recordings": [{"name": "x" * 500}]}


def decode(b):
    assert len(b) <= B.MAX_STATUS
    return json.loads(b)


# -- encoding ----------------------------------------------------------------

def test_status_is_compact_and_leaves_out_the_recordings_list():
    d = decode(B.encode_status(status("recording", 312.7)))
    assert d == {"s": "recording", "t": 312, "max": 5400, "saving": 0}


def test_unreachable_recorder_is_offline():
    assert decode(B.encode_status(None)) == {"s": "offline"}


def test_command_error_wins_over_the_recorders_own():
    d = decode(B.encode_status(status(error="old"), err="Not enough disk space"))
    assert d["err"] == "Not enough disk space"
    assert decode(B.encode_status(status(error="old")))["err"] == "old"


def test_long_errors_and_busy_lists_are_cut_to_fit():
    busy = [f"ffmpeg {1000 + i}" for i in range(40)]
    d = decode(B.encode_status(status(), err="é" * 300, busy=busy))
    assert d["busy"] == busy[:1]
    assert d["err"].endswith("…") and d["err"].startswith("é")


def test_busy_names():
    procs = [{"pid": 12, "command": "/usr/bin/ffmpeg -i udp://@:8554 out.ts"},
             {"pid": 13, "command": "python3 /home/x/fake_rpicam_vid.py -o -"},
             {"pid": 14, "command": ""}]
    assert B.busy_names(procs) == ["ffmpeg 12", "fake_rpicam_vid.py 13", "? 14"]


def test_commands():
    assert B.command_to_request(b"start") == ("/api/start", {})
    assert B.command_to_request(b"start-force\n") == ("/api/start", {"force": True})
    assert B.command_to_request(bytearray(b"STOP")) == ("/api/stop", {})
    assert B.command_to_request(b"rm -rf") is None


# -- Remote: when the watch hears ----------------------------------------------

class StubClient:
    def __init__(self):
        self.st = status()
        self.reply = None
        self.posts = []

    def status(self):
        return self.st

    def post(self, path, body):
        self.posts.append((path, body))
        return self.reply or (self.st, None, None)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def run(coro):
    return asyncio.run(coro)


def test_notifies_on_change_and_every_ten_seconds_only():
    c, clock, sent = StubClient(), Clock(), []
    r = B.Remote(c, sent.append, clock)
    run(r.poll())
    assert len(sent) == 1
    c.st = status("idle")
    clock.t = 3
    run(r.poll())
    assert len(sent) == 1                       # nothing new
    c.st = status("recording", 1)
    clock.t = 4
    run(r.poll())
    assert decode(sent[-1])["s"] == "recording"
    c.st = status("recording", 5)
    clock.t = 8
    run(r.poll())
    assert len(sent) == 2                       # only the clock moved
    clock.t = 14.5
    run(r.poll())
    assert len(sent) == 3 and decode(sent[-1])["t"] == 5


def test_busy_start_then_force():
    c, sent = StubClient(), []
    r = B.Remote(c, sent.append)
    run(r.poll())
    c.reply = (None, "The camera is in use.", [{"pid": 7, "command": "ffmpeg -x"}])
    run(r.command(b"start"))
    d = decode(sent[-1])
    assert d["busy"] == ["ffmpeg 7"] and d["s"] == "idle"
    run(r.poll())                               # still idle: the prompt stays
    assert decode(r.value)["busy"] == ["ffmpeg 7"]
    c.st, c.reply = status("recording", 0), None
    run(r.command(b"start-force"))
    assert c.posts[-1] == ("/api/start", {"force": True})
    assert "busy" not in decode(sent[-1]) and decode(sent[-1])["s"] == "recording"


def test_error_clears_when_the_state_changes():
    c, sent = StubClient(), []
    r = B.Remote(c, sent.append)
    run(r.poll())
    c.reply = (None, "Not enough disk space", None)
    run(r.command(b"start"))
    assert decode(sent[-1])["err"] == "Not enough disk space"
    c.st = status("recording", 0)
    run(r.poll())
    assert "err" not in decode(sent[-1])


def test_unknown_command_is_reported_not_sent():
    c, sent = StubClient(), []
    r = B.Remote(c, sent.append)
    run(r.command(b"reboot"))
    assert c.posts == [] and "unknown command" in decode(sent[-1])["err"]


# -- against the real recorder -------------------------------------------------

def test_recorder_down():
    client = B.RecorderClient("http://127.0.0.1:9", timeout=2)
    assert client.status() is None
    st, err, busy = client.post("/api/start", {})
    assert st is None and "isn't answering" in err and busy is None


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    import recorder as R
    monkeypatch.setattr(R, "camera_users", lambda: [])
    monkeypatch.setattr(R, "CAMERA", "usb")
    monkeypatch.setattr(R, "find_usb_device", lambda: "/dev/v4l/by-id/usb-fake-video-index0")
    monkeypatch.setattr(R, "find_usb_audio", lambda dev: None)
    rec = R.Recorder(tmp_path, camera=FAKE)
    server = ThreadingHTTPServer(("127.0.0.1", 0), R.make_handler(rec))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield R, rec, B.RecorderClient(f"http://127.0.0.1:{server.server_port}")
    if rec.state == R.RECORDING:
        rec.stop()
    server.shutdown()


def test_start_and_stop_through_the_recorder(recorder):
    R, rec, client = recorder
    sent = []
    r = B.Remote(client, sent.append)
    run(r.poll())
    assert decode(sent[-1])["s"] == "idle"
    run(r.command(b"start"))
    assert rec.state == R.RECORDING and decode(sent[-1])["s"] == "recording"
    run(r.command(b"stop"))
    assert rec.state in (R.FINISHING, R.IDLE)
    assert decode(sent[-1])["s"] in ("finishing", "idle")


def test_busy_camera_through_the_recorder(recorder, monkeypatch):
    R, rec, client = recorder
    held = [{"pid": 4242, "command": "/usr/bin/ffmpeg -f mpegts"}]
    monkeypatch.setattr(R, "camera_users", lambda: held)
    closed = []

    def close():
        closed.append(held[:])
        held.clear()
        return []
    monkeypatch.setattr(R, "close_camera_users", close)
    sent = []
    r = B.Remote(client, sent.append)
    run(r.command(b"start"))
    assert decode(sent[-1])["busy"] == ["ffmpeg 4242"] and rec.state == R.IDLE
    run(r.command(b"start-force"))
    assert closed and rec.state == R.RECORDING
    assert "busy" not in decode(sent[-1])
