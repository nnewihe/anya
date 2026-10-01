"""
Tests for recorder.py against fake_rpicam_vid.py (needs ffmpeg, not a Pi).

    python -m pytest pi/recorder/test_recorder.py
"""

import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import recorder as R  # noqa: E402

FAKE = str(HERE / "fake_rpicam_vid.py")

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def wait_for(cond, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def saved(rec):
    return wait_for(lambda: rec.status()["saving"] == 0 and rec.state == R.IDLE)


@pytest.fixture
def rec(tmp_path):
    return R.Recorder(tmp_path, camera=FAKE)


@pytest.fixture
def server(rec):
    srv = R.ThreadingHTTPServer(("127.0.0.1", 0), R.make_handler(rec))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def post(url):
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_start_stop_gives_an_mp4(rec, tmp_path):
    assert rec.start() is None
    assert rec.status()["state"] == R.RECORDING
    time.sleep(2)
    assert rec.stop() is None
    assert rec.state == R.IDLE
    assert saved(rec)
    mp4s = list(tmp_path.glob("*.mp4"))
    assert len(mp4s) == 1 and not list(tmp_path.glob("*.ts"))
    assert 1.5 < R.probe_duration(mp4s[0]) < 3.5
    st = rec.status()
    assert st["error"] is None
    assert [r["name"] for r in st["recordings"]] == [mp4s[0].name]


def test_http_start_twice_and_stop_idle_are_refused(server, rec):
    code, body = post(server + "/api/stop")
    assert code == 409 and body["error"] == "not recording"
    code, body = post(server + "/api/start")
    assert code == 200 and body["state"] == R.RECORDING
    code, body = post(server + "/api/start")
    assert code == 409 and "already recording" in body["error"]
    time.sleep(1)
    code, body = post(server + "/api/stop")
    assert code == 200 and body["state"] == R.IDLE
    assert saved(rec)
    with urllib.request.urlopen(server + "/") as r:
        assert b"Court Recorder" in r.read()


def test_camera_stopping_by_itself_is_reported(rec, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CAMERA_SECONDS", "1")
    assert rec.start() is None
    assert saved(rec)
    st = rec.status()
    assert "camera stopped by itself" in st["error"]
    # What it did record is still kept.
    assert len(list(tmp_path.glob("*.mp4"))) == 1


def test_camera_that_cannot_start_is_reported(rec, monkeypatch):
    monkeypatch.setenv("FAKE_CAMERA_FAIL", "1")
    assert rec.start() is None
    assert saved(rec)
    assert "no cameras available" in rec.status()["error"]


def test_missing_camera_binary(tmp_path):
    rec = R.Recorder(tmp_path, camera=str(tmp_path / "no-such-rpicam-vid"))
    assert "could not run" in rec.start()
    assert rec.state == R.IDLE


def test_low_disk_refuses_to_start(rec, monkeypatch):
    monkeypatch.setattr(R, "MIN_FREE_GB", 1e9)
    assert "GB free" in rec.start()
    assert rec.state == R.IDLE


def test_leftover_ts_is_recovered_at_startup(tmp_path):
    ts = tmp_path / "2026-01-01_100000.ts"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=60", "-t", "2", "-c:v", "libx264",
                    "-preset", "ultrafast", "-f", "mpegts", str(ts)], check=True)
    rec = R.Recorder(tmp_path, camera=FAKE)
    rec.recover()
    assert not ts.exists()
    assert abs(R.probe_duration(ts.with_suffix(".mp4")) - 2) < 0.2
    assert rec.error is None


def test_unreadable_ts_is_left_alone(tmp_path):
    ts = tmp_path / "2026-01-01_100000.ts"
    ts.write_bytes(b"not video")
    rec = R.Recorder(tmp_path, camera=FAKE)
    rec.recover()
    assert ts.exists() and "unreadable" in rec.error
