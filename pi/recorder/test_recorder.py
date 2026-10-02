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
from anya_pi import config as CFG  # noqa: E402
from anya_pi import jobs as J  # noqa: E402

FAKE = str(HERE / "fake_rpicam_vid.py")
REAL_CAMERA_USERS = R.camera_users      # before the autouse stub replaces it

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


@pytest.fixture(autouse=True)
def no_other_camera_users(monkeypatch):
    monkeypatch.setattr(R, "camera_users", lambda: [])


@pytest.fixture
def rec(tmp_path):
    return R.Recorder(tmp_path, camera=FAKE)


@pytest.fixture
def cfg(tmp_path):
    c = CFG.Config(root=tmp_path / "srv")
    c.youtube.enabled = True
    return c


def post_json(url, body):
    req = urllib.request.Request(url, method="POST", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


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


BUSY = [{"pid": 4242, "command": "rpicam-hello -t 0"}]


def test_busy_camera_asks_first_then_closes_on_force(server, rec, monkeypatch):
    monkeypatch.setattr(R, "camera_users", lambda: list(BUSY))
    closed = []
    monkeypatch.setattr(R, "close_camera_users", lambda: closed.append(1) or [])
    code, body = post(server + "/api/start")
    assert code == 409 and body["busy"] == BUSY and rec.state == R.IDLE
    assert not closed
    code, body = post_json(server + "/api/start", {"force": True})
    assert code == 200 and body["state"] == R.RECORDING and closed == [1]
    rec.stop()
    assert saved(rec)


def test_camera_that_will_not_close_is_reported(rec, monkeypatch):
    monkeypatch.setattr(R, "camera_users", lambda: list(BUSY))
    monkeypatch.setattr(R, "close_camera_users", lambda: list(BUSY))
    err = rec.start(force=True)
    assert "could not close rpicam-hello -t 0 (pid 4242)" in err
    assert rec.state == R.IDLE


def test_close_camera_users_falls_back_to_plain_pkill(monkeypatch):
    """The sudo rule missing (first command does nothing) -> the plain one."""
    victim = subprocess.Popen(["sleep", "31415"])
    try:
        monkeypatch.setattr(R, "CAMERA_USERS_CMD", ["pgrep", "-f", "sleep 31415"])
        monkeypatch.setattr(R, "CLOSE_CAMERA_CMDS",
                            (["false"], ["pkill", "-f", "sleep 31415"]))
        monkeypatch.setattr(R, "CLOSE_WAIT_S", 0.5)
        monkeypatch.setattr(R, "camera_users", REAL_CAMERA_USERS)
        assert R.camera_users()[0]["pid"] == victim.pid
        assert R.close_camera_users() == []
        assert victim.wait(5) is not None
    finally:
        victim.kill()


def test_recording_stops_at_the_limit_and_is_queued(tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(R, "MAX_RECORDING_S", 2)
    rec = R.Recorder(tmp_path / "rec", camera=FAKE, cfg=cfg)
    assert rec.start() is None
    assert cfg.recording_flag.is_file()
    assert wait_for(lambda: rec.state == R.IDLE, timeout=10)
    assert saved(rec)
    assert not cfg.recording_flag.exists()
    st = rec.status()
    assert st["notice"].startswith("Stopped at the")
    mp4 = next((tmp_path / "rec").glob("*.mp4"))
    assert 1.5 < R.probe_duration(mp4) < 3.5
    job = J.Queue(cfg).get(mp4.stem)
    assert job["status"] == J.PENDING and job["chapters"] == [str(mp4)]
    assert job["recording"]["source"] == "picam"
    assert job["recording"]["start"] == R.recording_start(mp4.stem).isoformat()
    assert job["youtube_raw"]["status"] == J.UP_PENDING
    assert st["recordings"][0]["chips"][0]["text"] == "Raw: waiting to upload"
    # Never queued twice.
    rec._enqueue(mp4)
    assert len(J.Queue(cfg).all()) == 1


def test_recovered_recording_is_queued_and_stale_flag_cleared(tmp_path, cfg):
    d = tmp_path / "rec"
    d.mkdir()
    ts = d / "2026-10-01_183005.ts"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=50", "-t", "2", "-c:v", "libx264",
                    "-preset", "ultrafast", "-f", "mpegts", str(ts)], check=True)
    cfg.ensure_dirs()
    cfg.recording_flag.write_text("left by a crash")
    rec = R.Recorder(d, camera=FAKE, cfg=cfg)
    assert not cfg.recording_flag.exists()
    rec.recover()
    job = J.Queue(cfg).get("2026-10-01_183005")
    assert job and job["recording"]["start"] == "2026-10-01T18:30:05"


def test_chips():
    assert R.chips(None) == []
    job = {"status": J.RUNNING, "stage": "4/9 Detecting players (near)", "progress": 0.42,
           "youtube_raw": {"status": J.UP_DONE, "url": "https://youtu.be/r",
                           "forced_private": True},
           "youtube": {"status": J.UP_DISABLED}}
    assert R.chips(job) == [
        {"text": "Raw on YouTube (private)", "url": "https://youtu.be/r", "kind": "ok"},
        {"text": "Highlights: Detecting players (near) 42%", "kind": "busy"}]
    job.update(status=J.PENDING, stage="paused while recording")
    assert R.chips(job)[1]["text"] == "Highlights: paused while recording"
    job.update(status=J.DONE, youtube={"status": J.UP_DONE, "url": "https://youtu.be/h"})
    assert R.chips(job)[1] == {"text": "Highlights on YouTube",
                               "url": "https://youtu.be/h", "kind": "ok"}
    job.update(status=J.NEEDS_CALIBRATION)
    assert R.chips(job)[1]["kind"] == "err"


def test_player_name_goes_with_the_recording(tmp_path, cfg):
    rec = R.Recorder(tmp_path / "rec", camera=FAKE, cfg=cfg)
    assert rec.start(player="  Mary-Jane  <O'Neil> ") is None
    assert rec.status()["player"] == "Mary-Jane O'Neil"
    time.sleep(1)
    rec.stop()
    assert saved(rec)
    mp4 = next((tmp_path / "rec").glob("*.mp4"))
    assert J.Queue(cfg).get(mp4.stem)["recording"]["player"] == "Mary-Jane O'Neil"
    st = rec.status()
    assert st["recordings"][0]["player"] == "Mary-Jane O'Neil"
    assert st["default_session"] == "Wimbledon Session"
    assert "free_gb" not in st and "size_mb" not in st["recordings"][0]


def test_player_name_survives_a_crash(tmp_path, cfg):
    d = tmp_path / "rec"
    d.mkdir()
    ts = d / "2026-10-01_183005.ts"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=50", "-t", "1", "-c:v", "libx264",
                    "-preset", "ultrafast", "-f", "mpegts", str(ts)], check=True)
    ts.with_suffix(".player").write_text("Andy\n")
    R.Recorder(d, camera=FAKE, cfg=cfg).recover()
    assert J.Queue(cfg).get(ts.stem)["recording"]["player"] == "Andy"


def test_http_start_passes_the_name(server, rec):
    code, body = post_json(server + "/api/start", {"player": "Andy"})
    assert code == 200 and body["player"] == "Andy"
    rec.stop()
    assert saved(rec)
