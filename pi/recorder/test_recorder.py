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
import types
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
REAL_CAMERA_USERS = R.camera_users      # before the autouse stubs replace them
REAL_FIND_AUDIO = R.find_usb_audio

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


@pytest.fixture(autouse=True)
def a_webcam_is_plugged_in(monkeypatch):
    monkeypatch.setattr(R, "CAMERA", "usb")
    monkeypatch.setattr(R, "find_usb_device", lambda: "/dev/v4l/by-id/usb-fake-video-index0")
    monkeypatch.setattr(R, "find_usb_audio", lambda dev: "plughw:CARD=OsmoAction4,DEV=0")


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
    assert len(mp4s) == 1 and not list(tmp_path.glob("*.mkv"))
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
        monkeypatch.setattr(R, "_users_cmd", lambda: ["pgrep", "-f", "sleep 31415"])
        monkeypatch.setattr(R, "_close_cmds",
                            lambda: (["false"], ["pkill", "-f", "sleep 31415"]))
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


def test_usb_webcam_saves_its_mjpeg_untouched():
    """Encoding live made the video choppy: recording only copies."""
    assert R.capture_suffix() == ".mkv"
    cmd = R.camera_command("/r/x.mkv")
    assert cmd[0] == "ffmpeg" and cmd[-1] == "/r/x.mkv"
    i = cmd.index("-i")
    assert cmd[i + 1] == "/dev/v4l/by-id/usb-fake-video-index0"
    assert cmd[cmd.index("-input_format") + 1] == "mjpeg"
    assert cmd[cmd.index("-c:v") + 1] == "copy" and "libx264" not in cmd
    assert cmd[-4:-1] == ["-f", "matroska", "-y"]
    assert cmd[cmd.index("-video_size") + 1] == "1920x1080"
    assert cmd[cmd.index("-framerate") + 1] == "30"


def test_usb_webcam_h264_is_copied_into_ts(monkeypatch):
    monkeypatch.setattr(R, "USB_FORMAT", "h264")
    assert R.capture_suffix() == ".ts"
    cmd = R.camera_command("/r/x.ts")
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[-4:-1] == ["-f", "mpegts", "-y"]


def test_pi_camera_command_is_still_available(monkeypatch):
    monkeypatch.setattr(R, "CAMERA", "picam")
    cmd = R.camera_command("/r/x.ts")
    assert cmd[0] == "rpicam-vid" and cmd[-2:] == ["-o", "/r/x.ts"]


def test_no_webcam_plugged_in_is_reported(rec, monkeypatch):
    monkeypatch.setattr(R, "find_usb_device", lambda: None)
    err = rec.start()
    assert "No USB camera found" in err and rec.state == R.IDLE
    assert rec.status()["error"] == err


def test_webcam_users_come_from_fuser(monkeypatch):
    monkeypatch.setattr(R, "camera_users", REAL_CAMERA_USERS)
    monkeypatch.setattr(R, "_users_cmd", lambda: ["sh", "-c", "echo ' 4242 4243'"])
    monkeypatch.setattr(R, "_command_of", lambda pid: f"prog{pid}")
    assert R.camera_users() == [{"pid": 4242, "command": "prog4242"},
                                {"pid": 4243, "command": "prog4243"}]


def test_close_commands_for_the_webcam(monkeypatch):
    monkeypatch.setattr(R.os.path, "realpath", lambda p: "/dev/video0")
    assert R._close_cmds() == (["sudo", "-n", "fuser", "-k", "-TERM", "/dev/video0"],
                               ["fuser", "-k", "-TERM", "/dev/video0"])
    assert R.close_hint() == "sudo fuser -k /dev/video0"


def _mjpeg(path, seconds=2, cut=0):
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=320x180:rate=30", "-t", str(seconds),
                    "-c:v", "mjpeg", "-pix_fmt", "yuvj422p", "-f", "matroska",
                    "-y", str(path)], check=True)
    if cut:                                 # a recording cut off mid-write
        data = path.read_bytes()
        path.write_bytes(data[:len(data) - cut])


def _stream(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=codec_name,pix_fmt,r_frame_rate",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return out.stdout.strip()


def test_mjpeg_is_converted_to_h264_after_stop(tmp_path):
    src = tmp_path / "2026-10-21_180000.mkv"
    _mjpeg(src, seconds=3)
    seen = []
    assert R.finish(src, progress=seen.append) is None
    mp4 = src.with_suffix(".mp4")
    assert not src.exists() and not src.with_suffix(".part.mp4").exists()
    codec, pix, rate = _stream(mp4).split(",")
    assert codec == "h264" and pix.endswith("420p") and rate == "30/1"
    assert abs(R.probe_duration(mp4) - 3) < 0.2
    assert seen and seen[-1] > 0.9


def test_a_cut_off_mjpeg_recording_still_converts(tmp_path):
    src = tmp_path / "2026-10-21_180000.mkv"
    _mjpeg(src, seconds=3, cut=4000)
    assert R.finish(src) is None
    assert R.probe_duration(src.with_suffix(".mp4")) > 1


def test_recording_from_the_webcam_ends_as_a_queued_mp4(tmp_path, cfg):
    rec = R.Recorder(tmp_path / "rec", camera=FAKE, cfg=cfg)
    assert rec.start(player="Andy") is None
    assert rec.file.suffix == ".mkv"
    time.sleep(2)
    rec.stop()
    assert saved(rec)
    files = sorted(p.name for p in (tmp_path / "rec").iterdir() if p.suffix != ".log")
    stem = rec.recordings()[0]["name"][:-4]
    assert files == [f"{stem}.mp4", f"{stem}.player"]
    assert _stream(tmp_path / "rec" / f"{stem}.mp4").startswith("h264,")
    assert J.Queue(cfg).get(stem)["recording"]["player"] == "Andy"


def test_conversions_show_on_the_page(rec):
    rec.converting = {"2026-10-21_180000": None, "2026-10-21_190000": 0.42}
    names = [(r["name"], r["chips"][0]["text"]) for r in rec.recordings()]
    assert names == [("2026-10-21_190000.mp4", "Converting to MP4 42%"),
                     ("2026-10-21_180000.mp4", "Waiting to convert to MP4")]


def test_recovery_converts_leftover_mkv(tmp_path, cfg):
    d = tmp_path / "rec"
    d.mkdir()
    _mjpeg(d / "2026-10-21_180000.mkv")
    R.Recorder(d, camera=FAKE, cfg=cfg).recover()
    assert (d / "2026-10-21_180000.mp4").exists()
    assert J.Queue(cfg).get("2026-10-21_180000")


def test_low_disk_stops_the_recording(rec, monkeypatch):
    assert rec.start() is None
    monkeypatch.setattr(R, "LOW_DISK_STOP_GB", 1e9)
    assert wait_for(lambda: rec.state == R.IDLE, timeout=10)
    assert saved(rec)
    assert "disk is nearly full" in rec.status()["notice"]


def test_the_webcam_microphone_is_recorded_with_the_video():
    cmd = R.camera_command("/r/x.mkv")
    a = cmd.index("alsa")
    assert cmd[a + 1:a + 5] == ["-thread_queue_size", "1024", "-i",
                                "plughw:CARD=OsmoAction4,DEV=0"]
    assert cmd[cmd.index("-ts") + 1] == "mono2abs"           # same clock as ALSA
    assert cmd.index("-ts") < cmd.index("-i")                # a v4l2 input option
    assert ["-map", "1:a", "-c:a", "pcm_s16le"] == cmd[cmd.index("1:a") - 1:
                                                         cmd.index("1:a") + 3]


def test_no_microphone_records_video_only(monkeypatch):
    monkeypatch.setattr(R, "find_usb_audio", lambda dev: None)
    cmd = R.camera_command("/r/x.mkv")
    assert "alsa" not in cmd and "-ts" not in cmd and "1:a" not in cmd


def test_microphone_is_found_on_the_webcams_usb_device(tmp_path, monkeypatch):
    """/sys: video0's interface and card2's interface share one USB device."""
    usb = tmp_path / "usb" / "1-1"
    for d in ("1-1:1.0", "1-1:1.3"):
        (usb / d).mkdir(parents=True)
    other = tmp_path / "usb" / "2-1" / "2-1:1.0"
    other.mkdir(parents=True)
    sysfs = {"/sys/class/video4linux/video0/device": str(usb / "1-1:1.0"),
             "/sys/class/sound/card1/device": str(other),
             "/sys/class/sound/card2/device": str(usb / "1-1:1.3")}
    ids = {"/sys/class/sound/card1": "Headset", "/sys/class/sound/card2": "OsmoAction4"}
    monkeypatch.setattr(R.os.path, "realpath",
                        lambda p: "/dev/video0" if p.startswith("/dev/") else sysfs.get(p, p))
    monkeypatch.setattr(R.glob, "glob", lambda pat: sorted(ids))
    real_path = R.Path
    monkeypatch.setattr(R, "Path", lambda *a: types.SimpleNamespace(
        read_text=lambda: ids[a[0]]) if a[1:] == ("id",) else real_path(*a))
    assert REAL_FIND_AUDIO("/dev/v4l/by-id/usb-DJI-video-index0") == \
        "plughw:CARD=OsmoAction4,DEV=0"
    monkeypatch.setattr(R, "USB_AUDIO", "none")
    assert REAL_FIND_AUDIO("/dev/v4l/by-id/usb-DJI-video-index0") is None


GOPRO_IP = "172.26.161.51"


@pytest.fixture
def gopro(monkeypatch):
    """A HERO8 on its USB network link; returns the HTTP calls made to it."""
    calls = []
    monkeypatch.setattr(R, "CAMERA", "gopro")
    monkeypatch.setattr(R, "find_gopro", lambda: GOPRO_IP)
    monkeypatch.setattr(R, "gopro_call", lambda ip, c: calls.append((ip, c)))
    return calls


def test_gopro_stream_is_copied_into_ts(gopro):
    assert R.capture_suffix() == ".ts"
    cmd = R.camera_command("/r/x.ts")
    assert gopro == [(GOPRO_IP, "START?res=1080")]
    assert cmd[0] == "ffmpeg" and cmd[-1] == "/r/x.ts"
    url = cmd[cmd.index("-i") + 1]
    assert url.startswith("udp://@0.0.0.0:8554?") and "timeout=10000000" in url
    assert cmd[cmd.index("-c") + 1] == "copy" and "libx264" not in cmd
    assert cmd[-4:-1] == ["-f", "mpegts", "-y"]
    assert "v4l2" not in cmd and "alsa" not in cmd
    # Its empty AC-3 track made the muxer refuse the header: video only.
    assert [a for a in cmd if a.startswith("0:")] == ["0:v"]


def test_gopro_recording_ends_as_a_queued_mp4_and_stops_the_stream(
        tmp_path, cfg, gopro):
    rec = R.Recorder(tmp_path / "rec", camera=FAKE, cfg=cfg)
    assert rec.start(player="Andy") is None
    assert rec.file.suffix == ".ts"
    time.sleep(2)
    rec.stop()
    assert saved(rec)
    assert gopro == [(GOPRO_IP, "START?res=1080"), (GOPRO_IP, "STOP")]
    mp4 = next((tmp_path / "rec").glob("*.mp4"))
    assert not list((tmp_path / "rec").glob("*.ts"))
    assert 1.5 < R.probe_duration(mp4) < 3.5
    assert J.Queue(cfg).get(mp4.stem)["recording"]["player"] == "Andy"


def test_gopro_going_quiet_stops_the_stream_too(rec, gopro, monkeypatch):
    monkeypatch.setenv("FAKE_CAMERA_SECONDS", "1")
    assert rec.start() is None
    assert saved(rec)
    assert "camera stopped by itself" in rec.status()["error"]
    assert gopro[-1] == (GOPRO_IP, "STOP")


def test_no_gopro_plugged_in_is_reported(rec, gopro, monkeypatch):
    monkeypatch.setattr(R, "find_gopro", lambda: None)
    err = rec.start()
    assert "No GoPro found" in err and rec.state == R.IDLE and gopro == []


def test_gopro_refusing_to_stream_is_reported(rec, monkeypatch):
    monkeypatch.setattr(R, "CAMERA", "gopro")
    monkeypatch.setattr(R, "find_gopro", lambda: GOPRO_IP)
    monkeypatch.setattr(R, "gopro_call", lambda ip, c: "HTTP Error 500")
    err = rec.start()
    assert err == f"The GoPro ({GOPRO_IP}) would not start streaming: HTTP Error 500"
    assert rec.state == R.IDLE


def test_gopro_address_comes_from_its_usb_network_link(monkeypatch):
    out = ("1: lo    inet 127.0.0.1/8 scope host lo\n"
           "3: wlan0    inet 192.168.1.40/24 brd 192.168.1.255 scope global wlan0\n"
           "5: enxd43260ab1234    inet 172.26.161.52/24 brd 172.26.161.255 scope global\n")
    monkeypatch.setattr(R.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    assert R.find_gopro() == GOPRO_IP
    monkeypatch.setattr(R.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out[:110]))
    assert R.find_gopro() is None
    monkeypatch.setattr(R, "GOPRO_IP", "172.20.110.51")
    assert R.find_gopro() == "172.20.110.51"


def test_gopro_call_reports_an_unreachable_camera():
    # Port 9 on localhost: nothing listens, so the connection is refused.
    assert R.gopro_call("127.0.0.1:9", "STOP")


def test_close_commands_for_the_gopro(gopro):
    assert R._users_cmd() == ["fuser", "8554/udp"]
    assert R._close_cmds() == (["sudo", "-n", "fuser", "-k", "-TERM", "8554/udp"],
                               ["fuser", "-k", "-TERM", "8554/udp"])
    assert R.close_hint() == "sudo fuser -k 8554/udp"


def test_sound_survives_the_conversion(tmp_path):
    src = tmp_path / "2026-10-21_180000.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=30", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000", "-t", "2",
                    "-map", "0:v", "-map", "1:a", "-c:v", "mjpeg", "-pix_fmt", "yuvj422p",
                    "-c:a", "pcm_s16le", "-f", "matroska", "-y", str(src)], check=True)
    assert R.finish(src) is None
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", str(src.with_suffix(".mp4"))],
                         capture_output=True, text=True).stdout.split()
    assert out == ["h264", "aac"]
