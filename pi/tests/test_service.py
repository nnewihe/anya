"""pi/anya_pi: ingest -> queue -> worker -> YouTube, without the pipeline or the network."""
import shutil
import signal
import subprocess
import threading
import time
import types
from pathlib import Path

import pytest

from anya_pi import config as CFG
from anya_pi import ingest as I
from anya_pi import jobs as J
from anya_pi import worker as W
from anya_pi import youtube as Y


@pytest.fixture
def cfg(tmp_path):
    c = CFG.Config(root=tmp_path / "srv")
    c.processing.keep_inbox_days = 0
    c.ensure_dirs()
    return c


def _card(tmp_path, size="320x180"):
    d = tmp_path / "card" / "DCIM" / "DJI_001"
    d.mkdir(parents=True)
    for name, secs in (("DJI_20260923180000_0001_D.MP4", 2),
                       ("DJI_20260923180002_0002_D.MP4", 2)):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc=size={size}:rate=30:duration={secs}",
                        "-pix_fmt", "yuv420p", str(d / name)], check=True)
    (d / "DJI_20260923180000_0001_D.LRF").write_bytes(b"preview")
    return tmp_path / "card"


needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


@needs_ffmpeg
def test_ingest_copies_once_and_queues(cfg, tmp_path):
    card = _card(tmp_path)
    before = sorted(p.name for p in card.rglob("*"))
    ids = I.ingest(cfg, card, log=lambda *_: None)
    assert ids == ["20260923_180000_0001"]
    job = J.Queue(cfg).get(ids[0])
    assert job["status"] == J.PENDING and len(job["chapters"]) == 2
    assert all(Path(c).is_file() for c in job["chapters"])
    assert not any(c.endswith(".LRF") for c in job["chapters"])
    # Re-plug: nothing new, and the card is untouched.
    assert I.ingest(cfg, card, log=lambda *_: None) == []
    assert sorted(p.name for p in card.rglob("*")) == before


@needs_ffmpeg
def test_ingest_flags_unsupported_4x3(cfg, tmp_path):
    card = _card(tmp_path, size="320x240")
    ids = I.ingest(cfg, card, log=lambda *_: None)
    job = J.Queue(cfg).get(ids[0])
    assert job["status"] == J.UNSUPPORTED and "16:9" in job["error"]
    assert job["chapters"] == []


def _queued(cfg, tmp_path):
    q = J.Queue(cfg)
    ch = tmp_path / "in.mp4"
    ch.write_bytes(b"v")
    return q, q.create("20260923_180000_0001",
                       {"start": "2026-09-23T18:00:00", "duration": 600.0}, [ch])


def test_worker_success_writes_reel_status_and_cleans_up(cfg, tmp_path, monkeypatch):
    from pipeline.anya2 import headless as H
    q, job = _queued(cfg, tmp_path)
    cfg.youtube.enabled = True

    def fake_build(videos, output, c, site=None, on_progress=None, dry_run=False):
        on_progress(4, 9, "Detecting players (near)", 0.5)
        Path(output).write_bytes(b"reel")
        (cfg.work / job["id"] / "big_interim.mp4").write_bytes(b"x")
        return [{"start": 1.0, "stop": 11.0}, {"start": 20.0, "stop": 25.0}], output

    monkeypatch.setattr(H, "build", fake_build)
    W.process(cfg, job, log=lambda *_: None)
    j = q.get(job["id"])
    assert j["status"] == J.DONE and j["segments"] == 2 and j["kept_s"] == 15.0
    assert (cfg.reels / "2026-09-23_1800_tennis.mp4").is_file()
    assert (cfg.reels / "2026-09-23_1800_tennis.segments.json").is_file()
    assert not (cfg.work / job["id"]).exists()
    assert j["youtube"]["status"] == J.UP_PENDING


def test_worker_needs_calibration(cfg, tmp_path, monkeypatch):
    from pipeline.anya2 import headless as H
    q, job = _queued(cfg, tmp_path)

    def fake_build(*a, **k):
        raise H.NotCalibrated("no corners")

    monkeypatch.setattr(H, "build", fake_build)
    W.process(cfg, job, log=lambda *_: None)
    assert q.get(job["id"])["status"] == J.NEEDS_CALIBRATION
    W.write_status(cfg, q)
    assert "needs_calibration" in (cfg.reels / "STATUS.txt").read_text()


def test_interrupted_job_is_resumed(cfg, tmp_path):
    q, job = _queued(cfg, tmp_path)
    job["status"] = J.RUNNING
    q.put(job)
    assert q.recover() == 1 and q.next_pending()["id"] == job["id"]


def test_upload_retries_then_records_url(cfg, tmp_path):
    q, job = _queued(cfg, tmp_path)
    reel = cfg.reels / "r.mp4"
    reel.write_bytes(b"reel")
    job.update(status=J.DONE, reel=str(reel), segments=3, kept_s=60,
               youtube={"status": J.UP_PENDING, "attempts": 0})
    q.put(job)
    cfg.youtube.enabled = True
    cfg.youtube_token.write_text("{}")
    calls = []

    def flaky(path, title, desc, token, privacy, playlist_id, log):
        calls.append(title)
        if len(calls) == 1:
            raise ConnectionError("no network at the court")
        return {"video_id": "abc", "url": "https://youtu.be/abc", "privacy": "private",
                "requested": privacy, "forced_private": True}

    W.upload_pending(cfg, log=lambda *_: None, uploader=flaky)
    assert q.get(job["id"])["youtube"]["status"] == J.UP_PENDING
    W.upload_pending(cfg, log=lambda *_: None, uploader=flaky)
    yt = q.get(job["id"])["youtube"]
    assert yt["status"] == J.UP_DONE and yt["forced_private"]
    assert calls[0] == "6:00 PM · Sep 23, 2026 · Wimbledon Session Highlights"
    assert (cfg.reels / "r.youtube.txt").read_text().strip() == "https://youtu.be/abc"


def test_resumable_retries_transient_only():
    class Req:
        def __init__(self, script):
            self.script = list(script)

        def next_chunk(self):
            x = self.script.pop(0)
            if isinstance(x, Exception):
                raise x
            return x

    prog = types.SimpleNamespace(progress=lambda: 0.5)
    ok = Req([ConnectionError("x"), (prog, None), (None, {"id": "v"})])
    assert Y.run_resumable(ok, sleep=lambda s: None, log=lambda *_: None) == {"id": "v"}
    with pytest.raises(FileNotFoundError):
        Y.run_resumable(Req([FileNotFoundError("gone")]), sleep=lambda s: None,
                        log=lambda *_: None)
    with pytest.raises(Y.UploadError):
        Y.run_resumable(Req([ConnectionError("x")] * 5), sleep=lambda s: None,
                        log=lambda *_: None, max_retries=3)


def test_config_file(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('root = "/x"\n[processing]\nbackend = "onnx"\n[youtube]\nenabled = true\n')
    c = CFG.load(p)
    assert str(c.root) == "/x" and c.processing.backend == "onnx" and c.youtube.enabled
    assert c.youtube.privacy == "unlisted"
    p.write_text('[youtube]\nprivcy = "public"\n')
    with pytest.raises(ValueError, match="privcy"):
        CFG.load(p)


def test_titles(cfg):
    job = {"recording": {"start": "2026-10-01T18:30:05"}}
    assert W.titles(job, cfg) == ("6:30 PM · Oct 1, 2026 · Wimbledon Session",
                                  "6:30 PM · Oct 1, 2026 · Wimbledon Session Highlights")
    job = {"recording": {"start": "2026-10-01T00:05:00"}}
    assert W.titles(job, cfg)[0].startswith("12:05 AM · Oct 1, 2026")


def test_raw_is_uploaded_and_then_the_highlights(cfg, tmp_path):
    q = J.Queue(cfg)
    rec = tmp_path / "2026-10-01_183005.mp4"
    rec.write_bytes(b"raw")
    job = q.create("2026-10-01_183005",
                   {"start": "2026-10-01T18:30:05", "duration": 5400.0, "source": "picam"},
                   [rec], upload_raw=True)
    cfg.youtube.enabled = True
    cfg.youtube_token.write_text("{}")
    calls = []

    def ok(path, title, desc, token, privacy, playlist_id, log):
        calls.append((Path(path).name, title))
        assert q.get(job["id"])[("youtube_raw" if path == str(rec) else "youtube")][
            "status"] == J.UP_UPLOADING
        return {"video_id": "v", "url": f"https://youtu.be/{len(calls)}",
                "privacy": "unlisted", "requested": privacy, "forced_private": False}

    W.upload_pending(cfg, log=lambda *_: None, uploader=ok)
    assert calls == [(rec.name, "6:30 PM · Oct 1, 2026 · Wimbledon Session")]
    assert q.get(job["id"])["youtube_raw"]["url"] == "https://youtu.be/1"

    reel = cfg.reels / "reel.mp4"
    reel.write_bytes(b"reel")
    q.update(job["id"], status=J.DONE, reel=str(reel), segments=4, kept_s=600,
             youtube={"status": J.UP_PENDING, "attempts": 0})
    W.upload_pending(cfg, log=lambda *_: None, uploader=ok)
    assert calls[1] == ("reel.mp4", "6:30 PM · Oct 1, 2026 · Wimbledon Session Highlights")
    j = q.get(job["id"])
    assert j["youtube"]["status"] == J.UP_DONE and j["youtube_raw"]["status"] == J.UP_DONE


def test_interrupted_upload_starts_again(cfg, tmp_path):
    q, job = _queued(cfg, tmp_path)
    q.update(job["id"], youtube_raw={"status": J.UP_UPLOADING, "attempts": 1})
    q.recover()
    assert q.get(job["id"])["youtube_raw"]["status"] == J.UP_PENDING


def test_modify_from_two_threads_keeps_both_writers_fields(cfg, tmp_path):
    q, job = _queued(cfg, tmp_path)

    def bump(key):
        for _ in range(50):
            q.modify(job["id"], lambda j: j.__setitem__(key, j.get(key, 0) + 1))

    ts = [threading.Thread(target=bump, args=(k,)) for k in ("a", "b")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    j = q.get(job["id"])
    assert j["a"] == 50 and j["b"] == 50


def test_processing_pauses_while_the_camera_records(cfg, tmp_path, monkeypatch):
    from pipeline import cancel
    q, job = _queued(cfg, tmp_path)
    cfg.poll_s = 1
    seen = []

    def fake_process(c, j, log):
        seen.append(j.get("stage"))
        if len(seen) == 1:
            # The recorder presses Start mid-run; the watcher must cancel us.
            cfg.recording_flag.write_text("x.ts")
            threading.Timer(3, cfg.recording_flag.unlink).start()
            end = time.time() + 10
            while time.time() < end:
                cancel.check()
                time.sleep(0.05)
            raise AssertionError("never cancelled")
        q.update(j["id"], status=J.DONE)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)   # end the loop

    monkeypatch.setattr(W, "process", fake_process)
    old = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        W.run(cfg, log=lambda *_: None)
    finally:
        signal.signal(signal.SIGTERM, old[0])
        signal.signal(signal.SIGINT, old[1])
        cancel.clear()
    assert seen == [None, "paused while recording"]
    assert q.get(job["id"])["status"] == J.DONE
