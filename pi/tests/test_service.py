"""pi/anya_pi: ingest -> queue -> worker -> YouTube, without the pipeline or the network."""
import shutil
import subprocess
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
    assert calls[0] == "Tennis 2026-09-23 18:00"
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
