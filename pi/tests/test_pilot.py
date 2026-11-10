"""Phone court calibration and the club playlist (pi/anya_pi calibrate, youtube)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from anya_pi import calibrate as K
from anya_pi import config as CFG
from anya_pi import jobs as J
from anya_pi import youtube as Y

GOOD = [[100, 500], [860, 500], [640, 150], [320, 150]]


@pytest.mark.parametrize("pts, why", [
    (GOOD[:3], "need 4"),
    ([[100, 500], [860, 500], [320, 150], [640, 150]], "cross over"),
    ([[320, 150], [640, 150], [860, 500], [100, 500]], "lower in the picture"),
    ([[860, 500], [100, 500], [320, 150], [640, 150]], "swapped"),
    ([[100, 600], [860, 500], [640, 150], [320, 150]], "outside the picture"),
    ("nonsense", "pairs"),
])
def test_bad_corners_are_refused(pts, why):
    with pytest.raises(K.BadCorners, match=why):
        K.check_corners(pts)


def test_good_corners_pass():
    assert K.check_corners(GOOD) == [tuple(map(float, p)) for p in GOOD]


@pytest.fixture
def cfg(tmp_path):
    c = CFG.Config(root=tmp_path / "srv")
    c.ensure_dirs()
    return c


@pytest.fixture
def video(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    pytest.importorskip("cv2")
    p = tmp_path / "2026-11-10_100000.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=1920x1080:rate=30", "-t", "4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_save_writes_profile_and_requeues_waiting_jobs(cfg, video, tmp_path):
    from pipeline.anya2 import site as S
    q = J.Queue(cfg)
    q.create("waiting", {"start": "2026-11-10T10:00:00"}, [video],
             status=J.NEEDS_CALIBRATION, error="no site")
    q.create("failed", {"start": "2026-11-10T10:00:00"}, [video], status=J.FAILED)

    info = K.frame(video, tmp_path / "f.png")
    assert (info["width"], info["height"], info["source_size"]) == (960, 540, [1920, 1080])

    out = K.save(cfg, video, GOOD)
    assert out["requeued"] == ["waiting"]
    assert q.get("waiting")["status"] == J.PENDING
    assert q.get("failed")["status"] == J.FAILED
    prof = json.loads((cfg.site_dir / "court.json").read_text())
    assert prof["points"] == [list(map(float, p)) for p in GOOD]
    assert prof["source_size"] == [1920, 1080]
    # The profile carries back onto the same recording with no shift, exactly
    # as one made by `site save` on a laptop would.
    other = tmp_path / "again" / video.name
    other.parent.mkdir()
    shutil.copy(video, other)
    assert S.apply(str(cfg.site_dir), str(other)) == pytest.approx(0.0, abs=0.5)
    assert not list(cfg.work.iterdir()), "scratch work dir left behind"

    K.save(cfg, video, GOOD)                       # again: the old one is kept
    assert (cfg.site_dir.with_name("site.prev") / "court.json").is_file()


def test_cli_prints_one_json_line(cfg, video, tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(f'root = "{cfg.root}"\n')
    root = Path(__file__).resolve().parents[2]
    env = {"PYTHONPATH": f"{root}:{root / 'pi'}", "PATH": "/usr/bin:/bin:/usr/local/bin:"
           + str(Path(shutil.which("ffmpeg")).parent)}
    run = lambda *a: subprocess.run([sys.executable, "-m", "anya_pi", "--config",
                                     str(cfg_file), *a], capture_output=True,
                                    text=True, env=env)
    r = run("calibrate-save", str(video), json.dumps(GOOD[::-1]))
    assert r.returncode == 2 and json.loads(r.stdout)["ok"] is False
    r = run("calibrate-save", str(video), json.dumps(GOOD))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1])["ok"] is True
    assert len(r.stdout.strip().splitlines()) == 1


# -- YouTube playlist --------------------------------------------------------

class _Req:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class _HttpError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.resp = type("R", (), {"status": status})()


class FakeYT:
    def __init__(self, refuse_position=False):
        self.items, self.lists, self.refuse = [], [], refuse_position

    def playlistItems(self):
        yt = self

        class PI:
            def insert(self, part, body):
                def go():
                    if yt.refuse and "position" in body["snippet"]:
                        raise _HttpError(400)
                    yt.items.append(body["snippet"])
                    return {"id": "x"}
                return _Req(go)
        return PI()

    def playlists(self):
        yt = self

        class PL:
            def insert(self, part, body):
                yt.lists.append(body)
                return _Req(lambda: {"id": "PLabc",
                                     "status": {"privacyStatus": body["status"]["privacyStatus"]}})
        return PL()


def test_new_videos_go_to_the_top_of_the_playlist():
    yt = FakeYT()
    Y.add_to_playlist(yt, "PLabc", "v1")
    assert yt.items == [{"playlistId": "PLabc", "position": 0,
                         "resourceId": {"kind": "youtube#video", "videoId": "v1"}}]


def test_a_resorted_playlist_still_gets_the_video():
    yt = FakeYT(refuse_position=True)
    Y.add_to_playlist(yt, "PLabc", "v1")
    assert yt.items[0]["resourceId"]["videoId"] == "v1"
    assert "position" not in yt.items[0]


def test_create_playlist():
    yt = FakeYT()
    pl = Y.create_playlist(None, "Wimbledon club court", privacy="unlisted", service=yt)
    assert pl == {"id": "PLabc", "privacy": "unlisted",
                  "url": "https://www.youtube.com/playlist?list=PLabc"}
    assert yt.lists[0]["status"] == {"privacyStatus": "unlisted"}
