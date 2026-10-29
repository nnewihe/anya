"""pipeline/dji.py: which files on a DJI card belong to which recording."""
import datetime as dt
import shutil
import subprocess

import pytest

from pipeline import dji


def _probe_for(meta):
    return lambda p: meta[p.rsplit("/", 1)[-1]]


def _m(dur, w=3840, h=2160, codec="hevc", pix="yuv420p", fps=29.97):
    return {"codec": codec, "pix_fmt": pix, "width": w, "height": h,
            "fps": fps, "duration": dur, "frames": None}


def _touch(d, name):
    p = d / name
    p.write_bytes(b"x")
    return p


def test_groups_consecutive_chapters_and_ignores_decoys(tmp_path):
    d = tmp_path / "DCIM" / "DJI_001"
    d.mkdir(parents=True)
    names = {
        "DJI_20260923180000_0001_D.MP4": _m(600.0),     # chapter 1
        "DJI_20260923181000_0002_D.MP4": _m(300.0),     # chapter 2: starts at +600s
        "DJI_20260923183000_0003_D.MP4": _m(120.0),     # new press of record
    }
    for n in names:
        _touch(d, n)
    for decoy in ("DJI_20260923180000_0001_D.LRF", "DJI_20260923182000_0004_D.JPG",
                  "._DJI_20260923180000_0001_D.MP4", "notes.MP4"):
        _touch(d, decoy)
    recs = dji.find_recordings(str(tmp_path), probe=_probe_for(names))
    assert [len(r.chapters) for r in recs] == [2, 1]
    assert recs[0].id == "20260923_180000_0001"
    assert recs[0].duration == 900.0
    assert all(p.endswith(".MP4") for r in recs for p in r.paths)


def test_same_timestamp_chapters_group():
    t = dt.datetime(2026, 9, 23, 18)
    a = dji.Chapter("a", t, 1, "hevc", "yuv420p", 3840, 2160, 29.97, 600, 1)
    b = dji.Chapter("b", t, 2, "hevc", "yuv420p", 3840, 2160, 29.97, 600, 1)
    assert len(dji.group([b, a])) == 1
    assert dji.group([b, a])[0].paths == ["a", "b"]


def test_gap_or_setting_change_splits():
    t = dt.datetime(2026, 9, 23, 18)
    a = dji.Chapter("a", t, 1, "hevc", "yuv420p", 3840, 2160, 29.97, 600, 1)
    late = dji.Chapter("b", t + dt.timedelta(seconds=610), 2, "hevc", "yuv420p",
                       3840, 2160, 29.97, 60, 1)
    other = dji.Chapter("c", t + dt.timedelta(seconds=600), 2, "h264", "yuv420p",
                        3840, 2160, 29.97, 60, 1)
    assert len(dji.group([a, late])) == 2
    assert len(dji.group([a, other])) == 2


@pytest.mark.parametrize("meta,msg", [
    (_m(10, 3840, 2880), "16:9"),
    (_m(10, pix="yuv420p10le"), "10-bit"),
])
def test_validate_rejects(meta, msg):
    c = dji.Chapter("x.MP4", dt.datetime(2026, 1, 1), 1, meta["codec"], meta["pix_fmt"],
                    meta["width"], meta["height"], meta["fps"], 10, 1)
    with pytest.raises(dji.UnsupportedFootage, match=msg):
        dji.validate(dji.Recording([c]))


def test_validate_warns_h264_and_60fps():
    c = dji.Chapter("x.MP4", dt.datetime(2026, 1, 1), 1, "h264", "yuv420p",
                    3840, 2160, 59.94, 10, 1)
    w = dji.validate(dji.Recording([c]))
    assert any("HEVC" in x for x in w) and any("fps" in x for x in w)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_real_files_probe_and_group(tmp_path):
    d = tmp_path / "DCIM" / "DJI_001"
    d.mkdir(parents=True)

    def make(name, secs):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        f"testsrc=size=320x180:rate=30:duration={secs}",
                        "-pix_fmt", "yuv420p", str(d / name)], check=True)
    make("DJI_20260923180000_0001_D.MP4", 2)
    make("DJI_20260923180002_0002_D.MP4", 2)
    make("DJI_20260923180100_0003_D.MP4", 1)
    (d / "DJI_20260923180000_0001_D.LRF").write_bytes(b"not a video")
    recs = dji.find_recordings(str(tmp_path))
    assert [len(r.chapters) for r in recs] == [2, 1]
    assert recs[0].chapters[0].width == 320 and recs[0].chapters[0].fps == 30
