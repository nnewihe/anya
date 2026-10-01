"""pipeline/anya2/run.py stream-copy cut: keyframe snapping and a lossless reel."""
import shutil
import subprocess

import pytest

from pipeline.anya2 import run as R
from pipeline.anya2.config import Anya2Config


def test_snap_moves_starts_back_and_merges_overlaps():
    kf = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    segs = [{"start": 1.5, "stop": 2.4}, {"start": 2.6, "stop": 3.5},
            {"start": 4.0, "stop": 4.8}]
    out = R.snap_to_keyframes(segs, kf)
    # 2.6 snaps to 2.0, before the first segment's end: merged, never replayed.
    assert out == [{"start": 1.0, "stop": 3.5}, {"start": 4.0, "stop": 4.8}]


def _probe(path, entries):
    import json
    return json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", entries, "-of", "json", path],
        capture_output=True, text=True, check=True).stdout)["streams"][0]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
@pytest.mark.parametrize("codec,tag", [("libx264", "avc1"), ("libx265", "hvc1")])
def test_copy_reel_keeps_every_frame_and_the_codec(tmp_path, codec, tag):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=12",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
                    "-c:v", codec, "-g", "30", "-bf", "2", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(src)], check=True)
    segs = R.snap_to_keyframes([{"start": 1.3, "stop": 3.0}, {"start": 6.2, "stop": 9.5}],
                               R.keyframe_times(str(src)))
    assert segs == [{"start": 1.0, "stop": 3.0}, {"start": 6.0, "stop": 9.5}]
    cfg = Anya2Config()
    cfg.copy_video = True
    out = tmp_path / "reel.mp4"
    R.cut(str(src), [{"start": 1.3, "stop": 3.0}, {"start": 6.2, "stop": 9.5}],
          str(out), cfg)
    v = _probe(str(out), "stream=width,height,codec_tag_string,nb_read_frames")
    assert (v["width"], v["height"], v["codec_tag_string"]) == (640, 360, tag)
    # (2.0 + 3.5) s * 30 fps = 165.  A copy never loses a frame; it may keep a
    # few past each end, which B-frame reordering drags in (3/segment here).
    n = int(v["nb_read_frames"])
    assert 165 <= n <= 165 + 2 * 4
    a = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=duration", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True, check=True).stdout
    assert abs(float(a) - n / 30.0) < 0.1          # audio follows the video

    # ...and it is the source's own AAC, packet for packet, never re-encoded.
    def hashes(path):
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                              "-show_packets", "-show_data_hash", "MD5",
                              "-show_entries", "packet=data_hash", "-of", "csv=p=0",
                              str(path)], capture_output=True, text=True,
                             check=True).stdout.split()
        return out
    src, reel = hashes(src), hashes(out)
    assert reel and all(h in set(src) for h in reel)


def test_audio_ranges_track_the_video_without_drift():
    pk = [(i * 0.021333, 0.021333) for i in range(3000)]          # 64 s of AAC
    segs = [{"start": 1.0, "stop": 0}, {"start": 20.0, "stop": 0},
            {"start": 40.0, "stop": 0}]
    vd = [5.005, 7.007, 3.003]                                    # 30000/1001 fps
    rng = R.audio_packet_ranges(pk, segs, vd)
    a = v = 0.0
    for (a0, a1), d in zip(rng, vd):
        a += a1 - a0
        v += d
        assert abs(a - v) <= 0.0107 + 1e-6                        # half a packet
