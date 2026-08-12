"""
proxy.py
========
One-time ffmpeg transcodes that the fast extraction paths decode instead of
the source video.

Decode dominates every one of these passes once inference comes down.  On a
4K source a frame costs ~6.7 ms to read, of which ~4.1 ms is reconstruction
that cannot be skipped even for frames we throw away (an H.264 frame is a
difference against its predecessors), so decoding *less often* barely helps
and decoding something *smaller* is the whole win.

Two shapes of proxy, both frame-exact against the source:

    ensure_proxy       whole frame, downscaled   (anya_near_telemetry: the
                       near player is large and every coordinate it records
                       is already in 960x540 analysis space)

    ensure_crop_proxy  a crop at NATIVE resolution (anya_far_telemetry: the
                       far player is ~25 px tall at 540p, which is why the
                       full pass runs a second native-resolution model call
                       on a band around the far baseline in the first place)

Both write a `<proxy>.build.json` sidecar carrying the parameters used.  A
frame-count check alone cannot tell a CRF 20 proxy from a CRF 14 one, so
without the sidecar a quality change would silently reuse the old file and
quietly invalidate any A/B comparing them.

Frame indices must map 1:1 to the source: every record the extractors write
is keyed by source frame number, and so is the ground truth.  `-fps_mode
passthrough` keeps ffmpeg from dropping or duplicating frames to hit a target
rate, and the result is verified against the source frame count before the
proxy is accepted.  Anything that does not come back frame-exact returns the
SOURCE path unchanged — a slow correct run beats a fast wrong one.
"""

import json
import os
import shutil
import subprocess
import time
from typing import Optional, Sequence, Tuple

try:                                        # package import (python -m pipeline.x)
    from .utilities import probe_video
    from .subproc import run as _run
except ImportError:                         # script import (python pipeline/x.py)
    from utilities import probe_video
    from subproc import run as _run

PROXY_SUFFIX      = "_proxy540.mp4"
FAR_BAND_SUFFIX   = "_farband.mp4"


def proxy_path_for(video_path: str, suffix: str = PROXY_SUFFIX) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{suffix}")


def _transcode(video_path: str, out: str, vf: str, want: dict,
               crf: int, preset: str, label: str, force: bool) -> str:
    """Build `out` from `video_path` with filter `vf`, or reuse a matching one.

    `want` is the full build description written to the sidecar; it must
    already carry the source frame count.  Returns `out` on success and
    `video_path` on any failure, so a caller can always just decode whatever
    comes back.
    """
    meta_path = out + ".build.json"
    src_n = int(want["frames"])

    if not force and os.path.isfile(out):
        try:
            have = json.load(open(meta_path)) if os.path.isfile(meta_path) else None
            if have == want and int(probe_video(out)["frame_count"]) == src_n:
                print(f"[{label}] Using cached proxy: {out}")
                return out
            print(f"[{label}] Cached proxy was built with {have} but {want} is "
                  f"wanted — rebuilding.")
        except Exception:
            pass

    if shutil.which("ffmpeg") is None:
        print(f"[{label}] WARN: ffmpeg not found — decoding the source directly.")
        return video_path

    tmp = out + ".part.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", video_path,
           "-vf", vf, "-fps_mode", "passthrough",
           "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
           "-pix_fmt", "yuv420p", "-an", tmp]
    print(f"[{label}] Building proxy ({vf}, crf {crf}, one-time)…")
    t0 = time.perf_counter()
    try:
        _run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as ex:
        print(f"[{label}] WARN: proxy transcode failed ({ex}) — using source.")
        if os.path.isfile(tmp):
            os.remove(tmp)
        return video_path

    try:
        proxy_n = int(probe_video(tmp)["frame_count"])
    except Exception:
        proxy_n = -1
    if proxy_n != src_n:
        print(f"[{label}] WARN: proxy has {proxy_n} frames vs source {src_n} "
              "— discarding it and using the source.")
        os.remove(tmp)
        return video_path

    os.replace(tmp, out)
    with open(meta_path, "w") as fh:
        json.dump(want, fh)
    print(f"[{label}] proxy → {out}  ({time.perf_counter() - t0:.1f}s, "
          f"{proxy_n} frames, crf {crf})")
    return out


def ensure_proxy(video_path: str, size: Tuple[int, int] = (960, 540),
                 crf: int = 20, preset: str = "veryfast",
                 force: bool = False, label: str = "PROXY") -> str:
    """Transcode `video_path` to a whole-frame `size` proxy once; return its path.

    CRF matters more than it looks.  A tennis ball mid-toss is small, fast and
    low-contrast — precisely what x264 spends its bit budget last on — and at
    CRF 20 the encoder was deleting it outright.  Measured on Data/38 at an
    identical 960x540 either way, so this is the RE-ENCODE and not the
    downscale: surviving toss-ROI detections per serve went
        CRF 20 -> [17, 4, 4, 12, 0, 2, 0, 1]      (4 serves with no toss at all)
        CRF 14 -> [25, 32, 25, 21, 10, 10, 14, 24]
        source -> [28, 34, 24, 18, 19, 12, 34, 21]
    Callers that care about the ball should pass crf <= 14.
    """
    w, h = size
    want = {"size": [w, h], "crf": int(crf), "preset": str(preset),
            "frames": int(probe_video(video_path)["frame_count"])}
    return _transcode(video_path, proxy_path_for(video_path, PROXY_SUFFIX),
                      f"scale={w}:{h}", want, crf, preset, label, force)


def ensure_crop_proxy(video_path: str, crop: Sequence[int],
                      suffix: str = FAR_BAND_SUFFIX,
                      crf: int = 14, preset: str = "veryfast",
                      force: bool = False, label: str = "PROXY",
                      extra: Optional[dict] = None) -> str:
    """Transcode a native-resolution [x1,y1,x2,y2] crop of `video_path`.

    No scaling: the point of this one is to keep source pixels on a subject
    that is small in the frame while paying to decode only the part of the
    frame it occupies.  The crop rectangle travels in the sidecar, so
    recalibrating the court (which moves the rectangle) rebuilds the proxy
    rather than silently reusing a band aimed somewhere else.

    ffmpeg needs even width/height for yuv420p; the rectangle is rounded
    outward and the effective one is returned to the caller in the sidecar.
    """
    x1, y1, x2, y2 = (int(v) for v in crop)
    w = (x2 - x1) & ~1
    h = (y2 - y1) & ~1
    want = {"crop": [x1, y1, w, h], "crf": int(crf), "preset": str(preset),
            "frames": int(probe_video(video_path)["frame_count"])}
    if extra:
        want.update(extra)
    out = _transcode(video_path, proxy_path_for(video_path, suffix),
                     f"crop={w}:{h}:{x1}:{y1}", want, crf, preset, label, force)
    return out
