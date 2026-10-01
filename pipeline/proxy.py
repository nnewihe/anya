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
import logging
import os
import shutil
import subprocess
import time
from typing import Optional, Sequence, Tuple

try:                                        # package import (python -m pipeline.x)
    from .utilities import probe_video
    from .subproc import run as _run
    from . import cancel as _cancel
    from . import workdir as _workdir
except ImportError:                         # script import (python pipeline/x.py)
    from utilities import probe_video
    from subproc import run as _run
    import cancel as _cancel
    import workdir as _workdir

# Every degradation below is recoverable, so none of them stops the run — but
# a windowed PyInstaller build owns no console, `sys.stdout` goes nowhere, and
# a `print` here is therefore invisible on exactly the machines that need it.
# The desktop app configures the root logger (desktop/applog.py) to a rotating
# file, so warnings routed here reach a tester's app.log; running from a
# terminal they still print as before.
_log = logging.getLogger("anya_tennis.proxy")


def _warn(msg: str) -> None:
    print(msg)
    _log.warning(msg)

PROXY_SUFFIX      = "_proxy540.mp4"
FAR_BAND_SUFFIX   = "_farband.mp4"


def proxy_path_for(video_path: str, suffix: str = PROXY_SUFFIX) -> str:
    d = _workdir.artifact_dir(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{suffix}")


def hwaccel_args() -> list:
    """`-hwaccel X` for every ffmpeg DECODE of a source, from ANYA_FFMPEG_HWACCEL.

    Off unless set.  On a Raspberry Pi 5 `drm` puts HEVC decode on the SoC's
    hardware decoder, which is the difference between the two 4K proxy decodes
    costing about as much as pose inference and costing a fraction of it.  An
    input the accelerator cannot take (H.264 on a Pi 5) is not an error:
    ffmpeg falls back to software decode by itself.  Frames come back in system
    memory, so every software filter below works unchanged.
    """
    v = os.environ.get("ANYA_FFMPEG_HWACCEL", "").strip()
    return ["-hwaccel", v] if v and v.lower() not in ("0", "off", "none") else []


def _cached(out: str, want: dict, label: str) -> bool:
    """True when `out` exists, was built as `want`, and is still frame-exact."""
    if not os.path.isfile(out):
        return False
    meta_path = out + ".build.json"
    try:
        have = json.load(open(meta_path)) if os.path.isfile(meta_path) else None
        if have == want and int(probe_video(out)["frame_count"]) == int(want["frames"]):
            return True
        print(f"[{label}] Cached proxy was built with {have} but {want} is "
              f"wanted — rebuilding.")
    except Exception:
        pass
    return False


def _encode_args(crf: int, preset: str) -> list:
    return ["-fps_mode", "passthrough", "-c:v", "libx264", "-crf", str(crf),
            "-preset", preset, "-pix_fmt", "yuv420p", "-an"]


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

    if not force and _cached(out, want, label):
        print(f"[{label}] Using cached proxy: {out}")
        return out

    if shutil.which("ffmpeg") is None:
        _warn(f"[{label}] WARN: ffmpeg not found — decoding the source "
              f"directly. Packaged builds bundle their own, so this means a "
              f"source run on a machine without one.")
        return video_path

    tmp = out + ".part.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", *hwaccel_args(), "-i", video_path,
           "-vf", vf, *_encode_args(crf, preset), tmp]
    print(f"[{label}] Building proxy ({vf}, crf {crf}, one-time)…")
    t0 = time.perf_counter()
    try:
        # Cancellable: this is a single multi-minute call on a long match, so
        # a run cancelled here has to reach into the child rather than wait
        # for it. See pipeline.cancel.
        _cancel.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as ex:
        # `capture_output=True` swallows ffmpeg's stderr into the exception,
        # so without this the reason is lost and all a tester's log shows is
        # that the run got slower and — on a source the fallback decoder
        # cannot read — wrong.
        err = ex.stderr or b""
        if isinstance(err, (bytes, bytearray)):
            err = err.decode("utf-8", "replace")
        _warn(f"[{label}] WARN: proxy transcode failed ({ex}) — using source. "
              f"ffmpeg said: {err.strip()[-2000:] or '(nothing)'}")
        if os.path.isfile(tmp):
            os.remove(tmp)
        return video_path

    try:
        proxy_n = int(probe_video(tmp)["frame_count"])
    except Exception:
        proxy_n = -1
    if proxy_n != src_n:
        _warn(f"[{label}] WARN: proxy has {proxy_n} frames vs source {src_n} "
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
    want, vf = _proxy_spec(video_path, size, crf, preset)
    return _transcode(video_path, proxy_path_for(video_path, PROXY_SUFFIX),
                      vf, want, crf, preset, label, force)


def _proxy_spec(video_path, size, crf, preset, frames=None):
    """(sidecar description, filter) of a whole-frame proxy.  ONE definition,
    shared with `ensure_proxies_once`, so a proxy built either way is the
    cache hit of the other."""
    w, h = size
    n = frames if frames is not None else int(probe_video(video_path)["frame_count"])
    return ({"size": [w, h], "crf": int(crf), "preset": str(preset),
             "frames": int(n)}, f"scale={w}:{h}")


def _crop_spec(video_path, crop, crf, preset, extra=None, frames=None):
    """(sidecar description, filter) of a crop proxy; see `_proxy_spec`."""
    x1, y1, x2, y2 = (int(v) for v in crop)
    w = (x2 - x1) & ~1
    h = (y2 - y1) & ~1
    n = frames if frames is not None else int(probe_video(video_path)["frame_count"])
    want = {"crop": [x1, y1, w, h], "crf": int(crf), "preset": str(preset),
            "frames": int(n)}
    if extra:
        want.update(extra)
    return want, f"crop={w}:{h}:{x1}:{y1}"


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
    want, vf = _crop_spec(video_path, crop, crf, preset, extra)
    out = _transcode(video_path, proxy_path_for(video_path, suffix),
                     vf, want, crf, preset, label, force)
    return out


def ensure_proxies_once(video_path: str, size: Tuple[int, int],
                        crop: Sequence[int], crf: int = 14,
                        preset: str = "veryfast",
                        crop_suffix: str = FAR_BAND_SUFFIX,
                        label: str = "PROXIES") -> bool:
    """Build the whole-frame AND the crop proxy from ONE decode of the source.

    Each proxy on its own is a full decode of a 4K source, and on a CPU without
    a fast decoder (a Raspberry Pi) the two of them cost about as much as the
    pose passes they feed.  One `split` filter graph halves that.

    Pure pre-warming: both files and sidecars are exactly what `ensure_proxy`
    and `ensure_crop_proxy` would have written for the same arguments (same
    spec functions), so the later calls are cache hits.  If either proxy is
    already cached, or anything goes wrong, this does nothing and returns
    False, and those calls build what is missing the ordinary way.  The crop
    is only a GUESS at the band `perceive.far` will want -- if the camera track
    later widens it, `far` rebuilds its proxy, which is what it would have done
    anyway.
    """
    if shutil.which("ffmpeg") is None:
        return False
    n = int(probe_video(video_path)["frame_count"])
    want_p, vf_p = _proxy_spec(video_path, size, crf, preset, frames=n)
    want_c, vf_c = _crop_spec(video_path, crop, crf, preset, frames=n)
    out_p = proxy_path_for(video_path, PROXY_SUFFIX)
    out_c = proxy_path_for(video_path, crop_suffix)
    if _cached(out_p, want_p, label) or _cached(out_c, want_c, label):
        return False
    tmp_p, tmp_c = out_p + ".part.mp4", out_c + ".part.mp4"
    enc = _encode_args(crf, preset)
    cmd = ["ffmpeg", "-v", "error", "-y", *hwaccel_args(), "-i", video_path,
           "-filter_complex", f"[0:v]split=2[a][b];[a]{vf_p}[p];[b]{vf_c}[c]",
           "-map", "[p]", *enc, tmp_p, "-map", "[c]", *enc, tmp_c]
    print(f"[{label}] Building both proxies from one decode ({vf_p} + {vf_c})…")
    t0 = time.perf_counter()
    try:
        _cancel.run(cmd, check=True, capture_output=True)
        ok = all(int(probe_video(t)["frame_count"]) == n for t in (tmp_p, tmp_c))
    except _cancel.Cancelled:
        for t in (tmp_p, tmp_c):
            if os.path.isfile(t):
                os.remove(t)
        raise
    except subprocess.CalledProcessError as ex:
        err = ex.stderr or b""
        if isinstance(err, (bytes, bytearray)):
            err = err.decode("utf-8", "replace")
        _warn(f"[{label}] WARN: single-decode proxies failed ({ex}); building "
              f"them one at a time. ffmpeg said: {err.strip()[-2000:] or '(nothing)'}")
        ok = False
    except Exception as ex:
        _warn(f"[{label}] WARN: single-decode proxies unreadable ({ex})")
        ok = False
    if not ok:
        for t in (tmp_p, tmp_c):
            if os.path.isfile(t):
                os.remove(t)
        return False
    for tmp, out, want in ((tmp_p, out_p, want_p), (tmp_c, out_c, want_c)):
        os.replace(tmp, out)
        with open(out + ".build.json", "w") as fh:
            json.dump(want, fh)
    print(f"[{label}] both proxies built ({time.perf_counter() - t0:.1f}s, {n} frames)")
    return True
