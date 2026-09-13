"""
join.py
=======
Turn the several files a GoPro wrote for one recording into the single video
path the rest of this codebase is built around.

A GoPro splits a continuous recording into ~4 GB chapters -- GX010123.MP4,
GX020123.MP4, ... -- and every one of them is a slice of ONE encode: same
codec, same resolution, same frame rate, same GOP settings.  Nothing about the
footage is discontinuous; only the container is.

Why join rather than teach the pipeline to read a list
------------------------------------------------------
Because "a list of files" is not a small change to this pipeline, it is a
different pipeline.  The single-source assumption is load-bearing in four
separate places:

  * Every artifact is `<stem>_something` under one `workdir.artifact_dir` --
    the court cache, the camera track, both proxies and their sidecars, the
    two pose npz, the tracks npz, walk and endsig, three event JSONs, the reel
    JSON.  A list input needs either a synthetic stem or per-chapter artifacts
    plus something that stitches them.
  * The time axis is a source frame index.  `anya2.contract` fixes `t` as
    seconds on THE source timeline, `perceive` decimates over
    `range(0, total, stride)`, `tracks` maps back with `src_f = f * stride`,
    and `camera.CameraTrack` is indexed by source frame.  Multi-input means a
    global-frame <-> (file, local frame) map inside every one of those.
  * `proxy` verifies frame-exactness against one source frame count and throws
    the proxy away if it does not match.  Joining first keeps that invariant
    true, because the join IS the source.
  * `anya2.run.cut` cuts `-ss/-t` from a single `-i`, and a `Segment` carries
    no file identity at all.

Whereas joining chapters is nearly free: `-c copy` remuxes without touching a
single compressed frame, so it is lossless and runs at disk speed.

What this module guarantees
---------------------------
The returned path is FRAME-EXACT against the inputs: its frame count equals
the sum of theirs, verified before the join is accepted.  Everything
downstream converts seconds through that count, so a drift here would not
fail, it would quietly move every detected point.  A mismatch raises.

One file in, that same path out, unchanged -- so a caller can hand this
whatever the user picked and stop thinking about it.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Callable, List, Optional, Sequence

import cv2

try:                                        # package import (python -m pipeline.x)
    from .videoio import open_video
    from .utilities import write_concat_list
    from . import cancel as _cancel
    from . import subproc as _subproc
    from . import workdir as _workdir
except ImportError:                         # script import (python pipeline/x.py)
    from videoio import open_video
    from utilities import write_concat_list
    import cancel as _cancel
    import subproc as _subproc
    import workdir as _workdir

_log = logging.getLogger("anya_tennis.join")

JOIN_SUFFIX = "_joined"

# GX010123.MP4 -- two-digit CHAPTER, then a four-digit number shared by every
# chapter of the same recording.  Lexical order happens to be right for this
# name shape, but only by accident of the chapter field coming first; parsing
# it means a rename or a differently-zero-padded sibling cannot reorder a
# match silently.
_GOPRO_RE = re.compile(r"^(G[XHLP])(\d{2})(\d{4})$", re.IGNORECASE)


def _probe(path: str) -> dict:
    """fps / frames / size / fourcc in one open.

    Not `utilities.probe_video`: that one prints a five-line banner per call
    (noise when this runs over eight chapters) and does not report the codec,
    which is the field that decides whether `-c copy` is safe.
    """
    cap = open_video(path, "JOIN")
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    if fps <= 0 or fps > 300:               # same clamp as probe_video
        fps = 30.0
    fourcc = "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4)) if cc else ""
    # Rounded, and not merely for tidiness: OpenCV derives fps from the
    # container's duration and frame count, so two chapters of ONE recording
    # come back 29.970054 and 29.969978.  Full precision here would make the
    # sidecar's cache key jitter and the uniformity check below reject exactly
    # the input this module exists for.
    return {"path": path, "fps": round(float(fps), 3), "frame_count": n,
            "width": w, "height": h, "fourcc": fourcc}


def order_inputs(paths: Sequence[str]) -> List[str]:
    """The order the chapters were recorded in.

    GoPro naming when every file matches it, basename order otherwise.  A
    caller-supplied order is NOT honoured: a file dialog returns its selection
    in whatever order the widget felt like, and a reversed match is a
    forty-minute mistake that looks like a working run.
    """
    base = [os.path.basename(p) for p in paths]
    stems = [os.path.splitext(b)[0] for b in base]
    m = [_GOPRO_RE.match(s) for s in stems]
    if all(m) and len({x.group(3) for x in m}) == 1:
        return [p for _, p in sorted(zip((x.group(2) for x in m), paths))]
    return [p for _, p in sorted(zip(base, paths))]


def join_path_for(paths: Sequence[str]) -> str:
    """Where the joined file goes: the artifact dir of the FIRST chapter.

    Deterministic in the input set, so a second run against a `tmp_anya` that
    is still there reuses the join -- and, because every artifact is keyed by
    this file's stem, the court calibration and pose passes with it.
    """
    first = paths[0]
    stem = os.path.splitext(os.path.basename(first))[0]
    return os.path.join(_workdir.artifact_dir(first),
                        f"{stem}{JOIN_SUFFIX}{len(paths)}.mp4")


def _describe(paths: Sequence[str]) -> List[dict]:
    """The sidecar's view of the inputs: identity plus enough to spot a change."""
    out = []
    for p in paths:
        st = os.stat(p)
        info = _probe(p)
        info.update({"path": os.path.abspath(p), "size": st.st_size,
                     "mtime": int(st.st_mtime)})
        out.append(info)
    return out


def _check_uniform(infos: Sequence[dict]) -> None:
    """Every chapter must share a stream shape, or `-c copy` lies.

    Concatenating streams that disagree does not fail: ffmpeg writes a file
    whose header describes the first input and whose later packets do not
    match it.  It plays back wrong rather than not at all, which is exactly
    the kind of thing that would be found four stages downstream as "the
    detector stopped working half way through".
    """
    ref = infos[0]
    # fps compares with a tolerance for the reason given in `_probe`: the
    # number is derived, not read, so identically-encoded chapters differ in
    # the noise.  A tenth of a frame is far below any real rate gap (29.97 vs
    # 30, 30 vs 59.94) and far above the derivation's error.
    same = {"fps": lambda a, b: abs(a - b) <= 0.1}
    for k, what in (("width", "width"), ("height", "height"),
                    ("fps", "frame rate"), ("fourcc", "codec")):
        eq = same.get(k, lambda a, b: a == b)
        bad = [i for i in infos[1:] if not eq(i[k], ref[k])]
        if bad:
            raise ValueError(
                f"cannot join: {os.path.basename(bad[0]['path'])} has "
                f"{what} {bad[0][k]} but {os.path.basename(ref['path'])} has "
                f"{ref[k]}. These are not chapters of one recording; join "
                f"them by hand with a re-encode, or run them separately.")


def _check_space(paths: Sequence[str], out: str) -> None:
    need = int(sum(os.path.getsize(p) for p in paths) * 1.02)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out))).free
    if free < need:
        raise RuntimeError(
            f"not enough free space to join: need about "
            f"{need / 1e9:.1f} GB on {os.path.dirname(os.path.abspath(out))}, "
            f"{free / 1e9:.1f} GB available.")


def _run_ffmpeg(cmd: List[str], tmp: str, expect_bytes: int,
                on_progress: Optional[Callable[[float], None]]) -> None:
    """`cancel.run`, plus a progress tick off the output file's size.

    Not `cancel.run` itself: this is the one ffmpeg call in the codebase long
    enough that a tester needs to see it moving, and a remux's output grows
    at a rate close enough to linear in wall time for the byte count to be an
    honest progress bar.  The cancel handling is deliberately identical.
    """
    proc = _subproc.popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        try:
            out, err = proc.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if _cancel.requested():
                proc.terminate()
                try:
                    proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                raise _cancel.Cancelled("cancelled by the user")
            if on_progress and expect_bytes > 0:
                try:
                    on_progress(min(0.99, os.path.getsize(tmp) / expect_bytes))
                except OSError:
                    pass                    # not created yet
    if proc.returncode != 0:
        msg = err or b""
        if isinstance(msg, (bytes, bytearray)):
            msg = msg.decode("utf-8", "replace")
        raise RuntimeError(
            f"ffmpeg could not join the files (exit {proc.returncode}). "
            f"It said: {msg.strip()[-2000:] or '(nothing)'}")


def resolve_input(paths, on_progress: Optional[Callable[[float], None]] = None,
                  force: bool = False, out: Optional[str] = None) -> str:
    """One video path from one-or-more inputs; the path every stage then uses.

    A single input is returned verbatim -- no join file, no sidecar, nothing
    written -- so every existing caller behaves exactly as it did.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [os.fspath(paths)]
    paths = [os.fspath(p) for p in paths]
    if not paths:
        raise ValueError("no input video given")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"input video not found: {missing[0]}")
    if len(paths) == 1:
        return paths[0]

    # Absolute from here down: the concat list ffmpeg reads resolves relative
    # entries against the CWD, and the sidecar records absolute paths, so a
    # relative input would compare unequal to its own cache entry.
    paths = [os.path.abspath(p) for p in order_inputs(paths)]
    out = out or join_path_for(paths)
    meta_path = out + ".build.json"

    infos = _describe(paths)
    _check_uniform(infos)
    want = {"sources": infos}
    total_frames = sum(i["frame_count"] for i in infos)

    if not force and os.path.isfile(out) and os.path.isfile(meta_path):
        try:
            if json.load(open(meta_path)) == want:
                if int(_probe(out)["frame_count"]) == total_frames:
                    print(f"[JOIN] Using cached join: {out}")
                    if on_progress:
                        on_progress(1.0)
                    return out
        except Exception:
            pass                            # unreadable sidecar: rebuild
        print("[JOIN] The cached join was built from a different set of "
              "files — rebuilding.")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is needed to join multiple video files and was not found "
            "on this machine.")

    # makedirs first: the space check stats the destination directory, and
    # with a work-dir override in play it may not exist yet.
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    _check_space(paths, out)

    print(f"[JOIN] Joining {len(paths)} files "
          f"({total_frames} frames, {total_frames / infos[0]['fps'] / 60:.1f} min):")
    for i, p in enumerate(paths, 1):
        print(f"  {i}. {os.path.basename(p)}")
    _log.info("joining %d files into %s: %s", len(paths), out,
              ", ".join(os.path.basename(p) for p in paths))

    tmp = out + ".part.mp4"
    lst = write_concat_list(paths, out + ".concat.txt")
    # -map/-ignore_unknown rather than utilities.concat_cmd's bare `-c copy`:
    # a GoPro MP4 carries a `gpmd` telemetry track and a `tmcd` timecode track
    # alongside the video and audio, and remuxing those into a plain mp4 is
    # either an error or a stream nothing downstream can read.  Naming the two
    # streams we want makes the rest somebody else's problem.
    # +faststart because every later pass seeks in this file.
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", lst,
           "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
           "-ignore_unknown", "-movflags", "+faststart", tmp]

    t0 = time.perf_counter()
    try:
        _run_ffmpeg(cmd, tmp, sum(i["size"] for i in infos), on_progress)
    except BaseException:
        for f in (tmp, lst):
            if os.path.isfile(f):
                os.remove(f)
        raise

    got = int(_probe(tmp)["frame_count"])
    if got != total_frames:
        os.remove(tmp)
        os.remove(lst)
        raise RuntimeError(
            f"the joined file has {got} frames but the {len(paths)} inputs "
            f"have {total_frames} between them. Every stage after this one "
            f"converts seconds through that count, so a join that is not "
            f"frame-exact would move every point it detects rather than "
            f"fail. Refusing to use it.")

    os.replace(tmp, out)
    os.remove(lst)
    with open(meta_path, "w") as fh:
        json.dump(want, fh, indent=1)
    if on_progress:
        on_progress(1.0)
    print(f"[JOIN] joined → {out}  ({time.perf_counter() - t0:.1f}s, "
          f"{got} frames)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Join GoPro chapter files into the single video the anya2 "
                    "pipeline takes. Prints the joined path.")
    ap.add_argument("video", nargs="+", help="the chapter files, in any order")
    ap.add_argument("-o", "--out", help="where to write the join "
                                        "(default: beside the first input)")
    ap.add_argument("-f", "--force", action="store_true",
                    help="rebuild even if a matching join is cached")
    a = ap.parse_args(argv)
    print(resolve_input(a.video, force=a.force, out=a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
