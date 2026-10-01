"""
dji.py
======
Find the recordings on a DJI Osmo Action card and say which files belong to
which one.

A DJI camera writes everything for a session into `DCIM/DJI_001/` (then
`DJI_002/`, ... once a folder fills):

    DJI_20260923180512_0001_D.MP4    the video
    DJI_20260923180512_0001_D.LRF    a low-bitrate preview of the same clip
    DJI_20260923181944_0002_D.JPG    a photo
    ...

The 14 digits are the camera clock at the file's start, the 4 digits a
sequence number that increments per FILE, and a long recording is split into
several files -- chapters of one encode, exactly as a GoPro does it.

Why the .LRF is not used
------------------------
It looks like a free 540p proxy and it is not one.  The far pass needs NATIVE
resolution (`anya2.perceive.far_band`), and every proxy the pipeline builds is
checked frame-exact against the source; a camera-made preview carries no such
guarantee.  It is ignored.

How chapters are grouped
------------------------
Two files are chapters of ONE recording when they are stream-compatible (same
codec, size and rate -- the same test `join._check_uniform` makes before a
`-c copy`) and EITHER share the timestamp in their name OR are consecutive in
sequence number with the second starting where the first ended, within
`CHAPTER_GAP_S`.  Both forms are accepted because what matters is the
continuity, not which clock DJI stamped a chapter with; a gap of more than a
few seconds is a new press of the record button, even on the same card and
the same settings.

`validate` then refuses footage the pipeline would silently mis-read: a
non-16:9 frame (the 960x540 analysis frame would squash it) and 10-bit video
(D-Log M / HLG -- the pose model was never shown it, and the proxies are
8-bit).
"""

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

# DJI_20260923180512_0001_D.MP4 -- the _D suffix is the camera's own; the
# suffix-less form some firmware writes is accepted too.
_DJI_RE = re.compile(r"^DJI_(\d{14})_(\d{4})(?:_D)?$", re.IGNORECASE)

VIDEO_EXTS = (".mp4", ".mov")

# A chapter boundary is a container split inside one encode, so the next file
# starts where the last one ended to within the camera clock's one-second
# resolution plus rounding.  Anything past this is a new recording.
CHAPTER_GAP_S = 3.0

ANALYSIS_ASPECT = 16 / 9
ASPECT_TOL = 0.02
MAX_FPS_WARN = 31.0          # 60 fps runs, but decodes twice the frames


@dataclass
class Chapter:
    path: str
    start: _dt.datetime
    seq: int
    codec: str
    pix_fmt: str
    width: int
    height: int
    fps: float
    duration: float
    size: int

    @property
    def end(self) -> _dt.datetime:
        return self.start + _dt.timedelta(seconds=self.duration)


@dataclass
class Recording:
    chapters: List[Chapter] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Stable across re-plugs: the first chapter's start and sequence."""
        c = self.chapters[0]
        return f"{c.start:%Y%m%d_%H%M%S}_{c.seq:04d}"

    @property
    def start(self) -> _dt.datetime:
        return self.chapters[0].start

    @property
    def paths(self) -> List[str]:
        return [c.path for c in self.chapters]

    @property
    def duration(self) -> float:
        return sum(c.duration for c in self.chapters)

    @property
    def size(self) -> int:
        return sum(c.size for c in self.chapters)

    def to_json(self) -> dict:
        return {"id": self.id, "start": self.start.isoformat(),
                "duration": round(self.duration, 3), "size": self.size,
                "chapters": [{"path": c.path, "size": c.size,
                              "duration": round(c.duration, 3),
                              "seq": c.seq} for c in self.chapters]}


def parse_name(path: str):
    """(start datetime, sequence) from a DJI file name, or None."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _DJI_RE.match(stem)
    if not m:
        return None
    try:
        start = _dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None
    return start, int(m.group(2))


def ffprobe(path: str) -> dict:
    """Stream facts without a decode: codec, pixel format, size, rate, length.

    ffprobe rather than OpenCV (`join._probe`) because OpenCV cannot report the
    pixel format, and 10-bit is the one thing `validate` most needs to catch.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,"
                          "avg_frame_rate,nb_frames,duration:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = (j.get("streams") or [{}])[0]

    def _rate(r):
        try:
            a, b = r.split("/")
            return float(a) / float(b) if float(b) else 0.0
        except Exception:
            return 0.0

    fps = _rate(s.get("avg_frame_rate", "")) or _rate(s.get("r_frame_rate", ""))
    dur = s.get("duration") or (j.get("format") or {}).get("duration") or 0
    return {"codec": s.get("codec_name", ""), "pix_fmt": s.get("pix_fmt", ""),
            "width": int(s.get("width") or 0), "height": int(s.get("height") or 0),
            "fps": round(fps, 3), "duration": float(dur),
            "frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None}


def _video_files(root: str) -> List[str]:
    out = []
    for d, _, files in os.walk(root):
        # macOS litter on a card that has been in a Mac: ._DJI_... is a
        # resource fork, not a video, and it matches the name pattern.
        for f in files:
            if f.startswith("._"):
                continue
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS and parse_name(f):
                out.append(os.path.join(d, f))
    return out


def _compatible(a: Chapter, b: Chapter) -> bool:
    return (a.codec == b.codec and a.pix_fmt == b.pix_fmt
            and a.width == b.width and a.height == b.height
            and abs(a.fps - b.fps) <= 0.1)


def _continues(prev: Chapter, cur: Chapter) -> bool:
    if not _compatible(prev, cur):
        return False
    if cur.start == prev.start:
        return True
    return (cur.seq == prev.seq + 1
            and abs((cur.start - prev.end).total_seconds()) <= CHAPTER_GAP_S)


def group(chapters: Iterable[Chapter]) -> List[Recording]:
    """Chapters -> recordings, in recording order."""
    recs: List[Recording] = []
    for c in sorted(chapters, key=lambda c: (c.start, c.seq)):
        if recs and _continues(recs[-1].chapters[-1], c):
            recs[-1].chapters.append(c)
        else:
            recs.append(Recording([c]))
    return recs


def chapter_of(path: str, probe=ffprobe) -> Optional[Chapter]:
    parsed = parse_name(path)
    if not parsed:
        return None
    p = probe(path)
    return Chapter(path=path, start=parsed[0], seq=parsed[1],
                   codec=p["codec"], pix_fmt=p["pix_fmt"],
                   width=p["width"], height=p["height"], fps=p["fps"],
                   duration=p["duration"], size=os.path.getsize(path))


def find_recordings(root: str, probe=ffprobe) -> List[Recording]:
    """Every recording under `root` (a card mount, or any folder of DJI files).

    A file ffprobe cannot read -- typically the last one on a card whose
    battery died mid-recording, which DJI leaves without a moov atom -- is
    skipped with a message rather than stopping the other recordings.
    """
    chapters = []
    for p in _video_files(root):
        try:
            c = chapter_of(p, probe)
        except (subprocess.CalledProcessError, ValueError, KeyError) as e:
            print(f"[DJI] skipping unreadable {p}: {e}")
            continue
        if c is not None and c.duration > 0:
            chapters.append(c)
    return group(chapters)


class UnsupportedFootage(ValueError):
    pass


def validate(rec: Recording) -> List[str]:
    """Raise on footage the pipeline would mis-read; return soft warnings."""
    c = rec.chapters[0]
    aspect = c.width / float(c.height or 1)
    if abs(aspect - ANALYSIS_ASPECT) > ASPECT_TOL:
        raise UnsupportedFootage(
            f"{os.path.basename(c.path)} is {c.width}x{c.height} "
            f"({aspect:.2f}:1). The analysis frame is 16:9, so a 4:3 recording "
            f"would be squashed; set the camera to a 16:9 resolution "
            f"(4K 16:9 recommended).")
    if "10" in c.pix_fmt or "12" in c.pix_fmt:
        raise UnsupportedFootage(
            f"{os.path.basename(c.path)} is {c.pix_fmt} (10-bit). Record in "
            f"8-bit Normal colour, not D-Log M or HLG.")
    warns = []
    if c.fps > MAX_FPS_WARN:
        warns.append(f"{c.fps:g} fps works but costs twice the decode of 30 fps")
    if c.codec != "hevc":
        warns.append(f"codec is {c.codec}; the Pi 5 hardware-decodes only HEVC "
                     f"(H.265), so this will decode in software")
    return warns


def main(argv=None):
    ap = argparse.ArgumentParser(description="List DJI recordings under a folder.")
    ap.add_argument("root")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    recs = find_recordings(a.root)
    if a.json:
        print(json.dumps([r.to_json() for r in recs], indent=1))
        return
    for r in recs:
        try:
            warns = validate(r)
            state = "ok" if not warns else "; ".join(warns)
        except UnsupportedFootage as e:
            state = f"UNSUPPORTED: {e}"
        print(f"{r.id}  {len(r.chapters)} file(s)  {r.duration / 60:6.1f} min  "
              f"{r.size / 1e9:6.2f} GB  {state}")


if __name__ == "__main__":
    main()
