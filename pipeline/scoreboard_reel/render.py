"""Render a spliced highlight cut with a burned-in scoreboard from tagged
points — port of src/scoreboard/scripts/render.mjs (a separate, standalone
reference project) from Node/ffmpeg-via-execFileSync to Python/ffmpeg-via-
subprocess, reskinned from the reference app's forest-green/Montserrat to
Anya's black/yellow brand.

For each tagged point: cut `start -> end` (dead time between points is never
kept), draw the scoreboard as it stood *entering* that point (via
scoring.replay_match), and concatenate every point into one video — same
per-segment-cut-then-concat shape as pipeline.utilities.create_highlights_ffmpeg,
which this doesn't reuse directly because burn-in needs a per-segment `-vf`
the cut-only helper has no hook for.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pipeline.utilities import probe_video

from .config import MatchFormat
from .scoring import display_columns, replay_match
from .tags import PointTag

# ---- Anya brand palette, ffmpeg 0xRRGGBB form (theme.py's black/yellow/sky) ----
C = {
    "bg": "0x000000",       # court black
    "accent": "0xE8FF3D",   # brand yellow
    "divider": "0xFFFFFF",  # thin divider, drawn at low alpha below
    "text": "0xFFFFFF",
    "server_dot": "0x49C5F1",  # sky accent, distinct from the yellow game box
}

_UNSAFE_DRAWTEXT_CHARS = re.compile(r"[:'\\%,\[\]=;]")


def _safe(text: str) -> str:
    """drawtext-safe: strip characters that break the filtergraph."""
    cleaned = _UNSAFE_DRAWTEXT_CHARS.sub(" ", str(text)).strip()
    return cleaned or " "


def find_font(explicit: Optional[str] = None) -> Optional[str]:
    """Prefer the bundled Montserrat; fall back to a system font so a burn-in
    still works (in a different typeface) on a machine that somehow lost the
    asset. Mirrors render.mjs's `findFont()`.

    Checks both the packaged (PyInstaller `_MEIPASS/assets/fonts`) and
    dev-run (`../../desktop/assets/fonts`) locations — same `_MEIPASS`-aware
    pattern desktop/app.py's `_logo_path()` uses, since a pure-Python
    module's `__file__` doesn't point at a real on-disk path once frozen.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(str(Path(meipass) / "assets" / "fonts" / "Montserrat-SemiBold.ttf"))
        candidates.append(str(Path(meipass) / "assets" / "fonts" / "Montserrat-Bold.ttf"))
    here = Path(__file__).resolve().parent.parent.parent / "desktop" / "assets" / "fonts"
    candidates += [
        str(here / "Montserrat-SemiBold.ttf"),
        str(here / "Montserrat-Bold.ttf"),
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _scoreboard_filters(snap: dict, names: Dict[str, str], font: str, width: int, height: int) -> str:
    """Build the drawbox/drawtext -vf filtergraph for one point's scoreboard
    bug, given its pre-point snapshot. Port of render.mjs's
    `scoreboardFilters()` — layout designed for 720p, scaled by height.
    """
    s = height / 720.0

    def px(n):
        return round(n * s)

    cols = display_columns(snap)

    x = px(36)
    h = px(96)
    y = height - px(36) - h  # bottom-left
    row_h = h / 2
    font_main = px(24)
    font_set = px(23)
    name_x = x + px(52)
    point_w = px(52)
    point_x = x + px(560) - point_w  # right accent box
    box_w = point_x + point_w - x

    y_a = y
    y_b = y + row_h

    def center_text(cy, fs):
        return round(cy + row_h / 2 - fs / 2 - px(2))

    f: List[str] = []

    # Background bar + left accent stripe + right game box + divider.
    f.append(f"drawbox=x={x}:y={y}:w={box_w}:h={h}:color={C['bg']}@0.85:t=fill")
    f.append(f"drawbox=x={x}:y={y}:w={px(5)}:h={h}:color={C['accent']}:t=fill")
    f.append(f"drawbox=x={point_x}:y={y}:w={point_w}:h={h}:color={C['accent']}@0.95:t=fill")
    f.append(f"drawbox=x={x}:y={y + row_h - 1}:w={box_w}:h=1:color={C['divider']}@0.15:t=fill")

    def draw_text(text, x_pos, y_pos, size, color, center=False):
        x_expr = f"{x_pos}-tw/2" if center else f"{x_pos}"
        f.append(
            f"drawtext=fontfile='{font}':text='{_safe(text)}':x={x_expr}:y={y_pos}:"
            f"fontsize={size}:fontcolor={color}:box=0"
        )

    # Server dot.
    if not snap["matchOver"]:
        dot = px(9)
        dx = x + px(26)
        server = snap["server"]
        dy = (y_a if server == "A" else y_b) + row_h / 2 - dot / 2
        f.append(f"drawbox=x={dx}:y={round(dy)}:w={dot}:h={dot}:color={C['server_dot']}:t=fill")

    # Names.
    draw_text(names.get("a", "Player A"), name_x, center_text(y_a, font_main), font_main, C["text"])
    draw_text(names.get("b", "Player B"), name_x, center_text(y_b, font_main), font_main, C["text"])

    # Set/game columns, right-aligned just left of the game box.
    col_w = px(38)
    start_x = point_x - px(10) - len(cols) * col_w
    for i, c in enumerate(cols):
        cx = start_x + i * col_w + col_w / 2
        color_a = C["text"] if c.get("current") else "0xD8E6DC"
        color_b = C["text"] if c.get("current") else "0xD8E6DC"
        draw_text(str(c["A"]), cx, center_text(y_a, font_set), font_set, color_a, center=True)
        draw_text(str(c["B"]), cx, center_text(y_b, font_set), font_set, color_b, center=True)

    # Current game points in the accent box.
    pcx = point_x + point_w / 2
    draw_text(snap["pointLabels"]["A"], pcx, center_text(y_a, font_main), font_main, C["text"], center=True)
    draw_text(snap["pointLabels"]["B"], pcx, center_text(y_b, font_main), font_main, C["text"], center=True)

    return ",".join(f)


def _run_ffmpeg(args: List[str]):
    result = subprocess.run(["ffmpeg", *args], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[-2000:]}")


def render_scoreboard_video(
    video_path: str,
    points: List[PointTag],
    fmt: MatchFormat,
    names: Dict[str, str],
    output_path: str,
    font_path: Optional[str] = None,
    on_progress: Optional[Callable[[int, int, str, Optional[float]], None]] = None,
) -> str:
    """Cut every tagged point tight, burn in the scoreboard standing entering
    it, and concatenate into `output_path`. Points must all have a winner
    (call sites should filter out pending/imported-but-unassigned rows
    first — this raises if any winner is missing).
    """
    video_path = os.path.abspath(video_path)
    points = [p for p in points if p.end > p.start]
    if not points:
        raise ValueError("No valid points to render (need end > start).")
    missing = [i for i, p in enumerate(points) if p.winner not in ("A", "B")]
    if missing:
        raise ValueError(f"{len(missing)} point(s) have no winner assigned (indices {missing[:5]}...).")

    font = find_font(font_path)
    if not font:
        raise RuntimeError(
            "No usable font found for the scoreboard overlay. Expected Montserrat under "
            "desktop/assets/fonts/, and no system fallback was found either."
        )

    info = probe_video(video_path)
    width, height = info["width"], info["height"]

    replay = replay_match([p.winner for p in points], fmt)
    snapshots = replay.snapshots

    n = len(points)
    work = tempfile.mkdtemp(prefix="anya_scoreboard_")
    try:
        seg_files = []
        for i, p in enumerate(points):
            if on_progress:
                on_progress(i, n, "burning scoreboard", i / n)
            snap = snapshots[i] if i < len(snapshots) else replay.final_state
            filters = _scoreboard_filters(snap, names, font, width, height)
            dur = max(0.0, p.end - p.start)
            seg = os.path.join(work, f"seg_{i:04d}.mp4")
            _run_ffmpeg([
                "-y",
                "-ss", f"{p.start:.3f}",
                "-i", video_path,
                "-t", f"{dur:.3f}",
                "-vf", filters,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "160k",
                "-movflags", "+faststart",
                seg,
            ])
            seg_files.append(seg)

        if on_progress:
            on_progress(n, n, "concatenating", None)

        list_file = os.path.join(work, "list.txt")
        with open(list_file, "w") as fh:
            for sf in seg_files:
                fh.write(f"file '{sf.replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n")

        output_path = os.path.abspath(output_path)
        _run_ffmpeg([
            "-y", "-f", "concat", "-safe", "0", "-i", list_file,
            "-c", "copy", "-movflags", "+faststart", output_path,
        ])
        return output_path
    finally:
        shutil.rmtree(work, ignore_errors=True)
