"""
Court calibration from a phone: the recorder page's /calibrate.

`pipeline.anya2.site save` wants a laptop with a screen and an OpenCV window.
At the court there is a phone and the Pi.  So the page asks the Pi for the
calibration frame of a recording, the four corners are tapped on the phone,
and the Pi writes the site profile itself -- the same files `site save` makes,
from the same frame:

    frame(video, out_png)   the 960x540 frame `site.save` stores as ref.png
                            (frame 300, or the middle of a shorter clip)
    save(cfg, video, pts)   corners -> <site>/court.json + ref.png, then
                            re-queue every job that was waiting for them

The corners are in that frame's pixels, in `utilities.init_court`'s order:
bottom-left, bottom-right, top-right, top-left (near baseline first, as the
camera sees it), singles corners.

These run in a subprocess of the recorder (`python -m anya_pi calibrate-...`),
so the recorder itself stays standard-library only and a crash in OpenCV
cannot take the Start/Stop page down with it.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

from . import jobs as J

ORDER = ["bottom-left", "bottom-right", "top-right", "top-left"]


class BadCorners(ValueError):
    pass


def check_corners(points, size=(960, 540)):
    """Four in-frame corners forming a convex quad, near baseline below far.

    Catches the taps that would otherwise give a confidently wrong court: a
    corner off the picture, the order mixed up (a self-crossing quad), or near
    and far swapped (the far baseline lower in the picture than the near one).
    """
    try:
        pts = [(float(x), float(y)) for x, y in points]
    except (TypeError, ValueError):
        raise BadCorners("corners must be four [x, y] pairs") from None
    if len(pts) != 4:
        raise BadCorners(f"need 4 corners, got {len(pts)}")
    w, h = size
    for (x, y), name in zip(pts, ORDER):
        if not (0 <= x <= w and 0 <= y <= h):
            raise BadCorners(f"the {name} corner is outside the picture")
    crosses = []
    for i in range(4):
        (ax, ay), (bx, by), (cx, cy) = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        crosses.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    if not (all(c > 0 for c in crosses) or all(c < 0 for c in crosses)):
        raise BadCorners("the corners cross over -- tap them in order: near left, "
                         "near right, far right, far left")
    bl, br, tr, tl = pts
    if min(bl[1], br[1]) <= max(tr[1], tl[1]):
        raise BadCorners("the near baseline must be lower in the picture than the "
                         "far one -- start with the corner closest to the camera")
    if bl[0] >= br[0] or tl[0] >= tr[0]:
        raise BadCorners("left and right are swapped -- start at the near LEFT corner")
    return pts


def frame(video, out_png):
    """Write the calibration frame of `video` to `out_png`; returns its info."""
    import cv2
    from pipeline.anya2 import site as S
    img = S._ref_color(str(video))
    tmp = f"{out_png}.tmp.png"
    if not cv2.imwrite(tmp, img):
        raise RuntimeError(f"could not write {out_png}")
    os.replace(tmp, out_png)
    h, w = img.shape[:2]
    return {"video": Path(video).name, "width": w, "height": h,
            "source_size": S._source_size(str(video))}


def current(cfg):
    """The saved profile's corners (or None), for the page to start from."""
    p = cfg.site_dir / "court.json"
    try:
        prof = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return {"points": prof.get("points"), "from_video": prof.get("from_video"),
            "source_size": prof.get("source_size"),
            "saved": _mtime(p)}


def _mtime(p):
    import datetime as dt
    return dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def save(cfg, video, points, log=print):
    """Make the site profile from `points` tapped on `video`'s frame.

    The profile is built in a scratch dir and swapped in whole, so a failure
    half-way leaves the old calibration in place, not a mix of the two.
    The previous profile is kept as <site>.prev.
    """
    from pipeline import workdir as WD
    from pipeline.anya2 import court as C
    from pipeline.anya2 import site as S

    video = str(video)
    pts = check_corners(points, C.ANALYSIS_SIZE)
    site = Path(cfg.site_dir)
    site.parent.mkdir(parents=True, exist_ok=True)
    cfg.work.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="calibrate-", dir=str(cfg.work)))
    staged = Path(tempfile.mkdtemp(prefix=f".{site.name}-new-", dir=str(site.parent)))
    try:
        WD.set_work_dir(str(scratch))
        try:
            # What utilities.init_court writes after the clicks; site.save then
            # reads it like any calibrated video.
            cache = C.court_cache_path(video)
            ref = S._ref_color(video)
            with open(cache, "w") as fh:
                json.dump({"points": pts, "frame_shape": list(ref.shape),
                           "analysis_size": list(C.ANALYSIS_SIZE),
                           "video": os.path.basename(video)}, fh, indent=2)
            prof = S.save(video, str(staged), prompt=False)
        finally:
            WD.clear_work_dir()
        prev = site.with_name(site.name + ".prev")
        if site.exists():
            shutil.rmtree(prev, ignore_errors=True)
            os.replace(site, prev)
        os.chmod(staged, 0o755)
        os.replace(staged, site)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(staged, ignore_errors=True)

    requeued = []
    q = J.Queue(cfg)
    for j in q.all():
        if j["status"] == J.NEEDS_CALIBRATION and j.get("chapters"):
            j.update(status=J.PENDING, error=None, stage=None)
            q.put(j)
            requeued.append(j["id"])
    log(f"[calibrate] site profile saved from {os.path.basename(video)}; "
        f"re-queued {len(requeued)} job(s)")
    return {"ok": True, "from_video": prof["from_video"],
            "far_band": prof["far_band"], "requeued": requeued}
