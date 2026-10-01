"""
The long-running processor: one job at a time, oldest first, then uploads.

Run by `anya-worker.service`; by hand:  python -m anya_pi worker [--once]

One at a time because the Pi has 4 GB and both the pose runtime and a 4K
decode want a good share of it; two concurrent runs would each take longer
than running them back to back.

Uploads run on their own thread: an 11 GB recording over the court Wi-Fi (or
no Wi-Fi at all) must not hold up processing, and processing must not hold up
the upload of the raw recording.

While the Pi camera records (pi/recorder writes <state>/recording), no job
starts and a running one is cancelled back to pending: processing pins all
four cores, and the camera's software encoder would drop frames.  The re-run
resumes from anya2's per-stage caches.
"""

import datetime as dt
import os
import shutil
import signal
import threading
import time
import traceback
from pathlib import Path

from . import jobs as J


def _apply_env(cfg):
    """The pipeline reads its runtime switches from the environment."""
    p = cfg.processing
    os.environ["ANYA_POSE_BACKEND"] = p.backend
    os.environ["ANYA_POSE_MODELS"] = str(cfg.models)
    os.environ["ANYA_FFMPEG_HWACCEL"] = p.hwaccel or ""
    os.environ["ANYA_SINGLE_DECODE_PROXIES"] = "1" if p.single_decode else "0"
    if p.threads:
        os.environ["OMP_NUM_THREADS"] = str(p.threads)
        try:
            import torch
            torch.set_num_threads(int(p.threads))
        except Exception:
            pass


def reel_name(job):
    start = dt.datetime.fromisoformat(job["recording"]["start"])
    return f"{start:%Y-%m-%d_%H%M}_tennis"


def write_status(cfg, q):
    """STATUS.txt in the reels share: what the Pi is doing, readable from any
    laptop without logging in to it."""
    lines = [f"anya Pi -- updated {J.now()}", ""]
    for j in reversed(q.all()[-30:]):
        s = j["status"]
        if s == J.RUNNING and j.get("stage"):
            s += f" ({j['stage']}"
            s += f", {j['progress']:.0%})" if j.get("progress") is not None else ")"
        for key, what in (("youtube_raw", "raw"), ("youtube", "youtube")):
            yt = (j.get(key) or {})
            if yt.get("status") not in (None, J.UP_DISABLED):
                s += f"  {what}: {yt['status']}"
                if yt.get("url"):
                    s += f" {yt['url']}"
                if yt.get("forced_private"):
                    s += " (PRIVATE -- see README)"
        lines.append(f"{j['id']}  {s}")
        if j.get("error") and j["status"] != J.DONE:
            lines.append(f"    {j['error'].splitlines()[0][:300]}")
    try:
        tmp = cfg.reels / "STATUS.txt.tmp"
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, cfg.reels / "STATUS.txt")
    except OSError:
        pass


class _Progress:
    """on_progress -> the job file (throttled) and the log."""

    def __init__(self, q, job, cfg, log):
        from pipeline.anya2.headless import ProgressPrinter
        self.q, self.job, self.cfg = q, job, cfg
        self.printer = ProgressPrinter(sink=log)
        self.last_write = 0.0
        self.last_label = None

    def __call__(self, i, n, label, frac=None):
        self.printer(i, n, label, frac)
        now = time.time()
        if now - self.last_write >= 15 or label != self.last_label:
            self.last_label = label
            self.q.update(self.job["id"], stage=f"{i}/{n} {label}", progress=frac)
            write_status(self.cfg, self.q)
            self.last_write = now


def process(cfg, job, log=print):
    from pipeline import workdir as WD
    from pipeline.anya2 import headless as H
    from pipeline.anya2 import site as S

    q = J.Queue(cfg)
    job = q.update(job["id"], status=J.RUNNING, started=J.now(), error=None,
                   stage="starting")
    write_status(cfg, q)
    _apply_env(cfg)
    work = cfg.work / job["id"]
    WD.set_work_dir(str(work))
    name = reel_name(job)
    out = cfg.reels / f"{name}.mp4"
    t0 = time.time()
    try:
        segs, reel = H.build(job["chapters"], str(out),
                             H.make_config(cfg.processing.device,
                                           cfg.processing.scale_height or None,
                                           copy_video=cfg.processing.copy_video),
                             site=str(cfg.site_dir) if cfg.site_dir.is_dir() else None,
                             on_progress=_Progress(q, job, cfg, log))
    except (H.NotCalibrated, S.NeedsCalibration) as e:
        job = q.update(job["id"], status=J.NEEDS_CALIBRATION, error=str(e),
                       finished=J.now())
        log(f"[worker] {job['id']}: needs calibration -- {e}")
        return job
    except Exception as e:                      # noqa: BLE001 -- recorded on the job
        from pipeline import cancel
        if isinstance(e, cancel.Cancelled):
            # Shutdown or a recording, not failure: `run` decides what the
            # job goes back to; either way it resumes from the caches.
            log(f"[worker] {job['id']}: stopped; will resume")
            raise
        job = q.update(job["id"], status=J.FAILED, finished=J.now(),
                       error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        log(f"[worker] {job['id']}: FAILED -- {e}")
        return job
    finally:
        WD.clear_work_dir()

    kept = sum(s["stop"] - s["start"] for s in segs)
    J.write_json(cfg.reels / f"{name}.segments.json", segs)
    job = q.update(job["id"], status=J.DONE, finished=J.now(), stage=None,
                   progress=None, reel=reel, segments=len(segs),
                   kept_s=round(kept, 1), elapsed_s=round(time.time() - t0),
                   youtube={"status": J.UP_PENDING if cfg.youtube.enabled and reel
                            else J.UP_DISABLED, "attempts": 0})
    # The work dir holds the join, both proxies and every cached stage: tens
    # of GB for a long match, and useless once the reel exists.
    shutil.rmtree(work, ignore_errors=True)
    dur = job["recording"].get("duration") or 0
    log(f"[worker] {job['id']}: done -- {len(segs)} points, {kept / 60:.1f} of "
        f"{dur / 60:.1f} min kept, {job['elapsed_s'] / 60:.0f} min to process "
        f"({job['elapsed_s'] / max(dur, 1):.1f}x realtime) -> {reel}")
    return job


def titles(job, cfg):
    """("6:30 PM · Oct 1, 2026 · Wimbledon Session", "... Highlights")."""
    start = dt.datetime.fromisoformat(job["recording"]["start"])
    hour = start.hour % 12 or 12
    base = (f"{hour}:{start:%M} {start:%p} · {start:%b} {start.day}, {start.year}"
            f" · {cfg.youtube.session_name}")
    return base, f"{base} Highlights"


def _upload_one(cfg, q, job, key, path, title, desc, uploader, log):
    """One attempt at one upload.  Failures leave it pending for the next
    pass until `max_attempts` (no network at the court is the normal case)."""
    attempts = int((job.get(key) or {}).get("attempts", 0)) + 1
    q.modify(job["id"], lambda j: j[key].update(status=J.UP_UPLOADING,
                                                attempts=attempts))
    try:
        log(f"[youtube] uploading {path} as {title!r}")
        res = uploader(path, title, desc, cfg.youtube_token,
                       privacy=cfg.youtube.privacy,
                       playlist_id=cfg.youtube.playlist_id, log=log)
    except Exception as e:                      # noqa: BLE001 -- retried next pass
        give_up = attempts >= cfg.youtube.max_attempts
        q.modify(job["id"], lambda j: j[key].update(
            status=J.UP_FAILED if give_up else J.UP_PENDING,
            error=f"{type(e).__name__}: {e}"))
        log(f"[youtube] {job['id']} {key}: attempt {attempts} failed -- {e}")
        return None
    q.modify(job["id"], lambda j: j[key].update(status=J.UP_DONE, error=None,
                                                uploaded=J.now(), **res))
    log(f"[youtube] {job['id']} {key}: {res['url']} ({res['privacy']})")
    return res


def upload_pending(cfg, log=print, uploader=None):
    """Upload every raw recording waiting for YouTube, oldest first, then
    every finished reel."""
    if not cfg.youtube.enabled:
        return
    if not cfg.youtube_token.is_file():
        log(f"[youtube] no token at {cfg.youtube_token}; see pi/README.md")
        return
    from . import youtube as Y
    uploader = uploader or Y.upload
    q = J.Queue(cfg)

    for job in q.all():
        yt = job.get("youtube_raw") or {}
        if yt.get("status") != J.UP_PENDING:
            continue
        src = (job.get("chapters") or [None])[0]
        if not src or not os.path.isfile(src):
            q.modify(job["id"], lambda j: j["youtube_raw"].update(
                status=J.UP_FAILED, error="recording file is missing"))
            continue
        dur = (job["recording"].get("duration") or 0) / 60
        _upload_one(cfg, q, job, "youtube_raw", src, titles(job, cfg)[0],
                    f"Full recording, {dur:.0f} min.", uploader, log)

    for job in q.all():
        yt = job.get("youtube") or {}
        if job["status"] != J.DONE or yt.get("status") != J.UP_PENDING:
            continue
        if not job.get("reel") or not os.path.isfile(job["reel"]):
            q.modify(job["id"], lambda j: j["youtube"].update(
                status=J.UP_FAILED, error="reel file is missing"))
            continue
        desc = (f"{job.get('segments', '?')} points, "
                f"{(job.get('kept_s') or 0) / 60:.0f} min of play from a "
                f"{(job['recording'].get('duration') or 0) / 60:.0f} min recording. "
                f"Dead time removed by anya.")
        res = _upload_one(cfg, q, job, "youtube", job["reel"], titles(job, cfg)[1],
                          desc, uploader, log)
        if res:
            Path(job["reel"]).with_suffix(".youtube.txt").write_text(res["url"] + "\n")


def _upload_loop(cfg, stopping, log):
    while not stopping["flag"]:
        try:
            upload_pending(cfg, log)
        except Exception as e:          # noqa: BLE001 -- never kill the thread
            log(f"[youtube] {e}")
        for _ in range(cfg.poll_s):
            if stopping["flag"]:
                return
            time.sleep(1)


def _pause_watch(cfg, stopping, running):
    """Cancel the running job while the camera records (see module doc)."""
    from pipeline import cancel
    while not stopping["flag"]:
        if running["id"] and cfg.recording_flag.exists():
            cancel.request()
        time.sleep(2)


def cleanup_inbox(cfg, log=print):
    """Drop copied originals of finished jobs after `keep_inbox_days`."""
    keep = dt.timedelta(days=cfg.processing.keep_inbox_days)
    for job in J.Queue(cfg).all():
        if job["status"] != J.DONE or not job.get("finished"):
            continue
        if dt.datetime.now() - dt.datetime.fromisoformat(job["finished"]) < keep:
            continue
        d = cfg.inbox / job["id"]
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            log(f"[worker] removed originals of {job['id']} (kept "
                f"{cfg.processing.keep_inbox_days} days)")
        # A Pi-camera recording is the only copy until YouTube has it.
        if (job["recording"].get("source") == "picam"
                and (job.get("youtube_raw") or {}).get("status") == J.UP_DONE):
            for c in job.get("chapters") or []:
                if os.path.isfile(c):
                    os.remove(c)
                    log(f"[worker] removed {c} (on YouTube, kept "
                        f"{cfg.processing.keep_inbox_days} days)")


def run(cfg, once=False, log=print):
    from pipeline import cancel
    cfg.ensure_dirs()
    q = J.Queue(cfg)
    n = q.recover()
    if n:
        log(f"[worker] resuming {n} interrupted job(s)")

    stopping = {"flag": False}
    running = {"id": None}

    def _stop(signum, frame):
        stopping["flag"] = True
        cancel.request()                # the pipeline checks in and raises

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    if not once:
        threading.Thread(target=_upload_loop, args=(cfg, stopping, log),
                         daemon=True).start()
        threading.Thread(target=_pause_watch, args=(cfg, stopping, running),
                         daemon=True).start()

    paused_logged = False
    while not stopping["flag"]:
        write_status(cfg, q)
        recording = cfg.recording_flag.exists()
        job = None if recording else q.next_pending()
        if recording and q.next_pending() and not paused_logged:
            log("[worker] the camera is recording; processing waits")
        paused_logged = recording
        if job:
            cancel.clear()
            running["id"] = job["id"]
            try:
                process(cfg, job, log)
            except cancel.Cancelled:
                if stopping["flag"]:
                    break               # left RUNNING; `recover` resumes it
                q.update(job["id"], status=J.PENDING,
                         stage="paused while recording")
                log(f"[worker] {job['id']}: paused while the camera records")
            finally:
                running["id"] = None
            write_status(cfg, q)
            continue                    # straight on to the next job
        if once:
            try:
                upload_pending(cfg, log)
            except Exception as e:      # noqa: BLE001 -- never kill the loop
                log(f"[youtube] {e}")
        cleanup_inbox(cfg, log)
        write_status(cfg, q)
        if once:
            break
        for _ in range(cfg.poll_s if not recording else 2):
            if stopping["flag"]:
                break
            time.sleep(1)
    log("[worker] stopped")
