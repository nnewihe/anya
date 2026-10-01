"""
The long-running processor: one job at a time, oldest first, then uploads.

Run by `anya-worker.service`; by hand:  python -m anya_pi worker [--once]

One at a time because the Pi has 4 GB and both the pose runtime and a 4K
decode want a good share of it; two concurrent runs would each take longer
than running them back to back.
"""

import datetime as dt
import os
import shutil
import signal
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
        yt = (j.get("youtube") or {})
        if yt.get("status") not in (None, J.UP_DISABLED):
            s += f"  youtube: {yt['status']}"
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
            self.job["stage"] = f"{i}/{n} {label}"
            self.job["progress"] = frac
            self.q.put(self.job)
            write_status(self.cfg, self.q)
            self.last_write = now


def process(cfg, job, log=print):
    from pipeline import workdir as WD
    from pipeline.anya2 import headless as H
    from pipeline.anya2 import site as S

    q = J.Queue(cfg)
    job.update(status=J.RUNNING, started=J.now(), error=None, stage="starting")
    q.put(job)
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
        job.update(status=J.NEEDS_CALIBRATION, error=str(e), finished=J.now())
        q.put(job)
        log(f"[worker] {job['id']}: needs calibration -- {e}")
        return job
    except Exception as e:                      # noqa: BLE001 -- recorded on the job
        from pipeline import cancel
        if isinstance(e, cancel.Cancelled):
            # Shutdown, not failure: leave it RUNNING so `recover` resumes it.
            log(f"[worker] {job['id']}: stopped for shutdown; will resume")
            raise
        job.update(status=J.FAILED, finished=J.now(),
                   error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        q.put(job)
        log(f"[worker] {job['id']}: FAILED -- {e}")
        return job
    finally:
        WD.clear_work_dir()

    kept = sum(s["stop"] - s["start"] for s in segs)
    J.write_json(cfg.reels / f"{name}.segments.json", segs)
    job.update(status=J.DONE, finished=J.now(), stage=None, progress=None,
               reel=reel, segments=len(segs), kept_s=round(kept, 1),
               elapsed_s=round(time.time() - t0),
               youtube={"status": J.UP_PENDING if cfg.youtube.enabled and reel
                        else J.UP_DISABLED, "attempts": 0})
    q.put(job)
    # The work dir holds the join, both proxies and every cached stage: tens
    # of GB for a long match, and useless once the reel exists.
    shutil.rmtree(work, ignore_errors=True)
    dur = job["recording"].get("duration") or 0
    log(f"[worker] {job['id']}: done -- {len(segs)} points, {kept / 60:.1f} of "
        f"{dur / 60:.1f} min kept, {job['elapsed_s'] / 60:.0f} min to process "
        f"({job['elapsed_s'] / max(dur, 1):.1f}x realtime) -> {reel}")
    return job


def upload_pending(cfg, log=print, uploader=None):
    """Upload every DONE reel waiting for YouTube; each failure just waits for
    the next loop (no network at the court is the normal case, not an error)."""
    if not cfg.youtube.enabled:
        return
    from . import youtube as Y
    uploader = uploader or Y.upload
    q = J.Queue(cfg)
    for job in q.all():
        yt = job.setdefault("youtube", {})
        if job["status"] != J.DONE or yt.get("status") != J.UP_PENDING:
            continue
        if not job.get("reel") or not os.path.isfile(job["reel"]):
            yt.update(status=J.UP_FAILED, error="reel file is missing")
            q.put(job)
            continue
        if not cfg.youtube_token.is_file():
            log(f"[youtube] no token at {cfg.youtube_token}; see pi/README.md")
            return
        start = dt.datetime.fromisoformat(job["recording"]["start"])
        title = f"{cfg.youtube.title_prefix} {start:%Y-%m-%d %H:%M}"
        desc = (f"{job.get('segments', '?')} points, "
                f"{(job.get('kept_s') or 0) / 60:.0f} min of play from a "
                f"{(job['recording'].get('duration') or 0) / 60:.0f} min recording. "
                f"Dead time removed by anya.")
        yt["attempts"] = int(yt.get("attempts", 0)) + 1
        try:
            log(f"[youtube] uploading {job['reel']}")
            res = uploader(job["reel"], title, desc, cfg.youtube_token,
                           privacy=cfg.youtube.privacy,
                           playlist_id=cfg.youtube.playlist_id, log=log)
        except Exception as e:                  # noqa: BLE001 -- retried next loop
            yt["error"] = f"{type(e).__name__}: {e}"
            if yt["attempts"] >= cfg.youtube.max_attempts:
                yt["status"] = J.UP_FAILED
            q.put(job)
            log(f"[youtube] {job['id']}: attempt {yt['attempts']} failed -- {e}")
            continue
        yt.update(status=J.UP_DONE, error=None, uploaded=J.now(), **res)
        q.put(job)
        Path(job["reel"]).with_suffix(".youtube.txt").write_text(res["url"] + "\n")
        log(f"[youtube] {job['id']}: {res['url']} ({res['privacy']})")


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


def run(cfg, once=False, log=print):
    from pipeline import cancel
    cfg.ensure_dirs()
    q = J.Queue(cfg)
    n = q.recover()
    if n:
        log(f"[worker] resuming {n} interrupted job(s)")

    stopping = {"flag": False}

    def _stop(signum, frame):
        stopping["flag"] = True
        cancel.request()                # the pipeline checks in and raises

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while not stopping["flag"]:
        write_status(cfg, q)
        job = q.next_pending()
        if job:
            cancel.clear()
            try:
                process(cfg, job, log)
            except cancel.Cancelled:
                break
            write_status(cfg, q)
            continue                    # straight on to the next job
        try:
            upload_pending(cfg, log)
        except Exception as e:          # noqa: BLE001 -- never kill the loop
            log(f"[youtube] {e}")
        cleanup_inbox(cfg, log)
        write_status(cfg, q)
        if once:
            break
        for _ in range(cfg.poll_s):
            if stopping["flag"]:
                break
            time.sleep(1)
    log("[worker] stopped")
