"""
Copy new recordings off a mounted DJI card (or any folder of DJI files) into
the inbox, and queue a job for each.

Run by `anya-ingest@.service` when a USB disk appears (see pi/bin/anya-ingest-device),
or by hand:  python -m anya_pi ingest /path/to/mount

NEVER deletes or writes anything on the card.  The mount is read-only anyway;
this is the second lock on that door.
"""

import os
import shutil
import subprocess
from pathlib import Path

from pipeline import dji

from . import jobs as J

# Free space to leave for one recording: the copy itself, a same-size join
# when it has several chapters, the proxies, and the reel.
SPACE_FACTOR = 2.3


def _copy(src, dst):
    """rsync when available: it resumes a partial file after an interrupted
    copy instead of starting a 10 GB chapter again."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        subprocess.run(["rsync", "--partial", "--inplace", "--times", "-q",
                        str(src), str(dst)], check=True)
    else:
        shutil.copy2(src, dst)


def _verify(src, dst):
    if os.path.getsize(src) != os.path.getsize(dst):
        raise IOError(f"size mismatch after copying {src}")
    a, b = dji.ffprobe(str(src)), dji.ffprobe(str(dst))
    if abs(a["duration"] - b["duration"]) > 0.5:
        raise IOError(f"{dst} does not read back as the same video")


def ingest(cfg, root, log=print):
    """Returns the job ids created."""
    cfg.ensure_dirs()
    q, ledger = J.Queue(cfg), J.Ledger(cfg)
    recs = dji.find_recordings(str(root))
    log(f"[ingest] {len(recs)} recording(s) under {root}")
    created = []
    for rec in recs:
        if ledger.seen(rec.id, rec.size):
            continue
        try:
            warns = dji.validate(rec)
        except dji.UnsupportedFootage as e:
            # Recorded, not copied: the job file is how the user finds out
            # (STATUS.txt in the share), and the ledger keeps a re-plug quiet.
            log(f"[ingest] {rec.id}: unsupported -- {e}")
            q.create(rec.id, rec.to_json(), [], status=J.UNSUPPORTED, error=str(e))
            ledger.add(rec.id, rec.size, status=J.UNSUPPORTED)
            created.append(rec.id)
            continue
        for w in warns:
            log(f"[ingest] {rec.id}: note: {w}")

        need = rec.size * SPACE_FACTOR
        free = shutil.disk_usage(cfg.root).free
        if free < need:
            log(f"[ingest] {rec.id}: NOT copied -- needs ~{need / 1e9:.0f} GB free, "
                f"{free / 1e9:.0f} GB available. Delete old reels/inbox and re-plug.")
            continue

        dest = cfg.inbox / rec.id
        copied = []
        log(f"[ingest] {rec.id}: copying {len(rec.chapters)} file(s), "
            f"{rec.size / 1e9:.1f} GB, {rec.duration / 60:.0f} min")
        try:
            for c in rec.chapters:
                d = dest / os.path.basename(c.path)
                _copy(c.path, d)
                _verify(c.path, d)
                copied.append(d)
        except (OSError, subprocess.CalledProcessError) as e:
            # No job and no ledger entry: the next plug-in retries the copy
            # (rsync resumes the partial file).
            log(f"[ingest] {rec.id}: copy failed ({e}); will retry next time")
            continue
        q.create(rec.id, rec.to_json(), copied)
        ledger.add(rec.id, rec.size, status="copied")
        created.append(rec.id)
        log(f"[ingest] {rec.id}: queued")
    return created
