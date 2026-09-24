"""
The job queue and the ingest ledger: small JSON files under <root>/state.

A directory of files rather than a database because there is one writer at a
time per file (ingest creates jobs, the worker advances them), the queue is a
handful of entries a week, and a person debugging a stuck Pi can `cat` it.
Every write is write-temp-then-rename, so a power cut mid-write leaves the old
file, never half of a new one.
"""

import datetime as dt
import json
import os
from pathlib import Path

# Job lifecycle.  Terminal states wait for a person (`anya-pi retry <id>`).
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
NEEDS_CALIBRATION = "needs_calibration"
UNSUPPORTED = "unsupported"

# YouTube sub-state, on a DONE job.
UP_DISABLED = "disabled"
UP_PENDING = "pending"
UP_DONE = "done"
UP_FAILED = "failed"


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


class Queue:
    def __init__(self, cfg):
        self.cfg = cfg

    def path(self, job_id):
        return self.cfg.jobs / f"{job_id}.json"

    def get(self, job_id):
        return read_json(self.path(job_id))

    def put(self, job):
        job["updated"] = now()
        write_json(self.path(job["id"]), job)
        return job

    def all(self):
        out = []
        if self.cfg.jobs.is_dir():
            for p in sorted(self.cfg.jobs.glob("*.json")):
                j = read_json(p)
                if j:
                    out.append(j)
        return sorted(out, key=lambda j: j.get("created", ""))

    def create(self, job_id, recording, chapters, status=PENDING, error=None):
        job = {"id": job_id, "status": status, "created": now(),
               "recording": recording, "chapters": [str(c) for c in chapters],
               "stage": None, "progress": None, "error": error,
               "reel": None, "youtube": {"status": UP_DISABLED}}
        return self.put(job)

    def next_pending(self):
        for j in self.all():
            if j["status"] == PENDING:
                return j
        return None

    def recover(self):
        """A job left RUNNING was interrupted (reboot, power cut): run it again.

        anya2 caches every stage in the work dir, so a re-run resumes from the
        last finished stage rather than from the start."""
        n = 0
        for j in self.all():
            if j["status"] == RUNNING:
                j["status"] = PENDING
                j["stage"] = "resuming after interruption"
                self.put(j)
                n += 1
        return n


class Ledger:
    """Recordings already copied off a card, so a re-plug copies nothing twice.

    Keyed by recording id (first chapter's timestamp + sequence) AND total
    size: a card that was formatted and re-used can repeat a sequence number,
    but not with the same timestamp and byte count.
    """

    def __init__(self, cfg):
        self.path = cfg.state / "ledger.json"

    def _load(self):
        return read_json(self.path, {})

    def seen(self, rec_id, size):
        e = self._load().get(rec_id)
        return bool(e) and int(e.get("size", -1)) == int(size)

    def add(self, rec_id, size, **extra):
        d = self._load()
        d[rec_id] = {"size": int(size), "at": now(), **extra}
        write_json(self.path, d)
