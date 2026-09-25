"""
python -m anya_pi <command>

    ingest PATH          copy new DJI recordings under PATH and queue them
    worker [--once]      process the queue (what anya-worker.service runs)
    status               list jobs
    retry ID             put a failed / needs-calibration job back in the queue
    reupload ID          queue a finished reel for YouTube again
    youtube-auth         one-time YouTube consent (run on a laptop with a browser)
"""

import argparse
import sys

from . import config as CFG
from . import jobs as J


def main(argv=None):
    ap = argparse.ArgumentParser(prog="anya-pi")
    ap.add_argument("--config", help=f"TOML settings (default {CFG.DEFAULT_PATH})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest")
    p.add_argument("path")
    p = sub.add_parser("worker")
    p.add_argument("--once", action="store_true", help="drain the queue, then exit")
    sub.add_parser("status")
    p = sub.add_parser("retry")
    p.add_argument("id")
    p = sub.add_parser("reupload")
    p.add_argument("id")
    p = sub.add_parser("youtube-auth")
    p.add_argument("--client-secrets", required=True)
    p.add_argument("--out", default="youtube_token.json")
    p.add_argument("--playlists", action="store_true",
                   help="also ask for playlist access (needed for playlist_id)")
    a = ap.parse_args(argv)

    if a.cmd == "youtube-auth":             # no config needed on the laptop
        from . import youtube as Y
        Y.authorize(a.client_secrets, a.out, playlists=a.playlists)
        return 0

    cfg = CFG.load(a.config)
    q = J.Queue(cfg)
    if a.cmd == "ingest":
        from . import ingest as I
        I.ingest(cfg, a.path)
        from .worker import write_status
        write_status(cfg, q)
    elif a.cmd == "worker":
        from . import worker as W
        W.run(cfg, once=a.once)
    elif a.cmd == "status":
        for j in q.all():
            yt = (j.get("youtube") or {}).get("status", "")
            print(f"{j['id']}  {j['status']:<18} {j.get('stage') or ''}  "
                  f"{'youtube: ' + yt if yt and yt != J.UP_DISABLED else ''}")
            if j.get("error") and j["status"] != J.DONE:
                print(f"    {j['error'].splitlines()[0][:200]}")
    elif a.cmd in ("retry", "reupload"):
        j = q.get(a.id)
        if not j:
            print(f"no job {a.id}", file=sys.stderr)
            return 1
        if a.cmd == "retry":
            if not j.get("chapters"):
                print("this job has no copied files (unsupported footage); "
                      "fix the camera settings and record again", file=sys.stderr)
                return 1
            j.update(status=J.PENDING, error=None, stage=None)
        else:
            if j["status"] != J.DONE:
                print("only a finished job can be uploaded", file=sys.stderr)
                return 1
            j["youtube"] = {"status": J.UP_PENDING, "attempts": 0}
        q.put(j)
        print(f"{a.id}: queued")
    return 0


if __name__ == "__main__":
    sys.exit(main())
