"""
python -m anya_pi <command>

    ingest PATH          copy new DJI recordings under PATH and queue them
    worker [--once]      process the queue (what anya-worker.service runs)
    status               list jobs
    retry ID             put a failed / needs-calibration job back in the queue
    reupload ID [--raw]  queue a finished reel (or, --raw, the recording) for YouTube again
    youtube-auth         one-time YouTube consent (run on a laptop with a browser)
    youtube-playlist T   create the club playlist every upload goes into
    calibrate-frame / calibrate-save   used by the recorder's /calibrate page
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
    p.add_argument("--raw", action="store_true",
                   help="the Pi-camera recording itself, not the reel")
    p = sub.add_parser("youtube-auth")
    p.add_argument("--client-secrets", required=True)
    p.add_argument("--out", default="youtube_token.json")
    p.add_argument("--playlists", action="store_true",
                   help="also ask for playlist access (needed for playlist_id)")
    p = sub.add_parser("youtube-playlist",
                       help="create the playlist every upload is added to")
    p.add_argument("title")
    p.add_argument("--description", default="")
    p.add_argument("--privacy", default="unlisted",
                   choices=["unlisted", "public", "private"])
    p.add_argument("--token", help="default: the config's [youtube] token")
    p = sub.add_parser("calibrate-frame", help="(the recorder page) write a "
                       "recording's calibration frame as PNG")
    p.add_argument("video")
    p.add_argument("out_png")
    p = sub.add_parser("calibrate-save", help="(the recorder page) save the site "
                       "profile from four corners")
    p.add_argument("video")
    p.add_argument("points", help='JSON: [[x,y] x4], bottom-left first')
    a = ap.parse_args(argv)

    if a.cmd == "youtube-auth":             # no config needed on the laptop
        from . import youtube as Y
        Y.authorize(a.client_secrets, a.out, playlists=a.playlists)
        return 0

    cfg = CFG.load(a.config)
    if a.cmd == "youtube-playlist":
        from . import youtube as Y
        pl = Y.create_playlist(a.token or cfg.youtube_token, a.title,
                               a.description, a.privacy)
        print(f"playlist created ({pl['privacy']}): {pl['url']}\n\n"
              f"Put this in /srv/anya/config.toml under [youtube]:\n\n"
              f'    playlist_id = "{pl["id"]}"\n\n'
              f"then: sudo systemctl restart anya-worker anya-recorder")
        return 0
    if a.cmd in ("calibrate-frame", "calibrate-save"):
        # The recorder page reads stdout: one JSON object, errors included.
        import contextlib
        import json
        from . import calibrate as K
        try:
            # pipeline code prints progress; keep stdout for the one JSON line
            with contextlib.redirect_stdout(sys.stderr):
                out = (K.frame(a.video, a.out_png) if a.cmd == "calibrate-frame"
                       else K.save(cfg, a.video, json.loads(a.points)))
        except K.BadCorners as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            return 2
        except Exception as e:              # noqa: BLE001 -- shown on the page
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
            return 1
        print(json.dumps(out))
        return 0
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
            ups = [f"{what}: {st}" for key, what in (("youtube_raw", "raw"),
                                                     ("youtube", "youtube"))
                   if (st := (j.get(key) or {}).get("status")) and st != J.UP_DISABLED]
            print(f"{j['id']}  {j['status']:<18} {j.get('stage') or ''}  "
                  f"{'  '.join(ups)}")
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
        elif a.raw:
            if not j.get("chapters"):
                print("this job has no recording to upload", file=sys.stderr)
                return 1
            j["youtube_raw"] = {"status": J.UP_PENDING, "attempts": 0}
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
