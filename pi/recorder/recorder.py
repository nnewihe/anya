#!/usr/bin/env python3
"""
Start/stop web page for recording tennis with the Raspberry Pi camera.

    python3 recorder.py [--dir /srv/anya/recordings] [--port 8080]

Open http://<pi>.local:8080 on a phone on the same network.

Start launches rpicam-vid, which records to <dir>/<YYYY-MM-DD_HHMMSS>.ts until
Stop is pressed, or for at most 90 minutes.  If another rpicam program already
has the camera, the page asks before closing it (`pkill rpicam`).  MPEG-TS
survives a crash or power cut (an MP4 without its index does not).  On stop
the file is rewrapped, without re-encoding, into an .mp4 next to it, and the
.ts is deleted only after the .mp4 reads back with the same duration.

Each .mp4 is then queued for the anya Pi worker (pi/anya_pi), which uploads it
to YouTube, cuts the dead time out, and uploads the highlights.  The page shows
how far each recording has got.  While the camera records, the worker pauses.

Standard library only (plus pi/anya_pi, itself stdlib): the Pi needs
rpicam-apps and ffmpeg for this part.
"""

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pi/, for anya_pi
from anya_pi import config as CFG  # noqa: E402
from anya_pi import jobs as J  # noqa: E402

# The camera settings.  Pi 5 has no hardware H.264 encoder, so encoding goes
# through libav (software).  If the log shows dropped frames, lower
# --framerate or --bitrate here.  Tuned for the HQ Camera (IMX477), which has
# a manual-focus lens: no --autofocus-mode.
CAMERA_ARGS = [
    "--width", "1920",
    "--height", "1080",
    # The HQ Camera (IMX477) tops out at 50 fps in its 2028x1080 mode; asking
    # for 60 makes it fall back to the lower-resolution 1332x990 mode.
    "--framerate", "50",
    "--shutter", "1000",
    "--gain", "20",
    "--sharpness", "1.5",
    "--denoise", "cdn_off",
    "--awb", "auto",
    "--bitrate", "16000000",            # 16 Mbps, about 7 GB an hour
    # A keyframe every second: the highlights are cut without re-encoding,
    # so each point starts up to one keyframe interval early.
    "--intra", "50",
    "--timeout", "0",                   # record until stopped
    "--codec", "libav",
    "--libav-format", "mpegts",
    "-n",                               # no preview window
]

MIN_FREE_GB = 5
MAX_RECORDING_S = 90 * 60
STOP_GRACE_S = 10        # after SIGINT, before SIGTERM
DURATION_TOLERANCE_S = 1.0

# Who else has the camera, and how to close them.  sudo first (a sudoers rule
# from install.sh) so it can close an rpicam-hello started from a login shell;
# the plain pkill covers processes of the service's own user.
CAMERA_USERS_CMD = ["pgrep", "-a", "rpicam"]
CLOSE_CAMERA_CMDS = (["sudo", "-n", "pkill", "rpicam"], ["pkill", "rpicam"])
CLOSE_WAIT_S = 3


class CameraBusy(Exception):
    def __init__(self, procs):
        super().__init__("camera in use")
        self.procs = procs

IDLE, RECORDING, FINISHING = "idle", "recording", "finishing"


def log(msg):
    print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def probe_duration(path):
    """Seconds, or None if ffprobe can't read the file."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def rewrap(ts):
    """Copy .ts into .mp4 without re-encoding; delete the .ts once the .mp4
    checks out.  Returns an error string, or None on success."""
    ts = Path(ts)
    mp4, tmp = ts.with_suffix(".mp4"), ts.with_suffix(".part.mp4")
    src_dur = probe_duration(ts)
    if src_dur is None:
        return f"{ts.name}: unreadable, left as is"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(ts), "-map", "0",
         "-c", "copy", "-movflags", "+faststart", str(tmp)],
        capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return f"{ts.name}: rewrap failed: {r.stderr.strip()[-300:]}"
    dur = probe_duration(tmp)
    if dur is None or abs(dur - src_dur) > DURATION_TOLERANCE_S:
        tmp.unlink(missing_ok=True)
        return f"{ts.name}: rewrapped file is {dur}s, source {src_dur:.1f}s; kept the .ts"
    os.replace(tmp, mp4)
    ts.unlink()
    log(f"saved {mp4.name} ({dur:.0f} s)")
    return None


def log_tail(path, n=5):
    try:
        lines = Path(path).read_text(errors="replace").strip().splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""


def camera_users():
    """[{pid, command}] of running rpicam programs."""
    try:
        r = subprocess.run(CAMERA_USERS_CMD, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in r.stdout.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit():
            out.append({"pid": int(pid), "command": cmd or "rpicam"})
    return out


def close_camera_users():
    """`pkill rpicam`; returns whatever is still running afterwards."""
    left = camera_users()
    for cmd in CLOSE_CAMERA_CMDS:
        if not left:
            break
        log(f"closing {', '.join(p['command'] for p in left)}: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        end = time.time() + CLOSE_WAIT_S
        while (left := camera_users()) and time.time() < end:
            time.sleep(0.2)
    return left


def recording_start(stem):
    try:
        return dt.datetime.strptime(stem, "%Y-%m-%d_%H%M%S")
    except ValueError:
        return None


def chips(job):
    """What the page shows for a recording's job: [{text, url?, kind}]."""
    if not job:
        return []
    out = []
    raw = job.get("youtube_raw") or {}
    st = raw.get("status")
    if st == J.UP_DONE:
        out.append({"text": "Raw on YouTube" + (" (private)" if raw.get("forced_private")
                                                else ""),
                    "url": raw.get("url"), "kind": "ok"})
    elif st == J.UP_UPLOADING:
        out.append({"text": "Uploading raw", "kind": "busy"})
    elif st == J.UP_PENDING:
        out.append({"text": "Raw: retrying upload" if raw.get("error")
                    else "Raw: waiting to upload", "kind": "busy"})
    elif st == J.UP_FAILED:
        out.append({"text": "Raw upload failed", "kind": "err"})

    s = job["status"]
    if s == J.PENDING:
        out.append({"text": "Highlights: paused while recording"
                    if job.get("stage") == "paused while recording"
                    else "Highlights: queued", "kind": "busy"})
    elif s == J.RUNNING:
        stage = (job.get("stage") or "").split(" ", 1)[-1]
        pct = f" {job['progress']:.0%}" if job.get("progress") is not None else ""
        out.append({"text": f"Highlights: {stage}{pct}", "kind": "busy"})
    elif s == J.NEEDS_CALIBRATION:
        out.append({"text": "Highlights: needs court calibration", "kind": "err"})
    elif s in (J.FAILED, J.UNSUPPORTED):
        out.append({"text": "Highlights failed", "kind": "err"})
    elif s == J.DONE:
        yt = job.get("youtube") or {}
        if yt.get("status") == J.UP_DONE:
            out.append({"text": "Highlights on YouTube" + (
                " (private)" if yt.get("forced_private") else ""),
                "url": yt.get("url"), "kind": "ok"})
        elif yt.get("status") in (J.UP_PENDING, J.UP_UPLOADING):
            out.append({"text": "Uploading highlights", "kind": "busy"})
        elif yt.get("status") == J.UP_FAILED:
            out.append({"text": "Highlights upload failed", "kind": "err"})
        else:
            out.append({"text": "Highlights ready", "kind": "ok"})
    return out


class Recorder:
    def __init__(self, out_dir, camera=None, cfg=None):
        """`cfg` (anya_pi.config.Config): queue each recording for the worker
        and pause it while recording.  None: just record."""
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.camera = camera or os.environ.get("RPICAM_VID", "rpicam-vid")
        self.cfg = cfg
        self.queue = J.Queue(cfg) if cfg else None
        if cfg:
            cfg.ensure_dirs()
            # A flag left by a crash would pause the worker forever.
            cfg.recording_flag.unlink(missing_ok=True)
        self.timer = None
        self.notice = None
        self.lock = threading.Lock()
        self.state = IDLE
        self.proc = None
        self.file = None
        self.started = None
        self.error = None
        self.pending_rewraps = 0
        self._durations = {}             # (name, mtime) -> seconds

    # -- recording -----------------------------------------------------
    def free_gb(self):
        return shutil.disk_usage(self.dir).free / 1e9

    def start(self, force=False):
        """Returns an error string, or None.  Raises CameraBusy when another
        rpicam program has the camera, unless `force`, which closes it."""
        if self.state == IDLE:
            # Our own camera process is gone when idle, so anything found
            # belongs to someone else.  Outside the lock: closing takes seconds.
            users = camera_users()
            if users and not force:
                raise CameraBusy(users)
            if users:
                left = close_camera_users()
                if left:
                    return ("could not close " + ", ".join(
                        f"{p['command']} (pid {p['pid']})" for p in left)
                        + ". On the Pi, run: sudo pkill rpicam")
        with self.lock:
            if self.state != IDLE:
                return f"already {self.state}"
            if self.free_gb() < MIN_FREE_GB:
                return f"only {self.free_gb():.1f} GB free; need {MIN_FREE_GB} GB"
            name = f"{dt.datetime.now():%Y-%m-%d_%H%M%S}"
            ts = self.dir / f"{name}.ts"
            logf = open(self.dir / f"{name}.log", "w")
            try:
                proc = subprocess.Popen(
                    [self.camera, *CAMERA_ARGS, "-o", str(ts)],
                    stdout=logf, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    # Its own session, so a Ctrl-C in the terminal or a
                    # systemd stop reaches only this server, which then stops
                    # the camera itself and marks the stop as intended.
                    start_new_session=True)
            except OSError as e:
                self.error = f"could not run {self.camera}: {e}"
                return self.error
            finally:
                logf.close()             # the child has its own copy
            self.proc, self.file, self.started = proc, ts, time.time()
            self.state, self.error, self.notice = RECORDING, None, None
            if self.cfg:
                self.cfg.recording_flag.write_text(ts.name + "\n")
            self.timer = threading.Timer(MAX_RECORDING_S, self._limit, args=(proc,))
            self.timer.daemon = True
            self.timer.start()
            log(f"recording {ts.name}")
            threading.Thread(target=self._watch, args=(proc,), daemon=True).start()
            return None

    def _limit(self, proc):
        if self.proc is proc and self.state == RECORDING:
            log(f"{MAX_RECORDING_S // 60}-minute limit reached")
            self.stop(reason="limit")

    def stop(self, reason=None):
        """Returns an error string, or None.  Returns once the camera has
        closed the file; the rewrap to .mp4 carries on in the background."""
        with self.lock:
            if self.state != RECORDING:
                return "not recording"
            self.state = FINISHING
            proc = self.proc
            if reason == "limit":
                self.notice = (f"Stopped at the {MAX_RECORDING_S // 60}-minute limit. "
                               f"Press Start to keep recording.")
        log("stopping")
        for sig, wait in ((signal.SIGINT, STOP_GRACE_S), (signal.SIGTERM, 5),
                          (signal.SIGKILL, 5)):
            if proc.poll() is not None:
                break
            proc.send_signal(sig)
            try:
                proc.wait(wait)
            except subprocess.TimeoutExpired:
                log(f"camera ignored {sig.name}")
        # _watch sees the exit, returns the state to idle and starts the
        # rewrap; wait for the first part so Start works as soon as we return.
        while self.state == FINISHING:
            time.sleep(0.02)
        return None

    def _watch(self, proc):
        """Runs for each recording: waits for the camera to exit, for any
        reason, and hands the file to the rewrap."""
        rc = proc.wait()
        if self.timer:
            self.timer.cancel()
        if self.cfg:
            self.cfg.recording_flag.unlink(missing_ok=True)
        with self.lock:
            stopped = self.state == FINISHING
            ts, started = self.file, self.started
            if not stopped:
                # Nobody pressed Stop: the camera failed or was unplugged.
                tail = log_tail(ts.with_suffix(".log"))
                self.error = (f"camera stopped by itself after "
                              f"{time.time() - started:.0f} s (exit {rc})"
                              + (f":\n{tail}" if tail else ""))
                log(self.error)
            self.state, self.proc, self.file, self.started = IDLE, None, None, None
            self.pending_rewraps += 1
        try:
            if ts.exists() and ts.stat().st_size > 0:
                err = rewrap(ts)
                if err:
                    self._set_error(err)
                else:
                    self._enqueue(ts.with_suffix(".mp4"))
            elif stopped:
                self._set_error(f"{ts.name}: the camera wrote nothing; "
                                f"see {ts.with_suffix('.log').name}")
        finally:
            with self.lock:
                self.pending_rewraps -= 1

    def _set_error(self, msg):
        log(msg)
        with self.lock:
            self.error = msg

    def recover(self):
        """Rewrap .ts files left by a crash or reboot."""
        for ts in sorted(self.dir.glob("*.ts")):
            if not ts.with_suffix(".mp4").exists():
                log(f"recovering {ts.name}")
                err = rewrap(ts)
                if err:
                    self._set_error(err)
                else:
                    self._enqueue(ts.with_suffix(".mp4"))

    def _enqueue(self, mp4):
        """Hand a finished recording to the anya worker (once)."""
        if not self.queue or self.queue.get(mp4.stem):
            return
        start = recording_start(mp4.stem) or dt.datetime.fromtimestamp(mp4.stat().st_mtime)
        self.queue.create(mp4.stem,
                          {"start": start.isoformat(timespec="seconds"),
                           "duration": probe_duration(mp4), "source": "picam"},
                          [mp4], upload_raw=self.cfg.youtube.enabled)
        log(f"queued {mp4.name} for processing")

    # -- status --------------------------------------------------------
    def _duration(self, p):
        st = p.stat()
        key = (p.name, st.st_mtime)
        if key not in self._durations:
            self._durations[key] = probe_duration(p)
        return self._durations[key]

    def recordings(self, limit=10):
        """The newest recordings, with how far the worker has got."""
        out = []
        files = [p for p in self.dir.glob("*.mp4") if not p.name.endswith(".part.mp4")]
        for p in sorted(files, reverse=True)[:limit]:
            job = self.queue.get(p.stem) if self.queue else None
            out.append({"name": p.name, "duration_s": self._duration(p),
                        "size_mb": round(p.stat().st_size / 1e6),
                        "chips": chips(job)})
        return out

    def status(self):
        with self.lock:
            s = {"state": self.state,
                 "file": self.file.name if self.file else None,
                 "elapsed_s": round(time.time() - self.started, 1) if self.started else None,
                 "error": self.error,
                 "notice": self.notice,
                 "max_s": MAX_RECORDING_S,
                 "saving": self.pending_rewraps}
        s["free_gb"] = round(self.free_gb(), 1)
        s["recordings"] = self.recordings()
        return s


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Court Recorder</title>
<style>
:root { --bg:#f6f6f4; --fg:#1b1b1b; --muted:#6b6b6b; --card:#fff; --line:#e2e2de;
        --go:#1f7a3a; --stop:#c62828; --warn:#8a5a00; --warnbg:#fff4dc;
        --info:#1d4f91; --infobg:#e6eefa; --ok:#1f7a3a; --okbg:#e3f3e7;
        --chip:#4b4b4b; --chipbg:#ececea; --bad:#b3261e; --badbg:#fbe5e3; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141414; --fg:#eee; --muted:#9a9a9a; --card:#1f1f1f; --line:#333;
          --go:#2e9d50; --stop:#e04444; --warn:#f0c060; --warnbg:#3a2e14;
          --info:#9cc2f5; --infobg:#17263a; --ok:#7fd49a; --okbg:#173022;
          --chip:#c9c9c9; --chipbg:#2b2b2b; --bad:#f2a49c; --badbg:#3a1d1a; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:16px/1.4 -apple-system, system-ui, sans-serif; }
main { max-width:480px; margin:0 auto; padding:24px 16px; }
h1 { font-size:20px; margin:0 0 20px; }
.status { display:flex; align-items:center; gap:10px; font-size:18px; margin-bottom:6px; }
.dot { width:14px; height:14px; border-radius:50%; background:var(--muted); }
.rec .dot { background:var(--stop); animation:blink 1s infinite; }
@keyframes blink { 50% { opacity:.25; } }
.timer { font:600 56px/1.1 ui-monospace, Menlo, monospace; margin:8px 0 24px;
         font-variant-numeric:tabular-nums; }
button { width:100%; padding:28px; font-size:26px; font-weight:700; border:0;
         border-radius:14px; color:#fff; background:var(--go); cursor:pointer; }
.rec button { background:var(--stop); }
button:disabled { opacity:.5; cursor:default; }
.err, .note { white-space:pre-wrap; padding:12px; border-radius:10px;
       margin-top:16px; font-size:14px; }
.err { background:var(--warnbg); color:var(--warn); }
.note { background:var(--infobg); color:var(--info); }
.limit { color:var(--muted); font-size:14px; margin:-18px 0 22px; }
.meta { color:var(--muted); font-size:14px; margin-top:14px; }
h2 { font-size:15px; margin:28px 0 8px; color:var(--muted); font-weight:600; }
ul { list-style:none; padding:0; margin:0; background:var(--card);
     border:1px solid var(--line); border-radius:10px; }
li { padding:10px 12px; border-top:1px solid var(--line); font-size:14px; }
li:first-child { border-top:0; }
.row { display:flex; justify-content:space-between; gap:8px; }
.row span:last-child { color:var(--muted); white-space:nowrap; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }
.chip { font-size:12px; padding:2px 8px; border-radius:999px;
        background:var(--chipbg); color:var(--chip); text-decoration:none; }
.chip.ok { background:var(--okbg); color:var(--ok); }
.chip.err { background:var(--badbg); color:var(--bad); }
a.chip { text-decoration:underline; }
</style></head>
<body><main id="app">
<h1>Court Recorder</h1>
<div class="status"><span class="dot"></span><span id="state">Connecting…</span></div>
<div class="timer" id="timer">0:00:00</div>
<div class="limit" id="limit"></div>
<button id="btn" disabled>Start</button>
<div class="note" id="note" hidden></div>
<div class="err" id="err" hidden></div>
<div class="meta" id="meta"></div>
<h2>Recent recordings</h2>
<ul id="list"><li><span>None yet</span></li></ul>
</main>
<script>
const $ = id => document.getElementById(id);
let state = null, elapsed = 0, base = 0, busy = false;

function hms(s) {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, x = s % 60;
  return h + ":" + String(m).padStart(2, "0") + ":" + String(x).padStart(2, "0");
}
function render(s) {
  state = s.state;
  const rec = state === "recording";
  $("app").classList.toggle("rec", rec);
  $("state").textContent = rec ? "Recording " + s.file.replace(/\.ts$/, "")
    : state === "finishing" ? "Stopping…"
    : s.saving ? "Saving to MP4…" : "Ready";
  $("btn").textContent = rec ? "Stop" : "Start";
  $("btn").disabled = busy || state === "finishing";
  if (rec) { elapsed = s.elapsed_s; base = performance.now(); } else { elapsed = 0; }
  $("limit").textContent = "Stops automatically at " + hms(s.max_s);
  $("note").hidden = !s.notice;
  $("note").textContent = s.notice || "";
  $("err").hidden = !s.error;
  $("err").textContent = s.error || "";
  $("meta").textContent = s.free_gb + " GB free (about " +
    Math.floor(s.free_gb / 7) + " h of recording)";
  const list = $("list");
  list.innerHTML = "";
  if (!s.recordings.length) list.innerHTML = "<li><span>None yet</span></li>";
  for (const r of s.recordings) {
    const li = document.createElement("li");
    li.innerHTML = '<div class="row"><span></span><span></span></div><div class="chips"></div>';
    const row = li.firstChild;
    row.children[0].textContent = r.name;
    row.children[1].textContent =
      (r.duration_s == null ? "?" : hms(r.duration_s)) + " · " +
      (r.size_mb >= 1000 ? (r.size_mb / 1000).toFixed(1) + " GB" : r.size_mb + " MB");
    for (const c of r.chips) {
      const el = document.createElement(c.url ? "a" : "span");
      el.className = "chip " + c.kind;
      el.textContent = c.text;
      if (c.url) { el.href = c.url; el.target = "_blank"; el.rel = "noopener"; }
      li.lastChild.appendChild(el);
    }
    if (!r.chips.length) li.lastChild.remove();
    list.appendChild(li);
  }
}
async function poll() {
  try {
    const r = await fetch("/api/status", {cache: "no-store"});
    render(await r.json());
  } catch (e) {
    $("state").textContent = "Can't reach the Pi";
  }
}
async function post(path, body) {
  const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"},
                               body: JSON.stringify(body || {})});
  return [r, await r.json()];
}
function showError(msg) { $("err").hidden = false; $("err").textContent = msg; }
$("btn").onclick = async () => {
  busy = true; $("btn").disabled = true;
  try {
    if (state === "recording") {
      const [r, body] = await post("/api/stop");
      if (!r.ok) showError(body.error);
      return;
    }
    let [r, body] = await post("/api/start");
    if (r.status === 409 && body.busy) {
      const who = body.busy.map(p => p.command + " (pid " + p.pid + ")").join("\n");
      if (!confirm("The camera is currently in use by:\n\n" + who +
                   "\n\nClose it and start a fresh recording?")) return;
      [r, body] = await post("/api/start", {force: true});
    }
    if (!r.ok) showError(body.error);
  } finally { busy = false; await poll(); }
};
setInterval(() => {
  if (state === "recording") $("timer").textContent = hms(elapsed + (performance.now() - base) / 1000);
  else $("timer").textContent = hms(0);
}, 250);
setInterval(poll, 1000);
poll();
</script></body></html>
"""


def make_handler(rec):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._send(200, rec.status())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/api/start", "/api/stop"):
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except (ValueError, json.JSONDecodeError):
                body = {}
            try:
                err = (rec.start(force=bool(body.get("force")))
                       if self.path == "/api/start" else rec.stop())
            except CameraBusy as e:
                return self._send(409, {"error": "The camera is in use.", "busy": e.procs})
            if err:
                self._send(409, {"error": err})
            else:
                self._send(200, rec.status())

        def log_message(self, fmt, *args):     # status polls every second: keep quiet
            pass

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", help="where recordings go (default <root>/recordings "
                                   "from the anya Pi config)")
    ap.add_argument("--config", help=f"anya Pi settings (default {CFG.DEFAULT_PATH})")
    ap.add_argument("--no-queue", action="store_true",
                    help="just record: don't hand recordings to the anya worker")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)

    cfg = CFG.load(a.config)
    rec = Recorder(a.dir or cfg.recordings, cfg=None if a.no_queue else cfg)
    rec.recover()
    server = ThreadingHTTPServer((a.host, a.port), make_handler(rec))

    def shutdown(signum, _frame):
        # systemd (or Ctrl-C): finish the current file before exiting.
        log(f"got {signal.Signals(signum).name}, shutting down")
        if rec.state == RECORDING:
            rec.stop()
        while rec.status()["saving"] or rec.state != IDLE:
            time.sleep(0.2)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log(f"serving http://{a.host}:{a.port}  recordings in {rec.dir}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
