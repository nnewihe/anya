#!/usr/bin/env python3
"""
Start/stop web page for recording tennis from a camera on the Raspberry Pi.

    python3 recorder.py [--dir /srv/anya/recordings] [--port 8080]

Open http://<pi>.local:8080 on a phone on the same network.

The camera is a GoPro HERO8 in webcam mode by default (ANYA_CAMERA=gopro):
over USB it is a network device, not a webcam, so the page asks it over HTTP
to stream and ffmpeg records the H.264 it sends by UDP.  ANYA_CAMERA=usb
records a USB webcam (the DJI Osmo Action 4 in webcam mode) with ffmpeg, and
ANYA_CAMERA=picam a ribbon-cable Pi camera through rpicam-vid.

Start records until Stop is pressed, for at most 90 minutes, or until the disk
is nearly full.  If another program already has the camera, the page asks
before closing it.  Recording only saves what the camera sends: the webcam's
MJPEG into <dir>/<YYYY-MM-DD_HHMMSS>.mkv (the Pi encoding H.264 live made the
video choppy), the GoPro's or a Pi camera's H.264 into .ts.  Both stay readable after a crash
or power cut, which an MP4 without its index does not.  After Stop the file
becomes an .mp4 in the background: the MJPEG is encoded to H.264, a .ts is
just rewrapped.  The original is deleted only once the .mp4 reads back with
the same duration, and anything left unconverted by a crash is converted the
next time the service starts.

Each .mp4 is then queued for the anya Pi worker (pi/anya_pi), which uploads it
to YouTube, cuts the dead time out, and uploads the highlights.  The page shows
how far each recording has got.  While the camera records, the worker pauses.

Standard library only (plus pi/anya_pi, itself stdlib): the Pi needs ffmpeg
and psmisc (fuser), or rpicam-apps for the Pi camera.
"""

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # pi/, for anya_pi
from anya_pi import config as CFG  # noqa: E402
from anya_pi import jobs as J  # noqa: E402

# Which camera: "gopro" (a HERO8 in webcam mode), "usb" (a webcam, e.g. the
# Osmo Action 4 in webcam mode) or "picam" (a ribbon-cable Pi camera).
CAMERA = os.environ.get("ANYA_CAMERA", "gopro")

# --- GoPro HERO8 (webcam mode) ------------------------------------------------
# Plugged in by USB, the HERO8 shows up as a network card, not a V4L2 webcam:
# the Pi gets an address 172.2x.1yy.z from it and the camera is 172.2x.1yy.51.
# GET /gp/gpWebcam/START?res=1080 makes it send H.264 in MPEG-TS by UDP to the
# Pi's port 8554, STOP ends it.  No sound comes with it.  ANYA_GOPRO_IP skips
# looking for the camera's address.
GOPRO_IP = os.environ.get("ANYA_GOPRO_IP", "")
GOPRO_RES = os.environ.get("ANYA_GOPRO_RES", "1080")     # 1080, 720 or 480
GOPRO_PORT = 8554
GOPRO_HTTP_TIMEOUT_S = 5
# The camera unplugged, asleep or overheated just stops sending: without a
# timeout ffmpeg would wait forever, and the page would say Recording.
GOPRO_SILENCE_S = 10

# --- USB webcam -------------------------------------------------------------
# The device: its permanent /dev/v4l/by-id name (the Pi 5 has a dozen video
# devices of its own, so /dev/video0 is not a safe guess).  Empty: the first
# webcam found.
USB_DEVICE = os.environ.get("ANYA_CAMERA_DEVICE", "")
# The Osmo Action 4 lists 1920x1080 at 30 fps as MJPEG or H.264 (v4l2-ctl
# --list-formats-ext), but on the Pi its H.264 will not stream: the driver
# refuses the buffers (VIDIOC_REQBUFS: Invalid argument), under ffmpeg and
# v4l2-ctl alike.  So it sends MJPEG, saved untouched while recording and
# encoded to H.264 after Stop: encoding live made the video choppy.  "h264"
# (copied as is) is for a webcam whose H.264 does stream.  The Osmo's webcam
# mode offers 30 fps only, at 720p or 1080p.  Exposure and focus are set on
# the camera, which exposes no controls to the Pi.
USB_FORMAT = os.environ.get("ANYA_CAMERA_FORMAT", "mjpeg")
USB_SIZE = os.environ.get("ANYA_CAMERA_SIZE", "1920x1080")
USB_FPS = "30"
# x264 speed for the conversion after Stop.  Nothing waits on it in real time,
# so this trades only conversion time (and the CPU it shares with processing).
CONVERT_PRESET = os.environ.get("ANYA_CONVERT_PRESET", "veryfast")
# Sound: the webcam's own microphone (the Osmo is also a USB audio device),
# found as the ALSA card on the same USB device as the video.  An ALSA name
# such as "plughw:CARD=OsmoAction4,DEV=0" picks one; "none" records no sound.
USB_AUDIO = os.environ.get("ANYA_AUDIO_DEVICE", "auto")


def capture_suffix():
    """What the camera's stream is saved as while recording."""
    return ".mkv" if CAMERA == "usb" and USB_FORMAT == "mjpeg" else ".ts"


def usb_args(device, audio=None):
    # Copy only: whatever the camera sends goes to disk as is.  Matroska takes
    # MJPEG and, like MPEG-TS, stays readable if the recording is cut off.
    container = "matroska" if USB_FORMAT == "mjpeg" else "mpegts"
    sound_in, sound_out, clock = [], [], []
    if audio:
        # ALSA stamps samples with the wall clock, V4L2 frames with the kernel's
        # monotonic one: mono2abs puts the video on the wall clock too, so the
        # two line up.  The sound is kept as raw PCM (no CPU); the conversion
        # after Stop makes it AAC.
        clock = ["-ts", "mono2abs"]
        sound_in = ["-f", "alsa", "-thread_queue_size", "1024", "-i", audio]
        sound_out = ["-map", "1:a",
                     "-c:a", "pcm_s16le" if container == "matroska" else "aac"]
    return ["-hide_banner", "-nostdin", "-loglevel", "warning",
            "-f", "v4l2", "-input_format", USB_FORMAT, "-video_size", USB_SIZE,
            "-framerate", USB_FPS, "-thread_queue_size", "512", *clock,
            "-i", device, *sound_in,
            "-map", "0:v", "-c:v", "copy", *sound_out,
            "-f", container, "-y"]


def find_usb_audio(video_device):
    """The ALSA capture device on the same USB device as the webcam, or None."""
    if USB_AUDIO == "none":
        return None
    if USB_AUDIO != "auto":
        return USB_AUDIO
    node = os.path.basename(os.path.realpath(video_device))      # video0
    iface = os.path.realpath(f"/sys/class/video4linux/{node}/device")
    usb = os.path.dirname(iface)                   # the USB device, not interface
    for card in sorted(glob.glob("/sys/class/sound/card[0-9]*")):
        if os.path.dirname(os.path.realpath(f"{card}/device")) != usb:
            continue
        try:
            card_id = Path(card, "id").read_text().strip()
        except OSError:
            continue
        # plughw: ALSA converts to whatever rate/channels ffmpeg asks for.
        return f"plughw:CARD={card_id},DEV=0"
    return None


def find_usb_device():
    """The webcam's device path, or None when none is plugged in."""
    if USB_DEVICE:
        return USB_DEVICE if os.path.exists(USB_DEVICE) else None
    found = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    return found[0] if found else None


def find_gopro():
    """The GoPro's IP address on its USB network link, or None."""
    if GOPRO_IP:
        return GOPRO_IP
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"\binet (172\.2\d\.1\d\d)\.\d+/", out)
    return f"{m.group(1)}.51" if m else None


def gopro_call(ip, command):
    """GET /gp/gpWebcam/<command> on the camera.  Returns an error, or None."""
    try:
        with urllib.request.urlopen(f"http://{ip}/gp/gpWebcam/{command}",
                                    timeout=GOPRO_HTTP_TIMEOUT_S) as r:
            r.read()
        return None
    except (OSError, ValueError) as e:      # URLError and HTTPError are OSErrors
        return str(e)


def gopro_args():
    # Copy only, as for the webcam.  fifo_size (in 188-byte packets, ~9 MB)
    # and overrun_nonfatal ride out a slow disk write; timeout (microseconds)
    # ends the recording once the camera goes quiet.  Video only: the stream
    # also lists an AC-3 track with no channels or sample rate, which the
    # MPEG-TS muxer refuses ("Could not write header"), so ffmpeg quit at once.
    url = (f"udp://@0.0.0.0:{GOPRO_PORT}?overrun_nonfatal=1&fifo_size=50000"
           f"&timeout={GOPRO_SILENCE_S * 1_000_000}")
    return ["-hide_banner", "-nostdin", "-loglevel", "warning", "-i", url,
            "-map", "0:v", "-c", "copy", "-f", "mpegts", "-y"]


def camera_off():
    """After a recording: the GoPro streams until told to stop."""
    if CAMERA != "gopro":
        return
    ip = find_gopro()
    err = gopro_call(ip, "STOP") if ip else "not found"
    if err:
        log(f"could not stop the GoPro's stream: {err}")


# --- Pi camera (rpicam-vid) ---------------------------------------------------
# Pi 5 has no hardware H.264 encoder, so encoding goes through libav
# (software).  If the log shows dropped frames, lower --framerate or --bitrate
# here.  Tuned for the HQ Camera (IMX477), which has a manual-focus lens: no
# --autofocus-mode.
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
LOW_DISK_STOP_GB = 2     # a running recording stops itself below this
MAX_RECORDING_S = 90 * 60
STOP_GRACE_S = 10        # after SIGINT, before SIGTERM
DURATION_TOLERANCE_S = 1.0

CLOSE_WAIT_S = 3


def camera_command(out, program=None):
    """The command that records to `out`, or an error string."""
    if CAMERA == "picam":
        return [program or "rpicam-vid", *CAMERA_ARGS, "-o", str(out)]
    if CAMERA == "gopro":
        ip = find_gopro()
        if not ip:
            return ("No GoPro found. Plug the HERO8 into the Pi with USB, turn it "
                    "on, and press Start again.")
        err = gopro_call(ip, f"START?res={GOPRO_RES}")
        if err:
            return f"The GoPro ({ip}) would not start streaming: {err}"
        return [program or "ffmpeg", *gopro_args(), str(out)]
    device = find_usb_device()
    if not device:
        return ("No USB camera found. Plug the Osmo Action 4 into the Pi, "
                "switch it to webcam mode, and press Start again.")
    audio = find_usb_audio(device)
    return [program or "ffmpeg", *usb_args(device, audio), str(out)]


def _users_cmd():
    """Lists who else has the camera: pgrep for rpicam programs; for a
    webcam, fuser on its device (prints the PIDs that hold it open); for the
    GoPro, fuser on the UDP port its stream arrives on."""
    if CAMERA == "picam":
        return ["pgrep", "-a", "rpicam"]
    if CAMERA == "gopro":
        return ["fuser", f"{GOPRO_PORT}/udp"]
    device = find_usb_device()
    return ["fuser", os.path.realpath(device)] if device else None


def _close_cmds():
    """How to close them.  sudo first (a sudoers rule from install.sh), so it
    can close a program started from a login shell; the plain command covers
    processes of the service's own user."""
    if CAMERA == "picam":
        cmd = ["pkill", "rpicam"]
    elif CAMERA == "gopro":
        cmd = ["fuser", "-k", "-TERM", f"{GOPRO_PORT}/udp"]
    else:
        device = find_usb_device()
        if not device:
            return ()
        cmd = ["fuser", "-k", "-TERM", os.path.realpath(device)]
    return (["sudo", "-n", *cmd], cmd)


def _command_of(pid):
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode(errors="replace").strip() or f"pid {pid}"
    except OSError:
        return f"pid {pid}"


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


def _encode_args():
    """MJPEG -> the H.264 the rest of anya expects: 4:2:0 (MJPEG decodes to
    4:2:2), a constant 30 fps, a keyframe every second (the highlights are cut
    without re-encoding, so each point starts up to one keyframe early)."""
    return ["-c:v", "libx264", "-preset", CONVERT_PRESET, "-b:v", "16M",
            "-maxrate", "20M", "-bufsize", "32M", "-pix_fmt", "yuv420p",
            "-r", USB_FPS, "-g", USB_FPS]


def finish(src, progress=None, log_path=None):
    """Turn a recording into <stem>.mp4 and delete `src` once the .mp4 checks
    out.  .mkv (MJPEG) is encoded to H.264, .ts is rewrapped as is.
    `progress(fraction)` is called as it goes.  Returns an error, or None."""
    src = Path(src)
    mp4, tmp = src.with_suffix(".mp4"), src.with_suffix(".part.mp4")
    encode = src.suffix == ".mkv"
    src_dur = probe_duration(src)
    # A recording cut off by a crash may not know its own length; MJPEG in
    # Matroska still converts, a .ts with no duration has nothing in it.
    if src_dur is None and not encode:
        return f"{src.name}: unreadable, left as is"
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", str(src),
           "-map", "0:v", "-map", "0:a?",
           *([*_encode_args(), "-c:a", "aac", "-b:a", "160k"] if encode
             else ["-c", "copy"]),
           "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(tmp)]
    if encode and shutil.which("nice"):
        cmd = ["nice", "-n", "10", *cmd]        # behind the camera and the page
    errf = open(log_path, "a") if log_path else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, text=True)
        for line in proc.stdout:
            key, _, val = line.strip().partition("=")
            # out_time_us; older ffmpeg calls it out_time_ms (also microseconds)
            if key in ("out_time_us", "out_time_ms") and progress and src_dur \
                    and val.isdigit():
                progress(min(int(val) / 1e6 / src_dur, 1.0))
        rc = proc.wait()
    finally:
        if log_path:
            errf.close()
    if rc != 0:
        tmp.unlink(missing_ok=True)
        tail = log_tail(log_path) if log_path else ""
        return f"{src.name}: converting to MP4 failed" + (f": {tail}" if tail else "")
    dur = probe_duration(tmp)
    if not dur or (src_dur and abs(dur - src_dur) > DURATION_TOLERANCE_S):
        tmp.unlink(missing_ok=True)
        return (f"{src.name}: the MP4 came out {dur}s long, the recording "
                f"{src_dur}s; kept the original")
    os.replace(tmp, mp4)
    src.unlink()
    log(f"saved {mp4.name} ({dur:.0f} s)")
    return None


def log_tail(path, n=5):
    try:
        lines = Path(path).read_text(errors="replace").strip().splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""


def camera_users():
    """[{pid, command}] of the programs holding the camera."""
    cmd = _users_cmd()
    if not cmd:
        return []
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    if cmd[0] == "pgrep":
        for line in r.stdout.splitlines():
            pid, _, command = line.strip().partition(" ")
            if pid.isdigit():
                out.append({"pid": int(pid), "command": command or "rpicam"})
    else:                                   # fuser: PIDs on stdout
        for pid in re.findall(r"\d+", r.stdout):
            if int(pid) != os.getpid():
                out.append({"pid": int(pid), "command": _command_of(pid)})
    return out


def close_hint():
    """What to run by hand when closing failed."""
    if CAMERA == "picam":
        return "sudo pkill rpicam"
    if CAMERA == "gopro":
        return f"sudo fuser -k {GOPRO_PORT}/udp"
    device = find_usb_device()
    return f"sudo fuser -k {os.path.realpath(device)}" if device else "unplug the camera"


def close_camera_users():
    """Close whatever holds the camera; returns what is still running after."""
    left = camera_users()
    for cmd in _close_cmds():
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


def clean_player(name):
    """A first name fit for a YouTube title: letters, spaces, - ' . only."""
    name = "".join(c for c in str(name or "") if c.isalpha() or c in " -'.")
    return " ".join(name.split())[:30]


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
        # Replaces the recording program (ffmpeg / rpicam-vid); the tests'
        # fake camera comes in this way.
        self.camera = camera or os.environ.get("ANYA_CAMERA_PROGRAM") or None
        self.cfg = cfg
        self.queue = J.Queue(cfg) if cfg else None
        if cfg:
            cfg.ensure_dirs()
            # A flag left by a crash would pause the worker forever.
            cfg.recording_flag.unlink(missing_ok=True)
        self.notice = None
        self.player = ""
        self.lock = threading.Lock()
        self.state = IDLE
        self.proc = None
        self.file = None
        self.started = None
        self.error = None
        self.pending_rewraps = 0
        self.converting = {}             # stem -> fraction done (None: waiting)
        self._convert_lock = threading.Lock()   # one conversion at a time
        self._durations = {}             # (name, mtime) -> seconds

    # -- recording -----------------------------------------------------
    def free_gb(self):
        return shutil.disk_usage(self.dir).free / 1e9

    def start(self, force=False, player=""):
        """Returns an error string, or None.  Raises CameraBusy when another
        rpicam program has the camera, unless `force`, which closes it.
        `player` (a first name) goes into the YouTube titles."""
        player = clean_player(player)
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
                        + f". On the Pi, run: {close_hint()}")
        with self.lock:
            if self.state != IDLE:
                return f"already {self.state}"
            if self.free_gb() < MIN_FREE_GB:
                return f"only {self.free_gb():.1f} GB free; need {MIN_FREE_GB} GB"
            name = f"{dt.datetime.now():%Y-%m-%d_%H%M%S}"
            ts = self.dir / f"{name}{capture_suffix()}"
            cmd = camera_command(ts, self.camera)
            if isinstance(cmd, str):
                self.error = cmd
                return cmd
            logf = open(self.dir / f"{name}.log", "w")
            # Beside the recording, so a crash-recovered one keeps its player.
            if player:
                (self.dir / f"{name}.player").write_text(player + "\n")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=logf, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    # Its own session, so a Ctrl-C in the terminal or a
                    # systemd stop reaches only this server, which then stops
                    # the camera itself and marks the stop as intended.
                    start_new_session=True)
            except OSError as e:
                camera_off()
                self.error = f"could not run {cmd[0]}: {e}"
                return self.error
            finally:
                logf.close()             # the child has its own copy
            self.proc, self.file, self.started = proc, ts, time.time()
            self.player = player
            self.state, self.error, self.notice = RECORDING, None, None
            if self.cfg:
                self.cfg.recording_flag.write_text(ts.name + "\n")
            log(f"recording {ts.name}")
            threading.Thread(target=self._watch, args=(proc,), daemon=True).start()
            threading.Thread(target=self._guard, args=(proc,), daemon=True).start()
            return None

    def _guard(self, proc):
        """Stops a recording at the time limit, or when the disk runs low:
        MJPEG is large, and a full disk would cut it off mid-write."""
        while self.proc is proc and self.state == RECORDING:
            if time.time() - self.started >= MAX_RECORDING_S:
                log(f"{MAX_RECORDING_S // 60}-minute limit reached")
                self.stop(reason="limit")
                return
            if self.free_gb() < LOW_DISK_STOP_GB:
                log(f"under {LOW_DISK_STOP_GB} GB free; stopping")
                self.stop(reason="disk")
                return
            time.sleep(1)

    def stop(self, reason=None):
        """Returns an error string, or None.  Returns once the camera has
        closed the file; the conversion to .mp4 carries on in the background."""
        with self.lock:
            if self.state != RECORDING:
                return "not recording"
            self.state = FINISHING
            proc = self.proc
            if reason == "limit":
                self.notice = (f"Stopped at the {MAX_RECORDING_S // 60}-minute limit. "
                               f"Press Start to keep recording.")
            elif reason == "disk":
                self.notice = ("Stopped: the Pi's disk is nearly full. The recording "
                               "so far is kept.")
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
        # conversion; wait for the first part so Start works as soon as we return.
        while self.state == FINISHING:
            time.sleep(0.02)
        return None

    def _watch(self, proc):
        """Runs for each recording: waits for the camera to exit, for any
        reason, and hands the file to the conversion."""
        rc = proc.wait()
        camera_off()
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
            has_video = ts.exists() and ts.stat().st_size > 0
            if has_video:
                self._claim(ts)          # in the same step as going idle
        if has_video:
            self._finish(ts)
        elif stopped:
            self._set_error(f"{ts.name}: the camera wrote nothing; "
                            f"see {ts.with_suffix('.log').name}")

    def _claim(self, src):
        """Count `src` as converting (call with self.lock held)."""
        self.pending_rewraps += 1
        self.converting[src.stem] = None

    def _finish(self, src):
        """Convert one claimed recording (queued behind any other), then queue
        it for the worker.  The page shows it as converting meanwhile."""
        try:
            with self._convert_lock:
                self.converting[src.stem] = 0.0
                err = finish(src, progress=lambda f: self.converting.__setitem__(src.stem, f),
                             log_path=src.with_suffix(".log"))
            if err:
                self._set_error(err)
            else:
                self._enqueue(src.with_suffix(".mp4"))
        finally:
            with self.lock:
                self.pending_rewraps -= 1
                self.converting.pop(src.stem, None)

    def _set_error(self, msg):
        log(msg)
        with self.lock:
            self.error = msg

    def recover(self):
        """Convert recordings left by a crash or reboot (main() runs this in
        the background: an MJPEG conversion can take as long as the match)."""
        for src in sorted([*self.dir.glob("*.ts"), *self.dir.glob("*.mkv")]):
            if not src.with_suffix(".mp4").exists():
                log(f"recovering {src.name}")
                with self.lock:
                    self._claim(src)
                self._finish(src)

    def _enqueue(self, mp4):
        """Hand a finished recording to the anya worker (once)."""
        if not self.queue or self.queue.get(mp4.stem):
            return
        start = recording_start(mp4.stem) or dt.datetime.fromtimestamp(mp4.stat().st_mtime)
        self.queue.create(mp4.stem,
                          {"start": start.isoformat(timespec="seconds"),
                           "duration": probe_duration(mp4), "source": "picam",
                           "player": self._player_of(mp4)},
                          [mp4], upload_raw=self.cfg.youtube.enabled)
        log(f"queued {mp4.name} for processing")

    def _player_of(self, mp4):
        try:
            return clean_player(mp4.with_suffix(".player").read_text())
        except OSError:
            return ""

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
        for stem, frac in sorted(dict(self.converting).items(), reverse=True):
            text = ("Waiting to convert to MP4" if frac is None
                    else f"Converting to MP4 {frac:.0%}")
            out.append({"name": f"{stem}.mp4", "duration_s": None,
                        "player": self._player_of(self.dir / f"{stem}.mp4"),
                        "chips": [{"text": text, "kind": "busy"}]})
        files = [p for p in self.dir.glob("*.mp4") if not p.name.endswith(".part.mp4")]
        for p in sorted(files, reverse=True)[:limit]:
            job = self.queue.get(p.stem) if self.queue else None
            player = ((job or {}).get("recording") or {}).get("player") \
                or self._player_of(p)
            out.append({"name": p.name, "duration_s": self._duration(p),
                        "player": player, "chips": chips(job)})
        return out

    def status(self):
        with self.lock:
            s = {"state": self.state,
                 "file": self.file.name if self.file else None,
                 "elapsed_s": round(time.time() - self.started, 1) if self.started else None,
                 "error": self.error,
                 "notice": self.notice,
                 "player": self.player if self.state == RECORDING else "",
                 "max_s": MAX_RECORDING_S,
                 "saving": self.pending_rewraps}
        s["recordings"] = self.recordings()
        s["default_session"] = (self.cfg.youtube.session_name if self.cfg
                                else CFG.YouTube().session_name)
        return s


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#070807">
<title>Anya Court Recorder</title>
<style>
/* After anyatennis.com: near-black, white type, the lime accent, pills. */
:root { --lime:#dfff00; --white:#fff; --dark:#070807; --card:#111311;
        --line:#2a2d2a; --field:#080908; --fieldline:#454845;
        --muted:rgba(255,255,255,.62); --stop:#ff4d4d;
        --warnbg:rgba(255,190,60,.12); --warn:#ffcf70;
        --okbg:rgba(223,255,0,.12); --badbg:rgba(255,77,77,.14); --bad:#ff8d8d; }
* { box-sizing:border-box; }
html, body { margin:0; min-height:100%; background:var(--dark); color:var(--white);
             font-family:Arial, Helvetica, sans-serif; }
main { max-width:480px; margin:0 auto; padding:20px 16px 40px; }
.logo { display:block; width:150px; height:auto; margin:4px auto 2px; }
.kicker { text-align:center; color:var(--lime); font-size:12px; font-weight:800;
          letter-spacing:.32em; margin:0 0 26px; }
.card { background:var(--card); border:1px solid rgba(223,255,0,.45);
        border-radius:18px; padding:20px; }
label { display:block; font-size:13px; font-weight:700; margin:0 0 7px; }
input { width:100%; padding:13px 14px; border:1px solid var(--fieldline);
        border-radius:9px; background:var(--field); color:var(--white);
        font:inherit; font-size:17px; outline:none; }
input:focus { border-color:var(--lime); }
input:disabled { opacity:.6; }
.hint { color:var(--muted); font-size:12px; margin:7px 0 20px; }
.status { display:flex; align-items:center; gap:10px; font-size:16px; font-weight:700; }
.dot { width:10px; height:10px; border-radius:50%; background:var(--lime);
       box-shadow:0 0 14px var(--lime); flex:none; }
.rec .dot { background:var(--stop); box-shadow:0 0 14px var(--stop);
            animation:blink 1s infinite; }
@keyframes blink { 50% { opacity:.25; } }
.timer { font-size:60px; font-weight:900; letter-spacing:-.03em; line-height:1;
         margin:14px 0 6px; font-variant-numeric:tabular-nums; }
.limit { color:var(--muted); font-size:13px; margin:0 0 20px; }
button { width:100%; padding:22px; font:inherit; font-size:22px; font-weight:900;
         letter-spacing:.04em; border:0; border-radius:999px; cursor:pointer;
         background:var(--lime); color:#000; transition:.2s ease; }
.rec button { background:var(--stop); color:#fff; }
button:disabled { opacity:.5; cursor:default; }
.err, .note { white-space:pre-wrap; padding:12px 14px; border-radius:9px;
              margin-top:16px; font-size:14px; }
.err { background:var(--warnbg); color:var(--warn); }
.note { background:var(--okbg); color:var(--lime); }
h2 { color:var(--lime); font-size:12px; font-weight:800; letter-spacing:.24em;
     text-transform:uppercase; margin:30px 0 10px; }
ul { list-style:none; padding:0; margin:0; background:var(--card);
     border:1px solid var(--line); border-radius:18px; overflow:hidden; }
li { padding:12px 16px; border-top:1px solid var(--line); font-size:14px; }
li:first-child { border-top:0; }
.row { display:flex; justify-content:space-between; gap:8px; }
.row span:last-child { color:var(--muted); white-space:nowrap; }
.who { color:var(--muted); font-size:12px; margin-top:2px; }
.chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.chip { font-size:12px; font-weight:700; padding:4px 10px; border-radius:999px;
        border:1px solid rgba(255,255,255,.25); color:var(--muted); text-decoration:none; }
.chip.ok { border-color:rgba(223,255,0,.55); color:var(--lime); background:var(--okbg); }
.chip.err { border-color:rgba(255,77,77,.5); color:var(--bad); background:var(--badbg); }
a.chip:hover { background:var(--lime); color:#000; }
footer { text-align:center; color:rgba(255,255,255,.4); font-size:12px; margin-top:28px; }
</style></head>
<body><main id="app">
<svg class="logo" viewBox="60 20 680 545" role="img" aria-label="Anya Tennis">
  <defs><mask id="ball">
    <circle cx="400" cy="240" r="130" fill="white"/>
    <path d="M 356 110 Q 298 240 356 370" fill="none" stroke="black" stroke-width="14" stroke-linecap="round"/>
    <path d="M 444 110 Q 502 240 444 370" fill="none" stroke="black" stroke-width="14" stroke-linecap="round"/>
    <circle cx="400" cy="240" r="44" fill="black"/>
  </mask></defs>
  <g fill="none" stroke="#fff" stroke-linecap="round" stroke-linejoin="round">
    <rect x="100" y="50" width="600" height="380" rx="6" stroke-width="1.5"/>
    <path d="M 100 122 L 100 50 L 172 50" stroke-width="5"/>
    <path d="M 628 50 L 700 50 L 700 122" stroke-width="5"/>
    <path d="M 100 358 L 100 430 L 172 430" stroke-width="5"/>
    <path d="M 628 430 L 700 430 L 700 358" stroke-width="5"/>
    <circle cx="400" cy="240" r="156" stroke-width="1.5" stroke-dasharray="8,7"/>
    <path d="M76 240H100M700 240H724M400 26V50M400 430V454" stroke-width="2.5"/>
    <path d="M76 145H92M76 335H92M708 145H724M708 335H724" stroke-width="1.5"/>
  </g>
  <circle cx="400" cy="240" r="130" fill="#fff" mask="url(#ball)"/>
  <text x="400" y="510" text-anchor="middle" fill="#fff" font-size="70" font-weight="500"
        letter-spacing="18" font-family="'Helvetica Neue', Helvetica, Arial, sans-serif">ANYA</text>
  <text x="400" y="552" text-anchor="middle" fill="#fff" font-size="13" letter-spacing="4"
        font-family="'Courier New', Courier, monospace">see the game differently</text>
</svg>
<p class="kicker">COURT RECORDER</p>

<div class="card">
  <label for="player">Your first name</label>
  <input id="player" maxlength="30" autocomplete="given-name" autocapitalize="words"
         placeholder="e.g. Andy">
  <p class="hint" id="titleHint">Videos are titled “6:30 PM · Oct 1, 2026 · Wimbledon Session”.</p>

  <div class="status"><span class="dot"></span><span id="state">Connecting…</span></div>
  <div class="timer" id="timer">0:00:00</div>
  <div class="limit" id="limit"></div>
  <button id="btn" disabled>Start</button>
  <div class="note" id="note" hidden></div>
  <div class="err" id="err" hidden></div>
</div>

<h2>Recent recordings</h2>
<ul id="list"><li><span>None yet</span></li></ul>
<footer>© 2026 Anya Tennis · Cut the dead time. See the action.</footer>
</main>
<script>
const $ = id => document.getElementById(id);
let state = null, elapsed = 0, base = 0, busy = false, fallback = "Wimbledon Session";

function store(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
function load(k) { try { return localStorage.getItem(k) || ""; } catch (e) { return ""; } }
function hms(s) {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, x = s % 60;
  return h + ":" + String(m).padStart(2, "0") + ":" + String(x).padStart(2, "0");
}
function player() { return $("player").value.trim(); }
function hint() {
  $("titleHint").textContent = "Videos are titled “6:30 PM · Oct 1, 2026 · " +
    (player() ? player() + " Session" : fallback) + "”.";
}
$("player").value = load("anya-player");
$("player").oninput = () => { store("anya-player", player()); hint(); };

function render(s) {
  state = s.state;
  fallback = s.default_session || fallback;
  hint();
  const rec = state === "recording";
  $("app").classList.toggle("rec", rec);
  $("state").textContent = rec ? "Recording" + (s.player ? " · " + s.player : "")
    : state === "finishing" ? "Stopping…"
    : "Ready";
  $("btn").textContent = rec ? "Stop" : "Start";
  $("btn").disabled = busy || state === "finishing";
  $("player").disabled = rec || state === "finishing";
  if (rec) { elapsed = s.elapsed_s; base = performance.now(); } else { elapsed = 0; }
  $("limit").textContent = "Stops automatically at " + hms(s.max_s);
  $("note").hidden = !s.notice;
  $("note").textContent = s.notice || "";
  $("err").hidden = !s.error;
  $("err").textContent = s.error || "";
  const list = $("list");
  list.innerHTML = "";
  if (!s.recordings.length) list.innerHTML = "<li><span>None yet</span></li>";
  for (const r of s.recordings) {
    const li = document.createElement("li");
    li.innerHTML = '<div class="row"><span></span><span></span></div>' +
                   '<div class="who"></div><div class="chips"></div>';
    const row = li.firstChild;
    row.children[0].textContent = r.name;
    row.children[1].textContent = r.duration_s == null ? "" : hms(r.duration_s);
    li.children[1].textContent = r.player || "";
    if (!r.player) li.children[1].remove();
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
    const name = player();
    let [r, body] = await post("/api/start", {player: name});
    if (r.status === 409 && body.busy) {
      const who = body.busy.map(p => p.command + " (pid " + p.pid + ")").join("\n");
      if (!confirm("The camera is currently in use by:\n\n" + who +
                   "\n\nClose it and start a fresh recording?")) return;
      [r, body] = await post("/api/start", {player: name, force: true});
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
                err = (rec.start(force=bool(body.get("force")),
                                 player=body.get("player") or "")
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
    threading.Thread(target=rec.recover, daemon=True).start()
    server = ThreadingHTTPServer((a.host, a.port), make_handler(rec))

    def shutdown(signum, _frame):
        # systemd (or Ctrl-C): close the current recording properly before
        # exiting.  A conversion in progress is dropped; the original stays and
        # is converted again at the next start.
        log(f"got {signal.Signals(signum).name}, shutting down")
        if rec.state == RECORDING:
            rec.stop()
        while rec.state != IDLE:
            time.sleep(0.2)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log(f"serving http://{a.host}:{a.port}  recordings in {rec.dir}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
