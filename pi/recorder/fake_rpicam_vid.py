#!/usr/bin/env python3
"""Stand-in for the camera (ffmpeg on a webcam, or rpicam-vid), for testing
recorder.py off the Pi.

Takes the same arguments, ignores all but the output (after `-o`, or last),
and writes a small real-time MPEG-TS test pattern there until SIGINT, like
the real camera.  Set
FAKE_CAMERA_SECONDS to make it stop by itself (a camera failure), and
FAKE_CAMERA_FAIL=1 to exit at once without writing anything.
"""

import os
import sys

args = sys.argv[1:]
out = args[args.index("-o") + 1] if "-o" in args else args[-1]
if os.environ.get("FAKE_CAMERA_FAIL"):
    print("ERROR: *** no cameras available ***", flush=True)
    sys.exit(1)
limit = os.environ.get("FAKE_CAMERA_SECONDS")
# .mkv: MJPEG, like the Osmo in webcam mode; .ts: H.264, like rpicam-vid.
codec = (["-c:v", "mjpeg", "-pix_fmt", "yuvj422p", "-f", "matroska"]
         if out.endswith(".mkv") else
         ["-c:v", "libx264", "-preset", "ultrafast", "-g", "30", "-f", "mpegts"])
os.execvp("ffmpeg", [
    "ffmpeg", "-v", "error", "-nostdin", "-re",
    "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30",
    *(["-t", limit] if limit else []),
    *codec, "-y", out])
