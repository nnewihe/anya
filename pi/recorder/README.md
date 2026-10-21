# Court recorder (Raspberry Pi + webcam)

A web page on the Pi with one big **Start / Stop** button. It records from a USB
webcam plugged into the Pi (a **DJI Osmo Action 4 in webcam mode**), then does the
rest of the work without anyone touching it:

```
phone ──▶ recorder.py :8080 ──▶ ffmpeg (webcam) ──▶ recordings/2026-10-01_183005.mp4
                                                   │  queued for anya-worker
                                                   ▼
             YouTube (unlisted)  "6:30 PM · Oct 1, 2026 · Andy Session"
             anya on the Pi      cuts the dead time out (~2–4× the recording's length)
             YouTube (unlisted)  "6:30 PM · Oct 1, 2026 · Andy Session Highlights"
```

- **Your first name** goes in the box above Start. It replaces "Wimbledon" in both
  YouTube titles, and the phone remembers it. Leave it empty to get
  `[youtube] session_name` ("Wimbledon Session").

- **Start** first checks whether another program already has the camera open
  (found with `fuser`). If one does, the page lists it and asks whether to close it.
  **OK** closes it and starts a fresh recording.
- **Recordings stop by themselves at 90 minutes.** Press Start again to keep going.
- **While recording, the camera writes MPEG-TS,** which survives a crash or power cut.
  Stop rewraps it to `.mp4` without re-encoding. A `.ts` left behind by a crash is
  converted the next time the service starts.
- **Each recording is then queued for `anya-worker`** (see `pi/README.md`):
  - It uploads the raw recording. This happens on its own thread, so slow or no court
    Wi-Fi never holds up processing.
  - It cuts the dead time out.
  - It uploads the highlights.

  The page shows how far each recording has got, with links once a video is on YouTube.
- **Processing pauses while the camera records.** It uses all four cores, and the
  camera's software encoder would drop frames. It resumes after Stop from the stage
  it reached.
- **The page has no login.** Anyone on the same network can use it. Don't expose port
  8080 to the internet.

## Install

This uses the anya Pi service. Get the code onto the Pi into `~/anya` (`git clone`/`git pull`
once this branch is pushed, or copy it from the Mac):

```bash
rsync -a --exclude __pycache__ pipeline walking pi nnewihe@biquet.local:~/anya/
```

Then on the Pi:

```bash
cd ~/anya && sudo pi/install.sh
```

The installer sets up everything:
- the anya user;
- the Python environment with the NCNN pose model;
- `/srv/anya/` (recordings, reels and queue);
- `anya-worker` and `anya-recorder`, which replaces the first hand-installed
  version, including `/opt/anya-recorder`;
- one sudo rule, which allows the `anya` user to run exactly `pkill rpicam`, so the
  page can close an `rpicam-hello` started from your own login.

Re-run it after every update.

The page is at **http://biquet.local:8080**. The logs:

```bash
journalctl -fu anya-recorder -u anya-worker
```

## One-time setup (only you can do these)

### 1. YouTube, as smarttenniswithandy@gmail.com
The steps are in `pi/README.md` → *YouTube*. In short:
1. Create a Google Cloud project and enable **YouTube Data API v3**.
2. Set the OAuth consent screen to **In production**. In *Testing*, the login expires
   after 7 days.
3. Create a *Desktop app* OAuth client.
4. On the Mac, run `youtube-auth` and **sign in as smarttenniswithandy@gmail.com**.
5. Copy the token to `/srv/anya/state/`.
6. Set `[youtube] enabled = true` in `/srv/anya/config.toml`, then run
   `sudo systemctl restart anya-worker anya-recorder`.

A recording made while YouTube was off isn't uploaded later on its own. Queue it with
`python -m anya_pi reupload <name> --raw`, run the same way as `retry` below.

> **Unlisted won't stick at first.** YouTube locks every upload made through an
> **unaudited** Google Cloud project to **private**, whatever the uploader asks for.
> The page marks these videos "(private)". Until the project passes YouTube's
> [API compliance audit](https://support.google.com/youtube/contact/yt_api_form),
> switch each one to Unlisted in YouTube Studio.
> The default quota covers about six uploads a day, so three sessions (raw + highlights each).

The title ends with the first name typed on the page ("Andy Session"). With the box
empty, it ends with `[youtube] session_name` from `/srv/anya/config.toml` instead
(default `"Wimbledon Session"`).

### 2. Court calibration
The court corners are clicked once on a computer with a screen, then reused for every
recording. On the Mac, with any recording from the Pi:

```bash
scp nnewihe@biquet.local:/srv/anya/recordings/<name>.mp4 .
```

```bash
python -m pipeline.anya2.site save <name>.mp4 site
```

Click the four **singles** corners. Then copy the profile to the Pi:

```bash
scp -r site nnewihe@biquet.local:/tmp/site
```

```bash
ssh nnewihe@biquet.local 'sudo rsync -a /tmp/site/ /srv/anya/site/ && sudo chown -R anya:anya /srv/anya/site'
```

Until this is done, recordings show **"needs court calibration"**, but their raw
uploads still go out. Once the site profile is in place, re-queue them:

```bash
sudo -u anya env PYTHONPATH=/opt/anya/src:/opt/anya/src/pi /opt/anya/venv/bin/python -m anya_pi retry <name>
```

If the camera mount moves, redo this.

## Camera settings

At the top of `recorder.py`:

| Setting | Value | Why |
|---|---|---|
| `CAMERA` | `"usb"` | The webcam. Set `ANYA_CAMERA=picam` in the service to go back to a ribbon-cable Pi camera (`CAMERA_ARGS` keeps its settings). |
| `USB_DEVICE` | empty | Uses the first webcam under `/dev/v4l/by-id/`, the permanent name. The Pi 5 has a dozen video devices of its own, so `/dev/video0` isn't a safe guess. Set `ANYA_CAMERA_DEVICE` to choose one. |
| `USB_FORMAT` | `"mjpeg"` | The Osmo lists H.264 too, but on the Pi it won't stream it (`VIDIOC_REQBUFS: Invalid argument`). So it sends MJPEG, and the Pi encodes H.264 as it records (x264, 16 Mbps, about 7 GB an hour, a keyframe every second). |
| `USB_PRESET` | `"veryfast"` | x264's speed setting. Use `"ultrafast"` if the Pi falls behind (see *Check on the first sessions*). |
| `USB_SIZE`, `USB_FPS` | 1920×1080, 30 | The Osmo's best webcam mode (`v4l2-ctl --list-formats-ext`). |

`ANYA_CAMERA_FORMAT`, `ANYA_CAMERA_PRESET` and `ANYA_CAMERA_SIZE` override these
without editing the code. Add them as `Environment=` lines in `anya-recorder.service`.

**Set exposure and focus on the Osmo itself.** It gives the Pi no controls. A fast
shutter keeps the ball sharp, so use the camera's Pro/manual mode if webcam mode
allows it, and turn stabilisation off: the court calibration assumes a fixed view.

Two things the camera, not the Pi, decides:
- **The picture.** MJPEG is the Osmo's own compression, before the Pi re-encodes it,
  so look at the first recording's sharpness.
- **Frame rate.** If the Osmo sends fewer than 30 frames a second (in low light, for
  example), the recording has fewer frames too.

The page refuses to start with less than 5 GB free. A recording is deleted from the
Pi `keep_inbox_days` (14) days after it's on YouTube and its highlights are done.

## Check on the first sessions
- **Dropped frames.** Count a 2-minute recording's frames. You should get about
  3600 (120 s × 30). If it's well short, the encode is falling behind: set
  `ANYA_CAMERA_PRESET=ultrafast`, or drop to `ANYA_CAMERA_SIZE=1280x720`. `top` while
  recording shows how busy `ffmpeg` keeps the CPU.
  ```bash
  ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 /srv/anya/recordings/<name>.mp4
  ```
- **Highlights quality.** anya was tuned on 4K action-camera footage. At 1080p the far
  player is half the size, so the first processed session is the real test. The Osmo
  records 4K on its own card, but webcam mode sends only 1080p at 30 fps.
- **Court calibration** has to be redone with a recording from the webcam: the view
  changed, so the old corners don't line up.
- **Processing speed.** The worker log reports "× realtime" for each job. If decoding
  errors appear, set `hwaccel = ""` in `/srv/anya/config.toml`: the Pi 5 decodes
  H.264 only in software.

## Tests (on a laptop, needs ffmpeg)

```bash
python -m pytest pi/recorder pi/tests
```

`fake_rpicam_vid.py` stands in for the camera. To click through the page on a laptop:

```bash
ANYA_CAMERA_PROGRAM=pi/recorder/fake_rpicam_vid.py python3 pi/recorder/recorder.py --dir /tmp/rec --no-queue
```
