# Court recorder (Raspberry Pi + GoPro)

A web page on the Pi with one big **Start / Stop** button. It records from a camera
plugged into the Pi by USB (a **GoPro HERO8 in webcam mode**), then does the rest of
the work without anyone touching it:

```
phone ──▶ recorder.py :8080 ──▶ ffmpeg (GoPro) ──▶ recordings/2026-10-01_183005.mp4
                                                   │  queued for anya-worker
                                                   ▼
             YouTube (unlisted)  "6:30 PM · Oct 1, 2026 · Andy Session"
             anya on the Pi      cuts the dead time out (~2–4× the recording's length)
             YouTube (unlisted)  "6:30 PM · Oct 1, 2026 · Andy Session Highlights"
```

- **Your first name** goes in the box above Start. It replaces "Wimbledon" in both
  YouTube titles, and the phone remembers it. Leave it empty to get
  `[youtube] session_name` ("Wimbledon Session").

- **The HERO8 isn't a webcam to the Pi.** Over USB it shows up as a network card:
  the Pi gets an address like `172.26.161.52` from it, and the camera is `.51` on the
  same network. **Start** asks the camera over HTTP to stream
  (`/gp/gpWebcam/START?res=1080`), and ffmpeg records the H.264 it sends by UDP to
  port 8554. When the recording ends, for any reason, the page sends `STOP`.
- **Start** first checks whether another program already has the camera's stream
  (whatever holds UDP port 8554, found with `fuser`). If one does, the page lists it
  and asks whether to close it. **OK** closes it and starts a fresh recording.
- **Recordings stop by themselves at 90 minutes,** or when the Pi's disk drops under
  2 GB free. Press Start again to keep going. If the camera stops sending for
  10 seconds (unplugged, asleep, overheated), the recording ends and the page says
  so; what was recorded is kept.
- **While recording, the Pi only saves what the camera sends**: the GoPro's H.264 goes
  untouched into a `.ts`, which stays readable even if the recording is cut off by a
  crash or power cut.
- **No sound.** The HERO8's webcam stream carries video only.
- **After Stop, it's rewrapped to an `.mp4` in the background** (no re-encoding, so
  it takes seconds). The `.ts` is deleted once the `.mp4` checks out. Anything left
  unconverted by a crash or reboot is converted the next time the service starts.
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
- sudo rules that allow the `anya` user to close exactly what holds the camera
  (`fuser -k -TERM 8554/udp` for the GoPro), so the page can close an ffmpeg started
  from your own login.

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

### GoPro HERO8 (the default)

On the camera, once:
- **Webcam mode needs HERO8 firmware 2.01 or later.**
- **Preferences → Connections → USB Connection: GoPro Connect** (not MTP), or the
  Pi sees a storage device instead of a network card.
- **Auto power off: Never.** Then plug it into the Pi and turn it on. It shows a
  webcam/USB screen once the Pi has the network link up.

Check from the Pi that the link is up and the camera answers:

```bash
ip -4 -o addr | grep ' 172\.2'
```

```bash
curl -s http://172.2X.1YY.51/gp/gpWebcam/STOP
```

Replace `172.2X.1YY` with the first three numbers from the `ip` line.

| Setting | Value | Why |
|---|---|---|
| `CAMERA` | `"gopro"` | Set `ANYA_CAMERA=usb` for a USB webcam (the Osmo, below), `picam` for a ribbon-cable Pi camera (`CAMERA_ARGS` keeps its settings). |
| `GOPRO_IP` | empty | Found from the Pi's own address on the camera's network (`172.2x.1yy.z` → `.51`). Set `ANYA_GOPRO_IP` if that guess is wrong. |
| `GOPRO_RES` | `"1080"` | Webcam mode's best: 1080p at 30 fps. `720` and `480` also work (`ANYA_GOPRO_RES`). The camera records 4K/60 to its own card, but not while it's a webcam. |
| `GOPRO_FOV` | `"linear"` | The lens, set after Start (`/gp/gpWebcam/SETTINGS?fov=4`). Webcam mode otherwise streams Wide, whose fisheye bends the court lines. `wide`, `narrow`, or empty to leave the camera's own (`ANYA_GOPRO_FOV`). If the camera refuses, it records anyway and the log says so. |
| `GOPRO_SILENCE_S` | 10 | How long the stream may go quiet before the recording ends. |

Things the camera decides, not the Pi:
- **Framing in Linear.** Linear crops the edges of the Wide view, so check that all
  four court corners are in the first recording; tilt the camera if not. Changing
  the lens means redoing the court calibration.
- **Bitrate and keyframes.** The stream is compressed harder than the camera's own
  recordings. The highlights are cut without re-encoding, so each point starts up
  to one keyframe early.
- **Heat.** HERO8s are known to overheat streaming for a long time, especially in
  the sun. If it shuts down, the recording ends after 10 seconds of silence.

### USB webcam (DJI Osmo Action 4)

| Setting | Value | Why |
|---|---|---|
| `CAMERA` | `"usb"` | Set `ANYA_CAMERA=usb` in the service. |
| `USB_DEVICE` | empty | Uses the first webcam under `/dev/v4l/by-id/`, the permanent name. The Pi 5 has a dozen video devices of its own, so `/dev/video0` isn't a safe guess. Set `ANYA_CAMERA_DEVICE` to choose one. |
| `USB_FORMAT` | `"mjpeg"` | The Osmo lists H.264 too, but on the Pi it won't stream it (`VIDIOC_REQBUFS: Invalid argument`). So it sends MJPEG, which is saved as is and converted after Stop. |
| `CONVERT_PRESET` | `"veryfast"` | x264's speed for the conversion after Stop (16 Mbps, about 7 GB an hour, a keyframe every second). Nothing waits on it live, so a slower preset only means a longer conversion. |
| `USB_SIZE`, `USB_FPS` | 1920×1080, 30 | The Osmo's best webcam mode (`v4l2-ctl --list-formats-ext`). **60 fps isn't possible**: webcam mode offers only 30. |
| `USB_AUDIO` | `"auto"` | The microphone on the same USB device as the webcam (`arecord -l` lists it as `OsmoAction4`). An ALSA name such as `plughw:CARD=OsmoAction4,DEV=0` picks one; `"none"` records no sound. |

`ANYA_CAMERA`, `ANYA_GOPRO_IP`, `ANYA_GOPRO_RES`, `ANYA_GOPRO_FOV`, `ANYA_CAMERA_FORMAT`,
`ANYA_CONVERT_PRESET`, `ANYA_CAMERA_SIZE` and `ANYA_AUDIO_DEVICE` override these
without editing the code. Add them as `Environment=` lines in `anya-recorder.service`.

**MJPEG is large while it waits to be converted.** Check the size of a short test
recording to know how much a 90-minute session needs. A 20-second file of *N* MB
works out to about *N* × 0.27 GB for 90 minutes. The page won't start with less than
5 GB free, and stops a recording when the disk gets under 2 GB.

**Set exposure and focus on the Osmo itself.** It gives the Pi no controls. A fast
shutter keeps the ball sharp, so use the camera's Pro/manual mode if webcam mode
allows it, and turn stabilisation off: the court calibration assumes a fixed view.

Two things the camera, not the Pi, decides:
- **The picture.** MJPEG is the Osmo's own compression, before the Pi converts it, so
  look at the first recording's sharpness.
- **Frame rate.** If the Osmo sends fewer than 30 frames a second (in low light, for
  example), the recording has fewer frames too.

A recording is deleted from the Pi `keep_inbox_days` (14) days after it's on YouTube
and its highlights are done.

## Check on the first sessions
- **Dropped frames.** Count a 2-minute recording's frames. You should get about
  3600 (120 s × 30). Recording only copies, so a big shortfall points at the
  camera or its USB link, not the Pi's CPU. The GoPro's stream is UDP, so lost
  packets also show up as smeared or blocky patches; look through the first one.
- **The network link.** The GoPro's link should not become the Pi's route to the
  internet, or YouTube uploads stall. `ip route` should show `default via` the
  court Wi-Fi's router, not a `172.2x` address.
  ```bash
  ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 /srv/anya/recordings/<name>.mp4
  ```
- **Highlights quality.** anya was tuned on 4K action-camera footage. At 1080p the far
  player is half the size, so the first processed session is the real test. The GoPro
  records 4K on its own card, but webcam mode sends only 1080p at 30 fps.
- **Court calibration** has to be redone with a recording from the GoPro: the view
  changed, so the old corners don't line up.
- **Conversion time.** The recorder log shows when each `.mp4` is saved
  (`journalctl -u anya-recorder`). The YouTube upload and processing start only after.
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
