# Court recorder (Raspberry Pi camera)

A web page on the Pi with one big **Start / Stop** button. It records with the
Pi's own camera, then does the rest of the work without anyone touching it:

```
phone ──▶ recorder.py :8080 ──▶ rpicam-vid ──▶ recordings/2026-10-01_183005.mp4
                                                   │  queued for anya-worker
                                                   ▼
             YouTube (unlisted)  "6:30 PM · Oct 1, 2026 · Andy Session"
             anya on the Pi      cuts the dead time out (~2–4× the recording's length)
             YouTube (unlisted)  "6:30 PM · Oct 1, 2026 · Andy Session Highlights"
```

- **Your first name** goes in the box above Start. It replaces "Wimbledon" in both
  YouTube titles, and the phone remembers it. Leave it empty to get
  `[youtube] session_name` ("Wimbledon Session").

- **Start** checks first whether another `rpicam` program (`rpicam-hello`, a
  hand-run `rpicam-vid`) has the camera. If one does, the page lists it and asks
  whether to close it. **OK** runs `pkill rpicam` and starts a fresh recording.
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

### 1. YouTube, as nnewihe@gmail.com
The steps are in `pi/README.md` → *YouTube*. In short:
1. Create a Google Cloud project and enable **YouTube Data API v3**.
2. Set the OAuth consent screen to **In production**. In *Testing*, the login expires
   after 7 days.
3. Create a *Desktop app* OAuth client.
4. On the Mac, run `youtube-auth` and **sign in as nnewihe@gmail.com**.
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

They're in `CAMERA_ARGS` at the top of `recorder.py`, tuned for the **HQ Camera
(IMX477)** on a **Pi 5**:

| Setting | Why |
|---|---|
| 1920×1080 at **50 fps** | The IMX477's fastest mode that covers the whole width at 1080p. At 60 fps it would drop to a lower-resolution mode. |
| no `--autofocus-mode` | The HQ Camera's lens is focused by hand. |
| `--codec libav --libav-format mpegts` | The Pi 5 has no hardware H.264 encoder. MPEG-TS survives a crash. |
| `--bitrate 16000000` | 16 Mbps, about 7 GB an hour (11 GB for 90 minutes). |
| `--intra 50` | A keyframe every second, so each point in the highlights starts at most a second early. |
| `--timeout 0`, `-n` | Record until stopped, with no preview window. |

The page refuses to start with less than 5 GB free. A recording is deleted from the
Pi `keep_inbox_days` (14) days after it's on YouTube and its highlights are done.

## Check on the first sessions
- **Dropped frames.** Count a 2-minute recording's frames. You should get about
  6000 (120 s × 50).
  ```bash
  ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 /srv/anya/recordings/<name>.mp4
  ```
- **Highlights quality.** anya was tuned on 4K action-camera footage. At 1080p the far
  player is half the size, so the first processed session is the real test.
- **Processing speed.** The worker log reports "× realtime" for each job. If decoding
  errors appear, set `hwaccel = ""` in `/srv/anya/config.toml`: the Pi 5 decodes
  H.264 only in software.

## Tests (on a laptop, needs ffmpeg)

```bash
python -m pytest pi/recorder pi/tests
```

`fake_rpicam_vid.py` stands in for the camera. To click through the page on a laptop:

```bash
RPICAM_VID=pi/recorder/fake_rpicam_vid.py python3 pi/recorder/recorder.py --dir /tmp/rec --no-queue
```
