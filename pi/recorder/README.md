# Court recorder (Raspberry Pi camera)

A web page on the Pi with one big **Start / Stop** button. It records with the
Pi's own camera through `rpicam-vid` and saves one `.mp4` per recording in
`/srv/anya/recordings/`, ready for anya to process.

```
phone ──http──▶ recorder.py :8080 ──▶ rpicam-vid ──▶ 2026-09-30_180000.ts
                                         Stop ──▶ ffmpeg -c copy ──▶ 2026-09-30_180000.mp4
```

- **Recording writes MPEG-TS.** If the Pi crashes or loses power mid-recording,
  everything up to that point can still be read. An MP4 without its index can't be.
- **Stop rewraps to `.mp4` without re-encoding.** It takes a few seconds per hour of
  video. The `.ts` is deleted only once the `.mp4` reads back with the same duration.
- **A `.ts` left behind by a crash is converted the next time the service starts.**
- **Each recording has a `<name>.log`** with `rpicam-vid`'s output. If the camera
  stops by itself (unplugged, or it errors), the page shows the error and the end of this log.

## Install (Raspberry Pi OS Bookworm)

```bash
sudo apt install -y rpicam-apps ffmpeg
```

```bash
sudo mkdir -p /opt/anya-recorder /srv/anya/recordings && sudo chown nnewihe:nnewihe /srv/anya/recordings
```

```bash
sudo cp pi/recorder/recorder.py /opt/anya-recorder/ && sudo cp pi/recorder/anya-recorder.service /etc/systemd/system/
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now anya-recorder
```

Then open **http://\<pi-hostname\>.local:8080** on a phone on the same Wi-Fi.
The service runs as the user `nnewihe` (the Pi `biquet`). For another user, change
`User=` in the unit.

Useful commands:

```bash
journalctl -fu anya-recorder
```

To try it without the service, run `python3 pi/recorder/recorder.py --dir ~/recordings`.

There is **no login**. Anyone on the same network can press Start or Stop, so
don't expose port 8080 to the internet.

## Camera settings

They are in the `CAMERA_ARGS` list at the top of `recorder.py`. They are the
original command's settings with these changes:

| Change | Why |
|---|---|
| `--timeout 0` | Record until Stop, where `15000` would stop after 15 s. |
| `--codec libav --libav-format mpegts` | The Pi 5 has no hardware H.264 encoder, so it encodes through libav (software). MPEG-TS survives a crash. |
| `--bitrate 16000000` | 16 Mbps, about 7 GB an hour. The software encoder's default is too low for a tennis ball. |
| `-n` | No preview window, since the Pi runs headless. |

The page refuses to start with less than 5 GB free.

## First checks on the Pi

1. **The Pi 5 keeps up with 1080p60.** Record for 2 minutes, then count the frames:

   ```bash
   ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 /srv/anya/recordings/<name>.mp4
   ```

   You should get about 6000 (120 s × 50). Also look in `<name>.log` for dropped-frame
   warnings. If frames drop, lower `--framerate` to 30 or `--bitrate`, or add
   `"--libav-video-codec-opts", "preset=ultrafast"`.
2. **The camera is the HQ Camera (IMX477).** It has a manual-focus lens, so there is
   no `--autofocus-mode`; focus the lens by hand. It reaches at most 50 fps at
   1080p (its 2028x1080 mode), so the frame rate is 50, not 60.

## Tests (on a laptop, needs ffmpeg)

```bash
python -m pytest pi/recorder/test_recorder.py
```

`fake_rpicam_vid.py` stands in for the camera. To click through the page on a
laptop, run:

```bash
RPICAM_VID=pi/recorder/fake_rpicam_vid.py python3 pi/recorder/recorder.py --dir /tmp/rec
```

## Next: processing at the end of the day

Not built yet. Recordings are named `YYYY-MM-DD_HHMMSS.mp4` in one folder, so a
nightly job (or a "Process today" button on this page) can run anya2 over the day's
files. The Pi worker in PR #11 reads only DJI files, so it would need a small
input path for these.
