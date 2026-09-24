# anya on a Raspberry Pi — Osmo Action 4 → dead-time-free reel

A Raspberry Pi 5 next to the court:

1. You plug in the DJI Osmo Action 4 (USB-C) after a session.
2. The Pi copies the new recordings off the camera. The camera is mounted read-only and nothing on it is deleted.
3. It runs the anya2 engine on each recording, one at a time.
4. The reel lands in a network share on the Pi.
5. Optionally, it uploads the reel to YouTube as unlisted.

```
camera ──USB──▶ udev ─▶ anya-ingest@  ─▶ /srv/anya/inbox/<id>/  +  state/jobs/<id>.json
                                                  │
                         anya-worker (always on) ◀┘  one job at a time
                             │  join chapters → site calibration → anya2 (NCNN pose)
                             ▼
                  /srv/anya/reels/2026-09-23_1800_tennis.mp4   (SMB share "anya-reels")
                  /srv/anya/reels/STATUS.txt                   (what the Pi is doing)
                             │  if [youtube] enabled
                             ▼
                  YouTube (unlisted) → reels/<name>.youtube.txt
```

## Hardware

- **Raspberry Pi 5** (4 GB works; the install sets up 4 GB of swap) with the **Active Cooler**.
  Processing pins all four cores for hours, and a Pi without a cooler throttles.
- **An SSD, not the SD card.** 4K footage is about 50 GB per hour. A multi-chapter
  recording briefly needs about twice that while it runs (the chapters are joined first).
  Use an NVMe HAT or a USB 3 SSD, and point `root` in the config at it if it isn't `/srv/anya`.
- The official 27 W supply (the Pi's USB ports share its power budget with the camera).

## Camera settings (Osmo Action 4)

| Setting | Use | Why |
|---|---|---|
| Resolution / aspect | **4K 16:9** | The analysis frame is 16:9. 4:3 footage is rejected (it would be squashed). The far player needs 4K. |
| Frame rate | **30 fps** (25 in PAL) | 60 works, but it doubles the decode for no gain. |
| Codec | **HEVC (H.265)** | The Pi 5 decodes HEVC in hardware and H.264 only in software. |
| Colour | **Normal**, 8-bit | D-Log M and HLG are 10-bit and are rejected. The pose model has never seen them. |
| Stabilisation | **Off** (RockSteady/HorizonSteady off) | It warps the view frame to frame, and the court calibration assumes a fixed camera. |
| Date/time | **Set it** (via the Mimo app) | Reel names and chapter grouping come from the file timestamps. |

## Install

On Raspberry Pi OS **Lite** (64-bit). The desktop edition's automounter competes with the ingest
unit for the camera.

```bash
git clone <this repo> ~/anya && cd ~/anya
```

```bash
sudo pi/install.sh
```

```bash
sudo smbpasswd -a anya
```

The first command gets the code. The second installs everything: code in `/opt/anya/src`, the venv,
`/srv/anya/`, the Samba share `anya-reels`, the udev rule and the worker service, and it exports
the NCNN pose model. The third sets the share password. Re-run `install.sh` after a `git pull` to update.

## One-time calibration (the site profile)

The camera doesn't move, so the court corners are clicked **once** and carried forward to every
recording. Each new recording is *registered* against the calibration frame, so a nudge of a few
pixels is corrected automatically. A knocked mount (more than 15 px) stops that job as
`needs_calibration`, so you don't get a confidently wrong reel.

On a laptop with this repo, using any recording from the mounted camera:

```bash
python -m pipeline.anya2.site save DJI_20260923180000_0001_D.MP4 site
```

Click the four **singles** corners in the window (bottom-left, bottom-right, top-right, top-left).
Then copy the folder to the Pi:

```bash
scp -r site pi@<pi>:/tmp/site
```

```bash
ssh pi@<pi> 'sudo rsync -a /tmp/site/ /srv/anya/site/ && sudo chown -R anya:anya /srv/anya/site'
```

If a job reports `needs_calibration` (the mount moved), redo this with a recording from the new
position, then run `anya-pi retry <id>` (see *Day to day*).

## Day to day

- **Plug the camera in** after the session, switched on. Copying starts on its own; unplug when
  `STATUS.txt` shows the recording as `pending`.
  *Not yet verified on a real Osmo Action 4:* that it appears as a USB drive (not MTP), and
  how it names and splits the chapters of a long recording. The first plug-in answers both; see
  Troubleshooting.
- **Reels:** open `\\<pi>\anya-reels` (Windows) or `smb://<pi>/anya-reels` (Mac).
  `STATUS.txt` there lists each job and its stage. Each reel has a `.segments.json` beside it with the kept time spans.
- **Logs:** `journalctl -fu anya-worker -u 'anya-ingest@*'`. The worker logs every stage's wall
  time and the final "× realtime", so the log is also the benchmark.
- **CLI:** everything below runs as the service user:

```bash
sudo -u anya env PYTHONPATH=/opt/anya/src:/opt/anya/src/pi /opt/anya/venv/bin/python -m anya_pi status
```

  Swap `status` for another command:

  | Command | What it does |
  |---|---|
  | `status` | Lists the queue. |
  | `retry <id>` | Re-queues a failed or needs-calibration job. |
  | `reupload <id>` | Uploads to YouTube again. |
  | `ingest /path` | Imports from a folder, e.g. an SD-card reader that didn't auto-trigger. |

- The copied originals in `inbox/` are deleted `keep_inbox_days` (14) after their reel is made.
  **The camera card is never touched.** Format it in the camera when you want the space back.

## Speed

Settings live in `/srv/anya/config.toml`; see `config.example.toml`. The defaults are the fast path:

| Setting | Default | Effect |
|---|---|---|
| `backend` | `"ncnn"` | Pose runs on NCNN, the ARM-optimised runtime. |
| `hwaccel` | `"drm"` | Hardware HEVC decode. |
| `single_decode` | `true` | One source decode builds both analysis proxies. |
| `copy_video` | `true` | The reel is cut straight from the original: same resolution, codec and quality, no re-encode (only the audio is re-encoded). Each point may start up to one keyframe interval (~1 s) early. |
| `scale_height` | `1080` | Used only with `copy_video = false`: re-encode with x264 at this height. |

**Expect roughly 2–4× the recording's length** on a Pi 5 (a 1-hour session takes 2–4 hours). Pose
inference and decode set that floor. The worker logs the real figure per job. A Hailo AI HAT+
backend (`ANYA_POSE_BACKEND=hailo`) is the planned next step for near-real-time.

**About the NCNN pose model.** An NCNN export has a fixed input size, and **must** match the
rectangle PyTorch would infer at. One exported with spare padding changed detections measurably, so
that isn't allowed. The near model is exported at install. The far model depends on the court's
position in the frame, so it is exported on first use (about 10 s) and cached in `/srv/anya/models`.
To check an export against PyTorch on a real recording:

```bash
python -m pipeline.anya2.export_pose parity --backend ncnn <video>
```

## YouTube (optional, unlisted)

1. In [Google Cloud Console](https://console.cloud.google.com/):
   1. Create a project and enable **YouTube Data API v3**.
   2. Configure the **OAuth consent screen** (External) and **publish it ("In production")**.
      In *Testing*, the token expires after 7 days and uploads start failing.
   3. Create an **OAuth client ID** of type *Desktop app* and download `client_secret.json`.
2. On a laptop with a browser and this repo, create the token:

   ```bash
   pip install google-api-python-client google-auth-oauthlib
   ```

   ```bash
   PYTHONPATH=pi python -m anya_pi youtube-auth --client-secrets client_secret.json --out youtube_token.json
   ```

   Add `--playlists` to the `youtube-auth` command if you'll set `playlist_id`.
3. Copy the token to the Pi and make it private to the service user:

   ```bash
   scp youtube_token.json pi@<pi>:/tmp/
   ```

   ```bash
   ssh pi@<pi> 'sudo install -o anya -g anya -m 600 /tmp/youtube_token.json /srv/anya/state/ && rm /tmp/youtube_token.json'
   ```

4. Set `[youtube] enabled = true` in `/srv/anya/config.toml`, then restart the worker:

   ```bash
   sudo systemctl restart anya-worker
   ```

Uploads run after processing and retry on their own while the Pi has no network. Each finished
upload writes `<reel>.youtube.txt` with the link.

**Uploads may be forced private.** Google locks every upload from an **unaudited** API project to
**private**, whatever privacy was requested. The worker reads back what YouTube applied, and
`STATUS.txt` shows `(PRIVATE -- see README)` when this happens. Until the project passes the
[YouTube API compliance audit](https://support.google.com/youtube/contact/yt_api_form), set each
video to Unlisted in YouTube Studio. Quota is about 6 uploads a day by default.

## Troubleshooting

- **Nothing happens when the camera is plugged in.** Run `journalctl -u 'anya-ingest@*'`.
  - If there's no entry at all, the camera isn't presenting as a USB drive. Check `lsblk` and the camera's USB mode.
  - As a fallback, put its microSD card in a USB reader. That triggers the same path.
- **`unsupported`**: the recording was 4:3 or 10-bit. Fix the camera settings.
- **Stuck `running` after a power cut**: nothing to do. The worker resumes it on start, from
  anya2's per-stage caches.
