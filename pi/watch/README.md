# Watch remote (LilyGo T-Watch 2020 → Pi, Bluetooth)

Start and Stop the court recorder from a **T-Watch 2020** (V1, V2 or V3) on your wrist, with no phone
and no Wi-Fi. The watch talks Bluetooth LE to the Pi; the Pi passes each press to the
recorder's own web page API, so everything after Stop (the `.mp4`, the YouTube
uploads, the highlights) happens exactly as it does from the page.

```
T-Watch 2020 ──BLE──▶ anya-watch (bridge.py) ──HTTP 127.0.0.1:8080──▶ anya-recorder ──▶ anya-worker
```

- **Hold the round button for one second** to start (lime) or stop (red). A plain tap
  does nothing, so it can't go off by accident during play.
- The watch shows the recording time, **Saving** while the Pi finishes the file, and
  any error the page would show (out of disk, camera gone).
- **If another program has the camera,** the watch asks *Camera in use … Close it and
  start recording?*, as the page does. **Close it** closes that program and starts.
- The page and the watch show the same state. Starting on one and stopping on the
  other works.
- Recordings started from the watch have no player name: the YouTube title ends with
  `[youtube] session_name` from `/srv/anya/config.toml`.
- **Only a paired watch can control the Pi.** The link must be authenticated
  (passkey pairing) and the Pi must have marked the watch trusted. Phones or other
  devices nearby can see the Pi advertising, but they can't press anything.
- **Battery.** The screen goes off after 20 s and the watch sleeps a minute later. The
  side button wakes it (a short press also turns the screen off); it reconnects in a second or two. While the screen is
  off, the first touch only wakes it.

## Install on the Pi

`pi/install.sh` installs and starts `anya-watch` along with the rest (it adds the
`anya` user to the `bluetooth` group and installs `dbus-next`). The Pi 5's built-in
Bluetooth is all it needs. Check it's running:

```bash
journalctl -fu anya-watch
```

It logs `advertising a5c0de00-…` and the recorder's state.

## Flash the watch (on the Mac)

Install PlatformIO once (`brew install platformio`), plug the watch in by USB, then
flash the build for its version (printed on the back):

```bash
cd pi/watch/firmware && pio run -e twatch2020-v3 -t upload
```

Use `-e twatch2020-v2` or `-e twatch2020-v1` for the older ones. The versions use
different pins, and the chip can't tell which watch it's in: the wrong build shows a
dark screen, and flashing the right one fixes it. The watch appears on the Mac as
`/dev/cu.usbserial-…` (its USB-serial chip); PlatformIO finds it by itself.

This replaces LilyGo's demo firmware. It's an ESP32 watch, so the S3's library and
board settings don't apply.

## Pair (once per watch)

On the Pi, over SSH:

```bash
sudo /opt/anya/venv/bin/python /opt/anya/src/pi/watch/bridge.py pair
```

The Pi is pairable for two minutes. Wake the watch near it: it connects and shows six
digits ("Type on the Pi: 123 456"). Type them at the prompt. The script then marks the
watch trusted, and the watch shows the Start button.

A watch that isn't paired shows **Not paired** and the digits; it keeps trying every
few seconds, so you can start `pair` after waking it.

```bash
sudo /opt/anya/venv/bin/python /opt/anya/src/pi/watch/bridge.py list
```

```bash
sudo /opt/anya/venv/bin/python /opt/anya/src/pi/watch/bridge.py unpair AA:BB:CC:DD:EE:FF
```

After re-flashing a watch with its storage erased, or swapping watches, `unpair` the
old entry first, then `pair` again. If the Pi forgets a watch, the watch drops its own
bond by itself and asks to pair again.

## The Bluetooth service

One primary service, `a5c0de00-7e9b-4c39-9a35-0c0a0f7e1a00`, advertised by UUID (no
name: 31 bytes don't hold both).

| Characteristic | UUID | Access | Content |
|---|---|---|---|
| command | `a5c0de01-…` | write, authenticated | `start`, `start-force`, `stop` |
| status | `a5c0de02-…` | read + notify, authenticated | `{"s":"recording","t":312,"max":5400,"saving":0}` plus `err` and `busy` when set |

`s` is the recorder's state (`idle`, `recording`, `finishing`) or `offline` when the
recorder doesn't answer. Status is sent on every change and every 10 s; the watch
counts the seconds itself in between. `busy` (e.g. `["ffmpeg 1234"]`) answers a
`start` when another program has the camera. The details are in `bridge.py`.

## Troubleshooting

- **Watch stuck on "Looking for the Pi…".** `journalctl -u anya-watch` should show
  `advertising`. `bluetoothctl show` should say `Powered: yes`; if Bluetooth is
  blocked, `sudo rfkill unblock bluetooth`.
- **"Recorder off" on the watch.** The bridge can't reach `127.0.0.1:8080`:
  `systemctl status anya-recorder`.
- **Writes refused in the log** (`not paired with bridge.py pair`). The device bonded
  some other way (Just Works, or before `pair` ran) and isn't trusted. `unpair` it,
  then `pair`.
- **Testing without the watch.** The nRF Connect app on a phone can see and use the
  service. Pair the phone with `bridge.py pair` (the phone shows the digits and
  you type them on the Pi, as with the watch), then write `start` (as text) to the
  command characteristic.

## Tests (on a laptop, needs ffmpeg)

```bash
python -m pytest pi/watch
```

They cover the status bytes and the commands, and drive the real recorder (with its
fake camera) through the same calls the watch makes, including the camera-busy
prompt. Bluetooth itself needs the Pi.
