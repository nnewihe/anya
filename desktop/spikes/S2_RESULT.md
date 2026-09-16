# S2 — streaming the preview video: **PASS**

*(One of three Phase 0 spikes — see README.md in this directory.)*

Run 7 September 2026, macOS 15.6 (Darwin 24.6.0), arm64, PyQt6 with Qt
multimedia's ffmpeg backend (ffmpeg 7.1.3, LGPL).

## Question

The gate screen streams its preview rather than bundling it, because the
installed app is already ~2 GB and a video inside the DMG cannot be changed
without a signed, notarized release. That is only safe if Qt can play a remote
file *from inside the shipped app*. `scoreboard_tab.py` only ever opens local
files, which is a different code path in every Qt multimedia backend, so
nothing we already ship answered this.

If it had failed, the fallback was bundling a ~15 MB clip via
`rally_app.spec`'s `datas` — a change to the spec and the DMG size, which is
why this was worth settling before the spec was finalised.

## Result

| | reached Playing | position advanced | error |
|---|---|---|---|
| Source tree | yes | 196 samples, full 10.0 s | none |
| PyInstaller bundle, unsigned | yes | 194 samples, full 10.0 s | none |
| PyInstaller bundle, **Developer ID + hardened runtime** | yes | 198 samples, full 10.0 s | none |

The signed run used the real `Developer ID Application: Anderson Nnewihe
(696S9GCN96)` identity and the app's own `entitlements.plist`, signed
inside-out with the same `find`/`codesign` sweep as `build_macos.sh`.
`codesign -d` confirmed `flags=0x10000(runtime)` — the hardened runtime was
actually in force, not merely requested.

Reaching `PlayingState` alone is not proof: a backend can report Playing on a
stream that never delivers a frame. The test also requires the position to
advance, which is what says bytes are actually arriving.

## What this settles

- **Keep streaming the preview.** No change to `rally_app.spec`, no growth in
  the DMG.
- **No extra TLS libraries are needed.** The bundle carries `libavcodec`,
  `libavformat`, `libavutil` and `libswresample` and no `libssl`/`libcrypto`:
  Qt's ffmpeg uses macOS SecureTransport, a system framework. Nothing has to be
  added to `datas` or to the signing sweep.
- **The hardened runtime does not block it**, consistent with the analysis in
  the plan: the four `com.apple.security.cs.*` keys are hardened-runtime keys,
  the app is not sandboxed, and network access is an App Sandbox concern.
  `entitlements.plist` needs no change.

## Residual, deliberately not run

**Notarization and stapling were not performed**, because submitting a binary
to Apple is an outward-facing action and this was a throwaway test bundle.

That residual is small and worth naming precisely: notarization is a Gatekeeper
*trust* signal attached after the fact. It does not alter what the process may
do at runtime — the hardened runtime does, and that is what was tested above,
in force. The remaining unknown is therefore first-launch Gatekeeper behaviour
on a downloaded DMG, which is already exercised by every existing release, not
anything about media or the network.

**Still worth doing once, on the real build:** open the gate screen on a clean
Mac from the notarized DMG and watch the preview play. Item 3 of the release
checklist in `desktop/README.md`.

## Reproducing

```bash
cd desktop
python3 spikes/s2_stream_video.py --headless          # source
python3 spikes/s2_stream_video.py <some-other-url>    # a candidate of your own
```

Note the default URL is a public test file that can rot: the two most commonly
cited ones (`commondatastorage.googleapis.com/gtv-videos-bucket` and
`w3schools`) both return **403** now. Before blaming Qt, check the URL with
`curl -sIL <url>` and confirm it answers 200 with `Content-Type: video/mp4` and
`Accept-Ranges: bytes`.
