"""
S2 — can QMediaPlayer stream an https MP4?

The gate screen's preview is streamed rather than bundled: the installed app is
already ~2 GB, and a video inside the DMG cannot be changed without a signed,
notarized release. That decision is only safe if Qt can actually play a remote
file from inside the shipped app.

It is not obvious that it can. scoreboard_tab.py only ever opens LOCAL files,
which is a different code path in every Qt multimedia backend. If this fails,
the fallback is bundling a ~15 MB clip via rally_app.spec's `datas`, and that
has to be decided before the spec is finalised.

    python3 spikes/s2_stream_video.py                 # default test file
    python3 spikes/s2_stream_video.py <url>           # a candidate of your own
    python3 spikes/s2_stream_video.py --headless      # no window, exit code only

Exit code 0 means it reached PlayingState and the position advanced. Anything
else is a failure with the reason printed.

IMPORTANT: running this from source only clears half the question. The other
half -- whether it still works inside the notarized .app, where the Qt
multimedia plugins are relocated by PyInstaller and the hardened runtime is in
force -- can only be answered by building and running the bundle. Do that
before treating S2 as passed.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

# A 10-second 360p h.264/AAC MP4 served over https with `Accept-Ranges: bytes`
# -- shaped like the preview will be, and about the same order of size.
#
# (The obvious candidate, commondatastorage.googleapis.com/gtv-videos-bucket,
# now returns 403; so does w3schools. Both are quoted all over the internet as
# open test files and neither is any more. If this URL rots too, any
# https-served progressive MP4 will do -- check with
# `curl -sIL <url>` that it answers 200 with Content-Type: video/mp4 and
# Accept-Ranges: bytes before blaming Qt.)
DEFAULT_URL = (
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/"
    "Big_Buck_Bunny_360_10s_1MB.mp4"
)

TIMEOUT_MS = 25_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default=DEFAULT_URL)
    ap.add_argument("--headless", action="store_true",
                    help="don't show a window; just report")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    state = {"played": False, "positions": [], "error": None, "status": []}

    window = QWidget()
    window.resize(880, 520)
    lay = QVBoxLayout(window)
    caption = QLabel("connecting…")
    lay.addWidget(caption)
    video = QVideoWidget()
    lay.addWidget(video, 1)

    player = QMediaPlayer()
    audio = QAudioOutput()
    audio.setMuted(True)
    player.setAudioOutput(audio)
    player.setVideoOutput(video)

    def on_error(err, msg=""):
        state["error"] = f"{err} {msg}".strip()
        print(f"  error: {state['error']}")

    def on_status(status):
        state["status"].append(status.name)
        print(f"  status: {status.name}")

    def on_state(st):
        print(f"  state:  {st.name}")
        if st == QMediaPlayer.PlaybackState.PlayingState:
            state["played"] = True

    def on_position(pos):
        if pos and (not state["positions"] or pos != state["positions"][-1]):
            state["positions"].append(pos)
            caption.setText(f"playing — {pos/1000:.1f}s")

    player.errorOccurred.connect(on_error)
    player.mediaStatusChanged.connect(on_status)
    player.playbackStateChanged.connect(on_state)
    player.positionChanged.connect(on_position)

    print(f"S2: streaming {args.url}")
    player.setSource(QUrl(args.url))
    player.play()

    if not args.headless:
        window.show()

    def finish():
        player.stop()
        # Two conditions, because reaching PlayingState alone is not proof:
        # a backend can report Playing on a stream that never delivers a
        # frame. An advancing position is what says data is actually arriving.
        advanced = len(state["positions"]) >= 2
        ok = state["played"] and advanced and state["error"] is None

        print()
        print(f"  reached PlayingState : {state['played']}")
        print(f"  position advanced    : {advanced} "
              f"({len(state['positions'])} distinct positions, "
              f"max {max(state['positions'], default=0)/1000:.1f}s)")
        print(f"  error                : {state['error'] or 'none'}")
        print()
        frozen = getattr(sys, "frozen", False)
        where = "packaged bundle" if frozen else "source tree"
        print(f"S2 {'PASS' if ok else 'FAIL'} ({where})")
        if ok and not frozen:
            print("Still to confirm: the same URL from inside the packaged, signed .app.")
        elif ok:
            print("Qt's bundled ffmpeg reaches the network from inside the bundle.")
        app.exit(0 if ok else 1)

    QTimer.singleShot(TIMEOUT_MS, finish)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
