"""
app.py — US Open themed Rally Detector GUI
"""

import os
import sys
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QLineEdit, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

sys.path.insert(0, str(Path(__file__).parent))

from rally_detector import collect_rally_segments
from utilities import create_highlights_ffmpeg

NAVY  = "#001E62"
GOLD  = "#FECB00"
WHITE = "#FFFFFF"


class _Worker(QThread):
    progress = pyqtSignal(int, int)   # (current, total)
    finished = pyqtSignal(str, int)   # (output_path, n_segments)
    error    = pyqtSignal(str)

    def __init__(self, video_path, output_path, start_frame):
        super().__init__()
        self.video_path  = video_path
        self.output_path = output_path
        self.start_frame = start_frame
        self._stopped    = False

    def run(self):
        try:
            def _cb(current, total):
                if not self._stopped:
                    self.progress.emit(current, total)

            segments = collect_rally_segments(
                self.video_path,
                headless=True,
                start_frame=self.start_frame,
                progress_cb=_cb,
            )
            if self._stopped:
                return

            create_highlights_ffmpeg(
                self.video_path,
                [(s, e) for s, e, _ in segments],
                self.output_path,
            )
            self.finished.emit(self.output_path, len(segments))
        except Exception as ex:
            self.error.emit(str(ex))

    def stop(self):
        self._stopped = True


class RallyDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rally Detector")
        self.setMinimumSize(560, 640)
        self.resize(580, 660)
        self._worker      = None
        self._output_path = ""
        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _setup_ui(self):
        self.setStyleSheet(f"QMainWindow, QWidget {{ background: {NAVY}; }}")

        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(36, 36, 36, 36)
        lay.setSpacing(18)

        lay.addLayout(self._logo_row())
        lay.addWidget(self._divider())

        lay.addWidget(self._label("INPUT VIDEO"))
        lay.addLayout(self._file_row("video"))

        lay.addWidget(self._label("OUTPUT VIDEO  (auto-generated if blank)"))
        lay.addLayout(self._file_row("output"))

        lay.addStretch()

        self._detect_btn = self._make_detect_btn()
        lay.addWidget(self._detect_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: rgba(255,255,255,0.12); border-radius: 4px; border: none; }}
            QProgressBar::chunk {{ background: {GOLD}; border-radius: 4px; }}
        """)
        lay.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: rgba(255,255,255,0.55); font-size: 12px;")
        lay.addWidget(self._status)

        self._result_panel = self._make_result_panel()
        self._result_panel.setVisible(False)
        lay.addWidget(self._result_panel)

    def _logo_row(self):
        row = QHBoxLayout()
        ball = QLabel("●")
        ball.setStyleSheet(f"color: {GOLD}; font-size: 34px; padding-right: 4px;")
        ball.setFixedWidth(46)
        col = QVBoxLayout()
        col.setSpacing(2)
        title = QLabel("Rally Detector")
        title.setStyleSheet(f"color: {WHITE}; font-size: 20px; font-weight: 700; letter-spacing: 0.03em;")
        sub = QLabel("US OPEN EDITION")
        sub.setStyleSheet(f"color: {GOLD}; font-size: 10px; letter-spacing: 0.16em;")
        col.addWidget(title)
        col.addWidget(sub)
        row.addWidget(ball)
        row.addLayout(col)
        row.addStretch()
        return row

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background: {GOLD}; border: none; max-height: 1px; min-height: 1px;")
        return line

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: rgba(255,255,255,0.60); font-size: 10px; font-weight: 700; letter-spacing: 0.08em;")
        return lbl

    def _file_row(self, kind):
        row = QHBoxLayout()
        row.setSpacing(8)
        edit = QLineEdit()
        edit.setStyleSheet(f"""
            QLineEdit {{
                background: rgba(255,255,255,0.07);
                border: 1px solid rgba(255,255,255,0.16);
                border-radius: 6px;
                color: {WHITE};
                padding: 9px 12px;
                font-size: 13px;
            }}
            QLineEdit:focus {{ border: 1px solid {GOLD}; }}
        """)
        btn = QPushButton("Browse")
        btn.setFixedWidth(88)
        btn.setStyleSheet(self._ghost_btn_css())

        if kind == "video":
            edit.setPlaceholderText("Select a tennis match video…")
            self._video_edit = edit
            btn.clicked.connect(self._browse_video)
            edit.textChanged.connect(self._refresh_detect_btn)
        else:
            edit.setPlaceholderText("match_rallies.mp4")
            self._output_edit = edit
            btn.clicked.connect(self._browse_output)

        row.addWidget(edit)
        row.addWidget(btn)
        return row

    def _make_detect_btn(self):
        btn = QPushButton("DETECT RALLIES")
        btn.setFixedHeight(52)
        btn.setEnabled(False)
        btn.setStyleSheet(self._detect_css(enabled=False))
        btn.clicked.connect(self._on_detect)
        return btn

    def _make_result_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background: rgba(0,0,0,0.28); border-radius: 8px;")
        row = QHBoxLayout(panel)
        row.setContentsMargins(16, 10, 16, 10)
        self._result_path_lbl = QLabel("")
        self._result_path_lbl.setStyleSheet("color: rgba(255,255,255,0.65); font-size: 12px; font-family: monospace;")
        self._result_path_lbl.setWordWrap(True)
        self._result_count_lbl = QLabel("")
        self._result_count_lbl.setFixedHeight(24)
        self._result_count_lbl.setStyleSheet(
            "background: #27ae60; color: white; font-size: 11px; font-weight: 700; "
            "padding: 2px 10px; border-radius: 4px;"
        )
        open_btn = QPushButton("Open Folder")
        open_btn.setFixedHeight(28)
        open_btn.setStyleSheet(self._ghost_btn_css())
        open_btn.clicked.connect(self._open_output_folder)
        row.addWidget(self._result_path_lbl, 1)
        row.addWidget(self._result_count_lbl)
        row.addWidget(open_btn)
        return panel

    # ── Style helpers ──────────────────────────────────────────────────────

    def _detect_css(self, enabled):
        if enabled:
            return f"""
                QPushButton {{
                    background: {GOLD}; color: {NAVY};
                    font-size: 14px; font-weight: 700; letter-spacing: 0.07em;
                    border-radius: 8px; border: none;
                }}
                QPushButton:hover  {{ background: #FFD933; }}
                QPushButton:pressed {{ background: #E6B800; }}
            """
        return f"""
            QPushButton {{
                background: rgba(254,203,0,0.22); color: rgba(254,203,0,0.38);
                font-size: 14px; font-weight: 700; letter-spacing: 0.07em;
                border-radius: 8px; border: none;
            }}
        """

    def _ghost_btn_css(self):
        return f"""
            QPushButton {{
                background: transparent;
                border: 1px solid rgba(255,255,255,0.28);
                border-radius: 6px;
                color: rgba(255,255,255,0.75);
                padding: 6px 12px;
                font-size: 12px;
            }}
            QPushButton:hover {{ border-color: {GOLD}; color: {GOLD}; }}
        """

    # ── Slots ──────────────────────────────────────────────────────────────

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.m4v);;All Files (*)"
        )
        if path:
            self._video_edit.setText(path)

    def _browse_output(self):
        video = self._video_edit.text().strip()
        default_dir = str(Path(video).parent) if video else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Rally Video As",
            os.path.join(default_dir, "rallies.mp4"),
            "MP4 Video (*.mp4)"
        )
        if path:
            self._output_edit.setText(path)

    def _refresh_detect_btn(self, text=None):
        enabled = bool(self._video_edit.text().strip()) and self._worker is None
        self._detect_btn.setEnabled(enabled)
        self._detect_btn.setStyleSheet(self._detect_css(enabled=enabled))

    def _on_detect(self):
        video = self._video_edit.text().strip()
        if not video or not os.path.isfile(video):
            self._set_status("Please select a valid video file.", error=True)
            return

        output = self._output_edit.text().strip()
        if not output:
            stem   = Path(video).stem
            output = str(Path(video).parent / f"{stem}_rallies.mp4")
        self._output_path = output

        self._result_panel.setVisible(False)
        self._progress.setValue(0)
        self._set_status("Initializing…")
        self._detect_btn.setEnabled(False)
        self._detect_btn.setStyleSheet(self._detect_css(enabled=False))
        self._detect_btn.setText("DETECTING…")

        self._worker = _Worker(video, output, start_frame=0)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_progress(self, current, total):
        if total > 0:
            self._progress.setValue(int(100 * current / total))
            self._set_status(f"Analyzing frame {current:,} / {total:,}")

    def _on_finished(self, output_path, n_segments):
        self._worker = None
        self._progress.setValue(100)
        noun = "rally" if n_segments == 1 else "rallies"
        self._set_status(f"Done — {n_segments} {noun} detected")
        self._detect_btn.setText("DETECT RALLIES")
        self._refresh_detect_btn()

        self._result_path_lbl.setText(os.path.basename(output_path))
        self._result_count_lbl.setText(f"{n_segments} RALL{'Y' if n_segments == 1 else 'IES'}")
        self._result_panel.setVisible(True)

    def _on_error(self, msg):
        self._worker = None
        self._progress.setValue(0)
        self._set_status(f"Error: {msg}", error=True)
        self._detect_btn.setText("DETECT RALLIES")
        self._refresh_detect_btn()

    def _set_status(self, text, error=False):
        color = "#e74c3c" if error else "rgba(255,255,255,0.55)"
        self._status.setStyleSheet(f"color: {color}; font-size: 12px;")
        self._status.setText(text)

    def _open_output_folder(self):
        folder = str(Path(self._output_path).parent)
        if sys.platform == "darwin":
            subprocess.run(["open", folder])
        elif sys.platform == "win32":
            subprocess.run(["explorer", folder])
        else:
            subprocess.run(["xdg-open", folder])


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = QFont("Helvetica Neue", 11)
    app.setFont(font)

    window = RallyDetectorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
