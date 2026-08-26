"""highlight_tab.py — "Highlight Reel" tab: cuts dead time from a raw match
video via ``pipeline.rally_reel``.

This is the desktop app's original (and only, pre-Scoreboard-tab) feature,
extracted verbatim out of app.py into its own QWidget so it can sit inside a
QTabWidget alongside scoreboard_tab.ScoreboardTab. Behavior is unchanged.
"""

import os
import shutil
import sys
import subprocess
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QLineEdit, QFrame,
    QCheckBox,
)
from pipeline import workdir as _workdir
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# ── which detection engine ───────────────────────────────────────────────
# anya2 is the primary path: three independent detectors (near serve, far
# serve, point end) on a shared player-tracking substrate, assembled by an
# orchestrator that applies tennis structure.  It is scored against
# ground_truth.json over 13 clips -- see pipeline/anya2/README.md.
#
# The legacy `rally_reel` remains selectable because it is what shipped, and
# because anya2 is newer inside the app than it is on the command line.  Set
# ANYA_ENGINE=legacy to fall back; nothing else changes.
#
# Both expose the same call:
#     build_reel(video, output_path=..., cfg=..., on_progress=cb) -> (segments, out)
# and the same one-time, main-thread court calibration.
import os as _os

ENGINE = _os.environ.get("ANYA_ENGINE", "anya2").strip().lower()

if ENGINE == "legacy":
    from pipeline.rally_reel import ReelConfig as EngineConfig, build_reel
    from pipeline.rally_reel.reel import ANALYSIS_SIZE
    from pipeline.utilities import init_court as _ensure_court

    def ensure_court(video):
        _ensure_court(video, analysis_size=ANALYSIS_SIZE)
else:
    from pipeline.anya2.config import Anya2Config as EngineConfig
    from pipeline.anya2.run import build_reel, ensure_court
    from pipeline.anya2.court import ANALYSIS_SIZE

from pipeline.utilities import probe_video

from applog import log_path, logger
from background import SleepBlocker, notify
from preflight import ensure_ffmpeg
from theme import BLACK, YELLOW, WHITE, ghost_btn_css, primary_btn_css, label_css, line_edit_css


class _Worker(QThread):
    """Runs the detection engine off the GUI thread (see ENGINE).

    `stage` carries rally_reel's own stage reporting straight through, so the
    UI never has to know the stage list — add or reorder a stage in the
    pipeline and this reflects it with no change here.

    Custom result signals are named `render_*`, NOT `finished`/`error` —
    QThread already has a built-in `finished` signal that Qt emits only
    after the OS thread has actually stopped, and `deleteLater` must be
    wired to *that* one. A same-named custom signal shadows the built-in
    one; emitting it manually from inside run() races the real thread
    teardown and can delete the QThread object while it's still alive,
    which crashes the app with "QThread: Destroyed while thread is still
    running".
    """
    stage         = pyqtSignal(int, int, str, float)  # (i, n, label, frac; -1 = busy)
    render_done   = pyqtSignal(str, int)               # (output_path, n_segments)
    render_failed = pyqtSignal(str)

    def __init__(self, video_path, output_path, cfg=None):
        super().__init__()
        self.video_path  = video_path
        self.output_path = output_path
        self.cfg         = cfg or EngineConfig()
        self._stopped    = False

    def run(self):
        try:
            def _on_progress(i, n, label, frac):
                if not self._stopped:
                    # Qt signals are strongly typed; -1.0 stands in for the
                    # "indeterminate" None a stage sends when it cannot
                    # report sub-progress.
                    self.stage.emit(i, n, label, -1.0 if frac is None else frac)

            # Court calibration already ran on the main thread (init_court
            # opens a cv2 window, unsafe off it); build_reel's stage 0 call
            # hits the disk cache and is windowless here.
            segments, out = build_reel(
                self.video_path,
                cfg=self.cfg,
                output_path=self.output_path,
                on_progress=_on_progress,
            )
            if self._stopped:
                return
            self.render_done.emit(out or self.output_path, len(segments))
        except Exception as ex:
            logger().exception("Highlight Reel render failed")
            self.render_failed.emit(str(ex))

    def stop(self):
        self._stopped = True


class HighlightReelTab(QWidget):
    """Pick a match video, click the four court corners once, get a reel of
    just the rallies.
    """

    def __init__(self):
        super().__init__()
        self._worker      = None
        self._output_path = ""
        self._cfg         = EngineConfig()
        self._sleep_blocker = SleepBlocker()
        # The tmp_anya directory for the run IN PROGRESS (or just finished) --
        # set in _on_detect, used by _cleanup_tmp_anya to know what to remove.
        # None outside of a run.
        self._tmp_anya = None
        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(36, 30, 36, 36)
        lay.setSpacing(18)

        lay.addWidget(self._label("INPUT VIDEO"))
        lay.addLayout(self._file_row("video"))

        lay.addWidget(self._label("OUTPUT VIDEO  (auto-generated if blank)"))
        lay.addLayout(self._file_row("output"))

        # Every calibration/detection file this run produces goes into a
        # tmp_anya folder beside the input video (see pipeline.workdir) --
        # unchecked (discard) by default, since a tester who never needs to
        # look at them shouldn't accumulate gigabytes of npz/json per video.
        # Checking it is for diagnosing a bad reel: the cached detections and
        # the court/exclusion calibration survive for inspection or reuse.
        self._keep_files_checkbox = QCheckBox(
            "Keep calibration and interim files after processing")
        self._keep_files_checkbox.setChecked(False)
        self._keep_files_checkbox.setStyleSheet(
            "color: rgba(255,255,255,0.7); font-size: 12px;")
        lay.addWidget(self._keep_files_checkbox)

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
            QProgressBar::chunk {{ background: {YELLOW}; border-radius: 4px; }}
        """)
        lay.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: rgba(255,255,255,0.55); font-size: 12px;")
        lay.addWidget(self._status)

        # Set once at job start and left visible for the whole run — unlike
        # self._status (overwritten every stage tick), this is the one place
        # a tester can check "is this normal?" without reading the README.
        self._estimate = QLabel("")
        self._estimate.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._estimate.setStyleSheet("color: rgba(255,255,255,0.38); font-size: 11px;")
        self._estimate.setVisible(False)
        lay.addWidget(self._estimate)

        self._result_panel = self._make_result_panel()
        self._result_panel.setVisible(False)
        lay.addWidget(self._result_panel)

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(label_css())
        return lbl

    def _file_row(self, kind):
        row = QHBoxLayout()
        row.setSpacing(8)
        edit = QLineEdit()
        edit.setStyleSheet(line_edit_css())
        btn = QPushButton("Browse")
        btn.setFixedWidth(88)
        btn.setStyleSheet(ghost_btn_css())

        if kind == "video":
            edit.setPlaceholderText("Select a tennis match video…")
            self._video_edit = edit
            btn.clicked.connect(self._browse_video)
            edit.textChanged.connect(self._refresh_detect_btn)
        else:
            edit.setPlaceholderText("match_rally_reel.mp4")
            self._output_edit = edit
            btn.clicked.connect(self._browse_output)

        row.addWidget(edit)
        row.addWidget(btn)
        return row

    def _make_detect_btn(self):
        btn = QPushButton(self._action_text())
        btn.setFixedHeight(52)
        btn.setEnabled(False)
        btn.setStyleSheet(primary_btn_css(enabled=False))
        btn.clicked.connect(self._on_detect)
        return btn

    def _action_text(self):
        return "BUILD RALLY REEL"

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
            f"background: {YELLOW}; color: {BLACK}; font-size: 11px; font-weight: 700; "
            "padding: 2px 10px; border-radius: 4px;"
        )
        open_btn = QPushButton("Open Folder")
        open_btn.setFixedHeight(28)
        open_btn.setStyleSheet(ghost_btn_css())
        open_btn.clicked.connect(self._open_output_folder)
        row.addWidget(self._result_path_lbl, 1)
        row.addWidget(self._result_count_lbl)
        row.addWidget(open_btn)
        return panel

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
        default_name = "rally_reel.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Output Video As",
            os.path.join(default_dir, default_name),
            "MP4 Video (*.mp4)"
        )
        if path:
            self._output_edit.setText(path)

    def _refresh_detect_btn(self, text=None):
        enabled = bool(self._video_edit.text().strip()) and self._worker is None
        self._detect_btn.setEnabled(enabled)
        self._detect_btn.setStyleSheet(primary_btn_css(enabled=enabled))

    def _on_detect(self):
        if not ensure_ffmpeg(self):
            return

        video = self._video_edit.text().strip()
        if not video or not os.path.isfile(video):
            self._set_status("Please select a valid video file.", error=True)
            return

        output = self._output_edit.text().strip()
        if not output:
            output = str(Path(video).parent / f"{Path(video).stem}_rally_reel.mp4")
        self._output_path = output

        # Every file this run creates -- court/exclusion calibration, pose
        # detections, tracks, each detector's events, the reel JSON, and the
        # scratch segments the cut passes through -- goes into tmp_anya beside
        # the input, reused across runs on the same video if it is still
        # there (a prior run only leaves it behind when "keep files" was
        # checked) and created fresh otherwise.  set_work_dir must be called
        # on the MAIN thread, and before calibration, because init_court
        # writes the court cache into it too.
        self._tmp_anya = str(Path(video).parent / "tmp_anya")
        try:
            _workdir.set_work_dir(self._tmp_anya)
            self._set_status("Court calibration…")
            ensure_court(video)
        except Exception as ex:
            self._set_status(f"Setup failed: {ex}", error=True)
            _workdir.clear_work_dir()
            return

        self._result_panel.setVisible(False)
        self._progress.setValue(0)
        self._set_status("Initializing…")
        self._set_estimate(video)
        self._detect_btn.setEnabled(False)
        self._detect_btn.setStyleSheet(primary_btn_css(enabled=False))
        self._detect_btn.setText("WORKING…")

        # Keep the machine awake for the duration of the (possibly long) job so
        # a backgrounded window keeps processing instead of sleeping mid-run.
        self._sleep_blocker.start()

        self._worker = _Worker(video, output, cfg=self._cfg)
        self._worker.stage.connect(self._on_stage)
        self._worker.render_done.connect(self._on_finished)
        self._worker.render_failed.connect(self._on_error)
        # QThread's own built-in `finished` — fires only after the OS thread
        # has actually stopped, unlike our render_done/render_failed signals
        # which we emit manually from inside run(). BOTH the deleteLater and
        # the drop of our own reference must be tied to this one: whichever
        # happens first destroys the QThread, and doing that while run() is
        # still on the stack is a Qt fatal, not an exception.
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.finished.connect(self._release_worker)
        self._worker.start()

    def _release_worker(self):
        # `self._worker is None` is also what re-enables the button, so this
        # has to refresh it — the completion slot no longer can.
        self._worker = None
        self._refresh_detect_btn()

    def _on_stage(self, i, n, label, frac):
        # Map (stage, fraction-within-stage) onto one continuous bar, so a
        # long stage still shows movement instead of sitting at a step.
        if frac < 0:                      # stage cannot report sub-progress
            pct = int(100 * i / max(1, n))
            self._progress.setValue(pct)
            self._set_status(f"Stage {i}/{n} — {label}…")
        else:
            pct = int(100 * (i - 1 + frac) / max(1, n))
            self._progress.setValue(max(0, pct))
            self._set_status(f"Stage {i}/{n} — {label}  {frac:.0%}")

    def _cleanup_tmp_anya(self):
        """Honor the checkbox: discard tmp_anya unless the tester asked to
        keep it. Called after EVERY terminal state -- success or failure --
        so a crash mid-run does not leave the override pointed at a directory
        that is about to be deleted out from under the next run.
        """
        _workdir.clear_work_dir()
        d, self._tmp_anya = self._tmp_anya, None
        if d and not self._keep_files_checkbox.isChecked():
            shutil.rmtree(d, ignore_errors=True)

    def _on_finished(self, output_path, n_segments):
        # NOT `self._worker = None` — render_done is emitted from inside run(),
        # so the OS thread is usually still alive when this slot runs and
        # dropping the last reference here calls ~QThread on a running thread:
        # "QThread: Destroyed while thread is still running", qFatal, SIGABRT.
        # The app died right at the moment the reel completed. See
        # _release_worker.
        self._cleanup_tmp_anya()
        self._sleep_blocker.stop()
        self._estimate.setVisible(False)
        self._progress.setValue(100)
        noun = "rally" if n_segments == 1 else "rallies"
        badge = f"{n_segments} RALL{'Y' if n_segments == 1 else 'IES'}"
        self._set_status(f"Done — {n_segments} {noun}")
        notify("Anya Tennis — reel complete",
               f"{n_segments} {noun} · {os.path.basename(output_path)}")
        self._detect_btn.setText(self._action_text())
        self._refresh_detect_btn()

        self._result_path_lbl.setText(os.path.basename(output_path))
        self._result_count_lbl.setText(badge)
        self._result_panel.setVisible(True)

    def _on_error(self, msg):
        # See _on_finished: the handle is released on `finished`, not here.
        self._cleanup_tmp_anya()
        self._sleep_blocker.stop()
        self._estimate.setVisible(False)
        self._progress.setValue(0)
        self._set_status(f"Error: {msg}  (details in {log_path()})", error=True)
        self._detect_btn.setText(self._action_text())
        self._refresh_detect_btn()

    def _set_status(self, text, error=False):
        color = "#e74c3c" if error else "rgba(255,255,255,0.55)"
        self._status.setStyleSheet(f"color: {color}; font-size: 12px;")
        self._status.setText(text)

    def _set_estimate(self, video_path):
        # Best-effort — a probe failure here shouldn't block the run itself,
        # build_reel will surface any real problem with the file.
        try:
            duration_sec = probe_video(video_path)["duration_sec"]
        except Exception:
            self._estimate.setVisible(False)
            return
        mins = duration_sec / 60
        # 1.6x the clip length, measured end to end on a cold 7.0-min 4K clip
        # (Data/21, M4): 11m06s wall with only the court corners cached.  Was
        # 3x before the partial-telemetry passes — every stage now runs its own
        # decimated extraction off a shared 540p proxy, and the full-resolution
        # stage-1 pass is skipped outright.  Bump this if a stage stops being
        # fast-pathed; a low estimate reads as a hang.
        self._estimate.setText(
            f"Video is {mins:.0f} min — first run typically takes ≈{mins * 1.6:.0f} min. "
            "Cached reruns on this video are much faster."
        )
        self._estimate.setVisible(True)

    def _open_output_folder(self):
        folder = str(Path(self._output_path).parent)
        if sys.platform == "darwin":
            subprocess.run(["open", folder])
        elif sys.platform == "win32":
            subprocess.run(["explorer", folder])
        else:
            subprocess.run(["xdg-open", folder])
