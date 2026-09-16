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
from pipeline import cancel as _cancel
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
    from pipeline.rally_reel.reel import ANALYSIS_SIZE, N_STAGES
    from pipeline.utilities import init_court as _ensure_court

    def ensure_court(video):
        _ensure_court(video, analysis_size=ANALYSIS_SIZE)
else:
    from pipeline.anya2.config import Anya2Config as EngineConfig
    from pipeline.anya2.run import build_reel, ensure_court, N_STAGES
    from pipeline.anya2.court import ANALYSIS_SIZE

from pipeline.utilities import probe_video
from pipeline import join as _join

from applog import log_path, logger
from background import SleepBlocker, notify
from preflight import ensure_ffmpeg
from theme import (BLACK, YELLOW, WHITE, danger_btn_css, ghost_btn_css,
                   primary_btn_css, label_css, line_edit_css)


class _JoinWorker(QThread):
    """Remuxes several GoPro chapter files into one, off the GUI thread.

    Its own thread and not part of `_Worker` because of an ordering
    constraint that cannot be relaxed: court calibration opens a cv2 window
    and so must run on the MAIN thread, but the corners have to be clicked on
    the JOINED file, because that is the one every later stage indexes into.
    So the sequence is join (here) -> calibrate (main thread, in
    `_begin_render`) -> detect (`_Worker`).

    Signal naming follows `_Worker`'s, and for the same reason -- see the note
    on `render_*` there.
    """
    stage        = pyqtSignal(int, int, str, float)
    join_done    = pyqtSignal(str)
    join_failed  = pyqtSignal(str)
    join_stopped = pyqtSignal()

    def __init__(self, videos):
        super().__init__()
        self.videos = list(videos)

    def run(self):
        try:
            out = _join.resolve_input(
                self.videos,
                on_progress=lambda f: self.stage.emit(
                    1, N_STAGES, f"Joining {len(self.videos)} video files", f))
            self.join_done.emit(out)
        except _cancel.Cancelled:
            logger().info("Highlight Reel join cancelled by the user")
            self.join_stopped.emit()
        except Exception as ex:
            logger().exception("Highlight Reel join failed")
            self.join_failed.emit(str(ex))


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
    stage            = pyqtSignal(int, int, str, float)  # (i, n, label, frac; -1 = busy)
    render_done      = pyqtSignal(str, int)              # (output_path, n_segments)
    render_failed    = pyqtSignal(str)
    render_cancelled = pyqtSignal()

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
                self.render_cancelled.emit()
                return
            self.render_done.emit(out or self.output_path, len(segments))
        except _cancel.Cancelled:
            # Not an error: the tester asked for this. Logged at info so a
            # truncated app.log still explains why the run has no reel at the
            # end of it.
            logger().info("Highlight Reel render cancelled by the user")
            self.render_cancelled.emit()
        except Exception as ex:
            logger().exception("Highlight Reel render failed")
            self.render_failed.emit(str(ex))

    def stop(self):
        """Ask the pipeline to stop at its next check-in.

        The flag alone is not enough and never was: build_reel does not consult
        it, so before pipeline.cancel existed this only suppressed the RESULT
        while the job ran to completion in the background -- the machine stayed
        pinned for the rest of a ten-minute render and the tester could not
        start another one. `_cancel.request()` is what actually stops the work;
        `_stopped` still suppresses the progress ticks so the bar does not
        twitch forward while the pipeline unwinds.
        """
        self._stopped = True
        _cancel.request()


class HighlightReelTab(QWidget):
    """Pick a match video, click the four court corners once, get a reel of
    just the rallies.
    """

    def __init__(self):
        super().__init__()
        self._worker      = None
        self._join_worker = None
        # The chapter files the tester picked, in recording order.  Empty
        # means "whatever is typed in the line edit", which is still the
        # normal single-video case.
        self._videos      = []
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
        # GoPro chapters beyond the first.  The line edit keeps showing a real
        # path so it stays typeable, and this says -- in RECORDING order, which
        # is not necessarily the order they were picked in -- what else is
        # going in.  A silently reordered match is a forty-minute mistake that
        # looks like a working run, so the order is shown rather than trusted.
        self._chapters_lbl = QLabel("")
        self._chapters_lbl.setWordWrap(True)
        self._chapters_lbl.setStyleSheet(
            "color: rgba(255,255,255,0.5); font-size: 11px;")
        self._chapters_lbl.setVisible(False)
        lay.addWidget(self._chapters_lbl)

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

        lay.addLayout(self._action_row())

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
            edit.textEdited.connect(self._forget_chapters)
        else:
            edit.setPlaceholderText("match_rally_reel.mp4")
            self._output_edit = edit
            btn.clicked.connect(self._browse_output)

        row.addWidget(edit)
        row.addWidget(btn)
        return row

    def _action_row(self):
        """The primary action, with Cancel beside it once a run is going.

        Cancel is built now and hidden rather than added when a job starts:
        appearing mid-run would reflow the progress bar and status line under
        a tester's cursor at exactly the moment they are watching them.
        """
        row = QHBoxLayout()
        row.setSpacing(10)

        self._detect_btn = QPushButton(self._action_text())
        self._detect_btn.setFixedHeight(52)
        self._detect_btn.setEnabled(False)
        self._detect_btn.setStyleSheet(primary_btn_css(enabled=False))
        self._detect_btn.clicked.connect(self._on_detect)
        row.addWidget(self._detect_btn, 1)

        self._cancel_btn = QPushButton("CANCEL")
        self._cancel_btn.setFixedHeight(52)
        self._cancel_btn.setFixedWidth(132)
        self._cancel_btn.setStyleSheet(danger_btn_css())
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.setToolTip(
            "Stop this run. Nothing is saved, but the calibration already done "
            "for this video is reused next time.")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        row.addWidget(self._cancel_btn)

        return row

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
        """Pick one video, or every chapter of one GoPro recording.

        Multi-select rather than a second "add another file" control: the
        chapters of one match are always picked together out of one folder,
        and the dialog already does that in one gesture.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video (or all chapters of one recording)", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.m4v);;All Files (*)"
        )
        if not paths:
            return
        self._videos = _join.order_inputs(paths)
        # setText, not textEdited -- `_forget_chapters` is wired to the latter
        # so that TYPING a path drops the chapter list while this does not.
        self._video_edit.setText(self._videos[0])
        self._show_chapters()

    def _show_chapters(self):
        extra = self._videos[1:]
        self._chapters_lbl.setVisible(bool(extra))
        if extra:
            self._chapters_lbl.setText(
                "+ " + ", ".join(os.path.basename(p) for p in extra)
                + f"  ({len(self._videos)} files, joined in this order)")

    def _forget_chapters(self, _text=None):
        """A hand-typed path replaces the whole selection, chapters included."""
        self._videos = []
        self._chapters_lbl.setVisible(False)

    def _input_videos(self):
        """What to run on: the picked chapters, or whatever is in the edit."""
        if self._videos:
            return list(self._videos)
        one = self._video_edit.text().strip()
        return [one] if one else []

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
        enabled = (bool(self._video_edit.text().strip())
                   and self._worker is None and self._join_worker is None)
        self._detect_btn.setEnabled(enabled)
        self._detect_btn.setStyleSheet(primary_btn_css(enabled=enabled))

    def _on_detect(self):
        if not ensure_ffmpeg(self):
            return

        videos = self._input_videos()
        bad = [v for v in videos if not os.path.isfile(v)]
        if not videos or bad:
            self._set_status(
                f"Cannot find {os.path.basename(bad[0])}." if bad
                else "Please select a valid video file.", error=True)
            return

        # Everything that is named after "the input" is named after the FIRST
        # chapter, never the join: the join lives in tmp_anya, which this run
        # deletes on its way out, so defaulting the reel into it would throw
        # away the only thing the run was for.
        first = videos[0]
        output = self._output_edit.text().strip()
        if not output:
            output = str(Path(first).parent / f"{Path(first).stem}_rally_reel.mp4")
        self._output_path = output

        # Every file this run creates -- the join, court/exclusion
        # calibration, pose detections, tracks, each detector's events, the
        # reel JSON, and the scratch segments the cut passes through -- goes
        # into tmp_anya beside the input, reused across runs on the same video
        # if it is still there (a prior run only leaves it behind when "keep
        # files" was checked) and created fresh otherwise.  set_work_dir must
        # be called on the MAIN thread, and before the join, because the
        # joined file goes into it too.
        self._tmp_anya = str(Path(first).parent / "tmp_anya")
        _workdir.set_work_dir(self._tmp_anya)

        self._result_panel.setVisible(False)
        self._progress.setValue(0)
        self._set_status("Initializing…")
        self._set_estimate(videos)
        self._detect_btn.setEnabled(False)
        self._detect_btn.setStyleSheet(primary_btn_css(enabled=False))
        self._detect_btn.setText("WORKING…")
        # Cleared at the START of a run, not the end of the last one: a
        # cancelled job can still be unwinding when the tester starts the next,
        # and clearing on the way out would race that unwind and leave the new
        # run cancelling itself. See pipeline.cancel.clear.
        _cancel.clear()
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("CANCEL")
        self._cancel_btn.setVisible(True)

        # Keep the machine awake for the duration of the (possibly long) job so
        # a backgrounded window keeps processing instead of sleeping mid-run.
        self._sleep_blocker.start()

        if len(videos) == 1:
            # No join, no second thread, no behaviour change whatsoever.
            self._begin_render(first)
            return
        self._set_status(f"Joining {len(videos)} video files…")
        self._join_worker = _JoinWorker(videos)
        self._join_worker.stage.connect(self._on_stage)
        self._join_worker.join_done.connect(self._begin_render)
        self._join_worker.join_failed.connect(self._on_error)
        self._join_worker.join_stopped.connect(self._on_cancelled)
        # As with _Worker: both the deleteLater and the drop of our own
        # reference hang off QThread's built-in `finished`, never off the
        # custom result signals, which are emitted from inside run().
        self._join_worker.finished.connect(self._join_worker.deleteLater)
        self._join_worker.finished.connect(self._release_join_worker)
        self._join_worker.start()

    def _release_join_worker(self):
        self._join_worker = None
        self._refresh_detect_btn()

    def _begin_render(self, video):
        """Calibrate on the MAIN thread, then start detection.

        Reached directly for a single video and from `_JoinWorker`'s
        `join_done` for several; either way `video` is now one file and
        nothing below here knows about chapters.  Calibration is here rather
        than inside the worker because `init_court` opens a cv2 window.
        """
        try:
            self._set_status("Court calibration…")
            ensure_court(video)
        except _cancel.Cancelled:
            self._on_cancelled()
            return
        except Exception as ex:
            self._on_error(f"Setup failed: {ex}")
            return
        self._set_status("Initializing…")

        self._worker = _Worker(video, self._output_path, cfg=self._cfg)
        self._worker.stage.connect(self._on_stage)
        self._worker.render_done.connect(self._on_finished)
        self._worker.render_failed.connect(self._on_error)
        self._worker.render_cancelled.connect(self._on_cancelled)
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
        self._cancel_btn.setVisible(False)
        notify("Anya Tennis — reel complete",
               f"{n_segments} {noun} · {os.path.basename(output_path)}")
        self._detect_btn.setText(self._action_text())
        self._refresh_detect_btn()

        self._result_path_lbl.setText(os.path.basename(output_path))
        self._result_count_lbl.setText(badge)
        self._result_panel.setVisible(True)

    def _on_cancel(self):
        """Cancel pressed. Ask, then wait — the thread reports back itself.

        Deliberately not a confirmation dialog: the button is already the
        second, quieter one in the row, and a modal on top of a ten-minute
        render that the tester has decided to abandon is one more thing in
        their way. What it does do is disable itself immediately, because the
        pipeline stops at its next check-in rather than instantly and a Cancel
        that still looked pressable would read as ignored.
        """
        if self._worker is None and self._join_worker is None:
            return
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("CANCELLING…")
        self._set_status("Cancelling — finishing the current step…")
        if self._worker is not None:
            self._worker.stop()
        else:
            # The join has no _stopped flag to suppress its result with: it
            # either produces the file or it does not, and pipeline.join
            # deletes its part-file and raises Cancelled on the way out.
            _cancel.request()

    def _on_cancelled(self):
        # See _on_finished: the handle is released on `finished`, not here.
        # A cancelled run leaves no reel, but tmp_anya still honours the
        # checkbox — a tester who cancelled BECAUSE the run looked wrong is
        # exactly the one who may want the interim files.
        self._cleanup_tmp_anya()
        self._sleep_blocker.stop()
        self._estimate.setVisible(False)
        self._progress.setValue(0)
        self._set_status("Cancelled — no reel was written.")
        self._cancel_btn.setVisible(False)
        self._detect_btn.setText(self._action_text())
        self._refresh_detect_btn()

    def _on_error(self, msg):
        # See _on_finished: the handle is released on `finished`, not here.
        self._cleanup_tmp_anya()
        self._sleep_blocker.stop()
        self._estimate.setVisible(False)
        self._progress.setValue(0)
        self._set_status(f"Error: {msg}  (details in {log_path()})", error=True)
        self._cancel_btn.setVisible(False)
        self._detect_btn.setText(self._action_text())
        self._refresh_detect_btn()

    def _set_status(self, text, error=False):
        color = "#e74c3c" if error else "rgba(255,255,255,0.55)"
        self._status.setStyleSheet(f"color: {color}; font-size: 12px;")
        self._status.setText(text)

    def _set_estimate(self, videos):
        # Best-effort — a probe failure here shouldn't block the run itself,
        # build_reel will surface any real problem with the file.  Summed
        # across chapters: the estimate is shown BEFORE the join exists, and
        # what a tester is waiting on is the whole match, not its first 4 GB.
        if isinstance(videos, (str, os.PathLike)):
            videos = [videos]
        try:
            duration_sec = sum(probe_video(v)["duration_sec"] for v in videos)
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
