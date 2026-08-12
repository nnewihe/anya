"""scoreboard_tab.py — "Scoreboard" tab: tag point winners against a raw
match video (from scratch, or seeded from pipeline.rally_reel's already-
detected point boundaries), then render a scored highlight video.

Port of src/scoreboard/src/App.tsx (a separate, standalone reference
project) from a browser tagging app to PyQt6. Feature 2 (tag from scratch)
and feature 3 (start from Highlight-Reel-tab segments, adjust, assign
winners) are the same screen — they differ only in how `self._points` gets
seeded; see `_on_import_segments` vs. manual `S`/`A`/`B` tagging.
"""

import os
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import QPointF, QSize, QThread, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QPolygonF, QShortcut
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSlider,
    QStackedWidget, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from pipeline.scoreboard_reel import (
    MatchFormat, PointTag, ProjectState, describe_score, import_segments_as_points,
    load_tags, render_scoreboard_video, replay_match, save_tags, segments_path_for,
    tags_path_for,
)
from applog import log_path, logger
from background import SleepBlocker, notify
from preflight import ensure_ffmpeg
from scoreboard_widget import ScoreboardPreview
from theme import BLACK, TEXT_DIM, WHITE, YELLOW, YELLOW_HOVER, ghost_btn_css, label_css, line_edit_css, primary_btn_css

COL_IDX, COL_START, COL_END, COL_WINNER, COL_SCORE = range(5)


def _fmt_time(s: float) -> str:
    if s is None or s != s:  # NaN guard
        return "0:00.0"
    m = int(s // 60)
    sec = s - m * 60
    return f"{m}:{sec:04.1f}"


def _parse_time(text: str, fallback: float) -> float:
    text = text.strip()
    try:
        if ":" in text:
            m, sec = text.split(":", 1)
            return float(m) * 60 + float(sec)
        return float(text)
    except ValueError:
        return fallback


_CHECK_GREEN = "#2ECC71"  # not a brand color (theme.py's black/yellow/sky) —
                          # a plain "selection confirmed" affordance, so it's
                          # kept local rather than added to the shared palette.
_check_icon_cache: Optional[QIcon] = None


def _check_icon() -> QIcon:
    """A small green checkmark, drawn on the fly (no asset file) and cached
    — built lazily since QPixmap/QIcon need a live QApplication, which
    doesn't exist yet at module import time.
    """
    global _check_icon_cache
    if _check_icon_cache is None:
        pix = QPixmap(16, 16)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(_CHECK_GREEN))
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPolyline(QPolygonF([QPointF(3, 8.5), QPointF(6.5, 12), QPointF(13, 3)]))
        painter.end()
        _check_icon_cache = QIcon(pix)
    return _check_icon_cache


class _RenderWorker(QThread):
    """Runs pipeline.scoreboard_reel.render_scoreboard_video off the GUI
    thread — same shape as highlight_tab._Worker.

    Custom result signals are named `render_*`, NOT `finished`/`error` —
    QThread already has a built-in `finished` signal that Qt emits only
    after the OS thread has actually stopped, and `deleteLater` must be
    wired to *that* one. A same-named custom signal shadows the built-in
    one; emitting it manually from inside run() races the real thread
    teardown and can delete the QThread object while it's still alive,
    which crashes the app with "QThread: Destroyed while thread is still
    running".
    """
    stage         = pyqtSignal(int, int, str, float)
    render_done   = pyqtSignal(str)
    render_failed = pyqtSignal(str)

    def __init__(self, video_path, points, fmt, names, output_path):
        super().__init__()
        self.video_path  = video_path
        self.points      = points
        self.fmt         = fmt
        self.names       = names
        self.output_path = output_path

    def run(self):
        try:
            def _on_progress(i, n, label, frac):
                self.stage.emit(i, n, label, -1.0 if frac is None else frac)

            out = render_scoreboard_video(
                self.video_path, self.points, self.fmt, self.names,
                self.output_path, on_progress=_on_progress,
            )
            self.render_done.emit(out)
        except Exception as ex:
            logger().exception("Scoreboard render failed")
            self.render_failed.emit(str(ex))


class ScoreboardTab(QWidget):
    def __init__(self):
        super().__init__()
        self._video_path: str = ""
        self._points: List[PointTag] = []
        self._names = {"a": "Player A", "b": "Player B"}
        self._format = MatchFormat()
        self._pending_start: Optional[float] = None
        self._render_worker: Optional[_RenderWorker] = None
        self._sleep_blocker = SleepBlocker()
        self._suspend_table_signal = False

        # Point-review mode: the video plays ONE point's clip [start, end] at
        # a time (stopping at the end, not looping) instead of the whole
        # match, so the user watches, adjusts boundaries if needed, and
        # confirms a winner before moving on. `_current_point` is tracked by
        # object identity (not table row index), since nudging re-sorts
        # `_points` and would otherwise silently point at the wrong point.
        self._current_point: Optional[PointTag] = None
        self._review_mode: bool = False

        self._setup_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._install_shortcuts()
        self._set_mode("manual")
        self._refresh_all()

    # ── UI construction ────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 10, 20, 16)
        root.setSpacing(8)

        # Load Video / Import segments / video name live in the QTabWidget's
        # corner (same row as the HIGHLIGHT REEL / SCOREBOARD tab labels),
        # not here — app.py pulls this via `self.load_row_widget` and wires
        # it up as a corner widget, shown only while this tab is active.
        self.load_row_widget = self._build_load_row_widget()

        # Built eagerly (not on first dialog open) so its fields exist for
        # _apply_format_to_combos()/_name_a_edit etc. regardless of whether
        # the user has opened Match Setup yet; not parented into any visible
        # layout until _open_modal() puts it in the dialog.
        self._match_setup_body = self._build_match_setup_body()

        # Two columns, side by side:
        #   left  — video, with its transport/tag/review controls directly
        #           below it (see _video_area()). Gets the majority of the
        #           width and the tab's FULL height (nothing else shares
        #           this column vertically), so this is where "maximize the
        #           video" actually comes from.
        #   right — Match Setup/Keyboard Shortcuts row, the scoreboard,
        #           the points list, Render Scoreboard Video, then
        #           Export/Import tags.json (see _side_column()).
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.addWidget(self._video_area())
        h_splitter.addWidget(self._side_column())
        h_splitter.setStretchFactor(0, 3)
        h_splitter.setStretchFactor(1, 1)
        # QSplitter panes are collapsible by default — it'll happily shrink
        # the side column below its actual minimumSizeHint (badge + standing
        # + table + render button + IO row all need real width), which
        # doesn't clip cleanly, it makes everything overlap. Locking this
        # pane's true computed minimum as a hard floor is what "video gets
        # max space, controls never overlap" actually requires; any squeeze
        # from a narrow window comes out of the video's share, not by
        # compressing the side column's own contents into each other.
        h_splitter.setCollapsible(1, False)
        h_splitter.setSizes([900, 340])
        root.addWidget(h_splitter, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(8)
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: rgba(255,255,255,0.12); border-radius: 4px; border: none; }}
            QProgressBar::chunk {{ background: {YELLOW}; border-radius: 4px; }}
        """)
        root.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: rgba(255,255,255,0.55); font-size: 12px;")
        root.addWidget(self._status)

    def _build_load_row_widget(self):
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        # A little breathing room from the tab labels this row now sits
        # beside (it's installed as the QTabWidget's corner widget in
        # app.py) rather than the row butting right up against them.
        row.addSpacing(24)
        load_btn = QPushButton("Load Video")
        load_btn.setStyleSheet(ghost_btn_css())
        load_btn.clicked.connect(self._on_load_video)
        self._import_btn = QPushButton("Import segments from Highlight Reel")
        self._import_btn.setStyleSheet(ghost_btn_css())
        self._import_btn.setEnabled(False)
        self._import_btn.clicked.connect(self._on_import_segments)
        self._video_lbl = QLabel("No video loaded")
        self._video_lbl.setStyleSheet("color: rgba(255,255,255,0.65); font-size: 12px; font-family: monospace;")
        row.addWidget(load_btn)
        row.addWidget(self._import_btn)
        row.addWidget(self._video_lbl)
        return w

    def _video_area(self):
        """Just the video — no overlay. Controls live in _control_bar(),
        added below it (not on top): a video surface can render through a
        native OS surface that always paints above ordinary Qt widgets
        regardless of stacking order, so anything meant to be seen reliably
        has to sit outside the video's own rect, not on top of it.
        """
        container = QWidget()
        container.setMinimumHeight(320)
        lay = QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self._video_widget = QVideoWidget()
        # Deliberately NOT styled via QSS: QVideoWidget renders frames through
        # its own video-sink surface, not QWidget::paintEvent, and any
        # setStyleSheet() call on it (even just a background color) forces Qt
        # to paint its background through the style-sheet engine on every
        # paint — which can cover the native video surface entirely, showing
        # a black/blank widget instead of the video. WA_StyledBackground is
        # explicitly turned off too, since the app-wide `QWidget { background
        # ... }` rule in app.py would otherwise still cascade to it by type.
        self._video_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        lay.addWidget(self._video_widget, 1)

        self._player = QMediaPlayer()
        self._audio = QAudioOutput()
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)
        self._player.positionChanged.connect(self._on_position_changed)

        lay.addWidget(self._control_bar())
        return container

    def _preview_widget(self):
        self._scoreboard_preview = ScoreboardPreview()
        # Opaque (matches the burned-in overlay look) but boxed and capped in
        # width so it reads as a badge, not a full-width bar.
        self._scoreboard_preview.setStyleSheet(f"border: 1px solid rgba(255,255,255,0.15); border-radius: 6px;")
        self._scoreboard_preview.setMaximumWidth(360)
        return self._scoreboard_preview

    def _control_bar(self):
        """The floating bar on the video has two pages, switched by the mode
        toggle at its top:

        - REVIEW POINTS (default whenever points exist): plays one point's
          clip at a time — [start, end], once, stopping at the end rather
          than looping — with Prev/Next navigation, fine start/end nudges,
          and A/B winner buttons that confirm the point and auto-advance.
        - TAG NEW POINT: the original free-scrub S/A/B/Z flow, for tagging a
          point from scratch (or adding one the detector missed).
        """
        bar = QFrame()
        bar.setStyleSheet("QFrame { background: rgba(0,0,0,0.62); border-radius: 10px; }")
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(18, 12, 18, 12)
        lay.setSpacing(8)

        lay.addLayout(self._mode_toggle_row())

        self._control_stack = QStackedWidget()
        self._control_stack.addWidget(self._review_page())  # index 0
        self._control_stack.addWidget(self._manual_page())  # index 1
        lay.addWidget(self._control_stack)
        return bar

    def _mode_toggle_row(self):
        row = QHBoxLayout()
        self._review_mode_btn = QPushButton("REVIEW POINTS")
        self._manual_mode_btn = QPushButton("TAG NEW POINT")
        for b in (self._review_mode_btn, self._manual_mode_btn):
            b.setCheckable(True)
            b.setFixedHeight(26)
        self._review_mode_btn.clicked.connect(lambda: self._set_mode("review"))
        self._manual_mode_btn.clicked.connect(lambda: self._set_mode("manual"))
        self._restyle_mode_buttons()
        row.addWidget(self._review_mode_btn)
        row.addWidget(self._manual_mode_btn)
        row.addStretch()
        return row

    def _restyle_mode_buttons(self):
        self._review_mode_btn.setStyleSheet(primary_btn_css(enabled=self._review_mode))
        self._manual_mode_btn.setStyleSheet(primary_btn_css(enabled=not self._review_mode))

    def _set_mode(self, mode: str):
        self._review_mode = (mode == "review")
        self._control_stack.setCurrentIndex(0 if self._review_mode else 1)
        self._review_mode_btn.setChecked(self._review_mode)
        self._manual_mode_btn.setChecked(not self._review_mode)
        self._restyle_mode_buttons()

        if self._review_mode:
            if self._current_point is None and self._points:
                self._current_point = self._first_pending() or self._points[0]
            if self._current_point is not None:
                self._player.setPosition(int(self._current_point.start * 1000))
                self._player.play()
        else:
            self._player.pause()
            self._pending_start = None
            self._pending_lbl.setText("No point in progress. Press S at the moment a point begins.")
        # Not just _sync_review_ui(): the table's "currently reviewed row"
        # highlight also depends on `_current_point`, which may have just
        # changed above — a full refresh keeps the table in sync too.
        self._refresh_all()

    def _review_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        nav = QHBoxLayout()
        prev_btn = QPushButton("‹ Prev  [,]")
        prev_btn.setStyleSheet(ghost_btn_css())
        prev_btn.clicked.connect(self._prev_point)
        self._point_counter_lbl = QLabel("No points yet")
        self._point_counter_lbl.setStyleSheet(f"color: {WHITE}; font-size: 13px; font-weight: 700;")
        next_btn = QPushButton("Next ›  [.]")
        next_btn.setStyleSheet(ghost_btn_css())
        next_btn.clicked.connect(self._next_point)
        insert_btn = QPushButton("+ Insert point at playhead")
        insert_btn.setStyleSheet(ghost_btn_css())
        insert_btn.clicked.connect(self._insert_point_at_playhead)
        delete_btn = QPushButton("Delete this point  [Del]")
        delete_btn.setStyleSheet(ghost_btn_css())
        delete_btn.clicked.connect(self._delete_current)
        nav.addWidget(prev_btn)
        nav.addWidget(self._point_counter_lbl, 1)
        nav.addWidget(next_btn)
        nav.addWidget(insert_btn)
        nav.addWidget(delete_btn)
        lay.addLayout(nav)

        lay.addLayout(self._seek_row())

        nudge = QHBoxLayout()
        # Asymmetric on purpose: you usually want to catch more of a point
        # by moving its start earlier and its end later, so those get more
        # (and bigger) options than the trim direction on each side.
        nudge.addWidget(self._dim_label("Start"))
        for label, delta in (("−5s", -5.0), ("−3s", -3.0), ("−1s", -1.0), ("+1s", 1.0)):
            b = QPushButton(label)
            b.setStyleSheet(ghost_btn_css())
            b.clicked.connect(lambda _, d=delta: self._nudge_current("start", d))
            nudge.addWidget(b)
        nudge.addSpacing(20)
        nudge.addWidget(self._dim_label("End"))
        for label, delta in (("−1s", -1.0), ("+1s", 1.0), ("+3s", 3.0), ("+5s", 5.0)):
            b = QPushButton(label)
            b.setStyleSheet(ghost_btn_css())
            b.clicked.connect(lambda _, d=delta: self._nudge_current("end", d))
            nudge.addWidget(b)
        nudge.addStretch()
        lay.addLayout(nudge)

        decide = QHBoxLayout()
        replay_btn = QPushButton("↺ Play again  [space]")
        replay_btn.setStyleSheet(ghost_btn_css())
        replay_btn.clicked.connect(self._replay_current)
        # Text/icon/style are set dynamically in _update_decide_buttons() —
        # player names once entered, a green check + yellow background once
        # selected — so these start blank rather than "A won [A]".
        self._decide_a_btn = QPushButton()
        self._decide_a_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)  # icon after text
        self._decide_a_btn.clicked.connect(lambda: self._assign_winner("A"))
        self._decide_b_btn = QPushButton()
        self._decide_b_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self._decide_b_btn.clicked.connect(lambda: self._assign_winner("B"))
        decide.addWidget(replay_btn)
        decide.addWidget(self._decide_a_btn)
        decide.addWidget(self._decide_b_btn)
        lay.addLayout(decide)

        return page

    def _seek_row(self):
        """The playhead: a draggable position slider scoped to the current
        point's [start, end] clip, plus a time readout and the split
        action. Uses a fixed 0-1000 normalized range instead of raw
        milliseconds so it never needs to be reconfigured as the reviewed
        point (and therefore its duration) changes — only the mapping
        between slider value and player position changes.
        """
        row = QHBoxLayout()
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ background: rgba(255,255,255,0.18); height: 4px; border-radius: 2px; }}
            QSlider::sub-page:horizontal {{ background: {YELLOW}; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {YELLOW}; width: 13px; height: 13px; margin: -5px 0; border-radius: 6px; }}
        """)
        self._seek_slider.sliderMoved.connect(self._on_seek_slider_moved)
        row.addWidget(self._seek_slider, 1)

        self._seek_time_lbl = QLabel("0:00.0 / 0:00.0")
        self._seek_time_lbl.setStyleSheet(f"color: {WHITE}; font-size: 11px; font-family: monospace;")
        row.addWidget(self._seek_time_lbl)

        self._split_btn = QPushButton("✂ Split at playhead  [X]")
        self._split_btn.setStyleSheet(ghost_btn_css())
        self._split_btn.clicked.connect(self._split_current_at_playhead)
        row.addWidget(self._split_btn)

        return row

    def _manual_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addLayout(self._transport_row())
        lay.addLayout(self._tag_row())

        self._pending_lbl = QLabel("No point in progress. Press S at the moment a point begins.")
        self._pending_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        lay.addWidget(self._pending_lbl)
        return page

    def _dim_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        return lbl

    def _side_column(self):
        """The right column, next to the video (see _setup_ui): Match Setup
        / Keyboard Shortcuts, the scoreboard, the points list, Render
        Scoreboard Video, then Export/Import tags.json — in that order, one
        row each.
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        lay.addLayout(self._setup_buttons_row())
        lay.addWidget(self._preview_widget())
        lay.addWidget(self._standing_block())
        lay.addWidget(self._points_table(), 1)
        lay.addLayout(self._render_row())
        lay.addLayout(self._io_row())

        self._result_panel = self._make_result_panel()
        self._result_panel.setVisible(False)
        lay.addWidget(self._result_panel)

        return w

    def _setup_buttons_row(self):
        row = QHBoxLayout()
        match_setup_btn = QPushButton("Match Setup")
        match_setup_btn.setStyleSheet(ghost_btn_css())
        match_setup_btn.clicked.connect(self._open_match_setup_dialog)
        shortcuts_btn = QPushButton("Keyboard Shortcuts")
        shortcuts_btn.setStyleSheet(ghost_btn_css())
        shortcuts_btn.clicked.connect(self._open_shortcuts_dialog)
        row.addWidget(match_setup_btn)
        row.addWidget(shortcuts_btn)
        row.addStretch()
        return row

    def _standing_block(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._standing_lbl = QLabel("")
        self._standing_lbl.setWordWrap(True)
        self._standing_lbl.setStyleSheet(f"color: {WHITE}; font-size: 13px;")
        lay.addWidget(self._standing_lbl)
        self._stat_lbl = QLabel("")
        self._stat_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        lay.addWidget(self._stat_lbl)
        return w

    def _io_row(self):
        row = QHBoxLayout()
        export_btn = QPushButton("Export tags.json")
        export_btn.setStyleSheet(ghost_btn_css())
        export_btn.clicked.connect(self._export_tags)
        import_btn = QPushButton("Import tags.json")
        import_btn.setStyleSheet(ghost_btn_css())
        import_btn.clicked.connect(self._import_tags_file)
        row.addWidget(export_btn)
        row.addWidget(import_btn)
        row.addStretch()
        return row

    def _transport_row(self):
        row = QHBoxLayout()
        play_btn = QPushButton("⏯ Play/Pause  [space]")
        play_btn.setStyleSheet(ghost_btn_css())
        play_btn.clicked.connect(self._toggle_play)
        back_btn = QPushButton("« 2s  [←]")
        back_btn.setStyleSheet(ghost_btn_css())
        back_btn.clicked.connect(lambda: self._seek(-2.0))
        fwd_btn = QPushButton("2s »  [→]")
        fwd_btn.setStyleSheet(ghost_btn_css())
        fwd_btn.clicked.connect(lambda: self._seek(2.0))
        slower_btn = QPushButton("–")
        slower_btn.setFixedWidth(28)
        slower_btn.setStyleSheet(ghost_btn_css())
        slower_btn.clicked.connect(lambda: self._change_rate(-0.25))
        faster_btn = QPushButton("+")
        faster_btn.setFixedWidth(28)
        faster_btn.setStyleSheet(ghost_btn_css())
        faster_btn.clicked.connect(lambda: self._change_rate(0.25))
        self._rate_lbl = QLabel("1.00×")
        self._rate_lbl.setStyleSheet(f"color: {WHITE}; font-size: 12px;")
        self._clock_lbl = QLabel("0:00.0")
        self._clock_lbl.setStyleSheet(f"color: {WHITE}; font-size: 12px; font-family: monospace;")

        for b in (play_btn, back_btn, fwd_btn):
            row.addWidget(b)
        row.addWidget(slower_btn)
        row.addWidget(self._rate_lbl)
        row.addWidget(faster_btn)
        row.addStretch()
        row.addWidget(self._clock_lbl)
        return row

    def _tag_row(self):
        row = QHBoxLayout()
        self._start_btn = QPushButton("▶ Point Start  [S]")
        self._start_btn.setStyleSheet(primary_btn_css(enabled=True))
        self._start_btn.clicked.connect(self._mark_start)
        self._a_btn = QPushButton("A won  [A]")
        self._a_btn.setStyleSheet(primary_btn_css(enabled=True))
        self._a_btn.clicked.connect(lambda: self._assign_winner("A"))
        self._b_btn = QPushButton("B won  [B]")
        self._b_btn.setStyleSheet(primary_btn_css(enabled=True))
        self._b_btn.clicked.connect(lambda: self._assign_winner("B"))
        undo_btn = QPushButton("Undo  [Z]")
        undo_btn.setStyleSheet(ghost_btn_css())
        undo_btn.clicked.connect(self._undo)
        row.addWidget(self._start_btn)
        row.addWidget(self._a_btn)
        row.addWidget(self._b_btn)
        row.addWidget(undo_btn)
        return row

    def _points_table(self):
        t = QTableWidget(0, 5)
        # A floor, not a target: QTableWidget scrolls its own rows
        # internally once content exceeds this, so it doesn't need (and
        # deliberately doesn't get) an unbounded natural minimum — that's
        # what was forcing the whole controls column too tall.
        t.setMinimumHeight(90)
        t.setHorizontalHeaderLabels(["#", "Start", "End", "Winner", "Score"])
        # Explicit narrow widths for everything but the score column: this
        # table now lives in the side column (~300-340px wide), not a
        # half-page-wide pane, and QHeaderView's default column widths
        # overflow that badly — Winner and Score would get clipped off the
        # right edge entirely rather than just look cramped.
        header = t.horizontalHeader()
        for col, width in ((COL_IDX, 24), (COL_START, 50), (COL_END, 50), (COL_WINNER, 58)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            t.setColumnWidth(col, width)
        header.setSectionResizeMode(COL_SCORE, QHeaderView.ResizeMode.Stretch)
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.setStyleSheet(f"""
            QTableWidget {{ background: rgba(255,255,255,0.04); color: {WHITE}; gridline-color: rgba(255,255,255,0.08); border: none; }}
            QHeaderView::section {{ background: rgba(255,255,255,0.08); color: {TEXT_DIM}; border: none; padding: 4px; font-size: 10px; }}
            QTableWidget::item:selected {{ background: rgba(232,255,61,0.20); }}
        """)
        t.itemChanged.connect(self._on_table_item_changed)
        # Clicking a row (any column except Winner, which has its own A/B
        # buttons) jumps point-review to that point — the table doubles as a
        # navigator into the review flow, not just a read-only summary.
        t.cellClicked.connect(self._on_cell_clicked)
        self._table = t
        return t

    def _on_cell_clicked(self, row: int, col: int):
        if col == COL_WINNER:
            return
        if 0 <= row < len(self._points):
            self._enter_review(self._points[row])

    # ── Match Setup / Keyboard Shortcuts popups ─────────────────────────────

    def _build_match_setup_body(self):
        """Built once (in _setup_ui, not lazily) so `self._name_a_edit` and
        the format combos always exist — _apply_format_to_combos() is called
        from _load_video()/_import_tags_file() regardless of whether the
        user has ever opened this dialog yet.
        """
        w = QWidget()
        form = QVBoxLayout(w)
        form.setSpacing(10)

        self._name_a_edit = self._name_field("Player A", "a")
        self._name_b_edit = self._name_field("Player B", "b")
        form.addLayout(self._field_row("Player A", self._name_a_edit))
        form.addLayout(self._field_row("Player B", self._name_b_edit))

        self._best_of_cb = QComboBox()
        self._best_of_cb.addItem("3 sets", 3)
        self._best_of_cb.addItem("5 sets", 5)
        self._best_of_cb.currentIndexChanged.connect(self._on_format_changed)
        form.addLayout(self._field_row("Best of", self._best_of_cb))

        self._set_to_cb = QComboBox()
        self._set_to_cb.addItem("First to 6 (standard)", 6)
        self._set_to_cb.addItem("First to 4 (short set)", 4)
        self._set_to_cb.addItem("First to 3", 3)
        self._set_to_cb.currentIndexChanged.connect(self._on_format_changed)
        form.addLayout(self._field_row("Games / set", self._set_to_cb))

        self._deuce_cb = QComboBox()
        self._deuce_cb.addItem("Advantage", False)
        self._deuce_cb.addItem("No-ad (sudden death)", True)
        self._deuce_cb.currentIndexChanged.connect(self._on_format_changed)
        form.addLayout(self._field_row("Deuce", self._deuce_cb))

        self._final_set_cb = QComboBox()
        self._final_set_cb.addItem("Tiebreak at 6-6", "tiebreak")
        self._final_set_cb.addItem("Super tiebreak (to 10)", "super")
        self._final_set_cb.addItem("Advantage set (win by 2)", "advantage")
        self._final_set_cb.currentIndexChanged.connect(self._on_format_changed)
        form.addLayout(self._field_row("Final set", self._final_set_cb))

        self._first_server_cb = QComboBox()
        self._first_server_cb.addItem("Player A", "A")
        self._first_server_cb.addItem("Player B", "B")
        self._first_server_cb.currentIndexChanged.connect(self._on_format_changed)
        form.addLayout(self._field_row("First server", self._first_server_cb))

        return w

    def _open_match_setup_dialog(self):
        self._apply_format_to_combos()
        self._open_modal("Match Setup", self._match_setup_body)

    _SHORTCUT_HELP = [
        # Kept in sync with _install_shortcuts() by hand — there are few
        # enough bindings that a generated table would be more machinery
        # than it's worth.
        ("Space", "Play / pause — or play the point again once it's finished"),
        ("A", "Record this player as the point winner, advance to the next point"),
        ("B", "Record the other player as the point winner, advance to the next point"),
        ("S", "Mark point start (Tag New Point mode)"),
        ("Z", "Undo the last tagged point (Tag New Point mode)"),
        (",", "Previous point"),
        (".", "Next point"),
        ("←  /  Shift+←", "Seek back 2s  /  0.1s"),
        ("→  /  Shift+→", "Seek forward 2s  /  0.1s"),
        ("[", "Slower playback"),
        ("]", "Faster playback"),
        ("Del  /  Backspace", "Delete the current point (Review mode), or undo the last "
                               "tagged point (Tag New Point mode)"),
        ("X", "Split the current point at the playhead"),
    ]

    def _build_shortcuts_body(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        for key, desc in self._SHORTCUT_HELP:
            row = QHBoxLayout()
            key_lbl = QLabel(key)
            key_lbl.setFixedWidth(150)
            key_lbl.setStyleSheet(f"color: {YELLOW}; font-size: 12px; font-weight: 700; font-family: monospace;")
            desc_lbl = QLabel(desc)
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet(f"color: {WHITE}; font-size: 12px;")
            row.addWidget(key_lbl)
            row.addWidget(desc_lbl, 1)
            lay.addLayout(row)
        return w

    def _open_shortcuts_dialog(self):
        self._open_modal("Keyboard Shortcuts", self._build_shortcuts_body())

    def _open_modal(self, title: str, body_widget: QWidget):
        """Centered, app-modal popup with an OKAY button. While open, the
        main window dims behind it (a semi-transparent overlay — Qt's own
        ApplicationModal already disables input to every other window, but
        doesn't dim anything on its own) and is fully disabled until OKAY.
        """
        top = self.window()
        overlay = QWidget(top)
        overlay.setStyleSheet("background: rgba(0,0,0,0.55);")
        overlay.setGeometry(top.rect())
        overlay.show()
        overlay.raise_()

        dialog = QDialog(top)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dialog.setStyleSheet(
            f"QDialog {{ background: {BLACK}; border: 1px solid rgba(255,255,255,0.18); border-radius: 10px; }}"
        )
        lay = QVBoxLayout(dialog)
        lay.setContentsMargins(28, 24, 28, 20)
        lay.setSpacing(18)
        lay.addWidget(self._label(title))
        lay.addWidget(body_widget)
        body_widget.show()

        ok_btn = QPushButton("OKAY")
        ok_btn.setStyleSheet(primary_btn_css(enabled=True))
        ok_btn.setFixedHeight(40)
        ok_btn.clicked.connect(dialog.accept)
        lay.addWidget(ok_btn)

        dialog.adjustSize()
        pg = top.geometry()
        dialog.move(pg.x() + (pg.width() - dialog.width()) // 2, pg.y() + (pg.height() - dialog.height()) // 2)

        dialog.exec()

        # Rescue body_widget from the dialog before it's garbage-collected —
        # for Match Setup this is the one persistent, reused form (its
        # fields/state need to survive to the next time this opens); for
        # Keyboard Shortcuts it's a fresh throwaway widget, so this is just
        # harmless extra reparenting.
        body_widget.setParent(self)
        body_widget.hide()
        overlay.hide()
        overlay.deleteLater()
        dialog.deleteLater()

    def _name_field(self, placeholder, key):
        edit = QLineEdit(self._names[key])
        edit.setStyleSheet(line_edit_css())
        edit.textChanged.connect(lambda text, k=key: self._on_name_changed(k, text))
        return edit

    def _field_row(self, label_text, widget):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(90)
        lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        if isinstance(widget, QComboBox):
            widget.setStyleSheet(f"""
                QComboBox {{ background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.16);
                             border-radius: 6px; color: {WHITE}; padding: 6px 10px; font-size: 12px; }}
            """)
        row.addWidget(lbl)
        row.addWidget(widget, 1)
        return row

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(label_css())
        return lbl

    def _render_row(self):
        row = QHBoxLayout()
        self._render_btn = QPushButton("RENDER SCOREBOARD VIDEO")
        self._render_btn.setFixedHeight(48)
        self._render_btn.setEnabled(False)
        self._render_btn.setStyleSheet(primary_btn_css(enabled=False))
        self._render_btn.clicked.connect(self._on_render)
        row.addWidget(self._render_btn)
        return row

    def _make_result_panel(self):
        panel = QFrame()
        panel.setStyleSheet("background: rgba(0,0,0,0.28); border-radius: 8px;")
        row = QHBoxLayout(panel)
        row.setContentsMargins(12, 8, 12, 8)
        self._result_path_lbl = QLabel("")
        self._result_path_lbl.setStyleSheet("color: rgba(255,255,255,0.65); font-size: 11px; font-family: monospace;")
        self._result_path_lbl.setWordWrap(True)
        open_btn = QPushButton("Open Folder")
        open_btn.setStyleSheet(ghost_btn_css())
        open_btn.clicked.connect(self._open_output_folder)
        row.addWidget(self._result_path_lbl, 1)
        row.addWidget(open_btn)
        return panel

    # ── Shortcuts ──────────────────────────────────────────────────────────

    def _install_shortcuts(self):
        bindings = [
            ("S", self._mark_start), ("A", lambda: self._assign_winner("A")),
            ("B", lambda: self._assign_winner("B")), ("Z", self._undo),
            ("Space", self._toggle_play),
            (",", self._prev_point), (".", self._next_point),
            ("Left", lambda: self._seek(-2.0)), ("Shift+Left", lambda: self._seek(-0.1)),
            ("Right", lambda: self._seek(2.0)), ("Shift+Right", lambda: self._seek(0.1)),
            ("[", lambda: self._change_rate(-0.25)), ("]", lambda: self._change_rate(0.25)),
            ("Del", self._delete_current), ("Backspace", self._delete_current),
            ("X", self._split_current_at_playhead),
        ]
        for key, slot in bindings:
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(self._guarded(slot))

    def _guarded(self, slot):
        """Wraps a shortcut handler so it no-ops while a text field has focus
        — mirrors App.tsx's `tag === 'INPUT'` guard.
        """
        def _wrapped():
            fw = self.focusWidget()
            if isinstance(fw, (QLineEdit,)) or (isinstance(fw, QComboBox)):
                return
            if isinstance(fw, QTableWidget) and fw.state() == QAbstractItemView.State.EditingState:
                return
            slot()
        return _wrapped

    # ── Video loading / playback ──────────────────────────────────────────

    def _on_load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.m4v);;All Files (*)"
        )
        if path:
            self._load_video(path)

    def _load_video(self, path: str):
        self._video_path = path
        self._video_lbl.setText(os.path.basename(path))
        self._player.setSource(QUrl.fromLocalFile(path))
        self._pending_start = None
        self._points = []
        self._current_point = None

        # Auto-restore a prior autosave for this video, if one exists.
        tpath = tags_path_for(path)
        if os.path.isfile(tpath):
            try:
                project = load_tags(tpath)
                self._names = project.names
                self._format = project.format
                self._points = project.points
                self._name_a_edit.setText(self._names.get("a", "Player A"))
                self._name_b_edit.setText(self._names.get("b", "Player B"))
                self._apply_format_to_combos()
            except Exception:
                pass

        self._import_btn.setEnabled(os.path.isfile(segments_path_for(path)))
        self._refresh_all()
        self._enter_review_default()

    def _apply_format_to_combos(self):
        f = self._format
        self._set_combo(self._best_of_cb, f.best_of)
        self._set_combo(self._set_to_cb, f.set_to)
        self._set_combo(self._deuce_cb, f.no_ad)
        self._set_combo(self._final_set_cb, f.final_set)
        self._set_combo(self._first_server_cb, f.first_server)

    def _set_combo(self, combo: QComboBox, value):
        idx = combo.findData(value)
        if idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _now(self) -> float:
        return self._player.position() / 1000.0

    def _seek(self, delta: float):
        ms = max(0, self._player.position() + int(delta * 1000))
        self._player.setPosition(ms)

    def _toggle_play(self):
        # In review mode, Space doubles as "play again" once the point's
        # clip has already played through to its end (rather than resuming
        # from the boundary, which would immediately re-trigger the
        # stop-at-end check in _on_position_changed and look like a no-op).
        if self._review_mode and self._current_point is not None:
            end_ms = int(self._current_point.end * 1000)
            if self._player.position() >= end_ms - 15:
                self._replay_current()
                return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _change_rate(self, delta: float):
        rate = max(0.25, min(4.0, round(self._player.playbackRate() + delta, 2)))
        self._player.setPlaybackRate(rate)
        self._rate_lbl.setText(f"{rate:.2f}×")

    def _on_position_changed(self, pos_ms: int):
        self._clock_lbl.setText(_fmt_time(pos_ms / 1000.0))
        if self._current_point is not None:
            self._sync_seek_slider(pos_ms)
        # Point-review clips play ONCE, stopping exactly at their end rather
        # than looping — the user re-watches via "Play again" [space] only
        # if they actually want to, instead of it replaying on its own.
        if (self._review_mode and self._current_point is not None
                and self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState):
            end_ms = int(self._current_point.end * 1000)
            if pos_ms >= end_ms:
                self._player.pause()
                self._player.setPosition(end_ms)

    def _sync_seek_slider(self, pos_ms: int):
        """Reflects the player's actual position on the playhead slider.
        `blockSignals` guards against feeding this back into
        _on_seek_slider_moved — QSlider.setValue() doesn't emit sliderMoved
        (that's user-drag-only) but does emit valueChanged, and nothing here
        listens to that, so this is belt-and-suspenders, not load-bearing.
        """
        p = self._current_point
        dur_ms = max(1, int((p.end - p.start) * 1000))
        frac = (pos_ms - int(p.start * 1000)) / dur_ms
        frac = max(0.0, min(1.0, frac))
        self._seek_slider.blockSignals(True)
        self._seek_slider.setValue(int(frac * 1000))
        self._seek_slider.blockSignals(False)
        self._seek_time_lbl.setText(
            f"{_fmt_time(max(0.0, pos_ms / 1000.0 - p.start))} / {_fmt_time(p.end - p.start)}"
        )

    def _on_seek_slider_moved(self, value: int):
        if self._current_point is None:
            return
        p = self._current_point
        target = p.start + (value / 1000.0) * (p.end - p.start)
        self._player.setPosition(int(target * 1000))

    # ── Point review (feature 3: seeded from Highlight Reel, or any tagged
    # points) ────────────────────────────────────────────────────────────────

    def _first_pending(self) -> Optional[PointTag]:
        for p in self._points:
            if p.winner is None:
                return p
        return None

    def _current_index(self) -> int:
        if self._current_point is None:
            return -1
        try:
            return self._points.index(self._current_point)
        except ValueError:
            return -1

    def _enter_review(self, point: PointTag):
        self._current_point = point
        if not self._review_mode:
            self._set_mode("review")  # _set_mode uses _current_point (already set) and refreshes
        else:
            self._player.setPosition(int(point.start * 1000))
            self._player.play()
            self._refresh_all()  # keeps the table's current-row highlight in sync too

    def _goto_point(self, index: int):
        if not self._points:
            return
        index = max(0, min(len(self._points) - 1, index))
        self._enter_review(self._points[index])

    def _next_point(self):
        self._goto_point(self._current_index() + 1)

    def _prev_point(self):
        self._goto_point(self._current_index() - 1)

    def _replay_current(self):
        if self._current_point is None:
            return
        self._player.setPosition(int(self._current_point.start * 1000))
        self._player.play()

    def _next_pending_after(self, point: PointTag) -> Optional[PointTag]:
        idx = self._points.index(point) if point in self._points else -1
        for p in self._points[idx + 1:]:
            if p.winner is None:
                return p
        for p in self._points:  # wrap around in case an earlier one is still pending
            if p.winner is None:
                return p
        return None

    def _enter_review_default(self):
        """Called after import/load/restore: jump straight into reviewing
        the first pending point, so the user lands in the point-based flow
        rather than a free-scrub view of the whole match.
        """
        if not self._points:
            self._current_point = None
            self._set_mode("manual")
            return
        self._enter_review(self._first_pending() or self._points[0])

    def _sync_review_ui(self):
        if not hasattr(self, "_point_counter_lbl"):
            return  # called before _control_bar() has finished building

        self._update_decide_buttons()

        # The playhead slider and split only make sense while an actual
        # point is being reviewed.
        has_current = self._current_point is not None
        self._split_btn.setEnabled(has_current)
        self._seek_slider.setEnabled(has_current)
        if not has_current:
            self._seek_slider.blockSignals(True)
            self._seek_slider.setValue(0)
            self._seek_slider.blockSignals(False)
            self._seek_time_lbl.setText("0:00.0 / 0:00.0")

        if not self._points:
            self._point_counter_lbl.setText("No points yet — import segments or tag one from scratch")
            return
        idx = self._current_index()
        if idx < 0:
            self._point_counter_lbl.setText(f"{len(self._points)} point(s) — click a row to review")
            return
        p = self._points[idx]
        status = "winner: " + p.winner if p.winner else "pending"
        self._point_counter_lbl.setText(
            f"Point {idx + 1} of {len(self._points)}  ·  {_fmt_time(p.start)} → {_fmt_time(p.end)}  ·  {status}"
        )

    def _decide_btn_css(self, selected: bool) -> str:
        if selected:
            return f"""
                QPushButton {{
                    background: {YELLOW}; color: {BLACK};
                    border: 1px solid {YELLOW}; border-radius: 8px;
                    padding: 10px; font-size: 13px; font-weight: 700;
                }}
                QPushButton:hover {{ background: {YELLOW_HOVER}; }}
            """
        # Idle AND "the other player was picked" both look like this — same
        # look as the Play again button, with the same yellow-text hover.
        return ghost_btn_css()

    def _update_decide_buttons(self):
        """Player names replace the bare A/B labels once entered; the button
        for whichever player actually won turns yellow-on-black with a green
        check appended, everything else (including the review page before
        any winner is picked) stays in the neutral ghost style.
        """
        if not hasattr(self, "_decide_a_btn"):
            return
        winner = self._current_point.winner if self._current_point else None
        for player, btn in (("A", self._decide_a_btn), ("B", self._decide_b_btn)):
            name = self._names.get("a" if player == "A" else "b") or f"Player {player}"
            selected = winner == player
            btn.setText(f"{name} won  [{player}]")
            btn.setIcon(_check_icon() if selected else QIcon())
            btn.setIconSize(QSize(14, 14))
            btn.setStyleSheet(self._decide_btn_css(selected))
        # The manual/"Tag New Point" buttons are a one-shot action (click ->
        # point is created and finalized immediately) with no reviewable
        # "selected" state to reflect, so they only ever need the name swap.
        if hasattr(self, "_a_btn"):
            self._a_btn.setText(f"{self._names.get('a') or 'Player A'} won  [A]")
            self._b_btn.setText(f"{self._names.get('b') or 'Player B'} won  [B]")

    def _nudge_current(self, target: str, delta: float):
        if self._current_point is None:
            return
        p = self._current_point
        if target == "start":
            p.start = max(0.0, p.start + delta)
        else:
            p.end = max(p.start + 0.05, p.end + delta)
        self._points.sort(key=lambda pt: pt.start)
        self._refresh_all()

        # Resume playback where the user just made the edit relevant, not a
        # blanket replay-from-start: a Start nudge is easiest to judge from
        # the new start; an End nudge is easiest to judge with a few
        # seconds of run-up into the new boundary — unless the point's now
        # too short for that lead-in, in which case just start on it.
        if target == "start":
            resume_at = p.start
        else:
            resume_at = p.end - 5.0 if (p.end - p.start) >= 5.0 else p.start
        self._player.setPosition(int(resume_at * 1000))
        self._player.play()

    def _split_current_at_playhead(self):
        """Splits the point being reviewed into two at the current playhead
        — for the case rally_reel (or manual tagging) merged two real points
        into one segment. Both halves come out pending: whatever winner the
        original point had no longer applies once it's two points.
        """
        if self._current_point is None:
            return
        p = self._current_point
        now = self._now()
        margin = 0.1  # keep both halves non-degenerate
        if now <= p.start + margin or now >= p.end - margin:
            self._set_status("Move the playhead inside the point (not at its very edge) to split it.", error=True)
            return
        second_half = PointTag(start=now, end=p.end, winner=None)
        p.end = now
        p.winner = None
        self._points.append(second_half)
        self._points.sort(key=lambda pt: pt.start)
        self._refresh_all()
        # The split shortens the current point but doesn't move the
        # playhead, so no positionChanged fires on its own — without this
        # the slider/time readout would keep showing the pre-split duration
        # until the next natural position tick.
        self._sync_seek_slider(self._player.position())

    def _delete_current(self):
        if self._review_mode:
            if self._current_point is None:
                return
            idx = self._current_index()
            self._points.remove(self._current_point)
            self._current_point = None
            self._refresh_all()
            if self._points:
                self._goto_point(min(idx, len(self._points) - 1))
            else:
                self._set_mode("manual")
        else:
            # Manual/free-scrub mode has no "current" point concept; Del
            # there removes the last tagged point, mirroring Undo.
            self._undo()

    # ── Tagging (feature 2: from scratch) ───────────────────────────────────

    def _mark_start(self):
        self._pending_start = self._now()
        self._pending_lbl.setText(
            f"● Point in progress — started at {_fmt_time(self._pending_start)}. Press A or B when it ends."
        )

    def _assign_winner(self, winner: str):
        if self._review_mode and self._current_point is not None:
            point = self._current_point
            point.winner = winner
            self._points.sort(key=lambda p: p.start)
            # Find the next pending point BEFORE refreshing, and switch
            # `_current_point` to it via _enter_review — which itself calls
            # _refresh_all() — so the table's current-row highlight lands on
            # the new point in one pass instead of flashing the old one.
            nxt = self._next_pending_after(point)
            if nxt is not None:
                self._enter_review(nxt)
            else:
                self._player.pause()
                self._refresh_all()
            return

        # Manual/free-scrub mode (feature 2): S marked a start, A/B ends it.
        if self._pending_start is None:
            return
        end = self._now()
        start = self._pending_start
        if end <= start:
            return
        self._points.append(PointTag(start=start, end=end, winner=winner))
        self._pending_start = None
        self._pending_lbl.setText("No point in progress. Press S at the moment a point begins.")
        self._points.sort(key=lambda p: p.start)
        self._refresh_all()

    def _undo(self):
        """Manual/Tag-New-Point mode only: removes the last tagged point.
        No-ops in review mode — its "Undo winner" button was removed, and
        popping an arbitrary list-end point via a stray Z press while
        reviewing would be a confusing, undocumented way to delete a point
        that has nothing to do with the one on screen; "Delete this point
        [Del]" already covers that mode.
        """
        if self._review_mode or not self._points:
            return
        self._points.pop()
        self._pending_start = None
        self._refresh_all()

    def _insert_point_at_playhead(self):
        now = self._now()
        p = PointTag(start=now, end=now + 3.0, winner=None)
        self._points.append(p)
        self._points.sort(key=lambda pt: pt.start)
        self._refresh_all()
        self._enter_review(p)

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if self._suspend_table_signal:
            return
        row, col = item.row(), item.column()
        if row >= len(self._points) or col not in (COL_START, COL_END):
            return
        p = self._points[row]
        if col == COL_START:
            p.start = max(0.0, _parse_time(item.text(), p.start))
        else:
            p.end = max(p.start + 0.05, _parse_time(item.text(), p.end))
        self._points.sort(key=lambda pt: pt.start)
        self._refresh_all()

    # ── Format / names ──────────────────────────────────────────────────────

    def _on_name_changed(self, key, text):
        self._names[key] = text
        self._refresh_all()

    def _on_format_changed(self, *_):
        self._format = MatchFormat(
            best_of=self._best_of_cb.currentData(),
            set_to=self._set_to_cb.currentData(),
            no_ad=self._deuce_cb.currentData(),
            final_set=self._final_set_cb.currentData(),
            tiebreak_to=7,
            super_to=10,
            first_server=self._first_server_cb.currentData(),
        )
        self._refresh_all()

    # ── Import / export ──────────────────────────────────────────────────────

    def _on_import_segments(self):
        path = segments_path_for(self._video_path) if self._video_path else ""
        if not os.path.isfile(path):
            picked, _ = QFileDialog.getOpenFileName(
                self, "Select rally_segments.json", "", "JSON Files (*.json);;All Files (*)"
            )
            if not picked:
                return
            path = picked

        try:
            imported = import_segments_as_points(path)
        except Exception as ex:
            QMessageBox.warning(self, "Import failed", str(ex))
            return

        if self._points:
            resp = QMessageBox.question(
                self, "Import segments",
                f"Replace the {len(self._points)} existing point(s) with {len(imported)} imported "
                "segment(s), or append them?",
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )
            if resp == QMessageBox.StandardButton.Cancel:
                return
            if resp == QMessageBox.StandardButton.Yes:
                self._points = imported
            else:
                self._points = self._points + imported
        else:
            self._points = imported

        self._points.sort(key=lambda p: p.start)
        self._current_point = None
        self._refresh_all()
        self._enter_review_default()

    def _export_tags(self):
        default_dir = str(Path(self._video_path).parent) if self._video_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export tags.json", os.path.join(default_dir, "tags.json"),
            "JSON Files (*.json)"
        )
        if not path:
            return
        save_tags(path, self._current_project())

    def _import_tags_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import tags.json", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            project = load_tags(path)
        except Exception as ex:
            QMessageBox.warning(self, "Import failed", str(ex))
            return
        self._names = project.names
        self._format = project.format
        self._points = project.points
        self._current_point = None
        self._name_a_edit.setText(self._names.get("a", "Player A"))
        self._name_b_edit.setText(self._names.get("b", "Player B"))
        self._apply_format_to_combos()
        self._refresh_all()
        self._enter_review_default()

    def _current_project(self) -> ProjectState:
        return ProjectState(
            names=dict(self._names), format=self._format,
            points=list(self._points),
            video_name=os.path.basename(self._video_path) if self._video_path else "",
        )

    def _autosave(self):
        if not self._video_path:
            return
        try:
            save_tags(tags_path_for(self._video_path), self._current_project())
        except Exception:
            pass  # autosave is best-effort, never blocks tagging

    # ── Refresh / rendering of derived state ────────────────────────────────

    def _refresh_all(self):
        # Replay only the unbroken prefix of resolved winners: a pending
        # point blocks the match from progressing past it (you can't know
        # who won game 4 before point 3 is settled), so the live standing
        # reflects the score entering the first unresolved point, not a
        # score that silently skips over it.
        winners_prefix = []
        for p in self._points:
            if p.winner is None:
                break
            winners_prefix.append(p.winner)
        replay = replay_match(winners_prefix, self._format)
        self._refresh_table(replay)
        self._scoreboard_preview.set_snapshot(replay.final_state, self._names)
        self._standing_lbl.setText(describe_score(replay.final_state, self._names))
        kept = sum(max(0.0, p.end - p.start) for p in self._points)
        pending_n = sum(1 for p in self._points if p.winner is None)
        self._stat_lbl.setText(
            f"{len(self._points)} point(s) · {pending_n} pending winner · kept {_fmt_time(kept)} of footage"
        )
        self._refresh_render_btn()
        self._sync_review_ui()
        self._autosave()

    def _refresh_table(self, replay):
        t = self._table
        self._suspend_table_signal = True
        # clearContents() (not just setRowCount) explicitly tears down the
        # per-row Winner cell widgets before rebuilding — _refresh_table runs
        # on every state change and each call constructs brand-new widgets,
        # so without an explicit clear, stale ones can survive and end up
        # visually overlapping the freshly laid-out row below them.
        t.clearContents()
        t.setRowCount(len(self._points))
        snap_i = 0
        for i, p in enumerate(self._points):
            is_current = p is self._current_point
            idx_item = QTableWidgetItem(str(i + 1))
            idx_item.setFlags(idx_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            t.setItem(i, COL_IDX, idx_item)

            t.setItem(i, COL_START, QTableWidgetItem(_fmt_time(p.start)))
            t.setItem(i, COL_END, QTableWidgetItem(_fmt_time(p.end)))

            winner_widget = self._winner_cell(i, p)
            t.setCellWidget(i, COL_WINNER, winner_widget)

            if p.winner is None:
                score_text = "pending"
            elif snap_i < len(replay.snapshots):
                snap = replay.snapshots[snap_i]
                snap_i += 1
                score_text = (f"{snap['games']['A']}-{snap['games']['B']}  "
                               f"{snap['pointLabels']['A']}-{snap['pointLabels']['B']}")
            else:
                score_text = "— (earlier point unresolved)"
            score_item = QTableWidgetItem(score_text)
            score_item.setFlags(score_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if p.winner is None:
                score_item.setForeground(Qt.GlobalColor.darkGray)
            t.setItem(i, COL_SCORE, score_item)

            # Currently-reviewed row gets a distinct highlight (independent
            # of QTableWidget's own selection, which we deliberately don't
            # drive from here — see _on_cell_clicked).
            if is_current:
                for c in (COL_IDX, COL_START, COL_END, COL_SCORE):
                    t.item(i, c).setBackground(Qt.GlobalColor.darkYellow)
        self._suspend_table_signal = False

    def _winner_cell(self, row: int, p: PointTag):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)
        for player in ("A", "B"):
            b = QPushButton(player)
            b.setFixedSize(24, 22)
            active = p.winner == player
            # Same default/selected split as the main decide buttons (black
            # + gray idle, yellow + black selected) — just without the
            # hover-color QSS or checkmark icon, no room for either at 24px.
            bg = YELLOW if active else BLACK
            fg = BLACK if active else TEXT_DIM
            b.setStyleSheet(f"QPushButton {{ background: {bg}; color: {fg}; border-radius: 4px; font-size: 11px; font-weight: 700; border: 1px solid rgba(255,255,255,0.15); }}")
            b.clicked.connect(lambda _, r=row, pl=player: self._set_row_winner(r, pl))
            lay.addWidget(b)
        return w

    def _set_row_winner(self, row: int, winner: str):
        if 0 <= row < len(self._points):
            self._points[row].winner = winner
            self._refresh_all()

    def _refresh_render_btn(self):
        ready = bool(self._points) and all(p.winner in ("A", "B") for p in self._points) and self._render_worker is None
        self._render_btn.setEnabled(ready)
        self._render_btn.setStyleSheet(primary_btn_css(enabled=ready))

    # ── Rendering ──────────────────────────────────────────────────────────

    def _on_render(self):
        if not self._video_path:
            return
        if not ensure_ffmpeg(self):
            return
        output = str(Path(self._video_path).parent / f"{Path(self._video_path).stem}_scoreboard_reel.mp4")
        self._result_panel.setVisible(False)
        self._progress.setValue(0)
        self._set_status("Rendering…")
        self._render_btn.setEnabled(False)
        self._render_btn.setText("RENDERING…")
        self._sleep_blocker.start()

        self._render_worker = _RenderWorker(
            self._video_path, list(self._points), self._format, dict(self._names), output
        )
        self._render_worker.stage.connect(self._on_render_stage)
        self._render_worker.render_done.connect(self._on_render_finished)
        self._render_worker.render_failed.connect(self._on_render_error)
        # QThread's own built-in `finished` — fires only after the OS thread
        # has actually stopped, unlike our render_done/render_failed signals
        # which we emit manually from inside run(). deleteLater must be tied
        # to this one so cleanup can never race the real thread teardown.
        self._render_worker.finished.connect(self._render_worker.deleteLater)
        self._render_worker.start()

    def _on_render_stage(self, i, n, label, frac):
        if frac is None or frac < 0:
            self._progress.setValue(int(100 * i / max(1, n)))
        else:
            self._progress.setValue(int(100 * (i + frac) / max(1, n)))
        self._set_status(f"{label} ({i}/{n})")

    def _on_render_finished(self, output_path):
        self._render_worker = None
        self._sleep_blocker.stop()
        self._progress.setValue(100)
        self._set_status("Done")
        notify("Anya Tennis — scoreboard reel complete", os.path.basename(output_path))
        self._render_btn.setText("RENDER SCOREBOARD VIDEO")
        self._output_path = output_path
        self._result_path_lbl.setText(os.path.basename(output_path))
        self._result_panel.setVisible(True)
        self._refresh_render_btn()

    def _on_render_error(self, msg):
        self._render_worker = None
        self._sleep_blocker.stop()
        self._progress.setValue(0)
        self._set_status(f"Error: {msg}  (details in {log_path()})", error=True)
        self._render_btn.setText("RENDER SCOREBOARD VIDEO")
        self._refresh_render_btn()

    def _set_status(self, text, error=False):
        color = "#e74c3c" if error else "rgba(255,255,255,0.55)"
        self._status.setStyleSheet(f"color: {color}; font-size: 12px;")
        self._status.setText(text)

    def _open_output_folder(self):
        import subprocess
        import sys
        folder = str(Path(self._output_path).parent)
        if sys.platform == "darwin":
            subprocess.run(["open", folder])
        elif sys.platform == "win32":
            subprocess.run(["explorer", folder])
        else:
            subprocess.run(["xdg-open", folder])
