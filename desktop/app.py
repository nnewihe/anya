"""
app.py — Anya Tennis desktop GUI (black & yellow)

A thin shell over ``pipeline.rally_reel`` and ``pipeline.scoreboard_reel``:
a QTabWidget hosting two tabs —

  * Highlight Reel  — pick a match video, click the four court corners once,
    get a reel of just the rallies (highlight_tab.HighlightReelTab).
  * Scoreboard      — tag point winners against a raw video (from scratch,
    or seeded from the Highlight Reel tab's already-detected point
    boundaries), then render a scored highlight video
    (scoreboard_tab.ScoreboardTab).

Colours and logo follow DESIGN.md and are shared with the mobile app.

The pipeline is *imported*, never copied — the repo root goes on sys.path and
``pipeline.X`` is imported as a package, so editing the pipeline takes effect
on the next run with no rebuild.
"""

import logging
import multiprocessing
import os
import shutil
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QTabWidget, QPushButton,
)
from PyQt6.QtCore import Qt, QSize, QUrl
from PyQt6.QtGui import QDesktopServices, QFont, QPixmap

# Import the pipeline as a PACKAGE: its modules use intra-package relative
# imports (`from .ball_tracker import …`), so the repo root — the parent of
# pipeline/ — goes on sys.path and modules are imported as `pipeline.X`.
# (Putting pipeline/ itself on the path and importing bare `rally_detector`
# breaks on those relative imports.) desktop/ itself also needs to be on the
# path so sibling modules (theme, highlight_tab, scoreboard_tab, ...) import
# as top-level names rather than a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from applog import setup_logging
from preflight import repair_path
from theme import BLACK, SURFACE_ALT, TEXT_DIM, WHITE, YELLOW, ghost_btn_css
from highlight_tab import HighlightReelTab
from scoreboard_tab import ScoreboardTab
from update_check import check_for_updates
from version import APP_VERSION

# Where the Download button sends a tester. The landing page rather than the
# GitHub release: it carries the install steps and the Apple-silicon-only
# caveat, and a release page's source-code zips and asset list are noise to
# someone who just wants the app.
DOWNLOAD_URL = "https://nnewihe.github.io/anya/"


def _logo_path():
    """Locate the Anya Tennis logo mark (shared with the mobile app).

    Resolves both the packaged (PyInstaller ``_MEIPASS/assets``) and dev-run
    (``../mobile/assets/images``) locations; returns "" if neither exists so
    the header degrades gracefully.
    """
    name = "anya_logo.png"
    here = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    candidates = [
        here / "assets" / name,
        Path(__file__).resolve().parent.parent / "mobile" / "assets" / "images" / name,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return ""


class RallyDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Anya Tennis — {APP_VERSION}")
        self.setMinimumSize(900, 780)
        self.resize(1240, 920)
        self._setup_ui()

        # Held until the thread's `finished` fires — see UpdateChecker's
        # docstring and HighlightReelTab._release_worker for why dropping the
        # last reference to a live QThread is fatal rather than merely untidy.
        self._update_checker = check_for_updates(self, self._on_update_available)
        if self._update_checker is not None:
            # Drop our reference once the thread is done. `check_for_updates`
            # has already scheduled deleteLater, so holding on past that leaves
            # a Python wrapper around a deleted C++ object — touching it raises
            # RuntimeError. Nothing does today; this keeps that true.
            self._update_checker.finished.connect(self._clear_update_checker)

    # ── UI construction ────────────────────────────────────────────────────

    def _setup_ui(self):
        # QMessageBox/QDialog are QWidget subclasses too, so the blanket
        # `QWidget { background }` rule below cascades to them — without an
        # explicit text color that leaves message-box text dark-on-black and
        # unreadable (only a button's default focus highlight stays
        # visible). Give dialogs their own readable, on-brand styling rather
        # than letting them silently inherit the app chrome's rule.
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {BLACK}; }}
            QMessageBox, QDialog {{ background: {BLACK}; }}
            QMessageBox QLabel {{ color: {WHITE}; }}
            QMessageBox QPushButton {{
                background: rgba(255,255,255,0.10); color: {WHITE};
                border: 1px solid rgba(255,255,255,0.28); border-radius: 6px;
                padding: 6px 16px; min-width: 64px;
            }}
            QMessageBox QPushButton:hover {{ border-color: {YELLOW}; color: {YELLOW}; }}
            QMessageBox QPushButton:default {{ background: {YELLOW}; color: {BLACK}; border: none; }}
        """)

        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(36, 14, 36, 0)
        lay.setSpacing(8)

        lay.addLayout(self._logo_row())
        lay.addWidget(self._divider())

        # Built hidden and added now so it can appear in place later without
        # reflowing anything: the check finishes seconds after launch, and a
        # banner that pushed the tab bar down while a tester was reaching for
        # it would be worse than no banner.
        self._update_banner = self._build_update_banner()
        lay.addWidget(self._update_banner)

        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; }}
            QTabBar::tab {{
                background: transparent; color: {TEXT_DIM};
                padding: 10px 18px; font-size: 12px; font-weight: 700;
                letter-spacing: 0.06em; border: none;
            }}
            QTabBar::tab:selected {{ color: {YELLOW}; border-bottom: 2px solid {YELLOW}; }}
        """)
        highlight_tab = HighlightReelTab()
        scoreboard_tab = ScoreboardTab()
        tabs.addTab(highlight_tab, "HIGHLIGHT REEL")
        tabs.addTab(scoreboard_tab, "SCOREBOARD")
        lay.addWidget(tabs, 1)

        # Load Video / Import segments / video name (Scoreboard-specific)
        # live in the same row as the tab labels themselves, not as a
        # separate row inside the tab body — Qt's corner-widget mechanism is
        # exactly this: a widget anchored in the tab bar's own row. Only
        # relevant while the Scoreboard tab is actually showing, so it's
        # swapped in/out on tab change rather than left visible over
        # Highlight Reel where "Load Video" would mean nothing.
        tabs.setCornerWidget(scoreboard_tab.load_row_widget, Qt.Corner.TopRightCorner)

        def _on_tab_changed(index):
            scoreboard_tab.load_row_widget.setVisible(tabs.widget(index) is scoreboard_tab)

        tabs.currentChanged.connect(_on_tab_changed)
        _on_tab_changed(tabs.currentIndex())

    # ── Update banner ──────────────────────────────────────────────────────

    def _build_update_banner(self):
        """A hidden strip that offers the newer build once one is found.

        Deliberately not a QMessageBox: the check lands a few seconds after
        launch, which is exactly when a tester is picking a video, and a modal
        dialog there would be an interruption they'd dismiss without reading.
        Dismissible, because someone mid-way through a 10-minute job should be
        able to make it go away and update afterwards.
        """
        bar = QWidget()
        bar.setVisible(False)
        bar.setStyleSheet(
            f"QWidget {{ background: {SURFACE_ALT}; border-radius: 6px; }}"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 8, 10, 8)
        row.setSpacing(10)

        self._update_label = QLabel()
        self._update_label.setStyleSheet(f"color: {WHITE}; font-size: 12px; background: transparent;")
        row.addWidget(self._update_label)
        row.addStretch()

        download = QPushButton("DOWNLOAD")
        download.setStyleSheet(ghost_btn_css())
        download.setCursor(Qt.CursorShape.PointingHandCursor)
        download.setToolTip(
            "Opens the download page. To install: quit Anya Tennis, open the "
            "downloaded file, and drag it onto Applications, replacing the old one."
        )
        download.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOWNLOAD_URL)))
        row.addWidget(download)

        # U+00D7, not a heavier ✕/✖: those live in fonts Qt may not fall back
        # to, and a missing glyph renders as some unrelated character rather
        # than nothing (offscreen it came out as "«").
        dismiss = QPushButton("×")
        dismiss.setStyleSheet(ghost_btn_css())
        dismiss.setCursor(Qt.CursorShape.PointingHandCursor)
        dismiss.setFixedWidth(30)
        dismiss.setToolTip("Hide until next launch")
        dismiss.clicked.connect(lambda: bar.setVisible(False))
        row.addWidget(dismiss)

        return bar

    def _clear_update_checker(self):
        self._update_checker = None

    def _on_update_available(self, version, html_url):
        # html_url is the GitHub release page. The button goes to DOWNLOAD_URL
        # instead — same build, but with the install steps around it — so this
        # is carried for the log and for a future "what changed" link rather
        # than used here.
        logging.getLogger("anya_tennis").info("update banner shown for %s (%s)", version, html_url)
        self._update_label.setText(
            f"<b style='color:{YELLOW}'>Anya Tennis {version}</b> is available "
            f"— you're on {APP_VERSION}."
        )
        self._update_banner.setVisible(True)

    def _logo_row(self):
        row = QHBoxLayout()
        logo_path = _logo_path()
        if logo_path:
            # anya_logo.png is the ball-mark only (no wordmark/tagline baked
            # in), so it's a QPixmap on a QLabel, not the QSvgWidget an .svg
            # asset would need.
            pixmap = QPixmap(logo_path).scaled(
                72, 72, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            logo = QLabel()
            logo.setPixmap(pixmap)
            logo.setFixedSize(QSize(72, 72))
            row.addWidget(logo)
        else:
            # Fallback if the asset can't be found (keeps the app usable).
            ball = QLabel("●")
            ball.setStyleSheet(f"color: {YELLOW}; font-size: 34px; padding-right: 4px;")
            ball.setFixedWidth(46)
            row.addWidget(ball)

        tagline_col = QVBoxLayout()
        tagline_col.setSpacing(2)

        tagline = QLabel("Watch your matches in minutes not hours.")
        tagline.setStyleSheet(f"color: {TEXT_DIM}; font-size: 18px;")
        tagline_col.addWidget(tagline)

        # Beta testers are handing over video of themselves or their
        # students — this needs to be visible up front, not buried in a
        # README they'll never open.
        trust = QLabel("Runs 100% on this computer — your video is never uploaded.")
        trust.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        tagline_col.addWidget(trust)

        row.addLayout(tagline_col)

        row.addStretch()

        # Beta build marker — the title bar isn't always visible (e.g. a
        # maximized window on some Linux WMs), so testers need a version
        # they can see and quote in a bug report without hunting for it.
        version = QLabel(APP_VERSION)
        version.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        row.addWidget(version, 0, Qt.AlignmentFlag.AlignBottom)

        return row

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background: {YELLOW}; border: none; max-height: 1px; min-height: 1px;")
        return line


def main():
    # FIRST statement in main(), before logging, Qt, or anything that could
    # spawn a worker. Windows has no fork(): multiprocessing re-launches the
    # program and unpickles the child's target, and in a frozen build "the
    # program" is AnyaTennis.exe, so a child re-runs main() and opens another
    # window — which spawns another child, without bound. freeze_support()
    # makes a re-launched child execute its worker payload and exit instead.
    # Nothing here calls multiprocessing directly, but torch and joblib both
    # do, and the failure mode is an unkillable cascade of app windows on a
    # tester's machine. It is a documented no-op on macOS and Linux.
    multiprocessing.freeze_support()

    # Must run before anything else can fail — it installs sys.excepthook so
    # even an error during QApplication/window construction gets logged
    # instead of vanishing (the packaged app has no console to print to).
    setup_logging()

    # Launched from Finder, this process inherits launchd's PATH, which has no
    # /opt/homebrew/bin on it — so ffmpeg looks missing on machines that have
    # it. Windows has the same symptom for a different reason (Explorer hands
    # down the PATH it started with, so a just-installed ffmpeg is invisible
    # until the next sign-in). Repair once here, before any tab can shell out.
    repair_path()

    # Logged because it is the one startup fact that cannot be checked from
    # outside the process: os.environ changes are invisible to `ps` on macOS
    # (which shows the initial env block), so without this line the only way
    # to know whether the repair worked on a tester's machine is to make them
    # start a job and see whether it fails.
    logging.getLogger("anya_tennis").info(
        "ffmpeg resolved to: %s  (PATH=%s)",
        shutil.which("ffmpeg") or "NOT FOUND", os.environ.get("PATH", ""),
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = QFont("Helvetica Neue", 11)
    app.setFont(font)

    window = RallyDetectorApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
