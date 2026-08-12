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
import os
import shutil
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QTabWidget,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QPixmap

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
from theme import BLACK, TEXT_DIM, WHITE, YELLOW
from highlight_tab import HighlightReelTab
from scoreboard_tab import ScoreboardTab
from version import APP_VERSION


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
    # Must run before anything else can fail — it installs sys.excepthook so
    # even an error during QApplication/window construction gets logged
    # instead of vanishing (the packaged app has no console to print to).
    setup_logging()

    # Launched from Finder, this process inherits launchd's PATH, which has no
    # /opt/homebrew/bin on it — so ffmpeg looks missing on machines that have
    # it. Repair once here, before any tab can shell out.
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
