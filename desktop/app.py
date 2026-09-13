"""
app.py — Anya Tennis desktop GUI (black & yellow)

A thin shell over ``pipeline.anya2`` and ``pipeline.scoreboard_reel``:
a QTabWidget hosting two tabs —

  * Highlight Reel  — pick a match video, click the four court corners once,
    get a reel of just the rallies (highlight_tab.HighlightReelTab).
  * Scoreboard      — tag point winners against a raw video (from scratch,
    or seeded from the Highlight Reel tab's already-detected point
    boundaries), then render a scored highlight video
    (scoreboard_tab.ScoreboardTab).

Since 0.2.0 the shell is a QStackedWidget rather than the tabs directly:
index 0 is the sign-in / pricing gate (gate_screen.GateScreen) and index 1 is
the app itself, built only once entitlement is confirmed. The tab modules are
therefore imported INSIDE _show_app() rather than at module scope -- they pull
in torch, ultralytics and sklearn transitively, and a signed-out launch should
not pay seconds of import and hundreds of MB of RSS to look at a paywall.

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
    QLabel, QFrame, QTabWidget, QPushButton, QStackedWidget,
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
from update_check import check_for_updates
from version import APP_VERSION

# HighlightReelTab and ScoreboardTab are deliberately NOT imported here -- see
# the module docstring and _show_app(). Everything below is cheap: no torch, no
# ultralytics, nothing that touches a model file.
import authworker
from account_dialog import AccountDialog
from entitlement import EntState
from gate_screen import GateScreen

# Where the Download button sends a tester. The landing page rather than the
# GitHub release: it carries the install steps and the Apple-silicon-only
# caveat, and a release page's source-code zips and asset list are noise to
# someone who just wants the app.
DOWNLOAD_URL = "https://nnewihe.github.io/anya/"

# Remembers that the pricing pre-announcement has been read and dismissed.
# Versioned in the key so a future announcement is a new key rather than
# something a past dismissal silently suppresses.
#
# No dots in it: QSettings uses "." as its group separator on macOS and escapes
# a literal one into U+00B7, so "going-paid-0.2.0" lands in the plist as
# "going-paid-0·2·0". It round-trips correctly, but a preference key
# containing a middle dot is the kind of thing someone later spends an hour on.
_NOTICE_KEY = "notice/going-paid-v2/dismissed"


class _FakeResult:
    """A minimal stand-in for entitlement.Entitlement.

    Used where a state is known without having evaluated a session: the two
    env-var development overrides, and the two paths that force the gate (a
    failed check, and signing out from the account dialog). Carrying the real
    class here would mean constructing one with no session to describe.
    """

    def __init__(self, state, reason="", plan=None, expires_at=None):
        self.state = state
        self.reason = reason
        self.plan = plan
        self.expires_at = expires_at


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

        self._session = None      # authstore.Session once signed in
        self._app_page = None     # the QTabWidget, built lazily by _show_app
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

        self._ent_worker = self._start_entitlement_check()

    # ── Entitlement ────────────────────────────────────────────────────────

    def _start_entitlement_check(self):
        """Ask, off the GUI thread, whether this machine has a paid session.

        Two escape hatches, both env-gated so neither can fire in a shipped
        build by accident. They exist so the gate and the app can each be
        worked on without standing up billing first.
        """
        if os.environ.get("ANYA_FORCE_GATE"):
            logging.getLogger("anya_tennis").info("ANYA_FORCE_GATE set; showing the gate")
            self._on_entitlement(_FakeResult(EntState.SIGNED_OUT), None)
            return None
        if os.environ.get("ANYA_FAKE_ENTITLED"):
            logging.getLogger("anya_tennis").info("ANYA_FAKE_ENTITLED set; skipping the gate")
            self._on_entitlement(_FakeResult(EntState.ENTITLED), None)
            return None

        worker = authworker.verify_entitlement(
            self, self._on_entitlement, self._on_entitlement_failed)
        worker.finished.connect(self._clear_ent_worker)
        return worker

    def _on_entitlement(self, result, session):
        self._session = session
        logging.getLogger("anya_tennis").info(
            "entitlement at launch: %s (%s)", result.state.value, result.reason)
        if result.state.allows_app:
            self._show_app()
        else:
            self._show_gate(result, session)

    def _on_entitlement_failed(self, message):
        # The check itself broke, which is not the same as being unentitled.
        # Fall back to the gate rather than to the app: failing open here would
        # make a broken check the easiest way past the paywall.
        logging.getLogger("anya_tennis").warning("entitlement check failed: %s", message)
        self._on_entitlement(_FakeResult(EntState.SIGNED_OUT), None)

    def _clear_ent_worker(self):
        self._ent_worker = None

    def _show_gate(self, result, session):
        self._gate.apply_state(result, session)
        self._gate.start_preview()
        self._stack.setCurrentWidget(self._gate)
        self._account_btn.setVisible(session is not None)

    def _on_gate_entitled(self, result, session):
        self._session = session
        self._gate.stop_preview()
        self._show_app()

    def _show_app(self):
        """Build the two tabs, importing them for the first time.

        The import is what costs: highlight_tab and scoreboard_tab pull in
        pipeline.*, and through it torch, ultralytics and sklearn — seconds of
        work and hundreds of MB. Doing it here rather than at module scope is
        what makes a signed-out launch cheap.

        It runs on the GUI thread on purpose. Importing torch off the main
        thread is not a documented-safe thing to do, and the honest fix is to
        say what is happening: the "Starting up" page is already showing, and
        it stays up for the couple of seconds this takes.

        rally_app.spec lists every pipeline.* module in `hiddenimports`, so
        PyInstaller still finds them all despite the import being invisible to
        static analysis. Do not remove those entries.
        """
        self._account_btn.setVisible(self._session is not None)

        if self._app_page is not None:
            self._stack.setCurrentWidget(self._app_page)
            return

        self._stack.setCurrentWidget(self._loading)
        QApplication.processEvents()   # paint "Starting up" before we block

        from highlight_tab import HighlightReelTab
        from scoreboard_tab import ScoreboardTab

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

        self._app_page = tabs
        self._stack.addWidget(tabs)
        self._stack.setCurrentWidget(tabs)

    def _open_account(self):
        if self._session is None:
            return
        dialog = AccountDialog(self._session, self)
        dialog.exec()
        if dialog.signed_out:
            self._session = None
            self._show_gate(_FakeResult(EntState.SIGNED_OUT), None)

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

        # Pre-announcement of the move to a paid app. Above the update banner
        # because it is the more consequential of the two, and unlike that one
        # it is shown immediately rather than after a network round trip —
        # there is nothing to look up.
        self._notice_banner = self._build_notice_banner()
        lay.addWidget(self._notice_banner)

        # Built hidden and added now so it can appear in place later without
        # reflowing anything: the check finishes seconds after launch, and a
        # banner that pushed the tab bar down while a tester was reaching for
        # it would be worse than no banner.
        self._update_banner = self._build_update_banner()
        lay.addWidget(self._update_banner)

        # The header, divider and update banner are outside the stack: they are
        # the same in both states, and rebuilding them per page would make the
        # logo jump on the transition into the app.
        self._stack = QStackedWidget()

        self._gate = GateScreen()
        self._gate.entitled.connect(self._on_gate_entitled)
        self._stack.addWidget(self._gate)

        self._loading = QLabel("Starting up…")
        self._loading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading.setStyleSheet(f"color: {TEXT_DIM}; font-size: 14px;")
        self._stack.addWidget(self._loading)

        # Neither page yet: the entitlement check has not answered. Showing the
        # gate first would flash a paywall at a paying customer every launch.
        self._stack.setCurrentWidget(self._loading)

        lay.addWidget(self._stack, 1)

    # ── Pricing pre-announcement ───────────────────────────────────────────

    def _build_notice_banner(self):
        """Tell testers, in advance, that the next version costs money.

        The point of shipping this build at all. Thirteen betas were tested by
        people who got nothing for it, and the worst possible way to introduce
        a price is for it to appear one morning without warning. This says it
        early, in the app they already have, and says the part that matters to
        them first: they are not the ones being asked to pay.

        Deliberately not a QMessageBox on launch. A modal would be dismissed
        unread by someone reaching for their video, which is exactly the
        failure the update banner's docstring already describes. The strip
        carries a one-line hook; the detail is one click away for anyone who
        wants it.

        Unlike the update banner, dismissing this is remembered across
        launches. That banner reappears every time because installing an
        update is actionable every time; an announcement is information, and
        re-showing it to someone who has read it and pressed the X is nagging.
        Nagging the people who did your QA for free is a poor way to open a
        conversation about money.
        """
        bar = QWidget()
        bar.setVisible(not self._notice_dismissed())
        bar.setStyleSheet(
            f"QWidget {{ background: {SURFACE_ALT}; border-radius: 6px; }}"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 8, 10, 8)
        row.setSpacing(10)

        label = QLabel(
            f"<b style='color:{YELLOW}'>Anya Tennis is becoming a paid app</b> "
            f"in the next version — and if you're reading this, your first year "
            f"is free."
        )
        label.setStyleSheet(f"color: {WHITE}; font-size: 12px; background: transparent;")
        row.addWidget(label)
        row.addStretch()

        details = QPushButton("WHAT'S CHANGING?")
        details.setStyleSheet(ghost_btn_css())
        details.setCursor(Qt.CursorShape.PointingHandCursor)
        details.clicked.connect(self._show_notice_details)
        row.addWidget(details)

        # U+00D7 for the same reason as the update banner's — see there.
        dismiss = QPushButton("×")
        dismiss.setStyleSheet(ghost_btn_css())
        dismiss.setCursor(Qt.CursorShape.PointingHandCursor)
        dismiss.setFixedWidth(30)
        dismiss.setToolTip("Hide this for good")
        dismiss.clicked.connect(self._dismiss_notice)
        row.addWidget(dismiss)

        return bar

    def _notice_settings(self):
        """Qt's own per-user settings store — no new dependency, and on macOS
        it is an ordinary plist under ~/Library/Preferences. The app has never
        needed to remember anything before; this is the first thing it does.

        The arguments matter. QSettings builds the macOS preference domain by
        reversing the organization and appending the application, so these two
        strings produce `com.anyatennis.app` — the bundle identifier
        rally_app.spec:371 already gives the app. Passing a display name like
        ("Anya Tennis", "Anya Tennis") instead invents a SECOND domain,
        `com.anya-tennis.Anya Tennis`, which is not the app's own, does not go
        away when the app is deleted, and would leave a stray plist on every
        tester's machine.
        """
        from PyQt6.QtCore import QSettings

        return QSettings("anyatennis.com", "app")

    def _notice_dismissed(self):
        try:
            return bool(self._notice_settings().value(_NOTICE_KEY, False, type=bool))
        except Exception:
            # A settings store that cannot be read must not stop the app from
            # launching. Showing the banner again is the harmless failure.
            return False

    def _dismiss_notice(self):
        self._notice_banner.setVisible(False)
        try:
            self._notice_settings().setValue(_NOTICE_KEY, True)
        except Exception:
            logging.getLogger("anya_tennis").info("could not persist notice dismissal")

    def _show_notice_details(self):
        from PyQt6.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setWindowTitle("Anya Tennis is becoming a paid app")
        box.setTextFormat(Qt.TextFormat.RichText)
        # Qt renders BOTH of a message box's text roles bold under Fusion, so
        # without this the four paragraphs below come out as a wall of bold
        # that is harder to read than plain text. Reset the weight here and let
        # the <b> in the copy do the emphasising. Scoped to this box rather
        # than the app-wide QMessageBox rule in _setup_ui, which the crash
        # dialog also uses and which wants its short text to stay prominent.
        box.setStyleSheet(
            f"QLabel {{ font-weight: 400; color: {WHITE}; }}"
            f"QLabel[text^='Anya Tennis is becoming'] {{ font-weight: 700; }}"
        )
        # setText is QMessageBox's HEADING slot and Qt renders it bold whatever
        # markup it contains; the body belongs in setInformativeText, which is
        # the regular-weight one. Putting four paragraphs in setText gives a
        # wall of bold that is harder to read than plain text would have been.
        box.setText("Anya Tennis is becoming a paid app — and you get a free year.")
        box.setInformativeText(
            "<p>You have been testing this through thirteen builds and "
            "found things that were broken. When the next "
            "version arrives, make an account with the same email address you "
            "use for beta feedback and a year is applied automatically. You "
            "will not be asked for a card.</p>"

            "<p><b>After that, and for everyone else:</b> $40 a year, or $5 a "
            "month. If it turns out not to be for you, there is a button in the "
            "app that cancels and refunds you in full, any time within 14 days "
            "of your first payment. No email, no form.</p>"

            "<p><b>Nothing about how it works changes.</b> Your video is still "
            "never uploaded — every part of finding the rallies still happens on "
            "this computer. Signing in checks your subscription and nothing "
            "else; paying happens on Stripe's own page in your browser.</p>"

            "<p><b>This build is unaffected.</b> Keep using it exactly as you "
            "are. Nothing here starts until you choose to update.</p>"
        )
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.exec()

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

        # Only meaningful once there is an account to open, so it is built
        # hidden and revealed by _show_app/_show_gate — same in-place pattern
        # as the update banner.
        self._account_btn = QPushButton("ACCOUNT")
        self._account_btn.setStyleSheet(ghost_btn_css())
        self._account_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._account_btn.setToolTip("Your plan, billing, and sign out")
        self._account_btn.setVisible(False)
        self._account_btn.clicked.connect(self._open_account)
        row.addWidget(self._account_btn, 0, Qt.AlignmentFlag.AlignBottom)

        # Build marker — the title bar isn't always visible (e.g. a maximized
        # window on some Linux WMs), so testers need a version they can see and
        # quote in a bug report without hunting for it.
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
