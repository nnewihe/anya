"""
gate_screen.py — what someone sees before they have paid.

The whole app when signed out: a one-minute preview, the price, and a way in.
Not a disabled version of the tabs. Two reasons that is the right shape here.
The product is one long-running action, so a half-enabled Highlight Reel tab
would be a Detect button that refuses — which reads as broken rather than as
locked. And app.py only imports the tab modules once entitlement is confirmed,
so a signed-out launch never pays for torch and ultralytics at all.

The left half is a still poster shipped in the bundle. It was a streamed MP4
once, on the reasoning that a video inside the DMG cannot be changed without a
signed, notarized release. What that bought in practice was a screen whose
main panel depended on the network: the stream 404'd for as long as the video
did not exist, and every launch spent a request finding that out. A still has
none of that, needs no QtMultimedia backend on the user's machine, and cannot
fail differently on Windows.

Nothing here is best-effort any more, which is the point. If the poster file
is missing the screen falls back to text, but the file ships in the bundle, so
that path is a guard rather than an expectation.
"""

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

import authworker
import firebase_config as cfg
from applog import logger
from entitlement import EntState
from theme import (
    OUTLINE, SURFACE, SURFACE_ALT, TEXT_DIM, WHITE, YELLOW,
    ghost_btn_css, label_css, line_edit_css, primary_btn_css,
)


class GateScreen(QWidget):
    """Emits `entitled(Entitlement, Session)` when the app should be shown."""

    entitled = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._session = None
        self._worker = None       # see authworker's docstring on holding these
        self._poll_worker = None
        # The unscaled poster; _rescale_poster fits a copy to the frame.
        self._poster_src = None
        self._create_mode = False
        self._setup_ui()

    # ── State in ───────────────────────────────────────────────────────────

    def apply_state(self, result, session):
        """Called by app.py with whatever the launch check found."""
        self._session = session
        state = result.state

        if state is EntState.SIGNED_OUT:
            self._show_signed_out()
        elif state is EntState.UNENTITLED:
            self._show_pricing("Choose a plan to start using Anya Tennis.")
        elif state is EntState.EXPIRED:
            self._show_pricing(self._expiry_copy(result))
        else:
            self.entitled.emit(result, session)

    def _expiry_copy(self, result):
        if result.reason == "clock rolled back":
            return ("Your Mac's clock is set earlier than we last saw it. "
                    "Connect to the internet to confirm your subscription.")
        if "grace" in result.reason or "offline" in result.reason:
            return ("You've been offline for a while. Connect to the internet "
                    "once to keep using Anya Tennis.")
        return "Your subscription has ended. Renew to keep making reels."

    # ── Layout ─────────────────────────────────────────────────────────────

    def _setup_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 18, 0, 18)
        outer.setSpacing(28)
        outer.addLayout(self._poster_column(), 3)
        outer.addWidget(self._panel(), 2)

    def _poster_column(self):
        col = QVBoxLayout()
        col.setSpacing(10)

        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background: {SURFACE}; border-radius: 10px; }}")
        frame.setMinimumHeight(320)
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(1, 1, 1, 1)

        self._poster = QLabel("Anya Tennis")
        self._poster.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._poster.setStyleSheet(f"color: {TEXT_DIM}; font-size: 13px; background: transparent;")
        # Ignored in both directions so a 1024x1536 pixmap cannot drive the
        # column's width: the scaled copy is computed from the frame's size in
        # _rescale_poster, which is the opposite dependency.
        self._poster.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        inner.addWidget(self._poster)

        # The poster is portrait, the column is not, so it has to be refitted
        # whenever the window resizes rather than scaled once at build time.
        self._poster_frame = frame
        frame.installEventFilter(self)

        col.addWidget(frame, 1)
        self._show_poster()

        return col

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        if obj is getattr(self, "_poster_frame", None) and event.type() == QEvent.Type.Resize:
            self._rescale_poster()
        return super().eventFilter(obj, event)

    def _panel(self):
        panel = QWidget()
        panel.setStyleSheet(f"QWidget {{ background: {SURFACE}; border-radius: 10px; }}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(26, 24, 26, 24)
        lay.setSpacing(12)

        self._heading = QLabel("SIGN IN")
        self._heading.setStyleSheet(
            f"color: {WHITE}; font-size: 17px; font-weight: 700; letter-spacing: 0.06em;"
            " background: transparent;")
        lay.addWidget(self._heading)

        self._blurb = QLabel("")
        self._blurb.setWordWrap(True)
        self._blurb.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 12px; background: transparent;")
        lay.addWidget(self._blurb)

        lay.addSpacing(4)

        # ── Sign-in form ──
        self._form = QWidget()
        self._form.setStyleSheet("QWidget { background: transparent; }")
        form_lay = QVBoxLayout(self._form)
        form_lay.setContentsMargins(0, 0, 0, 0)
        form_lay.setSpacing(8)

        form_lay.addWidget(self._label("EMAIL"))
        self._email = QLineEdit()
        self._email.setStyleSheet(line_edit_css())
        self._email.setPlaceholderText("you@example.com")
        self._email.returnPressed.connect(self._submit)
        form_lay.addWidget(self._email)

        form_lay.addWidget(self._label("PASSWORD"))
        self._password = QLineEdit()
        self._password.setStyleSheet(line_edit_css())
        self._password.setEchoMode(QLineEdit.EchoMode.Password)
        self._password.returnPressed.connect(self._submit)
        form_lay.addWidget(self._password)

        self._submit_btn = QPushButton("SIGN IN")
        self._submit_btn.setProperty("anyaPrimary", True)
        self._submit_btn.setStyleSheet(primary_btn_css())
        self._submit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._submit_btn.setFixedHeight(40)
        self._submit_btn.clicked.connect(self._submit)
        form_lay.addSpacing(4)
        form_lay.addWidget(self._submit_btn)

        self._google_btn = QPushButton("CONTINUE WITH GOOGLE")
        self._google_btn.setStyleSheet(ghost_btn_css())
        self._google_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._google_btn.setFixedHeight(36)
        self._google_btn.clicked.connect(self._google)
        form_lay.addWidget(self._google_btn)

        switch_row = QHBoxLayout()
        self._switch_btn = QPushButton("Create an account")
        self._switch_btn.setStyleSheet(self._link_css())
        self._switch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._switch_btn.setFlat(True)
        self._switch_btn.clicked.connect(self._toggle_mode)
        switch_row.addWidget(self._switch_btn)
        switch_row.addStretch()

        self._forgot_btn = QPushButton("Forgot password")
        self._forgot_btn.setStyleSheet(self._link_css())
        self._forgot_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._forgot_btn.setFlat(True)
        self._forgot_btn.clicked.connect(self._forgot)
        switch_row.addWidget(self._forgot_btn)
        form_lay.addLayout(switch_row)

        lay.addWidget(self._form)

        # ── Pricing ──
        self._pricing = self._pricing_block()
        self._pricing.setVisible(False)
        lay.addWidget(self._pricing)

        lay.addStretch()

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 12px; background: transparent;")
        lay.addWidget(self._status)

        legal = QLabel(
            f'<a href="{cfg.TERMS_URL}" style="color:{TEXT_DIM}">Terms</a> · '
            f'<a href="{cfg.PRIVACY_URL}" style="color:{TEXT_DIM}">Privacy</a>')
        legal.setOpenExternalLinks(True)
        legal.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent;")
        lay.addWidget(legal)

        return panel

    def _pricing_block(self):
        box = QWidget()
        box.setStyleSheet("QWidget { background: transparent; }")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self._annual_btn = QPushButton(f"{cfg.PRICE_ANNUAL_DISPLAY}  —  BEST VALUE")
        self._annual_btn.setProperty("anyaPrimary", True)
        self._annual_btn.setStyleSheet(primary_btn_css())
        self._annual_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._annual_btn.setFixedHeight(44)
        self._annual_btn.clicked.connect(lambda: self._checkout("annual"))
        lay.addWidget(self._annual_btn)

        self._monthly_btn = QPushButton(f"{cfg.PRICE_MONTHLY_DISPLAY}")
        self._monthly_btn.setStyleSheet(ghost_btn_css())
        self._monthly_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._monthly_btn.setFixedHeight(38)
        self._monthly_btn.clicked.connect(lambda: self._checkout("monthly"))
        lay.addWidget(self._monthly_btn)

        promise = QLabel(
            f"Cancel within {cfg.REFUND_WINDOW_DAYS} days for a full refund — "
            "one button, no email required.")
        promise.setWordWrap(True)
        promise.setStyleSheet(
            f"color: {YELLOW}; font-size: 11px; background: transparent;")
        lay.addWidget(promise)

        self._waiting = QLabel("")
        self._waiting.setWordWrap(True)
        self._waiting.setVisible(False)
        self._waiting.setStyleSheet(
            f"color: {WHITE}; font-size: 12px; background: transparent;")
        lay.addWidget(self._waiting)

        self._recheck_btn = QPushButton("I'VE PAID — CHECK NOW")
        self._recheck_btn.setStyleSheet(ghost_btn_css())
        self._recheck_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._recheck_btn.setVisible(False)
        self._recheck_btn.clicked.connect(self._start_polling)
        lay.addWidget(self._recheck_btn)

        self._signout_btn = QPushButton("Sign out")
        self._signout_btn.setStyleSheet(self._link_css())
        self._signout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._signout_btn.setFlat(True)
        self._signout_btn.clicked.connect(self._sign_out)
        lay.addWidget(self._signout_btn)

        return box

    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(label_css() + " background: transparent;")
        return lbl

    @staticmethod
    def _link_css():
        return (f"QPushButton {{ background: transparent; border: none; color: {TEXT_DIM};"
                f" font-size: 11px; text-align: left; padding: 2px 0; }}"
                f" QPushButton:hover {{ color: {YELLOW}; }}")

    # ── Modes ──────────────────────────────────────────────────────────────

    def _show_signed_out(self):
        self._form.setVisible(True)
        self._pricing.setVisible(False)
        self._heading.setText("CREATE AN ACCOUNT" if self._create_mode else "SIGN IN")
        self._blurb.setText(
            f"{cfg.PRICE_ANNUAL_DISPLAY} or {cfg.PRICE_MONTHLY_DISPLAY}. "
            f"Full refund within {cfg.REFUND_WINDOW_DAYS} days.")
        # A build made from a checkout without desktop/oauth_client.py can do
        # email sign-in but not Google. Hide the button rather than offering
        # one that always fails — see oauth_client.example.py.
        self._google_btn.setVisible(cfg.google_configured())

        if not cfg.is_configured():
            self._set_status(
                "This build isn't configured for sign-in yet.", error=True)
            for btn in (self._submit_btn, self._google_btn):
                self._set_enabled(btn, False)

    def _show_pricing(self, blurb):
        self._form.setVisible(False)
        self._pricing.setVisible(True)
        self._heading.setText("CHOOSE A PLAN")
        self._blurb.setText(blurb)

    def _toggle_mode(self):
        self._create_mode = not self._create_mode
        self._submit_btn.setText("CREATE ACCOUNT" if self._create_mode else "SIGN IN")
        self._switch_btn.setText(
            "I already have an account" if self._create_mode else "Create an account")
        self._heading.setText("CREATE AN ACCOUNT" if self._create_mode else "SIGN IN")
        self._set_status("")

    def _busy(self, on, message=""):
        for w in (self._submit_btn, self._google_btn, self._switch_btn,
                  self._forgot_btn, self._annual_btn, self._monthly_btn):
            self._set_enabled(w, not on)
        if message:
            self._set_status(message)

    @staticmethod
    def _set_enabled(widget, enabled):
        """Disable a button AND make it look disabled.

        primary_btn_css() picks its colours from an argument rather than from a
        QSS `:disabled` rule, so setEnabled() alone leaves a filled yellow
        button that ignores clicks — which reads as broken rather than as
        unavailable. ghost_btn_css() does carry its own :disabled rule, so
        those only need the flag.
        """
        widget.setEnabled(enabled)
        if widget.property("anyaPrimary"):
            widget.setStyleSheet(primary_btn_css(enabled=enabled))

    def _set_status(self, text, error=False):
        colour = "#E74C3C" if error else TEXT_DIM
        self._status.setStyleSheet(
            f"color: {colour}; font-size: 12px; background: transparent;")
        self._status.setText(text)

    # ── Actions ────────────────────────────────────────────────────────────

    def _submit(self):
        email = self._email.text().strip()
        password = self._password.text()
        if not email or not password:
            self._set_status("Enter your email and password.", error=True)
            return

        self._busy(True, "Signing in…")
        start = authworker.create_account if self._create_mode else authworker.sign_in_with_password
        self._worker = start(self, email, password, self._on_signed_in, self._on_auth_failed)
        self._worker.finished.connect(self._release_worker)

    def _google(self):
        self._busy(True, "Finishing sign-in in your browser…")
        self._worker = authworker.sign_in_with_google(
            self, lambda url: QDesktopServices.openUrl(QUrl(url)),
            self._on_signed_in, self._on_auth_failed)
        self._worker.finished.connect(self._release_worker)

    def _forgot(self):
        email = self._email.text().strip()
        if not email:
            self._set_status("Enter your email first, then press this.", error=True)
            return
        self._busy(True)
        self._worker = authworker.send_password_reset(self, email, self._on_reset_sent)
        self._worker.finished.connect(self._release_worker)

    def _checkout(self, plan):
        if self._session is None:
            return
        self._busy(True, "Opening checkout in your browser…")
        self._worker = authworker.start_checkout(
            self, self._session, plan,
            self._on_checkout_url, self._on_auth_failed)
        self._worker.finished.connect(self._release_worker)

    def _sign_out(self):
        import authstore
        authstore.clear()
        self._session = None
        self._create_mode = False
        self._email.clear()
        self._password.clear()
        self._show_signed_out()
        self._set_status("")

    def _start_polling(self):
        if self._session is None:
            return
        self._stop_polling()
        self._waiting.setVisible(True)
        self._waiting.setText("Waiting for your payment to go through…")
        self._recheck_btn.setVisible(True)
        self._poll_worker = authworker.poll_for_entitlement(
            self, self._session, self._on_entitled_after_checkout,
            self._on_poll_timeout, self._on_auth_failed)
        self._poll_worker.finished.connect(self._release_poll_worker)

    def _stop_polling(self):
        if self._poll_worker is not None:
            self._poll_worker.stop()

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_signed_in(self, result, session):
        self._busy(False)
        self._password.clear()
        self._session = session
        self.apply_state(result, session)

    def _on_auth_failed(self, message):
        self._busy(False)
        self._waiting.setVisible(False)
        # An empty message means the user cancelled on purpose; saying anything
        # would be scolding them for using the Cancel button.
        self._set_status(message, error=bool(message))

    def _on_reset_sent(self):
        self._busy(False)
        # Same wording whether or not the address exists — see auth.py on
        # email-enumeration protection.
        self._set_status("If that email has an account, a reset link is on its way.")

    def _on_checkout_url(self, url):
        self._busy(False)
        QDesktopServices.openUrl(QUrl(url))
        self._start_polling()

    def _on_entitled_after_checkout(self, result, session):
        self._waiting.setVisible(False)
        self._recheck_btn.setVisible(False)
        self._session = session
        self.entitled.emit(result, session)

    def _on_poll_timeout(self):
        self._waiting.setText(
            "Still waiting. If you finished paying, press the button below.")
        self._recheck_btn.setVisible(True)

    def _release_worker(self):
        # Dropping the last reference to a live QThread from a slot is what
        # caused the beta.4 SIGABRT — see HighlightReelTab._release_worker.
        self._worker = None

    def _release_poll_worker(self):
        self._poll_worker = None

    # ── Poster ─────────────────────────────────────────────────────────────

    def _show_poster(self):
        if self._poster_src is None:
            self._poster_src = self._poster_pixmap()
        if self._poster_src is None:
            # Only reachable if the bundled file went missing.
            self._poster.setText(
                "Watch your matches in minutes, not hours.\n"
                "Point Anya at a full match; get back just the rallies.")
            return
        self._poster.setText("")
        self._rescale_poster()

    def _rescale_poster(self):
        """Fit the poster to the frame, preserving its aspect.

        Scaled from the ORIGINAL every time rather than from the last scaled
        copy: repeatedly rescaling a rescaled pixmap compounds the resampling,
        and a window dragged wider then narrower again would end up visibly
        softer than one that was never touched.
        """
        if self._poster_src is None or not hasattr(self, "_poster_frame"):
            return
        area = self._poster_frame.size()
        if area.width() < 2 or area.height() < 2:
            return
        self._poster.setPixmap(self._poster_src.scaled(
            area.width() - 2, area.height() - 2,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    @staticmethod
    def _poster_pixmap():
        import sys
        from pathlib import Path
        here = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        for candidate in (here / "assets" / "preview_poster.jpg",
                          Path(__file__).resolve().parent / "assets" / "preview_poster.jpg"):
            if candidate.is_file():
                pm = QPixmap(str(candidate))
                if not pm.isNull():
                    return pm
        return None

    def closeEvent(self, event):
        self._stop_polling()
        super().closeEvent(event)
