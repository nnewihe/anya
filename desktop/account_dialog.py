"""
account_dialog.py — the signed-in account: plan, renewal, refund, sign out.

Everything money-related that is not the gate. Deliberately a small dialog
rather than a third tab: it is somewhere you go once a year, and a tab would
put billing permanently beside the two things the app is actually for.

Stripe's own hosted billing portal does the heavy lifting for card changes,
invoices and cancel-at-period-end, so the only thing built here is the piece
Stripe has no opinion about: the one-time 14-day full refund.

That button appears only when the server says it should. The server checks the
same conditions again when it is pressed — hiding it is a courtesy to the user,
not the enforcement. See functions/src/callables.ts.
"""

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
)

import authstore
import authworker
import firebase_config as cfg
from theme import SURFACE, TEXT_DIM, WHITE, YELLOW, danger_btn_css, ghost_btn_css


def _date(unix_seconds):
    if not unix_seconds:
        return "—"
    import datetime
    return datetime.datetime.fromtimestamp(int(unix_seconds)).strftime("%-d %B %Y")


_PLAN_NAMES = {"annual": "Annual", "monthly": "Monthly"}
_SOURCE_NAMES = {
    "grandfathered": "Beta tester — free for a year",
    "comp": "Complimentary",
}


class AccountDialog(QDialog):
    """`signed_out` is checked by the caller after exec() to know whether to
    drop back to the gate."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self._session = session
        self._worker = None
        self.signed_out = False

        self.setWindowTitle("Anya Tennis — Account")
        self.setMinimumWidth(430)
        self._setup_ui()
        self._load()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(10)

        email = QLabel(self._session.email or "Signed in")
        email.setStyleSheet(f"color: {WHITE}; font-size: 15px; font-weight: 700;")
        lay.addWidget(email)

        self._plan = QLabel("Loading your subscription…")
        self._plan.setWordWrap(True)
        self._plan.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        lay.addWidget(self._plan)

        lay.addSpacing(8)

        self._portal_btn = QPushButton("MANAGE SUBSCRIPTION")
        self._portal_btn.setStyleSheet(ghost_btn_css())
        self._portal_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._portal_btn.setFixedHeight(36)
        self._portal_btn.setToolTip(
            "Opens Stripe in your browser to change your card, see invoices, "
            "or cancel at the end of the period.")
        self._portal_btn.clicked.connect(self._open_portal)
        lay.addWidget(self._portal_btn)

        # Built hidden and added in place, so it can appear without reflowing
        # the dialog once the server answers — same reasoning as the update
        # banner in app.py.
        self._refund_btn = QPushButton("CANCEL & REFUND")
        self._refund_btn.setStyleSheet(danger_btn_css())
        self._refund_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refund_btn.setFixedHeight(36)
        self._refund_btn.setVisible(False)
        self._refund_btn.clicked.connect(self._refund)
        lay.addWidget(self._refund_btn)

        self._refund_note = QLabel("")
        self._refund_note.setWordWrap(True)
        self._refund_note.setVisible(False)
        self._refund_note.setStyleSheet(f"color: {YELLOW}; font-size: 11px;")
        lay.addWidget(self._refund_note)

        lay.addSpacing(6)

        row = QHBoxLayout()
        signout = QPushButton("SIGN OUT")
        signout.setStyleSheet(ghost_btn_css())
        signout.setCursor(Qt.CursorShape.PointingHandCursor)
        signout.clicked.connect(self._sign_out)
        row.addWidget(signout)

        everywhere = QPushButton("SIGN OUT EVERYWHERE")
        everywhere.setStyleSheet(ghost_btn_css())
        everywhere.setCursor(Qt.CursorShape.PointingHandCursor)
        everywhere.setToolTip(
            "Signs you out on every computer. Use this if you think someone "
            "else has access to your account.")
        everywhere.clicked.connect(self._sign_out_everywhere)
        row.addWidget(everywhere)
        row.addStretch()

        close = QPushButton("CLOSE")
        close.setStyleSheet(ghost_btn_css())
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        lay.addWidget(self._status)

        self.setStyleSheet(f"QDialog {{ background: {SURFACE}; }}")

    # ── Loading ────────────────────────────────────────────────────────────

    def _load(self):
        self._worker = authworker.fetch_account(
            self, self._session, self._on_loaded, self._on_failed)
        self._worker.finished.connect(self._release_worker)

    def _on_loaded(self, info):
        source = (info.get("entitlement") or {}).get("source")
        plan = info.get("plan")
        until = (info.get("entitlement") or {}).get("until")

        if source in _SOURCE_NAMES:
            # A grandfathered or comped account has no Stripe subscription, so
            # "Manage subscription" would open a portal with nothing in it.
            self._plan.setText(f"{_SOURCE_NAMES[source]} · through {_date(until)}")
            self._portal_btn.setVisible(False)
        else:
            name = _PLAN_NAMES.get(plan, "Subscription")
            status = info.get("status") or ""
            renew = ("Ends" if info.get("cancelAtPeriodEnd") else "Renews")
            line = f"{name} · {renew} {_date(info.get('currentPeriodEnd'))}"
            if status == "past_due":
                line += "  ·  There's a problem with your card — please update it."
            self._plan.setText(line)

        refund = info.get("refund") or {}
        if refund.get("eligible"):
            self._refund_btn.setVisible(True)
            self._refund_note.setVisible(True)
            self._refund_note.setText(
                f"You can cancel for a full refund until {_date(refund.get('deadline'))}. "
                "This offer can only be used once.")
        elif refund.get("reason") == "already_used":
            self._refund_note.setVisible(True)
            self._refund_note.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
            self._refund_note.setText("The one-time refund has already been used on this account.")

    def _on_failed(self, message):
        self._plan.setText(
            "Couldn't load your subscription details. You can still use the app.")
        self._status.setText(message)

    # ── Actions ────────────────────────────────────────────────────────────

    def _open_portal(self):
        self._status.setText("Opening Stripe in your browser…")
        # functions_client.create_portal_session already unwraps the response
        # and hands back the URL STRING -- it is the one callable that does,
        # while cancel_and_refund, revoke_sessions and get_entitlement all
        # return the whole dict. Unwrapping it a second time here is what threw
        # "'str' object has no attribute 'get'" the first time anyone pressed
        # Manage Subscription against a real subscription.
        self._worker = authworker.open_portal(
            self, self._session, self._open, self._on_action_failed)
        self._worker.finished.connect(self._release_worker)

    def _open(self, url):
        self._status.setText("")
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _refund(self):
        confirm = QMessageBox(self)
        confirm.setWindowTitle("Cancel and refund")
        confirm.setText(
            "Cancel your subscription and refund what you paid?\n\n"
            "Your subscription ends straight away and the full amount goes back "
            "to your card, usually within a few business days.\n\n"
            "This offer can only be used once — if you subscribe again later, "
            "it won't be available.")
        confirm.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Ok)
        confirm.setDefaultButton(QMessageBox.StandardButton.Cancel)
        confirm.button(QMessageBox.StandardButton.Ok).setText("Cancel & refund")
        if confirm.exec() != QMessageBox.StandardButton.Ok:
            return

        self._refund_btn.setEnabled(False)
        self._status.setText("Processing your refund…")
        self._worker = authworker.cancel_and_refund(
            self, self._session, self._on_refunded, self._on_refund_failed)
        self._worker.finished.connect(self._release_worker)

    def _on_refunded(self, _result):
        QMessageBox.information(
            self, "Refunded",
            "Your subscription has been cancelled and the payment refunded.\n\n"
            "It usually reaches your card within a few business days. "
            "Thanks for giving Anya a try.")
        self._finish_sign_out()

    def _on_refund_failed(self, message):
        self._refund_btn.setEnabled(True)
        self._status.setText(message)

    def _on_action_failed(self, message):
        self._status.setText(message)

    def _sign_out(self):
        self._finish_sign_out()

    def _sign_out_everywhere(self):
        self._status.setText("Signing out everywhere…")
        self._worker = authworker.revoke_sessions(
            self, self._session,
            lambda _r: self._finish_sign_out(), self._on_action_failed)
        self._worker.finished.connect(self._release_worker)

    def _finish_sign_out(self):
        authstore.clear()
        self.signed_out = True
        self.accept()

    def _release_worker(self):
        # See gate_screen and HighlightReelTab._release_worker: dropping a live
        # QThread's last reference from a slot is fatal, not untidy.
        self._worker = None
