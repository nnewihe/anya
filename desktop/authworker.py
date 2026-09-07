"""
authworker.py — everything in auth/entitlement/functions_client, off the GUI
thread.

Same contract as update_check.UpdateChecker, deliberately: a QThread subclass
with typed pyqtSignals, and a module-level factory that STARTS the thread and
RETURNS it so the owner can hold a reference.

That last part is not tidiness. Dropping the last reference to a live QThread
from a slot is what caused the beta.4 SIGABRT — see
HighlightReelTab._release_worker and the comment at update_check.py:110. Every
worker here follows the same shape: the owner assigns it to an attribute,
connects `finished` to a slot that clears the attribute, and never touches it
after that.

Five seconds of blocking network on the main thread would freeze the window
long enough to draw the macOS beachball, and sign-in is a button press, so
this matters more here than it did for the update check.
"""

import time

from PyQt6.QtCore import QThread, pyqtSignal

import auth
import authstore
import entitlement as ent_mod
import functions_client
import oauth_loopback
from applog import logger
from authstore import Session


class _Worker(QThread):
    """Shared plumbing: one `failed(str)` signal and a blanket try/except.

    Subclasses implement `work()`. Nothing a worker does may take the app down
    — a failed sign-in is a message in the gate screen, never a traceback out
    of a thread.
    """

    failed = pyqtSignal(str)

    def run(self):
        try:
            self.work()
        except Exception as exc:  # noqa: BLE001 — see the docstring
            logger().exception("auth worker failed: %s", exc)
            self.failed.emit("Something went wrong. Please try again.")

    def work(self):
        raise NotImplementedError


def _start(worker, on_failed=None, **signals):
    """Wire signals, start, and hand the thread back to the caller to hold."""
    for name, slot in signals.items():
        if slot is not None:
            getattr(worker, name).connect(slot)
    if on_failed is not None:
        worker.failed.connect(on_failed)
    worker.finished.connect(worker.deleteLater)
    worker.start()
    return worker


def _session_from(result, previous=None):
    """Build (or update) a Session from an auth result and persist it."""
    session = Session(
        refresh_token=result["refresh_token"],
        uid=result["uid"],
        email=result["email"],
        last_id_token=result["id_token"],
        hwm=int(time.time()),
        offline_launches=0,
        **({"salt": previous.salt} if previous else {}),
    )
    authstore.save(session)
    return session


# ── Launch: do we already have an entitled session? ────────────────────────

class VerifyWorker(_Worker):
    """The launch check. Emits (EntState value, Session|None).

    Runs on every start. On success it refreshes the token, which is also how a
    subscription bought on another machine, a cancellation, or a comped grant
    reaches this install.
    """

    done = pyqtSignal(object, object)  # (Entitlement, Session|None)

    def work(self):
        session = authstore.load()
        if session is None:
            self.done.emit(ent_mod.Entitlement(ent_mod.EntState.SIGNED_OUT), None)
            return

        authstore.touch_clock(session)
        result, session = ent_mod.verify_online(session)

        if session is None:
            # Positively rejected by the server (revoked, disabled, deleted).
            # Grace does not apply to a revoked account.
            authstore.clear()
        else:
            authstore.save(session)

        self.done.emit(result, session)


def verify_entitlement(parent, on_done, on_failed=None):
    return _start(VerifyWorker(parent), on_failed, done=on_done)


# ── Sign in / sign up ──────────────────────────────────────────────────────

class PasswordWorker(_Worker):
    """Email + password, for both sign-in and account creation."""

    done = pyqtSignal(object, object)  # (Entitlement, Session)

    def __init__(self, parent, email, password, create):
        super().__init__(parent)
        self._email, self._password, self._create = email, password, create

    def work(self):
        try:
            fn = auth.sign_up if self._create else auth.sign_in
            result = fn(self._email, self._password)
        except auth.AuthError as exc:
            self.failed.emit(exc.message)
            return

        session = _session_from(result)

        if self._create:
            # Best-effort: an account that works but never got its
            # verification email is far better than a sign-up that fails at
            # the last step.
            try:
                auth.send_email_verification(result["id_token"])
            except auth.AuthError as exc:
                logger().info("could not send verification email: %s", exc.code)

        # A brand-new account has no entitlement claim yet, and a grandfathered
        # one has its grant sitting in Firestore waiting to be minted. Ask the
        # server, then refresh so the claim is actually in our token.
        try:
            functions_client.get_entitlement(result["id_token"])
            refreshed = auth.refresh(session.refresh_token)
            session = _session_from(refreshed, previous=session)
        except (functions_client.FunctionError, auth.AuthError) as exc:
            # Not fatal: the user is signed in, and the gate will show the
            # pricing rather than the app. Worth logging because a
            # grandfathered tester seeing a paywall lands here.
            logger().info("post-sign-in entitlement sync failed: %s", exc)

        self.done.emit(ent_mod.evaluate(session, online_ok=True), session)


def sign_in_with_password(parent, email, password, on_done, on_failed):
    return _start(PasswordWorker(parent, email, password, create=False),
                  on_failed, done=on_done)


def create_account(parent, email, password, on_done, on_failed):
    return _start(PasswordWorker(parent, email, password, create=True),
                  on_failed, done=on_done)


class PasswordResetWorker(_Worker):
    done = pyqtSignal()

    def __init__(self, parent, email):
        super().__init__(parent)
        self._email = email

    def work(self):
        try:
            auth.send_password_reset(self._email)
        except auth.AuthError as exc:
            # Never reveal whether the address exists: that is the same
            # account-enumeration oracle auth.py's error mapping avoids.
            logger().info("password reset returned %s", exc.code)
        self.done.emit()


def send_password_reset(parent, email, on_done):
    return _start(PasswordResetWorker(parent, email), None, done=on_done)


# ── Google ─────────────────────────────────────────────────────────────────

class GoogleWorker(_Worker):
    """Browser sign-in. Emits (Entitlement, Session)."""

    done = pyqtSignal(object, object)

    def __init__(self, parent, open_url):
        super().__init__(parent)
        self._open_url = open_url

    def work(self):
        try:
            google_token = oauth_loopback.google_id_token(self._open_url)
        except oauth_loopback.OAuthCancelled:
            self.failed.emit("")  # empty: the user meant to stop, don't scold
            return
        except oauth_loopback.OAuthError as exc:
            self.failed.emit(str(exc))
            return

        try:
            result = auth.sign_in_with_idp(
                "google.com", auth.google_post_body(google_token))
        except auth.AuthError as exc:
            self.failed.emit(exc.message)
            return

        session = _session_from(result)
        try:
            functions_client.get_entitlement(result["id_token"])
            session = _session_from(auth.refresh(session.refresh_token), previous=session)
        except (functions_client.FunctionError, auth.AuthError) as exc:
            logger().info("post-sign-in entitlement sync failed: %s", exc)

        self.done.emit(ent_mod.evaluate(session, online_ok=True), session)


def sign_in_with_google(parent, open_url, on_done, on_failed):
    return _start(GoogleWorker(parent, open_url), on_failed, done=on_done)


# ── Checkout ───────────────────────────────────────────────────────────────

class CheckoutWorker(_Worker):
    """Ask the server for a hosted Checkout URL."""

    done = pyqtSignal(str)

    def __init__(self, parent, id_token, plan):
        super().__init__(parent)
        self._id_token, self._plan = id_token, plan

    def work(self):
        try:
            url = functions_client.create_checkout_session(self._id_token, self._plan)
        except functions_client.FunctionError as exc:
            self.failed.emit(exc.message)
            return
        if not url:
            self.failed.emit("Couldn't start checkout. Please try again.")
            return
        self.done.emit(url)


def start_checkout(parent, id_token, plan, on_done, on_failed):
    return _start(CheckoutWorker(parent, id_token, plan), on_failed, done=on_done)


class CheckoutPollWorker(_Worker):
    """Wait for the entitlement claim to appear after a browser checkout.

    Polling the token-refresh endpoint, rather than a listener or a deep link.
    It works because custom claims are baked into a token when it is MINTED:
    a token issued before Stripe's webhook ran will never show `ent` no matter
    how long you hold it, and a refresh is the only thing that re-reads them.
    The latency is just the webhook round trip, typically a second or two.

    The alternatives were worse for this app. A Firestore realtime listener
    needs the gRPC client SDK, which is a large PyInstaller problem; the
    Firestore REST API has no usable long-poll; and an `anya://` URL scheme
    means registry keys on Windows and CFBundleURLTypes on macOS for something
    this achieves with one endpoint we already call.

    Long-lived, so it has an explicit stop() — the gate screen calls it from
    closeEvent, and Cancel.
    """

    done = pyqtSignal(object, object)   # (Entitlement, Session)
    timed_out = pyqtSignal()

    # Tight at first, because the webhook usually lands within seconds, then
    # backing off so a user who wandered off isn't generating a request every
    # three seconds for ten minutes.
    FAST_INTERVAL_S = 3
    SLOW_INTERVAL_S = 6
    FAST_PHASE_S = 60
    TOTAL_S = 600

    def __init__(self, parent, session):
        super().__init__(parent)
        self._session = session
        self._stop = False

    def stop(self):
        self._stop = True

    def work(self):
        started = time.time()
        while not self._stop:
            elapsed = time.time() - started
            if elapsed > self.TOTAL_S:
                self.timed_out.emit()
                return

            try:
                result = auth.refresh(self._session.refresh_token)
            except auth.AuthError as exc:
                if exc.code != "NETWORK":
                    self.failed.emit(exc.message)
                    return
                result = None   # offline: keep waiting, the browser may be fine

            if result is not None:
                session = _session_from(result, previous=self._session)
                self._session = session
                found = ent_mod.evaluate(session, online_ok=True)
                if found.allows_app:
                    logger().info("entitlement appeared after checkout")
                    self.done.emit(found, session)
                    return

            interval = (self.FAST_INTERVAL_S if elapsed < self.FAST_PHASE_S
                        else self.SLOW_INTERVAL_S)
            # Sleep in short slices so stop() and quit() are responsive; a
            # single six-second sleep would make Cancel feel broken and would
            # delay app shutdown by the same amount.
            for _ in range(interval * 10):
                if self._stop:
                    return
                self.msleep(100)


def poll_for_entitlement(parent, session, on_done, on_timeout, on_failed):
    return _start(CheckoutPollWorker(parent, session), on_failed,
                  done=on_done, timed_out=on_timeout)


# ── Account screen ─────────────────────────────────────────────────────────

class AccountWorker(_Worker):
    """One-shot callable invocations for the account dialog."""

    done = pyqtSignal(object)

    def __init__(self, parent, fn, id_token):
        super().__init__(parent)
        self._fn, self._id_token = fn, id_token

    def work(self):
        try:
            self.done.emit(self._fn(self._id_token))
        except functions_client.FunctionError as exc:
            self.failed.emit(exc.message)


def fetch_account(parent, id_token, on_done, on_failed):
    return _start(AccountWorker(parent, functions_client.get_entitlement, id_token),
                  on_failed, done=on_done)


def open_portal(parent, id_token, on_done, on_failed):
    return _start(AccountWorker(parent, functions_client.create_portal_session, id_token),
                  on_failed, done=on_done)


def cancel_and_refund(parent, id_token, on_done, on_failed):
    return _start(AccountWorker(parent, functions_client.cancel_and_refund, id_token),
                  on_failed, done=on_done)


def revoke_sessions(parent, id_token, on_done, on_failed):
    return _start(AccountWorker(parent, functions_client.revoke_sessions, id_token),
                  on_failed, done=on_done)
