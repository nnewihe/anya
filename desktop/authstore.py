"""
authstore.py — the signed-in session, persisted between launches.

The one place in the app that reads or writes credentials. Everything else
goes through load() / save() / clear(), so swapping the file for the macOS
Keychain later is a change to this module and nothing else.

Why a file and not the Keychain today: `keyring` would add a dependency to
requirements.txt AND constraints-windows.txt, four backend hiddenimports in
rally_app.spec, and a transitive pywin32-ctypes on Windows — for a secret whose
entire blast radius is one $40/yr subscription, and which the server can revoke
(see "sign out everywhere"). The security win that actually mattered was
keeping this out of log_dir(): testers are told to email app.log, and a
refresh token in a directory we ask people to mail out is a token we gave away.
See applog.app_data_dir().

On tamper-resistance, plainly: this is a Python app in a PyInstaller bundle.
Anyone willing to unpack the PYZ can delete the entitlement check in ten
minutes, and no amount of cleverness here changes that. The HMAC below is not
DRM. It exists to stop the one thing that really happens — someone opening
session.json in a text editor and changing a date or a `true` — and to make a
corrupted file fail closed rather than silently granting access. The goal is
that paying is easier than not paying.
"""

import hashlib
import hmac
import json
import os
import secrets
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from applog import app_data_dir, logger

_FILENAME = "session.json"
_VERSION = 1

# Not a secret — it is compiled into a file anyone can unpack, and the module
# docstring says so.
#
# It is mixed with a per-install salt stored in the same file. Be clear about
# what that does and does not buy: the salt does NOT stop someone copying a
# whole session.json to another machine (the tag travels with it and still
# validates), and it does not stop anyone who reads this module from
# recomputing a tag over edited content. What the tag catches is a file edited
# by hand and left with its old tag, and a file half-written or corrupted on
# disk — both of which then fail closed instead of being read as an
# entitlement. Sharing an account is a server-side problem, addressed by
# revoking the refresh token, not by anything in this file.
_TAG_KEY = b"anya-tennis/session-integrity/v1"


@dataclass
class Session:
    """Everything needed to resume a signed-in state offline.

    `last_id_token` is kept whole and verbatim rather than picked apart,
    because its `iat` is a server-stamped timestamp: it is what makes the
    offline grace window start at a moment the user cannot move by editing a
    local file. See entitlement.evaluate().
    """

    refresh_token: str
    uid: str
    email: str = ""
    last_id_token: str = ""
    # Highest wall-clock second ever observed. Rolling the system clock back
    # below this is a tripwire, not a way to extend anything.
    hwm: int = 0
    # Consecutive launches where the online check failed. Bounds the grace
    # window in usage as well as in time — a lying clock cannot fake this.
    offline_launches: int = 0
    salt: str = field(default_factory=lambda: secrets.token_hex(16))

    def redacted(self):
        """Safe to log: identifies the session without carrying a credential."""
        return {
            "uid": self.uid,
            "email": self.email,
            "hwm": self.hwm,
            "offline_launches": self.offline_launches,
            "has_refresh_token": bool(self.refresh_token),
        }


def session_path() -> Path:
    return app_data_dir() / _FILENAME


# ── Integrity tag ──────────────────────────────────────────────────────────

def _canonical(payload: dict) -> bytes:
    """Deterministic bytes for the tag: sorted keys, no whitespace, no tag."""
    body = {k: v for k, v in payload.items() if k != "tag"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tag(payload: dict) -> str:
    salt = (payload.get("salt") or "").encode("utf-8")
    key = hashlib.sha256(_TAG_KEY + salt).digest()
    return hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()


# ── Load / save / clear ────────────────────────────────────────────────────

def load():
    """The stored Session, or None.

    None for every uninteresting case alike: no file, unreadable, truncated,
    wrong version, failed integrity check. The caller's response to all of
    them is identical — show the sign-in screen — and distinguishing them
    would only give a tamperer a diagnostic.
    """
    path = session_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger().info("session file unreadable (%s); signing out", exc)
        return None

    if not isinstance(raw, dict) or raw.get("v") != _VERSION:
        logger().info("session file has unexpected version; signing out")
        return None

    stored_tag = raw.get("tag") or ""
    if not hmac.compare_digest(stored_tag, _tag(raw)):
        logger().warning("session file failed its integrity check; signing out")
        return None

    fields = {f for f in Session.__dataclass_fields__}
    try:
        return Session(**{k: v for k, v in raw.items() if k in fields})
    except TypeError as exc:
        logger().info("session file missing required fields (%s); signing out", exc)
        return None


def save(session: Session) -> bool:
    """Write atomically with 0600. Returns False rather than raising.

    A machine where this cannot be written (read-only home, locked-down
    permissions) must still be able to *use* the app for this launch — it just
    won't stay signed in. Same best-effort contract as applog's file handler.
    """
    payload = asdict(session)
    payload["v"] = _VERSION
    payload["tag"] = _tag(payload)
    blob = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")

    path = session_path()
    tmp_fd = tmp_name = None
    try:
        # Create the temp file in the destination directory so os.replace is a
        # rename within one filesystem, and open it 0600 from the start — a
        # chmod after the write leaves a window where the token is readable.
        tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".session-")
        os.fchmod(tmp_fd, 0o600)
        with os.fdopen(tmp_fd, "wb") as fh:
            tmp_fd = None  # fdopen owns it now
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
        return True
    except OSError as exc:
        logger().warning("could not save session: %s", exc)
        return False
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def clear() -> None:
    try:
        session_path().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger().warning("could not remove session file: %s", exc)


def touch_clock(session: Session, now=None) -> Session:
    """Advance the high-water mark. Call on every launch and every verify.

    Moving the clock forward only ever shortens the grace window, so there is
    nothing to defend against there; this exists so that moving it *back* is
    detectable.
    """
    now = int(now if now is not None else time.time())
    session.hwm = max(session.hwm, now)
    return session
