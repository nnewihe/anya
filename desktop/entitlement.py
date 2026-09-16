"""
entitlement.py — is this user allowed to use the app, right now, offline?

The whole design turns on one constraint: a highlight reel takes about eleven
minutes of local computation, and the app's promise is that it runs on your
machine. A subscription check that can fail a render because a hotel wifi
dropped would be worse than no subscription check at all. So:

  * Entitlement is evaluated at LAUNCH and on the transition into the app.
    Never inside the render loop. Do not add a call to this module in
    highlight_tab._on_detect or the scoreboard render row — once the tabs are
    on screen, a render finishes.
  * A verified user keeps working offline for GRACE_DAYS after their last
    successful check.

Where the grace window starts is the interesting part. It is NOT a timestamp
this app writes down — that would be a number in a JSON file, and moving it
forward would extend access indefinitely. It is the `iat` of the last Firebase
ID token, stamped by Google's token service and carried inside a signed JWT.
The app cannot mint one, so it cannot move the start of its own window. (The
token's own `exp` is one hour, which is why grace cannot simply be "is the
token still valid".)

Three further bounds close the obvious gaps:

  * `hwm` — the highest wall clock ever seen. Setting the clock back below it
    fails closed. (Setting it forward only shortens grace, so it is not an
    attack, and we do not care.)
  * `offline_launches` — a count, which no clock can lie about, so the window
    is bounded in usage as well as in time.
  * `entExp` — the entitlement's own expiry from the custom claim. Grace
    extends a lapsed check, never a lapsed subscription beyond its own slack.

None of this is DRM, and it is not trying to be. See authstore.py's docstring.
"""

import enum
import time

from auth import AuthError, decode_jwt_payload, refresh, token_looks_like_ours
from applog import logger

GRACE_DAYS = 14
GRACE_SECONDS = GRACE_DAYS * 24 * 60 * 60

# Small tolerance so ordinary NTP corrections and daylight-saving oddities
# don't read as a rollback. Five minutes is far more than any legitimate
# adjustment and far less than any useful cheat.
CLOCK_SKEW_TOLERANCE_S = 300

# Belt to the clock's braces. Thirty launches is more than a fortnight of
# ordinary use, so a genuinely offline subscriber will never reach it.
MAX_OFFLINE_LAUNCHES = 30


class EntState(enum.Enum):
    SIGNED_OUT = "signed_out"      # no session at all
    UNENTITLED = "unentitled"      # signed in, never subscribed
    ENTITLED = "entitled"          # verified online, currently paid
    GRACE = "grace"                # can't reach the server, still inside grace
    EXPIRED = "expired"            # subscription lapsed, or grace ran out

    @property
    def allows_app(self):
        return self in (EntState.ENTITLED, EntState.GRACE)


class Entitlement:
    """The answer, plus enough context for the UI to say something useful."""

    def __init__(self, state, plan=None, expires_at=None, reason=""):
        self.state = state
        self.plan = plan                # "a" | "m" | None
        self.expires_at = expires_at    # entExp, unix seconds
        self.reason = reason            # short, for logs and the gate copy

    @property
    def allows_app(self):
        return self.state.allows_app

    def __repr__(self):
        return f"<Entitlement {self.state.value} plan={self.plan} reason={self.reason!r}>"


def claims_of(session):
    """Decoded claims of the session's cached ID token, or None.

    Returns None for a missing, malformed, or foreign token — the caller
    treats all three as "we have nothing to go on".
    """
    if not session or not session.last_id_token:
        return None
    try:
        claims = decode_jwt_payload(session.last_id_token)
    except ValueError:
        logger().info("cached ID token is not decodable")
        return None
    if not token_looks_like_ours(claims, uid=session.uid):
        logger().info("cached ID token is not for this project/user")
        return None
    return claims


def evaluate(session, now=None, online_ok=False):
    """Decide what a session is entitled to. Pure — no network, no clock reads
    beyond `now`, no I/O. This is the function the tests exercise exhaustively.

    `online_ok` says whether the token in `session` was just refreshed against
    Google. When it is True the answer is simply whatever the fresh claim says.
    When it is False we are working from a cached token and the grace rules
    below apply.
    """
    now = int(now if now is not None else time.time())

    if session is None or not session.refresh_token:
        return Entitlement(EntState.SIGNED_OUT, reason="no session")

    claims = claims_of(session)
    if claims is None:
        # Signed in as far as we know, but with nothing to prove entitlement.
        return Entitlement(EntState.UNENTITLED, reason="no usable token")

    ent = claims.get("ent")
    ent_exp = claims.get("entExp")
    plan = claims.get("pl")
    iat = claims.get("iat")

    if not ent or not isinstance(ent_exp, (int, float)):
        return Entitlement(EntState.UNENTITLED, plan=plan, reason="no entitlement claim")
    ent_exp = int(ent_exp)

    # A subscription that has run out is expired whether we are online or not.
    # entExp already carries a few days of issuer slack (see functions/), so
    # there is no need to be generous a second time here.
    if now >= ent_exp:
        return Entitlement(EntState.EXPIRED, plan, ent_exp, reason="subscription lapsed")

    if online_ok:
        return Entitlement(EntState.ENTITLED, plan, ent_exp, reason="verified online")

    # ── Offline: the four bounds ───────────────────────────────────────────
    if not isinstance(iat, (int, float)):
        # Every Firebase token has an iat; one without it is not something to
        # extend trust to.
        return Entitlement(EntState.EXPIRED, plan, ent_exp, reason="token has no iat")

    if now < session.hwm - CLOCK_SKEW_TOLERANCE_S:
        logger().warning(
            "system clock is %d s behind the highest previously seen time; "
            "requiring an online check", session.hwm - now,
        )
        return Entitlement(EntState.EXPIRED, plan, ent_exp, reason="clock rolled back")

    if session.offline_launches >= MAX_OFFLINE_LAUNCHES:
        return Entitlement(EntState.EXPIRED, plan, ent_exp, reason="too many offline launches")

    if now >= int(iat) + GRACE_SECONDS:
        return Entitlement(EntState.EXPIRED, plan, ent_exp, reason="grace window elapsed")

    if now >= ent_exp + GRACE_SECONDS:
        # Unreachable while the `now >= ent_exp` check above stands, and kept
        # deliberately: it is the invariant that grace may extend a missed
        # CHECK but never a lapsed SUBSCRIPTION, and it must survive anyone
        # loosening the earlier branch.
        return Entitlement(EntState.EXPIRED, plan, ent_exp, reason="past entitlement grace")

    days_left = (int(iat) + GRACE_SECONDS - now) // 86400
    return Entitlement(
        EntState.GRACE, plan, ent_exp,
        reason=f"offline, {days_left} day(s) of grace left",
    )


def verify_online(session, opener=None, now=None):
    """Refresh the token against Google, then evaluate.

    Returns (entitlement, session). The session comes back mutated with the
    new tokens and an updated clock high-water mark, and the caller is
    expected to authstore.save() it. On a network failure the session is
    returned with offline_launches incremented and the cached token intact —
    which is exactly the input evaluate() needs to apply the grace rules.
    """
    now = int(now if now is not None else time.time())

    if session is None or not session.refresh_token:
        return Entitlement(EntState.SIGNED_OUT, reason="no session"), session

    try:
        fresh = refresh(session.refresh_token, opener=opener)
    except AuthError as exc:
        if exc.code == "NETWORK":
            session.offline_launches += 1
            session.hwm = max(session.hwm, now)
            ent = evaluate(session, now=now, online_ok=False)
            logger().info(
                "entitlement check offline (attempt %d): %s",
                session.offline_launches, ent.reason,
            )
            return ent, session
        # TOKEN_EXPIRED, USER_NOT_FOUND, USER_DISABLED: the server has
        # positively rejected this session. Grace does not apply to a
        # revoked account — that is the whole point of being able to revoke.
        logger().info("refresh token rejected (%s); signing out", exc.code)
        return Entitlement(EntState.SIGNED_OUT, reason=f"rejected: {exc.code}"), None

    session.refresh_token = fresh["refresh_token"]
    session.last_id_token = fresh["id_token"]
    session.uid = fresh["uid"] or session.uid
    session.email = fresh["email"] or session.email
    session.offline_launches = 0
    session.hwm = max(session.hwm, now)

    ent = evaluate(session, now=now, online_ok=True)
    logger().info("entitlement verified online: %s", ent.state.value)
    return ent, session
