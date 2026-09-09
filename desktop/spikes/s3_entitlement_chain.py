"""
S3 — webhook -> custom claim -> refreshed ID token that carries `ent`.

The whole paid product rests on one mechanism: Stripe tells the webhook that
money moved, the webhook mints a custom claim, and the desktop app sees that
claim only after it REFRESHES its token. If any link is wrong, a paying
customer stares at a paywall.

This drives the real desktop client (auth.py, functions_client.py,
entitlement.py) and the real Cloud Functions against the Firebase emulator
suite. What it does not use is a real Stripe account: instead it posts a
correctly SIGNED synthetic `customer.subscription.updated` event, which is the
same bytes Stripe would send and exercises signature verification, the
idempotency lock, applyEntitlement, and setCustomUserClaims.

That is deliberate rather than lazy. The event carries `metadata.uid`, so
`uidFor()` resolves without calling out to Stripe, and the handler never
touches the network. It means this spike proves the entitlement chain without
anyone's live keys -- and leaves exactly one thing for the real Stripe account
to prove, which is that Checkout produces such an event in the first place.

Prerequisites, in another terminal:

    cd functions && npm run build
    firebase emulators:start --only functions,firestore,auth --project demo-anya

Then:

    python3 spikes/s3_entitlement_chain.py
"""

import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --real drives the LIVE Firebase project instead of the emulator. It runs
# only what does not depend on deployed Cloud Functions -- real sign-up, real
# token refresh, and the real Firestore rules -- and says plainly what it is
# skipping. Worth running even so: it is the only thing that checks the rules
# actually deployed, and the emulator has been wrong about rules before.
REAL = "--real" in sys.argv

PROJECT = os.environ.get("ANYA_SPIKE_PROJECT", "demo-anya")
AUTH_HOST = os.environ.get("ANYA_AUTH_EMULATOR_HOST", "127.0.0.1:9099")
FUNCTIONS_HOST = os.environ.get("ANYA_FUNCTIONS_EMULATOR_HOST", "127.0.0.1:5001")
FIRESTORE_HOST = os.environ.get("FIRESTORE_EMULATOR_HOST", "127.0.0.1:8080")
REGION = "us-central1"
WEBHOOK_SECRET = os.environ.get(
    "STRIPE_WEBHOOK_SECRET",
    "whsec_PLACEHOLDER_replace_via_STRIPE_SETUP_md" if "--real" in sys.argv
    else "whsec_spike_secret",
)

# Point the client modules at the emulator BEFORE importing them: both read
# their environment at import time. Under --real, leave the environment alone
# so firebase_config's own committed defaults (the live project) apply.
if not REAL:
    os.environ["ANYA_AUTH_EMULATOR_HOST"] = AUTH_HOST
    os.environ["ANYA_FIREBASE_PROJECT"] = PROJECT
    os.environ["ANYA_FIREBASE_API_KEY"] = "emulator-key"   # emulator ignores it
    os.environ["ANYA_FUNCTIONS_BASE"] = f"http://{FUNCTIONS_HOST}/{PROJECT}/{REGION}"

import auth                 # noqa: E402
import firebase_config as cfg  # noqa: E402
import entitlement as ent   # noqa: E402
import functions_client     # noqa: E402
from authstore import Session  # noqa: E402

DAY = 86_400
PASS, FAIL = "  \033[32mPASS\033[0m", "  \033[31mFAIL\033[0m"
_failures = []


def check(label, ok, detail=""):
    print(f"{PASS if ok else FAIL}  {label}{('  — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(label)
    return ok


def step(text):
    print(f"\n\033[1m{text}\033[0m")


# ── A synthetic Stripe event, signed the way Stripe signs ──────────────────

def signed_webhook(event, secret=WEBHOOK_SECRET):
    """Post an event to the emulated stripeWebhook with a valid signature.

    Stripe's scheme: sign "<timestamp>.<raw body>" with HMAC-SHA256 and send it
    as `Stripe-Signature: t=<ts>,v1=<hex>`. Reproducing it here is what proves
    the handler's use of req.rawBody is right -- a JSON round-trip anywhere in
    the path would change the bytes and break the signature, which is exactly
    the bug this guards against.
    """
    body = json.dumps(event, separators=(",", ":")).encode()
    ts = int(time.time())
    mac = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()

    req = urllib.request.Request(
        f"http://{FUNCTIONS_HOST}/{PROJECT}/{REGION}/stripeWebhook",
        data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Stripe-Signature": f"t={ts},v1={mac}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def subscription_event(uid, *, status="active", period_end=None, event_id=None,
                       price="price_annual_spike"):
    period_end = period_end or int(time.time()) + 365 * DAY
    return {
        "id": event_id or f"evt_{secrets.token_hex(8)}",
        "object": "event",
        "created": int(time.time()),
        "type": "customer.subscription.updated",
        "data": {"object": {
            "id": "sub_spike_1",
            "object": "subscription",
            "status": status,
            # metadata.uid is what subscription_data.metadata sets at checkout;
            # with it present uidFor() never has to call Stripe.
            "metadata": {"uid": uid},
            "customer": "cus_spike_1",
            "current_period_end": period_end,
            "cancel_at_period_end": False,
            "items": {"object": "list", "data": [
                {"id": "si_1", "price": {"id": price, "object": "price"}}]},
        }},
    }


def firestore_request(path, id_token, method="GET", body=None):
    """Talk to REAL Firestore as the signed-in user, so the deployed rules --
    not the emulator's copy of them -- are what answers."""
    url = (f"https://firestore.googleapis.com/v1/projects/{cfg.PROJECT_ID}"
           f"/databases/(default)/documents/{path}")
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {id_token}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code


def delete_account(id_token):
    req = urllib.request.Request(
        f"{cfg.IDENTITY_BASE}:delete?key={cfg.WEB_API_KEY}",
        data=json.dumps({"idToken": id_token}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=20)
        return True
    except Exception:
        return False


def main_real():
    """Everything provable against the live project before functions deploy."""
    print(f"S3 --real: live project {cfg.PROJECT_ID}")
    print(f"    auth={cfg.IDENTITY_BASE}")
    email = f"spike-{secrets.token_hex(4)}@example.com"

    step("1. Real sign-up against Google's Identity Toolkit")
    created = auth.sign_up(email, "spike-password-123")
    uid = created["uid"]
    check("signUp returned a session", bool(created["id_token"]), f"uid={uid}")

    session = Session(refresh_token=created["refresh_token"], uid=uid,
                      email=created["email"], last_id_token=created["id_token"],
                      hwm=int(time.time()))
    claims = auth.decode_jwt_payload(session.last_id_token)
    check("the token is structurally ours",
          auth.token_looks_like_ours(claims, uid),
          f"iss={claims.get('iss')}")
    check("a new account carries no entitlement claim", "ent" not in claims)
    check("evaluate() reads that as UNENTITLED",
          ent.evaluate(session, online_ok=True).state is ent.EntState.UNENTITLED)

    step("2. Real token refresh (the mechanism the checkout poll rides on)")
    before = session.refresh_token
    refreshed = auth.refresh(session.refresh_token)
    check("refresh returned a new ID token", bool(refreshed["id_token"]))
    check("uid survives the refresh", refreshed["uid"] == uid)
    check("the snake_case response normalised correctly",
          refreshed["email"] == email, refreshed["email"])
    session.refresh_token = refreshed["refresh_token"]
    session.last_id_token = refreshed["id_token"]

    step("3. The DEPLOYED Firestore rules")
    own = firestore_request(f"users/{uid}", session.last_id_token)
    check("a user may READ its own document", own in (200, 404),
          f"HTTP {own} (404 = permitted, document not created yet)")

    wrote = firestore_request(
        f"users?documentId={uid}", session.last_id_token, "POST",
        {"fields": {"entitlement": {"mapValue": {"fields": {
            "active": {"booleanValue": True}}}}}})
    check("a user may NOT write its own document", wrote == 403,
          f"HTTP {wrote} — this is what stops a patched client granting itself a year")

    other = firestore_request("users/some-other-uid", session.last_id_token)
    check("a user may not read ANOTHER user's document", other == 403, f"HTTP {other}")

    gf = firestore_request("grandfathered/deadbeef", session.last_id_token)
    check("the grandfathered allowlist is unreadable", gf == 403, f"HTTP {gf}")

    ev = firestore_request("stripeEvents/evt_x", session.last_id_token)
    check("the webhook idempotency log is unreadable", ev == 403, f"HTTP {ev}")

    step("4. The DEPLOYED callables")
    try:
        info = functions_client.get_entitlement(session.last_id_token)
        check("getEntitlement answered", isinstance(info, dict), json.dumps(info)[:100])
        check("reports not entitled", not (info.get("entitlement") or {}).get("active"))
        check("no refund offered without a payment",
              not (info.get("refund") or {}).get("eligible"),
              (info.get("refund") or {}).get("reason"))
    except functions_client.FunctionError as exc:
        check("getEntitlement answered", False, f"{exc.status}: {exc.message}")

    step("5. The DEPLOYED webhook -> claim -> refreshed token")
    # Signed with whatever STRIPE_WEBHOOK_SECRET the deploy is carrying. While
    # that is still the placeholder this proves the whole chain without a
    # Stripe account; once the real secret is set, rerun and it proves it with
    # Stripe's own signature.
    hook = (f"https://{REGION}-{cfg.PROJECT_ID}.cloudfunctions.net/stripeWebhook")
    body = json.dumps(subscription_event(uid), separators=(",", ":")).encode()
    ts = int(time.time())
    mac = hmac.new(WEBHOOK_SECRET.encode(), f"{ts}.".encode() + body,
                   hashlib.sha256).hexdigest()
    req = urllib.request.Request(
        hook, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Stripe-Signature": f"t={ts},v1={mac}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            status, text = r.status, r.read().decode()
    except urllib.error.HTTPError as exc:
        status, text = exc.code, exc.read().decode()
    check("signed event accepted in production", status == 200, f"HTTP {status} {text[:60]}")

    got = {}
    for attempt in range(30):
        refreshed = auth.refresh(session.refresh_token)
        session.refresh_token = refreshed["refresh_token"]
        session.last_id_token = refreshed["id_token"]
        got = auth.decode_jwt_payload(session.last_id_token)
        if got.get("ent"):
            break
        time.sleep(1)
    check("a refreshed token carries ent=1", got.get("ent") == 1,
          f"after {attempt + 1} refresh(es)")
    check("and entExp", isinstance(got.get("entExp"), int), f"entExp={got.get('entExp')}")
    check("evaluate() now unlocks the app",
          ent.evaluate(session, online_ok=True).allows_app)

    step("6. Cleanup")
    check("probe account deleted", delete_account(session.last_id_token))
    print(f"    (removing Firestore docs for {uid} — needs the firebase CLI)")
    import subprocess
    for path in (f"users/{uid}",):
        subprocess.run(["firebase", "firestore:delete", path, "--force",
                        "--project", cfg.PROJECT_ID],
                       capture_output=True, timeout=120)
    print("    done")

    print()
    if _failures:
        print(f"\033[31mS3 --real FAIL — {len(_failures)} check(s):\033[0m")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("\033[32mS3 --real PASS\033[0m — the whole chain, in production.")
    print("Signed webhook -> custom claim -> refreshed token -> unlocked app, against")
    print("the live project, the deployed rules and the deployed functions.")
    print()
    print("The one thing still standing in for Stripe is the SIGNATURE: the event is")
    print("signed with whatever STRIPE_WEBHOOK_SECRET the deploy carries. Set the real")
    print("one (functions/STRIPE_SETUP.md) and rerun, and even that is Stripe's own.")
    return 0


def main():
    if REAL:
        return main_real()
    email = f"spike-{secrets.token_hex(4)}@example.com"
    print(f"S3: entitlement chain against the emulator  (project={PROJECT})")
    print(f"    auth={AUTH_HOST}  functions={FUNCTIONS_HOST}  user={email}")

    # ── 1. Sign up through the real client ─────────────────────────────────
    step("1. Create an account (desktop/auth.py -> Identity Toolkit)")
    created = auth.sign_up(email, "spike-password-123")
    uid = created["uid"]
    check("signUp returned a session", bool(created["id_token"] and created["refresh_token"]),
          f"uid={uid}")

    session = Session(refresh_token=created["refresh_token"], uid=uid,
                      email=created["email"], last_id_token=created["id_token"],
                      hwm=int(time.time()))

    claims = auth.decode_jwt_payload(session.last_id_token)
    check("a brand-new token carries NO entitlement claim", "ent" not in claims,
          f"claims={sorted(claims)}")
    check("evaluate() reads that as UNENTITLED",
          ent.evaluate(session, online_ok=True).state is ent.EntState.UNENTITLED)

    # ── 2. getEntitlement is callable and honest ───────────────────────────
    step("2. getEntitlement (functions_client -> onCall)")
    try:
        info = functions_client.get_entitlement(session.last_id_token)
        check("callable answered", isinstance(info, dict), json.dumps(info)[:110])
        check("reports not entitled", not (info.get("entitlement") or {}).get("active"))
        check("refund not offered without a payment",
              not (info.get("refund") or {}).get("eligible"),
              (info.get("refund") or {}).get("reason"))
    except functions_client.FunctionError as exc:
        check("callable answered", False, f"{exc.status}: {exc.message}")

    # ── 3. The webhook ─────────────────────────────────────────────────────
    step("3. Stripe webhook -> applyEntitlement -> setCustomUserClaims")
    status, body = signed_webhook(subscription_event(uid))
    check("signed event accepted", status == 200, f"HTTP {status} {body[:70]}")

    bad = json.dumps(subscription_event(uid)).encode()
    req = urllib.request.Request(
        f"http://{FUNCTIONS_HOST}/{PROJECT}/{REGION}/stripeWebhook", data=bad,
        method="POST", headers={"Content-Type": "application/json",
                                "Stripe-Signature": "t=1,v1=deadbeef"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            code = r.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    check("UNSIGNED event rejected", code == 400, f"HTTP {code}")

    # ── 4. The claim only appears after a refresh ──────────────────────────
    step("4. The claim reaches the client only via a token refresh")
    stale = auth.decode_jwt_payload(session.last_id_token)
    check("the token held since sign-up still shows nothing", "ent" not in stale,
          "claims are baked in at mint time — this is why the app polls")

    got = None
    for attempt in range(20):
        refreshed = auth.refresh(session.refresh_token)
        session.refresh_token = refreshed["refresh_token"]
        session.last_id_token = refreshed["id_token"]
        got = auth.decode_jwt_payload(session.last_id_token)
        if got.get("ent"):
            break
        time.sleep(0.5)

    check("a REFRESHED token carries ent=1", got.get("ent") == 1,
          f"after {attempt + 1} refresh(es)")
    check("it carries entExp", isinstance(got.get("entExp"), int),
          f"entExp={got.get('entExp')}")
    check("it carries the plan letter", got.get("pl") in ("a", "m"),
          f"pl={got.get('pl')}")
    check("the claim payload stays well under the 1000-byte cap",
          len(json.dumps({k: got[k] for k in ("ent", "entExp", "pl") if k in got})) < 100)

    # ── 5. The client agrees ───────────────────────────────────────────────
    step("5. entitlement.evaluate() on the refreshed session")
    result = ent.evaluate(session, online_ok=True)
    check("state is ENTITLED", result.state is ent.EntState.ENTITLED, result.reason)
    check("the app would unlock", result.allows_app)

    now = int(time.time())
    offline = ent.evaluate(session, now=now + 13 * DAY, online_ok=False)
    check("13 days offline -> GRACE", offline.state is ent.EntState.GRACE, offline.reason)
    late = ent.evaluate(session, now=now + 15 * DAY, online_ok=False)
    check("15 days offline -> EXPIRED", late.state is ent.EntState.EXPIRED, late.reason)

    # ── 6. Idempotency and revocation ──────────────────────────────────────
    step("6. Idempotency and revocation")
    evt = subscription_event(uid, event_id="evt_replay_me")
    first = signed_webhook(evt)[0]
    second = signed_webhook(evt)[0]
    check("a replayed delivery is a no-op, still 200",
          first == 200 and second == 200, f"{first} then {second}")

    signed_webhook(subscription_event(uid, status="canceled"))
    for attempt in range(20):
        refreshed = auth.refresh(session.refresh_token)
        session.refresh_token = refreshed["refresh_token"]
        session.last_id_token = refreshed["id_token"]
        after = auth.decode_jwt_payload(session.last_id_token)
        if not after.get("ent"):
            break
        time.sleep(0.5)
    check("cancelling revokes the claim", not after.get("ent"),
          f"after {attempt + 1} refresh(es)")
    check("evaluate() now refuses",
          not ent.evaluate(session, online_ok=True).allows_app)

    # ── 7. Grandfathering ──────────────────────────────────────────────────
    step("7. A beta tester's free year (beforeUserCreated -> getEntitlement)")
    gf_email = f"tester-{secrets.token_hex(4)}@example.com"
    digest = hashlib.sha256(gf_email.strip().lower().encode()).hexdigest()

    # Seed the allowlist the way scripts/seed-grandfathered.ts would — i.e.
    # with admin credentials. `Bearer owner` is the emulator's documented
    # stand-in for those.
    #
    # Worth doing the unauthenticated attempt first: firestore.rules says
    # `allow read, write: if false` on this collection, and a 403 here is the
    # rules actually enforcing that. If this ever starts succeeding, the
    # allowlist has become world-writable and anyone can grant themselves a
    # free year.
    def _seed(headers):
        req = urllib.request.Request(
            f"http://{FIRESTORE_HOST}/v1/projects/{PROJECT}/databases/(default)"
            f"/documents/grandfathered?documentId={digest}",
            data=json.dumps({"fields": {"note": {"stringValue": "beta tester"}}}).encode(),
            method="POST", headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status
        except urllib.error.HTTPError as exc:
            return exc.code

    check("rules refuse an unauthenticated write to the allowlist",
          _seed({}) == 403, "firestore.rules is doing its job")
    check("allowlist seeded with admin credentials",
          _seed({"Authorization": "Bearer owner"}) == 200, digest[:16] + "…")

    gf = auth.sign_up(gf_email, "spike-password-123")
    gf_session = Session(refresh_token=gf["refresh_token"], uid=gf["uid"],
                         email=gf["email"], last_id_token=gf["id_token"],
                         hwm=int(time.time()))

    # beforeUserCreated cannot mint a claim -- the user does not exist yet --
    # so the grant sits in Firestore until the first getEntitlement call
    # reconciles it. This is exactly the path a real tester takes.
    info = functions_client.get_entitlement(gf_session.last_id_token)
    check("getEntitlement reports the free year",
          (info.get("entitlement") or {}).get("source") == "grandfathered",
          json.dumps(info.get("entitlement")))

    gf_claims = {}
    for attempt in range(20):
        refreshed = auth.refresh(gf_session.refresh_token)
        gf_session.refresh_token = refreshed["refresh_token"]
        gf_session.last_id_token = refreshed["id_token"]
        gf_claims = auth.decode_jwt_payload(gf_session.last_id_token)
        if gf_claims.get("ent"):
            break
        time.sleep(0.5)
    check("the reconciled claim reaches the token", gf_claims.get("ent") == 1,
          f"after {attempt + 1} refresh(es)")
    check("the tester is entitled without ever paying",
          ent.evaluate(gf_session, online_ok=True).allows_app)
    check("and is offered no refund (they paid nothing)",
          not (info.get("refund") or {}).get("eligible"),
          (info.get("refund") or {}).get("reason"))

    # ── Verdict ────────────────────────────────────────────────────────────
    print()
    if _failures:
        print(f"\033[31mS3 FAIL — {len(_failures)} check(s):\033[0m")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("\033[32mS3 PASS\033[0m — webhook to claim to refreshed token to unlocked app.")
    print("Left for a real Stripe account: that Checkout emits the event at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
