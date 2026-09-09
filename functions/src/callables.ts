/**
 * The four things the desktop app asks the server to do.
 *
 * All `onCall`: the callable protocol is a plain POST of {"data": {...}} with
 * the Firebase ID token in the Authorization header, which the desktop client
 * builds by hand (there is no Firebase client SDK for Python). The important
 * property is that `request.auth.uid` is derived from a verified token, so no
 * caller can act for another user by asking nicely.
 */
import { getAuth } from "firebase-admin/auth";
import { getFirestore } from "firebase-admin/firestore";
import { HttpsError, onCall } from "firebase-functions/v2/https";
import { logger } from "firebase-functions";

import {
  CANCEL_URL, DAY_SECONDS, PRICE_ANNUAL, PRICE_MONTHLY, REFUND_WINDOW_DAYS,
  REGION, STRIPE_SECRET_KEY, SUCCESS_URL,
} from "./config";
import { applyEntitlement, revoked } from "./entitlement";
import { customerFor, stripe } from "./stripeClient";

const opts = { region: REGION, secrets: [STRIPE_SECRET_KEY] };

/** The plan letter stored on the document, for re-minting a claim.
 *  A grandfathered or comped grant has no Stripe price and so no plan. */
function planOf2(user: Record<string, any>): "a" | "m" | null {
  const plan = user.subscription?.plan;
  return plan === "annual" ? "a" : plan === "monthly" ? "m" : null;
}

function requireUid(auth: { uid: string } | undefined): string {
  if (!auth?.uid) {
    throw new HttpsError("unauthenticated", "Please sign in first.");
  }
  return auth.uid;
}

async function userDoc(uid: string) {
  const snap = await getFirestore().doc(`users/${uid}`).get();
  return (snap.data() ?? {}) as Record<string, any>;
}

/** Hosted Checkout. Returns a URL the desktop app opens in the system browser.
 *
 *  The app does not watch the redirect — it polls its own ID token for the
 *  entitlement claim (see desktop/authworker.CheckoutPollWorker), which is why
 *  success_url can be an ordinary web page rather than a deep link. */
export const createCheckoutSession = onCall(opts, async (request) => {
  const uid = requireUid(request.auth);
  const plan = request.data?.plan === "monthly" ? "monthly" : "annual";
  const email = request.auth!.token.email ?? "";

  const existing = await userDoc(uid);
  const customerId = await customerFor(uid, email, existing.stripeCustomerId);

  const price = plan === "annual" ? PRICE_ANNUAL.value() : PRICE_MONTHLY.value();
  if (!price) {
    throw new HttpsError("failed-precondition", "Billing isn't configured yet.");
  }

  const session = await stripe().checkout.sessions.create({
    mode: "subscription",
    customer: customerId,
    line_items: [{ price, quantity: 1 }],
    client_reference_id: uid,
    subscription_data: { metadata: { uid } },
    allow_promotion_codes: true,
    // Stripe Tax. We are merchant of record on a direct Stripe integration, so
    // VAT/sales tax is ours to collect; `address: 'auto'` is what makes Stripe
    // collect the address it needs to compute a rate.
    automatic_tax: { enabled: true },
    customer_update: { address: "auto" },
    success_url: SUCCESS_URL.value() || "https://nnewihe.github.io/anya/",
    cancel_url: CANCEL_URL.value() || "https://nnewihe.github.io/anya/",
  });

  await getFirestore().doc(`users/${uid}`).set(
    { email, stripeCustomerId: customerId, updatedAt: Math.floor(Date.now() / 1000) },
    { merge: true }
  );

  logger.info("checkout session created", { uid, plan });
  return { url: session.url };
});

/** Stripe's own hosted billing portal: card changes, invoices, and
 *  cancel-at-period-end, none of which we have to build or maintain. */
export const createPortalSession = onCall(opts, async (request) => {
  const uid = requireUid(request.auth);
  const user = await userDoc(uid);
  if (!user.stripeCustomerId) {
    throw new HttpsError("failed-precondition", "No subscription to manage yet.");
  }

  const session = await stripe().billingPortal.sessions.create({
    customer: user.stripeCustomerId,
    return_url: SUCCESS_URL.value() || "https://nnewihe.github.io/anya/",
  });
  return { url: session.url };
});

/** What the account screen shows. The claim says WHETHER you are entitled;
 *  this says when it renews and whether the refund button should appear. */
export const getEntitlement = onCall({ region: REGION }, async (request) => {
  const uid = requireUid(request.auth);
  let user = await userDoc(uid);

  // Reconcile a grant that has a Firestore entitlement but no claim yet.
  // beforeUserCreated cannot mint one — the user does not exist at that point
  // — so a grandfathered tester arrives here entitled on paper and unentitled
  // in their token. This is the first call the app makes after signing in, so
  // it is the right place to close that gap. Also self-heals the rarer case of
  // a webhook whose setCustomUserClaims failed after its Firestore write.
  if (user.pendingClaim && user.entitlement?.active) {
    await applyEntitlement(uid, user.entitlement, planOf2(user), { pendingClaim: false });
    user = await userDoc(uid);
    logger.info("minted a pending entitlement claim", { uid, source: user.entitlement?.source });
  }

  const sub = user.subscription ?? null;

  return {
    entitlement: user.entitlement ?? { active: false, until: 0, source: "stripe" },
    plan: sub?.plan ?? null,
    status: sub?.status ?? null,
    currentPeriodEnd: sub?.currentPeriodEnd ?? null,
    cancelAtPeriodEnd: sub?.cancelAtPeriodEnd ?? false,
    refund: refundEligibility(user),
    email: user.email ?? request.auth!.token.email ?? "",
  };
});

/** Whether the one-time refund is still on the table, and if not, why.
 *
 *  Deliberately NOT conditioned on whether the app has been used. Enforcing
 *  that would mean the desktop app reporting completed renders — the first
 *  usage telemetry it has ever sent — against a product whose headline promise
 *  is that it runs entirely on your machine. The 14-day window plus a
 *  permanent one-shot flag does the job; abuse is bounded at one subscription
 *  per account and Stripe Radar catches serial refunders. */
function refundEligibility(user: Record<string, any>) {
  const now = Math.floor(Date.now() / 1000);
  const firstPaymentAt: number | null = user.firstPaymentAt ?? null;

  if (user.refundUsed) return { eligible: false, reason: "already_used", deadline: null };
  if (!firstPaymentAt) return { eligible: false, reason: "no_payment", deadline: null };

  const deadline = firstPaymentAt + REFUND_WINDOW_DAYS * DAY_SECONDS;
  if (now >= deadline) return { eligible: false, reason: "window_closed", deadline };
  return { eligible: true, reason: null, deadline };
}

/** The self-serve 14-day full refund.
 *
 *  Every precondition is checked here and nowhere else. The desktop app hides
 *  the button when getEntitlement says so, but that is a courtesy — a client
 *  that calls this directly gets the same three checks. */
export const cancelAndRefund = onCall(opts, async (request) => {
  const uid = requireUid(request.auth);
  const user = await userDoc(uid);

  const eligibility = refundEligibility(user);
  if (!eligibility.eligible) {
    throw new HttpsError(
      "failed-precondition",
      eligibility.reason === "already_used"
        ? "This account has already used its one-time refund."
        : eligibility.reason === "window_closed"
        ? `Refunds are available within ${REFUND_WINDOW_DAYS} days of your first payment.`
        : "There's no payment on this account to refund.",
      { reason: eligibility.reason }
    );
  }

  const subscriptionId: string | undefined = user.subscription?.id;
  if (!subscriptionId) {
    throw new HttpsError("failed-precondition", "There's no active subscription to cancel.");
  }

  const s = stripe();

  // Refund the most recent paid invoice's payment. Idempotency-keyed on the
  // uid so a double-click, or a retry after a timeout, cannot refund twice.
  const invoices = await s.invoices.list({
    customer: user.stripeCustomerId, status: "paid", limit: 1,
  });
  const paymentIntent = invoices.data[0]?.payment_intent;
  if (!paymentIntent) {
    throw new HttpsError("failed-precondition", "Couldn't find the payment to refund.");
  }

  await s.refunds.create(
    {
      payment_intent: typeof paymentIntent === "string" ? paymentIntent : paymentIntent.id,
      reason: "requested_by_customer",
    },
    { idempotencyKey: `refund_${uid}` }
  );

  await s.subscriptions.cancel(subscriptionId);

  // Set refundUsed BEFORE anything can fail afterwards, and set it
  // permanently: this is the flag that makes the offer one-time.
  await applyEntitlement(uid, revoked(), null, {
    refundUsed: true,
    refundedAt: Math.floor(Date.now() / 1000),
  });

  logger.info("refund issued and subscription cancelled", { uid });
  return { ok: true };
});

/** Sign out everywhere: revoke every refresh token for this account.
 *
 *  The kill switch for a leaked session.json. desktop/authstore.py is explicit
 *  that a copied session file still validates locally — this is the answer to
 *  that, and it is why it ships in the first release rather than later. */
export const revokeSessions = onCall({ region: REGION }, async (request) => {
  const uid = requireUid(request.auth);
  await getAuth().revokeRefreshTokens(uid);
  logger.info("refresh tokens revoked", { uid });
  return { ok: true };
});
