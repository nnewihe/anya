/**
 * The Stripe webhook — the authoritative source of entitlement.
 *
 * Nothing else grants a subscription. The checkout callable creates a session
 * and returns a URL; whether that turned into money is something only Stripe
 * can tell us, and it tells us here.
 *
 * Two properties this has to hold, because Stripe guarantees neither:
 *
 *   * Idempotency. Stripe retries on any non-2xx, and a retry of an already
 *     processed event must be a no-op. A transactional create of
 *     stripeEvents/{id} is the lock: `already-exists` means "seen it", and we
 *     return 200 so Stripe stops retrying.
 *   * Ordering. Events can arrive out of order, so a stale
 *     subscription.updated must not overwrite a newer one. Every write records
 *     the event's `created` and refuses to go backwards.
 */
import * as admin from "firebase-admin";
// FieldValue comes from the modular entry point, NOT from `admin.firestore.
// FieldValue`. The namespaced form still TYPE-CHECKS against firebase-admin
// v12's declarations but is `undefined` at runtime, so it compiles cleanly and
// then throws "Cannot read properties of undefined (reading
// 'serverTimestamp')" on the first webhook delivery. tsc cannot catch it;
// only running the thing can.
import { FieldValue } from "firebase-admin/firestore";
import { onRequest } from "firebase-functions/v2/https";
import { logger } from "firebase-functions";
import type Stripe from "stripe";

import {
  PRICE_ANNUAL, PRICE_MONTHLY, REGION, STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET,
} from "./config";
import { applyEntitlement, entitlementFromSubscription, planOf, revoked } from "./entitlement";
import { stripe } from "./stripeClient";

/** Claim the event id, or report that someone already did.
 *  `create` fails if the document exists, which is exactly the atomic
 *  test-and-set this needs — no transaction required.
 *
 *  ONLY an already-exists failure means "seen it". Every other error has to
 *  propagate: a catch-all here reports a duplicate, the caller answers 200,
 *  and Stripe — which stops retrying on any 2xx — never delivers that event
 *  again. A transient Firestore blip would silently cost a customer the
 *  subscription they just paid for. Ask me how I know. */
async function claimEvent(event: Stripe.Event): Promise<boolean> {
  try {
    await admin.firestore().doc(`stripeEvents/${event.id}`).create({
      type: event.type,
      created: event.created,
      at: FieldValue.serverTimestamp(),
    });
    return true;
  } catch (err) {
    // gRPC ALREADY_EXISTS is 6; the Admin SDK also surfaces it as a string.
    const code = (err as { code?: number | string }).code;
    if (code === 6 || code === "already-exists") return false;
    logger.error("could not claim stripe event", {
      id: event.id, code, err: String(err),
    });
    throw err;
  }
}

/** Resolve a Stripe customer or subscription back to a Firebase uid.
 *  Tried in order of reliability: the metadata we set at creation, then the
 *  customer's metadata, then a Firestore lookup by customer id. */
async function uidFor(
  sub: Stripe.Subscription | null,
  customerId: string | null
): Promise<string | null> {
  if (sub?.metadata?.uid) return sub.metadata.uid;

  if (customerId) {
    try {
      const customer = await stripe().customers.retrieve(customerId);
      if (!customer.deleted && customer.metadata?.firebaseUid) {
        return customer.metadata.firebaseUid;
      }
    } catch {
      // fall through
    }
    const q = await admin.firestore()
      .collection("users").where("stripeCustomerId", "==", customerId).limit(1).get();
    if (!q.empty) return q.docs[0].id;
  }
  return null;
}

/** True if we have already applied something newer than this event. */
async function isStale(uid: string, eventCreated: number): Promise<boolean> {
  const snap = await admin.firestore().doc(`users/${uid}`).get();
  const last = (snap.data()?.lastEventCreated as number | undefined) ?? 0;
  return eventCreated < last;
}

async function applySubscription(
  uid: string, sub: Stripe.Subscription, eventCreated: number
): Promise<void> {
  if (await isStale(uid, eventCreated)) {
    logger.info("ignoring out-of-order subscription event", { uid, eventCreated });
    return;
  }

  const priceId = sub.items.data[0]?.price?.id ?? "";
  const plan = planOf(priceId, PRICE_ANNUAL.value(), PRICE_MONTHLY.value());
  const ent = sub.status === "canceled" || sub.status === "incomplete_expired"
    ? revoked()
    : entitlementFromSubscription({
        status: sub.status,
        current_period_end: sub.current_period_end,
      });

  await applyEntitlement(uid, ent, plan, {
    subscription: {
      id: sub.id,
      status: sub.status,
      priceId,
      plan: plan === "a" ? "annual" : plan === "m" ? "monthly" : null,
      currentPeriodEnd: sub.current_period_end,
      cancelAtPeriodEnd: sub.cancel_at_period_end,
    },
    lastEventCreated: eventCreated,
  });

  logger.info("entitlement applied", { uid, status: sub.status, active: ent.active });
}

export const stripeWebhook = onRequest(
  { region: REGION, secrets: [STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET] },
  async (req, res) => {
    const signature = req.headers["stripe-signature"];
    let event: Stripe.Event;
    try {
      // rawBody, not req.body: the signature is over the exact bytes Stripe
      // sent, and any JSON round-trip changes them. Firebase provides rawBody
      // on onRequest for precisely this.
      event = stripe().webhooks.constructEvent(
        req.rawBody, signature as string, STRIPE_WEBHOOK_SECRET.value()
      );
    } catch (err) {
      logger.warn("webhook signature verification failed", { err: String(err) });
      res.status(400).send("invalid signature");
      return;
    }

    if (!(await claimEvent(event))) {
      logger.info("duplicate webhook delivery ignored", { id: event.id, type: event.type });
      res.status(200).send("already processed");
      return;
    }

    try {
      await handle(event);
      res.status(200).send("ok");
    } catch (err) {
      // Release the idempotency claim so Stripe's retry can actually retry;
      // leaving it in place would turn a transient failure into a permanently
      // dropped event.
      await admin.firestore().doc(`stripeEvents/${event.id}`).delete().catch(() => {});
      logger.error("webhook handler failed", { id: event.id, type: event.type, err: String(err) });
      res.status(500).send("handler error");
    }
  }
);

async function handle(event: Stripe.Event): Promise<void> {
  switch (event.type) {
    case "checkout.session.completed": {
      const session = event.data.object as Stripe.Checkout.Session;
      const uid = session.client_reference_id
        ?? (await uidFor(null, session.customer as string | null));
      if (!uid || !session.subscription) return;

      const sub = await stripe().subscriptions.retrieve(session.subscription as string);

      // firstPaymentAt anchors the 14-day refund window and must be written
      // exactly once — a later renewal must not restart the clock.
      const ref = admin.firestore().doc(`users/${uid}`);
      const existing = await ref.get();
      if (!existing.data()?.firstPaymentAt) {
        await ref.set(
          {
            firstPaymentAt: event.created,
            refundUsed: existing.data()?.refundUsed ?? false,
            stripeCustomerId: session.customer,
          },
          { merge: true }
        );
      }

      await applySubscription(uid, sub, event.created);
      return;
    }

    case "customer.subscription.created":
    case "customer.subscription.updated":
    case "customer.subscription.deleted": {
      const sub = event.data.object as Stripe.Subscription;
      const uid = await uidFor(sub, sub.customer as string);
      if (!uid) {
        logger.warn("subscription event with no resolvable uid", { id: sub.id });
        return;
      }
      await applySubscription(uid, sub, event.created);
      return;
    }

    case "invoice.paid": {
      // A renewal. Re-read the subscription rather than trusting the invoice's
      // copy of the period, so entitlement.until always comes from one place.
      const invoice = event.data.object as Stripe.Invoice;
      if (!invoice.subscription) return;
      const sub = await stripe().subscriptions.retrieve(invoice.subscription as string);
      const uid = await uidFor(sub, invoice.customer as string);
      if (uid) await applySubscription(uid, sub, event.created);
      return;
    }

    case "invoice.payment_failed": {
      // Deliberately does NOT revoke. Stripe retries a failed card for days,
      // and the subscription goes `past_due` — which entitlementFromSubscription
      // still treats as live. Access lapses on its own when `until` passes.
      // Record it so the account screen can say something useful.
      const invoice = event.data.object as Stripe.Invoice;
      const uid = await uidFor(null, invoice.customer as string);
      if (uid) {
        await admin.firestore().doc(`users/${uid}`).set(
          { paymentProblem: true, updatedAt: Math.floor(Date.now() / 1000) },
          { merge: true }
        );
        logger.info("payment failed; entitlement left to lapse naturally", { uid });
      }
      return;
    }

    case "charge.refunded": {
      // Belt and braces: cancelAndRefund already revokes, but a refund issued
      // by hand from the Stripe dashboard has to revoke too, and has to burn
      // the one-time offer so it cannot then be claimed in the app.
      const charge = event.data.object as Stripe.Charge;
      const uid = await uidFor(null, charge.customer as string);
      if (uid) {
        await applyEntitlement(uid, revoked(), null, {
          refundUsed: true,
          refundedAt: event.created,
          lastEventCreated: event.created,
        });
        logger.info("entitlement revoked after refund", { uid });
      }
      return;
    }

    default:
      logger.debug("unhandled stripe event", { type: event.type });
  }
}
