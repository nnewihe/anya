/**
 * The single place entitlement is granted or revoked.
 *
 * Every webhook branch and every callable ends here, so there is exactly one
 * function that decides what a user is allowed and exactly one that mints the
 * claim. Two writes have to stay in step — the Firestore document (what the
 * account screen reads) and the custom claim (what the desktop app trusts
 * offline) — and splitting that logic across call sites is how they drift.
 */
import { getAuth } from "firebase-admin/auth";
import { getFirestore } from "firebase-admin/firestore";
import { DAY_SECONDS, ISSUER_SLACK_DAYS } from "./config";

export type EntSource = "stripe" | "grandfathered" | "comp";

export interface Entitlement {
  active: boolean;
  /** Unix seconds. The desktop app's grace math is anchored to this; see
   *  desktop/entitlement.py. */
  until: number;
  source: EntSource;
}

/** Stripe statuses that should keep the app working.
 *
 *  `past_due` is deliberately included: Stripe retries a failed card for days,
 *  and cutting someone off the moment the first attempt fails would lock out a
 *  paying customer over an expired card they are about to update. Entitlement
 *  lapses naturally when `until` passes instead. */
const LIVE_STATUSES = new Set(["active", "trialing", "past_due"]);

export function entitlementFromSubscription(sub: {
  status: string;
  current_period_end: number;
}): Entitlement {
  return {
    active: LIVE_STATUSES.has(sub.status),
    until: sub.current_period_end + ISSUER_SLACK_DAYS * DAY_SECONDS,
    source: "stripe",
  };
}

export function revoked(source: EntSource = "stripe"): Entitlement {
  return { active: false, until: 0, source };
}

/**
 * Write the entitlement to Firestore AND mint it into the user's custom claims.
 *
 * The claim is what the desktop app reads, and claims are serialised into every
 * ID token with a 1000-byte cap, so the shape is deliberately terse:
 * `{ent, entExp, pl}`. `entExp` rather than a bare boolean is what makes the
 * token self-describing and is the entire basis of the offline grace window.
 *
 * Note the client only sees a new claim after it refreshes its token — claims
 * are baked in at mint time. That is not a limitation to work around; it is the
 * mechanism the desktop app polls on after checkout.
 */
export async function applyEntitlement(
  uid: string,
  ent: Entitlement,
  plan: "a" | "m" | null,
  extra: Record<string, unknown> = {}
): Promise<void> {
  const db = getFirestore();

  await db.doc(`users/${uid}`).set(
    {
      entitlement: ent,
      updatedAt: Math.floor(Date.now() / 1000),
      ...extra,
    },
    { merge: true }
  );

  // Preserve any claims we don't own rather than replacing the whole object:
  // setCustomUserClaims overwrites, and a future unrelated claim would vanish.
  const user = await getAuth().getUser(uid);
  const existing = { ...(user.customClaims ?? {}) };
  delete existing.ent;
  delete existing.entExp;
  delete existing.pl;

  await getAuth().setCustomUserClaims(
    uid,
    ent.active
      ? { ...existing, ent: 1, entExp: ent.until, ...(plan ? { pl: plan } : {}) }
      : existing
  );
}

/** "a" | "m" from a Stripe price id, or null if it is neither of ours. */
export function planOf(priceId: string, annual: string, monthly: string): "a" | "m" | null {
  if (priceId && priceId === annual) return "a";
  if (priceId && priceId === monthly) return "m";
  return null;
}
