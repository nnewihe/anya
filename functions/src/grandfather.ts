/**
 * Free year for the beta testers.
 *
 * Thirteen beta builds were tested by people who got nothing for it, and the
 * first thing they would meet after updating is a price. This runs before a
 * user record is created and, for an email on the list, grants a year outright
 * so they never see one.
 *
 * The list is keyed by sha256 of the lowercased, trimmed email rather than by
 * the email itself. The project id is public and the rules on this collection
 * are `false` for everyone, but a rules mistake should leak a set of digests
 * rather than a customer list.
 *
 * Seed it with functions/scripts/seed-grandfathered.ts.
 */
import { getFirestore } from "firebase-admin/firestore";
import { createHash } from "crypto";
import { beforeUserCreated } from "firebase-functions/v2/identity";
import { logger } from "firebase-functions";

import { DAY_SECONDS, REGION } from "./config";

export function emailHash(email: string): string {
  return createHash("sha256").update(email.trim().toLowerCase(), "utf8").digest("hex");
}

export const onUserCreated = beforeUserCreated({ region: REGION }, async (event) => {
  const email = event.data?.email;
  const uid = event.data?.uid;
  if (!email || !uid) return;

  const hit = await getFirestore().doc(`grandfathered/${emailHash(email)}`).get();
  const now = Math.floor(Date.now() / 1000);

  const base = {
    email,
    createdAt: now,
    updatedAt: now,
    refundUsed: false,
    firstPaymentAt: null,
  };

  if (!hit.exists) {
    await getFirestore().doc(`users/${uid}`).set(base, { merge: true });
    return;
  }

  // The claim itself cannot be minted here — the user does not exist yet, so
  // setCustomUserClaims has nothing to attach to. Write the entitlement to the
  // document now; the first getEntitlement call reconciles it into a claim.
  // (A blocking function CAN return customClaims, but only for the session it
  // is creating, which would silently expire on the next refresh.)
  await getFirestore().doc(`users/${uid}`).set(
    {
      ...base,
      entitlement: { active: true, until: now + 365 * DAY_SECONDS, source: "grandfathered" },
      pendingClaim: true,
    },
    { merge: true }
  );
  logger.info("grandfathered tester granted a free year", { uid });
});
