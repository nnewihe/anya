/**
 * Lazily-constructed Stripe client.
 *
 * Constructed on first use rather than at module load because the secret is
 * only bound while a function that declares it is executing; building the
 * client at import time throws in every other context, including deploy-time
 * analysis.
 */
import Stripe from "stripe";
import { STRIPE_SECRET_KEY } from "./config";

let client: Stripe | null = null;

export function stripe(): Stripe {
  if (!client) {
    client = new Stripe(STRIPE_SECRET_KEY.value(), {
      // Pinned so a Stripe-side API upgrade cannot change response shapes
      // under a deployed function.
      apiVersion: "2025-02-24.acacia",
      // Cloud Functions can be frozen between invocations; a shorter timeout
      // with retries beats hanging until the function times out.
      timeout: 20_000,
      maxNetworkRetries: 2,
    });
  }
  return client;
}

/** Find-or-create the Stripe Customer for a Firebase uid.
 *
 *  The uid goes in metadata as well as the Firestore document so that the link
 *  is recoverable from either side — a webhook that arrives before the document
 *  is written can still find its user. */
export async function customerFor(
  uid: string,
  email: string,
  existingId?: string
): Promise<string> {
  const s = stripe();

  if (existingId) {
    try {
      const found = await s.customers.retrieve(existingId);
      if (!found.deleted) return existingId;
    } catch {
      // Deleted in the dashboard, or from the other Stripe mode. Fall through
      // and make a new one rather than failing the checkout.
    }
  }

  const created = await s.customers.create({
    email,
    metadata: { firebaseUid: uid },
  });
  return created.id;
}
