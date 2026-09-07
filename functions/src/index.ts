/**
 * Billing and entitlements for the Anya Tennis desktop app.
 *
 * Node rather than Python on purpose: stripe-node is the best-maintained SDK,
 * and this repo's Python is a 2 GB ML stack pinned to exact versions in
 * desktop/constraints-windows.txt. Keeping the two dependency graphs apart is
 * worth more than running one language.
 *
 * The desktop client (desktop/auth.py, desktop/entitlement.py) never writes to
 * Firestore and never talks to Stripe. It reads its own user document, calls
 * the callables here, and trusts the custom claim these functions mint.
 */
import * as admin from "firebase-admin";

admin.initializeApp();

export {
  createCheckoutSession,
  createPortalSession,
  getEntitlement,
  cancelAndRefund,
  revokeSessions,
} from "./callables";

export { stripeWebhook } from "./webhook";
export { onUserCreated } from "./grandfather";
