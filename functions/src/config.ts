/**
 * Shared configuration and secret declarations.
 *
 * Secrets are gen-2 `defineSecret` bindings backed by Google Secret Manager,
 * set once with `firebase functions:secrets:set STRIPE_SECRET_KEY`. Never
 * functions.config() (deprecated) and never a .env file — the repo's
 * .gitignore blocks those, but the real reason is that Secret Manager gives
 * per-function access grants and an audit trail.
 *
 * The Admin SDK needs no key here at all: on Cloud Functions it authenticates
 * as the runtime's own service account. That is why no service-account JSON
 * exists anywhere in this repo.
 */
import { defineSecret, defineString } from "firebase-functions/params";

export const STRIPE_SECRET_KEY = defineSecret("STRIPE_SECRET_KEY");
export const STRIPE_WEBHOOK_SECRET = defineSecret("STRIPE_WEBHOOK_SECRET");

/** Stripe Price ids for the two plans. Not secret — they identify a price,
 *  they don't authorize a charge. Set with `firebase functions:config` params
 *  or the FIREBASE_CONFIG env; kept as params so test and live modes differ by
 *  configuration rather than by a code change. */
//
// Every one of these carries an explicit default. A `defineString` with no
// default is not merely undefined at runtime -- it makes the CLI PROMPT for a
// value, which hangs `firebase emulators:start` and `firebase deploy` on an
// interactive question. That is a genuinely confusing failure: the emulator
// reports "valid functions are <nothing>" while actually sitting on a prompt.
// An empty default is safe here because createCheckoutSession already refuses
// with `failed-precondition` when the price is empty.
export const PRICE_ANNUAL = defineString("PRICE_ANNUAL", { default: "" });
export const PRICE_MONTHLY = defineString("PRICE_MONTHLY", { default: "" });

/** Where Stripe sends the browser after checkout. Plain pages on Firebase
 *  Hosting; the desktop app is not watching this redirect (it polls its own
 *  token — see CheckoutPollWorker), so these only need to tell a human what
 *  happened. */
export const SUCCESS_URL = defineString("SUCCESS_URL", {
  default: "https://nnewihe.github.io/anya/",
});
export const CANCEL_URL = defineString("CANCEL_URL", {
  default: "https://nnewihe.github.io/anya/",
});

export const REGION = "us-central1";

/** Days of slack added to a Stripe period end before entitlement lapses.
 *  Covers the gap between a renewal charge and its webhook, so a paying
 *  customer is never locked out for the minutes in between. */
export const ISSUER_SLACK_DAYS = 3;

/** The refund window, in days. Mirrored in desktop/firebase_config.py for
 *  display only — this constant is the one that decides. */
export const REFUND_WINDOW_DAYS = 14;

export const DAY_SECONDS = 24 * 60 * 60;
