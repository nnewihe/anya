# functions/ — billing and entitlements

The server half of the paid release. The desktop app (`desktop/`) never talks
to Stripe and never writes to Firestore; it calls the callables here and trusts
the custom claim they mint.

Node/TypeScript rather than Python on purpose: `stripe-node` is the
best-maintained SDK, and this repo's Python is a 2 GB ML stack pinned to exact
versions in `desktop/constraints-windows.txt`. Keeping the two dependency
graphs apart is worth more than running one language everywhere.

## What's here

| File | |
|---|---|
| `src/index.ts` | Exports. `admin.initializeApp()` lives here and nowhere else. |
| `src/callables.ts` | `createCheckoutSession`, `createPortalSession`, `getEntitlement`, `cancelAndRefund`, `revokeSessions`. |
| `src/webhook.ts` | `stripeWebhook` — the authoritative source of entitlement. |
| `src/entitlement.ts` | The only place a grant is written. Firestore doc + custom claim, together. |
| `src/grandfather.ts` | `beforeUserCreated`: a free year for the beta testers. |
| `src/stripeClient.ts` | Lazy Stripe client, find-or-create Customer. |
| `src/config.ts` | Secret and parameter declarations. |
| `scripts/seed-grandfathered.ts` | Seed the allowlist; comp an account. |

## First-time setup

```bash
npm install

# Secrets live in Google Secret Manager, never in a file.
firebase functions:secrets:set STRIPE_SECRET_KEY
firebase functions:secrets:set STRIPE_WEBHOOK_SECRET

# Non-secret parameters (price ids, redirect URLs).
firebase functions:config:export   # or set them at deploy time
```

For the Stripe side — Product, both Prices, Stripe Tax, the Customer portal,
webhook forwarding, and the test cards — follow **[STRIPE_SETUP.md](STRIPE_SETUP.md)**.
Do that in test mode first; it ends with a full subscribe, unlock and refund.

## Developing

Everything below runs against emulators. No live keys, no real money.

```bash
npm run serve                       # functions + firestore + auth emulators

# In a second terminal:
stripe listen --forward-to http://127.0.0.1:5001/anya-tennis/us-central1/stripeWebhook
stripe trigger checkout.session.completed
stripe trigger customer.subscription.deleted
stripe trigger charge.refunded
```

After each trigger, check `users/{uid}` in the emulator UI and confirm the
custom claim was minted. The claim — not the document — is what the desktop app
trusts, and the two drifting apart is the failure mode `applyEntitlement`
exists to prevent.

## Things that will bite

- **A new claim only appears in a freshly minted ID token.** After a webhook
  grants entitlement, the client has to refresh. That is not a bug to work
  around; it is the mechanism `CheckoutPollWorker` polls on.
- **`beforeUserCreated` cannot set a durable custom claim** — the user does not
  exist yet. Grandfathered accounts are written with `pendingClaim: true` and
  reconciled on the first `getEntitlement` call.
- **The webhook must use `req.rawBody`.** The signature is over the exact bytes
  Stripe sent; any JSON round-trip breaks it.
- **`invoice.payment_failed` must not revoke.** Stripe retries a failed card for
  days and the subscription sits in `past_due`, which still counts as live.
  Entitlement lapses on its own when `until` passes.
