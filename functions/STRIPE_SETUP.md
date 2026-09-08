# Stripe test mode — setup

Everything here happens in **test mode**. No real card is charged and no real
money moves. Nothing in this document touches live keys; the last section says
what changes when you eventually flip over.

By the end you will have taken a full subscription through Checkout with a test
card, watched the webhook mint an entitlement, seen the desktop app unlock, and
refunded it again.

Prerequisites: the emulator suite runs (`cd functions && ./scripts/emulator.sh`)
and the S3 spike passes. If it doesn't, fix that first — this builds on it.

---

## 1. Install the Stripe CLI

```bash
brew install stripe/stripe-cli/stripe
```

```bash
stripe login
```

That opens a browser and pairs the CLI with your account. It defaults to test
mode, which is what we want.

**Check you are in test mode before anything else.** Everything below assumes
it, and the difference is invisible until you have charged a real card:

```bash
stripe config --list
```

Keys shown should begin `sk_test_` / `pk_test_`. In the Dashboard, the
**Test mode** toggle (top right) must be on — test and live have entirely
separate products, prices, customers and webhook endpoints, and an object
created in one does not exist in the other.

---

## 2. Create the Product and two Prices

One Product, two recurring Prices. The amounts are the promotional launch
prices.

```bash
stripe products create \
  --name="Anya Tennis" \
  --description="Turn a full match video into a reel of just the rallies."
```

Note the returned `prod_…` id, then:

```bash
stripe prices create \
  --product=prod_XXXXXXXX \
  --currency=usd \
  --unit-amount=4000 \
  -d "recurring[interval]=year" \
  --nickname="Annual"
```

```bash
stripe prices create \
  --product=prod_XXXXXXXX \
  --currency=usd \
  --unit-amount=500 \
  -d "recurring[interval]=month" \
  --nickname="Monthly"
```

`--unit-amount` is in **cents**: 4000 = $40.00, 500 = $5.00. Getting this wrong
by a factor of 100 is the classic first mistake.

Keep both `price_…` ids. They are what `PRICE_ANNUAL` and `PRICE_MONTHLY`
must be set to, and `planOf()` in `entitlement.ts` matches on them exactly to
decide whether a subscriber is on `"a"` or `"m"`. A mismatch is not an error —
the plan just silently comes back `null` and the account screen says
"Subscription" instead of "Annual".

---

## 3. Turn on Stripe Tax

`createCheckoutSession` passes `automatic_tax: { enabled: true }`. If Stripe Tax
is not configured, **Checkout fails outright** rather than degrading — so this
step is not optional.

In the Dashboard (test mode), go to **Settings → Tax**:

1. Set your **origin address**.
2. Set a **default tax code** on the Product. For downloadable/SaaS software the
   usual choices are `txcd_10103001` (SaaS) or `txcd_10202003` (downloadable
   software). Which one applies changes whether some US states tax it at all —
   this is a question for your accountant, not for this file.
3. Add at least one **registration** in test mode so there is somewhere for tax
   to be calculated. Test-mode registrations are free and fictional.

The session also sends `customer_update: { address: 'auto' }`, which is what
makes Checkout collect the address Stripe Tax needs to compute a rate.

---

## 4. Configure the Billing Portal — the step everyone forgets

`createPortalSession` will fail with **"No configuration provided"** until the
portal has been saved once *in test mode*. It is a per-mode setting and saving
it in live mode does not help.

Go to **Settings → Billing → Customer portal**, and turn on at minimum:

- Customers can **cancel subscriptions** (this is the after-14-days path;
  "at end of billing period" is the right choice — the app's terms promise
  access until the paid period ends)
- Customers can **update payment methods** (the `past_due` recovery path)
- Customers can **view invoice history**

Press **Save**. Nothing else in this setup produces an error message as
unhelpful as this one does when skipped.

---

## 5. Point the emulator at your test keys

Get your test secret key:

```bash
stripe config --list
```

Then edit `functions/.env.demo-anya` (gitignored; created by
`scripts/emulator.sh` on first run and **never overwritten after that**):

```
PRICE_ANNUAL=price_XXXXXXXXXXXX
PRICE_MONTHLY=price_YYYYYYYYYYYY
SUCCESS_URL=https://nnewihe.github.io/anya/
CANCEL_URL=https://nnewihe.github.io/anya/
STRIPE_SECRET_KEY=sk_test_XXXXXXXXXXXX
STRIPE_WEBHOOK_SECRET=whsec_from_step_6
```

`STRIPE_WEBHOOK_SECRET` comes from step 6 — do that next, then come back and
fill it in.

> Never put `sk_live_` in this file. It is gitignored, but the real protection
> is not putting it there: the emulator would then be issuing real charges
> against fake accounts.

---

## 6. Forward webhooks to the emulator

With the emulator running, in a **separate terminal**:

```bash
stripe listen --forward-to http://127.0.0.1:5001/demo-anya/us-central1/stripeWebhook
```

It prints:

```
> Ready! Your webhook signing secret is whsec_xxxxxxxxxxxx
```

Put **that** value in `STRIPE_WEBHOOK_SECRET` and restart the emulator. It is
**not** the same as the signing secret of a Dashboard webhook endpoint — a
`stripe listen` session generates its own, and mixing them up produces a
signature failure that looks exactly like a code bug.

Leave `stripe listen` running for everything below.

---

## 7. Take a subscription end to end

The desktop app can drive this, but the tighter loop is to create the Checkout
session the same way the app does — through the callable — so that
`client_reference_id` and `subscription_data.metadata.uid` are set:

```bash
cd desktop && python3 spikes/s3_entitlement_chain.py
```

That gets you a signed-in uid and its ID token. Then call the real callable and
open the URL it returns. Or simply run the app against the emulator:

```bash
cd desktop
ANYA_AUTH_EMULATOR_HOST=127.0.0.1:9099 \
ANYA_FIREBASE_PROJECT=demo-anya \
ANYA_FIREBASE_API_KEY=emulator-key \
ANYA_FUNCTIONS_BASE=http://127.0.0.1:5001/demo-anya/us-central1 \
ANYA_NO_UPDATE_CHECK=1 \
python3 app.py
```

Create an account, choose a plan, and pay with:

| Card | Number | What it does |
|---|---|---|
| Success | `4242 4242 4242 4242` | Charges cleanly |
| 3-D Secure | `4000 0025 0000 3155` | Forces an authentication step |
| Declined | `4000 0000 0000 0002` | Generic decline |
| Insufficient funds | `4000 0000 0000 9995` | Declines at capture — good for `invoice.payment_failed` |

Any future expiry, any CVC, any postcode.

**What to watch for, in order:**

1. `stripe listen` logs `checkout.session.completed` forwarded, `[200]`.
2. The emulator logs `entitlement applied`.
3. The gate screen transitions on its own within a few seconds — that is
   `CheckoutPollWorker` seeing `ent` appear in a freshly minted token.
4. Both tabs appear.

If (1) and (2) happen but (3) does not, the claim was written and the client
isn't seeing it — check that the poll worker is running, not that the webhook
is broken.

---

## 8. Exercise the paths that are hard to reach by hand

### The 14-day refund

Open **Account → Cancel & refund** in the app. Then confirm in the Dashboard
that a refund exists and the subscription is cancelled, and that the button
does not come back (the `refundUsed` flag is permanent).

To test the window *closing*, use a **Test Clock** rather than waiting a
fortnight:

```bash
stripe test_helpers test_clocks create --frozen-time=$(date +%s)
```

Create the customer and subscription attached to that clock, then advance it
past 14 days and confirm `getEntitlement` reports
`refund.reason = "window_closed"`.

### Renewal

Advance a test clock past `current_period_end` and confirm `invoice.paid`
arrives and `entitlement.until` moves forward.

### Failed payment

Use `4000 0000 0000 9995`, then confirm `invoice.payment_failed` arrives and
that entitlement is **not** revoked — the subscription should sit in `past_due`
and access should lapse naturally when `until` passes. Revoking here would lock
out a customer over a card they are about to update.

### Dashboard-issued refund

Refund the payment from the Dashboard rather than the app, and confirm
`charge.refunded` revokes the claim and burns the one-time offer.

---

## 9. A note on `stripe trigger`

`stripe trigger checkout.session.completed` is useful for checking that the
endpoint is reachable and that signature verification passes. It will **not**
produce an entitlement, and that is expected, not a bug: `trigger` fabricates
its own objects, so there is no `metadata.uid`, no `client_reference_id`, and no
Stripe customer carrying a `firebaseUid`. `uidFor()` has nothing to resolve and
the handler logs `subscription event with no resolvable uid`.

Use `trigger` for plumbing, and a real Checkout for the chain.

---

## 10. Webhook events the code handles

If you create a Dashboard endpoint later (for a deployed environment rather
than `stripe listen`), subscribe exactly these:

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`
- `invoice.paid`
- `invoice.payment_failed`
- `charge.refunded`

Anything else is logged and ignored.

---

## 11. When you go live

Not yet — but so it is written down:

1. Recreate the Product and both Prices **in live mode**. Test-mode ids do not
   exist there.
2. Create a live webhook endpoint pointing at the deployed function URL, and
   take its signing secret from the Dashboard.
3. Store the live values as **secrets**, never in a file:
   ```bash
   firebase functions:secrets:set STRIPE_SECRET_KEY
   firebase functions:secrets:set STRIPE_WEBHOOK_SECRET
   ```
4. Set `PRICE_ANNUAL` / `PRICE_MONTHLY` to the live price ids.
5. Configure the Customer portal **again**, in live mode.
6. Complete your Stripe Tax registrations for real, and confirm your obligations
   with an accountant first — you are merchant of record on a direct
   integration, so the tax is yours to collect and remit. Stripe Tax calculates
   and warns you about thresholds; outside its filing regions it does not file
   returns.
7. Do one real purchase with a real card, then refund it.

---

## Things that will bite

| Symptom | Cause |
|---|---|
| Signature verification fails | Using a Dashboard endpoint's secret with `stripe listen`, or vice versa. They are different. |
| `No configuration provided` from the portal | The Customer portal was never saved in **test** mode (step 4). |
| Checkout errors immediately | Stripe Tax is on in code but not configured in the Dashboard (step 3). |
| Webhook is `[200]` but nothing unlocks | Almost always a `trigger`-fabricated event with no `metadata.uid` (step 9). |
| Plan shows as "Subscription", not "Annual" | `PRICE_ANNUAL` doesn't match the real price id. Silent by design. |
| Emulator says `valid functions are` (nothing) | A Firebase param has no value and the CLI is sitting on a prompt. |
| Amounts are 100× wrong | `--unit-amount` is in cents. |
