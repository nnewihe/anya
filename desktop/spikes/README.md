# Phase 0 spikes

Three questions that could each have invalidated the paid-release design, asked
before the production code was trusted. All three now pass. Keep them: they are
the fastest way to re-answer the same questions after a dependency bump.

| | Question | Status |
|---|---|---|
| **S1** | Google sign-in: loopback + PKCE, and `signInWithIdp` | **PASS** (mechanics + request shape). Console wiring still needs a live project. |
| **S2** | Can `QMediaPlayer` stream an https MP4 from inside the shipped app? | **PASS**, including signed + hardened runtime. See `S2_RESULT.md`. |
| **S3** | Stripe webhook → custom claim → refreshed token → unlocked app | **PASS**, 24 checks, against the emulator suite. |

## Running them

S1 and S3 need the emulator suite:

```bash
cd functions && ./scripts/emulator.sh
```

Then, in another terminal:

```bash
cd desktop
python3 spikes/s2_stream_video.py --headless    # needs nothing
python3 spikes/s1_google_signin.py
python3 spikes/s3_entitlement_chain.py
```

`scripts/emulator.sh` exists because three separate things make a bare
`firebase emulators:start` fail in ways that are hard to read — see its header.

## What these found

Two bugs that `tsc` could not have caught, both of which would have shipped:

1. **`admin.firestore.FieldValue` is `undefined` at runtime** in firebase-admin
   v12, while still type-checking against its declarations. Every webhook
   delivery threw `Cannot read properties of undefined (reading
   'serverTimestamp')`. Fixed by importing `FieldValue` from
   `firebase-admin/firestore`. Nothing short of executing the function finds
   this.
2. **`claimEvent` swallowed every error as "duplicate"**, so a Firestore blip
   would answer Stripe 200 — and Stripe stops retrying on any 2xx. A customer
   would have paid and never been entitled. It now rethrows anything that is
   not `already-exists`, so the delivery is retried.

And one that was only a nuisance, but an expensive one to diagnose: a Firebase
`param` with no default makes the CLI *prompt* for a value, which hangs the
emulator while it reports `valid functions are <nothing>`. All params now carry
explicit defaults.

## What is still open

Only what genuinely requires accounts:

- **S1(b)** — that a real Google *Desktop* OAuth client and the Firebase
  whitelist accept each other. `python3 spikes/s1_google_signin.py --real`
  answers it in one run once the project exists. The failure to expect is
  `INVALID_IDP_RESPONSE`, which almost always means the desktop client ID is
  not whitelisted under Auth → Google → Web SDK configuration.
- **S3's first link** — that Stripe Checkout emits
  `customer.subscription.updated` with `metadata.uid` set. Everything
  downstream of that event is proven.
- **S2's last mile** — watching the preview play in the real notarized DMG on
  a clean Mac. Release-checklist item 3.
