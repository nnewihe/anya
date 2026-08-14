# Dead/Live Detection — two-stream GRU

Detects **point transitions** in tennis footage by classifying short windows of
near-player pose + motion as `live` (point in play) or `dead` (between points).
The product target is not window accuracy — it is catching the **live→dead
transition (point-end)** on a full match timeline, without firing mid-rally.

Built by extending the existing active/dead pipeline rather than starting over,
so the 743 hand-audited window labels in `labels.json` stay valid.

---

## Resolved ground-truth schema

The spec assumed a single `/Volumes/Anya/Data/ground_truth.json` with live/dead
segments. The data does not look like that. What is actually on disk:

| | |
|---|---|
| Location | **per clip**: `<clip>/ground_truth.json` — 15 clips, plus `<clip>/derived_ground_truth.json` for clip 68 |
| Shape | `{"rallies": [{"start": 760, "end": 1326, "serve": "near"}]}` |
| Units | **frame indices**, in each clip's own fps (29.97 / 59.94 / 119.88 across the set) |
| Seconds variant | clip 68 only: `{"start_s": 7.5, "end_s": 22.87, "serve": ...}` plus a `_derivation` block |
| Labelled state | **live only** — dead time is the *complement* of the rally list |

Two consequences the rest of the pipeline gets wrong if it reads the file directly:

1. **Dead is implicit.** Gaps between rallies are dead, but so are pre-roll and
   post-roll, which are not between-point time at all. `dead_segments()` returns
   them explicitly with an `edge` flag so callers can drop the head/tail.

2. **`_near_rallies()` is not a timeline.** `optimize_energy._near_rallies`
   filters to `serve == "near"`, which is correct for *training* (the features
   describe the near player) but wrong for the *timeline*: a far-serve rally is
   still live. Only 126 of 277 rallies are near, so scoring against a near-only
   timeline would mark every far rally dead and wreck the false-fire rate.
   `parse_ground_truth.py` builds the timeline from **all** rallies.

```bash
python pipeline/parse_ground_truth.py
```

Coverage note: clips 63, 37 and 35 have live fractions of 1.4% / 4.5% / 7.0%,
which is almost certainly partial labelling rather than genuinely dead footage.
They are poor choices for false-fire measurement — their "dead" time likely
contains unlabelled play.

---

## Two-stream features

Per frame, two concatenated groups (`n_pose=51`, `n_global=8`), tagged into
`windows.npz` so training and evaluation never guess the layout.

**Stream A — pose (51 dims)** — 17 COCO keypoints from `yolov8n-pose`,
normalized bbox-relative `((x-bx)/bw, (y-by)/bh, conf)`. Unchanged from
`extract_pose.py`. This is YOLO-17, *not* MediaPipe-33: the cache already exists
for 11 clips and near-player identity is already solved by IoU against the
telemetry bbox.

**Stream B — global trajectory (8 dims)** — `cx, cy, bw, bh, dcx, dcy, speed,
disp` in full-frame coords, from the cached `near_bbox`. Stream A deletes court
position and coverage by construction; Stream B restores it.

### The stair-step trap

`energy_telemetry_cache.json` runs the player detector every `PLAYER_STRIDE=10`
frames and **holds** the bbox in between, so the cached center track is a
staircase. Differencing it frame-to-frame — exactly what the spec's "frame-to-
frame delta of center" asks for — gives nine zeros and one spike per cycle, and
aliases identically whether the player is sprinting or standing still.

`global_features.py` therefore **de-steps first**: repeated bboxes collapse to
knots, the track is linearly interpolated between them, then differenced.
Genuine detection gaps stay `NaN` and are never interpolated across. Velocity is
scaled to units/second so 30/60/120 fps clips align.

---

## What the motion signal actually does

Measured on the cached spans, mean bbox speed against time from rally end:

| t − end | −2s | −1s | **0** | **+1s** | +2s | +3s | +5s |
|---|---|---|---|---|---|---|---|
| speed | 0.153 | 0.091 | **0.083** | **0.123** | 0.092 | 0.066 | 0.059 |

**Dead time is faster than live time in the second after the point ends.** The
player is still decelerating, walking down the ball, turning around. Motion only
becomes discriminative ~3s in.

So Stream B does *not* sharpen boundary localization — it separates deep-live
from deep-dead, which is what suppresses false fires. This is also why the tuned
hysteresis lands at ~2.7s of required dead evidence: that is the delay at which
the signal actually appears.

---

## Timeline extraction (required, not optional)

The telemetry cache only covers `[rally_start, rally_end + 6s]`, so **every dead
window in the original training set is the 6 seconds after a point** — the most
live-looking dead there is. Genuine deep-dead behaviour (standing at the fence,
toweling off, changing ends) is absent from the cache entirely, and false-fires-
per-live-minute cannot be measured without full coverage.

```bash
python pipeline/extract_timeline.py --clips 21
```

One YOLO pose forward per frame. Clip 21 (12.6k frames) takes minutes and yields
64% pose / 70% bbox coverage — materially lower than in-rally, because the near
player often leaves frame during dead time. Clip 58 is 200k frames; bound it with
`--start/--end/--max_frames` or pick a shorter clip.

---

## Results

Two held-out clips, each excluded from training via `--holdout`, so these are
honest. Tolerance ±2s, hysteresis swept per clip.

| | clip 21 | clip 22 |
|---|---|---|
| length / live | 7.0 min / 2.3 live-min | 7.0 min / 2.5 live-min |
| point-ends | 12 | 15 |
| far-serve rallies | 1 of 12 (8%) | 4 of 15 (27%) |

**Window level** (the inflated metric the spec warns about):

| features | clip 21 acc | clip 21 dead-F1 | clip 22 acc | clip 22 dead-F1 |
|---|---|---|---|---|
| pose only | — | — | 0.638 | 0.683 |
| two-stream | 0.852 | 0.899 | 0.777 | 0.845 |

**Event level** (primary):

| features | clip | N | M | P | R | F1 | median timing | false fires / live-min |
|---|---|---|---|---|---|---|---|---|
| pose only | 21 | 16 | 8 | 0.471 | 0.667 | 0.552 | −0.62s | 0.00 |
| two-stream | 21 | 16 | 5 | 0.769 | 0.833 | **0.800** | −0.42s | 0.88 |
| pose only | 22 | 6 | 5 | 0.364 | 0.533 | **0.432** | −0.63s | 1.98 |
| two-stream | 22 | 10 | 12 | 0.375 | 0.400 | 0.387 | −0.42s | 3.17 |

**Read this carefully — the headline does not replicate.** Stream B improves
window-level accuracy consistently (LOCO 0.712 → 0.730; clip 22 0.638 → 0.777),
but its large event-level gain on clip 21 (+0.248 F1) **reverses on clip 22**
(−0.045). With 2 clips and 27 total point-ends, the event-level difference
between feature sets is not established. What is established:

- Window accuracy is a poor proxy for event performance, exactly as the spec
  argues — clip 22 scores 0.777 window accuracy and 0.387 event F1.
- Stream B's window-level benefit is real and consistent.
- Stream B's event-level benefit is **unproven**. Do not quote the 0.800.

### Why clip 22 is worse: the far-serve hole

Mean predicted `P(live)` inside rallies, by serve side:

| clip | near rallies | far rallies |
|---|---|---|
| 21 | 0.625 | 0.568 |
| 22 | 0.582 | **0.473** |

Training uses `_near_rallies` — near-serve rallies only — but the timeline counts
far-serve rallies as live, because they are. During a far serve the near player
is the *receiver*: largely stationary at the baseline, which is what dead time
looks like to a near-player model. In clip 22 the far-serve average falls below
the 0.5 threshold, so the model calls those live rallies dead and false-fires
point-ends inside them. Clip 22 has 27% far-serve rallies to clip 21's 8%, which
accounts for much of the gap between the two clips.

This is a structural gap, not a tuning problem. Options: extend pose extraction
to far rallies and train on both; add far-player or ball features; gate the
detector on a serve-side classifier; or scope the product to near-serve points
and say so.

Hysteresis saturates around N=16–20 on clip 21 and degrades at N=24. Optimal N
differs per clip (16 vs 10), so N/M should be tuned on a validation set, not
baked in.

---

## Running it

```bash
python pipeline/parse_ground_truth.py                       # schema + timelines
python pipeline/extract_pose.py                             # Stream A cache (once)
python pipeline/make_windows.py                             # two-stream windows
python pipeline/train_active.py --holdout 21 --skip_loco --out /tmp/m.pt
python pipeline/extract_timeline.py --clips 21              # full-video pass
python pipeline/evaluate_events.py --clip 21 --model /tmp/m.pt --sweep
```

### Config flags

| flag | default | effect |
|---|---|---|
| `make_windows --no_global` | off | drop Stream B (51-dim pose-only tensor) |
| `make_windows --deadband_sec` | 0.3 | ± band around the transition flagged `boundary` |
| `make_windows --transitions` | none | label by hand-marked player transition instead of ball-based GT end |
| `train_active --no_global` | off | pose-only ablation from a two-stream npz |
| `train_active --holdout` | none | **exclude clips from training** — required before evaluating those clips |
| `train_active --include_boundary` | off | keep ambiguous straddling windows in training |
| `evaluate_events --stride` | 5 | frames between window ends |
| `evaluate_events --tol` | 2.0 | ±T seconds for matching a predicted point-end |
| `evaluate_events --n_dead/--m_live` | 4/3 | hysteresis; `--sweep` grid-searches instead |

---

## Design decisions

- **2s / 60-frame windows**, not the spec's 1s/30 — preserves the 743 hand
  labels, and 2s carries more context for a problem whose ceiling is context.
- **Bidirectional = False.** The spec flags this as a fork; the GRU is
  unidirectional and windows are scored at their **end** frame, so the design
  stays streaming-compatible. Offline-only accuracy is left on the table
  deliberately.
- **Class-weighted BCE** via `pos_weight` = neg/pos on the training split.
- **Split by clip**, never by window — adjacent windows from one rally overlap
  heavily, so shuffling them across a split leaks.

### Bug fixed along the way

`augment()` mirrored poses by **negating** normalized x. Pose x is bbox-relative
in `[0,1]`, so the mirror is `1-x`; negating mapped augmented samples into
`[-1,0]`, a range no real sample occupies — the augmented half of every batch was
off-manifold noise. Now mirrors correctly, gated on the present flag so
zero-filled missing frames are not turned into fabricated observations, and
Stream B mirrors consistently (`cx → 1-cx`, `dcx → -dcx`).

---

## Known ceiling

Pose + bbox cannot separate a serve-and-volley approach walk from a
between-points walk; frame-to-frame they are the same. Only context — ball in
play, how long the state has persisted — resolves it. Hysteresis buys back a
lot of this (event F1 0.42 → 0.80 across the sweep on clip 21), and no amount of
per-window modelling will close the rest.

The far-serve hole above is a *second*, separate ceiling, and probably the
larger one right now: it is not that the model is imprecise on far rallies, it
is that a near-player-only feature set has no way to know a far rally is live.

---

# Three-state model: {dead, transition, active}

Same two signals — near-player pose (Stream A) and near-player bbox kinematics
(Stream B) — with a third `transition` class as an explicit buffer.
`transition` is INTERNAL: the product still emits binary point-end timestamps.

Files: `make_state_windows.py`, `train_state3.py`, `evaluate_events.py --sweep`.

## Data used

**Only clips with a hand-written `ground_truth.json`.** `gt_path()` enforces this
at a single choke point, because several consumers discover clips by scanning
for *cache files* rather than ground truth — a stale cache would otherwise
silently rejoin the dataset.

| clip | why excluded |
|---|---|
| 35, 37, 63 | incompletely labelled (7.0% / 4.5% / 1.4% of the clip marked live) |
| 68 | no `ground_truth.json`; derived labels only, `bootstrap_matched_of_total: 10/45` |
| 23 | has ground truth but no court cache, so it cannot be extracted |

Leaves **10 clips, 3106 windows** (dead 876 / transition 907 / active 1323).
216 rallies remain labelled overall (102 near / 114 far), i.e. ~216 independent
transition events — that, not the window count, is the real sample size.

## The margin mistake, and what it cost

`extract_timeline --mode rallies --margin_sec` controls how much dead context is
decoded around each rally. The first run used **6s to save compute, which
recreated the exact flaw the whole pass existed to fix**: every span clip's dead
class was hard-capped at 6.0s from a transition — shallow, live-looking dead —
while the two full-coverage clips reached 15.9s and 53.0s.

Re-running at `--margin_sec 20` (2.1h vs 1.6h — the saving was never worth it)
lifted max dead depth to 12.4–34.0s per clip. Effect, holding everything else
fixed:

| configuration | clips | mean event F1 | false fires / live-min |
|---|---|---|---|
| binary, near-only, no deep-dead | 10 | 0.594 | 2.03 |
| 3-class, shallow dead (6s cap) | 13 | 0.613 | 2.07 |
| **3-class, deep dead (20s margin)** | **10** | **0.664** | **1.23** |

Better on *fewer, cleaner* clips. Coverage depth mattered more than clip count.

## Results (held-out clips 21 and 22)

| clip | P | R | F1 | median timing | false fires/live-min |
|---|---|---|---|---|---|
| 21 | 0.786 | 0.917 | 0.846 | +0.23s | 0.88 |
| 22 | 0.500 | 0.467 | 0.483 | −0.37s | 1.59 |

Window-level LOCO across 10 clips: 0.622 (majority-class baseline 0.43).
Per-clip range 0.461–0.800 — variance remains high.

Clip 22 is the persistent hard case in every configuration tried (binary 0.387,
shallow 0.467, deep 0.483). It has the higher share of far-serve rallies.

## Stream B no longer earns its keep

In the binary pipeline the global-motion stream helped window accuracy
consistently (LOCO 0.712 → 0.730). **In the 3-class deep-dead configuration that
reverses:**

| | LOCO window acc (10 clips) | event F1 (2 clips) | false fires/live-min |
|---|---|---|---|
| pose only (Stream A) | **0.638** | 0.659 | 1.61 |
| two-stream (A + B) | 0.622 | 0.664 | 1.23 |

Pose alone is doing the work. Event F1 is a wash (0.659 vs 0.664) and the
per-clip ranking flips; the only apparent edge for Stream B is aggregate false
fires, driven entirely by clip 22.

Plausible reason: once genuine deep-dead is in the training set, posture alone
separates standing/walking from playing, while Stream B's 8 dims add
detector-jitter noise that a small model (hidden 48) on 3106 windows
over-weights. **Do not assume the bbox stream is pulling weight** — at this data
scale there is no evidence it is.

## Transition band: measured, and asymmetric

Held-out P(active) versus offset from the true point-end:

| offset | −4s…−0.5s | +0.0s | +1.0s | +1.5s | +2.0s onward |
|---|---|---|---|---|---|
| P(active) | 0.79–0.91 | 0.70 | 0.51 | 0.33 | 0.22–0.33 |

Confidently live right up to −0.5s, crossing 0.5 at ~+1.1s, settled by +2.0s.
The ambiguity is entirely *after* the event, for two independent reasons that
agree: a window labelled at its END frame spans [E−2s, E], so windows ending
before a transition are pure, and only windows ending in (t, t+2s) are mixed.

Defaults are therefore **`--band_before_sec 0.5` / `--band_after_sec 2.0`**, not
a symmetric band. Switching from ±1.5s to −0.5s/+2.0s improved median timing
from +1.27s to +0.20s and recall from 0.583 to 0.750. Scale `band_after` with
window length; do not exceed ~2.5s, since even at +5s the model only calls dead
~50% of the time and a wider band starts consuming genuine dead.

## Detection, not prediction

There is no anticipatory signal. P(active) holds 0.79–0.91 until −0.5s and then
falls off a cliff — what ends a point is the ball, which is not an input, and
the near player's reaction is a consequence. The model *detects* the start of
dead time (median timing within ~0.25s) and cannot *predict* it. Real-time use
carries ~1.5–2.5s of confirmation latency; offline segmentation is unaffected.

The binding product risk is recall, not timing: at R≈0.47–0.92 depending on
clip, missed point-ends merge two points into one segment.

---

## Next steps, in priority order

1. **Clip 58.** 81 of the 216 remaining rallies (37.5%) and not yet extracted
   (~2h at `--margin_sec 20`). The single largest available increase in sample
   size. Caveat: it scored at chance in the binary LOCO, though that may have
   been the shallow-dead artifact rather than anything intrinsic.
2. **More eval clips.** 27 point-ends across 2 held-out clips cannot separate
   feature sets — it is why the Stream B question is unresolved. Extract full
   timelines for 2–3 more clips and hold them out.
3. **Diagnose clip 22.** Hard in every configuration (0.387 → 0.483). Higher
   far-serve share is the leading hypothesis; confirm by scoring its near-serve
   and far-serve point-ends separately.
4. **Clip 24 pose coverage** was 48% at 6s margin (others 64–100%); worth
   checking its court calibration.
5. **Drop Stream B, or justify it.** Pose-only matches it on 10-fold LOCO and on
   event F1. Either find a configuration where the bbox stream demonstrably
   helps, or simplify the model to one signal.

## Not done

- Broadcast-cut filtering (spec §6). Not implemented and not flagged in the
  manifest — a 2s window can still straddle a replay or graphic.
- Per-window hand-label override worksheet for the new timeline-sourced windows
  (`audit_windows.py` still targets the rally-span set).
- `optimize_energy.py --optimize` is broken independently of this work: it calls
  a `PointStartSystem(..., params=..., verbose=...)` API that no copy in the repo
  provides. `--extract` works; the parameter search does not.
