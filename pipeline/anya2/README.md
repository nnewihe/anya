# anya2 — point boundaries, rebuilt as three independent detectors

Three questions, three detectors, one shared substrate. Nothing here imports
from `pipeline/rally_reel/` or `pipeline/anya_*.py`; those keep shipping
untouched so every number below is A/B-able against them.

| detector | question | status |
|---|---|---|
| `near_serve.py` | did a point start with a near-side serve? | **built** |
| `far_serve.py` | did a point start with a far-side serve? | **built** |
| `point_end.py` | did the point end? | **built** (pose only) |
| `orchestrator.py` | **agent 4** — turn the three streams into a watchable reel | **built** |

## Substrate

| module | what it owns |
|---|---|
| `court.py` | image→court-metres map, and the two player gates |
| `camera.py` | per-frame warp back onto the calibration frame — the map is a function of time |
| `tracks.py` | ≤2 near + ≤2 far player tracks, with a per-frame eligibility flag |
| `balls.py` | ball detection with exclusion zones applied at source |
| `perceive.py` | the two pose ROIs (near 540p whole-frame, far native-res band) |
| `contract.py` | `Event` and `Requirement` — the only things detectors exchange |
| `eval.py` | one event-matching harness, three modes |

The two **player gates** are the user's rules, evaluated on the bounding box's
bottom-centre projected through the ground-plane homography:

- **near vs far** — whichever baseline the box bottom is closer to in y.
- **in bounds** — the box x-centre inside the doubles court plus 3 ft
  (`-2.284 … 10.514` m in the singles-origin court frame).

Doubles needs no re-calibration: the cached corners are the singles corners and
the homography is a full plane map, so the alley is analytic.

Tracking uses a **looser** zone than the gate, on purpose. Players walk off
court to the ball carts; a court gate would break the track and then
mis-reacquire it. So a player off court is still *tracked*, just not *eligible*.

### The map is a function of time

Calibration is four corners clicked once, on one reference frame. That is one
homography, and applying it to every frame is correct exactly as long as the
camera does not move. When the camera *is* bumped, nothing errors: the corners
still load, every projection still returns a number, and the numbers are simply
wrong from that frame on.

They are wrong in the direction that costs most. At far-court depth the 960×540
analysis frame is only ~4–5 px per court metre, so a 20 px jostle moves the far
baseline by four metres — `side()` starts calling far players near, `in_bounds()`
starts rejecting players who are on the court, and the far-serve band gates on a
`court_y` that no longer means what it meant at calibration.

`camera.py` keeps the single clicked calibration and makes the map time-varying
instead:

    court metres  =  H_ref  @  W_t  @  (image point at frame t)

`W_t` registers frame t against the frame the corners were clicked on — ORB plus
a RANSAC homography over the 540p proxy `perceive` already builds, at 5 Hz, so it
costs no extra decode and no model call. Every sample is registered against the
**one** reference, never chained, so nothing accumulates and a static camera
gives identity warps forever. `court.Geometry` composes the two; with no cached
track it *is* `H_ref`, through the same code path rather than a branch beside it.

Two details that are not incidental:

- **Ground-plane inliers are re-fit separately.** A single homography fits the
  whole scene only under pure rotation; a real bump translates a little too, and
  then the fence, the stands and the court surface each want a different one.
  Downstream `W_t` is only ever used on ground points, so the inliers landing
  inside the court polygon are re-fit alone when there are ≥20 of them.
- **The far band is the union over the whole track.** `perceive.far_band` becomes
  a fixed ffmpeg crop, and a crop cannot follow a camera. A far player who leaves
  it is not mis-projected, they are *absent*. So the band is taken around the
  reference band under every sampled warp at once. The cached far pose pass now
  records its crop and re-runs when the band moves.

Measured on a synthetic 30 s clip with a known 21 px pan/tilt/roll at t=15 s
(`--self-test` covers the same math analytically):

| | max court error |
|---|---|
| fixed homography | **2.63 m** |
| tracked | **0.10 m** |

The jostle is reported at frame 450 — the frame it was injected at.

#### What Data/77 actually turned out to be

The synthetic clip validated a code path real footage cannot support, and the
69-minute match said so. Two corrections came out of it.

**A real court is flat paint.** Across 414 samples the global fit had a median
of **1365** inliers and the *on-court* subset a median of **18** — see
`MIN_GROUND_INLIERS` for the numbers and what the original 20 did with them.
The synthetic court had speckle texture inside the quad, so it never exercised
the case where the ground re-fit is unsupportable, which on real footage is
almost always. The floor is now 60 plus an agreement gate.

**Data/77 has no jostle.** Its largest *sustained* level change over 69 minutes
is 0.92 px. What it has is monotonic drift — a settling mount — of the far
baseline:

| | 0–10 min | 30–40 min | 60–70 min |
|---|---|---|---|
| far-baseline offset | 0.2 px | 1.0 px | **2.2 px** |

Two pixels sounds like nothing and is not. The court's whole 23.77 m length
occupies ~117 px in the analysis frame, and the far end is where the perspective
compression is worst, so the same drift reads as **0.02 m of error at the near
corners and 2.4 m at the far ones** by the end of the match. That asymmetry is
the entire story of why a fixed homography fails quietly rather than obviously.

#### Data/78 — the clip with an actual jostle

Data/77 has drift; Data/78 has a bump, and it is what the module was built for.
The camera is dead still for ten minutes, is knocked at 10:53, and is still
again for the next thirty:

| | corner shift |
|---|---|
| 0–10 min | 0.5–1.0 px |
| **10:53** | → **45.2 px** (dx +24.1, dy −9.4 at the far baseline) |
| 12–42 min | 45.8–46.1 px, flat |
| **~43:30** | → **52.1 px**, a second smaller knock |

Sub-pixel stable across fifteen consecutive 2-minute windows, which is the
strongest evidence available that the estimate is real: noise does not hold
46.0 for half an hour. Under the fixed homography that is a **median 18.8 m**
court error, and the far player lands a median **9.15 m** from where they are.

Scored against the clip's own `tags.json` point starts (±3 s), both arms run off
the SAME cached pose passes so the court map is the only variable:

| point recall | fixed | tracked |
|---|---|---|
| before the jostle | 88.2% | **88.2%** |
| after | 65.8% | **89.5%** |
| overall | 72.7% | **89.1%** |

The first row is the control and matters as much as the second: the correction
does nothing when there is nothing to correct, which is what a change of
coordinates that is the identity on a still camera has to look like. Far
slot-frames rise +21% / +45% / +60% across the three thirds, far serves 37 → 76,
and the reel goes 960.7 s → 1195.2 s — the broken run was dropping about four
minutes of real rally.

#### Both halves of the fix, separated

The table above reuses the cached far dets, cut from a band aimed before the
camera moved — post-jostle that band is missing 99 px on the right and 83 px at
the bottom of where the far player now is. So it exercises the corrected
geometry but not the union band. Re-running the far pose pass over the union
band separates the two:

| arm | recall pre | recall post | far events/pt pre | post |
|---|---|---|---|---|
| A — fixed H, old band | 88.2% | **65.8%** | 0.53 | 0.74 |
| B — tracked H, old band | 88.2% | 89.5% | 0.59 | 1.74 |
| C — tracked H, union band | 88.2% | **94.7%** | 0.71 | 1.58 |

Both halves are load-bearing and they fix different things. The geometry
recovers 24 points of recall (A→B): the detections existed, and the court map
was throwing them away. The band recovers 5 more (B→C): those detections did
not exist, because the crop no longer contained the player — persons/sample
0.99 → 1.11 and empty frames 22.7% → 14.2%. Nothing downstream can recover a
frame the crop never saw, which is why the band is sized from the track rather
than from the calibration.

**The unresolved part.** Arm C's far-serve rate after the jostle is 1.58/point
against 0.71 before it. Widening the band explains some of it — the same ratio
in arm A is 0.53 → 0.74, and a wider crop admits more people — but not a factor
of 2.2. Near-side density is roughly flat across the same boundary (1.24 → 0.87),
so this is specific to the far side after the camera moved. It may be real (more
far serves in the later part of the match) or it may be false positives that the
recall number cannot see. `tags.json` records point starts only and misses
faults, so serve-level precision cannot be scored from it, and this stays open
until there is serve-level truth on this clip.

**One honest negative.** On the two coarse far-side backstops the corrected
geometry is marginally *worse* — `_trackable` survival 95.69% → 95.42% over
16,072 far detections. The correction pushes the far player ~1.8 m deeper in the
last third, toward `FAR_BACK_M`'s 12 m cutoff, and that 12 was measured against
the *uncorrected* projection: it absorbed some of this error. Near-side
eligibility moves the other way, +1.7 points in the last third. Correcting the
geometry without revisiting the constants that were fitted around its absence is
a job only half done, and `FAR_BACK_M` is the one to revisit first.

Turn it off with `ANYA_CAMERA_TRACK=0`, which restores the previous behaviour
everywhere at once: nothing estimates a track and nothing reads a cached one.

## Near-side serve — results

Scored against `ground_truth.json` with `eval.py`, ±2.0 s, greedy one-to-one,
restricted to each clip's labelled span. **All 13 trusted clips, 107 near
serves.**

| clip | labelled | recall | precision |
|---|---|---|---|
| 21 | 11 | 90.9% | 90.9% |
| 22 | 11 | 100% | 91.7% |
| 23 | 0 | — | **0 fires** |
| 24 | 2 | 100% | 66.7% |
| 25 (doubles) | 5 | 100% | 100% |
| 26 | 2 | 100% | 100% |
| **35 (out-of-sample)** | **5** | **100%** | **100%** |
| 36 | 8 | 100% | 100% |
| 38 | 8 | 100% | 100% |
| 40 (doubles) | 0 | — | **0 fires** |
| 43 | 6 | 100% | 100% |
| 50 | 5 | 100% | 71.4% |
| **58 (holdout)** | **44** | **65.9%** | **55.8%** |
| pooled | **107** | **90.7%** | **76.4%** |

Nine of eleven scored clips are at 100% recall. **Clips 23 and 40 have no near
serves and the detector fires zero times on either** — 40 being doubles, so two
near players stand at the baseline through 13 far-serve rallies without
producing a start.

### How it got there

Three fixes, in order of how much they mattered. The first two are defects in
the seed that eyeball review could not have found, because both are invisible
without labels.

1. **The trophy is a phase, not an instant.** The seed multiplies three shape
   terms sample-by-sample. On Data/38's five missed serves every term clears its
   threshold comfortably — `hi_head` +0.19…+0.27 against a 0.10 line — while the
   product never exceeds 0.30, because the tossing arm reaches full extension a
   sample or two before the racket arm settles. Each term is now dilated ±0.20 s
   before the product. Recall 79% → 100% on the nine short clips.

2. **Trophy onset is not the point start**, and lands +1.63 s late. Walking back
   to the last instant of the ready stance gets to +1.13 s; the rest is a
   labelling convention. Two other anchors were tried and are worse — starting
   the ready run scores sd 2.46–2.68 against the hand split's 1.41.

3. **The eval was scoring unlabelled regions.** Clip 38 is labelled to 206 s of
   420 s, and all three of its apparent false positives lived in the tail.

### What clip 58 showed, and why it was worth holding out

Clip 58 is a 55-minute match; the other eleven are 7-minute snippets. Every
parameter was fixed before its perception pass finished. It scores 65.9% / 55.8%
against ~100% / ~93% on the clips used for tuning, and the gap decomposes into
two findings, neither of which is a threshold problem:

**The label lead does not transfer.** Fitted on the nine short clips the
hand-split lead is 1.13 s; clip 58 wants 2.4 s. Per-clip leads genuinely span
0.9–2.4 s, and at a ±2.0 s tolerance no constant satisfies both ends: at 1.13
clip 21 scores 100% and clip 58 57%; at 2.00 clip 21 falls to 91% and clip 58
reaches 68%. Clips 22–50 do not move at all. `SERVE_LEAD_S` is now the median
over all ten clips (1.63) — a compromise, and worth only ~9 points on clip 58.
**A tighter tolerance would need a per-clip lead estimated from the clip's own
detections, not this constant.**

**The rest is overheads, and pose cannot fix it.** Of clip 58's 24 false
positives, **23 fall inside a labelled rally** rather than in dead time — and
all 23 are struck from *inside the serve zone*, at the same court depth as real
serves (median court_y −0.62 m for both). These are baseline overheads and high
defensive balls during a point. The seed predicted this exactly: a smash and a
serve are not different at the joints, and the court gate is the only ball-free
separator there is. It has nothing left to give here, because the player really
is standing where a server stands.

That is not a tuning problem, and no threshold will move it. **The fix is
structural: do not look for a serve while a point is live.** The detector
already declares this — `Requirement(windows="between_points")` — and honouring
it is the composition layer's job, once the point-end detector exists. Expect
most of clip 58's precision gap to close then, and treat 55.8% as the floor a
detector reaches with no knowledge of whether a point is in progress.

## Far-side serve — results

Same scoring: ±2.0 s, greedy one-to-one, restricted to the labelled span.
**Eleven clips carry a far serve, 129 in all.** Clip 58 was held out until its
far pass finished; clip 35 was out of the corpus entirely until relabelled.

| clip | labelled | recall | precision |
|---|---|---|---|
| 23 | 15 | 100% | 78.9% |
| 24 | 12 | 100% | 92.3% |
| 25 (doubles) | 10 | 100% | 71.4% |
| 26 | 11 | 100% | 45.8% |
| **35 (out-of-sample)** | **15** | **93.3%** | **93.3%** |
| 36 | 9 | 100% | 75.0% |
| 40 (doubles) | 13 | 100% | 81.2% |
| **seven far-dominant clips** | **85** | **98.8%** | **74.3%** |
| 21 | 1 | 0% | 0% (14 fires) |
| 22 | 4 | 75% | 18.8% |
| 50 | 2 | 0% | 0 fires |
| 38 | 0 | — | 9 fires |
| 43 | 0 | — | 3 fires |
| **58 (holdout)** | **37** | **51.4%** | **27.5%** |
| pooled, all 13 clips | **129** | **79.8%** | **64.8%** |

Bias +0.07 s. **Every far serve on every far-dominant clip is found.** The clips
that look terrible are near-dominant — 1, 4 and 2 far serves against 11, 11 and 5
near ones, and clips 38 and 43 with NO far serves at all — where the far player
spends the whole clip RETURNING, and a return is a serve motion the detector has
no way to place inside a point.

Clips 38 and 43 are the cleanest statement of that: zero labelled far serves, and
the detector fires 9 and 3 times inside their labelled spans. There is no
threshold that fixes this, because the motion really is a serve motion.

### Clip 58, the holdout — and what it says about the scoring

Clip 58's far pass finished after the parameters were fixed. It scores **21.6%
recall / 11.6% precision** on 37 far serves, against 100% on the six clips used
to tune. That is a worse collapse than the near detector's, and it decomposes
into two separate things:

**Most of it is the label lead again, not the detector.** 30 of the 37 serves
have a detection within 12 s; the median nearest-detection error is +2.32 s with
sd 2.31. An oracle per-clip lead of +2.25 s would give 24/37 = 65%. The threshold
is not implicated at all — at threshold **zero**, recall is still only 32%.

**The rest is genuine signal absence.** The trophy shape clears its floor
anywhere in the far band for only 23 of 37 serves. On clips 25 and 40 the same
diagnostic said 10/10 and 13/13 and the fault was slot assignment; here the
signal simply is not there for a third of the serves.

### Detection quality vs label-convention alignment

The lead exists only to match a labelling convention — `ground_truth.json`'s
`start` is a point boundary marked before the server moves. It is an EVALUATION
alignment parameter, not a product one: a reel needs pre-roll, which is a
user-facing choice, not an estimate. So it is worth reading recall at a wider
tolerance and with a per-clip lead removed, which separates "did we find the
serve" from "did we time it to this labeller's convention":

| | R @±2 s | R @±4 s | R @±2 s, per-clip lead |
|---|---|---|---|
| **near**, 9 short clips | 100% | 100% | 100% |
| **near**, clip 58 | 66% | 86% | 80% |
| **near**, pooled (102) | 84% | 94% | 91% |
| **far**, 6 far-dominant clips | 100% | 100% | 100% |
| **far**, clip 58 | 22% | 54% | 59% |
| **far**, pooled (114) | 71% | 82% | 83% |

The per-clip-lead column is an upper bound a deployment cannot fit, since it
needs labels. It is here to attribute error, not to claim performance.

Read that way: **the near detector finds essentially every serve it is shown and
the residual is convention.** The far detector does the same on far-dominant
clips, but on a full-length match a third of its serves are genuinely invisible
to it even with perfect timing — that is the honest limit, and it is the next
thing to attack on the far side.

### Why it is not the near detector with a flag

Two of the near trophy's three terms do not survive the change of viewpoint and
scale. Measured on Data/23's 15 far serves, per pose sample, against every
non-serve sample in the clip:

| term | serve | non-serve |
|---|---|---|
| wrist above head | 13% | **0%** (p99 = +0.002) |
| lower wrist near the shoulder | 11% | 2% |
| **hand split** | 35% | **58% — inverted** |

The hand split is the sharpest cue the near detector has and is worse than
useless here: at the trophy the far player's arms extend VERTICALLY, so their
horizontal separation is small exactly when the near view has it large, while
ordinary walking swings the arms apart. The far trophy is elevation-only and
never reads `gap`.

The serve-zone court band is dropped too. A far player's ground point is 22-32
px up the frame, where two pixels of box-bottom error is metres of court: their
`court_y` spans 19.7-28.6 m while really moving about three, and sits a median
4 m behind the baseline they are standing on.

### Three substrate defects it exposed

Each was found by measurement, and each was costing recall outright:

1. **Crop-truncated boxes were projected as ground points.** A person straddling
   the bottom of the far band is cut off, so their box bottom is the *crop
   boundary*, not their feet. 23–61% of far-band detections are truncated this
   way — and being closest to the camera they carry systematically *higher*
   confidence than the real far player (0.82 vs 0.54 on Data/40). They won every
   competition for the two far slots and pushed the server out of the tracks:
   clip 40's serve trophy was present at full strength in the band for all 13
   labelled serves while the tracks caught six. Fixing it took clip 40 from
   46% to 100% recall.

2. **`FAR_BACK_M` was deleting a third of clip 23's far detections** — real
   players rejected for standing where the homography said they were. Clip 23
   went 66.7% → 100% on that alone. Re-measured after the truncation fix: the
   spread is genuine homography noise, so 12 m stands.

3. **The claim score preferred the biggest box**, which is a *depth* proxy and is
   side-specific. On the near side bigger means closer to the near baseline, so
   it was accidentally right; on the far side bigger means closer to the net, and
   the server is by definition the deepest player. Replaced with proximity to the
   player's own baseline — right on both sides at once. Clip 25 went 30% → 100%,
   with no near-side regression.

### Measured dead ends — do not retry these

Two local cues were tested for separating far serves from far false positives,
because if one worked it would fix the near-dominant clips without any context:

  box height / that slot's clip median   **AUC 51%** — no separation at all
  projected court_y at the trophy        AUC 66% — real, far too weak to gate on

So there is no local depth cue. This agrees with every previous far-side attempt
(DESIGN.md's velocity dead end, and the far-gate taxonomy's finding that the
far player's ordinary play looks like a serve from that viewpoint).

### The residual is structural, and it is the same one the near detector has

Over all 67 false positives inside labelled spans on eleven clips, **79% fall in
a live point** — 75% in-rally, 4% the far player reacting to a near serve —
against 21% idle raises in dead time. The shipped `anya_far_serve` measured the
same taxonomy at 56% over 14 clips; ours is more concentrated because recall is
higher.

None of that is visible from inside a far-player pose crop. The module declares
`Requirement(windows="between_points")` and leaves the arbitration to the
composition layer, which will have the near serves and the point ends that
settle it. **Treat 52% pooled precision as the floor for a serve detector with
no knowledge of whether a point is in progress** — and note that the near
detector bottomed out at 55.8% for exactly the same reason.

## Point end — results

**Pose only. The ball is not read anywhere in the module**, at the user's
request: the shipped policy makes the ball trace primary, and that is not
dependable on clay, where the ball is low-contrast against the surface for much
of its flight.

Scored ±2.0 s over all **236 labelled ends on 13 clips**:

| | recall | precision | bias | truncations |
|---|---|---|---|---|
| shipped **ball-trace** policy (clips 21/22/23) | 48% | 65% | +0.87 s | 0 |
| **anya2 pose-only** (same 3 clips) | **57.1%** | 55.8% | −0.24 s | **0** |
| **anya2 pose-only** (all 13 clips) | 49.6% | 40.2% | −0.27 s | **0** |
| **anya2 pose-only** (clip 35, out-of-sample) | 60.0% | 60.0% | −0.92 s | **0** |

Better recall than the ball-based policy with no ball at all; behind on
precision. Per-clip recall spans 30.8%–78.6%; clip 58 (the 55-minute match) and
clip 40 (doubles) are the weak ones at ~30–38%.

### The point-end objective

`orchestrator.point_end_score` is **the** metric, at the user's direction;
everything else in `score_reel` is diagnostic. It scores **the point end the
system chose** — `Segment.end_t`, before pre- or post-roll — against the
labelled end. Deliberately not the cut point: roll is a separate, later decision
about how much air to leave around a correct answer, and folding it in lets a
tuning run paper over a bad end by padding it.

That does **not** make the two independent. Roll cannot shift a scored `end_t`
directly, but it changes where segments end, `smooth` then merges a different
set of them, and the segment covering a given rally is no longer the same one —
moving post-roll 1.0 → 2.0 took PES +0.051 → +0.008. The coupling runs through
`merge_gap_s`, not the arithmetic, so re-check PES after moving roll.

With `e = end_t − gt_end`:

| range | score |
|---|---|
| `\|e\| ≤ 2 s` | **1.0**, flat — a point end is a moment a couple of seconds wide |
| `e > +2 s` | linear from 1 down to **0** at the next labelled point start; never negative |
| `e < −2 s` | `2 − exp((\|e\| − 2) / 2)` — crosses zero at 3.39 s early, −5.4 at 6 s |

**The penalty saturates at −10, which the corpus forced.** Uncapped, one point
of 208 was 88.2% of the objective and the worst three were 98.9% — that is
tuning against a single moment of clip 58. The cap is also the honest shape: an
end 15 s early and one 21 s early destroyed the same rally. Set
`EARLY_MAX_PENALTY = None` for the pure exponential; `pes_errors` is reported
either way.

Tuned against it over 208 ends on 11 clips, **only the hysteresis moved** —
`LIVE_HI/LO` 0.50/0.35 → **0.40/0.25**. `FAR_VETO_W = 0.8`, `TURN_HOLD_S = 0`,
confident pairing, `LIVE_SMOOTH_S = 4.0` and `est_duration_pct = 85` were all
re-checked at the new hysteresis and all held.

| hi / lo | PES | within ±2 s | >2 s early | >2 s late | median err |
|---|---|---|---|---|---|
| 0.50 / 0.35 | +0.040 | 71 | 49 | 88 | +0.9 s |
| **0.40 / 0.25** | +0.051 | **75** | 41 | 92 | +1.2 s |
| 0.35 / 0.20 | **+0.102** | 64 | 31 | 113 | +2.5 s |
| 0.30 / 0.15 | +0.064 | 57 | 23 | 128 | +5.1 s |

**0.40/0.25 is not the top of the PES column, and was chosen knowing that.**
0.35/0.20 maximises the objective, but not by putting more ends on the labelled
time — it does the opposite. Points inside the ±2 s plateau *fall* from 71 to
64 and median error grows to +2.5 s; what improves is only that fewer ends land
early, bought by holding the live state open longer so every falling edge
arrives later. The objective rewards that because late costs it only linearly.
0.40/0.25 is the best row on the column that says *the end was right*.

**The histogram settles it.** Binned by how early, the mistimed ends are one
mode against the plateau edge and nothing beyond it:

| s early | 0.50/0.35 | 0.40/0.25 | 0.35/0.20 |
|---|---|---|---|
| 2–3 | 23 | 21 | 14 |
| 3–4 | 11 | 5 | 5 |
| 4–5 | 2 | 2 | 1 |
| 5–6 | 2 | 2 | 1 |
| 6+ | 0 | 0 | 0 |

Every genuinely mistimed end in the corpus is **under 6 s early**, at every
setting, median ~2.7 s and p90 ~4.0 s. There is no catastrophic mid-rally tail
for the exponential to punish — at 2.7 s early the score is still +0.35 — so the
penalty barely engages, and the PES ranking is driven mostly by the eleven
points with **no segment at all**, scored as full-duration truncations, which no
hysteresis setting changes (11, 11, 10 across the three rows). Optimising PES
here is largely optimising a serve-recall problem through the wrong knob.

### Near serve — the wrist-separation term is gone

The trophy was a product of three terms and one of them, `split` (the two
wrists ≥ 0.149 body heights apart), **assumed a measurement the perception
cannot make**. A player who serves side-on with the racket arm on the far side
of their body occludes that arm through the toss, and the pose model puts *both*
wrist keypoints on the visible arm. On Data/43's two missed serves the wrists sit
**18 px apart on a 221 px body** (0.081 BH) and **8 px on a 229 px body**
(0.037 BH). The other two trophy terms score 1.000 on the same samples, so the
product was zero for want of evidence that was never there — and no threshold
recovers it: at a `split` minimum of 0.04 the first serve reaches only 0.343 and
the second is still exactly 0.000.

`TROPHY_USE_SPLIT = False`. Scored over 96 labelled near serves on 11 clips at
the ±5 s tolerance that credits the label-timing cases:

| | recall | precision | F1 | misses |
|---|---|---|---|---|
| with `split` | 93.8% | **80.4%** | 86.5 | 6 |
| **without** | **97.9%** | 76.4% | 85.8 | **2** |

Both Data/43 misses go, as do 36@271.6 and 38@59.8. The two survivors are a
**0.9-second labelled "rally"** and a matching reshuffle — neither reachable by
any threshold. End to end the objective goes **+0.141 → +0.236** and points the
reel never covers drop **8 → 5**, because serve recall feeds straight into
coverage, which dominates the objective.

The cost is 7 new false positives, and they are exactly the family `split`
existed to reject: **5 of 7 are mid-rally**, a player with both arms up during
live play. `lo_elev` does not cover that case — it saturates at 1.0 above
−0.052 BH, so it only rejects a genuinely *one-armed* gesture (hand to cap, a
wave), which it still does.

Two things were measured and are worth recording:

* **`live_gate_near` does not catch them.** Swept 0.5→0.9, **5 of 7 survive at
  every setting** and PES gets *worse* (+0.236 → +0.207). The live score is low
  at these moments despite them sitting inside labelled rallies.
* **Most are harmless anyway.** Two are dropped by the orchestrator and four
  land inside segments that are majority live tennis, where a spurious start
  just re-anchors a segment the reel was keeping. Only **two** do real damage
  (36@380.8, 38@193.6), stretching a segment across mostly-dead footage. Both
  fire *between* points, 4–6 s before a real serve, so the targeted fix is a
  tighter start-merge window rather than restoring `split`.

Reviewed on video, at least two of the seven (58@46.4, 24@382.8) look like real
serves inside over-long labelled rallies — 58's "rally 26–68 s" is 42 seconds
and almost certainly spans more than one point — so the precision drop above is
probably overstated.

### The nine points the reel never covers

They carry 78% of the objective's penalty, so they are worth naming. Every one
fails the same way twice: **no serve was detected at all**, and `recover_missed`
— the live-score second line of defence — did not fire either, because live
never reaches its 0.75 threshold inside the rally (observed max 0.22–0.92,
median 0.06–0.68). No orchestrator rule drops any of them; nothing reaches the
in-rally gate, the rapid-repeat filter or the service-run DP to be dropped.

| clip / pt | side | best serve candidate | shortfall | near trk | far trk |
|---|---|---|---|---|---|
| 26 / 9 | far | far @ 319.9 s, p=0.988 | *fires, 1.3 s after a 3.4 s rally* | 1.00 | 0.60 |
| 35 / 0 | far | far @ **−0.6 s**, p=0.665 | +0.085 | 0.28 | 1.00 |
| 36 / 9 | near | near @ 270.7 s, p=0.616 | +0.084 | 1.00 | 0.96 |
| 36 / 10 | near | near @ 280.2 s, p=0.692 | **+0.008** | 1.00 | 1.00 |
| 43 / 0 | near | none at any score | — | 1.00 | 0.83 |
| 50 / 5 | far | far @ 150.4 s, p=0.265 | +0.485 | 0.21 | **0.06** |
| 58 / 4 | near | far @ 239.7 s, p=0.465 | +0.285 | 0.90 | 1.00 |
| 58 / 60 | far | none at any score | — | 1.00 | **0.50** |
| 58 / 61 | far | none at any score | — | 1.00 | **0.58** |

Four distinct causes, and only one of them is about point ends:

1. **Threshold near-misses (36/9, 36/10)** — tracking is perfect and the
   detector scored 0.616 and 0.692 against a 0.70 bar. 36/10 misses by **0.008**.
   Dropping the near threshold to 0.61 recovers both, at whatever precision cost
   the near detector's own sweep says.
2. **Far-player tracking (50/5, 58/60, 58/61)** — far coverage of 0.06, 0.50 and
   0.58 inside the rally. Clip 50's far coverage is 0.20 *clip-wide*. A far serve
   cannot be detected from a far player who is not there; this is a perception
   problem, not a detector one.
3. **Label artifacts (35/0, 26/9)** — 35/0 starts at 0.1 s, so the serve happened
   before the clip did, and the detector still found it at −0.6 s. 26/9 is a
   3.4 s labelled rally whose serve is detected 1.3 s *after* it ends, i.e. the
   detection belongs to the next labelled rally. Both are scoring artifacts.
4. **Genuine detector failures (43/0, 58/4)** — good tracking, no candidate at
   any score. Two of 208.

The corpus context that makes this legible: **30% of labelled points have a
neighbouring point within 5 s** and the median inter-point gap is 10.3 s, when
real tennis runs 15–25 s. The labels mark live ball-in-play stretches, so a
fault or a let becomes two adjacent "points". Points with a sub-5 s neighbour are
3× likelier to end up uncovered (8.1% against 2.7%), which is a real effect but
not the dominant one — 62 points have such a neighbour and only 5 are uncovered.

**The actionable item is `recover_live_thr`.** It exists precisely to catch
points no serve detector found, and on all nine it never fires. Lowering it from
0.75 would recover 43/0 (live peaks 0.92) and 35/0 (0.85) at least, and is a far
cheaper fix than moving a serve threshold.

### The truncation column was measuring the wrong thing

**"Zero truncations on every clip" was a metric artifact, and the reel truncated
anyway.** `eval.py` counts a truncation only when a *matched* end lands more
than 2 s early. At 40.2% precision the majority of emitted ends match no label
at all, so every end that cut a rally short by landing mid-point was invisible
to the one column that existed to catch it.

What made that harmful rather than merely wrong was `_pair_once`, which took
`cand[0]` — the **earliest** end in the point's window, unconditionally. With
most candidates being false positives, that rule systematically preferred a
mid-rally artifact over the real end whenever both fell in the window.

`score_reel` now carries the reel-level question instead: for a rally the reel
got *some* of, how many seconds of its **tail** are missing. Three fixes,
decomposed over all 208 labelled ends on 11 clips:

| | recall | precision | whole | trunc pts | trunc_s | dead_s |
|---|---|---|---|---|---|---|
| shipped before | 47.1% | 36.3% | 108 | 21 | 127 | 926 |
| + `turn_away` hold gate | 48.1% | 37.6% | 115 | 21 | 122 | 959 |
| + confidence-aware pairing | 47.1% | 36.3% | 117 | 20 | 106 | 1100 |
| + far-veto weight 0.8 | 49.5% | 37.7% | 119 | 18 | 119 | 998 |
| **all three** | **50.5%** | **39.3%** | **128** | **17** | **93** | 1213 |

Each is positive alone and they compose. 12 of the 17 remaining truncated points
are on clip 58.

1. **The union vetoed a player it knows nothing about.** All five members —
   walking and `near_end`'s four — are computed from the *near* pose shim, yet
   `raw = max(near, far) * (1 - union)` let near-player posture erase the far
   player's activity. The commonest truncating case is exactly that: the near
   player hits an approach and stands watching (`settle` saturates) while the
   far player sprints to run it down. Exempting the far term *outright* is the
   opposite error and costs more than it buys — detected segments run 24.3 s
   against 15.7 s, because nothing then vetoes the far player walking to the
   ball after the point. The union is *weak* evidence about the far player, not
   none; `FAR_VETO_W = 0.8` is where recall and precision both peak.

2. **`turn_away` fired on forehands — and the gate for it does NOT ship.** The
   `conf` ramp already handled a player seen edge-on, but not rotation *past*
   square: an open-stance takeback inverts the shoulder order with real
   magnitude. Ungated, the signal fires above 0.5 on **3.3% of live tennis**,
   and eroding it over a 2 s hold window cuts that 8.5× to 0.39%. That
   measurement is real and reproducible — and the gate still loses end to end,
   on every arm and every column, producing *more* badly-early ends rather than
   fewer (see `near_end.TURN_HOLD_S`, which ships at **0.0**). Suppressing the
   signal raises the live score and moves ends later. A half-window phase shift
   was tried, on the theory that the centred erosion was merely delaying genuine
   turns, and did not help either. The row above is its score under the earlier
   reel-based metric, kept because it is what the decomposition measured; it is
   not a claim that the gate is on.

3. **Pairing threw away the discriminator it already had.** `detect_ends`
   computes `p` = how far the live score falls *and stays fallen*; a real end
   scores near 1, a mid-rally dip low. Picking the argmax instead of the
   earliest costs nothing and needs no new evidence.

### Four measurements, in the order they killed the obvious designs

1. **Instantaneous activity does not separate live from dead** — AUC 38–75%
   per sample, at or below chance on the hardest clips. A rally contains long
   quiet beats; dead time contains a player walking to the ball. Motion does not
   stop at the end of a point, it changes character.

2. **Sustained quiet is nearly perfect and useless for timing.** Both players
   quiet for 1.5 s covers 0.0–1.2% of live play and 7–21% of dead time — close
   to proof the point is over. But the first such window after a labelled end
   arrives a **median +78 s** later. Quiet marks changeovers, not point ends.

3. **Walking has the coverage but not the timing.** A walk onset follows 212 of
   216 ends — essentially every one — but at median +5.1 s, p75 +15.9 s, because
   the walk is a *consequence* of the end. `near_end`'s four pose signals cover
   the gap between the last ball and the first step, and their union with
   walking is far better than any part (clip 58: p75 +6.7 s vs +24.8 s for
   walking alone) — even though every corroborator is individually *worse* than
   walking. They are worse on average and earlier where walking is late, which
   is all a `max()` asks of them.

4. **But the union is a weak live/dead signal on its own** (frame AUC 60%; its
   parts 48–59%, `settle` below chance). That is not a failure of the signals —
   it says the question they were built for is not this one. What carries
   live/dead is **player activity integrated over seconds**:

   | | AUC (live > dead) |
   |---|---|
   | near activity, 8 s | 79.5% |
   | far activity, 8 s | 75.1% |
   | max(near, far) − union | 82.6% |
   | **max(near, far) × (1 − union)** | **86.7%** |

   A product, not a sum: the union's job is to **veto** activity, not to be
   traded against it. A player walking to the ball is active and emphatically
   not playing, and only a multiplicative term can say so.

   The far-activity term exists only because anya2 tracks the far player —
   no previous point-end work here could, and memory records "far-serve rallies
   read as dead" as the biggest error source of the dead/live GRU.

5. **The end is the FALLING EDGE of that score, not the onset of a dead state.**
   Scored as dead-state onsets the detector emitted 504 candidates for 216 ends,
   and no local feature separated the good ones (run duration AUC 53%, depth
   64%, quiet overlap 1%). What does separate them is that a real end is
   *preceded by play* — which a falling edge encodes and an onset does not.

### Known gaps

- **Precision, and clips 58 and 40.** Half the corpus' ends are on clip 58 and
  it scores 38.3%/25.0%. The pooled row is dominated by it.
- **The near-slot shim.** The walking classifier is fed from anya2's near tracks
  through a shim npz; near coverage varies 42%–88% by clip, and on clip 23 the
  `settle` and `stance_drop` signals produce no usable onsets at all.
- The ball has **not** been tried as a minor corroborator. It may be worth a
  bounded experiment for precision, but only if it can be shown to earn its cost
  on clay.

## Can far-player depth be made reliable, and does it help?

**Yes to the first, no to the second, and the second is the one that matters.**

*Reliable:* raw per-frame `court_y` for a far player is noise-dominated — 11.9%
of consecutive samples imply a speed above 9 m/s, which nobody can run, and the
p90 implied speed is 12.3 m/s. A 0.7 s median filter fixes it: **impossible
samples 11.9% to 4.2%, p90 implied speed 12.3 to 3.2 m/s.** So depth *is*
recoverable, and the box-bottom projection is the better of the two estimators
available — a box-height proxy is worse (20% impossible against 12%).

*But it does not strengthen the far serve detector.* Every depth-derived
quantity tested sits at chance for separating a true far serve from a far false
positive:

| quantity | AUC |
|---|---|
| depth at the trophy, relative to the clip median (raw) | 41% |
| the same, median-filtered 0.5 / 1.5 / 3.0 s | 41 / 46 / 47% |
| step into the court during the strike | 47% |
| depth change from −1 s to +2 s | 55% |
| pre-serve drift, −4 s to −1 s | 47% |

The reason is structural, not a measurement failure: **the false positives are
the far player RETURNING, and a returner stands at essentially the same depth as
a server.** Depth cannot separate two things that happen in the same place. What
*does* separate them is what the player was doing before the racket went up —
stillness (AUC 78%) and a sustained ready stance (79%) — which is what the
refinement below uses instead.

## Far-serve refinement: how the ready phase is read, and pre-serve stillness

Two changes, both from measuring what actually separates a far serve from the
far player's *return* — which is where the false positives were.

**1. The ready term was inert.** It read the **max** of `ready` anywhere in the
6 s before the trophy. Over 105 true far serves and 101 false positives its
median is **1.00 on both**, separating at AUC 62% — any 6-second window contains
some quiet moment, so a returner who stood still once five seconds ago scored a
perfect ready. Read instead as the **mean over `[k−2.0s, k−0.3s]`**, ending at
the trophy, it separates at **AUC 79%** (true 0.90, false 0.62). The difference
is "was quiet at some point recently" versus "was quiet right up until the racket
went up", and only the second is a service stance.

**2. A server is stationary before serving; a returner has just been running.**
New term, and the largest single discriminator found: the median of the player's
own translation (box centre, body heights/s) over `[k−5s, k−1s]` is **0.15 for
true serves and 0.31 for false positives, AUC 78%**. It is the mirror of ready
and independent of it — **ready is about the arms, this is about the feet** — and
it is local to this detector, needing no other agent. It enters multiplicatively
at weight 0.30 so it can veto a candidate that is clearly mid-rally without
deleting one whose track was noisy; an untracked stretch scores 0.5, because not
knowing where the player was is not evidence either way.

Threshold re-swept, since both terms multiply the score down:

| ready | stillness | thr | recall | precision | F1 |
|---|---|---|---|---|---|
| max-6s | — | 0.90 | 81.4% | 51.0% | 62.7 |
| max-6s | 0.3 | 0.85 | 80.6% | 63.0% | 70.7 |
| **mean** | **0.3** | **0.75** | **79.8%** | **64.8%** | **71.8** |
| mean | 0.3 | 0.85 | 70.5% | 72.8% | 71.7 |

**Corpus: 81.4%/51.0% → 79.8%/64.8%** — 1.6 points of recall for **13.8 of
precision**. Per clip:

| clip | before | after |
|---|---|---|
| **21** (1 far serve) | 0% / 0%, **14 fires** | 0% / —, **0 fires** |
| 26 | 100% / 45.8% | 100% / 64.7% |
| 36 | 100% / 75.0% | **100% / 100%** |
| 25 | 100% / 71.4% | 100% / 83.3% |
| 24 | 100% / 92.3% | 91.7% / **100%** |
| 38, 43 (no far serves) | 9 and 3 fires | **2 and 0** |

Clip 21's fourteen phantom far serves — the case this refinement was aimed at —
are **gone entirely**, and the eleven that were the far player returning are
exactly the ones the stillness term rejects.

Reel effect is small and in the tightening direction: whole points 143 → 137,
live retained 89.5% → 87.9%, reel 47.5% → 45.5% of source. Fewer far detections
means fewer anchors, so a few points lose their start.

## Roll tightened to 1 s, and a far-side leak found (2026-08-25)

**Roll is now 1.0 s / 1.0 s** at the user's direction. Corpus effect:

| | pre/post 3.5/4.0 | **1.0/1.0** |
|---|---|---|
| points whole | 207 | **143** |
| partial | 22 | **85** |
| live retained | 95.3% | **89.5%** |
| reel | 61.6% of source | **47.5% of source** |

### Debugging Data/21's far false positives

Clip 21 has **11 near serves and 1 far**, and the far detector fired **14 times
at confidence 0.92–1.00** — while missing the one real far serve entirely. The
14 split into two unrelated causes:

**Three were a near player occupying a far slot.** Rendering the frame settled
it: at t=60.3 s the near player had moved forward, appeared inside the far-band
crop, and was tracked as a far player with a **69 px box where the far baseline
allows 26**. The existing crop-truncation filter missed them because their boxes
sit just inside the crop edge rather than on it.

The fix is geometric, in `tracks._height_plausible`. A person on the court has a
predictable pixel height — the court's own width at their depth gives
px-per-metre, and a player is ~1.75 m. Measured across four clips and both sides
the observed/expected ratio has median 0.82–1.06 and **p90 never above 1.24**, so
a box beyond 1.40× is a contradiction: not a player at that depth, but a closer
one mis-projected. The bound is one-sided — a box *smaller* than expected is
ordinary (crouching, partial detection, a clipped head) and rejecting those would
cost real tracking.

**Corpus effect: far precision 47.3% → 51.0% for 0.8 points of recall.** Near is
untouched.

**The other eleven are the far player returning a near serve** — genuinely the
far player, genuinely a serve-like motion, in a live point. Those are contextual
and the orchestrator's live gate is the only thing that can see them. It is set
at 0.80 and catches 3 of the 11; their median live score is 0.72 against 0.17 for
the dead-time ones, so a gate at 0.35 would catch 9 of 14 on this clip — but
corpus-wide that costs far recall 82% → 54%. **Not worth it**, and recorded here
so it is not retried:

| live gate | far recall | far precision |
|---|---|---|
| 0.80 (current) | 79.1% | 54.0% |
| 0.60 | 76.0% | 56.3% |
| 0.50 | 71.3% | 59.0% |
| 0.40 | 62.8% | 60.0% |

Also tested and rejected on this clip: dropping starts that contradict the
service run, and raising `min_service_run` to 4 or 5 — **neither changed clip 21
at all**, because the phantoms cluster into runs of 2–3 that any game-length
minimum accepts as legitimate.

## Toss evidence, and two confidence rules (2026-08-25)

### The pose toss score (far_serve.toss_score)

A **separate** score, never folded into the serve `p`, so the orchestrator can
arbitrate. Read from the tossing arm only: the wrist started low, is offset
laterally at the top, is held above head height, and how far above it reaches —
four ramps, averaged.

| | AUC vs far false positives |
|---|---|
| each component alone | 64–66% |
| **the mean of the four** | **75%** |
| correlation with the serve score `p` | **+0.04** |

That last row is the point: it is independent evidence, not a restatement of the
trophy. Two findings worth keeping — the toss arm is **not** vertical (true
serves have a *larger* lateral offset, 0.148 vs 0.090 bh, because the ball goes
up and forward across the body), and the four components must be **averaged, not
multiplied** (the product scores the same AUC but is degenerate, costing 53% of
true serves to cut 88% of false ones).

### The ball toss tracker — re-aimed, now works, still not shipped

A first attempt concluded the ball was undetectable at far-court range. **That
was wrong, and for two aiming bugs rather than any limit of the ball model:**

- **The window was before the toss.** `far_serve`'s event time is lead-corrected
  to ~0.9 s *before* the trophy, so `[t−0.2, t+1.0]` covered `trophy−1.1s …
  trophy+0.1s` — while the ball is still in the hand. It is airborne from about
  `trophy+0.0` to `+0.6`.
- **The ROI was above the head.** At this range the toss does not rise far above
  the head in *image* terms; it sits just above the tossing hand.

Re-aimed at the **tossing wrist**, anchored on the **trophy**, at native
resolution with SAHI tiling (160 px tiles, 30% overlap, imgsz 320, distance NMS):

| | frames with a ball per serve |
|---|---|
| head-centred ROI, wrong window | 0–0.5 |
| wrist-centred ROI, single shot | 5.0 |
| **wrist-centred ROI + SAHI** | **8.0** |

**But ball presence does not discriminate.** A returner also has a ball near
them — the one they are about to hit. Median frames-with-ball is 8.0 true against
6.0 false, and "≥3 frames with a ball" is 71% true against **83% false**.

**The arc does, but only through direction:**

| trajectory feature | AUC |
|---|---|
| total rise of the highest ball | 49% |
| peak height above the wrist | 45% |
| **fraction of time the ball is climbing** | **73–75%** |

**And it is redundant with the pose toss.** On the same 19 true / 12 false
sequences the ball arc scores 75%, the pose toss 83%, they correlate at **+0.49**,
and the best blend (0.3 arc + 0.7 pose) reaches **83% — no better than pose
alone**. Both watch the same event; the arm is simply easier to see than the ball.

So a pass costing tiled ball inference over ~18 native-resolution frames per
candidate, with a video seek each, buys nothing over a signal already computed
for free. `far_toss_ball.py` is complete and tested; enable it only if that
changes — a closer camera, a better small-object model, or a clip where far pose
degrades. The sample is small (19/12), so this is a reason not to spend the
compute, not proof the ball can never help.

### Rule 1 — toss evidence adjusts far confidence
### Rule 2 — a far serve among near serves is suspect

Both **adjust confidence, never delete**. A hard gate on toss below 0.40 removes
54% of false positives but 18% of true serves, and the brief is precision without
giving up recall. Rule 2 fires when ≥3 of a far candidate's 4 neighbouring starts
are near-side — measured to flag 18% of remaining far false positives at **zero**
true-serve cost, where the 2-of-4 setting starts costing 7%.

Rule 1's bound is asymmetric (+0.15 boost, −0.35 penalty): seeing a toss is
strong evidence *for* a serve, while not seeing one is weak evidence against,
since the arm may be occluded. Both run **before** the service-run DP, because
rule 2 reads the sides as detected and the DP would otherwise have overwritten
them.

**Effect on far starts leaving the orchestrator** (87 matched, 33 unmatched):

| | TP median | FP median | AUC |
|---|---|---|---|
| raw `p` | 0.97 | 0.94 | 64% |
| **adjusted_p** | **1.00** | **0.68** | **74%** |

A consumer thresholding `adjusted_p ≥ 0.70` keeps 80% of true far serves and 45%
of false ones — where raw `p` could barely tell them apart.

## Acquisition and inference cost

Everything above the two pose passes is numpy; the passes ARE the cost.

| pass | what | cost, clip 58 (56 min) |
|---|---|---|
| near | 540p whole-frame proxy, pose @15 Hz, **imgsz 640** | 784s → **484s** |
| far | native-res baseline band, pose @15 Hz, **imgsz 960** | 534s |
| everything else | walking model, near_end, tracks, 3 detectors, reel | seconds |

**~0.30× realtime** end to end, down from 0.39×.

**The near pass had headroom; the far pass has none.** That asymmetry is the
whole story of this pipeline, and it is the same fact that forced two ROIs in
the first place — near players are 72–275 px and far players 22–32 px, right at
the model's detection size.

- **near 960 → 640**: clip 35 near serve 100%/100% unchanged, pass 140s → 57s;
  clip 58 near serve 79.5%→81.8% recall, point end within noise, pass 784s →
  484s. Adopted.
- **far 960 → 768**: a micro-benchmark said this was free (35.2 vs 52.6 ms/frame,
  1.00 vs 1.02 persons/frame). End-to-end it cost **recall 100% → 86.7%**.
  Rejected. Persons-per-frame over a short sample cannot price this, because the
  frames that break are the ones where the player is smallest — which is
  disproportionately the serves. **Only an end-to-end recall number can price a
  far-side change.**
- **far 15 Hz → 10 Hz**: cannot be done alone. `tracks._stack` requires both
  passes on one timeline, since side membership is decided by the court rather
  than by which pass found a person. Attempted, `tracks` refuses — and the
  refusal produced a tempting false positive: with tracks failing, the far
  detector scored the *stale* 15 Hz tracks and reported identical accuracy for
  30% less time. Decoupling means resampling the far stream onto the near
  timeline: a substrate change, not a parameter.

### Known gap

11 of 13 clips still take their near detections from the legacy
`_end_walk_dets.npz` (540p, imgsz 960) rather than `perceive.near`; only clips
35 and 58 use the new pass. The corpus is therefore mixed, and a full
regeneration at imgsz 640 would shift the near-serve and point-end numbers by a
little in either direction.

## Agent 4 — the orchestrator

Takes the three event streams and produces the reel. It detects nothing; it
imposes the structure tennis has and the detectors cannot see.

**Over 236 labelled points on 13 clips (133 min of source):**

| | |
|---|---|
| points wholly inside a segment | **207 / 236 (88%)** |
| partially cut | 22 |
| **missing entirely** | **7 (3.0%)** |
| **live tennis retained** | **95.3%** |
| reel length | **61.6% of source** |
| segments | 135 (≈1.2 cuts/min) |
| points recovered from live play alone | 14 |

### The measurement that decides the architecture

The three detectors do not have comparable recall — near serve 90.7%, far serve
82.2%, **point end 49.6%**. So **the reel is built from starts, and ends only
trim.** A point whose end was never detected still becomes a segment; it runs for
the clip's own typical point length instead. Requiring a start/end pair would
silently drop half the points.

That asymmetry also sets the error budget: a missed end costs some dead time at
one segment's tail, while a missed start loses the whole point.

### What tennis knows that the detectors don't

- **Service runs.** One player serves a whole game, so the side sequence is
  `NNNNN FFFFF`, never `NNFNN`. A segmental DP (exact, not greedy — a greedy pass
  commits to an early wrong side and never recovers) relabels isolated flips. It
  only ever changes a *label*, never drops a start.
- **Deuce/ad alternation.** Within a game the server alternates court every
  point, always. In the video it is much weaker: re-measured over all 14
  clip-sides with at least 6 labelled serves, court-x flips sign **a mean of 63%
  of the time**, from 20% (clip 43 near) to 91% (clip 58 near). An earlier
  version of this file quoted the 91% as representative — that is the best case,
  not the corpus. Median-filtering the court track changes it not at all. At 63%
  it is barely above chance, so it only ever *flags* a suspected missing point.
- **A point is not live twice.** A serve struck while a point is already running
  is spurious — and agent 3's live score says so. Gating on the median live score
  in the 3 s before a detection drops **26% of far false positives for 4% of true
  far serves**. It is applied to the far stream only: on the near stream the same
  gate is AUC 55% and costs more than it buys, which is the FP taxonomy showing
  through (79% of far false positives are the returner mid-rally; near ones are
  the server's own repeated motions in dead time).
- **Rhythm.** Real start-to-start spacing is median 27 s; only 5% fall under 8 s.

### Recovering points nobody detected

15 of 236 points had no serve detection at all. Nothing in the serve streams can
recover those, but the live score is a different measurement and does not care
whether a serve was seen: any sustained run of live play that no segment covers
is kept, marked `recovered`, with **no side** — there is no detection to say who
served, and the reel should gain the tennis without inventing a fact about it.
This runs last, so detected points keep their serve-anchored boundaries.

**Missing points 15 → 7; live retained 88.8% → 93.1%** before roll tuning.

### Smoothness is an output, not a side effect

Two choices a pure accuracy metric would make differently:

- **When in doubt, keep footage.** An extra second of a player walking is barely
  noticeable; a cut landing mid-rally is jarring and loses the point.
- **A cut costs something.** Two segments separated by a couple of seconds of
  dead time are worse to watch than one continuous segment spanning them, so
  segments closer than `merge_gap_s` are joined and sub-4 s flashes are dropped
  or padded.

Roll turned out to be the dominant lever — far more than any end-estimation
tuning (moving the duration percentile from 85 to 97 buys 2 whole points; +1 s of
post-roll buys 17). The ends are close to right, they were just being cut a
second early:

| pre / post | whole points | live kept | reel |
|---|---|---|---|
| 2.5 / 2.0 | 179 | 93.1% | 54.9% |
| 2.5 / 3.0 | 196 | 94.6% | 57.3% |
| 3.0 / 3.5 | 202 | 94.9% | 59.5% |
| **3.5 / 4.0** | **207** | **95.3%** | **61.6%** |
| 3.5 / 5.0 | 212 | 95.8% | 63.7% |

That table is the exchange rate — lower both to tighten the reel.

### Caveats

- **Clip 38 reads 189% of span** because its labels stop at 206 s of a 420 s clip
  while the tennis (and the detectors) continue. That is a labelling gap, not a
  reel that doubled the match.
- Clip 58 still carries most of the residue: 16 of the 22 partial points and 4 of
  the 7 missing.
- `score_reel` measures the reel, not the detectors — points whole/partial/
  missing, live retained, dead per point, cuts per minute.

## Two corrections applied 2026-08-25

**1. Clip 58's labels carry the lead already.** Its author confirmed it, and it
measures that way: the median signed error from a labelled serve to the nearest
detection is between −0.93 s and +0.87 s on **every other clip**, and +0.80 s
(near) / +2.63 s (far) on clip 58. Scoring it against the corpus convention was
charging the detector for a labelling choice. `eval.LABEL_CONVENTION_S` now
brings each clip's labels onto the corpus convention at the corpus's own choke
point — the detectors get no per-clip parameter, because a detector must not
know which clip it is looking at. `--raw-labels` scores without it.

Clip 58 alone: near **65.9% → 79.5%** recall, far **21.6% → 51.4%**, both with a
bias of ≈0.00 s.

Worth stating plainly rather than smoothing over: the two modes needed different
corrections (+0.80 vs +2.63). A single built-in lead would shift both equally.
The near and far detectors anchor on different events — hands-together versus
trophy onset — so part of that gap is detector-side anchor bias that is only
visible on the clip with the largest offset. **These numbers are measured, not
derived from a stated convention.**

**2. The near trophy bands are 35% looser.** Some servers simply do not make a
textbook trophy: a compact service motion carries the racket lower and splits the
hands less, and the original bands scored those near zero however clean the rest
of the sequence was. Swept over all 107 near serves:

| loosen | recall | precision | F1 |
|---|---|---|---|
| 0.00 (original) | 86.9% | 78.2% | 82.3 |
| 0.20 | 88.8% | 77.9% | 83.0 |
| **0.35** | **90.7%** | **76.4%** | 82.9 |
| 0.50 | 89.7% | 74.4% | 81.4 |

0.35 is where recall peaks — 0.50 gives it back, so it is a maximum, not a slope
being ridden. F1 is flat from 0.20–0.35, so the choice between them is the
recall/precision preference and not a measurement: for a point START a miss loses
a whole point from the reel, while an extra start is something the composition
layer can arbitrate.

Two things the sweep settled that are worth not re-testing:

- **`TROPHY_MIN` is inert.** Swept from 0.35 down to 0.16 it changes *nothing* —
  the run threshold is not binding because the final probability threshold
  already dominates it. Lowering it looks like loosening while doing nothing.
- **Loosening the FAR trophy does not help.** The same sweep moves far recall
  only 82.2% → 83.7% at heavy loosening while costing precision. The far limit is
  resolution — the trophy is absent from the band entirely for 14 of clip 58's 37
  serves — not the threshold. Far bands are left alone.

## Clip 35 — the one clip none of this was built on

Clip 35 sat in `parse_ground_truth.EXCLUDED` for the whole of this work (7.0%
of it was labelled live), so **not one threshold, weight or window in anya2 was
chosen with any knowledge of it**. It was relabelled on 2026-08-24 — 20 rallies,
34.8% live, denser than clip 38 which was never excluded — and rejoined the
corpus. It is also the fastest source in the corpus at 119.88 fps, twice
anything else, so it exercises the stride logic at 8 rather than the usual 2-4.

| detector | labelled | recall | precision | vs pooled |
|---|---|---|---|---|
| near serve | 5 | **100%** | **100%** | above |
| far serve | 15 | **93.3%** | **93.3%** | well above |
| point end | 20 | **60.0%** | **60.0%** | above |

All three land at or above their corpus averages, and the far-serve precision is
the second-best on any clip. That is the strongest evidence available here that
the constructions are not fitted to the clips they were built on — with the
important caveat that 5 near serves is a small sample, and that clip 35 is a
7-minute snippet, which is the format the detectors do well on. The failure the
holdouts exposed is specific to FULL-LENGTH MATCHES (clip 58), and clip 35 does
not test that.

## Usage

```bash
python -m pipeline.anya2.camera   --self-test          # no video needed
python -m pipeline.anya2.camera   /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.camera   /Volumes/Anya/Data/21/snippet.mp4 --report
python -m pipeline.anya2.perceive /Volumes/Anya/Data/21/snippet.mp4 --roi near
python -m pipeline.anya2.perceive /Volumes/Anya/Data/21/snippet.mp4 --roi far
python -m pipeline.anya2.tracks   /Volumes/Anya/Data/21/snippet.mp4 \
    --dets   /Volumes/Anya/Data/21/snippet_anya2_near_dets.npz \
    --far-dets /Volumes/Anya/Data/21/snippet_anya2_far_dets.npz
python -m pipeline.anya2.near_serve /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.far_serve  /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.eval --mode near_serve --arm anya2:_anya2_near_serve.json
python -m pipeline.anya2.far_serve  /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.point_end  /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.eval --mode far_serve  --arm anya2:_anya2_far_serve.json
python -m pipeline.anya2.eval --mode point_end  --arm anya2:_anya2_point_end.json
python -m pipeline.anya2.orchestrator /Volumes/Anya/Data/21/snippet.mp4
```

## Known gaps

- **Precision on a full match is gated by the composition layer**, not by this
  detector — see clip 58 above. Until a point-end detector exists there is no
  way to suppress mid-rally overheads, and 55.8% is the floor.
- **The lead is a compromise across clips** and should become a per-clip
  estimate if the tolerance ever tightens below ±2 s.
- **The far ROI is built but unused.** `perceive.far` derives the band from the
  homography and grows it upward by 2.8 m of court scale (a ground-strip band is
  ~55 px on a 4K clip and clips the server's raised arm). Nothing reads it yet.
- **The end signals still project through the fixed homography.** `walking.predict`
  and `near_end.frame_signals` each call `load_homography` themselves, so after a
  jostle their `court_x`/`court_y` carry a constant offset and their `speed`/`acc`
  carry a one-sample spike at the jostle instant. anya2's own gates are corrected;
  these two feed agent 3 and are not. Threading `Geometry` through them means
  changing signatures in `walking/`, which is why it is listed here rather than
  done.
- **Slot flapping on clip 38** — 31 identity switches over 420 s on a singles
  clip. The doubles clips are stable (2 and 1), which is what the "who served"
  answer depends on, so this is not currently costing anything measurable.
