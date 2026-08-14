# Point detection — design

Goal: given a fixed-camera tennis video shot from behind the near baseline,
emit `[start, end]` for every point and cut a highlights video containing only
those segments.

Status: near-side starts and point ends have working components. Far-side
starts do not exist in this stack (they exist in the older `point_segmenter.py`
stack, measured — see §5). This document is the plan to close the gap and the
places where the data argues against the plan as first stated.

---

## 1. The metric

Not per-serve accuracy. **Kept-video coverage of ground-truth rallies, plus
stray dead time kept.** A point start that lands 2 s early is invisible in the
product; a missed point start loses a whole rally. `pipeline/trace_eval.py
--mode segments` already computes this (rallies-started, full-cover, mean
coverage, stray-kept) and merges segments at 1.0 s like the ffmpeg export.
Reuse it — do not invent a second scorer.

Secondary metric, only for the side prior in §7: per-point serve-side accuracy.

---

## 2. The four quadrants

| | point start | point end |
|---|---|---|
| **near serve** | `anya_near_serve.py` — recall 22/22, precision unresolved (§4) | ball trace + walking (§6) |
| **far serve** | does not exist here; port + extend (§5) | ball trace + walking (§6) |

Point *end* is one problem, not two — the end detector never needs to know who
served. Only the start detector is side-dependent. That is the main structural
simplification available.

---

## 3. What exists today

- **`pipeline/anya_telemetry.py`** — slim perception pass. Per frame: near box
  + world feet, far box + smoothed world feet, and **raw unfiltered whole-court
  ball detections**. Ball `imgsz` 960. Writes `<stem>_anya_telemetry.jsonl`.
  Everything below is a pure post-processor over this file — re-runs in seconds.
- **`pipeline/anya_near_serve.py`** — graded near-serve probability from dwell ×
  toss × ratio-jerk.
- **`walking/`** — learned near-player walking classifier. Cross-clip frame F1
  0.82–0.84 (indoor hard ↔ outdoor clay), beating the old hand-tuned
  cadence rule at 0.58. Emits intervals with `mean_prob` **and
  `detection_coverage`**.
- **`pipeline/point_segmenter.py`** — the previous-generation full cutter,
  including a **measured far-serve detector** (§5). Model-free stage 2.

---

## 4. Near-side start — the open blocker

Recall is solved: 22/22 across folders 21 and 22, constants untouched between
runs. Precision is not, and the cause is known: **the three cues are behaving as
one cue.** Jerk saturates at 1.00 on 12–15 % of *all* frames and toss on ~6 %,
so `P = J·(0.35+0.65T)·(0.40+0.60D)` collapses to a dwell-only score. True
positives quantize to exactly 0.879 / 0.985 — those are dwell 0.80 / 0.98. On
folder 22 a non-serve scores 0.985, tied with the best true positives. No
threshold separates because there is only one real dimension.

### Decision: fix jerk's dynamic range; leave toss alone

**Jerk is the cue that must be de-saturated.** Raise `jerk_hi` and `amp_hi`
until the score genuinely sweeps 0 → 1 across ordinary play rather than pinning
at 1.00 on 12–15 % of all frames. The goal is a *range*, not a higher bar: once
jerk varies, it becomes a real second dimension and a serve threshold can be
chosen on `P` meaningfully.

Method: histogram `jerk` and `amp` over all frames of 21+22, set `amp_hi` and
`jerk_hi` near the upper percentiles of ordinary rally play (not at the serve
values), and confirm the saturation rate drops to a few percent before
re-scoring. Seconds per iteration — both telemetry files are on disk, no
perception pass.

Do **not** lower `jerk_hi`. An earlier read of folder 21 alone suggested
12.0 → 7; the saturation measurement shows that is backwards.

**Toss stays as it is.** A toss either happens or it does not — it is a
genuinely near-binary event, and its ~6 % all-frame saturation is the cue
behaving correctly rather than a calibration fault. Grading it harder would
manufacture variance that is not in the world. After the jerk fix, `P` is
expected to read as jerk × dwell modulated by a near-binary toss, and that is
the intended shape.

Fit jointly on 21+22; hold out the rest (§8).

---

## 5. Far-side start

### The proposed signal already exists and has been measured

The plan as stated — sustained ball-quiet, then far player at the baseline,
then a trace shortly after coming toward the near side — is, to within the
baseline-dwell term, exactly the far branch of `detect_serve_events` in
`point_segmenter.py`:

> fresh trace onset in the far region (y ≤ far-baseline cutoff, calibrated from
> observed far-player feet), serve motion (downward + horizontal,
> perspective-scaled) within the trace's first 2.5 s, ≥ 4 s of no ball activity
> before onset (micro-blips < 0.25 s don't reset the quiet clock), far player
> present within 1 s.

Measured: **folder 23 (all-far, hand-labelled) 14/15 rallies recovered**; the
one miss never grew a ball trace at all — a perception limit, not a gate.
Folder 68: 16/22 far starts. Trace onsets lag labelled serve-motion start by
+1 to +4.2 s, covered by `far_pre_roll_s = 4.5`.

**Recommendation: port that branch rather than re-deriving it.** It carries
several findings that cost real effort to establish and will otherwise be paid
for twice:

- **Do not gate on far *world distance*.** The homography-projected far
  distance flip-flops between ±20 ft regimes faster than any self-calibration
  band tracks. That gate rejected 6/10 real onsets; the whole self-cal
  machinery was deleted.
- **Do not tighten the far-origin cutoff.** Sweeping the pad over
  {45, 90, 150} px left coverage/started/full completely flat and only raised
  stray dead time. The gate never gated a real point start.
- **Check the court cache first when a whole side goes dark.** Folder 68's
  miscalibrated corners projected the far player's feet to the near half and
  discarded them: far box coverage ~0 %, every trace onset logged
  `no_far_player`. Fixing the corners took it 12/45 → 30/45. The failure
  signature is unmistakable once you know it.

### What is added — decided

The existing far detector is kept as-is and **two confidence terms are layered
on top**. Neither is a gate.

**1. Far-baseline dwell — confidence only, never a gate.** The existing
detector checks far-player presence within 1 s of onset. A longer settled dwell
raises confidence that an onset is a serve rather than a mid-rally shot,
because a player who has just hit a mid-rally ball is moving and a server is
not. That is the intended fix for the known failure mode: **far events
double-firing inside long far rallies**, where a shot more than
`min_serve_separation` (8 s) after the serve passes the quiet gate as a fresh
onset (folder 68: 419 / 683 / 727 / 774 / 853 s).

But **quick servers exist** — some players take the ball and go with barely a
pause. A dwell *requirement* would delete exactly those serves, and a missed
start loses a whole rally under the §1 metric while a low-confidence start
costs nothing. So dwell contributes a graded bonus with a floor: short dwell
means "no boost," not "not a serve."

**2. Lateral stability instead of baseline proximity.** Do not tighten the
far player's *distance to the baseline* — that is the depth axis, and it is
the one the homography estimates worst. At far-court distance perspective
compresses many feet of world depth into a few pixels of feet position, so
small pixel error becomes large world error; this is the same effect that made
the far world-distance band flip-flop between ±20 ft regimes and got it deleted
(above). Lateral position does not suffer that: world *x* is recovered across
the court's width, where the pixel-to-world scale is far better conditioned.

So the cue is **minimal lateral movement in the run-up to the onset** — the
server's *x* should be roughly constant, which is exactly what distinguishes a
player about to serve from one recovering across the court. Score it on the
spread of world *x* over a short pre-onset window (the same RMS-scatter shape
the near-side dwell already uses in `anya_near_serve.py`, but on *x* alone),
and read *y* only loosely, as the existing far-region cutoff already does.

**3. Grade it, don't gate it.** Emit a continuous `P_far` from (quiet-duration
× lateral-stability × dwell-bonus × trace-onset geometry) and let the consumer
and the side prior (§7) move the threshold. Carry §4's lesson across: check
each cue's *saturation rate over all frames* before trusting the product, or
this ships as a multi-cue score that is secretly one cue again.

**4. Build the cheap double-fire fix first.** Suppress any far onset whose
timestamp falls inside an already-emitted segment's `[serve, end]`. It costs
almost nothing and targets the same 5 folder-68 false events. Measure it alone,
then measure what dwell and lateral stability add on top — otherwise there is
no way to attribute the gain.

### The one thing that cannot be fixed by better logic

Ball-detection consistency is the dominant blocker and it is a perception
problem, not a gating problem. Live-trace fraction measured across three
detection regimes: **92 % / 36 % / 7 %** (folders 21 ungated, 23 gated native,
68 gated native). On folder 68 only ~7 % of rally records have any detection at
all, and enrichment work found a ceiling at ~36 % of rally records having *any*
detection. A far-serve detector whose primary evidence is a ball trace inherits
that ceiling. Expect far-side recall to track the ball detector, folder by
folder, more strongly than it tracks anything in this document.

---

## 6. Point end — one detector for both sides

### Decided ordering: walking is primary, ball is confirmatory

**Primary — near player walking.** This is the strongest available signal and
the only one measured end-to-end (cross-clip frame F1 0.82–0.84, and it
transfers across surface, camera and players). Ball detection by contrast swings
92 % / 36 % / 7 % live across folders, so anchoring the primary decision to the
ball would make point-end quality a function of which folder you are in.

**Confirmatory — ball quiet, or a lone detection.** Raises confidence in a
walking-derived end. The test is *at most one ball detection inside a window*,
not zero: during dead time a single stray detection is common (a player
bouncing the ball, picking one up, a false positive on a stationary object),
whereas a live rally produces a sustained chain. A "≤1 detection in the window"
predicate is robust in a way that "no detections" is not.

**Last resort — prolonged near-player stationarity.** If neither walking nor
ball evidence resolves, a long stationary stretch marks dead time, and the
point end is placed at the **start** of that stretch. This is the correct
reading of the ambiguity: a stationary period may *begin* as dead time and then
transition into the ready position for the next point, so its leading edge is
the safe boundary. It is a floor on quality, not a preferred path — it only
fires when the two better signals are silent.

**Hard ceiling — the next point's start, minus pre-roll.** The search for a
point end is bounded between consecutive serve events. That is the offline
advantage and it stays.

```
end_walk  = onset of first walking interval after point start
            (gate on detection_coverage — a low-coverage interval is a
            guess spanning a hole in the input, not an observation)
confirm   = (ball detections in [end_walk - w, end_walk + w]) <= 1
end_still = start of a prolonged stationary stretch          (last resort)
ceiling   = next point's start - pre_roll                    (hard bound)

end = min(ceiling, end_walk if walking fired else end_still)
```

### One implementation detail this ordering forces

The walking classifier's context windows run to 8 s, centred — which is what
makes long walks reliable, and why nearly every miss is a walk under 3 s. Its
*onset timing* is therefore coarser than the ~0.5 s a cut boundary wants.

So walking should decide **that** the point ended, and the ball evidence should
refine **where**: having detected a walk onset, search backwards from it for the
last real ball detection (anchored to the last detection, `t - tsd`, **not** the
coasted prediction — coasted ends run ~2 s long) and place the boundary there
plus a pad. That keeps walking as the decision-maker while recovering the
precision it cannot supply on its own. If the backward search finds nothing, the
walk onset stands as the boundary.

### Consequence of counting a fault as its own point (§10, Q2)

A fault is a ~2 s point: serve, ball into the net or long, server stays put.
**Walking will not fire on it** — nobody walks after a fault. So for faults the
confirmatory signal becomes the only signal, and the end must come from the
short trace dying plus the ≤1-detection window. This is a real gap in a
walking-primary design and needs explicit handling: after a serve event, if no
walking onset appears within a few seconds and ball activity has already gone
quiet, close the point on the ball evidence alone.

### Known residual

With the previous generation's fusion, folder 23 left 4 tails 0.5–2 s short of
the GT end (trace dies, near player stops moving) at `end_pad_s = 2.0`. Tails
being slightly short is the benign direction for a highlights product, and
walking-primary should improve exactly these cases — this is the first real
test of whether the learned classifier beats the old direction-reversal cue.

---

## 7. The serve-side prior — two games per serving side

### The structure, derived

Serve side is **not** per-game alternation. It is **two games per side**, and it
follows from two rules acting together:

- the server alternates every game;
- the players change ends after every **odd** game (1, 3, 5, …).

Walk it through with A starting at the near end:

| game | server | server's end | camera sees | ends change after? |
|---|---|---|---|---|
| 1 | A | near | **NEAR** | yes → A far, B near |
| 2 | B | near | **NEAR** | no |
| 3 | A | far | **FAR** | yes → A near, B far |
| 4 | B | far | **FAR** | no |
| 5 | A | near | **NEAR** | yes |

Pattern: `NN FF NN FF …` — each player serves once from each end before the
camera-side flips. So a run is **two games**, minimum 8 points (two 4-point
games), typically 10–14 with deuces.

### The ground truth confirms it exactly

```
58  NNNNNNNNNNNFFFFFFFFFFNNNNNNNNNNNNNFFFFFFFFFFFFFFNNNNNNNNNNNNFFFFFFFFFFFFFNNNNNNNN
    runs: 11, 10, 13, 14, 12, 13, 8
36  FFFFFFFFFNNNNNNNN                     runs:  9,  8
25  NNNNNFFFFFFFFFF                       runs:  5, 10
```

Every full run is 8–14, and the **shortest is exactly 8** — the theoretical
minimum. Runs of 5 and 9 appear only at clip boundaries, where the clip starts
or ends mid-block. That is a clean validation of the two-game model, and it
means the prior can be built from the *rule* rather than fitted to data, with
the corpus used to confirm rather than to estimate.

Encoded prior: a side switch is heavily penalised before 8 points into a run,
cheap from ~8–10 onward, and increasingly expected past ~14. Runs that are
truncated by the start or end of a clip must be exempt from the minimum.

### Tiebreaks are a different regime

In a tiebreak the server alternates every 2 points after the first, and ends
change every 6 points. Working it the same way gives runs of **1–2**, not 8–14:

```
pt   1   2 3   4 5   6 | 7   8 9  ...      (| = end change after point 6)
side N   F F   N N   F | N   F F
```

So the run-length prior is bimodal, and a hard global minimum-run-of-8 would
mis-label an entire tiebreak. Two options, in order of preference:

1. Detect the regime — a confident short-run stretch after a long
   sequence of 8–14 runs is a tiebreak — and switch the transition penalty.
2. Simply floor the penalty so a strong-enough emission can always override the
   prior, accepting degraded smoothing during tiebreaks.

Option 2 is the safe default for a first build. Tiebreaks are a small fraction
of points and the §1 metric barely moves; the cost of getting option 1 wrong is
larger than the benefit.

### The prior is worth more than a smoother

Near-serve detection is strong (§4 — recall solved) and far-serve detection is
weak and ball-limited (§5). The prior converts one into the other: **a
confidently-near run implies the next run is far, which licenses lowering the
far threshold inside that block.** That is a recall gain on precisely the side
that needs it, and it is the main reason to build the prior at all — not
label-tidying.

Implementation: HMM over point-level side, transition penalty from the
two-game run-length model above, emissions = `P_near` and `P_far` at each
detected start. `hmm_filter_events` already exists in `point_segmenter.py`.
Carry over its hard-won rule: **a trace-confirmed event is shielded from
side-based dropping.** A confirmed trace proves a point started; it does not
prove which side served — from a low camera behind the near baseline, a near
serve's post-bounce descent into the far court is kinematically identical to a
far serve, and no pixel-speed cap separates them (measured). Dropping confirmed
events on side disagreement threw away real boundaries.

### Where the prior does nothing

Folders 21 (11 near / 1 far), 23 (15 far), 37, 38, 40, 43, 63 are effectively
single-side clips. The prior degenerates to "the side is constant" — still a
useful smoother, but the alternation term never fires. **The prior can only be
validated on 58, 36, 25, 24, 26, 22, 50.** Do not report a prior that was only
exercised on single-side clips.

---

## 8. Validation corpus — larger than previously used

Previous work validated on folders 21, 23 and 68. The drive actually holds
**15 folders with `ground_truth.json` carrying per-rally `serve: near|far`**,
plus folder 68's `tags.json`:

| folder | rallies | near | far | | folder | rallies | near | far |
|---|---|---|---|---|---|---|---|---|
| 21 | 12 | 11 | 1 | | 37 | 3 | 0 | 3 |
| 22 | 15 | 11 | 4 | | 38 | 8 | 8 | 0 |
| 23 | 15 | 0 | 15 | | 40 | 13 | 0 | 13 |
| 24 | 14 | 2 | 12 | | 43 | 6 | 6 | 0 |
| 25 | 15 | 5 | 10 | | 50 | 7 | 5 | 2 |
| 26 | 13 | 2 | 11 | | 58 | 81 | 44 | 37 |
| 35 | 4 | 1 | 3 | | 63 | 9 | 0 | 9 |
| 36 | 17 | 8 | 9 | | **total** | **232** | **103** | **129** |

**129 labelled far-serve rallies.** The far detector has never been measured
against most of them. Format is start/end **frames** plus a side label; folder
58 is a full `match.mp4` (~198 k frames) and is the only clip with enough
structure to exercise the side prior.

Proposed split — fix it before tuning anything, and write it down:

- **fit**: 21, 22 (already used for near-serve constants and for the walking
  classifier — they are burned as test data either way)
- **dev**: 23, 24, 25, 26, 36
- **held out, untouched until the end**: 58, 68, and the small clips
  (35, 37, 38, 40, 43, 50, 63)

Folder 58 is the most valuable single asset in the corpus and should not be
looked at during tuning.

Caveats before trusting the corpus: several folders have no cached court
corners, and a wrong court cache silently kills an entire side (§5). Verify
`<stem>_court_cache.json` per folder first. Rally counts are also small enough
per folder that per-folder precision estimates are noise — the near-serve
threshold that looked clean on folder 21's n=11 did not survive folder 22.

---

## 9. Build order

1. **Give jerk a real dynamic range** and re-fit near-serve on 21+22; leave
   toss as-is. Every downstream threshold depends on this. (§4)
2. **Court-cache audit** across the 15 GT folders. Cheap, and §5's failure
   signature says an un-audited folder produces meaningless far numbers.
3. **Port the far-serve branch** from `point_segmenter.py` onto the
   `anya_telemetry.jsonl` boundary; measure on dev folders. Establishes the
   baseline the new work has to beat. (§5)
4. **In-segment suppression** of far onsets. Cheap double-fire fix, measured
   alone so the next two steps can be attributed. (§5.4)
5. **Lateral-stability and dwell-bonus terms**, both graded, both
   saturation-checked, neither a gate. Measure the delta over step 4. (§5.1–2)
6. **Point end**: walking-primary, ball-confirmatory (≤1 detection in window),
   prolonged-stationarity as last resort, next-start as ceiling, ball-anchored
   backward refinement of the boundary. Includes explicit fault handling.
   Replaces the direction-reversal cues. (§6)
7. **Side-prior HMM** with the two-game run-length model and a floored
   transition penalty. Use it to lower the far threshold inside runs implied
   far. Validate only on mixed-side clips. (§7)
8. Report on held-out folders once, at the end.

---

## 10. Decisions taken, and what remains open

### Settled

**A fault is its own point.** The detector fires on two serves, so it emits two
points — the representation matches what the system can actually observe. The
serve refractory window therefore stays short (~3 s), not ~25 s. Consequence for
the end detector in §6: a fault produces a ~2 s point with no walking, so it
must be closable on ball evidence alone.

**Folder 58 is held out**, untouched until the final report. (§8)

**Far-side dwell and lateral stability are confidence terms, not gates** —
quick servers must survive. (§5)

**Walking is the primary point-end signal**, ball evidence confirmatory,
stationarity the last resort. (§6)

**Jerk gets a wider dynamic range; toss is left alone.** (§4)

### Still open

**Q1.** End changes produce 60–90 s of walking, off-screen and stationary all
at once. Detect and exclude explicitly, or is the next-start ceiling enough?
Folder 23's longest dead-time gap is 69 s and folder 36's is 51 s, so the case
is in the corpus. Leaning: the ceiling is enough, but confirm on dev folders
before assuming it.

**Q2.** Does the near-side *physical player* swapping at an end change break
tracking? It should not — "near" is a camera role, not a person — but
`select_near.py` scores continuity in court metres and rejects candidates
implying more than 9 m/s. Worth confirming it re-acquires cleanly rather than
silently holding a stale track across the change. This matters more now that
walking is the primary end signal.

**Q3.** How wide is the confirmatory window for the "≤1 ball detection" test in
§6? Too narrow and a rally's own gaps read as dead; too wide and it never
fires. Sweep on dev folders against the §1 metric.

**Q4.** Are lets treated like faults (their own point)? The fault decision
implies yes by symmetry, but a let has no ball flight to speak of and may
produce no detectable serve event at all.
