# Rally confidence — refocusing agent 3, and moving the reasoning into agent 4

> Status: **design, not built.** Nothing in this document is measured on the
> construction it proposes. Every number quoted is from the code and results
> that exist today, and is cited so it can be re-checked.

## The change in one paragraph

Agent 3 stops being a point-end **detector** and becomes a **rally confidence
score**: one continuous value per frame, for the entire video, answering "is a
point in play right now". It emits no events. The orchestrator takes three
inputs — near-serve starts, far-serve starts, and this curve — and is where
point starts and point ends are actually decided. Point ends are decided
**conservatively**: overrunning into dead time is cheap, truncating live tennis
is not.

## Why: the detector was throwing the signal away

The orchestrator's own header records the measurement that shaped it:

    near serve 90.7%      far serve 82.2%      point end 49.6%

and concludes that the reel must be built from starts, with ends only trimming.
That conclusion is correct given a 49.6% end stream. But the end stream is not
weak because ends are hard to see. It is weak because a rich curve was being
collapsed into discrete events at its falling edge, and half of them did not
survive the collapse.

The curve itself is not weak. `point_end.live_score()` separates live from dead
at **AUC 86.7%** (`pipeline/anya2/README.md`, "Point end — results", table 4):

| construction | AUC |
|---|---|
| near activity, 8 s | 79.5% |
| far activity, 8 s | 75.1% |
| max(near, far), 8 s | 78.7% |
| max(near, far) − union, 8 s | 82.6% |
| **max(near, far) × (1 − union)** | **86.7%** |

An 86.7% frame-level signal reduced to a 49.6%-recall event list is the whole
problem. Do not collapse it.

## This is a promotion, not a new build

`live_score()` already exists and the orchestrator already consumes it
directly — `pipeline/anya2/orchestrator.py:697` calls `PE.end_signal()` and
`PE.live_score()` and uses the result for in-rally serve suppression
(`suppress_in_rally`), the far-side live gate (`live_gate_far`), and
`recover_missed`.

So the work is:

1. Move the curve out of `point_end.py`'s private scope into its own module
   with a published contract and a cached artifact.
2. Delete `detect_ends()` and the `_anya2_point_end.json` event stream.
3. Rewrite `pair_ends()` to reason over the curve instead of matching events.

Scope discipline: **the near and far serve detectors are not touched.** They are
validated and they stay as they are.

## Rally confidence: the contract

Artifact `<stem>_anya2_rally.npz`, at pose rate:

| channel | meaning |
|---|---|
| `raw` | per-frame rally evidence, **unsmoothed** |
| `conf` | windowed confidence in [0, 1] — the headline signal |
| `valid` | coverage: is this frame measurable at all |
| `scale` | the per-clip normaliser actually applied |
| components | `near_act`, `far_act`, `union`, and the union's parts, for diagnosis |

### Four defects to fix in the promotion

These only matter once the curve is a published signal rather than a private
intermediate. All four are live in the current code.

**1. The window is no longer a compromise, so stop compromising.**
`LIVE_SMOOTH_S = 4.0` today, and its comment says why: 8 s separates live from
dead better but *blurs the edge*, and the edge was what the falling-edge
detector had to time — best F1 40.3% at 8 s against 42.8% at 4 s. **Once the
orchestrator does the edge-finding, the score no longer has to be sharp; it
should maximise separation.** That is why `raw` is stored alongside `conf`: the
orchestrator can re-window to whatever it needs, and the stored `conf` stops
being a constraint. **5 s is the default headline window** — between the two
values already explored, and to be swept against the frame-level metric below,
not assumed.

**2. The normaliser is per-clip and will bite.** `LIVE_SCALE_PCT = 90` divides
by the clip's own 90th percentile of the smoothed score. The reason is sound —
activity is in body heights per second, so its absolute level depends on how
large the players are in frame, and a fixed threshold would mean a different
thing on every camera. But the consequence is that **the score is not comparable
across clips, and is undefined on a clip with little live play**: a clip that is
90% dead time normalises its dead time to 1.0. Acceptable for a private
intermediate; not acceptable for a published signal that a conservative end
policy will threshold. Either find a scale-free formulation or record `scale`
explicitly and make every downstream threshold state which side of it it lives
on.

**3. "Not live" and "cannot tell" are different, and today they are the same
number.** With both players untracked, activity is low, and low activity reads
as *dead*. That is the failure already recorded in memory as "absent is not
neutral": an unmeasurable window scored as a vote against is not an abstention.
Near-slot coverage varies **42%–88% by clip** (README, "Known gaps"), so this is
not hypothetical. The `valid` channel exists so the orchestrator can **refuse to
place an end on unmeasured footage** rather than inferring one from silence.

**4. The score is near-biased.** Far activity is in the product, but the
non-rally union is entirely near-side — the walking classifier plus `near_end`'s
four signals, all fed from the near track through a shim. Worth measuring
whether a far-side union term exists at all. Related open item from the README:
on clip 23 the `settle` and `stance_drop` signals produce no usable onsets.

### What rally confidence is built from

Player kinematics, as today: `max(near activity, far activity) × (1 − non-rally
union)`. A product and not a sum, because the union's job is to **veto**
activity rather than be traded against it — a player walking to the ball is
active and emphatically not playing, and only a multiplicative term can say so.
This is the same arbitration shape the near serve detector's swing term uses.

The far-activity term exists only because anya2 tracks the far player. No
previous point-end work here could, and memory records "far-serve rallies read
as dead" as the biggest error source of the earlier dead/live GRU.

## Orchestrator: where the reasoning moves

`pair_ends()` is today a nearest-event matcher over a sparse event list, with
fallbacks to an 85th-percentile duration estimate when no end was detected. It
is replaced by inference over the curve.

**A start is an anchor, not a hypothesis.** A serve detection accompanied by a
*rise* in rally confidence is a confirmed point. A serve detection with no rise
is a candidate false positive. This is the population the corpus already
identifies: **23 of 24 near and 79% of far false positives fall inside a live
point**, both detectors declare `windows="between_points"`, and neither can
enforce it alone. `suppress_in_rally` does a version of this today as a gate;
here the curve becomes primary evidence rather than a filter.

**An end is where confidence falls and stays down.** Conservatism is expressed
as asymmetric hysteresis plus a dwell requirement — *prove the point is over* —
not as post-roll padding after the fact. The supporting measurement already
exists: both players quiet for 1.5 s covers **0.0–1.2% of live play** and 7–21%
of dead time, which is close to proof the point has ended. It was rejected for
timing because the first such window arrives a **median +78 s** after the
labelled end. Under a conservative policy that near-proof becomes usable as a
*bound* even though it is useless as a *timestamp*.

**The next start bounds the previous end.** Already present as
`next_start_guard_s`, but as a guard rail rather than a constraint inside the
inference. It should be the latter.

**Service-run structure stays.** It is independent evidence and it works.
Structure relabels a point, it never deletes one — that invariant holds today
without exception and must survive this rewrite.

**Conservative, stated precisely:** given a choice between an end that may
truncate live tennis and one that may include dead time, take the dead time.
The current pose-only detector achieves **zero truncations on every clip** at
49.6% recall. **Zero truncations is the property to preserve**; recall is the
thing being bought.

## Should the ball feed rally confidence?

**No, not in v1.** The repo has run this experiment three times and it lost
each time.

1. **Surface.** `point_end.py`'s opening argument: the ball is low-contrast
   against clay for much of its flight. A point-end policy whose primary
   evidence disappears on one surface is not a policy.
2. **It was built and removed.** The orchestrator's rule-1 note records a ball
   toss detector re-aimed at the tossing wrist with native-resolution SAHI
   tiling. It detected the ball well (8 frames per serve against 0–0.5 for a
   head-centred ROI) and separated true from false serves at **AUC 75%** — but
   it correlated **+0.49** with the pose toss and the best blend of the two was
   **no better than pose alone**. It cost tiled inference over ~18 native frames
   per candidate for nothing.
3. **Detection consistency is not stable across clips.** Memory from the Phase 0
   spike records live-ball detection consistency of **7% / 36% / 92%** across
   clips. At that spread, a ball term inside a shared confidence score is
   substantially a *clip-quality* term wearing a rally-confidence costume: it
   would look strong on the 92% clip and quietly invert on the 7% one.

**The honest counter-argument.** The old shipped `end_policy="trace"` reaches
62%/76% with **zero truncations**, and zero truncations is exactly the
conservative property this design is asking for. It costs ~3x clip length, and
it is hard-court evidence.

**What to do instead.** Design the rally-confidence interface so a ball channel
is *addable* as one more evidence stream, and queue one narrow experiment rather
than a trace:

- **Net-crossing count within the window** — a coarse directional-motion
  statistic over filtered detections, not a Kalman trace.
- **Strictly additive, low weight.** Never multiplicative: a ball term that can
  veto would take the whole score down with it wherever the ball is invisible.
- **Gated on a measured per-clip ball-detection consistency estimate**, so the
  channel can **abstain** rather than mislead.
- **Precondition:** measure that consistency across the whole corpus first. If
  the 7%–92% spread reproduces, the channel ships **off** by default and the
  experiment is closed.

## Evaluation

The current harness (`eval.py`, ±2.0 s, greedy one-to-one, restricted to the
labelled span) matches events and **cannot score a curve**. Two metrics, not one:

**Rally confidence — frame level.** Live/dead AUC and best-F1 against
`parse_ground_truth.live_mask`, which already produces a per-frame boolean
timeline from the existing labels. Reported per clip and pooled, and **broken
out by surface**. The 86.7% AUC above is the number to beat, and any change that
does not move it is not an improvement however good it looks on a reel.

**Orchestrator — event level.** Keep the existing ±2.0 s event metric for point
starts and ends, and add **truncation count as the headline number**, given the
conservative brief. A design that raises end recall while introducing
truncations has lost, not won.

Both arms must run off the **same cached pose passes**, so that the construction
is the only variable — the same discipline the camera-tracking A/B used.

**Watch for artifact staleness.** anya2 detection JSONs go stale against
regenerated tracks silently. Check mtimes before trusting any cached result;
this has already produced misleading numbers once (clip 21's on-disk
`_anya2_near_serve.json` predated its `tracks.npz`).

## Open questions, to settle before tuning

**1. Surface coverage is not recorded anywhere machine-readable.** *Deferred at
the user's direction: surface differences are explicitly not a blocker for this
work, and the manifest is no longer step 1 of the sequencing below.* The
observation is kept because it is the axis the ball decision would turn on if
the ball is ever revisited, and because the table beneath it is a live caution
against reading surface into results. It exists
only in prose: `walking/README.md:6` says Data/21 is indoor hard and Data/22 is
outdoor clay; `ios_tracker/labeling/README.md:8` says the 29-match library spans
"clay and hard, indoor and outdoor" without saying which is which. **Write the
surface per clip into a manifest before any of this is tuned.** It is the axis
the ball decision turns on and the axis the walking classifier — the one
component in anya2 ever shown to transfer across surfaces, leave-one-clip-out
frame F1 0.82–0.84 — was validated on.

A caution against assuming surface explains failures, from this session's
verification run:

| clip | surface | near R/P | far R/P |
|---|---|---|---|
| 21 | indoor hard | 100% / 100% (11) | 0% / 0% (1 GT, 3 fires) |
| 22 | outdoor clay | 100% / 91.7% (11) | 100% / 75.0% (3) |
| 23 | unrecorded | — (0 GT, 0 fires) | 93.3% / 87.5% (15) |

Far serve scored **best on the clay clip and worst on the indoor hard one**.
The axis that explains clip 21 is near/far dominance — 11 near serves against 1
far — not surface.

**2. Does a far-side non-rally union exist?** Defect 4 above. Unknown until
measured.

**3. What replaces the per-clip normaliser?** Defect 2 above. A scale-free
formulation is preferable to a recorded scale, but none is proposed here.

## Sequencing

1. **Frame-level eval harness** against `live_mask`. Before any change, so the
   86.7% baseline is reproduced on the current code first.
2. **Promote the curve** to `rally.py` with the four defects fixed. Re-measure.
   No orchestrator change yet; the frame metric alone says whether this is good.
3. **Rewrite `pair_ends`** over the curve. Event metric plus truncation count.
4. **Delete** `detect_ends()`, `_anya2_point_end.json`, and the `use_end` path
   that reads it — only after 3 beats the current numbers.
5. **Ball, deferred.** Closed for now. Reopening it requires the surface
   manifest and a consistency measurement that does not reproduce the 7%–92%
   spread.

The surface manifest was step 1 in the first draft of this plan and was dropped
at the user's direction. Results are still reported per clip, so a surface
effect stays visible after the fact if one appears; nothing here is blocked
waiting to name it in advance.

Steps 2 and 3 are separately measurable, and should stay separate. The camera
work is the precedent: splitting "corrected geometry" from "corrected band"
showed the two fixed different things and both were load-bearing. A combined
change that improves the reel tells you nothing about which half did it.
