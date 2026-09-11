# Rally confidence — refocusing agent 3, and moving the reasoning into agent 4

> Status: **step 1 done, the rest is design.** "Baseline, measured" below is
> measured, by `rally_eval.py`, on the current construction. Everything from
> "Rally confidence: the contract" onward is design and is not measured on the
> construction it proposes.

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

## Baseline, measured

`rally_eval.py`, all 13 trusted clips, scoring restricted to each clip's
labelled span. **This is the number to beat.**

| smoothing | mean per-clip AUC | pooled AUC | pooled best-F1 |
|---|---|---|---|
| 4 s (the module's own) | 87.2% | 82.9% | 69.7% |
| **5 s** | **88.2%** | **83.6%** | **70.8%** |
| 8 s | **88.5%** | **84.7%** | **71.6%** |

The 4 s row reproduces the documented 86.7% (measured at 86.6% on the 11 clips
whose artifacts existed before clips 21 and 23 had their `walk`/`endsig` passes
regenerated). The harness is measuring what the original measurement measured.

**Per clip, at 4 s** — the spread matters more than the mean:

| clip | AUC | | clip | AUC |
|---|---|---|---|---|
| 21 | 95.6% | | 36 | 85.4% |
| 22 | 92.3% | | 38 | 93.6% |
| 23 | 85.3% | | 40 | 81.6% |
| 24 | 89.5% | | 43 | 93.0% |
| 25 | **75.5%** | | 50 | 88.8% |
| 26 | 85.3% | | 58 | **80.9%** |
| 35 | 87.0% | | | |

Clip 58 is **46% of all scored frames** (49,364 of 108,248), so the pooled row is
largely clip 58's row. Clips 25, 40 and 58 are the weak ones and are where the
headroom is.

### The window question is answered: longer is better

Defect 1 predicted this and the sweep confirms it. **8 s beats 4 s on all three
metrics, including best-F1** — so there is no separation/threshold trade to
manage at the frame level, which is the level the orchestrator will consume.

This does not contradict `point_end.py`'s comment that 8 s scores a worse F1
than 4 s. That comment is about **event** F1 for the falling edge, a different
metric answering a different question. Frame separation genuinely prefers the
longer window; edge *timing* preferred the shorter one. Moving the edge-finding
into the orchestrator is precisely what dissolves the conflict.

5 s captures most of the gain (+1.0 mean AUC over 4 s against 8 s's +1.3) and
stays closer to the edge. **The default is 5 s and `raw` is stored regardless**,
so the orchestrator can re-window without a re-run.

### Defects 3 and 4, quantified

Both were argued from first principles in the first draft. Measured now:

**Defect 3 — unmeasurable frames are scored as confidently dead.** This was
argued in the first draft as a serious error and **the measurement does not
support that.** The honest version follows, because the first version of this
section was wrong.

Pooled over the corpus minus clip 58, **9.0%** of scored frames have neither a
near nor a far player tracked. Of those, **95% have no player box in any slot at
all** — so this is absence of a detection, not an artifact of the activity math.
The gates are not the cause: across every short-run unmeasurable frame in the
corpus, `MIN_H_PX` killed **0** detections, the tracking zone **79**, and height
plausibility **1**.

Split by how long the unmeasurable stretch lasts, it is two unrelated
populations:

| run length | runs | % of unmeasurable frames | of which labelled LIVE |
|---|---|---|---|
| 0–1 s | 79 | 7.2% | 30.4% |
| 1–3 s | 32 | 14.3% | 22.7% |
| 3–10 s | 11 | 15.9% | 5.3% |
| **>10 s** | **7** | **62.6%** | **0.0%** |

**Two thirds of it is seven long runs containing no live tennis whatsoever.**
Changeovers and breaks, players genuinely off court. Scoring those as dead is
*correct*, and a `valid` channel that removed them from the denominator would
discard frames the curve already gets right.

The short runs are the ones with live frames in them, and they are not detection
failures either. **81% of short-run unmeasurable frames have BOTH ROIs empty at
the same sample** — the near 540p whole-frame pass and the far native-res band
pass, two independent passes, finding nobody simultaneously. The footage says
why: the court is empty. Clip 50 at 117.5 s and 159.5 s is a calibrated court
with nobody on it and play happening on the adjacent court; clip 22 at 43.2 s
and clip 24 at 277.0 s each show a single player walking near the net with the
near half empty and a ball sitting on the ground. That is between-point footage
carrying a live label.

Measured rather than eyeballed: of the live-labelled unmeasurable frames,
**64% sit within 3 s of an edge of their own labelled rally** (median 2.5 s),
which is label over-extension at rally boundaries.

**Total harm: 334 frames, 0.57% of scored frames**, of which 121 (**0.21%**) are
deeper than 3 s inside a labelled rally and not explained by boundary slop.

**So the `valid` channel is demoted.** It is not load-bearing for AUC and it is
not step-2 work. What survives the correction is the inverse, and it is more
useful than the original claim: **absence is not an abstention, it is evidence
of dead** — 0.0% live across every long unmeasurable run in the corpus. A
conservative end policy should read "nobody is on the court" as strong evidence
the point is over, which is a signal for the ORCHESTRATOR rather than a
correction to the curve.

**Defect 4 — the union is near-side and the near player is often the absent
one.** Frames where ONLY the far player is tracked: clip 24 **48.7%**, clip 23
**47.1%**, clip 35 **44.0%**. On roughly half of those three clips the
non-rally union — walking classifier plus `near_end`'s four signals, all fed
from the near track through a shim — is being computed from a player who is not
there. A far-side union term is not a refinement; on these clips it is most of
the footage.

## Rally confidence: the contract

Artifact `<stem>_anya2_rally.npz`, at pose rate:

| channel | meaning |
|---|---|
| `raw` | per-frame rally evidence, **unsmoothed** |
| `conf` | windowed confidence in [0, 1] — the headline signal |
| `valid` | coverage: is this frame measurable at all (diagnostic; see defect 3) |
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

**3. "Not live" and "cannot tell" are different — but measurement says this
barely matters. DEMOTED.** The draft argued that an unmeasurable window scored
as dead is a vote against rather than an abstention. It was measured (see
"Defects 3 and 4, quantified") and it is worth **0.57% of scored frames**, two
thirds of which are long empty-court stretches containing **no live tennis at
all**. Absence turns out to be good evidence of dead, not a missing measurement.
The `valid` channel is still emitted — it costs nothing and the orchestrator
should know what it is standing on — but it is **not a step-2 priority and is
not expected to move AUC.**

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
