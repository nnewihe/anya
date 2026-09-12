# Rally confidence — refocusing agent 3, and moving the reasoning into agent 4

> **Numbers corrected 2026-09-12.** Every reel figure below was first measured
> against serve-event JSONs that predated this branch's far-serve work, and was
> therefore stale. Regenerated from the current detectors the reel is
> **140/154 whole points, 95.7% live retained, 71.6% of span, 0 truncations**.
> The tables below keep the arms' RELATIVE ordering, which is what each
> decision turned on, but their absolute values are the stale ones except where
> a row says otherwise. Regenerate the event JSONs before trusting any reel
> number: they go stale silently and a stale run looks exactly like a real one.

> Status: **done.** All four steps. `rally.py` is built and measured
> (`rally_eval.py` scores the curve), the orchestrator ends points off that
> curve (`reel_eval.py` scores the reel), and `point_end.py` is deleted.
> **Clip 58 is excluded from every corpus number below**, at the user's
> direction — it was 46% of all scored frames. It is still run as a holdout
> where that is informative, and labelled.
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

## Step 2 result: `rally.py`

Built, measured, shipped. **Corpus minus clip 58, 12 clips, 58,884 scored
frames.**

| arm | mean per-clip AUC | pooled AUC | pooled best-F1 |
|---|---|---|---|
| `current` — `point_end.live_score`, 4 s | 87.7% | 86.7% | 73.2% |
| `rally_noabs` — 5 s, absence off | 88.6% | 87.4% | 73.5% |
| **`rally` — 5 s + absence** | **89.7%** | **88.6%** | **75.7%** |

The ablation arm exists so the two changes separate: the window is worth +0.9
mean AUC and **the absence term is worth +1.1**, so the new term is the larger
of the two. Both arms run off the same cached pose passes, so the construction
is the only variable.

### The absence term, and how the obvious version of it earned nothing

The user's observation — longer stretches with the players unmeasured mean dead
time is more likely — is correct and is now in the construction. What the
measurement changed is *which* absence carries it.

Absence is applied **after** the smoothing. Before it, it is a no-op: with
nobody tracked, activity is already zero and `raw` is already zero. What the
term actually removes is the **smoothing leak** — a 5 s window at the edge of a
long empty stretch spreading real activity into footage with no one on court.
Killing that leak sharpens the edge the orchestrator has to find, which is why
it beats simply widening the window.

**A joint "nobody on court" term was built first and is worth nothing.** The
live-prevalence table says it should be decisive — both-absent is 0.0% live
from 3 s on, while near-only absence is still 15.6% live at 15 s:

| w_near | w_far | w_both | mean per-clip AUC |
|---|---|---|---|
| 0 | 0 | 0 | 88.65% (off) |
| 0 | 0 | 1.0 | 88.67% ← the "decisive" term |
| 0 | 0.25 | 0 | 88.70% |
| 0.25 | 0 | 0 | 89.66% ← the near term alone |
| **0.25** | **0.25** | **0** | **89.71%** ← shipped |

**Marginal prevalence is not incremental value.** Both sides absent already
drives `raw` to zero over a wide window, so there is nothing left for a veto to
remove. The case only the near term can see is the near player gone while the
far player is still tracked and still moving, holding `max(near, far)` up over
footage where the point is long over. **The near player is the one that matters,
exactly as predicted — but the table that seemed to refute that was measuring
the wrong thing.**

**The weight is an interior optimum, not a slope.** At w_near 1.00 mean AUC
collapses to 83.1%, because a full veto deletes real live play wherever the near
player is briefly untracked — and under 3 s, absence is evidence of *live*
(every column at or above the 35.9% base rate). 0.25 / 0.50 / 0.75 / 1.00 scores
89.7 / 90.0 / 88.9 / 83.1.

### One honest negative: 0.50 scores higher and is not shipped

| | mean | clips improved | clip 35 (out-of-sample) | clip 58 (holdout) |
|---|---|---|---|---|
| **0.25 / 0.25** | +1.1 | **12 / 12** | +0.1 | +1.1 |
| 0.50 / 0.25 | **+1.4** | 7 / 12 | **−1.1** | **+2.0** |

0.50 buys +0.3 of mean AUC by regressing five clips, one of them the corpus's
designated out-of-sample clip. **Clip 58 prefers 0.50**, and that is the one
piece of evidence pointing the other way — recorded rather than dropped. A
setting that improves every clip it is measured on transfers more credibly than
one that is 0.3 better on average and worse in five places. Revisit if the
orchestrator turns out to want the sharper edge more than the safer curve.

### Per clip, shipped arm against the current construction

| clip | current | rally | | clip | current | rally |
|---|---|---|---|---|---|---|
| 21 | 95.6% | **96.4%** | | 36 | 85.4% | **89.5%** |
| 22 | 92.3% | **93.9%** | | 38 | 93.6% | **97.1%** |
| 23 | 85.3% | **85.9%** | | 40 | 81.6% | **82.0%** |
| 24 | 89.5% | **92.5%** | | 43 | 93.0% | **96.6%** |
| 25 | 75.5% | **77.2%** | | 50 | 88.8% | **89.7%** |
| 26 | 85.3% | **88.2%** | | | | |
| 35 | 87.0% | **87.5%** | | | | |

Clips 25 (77.2%) and 40 (82.0%) remain the weak ones and both are doubles. That
is the next thing to look at, and it is a tracking question rather than a curve
question.

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

## Doubles: the veto has to describe the player who is playing

Found while investigating why clips 25 and 40 were the two weakest on the
curve. **The activity term was already per-team max** — `nanmax` over
`NEAR_SLOTS` — but the non-rally union was not. `run._end_signals` builds one
shim from whichever near slot has better coverage, and that single player's
union was then applied multiplicatively to the whole court. In singles that is
the only player there. In doubles, **a partner standing at the net vetoes the
activity of the partner hitting the ball.**

Mean union during **live** play:

| clip | union \| live |
|---|---|
| **25 (doubles)** | **0.413** |
| **40 (doubles)** | **0.388** |
| singles range | 0.09 – 0.37 |

The two doubles clips are the two highest in the corpus — the veto fires
hardest exactly where it is least entitled to. Both near slots are tracked on
36–38% of frames there against under 5% on most singles clips, so there are
genuinely two players to choose between.

`rally.team_union(mode="active")` now picks, per frame, the union of the slot
that supplied the max activity. Activity and veto describe one person.

**The obvious alternative is worse.** `mode="min"` — a side is non-rally only if
every player on it looks non-rally — weakens the veto uniformly instead of
re-aiming it, which helps where the wrong player was picked and hurts where the
right one was:

| arm | clip 25 | clip 40 | mean per-clip AUC |
|---|---|---|---|
| shim (one player) | 77.2% | 82.0% | 89.7% |
| min | 78.4% | **80.6%** | 89.6% |
| **active** | **79.2%** | **83.9%** | **90.0%** |

Curve AUC is now **90.0% mean per-clip, 89.1% pooled, best-F1 76.1%**.

## Step 3 result: ends off the curve

`orchestrator.pair_ends_curve`, selected by `ReelConfig.end_policy="curve"`.
The end of a point is where rally confidence **falls and stays fallen** —
the first sample whose following `end_dwell_s` are all below `end_lo`, searched
between `min_point_s` after the serve and the next serve. The old
event-matching path is untouched and still reachable with `end_policy="events"`.

Both arms read the same cached serve detections and the same pose passes, so
the end policy is the only variable. **12 clips, 154 labelled points.**

| arm | whole points | live kept | reel % of span | dead/pt | end R | end P | **trunc** |
|---|---|---|---|---|---|---|---|
| `events` (shipped) | 129 / 154 | 94.8% | 62.5% | 7.1 s | 27.3% | 70.0% | **0** |
| `curve`, shim union | 138 / 154 | 95.6% | 68.5% | 8.6 s | 15.6% | 58.5% | **0** |
| **`curve`, active union** | **137 / 154** | **95.6%** | 69.1% | 8.7 s | 20.1% | 67.4% | **0** |

**+8 whole points and +0.8 live retained, at zero truncations, for 6.6 points of
extra reel length.** Per clip, seven gain a whole point, four are flat, and one
(clip 40) loses one.

The doubles fix costs one whole point against the shim union and buys back most
of the end-accuracy gap it opened — precision 58.5% → 67.4% against the events
arm's 70.0%, recall 15.6% → 20.1%. **Ends placed on the corrected curve are
materially more accurate**, which matters for step 4 and for anything later that
wants the end timestamp rather than the segment.

### This is a trade, not a clean win

End *event* accuracy gets worse — recall 25.3% → 15.6%, precision 69.6% →
58.5%, and the number of points falling back to an estimated duration rises
from 29 to 40. The ends the curve policy places are **later**, so fewer land
inside the ±2 s window.

Under the stated brief that is the right direction: truncations stay at zero,
a late end costs dead time, and **a whole point is the unit a viewer notices**.
It is recorded as a trade because a future change that improves end-event
accuracy should not be assumed to improve the reel.

**One honest negative, and it is clip 40 again.** Whole points 11 → 10 and live
retention 96.2% → 87.0%. Its ends got much *better* as events — recall 23.1% →
53.8%, precision 42.9% → 77.8% — and its reel tightened from 45.4% of span to
36.9%. So the policy is now cutting clip 40 close to where its labelled ends
are, and losing live tennis in the last second before them. No end is more than
2 s early (truncations stay 0), so this is post-roll territory rather than a
policy failure: clip 40 would likely be recovered by a longer post-roll, which
is a reel setting and not an end-detection one. The other doubles clip, 25,
improves on every axis — live 93.7% → 95.8%, end precision 75.0% → 83.3%.

### The curve is doing the work, and the optimum is interior

Two controls, because "keep more footage, catch more points" is the obvious
confound:

| end_lo | whole / 154 | live kept | reel % of span |
|---|---|---|---|
| null — curve never read | 101 | 87.4% | 55.4% |
| 0.25 | 122 | 93.1% | 62.0% |
| 0.20 | 131 | 94.4% | 64.6% |
| **0.15** | **137** | **95.6%** | 69.1% |
| 0.10 | 135 | 94.4% | 69.0% |

(Re-swept on the corrected curve; the null row is from the previous sweep and
does not depend on the union change.)

The **null** arm sets every end from the clip's own typical duration and never
reads the curve. At 101 whole points it establishes that the gain is the curve,
not the fallback. And **0.10 produces a longer reel and fewer whole points than
0.15**, so the peak is not a length effect. Dwell is flat 1.5–3.0 s; 1.5 s is
taken because it reaches the same retention in a shorter reel.

### The fall is measured against the point's own peak

`end_mode="relative"` is the default: a point ends where confidence has fallen
to `end_rel` (0.10) of its **own running peak**, sustained for `end_dwell_s`
(2.5 s). Running peak accumulated from the serve, not a max over the window, so
the bar can only rise as the rally develops — a max would let a burst *after* a
quiet stretch retroactively raise the threshold and turn an ended point back
into a live one.

**Why relative, given the numbers are a wash:**

| mode | whole / 154 | live kept | reel % | end R | end P | trunc |
|---|---|---|---|---|---|---|
| `events` (old policy) | 129 | 94.8% | 62.5% | 27.3% | 70.0% | 0 |
| `absolute` 0.15 | 137 | 95.6% | 69.1% | 20.1% | 67.4% | 0 |
| **`relative` 0.10 / 2.5** | **137** | **95.9%** | 72.6% | 13.0% | 50.0% | **0** |
| `both` 0.35 / 0.15 | 137 | **96.3%** | 69.7% | 19.5% | 66.7% | 0 |

All three curve modes reach the same **137 whole points at zero truncations**.
The decision is therefore not made on these numbers. `conf` is normalised per
clip (`rally.SCALE_PCT`), so an absolute level means a different thing on every
clip and is only as portable as that normaliser — defect 2, still open. A ratio
against the point's own peak is **scale-free**, and that is the whole reason to
prefer it.

The corpus cannot test the claim: all 12 clips are in-sample. This is a choice
made on the construction, not on the measurement, and it is worth saying so.

**`both` is available and is not the default.** It retains the best live
fraction, but it keeps the absolute threshold and therefore the portability it
was meant to remove — at `lo=0.15` the absolute term is the binding one, and
sweeping `end_rel` from 0.25 to 0.45 changes nothing at all.

**What relative pays for its independence:** a longer reel (72.6% of span
against 69.7%), weaker end-event precision because its ends land later, and
**45 of 154 points falling back to an estimated duration** against 29 under the
events policy. The relative rule simply fires less often.

**The clip-40 regression is gone.** The doubles clip that lost a whole point and
9 points of live retention under the absolute rule recovers completely: whole
10 → 11, live 87.0% → 95.9%, and its ends are the most accurate in the corpus
(recall 53.8%, precision 77.8%).

**The regression moved to clip 43**, which loses one whole point (6 → 5) and
6 points of live retention. Its cause is the fallback, not the rule: the
relative fall never triggers there, so all its points use an estimated
duration, and with fewer than three curve ends to learn from
`estimate_point_s` drops to the 9.0 s global default. A clip where the rule
never fires is the case the fallback is worst at, and that is the next thing to
improve.

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

## Step 4: the old path is deleted

`point_end.py` is gone. What went with it:

- `detect_ends()` and `detect_video()` — the falling-edge detector
- the `_anya2_point_end.json` event stream, and the orchestrator's reading of it
- `ReelConfig.use_end` and `ReelConfig.end_policy`, plus `pair_ends` /
  `_pair_once`, the event-matching pairing
- `Anya2Config.end` (`PointEndConfig`), replaced by `RallyConfig`

What did **not** go: the pose machinery behind the curve —
`player_activity`, `quiet_mask`, `end_signal`, `UNION_NAMES` and their
constants — moved into `rally.py` unchanged. Those were never the weak part.
The detector was.

`contract.POINT_END` also stays: `eval.py --mode point_end` uses it to name the
ground-truth ends, which is still how the orchestrator's ends are scored.

**Two verifications, both required before deleting anything:**

1. The reel is bit-identical across the deletion — 137/154 whole, 95.9% live,
   0 truncations, before and after.
2. `rally_eval --arm current` reconstructs the pre-redesign curve exactly
   (4 s smoothing, single-player shim union, absence off) and still scores
   **87.7% mean per-clip AUC**, the number this work started from. The
   historical baseline remains runnable with the module that produced it
   deleted.

The orchestrator now treats the curve as **required rather than optional**.
There is no second construction to fall back to, so a clip whose rally artifact
cannot be built raises instead of silently emitting every point at its default
duration.

`desktop/rally_app.spec` had a PyInstaller hidden import for the deleted
module; it now names `pipeline.anya2.rally`. Nothing else outside `pipeline/anya2`
referenced it — `pipeline/rally_reel/*` has its own unrelated `end_policy` and is
untouched.

### Where this leaves the whole redesign

| | start | now |
|---|---|---|
| curve, mean per-clip AUC | 87.7% | **90.0%** |
| whole points | 129 / 154 | **137 / 154** |
| live retained | 94.8% | **95.9%** |
| truncations | 0 | **0** |

### What is still open

- **Defect 2, the per-clip normaliser**, is mitigated rather than fixed. The
  end rule no longer depends on it (the fall is relative to the point's own
  peak), but `conf` itself is still divided by the clip's 90th percentile, so
  anything else that thresholds it inherits the hazard.
- ~~`estimate_point_s` on clips where the rule never fires.~~ **Fixed — see
  below.**
- **The ball.** Still deferred, unchanged.

## The fallback, replaced

When the relative fall never happens inside a point's window, something has to
end it anyway. That fallback was the weakest link, and measured it was wrong in
**both directions at once**:

| clip | curve ends | duration it assumed | that clip's GT p85 | |
|---|---|---|---|---|
| 43 | 0 | 9.0 s | 15.5 s | too short |
| 38 | 1 | 9.0 s | 10.8 s | too short |
| 24 | 4 | **36.4 s** | 20.1 s | too long |
| 25 | 5 | **36.1 s** | 12.3 s | too long |

Under three curve ends it fell to a global **9.0 s** against a corpus median of
7.3 s and p85 of **13.8 s**, so it truncated. At three or more it took the 85th
percentile of the clip's own *curve* durations — but those are biased long,
because a point whose fall is found is disproportionately a point that clearly
ended. The percentile of a long-biased sample overshot by 16 s on clip 24.

**The replacement asks the curve again with the bar removed.** `end_backoff =
"quietest"` takes the quietest sustained stretch in the window — the lowest
`end_dwell_s`-length moving average. Same evidence, same window, same dwell;
only the acceptance test is gone, so it always has an answer. No prior about
how long a tennis point lasts enters the decision at all.

**Ties break late.** A rally can have two similarly quiet moments — a lull
mid-point and the real end — and a plain `argmin` picks whichever is lower by a
rounding error, which is a coin flip between truncating the point and ending it.
`end_quiet_tol` (0.25) takes the **last** stretch within that fraction of the
quietest, scaled by the window's own spread so a flat window resolves to the end
of the window rather than to noise.

| backoff | whole / 154 | live kept | reel % | blind estimates | trunc |
|---|---|---|---|---|---|
| `duration` (old) | 137 | 95.9% | 72.6% | **45** | 0 |
| `quietest`, tol 0 | 134 | 95.3% | 68.0% | 9 | 0 |
| **`quietest`, tol 0.25** | **139** | **96.5%** | 74.3% | **8** | **0** |
| `quietest`, tol 0.4 | 140 | 96.6% | 75.5% | 8 | 0 |

Blind estimates fall from **45 of 154 to 8**. tol 0.4 and 0.6 buy one more whole
point and saturate — at that point the rule is converging on "run to the end of
the window", which is the behaviour the duration prior existed to avoid, so 0.25
is taken as the last value that is still choosing a moment rather than giving up.

**Clip 43 is fully recovered** — the regression introduced by the relative fall.
Whole points 5 → 6 (all of them), live retention 93.9% → 99.9%.

**The duration prior is now inert.** Setting `default_point_s` to 9.0, 13.8 or
20.0 s gives *identical* results on every clip — the quietest backoff absorbs
every case that used to reach it. `estimate_point_s`, `default_point_s`,
`est_duration_pct` and `est_duration_pad_s` survive only for the `duration`
comparison arm and for a window too short to hold one dwell. Nothing on the
shipped path consults them.

### Where the redesign now stands

| | start | now |
|---|---|---|
| curve, mean per-clip AUC | 87.7% | **90.0%** |
| whole points | 129 / 154 | **140 / 154** |
| live retained | 94.8% | **95.7%** |
| blind duration estimates | 29 / 154 | **10 / 154** |
| truncations | 0 | **0** |

(Both columns re-measured on fresh serve events; see the correction note at the
top of this file.)

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
