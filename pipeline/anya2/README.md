# anya2 — point boundaries, rebuilt as three independent detectors

Three questions, three detectors, one shared substrate. Nothing here imports
from `pipeline/rally_reel/` or `pipeline/anya_*.py`; those keep shipping
untouched so every number below is A/B-able against them.

| detector | question | status |
|---|---|---|
| `near_serve.py` | did a point start with a near-side serve? | **built** |
| `far_serve.py` | did a point start with a far-side serve? | **built** |
| `point_end.py` | did the point end? | not yet |

## Substrate

| module | what it owns |
|---|---|
| `court.py` | image→court-metres map, and the two player gates |
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

## Near-side serve — results

Scored against `ground_truth.json` with `eval.py`, ±2.0 s, greedy one-to-one,
restricted to each clip's labelled span. **All 12 trusted clips, 102 near
serves.**

| clip | labelled | recall | precision |
|---|---|---|---|
| 21 | 11 | 90.9% | 90.9% |
| 22 | 11 | 100% | 91.7% |
| 23 | 0 | — | **0 fires** |
| 24 | 2 | 100% | 66.7% |
| 25 (doubles) | 5 | 100% | 100% |
| 26 | 2 | 100% | 100% |
| 36 | 8 | 100% | 100% |
| 38 | 8 | 100% | 100% |
| 40 (doubles) | 0 | — | **0 fires** |
| 43 | 6 | 100% | 100% |
| 50 | 5 | 100% | 71.4% |
| **58 (holdout)** | **44** | **65.9%** | **55.8%** |
| pooled | 102 | 84.3% | 75.4% |

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
**Nine clips carry a far serve; 77 of the corpus' 114.** Clip 58's other 37 are
the holdout and its far pass is still running.

| clip | labelled | recall | precision |
|---|---|---|---|
| 23 | 15 | 100% | 78.9% |
| 24 | 12 | 100% | 92.3% |
| 25 (doubles) | 10 | 100% | 71.4% |
| 26 | 11 | 100% | 45.8% |
| 36 | 9 | 100% | 75.0% |
| 40 (doubles) | 13 | 100% | 81.2% |
| **six far-dominant clips** | **70** | **100%** | **71.4%** |
| 21 | 1 | 0% | 0% (14 fires) |
| 22 | 4 | 75% | 18.8% |
| 50 | 2 | 0% | 0 fires |
| pooled | 77 | 94.8% | 57.0% |

Bias +0.07 s. **Every far serve on every far-dominant clip is found.** The three
clips that look terrible are near-dominant — 1, 4 and 2 far serves against 11, 11
and 5 near ones — where the far player spends the clip RETURNING, and a return
is a serve motion the detector has no way to place in a point.

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

Of the far detector's false positives, **80% fall in a live point** — 77%
in-rally, 3% the far player reacting to a near serve — against 20% idle raises
in dead time. The shipped `anya_far_serve` measured the same taxonomy at 56%
over 14 clips; ours is more concentrated because recall is higher.

None of that is visible from inside a far-player pose crop. The module declares
`Requirement(windows="between_points")` and leaves the arbitration to the
composition layer, which will have the near serves and the point ends that
settle it. **Treat 57% pooled precision as the floor for a serve detector with
no knowledge of whether a point is in progress** — and note that the near
detector bottomed out at 55.8% for exactly the same reason.

## Usage

```bash
python -m pipeline.anya2.perceive /Volumes/Anya/Data/21/snippet.mp4 --roi near
python -m pipeline.anya2.perceive /Volumes/Anya/Data/21/snippet.mp4 --roi far
python -m pipeline.anya2.tracks   /Volumes/Anya/Data/21/snippet.mp4 \
    --dets   /Volumes/Anya/Data/21/snippet_anya2_near_dets.npz \
    --far-dets /Volumes/Anya/Data/21/snippet_anya2_far_dets.npz
python -m pipeline.anya2.near_serve /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.far_serve  /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.eval --mode near_serve --arm anya2:_anya2_near_serve.json
python -m pipeline.anya2.eval --mode far_serve  --arm anya2:_anya2_far_serve.json
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
- **Slot flapping on clip 38** — 31 identity switches over 420 s on a singles
  clip. The doubles clips are stable (2 and 1), which is what the "who served"
  answer depends on, so this is not currently costing anything measurable.
