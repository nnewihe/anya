# anya2 — point boundaries, rebuilt as three independent detectors

Three questions, three detectors, one shared substrate. Nothing here imports
from `pipeline/rally_reel/` or `pipeline/anya_*.py`; those keep shipping
untouched so every number below is A/B-able against them.

| detector | question | status |
|---|---|---|
| `near_serve.py` | did a point start with a near-side serve? | **built** |
| `far_serve.py` | did a point start with a far-side serve? | not yet |
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

Scored against `ground_truth.json` with `eval.py`, ±2.0 s, greedy one-to-one.
**Nine trusted clips carrying a near serve; clip 58 (44 more) is the holdout.**

| | recall | precision | bias |
|---|---|---|---|
| seed, ported as-is | 50.0% | 55.8% | +1.45 s |
| + calibrated point-start | 79.3% | 100% | +0.04 s |
| + trophy dilation, threshold 0.70 | **100% (58/58)** | **93.5%** | +0.10 s |

Per clip: 21, 24, 25, 26, 36, 38, 43, 50 all at 100% recall; 22 at 100%. Four
false positives total (22:1, 24:1, 50:2). **Clips 23 and 40 have no near serves
at all and the detector fires zero times on either** — including 40, which is
doubles, so two near players stand at the baseline through 13 far-serve rallies
without producing a start.

Three things got it there, in order of how much they mattered:

1. **The trophy is a phase, not an instant.** The seed multiplies three shape
   terms sample-by-sample. On Data/38's five missed serves every term clears its
   threshold comfortably — `hi_head` +0.19…+0.27 against a 0.10 line — while the
   product never exceeds 0.30, because the tossing arm reaches full extension a
   sample or two before the racket arm settles. Each term is now dilated ±0.20 s
   before the product. This alone took recall 79% → 100%.

2. **Trophy onset is not the point start.** The seed reported it, and it lands
   +1.63 s after the label. Walking back to the last instant of the ready stance
   (hands still together on the grip) gets to +1.13 s; the rest is a labelling
   convention — `start` is marked before the server's hands move, so no
   definition taken from the serve motion can reach it. Corrected as one fitted
   constant, validated leave-one-clip-out: **44/45 (98%)**.

3. **The eval was scoring unlabelled regions.** Clip 38 is labelled to 206 s and
   then stops, leaving 214 s unlabelled — where all three of its apparent false
   positives lived. `eval.py` now restricts scoring to the labelled span and
   reports what fell outside it.

## Usage

```bash
python -m pipeline.anya2.perceive /Volumes/Anya/Data/21/snippet.mp4 --roi near
python -m pipeline.anya2.tracks   /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.near_serve /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.eval --mode near_serve --arm anya2:_anya2_near_serve.json
```

## Known gaps

- **Clip 58 is unscored** — 44 of the 102 labelled near serves. Its perception
  pass had not finished when this was written, and until it has, every number
  above is over 58 of 102 serves.
- **The far ROI is built but unused.** `perceive.far` derives the band from the
  homography and grows it upward by 2.8 m of court scale (a ground-strip band is
  ~55 px on a 4K clip and clips the server's raised arm). Nothing reads it yet.
- **Slot flapping on clip 38** — 31 identity switches over 420 s on a singles
  clip. The doubles clips are stable (2 and 1), which is what the "who served"
  answer depends on, so this is not currently costing anything measurable.
