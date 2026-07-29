# Near-player walking classifier

Detects when the **near-side player is walking** in a tennis video, as opposed to
stationary or actively playing. Trained on hand-labelled intervals from
`/Volumes/Anya/Data/21/snippet.mp4` (indoor hard) and
`/Volumes/Anya/Data/22/snippet.mp4` (outdoor clay).

## Results

Trained on two hand-labelled clips: **21** (indoor hard court, 21 intervals, 129 s
of walking in 420 s) and **22** (outdoor clay, different camera and players, 15
intervals, 109 s in 420 s).

### Cross-clip — the number that matters

Leave-one-clip-out: train on one clip, test on the other. Post-processing is
tuned on out-of-fold probabilities *within the training clip* and then frozen, so
the test clip informs nothing.

| test clip | trained on | precision | recall | F1 (frame) | F1 (second) | events |
|---|---|---|---|---|---|---|
| 21 (indoor hard) | 22 (outdoor clay) | 0.956 | 0.756 | **0.844** | 0.845 | 18/21, prec 0.95 |
| 22 (outdoor clay) | 21 (indoor hard) | 0.738 | 0.921 | **0.819** | 0.805 | 13/15, prec 0.87 |

The classifier transfers across surface, lighting, camera geometry and players —
and cross-clip scores *better* than within-clip CV, because a whole second clip is
a stronger training signal than 12 blocks of the same one. The clearest evidence:
the 183–191 s walk in clip 21, which within-clip CV missed completely (p ≈ 0.05),
is caught at 0.76 recall by the model that only ever saw clay.

The two operating points differ (clip 21 tested: precision-heavy; clip 22 tested:
recall-heavy) because the hysteresis thresholds were tuned on the other clip's
probability distribution. Expect to re-tune thresholds per deployment, or pool
more clips.

### Within-clip

5-fold `GroupKFold` over contiguous 30 s blocks, model and post-processing fitted
per fold on the other folds only.

| | precision | recall | F1 |
|---|---|---|---|
| clip 21, per second | 0.872 | 0.785 | **0.826** |
| clip 22, per second | 0.884 | 0.819 | **0.850** |
| prior hand-tuned cadence/speed rule on clip 21 (`codex/`, reported) | 0.619 | 0.543 | 0.579 |
| single best court-speed threshold, clip 21, same CV | 0.590 | 0.775 | 0.670 |
| clip 21, per frame, ±0.5 s boundary guard | 0.922 | 0.828 | 0.872 |

Per-fold F1: clip 21 mean 0.833 (sd 0.090), clip 22 mean 0.813 (sd 0.098).
Event level (IoU ≥ 0.3): clip 21 16/21 walks at precision 0.889 and 0.29 false
events per minute; clip 22 12/15.

### Remaining failure modes

- **Short events.** Nearly every miss is under 3 s (clip 21: 163–165 s, 311–313 s;
  clip 22: 170–172 s, 210–212 s). The 8 s context window that makes long walks
  reliable dilutes brief ones.
- **Pose loss at extreme scale.** Clip 21's 163–165 s walk has 21 % detection
  coverage — the player fills the frame and keypoints drop out.
- **One candidate label gap.** On clip 22, 387.5–397.7 s is predicted walking at
  p = 0.88 and is unlabelled; the near player (after the end change) appears to be
  walking back into position. Worth a look before counting it as an error.

## Pipeline

```
video ──► extract_pose.py ──► select_near.py ──► features.py ──► model ──► evaluate.py
          all person poses     the near player    373 window      HistGB     hysteresis
          (+ hi-res rescue)    per frame          features                   + intervals
```

1. **`extract_pose.py`** — yolov8n-pose on every frame at 960×540, caching *every*
   person detection (`<clip>/<stem>_walk_dets.npz`). `--rescue` re-runs the frames
   that came back empty at 1920 px: a player walking to the ball carts is ~60 px
   tall and invisible to the 960 px pass. That pass alone took blind frames from
   25.4 % to 7.7 % and recovered a whole labelled walk (82–92 s went from recall
   0.00 to 0.89).
2. **`select_near.py`** — picks the near player per frame. There is no "must be on
   the singles court" gate: players walk off court to the carts and change ends,
   and gating on the court loses exactly the frames that carry walking labels.
   Instead, the far half is excluded (that is the opponent), the near half is
   preferred, and the surrounding floor stays eligible. Continuity is scored in
   **court metres**, and a candidate implying more than 9 m/s from the current
   track is rejected outright rather than accepted as a bad match — the detector
   often finds only one of two nearby people, and a penalised-but-still-best
   candidate silently teleports the track. This fix alone moved frame F1 from
   0.817 to 0.826 and turned the 183–191 s region from garbage (3.8 m/s mean
   "speed", 14 m of jitter) into a real 1.4 m/s walk.
3. **`features.py`** — per-frame court position (via the cached court homography),
   speed/acceleration, gait descriptors normalised by body height (ankle
   separation, knee angle, hip bob, wrist motion relative to the torso), then
   statistics over centred 0.5/1/2/4/8 s windows plus net-vs-path displacement,
   straightness, speed duty cycle, and cadence spectra. 373 features.
4. **model** — `HistGradientBoostingClassifier` (NaN-native, so detection gaps
   flow through as missing rather than as zeros), decoded with two-threshold
   hysteresis (0.45 / 0.225, minimum duration 2 s).

## What the ablation showed

| dropped feature group | F1 | worst fold |
|---|---|---|
| nothing | 0.728 | 0.497 |
| **pixel body scale (`box_h`, `aspect`)** | **0.754** | 0.518 |
| gait cadence spectra | 0.736 | 0.540 |
| absolute court position | 0.722 | 0.447 |
| posture (knee, hip bob, torso lean) | 0.660 | 0.321 |

Body scale in pixels is a camera-and-depth fingerprint: it tells the model *where
in this clip* a sample came from, and it is dropped. Posture is the most valuable
group by a distance — the top features are consistently knee-angle spread and
ankle separation over 8 s, i.e. *how the legs move*, not how fast the player
travels. Cadence spectra, the basis of the prior hand-tuned rule, add almost
nothing here.

## Usage

```bash
# one-off, per clip: detect, rescue the empty frames, pick the near player
python -m walking.extract_pose /path/clip.mp4
python -m walking.extract_pose /path/clip.mp4 --rescue
python -m walking.select_near  /path/clip.mp4

# train + score: within-clip CV per clip, plus leave-one-clip-out across them
python -m walking.train --clips 21 22

# inference on any clip with the same fixed camera + court cache
python -m walking.predict /path/clip.mp4 --out walking.json --jsonl walking.jsonl \
    --overlay review.mp4 --overlay-start 140 --overlay-seconds 60
```

`predict` emits intervals with `mean_prob` **and `detection_coverage`**, the share
of the interval in which a person was actually tracked. Low-coverage intervals are
guesses spanning a hole in the input; gate on that field rather than trusting the
probability alone.

Requires per clip: the video, and `<stem>_court_cache.json` with the four singles
corners in 960×540 order (near-left, near-right, far-right, far-left).

## Limitations

- **Two clips.** Cross-clip transfer is demonstrated (indoor hard <-> outdoor
  clay), but on a sample of two: both are singles, both are shot from behind the
  near baseline at a similar height, and clip 22's players overlap clip 21's. A
  third clip with a different framing (higher/lower camera, doubles, a different
  pair) is the next thing that would move the estimate.
- Decision thresholds do not transfer cleanly even when the model does: the two
  LOCO runs land on quite different precision/recall points. Pool more clips, or
  re-tune the threshold per deployment.
- Short walks (under ~3 s) are the dominant miss.
- Interval boundaries carry the label author's reaction time; the ±0.5 s
  boundary-guarded row exists to show how much of the residual error is that.
- A fixed camera is assumed — one homography per clip, no per-frame court
  tracking.
