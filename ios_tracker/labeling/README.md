# Labeling + scoring harness

Ground truth for the ball tracker, so tuning is measured instead of guessed.

## The library workflow (start here)

Tuning on one clip overfits one court. The library takes 1–2 rallies from each
match in `/Volumes/Anya/Data` — 29 matches across clay and hard, indoor and
outdoor, 5 resolutions, 30/60/120 fps — so a weight has to earn its keep
everywhere.

```bash
python3 build_library.py --data /Volumes/Anya/Data --out library   # ~60 min, automated
python3 label_library.py --lib library --all                        # you; resumable
python3 score_library.py --lib library                              # honest number
python3 score_library.py --lib library --sweep ACCEL_WEIGHT=0.002,0.01,0.03
```

`build_library.py` finds rallies **without using the tracker** — it looks for
windows where non-clutter detections move. Seeding the training set from the
tracker's own output would teach it only what it already knows and hide its
failures. Each clip is padded into dead time either side, because drawing a ball
where there is none is the failure mode that matters and it is invisible unless
the library contains empty frames.

Clips are short by design: a sweep re-solves every clip on every trial, which is
seconds over ~5 min of clips and unusable over 12 h of source.

`step` in the manifest keeps labelling effort at ~30 labels per second of video
regardless of source frame rate — a 120 fps clip would otherwise cost 4× the
keystrokes for the same tennis.

`score_library.py` reports **macro-F1** (mean of per-clip F1), not F1 pooled over
frames: pooling lets a few long easy clips drown out a court where the tracker is
blind. Watch `worst F1` as closely as the mean — great on 9 courts and blind on
the 10th is not robust.

Clips and candidate caches are gitignored (regenerable); the manifest and labels
are the irreplaceable part and stay in git.

## Why this exists

The video harness reports `live trace: NN%`. That number is not quality. A
tracker that confidently draws a trajectory across empty court scores 100% and
is worthless — which is not hypothetical: dropping the detector threshold to
0.03 raised coverage from 29% to 67% while the solver quietly stitched noise
into a fictional 17-second trajectory. Only rendering it on the video revealed
that. These tools replace that eyeball loop with numbers:

| metric | meaning | how it's gamed |
| --- | --- | --- |
| `recall` | of frames where the ball IS visible, how many we found | draw everywhere |
| `precision` | of the points we drew, how many were on the ball | draw almost nothing |
| `ghost rate` | frames with no ball where we drew anyway | — |
| **`F1`** | harmonic mean of recall & precision | **can't be gamed either way** |

Tune on `F1`; watch `ghost rate` for fiction.

## 1. Label

```bash
python3 label_ball.py --video /tmp/clip.mov --out labels.json --start 900 --end 1150
```

Runs the real Core ML detector at conf 0.02 and draws its candidates as numbered
circles — press a **digit** to accept one rather than clicking a 14 px ball.
Click only when the detector missed it entirely.

```
0-9        accept that candidate (auto-advances)
click      place the ball manually (auto-advances)
n          no ball visible in this frame (auto-advances)
u          clear this frame
a / d      prev / next frame        [ / ]   jump 10
z          toggle magnifier         s  save        q  save & quit
```

Resumable, and unlabelled frames are simply skipped by the scorer — so label a
couple of rallies (a few hundred frames) and stop. **Label the no-ball frames
too** (`n`): without them `ghost rate` is blind, and ghosts are the failure mode
that matters here.

Candidates are cached to `labels.json.cands.json`; delete it to recompute.

## 2. Score

```bash
DUMP_CSV=/tmp/trace.csv ../run_video_check.sh /tmp/clip.mov
python3 score_trace.py --labels labels.json --trace /tmp/trace.csv
```

`--tol` is the hit radius in source px (default 25, about one ball width).

## 3. Tune

```bash
python3 sweep.py --video /tmp/clip.mov --labels labels.json \
    --param ACCEL_WEIGHT=0.0015,0.004,0.01 \
    --param FRAME_REWARD=0.5,1.0,2.0
```

Each trial runs the **real Swift solver** with `VITERBI_*` env overrides
(`ViterbiConfig.fromEnvironment`), so there's no rebuild per trial. Ranked by F1.
Apply a winner by editing the matching default in `ViterbiConfig`
(`BallTracker/Tracking/ViterbiTracker.swift`).

One trial ≈ one decode+solve (~15 s per 60 s of video). Keep grids small.

## Where to start

`ACCEL_WEIGHT` vs `FRAME_REWARD` is the tradeoff that decides whether the solver
chains noise: the reward per tracked frame has to be worth *less* than the cost
of an implausible link, or fiction pays. Every weight in `ViterbiConfig` is
currently one engineer's judgment against a single 61 s clip — none of it is
validated. Once labels exist, the open question worth settling first is whether
`BallDetector.solverConf = 0.03` plus a harsher `ACCEL_WEIGHT` beats the current
conservative `0.10`; that's the regime where a global solver should earn its
keep, but it has never been measured.
