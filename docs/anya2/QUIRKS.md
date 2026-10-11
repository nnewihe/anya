# anya2 — known quirks (preserved on purpose)

The October 2026 readability rewrite had to produce **byte-identical output**,
so these were kept exactly as they were. Each is marked at its site in the
code with `QUIRKS.md #n`. They are candidates for you to decide on later. Fixing
any of them will change results, so re-run the equivalence check
(`pipeline/anya2/README.md`, "Changing things safely") and expect a diff.

| # | Where | What happens | Effect today |
|---|---|---|---|
| 1 | `run._run_detectors` | `cfg.near.threshold or near_serve.THRESHOLD` (same for `far.threshold` and `end.live_hi`). A value of `0.0` is falsy, so it **falls back to the default** instead of meaning "accept everything". | None with default config. It only matters if someone sets a threshold of exactly 0. |
| 2 | `run.build_reel` | Writes `cfg.reel.use_near/use_far/use_end` back onto the **caller's** config object. | A config reused across runs carries those flags along. Harmless today. |
| 3 | `orchestrator._live_score` | The point-end detector honours `cfg.end.smooth_s`, but the orchestrator's own live score (used to suppress in-rally serves and to recover missed points) always uses `point_end.LIVE_SMOOTH_S` = 4.0. | Changing `smooth_s` changes the ends but not suppression or recovery. |
| 4 | `orchestrator._summary` | The reel JSON's `end_source` always contains `"next-serve": 0, "default": 0`, sources that no longer exist. The real ones are `detected`, `estimated` and `recovered`. | Cosmetic, but anything reading the JSON sees two dead keys. |
| 5 | `far_serve.detect_serves` | The far serve's `t` is the trophy-run **start** minus 0.90 s. The near serve's `t` comes from a walk-back from the trophy **peak**. | Intentional per side, but easy to miss when comparing the two. |
| 6 | `camera._register_samples` | When a frame fails against the reference, the fallback match against the last good frame uses the **approximate** (FLANN) matcher only, with no brute-force retry. | Very rarely, a frame that could have registered is filled from a neighbour instead. |
| 7 | `config.PerceiveConfig` | `near_imgsz` and `far_imgsz` exist but are **never read**: run.py does not pass them, so `perceive.NEAR_IMGSZ`/`FAR_IMGSZ` always apply. | Setting them does nothing. |
| 8 | `far_serve` | `READY_BACK_MIN_S`/`READY_BACK_MAX_S` are defined but unused (the far ready score uses `READY_MEAN_FROM_S/TO_S`). | None. |
| 9 | `far_serve` vs `near_serve` | The ready score is smoothed differently. The far side uses a plain moving mean that counts NaN as 0; the near side uses a NaN-skipping mean. | Intentional or not, it is part of the tuned behaviour. |

## Removed in the rewrite (no effect on output)

- `point_end.quiet_mask` and `parts["quiet"]`: computed on every run, never read.
- Duplicate `_runs`, `_movmax` and `_ramp` copies in near_serve and near_end, replaced by `signals.py`.
- `perceive`'s `stride * (1 if … else 1)`.

`court.side`, `court.is_far`, `court.eligible`, `Geometry.project_at` and
`Geometry.to_court_at` are unused by the pipeline but kept, and marked as such,
for outside callers.
