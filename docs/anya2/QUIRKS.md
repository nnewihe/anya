# anya2 — quirks found in the readability rewrite, and what happened to each

The October 2026 readability rewrite had to produce byte-identical output, so it
kept nine odd behaviours as they were. They were then fixed on the branch
`claude/anya2-quirks`, one commit per group, so any single one can be reverted.

| # | Where | Was | Now | Changes results? |
|---|---|---|---|---|
| 1 | `run._run_detectors` | `threshold or DEFAULT`: a threshold of 0.0 silently meant "default" | Only `None` means default (`_or_default`) | No, at default settings |
| 2 | `run.build_reel` | Wrote `use_near/use_far/use_end` into the caller's config | Works on a copy (`_reel_config`) | No |
| 3 | `orchestrator._live_score` | Always used 4.0 s smoothing, whatever `cfg.end.smooth_s` said | Uses the same smoothing as the point-end detector (`ReelConfig.live_smooth_s`) | No, at default settings |
| 4 | `orchestrator._summary` | `end_source` reported two dead sources (`next-serve`, `default`) | Counts `detected` / `estimated` / `recovered` | Reel JSON summary only; segments identical |
| 5 | `near_serve.serve_onset` | Point start = last "hands together" sample − 1.63 s | Point start = trophy peak − 1.75 s (ported from the unmerged 2026-09-05 work; evidence in NOTES) | **Yes**: near serves retimed ~0.2–0.5 s later; reel pre-roll raised 1.0 → 1.5 s to keep whole points |
| 6 | `camera._register_via_last_good` | Fallback match used only the approximate matcher | Retries with brute force, like the main path | **Possibly**, on frames where approximate matching fails |
| 7 | `config.PerceiveConfig` | `near_imgsz` / `far_imgsz` were never read | Passed to the pose passes (None = 640 / 960) | No, at default settings |
| 8 | `far_serve` | `READY_BACK_MIN_S/MAX_S` defined but unused | Deleted | No |
| 9 | `far_serve.serve_features` | Ready smoothing counted a missing sample as 0 | NaN-skipping mean, as on the near side | **Yes**: far scores move up to ~0.16 |

## Verification status

- **#1–#4, #7, #8:** verified on Data/21, 25, 35, 40. The segments are
  identical; the only change is the `end_source` summary.
- **#5, #6, #9:** scored against `main` over the 13 trusted clips
  (`pipeline.anya2.eval` + `orchestrator.score_reel`):

  | Arm | Near R / P | Far R / P | Reel whole / partial / missing | Reel s | Dead s |
  |---|---|---|---|---|---|
  | main, pre-roll 1.0 | 86.0 / 74.8 | 75.8 / 62.6 | 132 / 89 / 14 | 3472.6 | 1129.8 |
  | #5+#9, pre-roll 1.0 | 88.8 / 76.6 | 78.1 / 61.3 | 130 / 92 / 13 | 3427.3 | 1092.5 |
  | main, pre-roll 1.5 | 86.0 / 74.8 | 75.8 / 62.6 | 138 / 84 / 13 | 3609.8 | 1235.3 |
  | **#5+#9, pre-roll 1.5** (this branch) | 88.8 / 76.6 | 78.1 / 61.3 | 137 / 85 / 13 | 3577.6 | 1211.0 |

  #5 starts near serves later, which cost 3 whole points at the old 1.0 s
  pre-roll; 1.5 s wins them back (+7 over #5 at 1.0 s) for ~150 s more reel.
  #6 changed nothing on Data/21, 58 or 78 (every frame registered first time).
  Point ends are unchanged everywhere.
- **#9** may still want `far_serve.THRESHOLD` re-swept, since 0.75 was tuned
  with the old smoothing.

## Removed in the rewrite itself (no effect on output)

- `point_end.quiet_mask` and `parts["quiet"]`: computed on every run, never read.
- Duplicate `_runs`, `_movmax` and `_ramp` copies in near_serve and near_end, replaced by `signals.py`.
- `perceive`'s `stride * (1 if … else 1)`.

`court.side`, `court.is_far`, `court.eligible`, `Geometry.project_at` and
`Geometry.to_court_at` are unused by the pipeline but kept for outside callers.
