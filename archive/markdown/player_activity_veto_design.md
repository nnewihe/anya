# Player-Activity Veto for ACTIVE → WAITING (Rule A)

## Problem

`Rule A (Ball Missing)` fires when `now - last_ball_seen_time > 2.5s`. The near player
frequently occludes the ball from the camera during a live rally, causing premature
ACTIVE → WAITING transitions.

## Scope

**Only Rule A is vetoable.** Rules B.i (All Stationary), B.ii (Near Player), and C (In Net)
are based on positive ball evidence and must remain authoritative.

---

## Feature Extraction (rolling 1.0s window)

Compute over the last ~1.0s of `telemetry_history` (roughly `fps` frames):

1. **Near-player world-space speed** — mean of frame-to-frame `hypot(Δwx, Δwy) / Δt`,
   after a light EMA smooth on `(wx, wy)`. Source: `near_player_world` in `TelemetryFrame`.

2. **Near-player aspect-ratio variance** — `w/h` per frame from `near_player_box`,
   compute variance and a "periodicity score" via autocorrelation at lag ≈ 0.5s
   (the half-period of a ~2 Hz walking stride).

3. **Far-player displacement** — total world-space path length of far player's feet
   centroid over the window. Source: `far_player_box` → homography on feet centroid.

4. **Last-ball-toward-player flag** — was the last observed `active_ball_candidate`
   trajectory heading into the near-player bounding box? Compute once at the moment
   `last_ball_seen_time` is updated (from last 2–3 candidate positions) and cache it.

---

## Veto Predicate

```python
veto = (
    near_speed_fts > SPEED_HI                          # e.g. 6 ft/s — unambiguous sprint
    or (near_speed_fts > SPEED_LO                      # e.g. 3 ft/s
        and ar_variance > AR_VAR_THRESH                # erratic body shape (lunges, turns)
        and not ar_is_periodic)                        # not a clean walk cycle
    or far_player_displacement_ft > FAR_DISP_THRESH    # e.g. 4 ft in last 1s
    or last_ball_toward_player                         # occlusion prior
)
```

The compound second clause separates "walking between points" (periodic, moderate speed,
steady AR) from "scrambling during a rally" (non-periodic, moderate-to-high speed, erratic AR).

---

## How the Veto Modifies Rule A

### Option 1 — Grace Extension (preferred)

Do not touch `last_ball_seen_time`. Make the timeout threshold variable:

```python
effective_timeout = BALL_TIMEOUT + (VETO_EXTENSION if veto else 0.0)

if (now - self.last_ball_seen_time) > effective_timeout:
    # fire Rule A
```

With `VETO_EXTENSION ≈ 1.5–2.0s`. This guarantees the point eventually times out even
if the player keeps moving — preventing a stuck-ACTIVE state.

### Option 2 — Timer Reset (what the user originally proposed)

Set `last_ball_seen_time = now` each frame the veto holds. Simpler, but unbounded.
An always-active player prevents Rule A from ever firing.

**If using Option 2, add a hard cap**: "veto can extend ACTIVE by at most 3 seconds total"
tracked via `self._veto_active_since`.

**Option 1 is recommended.** Option 2 is viable only with the hard cap.

---

## Thresholds (starting points — tune on labeled data)

| Symbol | Value | Meaning |
|---|---|---|
| `SPEED_HI` | 6.0 ft/s | Unambiguously scrambling |
| `SPEED_LO` | 3.0 ft/s | Possibly moving |
| `AR_VAR_THRESH` | 0.04 | Variance of (w/h) — tune empirically |
| `AR_PERIOD_LAG` | ≈ 0.5 s | Walking stride half-period |
| `AR_PERIOD_CORR` | 0.6 | Autocorrelation above this ⇒ "periodic" (walking) |
| `FAR_DISP_THRESH` | 4.0 ft | Far player active in the window |
| `VETO_EXTENSION` | 1.5 s | Grace added to BALL_TIMEOUT (Option 1) |
| `VETO_MAX_HOLD` | 3.0 s | Hard cap on veto duration (Option 2) |

---

## New State to Add to `TransitionEngine`

```python
self._near_pos_history: deque  # (t, wx, wy)  — pruned to ~1.5s
self._near_box_history: deque  # (t, w, h)    — for AR variance/periodicity
self._far_pos_history:  deque  # (t, wx, wy)  — from far-box feet centroid
self._last_ball_toward_player: bool  # set when last_ball_seen_time is updated
self._veto_active_since: Optional[float]  # for Option 2 hard cap
```

---

## Where It Plugs In (`_check_active`)

In `anya_transitions.py`, inside `_check_active`, right before Rule A:

```python
# Update rolling histories
update_near_pos_history(frame, now)
update_near_box_history(frame, now)
update_far_pos_history(frame, now)

veto = compute_player_activity_veto(now)

# Rule A: Missing (with optional veto)
effective_timeout = self.BALL_TIMEOUT + (VETO_EXTENSION if veto else 0.0)
if (now - self.last_ball_seen_time) > effective_timeout:
    self.last_transition_time = self.last_ball_seen_time
    return self._perform_transition(now, "Ball Missing")
```

Also: update `last_ball_seen_time` logic to set `_last_ball_toward_player` when
`active_ball_candidates` is non-empty (check trajectory direction vs. near-player centroid).

---

## Debug / Observability

Push all veto features into `self.last_active_debug` so `render_active_debug` can display them:

```python
self.last_active_debug["veto_active"]        = veto
self.last_active_debug["near_speed_fts"]     = near_speed_fts
self.last_active_debug["ar_variance"]        = ar_variance
self.last_active_debug["ar_is_periodic"]     = ar_is_periodic
self.last_active_debug["far_disp_ft"]        = far_player_displacement_ft
self.last_active_debug["ball_toward_player"] = last_ball_toward_player
```

Add a corresponding row in `render_active_debug` and in the CSV columns in `run_anya.py`.

---

## Known Failure Modes

| Scenario | Risk | Mitigation |
|---|---|---|
| Ball goes into net, player walks to fetch it | Walking is periodic → veto doesn't fire. Safe. | — |
| Point ends on a winner, player fist-pumps / turns | AR variance spikes, veto may fire | Option 1 hard timeout still ends it within `VETO_EXTENSION` seconds |
| Player stationary at baseline, ball occluded | Speed low, AR stable → motion-veto doesn't fire | Relies on `last_ball_toward_player` or far-player motion |
| Camera jitter on near-player box | AR variance noise | Smooth box with `BoxSmoother` before computing AR |

---

## Validation Plan

1. Log all veto features to the CSV for existing test clips.
2. Manually label 20–30 ACTIVE → WAITING transitions as "correct" or "premature due to occlusion".
3. Verify veto would flip the premature-occlusion cases without creating new false negatives on real point-ends.
4. Tune `SPEED_LO`, `AR_VAR_THRESH`, `VETO_EXTENSION` on that labeled set.

---

## Existing Code Issues Noted (unrelated to veto)

- `_stub_init_court` at `anya_base.py:158` is unreferenced.
- `render_debug_panel` reads `window_detections` and `any_ball_near_player` from `debug`
  (`run_anya.py:121–123`) but neither key is set in `last_active_debug` — CSV always 0/False.
- `_post_active_next_state` is defined but never called; `_perform_transition` always
  returns `"WAITING"` even though the bypass-to-ARMED logic exists.
- `last_transition_time` is set in `_check_active` but `run_anya.py` never reads it,
  so the "rewind output-video writing" behavior (described in `anya_transitions.py:13–17`)
  is not implemented.
