# full_anya.py — Unified Near + Far Serve Detection Design

## Overview

`full_anya.py` merges `run_anya.py` (near-side) and `far_anya.py` (far-side) into a single
pipeline. Both players run their individual WAITING/ARMED state machines simultaneously.
When either player triggers ACTIVE, the entire system enters a shared ACTIVE state. After
the point ends, each player resumes their own state from WAITING. A serving-side filter
suppresses spurious detections by requiring ≥ 8 consecutive serves on a side before
allowing the serving side to switch.

---

## 1. State Model

### Per-player states (independent)
| State | Description |
|-------|-------------|
| `WAITING` | Player not in ready position |
| `ARMED` | Player at baseline, still, toss being monitored |

### System-level state (shared)
| State | Description |
|-------|-------------|
| `IDLE` | Neither player is ACTIVE; both run their individual machines |
| `ACTIVE` | A serve is in progress; triggered by either player going ACTIVE |

When `system_state == ACTIVE`, the individual near/far states are frozen. `active_player`
records which side triggered the point (`"near"` or `"far"`).

When ACTIVE ends, both players reset to `WAITING` and the system returns to `IDLE`.

### State diagram

```
           [NEAR: WAITING ⟷ ARMED]
                        │
                        │ serve score ≥ threshold
                        ▼
IDLE ──────────────── ACTIVE ──────────────────▶ IDLE
                        ▲
                        │ serve score ≥ threshold
           [FAR:  WAITING ⟷ ARMED]
```

---

## 2. File Structure

```
full_anya.py
├── Constants              (merged NEAR_ / FAR_ prefixed where they differ)
├── FullTelemetryFrame     (dataclass: both near + far player data per frame)
├── FullTelemetryProvider  (sensor layer: runs near + far detection each frame)
├── NearSubEngine          (WAITING/ARMED logic copied from run_anya TransitionEngine)
├── FarSubEngine           (WAITING/ARMED logic copied from far_anya FarTransitionEngine)
├── FullTransitionEngine   (orchestrates sub-engines + shared ACTIVE state)
├── ServingSideFilter      (spurious detection suppressor)
├── render_frame()         (combined overlay)
├── _collect_full_segments()
└── run_full_anya_pipeline() / __main__
```

---

## 3. Constants

All existing near-side and far-side constants are preserved verbatim.
Where the same logical constant differs between sides, prefix it:

```python
# Near side
NEAR_ACTIVE_BALL_CONF      = 0.15
NEAR_TOSS_BALL_CONF        = 0.10
NEAR_TOSS_BALL_IMGSZ       = 320
NEAR_ACTIVE_BALL_IMGSZ     = 960
NEAR_PLAYER_PERSIST_FRAMES = 5
NEAR_PLAYER_MISSING_GRACE  = 5

# Far side  (unchanged from far_anya.py)
FAR_ACTIVE_BALL_CONF       = 0.10
FAR_TOSS_BALL_CONF         = 0.05
FAR_TOSS_BALL_IMGSZ        = 480
FAR_ACTIVE_BALL_IMGSZ      = 960
FAR_PLAYER_PERSIST_FRAMES  = 20
FAR_PLAYER_MISSING_GRACE   = 15
```

Shared constants (same value both sides) keep their existing names.

---

## 4. FullTelemetryFrame

Combines both players' data into a single per-frame dataclass. Fields that already exist in
either `AnyaTelemetryFrame` or `FarTelemetryFrame` are kept as-is, just merged:

```python
@dataclass
class FullTelemetryFrame:
    frame_id:   int
    timestamp:  float
    system_state: str          # "IDLE" or "ACTIVE"
    active_player: str         # "near", "far", or ""

    # Near player
    near_player_box:        Optional[Tuple[int,int,int,int]] = None
    near_player_world:      Optional[Tuple[float,float]]     = None
    near_state:             str                              = "WAITING"
    near_trophy_score:      float                            = 0.0
    near_toss_ball_candidates: List[dict]                    = None
    near_z_box:             Optional[Tuple[int,int,int,int]] = None

    # Far player
    far_player_box:         Optional[Tuple[int,int,int,int]] = None
    far_player_world:       Optional[Tuple[float,float]]     = None
    far_state:              str                              = "WAITING"
    far_trophy_score:       float                            = 0.0
    far_toss_ball_candidates: List[dict]                     = None
    far_z_box:              Optional[Tuple[int,int,int,int]] = None
    far_mhi_toss_score:     float                            = 0.0

    # Shared ACTIVE state
    active_ball_candidates: List[dict]                       = None
```

---

## 5. FullTelemetryProvider

Runs both near-side and far-side sensor work each frame. Internally it merges the logic
from `AnyaTelemetryProvider` (in `anya_base.py`) and `FarTelemetryProvider` (in
`far_anya.py`) without modification to either.

### Initialisation

```python
class FullTelemetryProvider:
    def __init__(self, video_path: str):
        # Shared
        self.court_vertices, self.frame_shape = init_court(...)
        self.H, self.H_inv = ...
        self.ball_model = YOLO(...)

        # Near-specific
        self.near_player_model  = YOLO("yolo26n.pt")
        self.near_trophy_model  = YOLO(Config.DEFAULT_NEAR_TROPHY_MODEL_PATH)
        self.near_player_roi    = init_near_player_roi(...)
        self._near_box_smoother = BoxSmoother(alpha_pos=0.30, alpha_size=0.12)

        # Far-specific
        self.far_player_model   = YOLO("yolo26n.pt")   # same weights, separate instance
        self.far_trophy_model   = YOLO(Config.DEFAULT_FAR_TROPHY_MODEL_PATH)
        self.far_player_roi     = init_far_player_roi(...)
        self._far_box_smoother  = BoxSmoother(alpha_pos=0.50, alpha_size=0.12)
        self.net_y_px           = self._compute_net_y_px()
        self._far_box_heights   = deque(maxlen=30)

        # Exclusion zones (shared ball model, computed once)
        self.static_exclusion_zones = create_auto_exclusion_zones(...)
        self.dynamic_exclusion_zones = []
```

### get_frame_telemetry(frame, system_state, active_player)

```
1. Run _track_near_players(frame)   → near_box, near_world
2. Run _track_far_players(frame)    → far_box, far_world
3. If system_state == "IDLE":
     - If near_state == "ARMED": run near toss ball detection (near ROI, NEAR_TOSS_BALL_CONF)
     - If far_state == "ARMED":  run far toss ball detection  (far ROI,  FAR_TOSS_BALL_CONF)
     - Run trophy detection for each ARMED player
     - Compute far MHI toss score
4. If system_state == "ACTIVE":
     - Run full-frame ball detection (conf = NEAR or FAR threshold based on active_player)
     - Filter exclusion zones and both player boxes
5. Return FullTelemetryFrame
```

---

## 6. Sub-engines

### NearSubEngine

Extracted verbatim from the `TransitionEngine` class in `run_anya.py`, covering only the
WAITING and ARMED logic (i.e. everything in `_check_waiting` and `_check_armed`).

Interface:
```python
class NearSubEngine:
    def tick(self, history: deque) -> str:
        """Returns 'WAITING', 'ARMED', or 'TRIGGER_ACTIVE'."""
```

`TRIGGER_ACTIVE` is returned when the existing `serve_score >= TRANSITION_SCORE_THRESHOLD`
condition is met (replacing the current `return "ACTIVE"`).

### FarSubEngine

Extracted verbatim from `FarTransitionEngine` in `far_anya.py`, again covering only
WAITING and ARMED.

Same interface — returns `TRIGGER_ACTIVE` instead of `"ACTIVE"`.

---

## 7. FullTransitionEngine

Orchestrates both sub-engines and the shared ACTIVE state.

```python
class FullTransitionEngine:
    def __init__(self, fps, near_ready_zone, far_baseline_strip):
        self.near_engine  = NearSubEngine(fps, near_ready_zone)
        self.far_engine   = FarSubEngine(fps, far_baseline_strip)
        self.system_state = "IDLE"
        self.active_player = ""
        self._active_engine = ActiveEngine(fps)  # existing ACTIVE logic

    def evaluate_transitions(self, history: deque) -> Tuple[str, str]:
        """Returns (system_state, active_player)."""
```

### evaluate_transitions logic

```
If system_state == "IDLE":
    near_result = near_engine.tick(history)
    far_result  = far_engine.tick(history)

    if near_result == "TRIGGER_ACTIVE":
        system_state  = "ACTIVE"
        active_player = "near"
        _active_engine.init(now)
        near_engine.reset()
        far_engine.reset()

    elif far_result == "TRIGGER_ACTIVE":
        system_state  = "ACTIVE"
        active_player = "far"
        _active_engine.init(now)
        near_engine.reset()
        far_engine.reset()

If system_state == "ACTIVE":
    result = _active_engine.tick(history, active_player)
    if result == "END_ACTIVE":
        system_state  = "IDLE"
        active_player = ""
        near_engine.reset_to_waiting()
        far_engine.reset_to_waiting()
```

### ActiveEngine

Extracted verbatim from the ACTIVE logic in `run_anya.py` (`_check_active`,
`_compute_energy_delta`, `_update_player_tracking`, etc.).

The only change: `_update_player_tracking` receives `active_player` to know whether to
read `near_player_box/world` or `far_player_box/world` from the telemetry frame.

For the energy bar, the **opponent** is always tracked:
- active_player = "near" → energy bar tracks far player (opponent)
- active_player = "far"  → energy bar tracks near player (opponent)

This is consistent with the existing far_anya.py behaviour where the near player drives
the energy bar.

---

## 8. ServingSideFilter

Suppresses spurious detections using the rule: a side must accumulate ≥ 8 consecutive
serves before a switch to the other side is considered confirmed.

```python
class ServingSideFilter:
    MIN_SERVES_TO_CONFIRM = 8

    def __init__(self):
        self.confirmed_side   = None   # "near" | "far" | None
        self.current_streak   = []     # list of "near"/"far" from most recent detections
        self.pending_side     = None   # candidate new side accumulating evidence
        self.pending_count    = 0

    def record(self, side: str) -> bool:
        """
        Record a serve detection on `side`. Returns True if this detection
        should be kept, False if it should be suppressed as spurious.
        """
```

### Filter logic

```
Case 1 — No confirmed side yet:
    Append to current_streak.
    If len(current_streak) >= MIN_SERVES_TO_CONFIRM and all same side:
        confirmed_side = side
    Return True (allow all until first confirmation)

Case 2 — Detection matches confirmed_side:
    Append to current_streak. Reset pending state.
    Return True

Case 3 — Detection is on the opposite side:
    If pending_side == side:
        pending_count += 1
    Else:
        pending_side  = side
        pending_count = 1

    If pending_count >= MIN_SERVES_TO_CONFIRM:
        confirmed_side = side   ← switch confirmed
        current_streak = [side] * pending_count
        pending_side  = None
        pending_count = 0
        Return True
    Else:
        Return False   ← suppress as spurious
```

Segments kept by the filter are emitted; suppressed segments are logged but not written
to the output CSV or highlights reel.

---

## 9. _collect_full_segments()

Replaces both `_collect_far_segments()` and `_collect_near_segments()`.

```python
def _collect_full_segments(video_path, provider, engine, side_filter, ...):
    for frame in video:
        tel = provider.get_frame_telemetry(frame, engine.system_state, engine.active_player)
        system_state, active_player = engine.evaluate_transitions(provider.telemetry_history)

        if a new ACTIVE segment just ended:
            keep = side_filter.record(active_player)
            if keep:
                emit segment

        render_frame(...)
```

---

## 10. Output

The output CSV and highlights reel gain a `side` column (`"near"` or `"far"`):

```csv
serve_number,timestamp,side,duration,...
1,3.92,far,2.1,...
2,12.4,far,1.9,...
...
```

---

## 11. What does NOT change

- All transition thresholds (serve score, energy, movement) stay exactly as-is in each sub-engine.
- The WAITING↔ARMED logic for each player is untouched.
- The ACTIVE ball-trace / energy-bar logic is untouched.
- The parabolic toss arc, MHI fallback, net-occlusion correction — all unchanged.
- CLI arguments are the same, plus an optional `--side {near,far,both}` flag to restrict output.

---

## 12. Open questions before implementation

1. **Simultaneous triggers**: If near and far both satisfy the serve threshold in the same
   frame-window, which takes priority? Proposal: first-to-trigger wins (near checked before
   far in the tick order). Is this acceptable?

2. **Side-filter bootstrap**: Before 8 detections are accumulated, all detections are emitted
   (no filter yet). Acceptable?

3. **Active player ROI for toss/trophy in ARMED**: Currently both players run ARMED detection
   simultaneously. Should we detect toss for both at the same time, or only for the player
   whose sub-engine is ARMED?

4. **Highlights reel**: Single output video with both sides interleaved in timestamp order, or
   separate near/far output files?
