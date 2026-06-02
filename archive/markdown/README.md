# Anya Pipeline — Version 0

Tennis rally detection system. Watches a match video, detects when each point starts (serve) and ends (ball dead / in net), and exports a highlight reel of active rallies.

## Architecture

The pipeline is split into three layers:

```
run_anya.py          — main loop, video I/O, debug visualisation
anya_base.py         — telemetry provider (YOLO inference, homography, exclusion zones)
anya_transitions.py  — state machine (WAITING → ARMED → ACTIVE → WAITING)
utilities.py         — shared config constants, court helpers
```

### State Machine

| State | Description |
|---|---|
| **WAITING** | Watching for the near player to take position at the baseline |
| **ARMED** | Player is at baseline; running trophy pose + ball toss detection |
| **ACTIVE** | Point is live; tracking ball trace and player energy |

### ACTIVE → WAITING Transition (Hybrid Model)

Point death uses a two-stage model:

1. **Ball Trace Gate** (primary): while a moving ball is detected inside the active zone, the point stays alive unconditionally. A detection is only counted as a "trace" if inter-frame speed is above `BALL_DEAD_SPEED_FTS` (3 ft/s) and below the physical maximum `MAX_BALL_SPEED_FTS` (176 ft/s).

2. **Energy Bar** (fallback): the instant no active trace is present, an energy bar starts at 1.0 and drains/boosts each frame based on near-player telemetry:
   - Sprinting → boost (+4.0/s)
   - Swing / split-step (normalised shape change > 35% of box height) → boost (+4.0/s)
   - Walking gait (oscillatory y-movement) → drain (−0.4/s)
   - Player missing from frame → drain (−0.4/s)
   - Standing still → drain (−0.25/s)
   - Moving (neutral) → small boost (+0.1/s)
   
   Point dies when energy reaches 0. If a ball trace reappears, the energy bar is discarded and reset to 1.0.

3. **Net Kill**: if any ball detection lands within ±2.5 ft of the net cord (y = 39 ft), the point is killed immediately — no energy bar.

## Key Configuration

All tuning knobs live in `utilities.py` (`Config` class) and `anya_transitions.py` (`TransitionEngine.__init__`):

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `ACTIVE_BALL_CONF` | `utilities.py` | 0.20 | YOLO confidence for in-rally ball detection |
| `TOSS_BALL_CONF` | `utilities.py` | 0.10 | YOLO confidence for serve toss detection |
| `BALL_TIMEOUT` | `anya_transitions.py` | 2.5s | DBSCAN history window for stationary cluster detection |
| `BALL_DEAD_SPEED_FTS` | `anya_transitions.py` | 3 ft/s | Min speed for a detection to count as an active trace |
| `MAX_BALL_SPEED_FTS` | `anya_transitions.py` | 176 ft/s | Physical upper bound — faster detections are discarded |
| `NET_Y_FT` | `anya_transitions.py` | 39 ft | Net cord world-space position |
| `NET_ZONE_HALF_WIDTH_FT` | `anya_transitions.py` | 2.5 ft | Half-width of net kill zone |
| `ENERGY_BOOST_SPRINT` | `anya_transitions.py` | 4.0/s | Energy boost rate while sprinting |
| `ENERGY_DECAY_STILL` | `anya_transitions.py` | 0.25/s | Energy drain rate while standing still |
| `ENERGY_DECAY_MISSING` | `anya_transitions.py` | 0.4/s | Energy drain rate when player is off-frame |
| `TRANSITION_SCORE_THRESHOLD` | `anya_transitions.py` | 0.55 | Minimum serve score to fire ARMED → ACTIVE |

## Running

```bash
python -m src.ai.run_anya path/to/video.mp4 --output highlights.mp4
```

Options:
- `--headless` — skip live preview windows (faster)
- `--start-frame N` — begin processing at frame N

Outputs:
- `highlights.mp4` — original-resolution video of active rallies only
- `highlights_telemetry.csv` — per-frame state log

## First-run Setup

On first run for a new video, two interactive windows will appear:

1. **Court corners** — click the 4 court corners (BL, BR, TR, TL). Results cached to `<video>_court_cache.json`.
2. **Active zone polygon** — click 8 points defining the ball-valid region. Results cached to `active_zone_config.json`.

## Model Weights

| Model | Path | Purpose |
|---|---|---|
| Player detection | `yolo26n.pt` | Locate near/far players each frame |
| Ball detection | `weights/ball/weights/best.pt` | Detect ball in ARMED (toss) and ACTIVE (rally) states |
| Trophy pose | `weights/trophy_pose_cls2/weights/best.pt` | Classify serve wind-up pose in ARMED state |
