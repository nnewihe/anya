# Anya Vision Core — Tennis Serve Detection & Rally Extraction

## Overview

**anya_vision_core.py** is a unified vision engine for detecting tennis serves and extracting rally segments from match videos. It implements a **three-state finite state machine** (WAITING → ARMED → ACTIVE) with energy-based event tracking, real-time pose/toss detection, and automated highlight video generation.

**Stage 1** of the two-stage Anya pipeline. Produces:
- `{video}_telemetry.csv` — Smoothed near-player bounding boxes (frame-by-frame)
- `{video}_serve_events.json` — Serve timestamps and frame indices
- `{video}_highlights.mp4` — Concatenated active rally segments

---

## System Architecture

### Three-State Finite State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                        STATE MACHINE                             │
└─────────────────────────────────────────────────────────────────┘

    ┌─────────┐
    │ WAITING │  Player at baseline, no serve in progress
    └────┬────┘
         │ Player holds "ready" position ≥ READY_WAIT_TIME_SEC (0.4s)
         │ AND within ±0.5 to ±3.5 ft from baseline
         ↓
    ┌─────────┐
    │  ARMED  │  Player ready, watching for toss & pose
    └────┬────┘  
         │
         ├─ EXITS TO WAITING if out-of-band > 25% over 2-second window
         │
         └─ ARMED → ACTIVE on serve detection (TRANSITION_SCORE_THRESHOLD ≥ 0.55)
             │       Serve score = 0.2×max_trophy + 0.8×max_toss
             │       REQUIRES: ball above head + trophy pose
             ↓
         ┌──────────┐
         │ ACTIVE   │  Rally in progress; energy-driven state
         └────┬─────┘
              │
              ├─ Ball energy increases if: v_ball > 15 ft/s (flying)
              ├─ Ball energy decreases if: occluded, dead, rolling
              │
              ├─ Player energy increases if: sprinting (v > 6 ft/s), active swing
              ├─ Player energy decreases if: walking, stationary, off-screen
              │
              ├─ Energy scaled by net proximity (3× boost at net for player)
              │
              └─ EXITS TO WAITING when:
                 • Energy → 0 (rally momentum lost)
                 • Ball missing > 20s (ACTIVE) or 6s (IDLE)
                 • Emergency override: if energy < 0.5 AND trophy pose > 0.6
```

---

## Key Data Structures

### `SystemState` (Enum)
```python
WAITING  → Awaiting player to ready position
ARMED    → Pose & toss detection mode
ACTIVE   → Rally tracking with energy decay
```

### `BoxSmoother` (Exponential Weighted Moving Average)
Smooths bounding box jitter across **position** and **size** independently:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `alpha_pos` | 0.35 | Weight for center-point EWA (responsive) |
| `alpha_size` | 0.12 | Weight for width/height EWA (stable) |
| `still_thresh` | 4.0 px | Player velocity threshold for size suppression |

**Key insight**: When player velocity < 4 px/frame, multiply `alpha_size` by 0.3 to suppress box jitter from tracking noise. Critical for preventing false "action" signals when player is stationary.

```python
# Pseudocode
velocity = hypot(new_cx - old_cx, new_cy - old_cy)
eff_alpha = 0.12 * 0.3 if velocity < 4.0 else 0.12
smoothed_w = (1 - eff_alpha) * old_w + eff_alpha * new_w
```

### `SideBuffer`
Maintains rolling buffers of detection scores (with timestamps) for one side:
- `trophy_scores` → Pose classification (ready position)
- `toss_scores` → Ball toss detection
- Auto-cleanup: discards detections older than `EVENT_WINDOW_SECONDS` (1.2s)

---

## State Behaviors & Algorithms

### ⏳ WAITING STATE

**Purpose**: Detect when the near-side player moves to baseline ready position.

**Detection Logic**:
1. Track near-side player (closest to baseline)
2. Convert pixel center (cx, y_feet) → world coordinates via homography
3. Check if player is **behind baseline** (world_y < 0) AND within **±0.5–3.5 ft**
4. If in zone for ≥ 0.4 seconds → transition to ARMED

**Assumptions**:
- **Court homography** is pre-calibrated via 4-point corner selection
- **Near-side** = closer to baseline (world_y-axis origin)
- **Ready distance** (-0.5 to -3.5 ft) = natural serve stance zone

---

### 🎯 ARMED STATE

**Purpose**: Detect serve initiation via ball toss + pose.

**Concurrent Detection Streams**:

#### 1. **Trophy Pose Detection**
- Crop player bounding box ± 30% padding
- Run pose classification model on crop
- Track max confidence over 1.2-second window
- **Class 1** = ready/serving position

#### 2. **Ball Toss Detection** (Consecutive-Frame Hysteresis)

Toss ROI region: `2 × player_width` horizontal, extends `player_height` above and below player head.

**State Machine** within ARMED:
```python
if best_ball_detected_in_toss_roi:
    dy = ball_y_current - ball_y_previous
    if dy < 0 and (time_delta > 0):  # Upward motion
        toss_gap_frames = 0
        toss_consecutive_frames += 1
        toss_ball_above_head_detected = True
    else:
        toss_gap_frames += 1
        if toss_gap_frames > 3:
            toss_consecutive_frames = 0
            toss_ball_above_head_detected = False

# Assign score
if toss_ball_above_head_detected:
    if toss_consecutive_frames >= 3:    score = 1.0
    elif toss_consecutive_frames >= 2:  score = 0.7
    else:                                 score = 0.0
else:
    score = 0.0
```

**Rationale**: Hysteresis (3-frame grace) suppresses bouncing ball false positives; requires 2–3 consecutive frames with upward motion AND detection of ball above head at some point.

#### 3. **Serve Transition Logic**
```python
max_trophy = max(trophy_scores over 1.2s window) or 0.0
max_toss   = max(toss_scores over 1.2s window) or 0.0
serve_score = 0.2 × max_trophy + 0.8 × max_toss

# Height validation (NEW)
if serve_score >= 0.55:
    if toss_min_y_px < player_box_top:  # Ball went above head
        TRANSITION: ARMED → ACTIVE
    else:
        RESET: reject toss attempt
```

**Assumptions**:
- Toss ball is detected in ROI → ball is above head (by ROI definition)
- Upward motion (dy < 0) = toss (not downward catch)
- 80% weight on toss, 20% on pose (toss is more reliable serve indicator)

#### 4. **ARMED Exit Conditions**
Player leaves ready zone for > 25% of 2-second window → ARMED → WAITING:
```python
# Track in_band history over ARMED_BAND_WINDOW_SEC (2.0s)
time_out_of_band = sum(dt for each frame where not in_band)
total_time = armed_band_history[-1].time - armed_band_history[0].time
out_ratio = time_out_of_band / total_time

if out_ratio > 0.25:  # 25% threshold
    TRANSITION: ARMED → WAITING
```

---

### ⚡ ACTIVE STATE — Energy-Driven Rally Tracking

**Core Concept**: Point energy starts at 1.0 and decays based on ball/player dynamics. Rally ends when energy → 0 or ball missing > timeout.

#### **Energy Computation**

```python
point_energy = clamp(
    point_energy + ball_delta + player_delta,
    min=0.0, max=1.0
)
```

Each frame (dt = 1/fps):

##### Ball Energy Delta
| Condition | Delta | Notes |
|-----------|-------|-------|
| Ball detected, v > 15 ft/s | +1000 × dt | "FLYING" — fast active play |
| Ball detected, v ≤ 15 ft/s | −0.3 × dt | "ROLLING" — dying play |
| Ball occluded, player in court | −0.03 × dt | "OCCLUDED (court)" — low decay |
| Ball occluded, player at baseline | −0.15 × dt | "OCCLUDED (baseline)" — moderate decay |
| Ball missing > 0.25s, neither above | −0.15 × dt | "LIKELY DEAD" — rally ending |

##### Player Energy Delta
| Condition | Delta | Notes |
|-----------|-------|-------|
| Off-screen | −0.5 × dt | Missing player killer |
| Sprinting (v > 6 ft/s) | +1000 × dt | High-speed movement |
| Active swing (Δshape > 15 px) | +1000 × dt | Large box change |
| Walking gait detected | −0.4 × dt | Rhythmic oscillation (see Gait Detector) |
| Slow/stationary (v < 2 ft/s) | −0.2 × dt | Low momentum |

**Shape change**: `|Δwidth| + |Δheight|` over last 5 detections.

#### **Net Proximity Scaling** (Depth-Dependent Boost)

As near-side player moves from baseline (0 ft) toward net (39 ft), priorities shift:

```python
net_proximity_factor = clamp(player_world_y / 39.0, 0, 1)

player_scale = 1.0 + net_proximity_factor × (3.0 - 1.0)  # 1.0–3.0×
ball_scale   = 1.0 - net_proximity_factor × (1.0 - 0)    # 1.0–0×

# At net: player energy 3× more valuable, ball energy ignored
player_delta *= player_scale
ball_delta   *= ball_scale
```

**Assumption**: Player action at net is more immediately game-relevant; ball often occluded by net frame.

#### **Gait Detection Algorithm**

Detects characteristic **walking rhythm** (oscillating up-and-down pattern):

1. Buffer last 45 frames of player feet y-position
2. Fit linear trend: `y_expected = y0 + (y45 - y0) × (frame / 45)`
3. Compute residuals: `residual[i] = actual[i] - expected[i]`
4. Count **sign reversals** (direction changes in residual):

```python
reversals = 0
for i in range(1, len(residuals)):
    if sign(residuals[i] - residuals[i-1]) changes:
        reversals += 1

# Valid gait: 2–8 reversals (typical walking cadence)
# Excludes: stationary (0 reversals), chaotic movement (9+ reversals)
```

Requires **minimum drift** of 10 px over 45 frames (excludes stationary players).

**Energy penalty**: −0.4 × dt (player walking ≠ rallying).

#### **Ball Velocity Computation** (Median-Filtered)

```python
# Stabilize velocity using median over 5 most recent frame-pairs
pairwise_velocities = []
for i in range(n-5, n):
    dist = hypot(pos[i].x - pos[i-1].x, pos[i].y - pos[i-1].y)
    dt   = pos[i].t - pos[i-1].t
    pairwise_velocities.append(dist / dt)

median_velocity = median(pairwise_velocities)
```

Prevents single-frame jitter from affecting serve/rally decisions.

#### **Ball Jump Validation**

Rejects physically impossible ball trajectories:
```python
dist = hypot(new_pos - old_pos)
dt   = time_delta
velocity = dist / dt

if velocity > MAX_BALL_SPEED_FT_SEC (180 ft/s):
    REJECT ball detection (too fast)
```

#### **Dead Ball Snapshotting**

On ARMED → ACTIVE transition, snapshot all visible balls (likely dead):
```python
self.dead_ball_refs = [(world_x, world_y) for all detected balls]
```

During ACTIVE, ball selection scoring **boosts** detections far from dead refs (avoids re-detecting stationary balls).

#### **ACTIVE Exit Conditions**

Rally ends and transitions to WAITING or ARMED bypass:

```python
if point_energy <= 0.0 or (ball_missing > timeout):
    if near_player_at_baseline AND trophy_pose > 0.6:
        EMERGENCY OVERRIDE: ACTIVE → ARMED (fast re-serve)
    else:
        ACTIVE → WAITING

# Log active segment: (segment_start, current_frame)
# Store for highlight video export
```

**Absolute timeouts**:
- Active player in court/sprinting: 20 seconds max
- Idle player at baseline: 6 seconds max

---

## Court Geometry & Coordinate Systems

### **Interactive Court Calibration**

```
User selects 4 corners in any order:
  ↓
Cached to JSON: {video_name}_court_cache.json
Cached layout: { points, frame_shape, analysis_size, video_name }
  ↓
Reloaded on subsequent runs (auto-invalidate if analysis_size changes)
```

### **Homography Transform**

4-point perspective transform maps **pixel space** → **world tennis court** (feet):

```python
src_pts = [BL, BR, TR, TL]  # Pixel corners (user-selected)
dst_pts = [
    [0,    0],              # Bottom-left (baseline, sideline)
    [27,   0],              # Bottom-right (baseline + width)
    [27,  78],              # Top-right (net + length)
    [0,   78],              # Top-left (net)
]
H, _ = cv2.findHomography(src_pts, dst_pts)
world_x, world_y = perspectiveTransform(pixel_x, pixel_y, H)
```

**Court dimensions** (singles):
- Width: 27 ft
- Depth (baseline to net): 78 ft

**Assumptions**:
- Perspective camera (not orthographic)
- Court fully visible in reference frame (frame 300 or closest)
- All four corners clearly identifiable

---

## Detection Pipelines

### **Player Tracking**

```python
# YOLO person detector (yolo26n.pt)
player_model(frame, conf=0.5, imgsz=480)

# Select nearest player to baseline (minimum world_y in "near" half)
for each detected person:
    world_x, world_y = homography_transform(cx, y_feet)
    if x in court bounds:
        candidates.append((world_x, world_y, box))

near_player = argmin(candidates, key=world_y)
```

**Exclusion**: Players outside horizontal court bounds ± 15 ft.

### **Ball Detection**

```python
# YOLO ball detector (best.pt)
ball_model(frame, conf=0.05, imgsz=960)

# Filter & rank
for each detected ball:
    reject if:
        • size > 20 px (too large → false positive)
        • inside player bounding box ± padding (hand/racket)
        • in exclusion zone (ball baskets, backgrounds)
        • outside court bounds ± 15 ft

    score based on:
        • proximity to last known position (continuity)
        • confidence level
        • distance from dead ball snapshots
        • in/out of court (bonus for in-court)

select = argmax(score)
```

**Exclusion zones** auto-generated:
- Sample 20 random frames
- DBSCAN cluster low-confidence detections (eps=30, min_samples=3)
- Exclude static clusters (ball baskets, backgrounds)

---

## Output Formats

### **telemetry.csv**

Frame-by-frame near-player tracking (BoxSmoother output only):

```csv
frame_id,x,y,w,h
0,480.50,270.25,45.00,95.00
1,481.20,270.80,45.10,94.90
...
```

- **x, y**: center coordinates (pixels)
- **w, h**: width, height (pixels)
- **Only written if detection exists** (no interpolation for missing frames)

### **serve_events.json**

Deterministic serve timestamps (ARMED → ACTIVE transitions):

```json
[
  {
    "frame_id": 1247,
    "timestamp": 41.567
  },
  {
    "frame_id": 2156,
    "timestamp": 71.867
  }
]
```

### **highlights.mp4**

Merged active segments with 1-second pre-roll buffer:

```
Rally 1: 1.23s → 8.45s (7.22s duration)
Rally 2: 15.67s → 22.34s (6.67s duration)
...
```

Generated via single-pass ffmpeg filter_complex or sequential concat (fallback).

---

## Key Assumptions & Limitations

| Assumption | Impact | Mitigations |
|-----------|--------|------------|
| **Court must be fully visible** | Homography fails if court corners off-frame | Validate with reference frame preview |
| **Near-side player = server** | System doesn't distinguish players | Run twice (flip court) for both sides |
| **Toss detected in ROI** | High false-neg if toss moves sideways | Expand ROI in config if needed |
| **Serve always generates pose** | Trophy model must fire > 0.6 | Retrain model or lower threshold |
| **Ball model trained on match footage** | Poor generalization to different courts/lighting | Synthetic data + domain adaptation |
| **FPS stable** | Energy decay assumes constant dt | Validate video FPS; reject unstable sources |
| **Singles court only** | Doubles baseline/net differ | Hardcoded 27×78 ft — need config param |
| **Real-time visualization skips frames** | Headless mode ~2–4× faster | Use `--headless` for batch processing |

---

## Energy Configuration Reference

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `ENERGY_BOOST_BALL_FAST` | 1000 | Flying ball multiplier |
| `ENERGY_BOOST_PLAYER_SPRINT` | 1000 | Sprinting multiplier |
| `ENERGY_BOOST_PLAYER_ACTION` | 1000 | Swing/step multiplier |
| `ENERGY_DECAY_BALL_ROLLING` | 0.3 | Slowing ball penalty |
| `ENERGY_DECAY_BALL_OCCLUDED` | 0.15 | Hidden ball penalty |
| `ENERGY_DECAY_BALL_ACTION_ZONE` | 0.03 | Court occlusion (low decay) |
| `ENERGY_DECAY_PLAYER_WALK` | 0.2 | Walking penalty |
| `ENERGY_DECAY_PLAYER_WALKING_GAIT` | 0.4 | Detected walking gait penalty |
| `ENERGY_DECAY_PLAYER_MISSING` | 0.5 | Off-screen penalty |
| `PLAYER_WALK_VELOCITY_THRESHOLD` | 2.0 px/frame | Min velocity for "active" |
| `PLAYER_SPRINT_VELOCITY_THRESHOLD` | 6.0 px/frame | Min velocity for "sprint" |
| `NET_PROXIMITY_PLAYER_SCALE` | 3.0 | Boost at net (multiplier) |
| `NET_PROXIMITY_BALL_SCALE` | 0 | Attenuation at net |
| `TRANSITION_SCORE_THRESHOLD` | 0.55 | Serve detection threshold |
| `MIN_BALL_VELOCITY_FT_SEC` | 15.0 ft/s | Distinguish flying vs. rolling |
| `ABSOLUTE_BALL_LOST_TIMEOUT_ACTIVE` | 20.0 s | Max rally duration (active) |
| `ABSOLUTE_BALL_LOST_TIMEOUT_IDLE` | 6.0 s | Max rally duration (idle) |

---

## Performance Notes

- **Analysis resolution**: 960×540 (downscaled from input)
- **Headless mode**: 2–4× faster than live preview (no OpenCV imshow)
- **YOLO models**: Ball (960px), Player (480px), Trophy (320px), Toss (320px)
- **Memory**: ~2–3 GB for 1-hour video in memory (position/ball buffers)
- **CPU**: ~20–40% on M-series Mac (depends on video resolution)

---

## Usage

```bash
# Interactive court calibration + live preview
python anya_vision_core.py path/to/match.mp4

# Headless processing + custom output
python anya_vision_core.py path/to/match.mp4 -o my_highlights.mp4 --headless

# Outputs
# • match_telemetry.csv
# • match_serve_events.json
# • my_highlights.mp4
# • my_highlights_timestamps.txt
```

---

## Future Extensions

- [ ] Multi-serve support (both sides, simultaneous detection)
- [ ] Doubles court geometry (wider baseline)
- [ ] Finer-grained shot classification (forehand, backhand, volley)
- [ ] Ball spin detection (topspin, slice)
- [ ] Player-specific energy models (different playing styles)
- [ ] Neural network energy replacement (end-to-end rally segmentation)

