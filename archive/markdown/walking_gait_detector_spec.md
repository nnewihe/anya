# Walking Gait Detector — Design Specification
### Module: `gait_detector.py` | Integrates with: `andy_vision_core.py`

---

## 1. Motivation & Product Hypothesis

Tennis points end not with a dramatic shot but with a quiet walk. After the final rally ball lands out, hits the net, or lands a winner, both players transition from active athletic movement to a relaxed walking pace within roughly 1–3 seconds. The hypothesis underlying this module is:

> **When the nearest-camera player sustains a walking gait for ≥ 1.5 seconds, the current point has ended and the video segment that follows is dead time eligible for splicing.**

This is a complementary dead-ball signal to the ball-energy and bounding-box-based signals already in `andy_vision_core.py`. Whereas those signals are ball-centric, this signal is player-centric and catches the human behavioral response to a point ending — which is biomechanically distinctive and robust to camera angle and ball occlusion.

---

## 2. Biomechanical Foundations

### 2.1 The Walking Gait Cycle — Key Anatomy

The gait cycle spans one full stride (heel-strike of one foot → next heel-strike of the same foot). It has two phases:

| Phase | Percent of Cycle | Description |
|---|---|---|
| **Stance** | ~60% | Foot on ground; weight-bearing |
| **Swing** | ~40% | Foot in air; limb advances forward |

Walking gait has a mandatory **double-support period** (~10–12% of cycle) where both feet are on the ground simultaneously. This is the kinematic boundary between walking and running — running has a float phase (zero ground contact), while walking never does.

### 2.2 Knee Angle Signature During Walking

The knee is the primary discriminating joint between walking, running, and standing still. Its sagittal-plane flexion/extension during normal walking follows a **biphasic** pattern:

```
Knee Angle (degrees flexion) vs. Gait Cycle %

70° |                          ___
    |                         /   \
    |                        /     \
40° |                       /       \
    |                      /         \
20° |           __/        /           \
    |          /  \       /             \___
 5° |_________/    \_____/
    0%    10%   20%   30%   40%   50%   60%   70%   80%   90%  100%
         IC    LR    MS    TS    PS    IS    MS    TS    IC
         
    IC=Initial Contact, LR=Loading Response, MS=Mid-Stance,
    TS=Terminal Stance, PS=Pre-Swing, IS=Initial Swing
```

**Key landmarks from literature:**

| Gait Event | Knee Flexion Angle |
|---|---|
| Initial contact (heel strike) | ~5° (range: −2° to 10°) |
| Loading response peak (stance) | ~15–20° |
| Terminal stance (near extension) | ~0–5° (near full extension) |
| Toe-off onset | ~40° |
| Mid-swing peak | **~65–70°** |
| Late swing pre-heel-strike | ~5° (rapid extension back) |

Sources:
- Musculoskeletal Key / Richards et al. (2017): knee flexion/extension is cyclic, ranging 0–70° in normal gait
- PMC / Biomechanics of MLKI (2020): stance peak flexion ≈ 20°; swing phase requires 60–70° for toe clearance
- Tandfonline (2020): walking max flexion 62.6°, min 1.7° at comfortable speed

**Discriminating features for walking vs. other states:**

| State | Knee Range | Rhythm | Hip Displacement |
|---|---|---|---|
| Walking | 0–70°, biphasic oscillation | ~0.9–1.1 Hz stride | Moderate lateral sway |
| Running | 10–130°, larger excursion | >1.3 Hz stride | Larger vertical bounce |
| Standing still / ready | Small (~0–15°), no rhythm | No periodicity | Near-zero |
| Lunging / split step | Single deep flexion >90° | Asymmetric | Large |

### 2.3 Why Knee Angle Is the Right Signal

Research validates MediaPipe Pose for markerless knee angle extraction:

- **Correlation with Kinovea (gold standard):** r = 0.941
- **Mean absolute error:** 5.88° — well within the ~20° range needed to discriminate walking from standing
- MediaPipe-based gait analysis has been validated against Vicon motion capture systems for stance time, swing time, and heel-strike / toe-off events

The key observable from a tennis court camera (side or near view) is the **rhythmic, periodic oscillation** of knee flexion angle at walking frequency (~0.9–1.2 Hz), with peak swings in the 40–70° range and near-full extension during stance (~5–15°). This is biomechanically distinct from:
- The athletic **ready stance** (knees bent ~20–30°, symmetric, no oscillation)
- The **run/sprint** (deeper flexion, higher frequency, vertical hip bounce)
- **Standing still** post-point (knees near 0°, no oscillation)

---

## 3. System Design

### 3.1 Module Role in `andy_vision_core.py`

```
andy_vision_core.py
    │
    ├── EnergyModel (ball tracking, serve detection)
    │       └── per-camera independent energy
    │
    ├── DeadBallDetector (existing snapshot logic)
    │       └── velocity + bounding box signal
    │
    └── GaitDetector  ←── NEW MODULE (gait_detector.py)
            └── knee angle oscillation from nearest player
            └── outputs: is_walking (bool), confidence (0–1), walk_duration_s (float)
```

`GaitDetector` is a **soft sensor** — its output feeds into the existing dead-ball state machine as an additional evidence signal. It does not gate recording independently; it corroborates or accelerates dead-ball triggering.

### 3.2 Inputs & Outputs

**Inputs:**
- `frame: np.ndarray` — current BGR video frame from near-side camera
- `fps: float` — frames per second of the video source
- `frame_idx: int` — current frame index (for timing)

**Outputs (per-frame, accumulated over time):**
```python
@dataclass
class GaitSignal:
    is_walking: bool          # True if sustained walk detected (≥ threshold duration)
    confidence: float         # 0.0–1.0 walk confidence this frame
    walk_duration_s: float    # seconds of continuous walking so far
    knee_angle_left: float    # degrees (nan if not visible)
    knee_angle_right: float   # degrees (nan if not visible)
    stride_freq_hz: float     # estimated stride frequency (nan if insufficient data)
    player_bbox: tuple        # (x1,y1,x2,y2) of nearest player or None
```

---

## 4. Algorithm Design

### 4.1 Player Selection — Nearest Camera Player

Use the existing YOLO person-detection output from `andy_vision_core.py` (no redundant inference). Select the player with the **largest bounding box area** — this is the nearest-camera player.

```python
def select_nearest_player(detections: list[Detection]) -> Detection:
    """
    Largest bounding box = nearest player.
    Filter to class 'person', confidence > 0.4.
    """
    persons = [d for d in detections if d.cls == 'person' and d.conf > 0.4]
    if not persons:
        return None
    return max(persons, key=lambda d: d.bbox_area())
```

If no player is detected for >1.0 seconds, suppress gait output (return `is_walking=False`).

### 4.2 Pose Estimation — MediaPipe Holistic / Pose

Use `mediapipe.solutions.pose` in **video mode** (`static_image_mode=False`) for temporal consistency. Crop the frame to the nearest-player bounding box (+ 20% padding) before MediaPipe inference to reduce compute and improve landmark quality.

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,        # balance of speed vs. accuracy
    smooth_landmarks=True,     # temporal smoothing
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

**Relevant landmark indices (MediaPipe 33-point model):**

| Landmark | Index |
|---|---|
| LEFT_HIP | 23 |
| LEFT_KNEE | 25 |
| LEFT_ANKLE | 27 |
| RIGHT_HIP | 24 |
| RIGHT_KNEE | 26 |
| RIGHT_ANKLE | 28 |

### 4.3 Knee Angle Computation

Compute the **included angle at the knee** using the hip–knee–ankle triplet:

```python
def compute_knee_angle(hip, knee, ankle) -> float:
    """
    Returns the angle (degrees) at the knee joint.
    0° = fully extended (straight leg), 90° = right-angle bend, 180° = hyperflexed.
    Convention: 180° = straight (anatomical extension) used in some literature.
    
    We use the SUPPLEMENTARY form: angle = 180° - included angle
    so that 0° = fully straight, 70° = mid-swing peak.
    This matches clinical reporting convention.
    """
    v1 = np.array([hip.x - knee.x, hip.y - knee.y])
    v2 = np.array([ankle.x - knee.x, ankle.y - knee.y])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    flexion_deg = 180.0 - angle_deg  # 0 = extended, positive = flexed
    return flexion_deg
```

Only use landmarks where `visibility > 0.6`. If both knees are below this threshold for >15 consecutive frames, set signal to uncertain.

### 4.4 Gait Feature Extraction

#### Feature 1: Knee Angle Oscillation (Primary Signal)

Maintain a **rolling buffer** of knee angles for both legs over a sliding window of 3.0 seconds (= `fps * 3` frames). Compute:

- `knee_range`: `max(buffer) - min(buffer)` — should be **30–80°** during walking
- `knee_oscillation_freq`: dominant FFT frequency of the angle time-series — should be **0.8–1.4 Hz** for walking (stride frequency; step frequency is 2× this)

```python
def estimate_oscillation_freq(angle_buffer: deque, fps: float) -> float:
    """
    FFT on recent knee angle time series.
    Returns dominant frequency in Hz. Returns nan if buffer < 1.5s.
    """
    arr = np.array(angle_buffer)
    if len(arr) < int(fps * 1.5):
        return float('nan')
    arr = arr - np.mean(arr)          # detrend
    freqs = np.fft.rfftfreq(len(arr), d=1.0/fps)
    power = np.abs(np.fft.rfft(arr))**2
    # constrain to walking-plausible band
    mask = (freqs >= 0.5) & (freqs <= 2.5)
    if not mask.any():
        return float('nan')
    dominant_freq = freqs[mask][np.argmax(power[mask])]
    return dominant_freq
```

#### Feature 2: Hip Lateral Oscillation (Secondary Signal)

Walking produces a characteristic side-to-side hip sway (~3–5 cm at normal walking speed). Track the normalized x-position of the midpoint between left and right hips over time. Oscillation amplitude in the range 0.01–0.06 of frame width at walking frequency supports gait.

#### Feature 3: Ankle Vertical Oscillation (Tertiary Signal)

During walking, ankles cycle up and down with the gait cycle. Compute the vertical range of ankle motion over the buffer window. Very low range = standing still. Very high range = running/lunging.

#### Feature 4: Body Center-of-Mass Velocity (Context Signal)

Estimate horizontal velocity of the player bounding box centroid. During active play, velocity is high and erratic. During walking, it is low (~0.5–2.0 px/frame depending on camera distance) and relatively constant in direction.

| CoM velocity | Interpretation |
|---|---|
| < 0.5 px/frame | Standing still |
| 0.5–3.0 px/frame | Walking (candidate) |
| > 3.0 px/frame | Running / moving fast |

### 4.5 Walking Classifier — Rule-Based Score

Combine features into a **walking confidence score** (0–1):

```python
def compute_walk_confidence(
    knee_range: float,
    knee_freq: float,
    hip_sway_amp: float,
    com_velocity: float,
    fps: float
) -> float:

    score = 0.0

    # Primary: knee oscillation range (weight 0.45)
    # Walking: range 25–75°. Outside = subtract.
    if 25 <= knee_range <= 80:
        score += 0.45 * np.clip((knee_range - 20) / 30.0, 0, 1)
    elif knee_range < 10:
        score -= 0.2  # standing still

    # Primary: oscillation frequency (weight 0.35)
    # Walking stride: 0.8–1.4 Hz
    if not np.isnan(knee_freq):
        if 0.75 <= knee_freq <= 1.5:
            score += 0.35
        elif 1.5 < knee_freq <= 2.5:
            score += 0.10   # fast walk / jog — partial credit

    # Secondary: hip sway (weight 0.10)
    if 0.008 <= hip_sway_amp <= 0.07:
        score += 0.10

    # Tertiary: CoM velocity (weight 0.10)
    if 0.3 <= com_velocity <= 3.5:
        score += 0.10
    elif com_velocity > 5.0:
        score -= 0.15  # running penalty

    return float(np.clip(score, 0.0, 1.0))
```

### 4.6 Temporal State Machine

Raw per-frame confidence is noisy. Apply a **temporal smoothing + threshold state machine**:

```
States: INACTIVE → CANDIDATE → WALKING → INACTIVE

INACTIVE  → CANDIDATE  : confidence > 0.55 for ≥ 3 consecutive frames
CANDIDATE → WALKING    : confidence > 0.55 maintained for ≥ WALK_MIN_DURATION_S (1.5s)
WALKING   → INACTIVE   : confidence drops below 0.35 for ≥ 0.4s (hysteresis)
CANDIDATE → INACTIVE   : confidence drops below 0.35 before WALK_MIN_DURATION_S
```

```python
WALK_MIN_DURATION_S = 1.5   # minimum sustained walk to trigger
WALK_ONSET_CONF     = 0.55  # confidence needed to start candidate
WALK_EXIT_CONF      = 0.35  # confidence below which walk ends
WALK_EXIT_HOLD_S    = 0.40  # must stay below exit threshold for this long
```

When `state == WALKING`, `is_walking = True` is emitted. The module also records the **walk start frame** for splicing reference.

---

## 5. Integration with `andy_vision_core.py`

### 5.1 Calling Convention

`GaitDetector` is instantiated once and called per-frame inside the main video processing loop:

```python
# In andy_vision_core.py initialization:
from gait_detector import GaitDetector, GaitSignal
gait_detector = GaitDetector(fps=fps, walk_min_duration_s=1.5)

# In per-frame loop:
gait_signal: GaitSignal = gait_detector.update(
    frame=frame,
    detections=yolo_detections,   # reuse existing YOLO output
    frame_idx=frame_idx
)

if gait_signal.is_walking:
    # candidate dead ball — feed into existing DeadBallDetector
    dead_ball_detector.add_gait_evidence(
        confidence=gait_signal.confidence,
        walk_duration_s=gait_signal.walk_duration_s
    )
```

### 5.2 Dead Ball Fusion Logic

The existing dead-ball detector uses energy-based signals. Gait evidence is fused as a **corroborating signal** with the following rules:

| Energy Signal | Gait Signal | Dead Ball Decision |
|---|---|---|
| Low energy | Walking ≥ 1.5s | **Strong dead ball** — splice immediately |
| Low energy | Not walking | Dead ball (existing behavior) — splice |
| Active energy | Walking ≥ 2.0s | **Override** — walking player overrides residual energy; splice |
| Active energy | Not walking | No dead ball — rally continues |

This means gait can **override a false-negative** from the energy model (e.g., the ball is still bouncing after a winner is hit, but the player is already walking).

The `walk_start_frame` from `GaitDetector` is used to backtrack the splice point to the moment the walk began (not the moment the 1.5s threshold triggers), subtracting ~0.2s latency buffer:

```python
splice_frame = gait_signal.walk_start_frame + int(fps * 0.2)
```

### 5.3 Per-Camera Behavior

`GaitDetector` runs on the **near-side camera only** (the camera for which a close-up player is most reliably visible). The far-side camera feed is not used for gait detection since:
- Players appear small, reducing MediaPipe landmark accuracy
- The alternating server means the walking player may be the far-side camera's near player on some points

If a dual-camera setup is active, `GaitDetector` instances per camera can be created, and gait from **either camera** can trigger dead-ball state — taking the **most positive** signal (consistent with the existing per-camera independent energy architecture).

---

## 6. Implementation Plan

### Phase 1 — Core Module (gait_detector.py)

- [ ] `GaitDetector` class with `__init__`, `update()`, `reset()` methods
- [ ] MediaPipe Pose integration with crop-to-player-bbox optimization
- [ ] `compute_knee_angle()` utility function
- [ ] Rolling angle buffer (`collections.deque`) with configurable window
- [ ] FFT-based oscillation frequency estimator
- [ ] `compute_walk_confidence()` score function
- [ ] Temporal state machine (INACTIVE → CANDIDATE → WALKING)
- [ ] `GaitSignal` dataclass output

### Phase 2 — Integration

- [ ] Hook `GaitDetector.update()` into the per-frame loop in `andy_vision_core.py`
- [ ] Add `gait_evidence` input to `DeadBallDetector`
- [ ] Fuse gait signal with energy signal per the fusion logic table above
- [ ] Backtrack splice point to `walk_start_frame`

### Phase 3 — Tuning & Validation

- [ ] Label 20–30 point endings in real match footage with: `[point_end_frame, walk_start_frame, gait_class (walking/running/standing)]`
- [ ] Run leave-one-video-out cross-validation on `walk_min_duration_s`, `WALK_ONSET_CONF`, knee range thresholds
- [ ] Measure: recall of walk detection (% of true dead balls caught by gait), false positive rate (gait fires during live play), latency (frames from walk start to detection trigger)
- [ ] Target metrics: recall ≥ 0.85, FP rate < 0.05 per point

### Phase 4 — Edge Cases

- [ ] Player temporarily out of frame (serve follow-through takes player off-screen): suppress gait for up to 0.5s
- [ ] Player bends to pick up a ball (deep knee flexion, not walking): spike filter on knee angle — single-frame excursions > 90° are masked
- [ ] Both players walking at different times: use only nearest-camera player for decision
- [ ] Player walking toward camera (frontal view): knee angle estimation degrades; fall back to hip vertical oscillation + CoM velocity only

---

## 7. Configuration Parameters

All tunable parameters exposed via a config dict for use with the existing parameter optimizer (`parameter_optimizer.py`):

```python
GAIT_DETECTOR_DEFAULTS = {
    # Timing
    "walk_min_duration_s":      1.5,   # minimum walk to call dead ball
    "buffer_window_s":          3.0,   # rolling analysis window
    "exit_hold_s":              0.40,  # hysteresis: stay below exit_conf for this long

    # Confidence thresholds
    "onset_confidence":         0.55,  # score to enter CANDIDATE
    "exit_confidence":          0.35,  # score to exit WALKING

    # Knee angle thresholds
    "knee_range_min_deg":       25.0,  # minimum oscillation range for walking
    "knee_range_max_deg":       80.0,  # maximum (above = running)
    "knee_freq_min_hz":         0.75,  # min stride frequency
    "knee_freq_max_hz":         1.50,  # max stride frequency

    # Player selection
    "player_min_confidence":    0.40,  # YOLO confidence for person
    "pose_visibility_thresh":   0.60,  # MediaPipe landmark visibility

    # MediaPipe
    "model_complexity":         1,     # 0=fast, 1=balanced, 2=accurate
    "bbox_padding_frac":        0.20,  # pad player bbox before crop

    # Fusion
    "gait_override_duration_s": 2.0,  # walk duration to override active energy
    "splice_latency_s":         0.20,  # backtrack offset from walk_start_frame
}
```

---

## 8. Computational Cost Considerations

MediaPipe Pose on a cropped player patch (typically ~200×400 px) runs at:
- ~5–8 ms/frame on CPU (model_complexity=1)
- ~2–3 ms/frame on GPU

At 30 fps, this adds ~150–240 ms of CPU processing per second — feasible for offline processing. For real-time operation on Raspberry Pi 5, use `model_complexity=0` (lighter model) and process every other frame.

YOLO detections are **reused** from the existing pipeline — no additional inference cost for player detection.

---

## 9. Known Limitations & Mitigations

| Limitation | Mitigation |
|---|---|
| MediaPipe struggles with side-on oblique angles (camera not perpendicular to motion plane) | Acceptable: near-side camera provides reasonable sagittal view of walking player |
| Knee angle error ~5.9° from MediaPipe vs. gold standard | Thresholds account for this; a 5° error doesn't change walk/stand/run classification |
| Player occlusion by net post or baseline | Suppress gait signal; don't emit false negatives — let energy model carry dead ball |
| Similar knee oscillation during excited ready-position bouncing | Bouncing has higher frequency (>2 Hz) and low CoM velocity — frequency gate filters it |
| Running after a point (chasing a ball that rolled away) | Walking frequency gate (< 1.5 Hz) and CoM velocity cap handle this |

---

## 10. File Structure

```
andy_vision/
├── andy_vision_core.py          # existing — integrates GaitDetector
├── gait_detector.py             # NEW — this module
├── gait_detector_test.py        # NEW — unit tests + validation harness
└── docs/
    └── walking_gait_detector_spec.md   # this file
```

---

## 11. Key References

1. Richards J., Chohan A., Erande R. (2017). *Biomechanics.* Musculoskeletal Key. — Knee gait cycle kinematics: 0–70° oscillation, biphasic pattern, five phases.

2. Mele M. et al. (2022). *Biomechanics of Multi-ligament Knee Injuries and Effects on Gait.* PMC (PMCID 2953341). — Stance peak ~20°, swing peak 60–70°, double-support phase anatomy.

3. Rowe P.J. et al. (2000). *Knee joint kinematics in gait and other functional activities.* Gait & Posture. — Walking requires <90° knee flexion; electrogoniometry validation.

4. Investigation of normal knee kinematics in walking and running. (2020). *J. Sports Biomechanics.* — Walking max flexion 62.6°, min 1.7°; running increases both values significantly.

5. Abid M. et al. (2019). *Knee Joint Biomechanical Gait Data Classification for Knee Pathology Assessment.* Applied Bionics and Biomechanics. — Feature extraction from gait waveforms; spatiotemporal parameters.

6. Bhattacharya S. et al. (2023). *Knee Flexion/Extension Angle Measurement for Gait Analysis Using MediaPipe Pose.* ResearchGate. — r=0.941 correlation, MAE=5.88° vs. Kinovea gold standard; validates MediaPipe for markerless knee gait analysis.

7. Kidziński Ł. et al. (2023). *Automated Gait Analysis Based on a Marker-Free Pose Estimation Model.* Sensors, 23(14):6489. — MediaPipe validated against Vicon for heel-strike, toe-off, stance/swing time; ICC good for most parameters at 25 fps.

8. PMC (2024). *Human Pose Estimation for Clinical Analysis of Gait Pathologies.* — MediaPipe for knee/hip angle extraction in pathological gait; confirms stride time, step time, cadence extractable.
