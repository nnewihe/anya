# Task: Implement Perspective-Aware Energy System for Point State Detection

## 1. Context & Objectives
We are detecting the ball using YOLO in a 2D pixel space (960 x 540). The camera is mounted ~10ft high on the back fence. 
**Goal:** Implement a robust `Point Energy` system in `TransitionEngine` that increases when a point is active and decays when the point is dead, using only 2D telemetry.

## 2. Configuration Updates (`utilities.py`)
Add the following constants to the `Config` class:
* `HORIZON_Y_PX`: 200 (The vertical pixel coordinate of the visual horizon for a 540p frame).
* `ENERGY_MAX`: 100.0
* `ENERGY_THRESHOLD_ACTIVE`: 40.0 (Point starts)
* `ENERGY_THRESHOLD_DEAD`: 15.0 (Point ends)
* `COHERENCE_THRESHOLD`: 0.7 (Minimum cosine similarity for trajectory validation)
* `ROLLING_Y_VAR_THRESHOLD`: 2.0 (Low variance in Y indicates the ball is on the ground)

## 3. Telemetry Processing Logic (`anya_transitions.py`)

Modify the `TransitionEngine` to implement the following four pillars of energy calculation:

### A. Perspective-Weighted Velocity (V_w)
Since pixel displacement is smaller at the far end of the court, we must normalize velocity. Because the camera is mounted on the back fence, the y-coordinate is an excellent proxy for depth.
1. Calculate raw pixel velocity: `V_px = sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)`.
2. Calculate weight: `W = 1 / max(1, y_t - Config.HORIZON_Y_PX)`.
3. Result: `V_w = V_px * W`.

### B. Trajectory Coherence Filter (C_t)
To ignore "teleporting" balls caused by false YOLO detections:
1. Calculate the current velocity vector `v_t` and the previous vector `v_{t-1}`.
2. Calculate Cosine Similarity: `C_t = (v_t • v_{t-1}) / (||v_t|| ||v_{t-1}||)`.
3. **Validation:** If `C_t < Config.COHERENCE_THRESHOLD` and displacement is large, treat the detection as "noise" and do not add energy.

### C. The Rolling Heuristic (H_r)
1. Maintain a short rolling buffer (e.g., 15 frames) of the ball's y-coordinate.
2. If the variance of y is very low and `V_w` is low, the ball is likely rolling on the ground.
3. **Penalty:** If `H_r` is active, apply a `DRIVE_TO_ZERO` decay multiplier.

### D. Temporal Decay & Missing Frames
1. If no ball is detected in the current frame, do not immediately zero the energy.
2. Apply a linear decay: `E_t = E_{t-1} - Base_Decay`.
3. If no ball is detected for > 1.5 seconds (e.g., 45 frames at 30fps), accelerate decay exponentially.

---

## 4. State Machine Implementation Details

### The Energy Equation
In `evaluate_transitions`, update the energy `E` for the current frame:
`E_t = clamp(E_{t-1} + Delta_active - Delta_decay, 0, 100)`

* **`Delta_active`**: Only applied if a valid, coherent ball detection exists. Scaled by `V_w`.
* **`Delta_decay`**: A constant friction value, multiplied by 2x if the ball is "Rolling" or 5x if the ball is absent for >30 frames.

### The Sliding Window Trigger
To prevent state "flickering" (e.g., a high lob briefly losing energy):
1. Maintain a 2.0-second sliding window of `E_t` (e.g., a `deque` of the last 60 energy values).
2. **Transition to ACTIVE:** Trigger if the *average* energy in the window crosses `ENERGY_THRESHOLD_ACTIVE`.
3. **Transition to DEAD:** Trigger if the *average* energy in the window drops below `ENERGY_THRESHOLD_DEAD`.

---

## 5. Implementation Instructions for Claude

1.  **Update `anya_transitions.py`**:
    * Create a method `_calculate_ball_telemetry_features(self, history)` that extracts `V_w` and `C_t`.
    * Refactor `_update_active_energy` to use the physics-based logic above.
    * Implement a `deque` to store the last 60 frames of calculated energy values to support the sliding window average.
2.  **Handle Missing Frames**: 
    * Ensure the system uses the `telemetry_history` buffer to bridge gaps. If the ball is missing for 5 frames, use the last known position to calculate "imaginary" decay until a new detection appears or the timeout is reached.
3.  **Visual Debugging (Optional but Recommended)**:
    * In `run_anya.py`, ensure the "Energy" bar in the UI reflects this new 0-100 scale and clearly marks the Active/Dead thresholds.

### Logic Edge Cases to Handle:
* **The Serve:** The energy will spike during the serve. Ensure the transition from `ARMED` to `ACTIVE` sets the initial energy to ~50 to prevent an immediate drop back to `DEAD`.
* **The Horizon:** Ensure `y_t - Config.HORIZON_Y_PX` never results in a division by zero; use a small epsilon or `max(1, diff)`.
