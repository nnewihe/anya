import os

# Define the content for the markdown file
md_content = """# Specification: Anya 3D Velocity Engine

## 1. Executive Summary
The goal is to upgrade the `anya_vision_core.py` framework from a 2D court-mapping system to a 3D trajectory and velocity engine. This system must detect a tennis serve, calculate the 3D velocity vector (x, y, z), and handle speeds between 40 mph and 120 mph using a single monocular camera feed (30/60 fps).

## 2. Geometric Constraints & Setup
* **Camera Position:** 6 feet high, 15 feet behind the baseline.
* **Coordinate System:**
    * **X-axis:** Along the baseline (0 = center mark).
    * **Y-axis:** Vertical (0 = court surface).
    * **Z-axis:** Into the court (0 = baseline).
* **Reference Object:** Standard tennis ball diameter (~0.22 ft / 6.7 cm).

## 3. Core Logic Upgrades

### 3.1 Camera Calibration (solvePnP)
Replace the existing `init_court` / `findHomography` logic.
* **Input:** 4 manual click points on the court (Baseline-Sideline intersections and Service-Line-Sideline intersections).
* **World Points:** * P1: `[-13.5, 0, 0]` (Left Baseline Corner)
    * P2: `[13.5, 0, 0]` (Right Baseline Corner)
    * P3: `[-13.5, 0, 60]` (Left Service Corner)
    * P4: `[13.5, 0, 60]` (Right Service Corner)
* **Algorithm:** Use `cv2.solvePnP` to derive the Rotation Vector (`rvec`) and Translation Vector (`tvec`). Convert `rvec` to a 3x3 Rotation Matrix (`R`) using `cv2.Rodrigues`.

### 3.2 3D Reconstruction via Optical Scaling
For every ball detection in the `ARMED` or `ACTIVE` states:
1. **Depth (Z_c) Calculation:** $Z_c = (focal\_length * ball\_real\_diam) / pixel\_diameter$.
2. **Camera Frame Coordinates:** Calculate $X_c$ and $Y_c$ using the pinhole camera model.
3. **World Frame Transformation:** $P_{world} = R^{-1} * (P_{camera} - tvec)$.

### 3.3 The Two-State Velocity Transition
Modify the `AnyaSystem` state machine to monitor the velocity vector $\mathbf{v} = [V_x, V_y, V_z]$:
* **TOSS State (Internal to ARMED):**
    * High $|V_y|$ (upward movement).
    * Low $|V_z|$ (minimal depth change).
    * Position must be above the Server's YOLO bounding box.
* **SERVE State (Trigger for ACTIVE):**
    * Detected when $V_z$ magnitude spikes and total velocity magnitude $||\mathbf{v}||$ exceeds 35-40 mph.
    * This transition marks the "Contact" event.

## 4. Technical Requirements

### 4.1 Frame Rate Handling
* The system must detect the ball at 30 or 60 fps. 
* At 120 mph, the ball moves ~2.9 ft (60 fps) or ~5.8 ft (30 fps) per frame.
* **Motion Blur Correction:** Use `max(width, height)` of the ball bounding box to determine `pixel_diameter` for depth, as the ball will streak along its vector.

### 4.2 Telemetry & Output
* **telemetry.csv:** Update columns to `[frame_id, world_x, world_y, world_z, vel_x, vel_y, vel_z, speed_mph]`.
* **Visualization:** Upon `finalize()`, generate a Matplotlib plot showing three subplots (Vx, Vy, Vz) over the duration of the serve (from toss to 1.0s post-contact).

## 5. Integration with anya_vision_core.py
* Retain the `BoxSmoother` for the player and racquet.
* Retain the `WAITING` state for player positioning.
* Replace the `point_energy` rally logic with the 3D physics-based state transition for the serve phase.
"""
