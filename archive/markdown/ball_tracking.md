# Specification: High-Fidelity Tennis Ball Tracking System

## 1. System Overview
This system is designed for a single-camera setup positioned behind the baseline. It prioritizes temporal consistency and physical logic over raw detection frequency. By using a pixel-based coordinate system, it avoids the inaccuracies of 3D mapping while maintaining high precision for trajectory analysis.

### Core Constraints
- **Primary Detector:** YOLO-based ball detection.
- **Coordinate Space:** Screen Pixels ($px$).
- **Velocity Units:** Pixels per second ($px/s$).
- **Default Temporal Resolution:** 60 FPS (Scalable to 30 FPS).
- **Camera Position:** Fixed, behind the baseline (Back fence).

---

## 2. State Machine: Point in Progress (PiP)
To ensure the tracker does not jump to "dead" balls or adjacent court activity, the system operates under a strict state machine.

### A. SEARCH (Idle)
- YOLO scans the full frame.
- **Trigger:** A point is flagged only when a detection exhibits a high-velocity vertical vector ($dy$) moving "up" the frame (away from the baseline), consistent with a serve.

### B. TRACKING (Active)
- **Hard Lock:** Once a serve is validated, the system locks to a specific `TrackID`.
- **Exclusion Zone:** All other detections (static balls, players on other courts) are filtered out based on velocity thresholds.

### C. EXTRAPOLATION (Occluded/Lost)
- Entered when YOLO confidence drops or detection is lost (e.g., hitting the net or far-court resolution limits).
- Uses the **Kalman Filter "Ghost"** to project position based on the last known velocity.

---

## 3. Kalman Filter Configuration
The Kalman Filter maintains the "Ghost" track during gaps in YOLO detections.

### State Vector ($S$)
$S = [x, y, dx, dy, w, h]$
- $x, y$: Centroid coordinates.
- $dx, dy$: Velocity (pixels per frame).
- $w, h$: Bounding box dimensions (used as a proxy for depth/perspective).

### Dynamic Process Noise ($Q$)
- **Normal Flight:** Low $Q$ (High trust in the physics model).
- **Hit Event Detection:** Momentary spike in $Q$ (Tells the filter to "reset" and trust new YOLO detections over the old trajectory).

---

## 4. Noise & Static Ball Suppression
To prevent the tracker from "jumping" to stationary balls on the court:
1. **The Static Buffer:** Any YOLO detection that remains at $[x, y]$ with $\Delta \approx 0$ for $>10$ frames is blacklisted.
2. **The Search Window:** YOLO detections are only considered if they appear within a $2.5 \times$ (Ball Diameter) radius of the Kalman-predicted "Ghost" position.
3. **Velocity Gating:** Detections moving at speeds or angles physically impossible for a tennis ball (based on current trajectory) are discarded.

---

## 5. Hit Event & Track Stitching
The system must logically bridge the gap when a ball is hit back from the far court.

### Velocity Flip Signature
- The "Return Hit" is identified by a **Y-Peak**: The moment the $y$-coordinate reaches its maximum displacement and the $dy$ vector flips sign (e.g., from $+15 px/f$ to $-10 px/f$).

### The "Handshake" Verification
When a ball reappears moving toward the camera:
1. **Probationary Period:** The new detection must persist for at least 3-5 consecutive frames.
2. **Vector Alignment:** The system checks if the new track's origin aligns with the "Ghost" track's projected endpoint at the time of the Y-Peak.
3. **Stitching:** If valid, the "Ghost" frames are backfilled with a smoothed trajectory (RTS Smoother), and the two tracks are merged into a single continuous ID.

---

## 6. Logic for Persistence & Abandonment
- **Persistence Limit:** The "Ghost" track will propagate for a maximum of **60 frames (1 second)**.
- **Abandonment:** If no "Handshake" is achieved within the persistence limit, the track is terminated to prevent "runaway ghosts."
- **Verification Priority:** A valid "Return" detection always overrides an existing "Ghost" projection if the spatial-temporal logic matches.