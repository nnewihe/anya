# Task: Upgrade Gait Detection to Kinematic Knee-Angle Analysis

## 1. Context & Goal
I am upgrading a "Gait Detector" module within the **Anti-Vision Two-State System**. The current version uses a simple Y-coordinate oscillation heuristic based on a YOLO bounding box (`near_box`). This is prone to false positives from camera shake or non-walking vertical movement.

**The Goal:** Transition to a more robust, biologically accurate gait detection method using **MediaPipe Pose**. We want to monitor the flexion and extension of the knee joint over a rolling **5-second window**.

## 2. Current Implementation (Baseline)
The existing logic uses a residual/reversal count on the bounding box's bottom-edge Y-coordinate:

```python
def _detect_walking_gait(self, near_box) -> bool:
    if near_box is None:
        self.gait_y_buffer.clear()
        return False
    self.gait_y_buffer.append(near_box[3])
    n = len(self.gait_y_buffer)
    if n < self.GAIT_BUFFER_FRAMES * 0.6:
        return False
    # ... logic for oscillation reversals ...
3. Technical Requirements for the UpgradeA. Pose Estimation IntegrationUse mediapipe.solutions.pose to extract landmarks.Specifically track the Left Knee (23, 25, 27) and Right Knee (24, 26, 28) (Hip, Knee, Ankle indices).Only process frames where landmark visibility/confidence is above 0.5.B. Angle Calculation LogicImplement a helper function to calculate the interior angle of the knee.Points: $A$ (Hip), $B$ (Knee), $C$ (Ankle).Formula: Use the dot product of vectors $BA$ and $BC$ or atan2 to find the angle $\theta$.Normalization: Ensure the angle maps to degrees (e.g., ~180° for a straight leg, ~120° for a flexed leg during swing phase).C. Buffer Management (The 5-Second Rule)Implement a deque or rolling buffer that stores knee angles (left/right) indexed by timestamps.The buffer must maintain exactly 5 seconds of historical data at the current FPS.D. Gait Signature AnalysisA valid gait cycle should be identified by:Periodic Flexion: The knee angle should cycle between "Extension" (~170°+) and "Flexion" (~130° or less).Alternating Legs: If possible, correlate left and right knee cycles (they should be roughly 180° out of phase).Frequency Check: Human walking gait usually occurs at 1.5Hz to 2.5Hz. Analyze the buffer to see if the peak-to-peak frequency falls within this range.Signal Smoothing: Use a simple moving average or Savitzky-Golay filter to remove "jitter" from the MediaPipe landmarks.4. Requested Python StructurePlease provide:A class GaitAnalyzer that maintains the MediaPipe state and the 5-second buffers.A method update(frame) that returns True if a walking gait is detected.Threshold constants (e.g., MIN_FLEXION_ANGLE, MAX_EXTENSION_ANGLE, MIN_CYCLES_IN_5S).A fallback mechanism: If landmarks are lost, how does the system gracefully handle the "Unknown" state?