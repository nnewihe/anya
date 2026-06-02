# System Specification: Probabilistic Hidden Markov Model (HMM) for Tennis Match Phase Tracking
## Platform: Anya Tennis Analytics (Upgrading the Anti-Vision Two-State Base Canvas)

This engineering specification outlines the mathematical framework, state mechanics, multimodal emission probabilities, and architectural blueprints required to build a robust frame-by-frame Hidden Markov Model (HMM). This system replaces the legacy heuristic "energy bar" decay mechanism, explicitly optimized to **minimize false positive serve detections** and **prevent clipping of active play (zero-tolerance for active-time truncation)**.

---

## 1. Architectural Context & Objectives

The system processes continuous metadata streams (YOLO coordinates, TrackNet trajectories, RNN action outputs, and acoustic events) to slice a tennis match video into clean segments. 

### Core Objectives:
1. **Zero Active-Time Truncation:** The system must never falsely transition from `Active_Rally` to `Waiting` while a point is alive.
2. **False Positive Serve Elimination:** Transitions to `Active_Rally` require synchronous multi-sensor verification to protect against random hand gestures or player motion hallucinations.
3. **Probabilistic Smoothing:** Replaces hard-coded linear decay with a recursive Bayesian tracker that maintains context despite temporal dropouts or physical occlusions (e.g., ball out-of-frame for 15 frames).

---

## 2. Mathematical Framework: First-Order Hidden Markov Model

The match is modeled as a first-order Hidden Markov Model where the true state of play $S_t$ at frame $t$ is hidden, and must be inferred from a continuous, multimodal observation vector $O_t$.

The system computes the posterior probability distribution over all states via a forward recursive update:

$$P(S_t \\mid O_{1:t}) = \\alpha \\cdot P(O_t \\mid S_t) \\sum_{S_{t-1}} P(S_t \\mid S_{t-1}) P(S_{t-1} \\mid O_{1:t-1})$$

Where:
* $\\alpha$ is a normalization scaling factor ensuring $\\sum_{S} P(S_t = S \\mid O_{1:t}) = 1.0$.
* $P(S_{t-1} \\mid O_{1:t-1})$ is the **Prior Belief** (the posterior distribution from the previous frame).
* $P(S_t \\mid S_{t-1})$ is the **Transition Probability**, defining the physical bounds of how tennis scores/phases progress.
* $P(O_t \\mid S_t)$ is the **Emission Probability**, representing how likely the computer vision and audio tracking signatures are given a specific match state.

---

## 3. State Space Definition ($S$)

The system tracks three distinct hidden states ($N=3$):

1. **`Waiting` ($S_W$):** Default state. Dead time between points. Players walking, picking up balls, changing ends, or standing inactive.
2. **`Ready_Armed` ($S_{RA}$):** The server is stationary in the serving region (0 to 3.5 feet behind the baseline) preparing to initiate a serve sequence.
3. **`Active_Rally` ($S_{AR}$):** Point is actively alive. From the instant of racket-ball impact on serve until the point is dead.

### Transition Matrix $A = \\{P(S_t \\mid S_{t-1})\\}$

The transition matrix enforces chronological constraints. Certain state transitions are physically impossible in a tennis match (e.g., jumping from `Waiting` directly to `Active_Rally` without an intervening `Ready_Armed` setup, or initiating a `Serve_Motion` during a rally).

| From State ($S_{t-1}$) \\ To State ($S_t$) | `Waiting` ($S_W$) | `Ready_Armed` ($S_{RA}$) | `Active_Rally` ($S_{AR}$) |
| :--- | :--- | :--- | :--- |
| **`Waiting` ($S_W$)** | $1 - p_{W\\to RA}$ | $p_{W\\to RA}$ | $0.0000$ (Mathematically Zero) |
| **`Ready_Armed` ($S_{RA}$)** | $p_{RA\\to W}$ | $1 - p_{RA\\to W} - p_{RA\\to AR}$ | $p_{RA\\to AR}$ (Requires Serve Trigger) |
| **`Active_Rally` ($S_{AR}$)** | $p_{AR\\to W}$ (Probabilistic Decay) | $0.0000$ (Mathematically Zero) | $1 - p_{AR\\to W}$ |

#### Parameter Commitments for Matrix $A$:
* $P(S_{AR} \\mid S_W) = 0.0$: Completely prevents hallucinated serves from triggering points when players are moving in dead time.
* $P(S_{RA} \\mid S_{AR}) = 0.0$: A player cannot enter a ready-to-serve state while a rally is happening.
* $p_{RA\\to AR}$: This probability spikes from baseline close-to-zero to **1.0** ONLY when the multi-modal 2-second windows sync up (see Section 5).

---

## 4. Multimodal Observation Space ($O_t$) & Emission Modeling

At each frame $t$, the observation vector $O_t$ is composed of three primary channels:
$$O_t = [O_{\\text{ball}}, O_{\\text{player}}, O_{\\text{serve}}]$$

Instead of binary logic, observations scale the emission probabilities $P(O_t \\mid S_t)$ continuously.

### 4.1 Ball Tracking Signal ($O_{\\text{ball}}$)
Evaluated relative to the user-defined active court polygon zone.

* **Rule 4.1.1: Active Fast Ball Trace**
  * *Condition:* Active TrackNet ball trajectory is detected inside the court polygon, with localized velocity $V_{\\text{ball}} > V_{\\text{threshold}}$.
  * *Emission Impact:* High probability for `Active_Rally`, low for `Waiting`, zero for `Ready_Armed`.
    $$P(O_{\\text{ball}} \\mid S_{AR}) \\gg P(O_{\\text{ball}} \\mid S_W)$$

* **Rule 4.1.2: Slow Ball In Court**
  * *Condition:* Ball is detected rolling or moving slowly ($V_{\\text{ball}} \\le V_{\\text{threshold}}$) inside the court zone.
  * *Emission Impact:* High probability for `Waiting` (dead time ball retrieval), low for `Active_Rally`.
    $$P(O_{\\text{ball}} \\mid S_W) \\gg P(O_{\\text{ball}} \\mid S_{AR})$$

* **Rule 4.1.3: Ball Appended to Player**
  * *Condition:* Ball coordinate remains within the bounding box proximity of a player for a sustained period ($>1.5$ seconds).
  * *Emission Impact:* High probability for `Waiting` (bouncing ball before serving) or `Ready_Armed`, low for `Active_Rally`.
    $$P(O_{\\text{ball}} \\mid S_{RA}) \\approx P(O_{\\text{ball}} \\mid S_W) > P(O_{\\text{ball}} \\mid S_{AR})$$

* **Rule 4.1.4: Sustained Ball Occlusion / Missing Track**
  * *Condition:* No ball detection is returned from the computer vision layer for a sustained window ($t_{\\text{missing}} > \\tau_{\\text{missing}}$, where $\\tau_{\\text{missing}} \\approx 2.0$ seconds).
  * *Emission Impact:* Acts as a smooth, probabilistic degradation of the active state. High probability of `Waiting` or `Ready_Armed`, low probability of `Active_Rally`.
    $$P(\\text{Missing For } t > \\tau \\mid S_W) \\gg P(\\text{Missing For } t > \\tau \\mid S_{AR})$$

### 4.2 Player Kinematics Signal ($O_{\\text{player}}$)
Tracks tracking center velocities ($V_p$) and spatial placement.

* **Rule 4.2.1: High Lateral/Vertical Kinematics**
  * *Condition:* High player bounding box displacement ($V_p > V_{\\text{active}}$) suggesting athletic movement.
  * *Emission Impact:* High probability for `Active_Rally`, lower for `Waiting`, absolute zero for `Ready_Armed`.
    $$P(O_{\\text{player}} \\mid S_{AR}) = 0.95, \\quad P(O_{\\text{player}} \\mid S_W) = 0.05, \\quad P(O_{\\text{player}} \\mid S_{RA}) = 0.00$$

* **Rule 4.2.2: Slow Linear Baseline Recovery (Top-to-Bottom / Bottom-to-Top)**
  * *Condition:* Player moves slowly along the longitudinal axis back towards the absolute back edges of the court screen space.
  * *Emission Impact:* Highly characteristic of dead time transition back to position. High probability for `Waiting`, low for `Active_Rally`, zero for `Ready_Armed`.
    $$P(O_{\\text{player}} \\mid S_W) \\gg P(O_{\\text{player}} \\mid S_{AR})$$

* **Rule 4.2.3: Consistent Walking Velocity (Left/Right pacing)**
  * *Condition:* Player drifts laterally along the back fence or middle court areas with a steady, non-accelerating walking pace.
  * *Emission Impact:* High probability for `Waiting`, low for `Active_Rally`, zero for `Ready_Armed`.

* **Rule 4.2.4: Baseline Stationarity Anchoring**
  * *Condition:* The base center of the player's bounding box resides entirely within the designated server zone: $[0 \\text{ ft}, 3.5 \\text{ ft}]$ strictly behind the baseline, and satisfies this condition for a sustained duration of $\\Delta t \\ge 1.0\\text{ second}$.
  * *Emission Impact:* Heavy mathematical pull towards the `Ready_Armed` state.
    $$P(O_{\\text{player}} \\mid S_{RA}) \\gg P(O_{\\text{player}} \\mid S_W) > P(O_{\\text{player}} \\mid S_{AR})$$

---

## 5. Slit-Window Multi-Sensor Serve Trigger Mechanics

The transition from `Ready_Armed` ($S_{RA}$) to `Active_Rally` ($S_{AR}$) is protected by a strict multimodal verification window. A single sensor cannot trigger a state change; instead, three separate observations must be logged within a moving temporal slice of $\\Delta t_{\\text{window}} \\approx 2.0\\text{ seconds}$.

### The Multi-Sensor Inputs:
1. **$E_{\\text{toss}}$ (Ball Toss Event):** Explicit detection of vertical ball divergence from player bounding box baseline, peaking and entering freefall.
2. **$E_{\\text{rnn}}$ (RNN Motion Match):** Recurrent Neural Network or 3D-CNN confidence output for the mechanics of a tennis service motion crossing a localized threshold (e.g., $Confidence > 0.85$).
3. **$E_{\\text{audio}}$ (Racket Impact Audio Spike):** Audio processing signature registering a high-frequency acoustic peak localized between 4kHz - 8kHz (matching graphite racket ball strikes).

### Mathematical Trigger Execution:
The state transition variable $p_{RA\\to AR}$ within Matrix $A$ is modeled dynamically based on the overlapping presence of these events within the 2-second buffer:

$$p_{RA\\to AR}(t) = g\\left( \\max_{t_1, t_2, t_3 \\in [t-2.0, t]} \\left\\{ E_{\\text{toss}}(t_1) \\cdot E_{\\text{rnn}}(t_2) \\cdot E_{\\text{audio}}(t_3) \\right\\} \\right)$$

If all three elements fire within the 2-second moving window, $p_{RA\\to AR}$ snaps to **1.0**, driving an absolute and immediate transition to `Active_Rally`. If only one or two fire, the probability scales down aggressively toward zero, completely eliminating false positive serve detections from random ball bounces or warm-up movements.

---

## 6. Implementation Blueprint (Python Code Specification)

Use this complete architectural blueprint to translate the specifications directly into your execution code.

## 7. Instructions for System Construction (Prompting Directives for Claude)

When implementing this HMM within the existing pipeline architecture, ensure you execute the code adhering to these specific deployment parameters:Prioritize Active State Retention: Ensure that A_base[2][2] (the probability of staying in Active_Rally) remains highly weighted ($\ge 0.98$). This ensures that once a point starts, the system heavily resists transitioning back to a Waiting state until massive, sustained physical and tracking data indicates the point has concluded.