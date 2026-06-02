import os

# Content for the Markdown file
markdown_content = """# Technical Brief: Anya Vision Active Phase Parameter Optimization

## 1. Objective
Create a Python evaluation and optimization script that tunes the `Config` parameters of the `AnyaSystem` (defined in `anya_vision_core.py`). The primary goal is to refine the **end-of-rally detection logic** to ensure high-quality highlight clips.

## 2. Evaluation Heuristics (The "Scoring Function")
The model's performance on a specific rally is measured by the delta between the Predicted End Time ($T_{pred}$) and the Ground Truth End Time ($T_{gt}$).

* **Early Termination ($T_{pred} < T_{gt}$)**: **HEAVY PENALTY**. We must never cut off the live action of a point.
* **Exact to 2s Late ($T_{gt} \leq T_{pred} \leq T_{gt} + 2.0s$)**: **MAXIMUM REWARD**. This is the "Golden Zone" for highlights, providing enough follow-through after the point ends.
* **Excessive Overrun ($T_{pred} > T_{gt} + 2.0s$)**: **PROGRESSIVE SLOW PENALTY**. We want to minimize "dead time" (players walking back), but a 3-4 second overrun is preferable to an early cut.

## 3. Data Environment & Structure
The script must crawl a local directory structure: `/Volumes/Anya/Data/{XX}/` where `{XX}` is a numeric folder.

Each folder contains:
* `snippet.mp4`: The 7-minute source video.
* `ground_truth.json`: A list of rally objects with `start`, `end`, and `serve` ("near" or "far").
* `snippet_court_cache.json`: Pre-selected court corners.

**Constraint**: Only evaluate rallies where `"serve": "near"`.

## 4. Technical Logic (Anya Vision Core)
The system uses an energy-based state machine. The `ACTIVE` phase ends when `point_energy` hits `0.0` or a hard timeout is reached. 
You should focus on tuning the following variables in the `Config` class:
* `ENERGY_DECAY_BALL_DEAD`: Base decay when the ball is missing.
* `ENERGY_DECAY_BALL_ROLLING`: Decay when the ball is slow.
* `ABSOLUTE_BALL_LOST_TIMEOUT_IDLE`: Hard cut-off when the player is stationary.
* `ENERGY_DECAY_PLAYER_WALK`: How much walking (picking up balls) drains the energy.

## 5. Requirements for the Script
1.  **Import & Setup**: Import `AnyaSystem` and `Config` from `anya_vision_core.py`.
2.  **Headless Processing**: Run the video processing loop (likely using `cv2.VideoCapture`) without rendering frames for speed.
3.  **Synchronization**: Match predicted segments to Ground Truth segments using the `start` frame (within a reasonable 2-3 second window).
4.  **Metric Calculation**: Implement a scoring algorithm based on the heuristics in Section 2.
5.  **Output**: Provide a summary report showing the average error (in seconds) per video and an overall "Anya Performance Score."
"""

# Write the file to the sandbox
with open("anya_optimization_brief.md", "w") as f:
    f.write(markdown_content)

print("Anya_optimization_brief.md generated successfully.")