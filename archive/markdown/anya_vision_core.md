# Objective
Refactor the provided tennis analysis scripts into a streamlined, two-stage pipeline. The goal is to consolidate video processing into a single pass and use deterministic serve detection to anchor a downstream state machine.

# Stage 1: Create `anya_vision_core.py`
Merge the telemetry extraction capabilities of `extract_telemetry.py` directly into `near_anya_v2.py` to create a unified vision engine.

**Requirements:**
1. **Integrate Bounding Box Smoothing:** Port the `BoxSmoother` class from `extract_telemetry.py` into the new script.
2. **Consolidate Tracking:** During the `_run_active_state`, `_run_waiting_state`, and `_run_armed_state` methods, apply the `BoxSmoother` to the `near_box` coordinates.
3. **Data Logging:** - Initialize a CSV writer to log the smoothed telemetry (`frame_id`, `x`, `y`, `w`, `h`) for every frame processed.
   - Initialize a JSON logger to record "Serve Events".
4. **Record Serves:** Whenever the system transitions from `SystemState.ARMED` to `SystemState.ACTIVE` (the serve detection), append the current `frame_id` and timestamp to the serve events JSON.
5. **Output:** The script must produce a `telemetry.csv` and a `serve_events.json` at the end of execution.

# Stage 2: Create `point_state_machine_v2.py`
Refactor `point_state_machine.py` to ingest the newly created *a priori* serve data, removing its reliance on bounding-box heuristics for serve detection.

**Requirements:**
1. **New Input:** Modify the script to accept `serve_events.json` alongside the bounding-box CSV.
2. **Remove Heuristics:** Strip out the `DELTA_A_SERVE_SPIKE` logic. The state machine should no longer use area volatility to guess when a serve happens.
3. **Anchor the States:** - Modify `_evaluate_transition` so that the transition from `PRE_POINT` (or `POINT_END`/`WALK_OFF`) to `SERVE` is triggered *strictly* when the current `frame_id` matches a frame listed in `serve_events.json`.
   - Ensure the `SERVE` state is held briefly before allowing RAI to take over for the `ACTIVE_MOVING` transition.
4. **Maintain Downstream Logic:** The rest of the state transitions (`ACTIVE_MOVING`, `ACTIVE_WAITING`, `POINT_END`, `WALK_OFF`) should continue to use RAI, Velocity, and Pacing Variance, but they will now be safely anchored by the true serve frame.
5. **Update Plotting:** Ensure the matplotlib output cleanly reflects these injected, absolute serve points.

Please provide the complete, ready-to-run code for both `anya_vision_core.py` and `point_state_machine_v2.py`.