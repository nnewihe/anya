# Objective
Write a Python script that processes a tennis match video, extracts the bounding box telemetry of the "near" player using YOLO, and exports the data to a CSV file. This CSV is the direct input for our downstream point state machine.

# Core Requirements
1. **Target Identification**: Isolate the "near" player (the player closest to the bottom of the camera frame) using homography and real-world court dimensions.
2. **Telemetry Extraction**: For each frame, extract the near player's bounding box center `x`, center `y`, width `w`, and height `h` in **pixels**. 
3. **CSV Export**: Save the telemetry to a CSV strictly adhering to this header: `frame_id, x, y, w, h`.

# Technical Specifications & Reference Logic
You must use the following established patterns for court mapping and player filtering (matching our existing video probe and analysis architecture):

* **Interactive Court Setup**: Implement an `init_court` function that allows the user to click the 4 court corners on a reference frame, caching these pixel coordinates in a JSON file for subsequent runs.
* **Homography Mapping**: Compute a homography matrix (`H`) mapping the 4 pixel corners to a real-world coordinate system (Court Width: 27.0 ft, Court Length: 78.0 ft).
* **Near Player Selection Pipeline**:
    1.  Run YOLO (e.g., `yolo26n.pt`) to detect `person` classes.
    2.  For each detection, extract the bottom-center of the bounding box (representing the player's feet).
    3.  Transform this pixel coordinate into real-world coordinates `(world_x, world_y)` using `H`.
    4.  Discard detections outside the expanded court boundaries (use a lateral padding of 15.0 ft).
    5.  Compare the real-world Y distance to the near baseline vs. the far baseline. Select the detection closest to the near baseline as the target player for that frame.

# Critical System Context: Post-Strike Idle Periods
The downstream state machine calculates a Rolling Activity Index (RAI) driven heavily by pixel velocity and area volatility (`delta_area`). 

During active play, after the player hits the ball, the RAI will naturally drop low because they typically stand still until the opponent strikes the ball. It is absolutely critical that your extraction script maintains highly stable, jitter-free bounding box dimensions during these stationary periods. Any bounding box flickering (rapid changes in `w` and `h` while the player is standing still) will trigger false spikes in `delta_area` and prematurely shift the state machine out of its waiting state. Please ensure the tracking logic (e.g., box smoothing or SORT implementation) accounts for this.