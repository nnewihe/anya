# Tennis walking detector

Detects whether the near-side player is walking once per elapsed second. It uses MediaPipe Pose landmarks 23/24 (hips) and 27/28 (ankles), projects them through an **image-to-court, metres** homography, and combines hip velocity, ankle separation, and cadence estimated from separation peaks.

```bash
pip install -r requirements.txt
python -m tennis_walking match.mp4 --homography image_to_court.npy \
  --output walking.jsonl --visualize walking_overlay.mp4 \
  --pose-model pose_landmarker_full.task
```

Each JSONL record contains `is_walking`, `confidence`, `speed_mps`, `stride_frequency_hz`, `foot_separation_m`, `hip_position`, and `ankle_position`. Missing pose data produces `is_walking: false` and confidence `0` for that second.

The default near-side gate accepts hip centres in the lower 60% of the image. MediaPipe evaluates up to two poses and selects the reliable pose with the lowest hip centre; adjust `--near-side-min-y` for a differently framed camera, or pass `0` to disable the gate. For an existing calibration object, call `run_video(..., homography=your_projector)` with a callable that converts `(pixel_x, pixel_y)` to `(court_x, court_y)`.

On MediaPipe Tasks installations, provide the `pose_landmarker` `.task` model with `--pose-model`. `--analysis-width` can speed inference, but its homography must be calibrated at that resized resolution.

`--walking-profile snippet21_cadence_speed_v1` applies the rule calibrated from this project's hand labels: cadence above 2.79 Hz and speed at most 12.71 m/s. It is intentionally clip-specific; use `confidence` (the default) for uncalibrated footage.

## Label walking intervals

To create near-player ground truth at real-time playback, run:

```bash
python -m tennis_walking.label_walks /path/to/match.mp4 --output walking_labels.json
```

Press `S` when walking starts and `E` when it ends. `Space` pauses playback, arrow keys step frames while paused, and `Q` saves completed intervals and exits.
