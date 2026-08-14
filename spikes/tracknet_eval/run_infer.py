"""
Standalone TrackNet eval on a short match clip.
Thin wrapper around repo/infer_on_video.py's functions -- swaps the
hardcoded 'cuda' device for mps/cpu so it runs on this Mac.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "repo"))

import torch
from model import BallTrackerNet
from infer_on_video import read_video, infer_model, remove_outliers, split_track, interpolation, write_track

HERE = Path(__file__).parent

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device = {device}")

model = BallTrackerNet()
model.load_state_dict(torch.load(HERE / "repo" / "model_best.pt", map_location=device))
model = model.to(device)
model.eval()

# infer_on_video.infer_model references a module-global `device` -- patch it in.
import infer_on_video
infer_on_video.device = device

t0 = time.time()
frames, fps = read_video('/Volumes/Anya/Data/21/snippet.mp4')
print(f"{len(frames)} frames @ {fps} fps")

ball_track, dists = infer_model(frames, model)
ball_track = remove_outliers(ball_track, dists)

subtracks = split_track(ball_track)
for r in subtracks:
    ball_subtrack = interpolation(ball_track[r[0]:r[1]])
    ball_track[r[0]:r[1]] = ball_subtrack

write_track(frames, ball_track, str(HERE / "sample_clip_tracked.avi"), fps)
print(f"done in {time.time() - t0:.1f}s -> sample_clip_tracked.avi")

detected = sum(1 for x, y in ball_track if x is not None)
print(f"ball detected in {detected}/{len(ball_track)} frames ({100 * detected / len(ball_track):.1f}%)")
