"""
Benchmark the existing YOLO ball detector (pipeline/anya_base.py's config) on the
same sample_clip.mp4 used for the TrackNet eval, for an apples-to-apples speed
comparison on this machine (MPS).
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from ultralytics import YOLO
from utilities import Config

HERE = Path(__file__).parent
MODEL_PATH = HERE.parents[1] / "pipeline" / "models" / "ball_best.pt"

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(HERE / "sample_clip.mp4"))
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f"{len(frames)} frames, imgsz={Config.ACTIVE_BALL_IMGSZ}, conf={Config.ACTIVE_BALL_CONF}")

# warmup (first call pays model/graph init cost)
model(frames[0], verbose=False, conf=Config.ACTIVE_BALL_CONF, imgsz=Config.ACTIVE_BALL_IMGSZ, device="mps")

t0 = time.time()
detected = 0
for f in frames:
    res = model(f, verbose=False, conf=Config.ACTIVE_BALL_CONF, imgsz=Config.ACTIVE_BALL_IMGSZ, device="mps")
    if res and res[0].boxes and len(res[0].boxes) > 0:
        detected += 1
elapsed = time.time() - t0

print(f"YOLO ball_best.pt: {elapsed:.1f}s total, {elapsed / len(frames) * 1000:.1f} ms/frame, "
      f"{len(frames) / elapsed:.2f} fps")
print(f"ball detected in {detected}/{len(frames)} frames ({100 * detected / len(frames):.1f}%)")
