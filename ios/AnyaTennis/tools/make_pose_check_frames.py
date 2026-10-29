"""
make_pose_check_frames.py
=========================
Real frames for checking the app's pose model against Ultralytics:
far-band crops (named far*) and 960x540 whole frames (named near*) from
corpus clips, as .png. Used by export_pose_coreml.py --check-frames and by
the Swift PoseModelTests, which also read reference.json: Ultralytics' own
people on each frame at perceive.py's settings.

    python ios/AnyaTennis/tools/make_pose_check_frames.py OUT_DIR
"""

import glob
import json
import os
import subprocess
import sys

DATA = os.environ.get("ANYA_DATA", "/Volumes/Anya/Data")

# (clip, far band crop w:h:x:y in source px) -- each clip's own reference band.
CLIPS = {"43": "774:199:902:669", "25": "1058:283:1592:612", "50": "733:189:1139:532", "23": "729:185:1022:595"}
SECONDS = (30, 95, 150, 210)


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for clip, band in CLIPS.items():
        video = os.path.join(DATA, clip, "snippet.mp4")
        for second in SECONDS:
            for name, vf in ((f"far{clip}_{second}", f"crop={band}:exact=1"),
                             (f"near{clip}_{second}", "scale=960:540:flags=bilinear")):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(second), "-i", video,
                                "-frames:v", "1", "-vf", vf, os.path.join(out, name + ".png")], check=True)
    write_reference(out)
    print(f"wrote {len(glob.glob(os.path.join(out, '*.png')))} frames and reference.json to {out}")


def write_reference(out):
    """Ultralytics predict on each frame, as perceive.py runs it."""
    import cv2
    from ultralytics import YOLO
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    model = YOLO(os.path.join(repo, "pipeline", "models", "yolov8n-pose.pt"))
    reference = {}
    for path in sorted(glob.glob(os.path.join(out, "*.png"))):
        name = os.path.basename(path)
        imgsz = 640 if name.startswith("near") else 960
        result = model.predict(cv2.imread(path), imgsz=imgsz, conf=0.20, iou=0.70, classes=[0], verbose=False)[0]
        reference[name] = {"imgsz": imgsz, "people": [
            {"box": box.tolist(), "conf": float(conf), "keypoints": kp.reshape(-1).tolist()}
            for box, conf, kp in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                     result.keypoints.data.cpu().numpy())]}
    with open(os.path.join(out, "reference.json"), "w") as f:
        json.dump(reference, f)


if __name__ == "__main__":
    main()
