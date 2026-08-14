import argparse
import numpy as np
from .detector import run_video

parser = argparse.ArgumentParser(description="Detect near-side tennis player walking.")
parser.add_argument("video")
parser.add_argument("--homography", required=True, help=".npy 3x3 image-to-court homography")
parser.add_argument("--output", default="walking.jsonl")
parser.add_argument("--visualize", help="Optional overlay video (.mp4)")
parser.add_argument("--near-side-min-y", type=float, default=0.40)
parser.add_argument("--pose-model", help="MediaPipe Tasks pose_landmarker .task model")
parser.add_argument("--analysis-width", type=int, help="Resize frames for inference; homography must use this size")
parser.add_argument("--walking-profile", choices=("confidence", "snippet21_cadence_speed_v1"), default="confidence")
args = parser.parse_args()
run_video(args.video, np.load(args.homography), args.output, visualize_path=args.visualize, near_side_min_image_y=args.near_side_min_y, pose_model_path=args.pose_model, analysis_width=args.analysis_width, walking_profile=args.walking_profile)
