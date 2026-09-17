import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_COURT_POINTS = [[99.0, 420.0], [941.0, 393.0], [564.0, 257.0], [419.0, 260.0]]
DEFAULT_EXCLUSION_ZONES = [
    [135, 262, 152, 279], [157, 268, 182, 287], [471, 208, 489, 225],
    [205, 260, 235, 277], [118, 118, 135, 134], [73, 282, 94, 300],
    [239, 263, 256, 280], [65, 125, 84, 145]
]

class CourtSpatialFilter:
    def __init__(self, court_points: list, exclusion_zones: list, margin_px: float = 60.0):
        self.court_poly = np.array(court_points, dtype=np.int32)
        self.exclusion_zones = exclusion_zones
        self.margin_px = margin_px

    def score_spatial_prior(self, x: float, y: float, raw_conf: float) -> float:
        pt = (float(x), float(y))
        for zone in self.exclusion_zones:
            if len(zone) == 4:
                x1, y1, x2, y2 = zone
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return 1e-4

        dist = cv2.pointPolygonTest(self.court_poly, pt, measureDist=True)
        if dist < -self.margin_px:
            spatial_weight = np.exp((dist + self.margin_px) / 20.0)
        else:
            spatial_weight = 1.0

        return max(raw_conf * spatial_weight, 1e-4)


class TennisViterbiTracker:
    def __init__(self, fps: float = 30.0, max_changes: int = 2, change_thresh_px: float = 25.0, miss_penalty: float = 12.0):
        self.fps = fps
        self.max_changes = max_changes
        self.change_thresh = change_thresh_px
        self.miss_penalty = miss_penalty
        self.gravity = np.array([0.0, 0.5 * 9.81 * ((1.0 / fps) ** 2) * 30.0])

    def solve_window(self, frame_candidates: list) -> list:
        T = len(frame_candidates)
        if T < 3:
            return [None] * T

        aug_dets = []
        for dets in frame_candidates:
            aug = [(x, y, conf, False) for (x, y, conf) in dets]
            aug.append((0.0, 0.0, 0.0, True))
            aug_dets.append(aug)

        V = [{} for _ in range(T)]
        BP = [{} for _ in range(T)]

        for i, c1 in enumerate(aug_dets[1]):
            for j, c0 in enumerate(aug_dets[0]):
                s0 = -self.miss_penalty if c0[3] else np.log(c0[2])
                s1 = -self.miss_penalty if c1[3] else np.log(c1[2])
                V[1][(i, j, 0)] = s0 + s1
                BP[1][(i, j, 0)] = None

        for t in range(2, T):
            for i, curr in enumerate(aug_dets[t]):
                emission = -self.miss_penalty if curr[3] else np.log(curr[2])
                for (prev_i, prev_j, c), prev_score in V[t-1].items():
                    j = prev_i
                    prev1 = aug_dets[t-1][j]
                    prev2 = aug_dets[t-2][prev_j]

                    is_change, trans_cost = self._evaluate_step(prev2, prev1, curr)
                    c_new = c + (1 if is_change else 0)
                    if c_new > self.max_changes:
                        continue

                    total = prev_score + emission - trans_cost
                    key = (i, j, c_new)
                    if key not in V[t] or total > V[t][key]:
                        V[t][key] = total
                        BP[t][key] = (prev_j, c)

        if not V[T-1]:
            return [None] * T

        best_key = max(V[T-1], key=V[T-1].get)
        path = []
        curr_key = best_key
        for t in range(T - 1, 1, -1):
            i, j, c = curr_key
            cand = aug_dets[t][i]
            path.append(None if cand[3] else (cand[0], cand[1]))
            prev_j, prev_c = BP[t][curr_key]
            curr_key = (j, prev_j, prev_c)

        i, j, c = curr_key
        path.append(None if aug_dets[1][i][3] else (aug_dets[1][i][0], aug_dets[1][i][1]))
        path.append(None if aug_dets[0][j][3] else (aug_dets[0][j][0], aug_dets[0][j][1]))
        path.reverse()
        return path

    def _evaluate_step(self, p2, p1, p0) -> tuple:
        if p2[3] or p1[3] or p0[3]:
            return False, 1.5
        p_pred = 2 * np.array(p1[:2]) - np.array(p2[:2]) + self.gravity
        err = np.linalg.norm(np.array(p0[:2]) - p_pred)  # Fixed typo
        is_change = err > self.change_thresh
        return is_change, (3.0 if is_change else (err / self.change_thresh) ** 2)


def load_and_scale_metadata(video_path: Path, target_w: int, target_h: int) -> tuple:
    parent = video_path.parent
    stem = video_path.stem

    court_pts = DEFAULT_COURT_POINTS
    excl_zones = DEFAULT_EXCLUSION_ZONES
    ref_w, ref_h = 960, 540  # Default reference resolution

    combined_json = video_path.with_suffix(".json")
    if combined_json.exists():
        with open(combined_json, "r") as f:
            data = json.load(f)
            court_pts = data.get("points", court_pts)
            excl_zones = data.get("exclusion_zones", excl_zones)
            if "analysis_size" in data:
                ref_w, ref_h = data["analysis_size"][0], data["analysis_size"][1]
    else:
        court_cache = parent / f"{stem}_court_cache.json"
        if court_cache.exists():
            with open(court_cache, "r") as f:
                cdata = json.load(f)
                court_pts = cdata.get("points", court_pts)
                if "analysis_size" in cdata:
                    ref_w, ref_h = cdata["analysis_size"][0], cdata["analysis_size"][1]

        excl_cache = parent / f"{stem}_exclusion_cache.json"
        if excl_cache.exists():
            with open(excl_cache, "r") as f:
                excl_zones = json.load(f)

    # Calculate resolution scale factors
    sx = target_w / float(ref_w)
    sy = target_h / float(ref_h)

    scaled_court = [[p[0] * sx, p[1] * sy] for p in court_pts]
    scaled_excl = [
        [int(z[0] * sx), int(z[1] * sy), int(z[2] * sx), int(z[3] * sy)]
        for z in excl_zones if len(z) == 4
    ]

    return scaled_court, scaled_excl


def main():
    parser = argparse.ArgumentParser(description="Tennis Ball Trace Detector")
    parser.add_argument("video_pos", nargs="?", type=str, help="Positional path to video")
    parser.add_argument("--video", type=str, help="Optional flag path to video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--window_sec", type=float, default=0.5, help="Rolling Viterbi window in seconds")
    args = parser.parse_args()

    input_file = args.video_pos or args.video
    if not input_file:
        sys.exit("Error: Please specify a video file path.")

    video_path = Path(input_file).resolve()
    if not video_path.exists():
        sys.exit(f"Error: Video file not found: {video_path}")

    # Inspect video metadata first
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    window_frames = max(3, int(fps * args.window_sec))

    court_points, exclusion_zones = load_and_scale_metadata(video_path, width, height)
    spatial_filter = CourtSpatialFilter(court_points, exclusion_zones)

    # PASS 1: Candidate Extraction
    print("Pass 1/2: Running YOLO detection & candidate extraction...")
    model = YOLO(args.model)
    all_candidates = []
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        results = model(frame, verbose=False)[0]
        frame_dets = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 32
            if cls_id != 32:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            
            adjusted_conf = spatial_filter.score_spatial_prior(cx, cy, conf)
            if adjusted_conf > 0.02:
                frame_dets.append((cx, cy, adjusted_conf))

        frame_dets.sort(key=lambda d: d[2], reverse=True)
        all_candidates.append(frame_dets[:8])

    cap.release()

    # Solve Viterbi across rolling windows
    print(f"Solving trajectory across {total_frames} frames...")
    tracker = TennisViterbiTracker(fps=fps)
    final_trace = [None] * total_frames

    for i in range(0, total_frames, window_frames):
        win_candidates = all_candidates[i : i + window_frames]
        win_trace = tracker.solve_window(win_candidates)
        for idx, pt in enumerate(win_trace):
            if i + idx < len(final_trace):
                final_trace[i + idx] = pt

    # PASS 2: Stream Render
    print("Pass 2/2: Rendering trace overlay to output file...")
    out_path = video_path.parent / f"{video_path.stem}_trace.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    tail_len = int(fps * 0.5)
    court_poly = np.array(court_points, dtype=np.int32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.polylines(frame, [court_poly], isClosed=True, color=(255, 255, 0), thickness=2)
        for zone in exclusion_zones:
            if len(zone) == 4:
                x1, y1, x2, y2 = zone
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        start_idx = max(0, frame_idx - tail_len)
        pts = [pt for pt in final_trace[start_idx : frame_idx + 1] if pt is not None]

        for k in range(1, len(pts)):
            ptA = (int(pts[k-1][0]), int(pts[k-1][1]))
            ptB = (int(pts[k][0]), int(pts[k][1]))
            cv2.line(frame, ptA, ptB, (0, 255, 255), 3)

        if final_trace[frame_idx]:
            cx, cy = int(final_trace[frame_idx][0]), int(final_trace[frame_idx][1])
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done! Rendered output saved to: {out_path}")

if __name__ == "__main__":
    main()