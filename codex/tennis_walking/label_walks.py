"""Interactive real-time annotation of near-player walking intervals."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2


@dataclass
class WalkingInterval:
    start_second: float
    end_second: float
    start_frame: int
    end_frame: int


def label_video(video_path: Path, output_path: Path, *, speed: float = 1.0, display_width: int = 1280) -> list[WalkingInterval]:
    """Play a clip and collect intervals with S (start) and E (end)."""
    if speed <= 0:
        raise ValueError("speed must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, display_width / source_width)
    size = (round(source_width * scale), round(source_height * scale))
    frame_number, current_start, intervals, paused = 0, None, [], False
    window_name = "Near-player walking labels"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        if not paused:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        elif 'frame' not in locals():
            break
        timestamp = frame_number / fps
        display = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale != 1.0 else frame.copy()
        state = f"OPEN interval since {current_start[0]:.2f}s" if current_start else "no open interval"
        cv2.rectangle(display, (0, 0), (size[0], 88), (0, 0, 0), -1)
        cv2.putText(display, f"{timestamp:7.2f}s  {state}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        cv2.putText(display, "S=start  E=end  Space=pause  Left/Right=step  Q=save & quit", (16, 63), cv2.FONT_HERSHEY_SIMPLEX, .56, (100, 230, 255), 1)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(max(1, round(1000 / (fps * speed)))) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            paused = not paused
        elif key in (ord('s'), ord('S')):
            if current_start is None:
                current_start = (timestamp, frame_number)
                print(f"start {timestamp:.2f}s")
            else:
                print("Ignored S: an interval is already open.")
        elif key in (ord('e'), ord('E')):
            if current_start is None:
                print("Ignored E: press S before ending an interval.")
            elif timestamp > current_start[0]:
                intervals.append(WalkingInterval(current_start[0], timestamp, current_start[1], frame_number))
                print(f"end {timestamp:.2f}s")
                current_start = None
        elif key in (81, 2424832):  # OpenCV left-arrow values vary by platform.
            paused = True
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number - 2))
        elif key in (83, 2555904):
            paused = True
    if current_start is not None:
        print(f"Discarded unclosed interval beginning at {current_start[0]:.2f}s")
    payload = {
        "video": str(video_path), "fps": fps, "label": "near_player_walking",
        "intervals": [asdict(interval) for interval in intervals],
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    capture.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(intervals)} intervals to {output_path}")
    return intervals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label near-player walking intervals in a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=Path("walking_labels.json"))
    parser.add_argument("--speed", type=float, default=1.0, help="Playback multiplier; 1.0 is real time.")
    parser.add_argument("--display-width", type=int, default=1280)
    args = parser.parse_args()
    label_video(args.video, args.output, speed=args.speed, display_width=args.display_width)
