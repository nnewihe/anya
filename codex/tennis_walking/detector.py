"""Near-side player walking detection using MediaPipe Pose.

The supplied homography must map image pixels to court coordinates in metres.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Callable, Deque, Iterable, Optional, Sequence, Union

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # permit importing metric utilities without MediaPipe installed
    mp = None


Point = tuple[float, float]
Projector = Union[np.ndarray, Callable[[Point], Point]]


@dataclass(frozen=True)
class WalkingSample:
    """One per-second result. Coordinates and speed are in court units/metres."""

    second: int
    is_walking: bool
    confidence: float
    speed_mps: Optional[float]
    stride_frequency_hz: Optional[float]
    foot_separation_m: Optional[float]
    hip_position: Optional[Point]
    ankle_position: Optional[Point]


def project_point(point: Point, homography: Projector) -> Point:
    """Project an image point through a 3x3 image-to-court homography."""
    if callable(homography):
        return tuple(map(float, homography(point)))  # type: ignore[arg-type]
    x, y = point
    projected = np.asarray(homography, dtype=float) @ np.array([x, y, 1.0])
    if abs(projected[2]) < 1e-9:
        raise ValueError("homography projects point to infinity")
    return (float(projected[0] / projected[2]), float(projected[1] / projected[2]))


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


class WalkingDetector:
    """Temporal walking classifier for a single, near-side MediaPipe pose.

    ``near_side_min_image_y`` is expressed as a fraction of frame height. Poses
    whose hip centre is above it are ignored, preventing the far-side player
    from producing an output. Set it to 0 to disable this image-space gate.
    """

    def __init__(
        self,
        homography: Projector,
        *,
        near_side_min_image_y: float = 0.40,
        min_visibility: float = 0.45,
        history_seconds: float = 6.0,
    ) -> None:
        if not 0.0 <= near_side_min_image_y <= 1.0:
            raise ValueError("near_side_min_image_y must be between 0 and 1")
        self.homography = homography
        self.near_side_min_image_y = near_side_min_image_y
        self.min_visibility = min_visibility
        self.history_seconds = history_seconds
        self._positions: Deque[tuple[float, Point]] = deque()
        self._separations: Deque[tuple[float, float]] = deque()
        self._speed_ema: Optional[float] = None

    def update(
        self, timestamp: float, landmarks: Optional[Sequence[object]], width: int, height: int
    ) -> tuple[Optional[dict], float, Optional[float], Optional[float]]:
        """Update metrics from normalized MediaPipe landmarks.

        Returns ``(metrics, confidence, stride_hz, separation_m)``. Metrics is
        None if no reliable near-side pose is available.
        """
        if not landmarks:
            return None, 0.0, None, None
        # BlazePose indices: 23/24 hips, 27/28 ankles.
        required = (23, 24, 27, 28)
        if len(landmarks) <= max(required) or any(
            getattr(landmarks[i], "visibility", 1.0) < self.min_visibility for i in required
        ):
            return None, 0.0, None, None
        image_points = [
            (float(getattr(landmarks[i], "x")) * width, float(getattr(landmarks[i], "y")) * height)
            for i in required
        ]
        hip_px = ((image_points[0][0] + image_points[1][0]) / 2, (image_points[0][1] + image_points[1][1]) / 2)
        if hip_px[1] < height * self.near_side_min_image_y:
            return None, 0.0, None, None
        hip = project_point(hip_px, self.homography)
        left_ankle = project_point(image_points[2], self.homography)
        right_ankle = project_point(image_points[3], self.homography)
        ankle = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)
        separation = _distance(left_ankle, right_ankle)
        speed = self._append_position(timestamp, hip)
        self._separations.append((timestamp, separation))
        self._trim(timestamp)
        stride_hz = self._stride_frequency()
        confidence = self._confidence(speed, stride_hz, separation)
        return {"hip": hip, "ankle": ankle, "speed": speed}, confidence, stride_hz, separation

    def _append_position(self, timestamp: float, position: Point) -> Optional[float]:
        if self._positions and timestamp <= self._positions[-1][0]:
            return self._speed_ema
        raw_speed: Optional[float] = None
        if self._positions:
            old_t, old_position = self._positions[-1]
            dt = timestamp - old_t
            if dt > 1e-3:
                raw_speed = _distance(position, old_position) / dt
                # A short EMA suppresses pose jitter without hiding walking.
                self._speed_ema = raw_speed if self._speed_ema is None else 0.35 * raw_speed + 0.65 * self._speed_ema
        self._positions.append((timestamp, position))
        return self._speed_ema

    def _trim(self, timestamp: float) -> None:
        cutoff = timestamp - self.history_seconds
        while self._positions and self._positions[0][0] < cutoff:
            self._positions.popleft()
        while self._separations and self._separations[0][0] < cutoff:
            self._separations.popleft()

    def _stride_frequency(self) -> Optional[float]:
        """Estimate cadence from maxima in ankle separation (one maximum/step)."""
        values = list(self._separations)
        if len(values) < 5 or values[-1][0] - values[0][0] < 1.0:
            return None
        peaks: list[tuple[float, float]] = []
        for i in range(1, len(values) - 1):
            prev_v, value, next_v = values[i - 1][1], values[i][1], values[i + 1][1]
            if value > 0.14 and value >= prev_v and value > next_v:
                if not peaks or values[i][0] - peaks[-1][0] >= 0.20:
                    peaks.append((values[i][0], value))
                elif value > peaks[-1][1]:
                    # Keep the stronger peak inside the refractory period.
                    peaks[-1] = (values[i][0], value)
        if len(peaks) < 2:
            return None
        intervals = np.diff([peak[0] for peak in peaks])
        interval = float(np.median(intervals))
        return 1.0 / interval if interval > 0 else None

    @staticmethod
    def _confidence(speed: Optional[float], stride: Optional[float], separation: float) -> float:
        if speed is None or stride is None:
            return 0.0
        # Soft ranges make confidence stable near thresholds.
        speed_score = _sigmoid((speed - 0.28) / 0.10) * _sigmoid((4.5 - speed) / 0.5)
        stride_score = _sigmoid((stride - 0.55) / 0.18) * _sigmoid((3.8 - stride) / 0.4)
        separation_score = _sigmoid((separation - 0.12) / 0.04) * _sigmoid((1.7 - separation) / 0.25)
        return float(speed_score * stride_score * separation_score)


def run_video(
    input_video: Union[str, Path], homography: Projector, output_jsonl: Union[str, Path], *,
    visualize_path: Optional[Union[str, Path]] = None, near_side_min_image_y: float = 0.40,
    pose_model_path: Optional[Union[str, Path]] = None, analysis_width: Optional[int] = None,
    walking_profile: str = "confidence",
) -> list[WalkingSample]:
    """Analyze video and write exactly one JSON object per elapsed second."""
    if mp is None:
        raise ImportError("Install mediapipe to run video analysis: pip install -r requirements.txt")
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if visualize_path:
        writer = cv2.VideoWriter(str(visualize_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"Cannot create visualization: {visualize_path}")
    detector = WalkingDetector(homography, near_side_min_image_y=near_side_min_image_y)
    has_legacy_api = hasattr(mp, "solutions")
    if not has_legacy_api and not pose_model_path:
        raise ValueError("pose_model_path is required with the MediaPipe Tasks API")
    if analysis_width and analysis_width <= 0:
        raise ValueError("analysis_width must be positive")
    if walking_profile not in {"confidence", "snippet21_cadence_speed_v1"}:
        raise ValueError("unknown walking_profile")
    analysis_height = height if not analysis_width else round(height * analysis_width / width)
    analysis_width = analysis_width or width
    if has_legacy_api:
        drawing, connections = mp.solutions.drawing_utils, mp.solutions.pose.POSE_CONNECTIONS
        pose_engine = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    else:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        options = vision.PoseLandmarkerOptions(
            # CPU makes batch/video jobs reliable on headless macOS hosts.
            base_options=python.BaseOptions(model_asset_path=str(pose_model_path), delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_poses=2,
            min_pose_detection_confidence=0.35, min_pose_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        pose_engine = vision.PoseLandmarker.create_from_options(options)
    results: list[WalkingSample] = []
    current_second = 0
    bucket: list[tuple[dict, float, Optional[float], Optional[float]]] = []
    with pose_engine as pose:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            second = int(timestamp)
            while second > current_second:
                results.append(_summarize_second(current_second, bucket, walking_profile))
                bucket, current_second = [], current_second + 1
            analysis_frame = frame if (analysis_width, analysis_height) == (width, height) else cv2.resize(frame, (analysis_width, analysis_height))
            rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
            if has_legacy_api:
                pose_result = pose.process(rgb)
                landmarks = pose_result.pose_landmarks.landmark if pose_result.pose_landmarks else None
            else:
                pose_result = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(timestamp * 1000))
                landmarks = _select_near_pose(
                    pose_result.pose_landmarks, analysis_height,
                    detector.near_side_min_image_y, detector.min_visibility,
                )
            metrics, confidence, stride, separation = detector.update(timestamp, landmarks, analysis_width, analysis_height)
            profile_metrics = {**metrics, "stride_frequency_hz": stride} if metrics else None
            effective_confidence = _profile_confidence(profile_metrics, confidence, walking_profile)
            if metrics:
                bucket.append((metrics, effective_confidence, stride, separation))
            if writer:
                if landmarks:
                    if has_legacy_api:
                        drawing.draw_landmarks(frame, pose_result.pose_landmarks, connections)
                    else:
                        _draw_task_landmarks(frame, landmarks)
                state = _is_walking(profile_metrics, effective_confidence, walking_profile)
                label = f"speed: {metrics['speed']:.2f} m/s" if metrics and metrics["speed"] is not None else "speed: unavailable"
                cv2.putText(frame, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30), 2)
                cv2.putText(frame, f"walking: {state} ({effective_confidence:.2f})", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 220, 30) if state else (20, 40, 220), 2)
                writer.write(frame)
            frame_index += 1
    if frame_index:
        results.append(_summarize_second(current_second, bucket, walking_profile))
    capture.release()
    if writer:
        writer.release()
    with open(output_jsonl, "w", encoding="utf-8") as stream:
        for sample in results:
            stream.write(json.dumps(asdict(sample)) + "\n")
    return results


_POSE_EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32))


def _draw_task_landmarks(frame: np.ndarray, landmarks: Sequence[object]) -> None:
    """Minimal skeleton renderer for the modern MediaPipe Tasks result type."""
    h, w = frame.shape[:2]
    for a, b in _POSE_EDGES:
        if a < len(landmarks) and b < len(landmarks):
            pa, pb = landmarks[a], landmarks[b]
            cv2.line(frame, (round(pa.x * w), round(pa.y * h)), (round(pb.x * w), round(pb.y * h)), (0, 220, 255), 2)
    for landmark in landmarks:
        cv2.circle(frame, (round(landmark.x * w), round(landmark.y * h)), 2, (0, 100, 255), -1)


def _select_near_pose(
    poses: Sequence[Sequence[object]], frame_height: int, near_side_min_image_y: float,
    min_visibility: float,
) -> Optional[Sequence[object]]:
    """Choose the lowest reliable hip centre from MediaPipe's multi-pose result."""
    candidates: list[tuple[float, Sequence[object]]] = []
    for pose in poses:
        if len(pose) <= 24:
            continue
        left, right = pose[23], pose[24]
        if min(getattr(left, "visibility", 1.0), getattr(right, "visibility", 1.0)) < min_visibility:
            continue
        hip_y = (float(left.y) + float(right.y)) / 2 * frame_height
        if hip_y >= frame_height * near_side_min_image_y:
            candidates.append((hip_y, pose))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _is_walking(metrics: Optional[dict], confidence: float, profile: str) -> bool:
    if profile == "confidence":
        return confidence >= 0.55
    if not metrics or metrics["speed"] is None:
        return False
    # Calibrated against snippet's hand-labelled near-player walking intervals.
    # A high cadence distinguishes walking from static/recovery frames; the
    # upper speed guard removes homography/pose jumps.
    return metrics["speed"] <= 12.71 and metrics.get("stride_frequency_hz", 0.0) > 2.79


def _profile_confidence(metrics: Optional[dict], confidence: float, profile: str) -> float:
    if profile == "confidence":
        return confidence
    # Empirical precision for each leaf of the snippet21 tuning rule.
    return 0.6513761468 if _is_walking(metrics, confidence, profile) else 0.1858974359


def _summarize_second(second: int, bucket: Iterable[tuple[dict, float, Optional[float], Optional[float]]], walking_profile: str = "confidence") -> WalkingSample:
    entries = list(bucket)
    if not entries:
        return WalkingSample(second, False, 0.0, None, None, None, None, None)
    metrics, confidence, stride, separation = entries[-1]
    second_confidence = float(np.median([e[1] for e in entries]))
    second_metrics = {**metrics, "stride_frequency_hz": stride}
    return WalkingSample(second, _is_walking(second_metrics, second_confidence, walking_profile), second_confidence, metrics["speed"], stride, separation, metrics["hip"], metrics["ankle"])
