"""
gait_analyzer.py

Kinematic knee-angle gait detection using MediaPipe Pose.
Standalone module – no dependency on any other project code.

Usage
-----
    analyzer = GaitAnalyzer(fps=30.0)
    for frame in video_frames:
        walking = analyzer.update(frame)   # BGR ndarray
    analyzer.release()
"""

import math
import time
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (Pose topology)
# ---------------------------------------------------------------------------
_LEFT_HIP,   _LEFT_KNEE,   _LEFT_ANKLE   = 23, 25, 27
_RIGHT_HIP,  _RIGHT_KNEE,  _RIGHT_ANKLE  = 24, 26, 28

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
VISIBILITY_THRESHOLD = 0.5    # minimum per-landmark confidence to accept

# Gait biomechanics
MIN_FLEXION_ANGLE    = 130.0  # knee must dip below this (°) in swing phase
MAX_EXTENSION_ANGLE  = 170.0  # knee must rise above this (°) in stance phase

# Frequency: human walking cadence 1.5–2.5 Hz.
# Over a 5-second window that yields 7.5–12.5 full cycles per leg.
MIN_CYCLES_IN_5S     = 7
MAX_CYCLES_IN_5S     = 13

SMOOTHING_WINDOW     = 5      # frames for moving-average smoothing
WINDOW_SECONDS       = 5.0   # rolling analysis window
LANDMARK_LOST_GRACE  = 1.0   # seconds before resetting on lost landmarks

# Minimum fraction of a full window that must be filled before analysis runs
MIN_FILL_FRACTION    = 0.60


# ---------------------------------------------------------------------------
# GaitAnalyzer
# ---------------------------------------------------------------------------

class GaitAnalyzer:
    """
    Maintains MediaPipe Pose state and a 5-second rolling knee-angle buffer.

    Call ``update(frame)`` each frame; returns ``True`` when a walking gait
    is detected, ``False`` otherwise.
    """

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = fps
        self._min_frames = int(fps * WINDOW_SECONDS * MIN_FILL_FRACTION)

        # MediaPipe Pose (re-entrant across frames)
        _mp = mp.solutions.pose
        self._pose = _mp.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Rolling buffer: (timestamp, left_angle, right_angle, both_visible)
        self._buffer: Deque[Tuple[float, float, float, bool]] = deque()

        self._last_landmark_time: Optional[float] = None
        self._is_walking = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> bool:
        """
        Process one BGR frame and return the current gait verdict.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (e.g. from cv2.VideoCapture.read()).

        Returns
        -------
        bool
            True if walking gait is currently detected.
        """
        now = time.monotonic()
        angles = self._extract_angles(frame)

        if angles is None:
            # Graceful degradation: hold verdict during grace period,
            # then reset if landmarks stay absent too long.
            if (
                self._last_landmark_time is not None
                and now - self._last_landmark_time > LANDMARK_LOST_GRACE
            ):
                self._buffer.clear()
                self._is_walking = False
            return self._is_walking

        left_angle, right_angle, both_visible = angles
        self._last_landmark_time = now

        self._buffer.append((now, left_angle, right_angle, both_visible))

        # Prune entries that have aged out of the window
        cutoff = now - WINDOW_SECONDS
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

        # Wait until the buffer is reasonably full before analysing
        if len(self._buffer) < self._min_frames:
            return False

        self._is_walking = self._analyse_buffer()
        return self._is_walking

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._buffer.clear()
        self._last_landmark_time = None
        self._is_walking = False

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()

    # ------------------------------------------------------------------
    # Angle geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _knee_angle(
        hip:   Tuple[float, float],
        knee:  Tuple[float, float],
        ankle: Tuple[float, float],
    ) -> float:
        """
        Interior angle at *knee* formed by the hip–knee–ankle triplet.

        Returns degrees in [0, 180].  A fully extended leg → ~180°;
        a flexed leg in mid-swing → ~120°.
        """
        ax, ay = hip[0]   - knee[0], hip[1]   - knee[1]
        bx, by = ankle[0] - knee[0], ankle[1] - knee[1]
        dot     = ax * bx + ay * by
        mag_a   = math.hypot(ax, ay)
        mag_b   = math.hypot(bx, by)
        if mag_a < 1e-9 or mag_b < 1e-9:
            return 180.0
        cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
        return math.degrees(math.acos(cos_theta))

    # ------------------------------------------------------------------
    # MediaPipe interface
    # ------------------------------------------------------------------

    def _extract_angles(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float, bool]]:
        """
        Run MediaPipe Pose on *frame* and return
        ``(left_angle, right_angle, both_visible)`` or ``None``.

        If only one leg is visible the invisible leg's angle is mirrored
        from the other; ``both_visible`` is set to ``False`` in that case.
        """
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if result.pose_landmarks is None:
            return None

        lm = result.pose_landmarks.landmark

        def vis_ok(*indices: int) -> bool:
            return all(lm[i].visibility >= VISIBILITY_THRESHOLD for i in indices)

        def pt(i: int) -> Tuple[float, float]:
            return lm[i].x, lm[i].y

        left_angle:  Optional[float] = None
        right_angle: Optional[float] = None

        if vis_ok(_LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE):
            left_angle = self._knee_angle(
                pt(_LEFT_HIP), pt(_LEFT_KNEE), pt(_LEFT_ANKLE)
            )

        if vis_ok(_RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE):
            right_angle = self._knee_angle(
                pt(_RIGHT_HIP), pt(_RIGHT_KNEE), pt(_RIGHT_ANKLE)
            )

        if left_angle is None and right_angle is None:
            return None

        both_visible = left_angle is not None and right_angle is not None

        # Mirror the invisible leg so the buffer stays a complete time-series
        if left_angle  is None: left_angle  = right_angle   # type: ignore[assignment]
        if right_angle is None: right_angle = left_angle

        return left_angle, right_angle, both_visible  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    @staticmethod
    def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
        """Causal moving average (no look-ahead)."""
        if len(signal) < window:
            return signal.copy()
        kernel = np.ones(window) / window
        padded = np.pad(signal, (window - 1, 0), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _count_flexion_peaks(signal: np.ndarray) -> int:
        """
        Local minima below MIN_FLEXION_ANGLE = one swing-phase event.
        """
        count = 0
        for i in range(1, len(signal) - 1):
            if signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
                if signal[i] < MIN_FLEXION_ANGLE:
                    count += 1
        return count

    @staticmethod
    def _count_extension_peaks(signal: np.ndarray) -> int:
        """
        Local maxima above MAX_EXTENSION_ANGLE = one stance-phase event.
        """
        count = 0
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                if signal[i] > MAX_EXTENSION_ANGLE:
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Alternation check
    # ------------------------------------------------------------------

    @staticmethod
    def _legs_are_alternating(
        left: np.ndarray, right: np.ndarray, both_mask: np.ndarray
    ) -> bool:
        """
        Return True if the two knee signals are roughly anti-phase.

        If fewer than 60 % of frames had both legs visible, this check is
        skipped (returns True) to avoid false negatives from occlusion.

        Anti-phase detection: the zero-lag cross-correlation should be
        negative (left flexing while right extends, and vice versa).
        """
        if both_mask.mean() < 0.60:
            return True  # not enough bilateral data – skip alternation check

        n = len(left)
        if n < 10:
            return True

        # Skip alternation check when one leg was mirrored the whole time
        # (signals nearly identical → std of difference ≈ 0)
        if np.std(left - right) < 1.0:
            return True

        l_norm = left  - left.mean()
        r_norm = right - right.mean()

        # Zero-lag cross-correlation
        zero_corr = float(np.dot(l_norm, r_norm))
        # Negative → out of phase → alternating legs
        return zero_corr < 0.0

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def _analyse_buffer(self) -> bool:
        """
        Analyse the rolling buffer and return True when all gait criteria pass:

        1. Amplitude  – at least one leg flexes below MIN_FLEXION_ANGLE
                        and extends above MAX_EXTENSION_ANGLE.
        2. Frequency  – flexion-peak count in [MIN_CYCLES_IN_5S, MAX_CYCLES_IN_5S].
        3. Alternation – left and right knees are anti-phase (when both visible).
        """
        left_raw   = np.array([e[1] for e in self._buffer], dtype=float)
        right_raw  = np.array([e[2] for e in self._buffer], dtype=float)
        both_mask  = np.array([e[3] for e in self._buffer], dtype=float)

        # Smooth to reduce MediaPipe jitter
        left  = self._moving_average(left_raw,  SMOOTHING_WINDOW)
        right = self._moving_average(right_raw, SMOOTHING_WINDOW)

        # ---- 1. Amplitude check -----------------------------------------
        # Require genuine flexion on at least one leg
        if left.min() > MIN_FLEXION_ANGLE and right.min() > MIN_FLEXION_ANGLE:
            return False

        # Require genuine extension on at least one leg
        if left.max() < MAX_EXTENSION_ANGLE and right.max() < MAX_EXTENSION_ANGLE:
            return False

        # ---- 2. Frequency / cycle-count check ---------------------------
        left_peaks  = self._count_flexion_peaks(left)
        right_peaks = self._count_flexion_peaks(right)
        best_peaks  = max(left_peaks, right_peaks)

        if not (MIN_CYCLES_IN_5S <= best_peaks <= MAX_CYCLES_IN_5S):
            return False

        # ---- 3. Alternation check ----------------------------------------
        if not self._legs_are_alternating(left, right, both_mask):
            return False

        return True