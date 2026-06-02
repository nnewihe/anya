"""
gait_analyzer.py

Kinematic knee-angle gait detection using MediaPipe Pose Landmarker (Tasks API).
Standalone module – no dependency on any other project code.

Usage
-----
    analyzer = GaitAnalyzer(fps=30.0)
    for frame in video_frames:
        walking = analyzer.update(frame)   # BGR ndarray
    analyzer.release()
"""

import math
import os
import time
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Model path — sits alongside this file
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (Pose topology — same as legacy API)
# ---------------------------------------------------------------------------
_LEFT_HIP,   _LEFT_KNEE,   _LEFT_ANKLE   = 23, 25, 27
_RIGHT_HIP,  _RIGHT_KNEE,  _RIGHT_ANKLE  = 24, 26, 28

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
VISIBILITY_THRESHOLD = 0.3    # minimum per-landmark confidence to accept

# Gait biomechanics (3-D world landmark angles)
MIN_FLEXION_ANGLE    = 155.0  # knee must dip below this (°) in swing phase
MAX_EXTENSION_ANGLE  = 155.0  # knee must rise above this (°) in stance phase

# Frequency: human walking cadence 1.5–2.5 Hz.
# Over a 3-second window that yields 4.5–7.5 full cycles per leg.
MIN_CYCLES_IN_5S     = 7
MAX_CYCLES_IN_5S     = 13

SMOOTHING_WINDOW     = 5      # frames for moving-average smoothing
WINDOW_SECONDS       = 3.0   # rolling analysis window
LANDMARK_LOST_GRACE  = 2.0   # seconds before resetting on lost landmarks

# Minimum fraction of a full window that must be filled before analysis runs
MIN_FILL_FRACTION    = 0.40

# Gait frequency thresholds (adjusted for real-world walking with smoothing)
MIN_CYCLES_ADJUSTED  = 3
MAX_CYCLES_ADJUSTED  = 14


# ---------------------------------------------------------------------------
# GaitAnalyzer
# ---------------------------------------------------------------------------

class GaitAnalyzer:
    """
    Maintains a MediaPipe PoseLandmarker (VIDEO mode) and a 5-second rolling
    knee-angle buffer.

    Call ``update(frame)`` each frame; returns ``True`` when a walking gait
    is detected, ``False`` otherwise.
    """

    def __init__(self, fps: float = 30.0, model_path: str = _MODEL_PATH) -> None:
        self._fps = fps
        self._min_frames = int(fps * WINDOW_SECONDS * MIN_FILL_FRACTION)
        self._frame_index = 0  # used to derive VIDEO-mode timestamps

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

        # Rolling buffer: (timestamp, left_angle, right_angle, both_visible)
        self._buffer: Deque[Tuple[float, float, float, bool]] = deque()

        self._last_landmark_time: Optional[float] = None
        self._is_walking = False
        self._last_walking_detected_time: Optional[float] = None

        # Frame statistics for diagnostics
        self._frame_count = 0
        self._success_count = 0

        # Debug state — populated on every update(), read by draw_debug()
        self.debug_info: dict = {
            "left_angle":    None,
            "right_angle":   None,
            "buffer_fill":   0.0,
            "peak_count":    0,
            "amplitude_ok":  False,
            "frequency_ok":  False,
            "alternation_ok": False,
            "is_walking":    False,
            # Normalized (0-1) landmark positions within the crop for skeleton drawing
            "landmarks":     {},   # {index: (x_norm, y_norm)}
        }

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
        self._frame_index += 1
        timestamp_ms = int(self._frame_index * 1000 / self._fps)
        self._frame_count += 1

        angles = self._extract_angles(frame, timestamp_ms)
        if angles is not None:
            self._success_count += 1

        if angles is None:
            if (
                self._last_landmark_time is not None
                and now - self._last_landmark_time > LANDMARK_LOST_GRACE
            ):
                elapsed = now - self._last_landmark_time
                print(f"[GAIT] Landmarks missing for {elapsed:.1f}s (> {LANDMARK_LOST_GRACE}s grace), "
                      f"clearing buffer with {len(self._buffer)} frames")
                self._buffer.clear()
                self._is_walking = False
            self.debug_info["left_angle"]  = None
            self.debug_info["right_angle"] = None
            self.debug_info["landmarks"]   = {}
            return self._is_walking

        left_angle, right_angle, both_visible = angles
        self._last_landmark_time = now

        buf_before = len(self._buffer)
        self._buffer.append((now, left_angle, right_angle, both_visible))

        # Prune entries that have aged out of the window
        cutoff = now - WINDOW_SECONDS
        pruned = 0
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()
            pruned += 1

        buf_after = len(self._buffer)

        # Debug: if buffer is growing slowly, log it
        if self._frame_count % int(self._fps * 2) == 0 and buf_after < 100:
            print(f"[GAIT DEBUG] F{self._frame_count}: buf {buf_before}→+1→{buf_after} "
                  f"(pruned {pruned})  cutoff age: {now - cutoff:.2f}s  "
                  f"oldest entry: {now - self._buffer[0][0]:.2f}s ago" if self._buffer else "empty")

        fill = len(self._buffer) / max(1, self._min_frames)
        self.debug_info["left_angle"]  = left_angle
        self.debug_info["right_angle"] = right_angle
        self.debug_info["buffer_fill"] = min(1.0, fill)

        if len(self._buffer) < self._min_frames:
            # Buffer not full yet; check if walking was detected within the window
            walking_in_window = (
                self._last_walking_detected_time is not None
                and now - self._last_walking_detected_time < WINDOW_SECONDS
            )
            self.debug_info["is_walking"] = walking_in_window
            return walking_in_window

        # Analyze the full buffer
        current_frame_is_walking = self._analyse_buffer()
        self.debug_info["is_walking"] = current_frame_is_walking

        # If walking detected, update timestamp
        if current_frame_is_walking:
            self._last_walking_detected_time = now

        # Return True if walking detected in the last 5 seconds
        walking_in_window = (
            self._last_walking_detected_time is not None
            and now - self._last_walking_detected_time < WINDOW_SECONDS
        )

        prev_walking = self._is_walking
        self._is_walking = walking_in_window

        # Log statistics every ~5 seconds
        if self._frame_count % int(self._fps * 5) == 0:
            det_rate = (self._success_count / self._frame_count * 100
                        if self._frame_count > 0 else 0.0)
            buf_timespan = (self._buffer[-1][0] - self._buffer[0][0]) if self._buffer else 0.0
            print(f"[GAIT STATS] Frames: {self._frame_count}  "
                  f"Valid detections: {self._success_count} ({det_rate:.0f}%)  "
                  f"Buffer: {len(self._buffer)}/90 ({fill*100:.0f}%)  "
                  f"Time span: {buf_timespan:.2f}s")

        if self._is_walking != prev_walking:
            state = "WALKING" if self._is_walking else "NOT WALKING"
            print(f"[GAIT] -> {state}  "
                  f"L={left_angle:.1f}°  R={right_angle:.1f}°  "
                  f"peaks={self.debug_info['peak_count']}  "
                  f"buf={len(self._buffer)}frames")

        return self._is_walking

    def reset(self) -> None:
        """Clear all accumulated state (does not re-create the landmarker)."""
        print(f"[GAIT] BUFFER RESET  had {len(self._buffer)} frames")
        self._buffer.clear()
        self._last_landmark_time = None
        self._last_walking_detected_time = None
        self._is_walking = False
        # frame_index is intentionally NOT reset — VIDEO mode timestamps must
        # be strictly increasing across the lifetime of the landmarker.

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def draw_debug(self, frame: np.ndarray, cx1: int, cy1: int, cx2: int, cy2: int) -> None:
        """
        Draw gait debug overlay onto *frame* (in-place).

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame being rendered.
        cx1, cy1, cx2, cy2 : int
            Pixel coordinates of the player crop within *frame*.
        """
        d = self.debug_info
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        # ---- 1. Skeleton: hip → knee → ankle for each leg -----------------
        _LIMBS = [
            (_LEFT_HIP,  _LEFT_KNEE),
            (_LEFT_KNEE, _LEFT_ANKLE),
            (_RIGHT_HIP,  _RIGHT_KNEE),
            (_RIGHT_KNEE, _RIGHT_ANKLE),
        ]
        _LEFT_COLOR  = (255, 180,   0)   # blue-ish
        _RIGHT_COLOR = (  0, 180, 255)   # orange-ish

        lm = d.get("landmarks", {})

        def to_frame(idx):
            if idx not in lm:
                return None
            nx, ny = lm[idx]
            return (int(cx1 + nx * crop_w), int(cy1 + ny * crop_h))

        for a_idx, b_idx in _LIMBS:
            pa = to_frame(a_idx)
            pb = to_frame(b_idx)
            if pa and pb:
                color = _LEFT_COLOR if a_idx in (_LEFT_HIP, _LEFT_KNEE) else _RIGHT_COLOR
                cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

        for idx in [_LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE,
                    _RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE]:
            pt = to_frame(idx)
            if pt:
                color = _LEFT_COLOR if idx in (_LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE) else _RIGHT_COLOR
                cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Angle labels next to each knee
        for knee_idx, angle_key in [(_LEFT_KNEE, "left_angle"), (_RIGHT_KNEE, "right_angle")]:
            pt = to_frame(knee_idx)
            angle = d.get(angle_key)
            if pt and angle is not None:
                cv2.putText(frame, f"{angle:.0f}", (pt[0] + 6, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"{angle:.0f}", (pt[0] + 6, pt[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # ---- 2. HUD panel --------------------------------------------------
        panel_x, panel_y = 10, 90   # top-left of panel (below state text)
        line_h = 20
        font   = cv2.FONT_HERSHEY_SIMPLEX
        fs     = 0.5

        walking   = d.get("is_walking", False)
        fill_pct  = d.get("buffer_fill", 0.0)
        left_a    = d.get("left_angle")
        right_a   = d.get("right_angle")
        peaks     = d.get("peak_count", 0)
        amp_ok    = d.get("amplitude_ok", False)
        freq_ok   = d.get("frequency_ok", False)
        alt_ok    = d.get("alternation_ok", False)

        # Detection rate: what fraction of frames had valid landmarks
        det_rate = (self._success_count / self._frame_count * 100
                    if self._frame_count > 0 else 0.0)

        hud_color = (0, 220, 0) if walking else (0, 80, 220)
        hud_label = "GAIT: WALKING" if walking else "GAIT: NOT WALKING"

        lines = [
            (hud_label, hud_color),
            (f"Buf: {fill_pct*100:.0f}%  Det: {det_rate:.0f}%  Peaks: {peaks}/{MIN_CYCLES_IN_5S}-{MAX_CYCLES_IN_5S}",
             (200, 200, 200)),
            (f"L: {left_a:.1f}  R: {right_a:.1f}" if left_a is not None else "L: --  R: --",
             (200, 200, 200)),
            (f"Amp:{'OK' if amp_ok else 'X'}  Freq:{'OK' if freq_ok else 'X'}  "
             f"Alt:{'OK' if alt_ok else 'X'}",
             (0, 220, 0) if (amp_ok and freq_ok and alt_ok) else (80, 80, 220)),
        ]

        # semi-transparent backing rect
        panel_h = line_h * len(lines) + 8
        panel_w = 260
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 4, panel_y - 14),
                      (panel_x + panel_w, panel_y + panel_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (text, color) in enumerate(lines):
            y = panel_y + i * line_h
            cv2.putText(frame, text, (panel_x, y), font, fs, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Angle geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _knee_angle(
        hip:   Tuple[float, ...],
        knee:  Tuple[float, ...],
        ankle: Tuple[float, ...],
    ) -> float:
        """
        Interior angle at *knee* formed by the hip–knee–ankle triplet.

        Accepts 2-D (x, y) or 3-D (x, y, z) tuples — the dot-product
        formula works in any dimensionality.  Passing 3-D world coordinates
        eliminates camera-projection foreshortening.

        Returns degrees in [0, 180].  A fully extended leg → ~180°;
        a flexed leg in mid-swing → ~60–90°.
        """
        a = tuple(h - k for h, k in zip(hip,   knee))
        b = tuple(v - k for v, k in zip(ankle, knee))
        dot   = sum(ai * bi for ai, bi in zip(a, b))
        mag_a = math.sqrt(sum(ai * ai for ai in a))
        mag_b = math.sqrt(sum(bi * bi for bi in b))
        if mag_a < 1e-9 or mag_b < 1e-9:
            return 180.0
        cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
        return math.degrees(math.acos(cos_theta))

    # ------------------------------------------------------------------
    # MediaPipe Tasks interface
    # ------------------------------------------------------------------

    def _extract_angles(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> Optional[Tuple[float, float, bool]]:
        """
        Run PoseLandmarker on *frame* and return
        ``(left_angle, right_angle, both_visible)`` or ``None``.

        Knee angles are computed from ``pose_world_landmarks`` (metric 3-D
        coordinates, hip-centred) so that the measurement is correct for all
        walking directions — including directly toward/away from the camera
        where 2-D projected angles are heavily foreshortened.

        The 2-D image landmarks are still used for the skeleton overlay.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            self.debug_info["landmarks"] = {}
            return None

        # 2-D image landmarks — used only for skeleton overlay
        lm_2d = result.pose_landmarks[0]

        # Cache all six leg landmarks (normalised crop coords) for skeleton overlay
        skel_indices = [
            _LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE,
            _RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE,
        ]
        self.debug_info["landmarks"] = {
            i: (lm_2d[i].x, lm_2d[i].y) for i in skel_indices
        }

        def vis_ok(*indices: int) -> bool:
            return all(
                (lm_2d[i].visibility or 0.0) >= VISIBILITY_THRESHOLD
                for i in indices
            )

        # 3-D world landmarks — used for angle computation (corrects foreshortening)
        # Fall back to 2-D if world landmarks are unavailable.
        if result.pose_world_landmarks:
            lm_3d = result.pose_world_landmarks[0]
            def pt(i: int) -> Tuple[float, ...]:
                return lm_3d[i].x, lm_3d[i].y, lm_3d[i].z
        else:
            lm_3d = None
            def pt(i: int) -> Tuple[float, ...]:  # type: ignore[misc]
                return lm_2d[i].x, lm_2d[i].y

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
        Count flexion events: transitions where signal crosses below MIN_FLEXION_ANGLE.
        More robust than strict local minima detection for noisy signals.
        """
        if len(signal) < 2:
            return 0

        count = 0
        was_above = signal[0] >= MIN_FLEXION_ANGLE

        for i in range(1, len(signal)):
            is_above = signal[i] >= MIN_FLEXION_ANGLE
            # Count each transition from "above threshold" to "below threshold"
            if was_above and not is_above:
                count += 1
            was_above = is_above

        return count

    # ------------------------------------------------------------------
    # Alternation check
    # ------------------------------------------------------------------

    @staticmethod
    def _legs_are_alternating(
        left: np.ndarray, right: np.ndarray, both_mask: np.ndarray
    ) -> bool:
        """
        Return True if left and right knees show sufficient alternation.

        Relaxed check: skip if bilateral visibility < 50%, or if signals
        are too similar (std < 5°). Otherwise require weak anti-phase
        (correlation < 0.2 instead of strictly < 0).
        """
        bilateral_fraction = both_mask.mean()

        # Skip check if we don't have good bilateral data
        if bilateral_fraction < 0.50:
            return True

        n = len(left)
        if n < 10:
            return True

        # Skip if legs move almost identically (mirrored/occluded)
        if np.std(left - right) < 5.0:
            return True

        # Check anti-phase: allow weak anti-phase (correlation < 0.2)
        # instead of requiring strict anti-phase (< 0.0)
        l_norm = left  - left.mean()
        r_norm = right - right.mean()
        denom = np.sqrt(np.dot(l_norm, l_norm) * np.dot(r_norm, r_norm))

        if denom < 1e-6:
            return True

        corr = float(np.dot(l_norm, r_norm)) / denom
        return corr < 0.2  # Allow weak positive correlation

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def _analyse_buffer(self) -> bool:
        """
        Return True when all gait criteria pass:

        1. Amplitude  – at least one leg flexes below MIN_FLEXION_ANGLE
                        and extends above MAX_EXTENSION_ANGLE.
        2. Frequency  – flexion-peak count in [MIN_CYCLES_IN_5S, MAX_CYCLES_IN_5S].
        3. Alternation – left and right knees are anti-phase (when both visible).
        """
        left_raw  = np.array([e[1] for e in self._buffer], dtype=float)
        right_raw = np.array([e[2] for e in self._buffer], dtype=float)
        both_mask = np.array([e[3] for e in self._buffer], dtype=float)

        left  = self._moving_average(left_raw,  SMOOTHING_WINDOW)
        right = self._moving_average(right_raw, SMOOTHING_WINDOW)

        # 1. Amplitude
        left_min, left_max   = left.min(), left.max()
        right_min, right_max = right.min(), right.max()

        flex_ok = not (left_min > MIN_FLEXION_ANGLE and right_min > MIN_FLEXION_ANGLE)
        ext_ok  = not (left_max < MAX_EXTENSION_ANGLE and right_max < MAX_EXTENSION_ANGLE)
        amp_ok  = flex_ok and ext_ok

        # 2. Frequency (using adjusted thresholds for smoothed signal)
        left_peaks  = self._count_flexion_peaks(left)
        right_peaks = self._count_flexion_peaks(right)
        best_peaks  = max(left_peaks, right_peaks)
        freq_ok = MIN_CYCLES_ADJUSTED <= best_peaks <= MAX_CYCLES_ADJUSTED

        # 3. Alternation
        alt_ok = self._legs_are_alternating(left, right, both_mask)

        self.debug_info["amplitude_ok"]   = amp_ok
        self.debug_info["frequency_ok"]   = freq_ok
        self.debug_info["alternation_ok"] = alt_ok
        self.debug_info["peak_count"]     = best_peaks

        # Log detailed analysis for debugging
        if self._frame_count % int(self._fps * 2) == 0:
            print(f"[GAIT ANALYSIS] "
                  f"Flex: L{left_min:.0f}° R{right_min:.0f}° (need <{MIN_FLEXION_ANGLE}°) = {flex_ok} | "
                  f"Ext: L{left_max:.0f}° R{right_max:.0f}° (need >{MAX_EXTENSION_ANGLE}°) = {ext_ok} | "
                  f"Peaks: {best_peaks} [{MIN_CYCLES_ADJUSTED}-{MAX_CYCLES_ADJUSTED}] = {freq_ok} | "
                  f"Alt: {alt_ok}")

        return amp_ok and freq_ok and alt_ok
