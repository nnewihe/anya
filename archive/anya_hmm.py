"""
anya_hmm.py
============
Probabilistic Hidden Markov Model (HMM) pipeline for tennis match phase tracking.

Replaces anya_transitions.py / TransitionEngine with a forward-pass Bayesian
tracker that maintains a posterior belief distribution over three hidden states
and updates it per frame via:

    belief[t] = normalise( emission[t]  *  (A[t].T  @  belief[t-1]) )

Three hidden states:
    0  WAITING      — dead time between points
    1  READY_ARMED  — server stationary at baseline, preparing to serve
    2  ACTIVE_RALLY — point live, from serve impact to point end

Serve trigger (READY_ARMED → ACTIVE_RALLY) is protected by a strict
2-second multi-sensor co-occurrence window: ball toss (E_toss), RNN/trophy
pose score (E_rnn), and an acoustic racket-impact spike (E_audio) must all
fire within the window for p_RA2AR to snap to 1.0.

Design commitments from spec §7:
    • A[2][2] = 1 − P_AR2W ≥ 0.98  — once active the system heavily resists stopping
    • No injected timeline offsets; emission degradation ends points naturally
    • All observation values normalised to float ∈ [0, 1] before HMM update
"""

import argparse
import csv
import math
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.ai.anya_base import AnyaTelemetryProvider, TelemetryFrame
from src.ai.utilities import create_highlights_ffmpeg, create_highlights_ffmpeg_multisource


# ── State indices ──────────────────────────────────────────────────────────────
S_WAITING      = 0
S_READY_ARMED  = 1
S_ACTIVE_RALLY = 2
STATE_NAMES    = ["WAITING", "ARMED", "ACTIVE"]   # must match AnyaTelemetryProvider names


# ══════════════════════════════════════════════════════════════════════════════
# Tunable parameters  (all top-level so they are easy to find and adjust)
# ══════════════════════════════════════════════════════════════════════════════

# ── Base HMM transition probabilities (per-frame) ─────────────────────────────
P_W2RA   = 0.010    # WAITING      → READY_ARMED
P_RA2W   = 0.008    # READY_ARMED  → WAITING
P_AR2W   = 0.002    # ACTIVE_RALLY → WAITING       A[2][2] = 0.998 ≥ 0.98 ✓

# ── Serve trigger (spec §5) ───────────────────────────────────────────────────
SERVE_WINDOW_SEC      = 2.0     # multi-sensor co-occurrence window
E_TOSS_THRESH         = 0.70    # min toss confidence to contribute
E_RNN_THRESH          = 0.20    # min trophy/RNN score to contribute
E_AUDIO_THRESH        = 0.20    # min audio spike confidence to contribute
SERVE_TRIGGER_PRODUCT = 0.04    # min(E_t * E_r * E_a) within window to fire trigger

# ── Ball velocity (perspective-normalised, see BallVelocityTracker) ───────────
BALL_VEL_HISTORY_SEC  = 0.40    # rolling window for velocity computation
BALL_VEL_FAST_THRESH  = 25.0    # normalised units/sec → ball is "fast"
BALL_VEL_SLOW_THRESH  = 8.0     # normalised units/sec → ball is "slow"
BALL_COAST_MAX_SEC    = 1.00    # max time to propagate position when YOLO misses
BALL_VEL_EMA_ALPHA    = 0.35    # EMA weight for velocity smoothing (higher = faster response)
SERVE_GRACE_SEC       = 3.0     # seconds after serve trigger where ACTIVE is held unconditionally

# ── Player kinematics (world-space: feet/sec) ──────────────────────────────────
PLAYER_VEL_HISTORY_SEC   = 2.0  # rolling window
PLAYER_VEL_ACTIVE_THRESH = 6.0   # ft/sec → "sprinting"
PLAYER_VEL_STILL_THRESH  = 3.0   # ft/sec → "walking / stationary"

# ── ARMED displacement backstop (Option A) ────────────────────────────────────
ARMED_DISPLACEMENT_WINDOW_SEC = 3.0   # window over which net travel is measured
ARMED_DISPLACEMENT_MAX_FT     = 4.0   # net displacement (ft) that signals walking away

# ── Back-wall retreat detection ───────────────────────────────────────────────
# Negative world-y = behind baseline (toward camera). If the player moves in
# that direction at sustained speed, the point is almost certainly over.
RETREAT_WINDOW_SEC      = 2.0   # rolling window to compute y-direction velocity
RETREAT_MIN_SPEED_FT_S  = 1.5    # ft/sec retreating before signal activates
RETREAT_STRONG_FT_S     = 3.5    # ft/sec retreating → full WAITING signal

# ── Ball occlusion ────────────────────────────────────────────────────────────
BALL_MISSING_TAU_SEC  = 4.0     # sustained absence threshold (spec Rule 4.1.4)

# ── Ball near-player detection ────────────────────────────────────────────────
BALL_NEAR_PLAYER_PAD  = 20      # pixels padding around player bounding box

# ── Refined observation signals (new trackers) ─────────────────────────────────
VEL_VARIANCE_WINDOW_SEC        = 2.0     # window for computing velocity COV
VEL_VARIANCE_HIGH_THRESHOLD    = 0.30    # COV > 0.30 → sporadic (ACTIVE)
VEL_VARIANCE_LOW_THRESHOLD     = 0.10    # COV < 0.10 → constant (WAITING)
BBOX_ASPECT_RATIO_CHANGE_PCTS  = 5.0    # 5% change in aspect ratio → postural adjust
BBOX_STABILITY_WINDOW_SEC      = 2.0    # window for tracking bbox changes
BALL_NEAR_PLAYER_DURATION_SEC  = 1.5    # prolonged near-player = WAITING signal
NET_TO_BASELINE_DISTANCE_FT    = 20.0   # net to baseline in world coordinates (approx)

# ── Audio spike detector ──────────────────────────────────────────────────────
AUDIO_LOW_HZ                  = 4000
AUDIO_HIGH_HZ                 = 8000
AUDIO_WINDOW_SEC              = 0.020  # energy integration window per audio bucket
AUDIO_NOISE_FLOOR_PERCENTILE  = 20     # p20 of band energy → maps to 0 (lower = more sensitive)
AUDIO_PEAK_PERCENTILE         = 99     # loudest 1% of impacts → maps to 1.0
AUDIO_SMOOTHING_FRAMES        = 3      # temporal smoothing across adjacent buckets

# ── Main-loop stride (skips inference on WAITING frames to save CPU) ──────────
WAITING_STRIDE = 1

# ── Dual-camera run filtering (identical to run_anya.py) ─────────────────────
MIN_SERVES_PER_RUN = 1
GAP_THRESHOLD_SEC  = 240.0


# ── Emission tables  [P(obs|WAITING), P(obs|READY_ARMED), P(obs|ACTIVE_RALLY)] ─
# Soft interpolation drivers; actual emission is a blend of these vectors
# weighted by continuous signal strengths.
_EMIT_FAST_BALL     = np.array([0.04, 0.01, 0.97], dtype=np.float64)
_EMIT_SLOW_BALL     = np.array([0.55, 0.20, 0.30], dtype=np.float64)  # relaxed: slow ball no longer strongly WAITING
_EMIT_BALL_NEAR     = np.array([0.60, 0.75, 0.20], dtype=np.float64)  # relaxed: ball near player still compatible with ACTIVE
_EMIT_BALL_MISS     = np.array([0.70, 0.50, 0.20], dtype=np.float64)  # relaxed: sustained miss decays ACTIVE gently
_EMIT_NO_BALL       = np.array([0.35, 0.33, 0.50], dtype=np.float64)  # relaxed: no ball slightly favours ACTIVE during rally

_EMIT_PLAYER_ACTIVE = np.array([0.05, 0.00, 0.95], dtype=np.float64)
_EMIT_PLAYER_WALK   = np.array([0.70, 0.00, 0.30], dtype=np.float64)  # relaxed: walking player still partly compatible with ACTIVE
_EMIT_PLAYER_ARMED  = np.array([0.28, 0.95, 0.01], dtype=np.float64)
_EMIT_PLAYER_NONE   = np.array([0.35, 0.33, 0.45], dtype=np.float64)  # relaxed: no player slightly favours ACTIVE


# ══════════════════════════════════════════════════════════════════════════════
# AudioSpikeDetector
# ══════════════════════════════════════════════════════════════════════════════

class AudioSpikeDetector:
    """
    Pre-loads audio from a video at startup and exposes a per-timestamp
    confidence score (0..1) for racket-ball impact events.

    Detection is based on RMS energy in the 4–8 kHz band (typical of a
    graphite racket striking a pressurised tennis ball). Energy is normalised
    against a percentile baseline so that sporadic loud hits produce
    confidence close to 1.0 and background noise produces ≈ 0.

    All parameters are tunable at construction. Falls back gracefully to
    always returning 0.0 when librosa / scipy are unavailable or the video
    has no audio track.
    """

    def __init__(
        self,
        video_path:            str,
        low_hz:                int   = AUDIO_LOW_HZ,
        high_hz:               int   = AUDIO_HIGH_HZ,
        window_sec:            float = AUDIO_WINDOW_SEC,
        noise_floor_pct:       int   = AUDIO_NOISE_FLOOR_PERCENTILE,
        peak_pct:              int   = AUDIO_PEAK_PERCENTILE,
        smoothing_frames:      int   = AUDIO_SMOOTHING_FRAMES,
    ):
        self.window_sec = window_sec
        self._energy: Dict[float, float] = {}
        self._enabled = False
        self._load(video_path, low_hz, high_hz, noise_floor_pct, peak_pct, smoothing_frames)

    def _load(self, path: str, low_hz: int, high_hz: int,
              noise_floor_pct: int, peak_pct: int, smooth: int):
        """
        Extract audio from the video file via ffmpeg (already a project dependency),
        then compute per-window RMS energy in the target frequency band using
        scipy alone — no librosa required.

        A temporary WAV file is created in the system temp directory and removed
        immediately after reading.
        """
        import subprocess
        import tempfile
        from scipy.signal import butter, sosfilt
        from scipy.io import wavfile

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", path,
                    "-vn",                        # no video
                    "-acodec", "pcm_s16le",       # 16-bit PCM
                    "-ar", "44100",               # resample to 44.1 kHz
                    "-ac", "1",                   # mono
                    tmp_path,
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace").strip()[-200:])

            sr, raw = wavfile.read(tmp_path)
            y = raw.astype(np.float64) / 32768.0   # normalise int16 → float

            nyq = sr / 2.0
            lo  = low_hz  / nyq
            hi  = min(high_hz / nyq, 0.999)
            sos = butter(4, [lo, hi], btype="bandpass", output="sos")
            y_band = sosfilt(sos, y)

            hop    = max(1, int(sr * self.window_sec))
            n_hops = len(y_band) // hop
            rms    = np.array(
                [np.sqrt(np.mean(y_band[i * hop:(i + 1) * hop] ** 2))
                 for i in range(n_hops)],
                dtype=np.float64,
            )

            if smooth > 1:
                rms = np.convolve(rms, np.ones(smooth) / smooth, mode="same")

            # Two-point normalization:
            #   noise_floor (p50 = median) → 0.0   ambient/silence
            #   peak        (p99)          → 1.0   loudest impacts
            # This removes the ambient energy baseline so genuine racket
            # strikes register close to 1.0 rather than ~0.15-0.20.
            noise_floor = np.percentile(rms, noise_floor_pct)
            peak        = np.percentile(rms, peak_pct)
            span        = peak - noise_floor
            if span > 1e-10:
                normed = np.clip((rms - noise_floor) / span, 0.0, 1.0)
            else:
                normed = np.zeros_like(rms)

            for i, conf in enumerate(normed):
                self._energy[round(i * self.window_sec, 6)] = float(conf)

            self._enabled = True
            n_spikes = int((normed >= 0.80).sum())
            print(f"[AUDIO] {n_hops} windows loaded — "
                  f"band {low_hz}–{high_hz} Hz, "
                  f"floor={noise_floor:.5f} (p{noise_floor_pct})  "
                  f"peak={peak:.5f} (p{peak_pct})  "
                  f"frames≥0.80: {n_spikes}")

        except Exception as exc:
            print(f"[AUDIO] Disabled — {exc}")
            print("[AUDIO] Serve trigger will use toss + RNN only (2-sensor mode)")

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @property
    def is_hardware_enabled(self) -> bool:
        """True when a real audio track was loaded; False when running in pass-through mode."""
        return self._enabled

    def confidence_at(self, timestamp: float) -> float:
        """
        Return 0..1 audio spike confidence at the given video timestamp.
        When audio could not be loaded, returns 1.0 as a neutral pass-through so
        that the serve trigger degrades to a 2-sensor (toss + RNN) check rather
        than being permanently blocked.
        """
        if not self._enabled:
            return 1.0   # pass-through: don't gate the serve trigger when audio unavailable
        bucket = round(round(timestamp / self.window_sec) * self.window_sec, 6)
        return self._energy.get(bucket, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# BallVelocityTracker
# ══════════════════════════════════════════════════════════════════════════════

class BallVelocityTracker:
    """
    Computes perspective-normalised ball velocity from successive detections.

        velocity = displacement_px / (elapsed_sec × mean_bbox_width_px)

    Dividing by bbox width compensates for perspective distortion: a ball near
    the far baseline has a smaller bbox than the same ball near the camera, so
    equal pixel displacement represents a larger physical distance there.

    Missed-detection robustness:
      - When YOLO returns no candidates, the last known position is propagated
        forward at the current velocity estimate for up to BALL_COAST_MAX_SEC.
        This keeps the velocity reading alive through brief occlusions.
      - Raw per-frame velocity is smoothed via EMA (alpha = BALL_VEL_EMA_ALPHA)
        to prevent single noisy detections from spiking or zeroing the output.
    """

    def __init__(
        self,
        history_sec:  float = BALL_VEL_HISTORY_SEC,
        coast_max:    float = BALL_COAST_MAX_SEC,
        ema_alpha:    float = BALL_VEL_EMA_ALPHA,
    ):
        self._buf: deque       = deque()   # (timestamp, cx, cy, bbox_width)
        self.history_sec       = history_sec
        self.coast_max         = coast_max
        self.ema_alpha         = ema_alpha
        self.last_velocity     = 0.0       # EMA-smoothed output
        self._raw_velocity     = 0.0       # instantaneous two-point velocity
        self._last_detect_time: float = -999.0

    def update(self, timestamp: float, candidates: list) -> float:
        if candidates:
            best = max(candidates, key=lambda c: c["conf"])
            bx1, by1, bx2, by2 = best["box"]
            cx  = (bx1 + bx2) / 2.0
            cy  = (by1 + by2) / 2.0
            bw  = max(float(bx2 - bx1), 1.0)
            self._buf.append((timestamp, cx, cy, bw))
            self._last_detect_time = timestamp
        else:
            # Coast: if within window, inject a synthetic position propagated
            # from the last known position using the current velocity estimate.
            gap = timestamp - self._last_detect_time
            if 0 < gap <= self.coast_max and self._buf:
                last_t, last_cx, last_cy, last_bw = self._buf[-1]
                dt_coast = timestamp - last_t
                if dt_coast > 0 and self._raw_velocity > 0:
                    # Direction is unknown so keep same position — this keeps
                    # the velocity tracker alive without inventing trajectory.
                    self._buf.append((timestamp, last_cx, last_cy, last_bw))

        cutoff = timestamp - self.history_sec
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if len(self._buf) < 2:
            # Decay EMA toward zero while no data
            self.last_velocity = self.last_velocity * (1.0 - self.ema_alpha)
            self._raw_velocity = self.last_velocity
            return self.last_velocity

        old, new = self._buf[0], self._buf[-1]
        dt = new[0] - old[0]
        if dt <= 0:
            return self.last_velocity

        dist_px         = math.hypot(new[1] - old[1], new[2] - old[2])
        avg_bw          = (old[3] + new[3]) / 2.0
        self._raw_velocity = (dist_px / avg_bw) / dt

        # EMA smooth
        self.last_velocity = (self.ema_alpha * self._raw_velocity
                              + (1.0 - self.ema_alpha) * self.last_velocity)
        return self.last_velocity

    def clear(self):
        self._buf.clear()
        self.last_velocity     = 0.0
        self._raw_velocity     = 0.0
        self._last_detect_time = -999.0


# ══════════════════════════════════════════════════════════════════════════════
# PlayerVelocityTracker
# ══════════════════════════════════════════════════════════════════════════════

class PlayerVelocityTracker:
    """
    Tracks near-player position in world coordinates (feet) and returns
    instantaneous velocity in ft/sec over a rolling window.

    World position is (wx, wy) from near_player_world. We track the x-center
    (lateral position along the baseline) and measure lateral speed in feet/sec.
    """

    def __init__(
        self,
        history_sec: float = PLAYER_VEL_HISTORY_SEC,
    ):
        self._buf: deque = deque()   # (timestamp, wx)  — x-center in feet
        self.history_sec   = history_sec
        self.last_velocity = 0.0

    def update(self, timestamp: float, world_pos) -> float:
        """
        Push a world-coordinate position (wx, wy) and return lateral velocity in ft/sec.
        """
        if world_pos is not None:
            wx = float(world_pos[0])  # x-center in feet
            self._buf.append((timestamp, wx))

        cutoff = timestamp - self.history_sec
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if len(self._buf) < 2:
            self.last_velocity = 0.0
            return 0.0

        old, new = self._buf[0], self._buf[-1]
        dt = new[0] - old[0]
        if dt <= 0:
            self.last_velocity = 0.0
            return 0.0

        displacement_ft = abs(new[1] - old[1])
        self.last_velocity = displacement_ft / dt
        return self.last_velocity

    def clear(self):
        self._buf.clear()
        self.last_velocity = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PlayerDisplacementTracker
# ══════════════════════════════════════════════════════════════════════════════

class PlayerDisplacementTracker:
    """
    Measures net player displacement in world coordinates (feet) over a rolling
    window.  Used as the ARMED → WAITING backstop: small serve-setup oscillations
    produce near-zero net displacement; genuine walking away accumulates to > 4 ft.

    World coordinates come from near_player_world (wx, wy) in feet.  When world
    coords are unavailable the tracker returns 0.0 (safe — won't trigger exit).
    """

    def __init__(
        self,
        window_sec: float = ARMED_DISPLACEMENT_WINDOW_SEC,
        max_ft:     float = ARMED_DISPLACEMENT_MAX_FT,
    ):
        self.window_sec = window_sec
        self.max_ft     = max_ft
        self._buf: deque = deque()   # (timestamp, wx, wy)

    def update(self, timestamp: float, world_pos) -> float:
        """Push a world-coord position and return net displacement in feet over the window."""
        if world_pos is not None:
            wx, wy = float(world_pos[0]), float(world_pos[1])
            self._buf.append((timestamp, wx, wy))

        cutoff = timestamp - self.window_sec
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if len(self._buf) < 2:
            return 0.0

        old = self._buf[0]
        new = self._buf[-1]
        return math.hypot(new[1] - old[1], new[2] - old[2])

    @property
    def is_walking_away(self) -> bool:
        """True when accumulated net displacement exceeds the threshold."""
        if len(self._buf) < 2:
            return False
        old = self._buf[0]
        new = self._buf[-1]
        return math.hypot(new[1] - old[1], new[2] - old[2]) >= self.max_ft

    def clear(self):
        self._buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
# VelocityVarianceTracker
# ══════════════════════════════════════════════════════════════════════════════

class VelocityVarianceTracker:
    """
    Measures velocity variance (coefficient of variation) over a 2-second window.
    Low COV (constant velocity) → walking (WAITING).
    High COV (sporadic velocity) → active play (ACTIVE).
    """

    def __init__(
        self,
        window_sec: float = VEL_VARIANCE_WINDOW_SEC,
        high_thresh: float = VEL_VARIANCE_HIGH_THRESHOLD,
        low_thresh: float = VEL_VARIANCE_LOW_THRESHOLD,
    ):
        self.window_sec = window_sec
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self._buf: deque = deque()  # (timestamp, velocity)

    def update(self, timestamp: float, velocity: float) -> float:
        """
        Push velocity and return COV (sigma / mean) over the window.
        Returns 0 if insufficient data.
        """
        self._buf.append((timestamp, velocity))
        cutoff = timestamp - self.window_sec
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if len(self._buf) < 2:
            return 0.0

        velocities = np.array([v for _, v in self._buf])
        mean = velocities.mean()
        if mean < 0.01:  # avoid division by near-zero
            return 0.0
        cov = velocities.std() / mean
        return float(cov)

    def is_sporadic(self) -> bool:
        """True if COV indicates sporadic movement (ACTIVE)."""
        cov = self.update(self._buf[-1][0] if self._buf else 0, 0)
        return cov > self.high_thresh

    def is_constant(self) -> bool:
        """True if COV indicates constant movement (WALKING)."""
        cov = self.update(self._buf[-1][0] if self._buf else 0, 0)
        return cov < self.low_thresh

    def clear(self):
        self._buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
# BboxAspectRatioTracker
# ══════════════════════════════════════════════════════════════════════════════

class BboxAspectRatioTracker:
    """
    Tracks changes in bounding-box aspect ratio (width / height).
    Changing aspect ratio (even if position static) indicates postural adjustment → ACTIVE.
    Static aspect ratio + static position → WAITING.
    """

    def __init__(
        self,
        window_sec: float = BBOX_STABILITY_WINDOW_SEC,
        change_pct_thresh: float = BBOX_ASPECT_RATIO_CHANGE_PCTS,
    ):
        self.window_sec = window_sec
        self.change_pct_thresh = change_pct_thresh
        self._buf: deque = deque()  # (timestamp, aspect_ratio)

    def update(self, timestamp: float, player_box) -> float:
        """
        Compute and push aspect ratio; return % change over window.
        """
        if player_box is not None:
            x1, y1, x2, y2 = player_box
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            aspect = w / h
            self._buf.append((timestamp, aspect))

        cutoff = timestamp - self.window_sec
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

        if len(self._buf) < 2:
            return 0.0

        old_aspect = self._buf[0][1]
        new_aspect = self._buf[-1][1]
        if old_aspect < 0.01:
            return 0.0
        pct_change = abs(new_aspect - old_aspect) / old_aspect * 100.0
        return float(pct_change)

    def has_changed(self) -> bool:
        """True if aspect ratio changed >5% over window."""
        pct_change = self.update(self._buf[-1][0] if self._buf else 0, None)
        return pct_change > self.change_pct_thresh

    def clear(self):
        self._buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
# TossDetector  (ported from anya_transitions._update_toss_detection)
# ══════════════════════════════════════════════════════════════════════════════

class TossDetector:
    """
    Standalone ball-toss detector. Returns a 0..1 confidence score each frame
    by tracking whether a ball candidate is moving upward above the player's head.
    Logic is preserved from TransitionEngine._update_toss_detection().
    """

    def __init__(self):
        self._consecutive: int             = 0
        self._gap:         int             = 0
        self._above_head:  bool            = False
        self._min_y_px:    Optional[float] = None
        self._last_ball:   Optional[dict]  = None

    def update(self, frame: TelemetryFrame, now: float) -> float:
        if not frame.toss_ball_candidates or frame.near_player_box is None:
            self._last_ball = None
            self._gap      += 1
            if self._gap > 3:
                self._consecutive = 0
                self._above_head  = False
            return 0.0

        ny1  = frame.near_player_box[1]
        best = max(frame.toss_ball_candidates, key=lambda x: x["conf"])
        bx1, by1, bx2, by2 = best["box"]
        cy = (by1 + by2) / 2.0

        is_moving_upward   = False
        is_ball_above_head = cy < ny1

        if self._last_ball is not None:
            dy  = cy - self._last_ball["y"]
            dtt = now - self._last_ball["time"]
            if dy < 0 and dtt > 0:
                is_moving_upward = True

        if is_ball_above_head:
            if self._min_y_px is None or cy < self._min_y_px:
                self._min_y_px = cy

        self._last_ball = {"y": cy, "time": now}

        if is_moving_upward and is_ball_above_head:
            self._gap         = 0
            self._consecutive += 1
            self._above_head  = True
        else:
            self._gap += 1
            if self._gap > 3:
                self._consecutive = 0
                self._above_head  = False

        if not self._above_head:
            return 0.0
        if self._consecutive >= 3:
            return 1.0
        if self._consecutive >= 2:
            return 0.7
        return 0.0

    @property
    def min_toss_y_px(self) -> Optional[float]:
        return self._min_y_px

    def reset(self):
        self._consecutive = 0
        self._gap         = 0
        self._above_head  = False
        self._min_y_px    = None
        self._last_ball   = None


# ══════════════════════════════════════════════════════════════════════════════
# ServeEventWindow
# ══════════════════════════════════════════════════════════════════════════════

class ServeEventWindow:
    """
    Maintains 2-second rolling buffers for E_toss, E_rnn, and E_audio.
    Computes p_RA2AR per spec §5:

        p_RA2AR(t) = g( max_{window} { E_toss * E_rnn * E_audio } )

    When the product of best-in-window values crosses SERVE_TRIGGER_PRODUCT,
    p_RA2AR snaps to 1.0. Sensors that don't meet their individual threshold
    contribute only 10% of their value to the product (soft gate).

    The window is only updated / queried when the HMM is in READY_ARMED state;
    it is reset on every ARMED entry to prevent stale events from firing
    immediately after a new serve setup begins.
    """

    def __init__(
        self,
        window_sec:      float = SERVE_WINDOW_SEC,
        toss_thresh:     float = E_TOSS_THRESH,
        rnn_thresh:      float = E_RNN_THRESH,
        audio_thresh:    float = E_AUDIO_THRESH,
        trigger_product: float = SERVE_TRIGGER_PRODUCT,
    ):
        self.window_sec      = window_sec
        self.toss_thresh     = toss_thresh
        self.rnn_thresh      = rnn_thresh
        self.audio_thresh    = audio_thresh
        self.trigger_product = trigger_product

        self._toss_buf:  deque = deque()    # (timestamp, score)
        self._rnn_buf:   deque = deque()
        self._audio_buf: deque = deque()

        self.last_scores: Dict[str, float] = {
            "e_toss": 0.0, "e_rnn": 0.0, "e_audio": 0.0, "p_ra2ar": 0.0,
        }

    def update(self, now: float, e_toss: float, e_rnn: float, e_audio: float,
               armed: bool = False) -> float:
        def _push(buf, t, v):
            buf.append((t, float(v)))
            cutoff = t - self.window_sec
            while buf and buf[0][0] < cutoff:
                buf.popleft()

        _push(self._toss_buf,  now, e_toss)
        _push(self._rnn_buf,   now, e_rnn)
        _push(self._audio_buf, now, e_audio)

        best_toss  = max((v for _, v in self._toss_buf),  default=0.0)
        best_rnn   = max((v for _, v in self._rnn_buf),   default=0.0)
        best_audio = max((v for _, v in self._audio_buf), default=0.0)

        # Soft threshold gate: sensors below threshold contribute only 10%
        t_c = best_toss  if best_toss  >= self.toss_thresh  else best_toss  * 0.10
        r_c = best_rnn   if best_rnn   >= self.rnn_thresh   else best_rnn   * 0.10
        a_c = best_audio if best_audio >= self.audio_thresh else best_audio * 0.10

        product = t_c * r_c * a_c

        if product >= self.trigger_product:
            p_ra2ar = 1.0
            if armed:
                print(f"[SERVE-WIN] TRIGGER FIRED  "
                      f"toss={best_toss:.2f}  rnn={best_rnn:.2f}  audio={best_audio:.2f}  "
                      f"product={product:.3f}")
        else:
            p_ra2ar = (product / self.trigger_product) * 0.15
            if armed and product > 0.01:
                print(f"[SERVE-WIN] partial  "
                      f"toss={best_toss:.2f}  rnn={best_rnn:.2f}  audio={best_audio:.2f}  "
                      f"product={product:.3f}/{self.trigger_product:.3f}  "
                      f"p_ra2ar={p_ra2ar:.4f}")

        self.last_scores = {
            "e_toss":  best_toss,
            "e_rnn":   best_rnn,
            "e_audio": best_audio,
            "p_ra2ar": p_ra2ar,
        }
        return p_ra2ar

    def reset(self):
        self._toss_buf.clear()
        self._rnn_buf.clear()
        self._audio_buf.clear()
        self.last_scores = {"e_toss": 0.0, "e_rnn": 0.0, "e_audio": 0.0, "p_ra2ar": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# ObservationComputer
# ══════════════════════════════════════════════════════════════════════════════

class ObservationComputer:
    """
    Translates raw TelemetryFrame data into a 3-vector emission probability:
        emission[s] = P(O_t | hidden state s)    for s ∈ {WAITING, READY_ARMED, ACTIVE_RALLY}

    The ball channel (spec §4.1) and player kinematics channel (spec §4.2) are
    computed independently and combined via element-wise product (independence
    assumption), then renormalised.

    Spec §7.3: all inputs are normalised to float ∈ [0, 1] before entering
    the emission calculation.
    """

    def __init__(self, fps: float):
        self.fps                  = fps
        self._ball_tracker        = BallVelocityTracker()
        self._player_tracker      = PlayerVelocityTracker()
        self._displacement_tracker = PlayerDisplacementTracker()
        self._vel_variance_tracker = VelocityVarianceTracker()
        self._bbox_aspect_tracker  = BboxAspectRatioTracker()
        self._retreat_buf: deque  = deque()   # (timestamp, world_y) for back-wall detection
        self._last_ball_time: float = 0.0
        self._ball_near_start_time: float = -999.0
        self._prev_in_serve_zone: bool = False  # tracks zone entry to reset displacement
        # Per-signal contributions to P(ACTIVE), each [0,1].
        # 1.0 = strongly supports ACTIVE, 0.0 = strongly supports WAITING.
        self.signal_debug: Dict[str, float] = {
            "ball_moving":      0.0,
            "ball_present":     0.0,
            "ball_not_near":    1.0,
            "vel_sporadic":     0.0,
            "bbox_changing":    0.0,
            "player_not_base":  0.5,
            "retreating":       0.0,   # 1.0 = walking toward back wall (WAITING)
            "ball_vel_raw":     0.0,
            "vel_cov_raw":      0.0,
            "bbox_pct_raw":     0.0,
            "retreat_speed":    0.0,
        }

    def compute(self, frame: TelemetryFrame, now: float) -> np.ndarray:
        ball_emit   = self._ball_emission(frame, now)
        player_emit = self._player_emission(frame)

        combined = ball_emit * player_emit
        total    = combined.sum()
        if total > 1e-10:
            combined /= total
        else:
            combined[:] = 1.0 / 3.0
        return combined

    # ── Ball channel ──────────────────────────────────────────────────────────

    def _ball_emission(self, frame: TelemetryFrame, now: float) -> np.ndarray:
        all_cands = (frame.active_ball_candidates or []) + (frame.toss_ball_candidates or [])
        ball_vel  = self._ball_tracker.update(now, all_cands)

        if all_cands:
            self._last_ball_time = now
        time_since_ball = now - self._last_ball_time

        has_ball       = bool(all_cands)
        sustained_miss = time_since_ball > BALL_MISSING_TAU_SEC

        # Rule 4.1.3 — ball coordinate overlaps player box
        ball_near = False
        if has_ball and frame.near_player_box is not None:
            x1, y1, x2, y2 = frame.near_player_box
            p = BALL_NEAR_PLAYER_PAD
            for c in all_cands:
                bx1, by1, bx2, by2 = c["box"]
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                if x1 - p <= cx <= x2 + p and y1 - p <= cy <= y2 + p:
                    ball_near = True
                    break

        # Track ball-near duration: prolonged = WAITING signal
        if ball_near:
            if self._ball_near_start_time < 0:
                self._ball_near_start_time = now
        else:
            self._ball_near_start_time = -999.0

        ball_near_duration = now - self._ball_near_start_time if self._ball_near_start_time > 0 else 0.0
        prolonged_ball_near = ball_near_duration > BALL_NEAR_PLAYER_DURATION_SEC

        # Rule 4.1.4 — sustained occlusion: strong WAITING signal when >2 seconds
        if sustained_miss:
            miss_alpha = float(np.clip(
                (time_since_ball - BALL_MISSING_TAU_SEC) / BALL_MISSING_TAU_SEC, 0.0, 1.0
            ))
            self.signal_debug["ball_present"]  = 0.0
            self.signal_debug["ball_moving"]   = 0.0
            self.signal_debug["ball_not_near"] = 1.0
            self.signal_debug["ball_vel_raw"]  = 0.0
            return np.array([
                min(0.80, 0.65 + 0.15 * miss_alpha),   # gentler WAITING ramp
                min(0.55, 0.45 + 0.10 * miss_alpha),
                max(0.10, 0.40 - 0.30 * miss_alpha),   # stays higher before decaying
            ], dtype=np.float64)

        if not has_ball:
            self.signal_debug["ball_present"]  = 0.5
            self.signal_debug["ball_moving"]   = 0.0
            self.signal_debug["ball_not_near"] = 1.0
            self.signal_debug["ball_vel_raw"]  = 0.0
            return _EMIT_NO_BALL.copy()

        # Rule 4.1.3 — ball held / bouncing near player (prolonged = strong WAITING signal)
        if ball_near:
            near_frac = float(np.clip(ball_near_duration / BALL_NEAR_PLAYER_DURATION_SEC, 0.0, 1.0))
            self.signal_debug["ball_present"]  = 1.0
            self.signal_debug["ball_moving"]   = 0.0
            self.signal_debug["ball_not_near"] = 1.0 - near_frac
            self.signal_debug["ball_vel_raw"]  = ball_vel
            if prolonged_ball_near:
                return np.array([0.88, 0.85, 0.05], dtype=np.float64)
            else:
                return _EMIT_BALL_NEAR.copy()

        # Rules 4.1.1 / 4.1.2 — continuous speed interpolation
        v_range     = max(BALL_VEL_FAST_THRESH - BALL_VEL_SLOW_THRESH, 1.0)
        speed_alpha = float(np.clip(
            (ball_vel - BALL_VEL_SLOW_THRESH) / v_range, 0.0, 1.0
        ))
        self.signal_debug["ball_present"]  = 1.0
        self.signal_debug["ball_moving"]   = speed_alpha
        self.signal_debug["ball_not_near"] = 1.0
        self.signal_debug["ball_vel_raw"]  = ball_vel
        return (_EMIT_SLOW_BALL + speed_alpha * (_EMIT_FAST_BALL - _EMIT_SLOW_BALL)).copy()

    # ── Player kinematics channel ─────────────────────────────────────────────

    def _player_emission(self, frame: TelemetryFrame) -> np.ndarray:
        now   = frame.timestamp
        p_box = frame.near_player_box
        p_vel = self._player_tracker.update(now, frame.near_player_world)
        self._displacement_tracker.update(now, frame.near_player_world)

        # Update refined observation signals
        vel_cov = self._vel_variance_tracker.update(now, p_vel)
        bbox_aspect_change = self._bbox_aspect_tracker.update(now, p_box)

        if p_box is None:
            return _EMIT_PLAYER_NONE.copy()

        # Rule 4.2.4 — player in serve zone (world-coord check)
        in_serve_zone = False
        world_y = None
        if frame.near_player_world is not None:
            wx, wy = frame.near_player_world
            world_y = wy
            in_serve_zone = READY_ZONE_Y_MIN <= wy <= READY_ZONE_Y_MAX

        if in_serve_zone:
            self.signal_debug["vel_sporadic"]    = 0.5
            self.signal_debug["bbox_changing"]   = 0.5
            self.signal_debug["player_not_base"] = 0.5
            self.signal_debug["vel_cov_raw"]     = vel_cov
            self.signal_debug["bbox_pct_raw"]    = bbox_aspect_change
            self._prev_in_serve_zone = True
            return _EMIT_PLAYER_ARMED.copy()

        self._prev_in_serve_zone = False
        # Velocity variance signal
        vel_sporadic_score = float(np.clip(vel_cov / max(VEL_VARIANCE_HIGH_THRESHOLD, 0.01), 0.0, 1.0))

        if vel_cov > VEL_VARIANCE_HIGH_THRESHOLD:
            emit_base = _EMIT_PLAYER_ACTIVE.copy()
        elif vel_cov < VEL_VARIANCE_LOW_THRESHOLD:
            emit_base = _EMIT_PLAYER_WALK.copy()
        else:
            v_range   = max(PLAYER_VEL_ACTIVE_THRESH - PLAYER_VEL_STILL_THRESH, 1.0)
            vel_alpha = float(np.clip(
                (p_vel - PLAYER_VEL_STILL_THRESH) / v_range, 0.0, 1.0
            ))
            emit_base = (_EMIT_PLAYER_WALK + vel_alpha * (_EMIT_PLAYER_ACTIVE - _EMIT_PLAYER_WALK))

        # Aspect ratio signal
        bbox_score = float(np.clip(bbox_aspect_change / max(BBOX_ASPECT_RATIO_CHANGE_PCTS, 0.01), 0.0, 1.0))

        if bbox_aspect_change > BBOX_ASPECT_RATIO_CHANGE_PCTS:
            emit_base = emit_base * np.array([0.8, 1.0, 1.2])
            emit_base /= emit_base.sum()
        elif p_vel < PLAYER_VEL_STILL_THRESH and bbox_aspect_change < 1.0:
            # Static position + static aspect = likely WAITING but not absolute
            emit_base = np.array([0.65, 0.10, 0.25])

        # ── Back-wall retreat signal ──────────────────────────────────────────
        # Track world_y over a short window and compute y-direction velocity.
        # Negative dy/dt (y decreasing) = moving behind baseline toward camera.
        retreat_speed = 0.0
        if world_y is not None:
            self._retreat_buf.append((now, world_y))
        cutoff = now - RETREAT_WINDOW_SEC
        while self._retreat_buf and self._retreat_buf[0][0] < cutoff:
            self._retreat_buf.popleft()

        if len(self._retreat_buf) >= 2:
            old_t, old_y = self._retreat_buf[0]
            new_t, new_y = self._retreat_buf[-1]
            dt = new_t - old_t
            if dt > 0:
                dy_dt = (new_y - old_y) / dt   # ft/sec; negative = toward back wall
                if dy_dt < -RETREAT_MIN_SPEED_FT_S:
                    retreat_speed = abs(dy_dt)
                    # Scale: min speed → 0.0 contribution, strong speed → 1.0
                    retreat_alpha = float(np.clip(
                        (retreat_speed - RETREAT_MIN_SPEED_FT_S)
                        / (RETREAT_STRONG_FT_S - RETREAT_MIN_SPEED_FT_S),
                        0.0, 1.0
                    ))
                    # Blend emission toward strong WAITING
                    _EMIT_RETREAT = np.array([0.92, 0.03, 0.05], dtype=np.float64)
                    emit_base = ((1.0 - retreat_alpha) * emit_base
                                 + retreat_alpha * _EMIT_RETREAT)
                    emit_base /= emit_base.sum()

        self.signal_debug["vel_sporadic"]    = vel_sporadic_score
        self.signal_debug["bbox_changing"]   = bbox_score
        self.signal_debug["player_not_base"] = 0.5   # baseline position signal removed
        self.signal_debug["retreating"]      = float(np.clip(
            retreat_speed / max(RETREAT_STRONG_FT_S, 0.01), 0.0, 1.0))
        self.signal_debug["vel_cov_raw"]     = vel_cov
        self.signal_debug["bbox_pct_raw"]    = bbox_aspect_change
        self.signal_debug["retreat_speed"]   = round(retreat_speed, 2)

        return emit_base

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def reset_ball(self):
        self._ball_tracker.clear()
        self._last_ball_time = 0.0

    def reset_player(self):
        self._player_tracker.clear()
        self._displacement_tracker.clear()
        self._vel_variance_tracker.clear()
        self._bbox_aspect_tracker.clear()
        self._ball_near_start_time = -999.0
        self._prev_in_serve_zone   = False


# ══════════════════════════════════════════════════════════════════════════════
# TennisHMM — forward Bayesian state tracker
# ══════════════════════════════════════════════════════════════════════════════

class TennisHMM:
    """
    Per-frame forward HMM update driver.

    The transition matrix A is rebuilt each frame so that p_RA2AR can be
    varied dynamically by the serve event window while keeping all other
    entries — especially A[2][2] = 1 − P_AR2W ≥ 0.98 — constant.

    Forbidden transitions are hard-coded to 0:
        A[WAITING][ACTIVE]     = 0    (spec §3, no direct W→AR jump)
        A[ACTIVE][READY_ARMED] = 0    (spec §3, can't arm during a rally)

    Public attributes written each step:
        belief          — posterior 3-vector [W, RA, AR]
        map_state_idx   — argmax(belief) as int
        map_state_name  — "WAITING" / "ARMED" / "ACTIVE"
        last_transition — timestamp of most recent MAP state change
        debug           — dict snapshot for HUD / CSV
    """

    # Seconds of pre-ARMED history to include in the sensor graph
    SCORE_HISTORY_SEC = 12.0

    def __init__(self, fps: float, output_dir: str = "."):
        self.fps         = fps
        self.output_dir  = output_dir   # directory for all file outputs (PNGs, etc.)
        self._toss       = TossDetector()
        self._serve_win  = ServeEventWindow()
        self._obs        = ObservationComputer(fps)

        self.belief               = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._prev_map: int       = S_WAITING

        self.active_start_time: float        = 0.0
        self.last_transition:   Optional[float] = None
        self._last_armed_exit_time: float    = -999.0   # for post-ARMED trigger grace
        self._serve_grace_until: float       = -999.0   # ACTIVE held unconditionally until this t

        # Rolling buffer: (timestamp, e_toss, e_rnn, e_audio) — kept for graphing
        self._score_history: deque = deque()

        self.debug: dict = {}

    # ── Transition matrix ─────────────────────────────────────────────────────

    def _build_A(self, p_ra2ar: float) -> np.ndarray:
        p_ra_stay = max(0.0, 1.0 - P_RA2W - p_ra2ar)
        return np.array([
            [1.0 - P_W2RA, P_W2RA,    0.0          ],  # WAITING row
            [P_RA2W,       p_ra_stay, p_ra2ar       ],  # READY_ARMED row
            [P_AR2W,       0.0,       1.0 - P_AR2W  ],  # ACTIVE_RALLY row
        ], dtype=np.float64)

    # ── Main step ─────────────────────────────────────────────────────────────

    def step(self, frame: TelemetryFrame, audio: AudioSpikeDetector) -> int:
        """Process one frame. Returns the MAP state index (0/1/2)."""
        now          = frame.timestamp
        current_map  = int(np.argmax(self.belief))

        # ── Serve event signals (only active during READY_ARMED) ─────────────
        e_toss  = self._toss.update(frame, now)
        e_rnn   = float(frame.trophy_score or 0.0)
        e_audio = audio.confidence_at(now)

        # Rolling sensor history (used for ARMED-entry graph)
        self._score_history.append((now, e_toss, e_rnn, e_audio))
        cutoff = now - self.SCORE_HISTORY_SEC
        while self._score_history and self._score_history[0][0] < cutoff:
            self._score_history.popleft()

        # Always feed the serve window so it builds up sensor history regardless
        # of whether we're currently ARMED — brief ARMED flickers (< 0.5s) would
        # otherwise never accumulate a full 2-second window.
        p_ra2ar_raw = self._serve_win.update(now, e_toss, e_rnn, e_audio,
                                              armed=(current_map == S_READY_ARMED)
                                              # grace logging handled separately below
                                              )

        # Grace period: treat as effectively ARMED if we recently exited ARMED
        # and the trigger fires during the serve window.  This handles the common
        # case where the HMM drifts back to WAITING before the racket-impact
        # audio spike arrives (~1s after toss).
        recently_armed = (
            current_map == S_WAITING
            and (now - self._last_armed_exit_time) < SERVE_WINDOW_SEC
        )
        effectively_armed = (current_map == S_READY_ARMED) or recently_armed

        if effectively_armed:
            p_ra2ar = p_ra2ar_raw
            # Pass armed=True for logging when actually in ARMED state (not grace)
            if recently_armed and p_ra2ar >= 1.0:
                ss = self._serve_win.last_scores
                print(f"[SERVE-WIN] TRIGGER FIRED (grace {now - self._last_armed_exit_time:.2f}s after ARMED exit)  "
                      f"toss={ss['e_toss']:.2f}  rnn={ss['e_rnn']:.2f}  audio={ss['e_audio']:.2f}  "
                      f"product={ss['e_toss'] * ss['e_rnn'] * ss['e_audio']:.3f}")

            # Toss height guard: block trigger if ball never cleared player head
            if p_ra2ar >= 1.0 and frame.near_player_box is not None:
                ny1       = frame.near_player_box[1]
                min_toss  = self._toss.min_toss_y_px
                if min_toss is not None and min_toss >= ny1:
                    print(f"[HMM] Toss height invalid (min_y={min_toss:.0f} >= ny1={ny1:.0f})"
                          " — serve trigger blocked")
                    p_ra2ar = 0.0
                    self._serve_win.reset()
                    self._toss.reset()

            # Spec §5: when all three sensors co-occur, the transition is
            # "absolute and immediate" — bypass the emission weighting entirely
            # and force the state.  Emission[AR] is often near-zero at the serve
            # moment (ball not yet detected as fast), which would block the
            # transition if we relied solely on the Bayesian update.
            if p_ra2ar >= 1.0:
                self.belief = np.array([0.0, 0.0, 1.0])
                self._on_transition(self._prev_map, S_ACTIVE_RALLY, now)
                self._prev_map = S_ACTIVE_RALLY
                ss = self._serve_win.last_scores
                self.debug = {
                    "belief_w":   0.0, "belief_ra": 0.0, "belief_ar": 1.0,
                    "e_toss":     round(ss["e_toss"],  3),
                    "e_rnn":      round(ss["e_rnn"],   3),
                    "e_audio":    round(ss["e_audio"], 3),
                    "p_ra2ar":    1.0,
                    "ball_vel":   round(self._obs._ball_tracker.last_velocity,   2),
                    "player_vel": round(self._obs._player_tracker.last_velocity, 2),
                }
                return S_ACTIVE_RALLY
        else:
            p_ra2ar = 0.0

        # ── Serve grace window: hold ACTIVE unconditionally ──────────────────
        # For SERVE_GRACE_SEC after the trigger fires, the system is committed
        # to the point being alive regardless of ball detection latency.
        if current_map == S_ACTIVE_RALLY and now < self._serve_grace_until:
            # Run observation compute to keep trackers warm, then discard result
            self._obs.compute(frame, now)
            self.belief = np.array([0.0, 0.0, 1.0])
            sd = self._obs.signal_debug
            self.debug = {
                "belief_w": 0.0, "belief_ra": 0.0, "belief_ar": 1.0,
                "e_toss": 0.0, "e_rnn": 0.0, "e_audio": 0.0, "p_ra2ar": 0.0,
                "ball_vel":   round(self._obs._ball_tracker.last_velocity,   2),
                "player_vel": round(self._obs._player_tracker.last_velocity, 2),
                "sig_ball_moving":      round(sd.get("ball_moving",      0.0), 3),
                "sig_ball_present":     round(sd.get("ball_present",     0.0), 3),
                "sig_ball_not_near":    round(sd.get("ball_not_near",    1.0), 3),
                "sig_vel_sporadic":     round(sd.get("vel_sporadic",     0.0), 3),
                "sig_bbox_changing":    round(sd.get("bbox_changing",    0.0), 3),
                "sig_player_not_base":  round(sd.get("player_not_base", 0.5), 3),
                "sig_ball_vel_raw":     round(sd.get("ball_vel_raw",     0.0), 1),
                "sig_vel_cov_raw":      round(sd.get("vel_cov_raw",      0.0), 3),
                "sig_bbox_pct_raw":     round(sd.get("bbox_pct_raw",     0.0), 1),
                "sig_retreating":       round(sd.get("retreating",       0.0), 3),
                "sig_retreat_speed":    round(sd.get("retreat_speed",    0.0), 2),
            }
            return S_ACTIVE_RALLY

        # ── Build transition matrix ───────────────────────────────────────────
        A = self._build_A(p_ra2ar)

        # ── Observation emission ──────────────────────────────────────────────
        emission = self._obs.compute(frame, now)    # normalised by ObservationComputer

        # ── Forward update: predict then correct ─────────────────────────────
        predicted = A.T @ self.belief
        updated   = emission * predicted

        # ── Normalise (spec §7.3) ─────────────────────────────────────────────
        total = updated.sum()
        if total > 1e-10:
            self.belief = updated / total
        else:
            self.belief = np.full(3, 1.0 / 3.0)

        # ── MAP state and transition detection ────────────────────────────────
        new_map = int(np.argmax(self.belief))
        if new_map != self._prev_map:
            self._on_transition(self._prev_map, new_map, now)
            self._prev_map = new_map

        # ── Debug snapshot ────────────────────────────────────────────────────
        ss = self._serve_win.last_scores
        sd = self._obs.signal_debug
        self.debug = {
            "belief_w":   round(float(self.belief[S_WAITING]),      4),
            "belief_ra":  round(float(self.belief[S_READY_ARMED]),  4),
            "belief_ar":  round(float(self.belief[S_ACTIVE_RALLY]), 4),
            "e_toss":     round(ss["e_toss"],  3),
            "e_rnn":      round(ss["e_rnn"],   3),
            "e_audio":    round(ss["e_audio"], 3),
            "p_ra2ar":    round(ss["p_ra2ar"], 4),
            "ball_vel":   round(self._obs._ball_tracker.last_velocity,   2),
            "player_vel": round(self._obs._player_tracker.last_velocity, 2),
            # Per-signal contributions to P(ACTIVE): 1.0=supports ACTIVE, 0.0=supports WAITING
            "sig_ball_moving":      round(sd.get("ball_moving",      0.0), 3),
            "sig_ball_present":     round(sd.get("ball_present",     0.0), 3),
            "sig_ball_not_near":    round(sd.get("ball_not_near",    1.0), 3),
            "sig_vel_sporadic":     round(sd.get("vel_sporadic",     0.0), 3),
            "sig_bbox_changing":    round(sd.get("bbox_changing",    0.0), 3),
            "sig_player_not_base":  round(sd.get("player_not_base", 0.5), 3),
            "sig_ball_vel_raw":     round(sd.get("ball_vel_raw",     0.0), 1),
            "sig_vel_cov_raw":      round(sd.get("vel_cov_raw",      0.0), 3),
            "sig_bbox_pct_raw":     round(sd.get("bbox_pct_raw",     0.0), 1),
            "sig_retreating":       round(sd.get("retreating",       0.0), 3),
            "sig_retreat_speed":    round(sd.get("retreat_speed",    0.0), 2),
        }
        return new_map

    @property
    def map_state_idx(self) -> int:
        return int(np.argmax(self.belief))

    @property
    def map_state_name(self) -> str:
        return STATE_NAMES[self.map_state_idx]

    # ── Transition side-effects ───────────────────────────────────────────────

    def _on_transition(self, old: int, new: int, now: float):
        print(f"[HMM] {STATE_NAMES[old]} → {STATE_NAMES[new]}  "
              f"t={now:.2f}s  belief={self.belief.round(3)}")

        if new == S_ACTIVE_RALLY:
            self.active_start_time   = now
            self.last_transition     = None
            self._serve_grace_until  = now + SERVE_GRACE_SEC
            self._obs.reset_ball()
            self._toss.reset()
            self._serve_win.reset()   # flush window so next serve starts clean

        if old == S_ACTIVE_RALLY:
            self.last_transition = now
            self._obs.reset_ball()
            self._obs.reset_player()
            self._print_active_exit_reason(now)

        if new == S_READY_ARMED:
            self._toss.reset()
            self._plot_armed_entry(now)

        if old == S_READY_ARMED and new == S_WAITING:
            # Record exit time so the serve trigger can still fire within the
            # serve window even if the HMM drifted out of ARMED before the
            # racket-impact audio spike arrives (~1s after toss).
            self._last_armed_exit_time = now
            # Serve attempt failed — hard-zero belief[AR] so partial-trigger
            # leakage (A[RA][AR] > 0) can't seed a spurious ACTIVE state.
            # This is physically certain: we KNOW no serve happened.
            self.belief[S_ACTIVE_RALLY] = 0.0
            total = self.belief.sum()
            if total > 1e-10:
                self.belief /= total
            # Do NOT reset the serve window here — sensor data (toss, rnn, audio)
            # accumulates across ARMED flickers; the 2-second rolling buffer
            # expires stale events naturally.

    def _print_active_exit_reason(self, now: float):
        """Print which signals drove the ACTIVE → WAITING transition."""
        # Read directly from signal_debug (fresh from this frame's compute() call).
        # self.debug is written AFTER _on_transition fires, so it would be stale here.
        sd       = self._obs.signal_debug
        duration = now - self.active_start_time

        ball_present = sd.get("ball_present", 1.0)
        ball_moving  = sd.get("ball_moving",  1.0)
        ball_near    = sd.get("ball_not_near", 1.0)
        vel_spor     = sd.get("vel_sporadic",  0.5)
        bbox_chg     = sd.get("bbox_changing", 0.5)
        retreating   = sd.get("retreating",    0.0)
        player_vel   = self._obs._player_tracker.last_velocity

        reasons = []

        if ball_present < 0.3:
            tsince = now - self._obs._last_ball_time
            reasons.append(f"ball missing {tsince:.1f}s (>{BALL_MISSING_TAU_SEC}s threshold)")
        elif ball_moving < 0.2:
            reasons.append(f"ball slow/stationary (vel={sd.get('ball_vel_raw', 0):.1f})")

        if ball_near < 0.35:
            reasons.append("ball near player (prolonged hold detected)")

        if vel_spor < 0.2:
            reasons.append(f"player velocity constant (COV={sd.get('vel_cov_raw', 0):.3f})")

        if bbox_chg < 0.2 and player_vel < PLAYER_VEL_STILL_THRESH:
            reasons.append(f"player static + bbox stable ({sd.get('bbox_pct_raw', 0):.1f}% change)")

        if retreating > 0.5:
            reasons.append(f"player retreating to back wall ({sd.get('retreat_speed', 0):.1f} ft/s)")

        if not reasons:
            reasons.append("combined emission drift (no single dominant signal)")

        w  = round(float(self.belief[S_WAITING]),      3)
        ra = round(float(self.belief[S_READY_ARMED]),  3)
        ar = round(float(self.belief[S_ACTIVE_RALLY]), 3)
        print(f"[HMM] ACTIVE→WAITING  t={now:.2f}s  duration={duration:.2f}s  "
              f"belief=[W={w} RA={ra} AR={ar}]")
        for r in reasons:
            print(f"         ↳ {r}")
        print(f"         signals: ball_present={ball_present:.2f}  ball_moving={ball_moving:.2f}  "
              f"not_near={ball_near:.2f}  vel_cov={sd.get('vel_cov_raw', 0):.3f}  "
              f"bbox={sd.get('bbox_pct_raw', 0):.1f}%  retreat={retreating:.2f}")

    def _plot_armed_entry(self, now: float):
        """
        Save a PNG of toss / RNN / audio scores for the last SCORE_HISTORY_SEC
        seconds leading up to this ARMED entry.  The file is written to
        self.output_dir as armed_entry_<timestamp>.png.

        Uses matplotlib if available; falls back to an ASCII sparkline if not.
        """
        history = list(self._score_history)
        if not history:
            return

        ts     = np.array([r[0] for r in history])
        toss   = np.array([r[1] for r in history])
        rnn    = np.array([r[2] for r in history])
        audio  = np.array([r[3] for r in history])
        t_rel  = ts - ts[-1]          # seconds relative to ARMED entry (ends at 0)

        try:
            import matplotlib
            matplotlib.use("Agg")     # non-interactive — safe in CV loop
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(t_rel, toss,  label="E_toss",  color="#2196F3", linewidth=1.4)
            ax.plot(t_rel, rnn,   label="E_rnn",   color="#FF9800", linewidth=1.4)
            ax.plot(t_rel, audio, label="E_audio", color="#4CAF50", linewidth=1.4)

            # Threshold reference lines
            ax.axhline(E_TOSS_THRESH,  color="#2196F3", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axhline(E_RNN_THRESH,   color="#FF9800", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axhline(E_AUDIO_THRESH, color="#4CAF50", linestyle="--", linewidth=0.8, alpha=0.6)

            ax.axvline(0, color="red", linestyle=":", linewidth=1.2, label="ARMED entry")
            ax.set_xlim(t_rel[0], 0.5)
            ax.set_ylim(-0.05, 1.10)
            ax.set_xlabel("Time relative to ARMED entry (s)")
            ax.set_ylabel("Score  (0–1)")
            ax.set_title(f"Sensor scores at ARMED entry  t={now:.2f}s")
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)

            fname = os.path.join(self.output_dir, f"armed_entry_{now:.2f}s.png")
            fig.tight_layout()
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            print(f"[HMM-PLOT] Saved → {fname}")

        except Exception as exc:
            # ASCII fallback — one row per sensor, 60 chars wide
            print(f"[HMM-PLOT] matplotlib unavailable ({exc}), printing ASCII sparkline")
            width = 60
            def _spark(values):
                bars = " ▁▂▃▄▅▆▇█"
                out  = []
                for v in values:
                    idx = int(np.clip(v, 0.0, 1.0) * (len(bars) - 1))
                    out.append(bars[idx])
                # Sample down / up to exactly `width` chars
                step = max(1, len(out) // width)
                return "".join(out[::step])[:width]

            t0_str = f"{t_rel[0]:.1f}s"
            print(f"[HMM-PLOT] t={now:.2f}s  ({t0_str} → 0.0s)")
            print(f"  toss  [{_spark(toss)}]  thresh={E_TOSS_THRESH:.2f}")
            print(f"  rnn   [{_spark(rnn)}]  thresh={E_RNN_THRESH:.2f}")
            print(f"  audio [{_spark(audio)}]  thresh={E_AUDIO_THRESH:.2f}")

    def reset(self):
        self.belief            = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._prev_map         = S_WAITING
        self.last_transition   = None
        self.active_start_time = 0.0
        self._toss.reset()
        self._serve_win.reset()
        self._obs.reset_ball()
        self._obs.reset_player()
        self._score_history.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Core segment-collection loop
# ══════════════════════════════════════════════════════════════════════════════

def _collect_segments_hmm(video_path, headless=False, start_frame=0, csv_path=None):
    """
    Run the HMM pipeline on a single video and return detected active segments.

    Parameters
    ----------
    video_path  : source video file
    headless    : suppress all OpenCV windows
    start_frame : seek to this frame before starting
    csv_path    : explicit CSV output path; defaults to <video_dir>/<stem>_hmm_telemetry.csv

    Returns
    -------
    active_segments : list of (start_sec, end_sec) in source-video time
    point_number    : total points detected
    csv_path        : path to the written telemetry CSV
    """
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    if csv_path is None:
        csv_path = os.path.join(video_dir, f"{video_stem}_hmm_telemetry.csv")

    _probe   = cv2.VideoCapture(video_path)
    orig_fps = _probe.get(cv2.CAP_PROP_FPS)
    _total   = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    _probe.release()
    if orig_fps <= 0 or orig_fps > 300:
        orig_fps = 30.0
    video_duration_sec = _total / orig_fps if _total > 0 else float("inf")

    # Initialise telemetry provider and HMM components
    provider         = AnyaTelemetryProvider(video_path)
    hmm              = TennisHMM(fps=provider.fps, output_dir=video_dir)
    audio            = AudioSpikeDetector(video_path)
    ready_band_poly  = _compute_ready_band_polygon(provider.H)

    # CSV writer
    _CSV_COLS = [
        "point", "frame", "timestamp", "hmm_state",
        "belief_waiting", "belief_armed", "belief_active",
        "e_toss", "e_rnn", "e_audio", "p_ra2ar",
        "ball_velocity", "player_velocity",
        "sig_ball_present", "sig_ball_vel_raw", "sig_ball_not_near",
        "sig_vel_cov_raw", "sig_retreat_speed", "sig_bbox_pct_raw",
        "serve_grace_active",
    ]
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLS)
    csv_writer.writeheader()

    video_time_offset     = start_frame / orig_fps
    active_segments       = []
    current_segment_start = 0.0
    last_telemetry_ts     = 0.0
    HIGHLIGHT_END_PAD_SEC = 1.0

    cap            = cv2.VideoCapture(video_path)
    point_number   = 0
    frame_in_point = 0
    prev_map_state = S_WAITING
    interrupted    = False

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[DEBUG] Seeking to frame {start_frame}")

    try:
        while cap.isOpened():
            success, orig_frame = cap.read()
            if not success:
                break

            frame = cv2.resize(orig_frame, (960, 540), interpolation=cv2.INTER_LINEAR)

            # Skip full inference on most WAITING frames to save CPU (same as run_anya.py)
            skip_inference = (
                provider.current_state == "WAITING"
                and provider.frame_counter % WAITING_STRIDE != 0
                and bool(provider.telemetry_history)
            )

            if skip_inference:
                provider.frame_counter += 1
                last = provider.telemetry_history[-1]
                telemetry = TelemetryFrame(
                    frame_id=provider.frame_counter,
                    timestamp=provider.frame_counter / provider.fps,
                    state="WAITING",
                    near_player_box=last.near_player_box,
                    near_player_world=last.near_player_world,
                    toss_ball_candidates=[],
                    active_ball_candidates=[],
                )
                provider.telemetry_history.append(telemetry)
            else:
                telemetry = provider.process_frame(frame)

            last_telemetry_ts = telemetry.timestamp

            # HMM forward update
            new_map = hmm.step(telemetry, audio)

            # Sync provider state on MAP transitions
            if new_map != prev_map_state:
                new_name = STATE_NAMES[new_map]
                old_name = STATE_NAMES[prev_map_state]

                if new_map == S_ACTIVE_RALLY:
                    point_number   += 1
                    frame_in_point  = 0
                    current_segment_start = video_time_offset + telemetry.timestamp

                elif prev_map_state == S_ACTIVE_RALLY:
                    # Use last_transition (set by HMM when it first left ACTIVE) as the
                    # natural segment end — no injected spacers (spec §7.2)
                    end_t = hmm.last_transition if hmm.last_transition is not None \
                            else telemetry.timestamp
                    padded_end = min(
                        video_time_offset + end_t + HIGHLIGHT_END_PAD_SEC,
                        video_duration_sec,
                    )
                    active_segments.append((current_segment_start, padded_end))

                provider.update_state(new_name)
                prev_map_state = new_map

            current_state = provider.current_state

            if current_state == "ACTIVE":
                frame_in_point += 1
            _write_csv_row_hmm(csv_writer, hmm, telemetry, point_number, frame_in_point, telemetry.timestamp)

            if not headless:
                render_frame_hmm(frame, telemetry, current_state, hmm,
                                 provider.exclusion_zones, provider.active_zone_polygon,
                                 ready_band_poly)
                debug_panel = render_debug_panel_hmm(current_state, hmm)
                cv2.imshow("Anya HMM Pipeline", frame)
                cv2.imshow("HMM Debug Panel", debug_panel)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERRUPTED] Ctrl-C — producing highlights from completed segments...")

    finally:
        if provider.current_state == "ACTIVE":
            padded_end = min(
                video_time_offset + last_telemetry_ts + HIGHLIGHT_END_PAD_SEC,
                video_duration_sec,
            )
            active_segments.append((current_segment_start, padded_end))

        cap.release()
        csv_file.close()
        if not headless:
            cv2.destroyAllWindows()

    print(f"[COLLECT-HMM] {os.path.basename(video_path)}: "
          f"{point_number} points, {len(active_segments)} segments")
    if interrupted:
        print("[COLLECT-HMM] (interrupted — segments cover completed detections only)")

    return active_segments, point_number, csv_path


# ══════════════════════════════════════════════════════════════════════════════
# Public pipeline entry points
# ══════════════════════════════════════════════════════════════════════════════

def run_anya_hmm_pipeline(video_path, output_path=None, headless=False, start_frame=0):
    """Single-camera HMM pipeline."""
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_hmm_highlights.mp4")

    # CSV always goes beside the input video, regardless of where output_path points
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    csv_path   = os.path.join(video_dir, f"{video_stem}_hmm_telemetry.csv")

    active_segments, point_number, _ = _collect_segments_hmm(
        video_path, headless, start_frame, csv_path=csv_path,
    )
    create_highlights_ffmpeg(video_path, active_segments, output_path)

    print(f"\n[DONE] Output video   : {output_path}")
    print(f"[DONE] Telemetry CSV  : {csv_path}")
    print(f"[DONE] Points recorded: {point_number}")


def run_anya_hmm_pipeline_dual(
    video_a,
    video_b,
    time_offset_sec=0.0,
    output_path=None,
    headless=False,
    start_frame_a=0,
):
    """
    Dual-camera HMM pipeline. Processes each video independently, filters
    serve runs, merges chronologically, and splices a single highlight reel.

    time_offset_sec : (video_A recording start) − (video_B recording start)
                      in seconds. Positive = video_A started after video_B.
    """
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_a))
        video_stem = os.path.splitext(os.path.basename(video_a))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_hmm_dual_highlights.mp4")

    print(f"\n{'='*60}")
    print(f"  DUAL HMM PIPELINE")
    print(f"  Video A : {os.path.basename(video_a)}")
    print(f"  Video B : {os.path.basename(video_b)}")
    print(f"  Offset  : {time_offset_sec:+.2f}s  (A start − B start)")
    print(f"{'='*60}\n")

    print(f"[DUAL-HMM] Processing Video A: {os.path.basename(video_a)}")
    segs_a, pts_a, csv_a = _collect_segments_hmm(video_a, headless, start_frame_a)

    print(f"\n[DUAL-HMM] Processing Video B: {os.path.basename(video_b)}")
    segs_b, pts_b, csv_b = _collect_segments_hmm(video_b, headless, start_frame=0)

    print(f"\n[DUAL-HMM] Filtering serve runs "
          f"(min {MIN_SERVES_PER_RUN}, gap ≤ {GAP_THRESHOLD_SEC:.0f}s) …")
    valid_a = _filter_by_serve_run(segs_a, "Video A")
    valid_b = _filter_by_serve_run(segs_b, "Video B")

    tagged_a = [(video_a, s, e, s + time_offset_sec) for s, e in valid_a]
    tagged_b = [(video_b, s, e, s)                   for s, e in valid_b]

    all_tagged = sorted(tagged_a + tagged_b, key=lambda x: x[3])
    if not all_tagged:
        print("\n[DUAL-HMM] No valid segments remain — no output produced.")
        return

    merged = [(src, s, e) for src, s, e, _ in all_tagged]
    create_highlights_ffmpeg_multisource(merged, output_path)

    print(f"\n[DONE] Output video      : {output_path}")
    print(f"[DONE] Video A CSV       : {csv_a}")
    print(f"[DONE] Video B CSV       : {csv_b}")
    print(f"[DONE] Video A points    : {pts_a}  ({len(valid_a)} valid segments)")
    print(f"[DONE] Video B points    : {pts_b}  ({len(valid_b)} valid segments)")
    print(f"[DONE] Total in output   : {len(merged)}")


# ── Serve-run helpers (identical to run_anya.py) ──────────────────────────────

def _group_segments_into_runs(segments, gap_threshold_sec=GAP_THRESHOLD_SEC):
    if not segments:
        return []
    runs        = []
    current_run = [segments[0]]
    for seg in segments[1:]:
        if seg[0] - current_run[-1][1] <= gap_threshold_sec:
            current_run.append(seg)
        else:
            runs.append(current_run)
            current_run = [seg]
    runs.append(current_run)
    return runs


def _filter_by_serve_run(segments, video_label, min_run=MIN_SERVES_PER_RUN,
                          gap_threshold_sec=GAP_THRESHOLD_SEC):
    runs           = _group_segments_into_runs(segments, gap_threshold_sec)
    valid_segments = []
    for i, run in enumerate(runs):
        if len(run) >= min_run:
            valid_segments.extend(run)
            print(f"[DUAL-HMM] {video_label}: run {i+1} — {len(run)} serves  (VALID)")
        else:
            print(f"[DUAL-HMM] {video_label}: run {i+1} — {len(run)} serves  "
                  f"(DISCARDED, fewer than {min_run})")
    return valid_segments


# ══════════════════════════════════════════════════════════════════════════════
# Ready-band polygon helper
# ══════════════════════════════════════════════════════════════════════════════

# World-space y bounds of the serve ready zone
# Positive y = inside the court; negative y = behind the near baseline
READY_ZONE_Y_MAX    =  1.0    # 1 ft inside the baseline (into the court)
READY_ZONE_Y_MIN    = -3.5    # 3.5 ft behind the baseline (outside the court)
READY_BAND_X_PAD_FT =  3.0    # lateral overhang beyond each doubles sideline


def _compute_ready_band_polygon(H: np.ndarray, court_width_ft: float = 27.0) -> Optional[np.ndarray]:
    """
    Project the serve ready band from world space into pixel space.

    The band runs from READY_ZONE_Y_MIN (3.5 ft behind baseline) to
    READY_ZONE_Y_MAX (1 ft inside the baseline) and extends laterally
    X_PAD feet past each sideline for full visibility.

    Returns an int32 (4, 1, 2) array suitable for cv2.fillPoly / cv2.polylines,
    or None if the homography cannot be inverted.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    x0 = -READY_BAND_X_PAD_FT
    x1 = court_width_ft + READY_BAND_X_PAD_FT

    # Four corners clockwise: court-side-left, court-side-right, back-right, back-left
    world_pts = np.array(
        [[x0, READY_ZONE_Y_MAX], [x1, READY_ZONE_Y_MAX],
         [x1, READY_ZONE_Y_MIN], [x0, READY_ZONE_Y_MIN]],
        dtype=np.float32,
    )
    pixel_pts = cv2.perspectiveTransform(world_pts.reshape(1, -1, 2), H_inv)
    return pixel_pts.reshape(-1, 1, 2).astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# CSV helper
# ══════════════════════════════════════════════════════════════════════════════

def _write_csv_row_hmm(csv_writer, hmm: TennisHMM, telemetry: TelemetryFrame,
                        point_number: int, frame_in_point: int, now: float = 0.0):
    d = hmm.debug
    csv_writer.writerow({
        "point":          point_number,
        "frame":          frame_in_point,
        "timestamp":      round(telemetry.timestamp, 4),
        "hmm_state":      hmm.map_state_name,
        "belief_waiting": d.get("belief_w",  0.0),
        "belief_armed":   d.get("belief_ra", 0.0),
        "belief_active":  d.get("belief_ar", 0.0),
        "e_toss":         d.get("e_toss",    0.0),
        "e_rnn":          d.get("e_rnn",     0.0),
        "e_audio":        d.get("e_audio",   0.0),
        "p_ra2ar":        d.get("p_ra2ar",   0.0),
        "ball_velocity":  d.get("ball_vel",  0.0),
        "player_velocity": d.get("player_vel", 0.0),
        "sig_ball_present":  round(d.get("sig_ball_present",  1.0), 3),
        "sig_ball_vel_raw":  round(d.get("sig_ball_vel_raw",  0.0), 2),
        "sig_ball_not_near": round(d.get("sig_ball_not_near", 1.0), 3),
        "sig_vel_cov_raw":   round(d.get("sig_vel_cov_raw",   0.0), 4),
        "sig_retreat_speed": round(d.get("sig_retreat_speed", 0.0), 2),
        "sig_bbox_pct_raw":  round(d.get("sig_bbox_pct_raw",  0.0), 2),
        "serve_grace_active": int(now < hmm._serve_grace_until),
    })


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def render_frame_hmm(frame, telemetry, state, hmm: TennisHMM,
                      exclusion_zones=None, active_zone_polygon=None,
                      ready_band_poly=None):
    """Debug overlay — reuses the same drawing logic as run_anya.py."""
    if state == "ACTIVE" and active_zone_polygon is not None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [active_zone_polygon], (144, 238, 144))
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        cv2.polylines(frame, [active_zone_polygon], True, (0, 200, 0), 1)

    # ── Ready band overlay (always visible for debugging) ─────────────────────
    # Colour logic:
    #   Cyan fill   — player is currently inside the band (WAITING or ARMED)
    #   Amber fill  — ARMED state but player is outside the band
    #   Dark grey   — ACTIVE state (band is inactive; shown faintly for context)
    if ready_band_poly is not None and state != "ACTIVE":
        in_band = False
        if telemetry.near_player_world is not None:
            _, wy = telemetry.near_player_world
            in_band = READY_ZONE_Y_MIN <= wy <= READY_ZONE_Y_MAX

        if state == "ARMED" and not in_band:
            fill_color   = (0, 165, 255)   # amber  — armed but player drifted out
            border_color = (0, 120, 200)
            alpha        = 0.25
        elif in_band:
            fill_color   = (255, 230, 0)   # cyan   — player inside band
            border_color = (180, 160, 0)
            alpha        = 0.22
        else:
            fill_color   = (160, 160, 160)  # grey  — WAITING, player outside
            border_color = (110, 110, 110)
            alpha        = 0.12

        overlay = frame.copy()
        cv2.fillPoly(overlay, [ready_band_poly], fill_color)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        cv2.polylines(frame, [ready_band_poly], True, border_color, 1, cv2.LINE_AA)

        # Label outer/inner edges
        far_pt  = tuple(ready_band_poly[3][0])
        near_pt = tuple(ready_band_poly[0][0])
        cv2.putText(frame, f"{abs(READY_ZONE_Y_MIN):.1f}ft back", far_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{READY_ZONE_Y_MAX:.1f}ft in", near_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, border_color, 1, cv2.LINE_AA)

    color = (0, 255, 0) if state == "ACTIVE" else (0, 255, 255) if state == "ARMED" else (180, 180, 180)
    b_ar  = hmm.debug.get("belief_ar", 0.0)
    cv2.putText(frame, f"HMM: {state}  [{b_ar:.2f}]", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if telemetry.near_player_box:
        x1, y1, x2, y2 = telemetry.near_player_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if state == "ACTIVE" and telemetry.far_player_box:
        x1, y1, x2, y2 = telemetry.far_player_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 105, 255), 2)
        cv2.putText(frame, "FAR", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 105, 255), 1, cv2.LINE_AA)

    if exclusion_zones:
        for x1, y1, x2, y2 in exclusion_zones:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    if state == "ARMED" and telemetry.z_box:
        x1, y1, x2, y2 = telemetry.z_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if state == "ACTIVE" and hmm is not None:
        trace = [(cx, cy) for _, cx, cy, _bw in hmm._obs._ball_tracker._buf]
        n = len(trace)
        if n >= 2:
            for i in range(1, n):
                age   = i / (n - 1)
                color = (0, int(120 * age), int(255 * age))
                pt1   = (int(trace[i - 1][0]), int(trace[i - 1][1]))
                pt2   = (int(trace[i][0]),     int(trace[i][1]))
                cv2.line(frame, pt1, pt2, color, max(1, int(3 * age)), cv2.LINE_AA)
        if n >= 1:
            cv2.circle(frame, (int(trace[-1][0]), int(trace[-1][1])),
                       5, (0, 200, 255), -1, cv2.LINE_AA)

        if telemetry.active_ball_candidates:
            for ball in telemetry.active_ball_candidates:
                bx1, by1, bx2, by2 = ball["box"]
                cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)),
                              (0, 255, 255), 2)


def render_debug_panel_hmm(state: str, hmm: TennisHMM) -> np.ndarray:
    panel = np.ones((600, 520, 3), dtype=np.uint8) * 240

    d     = hmm.debug
    x0    = 15
    bar_w = 200
    bar_h = 14
    lh    = 28
    fs    = 0.5

    cv2.putText(panel, f"HMM STATE: {state}", (x0, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)

    # ── Belief probability bars ───────────────────────────────────────────────
    y = 55
    belief_rows = [
        ("Waiting",      d.get("belief_w",  0.0), (100, 100, 200)),
        ("Ready-Armed",  d.get("belief_ra", 0.0), (0, 190, 220)),
        ("Active-Rally", d.get("belief_ar", 0.0), (0, 200, 80)),
    ]
    for label, val, col in belief_rows:
        cv2.putText(panel, f"{label}:", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (30, 30, 30), 1, cv2.LINE_AA)
        bx = x0 + 100
        cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + bar_w, y + 2), (180, 180, 180), -1)
        fill = int(val * bar_w)
        if fill > 0:
            cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + fill, y + 2), col, -1)
        cv2.putText(panel, f"{val:.3f}", (bx + bar_w + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
        y += lh

    # ── Serve sensor signals ──────────────────────────────────────────────────
    y += 8
    cv2.putText(panel, "SERVE SENSORS", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 2, cv2.LINE_AA)
    y += lh

    sensor_rows = [
        ("E_toss",  d.get("e_toss",  0.0), (0, 190, 220)),
        ("E_rnn",   d.get("e_rnn",   0.0), (0, 120, 255)),
        ("E_audio", d.get("e_audio", 0.0), (0, 180, 180)),
        ("p_RA2AR", d.get("p_ra2ar", 0.0), None),
    ]
    for label, val, col in sensor_rows:
        if col is None:
            col = (0, 220, 0) if val >= 0.55 else (0, 140, 255)
        cv2.putText(panel, f"{label}:", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 1, cv2.LINE_AA)
        bx = x0 + 75
        cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + bar_w, y + 2), (190, 190, 190), -1)
        fill = int(val * bar_w)
        if fill > 0:
            cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + fill, y + 2), col, -1)
        cv2.putText(panel, f"{val:.3f}", (bx + bar_w + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
        y += lh

    # ── Velocity readouts ─────────────────────────────────────────────────────
    y += 6
    bv = d.get("ball_vel",   0.0)
    pv = d.get("player_vel", 0.0)
    cv2.putText(panel, f"Ball vel (norm): {bv:.1f}   Player vel: {pv:.1f} ft/s",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)

    # ── Active contribution signals ───────────────────────────────────────────
    y += 22
    cv2.line(panel, (x0, y - 6), (panel.shape[1] - x0, y - 6), (180, 180, 180), 1)
    cv2.putText(panel, "ACTIVE CONTRIBUTIONS", (x0, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    y += lh

    # Each entry: (label, debug_key, raw_key, raw_unit, is_active_signal)
    # is_active_signal=True → green bar (supports ACTIVE)
    # is_active_signal=False → red bar (supports WAITING when high)
    sig_rows = [
        ("Ball moving",       "sig_ball_moving",     "sig_ball_vel_raw",  "u/s", True),
        ("Ball present",      "sig_ball_present",    None,                "",    True),
        ("Ball not near plyr","sig_ball_not_near",   None,                "",    True),
        ("Vel sporadic",      "sig_vel_sporadic",    "sig_vel_cov_raw",   "cov", True),
        ("Bbox changing",     "sig_bbox_changing",   "sig_bbox_pct_raw",  "%",   True),
        ("Plyr not baseline", "sig_player_not_base", None,                "",    True),
        ("Retreating",        "sig_retreating",      "sig_retreat_speed", "f/s", False),
    ]

    bar_w2 = 160
    for label, key, raw_key, unit, is_active in sig_rows:
        val = float(d.get(key, 0.0))
        # Color: green=supports ACTIVE, red=supports WAITING.
        # For is_active=True: high val → ACTIVE (green). For is_active=False: high val → WAITING (red).
        active_score = val if is_active else (1.0 - val)
        if active_score >= 0.65:
            col = (30, 160, 30)    # green — supports ACTIVE
        elif active_score >= 0.35:
            col = (0, 140, 220)    # blue/neutral
        else:
            col = (40, 40, 200)    # red — supports WAITING

        cv2.putText(panel, f"{label}:", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        bx = x0 + 148
        cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + bar_w2, y + 2), (190, 190, 190), -1)
        fill = int(val * bar_w2)
        if fill > 0:
            cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + fill, y + 2), col, -1)

        # Midpoint tick (neutral line)
        mid_x = bx + bar_w2 // 2
        cv2.line(panel, (mid_x, y - bar_h + 2), (mid_x, y + 2), (120, 120, 120), 1)

        pct_str = f"{val:.2f}"
        if raw_key and raw_key in d:
            pct_str += f" ({d[raw_key]:.1f}{unit})"
        cv2.putText(panel, pct_str, (bx + bar_w2 + 4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        y += lh - 2

    return panel


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anya HMM Tennis Point Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single camera:
    python -m src.ai.anya_hmm video.mp4
    python -m src.ai.anya_hmm video.mp4 --output out.mp4 --headless

  Dual camera:
    python -m src.ai.anya_hmm cam_a.mp4 cam_b.mp4 --time-offset 12.5
    python -m src.ai.anya_hmm cam_a.mp4 cam_b.mp4 --time-offset -5 --output spliced.mp4 --headless
""",
    )
    parser.add_argument(
        "input", nargs="+", metavar="VIDEO",
        help="1 video (single-camera) or 2 videos (dual-camera, A then B).",
    )
    parser.add_argument("--output", default=None,
                        help="Output MP4 path (default: derived from first input).")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display windows.")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start processing from this frame in video_A (default: 0).")
    parser.add_argument(
        "--time-offset", type=float, default=0.0, metavar="SECONDS",
        help="[Dual only] (video_A start) − (video_B start) in seconds (default: 0.0).",
    )
    args = parser.parse_args()

    if len(args.input) == 1:
        run_anya_hmm_pipeline(args.input[0], args.output, args.headless, args.start_frame)
    elif len(args.input) == 2:
        run_anya_hmm_pipeline_dual(
            args.input[0], args.input[1],
            time_offset_sec=args.time_offset,
            output_path=args.output,
            headless=args.headless,
            start_frame_a=args.start_frame,
        )
    else:
        parser.error("Provide 1 or 2 input videos.")
