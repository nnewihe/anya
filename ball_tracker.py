"""
ball_tracker.py
===============
Trajectory-coherent single-ball tracker used by the Anya state machine to decide
when a tennis point is alive.

A tennis point has exactly one ball.  The ACTIVE state stays alive only while a
*moving* ball trace exists; the point ends the moment that trace stops or
disappears.  This module turns the per-frame whole-court YOLO detections
(noisy, identity-less, peppered with false positives) into a single coherent
trajectory and answers one question each frame:

    has_moving_trace -> is the ball still in play right now?

Pipeline per frame (`BallTrackManager.update`):
    1. Predict the confirmed track forward (three-model IMM: smooth CV + isotropic
       racket-impact model + anisotropic court-bounce model).  The gate widens
       automatically at contacts because the impact models' rising probability
       inflates the blended covariance P.
    2. Gate detections to the blended prediction and associate the nearest one
       -> IMM update.
    3. Feed unassociated detections to short-lived tentative seeds; a seed that
       collects enough *moving* hits is promoted to the confirmed track
       (this handles birth at serve and re-acquisition after a loss).
    4. The point is alive iff the confirmed track was detected within
       `miss_timeout_s` (coasting through brief occlusions is allowed) AND it
       has actually moved within the recent `motion_window_s` (so a ball that
       rolls to rest in view, or a stationary false ball, never sustains play).

The module depends only on numpy + filterpy so it can be unit-tested with
synthetic detection streams (see __main__) without any model weights or video.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter, IMMEstimator


# A detection is a pixel centre plus its YOLO confidence.
Detection = Tuple[float, float, float]   # (x, y, conf)


def _no_perspective(_y: float) -> float:
    """Default perspective scale: pixel thresholds are used as-is."""
    return 1.0


def make_image_row_perspective(frame_height: float, far_floor: float = 0.35) -> Callable[[float], float]:
    """
    Cheap perspective model needing only the analysis-frame height.

    Returns a multiplier in (far_floor, 1.0] that scales pixel thresholds by
    image row: ~1.0 near the bottom of the frame (near side, many px/ft) and
    shrinking toward `far_floor` at the top (far side, few px/ft).  A real
    ball's pixel motion shrinks the same way with distance, so gating and
    motion thresholds stay physically consistent across the court.
    """
    h = max(1.0, float(frame_height))

    def scale(y: float) -> float:
        return max(far_floor, min(1.0, float(y) / h))

    return scale


@dataclass
class TrackStatus:
    """Per-frame answer handed back to the state machine."""
    has_moving_trace: bool                       # is the point alive right now?
    state: str                                   # none | tentative | moving | coasting | stopped | lost | fading
    position: Optional[Tuple[float, float]]      # current (predicted) ball pixel position
    speed_px_s: float                            # IMM-blended speed magnitude, raw pixels/second
    time_since_detection: float                  # seconds since the last real detection
    coasting: bool                               # True while predicting through a detection gap
    ball_count: int                              # raw detections received this frame
    maneuver_prob: float                         # total non-flight probability (1 - μ₀); spikes at any contact
    racket_prob: float                           # IMM probability of the racket-impact model (μ₁)
    bounce_prob: float                           # IMM probability of the court-bounce model (μ₂)
    trace: List[Tuple[float, float]]             # recent trajectory polyline for overlay


class _ConfirmedTrack:
    """
    One coherent ball trajectory backed by a three-model IMM filter.

    Model 0 (smooth):  tight isotropic process noise — ball in free flight.
    Model 1 (racket):  loose *isotropic* noise on all state components — the
        racquet redirects the ball in any direction; both vx and vy can flip.
        Large Q_pos_x means the innovation covariance S_xx is wide, so even a
        full-reversal residual (|Δx|≈60 px) is accommodated without a gate miss.
    Model 2 (bounce):  *anisotropic* noise — tight Q_pos_x / Q_vx (horizontal
        motion continues through the bounce) but loose Q_pos_y / Q_vy (vertical
        velocity reverses).  S_xx stays narrow so the IMM gives M2 higher
        likelihood when z_x is small, distinguishing it from M1.

    Discrimination physics:
      • Racquet hit  (|z_x| large, z_y ≈ small) → M1 wins:
            S_xx_M1 >> S_xx_M2  → M1 far less penalised by the large x-residual.
      • Court bounce (z_x ≈ 0, |z_y| large)   → M2 wins:
            S_xx_M2 <  S_xx_M1  → M2 rewarded for correctly predicting x;
            S_yy identical (same Q_pos_y / Q_vy) → y neutral between M1 and M2.
    """

    def __init__(self, fps: float, x: float, y: float, vx: float, vy: float, t: float,
                 motion_window_s: float, corroboration_window_s: float,
                 q_smooth: float = 5.0,
                 q_racket: float = 300.0,
                 q_pos: float = 1.0,
                 q_bounce_vx: float = 20.0,
                 q_bounce_vy: float = 300.0,
                 mu_init=None, M=None,
                 perspective_scale: Optional[Callable[[float], float]] = None):
        self.fps = float(fps)
        self.dt = 1.0 / max(self.fps, 1e-6)
        self.motion_window_s = float(motion_window_s)
        self.corroboration_window_s = float(corroboration_window_s)
        # Perspective model: as the ball recedes to the far side it covers fewer
        # pixels per second, so its real per-frame motion shrinks.  We scale each
        # model's process noise Q linearly by this factor (floored at far_floor)
        # so the filter stiffens on the far side and stops straying into clutter
        # when the ball is small, while staying responsive on the near side.
        self.persp = perspective_scale or _no_perspective

        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1,       0],
                      [0, 0, 0,       1]], dtype=float)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        R  = np.eye(2, dtype=float) * 10.0
        P0 = np.eye(4, dtype=float) * 100.0
        x0 = np.array([[float(x)], [float(y)], [float(vx)], [float(vy)]], dtype=float)

        # Last *measured* (not predicted) position — used by the fallback gate.
        self._last_measured_pos: Tuple[float, float] = (float(x), float(y))

        # Derived position-component noise for the racket model.  Using q_racket/10
        # keeps the same scale ratio as the velocity noise (300 → 30) and ensures
        # S_xx_M1 is large enough to absorb full-reversal residuals without a gate miss.
        q_racket_pos = float(q_racket) / 10.0

        # Model 0 — smooth in-flight CV: tight on all state components.
        kf0 = KalmanFilter(dim_x=4, dim_z=2)
        kf0.F, kf0.H, kf0.R = F.copy(), H.copy(), R.copy()
        kf0.Q = np.diag([float(q_pos), float(q_pos), float(q_smooth), float(q_smooth)])
        kf0.P, kf0.x = P0.copy(), x0.copy()

        # Model 1 — racket impact: isotropic high-Q on position AND velocity.
        # A racquet strike can redirect the ball in any direction; both components
        # need wide uncertainty.  Large Q_pos makes S_xx wide → full-reversal
        # x-residuals are accommodated with minimal likelihood penalty.
        kf1 = KalmanFilter(dim_x=4, dim_z=2)
        kf1.F, kf1.H, kf1.R = F.copy(), H.copy(), R.copy()
        kf1.Q = np.diag([q_racket_pos, q_racket_pos, float(q_racket), float(q_racket)])
        kf1.P, kf1.x = P0.copy(), x0.copy()

        # Model 2 — court bounce: anisotropic Q.
        # Tight Q_pos_x / Q_vx (horizontal motion continues) → narrow S_xx → high
        # likelihood when z_x is small (ball keeps going sideways through the bounce).
        # Large Q_pos_y / Q_vy (vertical velocity reverses) → same S_yy as M1 →
        # y-residuals are neutral between M1 and M2, letting the x-component decide.
        kf2 = KalmanFilter(dim_x=4, dim_z=2)
        kf2.F, kf2.H, kf2.R = F.copy(), H.copy(), R.copy()
        kf2.Q = np.diag([float(q_pos), q_racket_pos,
                         float(q_bounce_vx), float(q_bounce_vy)])
        kf2.P, kf2.x = P0.copy(), x0.copy()

        _mu = np.array(mu_init, dtype=float) if mu_init is not None \
              else np.array([0.90, 0.05, 0.05])
        _M  = np.array(M, dtype=float) if M is not None \
              else np.array([[0.92, 0.04, 0.04],
                             [0.70, 0.25, 0.05],
                             [0.70, 0.05, 0.25]])

        self.imm = IMMEstimator([kf0, kf1, kf2], mu=_mu, M=_M)

        # Base (near-side, full-scale) process-noise matrices.  predict() rescales
        # each filter's live Q from these by the current perspective factor so the
        # noise tracks the ball's apparent pixel speed as it changes court depth.
        self._base_Q = [kf0.Q.copy(), kf1.Q.copy(), kf2.Q.copy()]

        self.last_detection_t = float(t)
        self.hits = 1
        # (t, x, y) using detected positions where available, predicted otherwise.
        self._history: Deque[Tuple[float, float, float]] = deque()
        self._history.append((float(t), float(x), float(y)))
        # Timestamps of *real* detections only, for corroboration scoring.
        self._det_times: Deque[float] = deque()
        self._det_times.append(float(t))

    def predict(self) -> None:
        # Linear perspective scaling: stiffen Q as the ball recedes (scale → far_floor).
        # Uses the current filtered y (depth proxy) before the state advances.
        scale = self.persp(self.y)
        for kf, baseQ in zip(self.imm.filters, self._base_Q):
            kf.Q = baseQ * scale
        self.imm.predict()

    def update(self, x: float, y: float, t: float) -> None:
        self.imm.update(np.array([[float(x)], [float(y)]], dtype=float))
        self.last_detection_t = float(t)
        self._last_measured_pos = (float(x), float(y))
        self.hits += 1
        self._det_times.append(float(t))

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.imm.x[0, 0]), float(self.imm.x[1, 0])

    @property
    def y(self) -> float:
        return float(self.imm.x[1, 0])

    def speed_px_s(self) -> float:
        return float(math.hypot(self.imm.x[2, 0], self.imm.x[3, 0]))

    def position_uncertainty(self) -> float:
        return float(np.trace(self.imm.P[:2, :2]))

    @property
    def maneuver_prob(self) -> float:
        """Total contact probability (1 - μ₀); spikes at any racquet hit or bounce."""
        return 1.0 - float(self.imm.mu[0])

    @property
    def racket_prob(self) -> float:
        """IMM weight of the racket-impact model (μ₁); spikes on full-direction reversals."""
        return float(self.imm.mu[1])

    @property
    def bounce_prob(self) -> float:
        """IMM weight of the court-bounce model (μ₂); spikes when only vy reverses."""
        return float(self.imm.mu[2])

    def record(self, t: float, now: float) -> None:
        """Append the current (predicted/updated) position and prune the window."""
        x, y = self.position
        self._history.append((float(t), x, y))
        cutoff = now - self.motion_window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        det_cutoff = now - self.corroboration_window_s
        while self._det_times and self._det_times[0] < det_cutoff:
            self._det_times.popleft()

    def recent_det_count(self) -> int:
        """Real detections inside the current motion window (corroboration)."""
        return len(self._det_times)

    def recent_span_px(self) -> float:
        """Max distance between the latest position and any position in the window."""
        if len(self._history) < 2:
            return 0.0
        _, lx, ly = self._history[-1]
        return max(math.hypot(hx - lx, hy - ly) for _, hx, hy in self._history)

    def trace(self) -> List[Tuple[float, float]]:
        return [(x, y) for _, x, y in self._history]

    def trace_with_time(self) -> List[Tuple[float, float, float]]:
        return list(self._history)


class _Tentative:
    """A candidate trajectory that has not yet earned a confirmed track."""

    __slots__ = ("points", "last_t")

    def __init__(self, x: float, y: float, t: float):
        self.points: List[Tuple[float, float, float]] = [(float(t), float(x), float(y))]
        self.last_t = float(t)

    def add(self, x: float, y: float, t: float) -> None:
        self.points.append((float(t), float(x), float(y)))
        self.last_t = float(t)

    @property
    def last_xy(self) -> Tuple[float, float]:
        return self.points[-1][1], self.points[-1][2]

    def expected_next(self, t: float) -> Tuple[float, float]:
        """Constant-velocity prediction of where the next point should land."""
        lx, ly = self.last_xy
        if len(self.points) < 2:
            return lx, ly
        t0, x0, y0 = self.points[-2]
        t1, x1, y1 = self.points[-1]
        seg_dt = t1 - t0
        if seg_dt <= 0:
            return lx, ly
        vx, vy = (x1 - x0) / seg_dt, (y1 - y0) / seg_dt
        dt = t - t1
        return lx + vx * dt, ly + vy * dt

    def span_px(self) -> float:
        if len(self.points) < 2:
            return 0.0
        _, lx, ly = self.points[-1]
        return max(math.hypot(px - lx, py - ly) for _, px, py in self.points)

    def velocity(self) -> Tuple[float, float]:
        """Mean velocity (px/s) across the seed, used to initialise the KF."""
        if len(self.points) < 2:
            return 0.0, 0.0
        t0, x0, y0 = self.points[0]
        t1, x1, y1 = self.points[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0, 0.0
        return (x1 - x0) / dt, (y1 - y0) / dt


class BallTrackManager:
    """
    Maintains a single confirmed ball trajectory and reports whether a moving
    trace is currently alive.  See module docstring for the per-frame pipeline.
    """

    def __init__(
        self,
        fps: float,
        *,
        perspective_scale: Optional[Callable[[float], float]] = None,
        gate_base_px: float = 50.0,
        gate_uncertainty_k: float = 0.6,
        seed_gate_px: float = 100.0,
        seed_coherence_px: float = 38.0,
        confirm_hits: int = 3,
        confirm_window_s: float = 0.6,
        miss_timeout_s: float = 2.0,       # coast this long without a detection before declaring lost
        motion_window_s: float = 0.5,
        move_thresh_px: float = 30.0,
        min_recent_dets: int = 3,
        corroboration_window_s: float = 2.0,  # alive while ≥min_recent_dets dets in this window
        # Hijack: allow a new seed to replace the confirmed track if no detection
        # has arrived for this many seconds.  Kept VERY SHORT (< corroboration_window_s)
        # so the serve ball can take over from the toss track almost immediately at
        # contact, while still protecting the track during brief legitimate occlusions.
        # The confirm_hits (3) + move_thresh seed requirement is the real anti-clutter
        # guard — a hijacking seed must itself be a coherent moving trajectory.
        hijack_after_s: float = 0.15,
        # IMM tuning knobs — three-model filter
        q_smooth: float = 5.0,        # M0: velocity noise for smooth free-flight
        q_maneuver: float = 300.0,    # M1: velocity noise for racket-impact model (isotropic)
        q_pos: float = 1.0,           # M0/M2: position-component noise (tight)
        q_bounce_vx: float = 20.0,    # M2: horizontal velocity noise (continues through bounce)
        q_bounce_vy: float = 300.0,   # M2: vertical velocity noise (flips at bounce)
        # Fallback gate: if the primary gate misses, search this multiple of gate_base_px
        # centered on the *last measured* position (contact point) to catch direction reversals.
        fallback_gate_k: float = 1.8,
        # Coast-expanding primary gate: while coasting through a detection gap (e.g. the
        # ball crossing/hitting the net on its way to the far side, where it goes small
        # and flickers out for several frames) the search radius around the *prediction*
        # grows by coast_gate_k · speed · time_since_detection — the distance the ball
        # could have travelled since it was last seen.  During normal per-frame tracking
        # time_since_detection ≈ dt so this term is negligible (far-side stray unaffected);
        # during a gap it opens just enough to re-catch the reappearing ball.  Capped so a
        # fast ball can't blow the gate open to the whole frame.
        coast_gate_k: float = 0.5,
        coast_gate_cap_px: float = 400.0,
    ):
        self.fps = float(fps)
        self.dt = 1.0 / max(self.fps, 1e-6)
        self.persp = perspective_scale or _no_perspective

        self.gate_base_px = float(gate_base_px)
        self.gate_uncertainty_k = float(gate_uncertainty_k)
        self.seed_gate_px = float(seed_gate_px)
        self.seed_coherence_px = float(seed_coherence_px)
        self.confirm_hits = int(confirm_hits)
        self.confirm_window_s = float(confirm_window_s)
        self.miss_timeout_s = float(miss_timeout_s)
        self.motion_window_s = float(motion_window_s)
        self.move_thresh_px = float(move_thresh_px)
        self.min_recent_dets = int(min_recent_dets)
        self.corroboration_window_s = float(corroboration_window_s)
        self.hijack_after_s = float(hijack_after_s)
        self.q_smooth = float(q_smooth)
        self.q_maneuver = float(q_maneuver)
        self.q_pos = float(q_pos)
        self.q_bounce_vx = float(q_bounce_vx)
        self.q_bounce_vy = float(q_bounce_vy)
        self.fallback_gate_k = float(fallback_gate_k)
        self.coast_gate_k = float(coast_gate_k)
        self.coast_gate_cap_px = float(coast_gate_cap_px)

        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.track: Optional[_ConfirmedTrack] = None
        self.tentatives: List[_Tentative] = []
        # Timestamp of the last real detection of the confirmed track — the
        # anchor the state machine rewinds to when the point dies.
        self.last_detection_time: Optional[float] = None
        self._last_trace: List[Tuple[float, float, float]] = []

    # ------------------------------------------------------------------
    def update(self, detections: List[Detection], now: float) -> TrackStatus:
        now = float(now)

        # 1. Predict the confirmed track forward.
        if self.track is not None:
            self.track.predict()

        # 2. Associate one detection to the confirmed track.
        used = [False] * len(detections)
        if self.track is not None and detections:
            tx, ty = self.track.position
            scale = self.persp(self.track.y)
            gate = self.gate_base_px * scale + self.gate_uncertainty_k * math.sqrt(
                max(self.track.position_uncertainty(), 0.0)
            )
            # Coast-expansion: widen the prediction-centred gate by the distance the
            # ball could have travelled since its last detection (speed · gap time),
            # capped.  Negligible during continuous tracking, opens during net-crossing
            # dropouts so the reappearing far-side ball is re-associated, not lost.
            tsd = now - self.track.last_detection_t
            gate += min(self.coast_gate_k * self.track.speed_px_s() * tsd,
                        self.coast_gate_cap_px)
            best_i, best_d = -1, gate
            for i, (dx, dy, _conf) in enumerate(detections):
                d = math.hypot(dx - tx, dy - ty)
                if d <= best_d:
                    best_d, best_i = d, i

            # Fallback gate: if the primary prediction-based gate failed, search around
            # the *last measured* position instead.  At a racquet hit or bounce the ball
            # doesn't teleport — it's still near the contact point even though velocity
            # reversed.  This guarantees the first post-maneuver detection reaches the
            # IMM, which then snaps mu[1]→1.0 and widens subsequent primary gates.
            if best_i < 0:
                lmx, lmy = self.track._last_measured_pos
                fb_gate = self.gate_base_px * self.fallback_gate_k * scale
                fb_best_i, fb_best_d = -1, fb_gate
                for i, (dx, dy, _conf) in enumerate(detections):
                    d = math.hypot(dx - lmx, dy - lmy)
                    if d <= fb_best_d:
                        fb_best_d, fb_best_i = d, i
                best_i = fb_best_i

            if best_i >= 0:
                dx, dy, _ = detections[best_i]
                self.track.update(dx, dy, now)
                used[best_i] = True
                self.last_detection_time = now

        # 3. Feed leftovers to tentative seeds and try to promote one.
        for i, (dx, dy, _conf) in enumerate(detections):
            if not used[i]:
                self._feed_tentative(dx, dy, now)
        self._prune_tentatives(now)
        self._try_promote(now)

        # 4. Record the (predicted/updated) position into the motion window.
        if self.track is not None:
            self.track.record(now, now)

        status = self._status(len(detections), now)

        # Drop a confirmed track once it is lost so a new seed can take over.
        if status.state == "lost":
            self.track = None

        return status

    # ------------------------------------------------------------------
    def _feed_tentative(self, x: float, y: float, now: float) -> None:
        # Match against each seed's constant-velocity *prediction*, not just its
        # last point.  A one-point seed uses a wide gate (the ball can be fast
        # and detections sparse); a seed with velocity uses a tight coherence
        # gate, so random false positives can't extend it into a fake trajectory.
        scale = self.persp(y)
        best, best_d = None, float("inf")
        for tnt in self.tentatives:
            ex, ey = tnt.expected_next(now)
            gate = (self.seed_gate_px if len(tnt.points) < 2 else self.seed_coherence_px) * scale
            d = math.hypot(x - ex, y - ey)
            if d <= gate and d < best_d:
                best_d, best = d, tnt
        if best is not None:
            best.add(x, y, now)
        else:
            self.tentatives.append(_Tentative(x, y, now))

    def _prune_tentatives(self, now: float) -> None:
        self.tentatives = [
            t for t in self.tentatives if now - t.points[0][0] <= self.confirm_window_s
        ]

    def _try_promote(self, now: float) -> None:
        # Allow promotion when there is no confirmed track, OR when the confirmed
        # track has received no detection for longer than hijack_after_s.
        #
        # hijack_after_s is kept SHORT (0.4 s default) so the serve ball can
        # take over from the toss-ball track quickly: at contact the ball's
        # velocity explodes — 100–200 px/frame, outside both gates — so the
        # toss track stalls immediately.  Without hijacking, the stalled track
        # would block promotion for the full miss_timeout_s (2 s).
        #
        # hijack_after_s < corroboration_window_s, so brief occlusions
        # (where real detections still sit inside the corroboration window) still
        # protect the track from premature replacement.  No seed can be promoted
        # during a genuine occlusion unless detections actually stop arriving.
        if self.track is not None:
            if now - self.track.last_detection_t <= self.hijack_after_s:
                return
        promotable = [
            t for t in self.tentatives
            if len(t.points) >= self.confirm_hits
            and t.span_px() > self.move_thresh_px * self.persp(t.last_xy[1])
        ]
        if not promotable:
            return
        best = max(promotable, key=lambda t: (len(t.points), t.span_px()))
        x, y = best.last_xy
        vx, vy = best.velocity()
        t_last = best.points[-1][0]
        self.track = _ConfirmedTrack(
            self.fps, x, y, vx, vy, t_last, self.motion_window_s,
            self.corroboration_window_s,
            q_smooth=self.q_smooth, q_racket=self.q_maneuver,
            q_pos=self.q_pos, q_bounce_vx=self.q_bounce_vx, q_bounce_vy=self.q_bounce_vy,
            perspective_scale=self.persp,
        )
        # Backfill the motion window and corroboration from the seed so the new
        # track immediately reads as a moving, well-corroborated trajectory.
        self.track._history.clear()
        self.track._det_times.clear()
        for (t, px, py) in best.points:
            self.track._history.append((t, px, py))
            self.track._det_times.append(t)
        self.last_detection_time = t_last
        self.tentatives.remove(best)

    # ------------------------------------------------------------------
    def _status(self, ball_count: int, now: float) -> TrackStatus:
        if self.track is None:
            self._last_trace = []
            return TrackStatus(
                has_moving_trace=False, state="none", position=None,
                speed_px_s=0.0, time_since_detection=0.0, coasting=False,
                ball_count=ball_count, maneuver_prob=0.0,
                racket_prob=0.0, bounce_prob=0.0, trace=[],
            )

        tsd = now - self.track.last_detection_t
        lost = tsd > self.miss_timeout_s
        coasting = (not lost) and tsd > (1.5 * self.dt)
        moving = self.track.recent_span_px() > self.move_thresh_px * self.persp(self.track.y)
        corroborated = self.track.recent_det_count() >= self.min_recent_dets

        self._last_trace = self.track.trace_with_time()

        if lost:
            state = "lost"
            alive = False
        elif not corroborated:
            # Too few real detections lately — a real rally ball is seen most
            # frames, so this is clutter or a dying tail, not live play.
            state = "fading"
            alive = False
        elif not moving:
            state = "stopped"
            alive = False
        else:
            state = "coasting" if coasting else "moving"
            alive = True

        return TrackStatus(
            has_moving_trace=alive,
            state=state,
            position=self.track.position,
            speed_px_s=self.track.speed_px_s(),
            time_since_detection=tsd,
            coasting=coasting,
            ball_count=ball_count,
            maneuver_prob=self.track.maneuver_prob,
            racket_prob=self.track.racket_prob,
            bounce_prob=self.track.bounce_prob,
            trace=self.track.trace(),
        )

    # ------------------------------------------------------------------
    def trace_points(self) -> List[Tuple[float, float, float]]:
        """(t, x, y) trajectory for the visual overlay; empty when no track."""
        return list(self._last_trace)


# =====================================================================
# Synthetic self-test — runs with numpy + filterpy only (no weights/video).
#   python ai/ball_tracker.py
# =====================================================================
def _run_self_test() -> int:
    fps = 30.0
    dt = 1.0 / fps

    def run(stream):
        """stream: list of (list_of_detections) per frame -> list of TrackStatus."""
        mgr = BallTrackManager(fps)
        out = []
        t = 0.0
        for dets in stream:
            out.append(mgr.update(dets, t))
            t += dt
        return out

    failures = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    # --- Scenario 1: a ball moving across the court stays alive. ---
    moving = []
    x, y = 100.0, 300.0
    for _ in range(40):
        x += 18.0  # ~540 px/s — a realistic groundstroke near side
        moving.append([(x, y, 0.9)])
    res = run(moving)
    check("moving ball becomes a live trace", any(s.has_moving_trace for s in res))
    check("moving ball is alive at the end", res[-1].has_moving_trace)
    check("moving ball reports state 'moving'", res[-1].state == "moving")

    # --- Scenario 2: ball moves, then stops dead in view -> trace ends. ---
    stop = []
    x = 100.0
    for _ in range(25):
        x += 18.0
        stop.append([(x, 300.0, 0.9)])
    for _ in range(25):                      # parked in place, still detected
        stop.append([(x, 300.0, 0.9)])
    res = run(stop)
    check("ball that stops in view ends the trace", not res[-1].has_moving_trace)
    check("stopped ball reports state 'stopped'", res[-1].state == "stopped")

    # --- Scenario 3: ball moves, then disappears -> ends after miss_timeout. ---
    disappear = []
    x = 100.0
    for _ in range(25):
        x += 18.0
        disappear.append([(x, 300.0, 0.9)])
    for _ in range(100):                     # ~3.3 s gap — exceeds miss_timeout_s=2.0 s
        disappear.append([])
    res = run(disappear)
    alive_idx = [i for i, s in enumerate(res) if s.has_moving_trace]
    check("disappearing ball eventually ends the trace", not res[-1].has_moving_trace)
    # The trace stays alive until detections age out of the corroboration window
    # (2.0 s) or miss_timeout fires (2.0 s) — both happen simultaneously.
    last_alive = alive_idx[-1] if alive_idx else 24
    coast_s = (last_alive - 24) * dt
    check("coast time is within miss_timeout (+1 frame)", coast_s <= 2.0 + dt + 1e-9)
    check("coast bridges at least a couple of frames", last_alive >= 24)

    # --- Scenario 4: lone flickering false positives never start a point. ---
    fp = []
    rng = np.random.default_rng(0)
    for _ in range(40):
        if rng.random() < 0.4:               # a stray detection some frames, random spot
            fp.append([(float(rng.integers(0, 900)), float(rng.integers(0, 500)), 0.3)])
        else:
            fp.append([])
    res = run(fp)
    check("scattered false positives never form a live trace",
          not any(s.has_moving_trace for s in res))

    # --- Scenario 5: a permanently stationary ball never sustains play. ---
    static = [[(500.0, 250.0, 0.8)] for _ in range(40)]
    res = run(static)
    check("a stationary ball never becomes a live trace",
          not any(s.has_moving_trace for s in res))

    # --- Scenario 6: brief occlusion (real gap) is bridged, point survives. ---
    occ = []
    x = 100.0
    for _ in range(20):
        x += 18.0
        occ.append([(x, 300.0, 0.9)])
    for _ in range(6):                       # ~0.2s occluded behind player
        x += 18.0
        occ.append([])
    for _ in range(20):                      # reappears on the predicted path
        x += 18.0
        occ.append([(x, 300.0, 0.9)])
    res = run(occ)
    check("trace survives a brief occlusion", res[-1].has_moving_trace)
    check("trace stays alive through the whole occluded rally",
          all(s.has_moving_trace for s in res[25:]))

    # --- Scenario 7: ball undergoes a ~180° direction reversal (racquet hit). ---
    # The IMM + fallback gate must absorb the velocity flip without a death blip.
    reversal = []
    x = 200.0
    for _ in range(25):                   # moving right at 30 px/frame
        x += 30.0
        reversal.append([(x, 300.0, 0.9)])
    reversal.append([])                   # one blank frame: ball at the racquet (hard to detect)
    for _ in range(25):                   # moving left after the hit
        x -= 30.0
        reversal.append([(x, 300.0, 0.9)])
    res = run(reversal)
    # Allow for the 1 blank frame, then alive throughout
    alive_after_hit = [s.has_moving_trace for s in res[26:]]
    max_dead = 0; cur = 0
    for a in alive_after_hit:
        cur = (cur + 1) if not a else 0
        max_dead = max(max_dead, cur)
    check("ball stays alive through a 180° direction reversal",
          all(alive_after_hit[-10:]))
    check("no prolonged death-blip at racquet impact (≤2 dead frames)",
          max_dead <= 2)
    mp_around = [s.maneuver_prob for s in res[24:34]]
    check("maneuver_prob spikes at racquet impact",
          max(mp_around) > 0.5)
    rp_around  = [s.racket_prob for s in res[24:34]]
    bp_around  = [s.bounce_prob for s in res[24:34]]
    check("racket_prob > bounce_prob at racquet impact (model discrimination)",
          max(rp_around) > max(bp_around))

    # --- Scenario 8: ball bounces off the court (vertical Vy-flip). ---
    # Horizontal velocity continues; vertical velocity reverses sign.
    bounce = []
    x2, y2 = 200.0, 100.0
    for _ in range(25):                   # descending: vy = +20 px/frame
        x2 += 15.0; y2 += 20.0
        bounce.append([(x2, y2, 0.9)])
    for _ in range(25):                   # ascending after bounce: vy = -20 px/frame
        x2 += 15.0; y2 -= 20.0
        bounce.append([(x2, y2, 0.9)])
    res = run(bounce)
    alive_after_bounce = [s.has_moving_trace for s in res[25:]]
    max_dead_b = 0; cur = 0
    for a in alive_after_bounce:
        cur = (cur + 1) if not a else 0
        max_dead_b = max(max_dead_b, cur)
    check("ball stays alive through a court bounce",
          all(alive_after_bounce[-10:]))
    check("no prolonged death-blip at bounce (≤2 dead frames)",
          max_dead_b <= 2)
    mp_bounce = [s.maneuver_prob for s in res[23:33]]
    check("maneuver_prob spikes at court bounce",
          max(mp_bounce) > 0.5)
    rp_bounce = [s.racket_prob for s in res[23:40]]
    bp_bounce = [s.bounce_prob for s in res[23:40]]
    check("bounce_prob > racket_prob at court bounce (model discrimination)",
          max(bp_bounce) > max(rp_bounce))

    # --- Scenario 9: serve toss confirmed → fast serve contact → rally. ---
    # The toss ball is slow (gets confirmed early in ACTIVE).  At contact the
    # ball's speed jumps to 100 px/frame — outside the primary AND the fallback
    # gate, so the toss track receives no updates.  Without the hijack path the
    # dead zone would last miss_timeout_s (3 s) and the whole point would be lost.
    toss_serve = []
    xt, yt = 400.0, 400.0
    for _ in range(20):            # toss: rising slowly  (20 px/frame upward)
        yt -= 20.0
        toss_serve.append([(xt, yt, 0.9)])
    toss_serve.append([])          # contact: 1 blank frame (motion blur / racket occlusion)
    xs, ys = xt, yt
    for _ in range(40):            # serve: 100 px/frame horizontal — well outside 90 px fallback gate
        xs += 100.0
        toss_serve.append([(xs, ys, 0.9)])
    res = run(toss_serve)
    # Allow the corroboration window (0.3 s ≈ 9 frames) for the toss track to age
    # out; the serve seed should be promoted shortly after that and stay alive.
    alive_post_contact = [s.has_moving_trace for s in res[22:]]
    check("tracker re-acquires ball after fast serve contact",
          any(alive_post_contact))
    check("serve/rally ball alive at end of sequence after fast serve",
          res[-1].has_moving_trace)

    # --- Scenario 10: low ball crosses the net near→far with SPARSE far-side dets. ---
    # A flat ball recedes toward the far side (y decreasing), goes small/occluded at
    # the net for ~7 frames, then reappears small and FLICKERING (one detection every
    # 4th frame) while continuing on a decelerated path.  These sparse far-side dets
    # are too thin to rebuild a fresh confirmed seed, so the existing track must absorb
    # them.  Without the coast-expanding gate the stale prediction's tight far-side
    # gate misses every flicker — the track stays nominally "alive" but its position
    # drifts hundreds of px off the true ball ("loses track").  The coast gate widens
    # with the detection gap and re-locks each flicker, holding position error small.
    # Run with a real perspective model so far-side scale/Q stiffening is in play.
    persp = make_image_row_perspective(540.0)
    mgr_net = BallTrackManager(fps, perspective_scale=persp)
    net_cross: List[list] = []
    truth: List[Tuple[float, float]] = []
    xn, yn = 150.0, 460.0
    for _ in range(16):                       # near side, descending toward net (detected)
        xn += 26.0; yn -= 11.0
        net_cross.append([(xn, yn, 0.9)]); truth.append((xn, yn))
    for _ in range(7):                        # at the net: small + occluded → no detections
        xn += 15.0; yn -= 6.0                 # ball physically decelerates while hidden
        net_cross.append([]); truth.append((xn, yn))
    for i in range(28):                       # far side: flickering dets, 1 every 4th frame
        xn += 15.0; yn -= 6.0
        net_cross.append([(xn, yn, 0.8)] if i % 4 == 0 else [])
        truth.append((xn, yn))
    out = []
    errs: List[float] = []
    tt = 0.0
    for dets, (tx_, ty_) in zip(net_cross, truth):
        s = mgr_net.update(dets, tt); tt += dt
        out.append(s)
        if s.position is not None:
            errs.append(math.hypot(s.position[0] - tx_, s.position[1] - ty_))
    check("trace position follows the ball across a sparse near→far net crossing",
          errs[-1] < 60.0)
    check("trace never drifts far off the true ball on the far side",
          max(errs[20:]) < 120.0)
    check("trace stays alive across the net crossing", out[-1].has_moving_trace)

    print()
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s) failed: {failures}")
        return 1
    print("SELF-TEST PASSED: all checks green.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_run_self_test())
