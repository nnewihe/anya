"""
far_anya_transitions.py
=======================
State machine for the far-side serve detector.

State summary
-------------
WAITING  : default.  Waits until the far player stands near the far baseline
           with their bounding-box top steady for ~1 second.

ARMED    : serve is expected.  Monitors the toss ball to fire ACTIVE.
           Falls back to WAITING if the far player walks away.

ACTIVE   : point in progress.  A trajectory-coherent Kalman tracker
           (BallTrackManager) is the sole authority on whether the ball is
           still in play.  When the moving trace goes dark, the near-player
           energy bar takes over: the point ends when the energy drains to 0.

Transition conditions
---------------------
WAITING → ARMED
    Far-player box centre is in the far half (cy < net_y_px − NET_BUFFER_PX)
    AND net displacement of the box centre over the last VELOCITY_WINDOW_SEC
    (1 s) is below MAX_VELOCITY_PX_S (50 px/s).  Net displacement (first →
    last position) is used so YOLO jitter does not accumulate.  No box-edge
    coordinates are used.

ARMED → WAITING
    Far-player net displacement in world space exceeds ARMED_MAX_TOTAL_FT ft
    over the last MOVEMENT_WINDOW_SEC seconds.

ARMED → ACTIVE
    Toss-based serve score ≥ TRANSITION_SCORE_THRESHOLD (0.55).
    Score = 0.05 × trophy + 0.95 × toss.
    Toss signal is the maximum of:
      • Consecutive-frame YOLO score (1 frame above head → 0.7, 2+ → 1.0).
      • Parabolic-arc score over a 1.5 s buffer (concave-down R² fit).
      • MHI secondary contribution (capped at MHI_MAX_CONTRIBUTION).

ACTIVE → WAITING
    Ball trace (BallTrackManager) is the primary authority.
    When the trace goes dark after MIN_POINT_DURATION_SEC:
      The near-player energy bar drives the remaining play duration.
      Energy decays while the receiver is walking/still/missing and
      grows while sprinting or making a swing motion.
      Point ends when energy reaches 0.
"""

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np

from ball_tracker import BallTrackManager, make_image_row_perspective


class FarTransitionEngine:
    def __init__(self, fps: float, net_y_px: float,
                 baseline_front_px: float = 0.0, baseline_behind_px: float = 0.0):
        self.fps      = fps
        self.net_y_px = net_y_px
        self._waiting_debug_counter: int = 0   # throttle WAITING debug prints

        # ── WAITING → ARMED ───────────────────────────────────────────────
        # Condition 1 — baseline zone (pixel space): far-player corrected foot
        #   pixel-y must lie between baseline_behind_px (6 ft behind, smaller y)
        #   and baseline_front_px (1 ft in front, larger y).
        self.NET_BUFFER_PX        = 0.0             # kept for render_frame overlay only
        self.baseline_front_px    = baseline_front_px   # larger pixel-y (1 ft inside court)
        self.baseline_behind_px   = baseline_behind_px  # smaller pixel-y (6 ft behind)

        # Condition 2 — low velocity: net displacement of box centre over
        # VELOCITY_WINDOW_SEC must be below MAX_VELOCITY_PX_S.
        # Net displacement (first→last position in window) is used rather than
        # cumulative path so that brief detection jitter does not accumulate.
        self.VELOCITY_WINDOW_SEC  = 1.0    # rolling window length (seconds)
        self.MAX_VELOCITY_PX_S    = 50.0   # px/s — above this = player is moving

        # Grace: tolerate short detection gaps without resetting the vel buffer
        self.VELOCITY_MISS_GRACE_FRAMES = 10

        self._vel_buffer:         deque = deque()   # (timestamp, cx, cy)
        self._vel_miss_frames:    int   = 0

        # Grace: tolerate brief exits from the baseline band before clearing the
        # velocity buffer — prevents a single out-of-band YOLO frame from wiping
        # a full second of accumulated data.
        self.BASELINE_MISS_GRACE  = 5
        self._baseline_miss_count: int = 0

        # ── ARMED → ACTIVE (toss-based) ───────────────────────────────────
        self.TRANSITION_SCORE_THRESHOLD = 0.55
        self.EVENT_WINDOW_SECONDS       = 1.2
        # MHI contribution cap: prevents MHI alone from firing the serve
        self.MHI_THRESHOLD             = 0.30
        self.MHI_MAX_CONTRIBUTION      = 0.50

        self.toss_consecutive_frames:       int             = 0
        self.toss_gap_frames:               int             = 0
        self.toss_ball_above_head_detected: bool            = False
        self.toss_min_y_px:                 Optional[float] = None
        self.last_toss_ball:                Optional[dict]  = None

        # Parabolic arc buffer: (timestamp, cy) for detections above player's head
        self.TOSS_ARC_WINDOW_SEC:   float = 1.5
        self.TOSS_ARC_MIN_POINTS:   int   = 3
        self.TOSS_ARC_R2_THRESHOLD: float = 0.80
        self._toss_arc_buffer:      deque = deque()

        self._toss_scores: deque = deque()

        self.last_serve_scores = {
            "toss_score":  0.0,
            "mhi_score":   0.0,
            "serve_score": 0.0,
        }

        # ── ACTIVE — ball-trace tracker ───────────────────────────────────
        self.SCREEN_HEIGHT_PX       = 540
        self.MIN_POINT_DURATION_SEC = 3.0

        self.ball_tracker = BallTrackManager(
            fps=fps,
            perspective_scale=make_image_row_perspective(self.SCREEN_HEIGHT_PX),
        )

        # Moving-ball trajectory for the visual overlay: (t, px, py)
        self._trace_ball_history:    deque = deque()
        self.active_start_time:      float = 0.0
        self.last_active_trace_time: float = 0.0

        # ── ACTIVE — near-player energy bar ───────────────────────────────
        # Drives point-end when the ball trace is dark (post MIN_POINT_DURATION).
        # Driven by the NEAR (receiver) player's movement — same logic as the
        # archived far_anya energy bar, but the near player is already the source
        # used in that archive.
        self.ENERGY_BOOST_SPRINT         = 4.0   # per second while receiver sprints
        self.ENERGY_BOOST_SWING          = 4.0   # per second on box-size swing detection
        self.ENERGY_DECAY_WALKING        = 0.3   # per second while receiver walks
        self.ENERGY_DECAY_STILL          = 0.2   # per second while receiver is still
        self.ENERGY_DECAY_MISSING        = 0.4   # per second while receiver not detected
        self.PLAYER_SPRINT_VELOCITY_FTS  = 7.0
        self.PLAYER_STILL_VELOCITY_FTS   = 2.0
        self.VELOCITY_WINDOW_SIZE        = 20
        self.ACTIVE_PLAYER_STRIDE        = 4
        self.PLAYER_MISSING_GRACE_FRAMES = 5
        self.PLAYER_EMA_ALPHA            = 0.25
        self.GAIT_BUFFER_FRAMES          = 45
        self.GAIT_MIN_REVERSALS          = 2
        self.GAIT_MAX_REVERSALS          = 8
        self.GAIT_MIN_DRIFT_PX           = 10.0

        self.energy_bar_mode:           bool  = False
        self.energy_bar_start_time:     float = 0.0
        self.point_energy:              float = 1.0

        self._energy_player_positions: deque = deque(maxlen=self.VELOCITY_WINDOW_SIZE)
        self._energy_player_boxes:     deque = deque(maxlen=5)
        self._energy_gait_y_buffer:    deque = deque(maxlen=self.GAIT_BUFFER_FRAMES)
        self._player_missing_frames:   int   = 0
        self._smoothed_player_world:   Optional[Tuple[float, float]] = None

        # ── Output signal ─────────────────────────────────────────────────
        self.last_transition_time: Optional[float] = None

        self.last_active_debug = {
            "state":                "none",
            "has_active_trace":     False,
            "time_since_detection": 0.0,
            "ball_speed_px_s":      0.0,
            "coasting":             False,
            "ball_count":           0,
            "maneuver_prob":        0.0,
            "racket_prob":          0.0,
            "bounce_prob":          0.0,
        }

    # ── Public entry point ────────────────────────────────────────────────────

    def evaluate_transitions(self, history: deque, current_state: str) -> str:
        if not history:
            return current_state
        if current_state == "WAITING":
            return self._check_waiting(history)
        if current_state == "ARMED":
            return self._check_armed(history)
        if current_state == "ACTIVE":
            return self._check_active(history)
        return current_state

    # ── WAITING → ARMED ───────────────────────────────────────────────────────

    def _check_waiting(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        far_box = frame.far_player_box
        if far_box is None:
            self._vel_miss_frames += 1
            if self._vel_miss_frames > self.VELOCITY_MISS_GRACE_FRAMES:
                self._vel_buffer.clear()
                self._vel_miss_frames = 0
            return "WAITING"

        x1, y1, x2, y2 = far_box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # ── Condition 1: far-player feet are in the baseline pixel band ────
        # Compare corrected foot pixel-y directly against pre-computed rows.
        # behind_px < front_px (far end of court is higher in the frame).
        foot_y_px = frame.far_player_foot_y_px
        near_baseline = (
            foot_y_px is not None
            and self.baseline_behind_px <= foot_y_px <= self.baseline_front_px
        )

        # ── Condition 2: low velocity over the last VELOCITY_WINDOW_SEC ──
        self._vel_miss_frames = 0
        if near_baseline:
            self._baseline_miss_count = 0
            self._vel_buffer.append((now, cx, cy))
        else:
            self._baseline_miss_count += 1

        # Prune to window + one frame so the oldest entry can be a full
        # second old (avoids the off-by-one where span never quite reaches 1 s)
        frame_dt = 1.0 / self.fps
        while (self._vel_buffer and
               now - self._vel_buffer[0][0] > self.VELOCITY_WINDOW_SEC + frame_dt):
            self._vel_buffer.popleft()

        span = (self._vel_buffer[-1][0] - self._vel_buffer[0][0]
                if len(self._vel_buffer) >= 2 else 0.0)

        # Velocity only meaningful once we have a full window
        velocity = 0.0
        if span >= self.VELOCITY_WINDOW_SEC:
            first_cx, first_cy = self._vel_buffer[0][1],  self._vel_buffer[0][2]
            last_cx,  last_cy  = self._vel_buffer[-1][1], self._vel_buffer[-1][2]
            net_disp = math.hypot(last_cx - first_cx, last_cy - first_cy)
            velocity = net_disp / span

        # Always-on debug print (every 30 frames) so every failure mode is visible
        self._waiting_debug_counter += 1
        if self._waiting_debug_counter % 30 == 0:
            foot_str = f"{foot_y_px:.0f}" if foot_y_px is not None else "?"
            print(f"[FAR WAIT DBG] foot_px={foot_str}  "
                  f"zone=[{self.baseline_behind_px:.0f}, {self.baseline_front_px:.0f}]px  "
                  f"near_baseline={near_baseline}  "
                  f"buf={len(self._vel_buffer)}  span={span:.2f}s  "
                  f"vel={velocity:.1f}px/s (max {self.MAX_VELOCITY_PX_S:.0f})")

        if not near_baseline:
            if self._baseline_miss_count > self.BASELINE_MISS_GRACE:
                self._vel_buffer.clear()
                self._baseline_miss_count = 0
            return "WAITING"

        if span < self.VELOCITY_WINDOW_SEC:
            return "WAITING"

        if velocity >= self.MAX_VELOCITY_PX_S:
            return "WAITING"

        print(f"[FAR TRANSITION] WAITING -> ARMED. "
              f"Far player at baseline (foot_px={foot_y_px:.0f}) for {span:.1f}s  "
              f"vel={velocity:.1f}px/s  cy={cy:.0f}")
        self._vel_buffer.clear()
        self._vel_miss_frames    = 0
        self._baseline_miss_count = 0
        return "ARMED"

    # ── ARMED → ACTIVE or ARMED → WAITING ─────────────────────────────────────

    def _check_armed(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        if frame.far_player_box is None:
            return "ARMED"

        fx1, fy1, fx2, fy2 = frame.far_player_box

        # YOLO toss score (relaxed: 1 frame above head → 0.7, 2+ → 1.0)
        yolo_toss = self._update_toss_detection(frame, fy1, now)
        if yolo_toss > 0:
            self._toss_scores.append((yolo_toss, now))

        # MHI secondary toss signal
        mhi_score = getattr(frame, "mhi_toss_score", 0.0)
        if mhi_score > self.MHI_THRESHOLD:
            scaled_mhi = mhi_score * self.MHI_MAX_CONTRIBUTION
            self._toss_scores.append((scaled_mhi, now))

        # Prune to event window
        while self._toss_scores and now - self._toss_scores[0][1] > self.EVENT_WINDOW_SECONDS:
            self._toss_scores.popleft()

        serve_score = max((s for s, _ in self._toss_scores), default=0.0)

        self.last_serve_scores = {
            "toss_score":  serve_score,
            "mhi_score":   mhi_score,
            "serve_score": serve_score,
        }

        if serve_score >= self.TRANSITION_SCORE_THRESHOLD:
            # Validate toss height: ball must have appeared above player's head
            if self.toss_min_y_px is not None and self.toss_min_y_px >= fy1:
                print(f"[FAR DEBUG] Toss height invalid: min_y={self.toss_min_y_px:.1f} "
                      f"must be < player_top={fy1}")
                self.toss_min_y_px = None
                return "ARMED"

            toss_h_str = (f"{self.toss_min_y_px:.1f}px (above {fy1})"
                          if self.toss_min_y_px is not None else "MHI only")
            print(f"[FAR TRANSITION] ARMED -> ACTIVE. "
                  f"Toss! Score={serve_score:.2f}  Toss height: {toss_h_str}")
            self._reset_armed_state()
            self._init_active(now)
            return "ACTIVE"

        return "ARMED"

    def _update_toss_detection(self, frame, fy1: float, now: float) -> float:
        """
        Toss detection — two independent signals, take the max.

        Signal 1 (consecutive-frame):
          Far-side uses a relaxed threshold: 1 consecutive YOLO frame above
          head with upward motion → 0.7, 2+ frames → 1.0.  (Near-side requires
          2–3 frames because the ball is easier to detect up close; at distance
          we settle for 1 confident frame.)

        Signal 2 (parabolic arc):
          Maintains a 1.5 s rolling buffer of (timestamp, cy) for detections
          above the player's head.  A concave-down parabola fit with R² ≥ 0.80
          scores 1.0; R² ≥ 0.60 scores 0.7.
        """
        if not frame.toss_ball_candidates:
            self.last_toss_ball   = None
            self.toss_gap_frames += 1
            if self.toss_gap_frames > 3:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False
            return 0.0

        best = max(frame.toss_ball_candidates, key=lambda x: x["conf"])
        bx1, by1, bx2, by2 = best["box"]
        cy = (by1 + by2) / 2.0

        is_moving_upward   = False
        is_ball_above_head = cy < fy1

        if self.last_toss_ball is not None:
            dy  = cy - self.last_toss_ball["y"]
            dtt = now - self.last_toss_ball["time"]
            if dy < 0 and dtt > 0:
                is_moving_upward = True

        if is_ball_above_head:
            if self.toss_min_y_px is None or cy < self.toss_min_y_px:
                self.toss_min_y_px = cy
            self._toss_arc_buffer.append((now, cy))

        while self._toss_arc_buffer and now - self._toss_arc_buffer[0][0] > self.TOSS_ARC_WINDOW_SEC:
            self._toss_arc_buffer.popleft()

        self.last_toss_ball = {"y": cy, "time": now}

        if is_moving_upward and is_ball_above_head:
            self.toss_gap_frames              = 0
            self.toss_consecutive_frames     += 1
            self.toss_ball_above_head_detected = True
        else:
            self.toss_gap_frames += 1
            if self.toss_gap_frames > 3:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False

        # Signal 1
        consecutive_score = 0.0
        if self.toss_ball_above_head_detected:
            if self.toss_consecutive_frames >= 2:
                consecutive_score = 1.0
            elif self.toss_consecutive_frames >= 1:
                consecutive_score = 0.7

        # Signal 2
        arc_score = 0.0
        if len(self._toss_arc_buffer) >= self.TOSS_ARC_MIN_POINTS:
            arc_score = self._score_toss_arc()

        return max(consecutive_score, arc_score)

    def _score_toss_arc(self) -> float:
        """Fit a concave-down parabola to the buffered (t, cy) points."""
        pts  = list(self._toss_arc_buffer)
        ts   = np.array([p[0] for p in pts], dtype=np.float64)
        cys  = np.array([p[1] for p in pts], dtype=np.float64)
        t0, t1 = ts[0], ts[-1]
        if t1 - t0 < 1e-6:
            return 0.0
        ts_norm  = (ts - t0) / (t1 - t0)
        coeffs   = np.polyfit(ts_norm, cys, 2)
        a        = coeffs[0]
        if a >= 0:
            return 0.0   # concave-up → not a toss arc
        cys_pred = np.polyval(coeffs, ts_norm)
        ss_res   = np.sum((cys - cys_pred) ** 2)
        ss_tot   = np.sum((cys - cys.mean()) ** 2)
        r2       = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        if r2 >= self.TOSS_ARC_R2_THRESHOLD:
            return 1.0
        if r2 >= 0.60:
            return 0.7
        return 0.0

    # ── ACTIVE → WAITING ──────────────────────────────────────────────────────

    def _check_active(self, history: deque) -> str:
        frame      = history[-1]
        now        = frame.timestamp
        candidates = frame.active_ball_candidates or []
        elapsed    = now - self.active_start_time

        # ── 1. Feed detections to tracker ────────────────────────────────
        detections = [
            (c["pixel_center"][0], c["pixel_center"][1], c.get("conf", 0.0))
            for c in candidates
        ]
        status = self.ball_tracker.update(detections, now)

        # ── 2. Sync moving-ball trace for overlay ─────────────────────────
        self._trace_ball_history.clear()
        self._trace_ball_history.extend(self.ball_tracker.trace_points())

        if status.has_moving_trace:
            self.last_active_trace_time = now

        # ── 3. Update near-player tracking for energy bar ─────────────────
        self._update_player_tracking(frame)

        # ── 4. Build debug snapshot ───────────────────────────────────────
        self.last_active_debug = {
            "state":                status.state,
            "has_active_trace":     status.has_moving_trace,
            "time_since_detection": status.time_since_detection,
            "ball_speed_px_s":      status.speed_px_s,
            "coasting":             status.coasting,
            "ball_count":           status.ball_count,
            "maneuver_prob":        status.maneuver_prob,
            "racket_prob":          status.racket_prob,
            "bounce_prob":          status.bounce_prob,
        }

        # ── 5. Trace alive → stay ACTIVE (reset energy bar if running) ────
        if status.has_moving_trace:
            if self.energy_bar_mode:
                print(f"[FAR ACTIVE] Ball trace restored at t={now:.2f}s. "
                      f"Resetting energy bar (was {self.point_energy:.2f}).")
                self.energy_bar_mode = False
                self.point_energy    = 1.0
                self._energy_player_positions.clear()
                self._energy_player_boxes.clear()
                self._energy_gait_y_buffer.clear()
            return "ACTIVE"

        # ── 6. Minimum duration grace ─────────────────────────────────────
        if elapsed < self.MIN_POINT_DURATION_SEC:
            return "ACTIVE"

        # ── 7. Trace dark → run energy bar ────────────────────────────────
        if not self.energy_bar_mode:
            print(f"[FAR ACTIVE] Trace dark at t={now:.2f}s. Entering energy bar mode.")
            self.energy_bar_mode       = True
            self.energy_bar_start_time = self.last_active_trace_time
            self.point_energy          = 1.0

        dt = 1.0 / self.fps
        energy_delta, status_label = self._compute_energy_delta(frame, dt)
        self.point_energy = max(0.0, min(1.0, self.point_energy + energy_delta))

        self.last_active_debug.update({
            "energy_bar_mode": self.energy_bar_mode,
            "point_energy":    self.point_energy,
            "energy_status":   status_label,
        })

        if self.point_energy <= 0.0:
            death_t = self.ball_tracker.last_detection_time
            if death_t is None:
                death_t = self.last_active_trace_time
            death_t = max(self.active_start_time, death_t)
            self.last_transition_time = death_t
            print(f"\n[FAR TRANSITION] ACTIVE -> WAITING (Energy Depleted [{status_label}]). "
                  f"Lasted {elapsed:.1f}s. Rewind to t={death_t:.2f}s.")
            self._reset_active_state()
            return "WAITING"

        return "ACTIVE"

    # ── Near-player energy bar helpers ────────────────────────────────────────

    def _update_player_tracking(self, frame) -> None:
        """Append near-player position and box to rolling buffers for energy bar."""
        near_box   = frame.near_player_box
        near_world = getattr(frame, "near_player_world", None)
        if near_box is None or near_world is None:
            self._player_missing_frames += 1
            self._energy_gait_y_buffer.clear()
            return
        self._player_missing_frames = 0

        wx, wy = near_world
        if self._smoothed_player_world is None:
            self._smoothed_player_world = (wx, wy)
        else:
            α = self.PLAYER_EMA_ALPHA
            self._smoothed_player_world = (
                α * wx + (1 - α) * self._smoothed_player_world[0],
                α * wy + (1 - α) * self._smoothed_player_world[1],
            )
        self._energy_player_positions.append(self._smoothed_player_world)
        self._energy_player_boxes.append(near_box)
        self._energy_gait_y_buffer.append(float(near_box[3]))   # y2 oscillation

    def _compute_energy_delta(self, frame, dt: float) -> Tuple[float, str]:
        """
        Return (energy_delta, status_label) for one frame.

        Priority order:
          1. Receiver missing        → fast decay  (MISSING)
          2. Walking gait detected   → slow decay  (WALKING)
          3. Receiver sprinting      → boost        (SPRINTING)
          4. Box-shape swing motion  → boost        (SWING)
          5. Receiver still          → slow decay  (STILL)
          6. Otherwise               → slight boost (MOVING)
        """
        if self._player_missing_frames > self.PLAYER_MISSING_GRACE_FRAMES:
            return -(self.ENERGY_DECAY_MISSING * dt), "MISSING"

        if self._detect_walking_gait():
            return -(self.ENERGY_DECAY_WALKING * dt), "WALKING"

        player_velocity_fts = 0.0
        if len(self._energy_player_positions) >= 5:
            old_p   = self._energy_player_positions[0]
            new_p   = self._energy_player_positions[-1]
            dist_ft = math.hypot(new_p[0] - old_p[0], new_p[1] - old_p[1])
            elapsed = len(self._energy_player_positions) * self.ACTIVE_PLAYER_STRIDE / self.fps
            player_velocity_fts = dist_ft / elapsed if elapsed > 0 else 0.0

        if player_velocity_fts > self.PLAYER_SPRINT_VELOCITY_FTS:
            return (self.ENERGY_BOOST_SPRINT * dt), f"SPRINTING {player_velocity_fts:.1f}ft/s"

        if len(self._energy_player_boxes) >= 5:
            old_b      = self._energy_player_boxes[0]
            new_b      = self._energy_player_boxes[-1]
            box_height = old_b[3] - old_b[1]
            if box_height > 0:
                dw = abs((new_b[2] - new_b[0]) - (old_b[2] - old_b[0]))
                dh = abs((new_b[3] - new_b[1]) - (old_b[3] - old_b[1]))
                if (dw + dh) / box_height > 0.25:
                    return (self.ENERGY_BOOST_SWING * dt), "SWING"

        if player_velocity_fts < self.PLAYER_STILL_VELOCITY_FTS:
            return -(self.ENERGY_DECAY_STILL * dt), f"STILL {player_velocity_fts:.1f}ft/s"
        return (0.1 * dt), f"MOVING {player_velocity_fts:.1f}ft/s"

    def _detect_walking_gait(self) -> bool:
        """Detect walking gait from oscillatory vertical movement of the near player's box."""
        ys = list(self._energy_gait_y_buffer)
        n  = len(ys)
        if n < self.GAIT_BUFFER_FRAMES * 0.6:
            return False
        if abs(ys[-1] - ys[0]) < self.GAIT_MIN_DRIFT_PX:
            return False
        residuals = [
            ys[i] - (ys[0] + (ys[-1] - ys[0]) * (i / (n - 1)))
            for i in range(n)
        ]
        reversals      = 0
        prev_direction = 0
        for i in range(1, len(residuals)):
            delta = residuals[i] - residuals[i - 1]
            if abs(delta) < 0.5:
                continue
            direction = 1 if delta > 0 else -1
            if prev_direction != 0 and direction != prev_direction:
                reversals += 1
            prev_direction = direction
        return self.GAIT_MIN_REVERSALS <= reversals <= self.GAIT_MAX_REVERSALS

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def _reset_armed_state(self) -> None:
        self._vel_buffer.clear()
        self._vel_miss_frames    = 0
        self._baseline_miss_count = 0
        self.toss_consecutive_frames       = 0
        self.toss_gap_frames               = 0
        self.toss_ball_above_head_detected = False
        self.toss_min_y_px                 = None
        self.last_toss_ball                = None
        self._toss_arc_buffer.clear()
        self._toss_scores.clear()
        self.last_serve_scores = {
            "toss_score":  0.0,
            "mhi_score":   0.0,
            "serve_score": 0.0,
        }

    def _reset_active_state(self) -> None:
        self._trace_ball_history.clear()
        self.active_start_time       = 0.0
        self.last_active_trace_time  = 0.0
        self.ball_tracker.reset()
        self.energy_bar_mode         = False
        self.energy_bar_start_time   = 0.0
        self.point_energy            = 1.0
        self._energy_player_positions.clear()
        self._energy_player_boxes.clear()
        self._energy_gait_y_buffer.clear()
        self._player_missing_frames  = 0
        self._smoothed_player_world  = None

    def _init_active(self, now: float) -> None:
        self._reset_active_state()
        self.active_start_time      = now
        self.last_active_trace_time = now
        self.last_transition_time   = None
