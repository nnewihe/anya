"""
anya_transitions.py  (REVISED — refined-priority build)
=======================================================
State machine logic. Determines transitions between WAITING, ARMED, and ACTIVE
based on the rolling telemetry buffer from anya_base.py.

This is a drop-in replacement for the original module. The public interface is
unchanged — run_anya.py still consumes:
    • evaluate_transitions(history, current_state) -> str
    • last_transition_time          (rewind anchor on ACTIVE -> WAITING)
    • last_active_debug             (HUD / CSV snapshot dict)
    • last_serve_scores             (ARMED HUD dict)
    • _trace_ball_history           (overlay polyline, (t, px, py))

WHAT CHANGED vs. the original, and why
--------------------------------------
Every signal is now routed by an explicit priority tier, collected in
`SignalPriorityConfig` so the knobs that matter are in one place instead of
scattered as magic numbers.

  Tier 1 — POINT START  (ARMED -> ACTIVE):  trophy pose (20%) + ball toss (80%)
  Tier 1 — POINT END    (ACTIVE -> WAITING): ball-trace movement (sole authority)
  Tier 2 — MODULATION   : near-player walking velocity nudges the point-end timeout
  Tier 3 — (audio)      : intentionally NOT wired in here; too noisy on rec. footage

1. POINT-END is still ball-trace-driven, but "is the trace alive" now also
   requires the IMM velocity to clear a *perspective-scaled* floor. A ball that
   rolls to rest in view can satisfy the span-based move test (it drifts >30 px
   over the window) yet be physically dead; the velocity floor rejects that.

2. Faster point START / shorter dead grace:
       MIN_POINT_DURATION_SEC  3.0 -> 2.0   (recreational serves are 1.5-2.5 s)
   Longer rally protection:
       TRACE_DEAD_TIMEOUT_SEC  3.0 -> 3.5   (bridges net cords / body occlusions)

3. Ball-toss detection (far-side serve, the weak Tier-1 signal) is made
   occlusion-tolerant: a confidence floor filters junk detections, a single
   dropped frame no longer resets the toss run, and a 2-frame run now confirms.

4. Near-player velocity (previously computed in telemetry but unused) becomes a
   Tier-2 tiebreaker: when the trace is gone, a stationary player shortens the
   dead-timeout (point is over sooner) while a player chasing/​walking lengthens
   it (don't truncate a long run-down). It can never *start* or *end* a point on
   its own — it only scales the ball-trace timeout within a bounded range.

5. fast_rearm: on ACTIVE -> WAITING, if the player is already back in the ready
   band, jump straight to ARMED (this wires up the pre-existing but previously
   unused _post_active_next_state helper), cutting dead time between points.

Audio (Signal 2) is deliberately left out of the decision path. It belongs in a
separate confirmatory module for quiet venues; forcing it into point-end logic
on noisy recreational footage costs more than it returns.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

import numpy as np

from ball_tracker import BallTrackManager, TrackStatus, make_image_row_perspective
from utilities import Config


# =====================================================================
# Refined signal priorities — every tunable knob that encodes "what we
# trust, and how much" lives here.
# =====================================================================
@dataclass
class SignalPriorityConfig:
    # ---- Tier 1: POINT START (ARMED -> ACTIVE) — serve detection ----
    trophy_weight: float = 0.2          # near-side trophy pose (confirmer)
    toss_weight:   float = 0.8          # ball toss (serve-specific anchor)
    serve_score_threshold: float = 0.55
    serve_event_window_s:  float = 1.2  # window over which trophy/toss scores persist
    trophy_conf_floor:     float = 0.0  # ignore trophy scores below this (0 = keep all)

    # Ball-toss gating (Phase 2 — occlusion-tolerant, far-side friendly)
    toss_conf_floor:     float = 0.5    # drop toss-ball detections below this YOLO conf
    toss_gap_tolerance:  int   = 1      # dropped frames allowed before a toss run resets (was 3)
    toss_confirm_frames: int   = 2      # consecutive good frames to confirm a toss (was 3)

    # ---- Tier 1: POINT END (ACTIVE -> WAITING) — ball-trace authority ----
    min_point_duration_s: float = 2.0   # earliest a point may end           (was 3.0)
    trace_dead_timeout_s: float = 3.5   # trace-dark grace before ending      (was 3.0)
    move_velocity_floor_px_s: float = 20.0  # perspective-scaled; below = not live motion

    # ---- Serve-trace directional confirmation ----
    # An ACTIVE window is only committed as a real segment when the ball trace
    # shows both a downward component (gravity after serve contact) and a
    # horizontal component (ball traveling toward/away from the net).
    # Windows where no such trace is seen (e.g. pre-serve ground bouncing) are
    # silently discarded — ST-GCN detection alone is not sufficient.
    trace_downward_px_s:   float = 40.0  # min net dy/dt required (px/s, positive = down)
    trace_horizontal_px_s: float = 30.0  # min |dx/dt| required (px/s)

    # ---- Tier 2: near-player velocity modulation of the dead-timeout ----
    use_player_velocity:     bool  = True
    player_velocity_window_s: float = 0.5   # robust to the ACTIVE player-detect stride
    player_stationary_ft_s:  float = 1.0    # <= this  -> standing around -> shorten timeout
    player_walking_ft_s:     float = 2.5    # >= this  -> chasing/walking  -> lengthen timeout
    stationary_timeout_mult: float = 0.8
    walking_timeout_mult:    float = 1.2

    # ---- Cycle ----
    fast_rearm: bool = True             # ACTIVE -> ARMED directly if player already at baseline


class TransitionEngine:
    """
    Parameterized by which player/baseline it gates on, so the far-side pass
    can reuse this exact algorithm (see FarSideTransitionEngine below) instead
    of a bespoke implementation:
      - player_box_attr / player_world_attr: TelemetryFrame fields to read for
        the gating player's box/world position (near_player_* by default).
      - baseline_y_ft: court-Y of the baseline being served from (0 = near).
      - direction: sign such that (world_y - baseline_y_ft) * direction > 0
        when the player is correctly positioned behind that baseline
        (-1 for near, since near_player_world.y is negative behind y=0).
    """

    def __init__(self, fps: float, cfg: Optional[SignalPriorityConfig] = None,
                 side_label: str = "near",
                 player_box_attr: str = "near_player_box",
                 player_world_attr: str = "near_player_world",
                 baseline_y_ft: float = 0.0,
                 direction: float = -1.0):
        self.fps = fps
        self.cfg = cfg or SignalPriorityConfig()

        self._side_label        = side_label
        self._player_box_attr   = player_box_attr
        self._player_world_attr = player_world_attr
        self._baseline_y_ft     = baseline_y_ft
        self._direction         = direction
        self._log_suffix        = "" if side_label == "near" else f" ({side_label} side)"

        # ------------------------------------------------------------------
        # WAITING / ARMED-band geometry (unchanged — point-start ZONE gating)
        # ------------------------------------------------------------------
        self.READY_MIN_DIST_FT   = -0.5
        self.READY_MAX_DIST_FT   = 3.5
        self.READY_WAIT_TIME_SEC = 0.4

        self.ARMED_BAND_WINDOW_SEC     = 2.0
        self.ARMED_OUT_RATIO_THRESHOLD = 0.25

        # ------------------------------------------------------------------
        # ACTIVE — ball-trace tracker (sole point-end authority)
        # ------------------------------------------------------------------
        self.SCREEN_HEIGHT_PX = 540   # analysis-frame height; drives perspective model

        self.ball_tracker = BallTrackManager(
            fps=fps,
            perspective_scale=make_image_row_perspective(self.SCREEN_HEIGHT_PX),
        )

        # ------------------------------------------------------------------
        # Persistent state — WAITING
        # ------------------------------------------------------------------
        self.near_ready_start_time: Optional[float] = None

        # ------------------------------------------------------------------
        # Persistent state — ARMED
        # ------------------------------------------------------------------
        self.armed_band_history: deque = deque()

        self.toss_consecutive_frames:       int             = 0
        self.toss_gap_frames:               int             = 0
        self.toss_ball_above_head_detected: bool            = False
        self.toss_min_y_px:                 Optional[float] = None
        self.last_toss_ball:                Optional[dict]  = None

        self._trophy_scores: deque = deque()
        self._toss_scores:   deque = deque()

        self.last_serve_scores = {
            "trophy_score": 0.0,
            "toss_score":   0.0,
            "serve_score":  0.0,
        }

        # ------------------------------------------------------------------
        # Persistent state — ACTIVE
        # ------------------------------------------------------------------
        self.active_start_time: float = 0.0
        self._trace_ball_history: deque = deque()      # (t, px, py) for the overlay
        self.last_active_trace_time: float = 0.0       # last frame with GENUINE motion
        self._trace_confirmed: bool = False             # True once a downward+horiz trace seen

        # Tier-2: near-player velocity (windowed, robust to player-detect stride)
        self._player_world_history: deque = deque()    # (t, world_y_ft)
        self.player_velocity_ft_s: float  = 0.0
        self.player_velocity_valid: bool  = False

        # Signal to the main loop: timestamp to truncate output on transition.
        self.last_transition_time: Optional[float] = None
        # Whether the last ACTIVE window should be committed as a segment.
        # False when the window is discarded (no confirmed serve trace).
        self.last_active_committed: bool = True

        # Debug snapshot for HUD / CSV (keys consumed by run_anya are preserved;
        # the three trailing keys are new and harmless to existing consumers).
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
            # --- new, refined-priority diagnostics ---
            "player_velocity_ft_s": 0.0,
            "slow_trace":           False,
            "effective_timeout_s":  self.cfg.trace_dead_timeout_s,
        }

    # ==================================================================
    # Public entry point
    # ==================================================================
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

    # ==================================================================
    # WAITING -> ARMED   (player settles into the ready band)
    # ==================================================================
    def _check_waiting(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        world = getattr(frame, self._player_world_attr, None)
        if world is None:
            self.near_ready_start_time = None
            return "WAITING"

        _, wy       = world
        signed_dist = (wy - self._baseline_y_ft) * self._direction
        in_zone     = self.READY_MIN_DIST_FT <= signed_dist <= self.READY_MAX_DIST_FT

        if in_zone:
            if self.near_ready_start_time is None:
                self.near_ready_start_time = now
            elapsed = now - self.near_ready_start_time
            if elapsed > self.READY_WAIT_TIME_SEC:
                print(f"[TRANSITION] WAITING -> ARMED{self._log_suffix}. "
                      f"Player held ready for {elapsed:.1f}s.")
                self.near_ready_start_time = None
                return "ARMED"
        else:
            self.near_ready_start_time = None

        return "WAITING"

    # ==================================================================
    # ARMED -> ACTIVE  or  ARMED -> WAITING   (Tier-1 serve detection)
    # ==================================================================
    def _check_armed(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        in_band = False
        world = getattr(frame, self._player_world_attr, None)
        if world is not None:
            _, wy       = world
            signed_dist = (wy - self._baseline_y_ft) * self._direction
            in_band     = self.READY_MIN_DIST_FT <= signed_dist <= self.READY_MAX_DIST_FT

        self.armed_band_history.append((now, in_band))
        while (self.armed_band_history and
               now - self.armed_band_history[0][0] > self.ARMED_BAND_WINDOW_SEC):
            self.armed_band_history.popleft()

        # Bail out of ARMED if the player has drifted out of the ready band for
        # too large a fraction of the recent window (they wandered off / reset).
        if len(self.armed_band_history) > 1:
            total_time = self.armed_band_history[-1][0] - self.armed_band_history[0][0]
            if total_time > 1.0:
                time_out = sum(
                    self.armed_band_history[i + 1][0] - self.armed_band_history[i][0]
                    for i in range(len(self.armed_band_history) - 1)
                    if not self.armed_band_history[i][1]
                )
                out_ratio = time_out / total_time
                if out_ratio > self.ARMED_OUT_RATIO_THRESHOLD:
                    print(f"[TRANSITION] ARMED -> WAITING{self._log_suffix}. "
                          f"Out of band {out_ratio:.0%} over {total_time:.1f}s.")
                    self._reset_armed_state()
                    return "WAITING"

        player_box = getattr(frame, self._player_box_attr, None)
        if not in_band or player_box is None:
            return "ARMED"

        nx1, ny1, nx2, ny2 = player_box

        # --- Trophy pose (Tier-1 confirmer, low weight) ---
        trophy_score = getattr(frame, "trophy_score", 0.0) or 0.0
        if trophy_score >= self.cfg.trophy_conf_floor and trophy_score > 0:
            self._trophy_scores.append((trophy_score, now))

        # --- Ball toss (Tier-1 anchor, high weight) ---
        toss_score = self._update_toss_detection(frame, ny1, now)
        if toss_score > 0:
            self._toss_scores.append((toss_score, now))

        for buf in (self._trophy_scores, self._toss_scores):
            while buf and now - buf[0][1] > self.cfg.serve_event_window_s:
                buf.popleft()

        max_trophy  = max((s for s, _ in self._trophy_scores), default=0.0)
        max_toss    = max((s for s, _ in self._toss_scores),   default=0.0)
        serve_score = self.cfg.trophy_weight * max_trophy + self.cfg.toss_weight * max_toss

        self.last_serve_scores = {
            "trophy_score": max_trophy,
            "toss_score":   max_toss,
            "serve_score":  serve_score,
        }

        if serve_score >= self.cfg.serve_score_threshold:
            # Sanity: the toss must have peaked above the player's head.
            if self.toss_min_y_px is not None and self.toss_min_y_px >= ny1:
                print(f"[DEBUG] Toss height invalid: min_y={self.toss_min_y_px:.1f} "
                      f"must be < player_top={ny1}")
                self.toss_min_y_px = None
                return "ARMED"

            toss_h_str = (f"{self.toss_min_y_px:.1f}px (above {ny1})"
                          if self.toss_min_y_px is not None else "N/A")
            print(f"[TRANSITION] ARMED -> ACTIVE{self._log_suffix}. "
                  f"Serve detected! Score: {serve_score:.2f}  "
                  f"(trophy {max_trophy:.2f}*{self.cfg.trophy_weight} + "
                  f"toss {max_toss:.2f}*{self.cfg.toss_weight})  "
                  f"Toss height: {toss_h_str}")
            self._reset_armed_state()
            self._init_active(now)
            return "ACTIVE"

        return "ARMED"

    # ------------------------------------------------------------------
    # Ball-toss detector (occlusion-tolerant — Phase 2 refinements)
    # ------------------------------------------------------------------
    def _update_toss_detection(self, frame, ny1: float, now: float) -> float:
        # Confidence floor: drop low-confidence toss-ball detections (shadows,
        # crowd glints, far-side noise) before any motion reasoning.
        candidates = [
            c for c in (frame.toss_ball_candidates or [])
            if c.get("conf", 0.0) >= self.cfg.toss_conf_floor
        ]

        if not candidates:
            self.last_toss_ball   = None
            self.toss_gap_frames += 1
            if self.toss_gap_frames > self.cfg.toss_gap_tolerance:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False
            return 0.0

        best = max(candidates, key=lambda x: x["conf"])
        bx1, by1, bx2, by2 = best["box"]
        cy = (by1 + by2) / 2.0

        is_moving_upward   = False
        is_ball_above_head = cy < ny1

        if self.last_toss_ball is not None:
            dy  = cy - self.last_toss_ball["y"]
            dtt = now - self.last_toss_ball["time"]
            if dy < 0 and dtt > 0:
                is_moving_upward = True

        if is_ball_above_head:
            if self.toss_min_y_px is None or cy < self.toss_min_y_px:
                self.toss_min_y_px = cy

        self.last_toss_ball = {"y": cy, "time": now}

        if is_moving_upward and is_ball_above_head:
            self.toss_gap_frames               = 0
            self.toss_consecutive_frames      += 1
            self.toss_ball_above_head_detected = True
        else:
            # A single dropped/ambiguous frame no longer wipes the toss run;
            # only a gap longer than the tolerance resets it.
            self.toss_gap_frames += 1
            if self.toss_gap_frames > self.cfg.toss_gap_tolerance:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False

        if not self.toss_ball_above_head_detected:
            return 0.0
        if self.toss_consecutive_frames >= self.cfg.toss_confirm_frames:
            return 1.0
        if self.toss_consecutive_frames >= 1:
            return 0.5
        return 0.0

    # ==================================================================
    # ACTIVE -> WAITING   (ball-trace authority + Tier-2 velocity modulation)
    # ==================================================================
    def _check_active(self, history: deque) -> str:
        frame      = history[-1]
        now        = frame.timestamp
        candidates = frame.active_ball_candidates or []

        # ---- 1. Feed this frame's ball detections to the tracker ----
        # Detections inside the gating player's box are excluded from the
        # trace: the racket/arm/body frequently throws off ball-shaped false
        # positives right where that player stands.
        pbox = getattr(frame, self._player_box_attr, None)
        if pbox is not None:
            nx1, ny1, nx2, ny2 = pbox
            candidates = [
                c for c in candidates
                if not (nx1 <= c["pixel_center"][0] <= nx2 and
                        ny1 <= c["pixel_center"][1] <= ny2)
            ]

        detections = [
            (c["pixel_center"][0], c["pixel_center"][1], c.get("conf", 0.0))
            for c in candidates
        ]
        status = self.ball_tracker.update(detections, now)

        # ---- 2. Sync the moving-ball trace for the visual overlay ----
        self._trace_ball_history.clear()
        self._trace_ball_history.extend(self.ball_tracker.trace_points())

        # ---- 3. Perspective-scaled velocity floor ----
        # has_moving_trace already requires recent span > move_thresh_px, but a
        # ball rolling to rest can drift far enough to pass that while being
        # physically dead. Require the IMM speed to clear a floor that shrinks
        # with image row (far-side balls legitimately move fewer px/s).
        persp_scale = 1.0
        if status.position is not None:
            persp_scale = self.ball_tracker.persp(status.position[1])
        velocity_floor = self.cfg.move_velocity_floor_px_s * persp_scale
        slow_trace      = status.has_moving_trace and status.speed_px_s < velocity_floor
        genuinely_moving = status.has_moving_trace and not slow_trace

        # Refresh the dead-timeout clock ONLY on genuine motion, so a slow/rolling
        # ball lets the timeout run instead of holding the point open forever.
        if genuinely_moving:
            self.last_active_trace_time = now
            # First time we see genuine motion: check if the direction is serve-like
            # (downward + horizontal). Only needs to be confirmed once per window.
            if not self._trace_confirmed:
                self._try_confirm_trace()

        # ---- 4. Tier-2: update windowed near-player velocity ----
        self._update_player_velocity(frame, now)

        # ---- 5. Debug snapshot ----
        elapsed        = now - self.active_start_time
        trace_dead_for = now - self.last_active_trace_time

        effective_timeout = self._effective_dead_timeout()

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
            "player_velocity_ft_s": self.player_velocity_ft_s,
            "slow_trace":           slow_trace,
            "effective_timeout_s":  effective_timeout,
        }

        # ---- 6. Decide ----
        # Stay ACTIVE while the ball genuinely moves, within the min-duration
        # grace, or within the (velocity-modulated) dead-timeout.
        if genuinely_moving:
            return "ACTIVE"
        if elapsed < self.cfg.min_point_duration_s:
            return "ACTIVE"
        if trace_dead_for < effective_timeout:
            return "ACTIVE"

        # ---- 7. Point over ----
        next_world = getattr(frame, self._player_world_attr, None)

        if not self._trace_confirmed:
            # No confirmed serve trace observed — this ACTIVE window was a false start
            # (e.g. pre-serve ground bouncing). Discard it without emitting a segment.
            print(f"\n[DISCARD] ACTIVE window at {self.active_start_time:.2f}s discarded "
                  f"(no confirmed serve trace after {elapsed:.1f}s{self._log_suffix})")
            self.last_active_committed = False
            self.last_transition_time  = None
            self._reset_active_state()
            if self.cfg.fast_rearm:
                return self._post_active_next_state(next_world, "WAITING")
            return "WAITING"

        # Trace was confirmed — commit the segment, rewind to last real detection.
        death_t = self.ball_tracker.last_detection_time
        if death_t is None:
            death_t = self.last_active_trace_time
        death_t = max(self.active_start_time, death_t)

        self.last_active_committed = True
        self.last_transition_time  = death_t
        vel_note = (f"player {self.player_velocity_ft_s:.1f} ft/s, "
                    f"timeout {effective_timeout:.1f}s"
                    if self.player_velocity_valid else f"timeout {effective_timeout:.1f}s")
        print(f"\n[TRANSITION] ACTIVE -> WAITING (trace {status.state}; {vel_note}). "
              f"Lasted {elapsed:.1f}s. Rewind to t={death_t:.2f}s.")

        self._reset_active_state()

        if self.cfg.fast_rearm:
            return self._post_active_next_state(next_world, "WAITING")
        return "WAITING"

    # ------------------------------------------------------------------
    # Tier-2 helpers — gating-player velocity
    # ------------------------------------------------------------------
    def _update_player_velocity(self, frame, now: float) -> None:
        """
        Maintain a short rolling window of the gating player's world-y position
        and derive a walking speed (ft/s). A window (not a frame-to-frame diff)
        is used because the player model only runs every ACTIVE_PLAYER_STRIDE
        frames in anya_base — consecutive frames often repeat a cached position,
        which a naive diff would read as 0 then spike. Over a ~0.5 s window
        there is at least one genuine refresh, giving a stable estimate.
        """
        world = getattr(frame, self._player_world_attr, None)
        if not self.cfg.use_player_velocity or world is None:
            return

        _, wy = world
        self._player_world_history.append((now, float(wy)))
        while (self._player_world_history and
               now - self._player_world_history[0][0] > self.cfg.player_velocity_window_s):
            self._player_world_history.popleft()

        if len(self._player_world_history) >= 2:
            t0, wy0 = self._player_world_history[0]
            t1, wy1 = self._player_world_history[-1]
            if t1 > t0:
                self.player_velocity_ft_s  = abs(wy1 - wy0) / (t1 - t0)
                self.player_velocity_valid = True

    def _effective_dead_timeout(self) -> float:
        """
        Scale the ball-trace dead-timeout by the player's movement state. Bounded:
        a stationary player ends the point a little sooner; a chasing/walking
        player gets a little more grace. Velocity NEVER ends or starts a point on
        its own — it only nudges the ball-trace timeout, and only when valid.
        """
        timeout = self.cfg.trace_dead_timeout_s
        if not (self.cfg.use_player_velocity and self.player_velocity_valid):
            return timeout
        if self.player_velocity_ft_s <= self.cfg.player_stationary_ft_s:
            timeout *= self.cfg.stationary_timeout_mult
        elif self.player_velocity_ft_s >= self.cfg.player_walking_ft_s:
            timeout *= self.cfg.walking_timeout_mult
        return timeout

    # ==================================================================
    # Helpers — reset / init
    # ==================================================================
    def _post_active_next_state(self, player_pos, default_state: str) -> str:
        """
        On ACTIVE -> WAITING, bypass WAITING if the player is already inside the
        ready band — go straight to ARMED for the next point. Cuts dead time
        between consecutive serves from the same end.
        """
        if player_pos is not None:
            _, wy       = player_pos
            signed_dist = (wy - self._baseline_y_ft) * self._direction
            if self.READY_MIN_DIST_FT <= signed_dist <= self.READY_MAX_DIST_FT:
                print(f"[BYPASS] Player already at baseline{self._log_suffix}. Jumping to ARMED.")
                self._reset_armed_state()
                return "ARMED"
        return default_state

    def _reset_armed_state(self) -> None:
        self.armed_band_history.clear()
        self.toss_consecutive_frames       = 0
        self.toss_gap_frames               = 0
        self.toss_ball_above_head_detected = False
        self.toss_min_y_px                 = None
        self.last_toss_ball                = None
        self._trophy_scores.clear()
        self._toss_scores.clear()
        self.last_serve_scores = {
            "trophy_score": 0.0,
            "toss_score":   0.0,
            "serve_score":  0.0,
        }

    def _reset_active_state(self) -> None:
        self._trace_ball_history.clear()
        self.active_start_time      = 0.0
        self.last_active_trace_time = 0.0
        self._trace_confirmed       = False
        self._player_world_history.clear()
        self.player_velocity_ft_s   = 0.0
        self.player_velocity_valid  = False
        self.ball_tracker.reset()

    def _try_confirm_trace(self) -> None:
        """
        Check whether the current trace shows a serve-like trajectory:
        net downward (gravity after contact) and net horizontal (ball traveling
        toward/away from the net). Uses the most recent 0.3 s of trace_points().
        Sets self._trace_confirmed = True on first qualifying observation.
        """
        pts = list(self._trace_ball_history)  # (t, x, y)
        if len(pts) < 2:
            return

        # Use the last 0.3 s; fall back to all available points if fewer.
        t_cutoff = pts[-1][0] - 0.3
        recent = [(t, x, y) for t, x, y in pts if t >= t_cutoff]
        if len(recent) < 2:
            recent = pts[-min(5, len(pts)):]
        if len(recent) < 2:
            return

        t0, x0, y0 = recent[0]
        t1, x1, y1 = recent[-1]
        dt = t1 - t0
        if dt <= 0:
            return

        dy_per_s = (y1 - y0) / dt   # positive = ball moving downward (pixel Y increases)
        dx_per_s = abs(x1 - x0) / dt

        if (dy_per_s >= self.cfg.trace_downward_px_s and
                dx_per_s >= self.cfg.trace_horizontal_px_s):
            self._trace_confirmed = True
            print(f"[TRACE-CONFIRM{self._log_suffix}] Serve trace confirmed: "
                  f"dy/dt={dy_per_s:.0f} px/s ↓, dx/dt={dx_per_s:.0f} px/s ↔")

    def _init_active(self, now: float) -> None:
        self._reset_active_state()
        self.active_start_time      = now
        self.last_active_trace_time = now   # dead-timeout clock starts at ACTIVE entry
        self.last_transition_time   = None


# =====================================================================
# Far-side variant — identical algorithm, far baseline + far-player fields
# =====================================================================
class FarSideTransitionEngine(TransitionEngine):
    """
    Far-side serve detector. Runs the same WAITING -> ARMED -> ACTIVE
    algorithm as TransitionEngine for the WAITING and ACTIVE phases
    (ready-band gating, ball-trace point-end), gating on far_player_box /
    far_player_world and the far baseline (Config.COURT_LENGTH_FT) instead of
    the near ones, with a wider ready band for far-court position noise.

    The ARMED -> ACTIVE decision is overridden entirely: there is no
    trophy/toss weighted score here — the far side has no ball-toss or
    trophy-pose signal of its own, so ARMED -> ACTIVE fires solely off
    frame.far_serve_score (see anya_base.FarSideTelemetryProvider /
    serve_stgcn.ServeSTGCNDetector), reusing cfg.serve_score_threshold as the
    probability cutoff.
    """

    def __init__(self, fps: float, cfg: Optional[SignalPriorityConfig] = None):
        super().__init__(
            fps, cfg,
            side_label="far",
            player_box_attr="far_player_box",
            player_world_attr="far_player_world",
            baseline_y_ft=Config.COURT_LENGTH_FT,
            direction=1.0,
        )
        # Wider ready band than the near side: far-court world-position
        # estimates are noisier (homography amplifies feet-pixel jitter more
        # at this distance), so a tight +/-baseline window misses real
        # ready-position dwells.
        self.READY_MIN_DIST_FT = -6.0
        self.READY_MAX_DIST_FT = 6.0

    # ==================================================================
    # ARMED -> ACTIVE  or  ARMED -> WAITING  (ST-GCN-only serve decision)
    # ==================================================================
    def _check_armed(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        in_band = False
        world = getattr(frame, self._player_world_attr, None)
        if world is not None:
            _, wy       = world
            signed_dist = (wy - self._baseline_y_ft) * self._direction
            in_band     = self.READY_MIN_DIST_FT <= signed_dist <= self.READY_MAX_DIST_FT

        self.armed_band_history.append((now, in_band))
        while (self.armed_band_history and
               now - self.armed_band_history[0][0] > self.ARMED_BAND_WINDOW_SEC):
            self.armed_band_history.popleft()

        if len(self.armed_band_history) > 1:
            total_time = self.armed_band_history[-1][0] - self.armed_band_history[0][0]
            if total_time > 1.0:
                time_out = sum(
                    self.armed_band_history[i + 1][0] - self.armed_band_history[i][0]
                    for i in range(len(self.armed_band_history) - 1)
                    if not self.armed_band_history[i][1]
                )
                out_ratio = time_out / total_time
                if out_ratio > self.ARMED_OUT_RATIO_THRESHOLD:
                    print(f"[TRANSITION] ARMED -> WAITING{self._log_suffix}. "
                          f"Out of band {out_ratio:.0%} over {total_time:.1f}s.")
                    self._reset_armed_state()
                    return "WAITING"

        player_box = getattr(frame, self._player_box_attr, None)
        if not in_band or player_box is None:
            return "ARMED"

        stgcn_score = getattr(frame, "far_serve_score", 0.0) or 0.0
        self.last_serve_scores = {"stgcn_score": stgcn_score, "serve_score": stgcn_score}

        if stgcn_score >= self.cfg.serve_score_threshold:
            print(f"[TRANSITION] ARMED -> ACTIVE{self._log_suffix}. "
                  f"Serve detected! ST-GCN score: {stgcn_score:.2f}")
            self._reset_armed_state()
            self._init_active(now)
            return "ACTIVE"

        return "ARMED"