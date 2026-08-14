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

    # Far-side motion-history toss fallback (blended into the toss score).  Only
    # consulted when serve_side == "far"; near side leaves this disabled.
    use_mhi_toss:         bool  = False
    mhi_threshold:        float = 0.30  # MHI must exceed this to contribute
    mhi_max_contribution: float = 0.50  # MHI at 1.0 adds this many toss points

    # ---- Tier 1: POINT END (ACTIVE -> WAITING) — ball-trace authority ----
    min_point_duration_s: float = 2.0   # earliest a point may end           (was 3.0)
    trace_dead_timeout_s: float = 3.5   # trace-dark grace before ending      (was 3.0)
    move_velocity_floor_px_s: float = 20.0  # perspective-scaled; below = not live motion

    # ---- Tier 2: near-player velocity modulation of the dead-timeout ----
    use_player_velocity:     bool  = True
    player_velocity_window_s: float = 0.5   # robust to the ACTIVE player-detect stride
    player_stationary_ft_s:  float = 1.0    # <= this  -> standing around -> shorten timeout
    player_walking_ft_s:     float = 2.5    # >= this  -> chasing/walking  -> lengthen timeout
    stationary_timeout_mult: float = 0.8
    walking_timeout_mult:    float = 1.2

    # ---- Cycle ----
    fast_rearm: bool = True             # ACTIVE -> ARMED directly if player already at baseline

    @classmethod
    def far(cls) -> "SignalPriorityConfig":
        """
        Forgiving configuration for far-side serve detection.

        The far player is small, the toss ball is faint, and the trophy pose is
        unreliable from the opposite camera angle, so we lean almost entirely on
        the toss (YOLO + motion-history fallback), confirm on a single frame, and
        drop the score threshold slightly.
        """
        return cls(
            trophy_weight=0.05,
            toss_weight=0.95,
            serve_score_threshold=0.50,
            toss_conf_floor=0.10,      # ball is faint at distance
            toss_gap_tolerance=2,      # tolerate more dropped frames
            toss_confirm_frames=1,     # a single good frame confirms
            use_mhi_toss=True,
        )


class TransitionEngine:
    def __init__(self, fps: float, cfg: Optional[SignalPriorityConfig] = None,
                 serve_side: str = "near"):
        if serve_side not in ("near", "far"):
            raise ValueError(f"serve_side must be 'near' or 'far', got {serve_side!r}")
        self.serve_side = serve_side
        self.fps = fps
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = SignalPriorityConfig.far() if serve_side == "far" else SignalPriorityConfig()

        # ------------------------------------------------------------------
        # WAITING / ARMED-band geometry (point-start ZONE gating).
        # The ready band is measured behind the *serving* baseline.  The far
        # side uses a wider band and more lenient out-of-band tolerance because
        # the far box jitters (net occlusion, small size) more than the near.
        # ------------------------------------------------------------------
        if serve_side == "far":
            self.READY_MIN_DIST_FT   = -1.0
            self.READY_MAX_DIST_FT   = 6.0
            self.READY_WAIT_TIME_SEC = 0.4
            self.ARMED_BAND_WINDOW_SEC     = 2.0
            self.ARMED_OUT_RATIO_THRESHOLD = 0.45   # tolerate more jitter than near (0.25)
            self.READY_MISS_GRACE_FRAMES   = 10     # missed far detections before timer resets
        else:
            self.READY_MIN_DIST_FT   = -0.5
            self.READY_MAX_DIST_FT   = 3.5
            self.READY_WAIT_TIME_SEC = 0.4
            self.ARMED_BAND_WINDOW_SEC     = 2.0
            self.ARMED_OUT_RATIO_THRESHOLD = 0.25
            self.READY_MISS_GRACE_FRAMES   = 0      # near box is reliable every frame

        # Baseline-behind sign convention used by the ready-band helpers.
        self._serve_baseline_ft = 0.0 if serve_side == "near" else Config.COURT_LENGTH_FT

        # Missed-detection grace counter for the ready hold timer (far side).
        self._ready_miss_frames = 0

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

        # Tier-2: near-player velocity (windowed, robust to player-detect stride)
        self._player_world_history: deque = deque()    # (t, world_y_ft)
        self.player_velocity_ft_s: float  = 0.0
        self.player_velocity_valid: bool  = False

        # Signal to the main loop: timestamp to truncate output on transition.
        self.last_transition_time: Optional[float] = None

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
    # Ready-band helpers (serve-side aware)
    # ==================================================================
    def _ready_behind_ft(self, world) -> float:
        """Signed distance (ft) the serving player stands *behind* their baseline.

        Positive = behind the court (the normal ready position).  Measured from
        the near baseline (y=0) for near serves and the far baseline (y=78) for
        far serves.
        """
        _, wy = world
        if self.serve_side == "near":
            return -wy                                   # behind near baseline → wy < 0
        return wy - self._serve_baseline_ft              # behind far baseline → wy > 78

    def _in_ready_band(self, world) -> bool:
        """True if the serving player's feet lie in the ready band behind baseline."""
        if world is None:
            return False
        b = self._ready_behind_ft(world)
        if self.serve_side == "near" and not (world[1] < 0):
            return False                                 # preserve exact near-side gate
        return self.READY_MIN_DIST_FT <= b <= self.READY_MAX_DIST_FT

    # ==================================================================
    # WAITING -> ARMED   (player settles into the ready band)
    # ==================================================================
    def _check_waiting(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp
        world = frame.serve_player_world

        # Missed-detection grace (far side): a dropped box doesn't immediately
        # reset the hold timer — the player rarely teleports out of the band.
        if world is None:
            self._ready_miss_frames += 1
            if self._ready_miss_frames > self.READY_MISS_GRACE_FRAMES:
                self.near_ready_start_time = None
                self._ready_miss_frames    = 0
            return "WAITING"

        if self._in_ready_band(world):
            self._ready_miss_frames = 0
            if self.near_ready_start_time is None:
                self.near_ready_start_time = now
            elapsed = now - self.near_ready_start_time
            if elapsed > self.READY_WAIT_TIME_SEC:
                print(f"[TRANSITION] WAITING -> ARMED ({self.serve_side}). "
                      f"Player held ready for {elapsed:.1f}s.")
                self.near_ready_start_time = None
                return "ARMED"
        else:
            # Player detected but out of zone — genuine, reset immediately.
            self.near_ready_start_time = None
            self._ready_miss_frames    = 0

        return "WAITING"

    # ==================================================================
    # ARMED -> ACTIVE  or  ARMED -> WAITING   (Tier-1 serve detection)
    # ==================================================================
    def _check_armed(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        in_band = self._in_ready_band(frame.serve_player_world)

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
                    print(f"[TRANSITION] ARMED -> WAITING. "
                          f"Out of band {out_ratio:.0%} over {total_time:.1f}s.")
                    self._reset_armed_state()
                    return "WAITING"

        if not in_band or frame.serve_player_box is None:
            return "ARMED"

        nx1, ny1, nx2, ny2 = frame.serve_player_box

        # --- Trophy pose (Tier-1 confirmer, low weight) ---
        trophy_score = getattr(frame, "trophy_score", 0.0) or 0.0
        if trophy_score >= self.cfg.trophy_conf_floor and trophy_score > 0:
            self._trophy_scores.append((trophy_score, now))

        # --- Ball toss (Tier-1 anchor, high weight) ---
        # Far side blends in a motion-history fallback: when YOLO misses the
        # faint ball, sustained head-region motion still contributes a partial
        # toss score (bounded so MHI can never fire a serve on its own).
        toss_score = self._update_toss_detection(frame, ny1, now)
        if self.cfg.use_mhi_toss:
            mhi = getattr(frame, "mhi_toss_score", 0.0) or 0.0
            if mhi >= self.cfg.mhi_threshold:
                toss_score = max(toss_score, self.cfg.mhi_max_contribution * mhi)
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
            print(f"[TRANSITION] ARMED -> ACTIVE ({self.serve_side}). "
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
        # Detections inside the near player's box are excluded from the trace:
        # the racket/arm/body frequently throws off ball-shaped false positives
        # right where the near player stands, and unlike the far side we have a
        # reliable box for them every frame.
        nbox = frame.near_player_box
        if nbox is not None:
            nx1, ny1, nx2, ny2 = nbox
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

        # ---- 7. Point over — rewind to the last real ball detection ----
        death_t = self.ball_tracker.last_detection_time
        if death_t is None:
            death_t = self.last_active_trace_time
        death_t = max(self.active_start_time, death_t)

        self.last_transition_time = death_t
        vel_note = (f"player {self.player_velocity_ft_s:.1f} ft/s, "
                    f"timeout {effective_timeout:.1f}s"
                    if self.player_velocity_valid else f"timeout {effective_timeout:.1f}s")
        print(f"\n[TRANSITION] ACTIVE -> WAITING (trace {status.state}; {vel_note}). "
              f"Lasted {elapsed:.1f}s. Rewind to t={death_t:.2f}s.")

        next_world = frame.serve_player_world
        self._reset_active_state()

        if self.cfg.fast_rearm:
            return self._post_active_next_state(next_world, "WAITING")
        return "WAITING"

    # ------------------------------------------------------------------
    # Tier-2 helpers — near-player velocity
    # ------------------------------------------------------------------
    def _update_player_velocity(self, frame, now: float) -> None:
        """
        Maintain a short rolling window of the near player's world-y position and
        derive a walking speed (ft/s). A window (not a frame-to-frame diff) is
        used because the player model only runs every ACTIVE_PLAYER_STRIDE frames
        in anya_base — consecutive frames often repeat a cached position, which a
        naive diff would read as 0 then spike. Over a ~0.5 s window there is at
        least one genuine refresh, giving a stable estimate.
        """
        if not self.cfg.use_player_velocity or frame.near_player_world is None:
            return

        _, wy = frame.near_player_world
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
    def _post_active_next_state(self, serve_pos, default_state: str) -> str:
        """
        On ACTIVE -> WAITING, bypass WAITING if the serving player is already
        inside the ready band — go straight to ARMED for the next point. Cuts
        dead time between consecutive serves from the same end.
        """
        if serve_pos is not None and self._in_ready_band(serve_pos):
            print("[BYPASS] Player already at baseline. Jumping to ARMED.")
            self._reset_armed_state()
            return "ARMED"
        return default_state

    def _reset_armed_state(self) -> None:
        self.armed_band_history.clear()
        self._ready_miss_frames            = 0
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
        self._player_world_history.clear()
        self.player_velocity_ft_s   = 0.0
        self.player_velocity_valid  = False
        self.ball_tracker.reset()

    def _init_active(self, now: float) -> None:
        self._reset_active_state()
        self.active_start_time      = now
        self.last_active_trace_time = now   # dead-timeout clock starts at ACTIVE entry
        self.last_transition_time   = None