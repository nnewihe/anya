"""
anya_transitions.py
===================
State machine logic. Determines transitions between WAITING, ARMED, and ACTIVE
based on the rolling telemetry buffer from anya_base.py.

ACTIVE → WAITING is driven purely by the ball trace.  Per-frame whole-court ball
detections are handed to a trajectory-coherent Kalman tracker
(ball_tracker.BallTrackManager), which fuses them into a single coherent ball
trajectory and reports whether a *moving* trace is alive right now.  The point
stays ACTIVE while that trace is alive and ends the moment it stops or
disappears — subject to a short minimum-point-duration grace so the tracker has
time to lock onto the served ball after the serve.

A moving trace requires the ball to have been detected recently (within
miss_timeout_s, so brief occlusions are bridged) AND to have actually moved
within the recent motion window — so a ball that rolls to rest in view, a
stationary stray detection, or scattered false positives never sustain play.

When the point ends, `last_transition_time` is set to the timestamp of the last
real ball detection (clamped to the point's start) — i.e. when the point
effectively died.  The main loop uses this to rewind output-video writing.
"""

from collections import deque
from typing import List, Optional, Tuple
import math

import numpy as np

from ball_tracker import BallTrackManager, TrackStatus, make_image_row_perspective


class TransitionEngine:
    def __init__(self, fps: float):
        self.fps = fps

        # ------------------------------------------------------------------
        # WAITING
        # ------------------------------------------------------------------
        self.READY_MIN_DIST_FT   = -0.5
        self.READY_MAX_DIST_FT   = 3.5
        self.READY_WAIT_TIME_SEC = 0.4

        # ------------------------------------------------------------------
        # ARMED
        # ------------------------------------------------------------------
        self.ARMED_BAND_WINDOW_SEC      = 2.0
        self.ARMED_OUT_RATIO_THRESHOLD  = 0.25
        self.TRANSITION_SCORE_THRESHOLD = 0.55
        self.EVENT_WINDOW_SECONDS       = 1.2

        # ------------------------------------------------------------------
        # ACTIVE — Ball-trace tracker (sole point-end authority)
        # ------------------------------------------------------------------
        self.SCREEN_HEIGHT_PX = 540   # analysis-frame height (px); drives perspective model

        # Minimum point duration before the point can end at all.
        self.MIN_POINT_DURATION_SEC = 3.0

        # How long the point stays ACTIVE after the ball trace goes dark.
        # Bridges long occlusions (player body, net post) and gives the tracker
        # time to re-acquire the ball after track loss.  The point ends only
        # when the trace has been absent for this long AND MIN_POINT_DURATION_SEC
        # has already passed.
        self.TRACE_DEAD_TIMEOUT_SEC = 3.0

        # Single-ball Kalman tracker.  Point-end tuning knobs live in
        # ball_tracker.py (miss_timeout_s, move_thresh_px, confirm_hits,
        # corroboration_window_s, …); pass overrides here if needed.
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
        # Persistent state — ACTIVE (ball tracking, pixel space)
        # ------------------------------------------------------------------
        self.active_start_time: float = 0.0

        # Moving-ball trajectory for the visual overlay: (t, px, py).
        # Synced from the tracker each frame so run_anya can draw the trail.
        self._trace_ball_history: deque = deque()

        # Timestamp of the most recent frame that had an active trace.
        self.last_active_trace_time: float = 0.0

        # ------------------------------------------------------------------
        # Signal to the main loop: timestamp to truncate output on transition.
        # ------------------------------------------------------------------
        self.last_transition_time: Optional[float] = None

        # Debug snapshot for HUD / CSV
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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # WAITING → ARMED
    # ------------------------------------------------------------------

    def _check_waiting(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        if frame.near_player_world is None:
            self.near_ready_start_time = None
            return "WAITING"

        _, wy   = frame.near_player_world
        dist_ft = abs(wy)
        in_zone = wy < 0 and self.READY_MIN_DIST_FT <= dist_ft <= self.READY_MAX_DIST_FT

        if in_zone:
            if self.near_ready_start_time is None:
                self.near_ready_start_time = now
            elapsed = now - self.near_ready_start_time
            if elapsed > self.READY_WAIT_TIME_SEC:
                print(f"[TRANSITION] WAITING -> ARMED. "
                      f"Player held ready for {elapsed:.1f}s.")
                self.near_ready_start_time = None
                return "ARMED"
        else:
            self.near_ready_start_time = None

        return "WAITING"

    # ------------------------------------------------------------------
    # ARMED → ACTIVE  or  ARMED → WAITING
    # ------------------------------------------------------------------

    def _check_armed(self, history: deque) -> str:
        frame = history[-1]
        now   = frame.timestamp

        in_band = False
        if frame.near_player_world is not None:
            _, wy   = frame.near_player_world
            dist_ft = abs(wy)
            in_band = wy < 0 and self.READY_MIN_DIST_FT <= dist_ft <= self.READY_MAX_DIST_FT

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
                    print(f"[TRANSITION] ARMED -> WAITING. "
                          f"Out of band {out_ratio:.0%} over {total_time:.1f}s.")
                    self._reset_armed_state()
                    return "WAITING"

        if not in_band or frame.near_player_box is None:
            return "ARMED"

        nx1, ny1, nx2, ny2 = frame.near_player_box

        trophy_score = getattr(frame, "trophy_score", 0.0) or 0.0
        if trophy_score > 0:
            self._trophy_scores.append((trophy_score, now))

        toss_score = self._update_toss_detection(frame, ny1, now)
        if toss_score > 0:
            self._toss_scores.append((toss_score, now))

        for buf in (self._trophy_scores, self._toss_scores):
            while buf and now - buf[0][1] > self.EVENT_WINDOW_SECONDS:
                buf.popleft()

        max_trophy  = max((s for s, _ in self._trophy_scores), default=0.0)
        max_toss    = max((s for s, _ in self._toss_scores),   default=0.0)
        serve_score = 0.2 * max_trophy + 0.8 * max_toss

        self.last_serve_scores = {
            "trophy_score": max_trophy,
            "toss_score":   max_toss,
            "serve_score":  serve_score,
        }

        if serve_score >= self.TRANSITION_SCORE_THRESHOLD:
            if self.toss_min_y_px is not None and self.toss_min_y_px >= ny1:
                print(f"[DEBUG] Toss height invalid: min_y={self.toss_min_y_px:.1f} "
                      f"must be < player_top={ny1}")
                self.toss_min_y_px = None
                return "ARMED"

            toss_h_str = (f"{self.toss_min_y_px:.1f}px (above {ny1})"
                          if self.toss_min_y_px is not None else "N/A")
            print(f"[TRANSITION] ARMED -> ACTIVE. "
                  f"Serve detected! Score: {serve_score:.2f}  "
                  f"Toss height: {toss_h_str}")
            self._reset_armed_state()
            self._init_active(now)
            return "ACTIVE"

        return "ARMED"

    def _update_toss_detection(self, frame, ny1: float, now: float) -> float:
        if not frame.toss_ball_candidates:
            self.last_toss_ball    = None
            self.toss_gap_frames  += 1
            if self.toss_gap_frames > 3:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False
            return 0.0

        best = max(frame.toss_ball_candidates, key=lambda x: x["conf"])
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
            self.toss_gap_frames              = 0
            self.toss_consecutive_frames     += 1
            self.toss_ball_above_head_detected = True
        else:
            self.toss_gap_frames += 1
            if self.toss_gap_frames > 3:
                self.toss_consecutive_frames       = 0
                self.toss_ball_above_head_detected = False

        if not self.toss_ball_above_head_detected:
            return 0.0
        if self.toss_consecutive_frames >= 3:
            return 1.0
        if self.toss_consecutive_frames >= 2:
            return 0.7
        return 0.0

    # ------------------------------------------------------------------
    # ACTIVE → WAITING  (pure ball-trace)
    # ------------------------------------------------------------------

    def _check_active(self, history: deque) -> str:
        frame      = history[-1]
        now        = frame.timestamp
        candidates = frame.active_ball_candidates or []

        # ---- 1. Feed this frame's ball detections to the tracker ----
        detections = [
            (c["pixel_center"][0], c["pixel_center"][1], c.get("conf", 0.0))
            for c in candidates
        ]
        status = self.ball_tracker.update(detections, now)

        # ---- 2. Sync the moving-ball trace for the visual overlay ----
        self._trace_ball_history.clear()
        self._trace_ball_history.extend(self.ball_tracker.trace_points())

        if status.has_moving_trace:
            self.last_active_trace_time = now

        # ---- 3. Update debug snapshot ----
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

        # ---- 4. Stay ACTIVE while a moving trace exists, or within the
        #         minimum-duration grace, or within the post-trace timeout.
        #         last_active_trace_time is initialised to active_start_time so
        #         the post-trace clock starts ticking from the moment ACTIVE
        #         begins if the trace never comes alive. ----
        elapsed        = now - self.active_start_time
        trace_dead_for = now - self.last_active_trace_time

        if status.has_moving_trace:
            return "ACTIVE"
        if elapsed < self.MIN_POINT_DURATION_SEC:
            return "ACTIVE"
        if trace_dead_for < self.TRACE_DEAD_TIMEOUT_SEC:
            return "ACTIVE"

        # ---- 5. Trace has been gone for TRACE_DEAD_TIMEOUT_SEC → point over.
        #         Rewind to the last real ball detection (clamped to point start). ----
        death_t = self.ball_tracker.last_detection_time
        if death_t is None:
            death_t = self.last_active_trace_time
        death_t = max(self.active_start_time, death_t)

        self.last_transition_time = death_t
        print(f"\n[TRANSITION] ACTIVE -> WAITING (trace {status.state}). "
              f"Lasted {elapsed:.1f}s. Rewind to t={death_t:.2f}s.")
        self._reset_active_state()
        return "WAITING"

    # ------------------------------------------------------------------
    # Helpers — reset / init
    # ------------------------------------------------------------------

    def _post_active_next_state(self, near_pos, default_state: str) -> str:
        """
        On ACTIVE → WAITING, bypass WAITING if the player is already inside
        the ready zone — go straight to ARMED for the next point.
        """
        if near_pos is not None:
            _, wy   = near_pos
            dist_ft = abs(wy)
            if wy < 0 and self.READY_MIN_DIST_FT <= dist_ft <= self.READY_MAX_DIST_FT:
                print("[BYPASS] Player already at baseline. Jumping to ARMED.")
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
        self.ball_tracker.reset()

    def _init_active(self, now: float) -> None:
        self._reset_active_state()
        self.active_start_time      = now
        self.last_active_trace_time = now
        self.last_transition_time   = None

