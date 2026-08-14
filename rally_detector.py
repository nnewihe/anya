"""
rally_detector.py
==================
Detects and clips active rally segments from a tennis video based on the
ball trace tracker.  Unlike run_anya.py, this bypasses the serve-detection
state machine (WAITING -> ARMED -> ACTIVE) and drives segment boundaries
directly from whether a moving ball trace is present.

Segment rules
-------------
  • A segment starts when the ball trace becomes active (has_moving_trace).
  • A segment ends when the trace goes inactive (with a 1 s end-pad).
  • Raw segments whose gap is < 4 s are merged into one segment.
    The gap is measured on the raw (unpadded) segment boundaries — the
    pre-roll added later does NOT count toward the gap.
  • A 1.5 s pre-roll is prepended to each final (post-merge) segment start.
"""

import argparse
import math
import os
from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

from anya_base import AnyaTelemetryProvider
from ball_tracker import BallTrackManager, make_image_row_perspective
from utilities import create_highlights_ffmpeg


RALLY_GAP_THRESHOLD_SEC = 4.0
RALLY_PRE_ROLL_SEC      = 1.5
RALLY_END_PAD_SEC       = 1.0

# ── Player-carry (velocity-coupling) suppression ──────────────────────────────
# When the near player walks while holding the ball, the ball's pixel velocity is
# coupled to the player's body velocity — same direction, same magnitude — so the
# ball appears to "ride along" with the player.  When the ball is actually struck
# (serve toss or stroke) it decouples: it flies far faster than the body and on
# its own heading.  We smooth both velocity vectors over a short window and treat
# the trace as a carried ball (suppress it) while it stays coupled to the player.
#
# coupling ratio = |v_ball - v_player| / |v_ball|
#   • ≈ 0  → ball rides with the body (carried)   → suppress
#   • ≈ 1+ → ball moves independently (struck)     → keep
COUPLING_WINDOW_SEC       = 0.40   # smoothing window for both velocity estimates
COUPLING_MIN_PLAYER_SPEED = 25.0   # px/s; the player must actually be walking
COUPLING_RATIO_MAX        = 0.50   # carried if ratio below this (and player moving)


class _SmoothedVelocity:
    """
    Sliding-window velocity estimator over a stream of (t, x, y) pixel samples.

    Velocity is the displacement between the oldest and newest samples inside a
    `window_sec` window divided by their time span — a cheap, jitter-tolerant
    smoother.  Samples older than the window are pruned on every add.
    """

    def __init__(self, window_sec: float):
        self.window_sec = float(window_sec)
        self._pts: Deque[Tuple[float, float, float]] = deque()

    def add(self, t: float, x: float, y: float) -> None:
        self._pts.append((float(t), float(x), float(y)))
        cutoff = float(t) - self.window_sec
        while self._pts and self._pts[0][0] < cutoff:
            self._pts.popleft()

    def velocity(self) -> Optional[Tuple[float, float]]:
        """(vx, vy) in px/s over the window, or None if span/samples insufficient."""
        if len(self._pts) < 2:
            return None
        t0, x0, y0 = self._pts[0]
        t1, x1, y1 = self._pts[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        return ((x1 - x0) / dt, (y1 - y0) / dt)


def _coupling_ratio(v_ball, v_player) -> Tuple[Optional[float], float]:
    """
    Return (ratio, player_speed) where ratio = |v_ball - v_player| / |v_ball|.

    ratio is None when it cannot be computed (missing velocity or a near-zero
    ball speed).  player_speed (px/s) is returned alongside so the caller can
    require the player to actually be moving before declaring a carried ball.
    """
    if v_ball is None or v_player is None:
        return None, 0.0
    bx, by = v_ball
    px, py = v_player
    ball_speed   = math.hypot(bx, by)
    player_speed = math.hypot(px, py)
    if ball_speed < 1e-6:
        return None, player_speed
    return math.hypot(bx - px, by - py) / ball_speed, player_speed


def _is_carried(v_ball, v_player) -> Tuple[bool, Optional[float]]:
    """
    Decide whether the ball is being carried (coupled to a walking player).

    Carried when the player is genuinely moving (≥ COUPLING_MIN_PLAYER_SPEED) and
    the ball's velocity closely matches it (ratio < COUPLING_RATIO_MAX).  Returns
    (carried, ratio) — ratio is surfaced for the debug overlay.
    """
    ratio, player_speed = _coupling_ratio(v_ball, v_player)
    if ratio is None:
        return False, None
    carried = ratio < COUPLING_RATIO_MAX and player_speed >= COUPLING_MIN_PLAYER_SPEED
    return carried, ratio


def _box_center(box):
    """Return the (cx, cy) center of a bounding box, or None."""
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _origin_side(ball_xy, near_box, far_box) -> str:
    """
    Classify a serve's origin as "near" or "far" by which player box the ball is
    closest to at the moment the segment opens.

    A near serve's ball originates at the near player (large-y, bottom of frame);
    a far serve's ball originates at the far player (small-y, top of frame).  We
    compare the ball-to-box-center distance for whichever boxes are present.
    Falls back to "near" when neither box is available (conservative: lets the
    near-side serve-gated pipeline own ambiguous cases).
    """
    if ball_xy is None:
        return "near"
    near_c = _box_center(near_box)
    far_c  = _box_center(far_box)
    if near_c is None and far_c is None:
        return "near"
    if far_c is None:
        return "near"
    if near_c is None:
        return "far"
    d_near = math.hypot(ball_xy[0] - near_c[0], ball_xy[1] - near_c[1])
    d_far  = math.hypot(ball_xy[0] - far_c[0],  ball_xy[1] - far_c[1])
    return "far" if d_far < d_near else "near"


def _merge_segments(segments, gap_threshold_sec=RALLY_GAP_THRESHOLD_SEC):
    """
    Merge adjacent segments whose gap < gap_threshold_sec.

    Each segment is (start, end, origin); the merged segment keeps the origin of
    its FIRST sub-segment (the serve that opened the run).
    """
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end, origin in segments[1:]:
        if start - merged[-1][1] < gap_threshold_sec:
            merged[-1][1] = end
        else:
            merged.append([start, end, origin])
    return [tuple(s) for s in merged]


def _apply_preroll(segments, pre_roll_sec=RALLY_PRE_ROLL_SEC):
    return [(max(0.0, start - pre_roll_sec), end, origin)
            for start, end, origin in segments]


def collect_rally_segments(video_path, headless=False, start_frame=0):
    """
    Run the trace-driven rally detector and return its segments.

    Returns a list of (start_sec, end_sec, origin) in source-video time, where
    origin ∈ {"near", "far"} is the serve side classified by the nearest player
    box at the moment the segment opened.  Performs no video writing — the caller
    (detect_rallies or the combined orchestrator) decides what to do with them.
    """
    # ── Probe video ───────────────────────────────────────────────────────
    _probe = cv2.VideoCapture(video_path)
    orig_fps    = _probe.get(cv2.CAP_PROP_FPS)
    total_frames = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    _probe.release()
    if orig_fps <= 0 or orig_fps > 300:
        orig_fps = 30.0
    video_duration_sec = total_frames / orig_fps if total_frames > 0 else float("inf")
    video_time_offset  = start_frame / orig_fps

    # ── Telemetry provider — forced into ACTIVE so ball detection runs ────
    telemetry_provider = AnyaTelemetryProvider(video_path)
    telemetry_provider.update_state("ACTIVE")

    # ── Ball tracker (independent from TransitionEngine) ──────────────────
    ball_tracker = BallTrackManager(
        fps=telemetry_provider.fps,
        perspective_scale=make_image_row_perspective(540),
    )

    # ── Main loop ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[RALLY] Seeking to frame {start_frame}")

    # Smoothed velocity windows for the carry-coupling test (Idea C).  The ball
    # window is fed the IMM-smoothed ball position; the player window is fed the
    # near-player-box centroid.
    ball_vel   = _SmoothedVelocity(COUPLING_WINDOW_SEC)
    player_vel = _SmoothedVelocity(COUPLING_WINDOW_SEC)

    raw_segments = []
    seg_start:  float = None
    seg_origin: str   = "near"
    last_ts:    float = 0.0
    interrupted = False

    try:
        while cap.isOpened():
            success, orig_frame = cap.read()
            if not success:
                break

            frame    = cv2.resize(orig_frame, (960,540), interpolation=cv2.INTER_LINEAR)
            telemetry = telemetry_provider.process_frame(frame)
            last_ts   = telemetry.timestamp

            near_box = telemetry.near_player_box
            far_box  = telemetry.far_player_box
            dets = [
                (c["pixel_center"][0], c["pixel_center"][1], c["conf"])
                for c in (telemetry.active_ball_candidates or [])
            ]
            status = ball_tracker.update(dets, telemetry.timestamp)

            # Feed the velocity windows every frame so they stay populated across
            # the carry → strike transition (Idea C).  Ball: IMM-smoothed position;
            # player: near-player-box centroid.
            if status.position is not None:
                ball_vel.add(telemetry.timestamp, status.position[0], status.position[1])
            if near_box is not None:
                player_vel.add(
                    telemetry.timestamp,
                    (near_box[0] + near_box[2]) / 2.0,
                    (near_box[1] + near_box[3]) / 2.0,
                )

            # Suppress balls whose velocity is coupled to a walking player
            # (carried, not struck).
            carried        = False
            coupling_ratio = None
            if status.has_moving_trace and status.position is not None:
                carried, coupling_ratio = _is_carried(
                    ball_vel.velocity(), player_vel.velocity()
                )
            trace_active = status.has_moving_trace and not carried

            if trace_active:
                if seg_start is None:
                    seg_start = video_time_offset + telemetry.timestamp
                    # Classify serve origin by the nearest player box at the
                    # moment the segment opens (the serve contact / toss).
                    seg_origin = _origin_side(status.position, near_box, far_box)
            else:
                if seg_start is not None:
                    raw_end = video_time_offset + (
                        ball_tracker.last_detection_time
                        if ball_tracker.last_detection_time is not None
                        else telemetry.timestamp
                    )
                    padded_end = min(raw_end + RALLY_END_PAD_SEC, video_duration_sec)
                    raw_segments.append((seg_start, padded_end, seg_origin))
                    seg_start = None

            if not headless:
                _render_overlay(frame, status, carried, coupling_ratio)
                cv2.imshow("Rally Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERRUPTED] Ctrl-C — saving completed segments...")

    finally:
        if seg_start is not None:
            raw_end = video_time_offset + (
                ball_tracker.last_detection_time
                if ball_tracker.last_detection_time is not None
                else last_ts
            )
            padded_end = min(raw_end + RALLY_END_PAD_SEC, video_duration_sec)
            raw_segments.append((seg_start, padded_end, seg_origin))

        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    print(f"[RALLY] Raw segments: {len(raw_segments)}")
    if interrupted:
        print("[RALLY] (interrupted — covers completed detections only)")

    # ── Post-process: merge, then add pre-roll ────────────────────────────
    merged = _merge_segments(raw_segments, RALLY_GAP_THRESHOLD_SEC)
    print(f"[RALLY] After merging (gap < {RALLY_GAP_THRESHOLD_SEC:.0f}s): {len(merged)} segment(s)")

    final = _apply_preroll(merged, RALLY_PRE_ROLL_SEC)
    n_far  = sum(1 for *_, o in final if o == "far")
    n_near = len(final) - n_far
    print(f"[RALLY] Segment origins: {n_near} near, {n_far} far")
    return final


def detect_rallies(video_path, output_path=None, headless=False, start_frame=0):
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_rallies.mp4")

    final = collect_rally_segments(video_path, headless, start_frame)

    # Standalone rally detector keeps every segment regardless of serve origin.
    create_highlights_ffmpeg(video_path, [(s, e) for s, e, _ in final], output_path)

    print(f"\n[DONE] Output   : {output_path}")
    print(f"[DONE] Segments : {len(final)}")


def _render_overlay(frame, status, carried: bool = False,
                    coupling_ratio: Optional[float] = None):
    """Minimal debug overlay — trace liveness, carry state, and ball trail."""
    suppressed = status.has_moving_trace and carried
    alive = status.has_moving_trace and not suppressed
    color = (0, 255, 0) if alive else ((0, 100, 255) if suppressed else (0, 255, 255))
    tag   = "ALIVE" if alive else ("SUPPRESSED" if suppressed else "DEAD")
    carry_txt = "carry" if carried else ("ratio=%.2f" % coupling_ratio if coupling_ratio is not None else "ratio=--")
    label = f"TRACE: {tag}  {status.speed_px_s:.0f}px/s  {carry_txt}"
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    trace = status.trace
    n = len(trace)
    if n >= 2:
        for i in range(1, n):
            age = i / (n - 1)
            c   = (0, int(120 * age), int(255 * age))
            cv2.line(frame,
                     (int(trace[i-1][0]), int(trace[i-1][1])),
                     (int(trace[i][0]),   int(trace[i][1])),
                     c, max(1, int(3 * age)), cv2.LINE_AA)
    if n >= 1:
        cv2.circle(frame, (int(trace[-1][0]), int(trace[-1][1])),
                   5, (0, 200, 255), -1, cv2.LINE_AA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rally Detector — clips active ball-trace segments from a tennis video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rally_detector.py match.mp4
  python rally_detector.py match.mp4 --output rallies.mp4 --headless
  python rally_detector.py match.mp4 --start-frame 9000 --headless
""",
    )
    parser.add_argument("video",        metavar="VIDEO",  help="Input tennis video")
    parser.add_argument("--output",     default=None,     help="Output MP4 path")
    parser.add_argument("--headless",   action="store_true")
    parser.add_argument("--start-frame", type=int, default=0,
                        metavar="N",    help="Start from frame N (default: 0)")
    args = parser.parse_args()

    detect_rallies(args.video, args.output, args.headless, args.start_frame)
