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

Post-processing filters (applied to merged segments)
------------------------------------------------------
  1. Ready-position gate — a NEW segment only opens when the originating
     player is within 4 ft (near) / 8 ft (far) of their baseline.
  2. Serving-pattern HMM — tennis serve sides are sticky (same server for
     an entire game, ~15 points on average).  A Viterbi-decoded HMM over the
     full segment sequence identifies segments whose observed origin disagrees
     with the globally inferred serving side.  Disagreeing segments that are
     also "weak" (no detectable racket contact, short duration) are dropped as
     spurious false positives.  Disagreeing but "strong" segments are kept and
     their origin is relabeled to the inferred side.
"""

import argparse
import math
import os
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from .anya_base import AnyaTelemetryProvider
from .ball_tracker import BallTrackManager, make_image_row_perspective
from .utilities import create_highlights_ffmpeg, open_video


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

# ── Serving-pattern HMM ───────────────────────────────────────────────────────
# HMM transition / emission probabilities fitted from 15 labeled matches in
# /Volumes/Anya/Data (203 stay-transitions, 14 switch-transitions across all GT).
#
# P(stay)   = 0.9355  →  average game ~15.5 points (includes deuce games).
# P(correct)= 0.85    →  probability the per-segment origin label is correct;
#                         estimated from diagnostic recall (9/9 on folder 63);
#                         conservative to avoid over-confident relabelling.
HMM_P_STAY    = 0.9355
HMM_P_CORRECT = 0.85

# Strength gate thresholds — distinguish a real (strong) point from a spurious
# (weak) ball-return or stray trace.  A real point always has at least one hard
# racket strike (the serve), which spikes the IMM's racket_prob (μ₁) for several
# consecutive frames; a ball-return is gentle and short.
#
# A segment is strong if it satisfies EITHER condition (OR gate):
#   • ≥ MIN_RACKET_FRAMES consecutive-ish frames with μ₁ > RACKET_SPIKE_THRESH
#   • Raw trace duration ≥ MIN_SEGMENT_SEC
RACKET_SPIKE_THRESH = 0.25   # μ₁ threshold for a hard racket contact
MIN_RACKET_FRAMES   = 5      # ~1 contact event at 60fps (noise-robust)
MIN_SEGMENT_SEC     = 2.5    # minimum raw trace duration for a "strong" segment


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


def _segment_is_strong(strength: dict) -> bool:
    """
    A segment is "strong" (likely a real point) if it has detectable racket contact
    OR sufficient raw trace duration.  Ball-returns are short and gentle (no strike).
    """
    return (strength["racket_frames"] >= MIN_RACKET_FRAMES
            or strength["duration"] >= MIN_SEGMENT_SEC)


def _viterbi(obs_sides: List[str],
             p_stay: float = HMM_P_STAY,
             p_correct: float = HMM_P_CORRECT) -> List[str]:
    """
    Viterbi decoding of the most likely serving-side state sequence.

    States / observations: "near"=0, "far"=1.
    Transition: symmetric sticky matrix — P(same→same) = p_stay.
    Emission:   P(obs=state) = p_correct  (label is usually right).
    Initial:    uniform (equal prior for who serves first).

    Returns the decoded state sequence — same length as obs_sides.
    """
    n = len(obs_sides)
    if n == 0:
        return []
    if n == 1:
        return list(obs_sides)          # single segment: no context, trust as-is

    sides = ["near", "far"]
    S = 2   # number of states

    def _idx(s): return 0 if s == "near" else 1

    # Log probabilities (avoid underflow)
    log_trans = np.log(np.array([
        [p_stay,        1.0 - p_stay],
        [1.0 - p_stay,  p_stay      ],
    ]))
    log_emit = np.log(np.array([
        [p_correct,        1.0 - p_correct],   # state=near: P(obs=near), P(obs=far)
        [1.0 - p_correct,  p_correct       ],   # state=far:  P(obs=near), P(obs=far)
    ]))
    log_init = np.log(np.array([0.5, 0.5]))

    obs = [_idx(o) for o in obs_sides]

    # Forward pass — delta[s] = log prob of best path ending in state s
    delta = log_init + log_emit[:, obs[0]]
    psi   = np.zeros((n, S), dtype=int)

    for t in range(1, n):
        for s in range(S):
            scores = delta + log_trans[:, s]
            best   = int(np.argmax(scores))
            delta[s]    = scores[best] + log_emit[s, obs[t]]
            psi[t, s]   = best

    # Backtrack
    path    = [0] * n
    path[n-1] = int(np.argmax(delta))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return [sides[s] for s in path]


def _hmm_filter(segments: list) -> list:
    """
    Apply the serving-pattern HMM filter to a list of merged segments.

    Input:  list of (start, end, origin, strength)
    Output: list of (start, end, origin)  — public format, strength stripped.

    For each segment where the observed origin disagrees with the Viterbi-decoded
    serving side (a structural anomaly):
      • Weak segment  → DROP  (spurious false positive: ball-return, stray trace).
      • Strong segment → KEEP, but relabel origin to the decoded side (misclassified
                         real point, e.g. a far point whose ball happened to start
                         near the net and was assigned "near" at open time).

    All agreeing segments are kept unchanged.  This means the filter only touches
    anomalies — it cannot remove a real game-run even if its run-length is short.
    """
    if not segments:
        return []

    obs_sides = [s[2] for s in segments]
    decoded   = _viterbi(obs_sides)

    kept = []
    n_dropped = 0
    n_relabeled = 0

    for (start, end, origin, strength), decoded_side in zip(segments, decoded):
        if origin == decoded_side:
            kept.append((start, end, origin))
        else:
            if _segment_is_strong(strength):
                kept.append((start, end, decoded_side))
                n_relabeled += 1
                print(f"[HMM] Relabelled  {start:.2f}s–{end:.2f}s "
                      f"{origin}→{decoded_side}  "
                      f"(racket_frames={strength['racket_frames']}, "
                      f"dur={strength['duration']:.1f}s)")
            else:
                n_dropped += 1
                print(f"[HMM] Dropped     {start:.2f}s–{end:.2f}s "
                      f"origin={origin} (decoded={decoded_side})  "
                      f"(racket_frames={strength['racket_frames']}, "
                      f"dur={strength['duration']:.1f}s) — weak/spurious")

    print(f"[HMM] Filter result: kept {len(kept)}, "
          f"dropped {n_dropped}, relabelled {n_relabeled}")
    return kept


def _merge_segments(segments, gap_threshold_sec=RALLY_GAP_THRESHOLD_SEC):
    """
    Merge adjacent segments whose gap < gap_threshold_sec.

    Each segment is (start, end, origin, strength).  The merged segment keeps
    the origin of its FIRST sub-segment (the serve that opened the run) and
    accumulates strength (summed racket_frames, summed duration).
    """
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end, origin, strength in segments[1:]:
        if start - merged[-1][1] < gap_threshold_sec:
            merged[-1][1] = end
            # Accumulate strength across sub-segments
            merged[-1][3]["racket_frames"] += strength["racket_frames"]
            merged[-1][3]["duration"]      += strength["duration"]
        else:
            merged.append([start, end, origin, strength])
    # Convert back to tuples with immutable strength dicts
    return [(s[0], s[1], s[2], dict(s[3])) for s in merged]


def _apply_preroll(segments, pre_roll_sec=RALLY_PRE_ROLL_SEC):
    """Apply pre-roll; segments may be 3-tuples (start, end, origin) or 4-tuples."""
    out = []
    for seg in segments:
        start, end, origin = seg[0], seg[1], seg[2]
        out.append((max(0.0, start - pre_roll_sec), end, origin))
    return out


def collect_rally_segments(video_path, headless=False, start_frame=0, progress_cb=None):
    """
    Run the trace-driven rally detector and return its segments.

    Returns a list of (start_sec, end_sec, origin) in source-video time, where
    origin ∈ {"near", "far"} is the serve side classified by the nearest player
    box at the moment the segment opened.  Performs no video writing — the caller
    (detect_rallies or the combined orchestrator) decides what to do with them.

    progress_cb: optional callable(current_frame, total_frames) called every 30 frames.
    """
    # ── Probe video ───────────────────────────────────────────────────────
    _probe = open_video(video_path, "RALLY")
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
    cap = open_video(video_path, "RALLY")
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[RALLY] Seeking to frame {start_frame}")

    # Smoothed velocity windows for the carry-coupling test.  The ball window is
    # fed the IMM-smoothed ball position; the player window is fed the near-player
    # box centroid.
    ball_vel   = _SmoothedVelocity(COUPLING_WINDOW_SEC)
    player_vel = _SmoothedVelocity(COUPLING_WINDOW_SEC)

    # raw_segments items: (start, end, origin, strength)
    # strength = {"racket_frames": int, "duration": float}
    raw_segments: list = []
    seg_start:         float = None
    seg_origin:        str   = "near"
    seg_racket_frames: int   = 0      # frames with μ₁ > RACKET_SPIKE_THRESH this seg
    last_ts:           float = 0.0
    interrupted = False
    frame_num   = start_frame

    try:
        while cap.isOpened():
            success, orig_frame = cap.read()
            if not success:
                break

            frame_num += 1
            frame    = cv2.resize(orig_frame, (960, 540), interpolation=cv2.INTER_LINEAR)
            telemetry = telemetry_provider.process_frame(frame)
            last_ts   = telemetry.timestamp

            if progress_cb is not None and frame_num % 30 == 0:
                progress_cb(frame_num - start_frame, total_frames - start_frame)

            near_box = telemetry.near_player_box
            far_box  = telemetry.far_player_box
            dets = [
                (c["pixel_center"][0], c["pixel_center"][1], c["conf"])
                for c in (telemetry.active_ball_candidates or [])
            ]
            status = ball_tracker.update(dets, telemetry.timestamp)

            # Feed the velocity windows every frame so they stay populated across
            # the carry → strike transition.  Ball: IMM-smoothed position;
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

            # Accumulate per-segment strength while a segment is open.
            if seg_start is not None and trace_active:
                if status.racket_prob > RACKET_SPIKE_THRESH:
                    seg_racket_frames += 1

            if trace_active:
                if seg_start is None:
                    seg_start         = video_time_offset + telemetry.timestamp
                    seg_origin        = _origin_side(status.position, near_box, far_box)
                    seg_racket_frames = 0
            else:
                if seg_start is not None:
                    raw_end = video_time_offset + (
                        ball_tracker.last_detection_time
                        if ball_tracker.last_detection_time is not None
                        else telemetry.timestamp
                    )
                    padded_end = min(raw_end + RALLY_END_PAD_SEC, video_duration_sec)
                    strength = {
                        "racket_frames": seg_racket_frames,
                        "duration":      raw_end - seg_start,
                    }
                    raw_segments.append((seg_start, padded_end, seg_origin, strength))
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
            strength = {
                "racket_frames": seg_racket_frames,
                "duration":      raw_end - seg_start,
            }
            raw_segments.append((seg_start, padded_end, seg_origin, strength))

        cap.release()
        if not headless:
            cv2.destroyAllWindows()

    print(f"[RALLY] Raw segments: {len(raw_segments)}")
    if interrupted:
        print("[RALLY] (interrupted — covers completed detections only)")

    # ── Post-process ──────────────────────────────────────────────────────
    # 1. Merge close gaps (mid-rally re-acquisitions → one segment).
    merged = _merge_segments(raw_segments, RALLY_GAP_THRESHOLD_SEC)
    print(f"[RALLY] After merging (gap < {RALLY_GAP_THRESHOLD_SEC:.0f}s): {len(merged)} segment(s)")

    # 2. HMM serving-pattern filter: drop weak anomalies, relabel strong ones.
    filtered = _hmm_filter(merged)
    print(f"[RALLY] After HMM filter: {len(filtered)} segment(s)")

    # 3. Prepend pre-roll to each final segment.
    final = _apply_preroll(filtered, RALLY_PRE_ROLL_SEC)
    n_far  = sum(1 for *_, o in final if o == "far")
    n_near = len(final) - n_far
    print(f"[RALLY] Segment origins: {n_near} near, {n_far} far")
    return final


def detect_rallies(video_path, output_path=None, headless=False, start_frame=0, progress_cb=None):
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_rallies.mp4")

    final = collect_rally_segments(video_path, headless, start_frame, progress_cb)

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
