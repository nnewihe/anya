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
import os

import cv2

from anya_base import AnyaTelemetryProvider
from ball_tracker import BallTrackManager, make_image_row_perspective
from utilities import create_highlights_ffmpeg


RALLY_GAP_THRESHOLD_SEC = 4.0
RALLY_PRE_ROLL_SEC      = 1.5
RALLY_END_PAD_SEC       = 1.0


def _merge_segments(segments, gap_threshold_sec=RALLY_GAP_THRESHOLD_SEC):
    """Merge adjacent segments whose gap < gap_threshold_sec."""
    if not segments:
        return []
    merged = [list(segments[0])]
    for start, end in segments[1:]:
        if start - merged[-1][1] < gap_threshold_sec:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(s) for s in merged]


def _apply_preroll(segments, pre_roll_sec=RALLY_PRE_ROLL_SEC):
    return [(max(0.0, start - pre_roll_sec), end) for start, end in segments]


def _in_player_box(cx, cy, box, padding=10):
    """Return True if (cx, cy) falls inside box expanded by padding pixels."""
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return (x1 - padding) <= cx <= (x2 + padding) and (y1 - padding) <= cy <= (y2 + padding)


def detect_rallies(video_path, output_path=None, headless=False, start_frame=0):
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_rallies.mp4")

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

    raw_segments = []
    seg_start: float = None
    last_ts:   float = 0.0
    interrupted = False

    try:
        while cap.isOpened():
            success, orig_frame = cap.read()
            if not success:
                break

            frame    = cv2.resize(orig_frame, (960, 540), interpolation=cv2.INTER_LINEAR)
            telemetry = telemetry_provider.process_frame(frame)
            last_ts   = telemetry.timestamp

            near_box = telemetry.near_player_box
            far_box  = telemetry.far_player_box
            dets = [
                (c["pixel_center"][0], c["pixel_center"][1], c["conf"])
                for c in (telemetry.active_ball_candidates or [])
                if not _in_player_box(c["pixel_center"][0], c["pixel_center"][1], near_box)
                and not _in_player_box(c["pixel_center"][0], c["pixel_center"][1], far_box)
            ]
            status = ball_tracker.update(dets, telemetry.timestamp)

            if status.has_moving_trace:
                if seg_start is None:
                    seg_start = video_time_offset + telemetry.timestamp
            else:
                if seg_start is not None:
                    raw_end = video_time_offset + (
                        ball_tracker.last_detection_time
                        if ball_tracker.last_detection_time is not None
                        else telemetry.timestamp
                    )
                    padded_end = min(raw_end + RALLY_END_PAD_SEC, video_duration_sec)
                    raw_segments.append((seg_start, padded_end))
                    seg_start = None

            if not headless:
                _render_overlay(frame, status)
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
            raw_segments.append((seg_start, padded_end))

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

    create_highlights_ffmpeg(video_path, final, output_path)

    print(f"\n[DONE] Output   : {output_path}")
    print(f"[DONE] Segments : {len(final)}")


def _render_overlay(frame, status):
    """Minimal debug overlay — trace liveness and ball trail."""
    alive = status.has_moving_trace
    color = (0, 255, 0) if alive else (0, 255, 255)
    label = f"TRACE: {'ALIVE' if alive else 'DEAD'}  {status.speed_px_s:.0f}px/s"
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
