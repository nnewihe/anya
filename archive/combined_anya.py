"""combined_anya.py
================
Unified tennis serve-detection pipeline that automatically identifies the serving
side (near or far camera) and delegates to the appropriate algorithm.

Five-state machine:
  WAITING    → both near-side and far-side WAITING detectors run in parallel
                (cheap; each strided independently)
  ARMED_NEAR → near-side ARMED logic only; far-side frozen
  ARMED_FAR  → far-side ARMED logic only; near-side frozen
  ACTIVE_NEAR → near-side ACTIVE (ball trace + energy bar); far frozen
  ACTIVE_FAR  → far-side ACTIVE; near frozen
  → WAITING on point end; loop

Tennis camera-side geometry
---------------------------
Players switch ends after every odd-total game; service alternates every game.
Combined, the serve side from the camera repeats for ~2 consecutive games before
flipping:

  Game 1: NEAR  (Player A serves from near baseline)
  Game 2: NEAR  (Player B serves from near baseline after end switch)
  Game 3: FAR   (Player A serves from far baseline)
  Game 4: FAR   (Player B serves from far baseline)
  Game 5: NEAR  ...

Two games × ≥4 points/game = ≥8 serves per camera-side run.
ServeTracker uses a threshold of 8 before accepting a predicted side flip.
"""

import argparse
import csv
import os
from collections import deque
from typing import List, Optional, Tuple

import cv2

from src.ai.anya_base import AnyaTelemetryProvider, TelemetryFrame
from src.ai.anya_transitions import TransitionEngine
from src.ai.far_anya import FarTelemetryProvider, FarTelemetryFrame, FarTransitionEngine
from src.ai.utilities import create_highlights_ffmpeg


# ─────────────────────────────────────────────────────────────────────────────
# Serve-side tracker
# ─────────────────────────────────────────────────────────────────────────────

class ServeTracker:
    """
    Tracks which camera side has been serving and biases WAITING polling rates.

    A side-flip prediction is only accepted once >= MIN_FOR_SWITCH consecutive
    serves from the current predicted side have been observed, guarding against
    stray false ARMED triggers on the wrong side.

    Polling strides (returned by get_strides):
      Predicted side  → stride 2  (check every other WAITING frame)
      Other side      → stride 4  (check every 4th WAITING frame)
      No prediction   → both stride 3 (equal, symmetric)

    Both sides always run — accuracy is never sacrificed; only frequency varies.
    """
    MIN_FOR_SWITCH = 8

    def __init__(self):
        self._predicted:   Optional[str] = None  # "near" | "far" | None
        self._consecutive: int           = 0

    @property
    def predicted(self) -> Optional[str]:
        return self._predicted

    def record_serve(self, side: str) -> None:
        """Record a detected serve. Call once per ARMED → ACTIVE transition."""
        if self._predicted is None:
            self._predicted   = side
            self._consecutive = 1
            print(f"[TRACKER] First serve detected: {side} side.")
        elif side == self._predicted:
            self._consecutive += 1
            print(f"[TRACKER] {side}-side serve #{self._consecutive} (consecutive).")
        else:
            if self._consecutive >= self.MIN_FOR_SWITCH:
                print(f"[TRACKER] Side flip: {self._predicted} → {side} "
                      f"(after {self._consecutive} consecutive serves).")
                self._predicted   = side
                self._consecutive = 1
            else:
                print(f"[TRACKER] {side}-side serve while {self._predicted} predicted "
                      f"({self._consecutive} < {self.MIN_FOR_SWITCH}); ignoring flip.")

    def get_strides(self) -> Tuple[int, int]:
        """Return (near_stride, far_stride) for WAITING-state polling."""
        if self._predicted is None:
            return 3, 3
        return (2, 4) if self._predicted == "near" else (4, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Reset helpers for frozen-side reactivation
# ─────────────────────────────────────────────────────────────────────────────

def _unfreeze_near(near_provider: AnyaTelemetryProvider,
                   near_engine: TransitionEngine) -> None:
    """Prepare the near-side provider+engine to re-enter WAITING from frozen state."""
    near_provider.telemetry_history.clear()
    near_engine.near_ready_start_time = None


def _unfreeze_far(far_provider: FarTelemetryProvider,
                  far_engine: FarTransitionEngine) -> None:
    """Prepare the far-side provider+engine to re-enter WAITING from frozen state."""
    far_provider.telemetry_history.clear()
    far_engine._movement_history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

_CSV_COLS = [
    "point", "frame", "timestamp", "state", "serve_side",
    "time_since_trace", "has_active_trace",
    "energy_bar_mode", "point_energy", "energy_status",
    "ball_count",
]


def _write_csv_row(csv_writer, engine, telemetry, point_number: int,
                   frame_in_point: int, serve_side: str) -> None:
    debug = engine.last_active_debug
    csv_writer.writerow({
        "point":            point_number,
        "frame":            frame_in_point,
        "timestamp":        round(telemetry.timestamp, 4),
        "state":            telemetry.state,
        "serve_side":       serve_side,
        "time_since_trace": round(debug.get("time_since_trace", 0.0), 3),
        "has_active_trace": debug.get("has_active_trace", False),
        "energy_bar_mode":  debug.get("energy_bar_mode", False),
        "point_energy":     round(debug.get("point_energy", 1.0), 3),
        "energy_status":    debug.get("energy_status", ""),
        "ball_count":       debug.get("ball_count", 0),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Core segment-collection loop
# ─────────────────────────────────────────────────────────────────────────────

def _collect_combined_segments(video_path: str, headless: bool = False,
                                start_frame: int = 0, csv_path: Optional[str] = None):
    """
    Run the combined pipeline on a single video.

    Returns
    -------
    active_segments : list of (start_sec, end_sec, serve_side)
    point_number    : total points detected
    csv_path        : path to the written telemetry CSV
    """
    if csv_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        csv_path   = os.path.join(video_dir, f"{video_stem}_combined_telemetry.csv")

    # ── Probe video properties ────────────────────────────────────────────────
    _probe   = cv2.VideoCapture(video_path)
    orig_fps = _probe.get(cv2.CAP_PROP_FPS)
    _total   = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    _probe.release()
    if orig_fps <= 0 or orig_fps > 300:
        orig_fps = 30.0
    video_duration_sec = _total / orig_fps if _total > 0 else float("inf")
    video_time_offset  = start_frame / orig_fps
    HIGHLIGHT_END_PAD_SEC = 1.0

    # ── Initialise providers (court setup, YOLO loading, exclusion zones) ─────
    mid_frame = max(0, _total // 2)

    print("[COMBINED] Initialising near-side provider …")
    near_provider = AnyaTelemetryProvider(video_path, court_frame_idx=mid_frame)
    near_engine   = TransitionEngine(fps=near_provider.fps)

    print("[COMBINED] Initialising far-side provider …")
    far_provider  = FarTelemetryProvider(video_path, court_frame_idx=mid_frame,
                                         active_zone_polygon=near_provider.active_zone_polygon)
    far_engine    = FarTransitionEngine(fps=far_provider.fps,
                                        far_baseline_strip=far_provider.far_baseline_strip)

    tracker = ServeTracker()

    # ── CSV writer ────────────────────────────────────────────────────────────
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLS)
    csv_writer.writeheader()

    # ── Tracking state ────────────────────────────────────────────────────────
    # Four combined states: WAITING | ARMED | ACTIVE_NEAR | ACTIVE_FAR
    # In ARMED both sides run simultaneously; first to fire ACTIVE wins.
    combined_state:        str                           = "WAITING"
    near_armed:            bool  = False   # near side currently in ARMED mode
    far_armed:             bool  = False   # far side currently in ARMED mode
    active_segments:       List[Tuple[float, float, str]] = []
    point_number:          int   = 0
    frame_in_point:        int   = 0
    current_segment_start: float = 0.0
    current_serve_side:    str   = "near"   # updated on every ACTIVE entry
    last_telemetry_ts:     float = 0.0

    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[COMBINED] Seeking to frame {start_frame}")

    interrupted = False

    try:
        while cap.isOpened():
            success, orig_frame = cap.read()
            if not success:
                break

            frame = cv2.resize(orig_frame, (960, 540), interpolation=cv2.INTER_LINEAR)

            # ── WAITING ───────────────────────────────────────────────────────
            # Both providers run in parallel; each with its own stride.
            # Near wins ties (both arm on the same frame).
            if combined_state == "WAITING":
                near_stride, far_stride = tracker.get_strides()

                # Near provider — skip or infer
                near_skip = (
                    near_provider.frame_counter % near_stride != 0
                    and bool(near_provider.telemetry_history)
                )
                if near_skip:
                    near_provider.frame_counter += 1
                    _last_near = near_provider.telemetry_history[-1]
                    near_tel   = TelemetryFrame(
                        frame_id=near_provider.frame_counter,
                        timestamp=near_provider.frame_counter / near_provider.fps,
                        state="WAITING",
                        near_player_box=_last_near.near_player_box,
                        near_player_world=_last_near.near_player_world,
                        toss_ball_candidates=[],
                        active_ball_candidates=[],
                    )
                    near_provider.telemetry_history.append(near_tel)
                else:
                    near_tel = near_provider.process_frame(frame)

                # Far provider — skip or infer
                far_skip = (
                    far_provider.frame_counter % far_stride != 0
                    and bool(far_provider.telemetry_history)
                )
                if far_skip:
                    far_provider.frame_counter += 1
                    _last_far = far_provider.telemetry_history[-1]
                    far_tel   = FarTelemetryFrame(
                        frame_id=far_provider.frame_counter,
                        timestamp=far_provider.frame_counter / far_provider.fps,
                        state="WAITING",
                        far_player_box=_last_far.far_player_box,
                        far_player_world=_last_far.far_player_world,
                        near_player_box=_last_far.near_player_box,
                        toss_ball_candidates=[],
                        active_ball_candidates=[],
                    )
                    far_provider.telemetry_history.append(far_tel)
                else:
                    far_tel = far_provider.process_frame(frame)

                last_telemetry_ts = near_tel.timestamp  # both counters in sync

                # Check both conditions; near wins ties
                near_result = near_engine.evaluate_transitions(
                    near_provider.telemetry_history, "WAITING")
                far_result  = far_engine.evaluate_transitions(
                    far_provider.telemetry_history, "WAITING")

                if near_result == "ARMED" or far_result == "ARMED":
                    # Both sides enter ARMED together regardless of which triggered.
                    # The non-triggering side will quickly self-drop if not at baseline.
                    combined_state = "ARMED"
                    near_armed = True
                    far_armed  = True
                    near_provider.update_state("ARMED")
                    far_provider.update_state("ARMED")
                    triggered = "near" if near_result == "ARMED" else "far"
                    print(f"[COMBINED] WAITING → ARMED  (triggered by {triggered} side)")

            # ── ARMED: both sides run simultaneously; first to ACTIVE wins ────
            elif combined_state == "ARMED":
                near_tel = None
                far_tel  = None

                if near_armed:
                    near_tel = near_provider.process_frame(frame)
                    last_telemetry_ts = near_tel.timestamp
                else:
                    near_provider.frame_counter += 1

                if far_armed:
                    far_tel = far_provider.process_frame(frame)
                    if near_tel is None:
                        last_telemetry_ts = far_tel.timestamp
                else:
                    far_provider.frame_counter += 1

                near_result = (near_engine.evaluate_transitions(
                                   near_provider.telemetry_history, "ARMED")
                               if near_armed else "WAITING")
                far_result  = (far_engine.evaluate_transitions(
                                   far_provider.telemetry_history, "ARMED")
                               if far_armed  else "WAITING")

                # Near wins ties
                if near_result == "ACTIVE":
                    point_number         += 1
                    frame_in_point        = 0
                    current_segment_start = video_time_offset + near_tel.timestamp
                    current_serve_side    = "near"
                    tracker.record_serve("near")
                    combined_state = "ACTIVE_NEAR"
                    near_provider.update_state("ACTIVE")
                    near_armed = False
                    if far_armed:
                        far_armed = False
                        far_provider.update_state("WAITING")
                        _unfreeze_far(far_provider, far_engine)
                    print(f"[COMBINED] ARMED → ACTIVE_NEAR  (point {point_number})")

                elif far_result == "ACTIVE":
                    point_number         += 1
                    frame_in_point        = 0
                    current_segment_start = video_time_offset + far_tel.timestamp
                    current_serve_side    = "far"
                    tracker.record_serve("far")
                    combined_state = "ACTIVE_FAR"
                    far_provider.update_state("ACTIVE")
                    far_armed = False
                    if near_armed:
                        near_armed = False
                        near_provider.update_state("WAITING")
                        _unfreeze_near(near_provider, near_engine)
                    print(f"[COMBINED] ARMED → ACTIVE_FAR  (point {point_number})")

                else:
                    # Update armed flags for any side that fell out
                    if near_armed and near_result == "WAITING":
                        near_armed = False
                        near_provider.update_state("WAITING")
                        _unfreeze_near(near_provider, near_engine)
                        print("[COMBINED] Near side fell out of ARMED")
                    if far_armed and far_result == "WAITING":
                        far_armed = False
                        far_provider.update_state("WAITING")
                        _unfreeze_far(far_provider, far_engine)
                        print("[COMBINED] Far side fell out of ARMED")
                    if not near_armed and not far_armed:
                        combined_state = "WAITING"
                        print("[COMBINED] ARMED → WAITING  (both fell out)")

            # ── ACTIVE_NEAR ───────────────────────────────────────────────────
            elif combined_state == "ACTIVE_NEAR":
                near_tel = near_provider.process_frame(frame)
                last_telemetry_ts = near_tel.timestamp
                far_provider.frame_counter += 1
                frame_in_point += 1

                near_result = near_engine.evaluate_transitions(
                    near_provider.telemetry_history, "ACTIVE")
                _write_csv_row(csv_writer, near_engine, near_tel,
                               point_number, frame_in_point, "near")

                if near_result == "WAITING":
                    end_t = (near_engine.last_transition_time
                             if near_engine.last_transition_time is not None
                             else near_tel.timestamp)
                    padded_end = min(video_time_offset + end_t + HIGHLIGHT_END_PAD_SEC,
                                     video_duration_sec)
                    active_segments.append((current_segment_start, padded_end, "near"))
                    combined_state = "WAITING"
                    near_provider.update_state("WAITING")
                    _unfreeze_far(far_provider, far_engine)
                    print("[COMBINED] ACTIVE_NEAR → WAITING")

            # ── ACTIVE_FAR ────────────────────────────────────────────────────
            elif combined_state == "ACTIVE_FAR":
                far_tel = far_provider.process_frame(frame)
                last_telemetry_ts = far_tel.timestamp
                near_provider.frame_counter += 1
                frame_in_point += 1

                far_result = far_engine.evaluate_transitions(
                    far_provider.telemetry_history, "ACTIVE")
                _write_csv_row(csv_writer, far_engine, far_tel,
                               point_number, frame_in_point, "far")

                if far_result == "WAITING":
                    end_t = (far_engine.last_transition_time
                             if far_engine.last_transition_time is not None
                             else far_tel.timestamp)
                    padded_end = min(video_time_offset + end_t + HIGHLIGHT_END_PAD_SEC,
                                     video_duration_sec)
                    active_segments.append((current_segment_start, padded_end, "far"))
                    combined_state = "WAITING"
                    far_provider.update_state("WAITING")
                    _unfreeze_near(near_provider, near_engine)
                    print("[COMBINED] ACTIVE_FAR → WAITING")

            # ── Display ───────────────────────────────────────────────────────
            if not headless:
                _render_frame(frame, combined_state, near_provider, far_provider,
                              near_engine, far_engine, tracker)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        interrupted = True
        print("\n[COMBINED] Ctrl-C — creating highlights from completed segments …")

    finally:
        # Flush any in-progress segment
        if combined_state in ("ACTIVE_NEAR", "ACTIVE_FAR"):
            padded_end = min(video_time_offset + last_telemetry_ts + HIGHLIGHT_END_PAD_SEC,
                             video_duration_sec)
            active_segments.append((current_segment_start, padded_end, current_serve_side))

        cap.release()
        csv_file.close()
        if not headless:
            cv2.destroyAllWindows()

    near_count = sum(1 for _, _, s in active_segments if s == "near")
    far_count  = sum(1 for _, _, s in active_segments if s == "far")
    print(f"[COMBINED] {os.path.basename(video_path)}: {point_number} points "
          f"({near_count} near, {far_count} far), {len(active_segments)} segments")
    if interrupted:
        print("[COMBINED] (interrupted — results cover completed detections only)")

    return active_segments, point_number, csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_combined_anya_pipeline(video_path: str, output_path: Optional[str] = None,
                                headless: bool = False, start_frame: int = 0) -> None:
    """
    Auto-detect serving side and produce a combined highlight reel.

    Mirrors the signature of run_anya_pipeline and run_far_anya_pipeline.
    """
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_combined_highlights.mp4")

    csv_path = os.path.splitext(output_path)[0] + "_telemetry.csv"

    active_segments, point_number, _ = _collect_combined_segments(
        video_path, headless, start_frame, csv_path=csv_path
    )

    # FFmpeg needs plain (start, end) tuples
    ffmpeg_segments = [(s, e) for s, e, _ in active_segments]
    create_highlights_ffmpeg(video_path, ffmpeg_segments, output_path)

    near_count = sum(1 for _, _, s in active_segments if s == "near")
    far_count  = sum(1 for _, _, s in active_segments if s == "far")

    print(f"\n[DONE] Output video     : {output_path}")
    print(f"[DONE] Telemetry CSV    : {csv_path}")
    print(f"[DONE] Points recorded  : {point_number}")
    print(f"[DONE] Near-side serves : {near_count}")
    print(f"[DONE] Far-side serves  : {far_count}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

_STATE_COLORS = {
    "WAITING":     (0, 255, 255),   # yellow
    "ARMED":       (0, 165, 255),   # orange
    "ACTIVE_NEAR": (0, 255, 0),     # green
    "ACTIVE_FAR":  (0, 220, 0),     # green
}


def _render_frame(frame, combined_state: str,
                  near_provider: AnyaTelemetryProvider,
                  far_provider:  FarTelemetryProvider,
                  near_engine:   TransitionEngine,
                  far_engine:    FarTransitionEngine,
                  tracker:       ServeTracker) -> None:
    color = _STATE_COLORS.get(combined_state, (200, 200, 200))
    cv2.putText(frame, f"STATE: {combined_state.replace('_', ' ')}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    pred_label = tracker.predicted or "?"
    cv2.putText(frame, f"Predicted: {pred_label}",
                (30, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    if combined_state == "WAITING":
        # Show both player boxes
        if near_provider.telemetry_history:
            nb = near_provider.telemetry_history[-1].near_player_box
            if nb:
                x1, y1, x2, y2 = nb
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, "NEAR", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1, cv2.LINE_AA)
        if far_provider.telemetry_history:
            fb = far_provider.telemetry_history[-1].far_player_box
            if fb:
                x1, y1, x2, y2 = fb
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 105, 255), 2)
                cv2.putText(frame, "FAR", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 105, 255), 1, cv2.LINE_AA)

    elif combined_state == "ARMED":
        # Both sides are active; draw both player boxes
        if near_provider.telemetry_history:
            nb = near_provider.telemetry_history[-1].near_player_box
            if nb:
                x1, y1, x2, y2 = nb
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, "NEAR", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1, cv2.LINE_AA)
        if far_provider.telemetry_history:
            fb = far_provider.telemetry_history[-1].far_player_box
            if fb:
                x1, y1, x2, y2 = fb
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 105, 255), 2)
                cv2.putText(frame, "FAR", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 105, 255), 1, cv2.LINE_AA)

    elif combined_state == "ACTIVE_NEAR":
        if near_provider.telemetry_history:
            nb = near_provider.telemetry_history[-1].near_player_box
            if nb:
                x1, y1, x2, y2 = nb
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        _draw_trace(frame, [(px, py) for _, px, py in near_engine._trace_ball_history])

    elif combined_state == "ACTIVE_FAR":
        if far_provider.telemetry_history:
            fb = far_provider.telemetry_history[-1].far_player_box
            if fb:
                x1, y1, x2, y2 = fb
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 105, 255), 2)
        _draw_trace(frame, [(px, py) for _, px, py in far_engine._trace_ball_history])

    cv2.imshow("Combined Anya Pipeline", frame)


def _draw_trace(frame, trace: list) -> None:
    n = len(trace)
    if n >= 2:
        for i in range(1, n):
            age   = i / (n - 1)
            color = (0, int(120 * age), int(255 * age))
            cv2.line(frame,
                     (int(trace[i - 1][0]), int(trace[i - 1][1])),
                     (int(trace[i][0]),     int(trace[i][1])),
                     color, max(1, int(3 * age)), cv2.LINE_AA)
    if n >= 1:
        cv2.circle(frame, (int(trace[-1][0]), int(trace[-1][1])),
                   5, (0, 200, 255), -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined Anya Tennis Serve-Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ai.combined_anya match.mp4
  python -m src.ai.combined_anya match.mp4 --output highlights.mp4 --headless
  python -m src.ai.combined_anya match.mp4 --start-frame 900 --headless
""",
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", default=None,
                        help="Output MP4 path (default: <video>_combined_highlights.mp4)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without OpenCV display windows")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Start processing from this frame number (default: 0)")
    args = parser.parse_args()

    run_combined_anya_pipeline(args.video, args.output, args.headless, args.start_frame)
