"""
far_anya.py
===========
Main loop, rendering, and CLI for the far-side serve detector.

Mirrors run_anya.py in structure:
  • _collect_far_segments  — core video loop; returns (segments, count, csv_path, timestamps)
  • run_far_anya_pipeline  — single-video public entry point
  • render_frame           — per-frame debug overlay
  • render_debug_panel     — separate debug window
  • CLI (__main__)

Usage
-----
  python far_anya.py video.mp4
  python far_anya.py video.mp4 --output far_highlights.mp4 --headless
  python far_anya.py video.mp4 --headless --start-frame 1800
"""

import argparse
import csv
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from far_anya_base import FarTelemetryProvider, FarTelemetryFrame
from far_anya_transitions import FarTransitionEngine
from utilities import create_highlights_ffmpeg


# ─────────────────────────────────────────────────────────────────────────────
# Core segment-collection loop
# ─────────────────────────────────────────────────────────────────────────────

def _in_intervals(t: float, intervals) -> bool:
    """True if video-time t falls inside any (start, end) interval."""
    if not intervals:
        return False
    for s, e in intervals:
        if s <= t <= e:
            return True
    return False


def _collect_far_segments(
    video_path: str,
    headless: bool = False,
    start_frame: int = 0,
    csv_path: Optional[str] = None,
    skip_intervals: Optional[List[Tuple[float, float]]] = None,
):
    """
    Run the far-side pipeline on a single video.

    Parameters
    ----------
    skip_intervals : optional list of (start_sec, end_sec) in source-video time
        that the far detector should NOT process (e.g. spans already claimed as
        near-end serving runs).  Inside these spans the frame is grabbed but not
        decoded or inferred — the far state machine is held in WAITING — which
        skips the expensive YOLO inference (and frame decode) for the bulk of
        the match's near-serving games.

    Returns
    -------
    active_segments : list of (start_sec, end_sec) in source-video time
    point_number    : total serves detected
    csv_path        : path to the written telemetry CSV
    timestamps      : list of serve-start timestamps (seconds)
    """
    if csv_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        csv_path   = os.path.join(video_dir, f"{video_stem}_far_telemetry.csv")

    # ── Probe video ───────────────────────────────────────────────────────
    _probe   = cv2.VideoCapture(video_path)
    orig_fps = _probe.get(cv2.CAP_PROP_FPS)
    _total   = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
    _probe.release()
    if orig_fps <= 0 or orig_fps > 300:
        orig_fps = 30.0
    video_duration_sec = _total / orig_fps if _total > 0 else float("inf")

    # ── Init pipeline ─────────────────────────────────────────────────────
    provider = FarTelemetryProvider(video_path)
    engine   = FarTransitionEngine(
        fps=provider.fps,
        net_y_px=provider.net_y_px,
        baseline_front_px=provider.baseline_front_px,
        baseline_behind_px=provider.baseline_behind_px,
    )

    # ── CSV writer ────────────────────────────────────────────────────────
    _CSV_COLS = [
        "serve", "frame", "timestamp", "state",
        "track_state", "has_active_trace", "time_since_detection",
        "ball_speed_px_s", "coasting", "ball_count",
        "maneuver_prob", "racket_prob", "bounce_prob",
        "energy_bar_mode", "point_energy", "energy_status",
    ]
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLS)
    csv_writer.writeheader()

    # ── Segment tracking ──────────────────────────────────────────────────
    video_time_offset:      float             = start_frame / orig_fps
    active_segments:        List[Tuple[float, float]] = []
    timestamps:             List[float]               = []
    current_segment_start:  float             = 0.0
    last_telemetry_ts:      float             = 0.0
    HIGHLIGHT_END_PAD_SEC                     = 1.0

    cap            = cv2.VideoCapture(video_path)
    point_number   = 0
    frame_in_point = 0

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[FAR] Seeking to frame {start_frame}")

    WAITING_STRIDE = 10
    ARMED_STRIDE   = 5
    interrupted    = False
    skipped_frames = 0

    try:
        while cap.isOpened():
            # Grab (demux) without decoding so masked spans cost almost nothing
            if not cap.grab():
                break

            # Video-time of the frame about to be processed
            this_video_t = video_time_offset + (provider.frame_counter + 1) / provider.fps

            # ── Skip-mask: frame falls inside a near-serving run ──────────
            if _in_intervals(this_video_t, skip_intervals):
                # Defensively close any open far ACTIVE segment
                if provider.current_state == "ACTIVE":
                    padded_end = min(
                        video_time_offset + last_telemetry_ts + HIGHLIGHT_END_PAD_SEC,
                        video_duration_sec,
                    )
                    active_segments.append((current_segment_start, padded_end))
                    engine._reset_active_state()
                if provider.current_state != "WAITING":
                    provider.update_state("WAITING")

                provider.frame_counter += 1
                stub = FarTelemetryFrame(
                    frame_id=provider.frame_counter,
                    timestamp=provider.frame_counter / provider.fps,
                    state="WAITING",
                    toss_ball_candidates=[],
                    active_ball_candidates=[],
                )
                provider.telemetry_history.append(stub)
                last_telemetry_ts = stub.timestamp
                skipped_frames += 1
                continue

            success, orig_frame = cap.retrieve()
            if not success:
                break

            frame = cv2.resize(orig_frame, (960, 540), interpolation=cv2.INTER_LINEAR)

            # Stride YOLO inference: 1-in-10 for WAITING, 1-in-5 for ARMED
            skip_inference = bool(provider.telemetry_history) and (
                (provider.current_state == "WAITING"
                 and provider.frame_counter % WAITING_STRIDE != 0)
                or (provider.current_state == "ARMED"
                    and provider.frame_counter % ARMED_STRIDE != 0)
            )

            if skip_inference:
                provider.frame_counter += 1
                last = provider.telemetry_history[-1]
                tel  = FarTelemetryFrame(
                    frame_id=provider.frame_counter,
                    timestamp=provider.frame_counter / provider.fps,
                    state=provider.current_state,
                    far_player_box=last.far_player_box,
                    far_player_world=last.far_player_world,
                    far_player_foot_y_px=last.far_player_foot_y_px,
                    near_player_box=last.near_player_box,
                    near_player_world=last.near_player_world,
                    z_box=last.z_box,
                    toss_ball_candidates=[],
                    active_ball_candidates=[],
                )
                provider.telemetry_history.append(tel)
            else:
                tel = provider.process_frame(frame)

            last_telemetry_ts = tel.timestamp

            new_state = engine.evaluate_transitions(
                provider.telemetry_history,
                provider.current_state,
            )

            old_state = provider.current_state
            if new_state != old_state:
                if new_state == "ACTIVE":
                    point_number    += 1
                    frame_in_point   = 0
                    serve_ts = video_time_offset + tel.timestamp
                    current_segment_start = serve_ts
                    timestamps.append(serve_ts)
                    print(f"[FAR] Serve #{point_number} detected at {serve_ts:.2f}s")
                elif old_state == "ACTIVE":
                    end_t = (engine.last_transition_time
                             if engine.last_transition_time is not None
                             else tel.timestamp)
                    padded_end = min(
                        video_time_offset + end_t + HIGHLIGHT_END_PAD_SEC,
                        video_duration_sec,
                    )
                    active_segments.append((current_segment_start, padded_end))
                provider.update_state(new_state)

            # Keep provider informed so it can skip player detection when trace is alive
            provider.energy_bar_active = engine.energy_bar_mode

            if provider.current_state == "ACTIVE":
                frame_in_point += 1
                _write_csv_row(csv_writer, engine, tel, point_number, frame_in_point)

            if not headless:
                render_frame(
                    frame, tel, provider.current_state, engine,
                    provider.exclusion_zones,
                    provider.active_zone_polygon,
                    provider.net_y_px,
                    engine.NET_BUFFER_PX,
                )
                debug_panel = render_debug_panel(provider.current_state, engine)
                cv2.imshow("Far Anya Pipeline", frame)
                cv2.imshow("Far Debug Panel",   debug_panel)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        interrupted = True
        print("\n[FAR] Ctrl-C — creating highlights from completed segments...")

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

    print(f"[FAR] {os.path.basename(video_path)}: "
          f"{point_number} far-side serves, {len(active_segments)} segments")
    if skip_intervals:
        print(f"[FAR] Skipped {skipped_frames} frames inside "
              f"{len(skip_intervals)} near-serving run span(s) (inference bypassed).")
    if interrupted:
        print("[FAR] (interrupted — results cover completed detections only)")

    return active_segments, point_number, csv_path, timestamps


# ─────────────────────────────────────────────────────────────────────────────
# CSV helper
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv_row(
    csv_writer,
    engine: FarTransitionEngine,
    tel: FarTelemetryFrame,
    point_number: int,
    frame_in_point: int,
):
    d = engine.last_active_debug
    csv_writer.writerow({
        "serve":                point_number,
        "frame":                frame_in_point,
        "timestamp":            round(tel.timestamp, 4),
        "state":                tel.state,
        "track_state":          d.get("state",                "none"),
        "has_active_trace":     d.get("has_active_trace",     False),
        "time_since_detection": round(d.get("time_since_detection", 0.0), 3),
        "ball_speed_px_s":      round(d.get("ball_speed_px_s",      0.0), 1),
        "coasting":             d.get("coasting",             False),
        "ball_count":           d.get("ball_count",           0),
        "maneuver_prob":        round(d.get("maneuver_prob",  0.0), 3),
        "racket_prob":          round(d.get("racket_prob",    0.0), 3),
        "bounce_prob":          round(d.get("bounce_prob",    0.0), 3),
        "energy_bar_mode":      d.get("energy_bar_mode",     False),
        "point_energy":         round(d.get("point_energy",  1.0), 3),
        "energy_status":        d.get("energy_status",       ""),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_far_anya_pipeline(
    video_path: str,
    output_path: Optional[str] = None,
    headless: bool = False,
    start_frame: int = 0,
):
    """Run the far-side serve detector on a single video."""
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_far_highlights.mp4")

    csv_path = os.path.splitext(output_path)[0] + "_telemetry.csv"

    segments, point_number, _, timestamps = _collect_far_segments(
        video_path, headless, start_frame, csv_path=csv_path
    )

    print("\n" + "=" * 50)
    print(f"  FAR-SIDE SERVES DETECTED: {point_number}")
    print("=" * 50)
    for i, ts in enumerate(timestamps, 1):
        mins = int(ts // 60)
        secs = ts % 60
        print(f"  Serve #{i:>3}: {mins}:{secs:05.2f}  ({ts:.2f}s)")
    print("=" * 50)

    if segments:
        create_highlights_ffmpeg(video_path, segments, output_path)
        print(f"\n[FAR] Output video  : {output_path}")
    else:
        print("\n[FAR] No segments to export.")

    print(f"[FAR] Telemetry CSV : {csv_path}")
    return timestamps


# ─────────────────────────────────────────────────────────────────────────────
# Debug rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(
    frame,
    tel: FarTelemetryFrame,
    state: str,
    engine: Optional[FarTransitionEngine] = None,
    exclusion_zones: Optional[list] = None,
    active_zone_polygon: Optional[np.ndarray] = None,
    net_y_px: Optional[float] = None,
    net_buffer_px: float = 40.0,
):
    """Per-frame debug overlay: state badge, player boxes, WAITING zone, balls, trace."""

    # WAITING zone indicator:
    #   cyan line  = net pixel line
    #   green line = effective zone ceiling (net_y − NET_BUFFER_PX)
    #                box centre must be above this to qualify for ARMED
    if net_y_px is not None:
        ny = int(net_y_px)
        cv2.line(frame, (0, ny), (frame.shape[1], ny), (0, 220, 220), 1)
        ceil_y = int(net_y_px - net_buffer_px)
        cv2.line(frame, (0, ceil_y), (frame.shape[1], ceil_y), (0, 200, 0), 1)

    # Translucent active zone in ACTIVE state
    if state == "ACTIVE" and active_zone_polygon is not None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [active_zone_polygon], (144, 238, 144))
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        cv2.polylines(frame, [active_zone_polygon], True, (0, 200, 0), 1)

    hud_color = (0, 255, 0) if state == "ACTIVE" else (0, 255, 255)
    cv2.putText(frame, f"FAR STATE: {state}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, hud_color, 2)

    # Far player (server) — orange box
    if tel.far_player_box:
        x1, y1, x2, y2 = tel.far_player_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(frame, "FAR-SRV", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)

    # Near player (receiver) — blue box
    if tel.near_player_box:
        x1, y1, x2, y2 = tel.near_player_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(frame, "NEAR-RCV", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 0, 0), 1, cv2.LINE_AA)

    # Exclusion zones — red boxes
    if exclusion_zones:
        for ex1, ey1, ex2, ey2 in exclusion_zones:
            cv2.rectangle(frame, (int(ex1), int(ey1)), (int(ex2), int(ey2)), (0, 0, 255), 2)

    # ARMED: toss zone and toss ball candidates
    if state == "ARMED":
        if tel.z_box:
            zx1, zy1, zx2, zy2 = tel.z_box
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
        if tel.toss_ball_candidates:
            for ball in tel.toss_ball_candidates:
                bx1, by1, bx2, by2 = ball["box"]
                cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)

    # ACTIVE: ball trace + live detections
    if state == "ACTIVE" and engine is not None:
        trace = [(px, py) for _, px, py in engine._trace_ball_history]
        n = len(trace)
        if n >= 2:
            for i in range(1, n):
                age       = i / (n - 1)
                color     = (0, int(120 * age), int(255 * age))
                thickness = max(1, int(3 * age))
                cv2.line(frame,
                         (int(trace[i - 1][0]), int(trace[i - 1][1])),
                         (int(trace[i][0]),     int(trace[i][1])),
                         color, thickness, cv2.LINE_AA)
        if n >= 1:
            cv2.circle(frame,
                       (int(trace[-1][0]), int(trace[-1][1])),
                       5, (0, 200, 255), -1, cv2.LINE_AA)

        if tel.active_ball_candidates:
            for ball in tel.active_ball_candidates:
                bx1, by1, bx2, by2 = ball["box"]
                cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)),
                              (0, 255, 255), 2)


def render_debug_panel(state: str, engine: FarTransitionEngine) -> np.ndarray:
    panel = np.ones((300, 500, 3), dtype=np.uint8) * 240

    if state == "ACTIVE":
        _render_active_panel(panel, engine)
    elif state == "ARMED":
        _render_armed_panel(panel, engine)
    else:
        cv2.putText(panel, "WAITING FOR FAR PLAYER", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 1)
    return panel


def _render_active_panel(panel: np.ndarray, engine: FarTransitionEngine):
    x0, y, lh, fs = 15, 35, 30, 0.5
    cv2.putText(panel, "FAR ACTIVE — BALL TRACE / ENERGY BAR", (x0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)

    d = engine.last_active_debug

    # Ball-trace row
    has_trace   = d.get("has_active_trace", False)
    tsd         = d.get("time_since_detection", 0.0)
    track_state = d.get("state", "none")
    trace_color = (0, 180, 0) if has_trace else (0, 0, 200)
    trace_label = f"YES  [{track_state}]" if has_trace else f"NO  ({tsd:.1f}s ago) [{track_state}]"
    cv2.putText(panel, f"Trace: {trace_label}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, trace_color, 1)
    y += lh

    # Ball speed
    speed = d.get("ball_speed_px_s", 0.0)
    cv2.putText(panel, f"Ball Speed: {speed:6.0f} px/s", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 80, 80), 1)
    y += lh

    # IMM contact probabilities
    mp = d.get("maneuver_prob", 0.0)
    rp = d.get("racket_prob",   0.0)
    bp = d.get("bounce_prob",   0.0)
    cv2.putText(panel, f"Maneuver: {mp:.2f}  Racket: {rp:.2f}  Bounce: {bp:.2f}",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, 140, 220) if mp > 0.4 else (80, 80, 80), 1)
    y += lh

    # Energy bar (shown only when active)
    energy_mode = d.get("energy_bar_mode", False)
    energy      = d.get("point_energy", 1.0)
    status      = d.get("energy_status", "--")
    bar_label   = "ENERGY BAR (near-player)" if energy_mode else "Energy (dormant)"
    bar_color   = (0, 180, 0) if not energy_mode else (
        (0, 0, 220) if energy < 0.3 else (0, 165, 255) if energy < 0.6 else (0, 200, 0)
    )
    cv2.putText(panel, f"{bar_label}: {energy:.2f}  [{status}]", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, bar_color, 2 if energy_mode else 1)
    y += 6
    bar_w  = 200
    bg_col = (180, 180, 180) if not energy_mode else (100, 100, 100)
    cv2.rectangle(panel, (x0, y), (x0 + bar_w, y + 14), bg_col, -1)
    if energy_mode and energy > 0:
        fill_col = (0, 0, 220) if energy < 0.3 else (0, 165, 255) if energy < 0.6 else (0, 200, 0)
        cv2.rectangle(panel, (x0, y), (x0 + int(energy * bar_w), y + 14), fill_col, -1)
    y += 22

    cv2.putText(panel, f"Balls detected: {d.get('ball_count', 0)}", (x0, y),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 80, 80), 1)


def _render_armed_panel(panel: np.ndarray, engine: FarTransitionEngine):
    x0, bar_w, bar_h, lh, label_w = 12, 200, 14, 30, 120
    cv2.putText(panel, "FAR ARMED — Toss Detection", (x0, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    scores = engine.last_serve_scores
    rows = [
        ("Toss",        scores.get("toss_score",  0.0), (0, 200, 200)),
        ("MHI",         scores.get("mhi_score",   0.0), (180, 180, 0)),
        ("Serve Score", scores.get("serve_score", 0.0), None),
    ]
    y = 65
    for label, value, color in rows:
        if color is None:
            color = (0, 220, 0) if value >= 0.55 else (0, 140, 255)
        cv2.putText(panel, f"{label}:", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        bx = x0 + label_w
        cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + bar_w, y + 2), (190, 190, 190), -1)
        cv2.rectangle(panel, (bx, y - bar_h + 2),
                      (bx + int(value * bar_w), y + 2), color, -1)
        cv2.putText(panel, f"{value:.3f}", (bx + bar_w + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if label == "Serve Score":
            thresh_x = bx + int(0.55 * bar_w)
            cv2.line(panel, (thresh_x, y - bar_h + 2), (thresh_x, y + 2), (0, 0, 0), 2)
        y += lh


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Far-side serve detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python far_anya.py video.mp4
  python far_anya.py video.mp4 --output far_out.mp4 --headless
  python far_anya.py video.mp4 --headless --start-frame 1800
""",
    )
    parser.add_argument("video", help="Input video file.")
    parser.add_argument("--output",      default=None,
                        help="Output highlights MP4 (default: <video>_far_highlights.mp4).")
    parser.add_argument("--headless",    action="store_true",
                        help="Run without display windows.")
    parser.add_argument("--start-frame", type=int, default=0, metavar="N",
                        help="Start processing from this frame number (default: 0).")
    args = parser.parse_args()

    run_far_anya_pipeline(
        args.video,
        output_path=args.output,
        headless=args.headless,
        start_frame=args.start_frame,
    )
