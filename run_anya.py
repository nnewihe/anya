import argparse
import csv
import os

import cv2
import numpy as np

from anya_base import AnyaTelemetryProvider, TelemetryFrame
from anya_transitions import TransitionEngine
from utilities import create_highlights_ffmpeg, create_highlights_ffmpeg_multisource


# ── Serve-run filtering constants ─────────────────────────────────────────────
# Serves from the same camera side cluster into "runs".  A run is a group of
# consecutive active segments whose inter-segment gap is ≤ GAP_THRESHOLD_SEC.
# Runs shorter than MIN_SERVES_PER_RUN are discarded as spurious detections.
MIN_SERVES_PER_RUN  = 8
GAP_THRESHOLD_SEC   = 240.0   # 4 minutes — spans short changeovers within a run


# ─────────────────────────────────────────────────────────────────────────────
# Core segment-collection loop (shared by single and dual pipeline modes)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_segments(video_path, headless=False, start_frame=0, csv_path=None,
                      serve_side="near", define_far_region=False):
    """
    Run the Anya pipeline on a single video and return the detected active segments.

    Parameters
    ----------
    video_path  : source video file
    headless    : suppress all OpenCV windows
    start_frame : seek to this frame before starting (default 0)
    csv_path    : explicit CSV output path; defaults to
                  <video_dir>/<stem>_telemetry_<serve_side>.csv
    serve_side  : "near" or "far" — which baseline originates the serve.  The
                  telemetry provider and transition engine are configured for that
                  side; near reproduces the original single-side pipeline exactly.

    Returns
    -------
    active_segments : list of (start_sec, end_sec) in source-video time
    point_number    : total points detected
    csv_path        : path to the written telemetry CSV
    """
    # ── Default CSV path ──────────────────────────────────────────────────
    if csv_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        csv_path   = os.path.join(video_dir, f"{video_stem}_telemetry_{serve_side}.csv")

    # ── Probe original video properties ──────────────────────────────────
    _probe   = cv2.VideoCapture(video_path)
    orig_fps = _probe.get(cv2.CAP_PROP_FPS)
    _probe.release()
    if orig_fps <= 0 or orig_fps > 300:
        orig_fps = 30.0

    # ── Initialize pipeline components ───────────────────────────────────
    telemetry_provider = AnyaTelemetryProvider(video_path, serve_side=serve_side,
                                                define_far_region=define_far_region)
    engine             = TransitionEngine(fps=telemetry_provider.fps, serve_side=serve_side)

    # ── CSV telemetry writer ──────────────────────────────────────────────
    _CSV_COLS = [
        "point", "frame", "timestamp", "state",
        "track_state", "has_active_trace", "time_since_detection",
        "ball_speed_px_s", "coasting", "ball_count",
        "maneuver_prob", "racket_prob", "bounce_prob",
    ]
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLS)
    csv_writer.writeheader()

    # ── Segment tracking ──────────────────────────────────────────────────
    _probe2 = cv2.VideoCapture(video_path)
    _total_frames      = int(_probe2.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = _total_frames / orig_fps if _total_frames > 0 else float("inf")
    _probe2.release()

    video_time_offset     = start_frame / orig_fps
    active_segments       = []
    current_segment_start: float = 0.0
    last_telemetry_ts:     float = 0.0
    HIGHLIGHT_END_PAD_SEC = 1.0

    # ── Main loop ─────────────────────────────────────────────────────────
    cap            = cv2.VideoCapture(video_path)
    point_number   = 0
    frame_in_point = 0

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[DEBUG] Seeking to frame {start_frame}")

    WAITING_STRIDE = 3
    interrupted    = False

    try:
        while cap.isOpened():
            success, orig_frame = cap.read()
            if not success:
                break

            frame = cv2.resize(orig_frame, (960, 540), interpolation=cv2.INTER_LINEAR)

            skip_inference = (
                telemetry_provider.current_state == "WAITING"
                and telemetry_provider.frame_counter % WAITING_STRIDE != 0
                and bool(telemetry_provider.telemetry_history)
            )

            if skip_inference:
                telemetry_provider.frame_counter += 1
                last = telemetry_provider.telemetry_history[-1]
                telemetry = TelemetryFrame(
                    frame_id=telemetry_provider.frame_counter,
                    timestamp=telemetry_provider.frame_counter / telemetry_provider.fps,
                    state="WAITING",
                    near_player_box=last.near_player_box,
                    near_player_world=last.near_player_world,
                    far_player_box=last.far_player_box,
                    far_player_world=last.far_player_world,
                    near_player_boxes=last.near_player_boxes,
                    far_player_boxes=last.far_player_boxes,
                    serve_player_box=last.serve_player_box,
                    serve_player_world=last.serve_player_world,
                    toss_ball_candidates=[],
                    active_ball_candidates=[],
                )
                telemetry_provider.telemetry_history.append(telemetry)
            else:
                telemetry = telemetry_provider.process_frame(frame)

            last_telemetry_ts = telemetry.timestamp

            new_state = engine.evaluate_transitions(
                telemetry_provider.telemetry_history,
                telemetry_provider.current_state,
            )

            old_state = telemetry_provider.current_state
            if new_state != old_state:
                if new_state == "ACTIVE":
                    point_number   += 1
                    frame_in_point  = 0
                    current_segment_start = video_time_offset + telemetry.timestamp
                elif old_state == "ACTIVE":
                    end_t = (engine.last_transition_time
                             if engine.last_transition_time is not None
                             else telemetry.timestamp)
                    padded_end = min(video_time_offset + end_t + HIGHLIGHT_END_PAD_SEC,
                                     video_duration_sec)
                    active_segments.append((current_segment_start, padded_end))
                telemetry_provider.update_state(new_state)

            current_state = telemetry_provider.current_state

            if current_state == "ACTIVE":
                frame_in_point += 1
                _write_csv_row(csv_writer, engine, telemetry, point_number, frame_in_point)

            if not headless:
                render_frame(frame, telemetry, current_state, engine,
                             telemetry_provider.exclusion_zones,
                             telemetry_provider.active_zone_polygon,
                             far_region_polygon=telemetry_provider.far_region_polygon,
                             serve_side=serve_side)
                debug_panel = render_debug_panel(current_state, engine, telemetry_provider)
                cv2.imshow(f"Anya Pipeline [{serve_side}]", frame)
                cv2.imshow(f"Debug Panel [{serve_side}]", debug_panel)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        interrupted = True
        print("\n[INTERRUPTED] Ctrl-C detected — creating highlights from completed segments...")

    finally:
        if telemetry_provider.current_state == "ACTIVE":
            padded_end = min(video_time_offset + last_telemetry_ts + HIGHLIGHT_END_PAD_SEC,
                             video_duration_sec)
            active_segments.append((current_segment_start, padded_end))

        cap.release()
        csv_file.close()
        if not headless:
            cv2.destroyAllWindows()

    print(f"[COLLECT] {os.path.basename(video_path)} [{serve_side}]: {point_number} points, "
          f"{len(active_segments)} segments")
    if interrupted:
        print("[COLLECT] (pipeline interrupted — segments cover completed detections only)")

    return active_segments, point_number, csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Single-video pipeline — near-origin and far-origin serves, merged by timestamp
# ─────────────────────────────────────────────────────────────────────────────

def _merge_overlapping_segments(segments, merge_gap_sec=0.0):
    """
    Sort segments by start time and merge any that overlap (or sit within
    merge_gap_sec of each other).  The two serve-side passes detect points
    independently, so the same rally can occasionally be picked up by both; this
    coalesces those into a single clip rather than cutting it twice.

    Parameters
    ----------
    segments      : list of (start_sec, end_sec)
    merge_gap_sec : gaps ≤ this many seconds are bridged into one segment

    Returns the merged, chronologically ordered list of (start_sec, end_sec).
    """
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s[0])
    merged  = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def run_anya_pipeline(video_path, output_path=None, headless=False, start_frame=0,
                      far_first=False, define_far_region=False):
    """
    Detect both near-origin and far-origin serves in a single video and build one
    highlight reel ordered by point timestamp.

    The pipeline runs in two passes over the same source: one configured for the
    near-side server (identical to the original single-side behaviour) and one for
    the far-side server.  The two segment lists are unioned, sorted by start time,
    and overlapping detections are merged before the highlight MP4 is cut.

    Parameters
    ----------
    far_first : process the far-side pass before the near-side pass.  The output
                reel is timestamp-ordered regardless; this only changes which side
                is analysed (and logged/CSV-written) first.
    define_far_region : force the interactive far-side detection region selector
                even if a cached region already exists for this video.
    """
    # ── Default output path ───────────────────────────────────────────────
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_highlights.mp4")

    out_stem = os.path.splitext(output_path)[0]
    order    = ["far", "near"] if far_first else ["near", "far"]

    results = {}   # serve_side -> (segments, point_number, csv_path)
    for serve_side in order:
        print(f"\n{'='*60}\n  PASS: {serve_side.upper()}-SIDE SERVES\n{'='*60}")
        csv_path = f"{out_stem}_telemetry_{serve_side}.csv"
        segs, points, _ = _collect_segments(
            video_path, headless, start_frame, csv_path=csv_path, serve_side=serve_side,
            define_far_region=define_far_region,
        )
        results[serve_side] = (segs, points, csv_path)

    near_segs, near_pts, near_csv = results["near"]
    far_segs,  far_pts,  far_csv  = results["far"]

    merged = _merge_overlapping_segments(near_segs + far_segs)

    create_highlights_ffmpeg(video_path, merged, output_path)

    print(f"\n[DONE] Output video    : {output_path}")
    print(f"[DONE] Near telemetry  : {near_csv}  ({near_pts} points, {len(near_segs)} segments)")
    print(f"[DONE] Far  telemetry  : {far_csv}  ({far_pts} points, {len(far_segs)} segments)")
    print(f"[DONE] Merged segments : {len(merged)}  (near+far, timestamp-ordered)")


# ─────────────────────────────────────────────────────────────────────────────
# Serve-run helpers for dual-camera mode
# ─────────────────────────────────────────────────────────────────────────────

def _group_segments_into_runs(segments, gap_threshold_sec=GAP_THRESHOLD_SEC):
    """
    Group temporally adjacent segments into runs.

    Two consecutive segments belong to the same run when the gap between the
    end of one and the start of the next is ≤ gap_threshold_sec.

    Returns a list of runs, each run being a list of (start_sec, end_sec) tuples.
    """
    if not segments:
        return []
    runs         = []
    current_run  = [segments[0]]
    for seg in segments[1:]:
        gap = seg[0] - current_run[-1][1]
        if gap <= gap_threshold_sec:
            current_run.append(seg)
        else:
            runs.append(current_run)
            current_run = [seg]
    runs.append(current_run)
    return runs


def _filter_by_serve_run(segments, video_label, min_run=MIN_SERVES_PER_RUN,
                          gap_threshold_sec=GAP_THRESHOLD_SEC):
    """
    Keep only segments that belong to runs of >= min_run consecutive serves.

    Tennis serves are taken from the same side for at least min_run serves
    before the players rotate ends/cameras.  Runs shorter than this threshold
    are likely false-positive detections or incomplete recordings and should be
    discarded.

    # TODO: End-of-set tiebreakers rotate serve every 2 points rather than
    #       holding serve for a full game.  The current >=8-serve-per-run filter
    #       and fixed gap threshold do not handle tiebreaker sequences correctly.
    #       Implement tiebreaker detection (e.g. score-aware or pattern-based)
    #       and apply a separate splicing logic for tiebreak segments.

    Parameters
    ----------
    segments          : list of (start_sec, end_sec)
    video_label       : human-readable label for log output (e.g. "Video A")
    min_run           : minimum number of serves required to keep a run
    gap_threshold_sec : maximum gap (seconds) allowed between serves in a run

    Returns filtered list of (start_sec, end_sec).
    """
    runs           = _group_segments_into_runs(segments, gap_threshold_sec)
    valid_segments = []
    for i, run in enumerate(runs):
        if len(run) >= min_run:
            valid_segments.extend(run)
            print(f"[DUAL] {video_label}: run {i+1} — {len(run)} serves  (VALID)")
        else:
            print(f"[DUAL] {video_label}: run {i+1} — {len(run)} serves  "
                  f"(DISCARDED, fewer than {min_run} required)")
    return valid_segments


# ─────────────────────────────────────────────────────────────────────────────
# Dual-camera pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_anya_pipeline_dual(video_a, video_b, time_offset_sec=0.0,
                            output_path=None, headless=False, start_frame_a=0):
    """
    Run the Anya pipeline on two camera angles and produce a spliced highlight reel.

    Each video is processed independently.  Active segments from each video are
    grouped into serve runs and any run shorter than MIN_SERVES_PER_RUN is
    discarded.  Surviving segments are merged in chronological wall-clock order
    (using time_offset_sec to align the two timelines) and cut from the
    appropriate source file.

    Parameters
    ----------
    video_a          : Path to video from camera A (the reference timeline).
    video_b          : Path to video from camera B.
    time_offset_sec  : (video_A recording start) − (video_B recording start) in
                       seconds.  Positive means video_A started *after* video_B.
                       Used only for chronological sorting of the merged segments;
                       each source file is still cut at its own local timestamps.
    output_path      : Output highlight MP4 (default: <video_a_dir>/<stem>_dual_highlights.mp4).
    headless         : Run without display windows.
    start_frame_a    : Start processing video_A from this frame; video_B always
                       starts at frame 0.

    # TODO: End-of-set tiebreakers rotate serve every 2 points rather than
    #       holding serve for a full game.  The current >=8-serve-per-run filter
    #       may incorrectly discard tiebreak segments or fail to isolate them
    #       from surrounding regular-play runs.  Implement tiebreaker detection
    #       and apply separate splicing logic for those sequences.
    """
    if output_path is None:
        video_dir  = os.path.dirname(os.path.abspath(video_a))
        video_stem = os.path.splitext(os.path.basename(video_a))[0]
        output_path = os.path.join(video_dir, f"{video_stem}_dual_highlights.mp4")

    print(f"\n{'='*60}")
    print(f"  DUAL PIPELINE")
    print(f"  Video A : {os.path.basename(video_a)}")
    print(f"  Video B : {os.path.basename(video_b)}")
    print(f"  Offset  : {time_offset_sec:+.2f}s  (A start − B start)")
    print(f"{'='*60}\n")

    # ── Process each video independently ─────────────────────────────────
    print(f"[DUAL] Processing Video A: {os.path.basename(video_a)}")
    segs_a, points_a, csv_a = _collect_segments(video_a, headless, start_frame_a)

    print(f"\n[DUAL] Processing Video B: {os.path.basename(video_b)}")
    segs_b, points_b, csv_b = _collect_segments(video_b, headless, start_frame=0)

    # ── Filter by serve-run length ────────────────────────────────────────
    print(f"\n[DUAL] Filtering serve runs (min {MIN_SERVES_PER_RUN} serves per run, "
          f"gap ≤ {GAP_THRESHOLD_SEC:.0f}s) …")
    valid_segs_a = _filter_by_serve_run(segs_a, "Video A")
    valid_segs_b = _filter_by_serve_run(segs_b, "Video B")

    # ── Tag with wall-clock time for chronological merge ──────────────────
    # Wall-clock reference: video_B start = 0.
    # video_A wall time   = local_timestamp + time_offset_sec
    # video_B wall time   = local_timestamp + 0
    tagged_a = [(video_a, s, e, s + time_offset_sec) for s, e in valid_segs_a]
    tagged_b = [(video_b, s, e, s)                   for s, e in valid_segs_b]

    all_tagged = sorted(tagged_a + tagged_b, key=lambda x: x[3])

    if not all_tagged:
        print("\n[DUAL] No valid segments remain after filtering — no output produced.")
        return

    # ── Strip wall-clock field → (source_path, start_sec, end_sec) ───────
    merged = [(src, s, e) for src, s, e, _ in all_tagged]

    create_highlights_ffmpeg_multisource(merged, output_path)

    print(f"\n[DONE] Output video   : {output_path}")
    print(f"[DONE] Video A CSV    : {csv_a}")
    print(f"[DONE] Video B CSV    : {csv_b}")
    print(f"[DONE] Video A points : {points_a}  ({len(valid_segs_a)} valid segments)")
    print(f"[DONE] Video B points : {points_b}  ({len(valid_segs_b)} valid segments)")
    print(f"[DONE] Total segments in output: {len(merged)}")


# ── CSV helper ─────────────────────────────────────────────────────────────────

def _write_csv_row(csv_writer, engine, telemetry, point_number, frame_in_point):
    """Write per-frame telemetry state and ball-tracker status."""
    debug = engine.last_active_debug

    csv_writer.writerow({
        "point":                point_number,
        "frame":                frame_in_point,
        "timestamp":            round(telemetry.timestamp, 4),
        "state":                telemetry.state,
        "track_state":          debug.get("state", "none"),
        "has_active_trace":     debug.get("has_active_trace", False),
        "time_since_detection": round(debug.get("time_since_detection", 0.0), 3),
        "ball_speed_px_s":      round(debug.get("ball_speed_px_s", 0.0), 1),
        "coasting":             debug.get("coasting", False),
        "ball_count":           debug.get("ball_count", 0),
        "maneuver_prob":        round(debug.get("maneuver_prob", 0.0), 3),
        "racket_prob":          round(debug.get("racket_prob",  0.0), 3),
        "bounce_prob":          round(debug.get("bounce_prob",  0.0), 3),
    })


# ── Debug visualisation ────────────────────────────────────────────────────────

def render_frame(frame, telemetry, state, engine=None, exclusion_zones=None,
                 active_zone_polygon=None, far_region_polygon=None, serve_side="near"):
    """Debug overlay — state badge, player boxes, balls, exclusion zones, and active polygon."""

    # Draw translucent light-green active-zone polygon in ACTIVE state
    if state == "ACTIVE" and active_zone_polygon is not None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [active_zone_polygon], (144, 238, 144))   # light green (BGR)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        cv2.polylines(frame, [active_zone_polygon], True, (0, 200, 0), 1)

    # Draw the user-defined far-side detection region (cyan outline), always visible.
    if far_region_polygon is not None:
        cv2.polylines(frame, [far_region_polygon], True, (255, 255, 0), 1, cv2.LINE_AA)

    color = (0, 255, 0) if state == "ACTIVE" else (0, 255, 255)
    cv2.putText(frame, f"STATE: {state}  ({serve_side} serve)", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Near players — blue boxes (supports multiple players, e.g. doubles)
    for box in (telemetry.near_player_boxes or []):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Far players — pink boxes in all states.
    for box in (telemetry.far_player_boxes or []):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 105, 255), 2)
        cv2.putText(frame, "FAR", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 105, 255), 1, cv2.LINE_AA)

    # Serving player — thick yellow highlight + label (the side under analysis).
    if telemetry.serve_player_box:
        x1, y1, x2, y2 = telemetry.serve_player_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(frame, f"SERVER ({serve_side})", (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw red bounding boxes for exclusion zones
    if exclusion_zones:
        for x1, y1, x2, y2 in exclusion_zones:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    # Draw yellow zone box for ARMED phase
    if state == "ARMED" and telemetry.z_box:
        x1, y1, x2, y2 = telemetry.z_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Draw green bounding boxes around detected balls
    if state == "ARMED" and telemetry.toss_ball_candidates:
        for ball in telemetry.toss_ball_candidates:
            bx1, by1, bx2, by2 = ball["box"]
            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 2)

    if state == "ACTIVE" and engine is not None:
        # ── Ball trace (fading orange trail, 1.5s window) ─────────────────
        trace = [(px, py) for _, px, py in engine._trace_ball_history]
        n = len(trace)
        if n >= 2:
            for i in range(1, n):
                age = i / (n - 1)          # 0 = oldest segment end, 1 = newest
                color = (0, int(120 * age), int(255 * age))   # dark → orange (BGR)
                thickness = max(1, int(3 * age))
                pt1 = (int(trace[i - 1][0]), int(trace[i - 1][1]))
                pt2 = (int(trace[i][0]),     int(trace[i][1]))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
        if n >= 1:
            tip = (int(trace[-1][0]), int(trace[-1][1]))
            cv2.circle(frame, tip, 5, (0, 200, 255), -1, cv2.LINE_AA)   # bright orange dot

        # ── Current-frame ball detections ─────────────────────────────────
        if telemetry.active_ball_candidates:
            for ball in telemetry.active_ball_candidates:
                bx1, by1, bx2, by2 = ball["box"]
                cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)),
                              (0, 255, 255), 2)   # cyan box around live detection


def render_debug_panel(state, engine, telemetry_provider):
    """
    Create a debug visualization panel showing timeout status and serve scores.
    Returns an image to display in a separate window.
    """
    panel_width, panel_height = 500, 300
    panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 240  # Light gray bg

    if state == "ACTIVE":
        render_active_debug(panel, engine)
    elif state == "ARMED":
        render_armed_debug(panel, engine)
    else:
        cv2.putText(panel, "WAITING FOR PLAYER", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 80), 1)

    return panel


def render_active_debug(panel, engine):
    """
    Visualizes the ball-tracker state driving ACTIVE -> WAITING.
    """
    x0, y, lh = 15, 35, 30
    fs = 0.5

    cv2.putText(panel, "ACTIVE — BALL TRACE", (x0, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)

    d = engine.last_active_debug

    # 1. Tracker state (none|tentative|moving|coasting|stopped|lost|fading)
    track_state = d.get("state", "none")
    alive       = d.get("has_active_trace", False)
    state_color = (0, 180, 0) if alive else (0, 0, 200)
    cv2.putText(panel, f"Track State: {track_state.upper()}",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, fs, state_color, 2)
    y += lh

    # 2. Moving trace alive/dead + time since last real detection
    tsd         = d.get("time_since_detection", 0.0)
    trace_color = (0, 180, 0) if alive else (0, 0, 200)
    trace_label = "ALIVE" if alive else "DEAD"
    cv2.putText(panel, f"Moving Trace: {trace_label}   (last seen {tsd:.2f}s ago)",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, fs, trace_color, 1)
    y += lh

    # 3. Ball speed (Kalman velocity magnitude, px/s)
    speed = d.get("ball_speed_px_s", 0.0)
    cv2.putText(panel, f"Ball Speed: {speed:6.0f} px/s",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 80, 80), 1)
    y += lh

    # 4. Coasting indicator (predicting through a brief detection gap)
    if d.get("coasting", False):
        cv2.putText(panel, "COASTING (bridging occlusion)",
                    (x0, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 140, 220), 1)
        y += lh

    # 5. IMM contact probabilities — maneuver (total), racket, bounce
    mp  = d.get("maneuver_prob", 0.0)
    rp  = d.get("racket_prob",  0.0)
    bp  = d.get("bounce_prob",  0.0)
    mp_color = (0, 140, 220) if mp > 0.4 else (80, 80, 80)
    cv2.putText(panel, f"Maneuver: {mp:.2f}  Racket: {rp:.2f}  Bounce: {bp:.2f}",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, fs, mp_color, 1)
    y += lh

    # 6. Raw detections this frame
    ball_count = d.get("ball_count", 0)
    cv2.putText(panel, f"Balls detected: {ball_count}",
                (x0, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 80, 80), 1)


def render_armed_debug(panel, engine):
    """Render serve scores during ARMED state."""
    x0      = 12
    bar_w   = 200
    bar_h   = 14
    lh      = 30
    label_w = 110

    cv2.putText(panel, "ARMED  —  Serve Detection", (x0, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    scores = engine.last_serve_scores
    rows = [
        ("Trophy Score", scores.get("trophy_score", 0.0), (0, 120, 255)),
        ("Toss Score",   scores.get("toss_score",   0.0), (0, 200, 200)),
        ("Serve Score",  scores.get("serve_score",  0.0), None),
    ]

    y = 65
    for label, value, color in rows:
        if color is None:
            color = (0, 220, 0) if value >= 0.55 else (0, 140, 255)
        cv2.putText(panel, f"{label}:", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        bx = x0 + label_w
        cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + bar_w, y + 2), (190, 190, 190), -1)
        cv2.rectangle(panel, (bx, y - bar_h + 2), (bx + int(value * bar_w), y + 2), color, -1)
        cv2.putText(panel, f"{value:.3f}", (bx + bar_w + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if label == "Serve Score":
            thresh_x = bx + int(0.55 * bar_w)
            cv2.line(panel, (thresh_x, y - bar_h + 2), (thresh_x, y + 2), (0, 0, 0), 2)
        y += lh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anya Tennis Point Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single camera:
    python -m src.ai.run_anya video.mp4
    python -m src.ai.run_anya video.mp4 --output out.mp4 --headless

  Dual camera:
    python -m src.ai.run_anya cam_a.mp4 cam_b.mp4 --time-offset 12.5
    python -m src.ai.run_anya cam_a.mp4 cam_b.mp4 --time-offset -5 --output spliced.mp4 --headless
""",
    )
    parser.add_argument(
        "input",
        nargs="+",
        metavar="VIDEO",
        help="Path to input video file(s). Provide one video for single-camera mode, "
             "or two videos (video_A video_B) for dual-camera mode.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output MP4 file (default: derived from first input video name).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying video windows.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="[Single/Dual mode] Start processing from this frame number in video_A (default: 0).",
    )
    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "[Dual mode only] Time offset in seconds: (video_A recording start) minus "
            "(video_B recording start).  Positive = video_A started after video_B.  "
            "Used for chronological sorting of the merged segment list (default: 0.0)."
        ),
    )
    parser.add_argument(
        "--far-first",
        action="store_true",
        help="[Single mode] Process far-side serves before near-side serves. The "
             "output reel is timestamp-ordered regardless; this only changes the "
             "analysis/logging order.",
    )
    parser.add_argument(
        "--define-far-region",
        action="store_true",
        help="[Single mode] Force the interactive far-side player detection "
             "region selector (4 points) even if a region is already cached for "
             "this video, e.g. to fence off an adjacent court or bench behind "
             "the far baseline. Cached to <video_dir>/far_region_config.json.",
    )
    args = parser.parse_args()

    if len(args.input) == 1:
        run_anya_pipeline(args.input[0], args.output, args.headless, args.start_frame,
                          far_first=args.far_first, define_far_region=args.define_far_region)
    elif len(args.input) == 2:
        run_anya_pipeline_dual(
            args.input[0], args.input[1],
            time_offset_sec=args.time_offset,
            output_path=args.output,
            headless=args.headless,
            start_frame_a=args.start_frame,
        )
    else:
        parser.error("Provide 1 or 2 input videos.")
