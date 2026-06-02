"""
point_end_detector.py
=====================
Stateless point-end detection pipeline.

Requires serve candidates produced by serve_detector.py.

Point-end detection uses a frame-by-frame probability model per rally window:

  • Ball detected         → P(point active) resets to 1.0.
  • Ball not found        → P decays at BALL_DECAY_RATE per second.
  • Ball not found AND player has minimal acceleration over a rolling window
    (still or constant velocity) → P decays at an additional
    PLAYER_DECAY_RATE per second.

  point_end_ts = first frame where P < P_THRESHOLD that is not subsequently
                 cancelled by a ball detection (which would reset P to 1.0).

  Step 1 — Serve windows        : derived from <stem>_final_candidates.json
                                   and <stem>_player_telemetry.json
  Step 2 — Rally ball scan      : <stem>_rally_ball_telemetry.json
  Step 3 — Point-end candidates : <stem>_point_end_candidates.json

Usage:
    python -m src.ai.point_end_detector path/to/video.mp4 [--highlights]
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import deque
from typing import List, Optional, Tuple

import cv2
from ultralytics import YOLO

from src.ai.utilities import Config, _is_in_exclusion_zone

# ── Pipeline constants ────────────────────────────────────────────────────────

ANALYSIS_SIZE          = (960, 540)
RALLY_BALL_IMGSZ       = 1280
RALLY_BALL_CONF        = 0.25

BALL_HISTORY_SEC       = 1.5   # rolling window for trace deques
TRACE_NEARBY_PX        = 40.0  # px radius — detections clustered tighter than this are stationary
TRACE_NEARBY_MIN_COUNT = 5     # min nearby hits to classify a detection as stationary (excluded from trace)

# Energy bar constants (ported from anya_transitions.py)
ENERGY_BOOST_SPRINT         = 4.0   # energy/s while sprinting
ENERGY_BOOST_SWING          = 4.0   # energy/s during swing / shape-change
ENERGY_DECAY_WALKING        = 0.5   # energy/s drain when walking gait detected
ENERGY_DECAY_STILL          = 0.3   # energy/s drain when standing still
ENERGY_DECAY_MISSING        = 0.5   # energy/s drain when player not detected
PLAYER_SPRINT_VELOCITY_FTS  = 7.0   # ft/s world-space → sprinting
PLAYER_STILL_VELOCITY_FTS   = 2.0   # ft/s world-space → standing still
VELOCITY_WINDOW_SIZE        = 20    # position samples for velocity smoothing
PLAYER_EMA_ALPHA            = 0.25  # EMA smoothing factor for world position
GAIT_BUFFER_FRAMES          = 45
GAIT_MIN_REVERSALS          = 2
GAIT_MAX_REVERSALS          = 8
GAIT_MIN_DRIFT_PX           = 10.0
SCREEN_HEIGHT_PX            = 540
BOTTOM_SCREEN_TOLERANCE_PX  = 8
PLAYER_MISSING_GRACE_FRAMES = 5

MIN_POINT_DURATION_SEC = 2.5
MAX_POINT_DURATION_SEC = 50.0
MIN_WINDOW_SEC         = 3.0


# ── Path helper ───────────────────────────────────────────────────────────────

def _out(video_path: str, suffix: str) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    s = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{s}_{suffix}")


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _in_player_box(cx: float, cy: float, box, padding: int = 10) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return x1 - padding <= cx <= x2 + padding and y1 - padding <= cy <= y2 + padding


def _court_crop_rect(court_points) -> Optional[Tuple[int, int, int, int]]:
    """Bounding rectangle of the four court corner pixels, clamped to frame."""
    if not court_points or len(court_points) < 4:
        return None
    xs = [v[0] for v in court_points]
    ys = [v[1] for v in court_points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


# ── Step 1 — Serve windows ────────────────────────────────────────────────────

def _serve_windows(serve_candidates, video_duration_sec: float) -> List[dict]:
    """
    Build per-rally search windows from serve candidates.

    Each window spans [serve_ts + MIN_POINT_DURATION_SEC, next_serve_ts] capped
    at MAX_POINT_DURATION_SEC.  Windows shorter than MIN_WINDOW_SEC are dropped
    (first-serve faults).
    """
    serves  = sorted(ts for ts, _ in serve_candidates)
    windows = []
    for i, ts in enumerate(serves):
        if i + 1 < len(serves):
            search_end = min(serves[i + 1], ts + MAX_POINT_DURATION_SEC)
        else:
            search_end = min(ts + MAX_POINT_DURATION_SEC, video_duration_sec)

        if search_end - ts < MIN_WINDOW_SEC:
            continue

        windows.append({
            "serve_ts":     round(ts, 4),
            "search_start": round(ts + MIN_POINT_DURATION_SEC, 4),
            "search_end":   round(search_end, 4),
        })
    return windows


# ── Step 2 — Rally ball scan ──────────────────────────────────────────────────

def step2_rally_ball_scan(
    video_path: str,
    windows,
    player_telemetry,
    exclusion_zones,
    ball_model,
    court_rect=None,
) -> List[dict]:
    """
    For each frame inside a search window, collect raw ball candidates.
    Returns list of {frame_id, timestamp, ball_candidates: [{pixel_center, conf}]}.
    """
    path = _out(video_path, "rally_ball_telemetry.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if data and "ball_candidates" in data[0]:
            print(f"[STEP 2] Loaded {len(data)} rally frames from {os.path.basename(path)}")
            return data
        print(f"[STEP 2] Stale cache (old format) — re-scanning …")
        os.remove(path)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    scan_ids: set = set()
    for w in windows:
        for fid in range(int(w["search_start"] * fps), int(w["search_end"] * fps) + 1):
            scan_ids.add(fid)

    player_by_id = {r["frame_id"]: r for r in player_telemetry}
    print(f"\n[STEP 2] Scanning {len(scan_ids)} frames for rally balls "
          f"(imgsz={RALLY_BALL_IMGSZ}) …")

    cap      = cv2.VideoCapture(video_path)
    frame_id = 0
    records  = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id not in scan_ids:
            continue

        frame    = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_LINEAR)
        ts       = round(frame_id / fps, 4)
        pdata    = player_by_id.get(frame_id, {})
        near_box = tuple(pdata["near_player_box"]) if pdata.get("near_player_box") else None

        candidates = _detect_rally_ball(frame, near_box, exclusion_zones, ball_model, court_rect)
        records.append({
            "frame_id":        frame_id,
            "timestamp":       ts,
            "ball_candidates": candidates,
        })

        if len(records) % 500 == 0:
            print(f"  … {len(records)}/{len(scan_ids)} rally frames scanned")

    cap.release()

    with open(path, "w") as f:
        json.dump(records, f)
    print(f"[STEP 2] {len(records)} rally frames → {os.path.basename(path)}")
    return records


def _detect_rally_ball(frame, near_box, exclusion_zones, ball_model,
                        court_rect=None) -> List[dict]:
    """Return list of valid ball candidates [{pixel_center, conf}] for this frame."""
    if court_rect is not None:
        rx1, ry1, rx2, ry2 = court_rect
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return []
    else:
        roi, rx1, ry1 = frame, 0, 0

    results = ball_model(roi, verbose=False, conf=RALLY_BALL_CONF, imgsz=RALLY_BALL_IMGSZ)
    if not (results and results[0].boxes):
        return []

    candidates = []
    for b in results[0].boxes:
        cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
        cx = rx1 + (cx1 + cx2) / 2.0   # translate back to full-frame space
        cy = ry1 + (cy1 + cy2) / 2.0
        if _is_in_exclusion_zone(cx, cy, exclusion_zones):
            continue
        if _in_player_box(cx, cy, near_box):
            continue
        candidates.append({"pixel_center": (cx, cy), "conf": float(b.conf[0])})
    return candidates


# ── Step 3 — Point-end candidates ────────────────────────────────────────────

def _detect_walking_gait(gait_y_buffer) -> bool:
    """
    Detect walking gait from oscillatory y-movement of player feet (pixel space).
    Ported from TransitionEngine._detect_walking_gait() in anya_transitions.py.
    """
    ys = list(gait_y_buffer)
    n  = len(ys)
    if n < GAIT_BUFFER_FRAMES * 0.6:
        return False
    if abs(ys[-1] - ys[0]) < GAIT_MIN_DRIFT_PX:
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

    return GAIT_MIN_REVERSALS <= reversals <= GAIT_MAX_REVERSALS


def _compute_energy_delta(
    near_box,
    player_pos_buf: deque,
    player_box_buf: deque,
    gait_y_buf: deque,
    missing_frames: int,
    dt: float,
    fps: float,
) -> Tuple[float, str]:
    """
    Return (energy_delta, status_label) for one frame.
    Ported from TransitionEngine._compute_energy_delta() in anya_transitions.py.

    Priority (high → low):
      1. Player missing                → drain ENERGY_DECAY_MISSING
      2. Near player clipped at frame bottom → drain ENERGY_DECAY_WALKING
      3. Walking gait detected         → drain ENERGY_DECAY_WALKING
      4. Sprinting (high velocity)     → boost ENERGY_BOOST_SPRINT
      5. Swing / split-step (shape Δ) → boost ENERGY_BOOST_SWING
      6. Standing still               → drain ENERGY_DECAY_STILL
      7. Moving (neutral)             → tiny boost 0.1/s
    """
    if missing_frames > PLAYER_MISSING_GRACE_FRAMES:
        return -(ENERGY_DECAY_MISSING * dt), "MISSING"

    if (near_box is not None and
            near_box[3] >= SCREEN_HEIGHT_PX - BOTTOM_SCREEN_TOLERANCE_PX):
        return -(ENERGY_DECAY_WALKING * dt), "WALKING_OFFSCREEN"

    if _detect_walking_gait(gait_y_buf):
        return -(ENERGY_DECAY_WALKING * dt), "WALKING"

    player_velocity_fts = 0.0
    if len(player_pos_buf) >= 5:
        old_p   = player_pos_buf[0]
        new_p   = player_pos_buf[-1]
        dist_ft = math.hypot(new_p[0] - old_p[0], new_p[1] - old_p[1])
        elapsed = len(player_pos_buf) / fps
        player_velocity_fts = dist_ft / elapsed if elapsed > 0 else 0.0

    if player_velocity_fts > PLAYER_SPRINT_VELOCITY_FTS:
        return (ENERGY_BOOST_SPRINT * dt), f"SPRINTING {player_velocity_fts:.1f}ft/s"

    if len(player_box_buf) >= 5:
        old_b      = player_box_buf[0]
        new_b      = player_box_buf[-1]
        box_height = old_b[3] - old_b[1]
        if box_height > 0:
            dw = abs((new_b[2] - new_b[0]) - (old_b[2] - old_b[0]))
            dh = abs((new_b[3] - new_b[1]) - (old_b[3] - old_b[1]))
            if (dw + dh) / box_height > 0.25:
                return (ENERGY_BOOST_SWING * dt), "SWING"

    if player_velocity_fts < PLAYER_STILL_VELOCITY_FTS:
        return -(ENERGY_DECAY_STILL * dt), f"STILL {player_velocity_fts:.1f}ft/s"

    return (0.1 * dt), f"MOVING {player_velocity_fts:.1f}ft/s"


def step3_point_end_candidates(
    video_path: str,
    windows,
    rally_ball_tel,
    player_telemetry,
    fps: float,
) -> List[dict]:
    """
    Two-stage hybrid model per rally window, mirroring anya_transitions.py:

    Stage 1 — Ball Trace Gate:
      Maintain two rolling 1.5-second buffers (ALL_BALL_HISTORY and
      TRACE_BALL_HISTORY).  Point is alive unconditionally while
      TRACE_BALL_HISTORY is non-empty.  energy_bar_start_time is updated
      every trace frame so it always records when the ball last moved.

    Stage 2 — Energy Bar:
      When the trace goes empty, an energy bar starts at 1.0 and decays or
      boosts each frame based on player kinematics (walking gait, sprint
      velocity, swing/shape-change, still).  Point ends when energy reaches
      zero.  point_end_ts is rewound to energy_bar_start_time (when the
      ball last disappeared), matching the anya_transitions rewind anchor.
    """
    path = _out(video_path, "point_end_candidates.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"[STEP 3] Loaded {len(data)} point-end candidates from {os.path.basename(path)}")
        return data

    print("\n[STEP 3] Computing point-end candidates via energy bar …")

    player_by_fid = {r["frame_id"]: r for r in player_telemetry}
    ball_sorted   = sorted(rally_ball_tel, key=lambda r: r["timestamp"])
    dt            = 1.0 / fps

    candidates = []

    for w in windows:
        window_records = [
            r for r in ball_sorted
            if w["search_start"] <= r["timestamp"] <= w["search_end"]
        ]

        # ── Per-window rolling state ──────────────────────────────────────
        all_ball_history:   deque = deque()
        trace_ball_history: deque = deque()

        energy               = 1.0
        energy_bar_mode      = False
        energy_bar_start_ts  = w["search_start"]   # rewind anchor
        smoothed_world: Optional[Tuple[float, float]] = None
        player_pos_buf: deque = deque(maxlen=VELOCITY_WINDOW_SIZE)
        player_box_buf: deque = deque(maxlen=5)
        gait_y_buf:     deque = deque(maxlen=GAIT_BUFFER_FRAMES)
        missing_frames        = 0

        point_end_ts = None
        frame_count  = 0

        ms, ss = divmod(w["serve_ts"], 60)
        print(f"  [WINDOW] serve {int(ms):02d}:{ss:05.2f} ({len(window_records)} frames)")

        for r in window_records:
            fid    = r["frame_id"]
            ts     = r["timestamp"]
            cutoff = ts - BALL_HISTORY_SEC
            frame_count += 1

            # ── 1. Update player tracking buffers ─────────────────────────
            pdata    = player_by_fid.get(fid, {})
            near_box = tuple(pdata["near_player_box"]) if pdata.get("near_player_box") else None
            near_wld = pdata.get("near_player_world")

            if near_box is None or near_wld is None:
                missing_frames += 1
                gait_y_buf.clear()
            else:
                missing_frames = 0
                wx, wy = near_wld
                if smoothed_world is None:
                    smoothed_world = (wx, wy)
                else:
                    α = PLAYER_EMA_ALPHA
                    smoothed_world = (
                        α * wx + (1 - α) * smoothed_world[0],
                        α * wy + (1 - α) * smoothed_world[1],
                    )
                player_pos_buf.append(smoothed_world)
                player_box_buf.append(near_box)
                gait_y_buf.append(float(near_box[3]))

            # ── 2. Update ball trace buffers ──────────────────────────────
            for c in r["ball_candidates"]:
                px, py = c["pixel_center"]
                nearby = sum(1 for _, hx, hy in all_ball_history
                             if math.hypot(px - hx, py - hy) < TRACE_NEARBY_PX)
                if nearby < TRACE_NEARBY_MIN_COUNT:
                    trace_ball_history.append((ts, px, py))
            for c in r["ball_candidates"]:
                px, py = c["pixel_center"]
                all_ball_history.append((ts, px, py))

            while all_ball_history   and all_ball_history[0][0]   < cutoff:
                all_ball_history.popleft()
            while trace_ball_history and trace_ball_history[0][0] < cutoff:
                trace_ball_history.popleft()

            has_active_trace = bool(trace_ball_history)

            # ── 3. Stage 1: active trace → keep alive ─────────────────────
            if has_active_trace:
                energy_bar_start_ts = ts   # advance rewind anchor to now
                if energy_bar_mode:
                    print(f"    Ball trace restored @ {ts:.2f}s — discarding energy bar "
                          f"(was {energy:.2f})")
                    energy_bar_mode = False
                energy = 1.0
                continue

            # ── 4. Stage 2: no trace → energy bar ─────────────────────────
            if not energy_bar_mode:
                print(f"    No ball trace @ {ts:.2f}s — entering energy bar "
                      f"(anchor={energy_bar_start_ts:.2f}s)")
                energy_bar_mode = True
                energy          = 1.0

            delta, status = _compute_energy_delta(
                near_box, player_pos_buf, player_box_buf, gait_y_buf,
                missing_frames, dt, fps,
            )
            energy = max(0.0, min(1.0, energy + delta))

            if frame_count % 30 == 0:
                print(f"    frame {frame_count:4d} @ {ts:7.2f}s — "
                      f"energy={energy:.2f} [{status}]")

            if energy <= 0.0:
                point_end_ts = energy_bar_start_ts
                me, se = divmod(point_end_ts, 60)
                print(f"    ✓ DETECTED  @ {int(me):02d}:{se:05.2f}  [{status}]\n")
                break

        if point_end_ts is None:
            point_end_ts = w["search_end"]
            method       = "next_serve_fallback"
            me, se = divmod(point_end_ts, 60)
            print(f"    ~ FALLBACK  @ {int(me):02d}:{se:05.2f} (energy never depleted)\n")
        else:
            method = "energy_bar"

        candidates.append({
            "serve_ts":     w["serve_ts"],
            "point_end_ts": round(point_end_ts, 4),
            "duration_sec": round(point_end_ts - w["serve_ts"], 4),
            "confidence":   1.0 if method == "energy_bar" else 0.0,
            "method":       method,
        })

    with open(path, "w") as f:
        json.dump(candidates, f)

    print(f"[STEP 3] {len(candidates)} point(s) from {len(windows)} window(s):")
    for c in candidates:
        ms, ss = divmod(c["serve_ts"],     60)
        me, se = divmod(c["point_end_ts"], 60)
        flag   = "✓" if c["method"] == "energy_bar" else "~"
        print(f"  {flag} serve {int(ms):02d}:{ss:05.2f} → end {int(me):02d}:{se:05.2f}  "
              f"({c['duration_sec']:.1f}s, conf={c['confidence']:.2f})")

    return candidates


# ── Debug video ───────────────────────────────────────────────────────────────

def step4_debug_video(
    video_path: str,
    windows,
    rally_ball_tel,
    player_telemetry,
    fps: float,
    output_path: Optional[str] = None,
) -> str:
    """
    Render a debug video for every frame inside the search windows showing:

      • Orange circles  — trace ball history (moving ball, drives Stage 1)
      • Yellow circles  — stationary ball detections (in all_ball_history only)
      • Blue rectangle  — near-player bounding box
      • Energy bar      — bottom strip of the frame:
            Green fill  = Stage 1 (active ball trace, energy pinned at 1.0)
            Color fill  = Stage 2 (energy bar mode: green → yellow → red)
      • Status label    — energy bar cause (WALKING, SPRINTING, STILL, etc.)
      • Mode label      — "TRACE" or "ENERGY BAR"
    """
    if output_path is None:
        output_path = _out(video_path, "debug.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, ANALYSIS_SIZE)

    # Build frame-id → ball candidates lookup
    ball_by_fid   = {r["frame_id"]: r for r in rally_ball_tel}
    player_by_fid = {r["frame_id"]: r for r in player_telemetry}

    # Per-window state (keyed on window index, reset per window)
    def _make_window_state():
        return dict(
            all_ball_history   = deque(),
            trace_ball_history = deque(),
            energy             = 1.0,
            energy_bar_mode    = False,
            energy_bar_start_ts= 0.0,
            smoothed_world     = None,
            player_pos_buf     = deque(maxlen=VELOCITY_WINDOW_SIZE),
            player_box_buf     = deque(maxlen=5),
            gait_y_buf         = deque(maxlen=GAIT_BUFFER_FRAMES),
            missing_frames     = 0,
            status             = "TRACE",
        )

    dt = 1.0 / fps

    # Map each fid to its window index so we can reset state at window boundaries
    fid_to_win = {}
    for wi, w in enumerate(windows):
        for fid in range(int(w["search_start"] * fps), int(w["search_end"] * fps) + 1):
            fid_to_win[fid] = wi

    states      = {}   # window_index → state dict
    prev_win_idx = None

    cap      = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        if frame_id not in ball_by_fid:
            continue

        win_idx = fid_to_win.get(frame_id)

        if win_idx is None:
            continue   # telemetry frame outside any window — skip

        frame = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_LINEAR)
        ts    = round(frame_id / fps, 4)
        w     = windows[win_idx]

        # Reset state when entering a new window
        if win_idx != prev_win_idx:
            states[win_idx] = _make_window_state()
            states[win_idx]["energy_bar_start_ts"] = w["search_start"]
        prev_win_idx = win_idx
        s = states[win_idx]

        r        = ball_by_fid.get(frame_id, {"ball_candidates": []})
        cutoff   = ts - BALL_HISTORY_SEC
        pdata    = player_by_fid.get(frame_id, {})
        near_box = tuple(pdata["near_player_box"]) if pdata.get("near_player_box") else None
        near_wld = pdata.get("near_player_world")

        # ── Replay player tracking ────────────────────────────────────────────
        if near_box is None or near_wld is None:
            s["missing_frames"] += 1
            s["gait_y_buf"].clear()
        else:
            s["missing_frames"] = 0
            wx, wy = near_wld
            if s["smoothed_world"] is None:
                s["smoothed_world"] = (wx, wy)
            else:
                α = PLAYER_EMA_ALPHA
                s["smoothed_world"] = (
                    α * wx + (1 - α) * s["smoothed_world"][0],
                    α * wy + (1 - α) * s["smoothed_world"][1],
                )
            s["player_pos_buf"].append(s["smoothed_world"])
            s["player_box_buf"].append(near_box)
            s["gait_y_buf"].append(float(near_box[3]))

        # ── Replay ball trace buffers ─────────────────────────────────────────
        trace_set_this_frame = set()
        for c in r["ball_candidates"]:
            px, py = c["pixel_center"]
            nearby = sum(1 for _, hx, hy in s["all_ball_history"]
                         if math.hypot(px - hx, py - hy) < TRACE_NEARBY_PX)
            if nearby < TRACE_NEARBY_MIN_COUNT:
                s["trace_ball_history"].append((ts, px, py))
                trace_set_this_frame.add((px, py))
        for c in r["ball_candidates"]:
            px, py = c["pixel_center"]
            s["all_ball_history"].append((ts, px, py))

        while s["all_ball_history"]   and s["all_ball_history"][0][0]   < cutoff:
            s["all_ball_history"].popleft()
        while s["trace_ball_history"] and s["trace_ball_history"][0][0] < cutoff:
            s["trace_ball_history"].popleft()

        has_active_trace = bool(s["trace_ball_history"])

        # ── Replay energy state ───────────────────────────────────────────────
        if has_active_trace:
            s["energy_bar_start_ts"] = ts
            s["energy_bar_mode"]     = False
            s["energy"]              = 1.0
            s["status"]              = "TRACE"
        else:
            if not s["energy_bar_mode"]:
                s["energy_bar_mode"] = True
                s["energy"]          = 1.0
            delta, status = _compute_energy_delta(
                near_box, s["player_pos_buf"], s["player_box_buf"],
                s["gait_y_buf"], s["missing_frames"], dt, fps,
            )
            s["energy"] = max(0.0, min(1.0, s["energy"] + delta))
            s["status"] = status

        # ── Draw overlay ──────────────────────────────────────────────────────
        vis = frame.copy()

        # Near-player box
        if near_box is not None:
            x1, y1, x2, y2 = near_box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 1)

        # Stationary ball detections (yellow, small)
        trace_pts = {(int(px), int(py)) for _, px, py in s["trace_ball_history"]}
        for c in r["ball_candidates"]:
            px, py = int(c["pixel_center"][0]), int(c["pixel_center"][1])
            if (px, py) not in trace_pts:
                cv2.circle(vis, (px, py), 5, (0, 220, 220), -1)   # yellow-ish (stationary)

        # Trace history trail (orange, size fades with age)
        trace_list = list(s["trace_ball_history"])
        n_trace    = len(trace_list)
        for i, (t_entry, px, py) in enumerate(trace_list):
            age_frac = i / max(n_trace - 1, 1)          # 0 = oldest, 1 = newest
            radius   = max(3, int(4 + 5 * age_frac))    # 4 px old → 9 px new
            alpha    = 0.4 + 0.6 * age_frac             # dim old, bright new
            colour   = (0, int(140 * alpha), int(255 * alpha))   # orange in BGR
            cv2.circle(vis, (int(px), int(py)), radius, colour, -1)

        # Energy bar (bottom 28 px)
        bar_h  = 28
        bar_y0 = ANALYSIS_SIZE[1] - bar_h
        cv2.rectangle(vis, (0, bar_y0), (ANALYSIS_SIZE[0], ANALYSIS_SIZE[1]),
                      (30, 30, 30), -1)

        energy  = s["energy"]
        bar_w   = int(ANALYSIS_SIZE[0] * energy)

        if has_active_trace:
            bar_colour = (60, 200, 60)       # green — trace active
        else:
            # green → yellow → red as energy decays
            r_ch = int(255 * (1.0 - energy))
            g_ch = int(255 * energy)
            bar_colour = (0, g_ch, r_ch)

        cv2.rectangle(vis, (0, bar_y0), (bar_w, ANALYSIS_SIZE[1]), bar_colour, -1)

        mode_label   = "TRACE" if has_active_trace else "ENERGY BAR"
        status_label = s["status"]
        bar_text     = f"{mode_label}  {energy:.2f}  [{status_label}]"
        cv2.putText(vis, bar_text, (8, bar_y0 + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        # Timestamp and window info
        mm, ss_ = divmod(ts, 60)
        ts_label = f"t={int(mm):02d}:{ss_:05.2f}  serve={w['serve_ts']:.2f}"
        cv2.putText(vis, ts_label, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(vis)

    cap.release()
    writer.release()

    print(f"[STEP 4] Debug video → {os.path.basename(output_path)}")
    return os.path.abspath(output_path)


# ── Highlight splicer ─────────────────────────────────────────────────────────

def create_point_highlights(
    video_path: str,
    point_end_candidates: List[dict],
    output_path: Optional[str] = None,
    pre_padding_sec: float = 2.0,
    post_padding_sec: float = 0.5,
) -> str:
    """
    Splice each point (serve_ts → point_end_ts) into a single continuous
    highlights video using ffmpeg.

    Each segment is padded by pre_padding_sec before the serve and
    post_padding_sec after the point end.  Where padding from adjacent
    segments would overlap, the earlier segment's t_end is capped at the
    later segment's t_start so there is never any repeated footage.

    Each segment is re-encoded with H.264 fast preset for clean cuts
    regardless of keyframe alignment, then concatenated with stream copy.
    """
    if not point_end_candidates:
        raise ValueError("No point-end candidates to splice.")

    if output_path is None:
        output_path = _out(video_path, "highlights.mp4")

    cap            = cv2.VideoCapture(video_path)
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    # Pre-compute padded boundaries for every point, then clamp consecutive
    # pairs so t_end[i] never exceeds t_start[i+1].
    cands  = sorted(point_end_candidates, key=lambda c: c["serve_ts"])
    starts = [max(0.0,            c["serve_ts"]     - pre_padding_sec)  for c in cands]
    ends   = [min(video_duration, c["point_end_ts"] + post_padding_sec) for c in cands]
    for i in range(len(cands) - 1):
        if ends[i] > starts[i + 1]:
            ends[i] = starts[i + 1]

    tmpdir = tempfile.mkdtemp(prefix="point_highlights_")
    try:
        segment_paths = []
        for i, c in enumerate(cands):
            t_start = starts[i]
            t_end   = ends[i]
            if t_end <= t_start:
                print(f"  [SKIP] point {i+1}: degenerate window ({t_start:.2f}–{t_end:.2f})")
                continue

            seg_path = os.path.join(tmpdir, f"seg_{i:04d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{t_start:.4f}",
                "-to", f"{t_end:.4f}",
                "-i", video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                seg_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                print(f"  [WARN] ffmpeg failed for point {i+1}:\n"
                      f"         {result.stderr.decode()[-200:]}")
                continue

            segment_paths.append(seg_path)
            ms, ss = divmod(t_start, 60)
            me, se = divmod(t_end,   60)
            print(f"  [{i+1:3d}] {int(ms):02d}:{ss:05.2f} → {int(me):02d}:{se:05.2f}  "
                  f"({t_end - t_start:.1f}s)")

        if not segment_paths:
            raise RuntimeError("All segments failed — no highlights produced.")

        concat_txt = os.path.join(tmpdir, "concat.txt")
        with open(concat_txt, "w") as f:
            for p in segment_paths:
                f.write(f"file '{p}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_txt,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed:\n{result.stderr.decode()[-400:]}"
            )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n[HIGHLIGHTS] {len(segment_paths)} point(s) → {os.path.basename(output_path)}")
    return os.path.abspath(output_path)


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_pipeline(video_path: str, serve_candidates=None) -> List[dict]:
    if serve_candidates is None:
        cands_path = _out(video_path, "final_candidates.json")
        if not os.path.exists(cands_path):
            raise FileNotFoundError(
                f"Serve candidates not found: {cands_path}\n"
                "Run serve_detector.py first."
            )
        with open(cands_path) as f:
            serve_candidates = [tuple(c) for c in json.load(f)]
        print(f"[INIT] Loaded {len(serve_candidates)} serve candidate(s)")

    tel_path = _out(video_path, "player_telemetry.json")
    if not os.path.exists(tel_path):
        raise FileNotFoundError(
            f"Player telemetry not found: {tel_path}\n"
            "Run serve_detector.py first."
        )
    with open(tel_path) as f:
        player_tel = json.load(f)
    print(f"[INIT] Loaded {len(player_tel)} player telemetry frames")

    excl_path  = _out(video_path, "exclusion_zones.json")
    excl_zones = []
    if os.path.exists(excl_path):
        with open(excl_path) as f:
            excl_zones = [tuple(z) for z in json.load(f)]

    court_rect  = None
    court_path  = _out(video_path, "court_cache.json")
    if os.path.exists(court_path):
        with open(court_path) as f:
            court_data = json.load(f)
        court_rect = _court_crop_rect(court_data.get("points"))
        if court_rect:
            print(f"[INIT] Court crop rect: {court_rect}")
        else:
            print("[INIT] court_cache.json found but points missing — full-frame inference")
    else:
        print("[INIT] No court_cache.json — full-frame inference")

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    video_duration = total_frames / fps

    ball_model = YOLO("weights/ball/weights/best.pt")

    windows    = _serve_windows(serve_candidates, video_duration)
    print(f"[INIT] {len(windows)} search window(s) (after dropping short windows)")

    rally_tel  = step2_rally_ball_scan(video_path, windows, player_tel, excl_zones, ball_model, court_rect)
    candidates = step3_point_end_candidates(video_path, windows, rally_tel, player_tel, fps)

    print(f"\n[DONE] {len(candidates)} point end(s) detected in {os.path.basename(video_path)}")
    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stateless point-end detection pipeline")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--highlights", action="store_true",
                        help="Create highlights video after detection")
    parser.add_argument("--debug", action="store_true",
                        help="Render debug video with ball trace and energy bar overlay")
    args = parser.parse_args()

    cands = run_pipeline(args.video)
    if args.highlights:
        create_point_highlights(args.video, cands)
    if args.debug:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        with open(_out(args.video, "final_candidates.json")) as f:
            serve_cands = [tuple(c) for c in json.load(f)]
        with open(_out(args.video, "player_telemetry.json")) as f:
            player_tel = json.load(f)
        with open(_out(args.video, "rally_ball_telemetry.json")) as f:
            rally_tel = json.load(f)
        windows = _serve_windows(serve_cands, total / fps)
        step4_debug_video(args.video, windows, rally_tel, player_tel, fps)
