#!/usr/bin/env python3
"""
tune_far_params.py — grid search over FarSideTransitionEngine parameters.

Usage:
    python tune_far_params.py /Volumes/Anya/Data/23/snippet.mp4 [--top N]

PHASE 1 (record): runs the pipeline once with permissive parameters so
  ST-GCN (only active in ARMED state) and ball detection (only in ACTIVE)
  fire on as many frames as possible.  Per-frame TelemetryFrame snapshots
  are saved to <video_dir>/<stem>_far_tune_cache.pkl.  Re-run with --rerecord
  to force a fresh recording.

PHASE 2 (grid search): replays FarSideTransitionEngine offline on the cached
  frames for every parameter combination.  No inference — only the state
  machine logic runs, so all combinations finish in seconds.

PHASE 3 (report): prints the top-N combinations sorted by F1, then recall.

NOTE: offline replay is an approximation.  ST-GCN scores were accumulated
  while the engine was in ARMED during the RECORD run; replaying with very
  different ARMED entry times may shift those scores slightly.  Use the
  results to identify 3-5 candidate combos, then validate with full pipeline
  runs.
"""

import argparse
import contextlib
import csv
import io
import json
import os
import pickle
import sys
from collections import deque
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import cv2

from anya_base import FarSideTelemetryProvider, TelemetryFrame
from anya_transitions import FarSideTransitionEngine, SignalPriorityConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WAITING_STRIDE = 6   # mirrors run_anya.py


def _load_gt(video_path: str) -> List[Dict]:
    """Return far-side GT rallies as list of {start_s, end_s}."""
    gt_path = os.path.join(os.path.dirname(os.path.abspath(video_path)),
                           "ground_truth.json")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"No ground_truth.json next to {video_path}")
    with open(gt_path) as f:
        data = json.load(f)
    rallies = []
    for r in data["rallies"]:
        if r.get("serve") == "far":
            fps = _probe_fps(video_path)
            rallies.append({"start_s": r["start"] / fps, "end_s": r["end"] / fps})
    return rallies


def _probe_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    cap.release()
    return fps


# ---------------------------------------------------------------------------
# Per-frame snapshot (what we persist to disk)
# ---------------------------------------------------------------------------

@dataclass
class FrameSnap:
    frame_id:               int
    timestamp:              float
    recorded_state:         str        # provider state DURING recording
    far_serve_score:        float
    far_player_box:         Optional[Tuple]
    far_player_world:       Optional[Tuple]
    active_ball_candidates: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PHASE 1 — record
# ---------------------------------------------------------------------------

def _record_phase_cfg() -> SignalPriorityConfig:
    """Config for the recording pass.

    Uses threshold=1.01 (never fires ARMED→ACTIVE) so ST-GCN runs throughout
    the entire ARMED period and the full score curve is captured in the cache.
    Replay can then test any threshold ≤ 1.0 against the cached scores.
    """
    return SignalPriorityConfig(
        far_serve_score_threshold=1.01,   # never fire — stay in ARMED
        trace_downward_px_s=1.0,
        trace_horizontal_px_s=1.0,
    )


def record(video_path: str, cache_path: str,
           start_frame: int = 0, end_frame: int = 0) -> List[FrameSnap]:
    fps = _probe_fps(video_path)
    n_frames = (end_frame - start_frame) if end_frame > start_frame else None
    region = (f" frames {start_frame}–{end_frame}"
              if end_frame > start_frame else "")
    print(f"[RECORD] Running far-side provider on {os.path.basename(video_path)}{region} …")
    print("[RECORD] This runs full inference — may take several minutes.")

    provider = FarSideTelemetryProvider(video_path)
    cfg      = _record_phase_cfg()
    engine   = FarSideTransitionEngine(fps=provider.fps, cfg=cfg)
    engine.ARMED_OUT_RATIO_THRESHOLD = 0.99   # never drop ARMED→WAITING
    engine.READY_MIN_DIST_FT         = -20.0  # wide band — catch any far-court position
    engine.READY_MAX_DIST_FT         =  20.0

    # Seek to start_frame and offset the provider counter so timestamps are
    # absolute video times (matching the frame numbers in ground_truth.json).
    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        provider.frame_counter = start_frame  # first process_frame call adds 1 → start_frame+1
        # Adjust so first timestamp = (start_frame+1)/fps ≈ start_frame/fps
        # (off by one frame, acceptable for a ~420s video at 60fps)

    snaps: List[FrameSnap] = []
    state  = "WAITING"
    frames_processed = 0

    with contextlib.redirect_stdout(io.StringIO()):  # suppress engine chatter
        while True:
            if n_frames is not None and frames_processed >= n_frames:
                break
            success, orig_frame = cap.read()
            if not success:
                break

            skip = (
                state == "WAITING"
                and provider.frame_counter % WAITING_STRIDE != 0
                and bool(provider.telemetry_history)
            )

            if skip:
                provider.frame_counter += 1
                last = provider.telemetry_history[-1]
                telemetry = TelemetryFrame(
                    frame_id=provider.frame_counter,
                    timestamp=provider.frame_counter / provider.fps,
                    state="WAITING",
                    far_player_box=last.far_player_box,
                    far_player_world=last.far_player_world,
                    toss_ball_candidates=[],
                    active_ball_candidates=[],
                )
                provider.telemetry_history.append(telemetry)
            else:
                frame = cv2.resize(orig_frame, (960, 540), interpolation=cv2.INTER_LINEAR)
                telemetry = provider.process_frame(frame, orig_frame=orig_frame)

            frames_processed += 1
            snaps.append(FrameSnap(
                frame_id=telemetry.frame_id,
                timestamp=telemetry.timestamp,
                recorded_state=state,
                far_serve_score=telemetry.far_serve_score or 0.0,
                far_player_box=telemetry.far_player_box,
                far_player_world=telemetry.far_player_world,
                active_ball_candidates=list(telemetry.active_ball_candidates or []),
            ))

            new_state = engine.evaluate_transitions(provider.telemetry_history, state)
            if new_state != state:
                provider.update_state(new_state)
                state = new_state

            if frames_processed % 3000 == 0:
                sys.stdout.write(f"\r  … {frames_processed} frames  (t={telemetry.timestamp:.0f}s)")
                sys.stdout.flush()

    cap.release()
    print(f"\n[RECORD] {len(snaps)} frames captured.")

    with open(cache_path, "wb") as f:
        pickle.dump(snaps, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[RECORD] Cache saved → {cache_path}")
    return snaps


# ---------------------------------------------------------------------------
# PHASE 2 — offline replay
# ---------------------------------------------------------------------------

def _make_tf(snap: FrameSnap) -> TelemetryFrame:
    return TelemetryFrame(
        frame_id=snap.frame_id,
        timestamp=snap.timestamp,
        state=snap.recorded_state,
        far_player_box=snap.far_player_box,
        far_player_world=snap.far_player_world,
        far_serve_score=snap.far_serve_score,
        active_ball_candidates=list(snap.active_ball_candidates),
        near_player_box=None,
        near_player_world=None,
        toss_ball_candidates=[],
    )


def replay(snaps: List[FrameSnap], fps: float, cfg: SignalPriorityConfig,
           out_ratio: float, ready_half: float) -> List[Tuple[float, float]]:
    """Replay cached snaps against a given parameter set.

    TRACE-CONFIRM is bypassed: the recording engine never entered ACTIVE
    (threshold=1.01), so active_ball_candidates are empty in all snaps and
    trace confirmation would never fire.  Every ARMED→ACTIVE transition is
    committed immediately, evaluating ARMED detection quality independently
    of ball tracking.
    """
    engine = FarSideTransitionEngine(fps=fps, cfg=cfg)
    engine.ARMED_OUT_RATIO_THRESHOLD = out_ratio
    engine.READY_MIN_DIST_FT         = -ready_half
    engine.READY_MAX_DIST_FT         =  ready_half

    history   = deque(maxlen=128)
    state     = "WAITING"
    segments: List[Tuple[float, float]] = []
    seg_start = 0.0
    v_dur     = snaps[-1].timestamp + 1.0

    with contextlib.redirect_stdout(io.StringIO()):
        for snap in snaps:
            tf = _make_tf(snap)
            history.append(tf)
            new_state = engine.evaluate_transitions(history, state)

            if new_state != state:
                if new_state == "ACTIVE":
                    seg_start = snap.timestamp
                elif state == "ACTIVE":
                    end_t = (engine.last_transition_time
                             if engine.last_transition_time is not None
                             else snap.timestamp)
                    segments.append((seg_start, min(end_t + 1.0, v_dur)))
            state = new_state

        if state == "ACTIVE":
            segments.append((seg_start, v_dur))

    return segments


# ---------------------------------------------------------------------------
# PHASE 3 — evaluation
# ---------------------------------------------------------------------------

def evaluate(segments: List[Tuple[float, float]],
             gt_rallies: List[Dict],
             pre_window_s: float = 5.0,
             post_window_s: float = 3.0) -> Dict:
    """
    A detection hits a GT rally if it overlaps
    [gt_start - pre_window_s, gt_end + post_window_s].
    Greedy one-to-one matching (each GT matched at most once).
    """
    windows = [(r["start_s"] - pre_window_s, r["end_s"] + post_window_s)
               for r in gt_rallies]
    matched = set()
    tp = fp = 0

    for seg_s, seg_e in segments:
        hit = None
        for i, (w_s, w_e) in enumerate(windows):
            if i in matched:
                continue
            if seg_s < w_e and seg_e > w_s:
                if hit is None or abs(seg_s - gt_rallies[i]["start_s"]) < abs(
                        seg_s - gt_rallies[hit]["start_s"]):
                    hit = i
        if hit is not None:
            tp += 1
            matched.add(hit)
        else:
            fp += 1

    fn   = len(gt_rallies) - len(matched)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

GRID = {
    "score_threshold": [0.40, 0.45, 0.50, 0.55, 0.60, 0.65],
    "out_ratio":       [0.25, 0.50, 0.75],
    "trace_down":      [15.0, 25.0],        # bypassed but kept for future use
    "trace_horiz":     [10.0, 20.0],        # bypassed but kept for future use
    "ready_half":      [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?",
                    default="/Volumes/Anya/Data/23/snippet.mp4")
    ap.add_argument("--start-frame", type=int, default=0,
                    help="First frame to process (default 0)")
    ap.add_argument("--end-frame",   type=int, default=0,
                    help="Last frame to process exclusive (default 0 = end of video)")
    ap.add_argument("--top",      type=int, default=30,
                    help="Show top-N combinations (default 30)")
    ap.add_argument("--rerecord", action="store_true",
                    help="Ignore existing cache and re-run recording")
    ap.add_argument("--pre-window", type=float, default=5.0,
                    help="Seconds before GT start that count as a hit (default 5)")
    ap.add_argument("--post-window", type=float, default=3.0,
                    help="Seconds after GT end that still count as a hit (default 3)")
    ap.add_argument("--csv-out",
                    help="Write full results table to this CSV path")
    args = ap.parse_args()

    video_path  = args.video
    start_frame = args.start_frame
    end_frame   = args.end_frame
    video_dir   = os.path.dirname(os.path.abspath(video_path))
    stem        = os.path.splitext(os.path.basename(video_path))[0]
    frame_tag   = (f"_{start_frame}_{end_frame}"
                   if start_frame or end_frame else "")
    cache_path  = os.path.join(video_dir, f"{stem}_far_tune_cache{frame_tag}.pkl")

    # Ground truth
    gt_rallies = _load_gt(video_path)
    print(f"[GT] {len(gt_rallies)} far-side rallies")
    for i, r in enumerate(gt_rallies):
        print(f"  {i+1:2d}. {r['start_s']:7.2f}s – {r['end_s']:7.2f}s")

    # Phase 1
    if args.rerecord or not os.path.exists(cache_path):
        snaps = record(video_path, cache_path,
                       start_frame=start_frame, end_frame=end_frame)
    else:
        print(f"[RECORD] Loading cache from {cache_path}")
        with open(cache_path, "rb") as f:
            snaps = pickle.load(f)
        print(f"[RECORD] {len(snaps)} frames loaded")

    # Summarise recording coverage
    armed_frames  = sum(1 for s in snaps if s.recorded_state == "ARMED")
    active_frames = sum(1 for s in snaps if s.recorded_state == "ACTIVE")
    scored_frames = sum(1 for s in snaps if s.far_serve_score > 0)
    print(f"[RECORD] coverage — ARMED:{armed_frames}  ACTIVE:{active_frames}  "
          f"scored(>0):{scored_frames} / {len(snaps)} total frames")

    # Phase 2 — grid search
    combos = list(product(
        GRID["score_threshold"],
        GRID["out_ratio"],
        GRID["trace_down"],
        GRID["trace_horiz"],
        GRID["ready_half"],
    ))
    fps = _probe_fps(video_path)
    print(f"\n[GRID] {len(combos)} combinations … (fps={fps:.4f})", flush=True)

    results = []
    for i, (thresh, orat, tdown, thoriz, ready_h) in enumerate(combos):
        cfg = SignalPriorityConfig(
            far_serve_score_threshold=thresh,
            trace_downward_px_s=tdown,
            trace_horizontal_px_s=thoriz,
        )
        segs  = replay(snaps, fps, cfg, out_ratio=orat, ready_half=ready_h)
        stats = evaluate(segs, gt_rallies, pre_window_s=args.pre_window,
                         post_window_s=args.post_window)
        results.append({
            "thresh":      thresh,
            "out_ratio":   orat,
            "trace_down":  tdown,
            "trace_horiz": thoriz,
            "ready_half":  ready_h,
            "n_segs":      len(segs),
            **stats,
        })
        if (i + 1) % 100 == 0:
            sys.stdout.write(f"\r  … {i+1}/{len(combos)} done")
            sys.stdout.flush()

    print(f"\r  {len(combos)}/{len(combos)} done         ")

    results.sort(key=lambda r: (-r["f1"], -r["recall"], r["fp"]))

    # Phase 3 — report
    hdr = (f"{'thresh':>7} {'out':>5} {'↓px':>6} {'↔px':>6} {'band':>5} "
           f"{'segs':>5} {'TP':>3} {'FP':>3} {'FN':>3} "
           f"{'prec':>5} {'rec':>5} {'F1':>5}")
    sep = "-" * len(hdr)
    print(f"\n{hdr}\n{sep}")
    for r in results[:args.top]:
        print(f"{r['thresh']:>7.2f} {r['out_ratio']:>5.2f} {r['trace_down']:>6.0f} "
              f"{r['trace_horiz']:>6.0f} {r['ready_half']:>5.1f} "
              f"{r['n_segs']:>5} {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} "
              f"{r['precision']:>5.2f} {r['recall']:>5.2f} {r['f1']:>5.2f}")

    best = results[0]
    print(f"\n[BEST]")
    print(f"  far_serve_score_threshold = {best['thresh']}")
    print(f"  ARMED_OUT_RATIO           = {best['out_ratio']}")
    print(f"  trace_downward_px_s       = {best['trace_down']}")
    print(f"  trace_horizontal_px_s     = {best['trace_horiz']}")
    print(f"  ready band                = ±{best['ready_half']} ft")
    print(f"  → F1={best['f1']:.3f}  P={best['precision']:.2f}  "
          f"R={best['recall']:.2f}  TP={best['tp']}  FP={best['fp']}  FN={best['fn']}")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"[CSV] Full results → {args.csv_out}")


if __name__ == "__main__":
    main()
