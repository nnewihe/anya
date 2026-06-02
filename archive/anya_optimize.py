"""
anya_optimize.py
================
Evaluation & optimization script for AnyaSystem active-phase parameters.

Usage:
    # Evaluate current config only
    python anya_optimize.py --mode eval

    # Run grid search over 4 key parameters (slow — runs full YOLO inference)
    python anya_optimize.py --mode optimize

    # Evaluate a specific override
    python anya_optimize.py --mode eval --decay-ball-dead 0.10 --decay-ball-rolling 0.25
"""

import os
import sys
import json
import time
import argparse
import itertools
from typing import Optional

import cv2
import numpy as np

# ---- resolve paths ---------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
# AnyaSystem resolves model weights relative to cwd; they live under
# /Users/tennis/Documents/Code/Laptop/weights/
_WEIGHTS_CWD = os.path.abspath(os.path.join(_HERE, "../.."))  # Laptop/
os.chdir(_WEIGHTS_CWD)

sys.path.insert(0, _HERE)

from anya_vision_core import AnyaSystem, Config

# =============================================================================
# Constants
# =============================================================================

DATA_ROOT = "/Volumes/Anya/Data"

# Scoring weights
EARLY_PENALTY_PER_SEC   = 10.0   # never cut active action
GOLDEN_ZONE_REWARD      = 10.0   # 0–2 s after GT end
GOLDEN_ZONE_MAX_SEC     = 2.0
OVERRUN_PENALTY_PER_SEC = 1.0    # progressive dead-time cost past 2 s
MISS_PENALTY            = -20.0  # rally not detected at all

START_MATCH_WINDOW_SEC  = 3.0    # max offset for GT–prediction start alignment

RESULTS_SAVE_PATH = os.path.join(_HERE, "anya_optimize_results.json")


# =============================================================================
# Scoring
# =============================================================================

def score_rally(pred_end_sec: float, gt_end_sec: float) -> float:
    """
    Score a single predicted rally end vs ground truth end.

    delta > 0 = system ran too long (overrun)
    delta < 0 = system cut early (very bad)
    """
    delta = pred_end_sec - gt_end_sec
    if delta < 0:
        return EARLY_PENALTY_PER_SEC * delta          # negative (heavy penalty)
    elif delta <= GOLDEN_ZONE_MAX_SEC:
        return GOLDEN_ZONE_REWARD                     # +10 (perfect)
    else:
        overrun = delta - GOLDEN_ZONE_MAX_SEC
        return GOLDEN_ZONE_REWARD - OVERRUN_PENALTY_PER_SEC * overrun


# =============================================================================
# Single-folder evaluation
# =============================================================================

def evaluate_folder(folder_path: str, verbose: bool = True) -> Optional[dict]:
    """
    Run AnyaSystem on snippet.mp4, compare to ground_truth.json (near serves only).
    Returns a dict with per-rally deltas, scores, and folder totals.
    Returns None if the folder cannot be processed.
    """
    video_path = os.path.join(folder_path, "snippet.mp4")
    gt_path    = os.path.join(folder_path, "ground_truth.json")

    if not os.path.isfile(video_path):
        if verbose:
            print(f"  [SKIP] No snippet.mp4 in {folder_path}")
        return None
    if not os.path.isfile(gt_path):
        if verbose:
            print(f"  [SKIP] No ground_truth.json in {folder_path}")
        return None

    with open(gt_path) as fh:
        gt_data = json.load(fh)

    near_rallies = [r for r in gt_data.get("rallies", []) if r.get("serve") == "near"]
    if not near_rallies:
        if verbose:
            print(f"  [SKIP] No near-serve rallies in {folder_path}")
        return None

    # ---- initialise system (reads cached court corners) --------------------
    try:
        system = AnyaSystem(video_path)
    except Exception as exc:
        if verbose:
            print(f"  [ERROR] AnyaSystem init failed: {exc}")
        return None

    fps = system.fps

    # ---- headless frame loop -----------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose:
            print(f"  [ERROR] Cannot open video: {video_path}")
        return None

    t0 = time.time()
    frames_processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            system.process_frame(frame)
            frames_processed += 1
    except Exception as exc:
        if verbose:
            print(f"  [ERROR] Frame processing failed at frame {frames_processed}: {exc}")
    finally:
        cap.release()

    elapsed = time.time() - t0
    system.finalize()

    # ---- match predicted segments to GT ------------------------------------
    rally_results = []
    folder_score  = 0.0

    for gt in near_rallies:
        gt_start_sec = gt["start"] / fps
        gt_end_sec   = gt["end"]   / fps

        best_match    = None
        best_dist     = float("inf")

        for (pred_start_f, pred_end_f) in system.active_segments:
            pred_start_sec = pred_start_f / fps
            pred_end_sec   = pred_end_f   / fps
            dist = abs(pred_start_sec - gt_start_sec)
            if dist < START_MATCH_WINDOW_SEC and dist < best_dist:
                best_dist  = dist
                best_match = (pred_start_sec, pred_end_sec)

        if best_match is None:
            s     = MISS_PENALTY
            delta = None
            tag   = "MISS"
        else:
            delta = best_match[1] - gt_end_sec
            s     = score_rally(best_match[1], gt_end_sec)
            if delta < 0:
                tag = f"EARLY {delta:+.2f}s"
            elif delta <= GOLDEN_ZONE_MAX_SEC:
                tag = f"GOLDEN {delta:+.2f}s"
            else:
                tag = f"OVERRUN {delta:+.2f}s"

        folder_score += s
        rally_results.append({
            "gt_start": round(gt_start_sec, 2),
            "gt_end":   round(gt_end_sec,   2),
            "pred_end": round(best_match[1], 2) if best_match else None,
            "delta":    round(delta, 2) if delta is not None else None,
            "score":    round(s, 2),
            "tag":      tag,
        })

        if verbose:
            gt_str = f"{gt_start_sec:.1f}s–{gt_end_sec:.1f}s"
            pred_str = f"pred_end={best_match[1]:.1f}s" if best_match else "no match"
            print(f"    Rally {gt_str} | {pred_str} | {tag} | score={s:.1f}")

    matched = sum(1 for r in rally_results if r["pred_end"] is not None)

    if verbose:
        print(f"  → Folder score={folder_score:.1f}  "
              f"matched={matched}/{len(near_rallies)}  "
              f"({elapsed:.0f}s elapsed)\n")

    return {
        "folder_score": round(folder_score, 2),
        "near_rallies": len(near_rallies),
        "matched":      matched,
        "rallies":      rally_results,
        "elapsed_sec":  round(elapsed, 1),
    }


# =============================================================================
# Full-dataset evaluation
# =============================================================================

def evaluate_all(verbose: bool = True) -> dict:
    """
    Run evaluate_folder() on every numeric folder under DATA_ROOT.
    Returns aggregate metrics and per-folder results.
    """
    folders = sorted(
        f for f in os.listdir(DATA_ROOT)
        if f.isdigit() and os.path.isdir(os.path.join(DATA_ROOT, f))
    )

    all_results  = {}
    total_score  = 0.0
    total_rallies = 0

    for folder_name in folders:
        folder_path = os.path.join(DATA_ROOT, folder_name)
        if verbose:
            print(f"\n{'='*60}")
            print(f"[Folder {folder_name}] {folder_path}")
            print(f"{'='*60}")

        result = evaluate_folder(folder_path, verbose=verbose)
        if result is None:
            continue

        all_results[folder_name]  = result
        total_score               += result["folder_score"]
        total_rallies             += result["near_rallies"]

    avg_score = total_score / total_rallies if total_rallies > 0 else 0.0

    return {
        "total_score":   round(total_score, 2),
        "avg_score":     round(avg_score, 4),
        "total_rallies": total_rallies,
        "folders":       all_results,
    }


# =============================================================================
# Optimization
# =============================================================================

PARAM_GRID = {
    "ENERGY_DECAY_BALL_DEAD":              [0.08, 0.15, 0.25],
    "ENERGY_DECAY_BALL_ROLLING":           [0.15, 0.30, 0.45],
    "ABSOLUTE_BALL_LOST_TIMEOUT_IDLE":     [4.0,  6.0,  9.0],
    "ENERGY_DECAY_PLAYER_WALK":            [0.10, 0.20, 0.35],
}


def apply_config(overrides: dict):
    for k, v in overrides.items():
        setattr(Config, k, v)


def restore_config(originals: dict):
    for k, v in originals.items():
        setattr(Config, k, v)


def run_grid_search(save_path: str = RESULTS_SAVE_PATH, verbose: bool = True) -> dict:
    """
    Evaluate all combinations in PARAM_GRID and return the best config.
    Results are incrementally saved to `save_path` so the run can be
    interrupted and analysed at any time.
    """
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    total  = len(combos)

    originals = {k: getattr(Config, k) for k in keys}

    # Load existing results if resuming
    all_search_results = []
    evaluated_combos   = set()
    if os.path.isfile(save_path):
        try:
            with open(save_path) as fh:
                saved = json.load(fh)
            all_search_results = saved.get("search_results", [])
            for r in all_search_results:
                key = tuple(r["params"][k] for k in keys)
                evaluated_combos.add(key)
            print(f"[RESUME] Loaded {len(all_search_results)} prior results from {save_path}")
        except Exception as exc:
            print(f"[WARN] Could not load saved results: {exc}")

    print(f"\n[OPTIMIZE] Grid search over {total} combinations "
          f"({len(evaluated_combos)} already done).\n")

    for idx, combo in enumerate(combos, 1):
        if combo in evaluated_combos:
            print(f"[{idx}/{total}] Skipping already-evaluated combo.")
            continue

        overrides = dict(zip(keys, combo))
        print(f"\n[{idx}/{total}] Evaluating: {overrides}")

        apply_config(overrides)
        try:
            metrics = evaluate_all(verbose=verbose)
        finally:
            restore_config(originals)

        all_search_results.append({
            "params":       overrides,
            "total_score":  metrics["total_score"],
            "avg_score":    metrics["avg_score"],
            "total_rallies": metrics["total_rallies"],
        })

        # Save incrementally
        try:
            with open(save_path, "w") as fh:
                json.dump({"search_results": all_search_results}, fh, indent=2)
        except Exception as exc:
            print(f"[WARN] Could not save results: {exc}")

    # ---- Find best --------------------------------------------------------
    if all_search_results:
        best = max(all_search_results, key=lambda r: r["avg_score"])
        print("\n" + "="*60)
        print("GRID SEARCH COMPLETE")
        print("="*60)
        print(f"Best avg_score : {best['avg_score']:.4f}")
        print(f"Best params    : {best['params']}")
        return best
    return {}


# =============================================================================
# CLI
# =============================================================================

def print_summary(metrics: dict):
    print("\n" + "="*60)
    print("ANYA PERFORMANCE REPORT")
    print("="*60)
    print(f"Total rallies evaluated : {metrics['total_rallies']}")
    print(f"Total score             : {metrics['total_score']:.2f}")
    print(f"Avg score per rally     : {metrics['avg_score']:.4f}")
    print("\nPer-folder breakdown:")
    for name, r in metrics.get("folders", {}).items():
        matched_str = f"{r['matched']}/{r['near_rallies']}"
        print(f"  Folder {name:>3}  score={r['folder_score']:>7.1f}  "
              f"matched={matched_str}  ({r['elapsed_sec']:.0f}s)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Anya Vision parameter evaluation & optimization."
    )
    parser.add_argument("--mode", choices=["eval", "optimize"], default="eval",
                        help="eval: baseline evaluation; optimize: grid search")
    parser.add_argument("--decay-ball-dead",     type=float, default=None)
    parser.add_argument("--decay-ball-rolling",  type=float, default=None)
    parser.add_argument("--timeout-idle",        type=float, default=None)
    parser.add_argument("--decay-player-walk",   type=float, default=None)
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-rally verbose output")
    args = parser.parse_args()

    verbose = not args.quiet

    # Apply any manual overrides
    overrides = {}
    if args.decay_ball_dead    is not None:
        overrides["ENERGY_DECAY_BALL_DEAD"]            = args.decay_ball_dead
    if args.decay_ball_rolling is not None:
        overrides["ENERGY_DECAY_BALL_ROLLING"]         = args.decay_ball_rolling
    if args.timeout_idle       is not None:
        overrides["ABSOLUTE_BALL_LOST_TIMEOUT_IDLE"]   = args.timeout_idle
    if args.decay_player_walk  is not None:
        overrides["ENERGY_DECAY_PLAYER_WALK"]          = args.decay_player_walk

    if overrides:
        print(f"[CONFIG] Applying overrides: {overrides}")
        apply_config(overrides)

    print("\n[CONFIG] Active parameters being evaluated:")
    for k in ["ENERGY_DECAY_BALL_DEAD", "ENERGY_DECAY_BALL_ROLLING",
              "ABSOLUTE_BALL_LOST_TIMEOUT_IDLE", "ENERGY_DECAY_PLAYER_WALK"]:
        print(f"  {k} = {getattr(Config, k)}")

    if args.mode == "eval":
        metrics = evaluate_all(verbose=verbose)
        print_summary(metrics)

        out_path = os.path.join(_HERE, "anya_eval_results.json")
        with open(out_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"\n[SAVED] Results → {out_path}")

    elif args.mode == "optimize":
        best = run_grid_search(verbose=verbose)
        print(f"\n[RESULT] Best configuration found:")
        print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
