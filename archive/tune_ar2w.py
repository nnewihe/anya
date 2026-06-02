"""
tune_ar2w.py
============
Grid-search optimizer for ACTIVE_RALLY → WAITING transition parameters.

Loads an extended telemetry CSV (produced by anya_hmm.py) and a ground-truth
JSON of true ACTIVE intervals, then replays the HMM belief update for every
combination of parameters in PARAM_GRID and ranks results by ACTIVE recall
(primary) and F1 (secondary).

Usage:
    python tune_ar2w.py \\
        --telemetry   path/to/hmm_telemetry.csv \\
        --ground-truth path/to/ground_truth.json \\
        --fps 30 \\
        [--top-n 10] \\
        [--out tune_results.csv]

Ground-truth JSON format (frame numbers, one entry per rally):
    {
        "rallies": [
            {"start": 760, "end": 1326, "serve": "near"},
            ...
        ]
    }

Telemetry CSV must include the signal columns added in anya_hmm.py v0.3+:
    sig_ball_present, sig_ball_vel_raw, sig_ball_not_near,
    sig_vel_cov_raw, sig_retreat_speed, sig_bbox_pct_raw, serve_grace_active
"""

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ── Constants mirrored from anya_hmm.py (kept in sync manually) ──────────────
# These are pure values with no runtime dependencies; duplicated here so the
# tuner can run without importing the full anya_hmm module (which pulls in
# cv2 and anya_base at module load time).

S_WAITING      = 0
S_READY_ARMED  = 1
S_ACTIVE_RALLY = 2

# HMM transition priors (fixed; only P_AR2W is grid-searched)
P_W2RA = 0.010
P_RA2W = 0.008

# Fixed thresholds (not in the grid)
BALL_VEL_FAST_THRESH         = 25.0
RETREAT_STRONG_FT_S          = 3.5
BBOX_ASPECT_RATIO_CHANGE_PCTS = 5.0
PLAYER_VEL_STILL_THRESH      = 3.0
PLAYER_VEL_ACTIVE_THRESH     = 6.0
SERVE_GRACE_SEC              = 3.0

# Emission tables [P(obs|WAITING), P(obs|READY_ARMED), P(obs|ACTIVE_RALLY)]
_EMIT_FAST_BALL     = np.array([0.04, 0.01, 0.97])
_EMIT_SLOW_BALL     = np.array([0.55, 0.20, 0.30])
_EMIT_BALL_NEAR     = np.array([0.60, 0.75, 0.20])
_EMIT_BALL_MISS     = np.array([0.70, 0.50, 0.20])
_EMIT_NO_BALL       = np.array([0.35, 0.33, 0.50])
_EMIT_PLAYER_ACTIVE = np.array([0.05, 0.00, 0.95])
_EMIT_PLAYER_WALK   = np.array([0.70, 0.00, 0.30])
_EMIT_PLAYER_ARMED  = np.array([0.28, 0.95, 0.01])


# ── Parameter grid ────────────────────────────────────────────────────────────

PARAM_GRID = {
    "P_AR2W":                        [0.001, 0.002, 0.003, 0.005],
    "BALL_MISSING_TAU_SEC":          [3.0, 4.0, 6.0, 8.0],
    "BALL_VEL_SLOW_THRESH":          [4.0, 6.0, 8.0],
    "BALL_NEAR_PLAYER_DURATION_SEC": [1.0, 1.5, 3.0, 5.0],
    "VEL_VARIANCE_LOW_THRESHOLD":    [0.05, 0.10, 0.15],
    "RETREAT_MIN_SPEED_FT_S":        [1.5, 2.5, 3.5],
}
# 4×4×3×4×3×3 = 1728 combinations


@dataclass
class ParamSet:
    P_AR2W: float
    BALL_MISSING_TAU_SEC: float
    BALL_VEL_SLOW_THRESH: float
    BALL_NEAR_PLAYER_DURATION_SEC: float
    VEL_VARIANCE_LOW_THRESHOLD: float
    RETREAT_MIN_SPEED_FT_S: float


@dataclass
class Score:
    recall: float
    precision: float
    f1: float
    false_exits: int
    params: ParamSet

    def sort_key(self):
        return (self.recall, self.f1, -self.false_exits)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_telemetry(path: str) -> dict:
    """
    Returns a dict of numpy arrays keyed by column name.
    Raises ValueError if required signal columns are absent (old CSV format).
    """
    required_signal_cols = {
        "sig_ball_present", "sig_ball_vel_raw", "sig_ball_not_near",
        "sig_vel_cov_raw", "sig_retreat_speed", "sig_bbox_pct_raw",
        "serve_grace_active",
    }

    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = required_signal_cols - fieldnames
        if missing:
            raise ValueError(
                f"Telemetry CSV is missing signal columns: {sorted(missing)}\n"
                "Re-run anya_hmm.py to regenerate the CSV with expanded columns."
            )
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError("Telemetry CSV is empty.")

    def col(name, dtype=float):
        return np.array([dtype(r[name]) for r in rows], dtype=dtype if dtype != float else np.float64)

    return {
        "timestamp":         col("timestamp"),
        "p_ra2ar":           col("p_ra2ar"),
        "sig_ball_present":  col("sig_ball_present"),
        "sig_ball_vel_raw":  col("sig_ball_vel_raw"),
        "sig_ball_not_near": col("sig_ball_not_near"),
        "sig_vel_cov_raw":   col("sig_vel_cov_raw"),
        "sig_retreat_speed": col("sig_retreat_speed"),
        "sig_bbox_pct_raw":  col("sig_bbox_pct_raw"),
        "serve_grace_active": col("serve_grace_active"),
    }


def load_ground_truth(path: str, fps: float) -> List[Tuple[float, float]]:
    """
    JSON with {"rallies": [{"start": <frame>, "end": <frame>, "serve": "near"|"far"}, ...]}.
    Frame numbers are converted to seconds using fps.
    """
    with open(path) as f:
        data = json.load(f)
    rallies = data.get("rallies", [])
    if not rallies:
        raise ValueError("Ground-truth JSON has no rallies.")
    return [(r["start"] / fps, r["end"] / fps) for r in rallies]


# ── Time-series pre-computation ───────────────────────────────────────────────

def precompute_histories(df: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-frame time-series that depend on history but NOT on thresholds:

    time_since_ball[i]    — seconds since last frame with a detected ball
    ball_near_duration[i] — duration of the current ball-near-player run (0 if not near)

    These are computed once; threshold comparisons happen inside build_emissions().
    """
    N = len(df["timestamp"])
    timestamps         = df["timestamp"]
    sig_ball_present   = df["sig_ball_present"]
    sig_ball_not_near  = df["sig_ball_not_near"]

    # time_since_ball: reset whenever sig_ball_present >= 1.0
    time_since_ball = np.zeros(N, dtype=np.float64)
    last_ball_t = timestamps[0] - 999.0
    for i in range(N):
        if sig_ball_present[i] >= 1.0:
            last_ball_t = timestamps[i]
        time_since_ball[i] = timestamps[i] - last_ball_t

    # ball_near_duration: accumulate while sig_ball_not_near < 1.0 (ball overlaps player box)
    ball_near_duration = np.zeros(N, dtype=np.float64)
    near_run_start = -999.0
    for i in range(N):
        if sig_ball_not_near[i] < 1.0:
            if near_run_start < 0:
                near_run_start = timestamps[i]
            ball_near_duration[i] = timestamps[i] - near_run_start
        else:
            near_run_start = -999.0
            ball_near_duration[i] = 0.0

    return time_since_ball, ball_near_duration


# ── Emission reconstruction ───────────────────────────────────────────────────

def build_emissions(df: dict, params: ParamSet,
                    time_since_ball: np.ndarray,
                    ball_near_duration: np.ndarray) -> np.ndarray:
    """
    Reconstruct the 3-vector emission P(obs | state) for every frame using the
    pre-computed history arrays and a candidate ParamSet.

    Returns ndarray[N, 3] — one row per frame, columns = [W, RA, AR].
    """
    N = len(df["timestamp"])
    emissions = np.empty((N, 3), dtype=np.float64)

    sig_ball_present   = df["sig_ball_present"]
    sig_ball_vel_raw   = df["sig_ball_vel_raw"]
    sig_ball_not_near  = df["sig_ball_not_near"]
    sig_vel_cov_raw    = df["sig_vel_cov_raw"]
    sig_retreat_speed  = df["sig_retreat_speed"]
    sig_bbox_pct_raw   = df["sig_bbox_pct_raw"]

    fast_thresh = BALL_VEL_FAST_THRESH
    vel_range   = max(fast_thresh - params.BALL_VEL_SLOW_THRESH, 1.0)

    for i in range(N):
        # ── Ball channel ──────────────────────────────────────────────────────
        tsb = time_since_ball[i]
        bnd = ball_near_duration[i]
        bvr = sig_ball_vel_raw[i]
        bnn = sig_ball_not_near[i]
        bpr = sig_ball_present[i]

        if tsb > params.BALL_MISSING_TAU_SEC:
            miss_alpha = float(np.clip(
                (tsb - params.BALL_MISSING_TAU_SEC) / params.BALL_MISSING_TAU_SEC, 0.0, 1.0
            ))
            ball_emit = np.array([
                min(0.80, 0.65 + 0.15 * miss_alpha),
                min(0.55, 0.45 + 0.10 * miss_alpha),
                max(0.10, 0.40 - 0.30 * miss_alpha),
            ])
        elif bpr < 1.0:
            ball_emit = _EMIT_NO_BALL.copy()
        elif bnd > params.BALL_NEAR_PLAYER_DURATION_SEC:
            ball_emit = np.array([0.88, 0.85, 0.05])
        elif bnn < 1.0:
            ball_emit = _EMIT_BALL_NEAR.copy()
        else:
            speed_alpha = float(np.clip((bvr - params.BALL_VEL_SLOW_THRESH) / vel_range, 0.0, 1.0))
            ball_emit   = _EMIT_SLOW_BALL + speed_alpha * (_EMIT_FAST_BALL - _EMIT_SLOW_BALL)

        # ── Player channel ────────────────────────────────────────────────────
        vel_cov      = sig_vel_cov_raw[i]
        retreat_spd  = sig_retreat_speed[i]
        bbox_pct     = sig_bbox_pct_raw[i]

        if vel_cov > 0.30:                                   # VEL_VARIANCE_HIGH_THRESHOLD fixed
            emit_base = _EMIT_PLAYER_ACTIVE.copy()
        elif vel_cov < params.VEL_VARIANCE_LOW_THRESHOLD:
            emit_base = _EMIT_PLAYER_WALK.copy()
        else:
            v_range   = max(PLAYER_VEL_ACTIVE_THRESH - PLAYER_VEL_STILL_THRESH, 1.0)
            vel_alpha = float(np.clip(
                (vel_cov - PLAYER_VEL_STILL_THRESH) / v_range, 0.0, 1.0
            ))
            emit_base = _EMIT_PLAYER_WALK + vel_alpha * (_EMIT_PLAYER_ACTIVE - _EMIT_PLAYER_WALK)

        if bbox_pct > BBOX_ASPECT_RATIO_CHANGE_PCTS:
            emit_base = emit_base * np.array([0.8, 1.0, 1.2])
            s = emit_base.sum()
            if s > 1e-10:
                emit_base /= s

        # Retreat blend
        if retreat_spd > params.RETREAT_MIN_SPEED_FT_S:
            retreat_alpha = float(np.clip(
                (retreat_spd - params.RETREAT_MIN_SPEED_FT_S)
                / max(RETREAT_STRONG_FT_S - params.RETREAT_MIN_SPEED_FT_S, 0.01),
                0.0, 1.0,
            ))
            _EMIT_RETREAT = np.array([0.92, 0.03, 0.05])
            emit_base = (1.0 - retreat_alpha) * emit_base + retreat_alpha * _EMIT_RETREAT
            s = emit_base.sum()
            if s > 1e-10:
                emit_base /= s

        # ── Combine ───────────────────────────────────────────────────────────
        combined = ball_emit * emit_base
        s = combined.sum()
        emissions[i] = combined / s if s > 1e-10 else np.full(3, 1.0 / 3.0)

    return emissions


# ── HMM forward pass ──────────────────────────────────────────────────────────

def _build_A(P_AR2W: float) -> np.ndarray:
    p_ra_stay = 1.0 - P_RA2W
    p_ra2ar   = 0.0   # serve trigger is handled separately
    return np.array([
        [1.0 - P_W2RA,  P_W2RA,    0.0         ],
        [P_RA2W,        p_ra_stay, p_ra2ar     ],
        [P_AR2W,        0.0,       1.0 - P_AR2W],
    ], dtype=np.float64)


def run_hmm(df: dict, emissions: np.ndarray, params: ParamSet) -> np.ndarray:
    """Forward-pass HMM replay. Returns per-frame state indices (0/1/2)."""
    N          = len(df["timestamp"])
    p_ra2ar    = df["p_ra2ar"]
    grace_flag = df["serve_grace_active"]
    A          = _build_A(params.P_AR2W)
    belief     = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    states     = np.empty(N, dtype=np.int32)
    grace_until = -999.0

    for i in range(N):
        t = df["timestamp"][i]

        if p_ra2ar[i] >= 1.0:
            belief      = np.array([0.0, 0.0, 1.0])
            grace_until = t + SERVE_GRACE_SEC
        elif grace_flag[i] or t < grace_until:
            belief = np.array([0.0, 0.0, 1.0])
        else:
            raw = emissions[i] * (A.T @ belief)
            s   = raw.sum()
            belief = raw / s if s > 1e-10 else np.full(3, 1.0 / 3.0)

        states[i] = int(np.argmax(belief))

    return states


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_states(states: np.ndarray, timestamps: np.ndarray,
                 ground_truth: List[Tuple[float, float]]) -> Tuple[float, float, float, int]:
    """
    Returns (recall, precision, f1, false_exits) for the ACTIVE class.
    false_exits = transitions AR→non-AR while ground truth says ACTIVE.
    """
    N = len(states)
    true_active = np.zeros(N, dtype=bool)
    for start, end in ground_truth:
        true_active |= (timestamps >= start) & (timestamps <= end)

    pred_active = (states == S_ACTIVE_RALLY)
    tp = float((pred_active & true_active).sum())
    fp = float((pred_active & ~true_active).sum())
    fn = float((~pred_active & true_active).sum())

    recall    = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1        = 2.0 * precision * recall / (precision + recall + 1e-10)

    false_exits = 0
    for i in range(1, N):
        if states[i - 1] == S_ACTIVE_RALLY and states[i] != S_ACTIVE_RALLY and true_active[i]:
            false_exits += 1

    return recall, precision, f1, false_exits


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search(df: dict, ground_truth: List[Tuple[float, float]]) -> List[Score]:
    time_since_ball, ball_near_duration = precompute_histories(df)
    timestamps = df["timestamp"]

    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))

    print(f"Searching {len(combos)} parameter combinations…")

    results = []
    for i, combo in enumerate(combos):
        if i % 200 == 0:
            print(f"  {i}/{len(combos)}", end="\r", flush=True)

        params = ParamSet(**dict(zip(keys, combo)))
        emissions = build_emissions(df, params, time_since_ball, ball_near_duration)
        states    = run_hmm(df, emissions, params)
        recall, precision, f1, false_exits = score_states(states, timestamps, ground_truth)
        results.append(Score(recall, precision, f1, false_exits, params))

    print(f"  {len(combos)}/{len(combos)} done.        ")
    return results


# ── Output ────────────────────────────────────────────────────────────────────

def print_top(results: List[Score], n: int):
    ranked = sorted(results, key=lambda s: s.sort_key(), reverse=True)
    header = (
        f"{'Rank':>4}  {'Recall':>6}  {'Prec':>6}  {'F1':>6}  {'FalseExit':>9}  "
        f"{'P_AR2W':>7}  {'MissTau':>7}  {'SlowV':>5}  "
        f"{'NearDur':>7}  {'VelCOV':>6}  {'RetSpd':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for rank, s in enumerate(ranked[:n], 1):
        p = s.params
        print(
            f"{rank:>4}  {s.recall:>6.3f}  {s.precision:>6.3f}  {s.f1:>6.3f}  "
            f"{s.false_exits:>9d}  "
            f"{p.P_AR2W:>7.4f}  {p.BALL_MISSING_TAU_SEC:>7.1f}  "
            f"{p.BALL_VEL_SLOW_THRESH:>5.1f}  "
            f"{p.BALL_NEAR_PLAYER_DURATION_SEC:>7.1f}  "
            f"{p.VEL_VARIANCE_LOW_THRESHOLD:>6.2f}  "
            f"{p.RETREAT_MIN_SPEED_FT_S:>6.1f}"
        )

    print(f"\nCurrent baseline params are:")
    print(f"  P_AR2W=0.002  BALL_MISSING_TAU_SEC=4.0  BALL_VEL_SLOW_THRESH=8.0")
    print(f"  BALL_NEAR_PLAYER_DURATION_SEC=1.5  VEL_VARIANCE_LOW_THRESHOLD=0.10  RETREAT_MIN_SPEED_FT_S=1.5")

    return ranked


def write_csv(results: List[Score], path: str):
    ranked = sorted(results, key=lambda s: s.sort_key(), reverse=True)
    fieldnames = [
        "rank", "recall", "precision", "f1", "false_exits",
        "P_AR2W", "BALL_MISSING_TAU_SEC", "BALL_VEL_SLOW_THRESH",
        "BALL_NEAR_PLAYER_DURATION_SEC", "VEL_VARIANCE_LOW_THRESHOLD",
        "RETREAT_MIN_SPEED_FT_S",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, s in enumerate(ranked, 1):
            p = s.params
            writer.writerow({
                "rank": rank,
                "recall": round(s.recall, 4),
                "precision": round(s.precision, 4),
                "f1": round(s.f1, 4),
                "false_exits": s.false_exits,
                "P_AR2W": p.P_AR2W,
                "BALL_MISSING_TAU_SEC": p.BALL_MISSING_TAU_SEC,
                "BALL_VEL_SLOW_THRESH": p.BALL_VEL_SLOW_THRESH,
                "BALL_NEAR_PLAYER_DURATION_SEC": p.BALL_NEAR_PLAYER_DURATION_SEC,
                "VEL_VARIANCE_LOW_THRESHOLD": p.VEL_VARIANCE_LOW_THRESHOLD,
                "RETREAT_MIN_SPEED_FT_S": p.RETREAT_MIN_SPEED_FT_S,
            })
    print(f"\nFull results written to: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tune AR→W parameters for anya_hmm.py")
    parser.add_argument("--telemetry",    required=True,       help="Path to *_hmm_telemetry.csv")
    parser.add_argument("--ground-truth", required=True,       help="Path to ground_truth.json")
    parser.add_argument("--fps",          type=float, default=30.0,
                        help="Video frame rate used to convert ground-truth frame numbers to seconds (default 30)")
    parser.add_argument("--top-n",        type=int, default=10, help="Number of top results to print")
    parser.add_argument("--out",          default=None,        help="Optional output CSV for all results")
    args = parser.parse_args()

    print(f"Loading telemetry:    {args.telemetry}")
    df = load_telemetry(args.telemetry)
    print(f"  {len(df['timestamp'])} frames  "
          f"({df['timestamp'][0]:.1f}s – {df['timestamp'][-1]:.1f}s)")

    print(f"Loading ground truth: {args.ground_truth}  (fps={args.fps})")
    gt = load_ground_truth(args.ground_truth, args.fps)
    total_active_sec = sum(e - s for s, e in gt)
    print(f"  {len(gt)} rally intervals  ({total_active_sec:.1f}s total active time)")

    results = grid_search(df, gt)
    ranked  = print_top(results, args.top_n)

    if args.out:
        write_csv(results, args.out)
    else:
        print("\n(Pass --out results.csv to save the full ranked table.)")

    best = ranked[0]
    print(f"\nBest param set (recall={best.recall:.3f}  F1={best.f1:.3f}  false_exits={best.false_exits}):")
    p = best.params
    print(f"  P_AR2W                        = {p.P_AR2W}")
    print(f"  BALL_MISSING_TAU_SEC          = {p.BALL_MISSING_TAU_SEC}")
    print(f"  BALL_VEL_SLOW_THRESH          = {p.BALL_VEL_SLOW_THRESH}")
    print(f"  BALL_NEAR_PLAYER_DURATION_SEC = {p.BALL_NEAR_PLAYER_DURATION_SEC}")
    print(f"  VEL_VARIANCE_LOW_THRESHOLD    = {p.VEL_VARIANCE_LOW_THRESHOLD}")
    print(f"  RETREAT_MIN_SPEED_FT_S        = {p.RETREAT_MIN_SPEED_FT_S}")
    print("\nApply these in anya_hmm.py to the corresponding top-level constants.")


if __name__ == "__main__":
    main()
