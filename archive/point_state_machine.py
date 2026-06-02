"""
Tennis Point State Machine from Bounding Box Telemetry.

Consumes YOLO bounding-box data (center x, y, width w, height h) and
identifies the lifecycle of a tennis point through six discrete states.

Outputs:
  - CSV of state events (timestamp, state, duration, start/end frame)
  - Matplotlib visualisation of RAI + state overlays
"""

import os
import csv
import enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ================================================================
# Human-in-the-Loop Configuration
# ================================================================

CONFIG = {
    "RAI_THRESHOLD_QUIET": 5.0,        # Below this RAI -> "quiet"
    "RAI_THRESHOLD_PLAY": 15.0,        # Above this RAI -> "playing"
    "DELTA_A_SERVE_SPIKE": 40.0,       # Min area-volatility spike for serve
    "T_GAP_SECONDS": 2.2,              # Quiet duration before POINT_END
    "SETTLE_WINDOW_FRAMES": 5,         # Hysteresis: frames to confirm transition
    "ALPHA_WEIGHT": 0.7,               # Velocity weight in RAI
    "BETA_WEIGHT": 0.3,                # Area-volatility weight in RAI
    "ROLLING_WINDOW_N": 15,            # Short rolling window (frames)
    "PACING_WINDOW_M": 60,             # Long rolling window for PV (frames)
    "PV_LOW_THRESHOLD": 2.0,           # Pacing variance below this -> uniform gait
    "FPS": 30,                         # Video frame-rate (used for time conversions)
}

# ================================================================
# States
# ================================================================

class State(enum.Enum):
    PRE_POINT      = "PRE_POINT"
    SERVE          = "SERVE"
    ACTIVE_MOVING  = "ACTIVE_MOVING"
    ACTIVE_WAITING = "ACTIVE_WAITING"
    POINT_END      = "POINT_END"
    WALK_OFF       = "WALK_OFF"

# Colour map for visualisation
STATE_COLOURS = {
    State.PRE_POINT:      "#bdc3c7",   # grey
    State.SERVE:          "#e74c3c",   # red
    State.ACTIVE_MOVING:  "#2ecc71",   # green
    State.ACTIVE_WAITING: "#f39c12",   # amber
    State.POINT_END:      "#3498db",   # blue
    State.WALK_OFF:       "#9b59b6",   # purple
}

# ================================================================
# Metrics Computation
# ================================================================

@dataclass
class FrameMetrics:
    """Per-frame derived metrics."""
    frame_id: int
    velocity: float = 0.0
    delta_area: float = 0.0
    rai: float = 0.0
    pacing_variance: float = 0.0


def compute_metrics(df: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    """
    Compute velocity, area volatility, RAI, and pacing variance from
    raw bounding-box telemetry.

    Parameters
    ----------
    df : DataFrame with columns [frame_id, x, y, w, h]
    cfg : configuration dictionary

    Returns
    -------
    DataFrame with added columns: velocity, delta_area, rai, pacing_variance
    """
    df = df.copy().sort_values("frame_id").reset_index(drop=True)

    N = cfg["ROLLING_WINDOW_N"]
    M = cfg["PACING_WINDOW_M"]
    alpha = cfg["ALPHA_WEIGHT"]
    beta = cfg["BETA_WEIGHT"]

    # Velocity: Euclidean distance between consecutive centroids
    dx = df["x"].diff().fillna(0.0)
    dy = df["y"].diff().fillna(0.0)
    df["velocity"] = np.sqrt(dx ** 2 + dy ** 2)

    # Area volatility: absolute change in bounding-box area
    area = df["w"] * df["h"]
    df["delta_area"] = area.diff().abs().fillna(0.0)

    # RAI (Rolling Activity Index): weighted sum, smoothed over N frames
    raw_rai = alpha * df["velocity"] + beta * df["delta_area"]
    df["rai"] = raw_rai.rolling(window=N, min_periods=1).mean()

    # Pacing Variance: rolling variance of velocity over longer window M
    df["pacing_variance"] = df["velocity"].rolling(window=M, min_periods=1).var().fillna(0.0)

    return df


# ================================================================
# State Machine
# ================================================================

@dataclass
class StateEvent:
    """One contiguous block in a single state."""
    state: State
    start_frame: int
    end_frame: int = 0
    duration_frames: int = 0
    duration_seconds: float = 0.0

    def close(self, end_frame: int, fps: float):
        self.end_frame = end_frame
        self.duration_frames = self.end_frame - self.start_frame + 1
        self.duration_seconds = self.duration_frames / fps


class TennisPointStateMachine:
    """
    Class-based state machine that walks through per-frame metrics
    and emits state labels + transition events.
    """

    def __init__(self, cfg: dict = CONFIG):
        self.cfg = cfg
        self.fps = cfg["FPS"]
        self.state: State = State.PRE_POINT
        self.events: List[StateEvent] = []
        self.frame_states: List[State] = []
        self.transition_log: List[str] = []

        # Debouncing / hysteresis bookkeeping
        self._candidate_state: Optional[State] = None
        self._candidate_count: int = 0

        # Time tracking for ACTIVE_WAITING -> POINT_END
        self._quiet_start_frame: Optional[int] = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the state machine over the full metrics DataFrame.
        Returns the DataFrame with an added 'state' column.
        """
        self.state = State.PRE_POINT
        self.events = []
        self.frame_states = []
        self.transition_log = []
        self._candidate_state = None
        self._candidate_count = 0
        self._quiet_start_frame = None

        current_event = StateEvent(state=self.state, start_frame=int(df.iloc[0]["frame_id"]))

        for _, row in df.iterrows():
            frame_id = int(row["frame_id"])
            rai = float(row["rai"])
            pv = float(row["pacing_variance"])
            delta_a = float(row["delta_area"])

            desired = self._evaluate_transition(frame_id, rai, pv, delta_a)

            if desired != self.state:
                # Try to debounce
                if desired == self._candidate_state:
                    self._candidate_count += 1
                else:
                    self._candidate_state = desired
                    self._candidate_count = 1

                # Only commit transition after settle window
                if self._candidate_count >= self.cfg["SETTLE_WINDOW_FRAMES"]:
                    old = self.state
                    self.state = desired
                    self._candidate_state = None
                    self._candidate_count = 0

                    # Close previous event, start new one
                    settle = self.cfg["SETTLE_WINDOW_FRAMES"]
                    transition_frame = frame_id - settle + 1
                    current_event.close(transition_frame - 1, self.fps)
                    self.events.append(current_event)
                    current_event = StateEvent(state=self.state, start_frame=transition_frame)

                    reason = self._transition_reason(old, self.state, rai, pv, delta_a, frame_id)
                    msg = f"[Frame {frame_id}]: {old.value} -> {self.state.value} ({reason})"
                    self.transition_log.append(msg)
                    print(msg)
            else:
                # Conditions reverted — reset candidate
                self._candidate_state = None
                self._candidate_count = 0

            self.frame_states.append(self.state)

        # Close final event
        current_event.close(int(df.iloc[-1]["frame_id"]), self.fps)
        self.events.append(current_event)

        df = df.copy()
        df["state"] = [s.value for s in self.frame_states]
        return df

    # ----------------------------------------------------------
    # Transition logic
    # ----------------------------------------------------------

    def _evaluate_transition(self, frame_id: int, rai: float, pv: float, delta_a: float) -> State:
        """
        Given current state and frame metrics, return the *desired* next state
        (before debouncing).
        """
        quiet = self.cfg["RAI_THRESHOLD_QUIET"]
        play  = self.cfg["RAI_THRESHOLD_PLAY"]
        t_gap_frames = self.cfg["T_GAP_SECONDS"] * self.fps
        pv_low = self.cfg["PV_LOW_THRESHOLD"]
        serve_spike = self.cfg["DELTA_A_SERVE_SPIKE"]

        is_quiet = rai < quiet
        is_active = rai >= play

        # Track how long we've been quiet
        if is_quiet:
            if self._quiet_start_frame is None:
                self._quiet_start_frame = frame_id
            quiet_duration = frame_id - self._quiet_start_frame
        else:
            self._quiet_start_frame = None
            quiet_duration = 0

        # ---- State-specific rules ----

        if self.state == State.PRE_POINT:
            # Stay PRE_POINT until a serve spike
            if delta_a >= serve_spike and rai >= quiet:
                return State.SERVE
            return State.PRE_POINT

        if self.state == State.SERVE:
            # After serve: move to active play once RAI is high
            if is_active:
                return State.ACTIVE_MOVING
            # If it calms down quickly, might be a false spike
            if is_quiet and quiet_duration > t_gap_frames:
                return State.POINT_END
            return State.SERVE

        if self.state == State.ACTIVE_MOVING:
            if is_quiet:
                return State.ACTIVE_WAITING
            # Walk-off detection: moderate RAI but ultra-low pacing variance
            if (not is_active) and pv < pv_low and rai >= quiet:
                return State.WALK_OFF
            return State.ACTIVE_MOVING

        if self.state == State.ACTIVE_WAITING:
            # Player moved again before T_gap expired -> back to active
            if is_active:
                return State.ACTIVE_MOVING
            # Quiet too long -> point ended
            if is_quiet and quiet_duration > t_gap_frames:
                return State.POINT_END
            return State.ACTIVE_WAITING

        if self.state == State.POINT_END:
            # New serve spike -> new point begins
            if delta_a >= serve_spike and rai >= quiet:
                return State.SERVE
            # Walk-off: moderate steady movement
            if (not is_quiet) and pv < pv_low:
                return State.WALK_OFF
            # Long quiet = pre-point
            if is_quiet and quiet_duration > t_gap_frames * 2:
                return State.PRE_POINT
            return State.POINT_END

        if self.state == State.WALK_OFF:
            # Walk-off ends with stillness -> pre-point
            if is_quiet and quiet_duration > t_gap_frames:
                return State.PRE_POINT
            # Sudden serve spike
            if delta_a >= serve_spike and rai >= quiet:
                return State.SERVE
            return State.WALK_OFF

        return self.state  # fallback

    @staticmethod
    def _transition_reason(old: State, new: State, rai: float, pv: float,
                           delta_a: float, frame: int) -> str:
        reasons = {
            (State.PRE_POINT, State.SERVE):       f"ΔA spike={delta_a:.1f}",
            (State.SERVE, State.ACTIVE_MOVING):    f"RAI={rai:.1f} above play threshold",
            (State.ACTIVE_MOVING, State.ACTIVE_WAITING): f"RAI={rai:.1f} dropped below quiet",
            (State.ACTIVE_WAITING, State.ACTIVE_MOVING): f"RAI={rai:.1f} resumed above play",
            (State.ACTIVE_WAITING, State.POINT_END):     f"RAI quiet for > T_gap",
            (State.POINT_END, State.PRE_POINT):    f"Extended quiet period",
            (State.POINT_END, State.SERVE):        f"New serve spike ΔA={delta_a:.1f}",
            (State.ACTIVE_MOVING, State.WALK_OFF): f"PV={pv:.2f} ultra-low, uniform gait",
            (State.WALK_OFF, State.PRE_POINT):     f"Walk-off ended, quiet",
            (State.POINT_END, State.WALK_OFF):     f"Moderate RAI, PV={pv:.2f} uniform gait",
            (State.SERVE, State.POINT_END):        f"RAI quiet for > T_gap after serve",
        }
        return reasons.get((old, new), f"RAI={rai:.1f}, PV={pv:.2f}, ΔA={delta_a:.1f}")


# ================================================================
# Output: CSV
# ================================================================

def write_events_csv(events: List[StateEvent], path: str, fps: float):
    """Write the state-event log to CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "state", "duration_in_state", "start_frame", "end_frame"])
        for ev in events:
            ts = f"{ev.start_frame / fps:.3f}"
            writer.writerow([ts, ev.state.value, f"{ev.duration_seconds:.3f}",
                             ev.start_frame, ev.end_frame])
    print(f"[CSV] Wrote {len(events)} state events to {path}")


# ================================================================
# Output: Visualisation
# ================================================================

def plot_state_timeline(df: pd.DataFrame, events: List[StateEvent],
                        out_path: str = "state_machine_plot.png",
                        cfg: dict = CONFIG):
    """
    Primary plot: RAI over time with colour-coded state overlays.
    """
    fps = cfg["FPS"]
    time = df["frame_id"].values / fps

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    # --- Panel 1: RAI with state backgrounds ---
    ax_rai = axes[0]
    for ev in events:
        t0 = ev.start_frame / fps
        t1 = ev.end_frame / fps
        ax_rai.axvspan(t0, t1, alpha=0.25, color=STATE_COLOURS[ev.state])

    ax_rai.plot(time, df["rai"].values, color="black", linewidth=0.8, label="RAI")
    ax_rai.axhline(cfg["RAI_THRESHOLD_QUIET"], color="red", ls="--", lw=0.7, label="Quiet threshold")
    ax_rai.axhline(cfg["RAI_THRESHOLD_PLAY"], color="green", ls="--", lw=0.7, label="Play threshold")
    ax_rai.set_ylabel("RAI")
    ax_rai.set_title("Tennis Point State Machine — RAI & State Timeline")
    ax_rai.legend(loc="upper right", fontsize=8)

    # --- Panel 2: Velocity ---
    ax_vel = axes[1]
    ax_vel.plot(time, df["velocity"].values, color="#2c3e50", linewidth=0.6)
    ax_vel.set_ylabel("Velocity (px/frame)")

    # --- Panel 3: Pacing Variance ---
    ax_pv = axes[2]
    ax_pv.plot(time, df["pacing_variance"].values, color="#8e44ad", linewidth=0.6)
    ax_pv.axhline(cfg["PV_LOW_THRESHOLD"], color="orange", ls="--", lw=0.7, label="PV low threshold")
    ax_pv.set_ylabel("Pacing Variance")
    ax_pv.set_xlabel("Time (s)")
    ax_pv.legend(loc="upper right", fontsize=8)

    # Shared legend for states
    patches = [mpatches.Patch(color=STATE_COLOURS[s], label=s.value) for s in State]
    fig.legend(handles=patches, loc="lower center", ncol=len(State), fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved state timeline to {out_path}")


# ================================================================
# Top-level runner
# ================================================================

def run_state_machine(input_csv: str,
                      output_csv: str = "state_events.csv",
                      output_plot: str = "state_machine_plot.png",
                      cfg: dict = CONFIG) -> Tuple[pd.DataFrame, List[StateEvent]]:
    """
    End-to-end pipeline:
      1. Load bounding-box CSV
      2. Compute metrics
      3. Run state machine
      4. Write CSV + plot

    Parameters
    ----------
    input_csv : path to CSV with columns [frame_id, x, y, w, h]
    output_csv : path for the state-event CSV output
    output_plot : path for the PNG/PDF plot
    cfg : configuration dictionary (override defaults)

    Returns
    -------
    (annotated_df, events)
    """
    print(f"[STATE MACHINE] Loading telemetry from {input_csv}")
    df = pd.read_csv(input_csv)

    required = {"frame_id", "x", "y", "w", "h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    df = compute_metrics(df, cfg)

    sm = TennisPointStateMachine(cfg)
    df = sm.run(df)

    write_events_csv(sm.events, output_csv, fps=cfg["FPS"])
    plot_state_timeline(df, sm.events, out_path=output_plot, cfg=cfg)

    print(f"\n[STATE MACHINE] Complete — {len(sm.events)} state segments, "
          f"{len(sm.transition_log)} transitions")
    return df, sm.events


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tennis Point State Machine")
    parser.add_argument("input_csv", help="Path to bounding-box CSV (frame_id, x, y, w, h)")
    parser.add_argument("--output-csv", default="state_events.csv",
                        help="Path for state-event output CSV")
    parser.add_argument("--output-plot", default="state_machine_plot.png",
                        help="Path for the plot image (PNG/PDF)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override video FPS (default: 30)")
    parser.add_argument("--t-gap", type=float, default=None,
                        help="Override T_gap in seconds")
    parser.add_argument("--rai-quiet", type=float, default=None,
                        help="Override RAI quiet threshold")
    parser.add_argument("--rai-play", type=float, default=None,
                        help="Override RAI play threshold")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.fps is not None:
        cfg["FPS"] = args.fps
    if args.t_gap is not None:
        cfg["T_GAP_SECONDS"] = args.t_gap
    if args.rai_quiet is not None:
        cfg["RAI_THRESHOLD_QUIET"] = args.rai_quiet
    if args.rai_play is not None:
        cfg["RAI_THRESHOLD_PLAY"] = args.rai_play

    run_state_machine(args.input_csv, args.output_csv, args.output_plot, cfg)
