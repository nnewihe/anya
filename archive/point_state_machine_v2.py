"""
point_state_machine_v2.py
==========================
Tennis Point State Machine — anchored to deterministic serve events.

Stage 2 of the two-stage Anya pipeline.

Key differences from point_state_machine.py
---------------------------------------------
* Accepts serve_events.json produced by anya_vision_core.py alongside the
  bounding-box CSV.
* DELTA_A_SERVE_SPIKE heuristic is completely removed.  The state machine
  no longer guesses serves from area volatility.
* PRE_POINT / POINT_END / WALK_OFF → SERVE transitions fire *only* when the
  current frame_id matches a frame recorded in serve_events.json.
* The SERVE state is held briefly (SERVE_HOLD_FRAMES) before RAI is allowed
  to drive the ACTIVE_MOVING transition — giving the signal time to settle.
* All downstream transitions (ACTIVE_MOVING, ACTIVE_WAITING, POINT_END,
  WALK_OFF) continue to use RAI, Velocity, and Pacing Variance unchanged.
* Matplotlib output marks each injected serve frame with a vertical line.

Usage
-----
  python point_state_machine_v2.py telemetry.csv serve_events.json
  python point_state_machine_v2.py telemetry.csv serve_events.json \\
      --output-csv state_events.csv --output-plot plot.png --fps 60
"""

from __future__ import annotations

import argparse
import csv
import enum
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ================================================================
# Configuration
# ================================================================

CONFIG = {
    "RAI_THRESHOLD_QUIET":  5.0,    # Below this RAI → "quiet"
    "RAI_THRESHOLD_PLAY":  15.0,    # Above this RAI → "playing"
    # NOTE: DELTA_A_SERVE_SPIKE has been intentionally removed.
    #       Serve detection is now fully delegated to serve_events.json.
    "T_GAP_SECONDS":        2.2,    # Quiet duration before POINT_END
    "SERVE_HOLD_FRAMES":   10,      # Frames SERVE state is held before RAI takes over
    "SETTLE_WINDOW_FRAMES": 5,      # Hysteresis: frames to confirm non-serve transitions
    "ALPHA_WEIGHT":         0.7,    # Velocity weight in RAI
    "BETA_WEIGHT":          0.3,    # Area-volatility weight in RAI
    "ROLLING_WINDOW_N":    15,      # Short rolling window (frames)
    "PACING_WINDOW_M":     60,      # Long rolling window for PV (frames)
    "PV_LOW_THRESHOLD":     2.0,    # PV below this → uniform gait / walk-off
    "FPS":                 30,      # Video frame-rate (used for time conversions)
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
    frame_id:       int
    velocity:       float = 0.0
    delta_area:     float = 0.0
    rai:            float = 0.0
    pacing_variance: float = 0.0


def compute_metrics(df: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    """
    Compute velocity, area volatility, RAI, and pacing variance from
    raw bounding-box telemetry.

    Note: delta_area is still computed for diagnostic purposes and is visible
    in the returned DataFrame, but it is never used for serve detection.

    Parameters
    ----------
    df  : DataFrame with columns [frame_id, x, y, w, h]
    cfg : configuration dictionary

    Returns
    -------
    DataFrame with added columns: velocity, delta_area, rai, pacing_variance
    """
    df    = df.copy().sort_values("frame_id").reset_index(drop=True)
    N     = cfg["ROLLING_WINDOW_N"]
    M     = cfg["PACING_WINDOW_M"]
    alpha = cfg["ALPHA_WEIGHT"]
    beta  = cfg["BETA_WEIGHT"]

    dx = df["x"].diff().fillna(0.0)
    dy = df["y"].diff().fillna(0.0)
    df["velocity"]  = np.sqrt(dx ** 2 + dy ** 2)

    area            = df["w"] * df["h"]
    df["delta_area"] = area.diff().abs().fillna(0.0)

    raw_rai  = alpha * df["velocity"] + beta * df["delta_area"]
    df["rai"] = raw_rai.rolling(window=N, min_periods=1).mean()

    df["pacing_variance"] = (
        df["velocity"].rolling(window=M, min_periods=1).var().fillna(0.0)
    )
    return df


# ================================================================
# Serve Events Loader
# ================================================================

def load_serve_frames(serve_events_path: str) -> Set[int]:
    """
    Load serve_events.json produced by anya_vision_core.py and return the
    set of frame_id integers at which a serve was detected.

    Parameters
    ----------
    serve_events_path : path to the JSON file

    Returns
    -------
    set of int  — frame IDs that anchor SERVE transitions
    """
    with open(serve_events_path, "r") as f:
        events = json.load(f)

    serve_frames = {int(ev["frame_id"]) for ev in events}
    print(f"[SERVE EVENTS] Loaded {len(serve_frames)} serve frame(s)"
          f" from {serve_events_path}")
    for ev in sorted(events, key=lambda e: e["frame_id"]):
        print(f"  frame {ev['frame_id']:>6}  |  {ev['timestamp']:.3f}s")

    return serve_frames


# ================================================================
# State Machine
# ================================================================

@dataclass
class StateEvent:
    """One contiguous block in a single state."""
    state:          State
    start_frame:    int
    end_frame:      int   = 0
    duration_frames: int  = 0
    duration_seconds: float = 0.0

    def close(self, end_frame: int, fps: float):
        self.end_frame        = end_frame
        self.duration_frames  = self.end_frame - self.start_frame + 1
        self.duration_seconds = self.duration_frames / fps


class TennisPointStateMachineV2:
    """
    Deterministic-serve state machine.

    Serve transitions are anchored to frames listed in serve_events.json.
    All other transitions use RAI, Velocity, and Pacing Variance.
    """

    def __init__(self, serve_frames: Set[int], cfg: dict = CONFIG):
        self.cfg          = cfg
        self.fps          = cfg["FPS"]
        self.serve_frames = serve_frames            # injected from JSON

        self.state:          State            = State.PRE_POINT
        self.events:         List[StateEvent] = []
        self.frame_states:   List[State]      = []
        self.transition_log: List[str]        = []

        # Debouncing for non-serve transitions
        self._candidate_state: Optional[State] = None
        self._candidate_count: int             = 0

        # Quiet-duration tracking
        self._quiet_start_frame: Optional[int] = None

        # SERVE hold counter — keeps the state in SERVE for a brief window
        # before RAI is allowed to push it to ACTIVE_MOVING
        self._serve_hold_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the state machine over the full metrics DataFrame.
        Returns the DataFrame with an added 'state' column.
        """
        self.state            = State.PRE_POINT
        self.events           = []
        self.frame_states     = []
        self.transition_log   = []
        self._candidate_state = None
        self._candidate_count = 0
        self._quiet_start_frame   = None
        self._serve_hold_counter  = 0

        current_event = StateEvent(
            state=self.state, start_frame=int(df.iloc[0]["frame_id"])
        )

        for _, row in df.iterrows():
            frame_id = int(row["frame_id"])
            rai      = float(row["rai"])
            pv       = float(row["pacing_variance"])
            # delta_a is available in the row for diagnostic use but is NOT
            # passed into _evaluate_transition — serve detection is frame-based.

            desired = self._evaluate_transition(frame_id, rai, pv)

            # Serve transitions are instantaneous (no debounce) to preserve
            # the precise a priori anchor point from the JSON.
            is_serve_transition = (desired == State.SERVE and
                                   self.state != State.SERVE)

            if desired != self.state:
                if is_serve_transition:
                    # Commit immediately — no settle window
                    old         = self.state
                    self.state  = desired
                    self._candidate_state = None
                    self._candidate_count = 0
                    self._serve_hold_counter = 0

                    current_event.close(frame_id - 1, self.fps)
                    self.events.append(current_event)
                    current_event = StateEvent(state=self.state,
                                               start_frame=frame_id)

                    reason = f"Injected serve at frame {frame_id}"
                    msg    = f"[Frame {frame_id}]: {old.value} -> {self.state.value} ({reason})"
                    self.transition_log.append(msg)
                    print(msg)

                else:
                    # Standard debounced transition
                    if desired == self._candidate_state:
                        self._candidate_count += 1
                    else:
                        self._candidate_state = desired
                        self._candidate_count = 1

                    if self._candidate_count >= self.cfg["SETTLE_WINDOW_FRAMES"]:
                        old         = self.state
                        self.state  = desired
                        self._candidate_state = None
                        self._candidate_count = 0

                        settle           = self.cfg["SETTLE_WINDOW_FRAMES"]
                        transition_frame = frame_id - settle + 1
                        current_event.close(transition_frame - 1, self.fps)
                        self.events.append(current_event)
                        current_event = StateEvent(state=self.state,
                                                   start_frame=transition_frame)

                        reason = self._transition_reason(
                            old, self.state, rai, pv, frame_id
                        )
                        msg = (f"[Frame {frame_id}]: "
                               f"{old.value} -> {self.state.value} ({reason})")
                        self.transition_log.append(msg)
                        print(msg)
                    else:
                        # Candidate hasn't settled yet — keep current state
                        pass

            else:
                # Desired matches current — reset candidate
                self._candidate_state = None
                self._candidate_count = 0

            self.frame_states.append(self.state)

        # Close final event
        current_event.close(int(df.iloc[-1]["frame_id"]), self.fps)
        self.events.append(current_event)

        df = df.copy()
        df["state"] = [s.value for s in self.frame_states]
        return df

    # ------------------------------------------------------------------
    # Transition logic
    # ------------------------------------------------------------------

    def _evaluate_transition(self, frame_id: int, rai: float, pv: float) -> State:
        """
        Return the desired next state for the current frame.

        Serve transitions are gated exclusively on frame_id membership in
        self.serve_frames.  All other transitions use RAI and Pacing Variance.
        """
        quiet       = self.cfg["RAI_THRESHOLD_QUIET"]
        play        = self.cfg["RAI_THRESHOLD_PLAY"]
        t_gap_frames = self.cfg["T_GAP_SECONDS"] * self.fps
        pv_low      = self.cfg["PV_LOW_THRESHOLD"]
        hold_limit  = self.cfg["SERVE_HOLD_FRAMES"]

        is_quiet  = rai < quiet
        is_active = rai >= play

        # Maintain quiet-duration counter (used by ACTIVE_WAITING, POINT_END, WALK_OFF)
        if is_quiet:
            if self._quiet_start_frame is None:
                self._quiet_start_frame = frame_id
            quiet_duration = frame_id - self._quiet_start_frame
        else:
            self._quiet_start_frame = None
            quiet_duration = 0

        # ------------------------------------------------------------------
        # Helper: is there a serve anchored at this exact frame?
        # ------------------------------------------------------------------
        def _is_serve_frame() -> bool:
            return frame_id in self.serve_frames

        # ------------------------------------------------------------------
        # State rules
        # ------------------------------------------------------------------

        if self.state == State.PRE_POINT:
            # Only a confirmed serve event can leave PRE_POINT
            if _is_serve_frame():
                return State.SERVE
            return State.PRE_POINT

        if self.state == State.SERVE:
            # Hold the SERVE state for SERVE_HOLD_FRAMES before yielding to RAI
            self._serve_hold_counter += 1
            if self._serve_hold_counter < hold_limit:
                return State.SERVE
            # After hold: RAI drives the transition to active play
            if is_active:
                return State.ACTIVE_MOVING
            # Serve that calmed down without activity → point ended quickly
            if is_quiet and quiet_duration > t_gap_frames:
                return State.POINT_END
            return State.SERVE

        if self.state == State.ACTIVE_MOVING:
            if is_quiet:
                return State.ACTIVE_WAITING
            # Walk-off: moderate motion but uniform, steady gait
            if (not is_active) and pv < pv_low and rai >= quiet:
                return State.WALK_OFF
            return State.ACTIVE_MOVING

        if self.state == State.ACTIVE_WAITING:
            # Player resumed movement before T_gap expired
            if is_active:
                return State.ACTIVE_MOVING
            # Quiet persisted beyond T_gap → point is over
            if is_quiet and quiet_duration > t_gap_frames:
                return State.POINT_END
            return State.ACTIVE_WAITING

        if self.state == State.POINT_END:
            # A new serve event anchors the next point — no heuristic needed
            if _is_serve_frame():
                return State.SERVE
            # Walk-off: moderate steady movement
            if (not is_quiet) and pv < pv_low:
                return State.WALK_OFF
            # Extended quiet → back to pre-point waiting
            if is_quiet and quiet_duration > t_gap_frames * 2:
                return State.PRE_POINT
            return State.POINT_END

        if self.state == State.WALK_OFF:
            # Walk-off ends with stillness
            if is_quiet and quiet_duration > t_gap_frames:
                return State.PRE_POINT
            # New serve event during walk-off (player rushes back)
            if _is_serve_frame():
                return State.SERVE
            return State.WALK_OFF

        return self.state  # fallback — should never reach here

    @staticmethod
    def _transition_reason(old: State, new: State,
                           rai: float, pv: float, frame: int) -> str:
        reasons = {
            (State.SERVE,          State.ACTIVE_MOVING):  f"RAI={rai:.1f} above play threshold",
            (State.ACTIVE_MOVING,  State.ACTIVE_WAITING): f"RAI={rai:.1f} dropped below quiet",
            (State.ACTIVE_WAITING, State.ACTIVE_MOVING):  f"RAI={rai:.1f} resumed above play",
            (State.ACTIVE_WAITING, State.POINT_END):      f"RAI quiet for > T_gap",
            (State.POINT_END,      State.PRE_POINT):      f"Extended quiet period",
            (State.ACTIVE_MOVING,  State.WALK_OFF):       f"PV={pv:.2f} ultra-low, uniform gait",
            (State.WALK_OFF,       State.PRE_POINT):      f"Walk-off ended, quiet",
            (State.POINT_END,      State.WALK_OFF):       f"Moderate RAI, PV={pv:.2f} uniform gait",
            (State.SERVE,          State.POINT_END):      f"RAI quiet for > T_gap after serve",
        }
        return reasons.get((old, new), f"RAI={rai:.1f}, PV={pv:.2f}")


# ================================================================
# Output: CSV
# ================================================================

def write_events_csv(events: List[StateEvent], path: str, fps: float):
    """Write the state-event log to CSV."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "state", "duration_in_state",
                          "start_frame", "end_frame"])
        for ev in events:
            ts = f"{ev.start_frame / fps:.3f}"
            writer.writerow([ts, ev.state.value, f"{ev.duration_seconds:.3f}",
                             ev.start_frame, ev.end_frame])
    print(f"[CSV] Wrote {len(events)} state events to {path}")


# ================================================================
# Output: Visualisation
# ================================================================

def plot_state_timeline(
    df: pd.DataFrame,
    events: List[StateEvent],
    serve_frames: Set[int],
    out_path: str = "state_machine_plot.png",
    cfg: dict = CONFIG,
):
    """
    Primary plot: RAI over time with colour-coded state overlays and vertical
    markers at each injected serve frame.
    """
    fps  = cfg["FPS"]
    time = df["frame_id"].values / fps

    fig, axes = plt.subplots(
        3, 1, figsize=(18, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
    )

    # ---- Panel 1: RAI with state backgrounds ----
    ax_rai = axes[0]

    for ev in events:
        t0 = ev.start_frame / fps
        t1 = ev.end_frame   / fps
        ax_rai.axvspan(t0, t1, alpha=0.25, color=STATE_COLOURS[ev.state])

    ax_rai.plot(time, df["rai"].values, color="black", linewidth=0.8, label="RAI")
    ax_rai.axhline(cfg["RAI_THRESHOLD_QUIET"], color="red",   ls="--", lw=0.7,
                   label="Quiet threshold")
    ax_rai.axhline(cfg["RAI_THRESHOLD_PLAY"],  color="green", ls="--", lw=0.7,
                   label="Play threshold")

    # Mark injected serve frames with vertical lines
    for sf in sorted(serve_frames):
        serve_t = sf / fps
        ax_rai.axvline(serve_t, color="#e74c3c", linewidth=1.4,
                       linestyle="-", alpha=0.85, zorder=5)

    # Dummy artist for the legend entry
    serve_line = plt.Line2D([0], [0], color="#e74c3c", linewidth=1.4,
                             label="Injected serve (a priori)")
    handles, labels_l = ax_rai.get_legend_handles_labels()
    ax_rai.legend(handles=handles + [serve_line],
                  labels=labels_l + ["Injected serve (a priori)"],
                  loc="upper right", fontsize=8)

    ax_rai.set_ylabel("RAI")
    ax_rai.set_title(
        f"Tennis Point State Machine v2 — RAI & State Timeline\n"
        f"({len(serve_frames)} deterministic serve event(s) injected from JSON)"
    )

    # ---- Panel 2: Velocity ----
    ax_vel = axes[1]
    ax_vel.plot(time, df["velocity"].values, color="#2c3e50", linewidth=0.6)
    for sf in sorted(serve_frames):
        ax_vel.axvline(sf / fps, color="#e74c3c", linewidth=1.0,
                       linestyle="-", alpha=0.6)
    ax_vel.set_ylabel("Velocity (px/frame)")

    # ---- Panel 3: Pacing Variance ----
    ax_pv = axes[2]
    ax_pv.plot(time, df["pacing_variance"].values, color="#8e44ad", linewidth=0.6)
    ax_pv.axhline(cfg["PV_LOW_THRESHOLD"], color="orange", ls="--", lw=0.7,
                  label="PV low threshold")
    for sf in sorted(serve_frames):
        ax_pv.axvline(sf / fps, color="#e74c3c", linewidth=1.0,
                      linestyle="-", alpha=0.6)
    ax_pv.set_ylabel("Pacing Variance")
    ax_pv.set_xlabel("Time (s)")
    ax_pv.legend(loc="upper right", fontsize=8)

    # Shared state legend
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

def run_state_machine(
    input_csv: str,
    serve_events_json: str,
    output_csv:  str = "state_events_v2.csv",
    output_plot: str = "state_machine_plot_v2.png",
    cfg: dict = CONFIG,
) -> Tuple[pd.DataFrame, List[StateEvent]]:
    """
    End-to-end pipeline:
      1. Load bounding-box CSV + serve events JSON
      2. Compute metrics
      3. Run serve-anchored state machine
      4. Write CSV + plot

    Parameters
    ----------
    input_csv         : path to CSV with columns [frame_id, x, y, w, h]
    serve_events_json : path to JSON produced by anya_vision_core.py
    output_csv        : path for the state-event CSV output
    output_plot       : path for the PNG/PDF plot
    cfg               : configuration dictionary (override defaults)

    Returns
    -------
    (annotated_df, events)
    """
    print(f"[STATE MACHINE v2] Loading telemetry from {input_csv}")
    df = pd.read_csv(input_csv)

    required = {"frame_id", "x", "y", "w", "h"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    serve_frames = load_serve_frames(serve_events_json)

    df = compute_metrics(df, cfg)

    sm = TennisPointStateMachineV2(serve_frames=serve_frames, cfg=cfg)
    df = sm.run(df)

    write_events_csv(sm.events, output_csv, fps=cfg["FPS"])
    plot_state_timeline(df, sm.events, serve_frames,
                        out_path=output_plot, cfg=cfg)

    print(f"\n[STATE MACHINE v2] Complete — {len(sm.events)} state segments, "
          f"{len(sm.transition_log)} transitions")
    return df, sm.events


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Tennis Point State Machine v2 — serve-event anchored.\n"
            "Requires both a bounding-box CSV and serve_events.json "
            "produced by anya_vision_core.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_csv",
        help="Path to bounding-box CSV (frame_id, x, y, w, h) from anya_vision_core.py",
    )
    parser.add_argument(
        "serve_events_json",
        help="Path to serve_events.json produced by anya_vision_core.py",
    )
    parser.add_argument(
        "--output-csv",
        default="state_events_v2.csv",
        help="Path for state-event output CSV (default: state_events_v2.csv)",
    )
    parser.add_argument(
        "--output-plot",
        default="state_machine_plot_v2.png",
        help="Path for the plot image (default: state_machine_plot_v2.png)",
    )
    parser.add_argument(
        "--fps",
        type=float, default=None,
        help="Override video FPS (default: 30)",
    )
    parser.add_argument(
        "--t-gap",
        type=float, default=None,
        help="Override T_gap in seconds (default: 2.2)",
    )
    parser.add_argument(
        "--rai-quiet",
        type=float, default=None,
        help="Override RAI quiet threshold (default: 5.0)",
    )
    parser.add_argument(
        "--rai-play",
        type=float, default=None,
        help="Override RAI play threshold (default: 15.0)",
    )
    parser.add_argument(
        "--serve-hold",
        type=int, default=None,
        help="Frames to hold SERVE state before RAI takes over (default: 10)",
    )
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
    if args.serve_hold is not None:
        cfg["SERVE_HOLD_FRAMES"] = args.serve_hold

    run_state_machine(
        input_csv         =args.input_csv,
        serve_events_json =args.serve_events_json,
        output_csv        =args.output_csv,
        output_plot       =args.output_plot,
        cfg               =cfg,
    )
