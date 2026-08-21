"""A non-monotonic per-point ENERGY BAR as the point-end policy.

The shipped `trace` policy ends a point on a fixed window of in-court ball
silence — 4.0 s alone, 2.0 s with corroborating walking.  That window is
memoryless: every returning detection resets the clock outright, so a rally is
judged only by its single longest dropout.  This module integrates the same
evidence instead of windowing it.

A bar starts at `energy_start` at every detected serve, holds flat through
`energy_hold_s` while the serve, the bounce and the first exchange make every
drain unreliable, and is then integrated forward in `energy_step_s` steps.
Rally activity charges it; the absence of an in-court ball trace discharges it;
walking makes that discharge steeper; so do three near-player signals that are
not walking (`pipeline/near_end.py` — turning away from the net, dropping out of
the ready stance, reaching into a pocket for the second ball); a missing near
player discharges it separately.  The point ends the first time the bar sits at
the floor for `energy_confirm_s`.  Three of those terms — the reversal cue, the
missing-player drain and the `settle` signal — tuned to zero on the corpus and
ship off; see ReelConfig for what was measured and why they are kept as knobs.

Four decisions carry the design:

  * EVERY NEAR-PLAYER TERM MULTIPLIES THE BALL-SILENCE DRAIN — none of them is
    an additive drain of its own.  With the ball visibly in play they drain
    nothing, however confident they are.  An additive walk term would
    reintroduce exactly the mid-rally-walking failure that `walk_ball_veto_s`
    exists to suppress, and mid-rally walking is common; the same holds for
    every near_end signal, since a player turns away, stands up and stops
    moving during live tennis too.  This is the load-bearing decision in the
    file — it is what lets more near-player evidence be added without each
    addition being a new way to cut a live rally in half.

  * THE MOTION GAIN IS THE NON-WALKING SHARE OF MOTION, `(1 - walk_prob)`
    gated.  Walking is player motion, so an ungated gain and the walk-amplified
    drain would be fighting over the same evidence.

  * WEIGHTS ARE PER SECOND and multiplied by dt.  `energy_step_s` is then a
    resolution knob rather than a strength knob, and changing it does not
    invalidate a fitted set of weights.  Nothing here may be applied per step.

  * `energy_start` AND `energy_floor` ARE FROZEN, never swept.  Scaling every
    weight by k with the two levels fixed is identical to rescaling
    time-to-drain by 1/k, so the parameters are unidentifiable together: a
    sweep over both is mostly duplicate configurations, and the duplicates
    manufacture leave-one-out fold disagreement out of nothing.

Steps are binned RELATIVE to the point start.  `deadtime_confidence` bins on
absolute `int(t)` seconds, which makes the same point behave differently
depending on where in the clip it happens to fall — an artefact, not a signal.

The end is stamped BACKWARDS at `drain_start + energy_stamp_s`, not where the
bar reaches the floor.  The point stopped when the evidence stopped; stamping at
the crossing would make the reported end a function of the drain rate, which
couples timing to every weight and leaves the sweep chasing two objectives with
one knob.  It is the same construction as the trace policy's
`gap_start + trace_stamp_s`, the arm with a +0.01 s event median.

The bar to clear is that same arm: over Data/21,22,23 it scores recall 62%,
precision 76%, point median +0.82 s and ZERO truncations.  Truncations are the
acceptance gate — an energy arm that buys recall with a truncation is a
regression, however good the other columns look.  As tuned this cleared it at
64% / 75% / +0.92 and became the default on 2026-08-21.

Adding the near_end signals the same day moves it again, over the wider
Data/21,22,23,24,43 (62 ends, detected starts):

    ball + walking only        61% / 73% / +1.38 / 0 truncations
    + near-player signals      66% / 77% / +1.04 / 0 truncations

and, unlike the weights above, that gain survives leave-one-clip-out (65% / 77%
/ 0, four of five folds picking the same configuration).  ReelConfig carries
both tables, including the leave-one-out result the ORIGINAL adoption was made
in spite of.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..near_end import SIGNAL_NAMES as NEAR_SIGNALS
from .config import ReelConfig


# ── evidence ─────────────────────────────────────────────────────────────

@dataclass
class EnergyEvidence:
    """Clip-global signals, in the form the per-step binner wants.

    Built once per clip and independent of every swept weight, so the tuner
    scores a whole grid without touching perception or re-binning.
    """
    pose_t: np.ndarray          # timestamps of pose records
    pose_near: np.ndarray       # bool, near box present at that pose record
    speed_t: np.ndarray         # timestamps where a world speed is defined
    speed: np.ndarray           # near-player world speed, ft/s
    reversals: np.ndarray       # direction-reversal timestamps
    look_t: np.ndarray          # timestamps where the ball model looked
    walk_prob: np.ndarray       # per-frame walking probability
    walk_fps: float
    intervals: List[Tuple[float, float]]        # in-court IMM trace spans
    interval_starts: np.ndarray
    # The four near-player point-end signals (pipeline/near_end.py), each [N] in
    # [0, 1] on the SAME rows as walk_prob — both are per near-player pose
    # sample — so they are indexed with walk_fps and need no timeline of their
    # own.  Empty when the module was not run, which is what makes them
    # optional to every caller.
    near: Dict[str, np.ndarray]


def _kinematics(meta: Dict, records: Sequence[Dict]):
    """Near-player speed and reversal times from `npw`, via PlayerKinematics.

    Deliberately built over the POSE records rather than the ball records
    `ball_trace.trace_intervals` feeds the tracker: `npw` only ever appears on a
    pose sample, so a ball-rate frame list would hand PlayerKinematics a stream
    that is ~75% empty at ball_fps 30 and pose_fps 15, and its sliding speed
    window is indexed on the records it is given.
    """
    from ..point_segmenter import (FrameRecord, MatchTelemetry, PlayerKinematics,
                                   SegmenterConfig)

    frames = [FrameRecord(f=int(r.get("f", 0)), t=float(r["t"]),
                          near_box=r.get("np"), near_world=r.get("npw"),
                          far_box=None, far_held=False, far_world=None,
                          balls=[], toss=[], trophy=0.0, stgcn=0.0)
              for r in records if r.get("pn")]
    if not frames:
        return np.empty(0), np.empty(0), np.empty(0)

    pose_fps = float(meta.get("pose_fps") or 15.0)
    match = MatchTelemetry({"fps": pose_fps, "stride": 1,
                            "analysis_size": meta.get("analysis_size", [960, 540])},
                           frames)
    kin = PlayerKinematics(match, SegmenterConfig())

    ts, sp = [], []
    for i, fr in enumerate(frames):
        v = kin.speed_near[i]
        if v is not None:
            ts.append(fr.t)
            sp.append(float(v))
    return (np.asarray(ts, dtype=float), np.asarray(sp, dtype=float),
            np.asarray(kin.rally_cues, dtype=float))


def build_evidence(meta: Dict, records: Sequence[Dict], walk_result: Dict,
                   intervals: Sequence[Tuple[float, float]],
                   near: Optional[Dict[str, np.ndarray]] = None) -> EnergyEvidence:
    """Everything the bar reads, from one telemetry pass and one walking pass.

    `near` is `pipeline.near_end.near_signals`' output — the four non-walking
    near-player signals.  Optional, and absent it the bar behaves exactly as it
    did before they existed, which is what lets an A/B be a weight change rather
    than a code path.
    """
    pose_t = np.asarray([float(r["t"]) for r in records if r.get("pn")], dtype=float)
    pose_near = np.asarray([bool(r.get("np")) for r in records if r.get("pn")], dtype=bool)
    look_t = np.asarray([float(r["t"]) for r in records if r.get("bn")], dtype=float)
    speed_t, speed, reversals = _kinematics(meta, records)
    ivals = [(float(a), float(b)) for a, b in intervals]
    walk_prob = np.asarray(walk_result["prob"], dtype=float)
    near_arrays: Dict[str, np.ndarray] = {}
    for name in NEAR_SIGNALS:
        arr = None if near is None else near.get(name)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if len(arr) != len(walk_prob):
            # The two come off the same pose npz, so a mismatch means they came
            # off DIFFERENT ones — a stale cache, most likely.  Silently
            # truncating would leave every signal offset from the walking it is
            # supposed to complement, and an offset signal still scores.
            raise ValueError(
                f"near_end signal '{name}' has {len(arr)} rows against "
                f"{len(walk_prob)} of walking: they must come from the same "
                f"near-player pose pass")
        near_arrays[name] = arr
    return EnergyEvidence(
        pose_t=pose_t, pose_near=pose_near,
        speed_t=speed_t, speed=speed, reversals=reversals, look_t=look_t,
        walk_prob=walk_prob,
        walk_fps=float(walk_result["fps"]),
        intervals=ivals,
        interval_starts=np.asarray([a for a, _ in ivals], dtype=float),
        near=near_arrays)


def _overlap(ev: EnergyEvidence, a: float, b: float) -> float:
    """Seconds of [a, b) covered by a trace interval."""
    if not ev.intervals:
        return 0.0
    total = 0.0
    i = max(0, int(np.searchsorted(ev.interval_starts, a, side="right")) - 1)
    for j in range(i, len(ev.intervals)):
        s, e = ev.intervals[j]
        if s >= b:
            break
        total += max(0.0, min(e, b) - max(s, a))
    return total


def _pose_rate_mean(arr: np.ndarray, fps: float, a: float, b: float) -> float:
    """Mean of a per-pose-sample array over [a, b) seconds.

    Every array on that timeline — walk_prob and the four near_end signals —
    is indexed the same way, by construction: they all come off the rows of one
    near-player pose npz.  One helper so a change to the indexing cannot apply
    to walking and miss the signals meant to complement it.
    """
    i0 = int(round(a * fps))
    i1 = max(i0 + 1, int(round(b * fps)))
    seg = arr[min(i0, len(arr)):min(i1, len(arr))]
    return float(seg.mean()) if seg.size else 0.0


def bin_point(ev: EnergyEvidence, start_t: float, stop_t: float,
              cfg: ReelConfig) -> Dict[str, np.ndarray]:
    """Per-step evidence for one point, binned relative to its start.

    Weight-independent by construction: the tuner bins once per point and then
    sweeps the whole grid over these arrays.
    """
    dt = float(cfg.energy_step_s)
    n = max(1, int(np.ceil((stop_t - start_t) / dt)))
    edges = start_t + dt * np.arange(n + 1)

    trace = np.empty(n)
    walk = np.empty(n)
    motion = np.empty(n)
    speed = np.empty(n)
    rev = np.empty(n)
    near_miss = np.empty(n)
    n_looks = np.empty(n, dtype=int)
    n_pose = np.empty(n, dtype=int)
    # Zeros where the module was not run, so every downstream reader sees the
    # full set of keys and a weight on an absent signal is simply inert.
    near_sig = {name: np.zeros(n) for name in NEAR_SIGNALS}

    pose_lo = np.searchsorted(ev.pose_t, edges[:-1], side="left")
    pose_hi = np.searchsorted(ev.pose_t, edges[1:], side="left")
    sp_lo = np.searchsorted(ev.speed_t, edges[:-1], side="left")
    sp_hi = np.searchsorted(ev.speed_t, edges[1:], side="left")
    look_lo = np.searchsorted(ev.look_t, edges[:-1], side="left")
    look_hi = np.searchsorted(ev.look_t, edges[1:], side="left")
    rev_lo = np.searchsorted(ev.reversals, edges[:-1], side="left")
    rev_hi = np.searchsorted(ev.reversals, edges[1:], side="left")

    for k in range(n):
        a, b = float(edges[k]), float(edges[k + 1])
        span = max(b - a, 1e-9)
        trace[k] = min(1.0, _overlap(ev, a, b) / span)
        n_looks[k] = int(look_hi[k] - look_lo[k])
        n_pose[k] = int(pose_hi[k] - pose_lo[k])

        walk[k] = _pose_rate_mean(ev.walk_prob, ev.walk_fps, a, b)
        for name in NEAR_SIGNALS:
            arr = ev.near.get(name)
            if arr is not None:
                near_sig[name][k] = _pose_rate_mean(arr, ev.walk_fps, a, b)

        s = ev.speed[sp_lo[k]:sp_hi[k]]
        speed[k] = float(s.mean()) if s.size else 0.0
        rev[k] = min(1.0, (rev_hi[k] - rev_lo[k]) /
                     max(cfg.energy_reversal_ref_hz * span, 1e-9))

        if n_pose[k]:
            seen = int(ev.pose_near[pose_lo[k]:pose_hi[k]].sum())
            near_miss[k] = 1.0 - seen / n_pose[k]
        else:
            near_miss[k] = 0.0

    # The gain is the NON-WALKING share of motion, so gain and the
    # walk-amplified drain can never both be near-maximal on one step.
    motion = np.clip(speed / max(cfg.energy_motion_ref_ft_s, 1e-9), 0.0, 1.0) * (1.0 - walk)

    return {
        "t": edges[:-1], "dt": dt, "n": n,
        "trace": trace, "walk": walk, "motion": motion, "speed": speed,
        "reversal": rev, "near_miss": near_miss, **near_sig,
        "n_looks": n_looks, "n_pose": n_pose,
        # A step the ball model never looked at is not evidence of silence, so
        # it holds the BALL TERM ONLY.  `deadtime_confidence` freezes the whole
        # second instead, which makes the bar hostage to ball sampling even
        # though the pose evidence in that step is perfectly good.
        "ball_held": n_looks < cfg.energy_min_looks_per_step,
        "near_held": n_pose == 0,
    }


# ── integration ──────────────────────────────────────────────────────────

@dataclass
class PointEnergy:
    """Integrated bar for one point, plus everything the onset rule reads."""
    energy: np.ndarray
    delta: np.ndarray
    gain: np.ndarray
    drain_ball: np.ndarray
    drain_near: np.ndarray
    idle: np.ndarray             # the near_end signals' combined, capped
                                 # contribution to the silence multiplier
    drain_start: np.ndarray      # start time of the current non-positive run
    floor_since: np.ndarray      # when the current floor visit began; nan if not
    n: int                       # steps actually integrated (see `stop_at_end`)


def integrate(bins: Dict, cfg: ReelConfig, stop_at_end: bool = False) -> PointEnergy:
    """Run the bar forward over one point's binned evidence.

    `stop_at_end` returns as soon as the onset rule is satisfiable, which is
    what makes a full-grid sweep affordable — most points end long before their
    span does.  The reel keeps it False so the debug rows cover the whole span.
    """
    n = int(bins["n"])
    dt = float(bins["dt"])
    t = bins["t"]
    trace, walk = bins["trace"], bins["walk"]
    motion, rev, near_miss = bins["motion"], bins["reversal"], bins["near_miss"]
    ball_held, near_held = bins["ball_held"], bins["near_held"]

    hold_s = float(cfg.energy_hold_s)
    floor, top = float(cfg.energy_floor), float(cfg.energy_max)
    w_motion, w_rev = float(cfg.energy_motion_weight), float(cfg.energy_reversal_weight)
    w_ball, boost = float(cfg.energy_ball_weight), float(cfg.energy_walk_boost)
    w_near = float(cfg.energy_near_missing_weight)
    # The four near_end signals boost the silence drain exactly as walking does,
    # each with its own weight, and their combined contribution is capped.
    # Without the cap four saturated signals could add ~4x the walk boost's
    # authority to a single step and drive the drain straight into
    # energy_max_drop_per_s, at which point the weights stop being separable —
    # the same identifiability trap the boost/cap pair already has.
    near_w = [(name, float(getattr(cfg, f"energy_{name}_weight", 0.0)))
              for name in NEAR_SIGNALS]
    near_cap = float(cfg.energy_near_signal_cap)
    up, down = float(cfg.energy_max_rise_per_s), float(cfg.energy_max_drop_per_s)
    confirm_s = float(cfg.energy_confirm_s)

    energy = np.zeros(n)
    delta = np.zeros(n)
    gain_a = np.zeros(n)
    dball_a = np.zeros(n)
    dnear_a = np.zeros(n)
    idle_a = np.zeros(n)
    dstart_a = np.full(n, np.nan)
    fsince_a = np.full(n, np.nan)

    e = float(cfg.energy_start)
    t0 = float(t[0]) if n else 0.0
    drain_start: Optional[float] = None
    floor_since: Optional[float] = None
    used = n

    for k in range(n):
        tk = float(t[k])
        if (k + 1) * dt <= hold_s:
            gain = dball = dnear = idle = 0.0
            de = 0.0
            drain_start = t0 + hold_s
        else:
            gain = w_motion * float(motion[k]) + w_rev * float(rev[k])
            idle = min(near_cap, sum(w * float(bins[name][k])
                                     for name, w in near_w if w))
            dball = 0.0 if ball_held[k] else (
                w_ball * (1.0 - float(trace[k]))
                * (1.0 + boost * float(walk[k]) + idle))
            dnear = 0.0 if near_held[k] else w_near * float(near_miss[k])
            rate = max(-down, min(up, gain - dball - dnear))
            de = rate * dt
            if de > 0.0:
                drain_start = None
            elif drain_start is None:
                drain_start = tk

        e = max(floor, min(top, e + de))
        if e <= floor + 1e-9:
            if floor_since is None:
                floor_since = tk
        else:
            floor_since = None

        energy[k] = e
        delta[k] = de
        gain_a[k] = gain
        dball_a[k] = dball
        dnear_a[k] = dnear
        idle_a[k] = idle
        dstart_a[k] = np.nan if drain_start is None else drain_start
        fsince_a[k] = np.nan if floor_since is None else floor_since

        if stop_at_end and floor_since is not None and tk - floor_since >= confirm_s:
            used = k + 1
            break

    return PointEnergy(energy, delta, gain_a, dball_a, dnear_a, idle_a,
                       dstart_a, fsince_a, used)


def point_end(bins: Dict, res: PointEnergy, start_t: float,
              cfg: ReelConfig) -> Optional[Dict]:
    """The first sustained floor visit, stamped back at the drain that caused it.

    Returns None when the bar never empties: that point contributes no onset and
    falls through to `find_point_end`'s guards, which is the honest reading — the
    evidence never became convincing.
    """
    confirm_s = float(cfg.energy_confirm_s)
    for k in range(res.n):
        fs = res.floor_since[k]
        if np.isnan(fs):
            continue
        tk = float(bins["t"][k])
        if tk - float(fs) < confirm_s:
            continue
        ds = res.drain_start[k]
        anchor = start_t + cfg.energy_hold_s if np.isnan(ds) else float(ds)
        t_end = min(tk, max(anchor, start_t + cfg.energy_hold_s) + cfg.energy_stamp_s)
        lo = max(0, k - int(round(cfg.energy_confirm_s / bins["dt"])) - 1)
        walking = float(np.mean(bins["walk"][lo:k + 1])) if k >= lo else 0.0
        # The non-walking near-player evidence promotes an end the same way
        # walking does, measured in walking's own units: `idle` and the walk
        # boost multiply the same drain, so half-confident walking is the
        # natural bar for "the player has visibly stopped playing".  With the
        # near_end weights at their default zero this term is zero and the
        # grading is bit-for-bit what it was before the signals existed.
        idle = float(np.mean(res.idle[lo:k + 1])) if k >= lo else 0.0
        # `idle > 0` is not redundant: with energy_walk_boost swept to zero the
        # bar to clear would otherwise be zero, and every end would grade high
        # on no evidence at all.
        idle_high = idle > 0.0 and idle >= 0.5 * float(cfg.energy_walk_boost)
        level = "high" if (walking >= 0.5 or idle_high) else "medium"
        reason = "energy drained"
        if walking >= 0.5:
            reason += ", walking"
        if idle_high:
            reason += ", near player idle"
        return {
            "t": round(float(t_end), 3),
            "source": "energy",
            "level": level,
            "reason": reason,
            "idle": round(idle, 4),
            "drain_start_s": round(float(anchor), 3),
            "cross_s": round(tk, 3),
            "min_energy": round(float(res.energy[:k + 1].min()), 4),
        }
    return None


# ── the reel's entry points ──────────────────────────────────────────────

def score_energy(starts: Sequence, duration: float, ev: EnergyEvidence,
                 cfg: ReelConfig) -> Tuple[List[Dict], List[Dict]]:
    """(debug rows, onset details) for every interval between point starts."""
    rows: List[Dict] = []
    details: List[Dict] = []
    for i, start in enumerate(starts):
        start_t = float(start.t)
        stop_t = float(starts[i + 1].t) if i + 1 < len(starts) else duration
        if stop_t <= start_t:
            continue
        bins = bin_point(ev, start_t, stop_t, cfg)
        res = integrate(bins, cfg, stop_at_end=not cfg.energy_debug_rows)
        found = point_end(bins, res, start_t, cfg)
        if found is not None:
            details.append({"point_index": i, **found})
        if not cfg.energy_debug_rows:
            continue
        for k in range(res.n):
            rows.append({
                **{name: round(float(bins[name][k]), 4) for name in NEAR_SIGNALS},
                "idle": round(float(res.idle[k]), 4),
                "point_index": i,
                "point_start_s": round(start_t, 3),
                "elapsed_s": round(k * bins["dt"], 3),
                "timestamp_s": round(float(bins["t"][k]), 3),
                "energy": round(float(res.energy[k]), 4),
                "delta": round(float(res.delta[k]), 5),
                "gain": round(float(res.gain[k]), 4),
                "drain_ball": round(float(res.drain_ball[k]), 4),
                "drain_near": round(float(res.drain_near[k]), 4),
                "trace_frac": round(float(bins["trace"][k]), 4),
                "walk_prob": round(float(bins["walk"][k]), 4),
                "motion": round(float(bins["motion"][k]), 4),
                "speed_ft_s": round(float(bins["speed"][k]), 3),
                "reversal_rate": round(float(bins["reversal"][k]), 4),
                "near_miss": round(float(bins["near_miss"][k]), 4),
                "n_looks": int(bins["n_looks"][k]),
                "n_pose": int(bins["n_pose"][k]),
                "ball_held": bool(bins["ball_held"][k]),
                "near_held": bool(bins["near_held"][k]),
            })
    return rows, details


def energy_onsets(details: Sequence[Dict]) -> List[Tuple[float, str]]:
    """Dead onsets as [(t, "energy")], one per point that emptied.

    Onsets rather than ends, exactly as `deadtime_confidence.deadtime_onsets`:
    `find_point_end` keeps ownership of point_min_s, point_max_s and
    next_serve_guard_s, so this policy inherits them instead of quietly
    reimplementing three guards that are already right.

    Non-monotonicity lives entirely below this line.  Within a point the bar may
    rise and fall freely; here only the first sustained floor visit matters, and
    `point_end` returns at most one per point, so a later recovery cannot
    retract or duplicate an end that has already been emitted.
    """
    return [(float(d["t"]), "energy") for d in details]
