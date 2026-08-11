"""Far-side serve detection from an anya_telemetry JSONL stream.

A far serve is claimed when the far player, while armed (held station at the
baseline — see anya_far_serve's earlier arming gate), raises a hand from
below to above the shoulder within a short window.  That pose-rise is the
primary gate.  A far-to-near ball trace is no longer required to fire: if one
follows within POSE_TRACE_CONFIRM_S it upgrades the detection's confidence
from MEDIUM to HIGH, but its absence does not suppress the detection.

This replaces ball-trace verification as the primary gate because ball
detection turned out to be the binding constraint (2 of the far serves we
could not recall had zero surviving ball detections anywhere near the serve),
while the far player is reliably tracked via the native-resolution ROI
(`fprw`/`fpr`, from anya_telemetry v2).  Feasibility was spot-checked before
building this: over 2s windows around 11 confirmed serves, wrist-above-
shoulder margin reached 5.8-27px with the arm raised across many sampled
frames; 10 negative-control windows mostly stayed at/below a ~5px noise
floor. Pose keypoints come from extract_far_pose.py, a separate offline pass
over the `fpr` crop — run that first, or this module has nothing to gate on.

The raise metric is normalised by torso length and median-smoothed over a
short trailing window before thresholding; see FarServeDetectorConfig for
why both matter.

Two things have to happen before the state machine sees a usable ball stream
(now only used for confirmatory scoring, not gating):

  1. Exclusion zones from the meta header are rescaled into analysis
     coordinates.  The zone cache is shared with the full-resolution pipeline,
     so a cached entry is in SOURCE video pixels while `all_balls` is in
     `analysis_size` pixels — applying them unscaled filters nothing.

  2. Static false positives that the cached zones missed are learned from the
     telemetry itself.  The zone scan only samples 50 frames, and on real
     clips it misses persistent blobs (scoreboard glints, line markers) that
     fire in thousands of frames and destroy any quiet-period test.

Run:
    python -m pipeline.extract_far_pose match_anya_telemetry.jsonl   # once
    python -m pipeline.anya_far_serve match_anya_telemetry.jsonl [--eval gt.json]
"""

import os
import json
import argparse
from enum import Enum
from collections import defaultdict, deque
from typing import List, Dict, Any, Optional, Tuple

try:                                        # package import (python -m pipeline.x)
    from .extract_far_pose import (far_pose_path_for, load_far_pose,
                                   L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST,
                                   L_HIP, R_HIP)
except ImportError:                         # script import (python pipeline/x.py)
    from extract_far_pose import (far_pose_path_for, load_far_pose,
                                  L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST,
                                  L_HIP, R_HIP)

# The original stillness gate was a per-frame velocity test (lateral speed
# under 1.5 ft/s for 0.6 s, differenced frame-to-frame at ~60 fps).  That
# admits only 0.025 ft of homography jitter per frame against an observed
# median of 1.2 ft/s, so it fired on 6 frames out of 25k.  The arming gate
# below replaces it with a displacement test over a 2 s window, which is
# insensitive to per-frame jitter, fed by a much sharper player track.


# Fast-path raise-gate settings, from a sweep over the ground-truthed corpus;
# see FarServeDetectorConfig.for_fast_path.  Kept as module constants so a
# re-sweep touches one place.
#
# Swept 16 (window, ratio) pairs against the clips where BOTH extractors have
# been run, which is the only fair comparison — an absolute F1 over the whole
# corpus rewards whichever setting emits fewer detections on clips the far
# detector cannot solve at all (see the clip-25 note below).
FAST_SMOOTH_WINDOW_S:  float = 0.20
FAST_RAISE_RATIO:      float = 0.65
FAST_MIN_TRACE_FRAMES: int   = 2       # a 10 fps ball stream rarely lands 3
                                       # detections inside one serve's flight


class SystemState(Enum):
    IDLE = "IDLE"              # Waiting for (armed + hand-raise) to fire
    CONFIRMING = "CONFIRMING"  # Serve already recorded; watching for a ball
                                # trace to upgrade confidence to HIGH


class FarServeDetectorConfig:
    # Ball filtering.  Defaults below were swept for F1 on Data/23; the far
    # serve ball is only a few pixels at 540p and scores 0.10-0.45, so the
    # confidence floor buys quiet periods without erasing the serve itself.
    BALL_CONF_MIN: float = 0.12            # Telemetry is extracted at 0.1
    STATIC_MIN_HITS: int = 40              # Blob calibration: min detections
    STATIC_MIN_RATE: float = 8.0           # ...per 10 s bucket it appears in
    STATIC_CELL_PX: int = 6                # Calibration grid size

    # Far-player arming.  The player must be visible in the far-baseline ROI
    # and hold station laterally for ARM_STABLE_S to be considered armed.
    FP_HYSTERESIS_S: float = 1.0           # Tolerated gap in ROI detection
    ARM_STABLE_S: float = 1.0              # Lateral hold required to arm
    ARM_MAX_DRIFT_FT: float = 3.0          # World-x drift allowed over that hold
    ARM_TO_TRACE_S: float = 2.5            # How long "armed" stays level-true
                                           # after the last qualifying sample

    # Hand-raise gate (primary trigger).
    #
    # The metric is scale-invariant: wrist elevation above the shoulder line,
    # divided by torso length (shoulder-to-hip).  The raw pixel margin it
    # replaces was not comparable across the clip — the far player's box
    # ranges from ~35 to ~92 px tall as they move and the camera drifts, so a
    # fixed pixel threshold meant quite different postures at either end.
    #
    # The signal is median-smoothed over a trailing window before thresholding.
    # Raw per-frame keypoints jitter hard at this crop size (observed swings
    # of -25px to +16px between adjacent frames, which is not physically
    # possible for an arm at 60 fps), and that jitter — not the absence of a
    # raise — was breaking the old consecutive-frame run counter.  With the
    # spikes filtered, a single threshold crossing of the smoothed signal is
    # enough: no consecutive-frame requirement.
    # RAISE_RATIO/SMOOTH_WINDOW_S below are the F1 peak of a 72-point sweep on
    # Data/23 (15 far serves, one clip) — treat as provisional until a second
    # clip is scored.  Raw ratios there sit at p50=-0.74 / p95=+0.17, so 0.35
    # is well clear of the resting distribution.  A longer median window blurs
    # the rise itself: 0.25s costs two serves outright.
    KPT_CONF_MIN: float = 0.30             # Per-keypoint confidence floor
    SMOOTH_WINDOW_S: float = 0.10          # Trailing median window (~6 frames)
    SMOOTH_MIN_SAMPLES: int = 3            # Below this the median is untrusted
    RAISE_RATIO: float = 0.35              # Wrist-above-shoulder, in torso lengths
    TORSO_FRAC_OF_BOX: float = 0.30        # Fallback scale when hips are unusable

    # Ball trace: no longer gates detection, only confirms it (HIGH vs MEDIUM
    # confidence) if a valid trace starts within POSE_TRACE_CONFIRM_S of the
    # hand-raise event.
    POSE_TRACE_CONFIRM_S: float = 2.5
    MIN_TRACE_FRAMES: int = 3              # Frames required to confirm trajectory
    MAX_TRACE_GAP_S: float = 0.6           # Max gap between ball detections
    FAR_COURT_Y_SPLIT: float = 270.0       # Analysis y < 270 is far court (540p)
    MIN_TRACE_DROP_PX: float = 8.0         # Net downward travel to confirm
    MAX_STEP_PX: float = 60.0              # Continuity gate between trace points
    MAX_TRACE_DURATION_S: float = 1.2      # A serve crosses the court in well
                                           # under this; past it the trace is
                                           # into the following rally shot and
                                           # confirmation is abandoned so a
                                           # fresh attempt can be made
    MIN_SERVE_SEPARATION_S: float = 3.0    # Refractory between reported serves

    # Rally-state exclusion.  A detected serve opens a point; while that point
    # is live, further hand-raises are suppressed — mid-rally overheads and
    # smashes look exactly like a serve to the pose gate, and on Data/23 four
    # of eight false positives were arm raises 4.8-9.1s into a live point.
    #
    # The point ends when the ball has been silent for POINT_END_QUIET_S
    # (bounded below by POINT_MIN_S, above by POINT_MAX_S).  Keep the quiet
    # threshold SHORT: a long-running point that never ends will swallow the
    # next real serve.  Sweeping to 3.0s scored a better F1 purely because one
    # false positive opened a point that ate the next two true serves — fewer
    # detections, but the wrong ones.
    POINT_MIN_S: float = 3.0               # A point cannot end before this
    POINT_END_QUIET_S: float = 1.5         # Ball silent this long ends the point
    POINT_MAX_S: float = 30.0              # Hard cap, so a stuck point recovers

    @classmethod
    def for_fast_path(cls) -> "FarServeDetectorConfig":
        """Preset for telemetry from anya_far_telemetry rather than the full pass.

        The fast path's pose crops are canonicalised and read out of a
        re-encoded band, so its raise signal is not the same signal these
        defaults were swept against: it crosses the 0.35 threshold far more
        often (67 times against the full pass's 44 on Data/23) while holding
        on to the true serves much better under a stricter gate.  So the
        settings move in opposite directions on the two streams — at
        RAISE_RATIO 0.55 the full pass drops to 12/15 recall, while the fast
        path is still 15/15.

        Only the raise gate is re-tuned; every other threshold is the shipped
        one.  Measured over the ten clips where both extractors have run, 77
        ground-truthed far serves (DESIGN.md 8.5 has the per-clip table):

            full pass, shipped thresholds   41/77 recall, 37 FP
            fast path, this preset          54/77 recall, 29 FP

        i.e. better on both axes, which is the bar it had to clear.  Raising
        recall two more serves costs nine more false positives
        (RAISE_RATIO 0.55 -> 56/77 at 38 FP) if that trade is wanted.

        A caution that applies to both presets: clip 25's 10 far serves are
        invisible to BOTH extractors (0/10 either way), so any threshold fitted
        on absolute corpus F1 gets rewarded for simply detecting less.  Fit
        against the full pass, clip for clip, not against the corpus total.
        """
        cfg = cls()
        cfg.SMOOTH_WINDOW_S = FAST_SMOOTH_WINDOW_S
        cfg.RAISE_RATIO     = FAST_RAISE_RATIO
        cfg.MIN_TRACE_FRAMES = FAST_MIN_TRACE_FRAMES
        return cfg


def scale_exclusion_zones(zones, meta) -> List[Tuple[float, float, float, float]]:
    """Rescales meta exclusion zones into analysis coordinates.

    Zones already inside the analysis frame are returned untouched.  Anything
    larger came from the full-resolution pipeline's shared cache and needs the
    source frame size, taken from `meta["source_size"]` when present and
    otherwise probed from the sibling video file.
    """
    if not zones:
        return []

    aw, ah = meta.get("analysis_size", [960, 540])
    max_x = max(z[2] for z in zones)
    max_y = max(z[3] for z in zones)
    if max_x <= aw and max_y <= ah:
        return [tuple(float(v) for v in z) for z in zones]

    src = meta.get("source_size")
    if not src:
        src = _probe_source_size(meta)
    if not src:
        print("[FAR-SERVE] WARN: exclusion zones are in source coordinates but "
              "the source size is unknown — zone filtering disabled.")
        return []

    sx, sy = aw / float(src[0]), ah / float(src[1])
    return [(z[0] * sx, z[1] * sy, z[2] * sx, z[3] * sy) for z in zones]


def _probe_source_size(meta) -> Optional[Tuple[int, int]]:
    video = meta.get("_video_path")
    if not video or not os.path.exists(video):
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(video)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (w, h) if w and h else None
    except Exception:
        return None


def ball_sampling_scales(meta) -> Tuple[float, float]:
    """(hits_scale, rate_scale) for a telemetry file's ball sampling.

    anya_telemetry runs the ball model on every frame; anya_far_telemetry runs
    it on a fraction of them (a reduced rate, and only where a point could be
    live).  The static-blob thresholds are counts of detections, so they mean
    something different against a sparser stream and have to be scaled or a
    persistent false positive stops looking persistent — which would leave a
    scoreboard glint keeping every point alive forever.

    `rate_scale` is the sampling rate (detections per unit time scale with
    it); `hits_scale` is the share of the clip's frames actually looked at,
    which is what the absolute STATIC_MIN_HITS floor scales with.  Both are
    1.0 for a full telemetry file, so its calibration is unchanged.
    """
    stride  = max(1, int(meta.get("ball_stride") or 1))
    total   = int(meta.get("total_frames") or 0)
    sampled = int(meta.get("ball_frames") or 0)
    rate_scale = 1.0 / stride
    hits_scale = (sampled / total) if (sampled and total) else rate_scale
    return hits_scale, rate_scale


def calibrate_static_blobs(records, cfg: FarServeDetectorConfig,
                           hits_scale: float = 1.0, rate_scale: float = 1.0):
    """Learns persistent false-positive cells from the whole telemetry stream.

    A real ball crosses any given cell a handful of times per pass; a static
    false positive fires in most frames it is visible for.  Scoring by
    detections-per-active-bucket separates the two cleanly (observed: static
    cells score 100+, the densest genuine ball cell scores under 4).

    The two scales carry a sub-sampled ball stream; see `ball_sampling_scales`.
    """
    hits = defaultdict(int)
    buckets = defaultdict(set)
    cell = cfg.STATIC_CELL_PX
    for r in records:
        bucket = int(r["t"] // 10)
        for b in r.get("all_balls", []):
            key = (int(b[0] // cell), int(b[1] // cell))
            hits[key] += 1
            buckets[key].add(bucket)
    min_hits = cfg.STATIC_MIN_HITS * hits_scale
    min_rate = cfg.STATIC_MIN_RATE * rate_scale
    return {
        k for k, n in hits.items()
        if n >= min_hits and n / len(buckets[k]) >= min_rate
    }


class FarServeDetector:
    """Far-side serve detector: quiet period, then a far-to-near ball trace."""

    def __init__(self, config: FarServeDetectorConfig = None):
        self.cfg = config or FarServeDetectorConfig()
        self.state = SystemState.IDLE

        self.exclusion_zones: List[Tuple[float, float, float, float]] = []
        self.static_cells = set()

        # Ball state.  Starts far in the past so a serve in the opening second
        # is reachable, unlike a 0.0 sentinel which both blocks the first
        # window and arms a spurious trigger at t=0.
        self.last_ball_t: float = -1e9

        # Far-player arming state.
        self.fp_samples: deque = deque()      # (t, world_x) inside the ROI
        self.last_fp_t: float = -1e9
        self.last_armed_t: float = -1e9
        self.arm_events: List[float] = []
        self.fp_key: str = "fprw"

        # Hand-raise state.
        self.pose_samples: deque = deque()    # (t, raw_ratio) — only non-None
        self.smoothed: Optional[float] = None      # current median
        self.prev_smoothed: Optional[float] = None # previous, for edge detection
        self.raise_events: List[float] = []
        self.pose_cache: Dict[int, Any] = {}  # frame -> far-pose record

        # Ball-trace confirmation state (only active in CONFIRMING).
        self.trace_points: List[Tuple[float, float, float]] = []  # (t, cx, cy)
        self.last_serve_t: float = -1e9
        self._pending_serve: Optional[Dict[str, Any]] = None

        # Rally state.  Independent of `state`: CONFIRMING only governs the
        # confidence upgrade window, while a point stays live long after it.
        self.point_live: bool = False
        self.point_start_t: float = -1e9
        self.last_any_ball_t: float = -1e9
        self.suppressed_raises: List[float] = []

        self.detected_serves: List[Dict[str, Any]] = []

    def set_exclusion_zones(self, zones):
        self.exclusion_zones = [tuple(float(v) for v in z) for z in zones]

    def set_static_cells(self, cells):
        self.static_cells = set(cells)

    def set_pose_cache(self, cache: Dict[int, Optional[float]]):
        self.pose_cache = cache

    def _is_in_exclusion_zone(self, cx: float, cy: float) -> bool:
        for (x1, y1, x2, y2) in self.exclusion_zones:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return True
        return False

    def _is_static_blob(self, cx: float, cy: float) -> bool:
        if not self.static_cells:
            return False
        # Test the 3x3 neighbourhood: a blob drifts a few pixels as the camera
        # settles and would otherwise leak through on the cell boundary.
        kx, ky = int(cx // self.cfg.STATIC_CELL_PX), int(cy // self.cfg.STATIC_CELL_PX)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (kx + dx, ky + dy) in self.static_cells:
                    return True
        return False

    def _filter_balls(self, all_balls) -> List[Tuple[float, float, float]]:
        """Drops low-confidence detections and known static false positives."""
        valid = []
        for (cx, cy, conf) in all_balls:
            if conf < self.cfg.BALL_CONF_MIN:
                continue
            if self._is_in_exclusion_zone(cx, cy):
                continue
            if self._is_static_blob(cx, cy):
                continue
            valid.append((cx, cy, conf))
        return valid

    def _pick_ball(self, valid_balls):
        """Highest-confidence ball, or the one continuing the current trace."""
        if not self.trace_points:
            return max(valid_balls, key=lambda b: b[2])
        _, px, py = self.trace_points[-1]
        near = [b for b in valid_balls
                if abs(b[0] - px) <= self.cfg.MAX_STEP_PX
                and abs(b[1] - py) <= self.cfg.MAX_STEP_PX]
        if not near:
            return None
        return max(near, key=lambda b: b[2])

    # -----------------------------------------------------------------
    # FAR-PLAYER ARMING
    # -----------------------------------------------------------------
    def _update_arming(self, record: Dict[str, Any], t: float) -> bool:
        """Arms at time t iff the far player held station over [t-2s, t].

        "Held station" means: present in the far-baseline ROI throughout (a
        detection gap of up to FP_HYSTERESIS_S is tolerated), and every world-x
        sample in the window within ARM_MAX_DRIFT_FT of the anchor sample's
        world-x, where the anchor is the sample at the start of the window.
        Arming is level-triggered, so a player who keeps standing still keeps
        re-arming and `last_armed_t` tracks the present.
        """
        cfg = self.cfg
        world = record.get(self.fp_key)
        if world:
            self.fp_samples.append((t, float(world[0])))
            self.last_fp_t = t

        # Keep one window's worth plus enough slack to find the anchor.
        horizon = cfg.ARM_STABLE_S + cfg.FP_HYSTERESIS_S
        while self.fp_samples and t - self.fp_samples[0][0] > horizon:
            self.fp_samples.popleft()

        # Player must be visible now (within hysteresis).
        if t - self.last_fp_t > cfg.FP_HYSTERESIS_S:
            return False

        anchor_t = t - cfg.ARM_STABLE_S
        anchor = None
        for s in self.fp_samples:
            if s[0] <= anchor_t:
                anchor = s        # latest sample at or before the window start
            else:
                break
        if anchor is None:
            return False

        window = [s for s in self.fp_samples if s[0] >= anchor[0]]
        if any(abs(wx - anchor[1]) > cfg.ARM_MAX_DRIFT_FT for _, wx in window):
            return False

        # No detection gap inside the window longer than the hysteresis.
        for (t0, _), (t1, _) in zip(window, window[1:]):
            if t1 - t0 > cfg.FP_HYSTERESIS_S:
                return False

        # Record rising edges only; `last_armed_t` carries the level.
        if t - self.last_armed_t > cfg.ARM_TO_TRACE_S:
            self.arm_events.append(t)
        self.last_armed_t = t
        return True

    # -----------------------------------------------------------------
    # HAND-RAISE GATE
    # -----------------------------------------------------------------
    def _raise_ratio(self, pose_rec) -> Optional[float]:
        """Wrist elevation above the shoulder, in torso lengths.

        Positive means the wrist is above the shoulder.  Taken as the better
        of the two arms, so it does not matter which hand tosses or hits.
        Normalising by torso length makes one threshold valid across the
        clip: the far player's on-screen size varies by ~2.5x.
        """
        if not pose_rec:
            return None
        cfg = self.cfg
        k = pose_rec["k"]

        def kp(i):
            x, y, c = k[3 * i], k[3 * i + 1], k[3 * i + 2]
            return (y, c) if c >= cfg.KPT_CONF_MIN else (None, c)

        (lsh_y, _), (rsh_y, _) = kp(L_SHOULDER), kp(R_SHOULDER)
        (lhp_y, _), (rhp_y, _) = kp(L_HIP), kp(R_HIP)

        shoulders = [y for y in (lsh_y, rsh_y) if y is not None]
        if not shoulders:
            return None
        shoulder_y = sum(shoulders) / len(shoulders)

        # Torso length, or a fraction of the box height when hips are not
        # confidently placed (common at this crop size).
        hips = [y for y in (lhp_y, rhp_y) if y is not None]
        scale = None
        if hips:
            scale = abs(sum(hips) / len(hips) - shoulder_y)
        if not scale or scale < 1.0:
            bh = pose_rec.get("bh")
            scale = bh * cfg.TORSO_FRAC_OF_BOX if bh else None
        if not scale or scale < 1.0:
            return None

        ratios = []
        for sh_i, wr_i in ((L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST)):
            (sh_y, _), (wr_y, _) = kp(sh_i), kp(wr_i)
            if sh_y is None or wr_y is None:
                continue
            ratios.append((sh_y - wr_y) / scale)
        return max(ratios) if ratios else None

    def _update_hand_raise(self, record: Dict[str, Any], t: float) -> bool:
        """True on the frame the smoothed raise ratio crosses the threshold.

        Edge-triggered on the median-smoothed signal, so it fires once per
        rise rather than on every frame the arm stays up.  The median does
        the work the old consecutive-frame counter was trying to do, but
        without letting a single jittery frame veto a real raise.
        """
        cfg = self.cfg
        ratio = self._raise_ratio(self.pose_cache.get(record["f"]))
        if ratio is not None:
            self.pose_samples.append((t, ratio))

        while self.pose_samples and t - self.pose_samples[0][0] > cfg.SMOOTH_WINDOW_S:
            self.pose_samples.popleft()

        self.prev_smoothed = self.smoothed
        if len(self.pose_samples) >= cfg.SMOOTH_MIN_SAMPLES:
            vals = sorted(v for _, v in self.pose_samples)
            self.smoothed = vals[len(vals) // 2]
        else:
            self.smoothed = None

        if self.smoothed is None or self.prev_smoothed is None:
            return False
        # Rising edge only: below-or-at threshold, then above.
        if self.prev_smoothed > cfg.RAISE_RATIO or self.smoothed <= cfg.RAISE_RATIO:
            return False

        self.raise_events.append(t)
        return True

    def _update_point_state(self, t: float, saw_ball: bool):
        """Closes a live point once the ball has gone quiet.

        Deliberately ball-quiet driven rather than kinematic: point_segmenter's
        richer find_point_end needs the v4 MatchTelemetry cache and its player
        kinematics, which this module does not carry.  The cap is the safety
        net for the case that matters — ball detection here is sparse enough
        that a point can otherwise appear to run forever.
        """
        if saw_ball:
            self.last_any_ball_t = t
        if not self.point_live:
            return
        duration = t - self.point_start_t
        if duration >= self.cfg.POINT_MAX_S:
            self.point_live = False
        elif (duration >= self.cfg.POINT_MIN_S and
              (t - self.last_any_ball_t) >= self.cfg.POINT_END_QUIET_S):
            self.point_live = False

    def _verify_far_to_near_trace(self) -> bool:
        """True when the trace starts in the far court and travels downward.

        In image space (0,0 top-left) far-to-near means increasing y.
        """
        if len(self.trace_points) < self.cfg.MIN_TRACE_FRAMES:
            return False

        start_y = self.trace_points[0][2]
        end_y = self.trace_points[-1][2]
        if start_y >= self.cfg.FAR_COURT_Y_SPLIT:
            return False
        if end_y < start_y + self.cfg.MIN_TRACE_DROP_PX:
            return False

        # Reject lateral drift dressed up as a drop: most steps must descend.
        ys = [p[2] for p in self.trace_points]
        down = sum(1 for a, b in zip(ys, ys[1:]) if b >= a)
        return down >= 0.6 * (len(ys) - 1)

    # -----------------------------------------------------------------
    # MAIN STATE MACHINE STEP
    # -----------------------------------------------------------------
    def process_frame(self, record: Dict[str, Any]):
        t = record["t"]
        valid_balls = self._filter_balls(record.get("all_balls", []))
        self._update_arming(record, t)
        armed = (t - self.last_armed_t) <= self.cfg.ARM_TO_TRACE_S
        raised = self._update_hand_raise(record, t)
        self._update_point_state(t, bool(valid_balls))

        if raised and self.point_live:
            # Mid-rally arm raise (overhead, smash, celebration) — not a serve.
            self.suppressed_raises.append(t)
            raised = False

        if self.state == SystemState.IDLE:
            # Primary gate: armed + a hand-raise rising edge. Ball trace is
            # not required — it only upgrades confidence, in CONFIRMING.
            if (armed and raised and
                    (t - self.last_serve_t) >= self.cfg.MIN_SERVE_SEPARATION_S):
                serve = {
                    "timestamp": round(t, 3),
                    "frame_start_t": t,
                    "frame_end_t": t,
                    "confidence": "MEDIUM",
                    "gate": "pose",
                }
                self.detected_serves.append(serve)
                self.last_serve_t = t
                self._pending_serve = serve
                self.trace_points = []
                self.state = SystemState.CONFIRMING
                # This serve opens a point; suppress raises until it ends.
                self.point_live = True
                self.point_start_t = t
                self.last_any_ball_t = t

        elif self.state == SystemState.CONFIRMING:
            if t - self._pending_serve["frame_start_t"] > self.cfg.POSE_TRACE_CONFIRM_S:
                # Confirmation window elapsed — leave the MEDIUM-confidence
                # detection as-is and go back to waiting for the next raise.
                self._pending_serve = None
                self.trace_points = []
                self.state = SystemState.IDLE
                return

            ball = self._pick_ball(valid_balls) if valid_balls else None
            if ball is not None:
                self.trace_points.append((t, ball[0], ball[1]))
                self.last_ball_t = t

                if self._verify_far_to_near_trace():
                    self._pending_serve["confidence"] = "HIGH"
                    self._pending_serve["frame_end_t"] = t
                    self._pending_serve = None
                    self.trace_points = []
                    self.state = SystemState.IDLE
                elif t - self.trace_points[0][0] > self.cfg.MAX_TRACE_DURATION_S:
                    # This attempt ran into the following rally shot — drop
                    # it and let a fresh trace attempt start, still within
                    # the same confirmation budget.
                    self.trace_points = []
            elif (self.trace_points and
                  (t - self.last_ball_t) > self.cfg.MAX_TRACE_GAP_S):
                self.trace_points = []


# =====================================================================
# TELEMETRY FILE PARSER & EXECUTION
# =====================================================================

def load_telemetry(telemetry_path: str):
    with open(telemetry_path, "r") as fh:
        first_line = fh.readline()
        meta = {}
        records = []
        try:
            header = json.loads(first_line)
            meta = header.get("meta", {})
        except json.JSONDecodeError:
            pass  # Fallback if header is missing
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if "f" in record:
                records.append(record)
    if meta.get("video"):
        meta["_video_path"] = os.path.join(
            os.path.dirname(os.path.abspath(telemetry_path)), meta["video"])
    return meta, records


def pick_far_player_source(records) -> str:
    """Prefers the native-resolution ROI track, falling back to v1's `fpw`.

    `fprw` comes from a full-resolution crop around the far baseline; `fpw` is
    a smoothed 540p full-frame track whose world position is far noisier, so
    the 3 ft arming tolerance means something quite different against it.
    """
    if any(r.get("fprw") for r in records):
        return "fprw"
    print("[FAR-SERVE] WARN: telemetry has no `fprw` (pre-v2 file) — arming "
          "against the noisier smoothed `fpw` track. Re-extract for v2.")
    return "fpw"


def load_pose_cache(telemetry_path: str, pose_path: str = None) -> Dict[int, Optional[float]]:
    """Loads the extract_far_pose.py cache, or warns and returns {} if absent.

    An empty cache means the hand-raise gate never fires (raise_run can't
    reach RAISE_MIN_CONSEC without margin samples) — this module has nothing
    left to gate on, so detect_far_serves will report zero serves rather than
    silently falling back to some other trigger.
    """
    pose_path = pose_path or far_pose_path_for(telemetry_path)
    if not os.path.isfile(pose_path):
        print(f"[FAR-SERVE] WARN: no pose cache at {pose_path} — the hand-raise "
              f"gate cannot fire. Run: python -m pipeline.extract_far_pose "
              f"{telemetry_path}")
        return {}
    return load_far_pose(pose_path)


def config_for(meta) -> FarServeDetectorConfig:
    """The right thresholds for whichever extractor produced this telemetry.

    The two streams want different raise-gate settings (see `for_fast_path`),
    and getting it wrong is silent — the detector runs happily either way and
    just scores worse — so the choice is made from the file's own provenance
    rather than left to the caller.
    """
    if meta.get("source") == "anya_far_telemetry":
        return FarServeDetectorConfig.for_fast_path()
    return FarServeDetectorConfig()


def detect_far_serves(telemetry_path: str,
                      config: FarServeDetectorConfig = None,
                      pose_path: str = None) -> List[Dict[str, Any]]:
    meta, records = load_telemetry(telemetry_path)
    cfg = config or config_for(meta)
    pose_cache = load_pose_cache(telemetry_path, pose_path)
    return _run_detector(meta, records, cfg, pose_cache).detected_serves


def _run_detector(meta, records, cfg, pose_cache=None) -> "FarServeDetector":

    detector = FarServeDetector(cfg)
    detector.set_exclusion_zones(scale_exclusion_zones(meta.get("exclusion_zones", []), meta))
    detector.set_static_cells(
        calibrate_static_blobs(records, cfg, *ball_sampling_scales(meta)))
    detector.fp_key = pick_far_player_source(records)
    detector.set_pose_cache(pose_cache or {})

    for record in records:
        detector.process_frame(record)
    return detector


def evaluate(serves, gt_path: str, fps: float,
             lead_s: float = 1.5, lag_s: float = 3.5):
    """Matches detections against ground-truth far-serve rally starts.

    `lead_s` is the wider side deliberately: ground truth marks the rally
    start (ball struck), while the pose gate fires on the toss/trophy phase
    that precedes it — so a correct detection legitimately lands ~1-1.5s
    *before* its ground-truth time.
    """
    rallies = json.load(open(gt_path))["rallies"]
    truth = [r["start"] / fps for r in rallies if r.get("serve") == "far"]
    times = sorted(s["timestamp"] for s in serves)

    matched, used = [], set()
    for g in truth:
        hit = next((i for i, t in enumerate(times)
                    if i not in used and (g - lead_s) <= t <= (g + lag_s)), None)
        if hit is not None:
            used.add(hit)
        matched.append((g, times[hit] if hit is not None else None))

    tp = len(used)
    fn = len(truth) - tp
    fp = len(times) - tp
    recall = tp / len(truth) if truth else 0.0
    precision = tp / len(times) if times else 0.0
    print(f"\n[EVAL] ground truth far serves: {len(truth)}   detections: {len(times)}")
    print(f"[EVAL] TP={tp}  FN={fn}  FP={fp}  "
          f"recall={recall:.2f}  precision={precision:.2f}")
    for g, m in matched:
        print(f"   GT {g:7.2f}s -> " + (f"HIT {m:.2f}s" if m else "MISS"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect far side serves from anya_telemetry JSONL.")
    parser.add_argument("telemetry_file", help="Path to _anya_telemetry.jsonl")
    parser.add_argument("--eval", help="Path to ground_truth.json for scoring")
    parser.add_argument("--conf", type=float, help="Override minimum ball confidence")
    parser.add_argument("--pose", help="Path to a far-pose cache (default: derived from telemetry path)")
    parser.add_argument("--full-preset", action="store_true",
                        help="Use the full-telemetry thresholds even on fast-path "
                             "telemetry (A/B runs; the preset is normally picked "
                             "from the file's provenance)")
    args = parser.parse_args()

    meta, records = load_telemetry(args.telemetry_file)
    cfg = FarServeDetectorConfig() if args.full_preset else config_for(meta)
    if args.conf is not None:
        cfg.BALL_CONF_MIN = args.conf

    pose_cache = load_pose_cache(args.telemetry_file, args.pose)
    detector = _run_detector(meta, records, cfg, pose_cache)
    serves = detector.detected_serves
    n_high = sum(1 for s in serves if s["confidence"] == "HIGH")
    print(f"[FAR-SERVE] far-player track: {detector.fp_key}   "
          f"arm events: {len(detector.arm_events)}   "
          f"raise events: {len(detector.raise_events)}   "
          f"suppressed in-rally: {len(detector.suppressed_raises)}")
    print(f"[{len(serves)} Far Serves Detected]  ({n_high} HIGH, {len(serves)-n_high} MEDIUM)")
    for s in serves:
        print(f"  -> Serve at t={s['timestamp']}s [{s['confidence']}]"
              f" (Window: {s['frame_start_t']:.2f}s - {s['frame_end_t']:.2f}s)")

    if args.eval:
        evaluate(serves, args.eval, meta.get("fps", 30.0))
