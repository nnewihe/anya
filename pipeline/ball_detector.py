"""
ball_detector.py
================
Standalone single-ball detector + tracker, the ball analogue of
``player_detector.py``.

Like ``player_detector.py`` it is a self-contained interactive script: it runs
YOLO (``ball_best.pt``) over the entire frame (no active-zone crop), filters
detections, resolves a single coherent ball, and visualises the result.
Detection is batched on the GPU (MPS/CUDA) at a reduced ``imgsz`` and on every
``stride``-th frame for speed; skipped frames are filled in by the tracker's
parabola fit.  It differs from the player script in three deliberate ways
requested for the ball:

  1. Exclusion zones are **automated**, not click-drawn.  We reuse the run_anya
     pipeline's ``create_auto_exclusion_zones`` (a DBSCAN scan over random frames
     that finds static ball-like clutter — ball baskets, logos, line markers) and
     cache the result to disk exactly as ``anya_base.py`` does.

  2. **At most one ball** is emitted per frame.  A tennis point has exactly one
     ball, so the tracker collapses the noisy identity-less per-frame detections
     to a single trajectory (or nothing).

  3. Legitimate tracks are **ballistic**: between impacts a tennis ball flies a
     smooth arc — in image space x(t) is ~linear and y(t) ~quadratic (projected
     gravity + perspective).  The tracker requires a candidate trace to fit a
     parabola over >= ``min_arc`` frames before it is believed, and it only
     changes trajectory (racket / ground / net impact) when *future* detections
     form a new valid arc that connects to the old one.  This "look forward"
     behaviour is why the detector runs offline in two passes over the video:
     pass 1 collects detections, the tracker resolves the whole detection stream
     with full lookahead, and pass 2 replays the frames drawing the resolved ball.

  4. Traces are gated to **this court** by a ground-plane homography.  The user
     clicks the 4 court corners once; a 2D region can't tell a ball over this
     court from a high ball over a neighbouring court (they share pixels at
     altitude), but the two are separable on the ground plane, where the ball
     sits at its *bounces*.  The tracker back-projects each trajectory's bounces
     to court feet and drops whole trajectories that bounce off-court.

The tracker itself (:class:`ParabolicBallTracker`) depends only on numpy, so it
is unit-testable with synthetic detection streams (see ``_run_self_test``) with
no model weights or video.

Run:
    python pipeline/ball_detector.py                 # synthetic self-test
    python pipeline/ball_detector.py <video.mp4>     # detect + visualise
"""

from __future__ import annotations

import math
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Reuse the pipeline's auto-exclusion + config helpers.  This file is meant to be
# run directly (``python pipeline/ball_detector.py``), which puts the pipeline dir
# on sys.path, so a plain import works; fall back to a package-relative import when
# imported as ``pipeline.ball_detector``.
try:  # pragma: no cover - import plumbing
    from utilities import (Config, create_auto_exclusion_zones,
                           load_cached_exclusion_zones, save_cached_exclusion_zones)
except ImportError:  # pragma: no cover - import plumbing
    from .utilities import (Config, create_auto_exclusion_zones,
                            load_cached_exclusion_zones, save_cached_exclusion_zones)


# A detection is a pixel centre plus its YOLO confidence.
Detection = Tuple[float, float, float]   # (x, y, conf)


def make_image_row_perspective(frame_height: float,
                               far_floor: float = 0.4) -> Callable[[float], float]:
    """Cheap perspective model from the analysis-frame height (see ball_tracker.py).

    Returns a multiplier in (far_floor, 1.0]: ~1.0 near the bottom of the frame
    (near side, many px/ft) shrinking toward `far_floor` at the top (far side,
    few px/ft).  A real ball's apparent motion shrinks the same way with depth,
    so scaling the motion/gate thresholds by this keeps them physically
    consistent — in particular it lowers the "is it moving?" bar on the far side
    so small, slow-looking far-court arcs aren't rejected as stationary.
    """
    h = max(1.0, float(frame_height))

    def scale(y: float) -> float:
        return max(far_floor, min(1.0, float(y) / h))

    return scale


# =====================================================================
# Ballistic segment: a parabola fitted to associated detections.
# =====================================================================
@dataclass
class _Segment:
    """One coherent ballistic arc.

    Holds the (frame, x, y) detection samples associated to the arc and a lazily
    refitted parametric model: x(t)=a*t+b (linear), y(t)=p*t^2+q*t+r (quadratic),
    with t measured in frames from the segment's first sample.  The RMS residual
    of that fit is the arc's quality score.
    """
    samples: List[Tuple[int, float, float]] = field(default_factory=list)  # (frame, x, y)
    _xc: Optional[np.ndarray] = field(default=None, repr=False)
    _yc: Optional[np.ndarray] = field(default=None, repr=False)
    _resid: float = field(default=0.0, repr=False)
    _dirty: bool = field(default=True, repr=False)

    # ---- construction -------------------------------------------------
    def add(self, frame: int, x: float, y: float) -> None:
        self.samples.append((int(frame), float(x), float(y)))
        self._dirty = True

    def copy(self) -> "_Segment":
        return _Segment(samples=list(self.samples))

    # ---- geometry -----------------------------------------------------
    @property
    def first_frame(self) -> int:
        return self.samples[0][0]

    @property
    def last_frame(self) -> int:
        return self.samples[-1][0]

    @property
    def n(self) -> int:
        return len(self.samples)

    def span_px(self) -> float:
        """Max distance between the last sample and any earlier sample."""
        if self.n < 2:
            return 0.0
        _, lx, ly = self.samples[-1]
        return max(math.hypot(sx - lx, sy - ly) for _, sx, sy in self.samples)

    # ---- fit ----------------------------------------------------------
    def _refit(self) -> None:
        f0 = self.samples[0][0]
        tt = np.array([s[0] - f0 for s in self.samples], dtype=float)
        xs = np.array([s[1] for s in self.samples], dtype=float)
        ys = np.array([s[2] for s in self.samples], dtype=float)

        # x(t) linear (needs >=2 pts); y(t) quadratic (needs >=3 pts).  With
        # fewer points we degrade gracefully to lower-order fits so the segment
        # can still predict while it is being seeded.
        with np.errstate(all="ignore"):
            Ax = np.vstack([tt, np.ones_like(tt)]).T
            self._xc = np.linalg.lstsq(Ax, xs, rcond=None)[0] if self.n >= 2 else np.array([0.0, xs[0]])

            if self.n >= 3:
                Ay = np.vstack([tt * tt, tt, np.ones_like(tt)]).T
                self._yc = np.linalg.lstsq(Ay, ys, rcond=None)[0]
            elif self.n == 2:
                slope = (ys[1] - ys[0]) / (tt[1] - tt[0] if tt[1] != tt[0] else 1.0)
                self._yc = np.array([0.0, slope, ys[0] - slope * tt[0]])
            else:
                self._yc = np.array([0.0, 0.0, ys[0]])

            # A Vandermonde fit over a long or degenerate t-range can be
            # ill-conditioned enough to blow up to inf/NaN; treat that as an
            # infinitely bad (rejected) fit rather than letting NaN propagate
            # into predict()/positions downstream.
            if not (np.all(np.isfinite(self._xc)) and np.all(np.isfinite(self._yc))):
                self._resid = float("inf")
                self._dirty = False
                return

            xp = Ax @ self._xc
            yp = np.vstack([tt * tt, tt, np.ones_like(tt)]).T @ self._yc
            resid = float(np.sqrt(np.mean((xp - xs) ** 2 + (yp - ys) ** 2)))
        self._resid = resid if math.isfinite(resid) else float("inf")
        self._dirty = False

    def _ensure_fit(self) -> None:
        if self._dirty:
            self._refit()

    @property
    def residual(self) -> float:
        self._ensure_fit()
        return self._resid

    def predict(self, frame: int) -> Tuple[float, float]:
        """Fitted (x, y) at an arbitrary frame index."""
        self._ensure_fit()
        t = float(frame - self.samples[0][0])
        x = self._xc[0] * t + self._xc[1]
        y = self._yc[0] * t * t + self._yc[1] * t + self._yc[2]
        return float(x), float(y)

    def vy_at(self, frame: int) -> float:
        """Fitted vertical image velocity (px/frame) at a frame.

        y(t) = p*t^2 + q*t + r  ->  vy = 2*p*t + q.  Positive means the ball is
        descending in the image (y increases downward), which is how a bounce is
        detected: an arc descending into a junction (vy > 0) that the next arc
        leaves ascending (vy < 0) marks a ground contact.
        """
        self._ensure_fit()
        t = float(frame - self.samples[0][0])
        return float(2.0 * self._yc[0] * t + self._yc[1])

    def resid_if_added(self, frame: int, x: float, y: float) -> float:
        """Residual the segment *would* have if (frame, x, y) were appended.

        Used to decide, during growth, whether a candidate detection keeps the
        arc parabolic — a point that spikes the residual is a different arc
        (impact) or clutter and is rejected.
        """
        trial = _Segment(samples=self.samples + [(int(frame), float(x), float(y))])
        return trial.residual


# =====================================================================
# ParabolicBallTracker: offline, look-forward, single-ball resolver.
# =====================================================================
class ParabolicBallTracker:
    """
    Resolve a noisy per-frame detection stream into at most one ballistic ball
    per frame.

    Strategy (fully offline / look-forward): grow every plausible ballistic
    segment by RANSAC-style forward extension — a segment only keeps a future
    detection if it stays inside the prediction gate *and* keeps the parabola
    fit tight, so growth is decided by whether future data fits the arc.  Then
    keep only *ballistic-valid* segments (>= ``min_arc`` points, low residual,
    real motion) and select a temporally non-overlapping set of them, best
    first.  Because accepted segments never share a frame, at most one ball is
    ever emitted; a genuine direction change (impact) simply appears as the next
    accepted segment, which only exists because future detections formed a new
    valid arc.
    """

    def __init__(
        self,
        fps: float = 30.0,
        *,
        min_arc: int = 3,           # frames a trace must fit a parabola over to be believed
        resid_tol: float = 12.0,    # max RMS fit residual (px) for a ballistic-valid segment
        grow_resid_tol: float = 18.0,  # max residual to accept a point mid-growth
        seed_gate_px: float = 90.0,    # gate for the 2nd point of a fresh seed (velocity unknown)
        two_pt_gate_px: float = 60.0,  # gate for the 3rd point (linear extrapolation)
        gate_px: float = 40.0,         # gate around the parabola prediction once fitted
        max_gap: int = 6,           # consecutive missed frames a segment may bridge (occlusion)
        move_thresh_px: float = 30.0,  # min motion span for a real (non-static) ball
        bridge_gap: int = 8,        # max frame gap between two arcs to interpolate the impact
        bridge_dist_px: float = 80.0,  # max endpoint distance to treat two arcs as one impact
        v_max_px_s: float = 2500.0,  # physical ball-speed ceiling at 960-wide (~60 mph serve)
        px_scale: float = 1.0,      # multiply every pixel threshold (frame_width / 960)
        # Perspective model: motion/gate thresholds shrink toward the far side
        # (top of frame) where the ball covers fewer pixels, so small far-court
        # arcs aren't wrongly rejected as too-short/stationary.  None -> constant 1.
        perspective_scale: Optional[Callable[[float], float]] = None,
        max_arc_s: float = 2.5,     # hard cap on one arc's duration (unimpeded tennis flight
                                     # is well under this; also keeps the t-range small so the
                                     # quadratic fit stays numerically well-conditioned)
        # ── Court gating via ground-plane homography (bounce-based) ─────
        # A 2D image polygon can't separate a ball over this court from a high
        # ball over a neighbouring court — they share pixels at altitude.  The
        # disambiguation lives on the ground plane: a ball is on the court
        # surface at its *bounces*, where the homography (image px -> court feet)
        # is exact.  We back-project each trajectory's bounces and reject whole
        # trajectories that bounce off-court.  `homography` is a 3x3 matrix
        # mapping image pixels to court feet (X across width, Y along length);
        # the court rectangle is [0,width]x[0,length] ft plus a margin.
        homography: Optional[np.ndarray] = None,
        court_width_ft: float = 27.0,     # singles width; margins cover the alleys
        court_length_ft: float = 78.0,
        court_margin_x_ft: float = 12.0,  # lateral tolerance (doubles alley + wide balls)
        court_margin_y_ft: float = 30.0,  # length tolerance (generous; side courts share Y)
        court_group_gap_s: float = 1.0,   # max time gap to chain arcs into one rally
        court_link_px: float = 60.0,      # max endpoint jump to chain arcs (960-wide px)
        court_fallback_margin_x_ft: float = 45.0,  # lateral cutoff for bounce-less chains
        # ── Soft exclusion zones (static-clutter rectangles) ────────────
        # Detections are NOT pre-filtered by these; instead a candidate segment
        # is rejected only if MORE than `zone_reject_frac` of its samples fall
        # inside a zone (i.e. it *is* the clutter).  A real ball merely passing
        # through a zone contributes only a few in-zone samples and survives.
        exclusion_zones: Optional[Sequence[Tuple[float, float, float, float]]] = None,
        zone_reject_frac: float = 0.5,
    ):
        self.fps = float(fps)
        self.min_arc = int(min_arc)
        self.max_arc_frames = max(int(min_arc), int(round(max_arc_s * self.fps)))
        self.persp = perspective_scale or (lambda _y: 1.0)
        self.homography = homography
        self.exclusion_zones = list(exclusion_zones) if exclusion_zones else []
        self.zone_reject_frac = float(zone_reject_frac)
        self.court_width_ft = float(court_width_ft)
        self.court_length_ft = float(court_length_ft)
        self.court_margin_x_ft = float(court_margin_x_ft)
        self.court_margin_y_ft = float(court_margin_y_ft)
        self.court_group_gap = max(1, int(round(court_group_gap_s * self.fps)))
        self.court_fallback_margin_x_ft = float(court_fallback_margin_x_ft)
        # Per-frame travel ceiling: a real ball can't jump more than this between
        # adjacent frames.  Fences off "teleport" chains where the loose seed gate
        # would otherwise link distant clutter detections into a fake parabola.
        self.v_max_step = float(v_max_px_s) / max(self.fps, 1e-6) * float(px_scale)
        # All *_px thresholds are calibrated at ~960-wide analysis frames; px_scale
        # rescales them to the working resolution so gating/residual/motion stay
        # physically consistent when the detector runs at native (e.g. 4K) size.
        s = float(px_scale)
        self.px_scale = s
        self.resid_tol = float(resid_tol) * s
        self.grow_resid_tol = float(grow_resid_tol) * s
        self.seed_gate_px = float(seed_gate_px) * s
        self.two_pt_gate_px = float(two_pt_gate_px) * s
        self.gate_px = float(gate_px) * s
        self.max_gap = int(max_gap)
        self.move_thresh_px = float(move_thresh_px) * s
        self.bridge_gap = int(bridge_gap)
        self.bridge_dist_px = float(bridge_dist_px) * s
        self.court_link_px = float(court_link_px) * s

    # ------------------------------------------------------------------
    def _grow(self, dets: Sequence[Sequence[Detection]], start_f: int,
              start_det: Detection) -> _Segment:
        """Grow a ballistic segment forward from one seed detection."""
        seg = _Segment()
        seg.add(start_f, start_det[0], start_det[1])
        misses = 0
        n = len(dets)
        for f in range(start_f + 1, n):
            if f - start_f > self.max_arc_frames:
                # A real unimpeded flight doesn't last this long — keep growing
                # here would just be chaining coincidental clutter and blows up
                # the quadratic fit's conditioning over the long t-range.
                break
            # Expected next position + gate depend on how well-formed the arc is.
            if seg.n >= 3:
                ex, ey = seg.predict(f)
                gate = self.gate_px
            elif seg.n == 2:
                # Linear extrapolation from the last two samples.
                (f0, x0, y0), (f1, x1, y1) = seg.samples[-2], seg.samples[-1]
                dfr = (f1 - f0) or 1
                ex = x1 + (x1 - x0) / dfr * (f - f1)
                ey = y1 + (y1 - y0) / dfr * (f - f1)
                gate = self.two_pt_gate_px
            else:
                _, ex, ey = seg.samples[-1]
                gate = self.seed_gate_px

            # Perspective: the ball covers fewer pixels on the far side, so scale
            # the gate and the physical step ceiling by image row.
            row = self.persp(ey)
            gate *= row
            # Widen the gate as an occlusion gap opens (ball travels while unseen).
            gate += 0.5 * gate * misses

            # Physical travel envelope since the last real sample (grows with the
            # gap): a candidate implying a supra-physical jump is clutter, not the ball.
            _, lx, ly = seg.samples[-1]
            max_step = self.v_max_step * row * (misses + 1)

            best, best_d = None, gate
            for (dx, dy, dc) in dets[f]:
                if math.hypot(dx - lx, dy - ly) > max_step:
                    continue
                d = math.hypot(dx - ex, dy - ey)
                if d <= best_d:
                    # Once the arc is fittable, reject points that break the parabola.
                    if seg.n >= 3 and seg.resid_if_added(f, dx, dy) > self.grow_resid_tol:
                        continue
                    best_d, best = d, (dx, dy)

            if best is not None:
                seg.add(f, best[0], best[1])
                misses = 0
            else:
                misses += 1
                if misses > self.max_gap:
                    break
        return seg

    def _frac_in_zone(self, seg: _Segment) -> float:
        """Fraction of a segment's samples that fall inside an exclusion zone."""
        if not self.exclusion_zones:
            return 0.0
        inz = 0
        for _, sx, sy in seg.samples:
            for (x1, y1, x2, y2) in self.exclusion_zones:
                if x1 <= sx <= x2 and y1 <= sy <= y2:
                    inz += 1
                    break
        return inz / max(1, seg.n)

    def _is_valid_segment(self, seg: _Segment) -> bool:
        """Whether a grown/trimmed segment qualifies as a real ballistic arc.

        Shared by candidate growth and selection so the bar is identical in both.
        The motion threshold is perspective-scaled by the arc's image row (a far
        arc covers fewer pixels), and a minimal-length (3-point) arc must be
        densely sampled so 3 scattered points can't fake a parabola.
        """
        if seg.n < self.min_arc or seg.residual > self.resid_tol:
            return False
        row = self.persp(sum(sy for _, _, sy in seg.samples) / seg.n)
        if seg.span_px() <= self.move_thresh_px * row:
            return False
        # Noise-triple guard: <= 2 frames per step (tolerates stride up to 2);
        # longer arcs are already well-corroborated and exempt.
        if seg.n < 4 and seg.last_frame - seg.first_frame > 2 * (seg.n - 1):
            return False
        # Soft exclusion: a segment living mostly inside a clutter zone is clutter;
        # one that only passes through keeps few in-zone samples and survives.
        if self._frac_in_zone(seg) > self.zone_reject_frac:
            return False
        return True

    def _candidate_segments(self, dets: Sequence[Sequence[Detection]]) -> List[_Segment]:
        """Grow a segment from every detection, then keep maximal ballistic-valid ones."""
        grown: List[_Segment] = []
        for f, frame_dets in enumerate(dets):
            for det in frame_dets:
                seg = self._grow(dets, f, det)
                if self._is_valid_segment(seg):
                    grown.append(seg)

        # De-duplicate: drop any segment whose *sample* membership is a subset of
        # a longer one (seeds from later points reproduce tails of earlier arcs).
        # Compare (frame, x, y) samples, not just frames — two spatially-distinct
        # balls can share the same frames, so a frame-set subset test would wrongly
        # discard a shorter real arc that overlaps a longer one in time.
        grown.sort(key=lambda s: s.n, reverse=True)
        kept: List[_Segment] = []
        kept_sets: List[set] = []
        for seg in grown:
            fset = set(seg.samples)
            if any(fset <= ks for ks in kept_sets):
                continue
            kept.append(seg)
            kept_sets.append(fset)
        return kept

    def _contiguous_runs(self, samples, occupied):
        """Split samples into runs that avoid claimed frames and big gaps.

        A run breaks whenever a frame is already claimed by a better segment or
        the frame-to-frame gap exceeds ``max_gap`` (a real occlusion bridge, not
        two arcs). Yields sample sublists.
        """
        def claimed(fr):
            return any(a <= fr <= b for a, b in occupied)

        run: List[Tuple[int, float, float]] = []
        prev_f = None
        for s in samples:
            fr = s[0]
            if claimed(fr) or (prev_f is not None and fr - prev_f > self.max_gap):
                if run:
                    yield run
                run = []
            if not claimed(fr):
                run.append(s)
                prev_f = fr
        if run:
            yield run

    def _select(self, candidates: List[_Segment]) -> List[_Segment]:
        """Accept segments best-first, trimming frames already claimed.

        Quality = (length, -residual): a longer/tighter arc wins any contested
        frame span.  A lower-priority candidate that overlaps an accepted one is
        not dropped whole — its unclaimed portion is re-checked and kept if it is
        still a valid ballistic arc.  This preserves the second arc after an
        impact (whose boundary frames the first arc may have absorbed) while
        still guaranteeing accepted segments never share a frame -> <=1 ball.
        """
        candidates = sorted(
            candidates,
            key=lambda s: (s.n, -s.residual),
            reverse=True,
        )
        chosen: List[_Segment] = []
        occupied: List[Tuple[int, int]] = []
        for seg in candidates:
            for run in self._contiguous_runs(seg.samples, occupied):
                trimmed = _Segment(samples=list(run))
                if self._is_valid_segment(trimmed):
                    chosen.append(trimmed)
                    occupied.append((trimmed.first_frame, trimmed.last_frame))
        chosen.sort(key=lambda s: s.first_frame)
        return chosen

    # ---- court gating (ground-plane, bounce-based) --------------------
    def _to_world(self, x: float, y: float) -> Tuple[float, float]:
        """Back-project an image point to court feet via the homography."""
        v = self.homography @ np.array([x, y, 1.0], dtype=float)
        w = v[2] if abs(v[2]) > 1e-9 else 1e-9
        return float(v[0] / w), float(v[1] / w)

    def _in_court(self, wx: float, wy: float, mx: float, my: float) -> bool:
        return (-mx <= wx <= self.court_width_ft + mx
                and -my <= wy <= self.court_length_ft + my)

    def _connected(self, a: _Segment, b: _Segment) -> bool:
        """Are arcs a (earlier) and b (later) the same ball across an impact?

        They must be close in time and land where the other takes off (a ball
        doesn't teleport at a bounce/hit).  Two different balls on different
        courts fail the spatial test, so they never chain into one trajectory.
        """
        gap = b.first_frame - a.last_frame
        if gap < 0 or gap > self.court_group_gap:
            return False
        ax, ay = a.predict(a.last_frame)
        bx, by = b.predict(b.first_frame)
        return math.hypot(bx - ax, by - ay) <= self.court_link_px

    def _court_gate(self, chosen: List[_Segment]) -> List[_Segment]:
        """Drop whole trajectories that bounce off-court.

        Chains consecutive arcs of one ball (time + spatial continuity), finds
        the ground contacts (junctions where one arc descends in and the next
        ascends out), back-projects them to court feet, and keeps a chain only
        if a contact lands in-court.  A chain with no detectable bounce falls
        back to its lowest on-screen point under a generous lateral cutoff.
        Off-court trajectories (e.g. rallies on a neighbouring court whose apex
        merely clips this court in the image) are removed entirely.
        """
        if self.homography is None or not chosen:
            return chosen

        chosen = sorted(chosen, key=lambda s: s.first_frame)
        chains: List[List[_Segment]] = []
        for seg in chosen:
            if chains and self._connected(chains[-1][-1], seg):
                chains[-1].append(seg)
            else:
                chains.append([seg])

        kept: List[_Segment] = []
        for chain in chains:
            bounces: List[Tuple[float, float]] = []
            for a, b in zip(chain, chain[1:]):
                # A bounce: descending into the junction, ascending out of it.
                if a.vy_at(a.last_frame) > 0 and b.vy_at(b.first_frame) < 0:
                    ax, ay = a.predict(a.last_frame)
                    bx, by = b.predict(b.first_frame)
                    bounces.append(self._to_world((ax + bx) / 2.0, (ay + by) / 2.0))

            if bounces:
                # Keep the rally iff at least one bounce is on this court.
                if any(self._in_court(wx, wy, self.court_margin_x_ft,
                                      self.court_margin_y_ft)
                       for wx, wy in bounces):
                    kept.extend(chain)
            else:
                # No clean bounce (single arc / hits only): the ground point is
                # ambiguous (airborne parallax), so only reject when the lowest
                # on-screen point projects egregiously wide of the court.
                low = max((smp for s in chain for smp in s.samples),
                          key=lambda smp: smp[2])
                wx, wy = self._to_world(low[1], low[2])
                if self._in_court(wx, wy, self.court_fallback_margin_x_ft,
                                  self.court_margin_y_ft + self.court_fallback_margin_x_ft):
                    kept.extend(chain)

        kept.sort(key=lambda s: s.first_frame)
        return kept

    # ------------------------------------------------------------------
    def resolve(self, dets: Sequence[Sequence[Detection]]
                ) -> Tuple[List[Optional[Tuple[float, float]]], List[str], List[_Segment]]:
        """
        Resolve the whole detection stream.

        Returns
        -------
        positions : list, one per frame — the emitted ball (x, y), or None.
        states    : list, one per frame — 'tracking' (real detection this frame),
                    'coasting' (fitted through an internal/occlusion gap),
                    'bridge' (interpolated impact between two arcs), or 'none'.
        segments  : the accepted ballistic segments (for trace overlays).
        """
        n = len(dets)
        positions: List[Optional[Tuple[float, float]]] = [None] * n
        states: List[str] = ["none"] * n

        segments = self._select(self._candidate_segments(dets))
        segments = self._court_gate(segments)

        for seg in segments:
            member_frames = {fr for fr, _, _ in seg.samples}
            for f in range(seg.first_frame, seg.last_frame + 1):
                positions[f] = seg.predict(f)
                states[f] = "tracking" if f in member_frames else "coasting"

        # Bridge the impact between consecutive arcs when they are close in time
        # and space (the ball is briefly undetectable at contact): linearly
        # interpolate across the blank frames so the ball doesn't blink out.
        for a, b in zip(segments, segments[1:]):
            gap = b.first_frame - a.last_frame
            if 1 < gap <= self.bridge_gap:
                ax, ay = a.predict(a.last_frame)
                bx, by = b.predict(b.first_frame)
                if math.hypot(bx - ax, by - ay) <= self.bridge_dist_px:
                    for f in range(a.last_frame + 1, b.first_frame):
                        t = (f - a.last_frame) / gap
                        positions[f] = (ax + t * (bx - ax), ay + t * (by - ay))
                        states[f] = "bridge"

        return positions, states, segments


# =====================================================================
# Highlight-reel assembly: live points, dead time cut, fixed pre/post-roll.
# =====================================================================
def compute_highlight_ranges(
    states: Sequence[str],
    fps: float,
    *,
    bridge_s: float = 3.0,       # merge live stretches closer than this into one clip
    preroll_s: float = 1.5,      # lead-in before every clip
    postroll_s: float = 1.0,     # tail after every clip's last live frame
    min_live_s: float = 0.3,     # drop groups with less real ball activity than this
) -> List[Tuple[int, int]]:
    """Turn per-frame live/dead states into a continuous set of highlight clips.

    A "live" frame is any non-``none`` state (a real detection, an intra-point
    coast, or an impact bridge — all already point-internal).  Live stretches
    within ``bridge_s`` of each other are grouped into one point (dead time
    inside a point is kept so active play is never chopped); the gap between
    distant points is what gets cut.  Each group is padded by a fixed
    ``preroll_s`` before and ``postroll_s`` after.  Ranges that then overlap are
    merged so the reel has no duplicated frames and flows continuously.

    Returns a sorted list of inclusive ``(start_frame, end_frame)`` clips.
    """
    n = len(states)
    if n == 0:
        return []

    # 1. Maximal runs of live frames.
    intervals: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if states[i] != "none":
            j = i
            while j + 1 < n and states[j + 1] != "none":
                j += 1
            intervals.append((i, j))
            i = j + 1
        else:
            i += 1
    if not intervals:
        return []

    # 2. Group live runs into points by the bridge gap.
    bridge_frames = bridge_s * fps
    groups: List[List[Tuple[int, int]]] = [[intervals[0]]]
    for (s, e) in intervals[1:]:
        prev_end = groups[-1][-1][1]
        if (s - prev_end) <= bridge_frames:
            groups[-1].append((s, e))
        else:
            groups.append([(s, e)])

    # 3. Per group: drop negligible activity, then pad by pre-/post-roll.
    min_live_frames = min_live_s * fps
    preroll = preroll_s * fps
    raw: List[List[int]] = []
    for g in groups:
        live = sum(e - s + 1 for s, e in g)
        if live < min_live_frames:
            continue
        gs, ge = g[0][0], g[-1][1]
        start = max(0, int(round(gs - preroll)))
        end = min(n - 1, int(round(ge + postroll_s * fps)))
        raw.append([start, end])

    # 4. Merge overlapping/adjacent ranges so the flow is continuous.
    raw.sort()
    merged: List[List[int]] = []
    for r in raw:
        if merged and r[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return [(a, b) for a, b in merged]


# =====================================================================
# AnyaBallDetector: interactive driver (mirrors AnyaTwoStateSystem).
# =====================================================================
class AnyaBallDetector:
    def __init__(self, video_path: str, *, imgsz: int = 1280, stride: int = 1,
                 batch_size: int = 16, device: Optional[str] = None,
                 half: Optional[bool] = None, ball_conf: float = 0.05):
        self.video_path = video_path
        _models = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model = self._load_model(os.path.join(_models, "ball_best.pt"))
        # Detection confidence: kept low so faint far-court balls are fed to the
        # tracker; the ballistic + soft-exclusion criteria reject the extra clutter.
        self.ball_conf = float(ball_conf)

        # ── Detection-speed knobs ───────────────────────────────────────
        # imgsz     : YOLO inference size (frames are downscaled so the longest
        #             side hits this, cutting pixels vs native 4K).
        # stride    : detect on every Nth frame; skipped frames are filled by the
        #             tracker's parabola fit (they read as coasting).
        # batch_size: frames per YOLO call, to keep the GPU fed.
        # device    : 'mps'/'cuda'/'cpu'; auto-picks the best available.
        # half      : FP16 inference (defaults on for GPU, off for CPU).
        self.imgsz = int(imgsz)
        self.stride = max(1, int(stride))
        self.batch_size = max(1, int(batch_size))
        self.device = device or self._pick_device()
        self.half = (self.device in ("mps", "cuda")) if half is None else bool(half)
        # Move the model onto the chosen device once so the exclusion-zone scan
        # (which calls the model internally) also benefits.
        try:
            self.model.to(self.device)
        except Exception as e:  # pragma: no cover - device edge cases
            print(f"[WARN] Could not move model to {self.device} ({e}); using CPU")
            self.device, self.half = "cpu", False
        print(f"[INFO] Detection: device={self.device} half={self.half} "
              f"imgsz={self.imgsz} stride={self.stride} batch={self.batch_size}")

        # Frame -> inference resize, set once video dims are known (see process_video).
        self._infer_size: Optional[Tuple[int, int]] = None   # (w, h) fed to YOLO
        self._det_scale: float = 1.0                         # resized = full * det_scale

        # Court corners (cached, click-selected once): the 4 corners of THIS
        # court, used to build the image->court-feet homography that gates the
        # ball trace by where its bounces land (see ParabolicBallTracker._court_gate).
        self.court_points: List[Tuple[int, int]] = []
        self.H: Optional[np.ndarray] = None

        # Automated static exclusion zones (DBSCAN scan; see utilities).
        self.exclusion_zones: List[Tuple[int, int, int, int]] = []

    @staticmethod
    def _load_model(path: str):
        from ultralytics import YOLO
        return YOLO(path)

    # ---- court-corner selection / caching + homography -----------------
    def select_court(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.court_points) < 4:
            self.court_points.append((x, y))

    def get_court_polygon(self, frame):
        window_name = "Select Court Corners (Click 4 corners)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.select_court)
        while True:
            display = frame.copy()
            for pt in self.court_points:
                cv2.circle(display, pt, 5, (0, 0, 255), -1)
            if len(self.court_points) == 4:
                cv2.polylines(display, [np.array(self.court_points, np.int32)],
                              True, (0, 255, 0), 2)
            cv2.imshow(window_name, display)
            if len(self.court_points) == 4:
                cv2.waitKey(1500)
                break
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyWindow(window_name)

    def _config_path(self) -> str:
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        return os.path.join(video_dir, f"{video_name}_ball_config.json")

    def save_config(self):
        config = {"court_points": [list(p) for p in self.court_points]}
        try:
            with open(self._config_path(), "w") as f:
                json.dump(config, f, indent=2)
            print(f"[INFO] Saved ball config to {self._config_path()}")
        except Exception as e:
            print(f"[WARN] Could not save ball config: {e}")

    def load_config(self) -> bool:
        path = self._config_path()
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "r") as f:
                config = json.load(f)
            self.court_points = [tuple(p) for p in config["court_points"]]
            if len(self.court_points) != 4:
                return False
            print(f"[INFO] Loaded ball config from {path}")
            return True
        except Exception as e:
            print(f"[WARN] Ball config unreadable ({e}), re-prompting")
            return False

    def _order_court_corners(self):
        """Derive (BL, BR, TR, TL) from the 4 clicked corners regardless of click
        order: image-y splits near (larger y) vs far (smaller y), image-x splits
        left vs right. Mirrors player_detector._order_court_corners.
        """
        pts = sorted(self.court_points, key=lambda p: p[1])
        far_pair, near_pair = pts[:2], pts[2:]
        TL, TR = sorted(far_pair, key=lambda p: p[0])
        BL, BR = sorted(near_pair, key=lambda p: p[0])
        return BL, BR, TR, TL

    def _build_homography(self):
        """Image->court-feet homography anchored on the singles rectangle."""
        if len(self.court_points) != 4:
            self.H = None
            return
        BL, BR, TR, TL = self._order_court_corners()
        src = np.array([BL, BR, TR, TL], dtype=np.float32)
        dst = np.array([
            [0, 0], [Config.COURT_WIDTH_FT, 0],
            [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT],
            [0, Config.COURT_LENGTH_FT],
        ], dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst)

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _set_infer_size(self, width: int, height: int) -> None:
        """Compute the aspect-preserving inference size (longest side = imgsz)."""
        scale = self.imgsz / float(max(width, height))
        scale = min(scale, 1.0)   # never upscale a small frame
        self._infer_size = (max(1, int(round(width * scale))),
                            max(1, int(round(height * scale))))
        self._det_scale = scale

    def _output_path(self, suffix: str) -> str:
        """Path for an output video next to the source video, named
        ``<stem>_<suffix>.mp4`` (deadtime_cutter.py / rally_detector.py style)."""
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        return os.path.join(video_dir, f"{video_name}_{suffix}.mp4")

    # ---- auto exclusion zones (mirrors anya_base startup) -------------
    def _init_exclusion_zones(self, width: int, height: int):
        cached = load_cached_exclusion_zones(self.video_path)
        if cached is not None:
            self.exclusion_zones = cached
            return
        print("\n[INFO] Scanning video for static exclusion zones...")
        try:
            # create_auto_exclusion_zones' clustering/padding defaults are tuned
            # for ~960-wide analysis frames (see anya_base.py).  Run the scan at
            # that resolution (aspect-preserving) so eps/padding behave as
            # intended, then scale the zones back up to full-resolution pixels so
            # they line up with our full-res detections.  Running the scan at full
            # 4K instead makes eps=5px cluster only near-identical points and
            # padding=5px negligible, yielding zones far too small.
            aw = Config.ANALYSIS_WIDTH
            sx = width / float(aw)
            ah = max(1, int(round(height / sx)))
            sy = height / float(ah)
            zones_small = create_auto_exclusion_zones(
                self.video_path, self.model,
                num_frames=50, conf=0.04, eps=12, padding=2,
                ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                analysis_size=(aw, ah),
            )
            self.exclusion_zones = [
                (int(round(x1 * sx)), int(round(y1 * sy)),
                 int(round(x2 * sx)), int(round(y2 * sy)))
                for (x1, y1, x2, y2) in zones_small
            ]
            print(f"[INFO] Found {len(self.exclusion_zones)} static exclusion zone(s)")
            save_cached_exclusion_zones(self.video_path, self.exclusion_zones)
        except Exception as e:
            print(f"[WARN] Could not compute static exclusion zones: {e}")
            self.exclusion_zones = []

    # ---- detection ----------------------------------------------------
    def _detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run YOLO on a batch of frames (downscaled to the inference size) and
        return per-frame ball detections remapped to full-resolution pixels.
        """
        if not frames:
            return []
        resized = [cv2.resize(f, self._infer_size, interpolation=cv2.INTER_AREA)
                   for f in frames]
        results = self.model(
            resized, conf=self.ball_conf,
            classes=[Config.DEFAULT_BALL_CLASS_INDEX], imgsz=self.imgsz,
            device=self.device, half=self.half, verbose=False,
        )
        inv = 1.0 / self._det_scale   # inference px -> full-resolution px
        batch_out: List[List[Detection]] = []
        for res in results:
            out: List[Detection] = []
            for box in res.boxes:
                x1_c, y1_c, x2_c, y2_c = map(float, box.xyxy[0])
                cx = (x1_c + x2_c) / 2.0 * inv
                cy = (y1_c + y2_c) / 2.0 * inv
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                # Exclusion zones are applied *softly* by the tracker (a segment
                # made mostly of in-zone detections is clutter and is rejected
                # there).  We keep every detection here so a real ball flying
                # through a zone — over a ball basket, a logo — is still tracked.
                out.append((cx, cy, conf))
            batch_out.append(out)
        return batch_out

    # ---- visualisation ------------------------------------------------
    def _draw_and_show(self, frame, position, state, trace, writer=None) -> bool:
        """Draw overlays, optionally write the frame to `writer`, and display it.

        Returns True if the user pressed 'q' (caller should stop the video).
        """
        if trace:
            pts = np.array([[int(px), int(py)] for px, py in trace], np.int32)
            cv2.polylines(frame, [pts], False, (0, 200, 255), 2)

        if position is not None:
            px, py = int(position[0]), int(position[1])
            colour = (0, 255, 0) if state == "tracking" else (0, 165, 255)
            cv2.circle(frame, (px, py), 8, colour, 2)
            cv2.circle(frame, (px, py), 2, colour, -1)

        for (ex1, ey1, ex2, ey2) in self.exclusion_zones:
            cv2.rectangle(frame, (int(ex1), int(ey1)), (int(ex2), int(ey2)), (0, 0, 255), 2)

        if len(self.court_points) == 4:
            cv2.polylines(frame, [np.array(self.court_points, np.int32)],
                          True, (255, 255, 0), 2)

        cv2.putText(frame, f"Ball: {state}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if state == "tracking" else (0, 165, 255), 3)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Tennis Ball Detection", frame)
        return (cv2.waitKey(1) & 0xFF) == ord('q')

    # ---- main loop ----------------------------------------------------
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read video")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._set_infer_size(width, height)

        # 1a. Court corners (cached, click-selected once) -> homography. Used to
        # gate the ball trace by where its bounces land; detections are NOT
        # filtered by it.
        if not self.load_config():
            self.get_court_polygon(first_frame)
            self.save_config()
        self._build_homography()

        # 1b. Exclusion zones (cached where possible). Detection runs over the
        # entire frame.
        self._init_exclusion_zones(width, height)

        # 2. Pass 1 — detect the ball on every `stride`-th frame, in batches.
        # Skipped frames get an empty detection list; the tracker fills them via
        # its parabola fit (they read as coasting), so per-frame output is intact.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        dets_per_frame: List[List[Detection]] = []
        batch_frames: List[np.ndarray] = []
        batch_slots: List[int] = []
        idx = 0

        def _flush_batch():
            if not batch_frames:
                return
            for slot, dets in zip(batch_slots, self._detect_batch(batch_frames)):
                dets_per_frame[slot] = dets
            batch_frames.clear()
            batch_slots.clear()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.stride == 0:
                dets_per_frame.append([])          # placeholder, filled on flush
                batch_slots.append(idx)
                batch_frames.append(frame)
                if len(batch_frames) >= self.batch_size:
                    _flush_batch()
            else:
                dets_per_frame.append([])          # skipped frame: no detection
            idx += 1
            if idx % 200 == 0:
                print(f"[INFO] Read {idx} frames...")
        _flush_batch()
        print(f"[INFO] Detection pass done: {idx} frames "
              f"({(idx + self.stride - 1) // self.stride} inferred)")

        # 3. Resolve the single ballistic ball with full lookahead.  Pixel
        # thresholds are calibrated at 960-wide frames; scale them to this
        # video's width so gating/residual/motion stay physically consistent,
        # and perspective-scale them by image row so far-court arcs aren't lost.
        # The court homography (if set) gates out trajectories that bounce
        # off-court (e.g. rallies on a neighbouring court).
        tracker = ParabolicBallTracker(
            fps=fps, px_scale=width / float(Config.ANALYSIS_WIDTH),
            perspective_scale=make_image_row_perspective(height),
            homography=self.H, exclusion_zones=self.exclusion_zones)
        positions, states, segments = tracker.resolve(dets_per_frame)
        alive = sum(1 for p in positions if p is not None)
        print(f"[INFO] Resolved {len(segments)} ballistic segment(s); "
              f"ball present in {alive}/{len(positions)} frames")

        # 4. Build the highlight reel: live points grouped into continuous clips,
        # dead time between distant points cut, fixed pre-/post-roll padding.
        highlight_ranges = compute_highlight_ranges(states, fps)
        in_highlight = bytearray(len(positions))
        for (a, b) in highlight_ranges:
            for j in range(a, b + 1):
                in_highlight[j] = 1
        hl_frames = sum(in_highlight)
        print(f"[INFO] Highlight reel: {len(highlight_ranges)} clip(s), "
              f"{hl_frames} frames ({hl_frames / fps:.1f}s of {len(positions) / fps:.1f}s)")

        # 5. Pass 2 — one decode pass that both (a) writes the full annotated trace
        # video and (b) writes the clean (un-annotated) frames selected for the
        # highlight reel.  The clean frame is captured before overlays are drawn.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        trace_path = self._output_path("ball_trace")
        highlight_path = self._output_path("highlights")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(trace_path, fourcc, fps, (width, height))
        hl_writer = cv2.VideoWriter(highlight_path, fourcc, fps, (width, height))

        f = 0
        trace_window = int(max(5, 0.4 * fps))
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if f < len(in_highlight) and in_highlight[f]:
                    hl_writer.write(frame)   # clean frame, before overlays
                trace = [positions[j] for j in range(max(0, f - trace_window), f + 1)
                         if positions[j] is not None]
                if self._draw_and_show(frame, positions[f], states[f], trace, writer):
                    break
                f += 1
        finally:
            writer.release()
            hl_writer.release()
            cap.release()
            cv2.destroyAllWindows()
        print(f"[INFO] Saved ball trace video to {trace_path}")
        print(f"[INFO] Saved highlight reel to {highlight_path}")


# =====================================================================
# Synthetic self-test — numpy only, no weights/video.
#   python pipeline/ball_detector.py
# =====================================================================
def _parabola(n, x0, y0, vx, vy, g, start_f=0, jitter=0.0, rng=None):
    """Generate n frames of a ballistic arc as single-detection frames."""
    out = []
    for k in range(n):
        x = x0 + vx * k
        y = y0 + vy * k + 0.5 * g * k * k
        if jitter and rng is not None:
            x += rng.normal(0, jitter)
            y += rng.normal(0, jitter)
        out.append((start_f + k, [(x, y, 0.9)]))
    return out


def _run_self_test() -> int:
    fps = 30.0
    failures: List[str] = []

    def check(name, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    def emissions_ok_single(positions):
        # positions is one-per-frame; single-ball is structurally guaranteed,
        # this just asserts the type contract.
        return all(p is None or (isinstance(p, tuple) and len(p) == 2) for p in positions)

    # --- Scenario 1: clean parabolic arc -> a single tracked ball. ---
    frames = [[] for _ in range(40)]
    for f, dets in _parabola(30, 100, 400, 18, -20, 1.5, start_f=2):
        frames[f] = dets
    tr = ParabolicBallTracker(fps)
    pos, st, segs = tr.resolve(frames)
    check("clean arc yields exactly one ballistic segment", len(segs) == 1)
    check("clean arc is tracked most frames", sum(p is not None for p in pos) >= 25)
    check("clean arc emits >=1 real detection frame", any(s == "tracking" for s in st))
    check("single-ball type contract holds (scenario 1)", emissions_ok_single(pos))

    # --- Scenario 2: brief occlusion mid-arc is bridged as ONE segment. ---
    frames = [[] for _ in range(40)]
    arc = dict(_parabola(34, 80, 420, 16, -22, 1.6, start_f=2))
    for f in range(2, 36):
        if 16 <= f <= 18:      # 3-frame occlusion hole
            continue
        frames[f] = arc[f]
    tr = ParabolicBallTracker(fps)
    pos, st, segs = tr.resolve(frames)
    check("occluded arc stays a single segment", len(segs) == 1)
    check("occlusion gap is coasted, not dropped",
          all(pos[f] is not None for f in range(16, 19)))
    check("coasting state reported inside the gap",
          any(st[f] == "coasting" for f in range(16, 19)))

    # --- Scenario 3: racket reversal — direction change only via a 2nd arc. ---
    frames = [[] for _ in range(60)]
    for f, dets in _parabola(24, 60, 460, 24, -26, 1.8, start_f=2):
        frames[f] = dets
    # After contact the ball reverses horizontal direction (vx flips sign).
    cx, cy = 60 + 24 * 23, 460 - 26 * 23 + 0.5 * 1.8 * 23 * 23
    for f, dets in _parabola(24, cx, cy, -24, -22, 1.8, start_f=27):
        frames[f] = dets
    tr = ParabolicBallTracker(fps)
    pos, st, segs = tr.resolve(frames)
    check("reversal produces two arcs (a committed direction change)", len(segs) == 2)
    check("never two balls at once (<=1 emission per frame)", emissions_ok_single(pos))
    # x-velocity sign genuinely flips between the two arcs.
    check("horizontal direction reverses between the arcs",
          len(segs) == 2 and _seg_vx(segs[0]) > 0 > _seg_vx(segs[1]))

    # --- Scenario 4: scattered false positives never form a track. ---
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(40):
        if rng.random() < 0.5:
            frames.append([(float(rng.integers(0, 900)), float(rng.integers(0, 500)), 0.3)])
        else:
            frames.append([])
    tr = ParabolicBallTracker(fps)
    pos, st, segs = tr.resolve(frames)
    check("scattered false positives yield no ballistic segment", len(segs) == 0)
    check("scattered false positives emit no ball", all(p is None for p in pos))

    # --- Scenario 5: a stray stationary ball alongside the real arc. ---
    frames = [[] for _ in range(40)]
    arc = dict(_parabola(30, 100, 400, 18, -20, 1.5, start_f=2))
    for f in range(2, 32):
        frames[f] = list(arc[f]) + [(700.0, 250.0, 0.8)]   # + a parked clutter ball
    tr = ParabolicBallTracker(fps)
    pos, st, segs = tr.resolve(frames)
    check("moving arc chosen over a parked stray (one segment)", len(segs) == 1)
    check("still exactly one ball emitted with a stray present", emissions_ok_single(pos))
    # The emitted ball follows the arc, not the static stray at x=700.
    tracked_x = [pos[f][0] for f in range(2, 32) if pos[f] is not None]
    check("emitted ball follows the arc, not the stationary stray",
          tracked_x and min(tracked_x) < 700 and max(tracked_x) < 700)

    # --- Scenario 7: court gating drops trajectories that bounce off-court. ---
    # Identity homography -> image coords are court feet; court is [0,27]x[0,78]
    # with the default 12ft lateral margin (in-court x in [-12, 39]).  Two rallies,
    # each a descending arc into a bounce then an ascending arc out:
    #   on-court  -> bounce at ~x=15 ft  (kept)
    #   neighbour -> bounce at ~x=60 ft  (past 39 -> whole rally dropped)
    def _mkseg(pts):
        s = _Segment()
        for (fr, x, y) in pts:
            s.add(fr, x, y)
        return s
    H = np.eye(3)
    A = _mkseg([(0, 10, 20), (1, 12, 30), (2, 14, 38), (3, 15, 42)])   # descend in
    B = _mkseg([(4, 16, 40), (5, 18, 30), (6, 20, 20), (7, 22, 10)])   # ascend out
    A2 = _mkseg([(60, 55, 20), (61, 57, 30), (62, 59, 38), (63, 60, 42)])  # neighbour
    B2 = _mkseg([(64, 61, 40), (65, 63, 30), (66, 65, 20), (67, 67, 10)])
    kept = ParabolicBallTracker(fps, homography=H)._court_gate([A, B, A2, B2])
    check("court gate keeps the on-court rally (bounce in court)",
          any(s is A for s in kept) and any(s is B for s in kept))
    check("court gate drops the neighbour rally (bounce off court)",
          not any(s is A2 for s in kept) and not any(s is B2 for s in kept))
    check("no homography leaves all trajectories untouched",
          len(ParabolicBallTracker(fps)._court_gate([A, B, A2, B2])) == 4)

    # --- Scenario 8: soft exclusion zones (keep pass-through, drop clutter). ---
    # A small zone the real arc briefly crosses, and a larger zone containing a
    # moving-clutter arc entirely.  The pass-through arc keeps few in-zone samples
    # (survives); the clutter arc is majority-in-zone (rejected).
    zones = [(235, 205, 290, 235), (400, 400, 480, 480)]
    frames = [[] for _ in range(50)]
    real = dict(_parabola(30, 60, 250, 12, -6, 0.5, start_f=2))   # sweeps across, clips zone 1
    for f, d in real.items():
        frames[f] = list(d)
    clutter = dict(_parabola(6, 410, 430, 8, 2, 0.2, start_f=40))  # entirely inside zone 2
    for f, d in clutter.items():
        frames[f] += list(d)
    segs_soft = ParabolicBallTracker(fps, exclusion_zones=zones).resolve(frames)[2]
    check("a real arc passing through a zone is kept",
          any(s.first_frame <= 10 <= s.last_frame for s in segs_soft))
    check("a moving-clutter arc living inside a zone is rejected",
          not any(s.first_frame >= 40 for s in segs_soft))
    # Without the zones the clutter arc would register as a segment.
    segs_open = ParabolicBallTracker(fps).resolve(frames)[2]
    check("without zones the same clutter arc does register",
          any(s.first_frame >= 40 for s in segs_open))

    # --- Scenario 9: perspective scaling recovers a small far-side arc. ---
    # A slow, small arc near the top of a 540px frame (span ~19px) is below the
    # flat move bar (30px) but above the perspective-scaled far-side bar (~12px).
    frames = [[] for _ in range(20)]
    fx, fy = 470.0, 40.0
    for k in range(8):
        fx += 2.6; fy += 0.5 + 0.1 * k
        frames[2 + k] = [(fx, fy, 0.6)]
    check("flat thresholds drop the small far-side arc",
          len(ParabolicBallTracker(fps).resolve(frames)[2]) == 0)
    check("perspective scaling recovers the small far-side arc",
          len(ParabolicBallTracker(
              fps, perspective_scale=make_image_row_perspective(540.0)
          ).resolve(frames)[2]) >= 1)

    # --- Scenario 6: highlight-reel range assembly. ---
    # 30 fps.  Three live blobs, plus a 3-frame flicker that must be dropped:
    #   A: frames 60-89
    #   B: frames 120-149, ~1.0s after A (< 3s) -> bridged into A's clip
    #   C: frames 400-429, a separate point (>3s gap)
    # Fixed 1.5s pre-roll (45 fr) and 1s post-roll (30 fr) padding.
    N = 460
    st6 = ["none"] * N
    for f in range(60, 90):
        st6[f] = "tracking"
    for f in range(120, 150):
        st6[f] = "tracking"
    for f in range(300, 303):     # tiny flicker -> dropped (< 0.3s)
        st6[f] = "tracking"
    for f in range(400, 430):
        st6[f] = "tracking"
    ranges = compute_highlight_ranges(st6, 30.0)
    check("highlight bridges <3s traces and drops the flicker (2 clips)",
          len(ranges) == 2)
    check("clip gets a 1.5s pre-roll (start = 60 - 45)",
          ranges and ranges[0][0] == 15)
    check("bridged clip runs through the gap to B's end + 1s post-roll (149 + 30)",
          ranges and ranges[0][1] == 179)
    check("second point also gets the 1.5s pre-roll (start = 400 - 45)",
          len(ranges) == 2 and ranges[1][0] == 355)
    check("the 300-302 flicker is not its own clip",
          all(not (a <= 301 <= b) for a, b in ranges))

    print()
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s) failed: {failures}")
        return 1
    print("SELF-TEST PASSED: all checks green.")
    return 0


def _seg_vx(seg: _Segment) -> float:
    """Fitted horizontal velocity (px/frame) of a segment."""
    seg._ensure_fit()
    return float(seg._xc[0])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        AnyaBallDetector(sys.argv[1]).process_video()
    else:
        sys.exit(_run_self_test())
