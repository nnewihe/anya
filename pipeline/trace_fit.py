"""
trace_fit.py
============
Piecewise-ballistic trajectory fitting over the replayed ball trace.

Between impacts (racket contact, court bounce) a tennis ball flies a smooth
arc: in image space x(t) is near-linear and y(t) near-quadratic (projected
gravity plus perspective).  The IMM tracker's causal state estimates are
noisy — differentiating them roughly doubles velocity noise — so this module
fits parametric curves to the RAW detections associated to the track,
segmented at the IMM's own impact signals:

  1. Collect (t, x, y) samples where a real detection was associated
     (ReplayFrame.det), skipping coasting frames entirely.
  2. Split at impact times — local maxima of racket_prob / bounce_prob —
     and at sample gaps longer than max_sample_gap_s.
  3. Per chunk, iteratively-reweighted least squares: y(t) quadratic,
     x(t) linear (upgraded to quadratic only when it clearly helps),
     dropping >3*MAD outliers between rounds.
  4. Chunks whose residual stays high are split at the worst residual and
     refit (depth-limited) — an undetected impact inside the chunk.

The fitted segments are the trace: pos(t)/vel(t) are analytic, residual RMS
is the per-segment quality score, and segment boundaries labeled "bounce"
are the anchors speed estimation maps through the court homography.

Run:
    python -m pipeline.trace_fit <telemetry.jsonl> [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .point_segmenter import (SegmenterConfig, MatchTelemetry, load_telemetry,
                              suppress_static_candidates, replay_ball_tracker,
                              ReplayFrame)


@dataclass
class FitConfig:
    racket_spike: float = 0.30      # impact split threshold on racket_prob
    bounce_spike: float = 0.30      # ... and on bounce_prob
    spike_min_sep_s: float = 0.25   # merge spikes closer than this
    max_sample_gap_s: float = 0.8   # never fit across a detection gap this long
    min_points: int = 5
    min_span_s: float = 0.16
    irls_rounds: int = 2
    outlier_k: float = 3.0          # drop residuals beyond k * MAD each round
    rms_split_px: float = 7.0       # refit-split chunks with RMS above this
    max_split_depth: int = 2
    x_quad_gain: float = 0.7        # quadratic x must cut x-RMS to this
                                    # fraction of linear's to be accepted


@dataclass
class FlightSegment:
    t0: float
    t1: float
    n: int
    cx: np.ndarray                  # np.polyfit coeffs (highest power first)
    cy: np.ndarray
    rms_px: float
    start_kind: str                 # "racket" | "bounce" | "appear" | "split"
    end_kind: str                   # "racket" | "bounce" | "vanish" | "split"

    def pos(self, t: float) -> Tuple[float, float]:
        return float(np.polyval(self.cx, t - self.t0)), \
               float(np.polyval(self.cy, t - self.t0))

    def vel(self, t: float) -> Tuple[float, float]:
        dx = np.polyval(np.polyder(self.cx), t - self.t0)
        dy = np.polyval(np.polyder(self.cy), t - self.t0)
        return float(dx), float(dy)

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def _impact_times(replay: Sequence[ReplayFrame], cfg: FitConfig
                  ) -> List[Tuple[float, str]]:
    """Local maxima of the IMM impact-model probabilities."""
    events: List[Tuple[float, str]] = []
    for kind, get, thr in (("racket", lambda fr: fr.racket_prob, cfg.racket_spike),
                           ("bounce", lambda fr: fr.bounce_prob, cfg.bounce_spike)):
        run_best: Optional[Tuple[float, float]] = None       # (prob, t)
        for fr in replay:
            p = get(fr)
            if p >= thr:
                if run_best is None or p > run_best[0]:
                    run_best = (p, fr.t)
            elif run_best is not None:
                events.append((run_best[1], kind))
                run_best = None
        if run_best is not None:
            events.append((run_best[1], kind))
    events.sort()
    merged: List[Tuple[float, str]] = []
    for t, kind in events:
        if merged and t - merged[-1][0] < cfg.spike_min_sep_s:
            # racket beats bounce when both fire — full reversal dominates
            if kind == "racket" and merged[-1][1] == "bounce":
                merged[-1] = (merged[-1][0], "racket")
            continue
        merged.append((t, kind))
    return merged


def _irls_fit(ts: np.ndarray, vs: np.ndarray, deg: int,
              cfg: FitConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively reweighted polyfit; returns (coeffs, residuals)."""
    keep = np.ones(len(ts), dtype=bool)
    coef = np.polyfit(ts, vs, deg)
    for _ in range(cfg.irls_rounds):
        res = vs - np.polyval(coef, ts)
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        new_keep = np.abs(res) <= cfg.outlier_k * 1.4826 * mad
        if new_keep.sum() < deg + 2:
            break
        keep = new_keep
        coef = np.polyfit(ts[keep], vs[keep], deg)
    return coef, vs - np.polyval(coef, ts)


def _fit_chunk(samples: List[Tuple[float, float, float]],
               start_kind: str, end_kind: str,
               cfg: FitConfig, depth: int = 0) -> List[FlightSegment]:
    if len(samples) < cfg.min_points:
        return []
    ts = np.array([s[0] for s in samples])
    xs = np.array([s[1] for s in samples])
    ys = np.array([s[2] for s in samples])
    t0 = float(ts[0])
    if ts[-1] - t0 < cfg.min_span_s:
        return []
    rel = ts - t0

    cx_lin, rx_lin = _irls_fit(rel, xs, 1, cfg)
    cx, rx = cx_lin, rx_lin
    if len(samples) >= cfg.min_points + 2:
        cx_q, rx_q = _irls_fit(rel, xs, 2, cfg)
        if np.sqrt(np.mean(rx_q ** 2)) < cfg.x_quad_gain * np.sqrt(np.mean(rx_lin ** 2)):
            cx, rx = cx_q, rx_q
    cy, ry = _irls_fit(rel, ys, 2, cfg)

    res = np.sqrt(rx ** 2 + ry ** 2)
    rms = float(np.sqrt(np.mean(res ** 2)))

    if rms > cfg.rms_split_px and depth < cfg.max_split_depth and \
            len(samples) >= 2 * cfg.min_points:
        # an undetected impact inside the chunk: split at the worst residual
        k = int(np.argmax(res))
        k = min(max(k, cfg.min_points - 1), len(samples) - cfg.min_points)
        left = _fit_chunk(samples[:k + 1], start_kind, "split", cfg, depth + 1)
        right = _fit_chunk(samples[k:], "split", end_kind, cfg, depth + 1)
        if left or right:
            return left + right

    return [FlightSegment(t0=t0, t1=float(ts[-1]), n=len(samples),
                          cx=cx, cy=cy, rms_px=rms,
                          start_kind=start_kind, end_kind=end_kind)]


def fit_flight_segments(match: MatchTelemetry,
                        cfg: Optional[SegmenterConfig] = None,
                        fcfg: Optional[FitConfig] = None,
                        replay: Optional[List[ReplayFrame]] = None,
                        t0: float = 0.0, t1: Optional[float] = None
                        ) -> List[FlightSegment]:
    cfg = cfg or SegmenterConfig()
    fcfg = fcfg or FitConfig()
    if match.meta:
        size = match.meta.get("analysis_size")
        if size:
            cfg.frame_height_px = float(size[1])
    suppress_static_candidates(match, cfg)
    if t1 is None:
        t1 = match.duration + 1.0
    if replay is None:
        replay = replay_ball_tracker(match, t0, t1, cfg)

    samples = [(fr.t, fr.det[0], fr.det[1]) for fr in replay
               if fr.det is not None]
    if not samples:
        return []
    impacts = _impact_times(replay, fcfg)

    # chunk boundaries: impact times + long sample gaps
    segments: List[FlightSegment] = []
    chunk: List[Tuple[float, float, float]] = []
    start_kind = "appear"
    imp_i = 0
    for s in samples:
        boundary_kind = None
        while imp_i < len(impacts) and impacts[imp_i][0] <= s[0]:
            if chunk and impacts[imp_i][0] > chunk[0][0]:
                boundary_kind = impacts[imp_i][1]
            imp_i += 1
        if chunk and s[0] - chunk[-1][0] > fcfg.max_sample_gap_s:
            segments += _fit_chunk(chunk, start_kind, "vanish", fcfg)
            chunk, start_kind = [], "appear"
        elif boundary_kind:
            chunk.append(s)                       # impact sample ends the arc
            segments += _fit_chunk(chunk, start_kind, boundary_kind, fcfg)
            chunk, start_kind = [], boundary_kind
            continue
        chunk.append(s)
    segments += _fit_chunk(chunk, start_kind, "vanish", fcfg)
    return segments


def write_segments_csv(segments: List[FlightSegment], path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t0", "t1", "dur_s", "n", "rms_px",
                    "start_kind", "end_kind",
                    "x0", "y0", "x1", "y1", "vx0", "vy0", "speed0_px_s"])
        for s in segments:
            x0, y0 = s.pos(s.t0); x1, y1 = s.pos(s.t1)
            vx, vy = s.vel(s.t0)
            w.writerow([f"{s.t0:.3f}", f"{s.t1:.3f}", f"{s.duration:.3f}",
                        s.n, f"{s.rms_px:.2f}", s.start_kind, s.end_kind,
                        f"{x0:.1f}", f"{y0:.1f}", f"{x1:.1f}", f"{y1:.1f}",
                        f"{vx:.0f}", f"{vy:.0f}",
                        f"{math.hypot(vx, vy):.0f}"])
    print(f"[FIT] Wrote {len(segments)} flight segments → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Piecewise-ballistic ball-trace fitting")
    parser.add_argument("telemetry")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    match = load_telemetry(args.telemetry)
    segs = fit_flight_segments(match)
    durs = sorted(s.duration for s in segs)
    rmss = sorted(s.rms_px for s in segs)
    if segs:
        total = sum(s.duration for s in segs)
        print(f"[FIT] {len(segs)} segments, {total:.0f}s fitted flight, "
              f"median dur {durs[len(durs)//2]:.2f}s, "
              f"median RMS {rmss[len(rmss)//2]:.2f}px, "
              f"p90 RMS {rmss[int(len(rmss)*0.9)]:.2f}px")
    if args.csv:
        write_segments_csv(segs, args.csv)
