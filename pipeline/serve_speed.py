"""
serve_speed.py
==============
Serve-speed estimation from fitted ball-flight segments + court geometry.

Principle: the homography maps image points to court coordinates ONLY for
points on the court plane.  A mid-air ball projects to a court position
deeper than its true ground point, so the two anchors used here are the only
two trustworthy ones:

  BOUNCE   the serve arc's end pixel, on the plane by definition
  CONTACT  the server's feet at contact (on the plane) plus a nominal
           contact height above them

The visible trace starts after contact (tracker confirmation lag), so
contact time is estimated by extrapolating the fitted arc backward to its
closest approach to the server's contact point in image space, clamped to a
plausible window.

    speed = sqrt(ground_distance² + contact_height²) / (t_bounce - t_contact)

reported in mph.  This is the average speed over the flight — radar-gun
"serve speed" is measured at contact and reads ~10-15% higher due to drag
deceleration; treat these numbers as a consistent relative measure.

Every estimate carries validity flags (bounce landed inside the plausible
service area, contact extrapolation stayed sane, fit quality) — consumers
should filter on `valid`.

Run:
    python -m pipeline.serve_speed /path/to/video.mp4 [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .point_segmenter import (SegmenterConfig, load_telemetry, segment_match,
                              replay_ball_tracker)
from .match_telemetry import telemetry_path_for
from .trace_fit import FitConfig, FlightSegment, fit_flight_segments
from .utilities import Config


@dataclass
class SpeedConfig:
    contact_height_ft: float = 8.5    # typical adult serve contact height
    window_before_s:   float = 1.0    # fit window around the serve event
    window_after_s:    float = 4.5
    min_seg_dur_s:     float = 0.25   # serve arc must last this long
    min_arc_px:        float = 60.0   # ... and cover this much image path
    min_arc_dy_px:     float = 25.0   # net vertical motion toward the
                                      # receiving court (sign per side)
    max_seg_rms_px:    float = 8.0
    max_back_extrap_s: float = 0.6    # how far before the trace the arc may
                                      # be extrapolated to find contact
    flight_s_range: Tuple[float, float] = (0.35, 1.15)  # plausible serve
                                      # contact→bounce time at club speeds
    # plausible bounce areas in court feet (y from the near baseline, net=39)
    bounce_y_far_serve:  Tuple[float, float] = (8.0, 45.0)   # lands near side
    bounce_y_near_serve: Tuple[float, float] = (33.0, 70.0)  # lands far side
    bounce_x_pad_ft: float = 3.0


@dataclass
class ServeSpeed:
    point: int
    side: str
    serve_t: float
    t_contact: float
    t_bounce: float
    bounce_world: Tuple[float, float]
    ground_dist_ft: float
    speed_mph: float
    seg_rms_px: float
    end_kind: str
    valid: bool
    reason: str          # "" when valid


def _homography(court_cache_path: str) -> np.ndarray:
    import cv2
    with open(court_cache_path, "r") as fh:
        pts = json.load(fh)["points"]
    BL, BR, TR, TL = [tuple(p) for p in pts]
    dst = np.array([[0, 0], [Config.COURT_WIDTH_FT, 0],
                    [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT],
                    [0, Config.COURT_LENGTH_FT]], dtype=np.float32)
    src = np.array([BL, BR, TR, TL], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def _world(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    import cv2
    pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), H)
    return float(pt[0][0][0]), float(pt[0][0][1])


def _server_anchor(match, serve_t: float, side: str
                   ) -> Optional[Tuple[float, float, float]]:
    """(feet_x, feet_y, head_y) of the serving player around serve_t."""
    for rec in match.slice(serve_t - 0.5, serve_t + 2.0):
        box = rec.near_box if side == "near" else rec.far_box
        if box is not None:
            return ((box[0] + box[2]) / 2.0, float(box[3]), float(box[1]))
    return None


def estimate_serve_speeds(video_path: str,
                          scfg: Optional[SpeedConfig] = None,
                          verbose: bool = True) -> List[ServeSpeed]:
    scfg = scfg or SpeedConfig()
    telemetry = telemetry_path_for(video_path)
    stem = os.path.splitext(os.path.abspath(video_path))[0]
    H = _homography(stem + "_court_cache.json")

    match = load_telemetry(telemetry)
    cfg = SegmenterConfig()
    segments = segment_match(match, cfg, verbose=False)
    from .ball_tracker import make_image_row_perspective
    persp = make_image_row_perspective(cfg.frame_height_px)

    out: List[ServeSpeed] = []
    for pt in segments:
        replay = replay_ball_tracker(
            match, pt.serve_t - scfg.window_before_s,
            pt.serve_t + scfg.window_after_s, cfg)
        flights = fit_flight_segments(match, cfg, FitConfig(), replay=replay,
                                      t0=pt.serve_t - scfg.window_before_s,
                                      t1=pt.serve_t + scfg.window_after_s)
        # The serve arc: substantial path, heading toward the opposite court
        # (near serve → up-frame, far serve → down-frame).  Pre-serve tosses
        # and ball bounces are short/slow/vertical; returns come later and
        # head the other way.  Take the fastest qualifying arc.
        def _is_serve_arc(s: FlightSegment) -> bool:
            if s.duration < scfg.min_seg_dur_s:
                return False
            x0, y0 = s.pos(s.t0)
            x1, y1 = s.pos(s.t1)
            # far-court arcs are perspective-compressed — scale the pixel
            # thresholds to the arc's depth or every far serve gets filtered
            sc = max(persp((y0 + y1) / 2.0), 0.35)
            if math.hypot(x1 - x0, y1 - y0) < scfg.min_arc_px * sc:
                return False
            dy = y1 - y0
            return dy < -scfg.min_arc_dy_px * sc if pt.side == "near" \
                else dy > scfg.min_arc_dy_px * sc

        candidates = [s for s in flights if _is_serve_arc(s)]
        serve_seg = max(
            candidates,
            key=lambda s: math.hypot(*(np.subtract(s.pos(s.t1), s.pos(s.t0))))
            / max(s.duration, 1e-6),
            default=None)
        if serve_seg is None:
            out.append(ServeSpeed(pt.point, pt.side, pt.serve_t, 0, 0, (0, 0),
                                  0, 0, 0, "", False, "no serve arc"))
            continue

        # ---- contact time: extrapolate the arc back toward the server ----
        anchor = _server_anchor(match, pt.serve_t, pt.side)
        t_contact = serve_seg.t0
        if anchor is not None:
            ax, _, head_y = anchor
            # contact point ≈ above the server's head; search the arc's
            # closest approach in [t0 - max_back_extrap, t0]
            taus = np.linspace(-scfg.max_back_extrap_s, 0.0, 25)
            best = None
            for tau in taus:
                x = float(np.polyval(serve_seg.cx, tau))
                y = float(np.polyval(serve_seg.cy, tau))
                d = math.hypot(x - ax, y - head_y)
                if best is None or d < best[0]:
                    best = (d, serve_seg.t0 + tau)
            t_contact = best[1]

        t_bounce = serve_seg.t1
        bx, by = serve_seg.pos(t_bounce)
        wx, wy = _world(H, bx, by)

        # ---- contact ground point ----
        # x from the server's feet through the homography, y CLAMPED to the
        # serving baseline: far-side feet pixels project with tens of feet
        # of depth error (the recurring lesson of this pipeline), while the
        # rules pin the server's depth at the baseline anyway.
        base_y = 0.0 if pt.side == "near" else Config.COURT_LENGTH_FT
        if anchor is not None:
            awx, _ = _world(H, anchor[0], anchor[1])
            cwx, cwy = min(max(awx, -2.0), Config.COURT_WIDTH_FT + 2.0), base_y
        else:
            cwx, cwy = Config.COURT_WIDTH_FT / 2.0, base_y

        dist = math.hypot(wx - cwx, wy - cwy)
        dt = t_bounce - t_contact
        speed_mph = 0.0
        if dt > 0.05:
            fps_ = math.sqrt(dist ** 2 + scfg.contact_height_ft ** 2) / dt
            speed_mph = fps_ * 0.681818

        # ---- validity ----
        reason = ""
        ylo, yhi = (scfg.bounce_y_far_serve if pt.side == "far"
                    else scfg.bounce_y_near_serve)
        if serve_seg.rms_px > scfg.max_seg_rms_px:
            reason = f"fit rms {serve_seg.rms_px:.1f}px"
        elif not (-scfg.bounce_x_pad_ft <= wx <=
                  Config.COURT_WIDTH_FT + scfg.bounce_x_pad_ft):
            reason = f"bounce x {wx:.0f}ft off court"
        elif not ylo <= wy <= yhi:
            reason = f"bounce y {wy:.0f}ft outside serve landing zone"
        elif serve_seg.end_kind == "vanish":
            reason = "arc lost before bounce"
        elif not scfg.flight_s_range[0] <= dt <= scfg.flight_s_range[1]:
            # a serve crosses the court in ~0.5-1.1 s; longer means the arc
            # sailed through an undetected bounce, shorter means a fragment
            reason = f"flight {dt:.2f}s implausible"
        elif not 35.0 <= speed_mph <= 140.0:
            reason = f"speed {speed_mph:.0f}mph implausible"

        out.append(ServeSpeed(pt.point, pt.side, pt.serve_t, t_contact,
                              t_bounce, (wx, wy), dist, speed_mph,
                              serve_seg.rms_px, serve_seg.end_kind,
                              reason == "", reason))

    if verbose:
        print(f"{'pt':>3} {'side':>4} {'serve_t':>8} {'speed':>7} "
              f"{'dist':>6} {'flight':>6} {'bounce(x,y)ft':>14} "
              f"{'rms':>5} {'end':>7}  status")
        for s in out:
            if s.t_bounce:
                print(f"{s.point:3d} {s.side:>4} {s.serve_t:8.1f} "
                      f"{s.speed_mph:6.1f}m {s.ground_dist_ft:5.1f}f "
                      f"{s.t_bounce - s.t_contact:5.2f}s "
                      f"({s.bounce_world[0]:5.1f},{s.bounce_world[1]:5.1f}) "
                      f"{s.seg_rms_px:5.1f} {s.end_kind:>7}  "
                      f"{'OK' if s.valid else s.reason}")
            else:
                print(f"{s.point:3d} {s.side:>4} {s.serve_t:8.1f} "
                      f"{'—':>7} {'—':>6} {'—':>6} {'—':>14} {'—':>5} "
                      f"{'—':>7}  {s.reason}")
        good = [s.speed_mph for s in out if s.valid]
        if good:
            good.sort()
            print(f"\n{len(good)}/{len(out)} serves measured: "
                  f"median {good[len(good)//2]:.0f} mph, "
                  f"range {good[0]:.0f}–{good[-1]:.0f} mph")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve-speed estimation")
    parser.add_argument("video", help="source video (telemetry + court cache beside it)")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    speeds = estimate_serve_speeds(args.video)
    if args.csv:
        import csv as _csv
        with open(args.csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["point", "side", "serve_t", "t_contact", "t_bounce",
                        "bounce_wx_ft", "bounce_wy_ft", "ground_dist_ft",
                        "speed_mph", "seg_rms_px", "end_kind", "valid", "reason"])
            for s in speeds:
                w.writerow([s.point, s.side, f"{s.serve_t:.2f}",
                            f"{s.t_contact:.2f}", f"{s.t_bounce:.2f}",
                            f"{s.bounce_world[0]:.1f}", f"{s.bounce_world[1]:.1f}",
                            f"{s.ground_dist_ft:.1f}", f"{s.speed_mph:.1f}",
                            f"{s.seg_rms_px:.2f}", s.end_kind,
                            int(s.valid), s.reason])
        print(f"[SPEED] Wrote {args.csv}")
