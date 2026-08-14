"""
serve_onset_lag.py  —  Phase 0 spike for the point-detection system.
====================================================================

Question this spike answers (the make-or-break assumption of the plan):

    If we define "serve impact" as the ONSET of the first ballistic ball
    segment of a rally, how well does that onset line up with the true serve?
    Specifically, what is the distribution of

        lag = ball_onset_time  -  ground_truth_rally_start_time

    across real rallies, and how often is the ball detected so LATE that we
    need the player-kinematics fallback to anchor the serve instead?

Why it matters: the point start is `serve_impact - 2s`.  If ball onset is a
stable, small offset from the true serve we can trust it (widen the pre-roll if
the offset is consistently positive).  If it is large / high-variance -- which
the dead-time-cutter notes warn about for near serves flying into the occluded
far court -- we need the player-magnitude trigger (serve_detector2 idea) as a
second anchor and take the earlier of the two.

This spike does NOT run YOLO.  It reuses the real perception output already on
disk (the dead-time cutter's `<video>_match_telemetry.jsonl`, which stores
per-frame ball detections + near/far player world-ft) and feeds it through the
REAL `ParabolicBallTracker` from ball_detector.py, so the segment structure it
measures is exactly what the production tracker would produce.

Run:
    python spikes/serve_onset_lag.py            # folder 68 (has telemetry)
    python spikes/serve_onset_lag.py --folder 68
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Make `pipeline` importable when run as `python spikes/serve_onset_lag.py`.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "pipeline"))

from ball_detector import ParabolicBallTracker  # noqa: E402
from utilities import Config  # noqa: E402
import cv2  # noqa: E402


# =====================================================================
# Ground-truth loading (frames vs seconds are both in the wild).
# =====================================================================
@dataclass
class Rally:
    start_s: float
    end_s: float
    serve: str


def load_ground_truth(folder: str, fps: float) -> List[Rally]:
    """Load GT rallies, normalising frames-or-seconds to seconds."""
    for name in ("ground_truth.json", "derived_ground_truth.json"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            out = []
            for r in data["rallies"]:
                if "start_s" in r:            # seconds schema (folder 68)
                    out.append(Rally(float(r["start_s"]), float(r["end_s"]),
                                     r.get("serve", "?")))
                else:                          # frames schema (folders 21, 23)
                    out.append(Rally(r["start"] / fps, r["end"] / fps,
                                     r.get("serve", "?")))
            return out
    raise FileNotFoundError(f"no ground_truth.json/derived_ground_truth.json in {folder}")


# =====================================================================
# Telemetry loading: per-frame ball detections + player world positions.
# =====================================================================
@dataclass
class Telemetry:
    fps: float
    n_frames: int
    balls: List[List[Tuple[float, float, float]]]     # per frame: [(x,y,conf), ...] @ 960-wide
    near_w: List[Optional[Tuple[float, float]]]        # per frame: near player (wx,wy) ft or None
    far_w: List[Optional[Tuple[float, float]]]         # per frame: far  player (wx,wy) ft or None


def load_telemetry(path: str) -> Telemetry:
    fps = 30.0
    n = 0
    balls: List[List[Tuple[float, float, float]]] = []
    near_w: List[Optional[Tuple[float, float]]] = []
    far_w: List[Optional[Tuple[float, float]]] = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "meta" in rec:
                fps = float(rec["meta"].get("fps", 30.0))
                n = int(rec["meta"].get("total_frames", 0))
                continue
            # Merge the whole-frame `balls` channel with the far-crop native-res
            # `fballs` re-detection.  The cutter notes far-serve starts collapse
            # (16->4 on this folder) without fballs, so a fair onset measurement
            # must include it; both are in 960-wide analysis coords.
            b = [(float(x), float(y), float(c)) for (x, y, c) in rec.get("balls", [])]
            b += [(float(x), float(y), float(c)) for (x, y, c) in rec.get("fballs", [])]
            balls.append(b)
            npw = rec.get("npw")
            fpw = rec.get("fpw")
            near_w.append((float(npw[0]), float(npw[1])) if npw else None)
            far_w.append((float(fpw[0]), float(fpw[1])) if fpw else None)
    return Telemetry(fps=fps, n_frames=n or len(balls), balls=balls,
                     near_w=near_w, far_w=far_w)


def load_ball_dets(path: str):
    """Load a `<stem>_ball_dets.jsonl` cache built by build_ball_dets.py.

    Returns (fps, width, court_points, per-frame [(x,y,conf), ...]) in
    FULL-RESOLUTION pixels.
    """
    fps, width, court_points = 30.0, 960, []
    balls: List[List[Tuple[float, float, float]]] = []
    with open(path) as f:
        meta = json.loads(f.readline())["meta"]
        fps = float(meta.get("fps", 30.0))
        width = int(meta.get("width", 960))
        court_points = meta.get("court_points", []) or []
        for line in f:
            balls.append([(float(x), float(y), float(c)) for (x, y, c) in json.loads(line)])
    return fps, width, court_points, balls


def homography_from_points(court_points, scale: float = 1.0) -> Optional[np.ndarray]:
    """Image->court-feet homography from full-res court corners (BL,BR,TR,TL order)."""
    if not court_points or len(court_points) != 4:
        return None
    pts = [(p[0] * scale, p[1] * scale) for p in court_points]
    ordered = sorted(pts, key=lambda p: p[1])
    far_pair, near_pair = ordered[:2], ordered[2:]
    TL, TR = sorted(far_pair, key=lambda p: p[0])
    BL, BR = sorted(near_pair, key=lambda p: p[0])
    src = np.array([BL, BR, TR, TL], dtype=np.float32)
    dst = np.array([[0, 0], [Config.COURT_WIDTH_FT, 0],
                    [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT],
                    [0, Config.COURT_LENGTH_FT]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def build_homography(court_cache_path: str) -> Optional[np.ndarray]:
    """Image(960-wide)->court-feet homography from a cached 4-corner click.

    Uses the same BL,BR,TR,TL ordering ball_detector._order_court_corners derives.
    """
    if not os.path.isfile(court_cache_path):
        return None
    with open(court_cache_path) as f:
        cache = json.load(f)
    pts = cache["points"]
    ordered = sorted(pts, key=lambda p: p[1])
    far_pair, near_pair = ordered[:2], ordered[2:]
    TL, TR = sorted(far_pair, key=lambda p: p[0])
    BL, BR = sorted(near_pair, key=lambda p: p[0])
    src = np.array([BL, BR, TR, TL], dtype=np.float32)
    dst = np.array([[0, 0], [Config.COURT_WIDTH_FT, 0],
                    [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT],
                    [0, Config.COURT_LENGTH_FT]], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


# =====================================================================
# Onset extraction: first live-ball frame of each detected rally.
# =====================================================================
def rally_onsets(states: List[str], fps: float, quiet_s: float = 3.0,
                 blip_s: float = 0.25) -> List[int]:
    """Frames where a fresh ball rally begins.

    A rally onset = a live frame (state != 'none') preceded by >= quiet_s of
    dead time, where micro-blips shorter than blip_s do not reset the quiet
    clock (mirrors the cutter's anti-flicker quiet gate).
    """
    quiet = int(round(quiet_s * fps))
    blip = int(round(blip_s * fps))
    n = len(states)
    live = [s != "none" for s in states]

    # Suppress micro-blips so a 1-2 frame false detection in dead time doesn't
    # look like a rally onset or reset the quiet clock.
    i = 0
    while i < n:
        if live[i]:
            j = i
            while j + 1 < n and live[j + 1]:
                j += 1
            if (j - i + 1) < blip:
                for k in range(i, j + 1):
                    live[k] = False
            i = j + 1
        else:
            i += 1

    onsets = []
    last_live = -10 ** 9
    for f in range(n):
        if live[f]:
            if f - last_live > quiet:
                onsets.append(f)
            last_live = f
    return onsets


# =====================================================================
# Player-kinematics serve trigger (the fusion fallback candidate).
# =====================================================================
def _world_speed_ftps(track, f0: int, f1: int, fps: float) -> List[Tuple[int, float]]:
    """Per-frame world speed (ft/s) of a player track over [f0, f1)."""
    pts = [(f, track[f]) for f in range(f0, min(f1, len(track))) if track[f] is not None]
    out = []
    for k in range(1, len(pts)):
        (fa, pa), (fb, pb) = pts[k - 1], pts[k]
        df = max(1, fb - fa)
        out.append((fb, np.hypot(pb[0] - pa[0], pb[1] - pa[1]) / df * fps))
    return out


def kinematics_serve_trigger(near_w, far_w, fps: float, serve_side: str,
                             search_f0: int, search_f1: int,
                             steady_s: float = 0.6, still_ftps: float = 1.2,
                             break_ftps: float = 3.5) -> Optional[int]:
    """Serve-motion onset from the SERVER's kinematics (serve_detector2 idea).

    Replicates the PRE_SERVE->POINT_STARTED shape: require the serving-side
    player to be STILL (speed < still_ftps) for >= steady_s, then return the
    first frame their speed BREAKS above break_ftps.  This is the real fallback
    anchor — not raw first-motion (which just fires on inter-point walking).
    Returns the break frame in [search_f0, search_f1), or None.
    """
    track = near_w if serve_side == "near" else far_w
    steady = int(round(steady_s * fps))
    spd = _world_speed_ftps(track, max(0, search_f0), search_f1, fps)
    if len(spd) < steady + 1:
        return None
    still_run = 0
    for f, s in spd:
        if s < still_ftps:
            still_run += 1
        else:
            if still_run >= steady and s > break_ftps:
                return f           # steadiness established, now it breaks -> serve
            still_run = 0
    return None


# =====================================================================
# Driver
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="68")
    ap.add_argument("--data-root", default="/Volumes/Anya/Data")
    ap.add_argument("--match-window-s", type=float, default=8.0,
                    help="max |onset - gt_start| to associate an onset with a rally")
    ap.add_argument("--prefer-telemetry", action="store_true",
                    help="use *_match_telemetry.jsonl even if a ball-dets cache exists")
    args = ap.parse_args()

    folder = os.path.join(args.data_root, args.folder)

    # Two perception sources: the cutter's telemetry (folder 68, 960-wide balls
    # + player world) or a headless ball-dets cache (folders 21/23, full-res
    # balls, no player world -> no kinematics fallback).
    dets_candidates = [f for f in os.listdir(folder) if f.endswith("_ball_dets.jsonl")]
    tel_candidates = [f for f in os.listdir(folder) if f.endswith("_match_telemetry.jsonl")]

    if dets_candidates and not args.prefer_telemetry:
        dets_path = os.path.join(folder, dets_candidates[0])
        print(f"[INFO] ball-dets cache: {dets_path}")
        fps, width, court_points, balls = load_ball_dets(dets_path)
        px_scale = width / float(Config.ANALYSIS_WIDTH)
        H = homography_from_points(court_points)      # full-res corners, scale 1.0
        near_w = far_w = [None] * len(balls)
        n_frames = len(balls)
    elif tel_candidates:
        tel_path = os.path.join(folder, tel_candidates[0])
        stem = tel_candidates[0].replace("_match_telemetry.jsonl", "")
        court_path = os.path.join(folder, f"{stem}_court_cache.json")
        print(f"[INFO] telemetry: {tel_path}")
        tel = load_telemetry(tel_path)
        fps, balls, near_w, far_w = tel.fps, tel.balls, tel.near_w, tel.far_w
        px_scale = 1.0                                # telemetry balls are 960-wide
        H = build_homography(court_path)
        n_frames = tel.n_frames
    else:
        print(f"[ERR] no *_ball_dets.jsonl or *_match_telemetry.jsonl in {folder}. "
              f"Run: python spikes/build_ball_dets.py {folder}/snippet.mp4")
        return 1

    print(f"[INFO] fps={fps:.3f} frames={n_frames} px_scale={px_scale:.3f} "
          f"ball-frames={sum(1 for b in balls if b)}")
    print(f"[INFO] homography: {'loaded (court gating on)' if H is not None else 'MISSING (no court gating)'}")

    gt = load_ground_truth(folder, fps)
    print(f"[INFO] ground-truth rallies: {len(gt)}")

    tracker = ParabolicBallTracker(fps=fps, px_scale=px_scale, homography=H)
    positions, states, segments = tracker.resolve(balls)
    print(f"[INFO] resolved {len(segments)} ballistic segment(s); "
          f"live in {sum(1 for s in states if s != 'none')}/{len(states)} frames")

    onsets = rally_onsets(states, fps)
    onset_times = [f / fps for f in onsets]
    print(f"[INFO] detected {len(onsets)} rally onset(s)\n")

    # Associate each GT rally with the FIRST onset searched forward from just
    # before its start (directional: the serve onset can't be the tail of the
    # previous point).  `pre_tol_s` lets the onset lead GT start slightly (GT
    # start is serve-motion; a tracked toss can register a hair earlier).
    pre_tol_s = 2.5
    rows = []
    matched = 0
    ball_lags, fused_lags = [], []
    for r in gt:
        cand = sorted(t for t in onset_times
                      if r.start_s - pre_tol_s <= t <= r.start_s + args.match_window_s)
        if not cand:
            rows.append((r, None, None, None))
            continue
        matched += 1
        t_onset = cand[0]
        f_onset = int(round(t_onset * fps))
        ball_lag = t_onset - r.start_s
        ball_lags.append(ball_lag)

        # Kinematics fallback: search the serving player's break in a window
        # that starts before the ball onset (the serve motion precedes the ball).
        kf = kinematics_serve_trigger(
            near_w, far_w, fps, r.serve,
            search_f0=int((r.start_s - 3.0) * fps),
            search_f1=int((r.start_s + args.match_window_s) * fps))
        kt = kf / fps if kf is not None else None
        # Fused anchor = earlier of ball-onset and kinematics trigger, but only
        # let kinematics pull EARLIER (it anchors the serve, never delays it).
        fused_t = t_onset if kt is None else min(t_onset, kt)
        fused_lags.append(fused_t - r.start_s)
        rows.append((r, ball_lag, kt - r.start_s if kt is not None else None,
                     fused_t - r.start_s))

    # ---- report -------------------------------------------------------
    print(f"{'GT start':>9} {'side':>4} {'ball_lag':>9} {'kin_lag':>8} {'fused_lag':>9}")
    for r, bl, kl, fl in rows:
        def fmt(v):
            return f"{v:+.2f}" if v is not None else "   --"
        print(f"{r.start_s:9.2f} {r.serve:>4} {fmt(bl):>9} {fmt(kl):>8} {fmt(fl):>9}")

    def stats(name, xs):
        if not xs:
            print(f"  {name}: no matches")
            return
        a = np.array(xs)
        print(f"  {name}: n={len(a)} mean={a.mean():+.2f}s median={np.median(a):+.2f}s "
              f"p10={np.percentile(a,10):+.2f} p90={np.percentile(a,90):+.2f} "
              f"min={a.min():+.2f} max={a.max():+.2f}")

    print(f"\n[SUMMARY] matched {matched}/{len(gt)} GT rallies "
          f"(window ±{args.match_window_s:.0f}s)")
    stats("ball-onset lag ", ball_lags)
    stats("fused-anchor lag", fused_lags)
    # How many need the fallback: ball onset arrives > 0.5s after GT start AND
    # the kinematics trigger beats it by a meaningful margin.
    late = sum(1 for r, bl, kl, fl in rows
               if bl is not None and bl > 0.5 and fl is not None and fl < bl - 0.3)
    print(f"[SUMMARY] rallies where kinematics fallback pulls the anchor earlier "
          f"by >0.3s: {late}/{matched}")
    print("\nInterpretation guide:")
    print("  lag ~ 0     -> ball onset ≈ GT serve start; 2s pre-roll is plenty.")
    print("  lag > 0     -> ball detected AFTER serve start; pre-roll must cover it.")
    print("  wide spread -> ball onset unreliable alone; fusion fallback earns its keep.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
