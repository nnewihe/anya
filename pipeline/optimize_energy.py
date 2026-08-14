"""
optimize_energy.py
==================
Optimize the energy-bar point-END parameters against near-side ground truth.

Two decoupled stages (so the search is fast):

  1. Telemetry cache  (build_telemetry) — run detection ONCE per clip over each
     near rally's span at 960x540, storing per-frame near_player_bbox and
     exclusion-filtered ball_pos. Cached to <clip>/energy_telemetry_cache.json.

  2. Optimize (optimize) — replay the EXACT production energy state-machine
     (PointStartSystem, driven by a params dict) over the cached telemetry for
     each trial, score predicted point-end vs GT end with an asymmetric
     seconds objective (early-ends penalised harder), and search with
     scipy.optimize.differential_evolution. Clips are weighted equally.

Ground truth: <clip>/ground_truth.json = {"rallies":[{"start","end","serve"}]}.
Only serve == "near" rallies are used. Frames are in each clip's own fps; the
objective converts to seconds so 30 and 60 fps clips weigh the same.

Usage:
    python pipeline/optimize_energy.py --extract        # build telemetry caches
    python pipeline/optimize_energy.py --optimize       # run the search
    python pipeline/optimize_energy.py --extract --optimize --loco
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# PointStartSystem/TrackData/MatchState moved to the archive module when
# anya_near_serve.py was rewritten as the graded near-serve scorer.
from anya_near_serve_archive import PointStartSystem, TrackData, MatchState
from utilities import _is_in_exclusion_zone, create_auto_exclusion_zones, load_cached_exclusion_zones

# ── Config ────────────────────────────────────────────────────────────────────

ANALYSIS_SIZE   = (960, 540)   # matches the cached court/exclusion coordinate space
PLAYER_STRIDE   = 10           # detect near player every N frames (matches production)
PLAYER_IMGSZ    = 640
BALL_IMGSZ      = 1280
BALL_CONF       = 0.25
MARGIN_SEC      = 6.0          # replay slack after GT end (bounds "no-end" penalty)
EARLY_WEIGHT    = 3.0          # asymmetric: ending before GT is this much worse than after

# The parameters under optimization (resolution-independent) and their bounds.
# The BOOST terms were added to the search: with fixed boosts the earlier fit
# ended points ~4s late because a rolling/retrieved ball kept pumping energy;
# letting the boosts weaken relative to the drain is the direct lever.
PARAM_SPACE = [
    ("ENERGY_DECAY_DEAD",    0.3, 6.0),
    ("ENERGY_DECAY_MISSING", 0.3, 6.0),
    ("ENERGY_DECAY_BASE",    0.05, 3.0),
    ("STILL_PROLONGED_SEC",  0.0, 3.0),
    ("BALL_TRACE_SEC",       0.2, 3.0),
    ("PLAYER_STILL_FTS",     0.5, 6.0),
    ("ENERGY_BOOST_BALL",    0.2, 6.0),
    ("ENERGY_BOOST_MOTION",  0.2, 6.0),
]
PARAM_NAMES = [p[0] for p in PARAM_SPACE]
BOUNDS      = [(lo, hi) for _, lo, hi in PARAM_SPACE]

TELEMETRY_CACHE = "energy_telemetry_cache.json"


# ── Clip discovery ────────────────────────────────────────────────────────────

def _video_path(clip_dir: str) -> Optional[str]:
    for name in ("snippet.mp4", "match.mp4"):
        p = os.path.join(clip_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _court_cache_path(video_path: str) -> str:
    d = os.path.dirname(video_path)
    s = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{s}_court_cache.json")


def _near_rallies(clip_dir: str) -> List[Dict]:
    gt = os.path.join(clip_dir, "ground_truth.json")
    if not os.path.isfile(gt):
        return []
    rallies = json.load(open(gt)).get("rallies", [])
    return [r for r in rallies if r.get("serve") == "near"]


def discover_clips(data_root: str) -> List[str]:
    """Clip dirs that have near rallies, a video, and a court-corner cache."""
    clips = []
    for name in sorted(os.listdir(data_root)):
        clip_dir = os.path.join(data_root, name)
        if not os.path.isdir(clip_dir):
            continue
        vid = _video_path(clip_dir)
        if vid is None or not _near_rallies(clip_dir):
            continue
        if not os.path.isfile(_court_cache_path(vid)):
            print(f"[skip] clip {name}: near rallies but no court cache — calibrate first")
            continue
        clips.append(clip_dir)
    return clips


def _load_corners(video_path: str) -> np.ndarray:
    """Court corners in 960x540 space from the shared court cache."""
    data = json.load(open(_court_cache_path(video_path)))
    return np.array(data["points"], dtype=np.float32)


# ── Stage 1: telemetry extraction ─────────────────────────────────────────────

def _exclusion_zones(video_path, ball_model, device):
    zones = load_cached_exclusion_zones(video_path)   # 960x540 space if present
    if zones is not None:
        return zones
    print("  building 960x540 exclusion zones ...")
    return create_auto_exclusion_zones(video_path, ball_model, analysis_size=ANALYSIS_SIZE)


def _detect_ball(frame, ball_model, device, zones):
    res = ball_model.predict(frame, conf=BALL_CONF, imgsz=BALL_IMGSZ, device=device, verbose=False)[0]
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        if _is_in_exclusion_zone(cx, cy, zones):
            continue
        return (float(cx), float(cy))
    return None


def _detect_near_player(frame, player_model, device, geom):
    res = player_model.predict(frame, classes=[0], imgsz=PLAYER_IMGSZ, device=device, verbose=False)[0]
    best, best_key = None, None
    for box in res.boxes:
        x, y, x2, y2 = box.xyxy[0].cpu().numpy()
        w, h = x2 - x, y2 - y
        wx, wy = geom._get_world_coords(x + w / 2.0, y + h)
        if -2.0 <= wx <= 29.0 and wy <= 38.0:      # near half
            key = abs(wy - 0.0)
            if best_key is None or key < best_key:
                best_key, best = key, (float(x), float(y), float(w), float(h))
    return best


def build_telemetry(clip_dir, ball_model, player_model, device, rescan=False):
    cache_path = os.path.join(clip_dir, TELEMETRY_CACHE)
    if os.path.isfile(cache_path) and not rescan:
        try:
            data = json.load(open(cache_path))
            print(f"[extract] {os.path.basename(clip_dir)}: cached")
            return data
        except Exception as e:
            print(f"[extract] {os.path.basename(clip_dir)}: cache unreadable ({e}) — re-extracting")

    vid = _video_path(clip_dir)
    corners = _load_corners(vid)
    rallies = _near_rallies(clip_dir)

    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    geom = PointStartSystem(corners, ANALYSIS_SIZE[0], ANALYSIS_SIZE[1], fps=int(round(fps)))
    zones = _exclusion_zones(vid, ball_model, device)
    margin = int(round(MARGIN_SEC * fps))

    print(f"[extract] {os.path.basename(clip_dir)}: {len(rallies)} near rallies, "
          f"fps={fps:.2f}, {len(zones)} zones")

    out_rallies = []
    for ri, r in enumerate(rallies):
        start, end = int(r["start"]), int(r["end"])
        span_end = min(total - 1, end + margin)
        frames = {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        cached_bbox = None
        for f in range(start, span_end + 1):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
            if (f - start) % PLAYER_STRIDE == 0:
                cached_bbox = _detect_near_player(frame, player_model, device, geom)
            ball = _detect_ball(frame, ball_model, device, zones)
            frames[str(f)] = {"near_bbox": cached_bbox, "ball": ball}
        out_rallies.append({"start": start, "end": end, "span_end": span_end, "frames": frames})
        print(f"    rally {ri+1}/{len(rallies)}  [{start}-{end}]  span_end={span_end}")

    cap.release()
    data = {"fps": fps, "corners": corners.tolist(), "rallies": out_rallies}
    json.dump(data, open(cache_path, "w"))
    print(f"[extract] {os.path.basename(clip_dir)}: cached -> {TELEMETRY_CACHE}")
    return data


# ── Stage 2: replay + objective ───────────────────────────────────────────────

def replay_rally(corners, fps, params, rally) -> int:
    """Seed a point ACTIVE at GT start, replay cached telemetry, return the
    predicted end frame (or span_end if energy never depletes)."""
    system = PointStartSystem(corners, ANALYSIS_SIZE[0], ANALYSIS_SIZE[1],
                              fps=int(round(fps)), params=params, verbose=False)
    start, span_end = rally["start"], rally["span_end"]
    system._trigger_active(start, "gt")
    frames = rally["frames"]
    for f in range(start + 1, span_end + 1):
        tel = frames.get(str(f))
        nb = tuple(tel["near_bbox"]) if (tel and tel["near_bbox"]) else None
        bp = tuple(tel["ball"]) if (tel and tel["ball"]) else None
        state = system.process_frame(TrackData(frame_idx=f, near_player_bbox=nb, ball_pos=bp))
        if state != MatchState.ACTIVE:
            return system.points[-1]["end_frame"]
    return span_end


def rally_penalty(pred_end, rally, fps) -> float:
    """Asymmetric end-time error in seconds; early ends weighted EARLY_WEIGHT."""
    err = (pred_end - rally["end"]) / fps
    return EARLY_WEIGHT * (-err) if err < 0 else err


def clip_mean_penalty(params, clip_tel) -> float:
    corners = np.array(clip_tel["corners"], dtype=np.float32)
    fps = clip_tel["fps"]
    pens = [rally_penalty(replay_rally(corners, fps, params, r), r, fps)
            for r in clip_tel["rallies"]]
    return float(np.mean(pens)) if pens else 0.0


def make_objective(clip_tels):
    def objective(x):
        params = dict(zip(PARAM_NAMES, x))
        return float(np.mean([clip_mean_penalty(params, ct) for ct in clip_tels]))
    return objective


def _run_de(clip_tels, maxiter, seed=0):
    result = differential_evolution(
        make_objective(clip_tels), BOUNDS,
        maxiter=maxiter, popsize=15, tol=1e-3, mutation=(0.5, 1.0),
        recombination=0.7, seed=seed, polish=True, disp=True,
    )
    return dict(zip(PARAM_NAMES, result.x)), float(result.fun)


# ── Reports ───────────────────────────────────────────────────────────────────

def per_clip_report(params, clip_tels, clip_names):
    print("\n  per-clip mean penalty (sec):")
    rows = []
    for name, ct in zip(clip_names, clip_tels):
        errs = []
        for r in ct["rallies"]:
            pe = replay_rally(np.array(ct["corners"], np.float32), ct["fps"], params, r)
            errs.append((pe - r["end"]) / ct["fps"])
        mp = clip_mean_penalty(params, ct)
        early = sum(1 for e in errs if e < -0.25)
        rows.append((name, mp, np.mean(errs), early, len(errs)))
        print(f"    {name:>6}: penalty={mp:5.2f}  mean_err={np.mean(errs):+5.2f}s  "
              f"early={early}/{len(errs)}")
    return rows


def coverage_report(clip_dirs, clip_names):
    """Detection coverage within each GT rally span (start..end, excluding the
    replay margin) — catches clips where the near player or ball isn't being
    detected, which would silently skew the fit."""
    print("\n[coverage] detection coverage within GT rally spans:")
    for name, c in zip(clip_names, clip_dirs):
        path = os.path.join(c, TELEMETRY_CACHE)
        if not os.path.isfile(path):
            print(f"  {name:>6}: no telemetry cache — run --extract")
            continue
        data = json.load(open(path))
        ball_fracs, play_fracs, low = [], [], []
        for r in data["rallies"]:
            fr = r["frames"]
            keys = [str(f) for f in range(r["start"], r["end"] + 1)]
            n = len(keys)
            if n == 0:
                continue
            nb = sum(1 for k in keys if fr.get(k, {}).get("near_bbox"))
            bb = sum(1 for k in keys if fr.get(k, {}).get("ball"))
            play_fracs.append(nb / n)
            ball_fracs.append(bb / n)
            if bb / n < 0.20 or nb / n < 0.50:
                low.append((r["start"], r["end"], bb / n, nb / n))
        if not ball_fracs:
            print(f"  {name:>6}: no rally frames")
            continue
        print(f"  {name:>6}: ball={np.mean(ball_fracs):.0%}  player={np.mean(play_fracs):.0%}  "
              f"rallies={len(ball_fracs)}  low_coverage={len(low)}")
        for s, e, bf, pf in low:
            print(f"          [{s}-{e}] ball={bf:.0%} player={pf:.0%}")


def leave_one_clip_out(clip_tels, clip_names, maxiter):
    print("\n[LOCO] leave-one-clip-out generalization:")
    gaps = []
    for i, held in enumerate(clip_names):
        train = [ct for j, ct in enumerate(clip_tels) if j != i]
        params, train_score = _run_de(train, maxiter, seed=i)
        test_score = clip_mean_penalty(params, clip_tels[i])
        gaps.append((held, train_score, test_score))
        print(f"  hold {held:>6}: train={train_score:5.2f}  test={test_score:5.2f}")
    mean_test = float(np.mean([t for _, _, t in gaps]))
    print(f"[LOCO] mean held-out penalty: {mean_test:.3f}")
    return gaps


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Optimize energy-bar point-end parameters (near side)")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--ball_model", default="/Users/tennis/Documents/Code/Laptop/src/anya/pipeline/models/ball_best.pt")
    ap.add_argument("--clips", nargs="*", default=None, help="Explicit clip folder names (default: auto-discover)")
    ap.add_argument("--extract", action="store_true", help="Build/refresh telemetry caches")
    ap.add_argument("--optimize", action="store_true", help="Run the parameter search")
    ap.add_argument("--coverage", action="store_true", help="Print detection coverage from cached telemetry and exit")
    ap.add_argument("--rescan_telemetry", action="store_true")
    ap.add_argument("--loco", action="store_true", help="Also run leave-one-clip-out generalization")
    ap.add_argument("--maxiter", type=int, default=60, help="differential_evolution generations")
    ap.add_argument("--out", default="/Volumes/Anya/Data/energy_params.json")
    args = ap.parse_args()

    if not (args.extract or args.optimize):
        args.extract = args.optimize = True

    if args.clips:
        clip_dirs = [os.path.join(args.data_root, c) for c in args.clips]
    else:
        clip_dirs = discover_clips(args.data_root)
    clip_names = [os.path.basename(c) for c in clip_dirs]
    n_rallies = sum(len(_near_rallies(c)) for c in clip_dirs)
    print(f"[init] {len(clip_dirs)} clips, {n_rallies} near rallies: {clip_names}")

    if args.coverage:
        coverage_report(clip_dirs, clip_names)
        return

    if args.extract:
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[init] loading models on {device}")
        ball_model = YOLO(args.ball_model)
        player_model = YOLO("yolov8n.pt")
        for c in clip_dirs:
            try:
                build_telemetry(c, ball_model, player_model, device, rescan=args.rescan_telemetry)
            except Exception as e:
                print(f"[WARN] extraction failed for {os.path.basename(c)}: {e} — skipping")

    if not args.optimize:
        return

    # Only optimize over clips that actually have a telemetry cache (an
    # uncalibrated or failed clip is simply excluded rather than aborting).
    ready = [(c, n) for c, n in zip(clip_dirs, clip_names)
             if os.path.isfile(os.path.join(c, TELEMETRY_CACHE))]
    missing = [n for c, n in zip(clip_dirs, clip_names) if not os.path.isfile(os.path.join(c, TELEMETRY_CACHE))]
    if missing:
        print(f"[optimize] no telemetry for {missing} — excluded")
    if not ready:
        print("[optimize] no telemetry caches available — run --extract first")
        return
    clip_tels, clip_names = [], []
    for c, n in ready:
        try:
            clip_tels.append(json.load(open(os.path.join(c, TELEMETRY_CACHE))))
            clip_names.append(n)
        except Exception as e:
            print(f"[optimize] could not read telemetry for {n}: {e} — excluded")
    if not clip_tels:
        print("[optimize] no readable telemetry caches — aborting")
        return
    n_rallies = sum(len(ct["rallies"]) for ct in clip_tels)

    print(f"\n[optimize] {len(clip_names)} clips, {n_rallies} near rallies: {clip_names}")
    print(f"[optimize] differential_evolution over {len(PARAM_NAMES)} params, maxiter={args.maxiter}")
    best_params, best_score = _run_de(clip_tels, args.maxiter)

    print(f"\n[optimize] best objective = {best_score:.4f}")
    for k in PARAM_NAMES:
        print(f"    {k:<22} = {best_params[k]:.4f}")
    rows = per_clip_report(best_params, clip_tels, clip_names)

    loco = None
    if args.loco:
        loco = leave_one_clip_out(clip_tels, clip_names, args.maxiter)

    out = {
        "objective": best_score,
        "early_weight": EARLY_WEIGHT,
        "params": best_params,
        "fixed": {"ENERGY_START": 0.6, "ENERGY_DEAD": 0.02},
        "n_clips": len(clip_dirs),
        "n_near_rallies": n_rallies,
        "per_clip": {name: {"penalty": mp, "mean_err_sec": me, "early": e, "n": n}
                     for name, mp, me, e, n in rows},
        "loco": [{"held": h, "train": tr, "test": te} for h, tr, te in loco] if loco else None,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
