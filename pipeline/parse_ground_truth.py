"""
parse_ground_truth.py
=====================
Stage 0 of the dead/live classifier: resolve the ground-truth schema into one
in-memory representation, and derive the live/dead timeline plus the transition
lists the window sampler and the event evaluator both key off.

Resolved schema (inspected, not assumed — see README):
  <clip>/ground_truth.json   {"rallies": [{"start": int, "end": int,
                                           "serve": "near"|"far"}]}
      `start`/`end` are FRAME INDICES in the source video. A seconds-based
      variant (`start_s`/`end_s`) is also parsed if a clip ever ships one.

ONLY `ground_truth.json` COUNTS. Clip 68 carries a `derived_ground_truth.json`
instead, whose own `_derivation` block reports
`"bootstrap_matched_of_total": "10/45"` — i.e. the segmenter agreed with only 10
of its 45 rallies, and the rest are model output, not labels. Training or
scoring against it would be measuring one model against another. Clips with only
a derived file are excluded from discovery; pass --allow_derived to opt in.

Two things the rest of the pipeline gets wrong if it reads the raw file itself:

  1. Only LIVE intervals are labelled. Dead time is the *complement* of the
     rally list, so it also swallows pre-roll, post-roll and any unlabelled gap.
     `dead_segments()` returns those gaps explicitly rather than by inference.

  2. `optimize_energy._near_rallies` filters to serve=="near", which is right for
     *training* (the pose features describe the near player) but wrong for the
     *timeline*: a far-serve rally is still live. Across the 15 labelled clips
     only 103 of 232 rallies are near, so scoring against a near-only timeline
     would count every far rally as dead and inflate the false-fire rate. The
     timeline built here uses ALL rallies; `serve` is kept per segment so callers
     can subset deliberately.

Usage:
    python pipeline/parse_ground_truth.py                 # schema + per-clip report
    python pipeline/parse_ground_truth.py --clips 22 58
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

import numpy as np

try:                                        # package import (python -m pipeline.x)
    from .videoio import open_video
except ImportError:                         # script import (python pipeline/x.py)
    from videoio import open_video

DATA_ROOT = "/Volumes/Anya/Data"
GT_NAME = "ground_truth.json"
GT_DERIVED = "derived_ground_truth.json"
# Flipped on only by --allow_derived. Kept module-level so every consumer
# (window builders, evaluator, extractors) inherits the same decision.
ALLOW_DERIVED = False

# Clips whose labels cannot be trusted. Excluded at the gt_path() choke point so
# they cannot re-enter through any consumer's own discovery — several of them
# scan for cache files, not for ground truth, and a stale cache left on disk
# would otherwise be picked up silently.
EXCLUDED = {
    # 35 was excluded at 7.0% live and was RELABELLED on 2026-08-24: it now
    # carries 20 rallies over 0-406s of a 420s clip, 34.8% live, with a median
    # inter-rally gap of 14s. That is denser than clip 38 (17.6%), which was
    # never excluded, so the reason no longer holds and it rejoins the corpus.
    "37": "incompletely labelled (4.5% marked live)",
    "63": "incompletely labelled (1.4% marked live)",
    "68": "no ground_truth.json — derived labels only, 10/45 bootstrap agreement",
}
ALLOW_EXCLUDED = False


def _fps_for(clip_dir: str) -> float:
    """fps from the telemetry cache if present, else from the video, else 30."""
    tel = os.path.join(clip_dir, "energy_telemetry_cache.json")
    if os.path.isfile(tel):
        try:
            return float(json.load(open(tel))["fps"])
        except Exception:
            pass
    for name in ("snippet.mp4", "match.mp4"):
        p = os.path.join(clip_dir, name)
        if os.path.isfile(p):
            try:
                import cv2
                cap = open_video(p, "GT")
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps and fps > 1:
                    return float(fps)
            except Exception:
                pass
    return 30.0


def _n_frames(clip_dir: str) -> Optional[int]:
    for name in ("snippet.mp4", "match.mp4"):
        p = os.path.join(clip_dir, name)
        if os.path.isfile(p):
            try:
                import cv2
                cap = open_video(p, "GT")
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if n > 0:
                    return n
            except Exception:
                pass
    return None


def gt_path(clip_dir: str) -> Optional[str]:
    """Path to the clip's usable ground truth, else None.

    Single choke point for label trust: clips in EXCLUDED (incompletely labelled,
    or labelled by a model rather than by hand) return None, so no consumer can
    train or score against them by accident.
    """
    if not ALLOW_EXCLUDED and os.path.basename(clip_dir.rstrip("/")) in EXCLUDED:
        return None
    p = os.path.join(clip_dir, GT_NAME)
    if os.path.isfile(p):
        return p
    if ALLOW_DERIVED:
        p = os.path.join(clip_dir, GT_DERIVED)
        if os.path.isfile(p):
            return p
    return None


def load_rallies(clip_dir: str, fps: Optional[float] = None) -> List[Dict]:
    """Rallies as frame-indexed live segments, whichever schema variant is on disk.

    Returns [{start, end, start_s, end_s, serve, schema}] sorted by start.
    """
    p = gt_path(clip_dir)
    if p is None:
        return []
    fps = fps or _fps_for(clip_dir)
    raw = json.load(open(p)).get("rallies", [])
    out = []
    for r in raw:
        if "start" in r and "end" in r:                 # frame-indexed
            s, e = int(r["start"]), int(r["end"])
            schema = "frames"
        elif "start_s" in r and "end_s" in r:           # seconds
            s, e = int(round(r["start_s"] * fps)), int(round(r["end_s"] * fps))
            schema = "seconds"
        else:
            continue
        if e <= s:
            continue
        out.append({"start": s, "end": e, "start_s": s / fps, "end_s": e / fps,
                    "serve": r.get("serve"), "schema": schema})
    return sorted(out, key=lambda r: r["start"])


def live_mask(rallies: List[Dict], n_frames: int) -> np.ndarray:
    """Boolean per-frame timeline: True = live (point in play)."""
    m = np.zeros(int(n_frames), dtype=bool)
    for r in rallies:
        a, b = max(0, r["start"]), min(int(n_frames) - 1, r["end"])
        if b >= a:
            m[a:b + 1] = True
    return m


def dead_segments(rallies: List[Dict], n_frames: int) -> List[Dict]:
    """Complement of the rally list — the implicit dead intervals, made explicit.

    Includes the head (before the first rally) and tail (after the last), which
    are dead but are typically pre/post-roll rather than genuine between-point
    time; callers that care can drop them via the `edge` flag.
    """
    segs, cursor = [], 0
    for r in rallies:
        if r["start"] > cursor:
            segs.append({"start": cursor, "end": r["start"] - 1, "edge": cursor == 0})
        cursor = max(cursor, r["end"] + 1)
    if n_frames and cursor <= n_frames - 1:
        segs.append({"start": cursor, "end": int(n_frames) - 1, "edge": True})
    return segs


def transitions(rallies: List[Dict]) -> Dict[str, List[Dict]]:
    """Point-start (dead->live) and point-end (live->dead) frames.

    A rally end is only a real live->dead transition if the next rally does not
    start immediately; back-to-back rallies (no dead gap) are reported under
    `merged` rather than silently emitting a zero-length dead state.
    """
    starts, ends, merged = [], [], []
    for i, r in enumerate(rallies):
        nxt = rallies[i + 1] if i + 1 < len(rallies) else None
        if i == 0 or rallies[i - 1]["end"] + 1 < r["start"]:
            starts.append({"frame": r["start"], "serve": r["serve"], "rally": i})
        if nxt is not None and nxt["start"] <= r["end"] + 1:
            merged.append({"frame": r["end"], "rally": i})
        else:
            ends.append({"frame": r["end"], "serve": r["serve"], "rally": i})
    return {"point_start": starts, "point_end": ends, "merged": merged}


def parse_clip(clip_dir: str) -> Optional[Dict]:
    r = load_rallies(clip_dir)
    if not r:
        return None
    fps = _fps_for(clip_dir)
    n = _n_frames(clip_dir) or (r[-1]["end"] + 1)
    tr = transitions(r)
    dead = dead_segments(r, n)
    live_f = int(sum(x["end"] - x["start"] + 1 for x in r))
    return {
        "clip": os.path.basename(clip_dir), "fps": fps, "n_frames": n,
        "schema": r[0]["schema"], "path": os.path.basename(gt_path(clip_dir)),
        "rallies": r, "dead": dead, "transitions": tr,
        "n_near": sum(1 for x in r if x["serve"] == "near"),
        "live_frames": live_f, "live_frac": live_f / n if n else 0.0,
    }


def discover(data_root: str) -> List[str]:
    return [os.path.join(data_root, d) for d in sorted(os.listdir(data_root))
            if os.path.isdir(os.path.join(data_root, d))
            and gt_path(os.path.join(data_root, d))]


def main():
    ap = argparse.ArgumentParser(description="Parse ground truth into live/dead timelines")
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--json_out", default=None, help="write parsed timelines to this path")
    ap.add_argument("--allow_derived", action="store_true",
                    help="also accept derived_ground_truth.json (model-inferred labels)")
    args = ap.parse_args()

    global ALLOW_DERIVED
    ALLOW_DERIVED = args.allow_derived

    dirs = ([os.path.join(args.data_root, c) for c in args.clips]
            if args.clips else discover(args.data_root))

    print("=== resolved ground-truth schema ===")
    sample = next((parse_clip(d) for d in dirs if parse_clip(d)), None)
    if sample is None:
        raise SystemExit("no ground truth found")
    print(f"file            : <clip>/{sample['path']}")
    print(f"top-level keys  : ['rallies'] (+ '_derivation' on the seconds variant)")
    print(f"rally record    : {json.dumps(sample['rallies'][0], default=str)}")
    print(f"units           : {sample['schema']}  (seconds variant converted via fps)")
    print(f"labelled state  : LIVE only — dead time is the complement\n")

    print(f"{'clip':>5} {'fps':>6} {'frames':>7} {'rallies':>8} {'near':>5} "
          f"{'starts':>7} {'ends':>5} {'merged':>7} {'dead segs':>10} {'live%':>6}")
    rows, tot = [], {"r": 0, "near": 0, "s": 0, "e": 0, "m": 0}
    for d in dirs:
        p = parse_clip(d)
        if not p:
            continue
        t = p["transitions"]
        print(f"{p['clip']:>5} {p['fps']:>6.2f} {p['n_frames']:>7} {len(p['rallies']):>8} "
              f"{p['n_near']:>5} {len(t['point_start']):>7} {len(t['point_end']):>5} "
              f"{len(t['merged']):>7} {len(p['dead']):>10} {p['live_frac']*100:>5.1f}%")
        tot["r"] += len(p["rallies"]); tot["near"] += p["n_near"]
        tot["s"] += len(t["point_start"]); tot["e"] += len(t["point_end"])
        tot["m"] += len(t["merged"])
        rows.append(p)

    print(f"\n[total] {len(rows)} clips  {tot['r']} rallies ({tot['near']} near / "
          f"{tot['r']-tot['near']} far)  {tot['s']} point-starts  {tot['e']} point-ends"
          + (f"  {tot['m']} merged (no dead gap)" if tot["m"] else ""))
    print("[note]  timeline uses ALL rallies; far-serve rallies are live too.")

    if args.json_out:
        json.dump({p["clip"]: p for p in rows}, open(args.json_out, "w"), indent=1, default=str)
        print(f"[done] -> {args.json_out}")


if __name__ == "__main__":
    main()
