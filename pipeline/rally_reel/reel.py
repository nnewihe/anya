"""End-to-end: video in, highlight reel of the active rallies out.

Stage order (every stage caches next to the video, so re-running after a
config change costs seconds rather than another perception pass):

    0  court corners      utilities.init_court          <stem>_court_cache.json
    1  telemetry          anya_telemetry                <stem>_anya_telemetry.jsonl
    2  far-player pose    anya_far_telemetry            <stem>_anya_far_telemetry.jsonl
                          (or extract_far_pose)         <stem>_far_pose.jsonl
    3  far serve starts   anya_far_serve                (in memory)
    4  near serve starts  anya_near_telemetry           <stem>_anya_near_telemetry.jsonl
                          + anya_near_serve             <stem>_near_serve_events.json
    5  walking / deadtime walking.predict               <stem>_walk_pose.npz
    6  segments           rally_reel.points             <stem>_rally_segments.json
    7  cut + concat       utilities.create_highlights   <stem>_rally_reel.mp4

Stage 0 is the only interactive step: the user clicks the four court corners
once (bottom-left, bottom-right, top-right, top-left).  Both `pipeline/` and
`walking/` read that same cache, so one calibration serves the whole run.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ in (None, ""):               # allow direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pipeline.rally_reel"

from ..anya_telemetry import extract_anya_telemetry, telemetry_path_for
from ..anya_far_serve import detect_far_serves
from ..extract_far_pose import extract_far_pose
from ..anya_near_serve import score_telemetry, NearServeConfig
from ..anya_near_telemetry import extract_near_telemetry
from ..anya_far_telemetry import extract_far_telemetry
from ..utilities import Config, init_court, create_highlights_ffmpeg, probe_video

from .config import ReelConfig
from .points import (build_segments, enforce_service_runs,
                     merge_serve_starts, walk_onsets)

ANALYSIS_SIZE = (960, 540)
SEGMENTS_SUFFIX = "_rally_segments.json"
REEL_SUFFIX = "_rally_reel.mp4"

N_STAGES = 7
STAGE_LABELS = {
    0: "court calibration",
    1: "telemetry",
    2: "far-player pose",    # includes its own perception pass under fast_far
    3: "far serve starts",
    4: "near serve starts",   # includes its own perception pass under fast_near
    5: "walking (dead time)",
    6: "segment assembly",
    7: "cutting reel",
}


def _emit(on_progress, stage: int, frac: Optional[float] = None):
    """Reports (stage, n_stages, label, frac) to a UI, if one is listening.

    `frac` is the fraction complete *within* the stage, or None where the
    stage cannot report sub-progress (a model call that returns only when
    done).  A caller should treat None as "indeterminate", not as zero.
    Never lets a display error break the pipeline.
    """
    if on_progress is None:
        return
    try:
        on_progress(stage, N_STAGES, STAGE_LABELS.get(stage, ""), frac)
    except Exception:
        pass


def _stem_path(video_path: str, suffix: str) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{suffix}")


def _walk_intervals(video_path: str, device: str = "mps") -> List[Dict]:
    """Walking intervals, as the dead-time proxy for point ends.

    Imported lazily: `walking` is a sibling top-level package with its own
    heavier deps (joblib, sklearn), and a caller that only wants segments
    from a cached JSON should not pay for them.
    """
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import numpy as np
    from walking.predict import predict_video
    from walking.evaluate import to_intervals

    res = predict_video(video_path, device=device)
    fps, prob, mask = res["fps"], res["prob"], res["is_walking"]
    valid = res["sig"]["valid"]
    return [{
        "start_second": a / fps,
        "end_second": (b + 1) / fps,
        "duration_s": (b - a + 1) / fps,
        "mean_prob": float(np.mean(prob[a:b + 1])),
        "detection_coverage": float(np.mean(valid[a:b + 1])),
    } for a, b in to_intervals(mask)]


def _near_blind_mask(records: List[Dict], fps: float,
                     cfg: ReelConfig) -> List[bool]:
    """Per-record: is the walking classifier uninformative right now?

    True when the near player has no fix at all for near_untracked_s, or is
    tracked but has stayed inside near_stationary_ft for near_stationary_s.
    Either way the walking signal carries no evidence about whether the
    point is still live, which is where ball-quiet is allowed to speak.
    """
    n = len(records)
    xs = [r.get("npw") for r in records]
    w_untr = max(1, int(cfg.near_untracked_s * fps))
    w_stat = max(1, int(cfg.near_stationary_s * fps))

    blind = [False] * n
    for i in range(n):
        lo = max(0, i - w_untr)
        if not any(xs[j] for j in range(lo, i + 1)):
            blind[i] = True
            continue
        lo = max(0, i - w_stat)
        seen = [xs[j] for j in range(lo, i + 1) if xs[j]]
        # Needs a real majority of the window tracked, otherwise "stationary"
        # is really just absence wearing a different hat.
        if len(seen) >= 0.5 * (i - lo + 1) and len(seen) >= 2:
            dx = max(p[0] for p in seen) - min(p[0] for p in seen)
            dy = max(p[1] for p in seen) - min(p[1] for p in seen)
            if dx <= cfg.near_stationary_ft and dy <= cfg.near_stationary_ft:
                blind[i] = True
    return blind


def _ball_quiet_onsets(telemetry_path: str, cfg: ReelConfig) -> List[tuple]:
    """Moments where the filtered ball track has gone silent for ball_quiet_s.

    Reuses anya_far_serve's ball filtering — confidence floor, rescaled
    exclusion zones, self-calibrated static blobs — so this sees the same
    cleaned ball stream the serve detector does rather than raw `all_balls`.

    In "gated" mode an onset only survives if the near player was untracked
    or stationary at that moment: a silent ball while the walking classifier
    is watching a visible, moving player is not evidence the point ended.
    """
    from ..anya_far_serve import (FarServeDetector, FarServeDetectorConfig,
                                  calibrate_static_blobs, load_telemetry,
                                  scale_exclusion_zones)

    meta, records = load_telemetry(telemetry_path)
    fcfg = FarServeDetectorConfig()
    det = FarServeDetector(fcfg)
    det.set_exclusion_zones(scale_exclusion_zones(meta.get("exclusion_zones", []), meta))
    det.set_static_cells(calibrate_static_blobs(records, fcfg))

    gated = cfg.ball_quiet_mode == "gated"
    blind = _near_blind_mask(records, meta.get("fps", 30.0), cfg) if gated else None

    onsets: List[tuple] = []
    last_ball: Optional[float] = None
    pending = False
    n_raw = 0
    for i, r in enumerate(records):
        t = r["t"]
        if det._filter_balls(r.get("all_balls", [])):
            last_ball, pending = t, False
        elif last_ball is not None and not pending and t - last_ball >= cfg.ball_quiet_s:
            pending = True     # one onset per silence, not one per frame
            n_raw += 1
            if blind is not None and not blind[i]:
                continue       # near player visible and moving — not our call
            onsets.append((last_ball + cfg.ball_quiet_s, "ball-quiet"))
    if gated:
        print(f"[REEL]   ball-quiet: {len(onsets)}/{n_raw} onset(s) passed the "
              f"near-blind gate")
    return onsets


def build_reel(video_path: str,
               cfg: Optional[ReelConfig] = None,
               output_path: Optional[str] = None,
               force_telemetry: bool = False,
               device: str = "mps",
               dry_run: bool = False,
               on_progress=None) -> Tuple[List, Optional[str]]:
    """Runs every stage. Returns (segments, output_path or None).

    `on_progress(stage, n_stages, label, frac)` is called at each stage
    boundary and periodically inside the long ones; see `_emit`.
    """
    cfg = cfg or ReelConfig()
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    output_path = output_path or _stem_path(video_path, REEL_SUFFIX)

    info = probe_video(video_path)
    duration = info["duration_sec"]

    # ── Stage 0: court corners (interactive once, then cached) ──────────
    # Up front, before any model loads, so the click prompt cannot appear
    # to hang behind a slow import.
    print("[REEL] Stage 0/7  court calibration")
    _emit(on_progress, 0)
    init_court(video_path, analysis_size=ANALYSIS_SIZE)

    # Which stages still need the shared full-resolution pass?  Ball-quiet dead
    # time reads it, and so do far serves and near serves whenever their fast
    # paths are turned off.  When nothing needs it, stages 1-2 are skipped
    # outright — that is where the speedup actually lands.
    needs_full = (cfg.ball_quiet_mode != "off"
                  or (cfg.use_far and not cfg.fast_far)
                  or not cfg.fast_near)

    # ── Stage 1: telemetry ──────────────────────────────────────────────
    telemetry = None
    if needs_full:
        print("[REEL] Stage 1/7  telemetry")
        _emit(on_progress, 1, 0.0)
        telemetry = extract_anya_telemetry(
            video_path, force=force_telemetry,
            progress_cb=lambda cur, tot: _emit(on_progress, 1, cur / max(1, tot)))
    else:
        print("[REEL] Stage 1/7  telemetry — skipped (nothing needs the full "
              "pass: fast_near, fast_far, ball_quiet_mode=off)")
        _emit(on_progress, 1, 1.0)

    # ── Stage 2: far-player pose ────────────────────────────────────────
    # `far_source` is whichever telemetry stage 3 should read.  Under fast_far
    # that is the far-only extractor's own JSONL, which writes its pose cache
    # at the path far_pose_path_for derives — so detect_far_serves finds it
    # unmodified, and picks its threshold preset from meta.source.
    far_source = telemetry
    if cfg.use_far:
        if cfg.fast_far:
            print("[REEL] Stage 2/7  far telemetry + pose (fast path)")
            # Indeterminate to start: the first run opens with a one-time band
            # proxy transcode that reports no progress, and a bar parked at 0%
            # for a minute reads as a hang.
            _emit(on_progress, 2)
            far_source = extract_far_telemetry(
                video_path, force=force_telemetry,
                progress_cb=lambda cur, tot: _emit(on_progress, 2,
                                                   cur / max(1, tot)))
        else:
            print("[REEL] Stage 2/7  far-player pose")
            _emit(on_progress, 2, 0.0)
            extract_far_pose(
                telemetry,
                progress_cb=lambda cur, tot: _emit(on_progress, 2, cur / max(1, tot)))
    else:
        print("[REEL] Stage 2/7  far-player pose — skipped (use_far=False)")
        _emit(on_progress, 2, 1.0)

    # ── Stage 3: far serves ─────────────────────────────────────────────
    far_serves = []
    if cfg.use_far:
        print("[REEL] Stage 3/7  far serve starts")
        _emit(on_progress, 3)
        far_serves = detect_far_serves(far_source)
    print(f"[REEL]   far serves: {len(far_serves)}")

    # ── Stage 4: near serves ────────────────────────────────────────────
    near_events = []
    if cfg.use_near:
        if cfg.fast_near:
            # Its own cheap perception pass (540p proxy, 5 fps player,
            # upscaled toss-ROI ball) plus the tuned low-rate scoring profile.
            print("[REEL] Stage 4/7  near serve starts (fast path)")
            # Indeterminate rather than 0%: on a video's first run this stage
            # opens with a one-time proxy transcode that reports no progress,
            # and a bar parked at 0% for a minute reads as a hang.
            _emit(on_progress, 4)
            near_telemetry = extract_near_telemetry(
                video_path,
                progress_cb=lambda cur, tot: _emit(on_progress, 4,
                                                   cur / max(1, tot)))
            near_cfg = NearServeConfig.for_low_rate(threshold=cfg.near_threshold)
            source = near_telemetry
        else:
            print("[REEL] Stage 4/7  near serve starts")
            _emit(on_progress, 4)
            near_cfg = NearServeConfig()
            near_cfg.threshold = cfg.near_threshold
            source = telemetry
        _, events_path = score_telemetry(source, near_cfg)
        with open(events_path) as fh:
            payload = json.load(fh)
        near_events = payload.get("events", payload if isinstance(payload, list) else [])
    print(f"[REEL]   near serves: {len(near_events)} "
          f"(p >= {cfg.near_threshold})")

    # ── Stage 5: walking / dead time ────────────────────────────────────
    print("[REEL] Stage 5/7  walking (dead-time proxy)")
    _emit(on_progress, 5)   # predict_video returns only when done
    walks = _walk_intervals(video_path, device=device)
    dead_onsets = walk_onsets(walks, cfg)
    n_walk = len(dead_onsets)
    print(f"[REEL]   walk intervals: {len(walks)} -> {n_walk} usable onset(s)")
    if cfg.ball_quiet_mode != "off":
        dead_onsets += _ball_quiet_onsets(telemetry, cfg)
        print(f"[REEL]   dead-time onsets: {n_walk} walk + "
              f"{len(dead_onsets) - n_walk} ball-quiet "
              f"(mode={cfg.ball_quiet_mode})")

    # ── Stage 6: assemble segments ──────────────────────────────────────
    print("[REEL] Stage 6/7  segment assembly")
    _emit(on_progress, 6)
    starts = merge_serve_starts(far_serves, near_events, cfg)

    if cfg.enforce_service_runs:
        starts = enforce_service_runs(starts, cfg)
        conflicts = [p for p in starts if p.side_conflict]
        if conflicts:
            verb = "dropped" if cfg.drop_side_conflicts else "relabelled"
            print(f"[REEL]   service runs (min {cfg.min_service_run}): "
                  f"{len(conflicts)} start(s) {verb} — " +
                  ", ".join(f"{p.t:.1f}s {p.detected_side}->{p.side}"
                            for p in conflicts))
            if cfg.drop_side_conflicts:
                starts = [p for p in starts if not p.side_conflict]

    segments = build_segments(starts, dead_onsets, duration, cfg)

    seg_path = _stem_path(video_path, SEGMENTS_SUFFIX)
    with open(seg_path, "w") as fh:
        json.dump({
            "video": os.path.basename(video_path),
            "duration_s": duration,
            "config": cfg.__dict__,
            "n_serve_starts": len(starts),
            "segments": [s.as_dict() for s in segments],
        }, fh, indent=2)

    kept = sum(s.end - s.start for s in segments)
    print(f"[REEL]   {len(starts)} serve start(s) -> {len(segments)} segment(s), "
          f"{kept:.1f}s of {duration:.1f}s kept ({kept / duration:.0%})")
    print(f"[REEL]   segments -> {seg_path}")

    # ── Stage 7: cut ────────────────────────────────────────────────────
    if dry_run:
        print("[REEL] Stage 7/7  skipped (--dry-run)")
        return segments, None

    print("[REEL] Stage 7/7  cutting reel")
    _emit(on_progress, 7)
    create_highlights_ffmpeg(
        video_path, [(s.start, s.end) for s in segments], output_path,
        pre_roll=0.0, merge_gap_sec=cfg.merge_gap_s,
    )
    return segments, output_path
