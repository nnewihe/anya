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
    5  walking / deadtime anya_end_telemetry            <stem>_anya_end_telemetry.jsonl
                          + walking.predict             <stem>_end_walk_pose.npz
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
from ..anya_end_telemetry import (EndExtractorConfig, extract_end_telemetry,
                                  end_dets_path_for, end_pose_path_for,
                                  end_telemetry_path_for)
from ..utilities import Config, init_court, create_highlights_ffmpeg, probe_video

from .config import ReelConfig
from .points import (build_segments, enforce_service_runs,
                     merge_serve_starts, usable_walk_intervals, walk_onsets)
from .deadtime_confidence import deadtime_onsets, score_deadtime
from . import energy as energy_policy
from . import ball_trace

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
    5: "walking (dead time)",  # includes its own perception pass under fast_end
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
    from ..workdir import artifact_dir
    d = artifact_dir(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{suffix}")


def _min_point_s(cfg: ReelConfig):
    """Per-policy floor on point length, or None for the shared cfg.point_min_s.

    The trace-based policies want a 4 s floor while walk-ball and legacy keep
    3 s; raising the shared value instead would confound any A/B between them.
    """
    if cfg.end_policy == "trace":
        return cfg.trace_point_min_s
    if cfg.end_policy == "energy":
        return cfg.energy_point_min_s
    return None


def _walk_model_path(cfg: ReelConfig) -> Optional[str]:
    """The walking model matching the pose rate this run will feed it.

    The shipped model was trained on 30 Hz window statistics; the fast path
    extracts at 15 Hz, and applying the 30 Hz model to it costs a real point
    end (Data/21 walk onsets 8/12 -> 7/12, median error -0.32s -> -1.13s).
    Retrained at 15 Hz it is back to 8/12 with FEWER false onsets, 18 against
    20 — even though its frame F1 against the baseline mask is LOWER (0.890 vs
    0.938).  Frame F1 is the proxy here; onsets are what stage 6 consumes.
    """
    if not cfg.fast_end or not cfg.walk_model_15hz:
        return None
    repo_root = Path(__file__).resolve().parents[2]
    p = repo_root / "walking" / "outputs" / "walking_model_15hz.joblib"
    return str(p) if p.is_file() else None


def _walk_intervals(video_path: str, device: str = "mps",
                    dets_npz: Optional[str] = None,
                    pose_npz: Optional[str] = None,
                    model_path: Optional[str] = None,
                    return_result: bool = False):
    """Walking intervals, as the dead-time proxy for point ends.

    `dets_npz`/`pose_npz` point the classifier at a pose pass someone else
    already ran — under fast_end that is anya_end_telemetry's decimated pass,
    which the near-player selector and the feature builder read unchanged
    because both take fps as a parameter.  With neither, walking.predict
    extracts its own full-rate pass as before.

    Imported lazily: `walking` is a sibling top-level package with its own
    heavier deps (joblib, sklearn), and a caller that only wants segments
    from a cached JSON should not pay for them.
    """
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import numpy as np
    from walking.predict import predict_video, MODEL_PATH
    from walking.evaluate import to_intervals

    if dets_npz is not None and pose_npz is not None and not os.path.isfile(pose_npz):
        from walking.select_near import select
        select(video_path, dets_npz=dets_npz, out=pose_npz)

    res = predict_video(video_path, device=device, pose_npz=pose_npz,
                        model_path=model_path or MODEL_PATH)
    fps, prob, mask = res["fps"], res["prob"], res["is_walking"]
    valid = res["sig"]["valid"]
    intervals = [{
        "start_second": a / fps,
        "end_second": (b + 1) / fps,
        "duration_s": (b - a + 1) / fps,
        "mean_prob": float(np.mean(prob[a:b + 1])),
        "detection_coverage": float(np.mean(valid[a:b + 1])),
    } for a, b in to_intervals(mask)]
    return (intervals, res) if return_result else intervals


def _near_end_signals(video_path: str, walk_result: Dict, cfg: ReelConfig):
    """The four near_end signals, or None when every weight on them is zero.

    They are pure arithmetic over the pose track the walking classifier just
    read, so this costs no perception and a fraction of a second.  Three of the
    four ship non-zero, so on a default run this always computes; the guard is
    for an ablation arm (`--energy-turn-away-weight 0 ...`), where computing
    evidence no term reads would be pure overhead.

    A failure here is NOT fatal.  These signals are additive evidence on a
    policy that shipped without them, so a bad pose cache should cost the reel
    the extra terms, not the reel.
    """
    from ..near_end import SIGNAL_NAMES, signals_for_video

    if not any(getattr(cfg, f"energy_{n}_weight", 0.0) for n in SIGNAL_NAMES):
        return None
    try:
        sigs = signals_for_video(video_path, pose_npz=end_pose_path_for(video_path)
                                 if cfg.fast_end else None,
                                 sig=walk_result.get("sig"))
    except Exception as e:
        print(f"[REEL]   WARN: near-end signals unavailable ({e}); the energy "
              f"bar falls back to walking alone")
        return None
    print("[REEL]   near-end signals: " + ", ".join(
        f"{n} {float(sigs[n].mean()):.2f}" for n in SIGNAL_NAMES))
    return sigs


def _near_blind_mask(records: List[Dict], fps: float,
                     cfg: ReelConfig) -> List[bool]:
    """Per-record: is the walking classifier uninformative right now?

    True when the near player has no fix at all for near_untracked_s, or is
    tracked but has stayed inside near_stationary_ft for near_stationary_s.
    Either way the walking signal carries no evidence about whether the
    point is still live, which is where ball-quiet is allowed to speak.

    Windows are in SECONDS over record timestamps, not in record counts.  The
    full telemetry has one record per frame so the two agreed; a fast-path
    file has records only where a model actually ran, and counting rows there
    would silently shrink every window by the sampling factor.

    `pn` marks a record that carries a player observation at all.  Records
    without it (the full pass, which looks every frame) count as observations,
    so this is unchanged against that file.
    """
    n = len(records)
    ts = [float(r.get("t", 0.0)) for r in records]
    xs = [r.get("npw") for r in records]
    looked = [bool(r.get("pn", True)) for r in records]

    blind = [False] * n
    lo_untr = lo_stat = 0
    for i in range(n):
        t = ts[i]
        while ts[lo_untr] < t - cfg.near_untracked_s:
            lo_untr += 1
        if not any(xs[j] for j in range(lo_untr, i + 1)):
            blind[i] = True
            continue
        while ts[lo_stat] < t - cfg.near_stationary_s:
            lo_stat += 1
        seen = [xs[j] for j in range(lo_stat, i + 1) if xs[j]]
        n_looked = sum(1 for j in range(lo_stat, i + 1) if looked[j])
        # Needs a real majority of the LOOKS to have found the player,
        # otherwise "stationary" is really just absence wearing a different hat.
        if len(seen) >= 0.5 * max(1, n_looked) and len(seen) >= 2:
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
    meta, records, filter_balls = _ball_stream(telemetry_path)
    def is_ball(r):
        return bool(filter_balls(r))

    gated = cfg.ball_quiet_mode == "gated"
    blind = _near_blind_mask(records, meta.get("fps", 30.0), cfg) if gated else None

    onsets: List[tuple] = []
    last_ball: Optional[float] = None
    pending = False
    n_raw = 0
    n_thin = 0
    # How many looks at the ball the window must contain before its silence
    # means anything.  Per-frame ball recall is clip-dependent (7% / 36% / 92%
    # across the corpus), so a window that was only sampled twice is silent by
    # sampling noise, not by evidence — and a false quiet ends the point early,
    # which is the one point-end error that loses tennis.  At the full pass's
    # 30 fps a 1.5 s window holds ~45 looks and this never binds; at 10 fps it
    # holds ~15 and still clears it.
    looks: List[float] = []       # timestamps of records that sampled the ball
    lo = 0
    for i, r in enumerate(records):
        t = r["t"]
        if r.get("bn", True):
            looks.append(t)
        if is_ball(r):
            last_ball, pending = t, False
        elif last_ball is not None and not pending and t - last_ball >= cfg.ball_quiet_s:
            while lo < len(looks) and looks[lo] < t - cfg.ball_quiet_s:
                lo += 1
            if len(looks) - lo < cfg.ball_quiet_min_looks:
                n_thin += 1
                continue       # too few looks to call this silence
            pending = True     # one onset per silence, not one per frame
            n_raw += 1
            if blind is not None and not blind[i]:
                continue       # near player visible and moving — not our call
            onsets.append((last_ball + cfg.ball_quiet_s, "ball-quiet"))
    if n_thin:
        print(f"[REEL]   ball-quiet: {n_thin} silence(s) dropped for fewer than "
              f"{cfg.ball_quiet_min_looks} looks in the window")
    if gated:
        print(f"[REEL]   ball-quiet: {len(onsets)}/{n_raw} onset(s) passed the "
              f"near-blind gate")
    return onsets


def _ball_stream(telemetry_path: str):
    """(meta, records, filter_balls) with the ball filtering ball-quiet uses.

    Factored out of `_ball_quiet_onsets` so every policy sees an identical
    ball stream — confidence floor, rescaled exclusion zones, self-calibrated
    static blobs — and a difference between them is never a difference in what
    counts as a ball.

    `filter_balls(record)` returns the surviving `(cx, cy, conf)` detections, so
    a caller that needs positions (the trace policy feeds them to the tracker)
    and a caller that only needs presence share one definition.  Use
    `bool(filter_balls(r))` for the presence question.

    The static-blob thresholds are DETECTION COUNTS, so they mean something
    different against a decimated stream and `ball_sampling_scales` is what
    makes them comparable.  Omitting it — as this did until 2026-08-19 — left a
    10 fps stream judged against full-rate thresholds, so a persistent false
    positive needed 3x its share of looks before it read as static and a
    scoreboard glint could keep a point alive.  The error scales with the
    sampling rate, so it also moves when ball_fps does.
    """
    from ..anya_far_serve import (FarServeDetector, FarServeDetectorConfig,
                                  ball_sampling_scales, calibrate_static_blobs,
                                  load_telemetry, scale_exclusion_zones)

    meta, records = load_telemetry(telemetry_path)
    fcfg = FarServeDetectorConfig()
    det = FarServeDetector(fcfg)
    det.set_exclusion_zones(scale_exclusion_zones(meta.get("exclusion_zones", []), meta))
    hits_scale, rate_scale = ball_sampling_scales(meta)
    det.set_static_cells(calibrate_static_blobs(records, fcfg, hits_scale, rate_scale))
    return meta, records, lambda r: det._filter_balls(r.get("all_balls", []))


def _walk_ball_onsets(telemetry_path: str, walks: List[Dict],
                      cfg: ReelConfig) -> List[tuple]:
    """Point ends under end_policy="walk-ball".  Returns [(t, source)].

    Two rules, evaluated per telemetry record, and which one owns a moment is
    decided by whether the walking classifier is saying anything there:

      A  "walk"        the near player is inside a usable walk interval AND no
                       ball has been seen for walk_ball_veto_s.  A ball seen
                       while the player walks vetoes the end and the point runs
                       on — mid-rally walking is common and is exactly what the
                       old union-of-onsets scheme could not survive.
      B  "ball-quiet"  no walk interval covers this moment and the ball has been
                       silent for no_walk_quiet_s.

    Rule A stamps the end at max(walk start, last ball) rather than at the
    moment the veto clears: the rally stopped when the ball did or when the
    walk began, and the veto window is confirmation, not duration.  Rule B
    stamps at last_ball + no_walk_stamp_s for the same reason.

    Both rules keep the `ball_quiet_min_looks` floor from the legacy path.  A
    window that was barely sampled is silent from sampling noise rather than
    from evidence, and per-frame ball recall runs 7%-92% across the corpus.
    """
    meta, records, filter_balls = _ball_stream(telemetry_path)
    def is_ball(r):
        return bool(filter_balls(r))

    usable = usable_walk_intervals(walks, cfg)
    spans = sorted((float(w["start_second"]), float(w["end_second"]))
                   for w in usable)

    onsets: List[tuple] = []
    last_ball: Optional[float] = None
    pending_a = pending_b = False
    n_veto = n_thin = 0
    looks: List[float] = []
    lo_look = 0
    si = 0                      # walk span cursor; records are time-ordered

    def enough_looks(t: float, window: float) -> bool:
        nonlocal lo_look
        while lo_look < len(looks) and looks[lo_look] < t - window:
            lo_look += 1
        return len(looks) - lo_look >= cfg.ball_quiet_min_looks

    for r in records:
        t = float(r["t"])
        if r.get("bn", True):
            looks.append(t)

        while si < len(spans) and spans[si][1] < t:
            si += 1
        span = spans[si] if si < len(spans) and spans[si][0] <= t else None

        if is_ball(r):
            if span is not None:
                n_veto += 1     # a ball while walking: the point continues
            last_ball, pending_a, pending_b = t, False, False
            continue
        if last_ball is None:
            continue

        quiet = t - last_ball
        if span is not None:
            pending_b = False
            if quiet >= cfg.walk_ball_veto_s and not pending_a:
                if not enough_looks(t, cfg.walk_ball_veto_s):
                    n_thin += 1
                    continue
                pending_a = True
                onsets.append((max(span[0], last_ball), "walk"))
        else:
            pending_a = False
            if quiet >= cfg.no_walk_quiet_s and not pending_b:
                if not enough_looks(t, cfg.no_walk_quiet_s):
                    n_thin += 1
                    continue
                pending_b = True
                onsets.append((last_ball + cfg.no_walk_stamp_s, "ball-quiet"))

    n_a = sum(1 for _, s in onsets if s == "walk")
    print(f"[REEL]   walk-ball: {n_a} walk end(s), {len(onsets) - n_a} "
          f"ball-quiet end(s); {n_veto} frame(s) of walking vetoed by a visible "
          f"ball; {n_thin} window(s) dropped for fewer than "
          f"{cfg.ball_quiet_min_looks} looks")
    return sorted(onsets, key=lambda x: x[0])


def build_reel(video_path: str,
               cfg: Optional[ReelConfig] = None,
               output_path: Optional[str] = None,
               force_telemetry: bool = False,
               device: str = "mps",
               dry_run: bool = False,
               segments_suffix: Optional[str] = None,
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
    wants_ball_end = (cfg.end_policy == "walk-ball"
                      or cfg.ball_quiet_mode != "off")
    needs_full = ((wants_ball_end and not cfg.fast_end)
                  or (cfg.use_far and not cfg.fast_far)
                  or not cfg.fast_near)
    if cfg.end_policy in ("trace", "energy") and not cfg.fast_end:
        # Both policies read an IMM trace, and the trace needs the dense ball
        # stream only anya_end_telemetry can be asked for.  The shared full pass
        # samples the ball far too slowly; failing here beats failing inside
        # trace_intervals with the sample rate as the proximate cause.
        raise ValueError(
            f"end_policy={cfg.end_policy!r} requires fast_end: its ball trace "
            f"needs the {cfg.trace_ball_fps:.0f} Hz stream from "
            f"anya_end_telemetry, which --no-fast-end does not produce.")

    # ── Stage 1: telemetry ──────────────────────────────────────────────
    telemetry = None
    if needs_full:
        print("[REEL] Stage 1/7  telemetry")
        _emit(on_progress, 1, 0.0)
        telemetry = extract_anya_telemetry(
            video_path, force=force_telemetry,
            progress_cb=lambda cur, tot: _emit(on_progress, 1, cur / max(1, tot)))
    else:
        print("[REEL] Stage 1/7  telemetry — skipped (every consumer has its "
              "own pass: fast_near, fast_far, fast_end)")
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
    end_telemetry = telemetry
    if cfg.fast_end:
        # One decode of the shared 540p proxy feeds both point-end signals:
        # the pose pass the walking classifier reads, and the whole-court ball
        # stream ball-quiet reads.
        print("[REEL] Stage 5/7  walking + ball quiet (fast path)")
        _emit(on_progress, 5)   # opens with a possible one-time proxy build
        ecfg = None
        end_out = None
        if cfg.end_policy in ("trace", "energy"):
            # The two trace-based policies are the only consumers that need a
            # dense ball stream, so they request one per-run rather than
            # raising the global default and slowing every other path down.
            # Their own cache path lets both rates coexist, so an A/B does not
            # re-extract per flip — and "energy" reuses the same cache file as
            # "trace", so flipping between them costs nothing.
            ecfg = EndExtractorConfig()
            ecfg.ball_fps = cfg.trace_ball_fps
            ecfg.ball_imgsz = cfg.trace_ball_imgsz
            end_out = end_telemetry_path_for(video_path, cfg.trace_ball_fps,
                                             cfg.trace_ball_imgsz)
        end_telemetry = extract_end_telemetry(
            video_path, force=force_telemetry, cfg=ecfg, out_path=end_out,
            progress_cb=lambda cur, tot: _emit(on_progress, 5,
                                               cur / max(1, tot)))
        walks, walk_result = _walk_intervals(
            video_path, device=device, dets_npz=end_dets_path_for(video_path),
            pose_npz=end_pose_path_for(video_path), model_path=_walk_model_path(cfg),
            return_result=True)
    else:
        print("[REEL] Stage 5/7  walking (dead-time proxy)")
        _emit(on_progress, 5)   # predict_video returns only when done
        walks, walk_result = _walk_intervals(video_path, device=device,
                                             return_result=True)
    # One ball stream for every consumer below.  `_ball_stream` exists so that a
    # difference between point-end policies is never a difference in what counts
    # as a ball, and the confidence score is now one of those consumers.
    ball_records, is_ball = None, None
    trace_details = []
    if (cfg.end_policy in ("walk-ball", "trace", "confidence", "energy")
            or cfg.ball_quiet_mode != "off"):
        ball_meta, ball_records, filter_balls = _ball_stream(end_telemetry)
        def is_ball(r):
            return bool(filter_balls(r))

    if cfg.end_policy == "confidence":
        # Scored in stage 6, where `starts` exists: the accumulator resets at
        # every point start, so it cannot be built before they are known.
        dead_onsets = []
        print(f"[REEL]   walk intervals: {len(walks)} -> confidence policy "
              f"(threshold {cfg.deadtime_score_threshold})")
    elif cfg.end_policy in ("trace", "energy"):
        intervals, tstats = ball_trace.trace_intervals(
            ball_meta, ball_records, filter_balls,
            _stem_path(video_path, "_court_cache.json"), cfg)
        print(f"[REEL]   trace: {tstats['n_pre_bridge']} interval(s) -> "
              f"{tstats['n_post_bridge']} after bridging {tstats['bridged_s']}s "
              f"at {cfg.trace_merge_gap_s}s; {tstats['alive_s']:.1f}s alive; gate "
              f"passed {tstats['gate_rate']:.0%} of {tstats['dets']} detection(s)")
        if cfg.end_policy == "energy":
            # Scored in stage 6 for the same reason "confidence" is: the bar
            # resets at every point start, so it cannot be built before the
            # starts are known.
            dead_onsets = []
            near_sigs = _near_end_signals(video_path, walk_result, cfg)
            energy_evidence = energy_policy.build_evidence(
                ball_meta, ball_records, walk_result, intervals, near=near_sigs)
            print(f"[REEL]   energy: start {cfg.energy_start} hold "
                  f"{cfg.energy_hold_s}s step {cfg.energy_step_s}s")
        else:
            looks = [float(r["t"]) for r in ball_records if r.get("bn")]
            dead_onsets, trace_details = ball_trace.trace_onsets(
                intervals, walks, cfg, look_times=looks,
                last_record_t=looks[-1] if looks else duration)
            n_hi = sum(1 for d in trace_details if d["level"] == "high")
            print(f"[REEL]   dead-time onsets: {n_hi} trace+walk (high) + "
                  f"{len(dead_onsets) - n_hi} trace-only (medium)")
    elif cfg.end_policy == "walk-ball":
        print(f"[REEL]   walk intervals: {len(walks)} -> "
              f"{len(usable_walk_intervals(walks, cfg))} usable")
        dead_onsets = _walk_ball_onsets(end_telemetry, walks, cfg)
    else:
        dead_onsets = walk_onsets(walks, cfg)
        n_walk = len(dead_onsets)
        print(f"[REEL]   walk intervals: {len(walks)} -> {n_walk} usable onset(s)")
        if cfg.ball_quiet_mode != "off":
            dead_onsets += _ball_quiet_onsets(end_telemetry, cfg)
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

    confidence_rows = []
    if ball_records is not None and cfg.end_policy != "energy":
        # Skipped under "energy": the accumulator is passive telemetry on every
        # other policy, but here the energy rows are the score worth reading and
        # this pass would be pure overhead.
        confidence_rows = score_deadtime(starts, duration, walk_result,
                                         ball_records, is_ball, cfg)
    if cfg.end_policy == "confidence":
        dead_onsets = deadtime_onsets(confidence_rows, cfg)
        print(f"[REEL]   dead-time onsets: {len(dead_onsets)} confidence "
              f"crossing(s) from {len(starts)} start(s)")

    energy_rows = []
    if cfg.end_policy == "energy":
        energy_rows, trace_details = energy_policy.score_energy(
            starts, duration, energy_evidence, cfg)
        dead_onsets = energy_policy.energy_onsets(trace_details)
        n_hi = sum(1 for d in trace_details if d["level"] == "high")
        print(f"[REEL]   dead-time onsets: {len(dead_onsets)} energy drain(s) "
              f"from {len(starts)} start(s) ({n_hi} with walking)")

    segments = build_segments(
        starts, dead_onsets, duration, cfg,
        min_point_s=_min_point_s(cfg))

    # Decorate ends with their graded confidence.  Keyed on the onset instant:
    # find_point_end returns min(t, hard_cap) and the cap path reports a
    # different end_method, so a trace-* method implies the exact instant here.
    by_t = {round(d["t"], 3): d for d in trace_details}
    for seg in segments:
        det = by_t.get(round(seg.end_t, 3))
        if det is not None and seg.end_method == det["source"]:
            seg.end_confidence, seg.end_reason = det["level"], det["reason"]
        else:
            seg.end_confidence, seg.end_reason = "", seg.end_method

    seg_path = _stem_path(video_path, segments_suffix or SEGMENTS_SUFFIX)
    with open(seg_path, "w") as fh:
        json.dump({
            "video": os.path.basename(video_path),
            "duration_s": duration,
            "config": cfg.__dict__,
            "n_serve_starts": len(starts),
            "segments": [s.as_dict() for s in segments],
            "end_confidence": {
                "policy": cfg.end_policy,
                "levels": {"high": "trace gap + walking",
                           "medium": "trace gap only",
                           "": "guard or cap decided the end"},
                "samples": trace_details,
            },
            "deadtime_confidence": {
                "threshold": cfg.deadtime_score_threshold,
                "samples": confidence_rows,
            },
            "energy": {
                "start": cfg.energy_start,
                "floor": cfg.energy_floor,
                "step_s": cfg.energy_step_s,
                "samples": energy_rows,
            },
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
