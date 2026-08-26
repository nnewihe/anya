"""
run.py
======
One call that takes a video and returns a cut reel -- the anya2 equivalent of
`pipeline.rally_reel.build_reel`, with the same signature so the desktop app
switches by changing an import.

    segments, out_path = build_reel(video, output, cfg=Anya2Config(),
                                    on_progress=cb)

Stages, and why they are in this order
--------------------------------------
  1 CALIBRATION   the court corners, prompted once per video and cached.  This
                  opens a cv2 window, so it MUST run on the caller's main
                  thread -- `ensure_court` is separated out for exactly that,
                  and the desktop app calls it before starting the worker.
  2 PERCEIVE      the two pose passes.  The whole cost of the pipeline; see
                  perceive.py for why one ROI cannot serve both ends.
  3 TRACKS        <=2 near + <=2 far player slots on one timeline.
  4 END SIGNALS   the walking classifier and near_end's four pose signals, both
                  read off the near track.  Agent 3 needs them; nothing else
                  does.
  5 DETECTORS     the three agents, independently.
  6 ORCHESTRATE   structure, rules, recovery, smoothing -> segments.
  7 CUT           ffmpeg.

Progress is reported as (stage_index, n_stages, label, fraction) to match what
`rally_reel` emits, so the app's existing progress handling works unchanged.
"""

import json
import os
import subprocess
import tempfile
from typing import Callable, List, Optional, Tuple

import numpy as np

from pipeline.anya2 import far_serve as FS
from pipeline.anya2 import near_serve as NS
from pipeline.anya2 import perceive as PC
from pipeline.anya2 import point_end as PE
from pipeline.anya2 import tracks as TR
from pipeline.anya2.config import Anya2Config
from pipeline.anya2.contract import dump_events
from pipeline.anya2.orchestrator import SEGMENTS_SUFFIX, build_reel as orchestrate

N_STAGES = 7


def _emit(cb, i, label, frac=None):
    if cb:
        try:
            cb(i, N_STAGES, label, frac)
        except Exception:
            pass


def ensure_court(video: str) -> None:
    """Prompt for the court corners if they are not cached.

    Separated from `build_reel` because it opens an OpenCV window: it has to run
    on the main thread, before any worker starts.  Idempotent once cached.
    """
    from pipeline.anya2 import court as C
    if os.path.isfile(C.court_cache_path(video)):
        return
    from pipeline.utilities import init_court
    init_court(video, analysis_size=C.ANALYSIS_SIZE)


def _stem(video: str) -> Tuple[str, str]:
    d = os.path.dirname(os.path.abspath(video))
    return d, os.path.splitext(os.path.basename(video))[0]


def _end_signals(video: str, force: bool = False) -> None:
    """Walking probabilities and near_end's four signals, cached beside the video.

    Both are computed from the NEAR track through a shim npz shaped like
    `walking.select_near`'s output, because that is what both consumers already
    read.  The near slot with the better coverage leads and the other fills its
    gaps -- on a changeover the same human moves between slots, and using one
    slot alone loses half the clip.
    """
    d, st = _stem(video)
    walk_p = os.path.join(d, f"{st}_anya2_walk.npz")
    sig_p = os.path.join(d, f"{st}_anya2_endsig.npz")
    if os.path.isfile(walk_p) and os.path.isfile(sig_p) and not force:
        return
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    from walking.predict import predict_video
    # near_end.py lives in pipeline/, not the repo root -- every other caller
    # in this codebase imports it as pipeline.near_end (`from ..near_end import
    # ...` inside pipeline.rally_reel, `from .near_end import ...` inside
    # pipeline/tune_energy.py).  `import near_end` alone only resolves when the
    # CALLER has also put pipeline/ itself on sys.path, which the CLI scripts
    # used to test this module did and this function did not -- the bug
    # surfaced as "No module named 'near_end'" the first time this ran from the
    # desktop app rather than from one of those scripts.
    from pipeline import near_end as NE

    z = TR.load(video)
    bb = z["bbox"]
    cov = [np.isfinite(bb[:, s, 0]).mean() for s in TR.NEAR_SLOTS]
    lead, other = (0, 1) if cov[0] >= cov[1] else (1, 0)
    kp = z["kp"][:, lead].copy()
    bx = bb[:, lead].copy()
    fill = ~np.isfinite(bx[:, 0]) & np.isfinite(bb[:, other, 0])
    kp[fill] = z["kp"][fill, other]
    bx[fill] = bb[fill, other]
    extra = {k: z[k] for k in ("stride", "src_fps", "n_src_frames") if k in z}
    shim = os.path.join(d, f"{st}_anya2_walk_pose.npz")
    np.savez_compressed(shim, kp=kp, bbox=bx,
                        on_court=z["eligible"][:, lead].astype(np.float32),
                        fps=np.float64(z["fps"]), **extra)
    r = predict_video(video, pose_npz=shim)
    np.savez_compressed(walk_p, prob=r["prob"], is_walking=r["is_walking"],
                        fps=np.float64(r["fps"]))
    sig = NE.signals_for_video(video, pose_npz=shim)
    np.savez_compressed(sig_p, **{k: np.asarray(sig[k], dtype=np.float32)
                                  for k in NE.SIGNAL_NAMES})


def cut(video: str, segments: List[dict], output: str,
        cfg: Optional[Anya2Config] = None, on_progress=None) -> str:
    """ffmpeg: encode each segment, then concat."""
    cfg = cfg or Anya2Config()
    if not segments:
        raise ValueError("no segments to cut")
    tmp = tempfile.mkdtemp(prefix="anya2_reel_")
    parts = []
    vf = (["-vf", f"scale=-2:{cfg.scale_height}"] if cfg.scale_height else [])
    for i, s in enumerate(segments):
        p = os.path.join(tmp, f"seg_{i:04d}.mp4")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{s['start']:.3f}", "-i", video,
               "-t", f"{s['stop'] - s['start']:.3f}", *vf,
               "-c:v", "libx264", "-crf", str(cfg.crf),
               "-preset", cfg.preset, "-pix_fmt", "yuv420p"]
        cmd += (["-c:a", "aac", "-b:a", "160k"] if cfg.keep_audio else ["-an"])
        cmd.append(p)
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            parts.append(p)
        _emit(on_progress, 6, f"Cutting segment {i + 1}/{len(segments)}",
              (i + 1) / len(segments))
    if not parts:
        raise RuntimeError("every segment failed to encode")
    lst = os.path.join(tmp, "concat.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        for p in parts:
            fh.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", output], capture_output=True, check=True)
    return output


def build_reel(video_path: str, output_path: Optional[str] = None,
               cfg: Optional[Anya2Config] = None,
               on_progress: Optional[Callable] = None,
               dry_run: bool = False) -> Tuple[List[dict], Optional[str]]:
    """Video in, cut reel out.  Signature-compatible with rally_reel.build_reel."""
    cfg = cfg or Anya2Config()
    d, st = _stem(video_path)
    output_path = output_path or os.path.join(d, f"{st}_anya2_reel.mp4")

    _emit(on_progress, 0, "Court calibration")
    ensure_court(video_path)

    _emit(on_progress, 1, "Detecting players (near)")
    near_npz = PC.near(video_path, device=cfg.perceive.device,
                       pose_fps=cfg.perceive.pose_fps, force=cfg.perceive.force)
    _emit(on_progress, 1, "Detecting players (far)", 0.5)
    far_npz = PC.far(video_path, device=cfg.perceive.device,
                     pose_fps=cfg.perceive.pose_fps, force=cfg.perceive.force)

    _emit(on_progress, 2, "Building player tracks")
    TR.build(video_path, near_npz, far_npz, verbose=False)

    _emit(on_progress, 3, "Player motion signals")
    _end_signals(video_path, force=cfg.perceive.force)

    _emit(on_progress, 4, "Detecting serves and point ends")
    if cfg.near.enabled:
        ev = NS.detect_video(video_path, verbose=False,
                             threshold=cfg.near.threshold or NS.THRESHOLD,
                             require_court=cfg.near.require_court,
                             lead_s=cfg.near.lead_s, refract_s=cfg.near.refract_s)
        dump_events(ev, os.path.join(d, f"{st}{NS.EVENTS_SUFFIX}"))
    if cfg.far.enabled:
        ev = FS.detect_video(video_path, verbose=False,
                             threshold=cfg.far.threshold or FS.THRESHOLD,
                             require_court=cfg.far.require_court,
                             lead_s=cfg.far.lead_s, refract_s=cfg.far.refract_s,
                             w_still=cfg.far.w_still)
        dump_events(ev, os.path.join(d, f"{st}{FS.EVENTS_SUFFIX}"))
    if cfg.end.enabled:
        ev = PE.detect_video(video_path, verbose=False,
                             hi=cfg.end.live_hi or PE.LIVE_HI,
                             lo=cfg.end.live_lo, smooth_s=cfg.end.smooth_s,
                             min_live_s=cfg.end.min_live_s)
        dump_events(ev, os.path.join(d, f"{st}{PE.EVENTS_SUFFIX}"))

    _emit(on_progress, 5, "Assembling the reel")
    # A disabled agent is disabled for the ORCHESTRATOR too, not merely skipped
    # here -- its events are cached on disk and would otherwise still be read.
    cfg.reel.use_near = cfg.near.enabled
    cfg.reel.use_far = cfg.far.enabled
    cfg.reel.use_end = cfg.end.enabled
    res = orchestrate(video_path, cfg=cfg.reel, verbose=False)
    with open(os.path.join(d, f"{st}{SEGMENTS_SUFFIX}"), "w") as fh:
        json.dump(res, fh, indent=1)
    segments = res["segments"]
    if dry_run or not segments:
        return segments, None

    _emit(on_progress, 6, "Cutting video", 0.0)
    out = cut(video_path, segments, output_path, cfg, on_progress)
    return segments, out
