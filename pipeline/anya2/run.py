"""
run.py
======
Video in, cut reel out.  The entry point the desktop app calls.

    segments, out_path = build_reel(video, output, cfg=Anya2Config(), on_progress=cb)

`video` is one path, or several GoPro chapter files of one recording (they are
joined into one file first, and every later stage sees a single video).

The nine stages, in order
-------------------------
  1 JOIN         chapter files -> one video (no-op for a single file)
  2 CALIBRATE    court corners, clicked once and cached.  Opens a window, so
                 the desktop app calls `ensure_court` on the main thread first.
  3 CAMERA       where each frame sits relative to the calibration frame
                 (camera.py).  Before perceive: the far crop is sized from it.
  4 PERCEIVE     the near and far pose passes (perceive.py) -- most of the cost
  5 TRACKS       <= 2 near + <= 2 far player slots (tracks.py)
  6 END SIGNALS  walking classifier + near_end's four signals, off the near
                 player (needed by point_end only)
  7 DETECT       near serves, far serves, point ends -> event JSON files
  8 ASSEMBLE     orchestrator.py -> segments (+ `<stem>_anya2_reel.json`)
  9 CUT          ffmpeg

Every artifact goes to `pipeline.workdir.artifact_dir(video)`: beside the video
normally, or the desktop app's tmp_anya folder.  The finished reel defaults to
beside the (first) input video.

Progress is reported as on_progress(stage, N_STAGES, label, fraction), with
stages numbered from 1.  Each stage announcement is also a cancel point.
"""

import json
import os
import tempfile
from typing import Callable, List, Optional, Tuple

import numpy as np

from pipeline import cancel
from pipeline import workdir
from pipeline.anya2 import camera
from pipeline.anya2 import far_serve
from pipeline.anya2 import near_serve
from pipeline.anya2 import perceive
from pipeline.anya2 import point_end
from pipeline.anya2 import tracks
from pipeline.anya2.config import Anya2Config
from pipeline.anya2.contract import dump_events
from pipeline.anya2.orchestrator import SEGMENTS_SUFFIX, build_reel as orchestrate

N_STAGES = 9

WALK_SUFFIX = "_anya2_walk.npz"            # stage 6 outputs
ENDSIG_SUFFIX = "_anya2_endsig.npz"
WALK_POSE_SUFFIX = "_anya2_walk_pose.npz"  # the near-player pose stage 6 reads


def build_reel(video_path, output_path: Optional[str] = None,
               cfg: Optional[Anya2Config] = None,
               on_progress: Optional[Callable] = None,
               dry_run: bool = False) -> Tuple[List[dict], Optional[str]]:
    """Run all nine stages.  Returns (segments, reel path or None).

    `dry_run` stops before cutting.  Same signature as rally_reel.build_reel.
    """
    cfg = cfg or Anya2Config()
    # Default output beside the FIRST input -- never in the artifact dir, which
    # under the desktop app is a temporary folder.
    first_input = video_path if isinstance(video_path, (str, os.PathLike)) else video_path[0]
    input_dir, input_stem = _dir_and_stem(first_input)
    output_path = output_path or os.path.join(input_dir, f"{input_stem}_anya2_reel.mp4")

    _emit(on_progress, 1, "Joining video files")
    video = _join(video_path, on_progress)
    _, stem = _dir_and_stem(video)
    artifacts = workdir.artifact_dir(video)

    _emit(on_progress, 2, "Court calibration")
    ensure_court(video)

    _emit(on_progress, 3, "Tracking the camera")
    camera.estimate(video, force=cfg.perceive.force,
                    sample_fps=cfg.perceive.camera_sample_fps or camera.SAMPLE_FPS,
                    on_progress=lambda f: _emit(on_progress, 3, "Tracking the camera", f))

    _emit(on_progress, 4, "Detecting players (near)")
    near_dets = perceive.near(video, device=cfg.perceive.device,
                              pose_fps=cfg.perceive.pose_fps, force=cfg.perceive.force)
    _emit(on_progress, 4, "Detecting players (far)", 0.5)
    far_dets = perceive.far(video, device=cfg.perceive.device,
                            pose_fps=cfg.perceive.pose_fps, force=cfg.perceive.force)

    _emit(on_progress, 5, "Building player tracks")
    tracks.build(video, near_dets, far_dets, verbose=False)

    _emit(on_progress, 6, "Player motion signals")
    _compute_end_signals(video, force=cfg.perceive.force)

    _emit(on_progress, 7, "Detecting serves and point ends")
    _run_detectors(video, cfg, artifacts, stem)

    _emit(on_progress, 8, "Assembling the reel")
    # A disabled detector is disabled for the orchestrator too: its stale event
    # file must not be read.  (This writes to the caller's cfg -- QUIRKS.md #2.)
    cfg.reel.use_near = cfg.near.enabled
    cfg.reel.use_far = cfg.far.enabled
    cfg.reel.use_end = cfg.end.enabled
    reel = orchestrate(video, cfg=cfg.reel, verbose=False)
    with open(os.path.join(artifacts, f"{stem}{SEGMENTS_SUFFIX}"), "w") as fh:
        json.dump(reel, fh, indent=1)
    segments = reel["segments"]
    if dry_run or not segments:
        return segments, None

    _emit(on_progress, 9, "Cutting video", 0.0)
    return segments, cut(video, segments, output_path, cfg, on_progress)


def ensure_court(video, on_progress=None) -> None:
    """Ask the user for the court corners unless they are cached.

    Opens an OpenCV window, so call it on the main thread before any worker.
    Corners are clicked on the JOINED video (what every later stage reads).
    """
    video = _join(video, on_progress)
    from pipeline.anya2 import court
    if os.path.isfile(court.court_cache_path(video)):
        return
    from pipeline.utilities import init_court
    init_court(video, analysis_size=court.ANALYSIS_SIZE)


def _run_detectors(video, cfg, artifacts, stem):
    """Stage 7: each enabled detector writes its `<stem>_anya2_*.json`.

    Note `x or DEFAULT` below: a threshold of 0.0 also means "default"
    (QUIRKS.md #1).
    """
    if cfg.near.enabled:
        events = near_serve.detect_video(
            video, verbose=False,
            threshold=cfg.near.threshold or near_serve.THRESHOLD,
            require_court=cfg.near.require_court,
            lead_s=cfg.near.lead_s, refract_s=cfg.near.refract_s)
        dump_events(events, os.path.join(artifacts, f"{stem}{near_serve.EVENTS_SUFFIX}"))
    if cfg.far.enabled:
        events = far_serve.detect_video(
            video, verbose=False,
            threshold=cfg.far.threshold or far_serve.THRESHOLD,
            require_court=cfg.far.require_court,
            lead_s=cfg.far.lead_s, refract_s=cfg.far.refract_s,
            w_still=cfg.far.w_still)
        dump_events(events, os.path.join(artifacts, f"{stem}{far_serve.EVENTS_SUFFIX}"))
    if cfg.end.enabled:
        events = point_end.detect_video(
            video, verbose=False,
            hi=cfg.end.live_hi or point_end.LIVE_HI,
            lo=cfg.end.live_lo, smooth_s=cfg.end.smooth_s,
            min_live_s=cfg.end.min_live_s)
        dump_events(events, os.path.join(artifacts, f"{stem}{point_end.EVENTS_SUFFIX}"))


def _compute_end_signals(video: str, force: bool = False) -> None:
    """Stage 6: walking probability and near_end's four signals, cached.

    Both read ONE near-player pose track in `walking.select_near`'s npz format,
    written here as `<stem>_anya2_walk_pose.npz` (see
    `_near_player_pose_for_end_signals`).
    """
    _, stem = _dir_and_stem(video)
    artifacts = workdir.artifact_dir(video)
    walk_path = os.path.join(artifacts, f"{stem}{WALK_SUFFIX}")
    endsig_path = os.path.join(artifacts, f"{stem}{ENDSIG_SUFFIX}")
    if os.path.isfile(walk_path) and os.path.isfile(endsig_path) and not force:
        return

    from pipeline.anya2.signals import walking_pose_helpers
    walking_pose_helpers()          # puts the repo root on sys.path for `walking`
    from walking.predict import predict_video
    from pipeline import near_end

    pose_path = _near_player_pose_for_end_signals(video, artifacts, stem)
    walking = predict_video(video, pose_npz=pose_path)
    np.savez_compressed(walk_path, prob=walking["prob"], is_walking=walking["is_walking"],
                        fps=np.float64(walking["fps"]))
    end_signals = near_end.signals_for_video(video, pose_npz=pose_path)
    np.savez_compressed(endsig_path, **{name: np.asarray(end_signals[name], dtype=np.float32)
                                        for name in near_end.SIGNAL_NAMES})


def _near_player_pose_for_end_signals(video, artifacts, stem):
    """Write one near-player track (kp, bbox, on_court, fps) and return its path.

    The near slot with more coverage leads; wherever it is empty, the other
    near slot fills in (on a changeover the same person can move slots).
    """
    track_data = tracks.load(video)
    boxes = track_data["bbox"]
    coverage = [np.isfinite(boxes[:, slot, 0]).mean() for slot in tracks.NEAR_SLOTS]
    lead, other = (0, 1) if coverage[0] >= coverage[1] else (1, 0)
    kp = track_data["kp"][:, lead].copy()
    lead_boxes = boxes[:, lead].copy()
    fill = ~np.isfinite(lead_boxes[:, 0]) & np.isfinite(boxes[:, other, 0])
    kp[fill] = track_data["kp"][fill, other]
    lead_boxes[fill] = boxes[fill, other]
    extra = {key: track_data[key] for key in ("stride", "src_fps", "n_src_frames")
             if key in track_data}
    path = os.path.join(artifacts, f"{stem}{WALK_POSE_SUFFIX}")
    np.savez_compressed(path, kp=kp, bbox=lead_boxes,
                        on_court=track_data["eligible"][:, lead].astype(np.float32),
                        fps=np.float64(track_data["fps"]), **extra)
    return path


def cut(video: str, segments: List[dict], output: str,
        cfg: Optional[Anya2Config] = None, on_progress=None) -> str:
    """Stage 9: encode each segment with ffmpeg, then concatenate."""
    cfg = cfg or Anya2Config()
    if not segments:
        raise ValueError("no segments to cut")
    # Under the desktop app the segment files go to its work dir (it decides
    # whether to keep them); otherwise to a temp dir removed at the end.
    work_dir = workdir.get_work_dir()
    if work_dir:
        segment_dir = os.path.join(work_dir, "cut_segments")
        os.makedirs(segment_dir, exist_ok=True)
    else:
        segment_dir = tempfile.mkdtemp(prefix="anya2_reel_")

    scale = (["-vf", f"scale=-2:{cfg.scale_height}"] if cfg.scale_height else [])
    parts = []
    for i, segment in enumerate(segments):
        cancel.check()
        part = os.path.join(segment_dir, f"seg_{i:04d}.mp4")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{segment['start']:.3f}", "-i", video,
               "-t", f"{segment['stop'] - segment['start']:.3f}", *scale,
               "-c:v", "libx264", "-crf", str(cfg.crf),
               "-preset", cfg.preset, "-pix_fmt", "yuv420p",
               "-vsync", "cfr"]
        cmd += (["-c:a", "aac", "-b:a", "160k"] if cfg.keep_audio else ["-an"])
        cmd.append(part)
        # cancel.run so Cancel can interrupt a long encode.
        if cancel.run(cmd, capture_output=True).returncode == 0:
            parts.append(part)
        _emit(on_progress, 9, f"Cutting segment {i + 1}/{len(segments)}",
              (i + 1) / len(segments))
    if not parts:
        raise RuntimeError("every segment failed to encode")
    # Audio is re-encoded at the join so it cannot drift from the picture
    # (see `concat_cmd` in pipeline.utilities).
    from pipeline.utilities import concat_cmd, write_concat_list
    concat_list = write_concat_list(parts, os.path.join(segment_dir, "concat.txt"))
    cancel.run(concat_cmd(concat_list, output, with_audio=cfg.keep_audio,
                          audio_bitrate="160k", quiet=True),
               capture_output=True, check=True)
    if not work_dir:
        import shutil
        shutil.rmtree(segment_dir, ignore_errors=True)
    return output


# ── helpers ──────────────────────────────────────────────────────────────

def _emit(callback, stage, label, fraction=None):
    """Report progress -- and check for Cancel, once per stage announcement."""
    cancel.check()
    if callback:
        try:
            callback(stage, N_STAGES, label, fraction)
        except Exception:
            pass


def _join(video, on_progress=None) -> str:
    """One source path from one-or-many inputs (chapters are joined, cached)."""
    from pipeline import join
    return join.resolve_input(
        video, on_progress=(lambda f: _emit(on_progress, 1, "Joining video files", f))
        if on_progress else None)


def _dir_and_stem(video: str) -> Tuple[str, str]:
    """The video's OWN folder and file stem (not the artifact dir)."""
    folder = os.path.dirname(os.path.abspath(video))
    return folder, os.path.splitext(os.path.basename(video))[0]
