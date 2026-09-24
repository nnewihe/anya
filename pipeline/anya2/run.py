"""
run.py
======
One call that takes a video and returns a cut reel -- the anya2 equivalent of
`pipeline.rally_reel.build_reel`, with the same signature so the desktop app
switches by changing an import.

    segments, out_path = build_reel(video, output, cfg=Anya2Config(),
                                    on_progress=cb)

`video` may be a single path or several: GoPro chapters of one recording are
joined into one file first (`pipeline.join`), and every stage after that sees
an ordinary single source video.  See that module for why the join happens
here rather than the pipeline learning to read a list.

Stages, and why they are in this order
--------------------------------------
  1 JOIN          several GoPro chapter files -> one video.  A no-op, and not
                  even a file, for the single-video case that is still the
                  normal one.
  2 CALIBRATION   the court corners, prompted once per video and cached.  This
                  opens a cv2 window, so it MUST run on the caller's main
                  thread -- `ensure_court` is separated out for exactly that,
                  and the desktop app calls it before starting the worker.
  3 CAMERA TRACK  where each frame sits relative to the frame those corners were
                  clicked on.  BEFORE perceive, and not merely for tidiness:
                  the far pose pass crops a fixed rectangle around the far
                  baseline, and that rectangle has to be sized to cover
                  everywhere the camera went, which is only knowable once the
                  track exists.  It also builds the 540p proxy that the near
                  pass then reuses, so it costs one decode, not two.
  4 PERCEIVE      the two pose passes.  The whole cost of the pipeline; see
                  perceive.py for why one ROI cannot serve both ends.
  5 TRACKS        <=2 near + <=2 far player slots on one timeline.
  6 END SIGNALS   the walking classifier and near_end's four pose signals, both
                  read off the near track.  Agent 3 needs them; nothing else
                  does.
  7 DETECTORS     the three agents, independently.
  8 ORCHESTRATE   structure, rules, recovery, smoothing -> segments.
  9 CUT           ffmpeg.

Progress is reported as (stage_index, n_stages, label, fraction), stages
numbered from ONE -- which is what the desktop app's bar already assumes when
it places a stage's fraction at `(i - 1 + frac) / n`.  It mattered little
while stage one was a cached calibration check; the join is minutes of I/O
with a real fraction to report, and off by one it would sit at zero for all
of them.
"""

import json
import os
import subprocess
import tempfile
from typing import Callable, List, Optional, Tuple

import numpy as np

from pipeline import cancel
from pipeline.anya2 import camera as CAM
from pipeline.anya2 import far_serve as FS
from pipeline.anya2 import near_serve as NS
from pipeline.anya2 import perceive as PC
from pipeline.anya2 import point_end as PE
from pipeline.anya2 import tracks as TR
from pipeline.anya2.config import Anya2Config
from pipeline.anya2.contract import dump_events
from pipeline.anya2.orchestrator import SEGMENTS_SUFFIX, build_reel as orchestrate

N_STAGES = 9


def _emit(cb, i, label, frac=None):
    # Every stage announces itself through here exactly once, which makes this
    # the one place a per-stage cancel check can live without a line per stage
    # -- and a stage added later inherits it. The long stages check in far more
    # often than this from the inside (see pipeline.cancel).
    cancel.check()
    if cb:
        try:
            cb(i, N_STAGES, label, frac)
        except Exception:
            pass


def ensure_court(video, on_progress=None) -> None:
    """Prompt for the court corners if they are not cached.

    Separated from `build_reel` because it opens an OpenCV window: it has to run
    on the main thread, before any worker starts.  Idempotent once cached.

    Accepts the same one-or-many `video` as `build_reel`, and joins first --
    the corners are clicked on the JOINED file, because that is what every
    later stage indexes into.  The join is cached, so `build_reel` calling
    `resolve_input` again a moment later costs a sidecar read.
    """
    video = _join(video, on_progress)
    from pipeline.anya2 import court as C
    if os.path.isfile(C.court_cache_path(video)):
        return
    from pipeline.utilities import init_court
    init_court(video, analysis_size=C.ANALYSIS_SIZE)


def _single_decode() -> bool:
    """ANYA_SINGLE_DECODE_PROXIES=1: build both proxies from one source decode.

    Off by default so the desktop app's behaviour is unchanged; the Raspberry
    Pi service turns it on, because there the source decode is the cost.
    """
    return os.environ.get("ANYA_SINGLE_DECODE_PROXIES", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _join(video, on_progress=None) -> str:
    """One source path from whatever the caller passed.

    A single path comes back untouched; several are remuxed into one file in
    the artifact dir.  Every stage below this line takes the result and knows
    nothing about chapters.
    """
    from pipeline import join as J
    return J.resolve_input(
        video, on_progress=(lambda f: _emit(on_progress, 1, "Joining video files", f))
        if on_progress else None)


def _stem(video: str) -> Tuple[str, str]:
    """The video's OWN directory and stem -- NOT the artifact dir.

    The DIRECTORY half is used only for the output video's default location,
    which must stay beside the input regardless of any work-dir override: the
    app's tmp_anya holds calibration and interim files, never the finished
    reel.  (With chapter inputs it is the FIRST CHAPTER that supplies it, not
    the join -- the join itself lives in tmp_anya.)  Every cache or event file
    uses `pipeline.workdir.artifact_dir` instead, paired with the stem half of
    this -- see `_end_signals` and `build_reel` below for the split.
    """
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
    from pipeline import workdir as WD
    _, st = _stem(video)
    d = WD.artifact_dir(video)
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


def keyframe_times(video: str) -> List[float]:
    """Video keyframe timestamps in seconds, from packet flags (no decode)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True).stdout
    kf = []
    for line in out.splitlines():
        t, _, flags = line.partition(",")
        if "K" in flags:
            try:
                kf.append(float(t))
            except ValueError:
                pass
    return sorted(kf)


def snap_to_keyframes(segments: List[dict], keyframes: List[float]) -> List[dict]:
    """Move each start back to the keyframe at or before it; merge overlaps.

    A stream copy can only begin on a keyframe.  Moving the start EARLIER only
    adds pre-roll (<= one GOP, 0.4-1 s on the corpus cameras) and never cuts
    into a point, which is the direction that matters.  A start pulled back
    past the previous segment's end would replay footage, so those two merge.
    """
    import bisect
    out: List[dict] = []
    for s in sorted(segments, key=lambda s: s["start"]):
        i = bisect.bisect_right(keyframes, s["start"] + 1e-6) - 1
        start = keyframes[i] if i >= 0 else 0.0
        if out and start <= out[-1]["stop"]:
            out[-1]["stop"] = max(out[-1]["stop"], s["stop"])
            continue
        out.append({**s, "start": start, "stop": s["stop"]})
    return out


def _codec(video: str) -> str:
    try:
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "csv=p=0", video],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def cut(video: str, segments: List[dict], output: str,
        cfg: Optional[Anya2Config] = None, on_progress=None) -> str:
    """ffmpeg: encode (or, with cfg.copy_video, stream-copy) each segment, then concat."""
    cfg = cfg or Anya2Config()
    if not segments:
        raise ValueError("no segments to cut")
    if cfg.copy_video:
        return _cut_copy(video, snap_to_keyframes(segments, keyframe_times(video)),
                         output, cfg, on_progress)
    from pipeline import workdir as WD
    # A work-dir override (the desktop app's tmp_anya) gets the scratch
    # segment files too; the app decides whether to keep them as part of the
    # same "keep interim files" choice that covers everything else there, so
    # this function does not clean up after itself in that case.  With no
    # override -- every CLI caller -- this used to leak a system temp
    # directory on every call; it now self-cleans, matching the legacy
    # cutter's behaviour in pipeline.utilities.create_highlights_ffmpeg.
    wd = WD.get_work_dir()
    if wd:
        tmp = os.path.join(wd, "cut_segments")
        os.makedirs(tmp, exist_ok=True)
    else:
        tmp = tempfile.mkdtemp(prefix="anya2_reel_")
    parts = []
    from pipeline.proxy import hwaccel_args
    vf = (["-vf", f"scale=-2:{cfg.scale_height}"] if cfg.scale_height else [])
    for i, s in enumerate(segments):
        cancel.check()
        p = os.path.join(tmp, f"seg_{i:04d}.mp4")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               *hwaccel_args(),
               "-ss", f"{s['start']:.3f}", "-i", video,
               "-t", f"{s['stop'] - s['start']:.3f}", *vf,
               "-c:v", "libx264", "-crf", str(cfg.crf),
               "-preset", cfg.preset, "-pix_fmt", "yuv420p",
               "-vsync", "cfr"]
        cmd += (["-c:a", "aac", "-b:a", "160k"] if cfg.keep_audio else ["-an"])
        cmd.append(p)
        # cancel.run rather than subprocess.run: one 4K segment can take tens
        # of seconds to encode, which would be the whole latency of Cancel.
        if cancel.run(cmd, capture_output=True).returncode == 0:
            parts.append(p)
        _emit(on_progress, 9, f"Cutting segment {i + 1}/{len(segments)}",
              (i + 1) / len(segments))
    if not parts:
        raise RuntimeError("every segment failed to encode")
    # The join re-encodes the audio (video is still copied) so sound cannot
    # walk away from picture over the length of the reel -- see the long note
    # above `concat_cmd` in pipeline.utilities for the measurements.
    from pipeline.utilities import concat_cmd, write_concat_list
    lst = write_concat_list(parts, os.path.join(tmp, "concat.txt"))
    cancel.run(concat_cmd(lst, output, with_audio=cfg.keep_audio,
                          audio_bitrate="160k", quiet=True),
               capture_output=True, check=True)
    if not wd:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return output


def _stream_duration(path: str) -> float:
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip())


def _has_audio(video: str) -> bool:
    return bool(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=index", "-of", "csv=p=0", video],
        capture_output=True, text=True).stdout.strip())


def _cut_copy(video: str, segments: List[dict], output: str,
              cfg: Anya2Config, on_progress=None) -> str:
    """The reel as a STREAM COPY of the source: original resolution, codec and
    bitrate, not one pixel re-encoded.  `segments` must already start on
    keyframes (`snap_to_keyframes`).

    WHY VIDEO AND AUDIO ARE BUILT SEPARATELY.  Copying each segment with its
    audio and joining them was tried first and is wrong: a copied segment's
    audio runs a few ms longer or shorter than its video, the concat demuxer
    offsets every following segment by the longer of the two, and the joined
    video came out with non-monotonic timestamps at the joins and a wrong frame
    rate (29.88 for 29.97 source, clip 21).  Padding each segment's audio to its
    video with -shortest lost 30 frames.  So:

      1. each segment's VIDEO is copied alone, and the video-only segments are
         joined -- every frame kept (5972/5972 on clip 21), source frame rate;
      2. the AUDIO is cut in ONE pass from the source, each range trimmed to the
         MEASURED length of its video segment, so the two tracks share every
         join exactly and cannot drift;
      3. the two are muxed without touching the video.

    Only the audio is encoded (AAC 256k, from the camera's own track).
    """
    from pipeline import workdir as WD
    from pipeline.utilities import write_concat_list
    wd = WD.get_work_dir()
    tmp = (os.path.join(wd, "cut_segments") if wd
           else tempfile.mkdtemp(prefix="anya2_reel_"))
    os.makedirs(tmp, exist_ok=True)
    parts, durs, kept = [], [], []
    for i, s in enumerate(segments):
        cancel.check()
        p = os.path.join(tmp, f"vseg_{i:04d}.mp4")
        # -ss a hair past the keyframe so the input seek lands ON it rather
        # than on the one before through float rounding.
        cancel.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{s['start'] + 0.001:.3f}", "-i", video,
                    "-t", f"{s['stop'] - s['start']:.3f}",
                    "-map", "0:v:0", "-c", "copy", "-an",
                    "-avoid_negative_ts", "make_zero", p],
                   capture_output=True, check=True)
        parts.append(p)
        durs.append(_stream_duration(p))
        kept.append(s)
        _emit(on_progress, 9, f"Cutting segment {i + 1}/{len(segments)}",
              (i + 1) / len(segments))
    lst = write_concat_list(parts, os.path.join(tmp, "vconcat.txt"))
    vid = os.path.join(tmp, "video_only.mp4")
    cancel.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", vid],
               capture_output=True, check=True)

    tag = ["-tag:v", "hvc1"] if _codec(video) == "hevc" else []
    if cfg.keep_audio and _has_audio(video):
        chains = [f"[0:a:0]atrim=start={s['start']:.6f}:duration={d:.6f},"
                  f"asetpts=PTS-STARTPTS[a{i}]" for i, (s, d) in enumerate(zip(kept, durs))]
        graph = ";".join(chains) + ";" + "".join(f"[a{i}]" for i in range(len(kept))) \
            + f"concat=n={len(kept)}:v=0:a=1[a]"
        gpath = os.path.join(tmp, "audio_graph.txt")
        with open(gpath, "w") as fh:          # a long match is hundreds of
            fh.write(graph)                   # segments: too long for argv
        aud = os.path.join(tmp, "audio.m4a")
        cancel.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video, "-filter_complex_script", gpath, "-map", "[a]",
                    "-c:a", "aac", "-b:a", "256k", aud],
                   capture_output=True, check=True)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", vid, "-i", aud, "-map", "0:v:0", "-map", "1:a:0",
               "-c", "copy", *tag, "-movflags", "+faststart", output]
    else:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", vid, "-c", "copy", *tag, "-movflags", "+faststart", output]
    cancel.run(cmd, capture_output=True, check=True)
    if not wd:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return output


def build_reel(video_path, output_path: Optional[str] = None,
               cfg: Optional[Anya2Config] = None,
               on_progress: Optional[Callable] = None,
               dry_run: bool = False) -> Tuple[List[dict], Optional[str]]:
    """Video in, cut reel out.  Signature-compatible with rally_reel.build_reel.

    `video_path` is one path, or several GoPro chapters of one recording that
    are joined into one before anything else runs.
    """
    cfg = cfg or Anya2Config()
    from pipeline import workdir as WD
    # The output default comes from the FIRST INPUT, not the join: a joined
    # file lives in the artifact dir, which under the desktop app is the
    # tmp_anya the run deletes on its way out -- defaulting the reel into it
    # would throw away the only thing the run was for.
    first = video_path if isinstance(video_path, (str, os.PathLike)) else video_path[0]
    src_dir, src_st = _stem(first)
    output_path = output_path or os.path.join(src_dir, f"{src_st}_anya2_reel.mp4")

    _emit(on_progress, 1, "Joining video files")
    video_path = _join(video_path, on_progress)
    _, st = _stem(video_path)
    d = WD.artifact_dir(video_path)

    _emit(on_progress, 2, "Court calibration")
    ensure_court(video_path)

    # Before perceive: `PC.far`'s crop rectangle is sized from this track, and
    # a crop is a fixed ffmpeg rectangle that cannot follow a moving camera.
    _emit(on_progress, 3, "Tracking the camera")
    if _single_decode():
        # Both proxies from one decode of the source, using the band as it
        # stands at calibration.  A camera that never moves (the fixed-mount
        # case this is for) leaves the band unchanged and both later calls are
        # cache hits; one that did move gets its far proxy rebuilt by
        # `PC.far`, exactly as without this.  See proxy.ensure_proxies_once.
        from pipeline import proxy as P
        from pipeline.anya2 import court as C
        try:
            P.ensure_proxies_once(video_path, C.ANALYSIS_SIZE,
                                  PC.far_band(video_path)[0], crf=14)
        except cancel.Cancelled:
            raise
        except Exception as e:                  # pre-warming only
            print(f"[proxies] single-decode build skipped: {e}")
    CAM.estimate(video_path, force=cfg.perceive.force,
                 sample_fps=cfg.perceive.camera_sample_fps or CAM.SAMPLE_FPS,
                 on_progress=lambda fr: _emit(on_progress, 3,
                                              "Tracking the camera", fr))

    _emit(on_progress, 4, "Detecting players (near)")
    near_npz = PC.near(video_path, device=cfg.perceive.device,
                       pose_fps=cfg.perceive.pose_fps, force=cfg.perceive.force)
    _emit(on_progress, 4, "Detecting players (far)", 0.5)
    far_npz = PC.far(video_path, device=cfg.perceive.device,
                     pose_fps=cfg.perceive.pose_fps, force=cfg.perceive.force)

    _emit(on_progress, 5, "Building player tracks")
    TR.build(video_path, near_npz, far_npz, verbose=False)

    _emit(on_progress, 6, "Player motion signals")
    _end_signals(video_path, force=cfg.perceive.force)

    _emit(on_progress, 7, "Detecting serves and point ends")
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

    _emit(on_progress, 8, "Assembling the reel")
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

    _emit(on_progress, 9, "Cutting video", 0.0)
    out = cut(video_path, segments, output_path, cfg, on_progress)
    return segments, out
