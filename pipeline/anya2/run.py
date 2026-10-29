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

import dataclasses
import json
import os
import subprocess
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
    if _single_decode():
        # Both proxies from one decode of the source, using the band as it
        # stands at calibration.  A camera that never moves (the fixed-mount
        # case this is for) leaves the band unchanged and both later calls are
        # cache hits; one that did move gets its far proxy rebuilt by
        # `perceive.far`, exactly as without this.  See proxy.ensure_proxies_once.
        from pipeline import proxy
        from pipeline.anya2 import court
        try:
            proxy.ensure_proxies_once(video, court.ANALYSIS_SIZE,
                                      perceive.far_band(video)[0], crf=14)
        except cancel.Cancelled:
            raise
        except Exception as e:                  # pre-warming only
            print(f"[proxies] single-decode build skipped: {e}")
    camera.estimate(video, force=cfg.perceive.force,
                    sample_fps=cfg.perceive.camera_sample_fps or camera.SAMPLE_FPS,
                    on_progress=lambda f: _emit(on_progress, 3, "Tracking the camera", f))

    _emit(on_progress, 4, "Detecting players (near)")
    near_dets = perceive.near(video, device=cfg.perceive.device,
                              pose_fps=cfg.perceive.pose_fps, force=cfg.perceive.force,
                              imgsz=cfg.perceive.near_imgsz)
    _emit(on_progress, 4, "Detecting players (far)", 0.5)
    far_dets = perceive.far(video, device=cfg.perceive.device,
                            pose_fps=cfg.perceive.pose_fps, force=cfg.perceive.force,
                            imgsz=cfg.perceive.far_imgsz)

    _emit(on_progress, 5, "Building player tracks")
    tracks.build(video, near_dets, far_dets, verbose=False)

    _emit(on_progress, 6, "Player motion signals")
    _compute_end_signals(video, force=cfg.perceive.force)

    _emit(on_progress, 7, "Detecting serves and point ends")
    _run_detectors(video, cfg, artifacts, stem)

    _emit(on_progress, 8, "Assembling the reel")
    reel = orchestrate(video, cfg=_reel_config(cfg), verbose=False)
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


def _reel_config(cfg):
    """The orchestrator's config for this run, without modifying the caller's.

    A disabled detector is disabled for the orchestrator too (its event file
    on disk may be stale), and the orchestrator's own live score uses the same
    smoothing as the point-end detector.
    """
    return dataclasses.replace(cfg.reel,
                               use_near=cfg.near.enabled,
                               use_far=cfg.far.enabled,
                               use_end=cfg.end.enabled,
                               live_smooth_s=cfg.end.smooth_s)


def _or_default(value, default):
    """`value` unless it is None.  (0.0 is a real value, not "unset".)"""
    return default if value is None else value


def _run_detectors(video, cfg, artifacts, stem):
    """Stage 7: each enabled detector writes its `<stem>_anya2_*.json`."""
    if cfg.near.enabled:
        events = near_serve.detect_video(
            video, verbose=False,
            threshold=_or_default(cfg.near.threshold, near_serve.THRESHOLD),
            require_court=cfg.near.require_court,
            lead_s=cfg.near.lead_s, refract_s=cfg.near.refract_s)
        dump_events(events, os.path.join(artifacts, f"{stem}{near_serve.EVENTS_SUFFIX}"))
    if cfg.far.enabled:
        events = far_serve.detect_video(
            video, verbose=False,
            threshold=_or_default(cfg.far.threshold, far_serve.THRESHOLD),
            require_court=cfg.far.require_court,
            lead_s=cfg.far.lead_s, refract_s=cfg.far.refract_s,
            w_still=cfg.far.w_still)
        dump_events(events, os.path.join(artifacts, f"{stem}{far_serve.EVENTS_SUFFIX}"))
    if cfg.end.enabled:
        events = point_end.detect_video(
            video, verbose=False,
            hi=_or_default(cfg.end.live_hi, point_end.LIVE_HI),
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
    """Stage 9: encode (or, with cfg.copy_video, stream-copy) each segment,
    then concatenate."""
    cfg = cfg or Anya2Config()
    if not segments:
        raise ValueError("no segments to cut")
    if cfg.copy_video:
        return _cut_copy(video, snap_to_keyframes(segments, keyframe_times(video)),
                         output, cfg, on_progress)
    # Under the desktop app the segment files go to its work dir (it decides
    # whether to keep them); otherwise to a temp dir removed at the end.
    work_dir = workdir.get_work_dir()
    if work_dir:
        segment_dir = os.path.join(work_dir, "cut_segments")
        os.makedirs(segment_dir, exist_ok=True)
    else:
        segment_dir = tempfile.mkdtemp(prefix="anya2_reel_")

    from pipeline.proxy import hwaccel_args
    scale = (["-vf", f"scale=-2:{cfg.scale_height}"] if cfg.scale_height else [])
    parts = []
    for i, segment in enumerate(segments):
        cancel.check()
        part = os.path.join(segment_dir, f"seg_{i:04d}.mp4")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               *hwaccel_args(),
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


def audio_packets(video: str) -> List[Tuple[float, float]]:
    """(pts, duration) of every packet of the first audio stream, in order."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "packet=pts_time,duration_time", "-of", "csv=p=0", video],
        capture_output=True, text=True, check=True).stdout
    pk = []
    for line in out.splitlines():
        a, _, b = line.partition(",")
        try:
            pk.append((float(a), float(b)))
        except ValueError:
            pass
    return sorted(pk)


def audio_packet_ranges(packets: List[Tuple[float, float]], segments: List[dict],
                        video_durs: List[float]) -> List[Tuple[float, float]]:
    """Whole-packet [inpoint, outpoint) audio ranges, one per video segment.

    A copied AAC stream can only be cut between packets (1024 samples, 21.3 ms
    at 48 kHz), so each range starts on the packet nearest its segment's start,
    and ends where the audio's RUNNING length lands nearest the video's running
    length.  Choosing against the running totals rather than per segment is
    what keeps the error from accumulating: every join is within half a packet
    (~11 ms) of the video, however many points the reel has.
    """
    import bisect
    starts = [p[0] for p in packets]
    ranges, a_total, v_total = [], 0.0, 0.0
    for s, vd in zip(segments, video_durs):
        v_total += vd
        i = bisect.bisect_left(starts, s["start"])
        if i > 0 and (i == len(starts)
                      or s["start"] - starts[i - 1] < starts[i] - s["start"]):
            i -= 1
        j = i
        # Take packets while taking the next one brings the running audio
        # length closer to the running video length.
        while j < len(packets) and abs(a_total + packets[j][1] - v_total) < abs(a_total - v_total):
            a_total += packets[j][1]
            j += 1
        if j == i:
            continue
        end = starts[j] if j < len(starts) else packets[j - 1][0] + packets[j - 1][1]
        ranges.append((starts[i], end))
    return ranges


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
      2. the AUDIO is copied too, packet for packet, with each range sized
         against the MEASURED video lengths (`audio_packet_ranges`), so every
         join is within half an AAC frame (~11 ms) of the video, and the error
         never accumulates;
      3. the two are muxed.

    Nothing is re-encoded: both tracks are the camera's own bits.
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
        # Audio is COPIED too -- the camera's own AAC packets, not re-encoded --
        # chosen per segment by `audio_packet_ranges` so the audio's running
        # length tracks the video's to within half an AAC frame at every join.
        #
        # Cut from an audio-ONLY copy of the track, not the source: in a file
        # with video the concat demuxer's inpoint seeks to the VIDEO keyframe
        # and lets in the audio packets from there (114 extra over 9 joins on
        # clip 21).  Alone, every audio packet is a seek point and the cut is
        # packet-exact.
        a_src = os.path.join(tmp, "source_audio.m4a")
        cancel.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", video, "-map", "0:a:0", "-c", "copy", a_src],
                   capture_output=True, check=True)
        alist = os.path.join(tmp, "aconcat.txt")
        src = a_src.replace("\\", "/").replace("'", "'\\''")
        with open(alist, "w", encoding="utf-8") as fh:
            for a0, a1 in audio_packet_ranges(audio_packets(a_src), kept, durs):
                # Both ends 1 us early, at full precision: the demuxer keeps a
                # packet whose pts rounds to just under the outpoint, and the
                # range's LENGTH -- which places the next range -- is unchanged.
                fh.write(f"file '{src}'\ninpoint {a0 - 1e-6:.9f}\n"
                         f"outpoint {a1 - 1e-6:.9f}\n")
        aud = os.path.join(tmp, "audio.m4a")
        cancel.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", alist,
                    "-map", "0:a:0", "-c", "copy", aud],
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


# ── helpers ──────────────────────────────────────────────────────────────

def _emit(callback, stage, label, fraction=None):
    """Report progress -- and check for Cancel, once per stage announcement."""
    cancel.check()
    if callback:
        try:
            callback(stage, N_STAGES, label, fraction)
        except Exception:
            pass


def _single_decode() -> bool:
    """ANYA_SINGLE_DECODE_PROXIES=1: build both proxies from one source decode.

    Off by default so the desktop app's behaviour is unchanged; the Raspberry
    Pi service turns it on, because there the source decode is the cost.
    """
    return os.environ.get("ANYA_SINGLE_DECODE_PROXIES", "0").strip().lower() in (
        "1", "true", "yes", "on")


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
