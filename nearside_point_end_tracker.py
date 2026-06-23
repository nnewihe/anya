"""
Near-side point-end tracker + video splicer.

Takes the serve-start events from nearside_serve_tracker_toss.py and, for each
serve, traces the ball forward (full-frame Kalman-filtered ball detection) to
find when the point ends — defined as a gap of `no_ball_gap_s` seconds with no
visible ball.  The original video is then spliced into per-point clips and saved
as a single concatenated output video.

Pipeline
--------
  1. INGEST  — accept list[NearServeEvent] (importable) or load from JSON
               (standalone).  save_events / load_events helpers included.

  2. SCAN    — single full-frame ball-detection pass over the union of all
               scan windows [first_contact, last_contact + max_point_frames].
               Detections stored per-frame; Kalman tracker run over the same
               window.  Exclusion zones passed in or recomputed automatically.

  3. ENDDETECT — for each serve, walk pre-computed detections forward from
               contact_frame.  When consecutive no-ball frames exceed the
               threshold, end_frame = last_ball_frame + 1.
               If the next serve arrives before that → end_frame = None.

  4. SPLICE  — copy [toss_start_frame − lead_in … end_frame + lead_out] from
               the original video into a single concatenated output file.

Reuses
------
  KalmanBallTracker, Detection, Track, find_static_exclusion_zones,
  apply_exclusion_zones          (farside_serve_detector_v2)

  NearConfig, NearServeEvent, CourtGeometry, init_court_geometry,
  _DEFAULT_BALL_MODEL            (nearside_serve_tracker_toss)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from farside_serve_detector_v2 import (
    Detection,
    KalmanBallTracker,
    find_static_exclusion_zones,
    apply_exclusion_zones,
)
from nearside_serve_tracker_toss import (
    NearConfig,
    NearServeEvent,
    CourtGeometry,
    init_court_geometry,
    save_events,        # single source of truth — defined in serve tracker
    load_events,        #   so both scripts share the same JSON schema
    _DEFAULT_BALL_MODEL,
)

try:
    import cv2 as _cv2
    from ultralytics import YOLO as _YOLO
except ImportError:  # pragma: no cover
    _cv2 = None      # type: ignore
    _YOLO = None     # type: ignore


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PointTrackerConfig(NearConfig):
    """Extends NearConfig with point-end detection and splicing parameters.

    All Kalman tracker parameters, analysis resolution, and player-detection
    settings are inherited from NearConfig → Config.
    """
    # --- End detection ---
    no_ball_gap_s: float = 2.0        # consecutive no-detection frames = point over
    max_point_s: float = 30.0         # safety cap on per-point scan window

    # --- Clip padding ---
    lead_in_s: float = 0.5            # seconds before toss_start_frame in clip
    lead_out_s: float = 1.0           # seconds after end_frame in clip

    # --- Full-frame ball pass (different from toss-crop pass in near tracker) ---
    full_ball_imgsz: int = 1280       # full resolution: don't miss far-court balls
    full_ball_conf: float = 0.20      # slightly lower than toss conf

    # --- Output ---
    output_suffix: str = "_points"    # appended to video stem before extension
    fourcc: str = "mp4v"


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class PointSegment:
    """One active point: from serve to end-of-rally.

    clip_start / clip_end are the actual frame indices used for the output
    video (padded by lead_in / lead_out).  end_frame is the first frame of the
    dead-ball gap (or None if the ball trace flowed into the next serve or the
    video ended without a clean gap).
    """
    serve_idx: int                    # 1-based
    contact_frame: int                # anchor from NearServeEvent
    toss_start_frame: int             # clip lead-in anchor
    end_frame: Optional[int]          # first frame of dead-ball gap; None = unclear
    clip_start: int                   # toss_start_frame − lead_in (≥0)
    clip_end: int                     # end_frame + lead_out (or fallback)
    fps: float
    notes: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> Optional[float]:
        """Active-rally duration from contact to end_frame (seconds)."""
        if self.end_frame is None:
            return None
        return (self.end_frame - self.contact_frame) / self.fps

    @property
    def clip_duration_s(self) -> float:
        return (self.clip_end - self.clip_start) / self.fps

    def _fmt(self, frame: int) -> str:
        s = frame / self.fps
        return f"{int(s // 60):02d}:{s % 60:06.3f}"

    def summary(self) -> str:
        dur = f"{self.duration_s:.2f}s" if self.duration_s is not None else "n/a"
        end_str = self._fmt(self.end_frame) if self.end_frame is not None else "NONE"
        return (f"#{self.serve_idx:>3}  contact={self._fmt(self.contact_frame)}  "
                f"end={end_str}  dur={dur}  "
                f"clip=[{self._fmt(self.clip_start)}, {self._fmt(self.clip_end)}]  "
                f"({self.clip_duration_s:.1f}s clip)  {self.notes}")



# --------------------------------------------------------------------------- #
# Stage 2 — Full-frame ball scan (single pass)
# --------------------------------------------------------------------------- #
def _full_frame_ball_pass(
    video_path: str,
    ball_model,
    cfg: PointTrackerConfig,
    scan_from: int,
    scan_to: int,
    excl_zones: list = None,
    start_frame: int = 0,
) -> dict[int, list[Detection]]:
    """Full-frame ball detection over [scan_from, scan_to) absolute frame indices.

    Returns a dict mapping absolute frame index → list[Detection].
    scan_from / scan_to are absolute (not relative to start_frame).
    """
    _zones = excl_zones or []

    def _not_excluded(cx, cy):
        return not any(x1 <= cx <= x2 and y1 <= cy <= y2
                       for (x1, y1, x2, y2) in _zones)

    W, H = cfg.frame_w, cfg.frame_h
    cap = _cv2.VideoCapture(video_path)
    total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))

    abs_scan_from = start_frame + scan_from
    abs_scan_to   = min(total, start_frame + scan_to)
    n_scan = abs_scan_to - abs_scan_from

    cap.set(_cv2.CAP_PROP_POS_FRAMES, abs_scan_from)

    stream: dict[int, list[Detection]] = {}
    fi = scan_from   # relative frame index (matches NearServeEvent frame numbering)

    print(f"[SCAN] Full-frame ball pass: frames {scan_from}–{scan_to - 1}  "
          f"({n_scan} frames, conf={cfg.full_ball_conf}) …")
    try:
        scanned = 0
        while cap.isOpened() and fi < scan_to:
            ret, orig = cap.read()
            if not ret:
                break
            frame = _cv2.resize(orig, (W, H), interpolation=_cv2.INTER_LINEAR)
            res = ball_model(frame, conf=cfg.full_ball_conf,
                             imgsz=cfg.full_ball_imgsz, verbose=False)
            dets = []
            if res and res[0].boxes:
                for b in res[0].boxes:
                    bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0
                    if _not_excluded(cx, cy):
                        dets.append(Detection(fi, cx, cy, bx2 - bx1, by2 - by1,
                                              float(b.conf[0])))
            stream[fi] = dets
            fi += 1
            scanned += 1
            if scanned % 300 == 0:
                pct = 100.0 * scanned / max(1, n_scan)
                total_dets = sum(len(v) for v in stream.values())
                print(f"[SCAN]   frame {fi:>6} ({pct:.0f}%)  "
                      f"total ball dets={total_dets}")
    finally:
        cap.release()

    total_dets = sum(len(v) for v in stream.values())
    print(f"[SCAN] Done: {scanned} frames scanned, {total_dets} ball detections.")
    return stream


# --------------------------------------------------------------------------- #
# Stage 2b — End detection
# --------------------------------------------------------------------------- #
def find_point_ends(
    serves: list[NearServeEvent],
    ball_stream: dict[int, list[Detection]],
    total_frames: int,
    cfg: PointTrackerConfig,
) -> list[PointSegment]:
    """Detect the end frame for each serve by scanning pre-computed ball
    detections forward from contact_frame.

    End rule: first frame where (current_frame − last_ball_frame) ≥ gap_frames.
    Special cases:
      • Next serve's contact_frame arrives first → end_frame = None
      • Last serve reaches end of scan window without gap → end_frame = None
    """
    serves_sorted = sorted(serves, key=lambda s: s.contact_frame)
    gap_frames = int(round(cfg.no_ball_gap_s * cfg.fps))
    max_point_frames = int(round(cfg.max_point_s * cfg.fps))
    lead_in_frames  = int(round(cfg.lead_in_s  * cfg.fps))
    lead_out_frames = int(round(cfg.lead_out_s * cfg.fps))

    segments: list[PointSegment] = []

    for i, ev in enumerate(serves_sorted):
        next_ev = serves_sorted[i + 1] if i + 1 < len(serves_sorted) else None
        # Scan up to the earlier of: next serve contact, or max_point cap.
        search_limit = ev.contact_frame + max_point_frames
        if next_ev is not None:
            search_limit = min(search_limit, next_ev.contact_frame)
        search_limit = min(search_limit, total_frames)

        last_ball_f = ev.contact_frame  # last frame with a ball detection
        # Seed: if there's no detection at contact itself, we still start here.
        if ball_stream.get(ev.contact_frame):
            last_ball_f = ev.contact_frame

        end_frame: Optional[int] = None
        for f in range(ev.contact_frame, search_limit):
            dets = ball_stream.get(f, [])
            if dets:
                last_ball_f = f
            else:
                gap = f - last_ball_f
                if gap >= gap_frames:
                    end_frame = last_ball_f + 1   # first frame of dead-ball gap
                    break

        # Determine clip bounds ----------------------------------------------- #
        clip_start = max(0, ev.toss_start_frame - lead_in_frames)

        if end_frame is not None:
            clip_end = min(total_frames, end_frame + lead_out_frames)
            notes = {"end_found": True, "last_ball_f": last_ball_f}
        elif next_ev is not None:
            # Ball trace flowed into next serve — clip until next toss start.
            clip_end = max(clip_start + 1,
                           min(total_frames, next_ev.toss_start_frame - 1))
            notes = {"end_found": False, "reason": "flowed_into_next_serve"}
        else:
            # Last serve, no clean gap before scan limit.
            clip_end = min(total_frames,
                           ev.contact_frame + max_point_frames + lead_out_frames)
            notes = {"end_found": False, "reason": "end_of_scan"}

        segments.append(PointSegment(
            serve_idx=i + 1,
            contact_frame=ev.contact_frame,
            toss_start_frame=ev.toss_start_frame,
            end_frame=end_frame,
            clip_start=clip_start,
            clip_end=clip_end,
            fps=cfg.fps,
            notes=notes,
        ))

    return segments


# --------------------------------------------------------------------------- #
# Stage 3 — Video splicing
# --------------------------------------------------------------------------- #
def splice_video(
    video_path: str,
    segments: list[PointSegment],
    cfg: PointTrackerConfig,
    output_path: str,
    start_frame: int = 0,
    draw_overlay: bool = True,
) -> None:
    """Concatenate the clip ranges from segments into a single output video.

    Each clip is stamped with a small HUD showing the point number, timestamp,
    and whether a clean end was detected (if draw_overlay=True).
    """
    if not segments:
        print("[SPLICE] No segments to splice.")
        return

    W, H = cfg.frame_w, cfg.frame_h
    fourcc = _cv2.VideoWriter_fourcc(*cfg.fourcc)
    writer = _cv2.VideoWriter(output_path, fourcc, cfg.fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"[SPLICE] Cannot open VideoWriter → {output_path}")

    total_clip_frames = sum(max(0, s.clip_end - s.clip_start) for s in segments)
    print(f"[SPLICE] Writing {len(segments)} clips "
          f"({total_clip_frames} frames total) → {output_path} …")

    cap = _cv2.VideoCapture(video_path)
    total_video_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))

    # Sort by clip_start so we seek forward only (efficient).
    seg_order = sorted(segments, key=lambda s: s.clip_start)
    written = 0

    for seg in seg_order:
        abs_start = start_frame + seg.clip_start
        abs_end   = start_frame + seg.clip_end
        abs_start = max(0, min(total_video_frames - 1, abs_start))
        abs_end   = max(abs_start, min(total_video_frames, abs_end))

        cap.set(_cv2.CAP_PROP_POS_FRAMES, abs_start)
        fi = seg.clip_start   # relative frame index for HUD timestamps

        for _ in range(abs_end - abs_start):
            ret, orig = cap.read()
            if not ret:
                break
            frame = _cv2.resize(orig, (W, H), interpolation=_cv2.INTER_LINEAR)

            if draw_overlay:
                _draw_splice_hud(frame, seg, fi, cfg)

            writer.write(frame)
            fi += 1
            written += 1

        print(f"[SPLICE]   point #{seg.serve_idx:>3}  "
              f"frames {seg.clip_start}–{seg.clip_end - 1}  "
              f"({seg.clip_end - seg.clip_start} frames)  "
              f"end={'OK' if seg.end_frame is not None else 'NONE'}")

    cap.release()
    writer.release()
    print(f"[SPLICE] Done — {written} frames written → {output_path}")


def _draw_splice_hud(frame, seg: PointSegment, fi: int, cfg: PointTrackerConfig):
    """Burn a small HUD into the top-left corner of a spliced frame."""
    t = fi / cfg.fps
    mm = int(t // 60); ss = t % 60
    end_ok = seg.notes.get("end_found", seg.end_frame is not None)
    lines = [
        f"POINT #{seg.serve_idx}",
        f"{mm:02d}:{ss:05.2f}",
        f"end: {'found' if end_ok else 'unclear'}",
    ]
    y = 18
    for txt in lines:
        (tw, th), bl = _cv2.getTextSize(txt, _cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        _cv2.rectangle(frame, (6, y - th - 2), (10 + tw, y + bl + 1), (0, 0, 0), -1)
        _cv2.putText(frame, txt, (8, y), _cv2.FONT_HERSHEY_SIMPLEX,
                     0.45, (230, 230, 230), 1, _cv2.LINE_AA)
        y += th + 8


# --------------------------------------------------------------------------- #
# Main orchestrator
# --------------------------------------------------------------------------- #
def run_point_tracker(
    video_path: str,
    serves: list[NearServeEvent],
    cfg: Optional[PointTrackerConfig] = None,
    ball_model_path: str = _DEFAULT_BALL_MODEL,
    output_path: Optional[str] = None,
    excl_zones: Optional[list] = None,
    recalibrate: bool = False,
    start_frame: int = 0,
    draw_overlay: bool = True,
) -> list[PointSegment]:
    """End-to-end point tracker: scan → end detection → splice.

    Parameters
    ----------
    video_path      : path to the original input video.
    serves          : list of NearServeEvent from nearside_serve_tracker_toss.
    cfg             : PointTrackerConfig (created with defaults if None).
    ball_model_path : path to the YOLO ball model weights.
    output_path     : where to save the spliced video; defaults to
                      <video_stem>_points.mp4 next to the input.
    excl_zones      : pre-computed static exclusion zones [(x1,y1,x2,y2), …].
                      Recomputed automatically if None.
    recalibrate     : force re-run of court-corner calibration.
    start_frame     : offset if the serve events were detected starting from a
                      non-zero frame (must match the value used in the serve tracker).
    draw_overlay    : burn point-number HUD into spliced frames.

    Returns
    -------
    list[PointSegment] — one per serve, with clip bounds and end-frame info.
    """
    if not _cv2 or not _YOLO:
        raise RuntimeError(
            "opencv-python and ultralytics are required.\n"
            "  pip install opencv-python ultralytics")

    if cfg is None:
        cfg = PointTrackerConfig()
    cfg.frame_w, cfg.frame_h = cfg.analysis_w, cfg.analysis_h

    if not serves:
        print("[PT] No serve events supplied — nothing to do.")
        return []

    # 1. FPS probe + total frame count
    cap = _cv2.VideoCapture(video_path)
    raw_fps = cap.get(_cv2.CAP_PROP_FPS)
    total_frames_abs = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if 0 < raw_fps < 300:
        cfg.fps = raw_fps
    total_frames = total_frames_abs - start_frame   # relative to start_frame
    print(f"[PT] Video: {total_frames_abs} frames @ {cfg.fps:.3f} fps  "
          f"(relative window: {total_frames} frames)")

    # 2. Ball model
    print(f"[PT] Loading ball model: {ball_model_path}")
    ball_model = _YOLO(ball_model_path)

    # 3. Static exclusion zones (recompute if not supplied)
    if excl_zones is None:
        print("[PT] Scanning for static exclusion zones …")
        try:
            from utilities import create_auto_exclusion_zones
            excl_zones = create_auto_exclusion_zones(
                video_path, ball_model,
                num_frames=20, conf=0.05,
                analysis_size=(cfg.frame_w, cfg.frame_h),
            )
        except Exception as _e:
            print(f"[PT] Exclusion zone scan skipped: {_e}")
            excl_zones = []
        print(f"[PT] Static exclusion zones: {len(excl_zones)}")
        for z in excl_zones:
            print(f"    zone  x=[{z[0]},{z[2]}]  y=[{z[1]},{z[3]}]")

    # 4. Determine scan window — union of all serve windows
    serves_sorted = sorted(serves, key=lambda s: s.contact_frame)
    max_point_frames = int(round(cfg.max_point_s * cfg.fps))
    scan_from = max(0, serves_sorted[0].contact_frame)
    scan_to   = min(total_frames,
                    serves_sorted[-1].contact_frame + max_point_frames)
    print(f"[PT] Scan window: relative frames {scan_from}–{scan_to}")

    # 5. Full-frame ball detection pass
    ball_stream = _full_frame_ball_pass(
        video_path, ball_model, cfg,
        scan_from=scan_from, scan_to=scan_to,
        excl_zones=excl_zones, start_frame=start_frame)

    # 6. Kalman tracking (for future visualisation / debugging)
    # Build a flat stream list covering [0, scan_to) for the tracker.
    flat_stream = [ball_stream.get(f, []) for f in range(scan_to)]
    zones_kalman = find_static_exclusion_zones(flat_stream, cfg)
    clean_stream = apply_exclusion_zones(flat_stream, zones_kalman, cfg)
    tracks = KalmanBallTracker(cfg).run(clean_stream)
    print(f"[PT] Kalman tracks: {len(tracks)}")

    # 7. End detection
    segments = find_point_ends(serves_sorted, ball_stream, total_frames, cfg)

    # 8. Print summary
    n_found = sum(1 for s in segments if s.end_frame is not None)
    print(f"\n{'='*70}")
    print(f"  POINTS: {len(segments)}  (end found: {n_found}  unclear: {len(segments)-n_found})")
    print(f"{'='*70}")
    for seg in segments:
        print(f"  {seg.summary()}")
    print(f"{'='*70}\n")

    # 9. Splice video
    if output_path is None:
        stem, ext = os.path.splitext(video_path)
        output_path = stem + cfg.output_suffix + (ext or ".mp4")

    splice_video(video_path, segments, cfg, output_path,
                 start_frame=start_frame, draw_overlay=draw_overlay)

    return segments


# --------------------------------------------------------------------------- #
# Synthetic offline demo  (no video / no models needed)
# --------------------------------------------------------------------------- #
def _run_synthetic_demo():
    """Verify end-detection logic with a fabricated ball stream and three
    synthetic serves.  No video file or YOLO model required."""
    cfg = PointTrackerConfig()
    cfg.fps = 30.0
    cfg.frame_w, cfg.frame_h = cfg.analysis_w, cfg.analysis_h

    rng = np.random.default_rng(0)
    fps = cfg.fps
    gap_f = int(cfg.no_ball_gap_s * fps)      # 60 frames

    def serve(contact, toss_start):
        return NearServeEvent(toss_start_frame=toss_start, apex_frame=toss_start + 8,
                              contact_frame=contact, fps=fps, track_id=0, score=0.9)

    # Serve A: clean end at frame 180 (ball last seen at 179, gap starts 180)
    s_a = serve(contact=30,  toss_start=10)
    # Serve B: ball flows directly into next serve (no gap before serve C)
    s_b = serve(contact=240, toss_start=220)
    # Serve C: reaches end of video with no gap
    s_c = serve(contact=450, toss_start=430)
    total = 600   # 20 s

    # Synthetic ball_stream: A has ball 30–179 (clean end), B has ball 240–449
    # (flows into C), C has ball 450–550 (then nothing, but scan ends before gap)
    ball_stream: dict[int, list] = {}
    for f in range(30, 180):
        ball_stream[f] = [Detection(f, 640, 360, 10, 10, 0.8)]
    for f in range(240, 450):
        ball_stream[f] = [Detection(f, 640, 360, 10, 10, 0.8)]
    for f in range(450, 551):
        ball_stream[f] = [Detection(f, 640, 360, 10, 10, 0.8)]

    segs = find_point_ends([s_a, s_b, s_c], ball_stream, total, cfg)

    print("\n[DEMO] Synthetic results:")
    for seg in segs:
        print(f"  {seg.summary()}")

    # Assertions
    assert segs[0].end_frame == 180, f"Expected end=180, got {segs[0].end_frame}"
    assert segs[1].end_frame is None, f"Expected None, got {segs[1].end_frame}"
    assert segs[2].end_frame is None, f"Expected None, got {segs[2].end_frame}"
    print("[DEMO] All assertions passed ✓")


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main():
    _run_synthetic_demo()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Near-side point-end tracker + video splicer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Accepts serve events as a JSON file (produced by save_events) or by importing
run_point_tracker() directly with a list[NearServeEvent].

Examples:
  # Synthetic demo (no video / model needed):
  python nearside_point_end_tracker.py

  # Real video + pre-saved serve events:
  python nearside_point_end_tracker.py video.mp4 --events serves.json

  # Custom output path and gap threshold:
  python nearside_point_end_tracker.py video.mp4 --events serves.json \\
      --output points.mp4 --gap 2.5

  # No HUD overlay on output clips:
  python nearside_point_end_tracker.py video.mp4 --events serves.json --no-overlay
""")
    p.add_argument("video", nargs="?", default=None,
                   help="Path to the original input video.")
    p.add_argument("--events", default=None, metavar="SERVES.json",
                   help="JSON file of NearServeEvents (from save_events).")
    p.add_argument("--ball-model", default=_DEFAULT_BALL_MODEL)
    p.add_argument("--output", default=None, metavar="OUT.mp4")
    p.add_argument("--gap", type=float, default=2.0, metavar="SECONDS",
                   help="Dead-ball gap threshold in seconds (default 2.0).")
    p.add_argument("--lead-in", type=float, default=0.5, metavar="SECONDS")
    p.add_argument("--lead-out", type=float, default=1.0, metavar="SECONDS")
    p.add_argument("--max-point", type=float, default=30.0, metavar="SECONDS",
                   help="Max scan window per point (safety cap, default 30 s).")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--conf", type=float, default=None,
                   help="Override ball detection confidence threshold.")
    p.add_argument("--no-overlay", action="store_true",
                   help="Suppress point-number HUD on spliced clips.")
    p.add_argument("--recalibrate", action="store_true",
                   help="Force court re-calibration (passed to serve tracker if used).")
    args = p.parse_args()

    if args.video is None:
        main()
    else:
        if args.events is None:
            p.error("--events SERVES.json is required when a video is supplied.\n"
                    "  Generate it with save_events() after running the near-side "
                    "serve tracker.")
        serves = load_events(args.events)
        _cfg = PointTrackerConfig()
        _cfg.no_ball_gap_s = args.gap
        _cfg.lead_in_s     = args.lead_in
        _cfg.lead_out_s    = args.lead_out
        _cfg.max_point_s   = args.max_point
        if args.conf is not None:
            _cfg.full_ball_conf = args.conf

        run_point_tracker(
            video_path=args.video,
            serves=serves,
            cfg=_cfg,
            ball_model_path=args.ball_model,
            output_path=args.output,
            recalibrate=args.recalibrate,
            start_frame=args.start_frame,
            draw_overlay=not args.no_overlay,
        )
