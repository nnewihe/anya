"""
anya_telemetry.py
=================
A minimal offline perception pass over a full match video, writing per-frame
telemetry to a JSONL file cached next to the video.

This is the slim sibling of match_telemetry.py.  Where that extractor runs
every serve-detection model (trophy pose, ST-GCN, toss ROI, native far crop)
and pre-filters ball detections, this one records only player geometry and
RAW ball detections, leaving every filtering and scoring decision to the
consumer.  match_telemetry.py is untouched and still feeds point_segmenter.py;
the two write to different cache files and can coexist on the same video.

Per-frame record (compact JSONL keys):
    f          frame index (in source video)
    t          timestamp seconds (frame / fps)
    np         near player box  [x1,y1,x2,y2] in 960x540 analysis coords, or null
    npw        near player world feet [wx, wy], or null
    fp         far  player box, or null (held boxes are carried through short
               detection gaps and are NOT flagged as such)
    fpw        far player world feet, SMOOTHED over a short window, or null
    fpr        far player box [x1,y1,x2,y2] in SOURCE-video pixels, from a
               second person-detection pass run at native resolution inside a
               band around the far baseline (`meta.far_roi`), or null.  Raw:
               no hold-through-gaps, no smoothing.
    fprw       world feet for `fpr`, or null
    fprc       confidence for `fpr`, or null
    all_balls  whole-court ball detections [[cx, cy, conf], ...], UNFILTERED —
               no active-zone test, no exclusion-zone test, no player-box
               test.  Crowd, scoreboard and other static false positives are
               all present; the consumer decides what to drop.

First line of the file is a meta header: {"meta": {...}}.  It carries
`exclusion_zones` — the auto-detected static false-ball boxes as
[x1, y1, x2, y2] in analysis coords — so a consumer can apply the filter
itself against the raw `all_balls` stream.

Run:
    python -m pipeline.anya_telemetry match.mp4 [--force] [--stride N]
"""

import argparse
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

_DEVICE = ('mps' if torch.backends.mps.is_available()
          else 'cuda' if torch.cuda.is_available() else 'cpu')

try:                                        # package import (python -m pipeline.x)
    from .utilities import (Config, init_court, create_auto_exclusion_zones,
                            load_cached_exclusion_zones,
                            save_cached_exclusion_zones, probe_video)
except ImportError:                         # script import (python pipeline/x.py)
    from utilities import (Config, init_court, create_auto_exclusion_zones,
                           load_cached_exclusion_zones,
                           save_cached_exclusion_zones, probe_video)

_MODELS_DIR = Path(__file__).parent / "models"

TELEMETRY_SUFFIX  = "_anya_telemetry.jsonl"
TELEMETRY_VERSION = 2   # v2 adds source_size + the native-resolution far-player
                        # ROI pass (`fpr`/`fprw`); v1 files lack both.


@dataclass
class ExtractorConfig:
    analysis_size: Tuple[int, int] = (960, 540)
    player_conf:    float = 0.2    # detection floor (far player needs the low floor)
    near_min_conf:  float = 0.5    # near-player candidates must clear this
    ball_conf:      float = Config.ACTIVE_BALL_CONF
    ball_imgsz:     int   = 960    # half of Config.ACTIVE_BALL_IMGSZ — the 1920
                                   # pass costs ~2.5x the wall clock for recall
                                   # this extractor's consumers don't need

    # Far-player robustness: hold the last known box through short detection
    # gaps, and smooth the world position (homography amplifies feet-pixel
    # jitter into feet of world noise at far-court distance).
    far_box_hold_s:      float = 0.7
    far_world_smooth_s:  float = 0.3

    # Far-player ROI pass: a second person-detection call on an uncropped,
    # full-resolution band around the far baseline.  At 540p the far player is
    # ~25 px tall and the box jitters by feet of world position; at native
    # resolution inside a tight crop the same player is several hundred px.
    far_roi_height_frac: float = 0.25   # band height as a fraction of frame height
    far_roi_conf:        float = 0.25
    far_roi_imgsz:        int  = 384    # crop is a few hundred px; imgsz=960
                                        # (needed for the full analysis frame)
                                        # would pad/upscale it for no benefit
    far_roi_edge_px:     int   = 4      # boxes whose feet sit this close to the
                                        # band's bottom edge are clipped, not landed


def telemetry_path_for(video_path: str) -> str:
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_stem}{TELEMETRY_SUFFIX}")


class AnyaTelemetryExtractor:
    """One-pass player-geometry + raw-ball extraction."""

    def __init__(self, video_path: str, cfg: Optional[ExtractorConfig] = None):
        self.video_path = video_path
        self.cfg = cfg or ExtractorConfig()

        info = probe_video(video_path)
        self.fps          = info["fps"]
        self.total_frames = info["frame_count"]
        self.source_size  = (info["width"], info["height"])

        self.player_model = YOLO(str(_MODELS_DIR / "yolo26n.pt"))
        self.ball_model   = YOLO(str(_MODELS_DIR / "ball_best.pt"))

        # Court geometry / homography — reuses the same disk caches as the
        # existing pipeline, prompting interactively only on first run.
        self.court_vertices, _ = init_court(video_path, analysis_size=self.cfg.analysis_size)
        self.H = self._compute_homography()

        # Exclusion zones are recorded in the meta header, not applied: this
        # extractor never drops a detection, but the zones are expensive to
        # recompute so they travel with the telemetry.
        cached = load_cached_exclusion_zones(video_path)
        if cached is not None:
            # The zone cache is shared with the full-resolution pipeline, so a
            # cached entry may be in source pixels while every coordinate in
            # this file is in analysis pixels.  Rescale rather than trust it.
            self.exclusion_zones = self._to_analysis_coords(cached)
        else:
            print("[ANYA-TELEM] Scanning video for static exclusion zones...")
            try:
                # padding matters: static false balls jitter ±10px around
                # their zone, and an unpadded (often 1px) zone misses them.
                self.exclusion_zones = create_auto_exclusion_zones(
                    video_path, self.ball_model,
                    num_frames=50, conf=0.04, eps=12, padding=8,
                    ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                    analysis_size=self.cfg.analysis_size,
                )
                save_cached_exclusion_zones(video_path, self.exclusion_zones)
            except Exception as e:
                print(f"[ANYA-TELEM] WARN: exclusion-zone scan failed: {e}")
                self.exclusion_zones = []
        print(f"[ANYA-TELEM] {len(self.exclusion_zones)} exclusion zone(s)")

        self.far_roi = self._far_roi()
        print(f"[ANYA-TELEM] far-player ROI (source px): {list(self.far_roi)}")

        # Rolling state
        self._far_world_history: deque = deque()          # (t, wx, wy)
        self._last_far_box: Optional[Tuple[int, int, int, int]] = None
        self._last_far_box_t: float = -1e9

    # ------------------------------------------------------------------
    def _far_roi(self) -> Tuple[int, int, int, int]:
        """The native-resolution inference band centred on the far baseline.

        Width is the far baseline's own width, height is a fraction of the
        frame height, centred on the baseline.  Returned as [x1, y1, x2, y2]
        in source-video pixels, clamped to the frame.
        """
        _, _, TR, TL = self.court_vertices
        aw, ah = self.cfg.analysis_size
        sw, sh = self.source_size
        sx, sy = sw / float(aw), sh / float(ah)

        x_left  = min(TL[0], TR[0]) * sx
        x_right = max(TL[0], TR[0]) * sx
        y_base  = ((TL[1] + TR[1]) / 2.0) * sy
        half_h  = (sh * self.cfg.far_roi_height_frac) / 2.0

        return (max(0, int(x_left)), max(0, int(y_base - half_h)),
                min(sw, int(x_right)), min(sh, int(y_base + half_h)))

    def _detect_far_player_roi(self, orig_frame):
        """Full-resolution person detection inside the far-baseline band.

        Returns (box_source_px, world_feet, conf) or (None, None, None).
        Detections whose feet sit on the band's bottom edge are rejected: the
        person continues below the crop, so their box bottom is the crop
        boundary rather than their feet, and the world position derived from
        it would be wrong.
        """
        rx1, ry1, rx2, ry2 = self.far_roi
        crop = orig_frame[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return None, None, None

        results = self.player_model(crop, verbose=False,
                                    conf=self.cfg.far_roi_conf,
                                    imgsz=self.cfg.far_roi_imgsz,
                                    device=_DEVICE)
        if not (results and results[0].boxes):
            return None, None, None

        aw, ah = self.cfg.analysis_size
        sw, sh = self.source_size
        inv_x, inv_y = aw / float(sw), ah / float(sh)
        band_h = ry2 - ry1
        fpad = Config.FAR_PLAYER_X_PAD_FT

        best = None
        for b in results[0].boxes:
            if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                continue
            cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()

            # Clipped by the band's bottom edge — feet are not really here.
            if cy2 >= band_h - self.cfg.far_roi_edge_px:
                continue

            x1, y1 = cx1 + rx1, cy1 + ry1
            x2, y2 = cx2 + rx1, cy2 + ry1

            # World position via the analysis-space homography.
            feet_x, feet_y = (x1 + x2) / 2.0, y2
            wx, wy = self._world(feet_x * inv_x, feet_y * inv_y)
            if not (-fpad <= wx <= Config.COURT_WIDTH_FT + fpad):
                continue

            conf = float(b.conf[0])
            if best is None or conf > best[2]:
                best = ((int(x1), int(y1), int(x2), int(y2)),
                        (round(wx, 2), round(wy, 2)), round(conf, 3))

        return best if best else (None, None, None)

    # ------------------------------------------------------------------
    def _to_analysis_coords(self, zones):
        """Rescales source-pixel exclusion zones into analysis coordinates."""
        aw, ah = self.cfg.analysis_size
        if not zones:
            return []
        if max(z[2] for z in zones) <= aw and max(z[3] for z in zones) <= ah:
            return list(zones)
        sx, sy = aw / float(self.source_size[0]), ah / float(self.source_size[1])
        print(f"[ANYA-TELEM] rescaling cached exclusion zones by "
              f"({sx:.3f}, {sy:.3f}) into analysis coords")
        return [[int(z[0] * sx), int(z[1] * sy), int(z[2] * sx), int(z[3] * sy)]
                for z in zones]

    # ------------------------------------------------------------------
    def _compute_homography(self):
        BL, BR, TR, TL = self.court_vertices
        dst = np.array([
            [0, 0], [Config.COURT_WIDTH_FT, 0],
            [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT], [0, Config.COURT_LENGTH_FT],
        ], dtype=np.float32)
        src = np.array([BL, BR, TR, TL], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H

    def _world(self, px: float, py: float) -> Tuple[float, float]:
        pt = cv2.perspectiveTransform(
            np.array([[[px, py]]], dtype=np.float32), self.H)
        return float(pt[0][0][0]), float(pt[0][0][1])

    # ------------------------------------------------------------------
    def _track_players(self, frame):
        """
        One full-frame player-model call classifying BOTH sides.

        Near player: highest-conf-eligible detection whose feet are closest to
        the near baseline (y=0), feet-x within the near-baseline span (+pad),
        conf >= near_min_conf.
        Far player:  detection whose feet are closest to the far baseline
        (y=COURT_LENGTH_FT), feet-x within the sidelines (+pad).

        Returns (near_box, near_world, far_box, far_world_raw).
        """
        results = self.player_model(frame, verbose=False,
                                    conf=self.cfg.player_conf,
                                    imgsz=Config.PLAYER_IMGSZ,
                                    device=_DEVICE)
        if not (results and results[0].boxes):
            return None, None, None, None

        cands = []
        for b in results[0].boxes:
            if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cx = (x1 + x2) / 2.0
            wx, wy = self._world(cx, y2)
            cands.append((x1, y1, x2, y2, wx, wy, float(b.conf[0])))

        if not cands:
            return None, None, None, None

        L   = Config.COURT_LENGTH_FT
        pad = Config.NEAR_PLAYER_X_PAD_FT
        near_cands = [
            c for c in cands
            if (c[6] >= self.cfg.near_min_conf and
                abs(c[5]) < abs(c[5] - L) and
                -pad <= c[4] <= Config.COURT_WIDTH_FT + pad)
        ]
        near_box = near_world = None
        if near_cands:
            near = min(near_cands, key=lambda c: abs(c[5]))
            near_box, near_world = near[:4], (near[4], near[5])

        fpad = Config.FAR_PLAYER_X_PAD_FT
        far_cands = [
            c for c in cands
            if (abs(c[5] - L) < abs(c[5]) and
                -fpad <= c[4] <= Config.COURT_WIDTH_FT + fpad and
                c[:4] != near_box)
        ]
        far_box = far_world = None
        if far_cands:
            far = min(far_cands, key=lambda c: abs(c[5] - L))
            far_box, far_world = far[:4], (far[4], far[5])

        return near_box, near_world, far_box, far_world

    def _smoothed_far_world(self, raw_world, now: float):
        if raw_world is not None:
            self._far_world_history.append((now, raw_world[0], raw_world[1]))
        while (self._far_world_history and
               now - self._far_world_history[0][0] > self.cfg.far_world_smooth_s):
            self._far_world_history.popleft()
        if not self._far_world_history:
            return None
        n = len(self._far_world_history)
        return (sum(s[1] for s in self._far_world_history) / n,
                sum(s[2] for s in self._far_world_history) / n)

    # ------------------------------------------------------------------
    def _detect_balls(self, frame) -> List[Tuple[float, float, float]]:
        """Every ball detection on the whole analysis frame, unfiltered."""
        res = self.ball_model(frame, verbose=False, conf=self.cfg.ball_conf,
                              imgsz=self.cfg.ball_imgsz, device=_DEVICE)
        out = []
        if res and res[0].boxes:
            for b in res[0].boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                out.append((round(cx, 1), round(cy, 1), round(float(b.conf[0]), 3)))
        return out

    # ------------------------------------------------------------------
    def extract(self, out_path: Optional[str] = None, stride: int = 1,
                max_frames: Optional[int] = None, start_frame: int = 0,
                progress_cb=None) -> str:
        """Run the pass and write the JSONL telemetry file. Returns its path.
        start_frame seeks before processing (windowed diagnostics — pair it
        with max_frames and a scratch out_path to probe a slice of the match
        without touching the full-match cache)."""
        out_path = out_path or telemetry_path_for(self.video_path)
        tmp_path = out_path + ".part"

        cap = cv2.VideoCapture(self.video_path)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        n_written = 0
        frame_idx = start_frame - 1

        with open(tmp_path, "w") as fh:
            meta = {
                "version":        TELEMETRY_VERSION,
                "video":          os.path.basename(self.video_path),
                "fps":            self.fps,
                "total_frames":   self.total_frames,
                "stride":         stride,
                "analysis_size":  list(self.cfg.analysis_size),
                "source_size":    list(self.source_size),
                "ball_imgsz":     self.cfg.ball_imgsz,
                "ball_conf":      self.cfg.ball_conf,
                "court_length_ft": Config.COURT_LENGTH_FT,
                "court_width_ft":  Config.COURT_WIDTH_FT,
                "exclusion_zones": [list(z) for z in self.exclusion_zones],
                "far_roi":        list(self.far_roi),
            }
            fh.write(json.dumps({"meta": meta}) + "\n")

            try:
                while cap.isOpened():
                    ok, orig_frame = cap.read()
                    if not ok:
                        break
                    frame_idx += 1
                    if stride > 1 and frame_idx % stride != 0:
                        continue
                    if max_frames is not None and n_written >= max_frames:
                        break

                    t = frame_idx / self.fps
                    frame = cv2.resize(orig_frame, self.cfg.analysis_size,
                                       interpolation=cv2.INTER_LINEAR)

                    near_box, near_world, far_box, far_world_raw = \
                        self._track_players(frame)

                    # Far-box hold through short detection gaps
                    if far_box is not None:
                        self._last_far_box, self._last_far_box_t = far_box, t
                    elif (self._last_far_box is not None and
                          t - self._last_far_box_t <= self.cfg.far_box_hold_s):
                        far_box = self._last_far_box

                    far_world = self._smoothed_far_world(far_world_raw, t)

                    # Independent native-resolution pass; deliberately NOT
                    # smoothed or hold-filled, so consumers see real gaps.
                    roi_box, roi_world, roi_conf = \
                        self._detect_far_player_roi(orig_frame)

                    all_balls = self._detect_balls(frame)

                    rec = {
                        "f": frame_idx,
                        "t": round(t, 4),
                        "np":  list(near_box) if near_box else None,
                        "npw": ([round(near_world[0], 2), round(near_world[1], 2)]
                                if near_world else None),
                        "fp":  list(far_box) if far_box else None,
                        "fpw": ([round(far_world[0], 2), round(far_world[1], 2)]
                                if far_world else None),
                        "fpr":  list(roi_box) if roi_box else None,
                        "fprw": list(roi_world) if roi_world else None,
                        "fprc": roi_conf,
                        "all_balls": [list(b) for b in all_balls],
                    }
                    fh.write(json.dumps(rec) + "\n")
                    n_written += 1

                    if n_written % 300 == 0:
                        pct = 100.0 * frame_idx / max(1, self.total_frames)
                        print(f"[ANYA-TELEM] frame {frame_idx}/{self.total_frames} "
                              f"({pct:.1f}%)  t={t:.1f}s")
                    if progress_cb is not None and n_written % 30 == 0:
                        progress_cb(frame_idx, self.total_frames)
            finally:
                cap.release()

        os.replace(tmp_path, out_path)
        print(f"[ANYA-TELEM] Wrote {n_written} records → {out_path}")
        return out_path


def extract_anya_telemetry(video_path: str, force: bool = False, stride: int = 1,
                           max_frames: Optional[int] = None,
                           start_frame: int = 0,
                           progress_cb=None) -> str:
    """Extract (or reuse cached) telemetry for video_path. Returns JSONL path."""
    out_path = telemetry_path_for(video_path)
    if not force and os.path.isfile(out_path):
        try:
            with open(out_path, "r") as fh:
                cached_ver = int(json.loads(fh.readline()).get("meta", {})
                                 .get("version", 0))
        except Exception:
            cached_ver = 0
        if cached_ver == TELEMETRY_VERSION:
            print(f"[ANYA-TELEM] Using cached telemetry: {out_path}  "
                  "(--force to re-extract)")
            return out_path
        print(f"[ANYA-TELEM] Cached telemetry is v{cached_ver}, current is "
              f"v{TELEMETRY_VERSION} — re-extracting.")
    extractor = AnyaTelemetryExtractor(video_path)
    return extractor.extract(stride=stride, max_frames=max_frames,
                             start_frame=start_frame, progress_cb=progress_cb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract minimal per-frame match telemetry "
                    "(players + raw whole-court ball detections)")
    parser.add_argument("video", help="Input tennis match video")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if a telemetry cache exists")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (quick tests; default 1)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after writing N records (quick tests)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Seek to this frame before processing (quick tests)")
    args = parser.parse_args()

    extract_anya_telemetry(args.video, force=args.force, stride=args.stride,
                           max_frames=args.max_frames,
                           start_frame=args.start_frame)
