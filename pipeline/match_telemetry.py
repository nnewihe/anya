"""
match_telemetry.py
==================
Stage 1 of the dead-time cutter: a single offline perception pass over a full
match video that extracts per-frame telemetry into a JSONL file cached next to
the video.

Unlike AnyaTelemetryProvider (which gates detectors behind the
WAITING/ARMED/ACTIVE state machine to stay real-time-ish), this extractor runs
EVERY detector on EVERY frame — processing is offline, so completeness beats
speed.  Decoupling perception from decision logic means the segmentation stage
(point_segmenter.py) can be re-run and re-tuned in seconds without touching
the video again.

Per-frame record (compact JSONL keys):
    f       frame index (in source video)
    t       timestamp seconds (frame / fps)
    np      near player box  [x1,y1,x2,y2] in 960x540 analysis coords, or null
    npw     near player world feet [wx, wy], or null
    fp      far  player box, or null (held boxes included — see fph)
    fph     1 when fp is a held (stale) box carried through a detection gap
    fpw     far player world feet, SMOOTHED over a short window, or null
    balls   whole-court ball detections [[cx, cy, conf], ...] filtered by the
            active-zone polygon and exclusion zones only (NOT by player boxes —
            the segmenter decides what to exclude)
    toss    toss-ball candidates above the near player [[cx, cy, conf], ...]
            (z-box + exclusion + player-box filtered, like the ARMED detector)
    ftoss   toss-ball candidates above the FAR player, detected on a crop of
            the NATIVE-resolution frame (the 960x540 analysis frame shrinks a
            far-side toss ball below the detection floor); coords are mapped
            back to analysis space.  Gated to the far-baseline band.
    fballs  ALL ball candidates in the native-res far crop (a taller crop
            that also covers the first flight of a struck serve below the
            head line).  Superset of ftoss; feeds the segmenter's ball-trace
            replay so far serves can be trace-confirmed.  Same gating.
    trophy  near-side trophy-pose probability (0.0 if the model is absent)
    stgcn   far-side ST-GCN serve probability (0.0 when not computed)

First line of the file is a meta header: {"meta": {...}}.

Run:
    python -m pipeline.match_telemetry match.mp4 [--force] [--stride N]
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
from ultralytics import YOLO

from .utilities import (Config, _is_in_exclusion_zone, init_court,
                        create_auto_exclusion_zones, load_cached_exclusion_zones,
                        save_cached_exclusion_zones, probe_video)

_MODELS_DIR = Path(__file__).parent / "models"

TELEMETRY_SUFFIX  = "_match_telemetry.jsonl"
TELEMETRY_VERSION = 4   # v2: ftoss; v3: fballs; v4: wide far gate + padded exclusions


@dataclass
class ExtractorConfig:
    analysis_size: Tuple[int, int] = (960, 540)
    player_conf:    float = 0.2    # detection floor (far player needs the low floor)
    near_min_conf:  float = 0.5    # near-player candidates must clear this
    ball_conf:      float = Config.ACTIVE_BALL_CONF
    toss_conf:      float = Config.TOSS_BALL_CONF

    # Far-player robustness: hold the last known box through short detection
    # gaps, and smooth the world position (homography amplifies feet-pixel
    # jitter into feet of world noise at far-court distance).
    far_box_hold_s:      float = 0.7
    far_world_smooth_s:  float = 0.3

    # Far-side ST-GCN gating: only run pose+GCN while the far player is within
    # this signed-distance band of the far baseline (positive = behind it).
    # Deliberately loose: the homography amplifies far-court pixel noise into
    # tens of feet, and per-video calibration offsets of +15..+25 ft are
    # common (measured across the ground-truth folders) — a tight gate makes
    # far serves invisible to stage 2, which self-calibrates its own band.
    stgcn_gate_min_ft: float = -15.0
    stgcn_gate_max_ft: float = 40.0
    stgcn_stride:      int   = 2     # forward passes every N gated frames
    stgcn_reset_gap_s: float = 1.0   # clear the joint window after this long ungated

    far_ball_imgsz: int = 480        # native far crop is ~340x460 — keep near-native

    trophy_stride: int = 2


def telemetry_path_for(video_path: str) -> str:
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_stem}{TELEMETRY_SUFFIX}")


def _define_active_zone(video_path: str, cache_path: str) -> np.ndarray:
    """Load the cached 8-point active-zone polygon or collect it interactively
    (same UI and cache file as AnyaTelemetryProvider, so caches are shared)."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                points = json.load(f)
            print(f"[TELEM] Loaded active zone from {cache_path}")
            return np.array(points, dtype=np.int32)
        except Exception as e:
            print(f"[TELEM] WARN: bad active-zone cache ({e}), re-selecting")

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read frame for active-zone definition.")

    frame = cv2.resize(frame, (960, 540))
    display = frame.copy()
    selected: list = []

    def _cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 8:
            selected.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            if len(selected) > 1:
                cv2.line(display, selected[-2], selected[-1], (0, 255, 0), 2)
            if len(selected) == 8:
                cv2.line(display, selected[-1], selected[0], (0, 255, 0), 2)
            cv2.imshow("Define 8-Sided Active Zone", display)

    cv2.namedWindow("Define 8-Sided Active Zone")
    cv2.setMouseCallback("Define 8-Sided Active Zone", _cb)
    print("Instructions: click 8 points to define the zone, then press 'q'.")
    while True:
        cv2.imshow("Define 8-Sided Active Zone", display)
        key = cv2.waitKey(1) & 0xFF
        if (key == ord("q") or key == 27) and len(selected) == 8:
            break
    cv2.destroyWindow("Define 8-Sided Active Zone")

    pts = np.array(selected, dtype=np.int32)
    with open(cache_path, "w") as f:
        json.dump(pts.tolist(), f)
    return pts


class MatchTelemetryExtractor:
    """One-pass full-telemetry extraction for the offline dead-time cutter."""

    def __init__(self, video_path: str, cfg: Optional[ExtractorConfig] = None,
                 enable_far_serve: bool = True):
        self.video_path = video_path
        self.cfg = cfg or ExtractorConfig()

        info = probe_video(video_path)
        self.fps          = info["fps"]
        self.total_frames = info["frame_count"]

        self.player_model = YOLO(str(_MODELS_DIR / "yolo26n.pt"))
        self.ball_model   = YOLO(str(_MODELS_DIR / "ball_best.pt"))

        # Trophy model is optional — toss detection alone can confirm a near
        # serve (toss weight 0.8 clears the 0.55 threshold by itself).
        self.trophy_model = None
        trophy_path = Config.DEFAULT_NEAR_TROPHY_MODEL_PATH
        if os.path.isfile(trophy_path):
            self.trophy_model = YOLO(trophy_path)
        else:
            print(f"[TELEM] Trophy model not found ({trophy_path}) — "
                  "near serve detection will use ball toss only.")

        # Far-side serve detector is optional (torch / mediapipe /
        # torch_geometric); without it far serves are simply not scored.
        self.far_serve_detector = None
        if enable_far_serve:
            try:
                from .serve_stgcn import ServeSTGCNDetector
                self.far_serve_detector = ServeSTGCNDetector()
                print("[TELEM] Far-side ST-GCN serve detector loaded.")
            except Exception as e:
                print(f"[TELEM] WARN: far-side serve detector unavailable ({e}). "
                      "Far serves will not be scored.")

        # Court geometry / homography / zones — reuses the same disk caches as
        # the existing pipeline, prompting interactively only on first run.
        self.court_vertices, _ = init_court(video_path, analysis_size=self.cfg.analysis_size)
        self.H = self._compute_homography()

        video_dir = os.path.dirname(os.path.abspath(video_path))
        self.active_zone_polygon = _define_active_zone(
            video_path, os.path.join(video_dir, "active_zone_config.json"))

        cached = load_cached_exclusion_zones(video_path)
        if cached is not None:
            self.exclusion_zones = cached
        else:
            print("[TELEM] Scanning video for static exclusion zones...")
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
                print(f"[TELEM] WARN: exclusion-zone scan failed: {e}")
                self.exclusion_zones = []
        print(f"[TELEM] {len(self.exclusion_zones)} exclusion zone(s)")

        # Rolling state
        self._far_world_history: deque = deque()          # (t, wx, wy)
        self._last_far_box: Optional[Tuple[int, int, int, int]] = None
        self._last_far_box_t: float = -1e9
        self._last_trophy: float = 0.0
        self._last_stgcn:  float = 0.0
        self._stgcn_last_gated_t: float = -1e9
        self._frames_since_stgcn: int = 0

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

    def _in_active_zone(self, cx: float, cy: float) -> bool:
        return cv2.pointPolygonTest(
            self.active_zone_polygon, (float(cx), float(cy)), False) >= 0

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
                                    imgsz=Config.PLAYER_IMGSZ)
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
        res = self.ball_model(frame, verbose=False, conf=self.cfg.ball_conf,
                              imgsz=Config.ACTIVE_BALL_IMGSZ)
        out = []
        if res and res[0].boxes:
            for b in res[0].boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                if (self._in_active_zone(cx, cy) and
                        not _is_in_exclusion_zone(cx, cy, self.exclusion_zones)):
                    out.append((round(cx, 1), round(cy, 1), round(float(b.conf[0]), 3)))
        return out

    def _detect_toss(self, frame, near_box) -> List[Tuple[float, float, float]]:
        """Toss-ball candidates in the zone box above the near player,
        mirroring AnyaTelemetryProvider's ARMED-state detector."""
        if near_box is None:
            return []
        nx1, ny1, nx2, ny2 = near_box
        pw, ph = nx2 - nx1, ny2 - ny1
        if pw <= 0 or ph <= 0:
            return []
        fh, fw = frame.shape[:2]

        # Zone box: bottom bisects the player box; 2x width, 1.5x height.
        pcx, pcy = (nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0
        zx1, zx2 = pcx - pw, pcx + pw
        zy2 = pcy
        zy1 = max(0.0, zy2 - ph * 1.5)

        rx1 = max(0,  int(nx1 - pw / 2))
        ry1 = max(0,  int(ny1 - ph))
        rx2 = min(fw, int(nx2 + pw / 2))
        ry2 = min(fh, int(ny1 + ph / 2))
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return []

        res = self.ball_model(roi, verbose=False, conf=self.cfg.toss_conf,
                              imgsz=Config.TOSS_BALL_IMGSZ)
        out = []
        if res and res[0].boxes:
            for b in res[0].boxes:
                cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
                cx = rx1 + (cx1 + cx2) / 2.0
                cy = ry1 + (cy1 + cy2) / 2.0
                in_z    = zx1 <= cx <= zx2 and zy1 <= cy <= zy2
                in_pbox = (nx1 - 15 <= cx <= nx2 + 15 and
                           ny1 - 15 <= cy <= ny2 + 15)
                if (in_z and not in_pbox and
                        not _is_in_exclusion_zone(cx, cy, self.exclusion_zones)):
                    out.append((round(cx, 1), round(cy, 1), round(float(b.conf[0]), 3)))
        return out

    def _detect_far_native_balls(self, orig_frame, frame, far_box, far_world,
                                 now: float):
        """Ball candidates around the FAR player from a NATIVE-resolution
        crop: at ~30 px of player height on the analysis frame a far-side
        ball is 1-2 px, below any detection floor, while the native crop
        keeps it at a detectable size.  Gated to the far-baseline band (same
        band as the ST-GCN scorer) so the extra model call only runs when a
        far serve is possible.  Returned coords are in analysis space.

        Returns (ftoss, fballs):
          ftoss  candidates inside the toss zone above the player (mirrors
                 _detect_toss for the near side),
          fballs every candidate in the crop outside the player box — the
                 crop extends below the box so the first flight of a struck
                 serve is captured for the segmenter's trace replay."""
        if far_box is None or far_world is None:
            return [], []
        dist = far_world[1] - Config.COURT_LENGTH_FT
        if not (self.cfg.stgcn_gate_min_ft <= dist <= self.cfg.stgcn_gate_max_ft):
            return [], []
        fx1, fy1, fx2, fy2 = far_box
        pw, ph = fx2 - fx1, fy2 - fy1
        if pw <= 0 or ph <= 0:
            return [], []
        ah, aw = frame.shape[:2]

        # Toss zone: bottom bisects the player box; 3x width, 2.5x height —
        # relatively wider than the near zone because the far box is small
        # and jittery, and the toss subtends more player-heights up there.
        pcx, pcy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
        zx1, zx2 = pcx - 1.5 * pw, pcx + 1.5 * pw
        zy2 = pcy
        zy1 = max(0.0, zy2 - ph * 2.5)

        # Crop: toss zone plus two player-heights below the box, so the
        # serve's first flight (down-image toward the near court) is seen.
        rx1 = max(0.0, fx1 - 1.5 * pw)
        ry1 = max(0.0, fy1 - ph * 2.5)
        rx2 = min(float(aw), fx2 + 1.5 * pw)
        ry2 = min(float(ah), fy2 + ph * 2.0)
        if rx2 <= rx1 or ry2 <= ry1:
            return [], []

        if orig_frame is not None:
            oh, ow = orig_frame.shape[:2]
            sx, sy = ow / float(aw), oh / float(ah)
        else:                       # native frame unavailable — degrade
            orig_frame, sx, sy = frame, 1.0, 1.0
            oh, ow = ah, aw
        nrx1, nry1 = max(0, int(rx1 * sx)), max(0, int(ry1 * sy))
        nrx2, nry2 = min(ow, int(rx2 * sx)), min(oh, int(ry2 * sy))
        roi = orig_frame[nry1:nry2, nrx1:nrx2]
        if roi.size == 0:
            return [], []

        res = self.ball_model(roi, verbose=False, conf=self.cfg.toss_conf,
                              imgsz=self.cfg.far_ball_imgsz)
        ftoss, fballs = [], []
        if res and res[0].boxes:
            for b in res[0].boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                cx = (nrx1 + (bx1 + bx2) / 2.0) / sx     # back to analysis coords
                cy = (nry1 + (by1 + by2) / 2.0) / sy
                in_pbox = (fx1 - 3 <= cx <= fx2 + 3 and
                           fy1 - 3 <= cy <= fy2 + 3)
                if in_pbox or _is_in_exclusion_zone(cx, cy, self.exclusion_zones):
                    continue
                cand = (round(cx, 1), round(cy, 1), round(float(b.conf[0]), 3))
                fballs.append(cand)
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    ftoss.append(cand)
        return ftoss, fballs

    def _score_trophy(self, frame, near_box, frame_idx: int) -> float:
        if self.trophy_model is None or near_box is None:
            return 0.0
        if frame_idx % self.cfg.trophy_stride != 0:
            return self._last_trophy
        nx1, ny1, nx2, ny2 = near_box
        pw, ph = nx2 - nx1, ny2 - ny1
        fh, fw = frame.shape[:2]
        pad_x, pad_y = int(pw * Config.DEFAULT_TROPHY_PAD), int(ph * Config.DEFAULT_TROPHY_PAD)
        tx1, ty1 = max(0, nx1 - pad_x), max(0, ny1 - pad_y)
        tx2, ty2 = min(fw, nx2 + pad_x), min(fh, ny2 + pad_y)
        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return self._last_trophy
        tr = self.trophy_model(crop, verbose=False, imgsz=Config.TROPHY_IMGSZ)
        if tr and hasattr(tr[0], "probs") and tr[0].probs is not None:
            idx = Config.DEFAULT_NEAR_TROPHY_CLASS_INDEX
            if idx < len(tr[0].probs.data):
                self._last_trophy = float(tr[0].probs.data[idx])
        return self._last_trophy

    def _score_far_serve(self, frame, orig_frame, far_box, far_world,
                         now: float) -> float:
        """ST-GCN far-serve probability. Only runs while the far player is
        within the gate band of the far baseline; the joint window is cleared
        after a long ungated stretch so stale poses don't leak into the score."""
        if self.far_serve_detector is None or far_box is None or far_world is None:
            return 0.0

        dist = far_world[1] - Config.COURT_LENGTH_FT   # positive = behind far baseline
        if not (self.cfg.stgcn_gate_min_ft <= dist <= self.cfg.stgcn_gate_max_ft):
            if now - self._stgcn_last_gated_t > self.cfg.stgcn_reset_gap_s:
                self.far_serve_detector.reset()
                self._last_stgcn = 0.0
            return 0.0

        self._stgcn_last_gated_t = now

        # Crop at native resolution — the far player is too small for a
        # resized frame to preserve the pose detail the model needs.
        fx1, fy1, fx2, fy2 = far_box
        if orig_frame is not None:
            oh, ow = orig_frame.shape[:2]
            sx = ow / float(self.cfg.analysis_size[0])
            sy = oh / float(self.cfg.analysis_size[1])
            nx1, ny1, nx2, ny2 = fx1 * sx, fy1 * sy, fx2 * sx, fy2 * sy
            npw, nph = nx2 - nx1, ny2 - ny1
            pad_x, pad_y = npw * Config.FAR_SERVE_LSTM_PAD, nph * Config.FAR_SERVE_LSTM_PAD
            cx1 = max(0, int(nx1 - pad_x)); cy1 = max(0, int(ny1 - pad_y))
            cx2 = min(ow, int(nx2 + pad_x)); cy2 = min(oh, int(ny2 + pad_y))
            crop = orig_frame[cy1:cy2, cx1:cx2]
        else:
            fh, fw = frame.shape[:2]
            pw, ph = fx2 - fx1, fy2 - fy1
            pad_x, pad_y = pw * Config.FAR_SERVE_LSTM_PAD, ph * Config.FAR_SERVE_LSTM_PAD
            cx1 = max(0, int(fx1 - pad_x)); cy1 = max(0, int(fy1 - pad_y))
            cx2 = min(fw, int(fx2 + pad_x)); cy2 = min(fh, int(fy2 + pad_y))
            crop = frame[cy1:cy2, cx1:cx2]

        self.far_serve_detector.update(crop)
        self._frames_since_stgcn += 1
        if self._frames_since_stgcn >= self.cfg.stgcn_stride:
            self._frames_since_stgcn = 0
            self._last_stgcn = self.far_serve_detector.score()
        return self._last_stgcn

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
                "court_length_ft": Config.COURT_LENGTH_FT,
                "court_width_ft":  Config.COURT_WIDTH_FT,
                "has_trophy":     self.trophy_model is not None,
                "has_far_serve":  self.far_serve_detector is not None,
                "has_far_toss":   True,
                "has_far_balls":  True,
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
                    far_held = False
                    if far_box is not None:
                        self._last_far_box, self._last_far_box_t = far_box, t
                    elif (self._last_far_box is not None and
                          t - self._last_far_box_t <= self.cfg.far_box_hold_s):
                        far_box, far_held = self._last_far_box, True

                    far_world = self._smoothed_far_world(far_world_raw, t)

                    balls  = self._detect_balls(frame)
                    toss   = self._detect_toss(frame, near_box)
                    ftoss, fballs = self._detect_far_native_balls(
                        orig_frame, frame, far_box, far_world, t)
                    trophy = self._score_trophy(frame, near_box, frame_idx)
                    stgcn  = self._score_far_serve(frame, orig_frame, far_box,
                                                   far_world, t)

                    rec = {
                        "f": frame_idx,
                        "t": round(t, 4),
                        "np":  list(near_box) if near_box else None,
                        "npw": ([round(near_world[0], 2), round(near_world[1], 2)]
                                if near_world else None),
                        "fp":  list(far_box) if far_box else None,
                        "fph": 1 if far_held else 0,
                        "fpw": ([round(far_world[0], 2), round(far_world[1], 2)]
                                if far_world else None),
                        "balls":  [list(b) for b in balls],
                        "toss":   [list(b) for b in toss],
                        "ftoss":  [list(b) for b in ftoss],
                        "fballs": [list(b) for b in fballs],
                        "trophy": round(trophy, 3),
                        "stgcn":  round(stgcn, 3),
                    }
                    fh.write(json.dumps(rec) + "\n")
                    n_written += 1

                    if n_written % 300 == 0:
                        pct = 100.0 * frame_idx / max(1, self.total_frames)
                        print(f"[TELEM] frame {frame_idx}/{self.total_frames} "
                              f"({pct:.1f}%)  t={t:.1f}s")
                    if progress_cb is not None and n_written % 30 == 0:
                        progress_cb(frame_idx, self.total_frames)
            finally:
                cap.release()

        os.replace(tmp_path, out_path)
        print(f"[TELEM] Wrote {n_written} records → {out_path}")
        return out_path


def extract_match_telemetry(video_path: str, force: bool = False, stride: int = 1,
                            max_frames: Optional[int] = None,
                            start_frame: int = 0,
                            enable_far_serve: bool = True,
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
            print(f"[TELEM] Using cached telemetry: {out_path}  (--force to re-extract)")
            return out_path
        print(f"[TELEM] Cached telemetry is v{cached_ver}, current is "
              f"v{TELEMETRY_VERSION} — re-extracting.")
    extractor = MatchTelemetryExtractor(video_path, enable_far_serve=enable_far_serve)
    return extractor.extract(stride=stride, max_frames=max_frames,
                             start_frame=start_frame, progress_cb=progress_cb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 of the dead-time cutter: extract full-match telemetry")
    parser.add_argument("video", help="Input tennis match video")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if a telemetry cache exists")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (quick tests; default 1)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after writing N records (quick tests)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Seek to this frame before processing (quick tests)")
    parser.add_argument("--no-far-serve", action="store_true",
                        help="Skip the far-side ST-GCN serve detector")
    args = parser.parse_args()

    extract_match_telemetry(args.video, force=args.force, stride=args.stride,
                            max_frames=args.max_frames,
                            start_frame=args.start_frame,
                            enable_far_serve=not args.no_far_serve)
