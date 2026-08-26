"""
anya_near_telemetry.py
======================
The near-serve-only sibling of anya_telemetry.py: same near-player geometry and
toss-ROI ball detections, at roughly a quarter of the compute.

anya_telemetry.py is a general-purpose pass — it feeds far-serve, ball-quiet
dead time and near-serve alike, so it runs three models on every frame at full
resolution.  This module serves exactly one consumer (anya_near_serve.py) and
spends nothing on what that consumer never reads.  It writes a DIFFERENT file
(`_anya_near_telemetry.jsonl`), so the full telemetry cache stays valid and the
two can be compared on the same video.

Where the savings come from, measured on Data/21 (3840x2160, 12594 frames):

  1. 540p proxy.  Decode of the 4K source costs ~6.7 ms/frame and dominated
     everything else once inference came down.  ~4.1 ms of that is frame
     reconstruction that cannot be skipped even for frames we throw away
     (H.264 frames are differences against their predecessors), so decoding
     less often barely helps.  Decoding something *smaller* does.  The proxy
     is transcoded once and reused; nothing here needs source resolution,
     because the near player is large and every coordinate is already in
     960x540 analysis space.

  2. Player at 5 fps, imgsz 480.  Dwell is time-based and indifferent to
     sample rate; the aspect-ratio range J uses a 1 s window, which holds 5
     samples at 5 fps.  Neither cue needs 30 fps.

  3. Ball only in the toss ROI, only while the player is ready.  The toss cue
     looks at one small box above the near player and nothing else, so the
     full-frame 960px ball pass is almost entirely wasted on it.  Ball
     inference still runs at the full 30 fps — the toss is a fast, short
     event — but only on a crop, and only inside ready windows.

The ready gate opens when the near player enters the ready zone and stays open
for `ready_hold_s` AFTER they leave it.  That trailing hold is load-bearing:
the player leaves the ready zone precisely when they begin serving, so a gate
that closed on zone exit would switch the ball model off at the toss.

Per-frame record (JSONL, one line each, meta header first):
    f      frame index in the source video
    t      timestamp seconds
    pn     true if this frame carries a FRESH player detection (5 fps);
           false on the 30 fps in-between frames, which repeat the held box
    np     near player box [x1,y1,x2,y2] in 960x540 analysis coords, or null
           (held between player samples so the toss ROI stays positioned)
    npw    near player world feet [wx, wy], or null
    balls  ball detections inside the toss-ROI crop, [[cx, cy, conf], ...] in
           analysis coords; absent outside ready windows

Run:
    python -m pipeline.anya_near_telemetry match.mp4 [--force]
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

_DEVICE = ('mps' if torch.backends.mps.is_available()
           else 'cuda' if torch.cuda.is_available() else 'cpu')

try:                                        # package import (python -m pipeline.x)
    from .utilities import (Config, init_court, create_auto_exclusion_zones,
                            load_cached_exclusion_zones,
                            save_cached_exclusion_zones, probe_video,
                            assert_decode_complete, open_video)
    from .proxy import PROXY_SUFFIX, proxy_path_for as _proxy_path_for
    from .proxy import ensure_proxy as _ensure_proxy
except ImportError:                         # script import (python pipeline/x.py)
    from utilities import (Config, init_court, create_auto_exclusion_zones,
                           load_cached_exclusion_zones,
                           save_cached_exclusion_zones, probe_video,
                           assert_decode_complete, open_video)
    from proxy import PROXY_SUFFIX, proxy_path_for as _proxy_path_for
    from proxy import ensure_proxy as _ensure_proxy
try:
    from . import workdir as _workdir
except ImportError:
    import workdir as _workdir


_MODELS_DIR = Path(__file__).parent / "models"

NEAR_TELEMETRY_SUFFIX = "_anya_near_telemetry.jsonl"
NEAR_TELEMETRY_VERSION = 1


@dataclass
class NearExtractorConfig:
    analysis_size: Tuple[int, int] = (960, 540)

    # --- proxy
    use_proxy:   bool = True
    # CRF 14, not the 20 this started at.  A tennis ball mid-toss is small,
    # fast and low-contrast — precisely what x264 spends its bit budget last
    # on — and at CRF 20 the encoder was deleting it.  Measured on Data/38
    # (identical 960x540 resolution either way, so this is the RE-ENCODE, not
    # the downscale): surviving toss-ROI detections per serve went
    #   CRF 20 -> [17, 4, 4, 12, 0, 2, 0, 1]      (4 serves with no toss)
    #   CRF 14 -> [25, 32, 25, 21, 10, 10, 14, 24]
    #   source -> [28, 34, 24, 18, 19, 12, 34, 21]
    # Native source resolution was no better than 960x540-from-source, so the
    # fix is quality, not pixels.  Costs ~30 MB/min of proxy on disk.
    proxy_crf:   int  = 14
    proxy_preset: str = "veryfast"

    # --- player pass (5 fps)
    player_fps:     float = 5.0
    player_imgsz:   int   = 480
    player_conf:    float = 0.2
    near_min_conf:  float = 0.5
    box_hold_s:     float = 0.45   # carry the box between 5 fps samples (0.2 s
                                   # apart) plus one tolerated dropout

    # --- ready gate (mirrors NearServeConfig's ready zone)
    zone_y_min_ft: float = -3.5
    zone_y_max_ft: float =  0.5
    zone_x_pad_ft: float =  3.0
    ready_hold_s:  float =  2.0    # keep the ball model on this long after the
                                   # player leaves the zone — they leave it BY
                                   # serving, which is the thing we came for

    # --- ball pass (30 fps, toss-ROI crop only)
    ball_conf:  float = Config.ACTIVE_BALL_CONF
    ball_imgsz: int   = 416
    # The toss ROI is cropped TIGHT and then resized up to (crop_w, crop_h),
    # rather than being read out of a fixed window at native analysis scale.
    # Both give the model the same 288x416 tensor, so it costs nothing — but
    # a tight crop fills that tensor with the ball instead of with court, and
    # the ball is magnified ~3x.  Measured on Data/38, same CRF-14 proxy, same
    # frames, same imgsz:
    #     tight ROI crop, upscaled  ->  7454 in-ROI detections, top conf 0.75
    #     fixed 288x416 window      ->   993 in-ROI detections, top conf 0.60
    # The ROI's aspect is fixed at 2:3 by construction (width is 2/3 of the
    # player height, height is one player height), so a single canonical size
    # keeps every batch shape-uniform without distorting the ball.
    crop_w: int = 288
    crop_h: int = 416
    roi_pad_frac: float = 0.10   # context around the ROI; the scorer still
                                 # applies the exact ROI test itself

    batch_size: int = 16


def near_telemetry_path_for(video_path: str) -> str:
    d = _workdir.artifact_dir(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{NEAR_TELEMETRY_SUFFIX}")


def proxy_path_for(video_path: str) -> str:
    """The 540p proxy path — shared with anya_far_telemetry's ball pass, so a
    video analysed by both fast paths is transcoded once."""
    return _proxy_path_for(video_path, PROXY_SUFFIX)


def ensure_proxy(video_path: str, size: Tuple[int, int] = (960, 540),
                 crf: int = 20, preset: str = "veryfast",
                 force: bool = False) -> str:
    """Transcode `video_path` to a `size` proxy once; return its path.

    Thin wrapper over `proxy.ensure_proxy` (which the far fast path shares) —
    see there for the frame-exactness contract and why CRF matters.
    """
    return _ensure_proxy(video_path, size=size, crf=crf, preset=preset,
                         force=force, label="NEAR-TELEM")


class NearTelemetryExtractor:
    """Two passes over the proxy: player track at 5 fps, then gated ball."""

    def __init__(self, video_path: str,
                 cfg: Optional[NearExtractorConfig] = None):
        self.video_path = video_path
        self.cfg = cfg or NearExtractorConfig()

        info = probe_video(video_path)
        self.fps          = info["fps"]
        self.total_frames = int(info["frame_count"])
        self.source_size  = (info["width"], info["height"])

        # Every 5 fps sample is a real source frame index, so the two passes
        # agree on frame numbering and so do the records.
        self.player_stride = max(1, int(round(self.fps / self.cfg.player_fps)))

        self.player_model = YOLO(str(_MODELS_DIR / "yolo26n.pt"))
        self.ball_model   = YOLO(str(_MODELS_DIR / "ball_best.pt"))

        self.court_vertices, _ = init_court(video_path,
                                            analysis_size=self.cfg.analysis_size)
        self.H = self._compute_homography()

        # Recorded in the header, not applied — anya_near_serve applies them to
        # the toss ROI itself, exactly as it does for the full telemetry.
        cached = load_cached_exclusion_zones(video_path)
        if cached is not None:
            self.exclusion_zones = self._to_analysis_coords(cached)
        else:
            print("[NEAR-TELEM] Scanning for static exclusion zones…")
            try:
                self.exclusion_zones = create_auto_exclusion_zones(
                    video_path, self.ball_model,
                    num_frames=50, conf=0.04, eps=12, padding=8,
                    ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                    analysis_size=self.cfg.analysis_size,
                )
                save_cached_exclusion_zones(video_path, self.exclusion_zones)
            except Exception as e:
                print(f"[NEAR-TELEM] WARN: exclusion-zone scan failed: {e}")
                self.exclusion_zones = []

        self.timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _compute_homography(self):
        BL, BR, TR, TL = self.court_vertices
        dst = np.array([
            [0, 0], [Config.COURT_WIDTH_FT, 0],
            [Config.COURT_WIDTH_FT, Config.COURT_LENGTH_FT],
            [0, Config.COURT_LENGTH_FT],
        ], dtype=np.float32)
        src = np.array([BL, BR, TR, TL], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H

    def _world(self, px: float, py: float) -> Tuple[float, float]:
        pt = cv2.perspectiveTransform(
            np.array([[[px, py]]], dtype=np.float32), self.H)
        return float(pt[0][0][0]), float(pt[0][0][1])

    def _to_analysis_coords(self, zones):
        aw, ah = self.cfg.analysis_size
        if not zones:
            return []
        if max(z[2] for z in zones) <= aw and max(z[3] for z in zones) <= ah:
            return list(zones)
        sx, sy = aw / float(self.source_size[0]), ah / float(self.source_size[1])
        return [[int(z[0] * sx), int(z[1] * sy), int(z[2] * sx), int(z[3] * sy)]
                for z in zones]

    # ------------------------------------------------------------------
    def _near_from_result(self, result):
        """Near player only: highest-eligible detection nearest the near baseline.

        Mirrors anya_telemetry._players_from_result's near branch exactly (same
        conf floor, same near-vs-far side test, same x padding) so the two
        extractors pick the same person; the far branch is simply absent.
        """
        if result is None or not result.boxes:
            return None, None

        L   = Config.COURT_LENGTH_FT
        pad = Config.NEAR_PLAYER_X_PAD_FT
        best = None
        for b in result.boxes:
            if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                continue
            conf = float(b.conf[0])
            if conf < self.cfg.near_min_conf:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            wx, wy = self._world((x1 + x2) / 2.0, y2)
            if not (abs(wy) < abs(wy - L)):          # far half — not our player
                continue
            if not (-pad <= wx <= Config.COURT_WIDTH_FT + pad):
                continue
            if best is None or abs(wy) < abs(best[2]):
                best = ((x1, y1, x2, y2), wx, wy)

        if best is None:
            return None, None
        return best[0], (round(best[1], 2), round(best[2], 2))

    def _in_ready_zone(self, world) -> bool:
        if world is None:
            return False
        wx, wy = world
        c = self.cfg
        return (c.zone_y_min_ft <= wy <= c.zone_y_max_ft and
                -c.zone_x_pad_ft <= wx <= Config.COURT_WIDTH_FT + c.zone_x_pad_ft)

    # ------------------------------------------------------------------
    def _toss_roi_centre(self, box) -> Tuple[float, float]:
        """Centre of anya_near_serve's toss ROI for a given player box.

        The ROI spans from 2/3 box-height above the box top down to 1/3 into
        the box; its centre is therefore one sixth of a box height above the
        box top.
        """
        x1, y1, x2, y2 = box
        ph = y2 - y1
        return (x1 + x2) / 2.0, y1 - ph / 6.0

    def _crop_window(self, box) -> Tuple[int, int, int, int]:
        """The toss ROI itself, padded — NOT clamped to the frame.

        Deliberately allowed to run off the frame edge: `_crop_pixels` pads
        the missing part, which keeps the window's size and aspect exactly as
        derived from the player box.  Clamping instead would change the scale
        factor per frame and quietly distort the mapping back to analysis
        coordinates.
        """
        x1, y1, x2, y2 = box
        ph = y2 - y1
        roi_w = ph * (2.0 / 3.0)
        cx = (x1 + x2) / 2.0
        roi_bottom = y2 - ph * (2.0 / 3.0)
        roi_top    = roi_bottom - ph
        px, py = roi_w * self.cfg.roi_pad_frac, ph * self.cfg.roi_pad_frac
        return (int(math.floor(cx - roi_w / 2.0 - px)),
                int(math.floor(roi_top - py)),
                int(math.ceil(cx + roi_w / 2.0 + px)),
                int(math.ceil(roi_bottom + py)))

    def _crop_pixels(self, frame, win):
        """Pixels for `win`, zero-padded where it falls outside the frame."""
        ah, aw = frame.shape[:2]
        x1, y1, x2, y2 = win
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        sx1, sy1 = max(0, x1), max(0, y1)
        sx2, sy2 = min(aw, x2), min(ah, y2)
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        if (sx1, sy1, sx2, sy2) == (x1, y1, x2, y2):
            out = frame[y1:y2, x1:x2]
        else:
            out = np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)
            out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frame[sy1:sy2, sx1:sx2]
        return np.ascontiguousarray(out)

    # ------------------------------------------------------------------
    def _pass_player(self, src: str, progress_cb=None) -> Dict[int, Dict]:
        """Pass A — near-player box/world every `player_stride` frames.

        Frames between samples are `grab()`-ed, not decoded: grab still has to
        reconstruct them (H.264 leaves no choice) but skips the YUV->BGR
        conversion and the copy into numpy, which is the part we can avoid.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        cap = open_video(src, "NEAR-TELEM")
        samples: Dict[int, Dict] = {}
        pend_idx: List[int] = []
        pend_img: List[np.ndarray] = []

        def flush():
            if not pend_img:
                return
            res = self.player_model(pend_img, verbose=False,
                                    conf=cfg.player_conf,
                                    imgsz=cfg.player_imgsz, device=_DEVICE)
            for fi, r in zip(pend_idx, res):
                box, world = self._near_from_result(r)
                samples[fi] = {"box": box, "world": world}
            pend_idx.clear()
            pend_img.clear()

        idx = -1
        try:
            while True:
                if not cap.grab():
                    break
                idx += 1
                if idx % self.player_stride:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != tuple(cfg.analysis_size):
                    frame = cv2.resize(frame, cfg.analysis_size,
                                       interpolation=cv2.INTER_LINEAR)
                pend_idx.append(idx)
                pend_img.append(frame)
                if len(pend_img) >= cfg.batch_size:
                    flush()
                    if progress_cb:
                        progress_cb(idx, self.total_frames)
            flush()
        finally:
            cap.release()

        # Pass A defines the coverage of everything after it: ready windows
        # come from these samples, and a short decode here reads as "the
        # player was never in the serve zone" rather than as a failure.
        assert_decode_complete("NEAR-TELEM pass A", src, idx,
                               self.total_frames - 1, self.fps)

        self.timings["pass_player_s"] = time.perf_counter() - t0
        n_found = sum(1 for s in samples.values() if s["box"])
        print(f"[NEAR-TELEM] pass A: {len(samples)} player samples "
              f"@{self.fps / self.player_stride:.2f} fps, {n_found} with a near "
              f"box ({self.timings['pass_player_s']:.1f}s)")
        return samples

    def _ready_windows(self, samples: Dict[int, Dict]) -> List[Tuple[float, float]]:
        """Merged [t_start, t_end] spans where the ball model should run.

        A span opens at the first in-zone sample and closes `ready_hold_s`
        after the last one; overlapping spans are merged.
        """
        hold = self.cfg.ready_hold_s
        spans: List[List[float]] = []
        for fi in sorted(samples):
            if not self._in_ready_zone(samples[fi]["world"]):
                continue
            t = fi / self.fps
            if spans and t <= spans[-1][1]:
                spans[-1][1] = t + hold
            else:
                spans.append([t, t + hold])
        return [(a, b) for a, b in spans]

    def _pass_ball(self, src: str, boxes: Dict[int, Optional[Tuple]],
                   windows: List[Tuple[float, float]],
                   progress_cb=None) -> Dict[int, List]:
        """Pass B — ball detections on the toss-ROI crop inside ready windows.

        Runs at the full frame rate: the toss is short and fast, and it is the
        one cue that still needs 30 fps.  Frames outside every window are
        grabbed and dropped.
        """
        cfg = self.cfg
        if not windows:
            self.timings["pass_ball_s"] = 0.0
            return {}
        t0 = time.perf_counter()
        cap = open_video(src, "NEAR-TELEM")
        out: Dict[int, List] = {}
        pend: List[Tuple[int, Tuple[int, int, int, int]]] = []
        pend_img: List[np.ndarray] = []
        n_infer = 0

        def flush():
            nonlocal n_infer
            if not pend_img:
                return
            res = self.ball_model(pend_img, verbose=False, conf=cfg.ball_conf,
                                  imgsz=cfg.ball_imgsz, device=_DEVICE)
            for (fi, win), r in zip(pend, res):
                ox, oy = win[0], win[1]
                # Undo the upscale: the crop was resized from (win) to
                # (crop_w, crop_h), so detections come back divided by that
                # same factor before the window origin is added.
                sx = (win[2] - win[0]) / float(cfg.crop_w)
                sy = (win[3] - win[1]) / float(cfg.crop_h)
                dets = []
                if r is not None and r.boxes:
                    for b in r.boxes:
                        bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                        dets.append((round(ox + (bx1 + bx2) / 2.0 * sx, 1),
                                     round(oy + (by1 + by2) / 2.0 * sy, 1),
                                     round(float(b.conf[0]), 3)))
                out[fi] = dets
            n_infer += len(pend_img)
            pend.clear()
            pend_img.clear()

        wi = 0
        idx = -1
        try:
            while wi < len(windows):
                if not cap.grab():
                    break
                idx += 1
                t = idx / self.fps
                while wi < len(windows) and t > windows[wi][1]:
                    wi += 1
                if wi >= len(windows) or t < windows[wi][0]:
                    continue
                box = boxes.get(idx)
                if box is None:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != tuple(cfg.analysis_size):
                    frame = cv2.resize(frame, cfg.analysis_size,
                                       interpolation=cv2.INTER_LINEAR)
                win = self._crop_window(box)
                crop = self._crop_pixels(frame, win)
                if crop is None:
                    continue
                # Upscale the tight ROI to the canonical tensor size: same
                # cost to the model, ~3x more pixels on the ball.
                crop = cv2.resize(crop, (cfg.crop_w, cfg.crop_h),
                                  interpolation=cv2.INTER_LINEAR)
                pend.append((idx, win))
                pend_img.append(crop)
                if len(pend_img) >= cfg.batch_size:
                    flush()
                    if progress_cb:
                        progress_cb(idx, self.total_frames)
            flush()
        finally:
            cap.release()

        # Window-bounded, so the bar is the last frame any window wanted
        # rather than the end of the file.
        assert_decode_complete(
            "NEAR-TELEM pass B", src, idx,
            min(self.total_frames - 1, int(windows[-1][1] * self.fps)), self.fps)

        self.timings["pass_ball_s"] = time.perf_counter() - t0
        print(f"[NEAR-TELEM] pass B: {n_infer} toss-ROI ball inferences over "
              f"{len(windows)} ready window(s) "
              f"({self.timings['pass_ball_s']:.1f}s)")
        return out

    # ------------------------------------------------------------------
    def _held_boxes(self, samples: Dict[int, Dict]) -> Dict[int, Optional[Tuple]]:
        """Per-source-frame near box, carried forward from the 5 fps samples.

        The toss ROI has to be positioned on every frame the ball model runs
        on, but the box is only measured five times a second; a box older than
        `box_hold_s` is treated as gone rather than stale.
        """
        hold_frames = int(round(self.cfg.box_hold_s * self.fps))
        held: Dict[int, Optional[Tuple]] = {}
        cur, cur_f = None, -10 ** 9
        for idx in range(self.total_frames):
            s = samples.get(idx)
            if s is not None and s["box"] is not None:
                cur, cur_f = s["box"], idx
            elif cur is not None and idx - cur_f > hold_frames:
                cur = None
            held[idx] = cur
        return held

    def extract(self, out_path: Optional[str] = None, progress_cb=None) -> str:
        cfg = self.cfg
        out_path = out_path or near_telemetry_path_for(self.video_path)
        tmp_path = out_path + ".part"
        wall0 = time.perf_counter()

        t0 = time.perf_counter()
        src = (ensure_proxy(self.video_path, cfg.analysis_size,
                            cfg.proxy_crf, cfg.proxy_preset)
               if cfg.use_proxy else self.video_path)
        self.timings["proxy_s"] = time.perf_counter() - t0

        samples = self._pass_player(src, progress_cb=progress_cb)
        held    = self._held_boxes(samples)
        windows = self._ready_windows(samples)
        ready_s = sum(b - a for a, b in windows)
        dur_s   = self.total_frames / self.fps
        print(f"[NEAR-TELEM] ready windows: {len(windows)}, {ready_s:.1f}s of "
              f"{dur_s:.1f}s ({ready_s / max(1e-9, dur_s):.0%} duty)")

        balls = self._pass_ball(src, held, windows, progress_cb=progress_cb)

        in_win = [False] * self.total_frames
        for a, b in windows:
            lo = max(0, int(a * self.fps))
            hi = min(self.total_frames - 1, int(b * self.fps) + 1)
            for i in range(lo, hi + 1):
                in_win[i] = True

        n = 0
        with open(tmp_path, "w") as fh:
            meta = {
                "version":        NEAR_TELEMETRY_VERSION,
                "video":          os.path.basename(self.video_path),
                "proxy":          os.path.basename(src) if src != self.video_path else None,
                "fps":            self.fps,
                "total_frames":   self.total_frames,
                "stride":         1,
                "player_stride":  self.player_stride,
                "player_fps":     self.fps / self.player_stride,
                "player_imgsz":   cfg.player_imgsz,
                "ball_imgsz":     cfg.ball_imgsz,
                "ball_conf":      cfg.ball_conf,
                "analysis_size":  list(cfg.analysis_size),
                "source_size":    list(self.source_size),
                "court_length_ft": Config.COURT_LENGTH_FT,
                "court_width_ft":  Config.COURT_WIDTH_FT,
                "exclusion_zones": [list(z) for z in self.exclusion_zones],
                "ready_windows":  [[round(a, 3), round(b, 3)] for a, b in windows],
                "ready_duty":     round(ready_s / max(1e-9, dur_s), 4),
            }
            fh.write(json.dumps({"meta": meta}) + "\n")

            for idx in range(self.total_frames):
                s = samples.get(idx)
                fresh = s is not None and s["box"] is not None
                box = held[idx]
                # Frames that carry neither a fresh player sample nor ball data
                # tell the scorer nothing; leaving them out keeps the file small
                # without changing any cue, since every cue is time-based.
                if not fresh and idx not in balls and not in_win[idx]:
                    continue
                rec = {
                    "f":  idx,
                    "t":  round(idx / self.fps, 4),
                    "pn": bool(fresh),
                    "np": list(box) if box else None,
                    "npw": (list(s["world"]) if fresh and s["world"] else None),
                }
                if idx in balls:
                    rec["balls"] = [list(b) for b in balls[idx]]
                fh.write(json.dumps(rec) + "\n")
                n += 1

        os.replace(tmp_path, out_path)
        self.timings["total_s"] = time.perf_counter() - wall0
        print(f"[NEAR-TELEM] Wrote {n} records → {out_path}")
        print(f"[NEAR-TELEM] timing: proxy {self.timings['proxy_s']:.1f}s + "
              f"player {self.timings['pass_player_s']:.1f}s + ball "
              f"{self.timings['pass_ball_s']:.1f}s = "
              f"{self.timings['total_s']:.1f}s total")
        return out_path


def extract_near_telemetry(video_path: str, force: bool = False,
                           cfg: Optional[NearExtractorConfig] = None,
                           out_path: Optional[str] = None,
                           progress_cb=None) -> str:
    """Extract (or reuse cached) near-only telemetry. Returns the JSONL path."""
    out_path = out_path or near_telemetry_path_for(video_path)
    if not force and os.path.isfile(out_path):
        try:
            with open(out_path) as fh:
                ver = int(json.loads(fh.readline()).get("meta", {}).get("version", 0))
        except Exception:
            ver = 0
        if ver == NEAR_TELEMETRY_VERSION:
            print(f"[NEAR-TELEM] Using cached near telemetry: {out_path}")
            return out_path
    return NearTelemetryExtractor(video_path, cfg=cfg).extract(
        out_path=out_path, progress_cb=progress_cb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Near-serve-only telemetry: 540p proxy, 5 fps player, "
                    "toss-ROI ball inside ready windows")
    ap.add_argument("video")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-proxy", action="store_true",
                    help="Decode the source directly instead of a 540p proxy")
    ap.add_argument("--player-fps", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c = NearExtractorConfig()
    if args.no_proxy:
        c.use_proxy = False
    if args.player_fps is not None:
        c.player_fps = args.player_fps
    extract_near_telemetry(args.video, force=args.force, cfg=c, out_path=args.out)
