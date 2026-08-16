"""
anya_far_telemetry.py
=====================
The far-serve-only sibling of anya_telemetry.py + extract_far_pose.py: the
same far-player track, hand-raise pose and ball stream that anya_far_serve.py
reads, for a fraction of the compute.

anya_telemetry.py is a general-purpose pass — it feeds far serve, near serve
and ball-quiet dead time alike, so it decodes the 4K source and runs three
models on every frame.  anya_far_serve.py then reads only four fields out of
it (`fpr`, `fprw`, `all_balls` and, through extract_far_pose, the pixels
inside `fpr`), and extract_far_pose decodes the whole source a second time to
crop those boxes.  This module serves that one consumer and spends nothing on
what it never reads.  It writes DIFFERENT files, so the full telemetry cache
stays valid and the two can be scored against each other on the same video.

    <stem>_anya_far_telemetry.jsonl            far telemetry (schema-compatible)
    <stem>_anya_far_telemetry_far_pose.jsonl   pose cache (v2 layout)

Both are what `detect_far_serves(<the telemetry path>)` already expects — the
pose path is exactly what `far_pose_path_for` derives — so the detector runs
against this pass unmodified.

Where the savings come from:

  1. A native-resolution BAND proxy.  The far player is ~25 px tall at 540p,
     which is why the full pass runs a second model call on an uncropped band
     around the far baseline in the first place.  That band is ~12% of a 4K
     frame, so transcoding it once turns a 6.7 ms/frame decode into well
     under 1 ms — and both far passes then read it instead of the source.
     Nothing is downscaled: these are the same source pixels the full pass
     crops out, just without the 88% of the frame it throws away.

  2. Far player at 5 fps.  Arming is a 1 s stationarity test with a 3 ft
     tolerance; it does not need 30 samples a second to decide the player is
     standing still.  The box is held between samples so the pose crop stays
     positioned — which is safe precisely because the frames that matter are
     the ones where the player is holding station.

  3. Pose at the full frame rate, batched, at imgsz 320.  The hand-raise gate
     medians over a 0.10 s window and needs >= 3 samples in it, so this is the
     one cue that genuinely wants every frame.  Two things make it cheaper
     anyway: the crop is canonicalised to one size per clip, which makes the
     call batchable (the full pass's crops are 3,323 distinct shapes, and a
     mixed-shape batch flips ultralytics' letterbox mode); and the model runs
     at imgsz 320 rather than the default 640, which was upscaling a ~110x160
     crop fourfold and paying for the pixels.

     Gating pose on the armed windows, which is the direct analogue of the
     near path's ready gate, turns out to buy very little: the far player
     stands still for most of a match, so the duty cycle is ~91%.  The gate
     is kept because it costs nothing and does bound the pass, but the real
     savings here are the band proxy, the batch and the smaller imgsz.

  4. Ball at a reduced rate, on the 540p proxy, only while a point can be
     live.  anya_far_serve uses the ball for two things: ending a point (a
     1.5 s quiet test, which does not care about a 0.1 s sampling grid) and
     upgrading a detection from MEDIUM to HIGH confidence.  Neither needs 30
     fps, and neither is consulted during the dead time between points, so
     the pass carries the detector's own point state and skips inference on
     frames where no point could be open.

The arming test here is deliberately LOOSER than the detector's (longer gap
tolerance, larger drift, shorter hold) and the windows it produces are padded
on both sides.  It decides where pose is available, so anything it misses the
detector can never see; anything extra costs only pose compute.

WHAT THIS PASS DOES NOT REPRODUCE.  The keypoints are not the full pass's
keypoints, and no amount of tuning the crop made them so.  Measured on Data/23
by raise-gate crossings at the shipped threshold (the full pass: 44):

    band proxy crf 14, crop resized to the canonical size      72
    band proxy crf 6                                           64
    band proxy crf 14, crop padded into the canvas (shipped)   67
    source pixels, no proxy, crop resized                      56
    source pixels, batch size 1                                56

So roughly half the difference is the band's re-encode and the rest is the
crop geometry; batching is not implicated at all (batch 1 and batch 16 agree
exactly, which is what the shape-uniform argument predicts).  What matters is
that the fast stream's raise signal is not a degraded copy of the full one but
a different one — it crosses more often at a low threshold yet keeps every
true serve at a high one, where the full pass starts losing them.  That is why
anya_far_serve carries a separate preset (`for_fast_path`) selected from this
file's `meta.source`, rather than a single set of thresholds for both.

Per-frame record (JSONL, meta header first).  Frames that carry none of the
three signals are omitted — every cue in the detector is time-based, so a
record with nothing in it changes no decision:
    f          frame index in the source video
    t          timestamp seconds
    fpr        far player box [x1,y1,x2,y2] in SOURCE pixels, or null.  Fresh
               on player-pass frames, held for `box_hold_s` in between.
    fprw       world feet [wx, wy] for a FRESH box only, or null — the arming
               test must not see a held sample as evidence of stillness
    fprc       confidence for a fresh box, or null
    all_balls  ball detections [[cx, cy, conf], ...] in 960x540 analysis
               coords, unfiltered.  Present ONLY on ball-pass frames; its
               absence means "not looked at", not "nothing there".

Run:
    python -m pipeline.anya_far_telemetry match.mp4 [--force]
    python -m pipeline.anya_far_serve match_anya_far_telemetry.jsonl [--eval gt.json]
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

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
    from .proxy import ensure_proxy, ensure_crop_proxy, FAR_BAND_SUFFIX
    from .extract_far_pose import FAR_POSE_VERSION, N_KP, far_pose_path_for
except ImportError:                         # script import (python pipeline/x.py)
    from utilities import (Config, init_court, create_auto_exclusion_zones,
                           load_cached_exclusion_zones,
                           save_cached_exclusion_zones, probe_video,
                           assert_decode_complete, open_video)
    from proxy import ensure_proxy, ensure_crop_proxy, FAR_BAND_SUFFIX
    from extract_far_pose import FAR_POSE_VERSION, N_KP, far_pose_path_for

_MODELS_DIR = Path(__file__).parent / "models"

FAR_TELEMETRY_SUFFIX  = "_anya_far_telemetry.jsonl"
FAR_TELEMETRY_VERSION = 1


@dataclass
class FarExtractorConfig:
    analysis_size: Tuple[int, int] = (960, 540)

    # --- band proxy (native resolution, far-baseline band only)
    use_band_proxy: bool = True
    band_crf:       int  = 14      # the far player is the subject here and the
                                   # band is small, so quality is cheap; see
                                   # proxy.ensure_proxy for what CRF 20 did to
                                   # the near path's ball
    band_preset:    str  = "veryfast"
    far_roi_height_frac: float = 0.25   # matches anya_telemetry's band exactly

    # --- pass A: far player inside the band
    player_fps:      float = 5.0
    far_roi_imgsz:   int   = 384   # same as the full pass's ROI call
    far_roi_conf:    float = 0.25
    far_roi_edge_px: int   = 4     # feet on the band's bottom edge are clipped,
                                   # not landed
    box_hold_s:      float = 0.45  # carry the box between 5 fps samples (0.2 s
                                   # apart) plus one tolerated dropout

    # --- arming (deliberately looser than FarServeDetectorConfig's)
    arm_stable_s:     float = 0.7   # detector: 1.0
    arm_max_drift_ft: float = 4.5   # detector: 3.0
    arm_gap_s:        float = 1.2   # detector: 1.0 (FP_HYSTERESIS_S)
    pose_lead_s:      float = 1.0   # pose available before the arming point, so
                                    # the 0.10 s median has samples to work with
    pose_tail_s:      float = 3.5   # detector: ARM_TO_TRACE_S = 2.5

    # --- pass B: pose, full rate, inside armed windows
    pose_conf:  float = 0.05   # crop is small, keep the floor permissive
    pad_px:     int   = 25     # padding around the fpr box, as extract_far_pose
    pose_imgsz: int   = 320    # the crop is ~110x160, so ultralytics' default
                               # 640 letterbox was upscaling it 4x and paying
                               # for the pixels.  Measured on Data/23: 320
                               # halves pass B (211s -> 103s) and the raise
                               # signal comes out CLEANER, not coarser (54
                               # gate crossings against 67)
    # How a variable-sized crop is made shape-uniform so the pose call can be
    # batched.  "pad" centres the native crop in one canvas per clip and adds
    # nothing else — no resampling at all, which is the point: "resize" scales
    # every crop to the canvas, and that second resample (the model then
    # letterboxes to pose_imgsz regardless) is visible in the keypoints as
    # jitter the raise gate reads as extra crossings.
    crop_mode: str = "pad"

    # --- pass C: ball, on the 540p proxy
    use_proxy540: bool  = True
    proxy_crf:    int   = 14
    ball_fps:     float = 10.0
    ball_imgsz:   int   = 960
    ball_conf:    float = Config.ACTIVE_BALL_CONF
    # Gate the ball pass on the detector's own point state: no point can be
    # open before the first armed frame of a window, and one closes on quiet
    # (or the hard cap).  Mirrors FarServeDetectorConfig's POINT_* values, one
    # batch of slack either way — overshooting costs compute, undershooting
    # would hide a ball the detector needed.
    gate_ball:      bool  = True
    point_max_s:    float = 30.0
    point_quiet_s:  float = 1.5
    ball_tail_s:    float = 3.0    # keep sampling this long past an armed
                                   # window even with no ball, so a serve at
                                   # the very end of one can still be confirmed

    batch_size: int = 16


def far_telemetry_path_for(video_path: str) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{FAR_TELEMETRY_SUFFIX}")


def _merge_spans(spans: List[List[float]]) -> List[Tuple[float, float]]:
    """Sorts and unions [start, end] spans."""
    out: List[List[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


class FarTelemetryExtractor:
    """Three passes: far player in the band, pose while armed, gated ball."""

    def __init__(self, video_path: str,
                 cfg: Optional[FarExtractorConfig] = None):
        self.video_path = video_path
        self.cfg = cfg or FarExtractorConfig()

        info = probe_video(video_path)
        self.fps          = info["fps"]
        self.total_frames = int(info["frame_count"])
        self.source_size  = (info["width"], info["height"])

        self.player_stride = max(1, int(round(self.fps / self.cfg.player_fps)))
        self.ball_stride   = max(1, int(round(self.fps / self.cfg.ball_fps)))

        self.player_model = YOLO(str(_MODELS_DIR / "yolo26n.pt"))
        self.pose_model   = YOLO(str(_MODELS_DIR / "yolov8n-pose.pt"))
        self.ball_model   = YOLO(str(_MODELS_DIR / "ball_best.pt"))

        self.court_vertices, _ = init_court(video_path,
                                            analysis_size=self.cfg.analysis_size)
        self.H = self._compute_homography()
        self.far_roi = self._far_roi()

        # Recorded in the header, not applied — anya_far_serve rescales and
        # applies them itself, exactly as it does for the full telemetry.
        cached = load_cached_exclusion_zones(video_path)
        if cached is not None:
            self.exclusion_zones = self._to_analysis_coords(cached)
        else:
            print("[FAR-TELEM] Scanning for static exclusion zones…")
            try:
                self.exclusion_zones = create_auto_exclusion_zones(
                    video_path, self.ball_model,
                    num_frames=50, conf=0.04, eps=12, padding=8,
                    ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                    analysis_size=self.cfg.analysis_size,
                )
                save_cached_exclusion_zones(video_path, self.exclusion_zones)
            except Exception as e:
                print(f"[FAR-TELEM] WARN: exclusion-zone scan failed: {e}")
                self.exclusion_zones = []

        self.timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Geometry — identical to anya_telemetry's, so `fpr`/`fprw` mean the
    # same thing in both files.
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

    def _far_roi(self) -> Tuple[int, int, int, int]:
        """The far-baseline band in source pixels, rounded to even dimensions.

        Same construction as anya_telemetry._far_roi — baseline width, a
        fraction of frame height, centred on the baseline — with width and
        height forced even because that is what the yuv420p transcode of the
        band needs.  Rounding here rather than inside the transcode keeps the
        band this module maps coordinates against and the one ffmpeg actually
        cut identical.
        """
        _, _, TR, TL = self.court_vertices
        aw, ah = self.cfg.analysis_size
        sw, sh = self.source_size
        sx, sy = sw / float(aw), sh / float(ah)

        x_left  = min(TL[0], TR[0]) * sx
        x_right = max(TL[0], TR[0]) * sx
        y_base  = ((TL[1] + TR[1]) / 2.0) * sy
        half_h  = (sh * self.cfg.far_roi_height_frac) / 2.0

        x1, y1 = max(0, int(x_left)), max(0, int(y_base - half_h))
        x2, y2 = min(sw, int(x_right)), min(sh, int(y_base + half_h))
        return (x1, y1, x1 + ((x2 - x1) & ~1), y1 + ((y2 - y1) & ~1))

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
    def _far_player_from_result(self, result, band_origin):
        """One band detection as (box_source_px, world_feet, conf).

        Mirrors anya_telemetry._far_player_from_roi: highest confidence, with
        detections whose feet sit on the band's bottom edge rejected because
        the person continues below the crop and their "feet" are really the
        crop boundary.
        """
        if result is None or not result.boxes:
            return None, None, None

        rx1, ry1, rx2, ry2 = self.far_roi
        ox, oy = band_origin
        aw, ah = self.cfg.analysis_size
        sw, sh = self.source_size
        inv_x, inv_y = aw / float(sw), ah / float(sh)
        band_h = ry2 - ry1
        fpad = Config.FAR_PLAYER_X_PAD_FT

        best = None
        for b in result.boxes:
            if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                continue
            cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
            if cy2 >= band_h - self.cfg.far_roi_edge_px:
                continue

            x1, y1 = cx1 + ox, cy1 + oy
            x2, y2 = cx2 + ox, cy2 + oy
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
    def _pass_player(self, src: str, band_origin, progress_cb=None) -> Dict[int, Dict]:
        """Pass A — far-player box/world every `player_stride` frames.

        Frames between samples are `grab()`-ed rather than decoded: grab still
        reconstructs them (H.264 leaves no choice) but skips the YUV->BGR
        conversion and the copy into numpy, which is the part that can be
        avoided.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        cap = open_video(src, "FAR-TELEM")
        samples: Dict[int, Dict] = {}
        pend_idx: List[int] = []
        pend_img: List[np.ndarray] = []

        def flush():
            if not pend_img:
                return
            res = self.player_model(pend_img, verbose=False,
                                    conf=cfg.far_roi_conf,
                                    imgsz=cfg.far_roi_imgsz, device=_DEVICE)
            for fi, r in zip(pend_idx, res):
                box, world, conf = self._far_player_from_result(r, band_origin)
                samples[fi] = {"box": box, "world": world, "conf": conf}
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
                pend_idx.append(idx)
                pend_img.append(frame)
                if len(pend_img) >= cfg.batch_size:
                    flush()
                    if progress_cb:
                        progress_cb(idx, self.total_frames)
            flush()
        finally:
            cap.release()

        # Armed windows come from these samples, so a short decode here is
        # indistinguishable from "the far player never settled to serve".
        assert_decode_complete("FAR-TELEM pass A", src, idx,
                               self.total_frames - 1, self.fps)

        self.timings["pass_player_s"] = time.perf_counter() - t0
        n_found = sum(1 for s in samples.values() if s["box"])
        print(f"[FAR-TELEM] pass A: {len(samples)} player samples "
              f"@{self.fps / self.player_stride:.2f} fps, {n_found} with a far "
              f"box ({self.timings['pass_player_s']:.1f}s)")
        return samples

    # ------------------------------------------------------------------
    def _armed_windows(self, samples: Dict[int, Dict]) -> List[Tuple[float, float]]:
        """Spans where the detector could plausibly consider the player armed.

        The detector arms when the far player is visible and their world-x has
        stayed within ARM_MAX_DRIFT_FT for ARM_STABLE_S, and holds that level
        for ARM_TO_TRACE_S afterwards.  This reproduces the test on the 5 fps
        track with every tolerance opened up (see the class docstring): pose
        exists wherever this says armed, so a miss here is a serve the
        detector can never see, while a false positive costs only pose frames.
        """
        cfg = self.cfg
        pts = [(fi / self.fps, s["world"][0])
               for fi, s in sorted(samples.items())
               if s["world"] is not None]
        spans: List[List[float]] = []
        for i, (t, _) in enumerate(pts):
            # Samples covering [t - arm_stable_s, t], anchored on the latest
            # sample at or before the window start.
            j = i
            while j > 0 and pts[j - 1][0] > t - cfg.arm_stable_s:
                j -= 1
            if j == 0 and pts[0][0] > t - cfg.arm_stable_s:
                continue                      # window opens before the track does
            win = pts[max(0, j - 1):i + 1]    # include the anchor, as the
                                              # detector's own test does
            if len(win) < 2:
                continue
            if any(t1 - t0 > cfg.arm_gap_s for (t0, _), (t1, _) in zip(win, win[1:])):
                continue
            anchor = win[0][1]
            if any(abs(wx - anchor) > cfg.arm_max_drift_ft for _, wx in win):
                continue
            spans.append([win[0][0] - cfg.pose_lead_s, t + cfg.pose_tail_s])
        return _merge_spans(spans)

    def _held_boxes(self, samples: Dict[int, Dict]) -> Dict[int, Optional[Tuple]]:
        """Per-source-frame far box, carried forward from the 5 fps samples.

        The pose crop has to be positioned on every frame pose runs on, but
        the box is only measured five times a second.  Holding is safe here
        for the same reason arming works at all: the frames that matter are
        the ones where the player is standing still.  A box older than
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

    # ------------------------------------------------------------------
    def _canonical_crop(self, samples: Dict[int, Dict]) -> Tuple[int, int]:
        """One crop size for the whole clip, from the clip's own boxes.

        extract_far_pose crops `fpr` + padding and hands that straight to the
        model, which produces thousands of distinct shapes over a clip — and
        ultralytics switches its letterbox mode for a mixed-shape batch
        (`pre_transform`: `auto=same_shapes and ...`), so batching those crops
        would move the very keypoints the raise gate reads.  Sizing one window
        from the median box instead makes every batch shape-uniform.

        Geometry is preserved: each frame's window is grown (never squashed)
        to this aspect around the box centre and then resized uniformly, so
        the vertical distances the raise metric compares are all scaled by the
        same factor — and `bh` is written in the same scaled units.

        Sized from the box-height p90 rather than the median, because the
        resize is only lossy in one direction.  The model letterboxes to
        `pose_imgsz` whatever it is handed, so upscaling here costs nothing and
        throws nothing away, while downscaling permanently removes detail from
        a subject that is already small.  At the median, half the clip's crops
        were being shrunk; at the p90, a tenth are.  That mattered: on Data/23
        the median-sized crop produced 73 raise-gate crossings against the
        full pass's 44, i.e. it was manufacturing keypoint jitter.
        """
        boxes = [s["box"] for s in samples.values() if s["box"]]
        if not boxes:
            return 128, 224
        pad = 2 * self.cfg.pad_px
        hs = sorted(b[3] - b[1] + pad for b in boxes)
        p90 = hs[int(0.90 * (len(hs) - 1))]
        aspect = median((b[2] - b[0] + pad) / float(b[3] - b[1] + pad)
                        for b in boxes)
        h = int(min(448, max(96, round(p90 / 2) * 2)))
        aspect = min(1.2, max(0.45, aspect))
        w = int(min(448, max(64, round(h * aspect / 2) * 2)))
        return w, h

    @staticmethod
    def _window_for(box, pad: int, aspect: float,
                    fixed: Optional[Tuple[int, int]] = None
                    ) -> Tuple[int, int, int, int]:
        """`box` padded, then grown around its centre.

        With `fixed=(w, h)` the window is that exact size whenever the padded
        box fits inside it (the crop is then native pixels in a canvas of the
        clip's one shape); otherwise it grows to `aspect` and the caller
        scales it down.
        """
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
        w, h = max(2.0, x2 - x1), max(2.0, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if fixed is not None and w <= fixed[0] and h <= fixed[1]:
            w, h = fixed
        elif w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        return (int(round(cx - w / 2)), int(round(cy - h / 2)),
                int(round(cx + w / 2)), int(round(cy + h / 2)))

    @staticmethod
    def _crop_pixels(frame, win):
        """Pixels for `win`, zero-padded where it falls outside the frame.

        Padding rather than clamping keeps the window's size and aspect
        exactly as derived from the box; clamping would change the scale
        factor per frame and quietly distort what `bh` means.
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = win
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        sx1, sy1 = max(0, x1), max(0, y1)
        sx2, sy2 = min(fw, x2), min(fh, y2)
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        if (sx1, sy1, sx2, sy2) == (x1, y1, x2, y2):
            return np.ascontiguousarray(frame[y1:y2, x1:x2])
        out = np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frame[sy1:sy2, sx1:sx2]
        return out

    @staticmethod
    def _keypoints(result) -> Optional[list]:
        """Flat [x,y,conf] * 17 for the highest-confidence detection."""
        if result is None or result.keypoints is None or len(result.boxes) == 0:
            return None
        k = result.keypoints.data.cpu().numpy()[0]
        out = []
        for i in range(N_KP):
            out.extend((round(float(k[i][0]), 1),
                        round(float(k[i][1]), 1),
                        round(float(k[i][2]), 3)))
        return out

    def _pass_pose(self, src: str, band_origin, boxes: Dict[int, Optional[Tuple]],
                   windows: List[Tuple[float, float]], crop_size: Tuple[int, int],
                   progress_cb=None) -> Dict[int, Dict]:
        """Pass B — pose on the far-player crop, full rate, inside `windows`."""
        cfg = self.cfg
        cw, ch = crop_size
        aspect = cw / float(ch)
        out: Dict[int, Dict] = {}
        if not windows:
            self.timings["pass_pose_s"] = 0.0
            return out

        t0 = time.perf_counter()
        cap = open_video(src, "FAR-TELEM")
        ox, oy = band_origin
        pend: List[Tuple[int, float]] = []       # (frame idx, scaled box height)
        pend_img: List[np.ndarray] = []
        n_infer = 0

        def flush():
            nonlocal n_infer
            if not pend_img:
                return
            res = self.pose_model(pend_img, verbose=False, conf=cfg.pose_conf,
                                  imgsz=cfg.pose_imgsz, device=_DEVICE)
            for (fi, bh), r in zip(pend, res):
                k = self._keypoints(r)
                if k:
                    out[fi] = {"k": k, "bh": bh}
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
                # Band coordinates: the boxes are in source pixels.
                bbox = (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)
                if cfg.crop_mode == "pad":
                    # The canvas is the clip's p90 crop, so it swallows the
                    # padded box whole on ~90% of frames and the pixels reach
                    # the model untouched.  The rest are the only ones that
                    # get resampled, and only downward.
                    win = self._window_for(bbox, cfg.pad_px, aspect,
                                           fixed=(cw, ch))
                    crop = self._crop_pixels(frame, win)
                    if crop is None:
                        continue
                    scale = 1.0
                    if crop.shape[0] != ch or crop.shape[1] != cw:
                        scale = ch / float(crop.shape[0])
                        crop = cv2.resize(crop, (cw, ch),
                                          interpolation=cv2.INTER_AREA)
                else:
                    win = self._window_for(bbox, cfg.pad_px, aspect)
                    crop = self._crop_pixels(frame, win)
                    if crop is None:
                        continue
                    scale = ch / float(win[3] - win[1])
                    crop = cv2.resize(crop, (cw, ch),
                                      interpolation=cv2.INTER_LINEAR)
                pend.append((idx, round((box[3] - box[1]) * scale, 1)))
                pend_img.append(crop)
                if len(pend_img) >= cfg.batch_size:
                    flush()
                    if progress_cb:
                        progress_cb(idx, self.total_frames)
            flush()
        finally:
            cap.release()

        # Window-bounded: the bar is the last frame any armed window wanted.
        assert_decode_complete(
            "FAR-TELEM pass B", src, idx,
            min(self.total_frames - 1, int(windows[-1][1] * self.fps)), self.fps)

        self.timings["pass_pose_s"] = time.perf_counter() - t0
        print(f"[FAR-TELEM] pass B: {n_infer} pose inferences at {cw}x{ch} over "
              f"{len(windows)} armed window(s), {len(out)} with keypoints "
              f"({self.timings['pass_pose_s']:.1f}s)")
        return out

    # ------------------------------------------------------------------
    def _pass_ball(self, src: str, windows: List[Tuple[float, float]],
                   progress_cb=None) -> Dict[int, List]:
        """Pass C — whole-court ball detections, every `ball_stride` frames.

        Runs on the 540p proxy, in analysis coordinates, so what it writes is
        directly comparable to the full pass's `all_balls`.

        With `gate_ball`, inference is skipped where the detector could not
        have a point open: no point starts outside an armed window, and one
        closes after `point_quiet_s` of silence or `point_max_s` outright.
        The state is updated at batch boundaries, so the gate lags by up to
        one batch — which overshoots (extra inference) and never undershoots.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        cap = open_video(src, "FAR-TELEM")
        out: Dict[int, List] = {}
        pend: List[int] = []
        pend_img: List[np.ndarray] = []
        n_infer = 0
        # `open_t` is when the current point could have started; `evidence_t`
        # is the last moment something kept it alive — a ball detection, or
        # simply still being inside an armed window.
        open_t: Optional[float] = None
        evidence_t = -1e9

        def flush():
            nonlocal n_infer, evidence_t
            if not pend_img:
                return
            res = self.ball_model(pend_img, verbose=False, conf=cfg.ball_conf,
                                  imgsz=cfg.ball_imgsz, device=_DEVICE)
            for fi, r in zip(pend, res):
                dets = []
                if r is not None and r.boxes:
                    for b in r.boxes:
                        bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                        dets.append((round((bx1 + bx2) / 2.0, 1),
                                     round((by1 + by2) / 2.0, 1),
                                     round(float(b.conf[0]), 3)))
                out[fi] = dets
                if dets:
                    evidence_t = max(evidence_t, fi / self.fps)
            n_infer += len(pend_img)
            pend.clear()
            pend_img.clear()

        wi = 0
        idx = -1
        try:
            while True:
                if not cap.grab():
                    break
                idx += 1
                if idx % self.ball_stride:
                    continue
                t = idx / self.fps
                while wi < len(windows) and t > windows[wi][1]:
                    wi += 1
                in_window = wi < len(windows) and windows[wi][0] <= t

                if cfg.gate_ball:
                    if in_window:
                        if open_t is None:
                            open_t = t
                        evidence_t = max(evidence_t, t)
                    elif open_t is None:
                        continue
                    elif (t - open_t > cfg.point_max_s or
                          t - evidence_t > cfg.point_quiet_s + cfg.ball_tail_s):
                        open_t = None
                        continue

                ok, frame = cap.retrieve()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != tuple(cfg.analysis_size):
                    frame = cv2.resize(frame, cfg.analysis_size,
                                       interpolation=cv2.INTER_LINEAR)
                pend.append(idx)
                pend_img.append(frame)
                if len(pend_img) >= cfg.batch_size:
                    flush()
                    if progress_cb:
                        progress_cb(idx, self.total_frames)
            flush()
        finally:
            cap.release()

        assert_decode_complete("FAR-TELEM pass C", src, idx,
                               self.total_frames - 1, self.fps)

        self.timings["pass_ball_s"] = time.perf_counter() - t0
        n_possible = self.total_frames // self.ball_stride
        print(f"[FAR-TELEM] pass C: {n_infer} ball inferences "
              f"@{self.fps / self.ball_stride:.2f} fps "
              f"({n_infer / max(1, n_possible):.0%} of the sampling grid, "
              f"gate={'on' if cfg.gate_ball else 'off'}) "
              f"({self.timings['pass_ball_s']:.1f}s)")
        self._n_ball_infer = n_infer
        return out

    # ------------------------------------------------------------------
    def extract(self, out_path: Optional[str] = None, progress_cb=None) -> str:
        cfg = self.cfg
        out_path = out_path or far_telemetry_path_for(self.video_path)
        pose_path = far_pose_path_for(out_path)
        wall0 = time.perf_counter()

        t0 = time.perf_counter()
        band_src, band_origin = self.video_path, (0, 0)
        if cfg.use_band_proxy:
            p = ensure_crop_proxy(self.video_path, self.far_roi,
                                  suffix=FAR_BAND_SUFFIX, crf=cfg.band_crf,
                                  preset=cfg.band_preset, label="FAR-TELEM")
            if p != self.video_path:
                band_src, band_origin = p, (self.far_roi[0], self.far_roi[1])
        if band_src == self.video_path:
            print("[FAR-TELEM] band proxy unavailable — cropping the source.")
        ball_src = (ensure_proxy(self.video_path, cfg.analysis_size,
                                 crf=cfg.proxy_crf, label="FAR-TELEM")
                    if cfg.use_proxy540 else self.video_path)
        self.timings["proxy_s"] = time.perf_counter() - t0

        # Pass A reads the band; when the proxy is unavailable it reads full
        # source frames, so the crop has to happen here instead.
        if band_src == self.video_path:
            samples = self._pass_player_from_source(progress_cb=progress_cb)
        else:
            samples = self._pass_player(band_src, band_origin,
                                        progress_cb=progress_cb)

        held    = self._held_boxes(samples)
        windows = self._armed_windows(samples)
        armed_s = sum(b - a for a, b in windows)
        dur_s   = self.total_frames / self.fps
        print(f"[FAR-TELEM] armed windows: {len(windows)}, {armed_s:.1f}s of "
              f"{dur_s:.1f}s ({armed_s / max(1e-9, dur_s):.0%} duty)")

        crop_size = self._canonical_crop(samples)
        pose = self._pass_pose(band_src, band_origin, held, windows, crop_size,
                               progress_cb=progress_cb)
        balls = self._pass_ball(ball_src, windows, progress_cb=progress_cb)

        # ---- write the pose cache (extract_far_pose's v2 layout) ----------
        tmp = pose_path + ".part"
        with open(tmp, "w") as fh:
            fh.write(json.dumps({"meta": {
                "version": FAR_POSE_VERSION,
                "source_telemetry": os.path.basename(out_path),
                "fps": self.fps,
                "pad_px": cfg.pad_px,
                "pose_conf": cfg.pose_conf,
                "pose_imgsz": cfg.pose_imgsz,
                "n_kp": N_KP,
                "crop_size": list(crop_size),
                "crop_mode": cfg.crop_mode,
                "coords": "canonical crop pixels (fpr box + pad_px, grown to "
                          "the clip's one crop aspect and resized); `bh` is "
                          "scaled to match",
            }}) + "\n")
            for idx in sorted(pose):
                rec = {"f": idx, "t": round(idx / self.fps, 4)}
                rec.update(pose[idx])
                fh.write(json.dumps(rec) + "\n")
        os.replace(tmp, pose_path)
        print(f"[FAR-TELEM] Wrote {len(pose)} pose records → {pose_path}")

        # ---- write the telemetry -----------------------------------------
        in_win = [False] * self.total_frames
        for a, b in windows:
            lo = max(0, int(a * self.fps))
            hi = min(self.total_frames - 1, int(b * self.fps) + 1)
            for i in range(lo, hi + 1):
                in_win[i] = True

        tmp = out_path + ".part"
        n = 0
        with open(tmp, "w") as fh:
            meta = {
                "version":         FAR_TELEMETRY_VERSION,
                # Provenance, so a consumer can tell this stream from the full
                # pass's without guessing from which keys happen to be present.
                # anya_far_serve reads it to pick its fast-path preset.
                "source":          "anya_far_telemetry",
                "video":           os.path.basename(self.video_path),
                "band_proxy":      (os.path.basename(band_src)
                                    if band_src != self.video_path else None),
                "ball_proxy":      (os.path.basename(ball_src)
                                    if ball_src != self.video_path else None),
                "fps":             self.fps,
                "total_frames":    self.total_frames,
                "stride":          1,
                "player_stride":   self.player_stride,
                "player_fps":      self.fps / self.player_stride,
                "ball_stride":     self.ball_stride,
                "ball_frames":     getattr(self, "_n_ball_infer", 0),
                "ball_imgsz":      cfg.ball_imgsz,
                "ball_conf":       cfg.ball_conf,
                "analysis_size":   list(cfg.analysis_size),
                "source_size":     list(self.source_size),
                "court_length_ft": Config.COURT_LENGTH_FT,
                "court_width_ft":  Config.COURT_WIDTH_FT,
                "exclusion_zones": [list(z) for z in self.exclusion_zones],
                "far_roi":         list(self.far_roi),
                "armed_windows":   [[round(a, 3), round(b, 3)] for a, b in windows],
                "armed_duty":      round(armed_s / max(1e-9, dur_s), 4),
                "pose_cache":      os.path.basename(pose_path),
            }
            fh.write(json.dumps({"meta": meta}) + "\n")

            for idx in range(self.total_frames):
                s = samples.get(idx)
                fresh = s is not None and s["box"] is not None
                has_ball = idx in balls
                if not fresh and not has_ball and not in_win[idx]:
                    continue
                rec = {
                    "f":    idx,
                    "t":    round(idx / self.fps, 4),
                    "fpr":  list(held[idx]) if held[idx] else None,
                    # World position is written for FRESH samples only: the
                    # arming test reads it as evidence the player is standing
                    # still, and a held box repeated 5 times would be evidence
                    # of nothing but the hold.
                    "fprw": list(s["world"]) if fresh and s["world"] else None,
                    "fprc": s["conf"] if fresh else None,
                }
                if has_ball:
                    rec["all_balls"] = [list(b) for b in balls[idx]]
                fh.write(json.dumps(rec) + "\n")
                n += 1

        os.replace(tmp, out_path)
        self.timings["total_s"] = time.perf_counter() - wall0
        print(f"[FAR-TELEM] Wrote {n} records → {out_path}")
        print(f"[FAR-TELEM] timing: proxy {self.timings['proxy_s']:.1f}s + "
              f"player {self.timings['pass_player_s']:.1f}s + pose "
              f"{self.timings['pass_pose_s']:.1f}s + ball "
              f"{self.timings['pass_ball_s']:.1f}s = "
              f"{self.timings['total_s']:.1f}s total "
              f"({1000 * self.timings['total_s'] / max(1, self.total_frames):.2f} "
              f"ms/frame)")
        return out_path

    # ------------------------------------------------------------------
    def _pass_player_from_source(self, progress_cb=None) -> Dict[int, Dict]:
        """Pass A without a band proxy: decode the source and cut the band.

        The fallback path for a machine with no ffmpeg, or a transcode that
        did not come back frame-exact.  Same model call, same coordinates —
        only the decode is expensive.
        """
        cfg = self.cfg
        t0 = time.perf_counter()
        rx1, ry1, rx2, ry2 = self.far_roi
        cap = open_video(self.video_path, "FAR-TELEM")
        samples: Dict[int, Dict] = {}
        pend_idx: List[int] = []
        pend_img: List[np.ndarray] = []

        def flush():
            if not pend_img:
                return
            res = self.player_model(pend_img, verbose=False,
                                    conf=cfg.far_roi_conf,
                                    imgsz=cfg.far_roi_imgsz, device=_DEVICE)
            for fi, r in zip(pend_idx, res):
                box, world, conf = self._far_player_from_result(r, (rx1, ry1))
                samples[fi] = {"box": box, "world": world, "conf": conf}
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
                crop = frame[ry1:ry2, rx1:rx2]
                if crop.size == 0:
                    continue
                pend_idx.append(idx)
                pend_img.append(np.ascontiguousarray(crop))
                if len(pend_img) >= cfg.batch_size:
                    flush()
                    if progress_cb:
                        progress_cb(idx, self.total_frames)
            flush()
        finally:
            cap.release()

        assert_decode_complete("FAR-TELEM pass A (from source)",
                               self.video_path, idx,
                               self.total_frames - 1, self.fps)

        self.timings["pass_player_s"] = time.perf_counter() - t0
        print(f"[FAR-TELEM] pass A (from source): {len(samples)} player samples "
              f"({self.timings['pass_player_s']:.1f}s)")
        return samples


def extract_far_telemetry(video_path: str, force: bool = False,
                          cfg: Optional[FarExtractorConfig] = None,
                          out_path: Optional[str] = None,
                          progress_cb=None) -> str:
    """Extract (or reuse cached) far-only telemetry. Returns the JSONL path.

    The sibling pose cache is written by the same pass and is only valid
    together with it, so a cache hit requires both files.
    """
    out_path = out_path or far_telemetry_path_for(video_path)
    if not force and os.path.isfile(out_path):
        try:
            with open(out_path) as fh:
                ver = int(json.loads(fh.readline()).get("meta", {}).get("version", 0))
        except Exception:
            ver = 0
        if ver == FAR_TELEMETRY_VERSION and os.path.isfile(far_pose_path_for(out_path)):
            print(f"[FAR-TELEM] Using cached far telemetry: {out_path}")
            return out_path
    return FarTelemetryExtractor(video_path, cfg=cfg).extract(
        out_path=out_path, progress_cb=progress_cb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Far-serve-only telemetry: native-resolution band proxy, "
                    "5 fps far player, pose while armed, gated ball")
    ap.add_argument("video")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-band-proxy", action="store_true",
                    help="Crop the band out of the source instead of transcoding it")
    ap.add_argument("--no-ball-gate", action="store_true",
                    help="Run the ball pass over the whole clip, not just where "
                         "a point could be live")
    ap.add_argument("--player-fps", type=float, default=None)
    ap.add_argument("--ball-fps", type=float, default=None)
    ap.add_argument("--pose-imgsz", type=int, default=None)
    ap.add_argument("--band-crf", type=int, default=None)
    ap.add_argument("--crop-mode", choices=("pad", "resize"), default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c = FarExtractorConfig()
    if args.no_band_proxy:
        c.use_band_proxy = False
    if args.no_ball_gate:
        c.gate_ball = False
    if args.player_fps is not None:
        c.player_fps = args.player_fps
    if args.ball_fps is not None:
        c.ball_fps = args.ball_fps
    if args.pose_imgsz is not None:
        c.pose_imgsz = args.pose_imgsz
    if args.band_crf is not None:
        c.band_crf = args.band_crf
    if args.crop_mode is not None:
        c.crop_mode = args.crop_mode
    if args.batch is not None:
        c.batch_size = args.batch
    extract_far_telemetry(args.video, force=args.force, cfg=c, out_path=args.out)
