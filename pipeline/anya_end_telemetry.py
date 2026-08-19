"""
anya_end_telemetry.py
=====================
The point-END sibling of anya_near_telemetry / anya_far_telemetry: everything
the dead-time signals need, off the shared 540p proxy instead of the 4K source.

Point end has two consumers and they used to pay for two full-rate passes
between them:

    near-player walking   walking.predict, on a pose pass over every frame of
                          the 4K source at imgsz 960          (14.3 ms/frame)
    ball quiet            rally_reel._ball_quiet_onsets, on `all_balls` from
                          the shared anya_telemetry pass       (~18 ms/frame
                          of ball inference plus 4K decode)

Both are read as *time-coarse* evidence — "is the near player walking", "has
the ball been silent for 1.5 s" — and neither needs 30 fps of 4K.  So this
module reads the proxy and runs each model at its own stride, in two sequential
passes (see `interleave` for why not one):

  1. Shared 540p proxy (pipeline/proxy.py, CRF 14).  4K decode is ~6.7 ms of
     which ~4.1 ms is reconstruction that cannot be skipped, so the win is
     decoding something smaller, not less often.  The near-serve fast path
     already builds this file, so on a default run it costs nothing.

  2. Pose at `pose_fps` (15 by default), NOT every frame.  walking.predict
     already scores at 15 Hz — it took every 2nd frame of a 30 fps clip — so
     the pose beneath it was being extracted at twice the rate anything read.
     15 Hz is also above Nyquist for the 0.7-4.0 Hz cadence band the features
     measure; 7.5 Hz is not, which is why this stops at 15 and does not go
     lower.  On 60 fps clips it is a 4x reduction rather than 2x.

     Unsampled frames are DROPPED, not held: DESIGN.md 8.3 measured a
     zero-order hold inflating a differentiated feature into spurious events.
     The npz is written decimated with `fps` set to the effective rate, so
     walking/select_near.py and walking/features.py — both of which take fps
     as a parameter — see a consistent, slower clip and need no changes.

  3. Ball at `ball_fps` (10 by default), full frame.  Unlike the near path's
     toss ROI there is no crop to make: a rally ball is anywhere on court.
     Quiet is a claim about a 1.5 s window, so it is the sample RATE that has
     to be defended, not the per-frame rate — see `ball_stride`/`ball_frames`
     in the meta, which anya_far_serve.ball_sampling_scales already uses to
     rescale its static-blob thresholds for a sparser stream.

`npw` (near player, court feet) comes from the pose pass rather than a third
model: the same near-player rule anya_telemetry applies, over the person boxes
pose already produced.  That is what `_near_blind_mask` gates ball-quiet on, so
the gate now reads the walking classifier's own input instead of a second,
independently-failing player track.

Two outputs, both keyed by SOURCE frame index:

    <stem>_anya_end_telemetry.jsonl   meta header, then per-frame records
        f, t          source frame index and timestamp
        pn            true if this frame carries a pose sample
        np, npw       near-player box (analysis px) and world feet, or null
        bn            true if this frame carries a ball sample
        all_balls     whole-court detections [[cx, cy, conf], ...], unfiltered,
                      analysis coords — same schema as anya_telemetry

    <stem>_end_walk_dets.npz          walking/extract_pose.py's schema, plus
        stride, src_fps               decimated to `pose_fps`

Run:
    python -m pipeline.anya_end_telemetry match.mp4 [--force]
"""

import argparse
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
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

_MODELS_DIR = Path(__file__).parent / "models"

END_TELEMETRY_SUFFIX = "_anya_end_telemetry.jsonl"
END_DETS_SUFFIX      = "_end_walk_dets.npz"
END_POSE_SUFFIX      = "_end_walk_pose.npz"
END_TELEMETRY_VERSION = 1

N_KP = 17
MAX_PERSONS = 8     # walking/extract_pose.py's slot count; the npz schema is
                    # shared with it, so this has to match or select_near sees
                    # a differently-shaped array.


@dataclass
class EndExtractorConfig:
    analysis_size: Tuple[int, int] = (960, 540)

    # --- proxy (shared with anya_near_telemetry; see proxy.ensure_proxy)
    use_proxy:    bool = True
    proxy_crf:    int  = 14
    proxy_preset: str  = "veryfast"

    # --- pose pass
    pose_fps:   float = 15.0
    pose_imgsz: int   = 960
    pose_conf:  float = 0.20     # walking/extract_pose.POSE_CONF
    # Absolute path, not the bare "yolov8n-pose.pt". Given a bare name
    # ultralytics looks in the CWD and then DOWNLOADS the weights from the
    # internet — so the packaged app ignored its own bundled copy and fetched
    # one at the start of every reel. That silently worked for anyone online
    # and failed outright for a tester who wasn't. _MODELS_DIR resolves under
    # sys._MEIPASS in the frozen app and to pipeline/models when run from
    # source, so both get the copy this build was tested against.
    pose_model: str   = str(_MODELS_DIR / "yolov8n-pose.pt")

    # Empty-frame rescue, OFF by default — matching the shipped path, which is
    # the only honest baseline to measure against.  walking/extract_pose.py has
    # a `rescue()` that re-runs empty frames at imgsz 1920 (blind frames 25.4%
    # -> 7.7% on Data/21, recovering a labelled walk from interval recall 0.00
    # -> 0.89), but `walking.predict` never calls it: it runs `extract()` and
    # stops.  So the rescue is a corpus-building step, not part of the reel.
    #
    # Turning it on here also costs more than it looks: at 1920 the activations
    # are ~4x a 960 batch, and batching 16 of them on MPS was measured spending
    # 20+ minutes on a clip whose main pass is ~2.  Hence the separate, much
    # smaller rescue batch.
    #
    # And off the 540p proxy there are no extra pixels to find — this re-runs
    # the SAME pixels upscaled.  If the proxy turns out to lose blind frames
    # against a source decode, the fix is a native-resolution near-band crop
    # proxy (proxy.ensure_crop_proxy), not a bigger imgsz.
    rescue_empty: bool = False
    rescue_imgsz: int  = 1920
    rescue_batch: int  = 4

    # --- near-player pick (mirrors anya_telemetry's near branch)
    near_min_conf: float = 0.5

    # --- ball pass
    ball_fps:   float = 10.0
    ball_imgsz: int   = 960
    ball_conf:  float = Config.ACTIVE_BALL_CONF

    batch_size: int = 16
    prefetch:   bool = True

    interleave: bool = True
    # Run both models over ONE decode (True) or one after the other over two
    # (False).  The obvious guess was that alternating models on MPS would cost
    # more than a second 540p decode; measured on Data/21 it is the other way
    # round — interleaved 12.84 ms/frame, sequential 14.11 — so the second
    # decode is the more expensive half and one decode stays the default.
    #
    # What that A/B does NOT explain is why each call is ~50% over its
    # reference cost either way (pose 18-19 ms/sample against 12.0 in the
    # batched full-rate pass, ball 17.5-19.6 against 11.4).  The two arms ran
    # against different background load, so treat both absolute numbers as
    # provisional until they are re-measured on an idle machine — the ranking
    # is what this flag records, not the magnitudes.


def end_telemetry_path_for(video_path: str,
                           ball_fps: Optional[float] = None) -> str:
    """Cache path, keyed by ball rate when it is not the default.

    The trace policy needs a denser ball stream than anything else does, and
    re-extracting on every policy flip would make an A/B unaffordable.  Keying
    the non-default rate into the filename lets both live on disk at once.
    """
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    base = os.path.join(d, f"{stem}{END_TELEMETRY_SUFFIX}")
    if ball_fps is None or abs(float(ball_fps) - EndExtractorConfig().ball_fps) < 1e-6:
        return base
    return base[:-len(".jsonl")] + f"_b{int(round(float(ball_fps)))}.jsonl"


def end_dets_path_for(video_path: str) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{END_DETS_SUFFIX}")


def end_pose_path_for(video_path: str) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{END_POSE_SUFFIX}")


def ensure_proxy(video_path: str, size: Tuple[int, int] = (960, 540),
                 crf: int = 14, preset: str = "veryfast",
                 force: bool = False) -> str:
    return _ensure_proxy(video_path, size=size, crf=crf, preset=preset,
                         force=force, label="END-TELEM")


class EndTelemetryExtractor:
    """One decode of the proxy; pose at `pose_fps` and ball at `ball_fps`."""

    def __init__(self, video_path: str,
                 cfg: Optional[EndExtractorConfig] = None):
        self.video_path = video_path
        self.cfg = cfg or EndExtractorConfig()

        info = probe_video(video_path)
        self.fps          = info["fps"]
        self.total_frames = int(info["frame_count"])
        self.source_size  = (info["width"], info["height"])

        # Strides are derived from the source rate, so a 60 fps clip samples
        # the same NUMBER OF TIMES PER SECOND as a 30 fps one.  Everything
        # downstream is time-based, and this is what makes the two comparable.
        self.pose_stride = max(1, int(round(self.fps / self.cfg.pose_fps)))
        self.ball_stride = max(1, int(round(self.fps / self.cfg.ball_fps)))

        self.pose_model = YOLO(self.cfg.pose_model)
        self.ball_model = YOLO(str(_MODELS_DIR / "ball_best.pt"))

        self.court_vertices, _ = init_court(video_path,
                                            analysis_size=self.cfg.analysis_size)
        self.H = self._compute_homography()

        cached = load_cached_exclusion_zones(video_path)
        if cached is not None:
            self.exclusion_zones = self._to_analysis_coords(cached)
        else:
            print("[END-TELEM] Scanning for static exclusion zones…")
            try:
                self.exclusion_zones = create_auto_exclusion_zones(
                    video_path, self.ball_model,
                    num_frames=50, conf=0.04, eps=12, padding=8,
                    ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
                    analysis_size=self.cfg.analysis_size,
                )
                save_cached_exclusion_zones(video_path, self.exclusion_zones)
            except Exception as e:
                print(f"[END-TELEM] WARN: exclusion-zone scan failed: {e}")
                self.exclusion_zones = []

        self.timings: Dict[str, float] = {}
        self.n_empty = 0
        self.n_rescued = 0

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
    def _near_from_boxes(self, boxes, confs):
        """Near player from person boxes: the eligible one nearest the near
        baseline, exactly as anya_telemetry._players_from_result picks it."""
        L   = Config.COURT_LENGTH_FT
        pad = Config.NEAR_PLAYER_X_PAD_FT
        best = None
        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            if conf < self.cfg.near_min_conf:
                continue
            wx, wy = self._world((x1 + x2) / 2.0, y2)
            if not (abs(wy) < abs(wy - L)):          # far half — not our player
                continue
            if not (-pad <= wx <= Config.COURT_WIDTH_FT + pad):
                continue
            if best is None or abs(wy) < abs(best[2]):
                best = ((int(x1), int(y1), int(x2), int(y2)), wx, wy)
        if best is None:
            return None, None
        return best[0], (round(best[1], 2), round(best[2], 2))

    def _balls_from_result(self, result) -> List[Tuple[float, float, float]]:
        if result is None or not result.boxes:
            return []
        out = []
        for b in result.boxes:
            if int(b.cls[0]) != Config.DEFAULT_BALL_CLASS_INDEX:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((round((x1 + x2) / 2.0, 1), round((y1 + y2) / 2.0, 1),
                        round(float(b.conf[0]), 3)))
        return out

    # ------------------------------------------------------------------
    def _frames(self, src: str, strides: Optional[Tuple[int, ...]] = None):
        """Yield (idx, frame) for frames any of `strides` wants, in order.

        Frames no pass wants are grabbed and dropped: grab() still has to
        reconstruct them (H.264 leaves no choice) but skips the YUV->BGR
        conversion and the copy into numpy, which is the part worth avoiding.
        """
        strides = strides or (self.pose_stride, self.ball_stride)
        cap = open_video(src, "END-TELEM")
        aw, ah = self.cfg.analysis_size
        idx = -1
        try:
            while True:
                if not cap.grab():
                    break
                idx += 1
                if all(idx % s for s in strides):
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if (frame.shape[1], frame.shape[0]) != (aw, ah):
                    frame = cv2.resize(frame, (aw, ah),
                                       interpolation=cv2.INTER_AREA)
                yield idx, frame
        finally:
            cap.release()
        # After the loop, not in `finally`: a consumer that stops early (or
        # raises) closes the generator here, and its own reason for stopping
        # is the one worth reporting. Reaching this line means the decode
        # itself ended, so a short read is the decoder's.
        assert_decode_complete("END-TELEM", src, idx,
                               self.total_frames - 1, self.fps)

    def _prefetched(self, src: str, strides: Optional[Tuple[int, ...]] = None,
                    depth: int = 4):
        """`_frames` on a reader thread, so decode overlaps inference.

        Measured worth 1.38x on the walking pose pass alone (DESIGN.md 8, the
        b89a4f6 arm): decode is CPU and inference is GPU, and they were running
        strictly one after the other.  Order is preserved (one reader, one
        queue) and exceptions cross the queue rather than vanishing.
        """
        q: "queue.Queue" = queue.Queue(maxsize=depth)
        SENTINEL = object()

        def _read():
            try:
                for item in self._frames(src, strides):
                    q.put(item)
            except BaseException as ex:
                q.put(ex)
            finally:
                q.put(SENTINEL)

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            t.join(timeout=5.0)

    # ------------------------------------------------------------------
    def _run_pose(self, idxs: List[int], imgs: List[np.ndarray],
                  kp, bx, cf, near: Dict[int, Dict]) -> None:
        cfg = self.cfg
        results = self.pose_model.predict(imgs, imgsz=cfg.pose_imgsz,
                                          conf=cfg.pose_conf, device=_DEVICE,
                                          classes=[0], verbose=False)
        rescue_idx, rescue_img = [], []
        for (fi, img), res in zip(zip(idxs, imgs), results):
            if res.keypoints is None or len(res.boxes) == 0:
                self.n_empty += 1
                if cfg.rescue_empty:
                    rescue_idx.append(fi)
                    rescue_img.append(img)
                continue
            self._store_pose(fi, res, kp, bx, cf, near)

        for lo in range(0, len(rescue_idx), cfg.rescue_batch):
            # Same pixels, larger letterbox — in small batches, because a 1920
            # batch of 16 does not fit MPS gracefully.
            chunk_i = rescue_idx[lo:lo + cfg.rescue_batch]
            res2 = self.pose_model.predict(rescue_img[lo:lo + cfg.rescue_batch],
                                           imgsz=cfg.rescue_imgsz,
                                           conf=cfg.pose_conf, device=_DEVICE,
                                           classes=[0], verbose=False)
            for fi, res in zip(chunk_i, res2):
                if res.keypoints is None or len(res.boxes) == 0:
                    continue
                self.n_rescued += 1
                self._store_pose(fi, res, kp, bx, cf, near)

    def _store_pose(self, fi: int, res, kp, bx, cf, near: Dict[int, Dict]) -> None:
        row = fi // self.pose_stride
        k = res.keypoints.data.cpu().numpy()          # [P, 17, 3]
        b = res.boxes.xyxy.cpu().numpy()              # [P, 4]
        c = res.boxes.conf.cpu().numpy()              # [P]
        order = np.argsort(-c)[:MAX_PERSONS]
        for slot, i in enumerate(order):
            kp[row, slot] = k[i]
            bx[row, slot] = b[i]
            cf[row, slot] = c[i]
        box, world = self._near_from_boxes(b[order], c[order])
        near[fi] = {"box": box, "world": world}

    # ------------------------------------------------------------------
    def extract(self, out_path: Optional[str] = None,
                dets_path: Optional[str] = None, progress_cb=None) -> str:
        cfg = self.cfg
        out_path = out_path or end_telemetry_path_for(self.video_path)
        dets_path = dets_path or end_dets_path_for(self.video_path)
        tmp_path = out_path + ".part"
        wall0 = time.perf_counter()

        t0 = time.perf_counter()
        src = (ensure_proxy(self.video_path, cfg.analysis_size,
                            cfg.proxy_crf, cfg.proxy_preset)
               if cfg.use_proxy else self.video_path)
        self.timings["proxy_s"] = time.perf_counter() - t0

        n_rows = (self.total_frames + self.pose_stride - 1) // self.pose_stride
        kp = np.full((n_rows, MAX_PERSONS, N_KP, 3), np.nan, dtype=np.float32)
        bx = np.full((n_rows, MAX_PERSONS, 4), np.nan, dtype=np.float32)
        cf = np.full((n_rows, MAX_PERSONS), np.nan, dtype=np.float32)
        near: Dict[int, Dict] = {}
        balls: Dict[int, List] = {}

        pose_idx: List[int] = []
        pose_img: List[np.ndarray] = []
        ball_idx: List[int] = []
        ball_img: List[np.ndarray] = []
        t_pose = t_ball = 0.0

        def flush_pose():
            nonlocal t_pose
            if not pose_img:
                return
            t = time.perf_counter()
            self._run_pose(pose_idx, pose_img, kp, bx, cf, near)
            t_pose += time.perf_counter() - t
            pose_idx.clear()
            pose_img.clear()

        def flush_ball():
            nonlocal t_ball
            if not ball_img:
                return
            t = time.perf_counter()
            res = self.ball_model(ball_img, verbose=False, conf=cfg.ball_conf,
                                  imgsz=cfg.ball_imgsz, device=_DEVICE)
            for fi, r in zip(ball_idx, res):
                balls[fi] = self._balls_from_result(r)
            t_ball += time.perf_counter() - t
            ball_idx.clear()
            ball_img.clear()

        def stream(strides):
            return (self._prefetched(src, strides) if cfg.prefetch
                    else self._frames(src, strides))

        def tick(idx):
            if progress_cb:
                progress_cb(idx, self.total_frames)
            el = time.perf_counter() - t0
            print(f"[END-TELEM]   {idx}/{self.total_frames}  "
                  f"{idx / max(1e-9, el):.0f} src-fps  "
                  f"empty {self.n_empty / max(1, len(near) + self.n_empty):.1%}",
                  flush=True)

        t0 = time.perf_counter()
        if cfg.interleave:
            for idx, frame in stream((self.pose_stride, self.ball_stride)):
                if idx % self.pose_stride == 0:
                    pose_idx.append(idx)
                    pose_img.append(frame)
                    if len(pose_img) >= cfg.batch_size:
                        flush_pose()
                        if idx // self.pose_stride % 1000 < cfg.batch_size:
                            tick(idx)
                if idx % self.ball_stride == 0:
                    ball_idx.append(idx)
                    ball_img.append(frame)
                    if len(ball_img) >= cfg.batch_size:
                        flush_ball()
            flush_pose()
            flush_ball()
        else:
            # Two passes, each with one hot model.  Costs a second decode of
            # the proxy (~1.2 ms/frame) and saves alternating between two
            # models on MPS, which measured far more than that — see
            # DESIGN.md 8.6.
            for idx, frame in stream((self.pose_stride,)):
                pose_idx.append(idx)
                pose_img.append(frame)
                if len(pose_img) >= cfg.batch_size:
                    flush_pose()
                    if idx // self.pose_stride % 1000 < cfg.batch_size:
                        tick(idx)
            flush_pose()
            for idx, frame in stream((self.ball_stride,)):
                ball_idx.append(idx)
                ball_img.append(frame)
                if len(ball_img) >= cfg.batch_size:
                    flush_ball()
                    if idx // self.ball_stride % 2000 < cfg.batch_size:
                        tick(idx)
            flush_ball()
        self.timings["pass_s"] = time.perf_counter() - t0
        self.timings["pose_infer_s"] = t_pose
        self.timings["ball_infer_s"] = t_ball

        n_pose = len(near)
        n_blind = n_pose - sum(1 for v in near.values() if v["box"] is not None)
        print(f"[END-TELEM] pose: {n_pose} samples @{self.fps / self.pose_stride:.2f} "
              f"fps, {self.n_empty} empty ({self.n_empty / max(1, n_pose):.1%}), "
              f"{self.n_rescued} rescued, {n_blind} with no near player "
              f"({t_pose:.1f}s)")
        print(f"[END-TELEM] ball: {len(balls)} samples "
              f"@{self.fps / self.ball_stride:.2f} fps, "
              f"{sum(1 for v in balls.values() if v)} with a detection "
              f"({t_ball:.1f}s)")

        np.savez_compressed(dets_path, kp=kp, box=bx, conf=cf,
                            fps=np.float64(self.fps / self.pose_stride),
                            src_fps=np.float64(self.fps),
                            stride=np.int64(self.pose_stride),
                            n_src_frames=np.int64(self.total_frames))
        print(f"[END-TELEM] pose dets → {dets_path}")

        n = 0
        with open(tmp_path, "w") as fh:
            meta = {
                "version":       END_TELEMETRY_VERSION,
                "source":        "anya_end_telemetry",
                "video":         os.path.basename(self.video_path),
                "proxy":         os.path.basename(src) if src != self.video_path else None,
                "fps":           self.fps,
                "total_frames":  self.total_frames,
                "stride":        1,
                # ball_stride/ball_frames are read by
                # anya_far_serve.ball_sampling_scales to rescale the
                # static-blob thresholds for this sparser stream.
                "ball_stride":   self.ball_stride,
                "ball_frames":   len(balls),
                "ball_fps":      self.fps / self.ball_stride,
                "ball_imgsz":    cfg.ball_imgsz,
                "ball_conf":     cfg.ball_conf,
                "pose_stride":   self.pose_stride,
                "pose_fps":      self.fps / self.pose_stride,
                "pose_imgsz":    cfg.pose_imgsz,
                "pose_empty":    self.n_empty,
                "pose_rescued":  self.n_rescued,
                "interleave":    cfg.interleave,
                "dets_npz":      os.path.basename(dets_path),
                "analysis_size": list(cfg.analysis_size),
                "source_size":   list(self.source_size),
                "court_length_ft": Config.COURT_LENGTH_FT,
                "court_width_ft":  Config.COURT_WIDTH_FT,
                "exclusion_zones": [list(z) for z in self.exclusion_zones],
            }
            fh.write(json.dumps({"meta": meta}) + "\n")

            for idx in sorted(set(near) | set(balls)):
                s = near.get(idx)
                rec = {
                    "f":  idx,
                    "t":  round(idx / self.fps, 4),
                    "pn": s is not None,
                    "np": (list(s["box"]) if s and s["box"] else None),
                    "npw": (list(s["world"]) if s and s["world"] else None),
                    "bn": idx in balls,
                }
                if idx in balls:
                    rec["all_balls"] = [list(b) for b in balls[idx]]
                fh.write(json.dumps(rec) + "\n")
                n += 1

        os.replace(tmp_path, out_path)
        self.timings["total_s"] = time.perf_counter() - wall0
        ms = 1000.0 * self.timings["pass_s"] / max(1, self.total_frames)
        print(f"[END-TELEM] Wrote {n} records → {out_path}")
        print(f"[END-TELEM] timing: proxy {self.timings['proxy_s']:.1f}s + "
              f"pass {self.timings['pass_s']:.1f}s "
              f"(pose {t_pose:.1f}s + ball {t_ball:.1f}s + decode) = "
              f"{self.timings['total_s']:.1f}s total, "
              f"{ms:.2f} ms/source-frame steady state")
        return out_path


def _rate_matches(out_path: str, cfg: Optional["EndExtractorConfig"]) -> bool:
    """Is the cached file's ball rate the one this run asked for?

    The version gate alone compares only the schema, so a file built with
    `--ball-fps 30` and a caller wanting 10 (or the reverse) read as a hit.
    Harmless until the trace policy made two rates coexist; now a silent reuse
    would hand the tracker a stream it cannot confirm on, or make a cheap run
    pay for a dense one.
    """
    want = (cfg or EndExtractorConfig()).ball_fps
    try:
        with open(out_path) as fh:
            meta = json.loads(fh.readline()).get("meta", {})
        fps, stride = float(meta["fps"]), int(meta["ball_stride"])
    except Exception:
        return True                  # unreadable meta is the version gate's problem
    return abs(fps / max(1, stride) - fps / max(1, round(fps / want))) < 0.05


def extract_end_telemetry(video_path: str, force: bool = False,
                          cfg: Optional[EndExtractorConfig] = None,
                          out_path: Optional[str] = None,
                          progress_cb=None) -> str:
    """Extract (or reuse cached) point-end telemetry. Returns the JSONL path.

    The pose npz is written by the same pass and is only valid together with
    it, so a cache hit requires both files.
    """
    out_path = out_path or end_telemetry_path_for(video_path)
    dets = end_dets_path_for(video_path)
    if not force and os.path.isfile(out_path):
        try:
            with open(out_path) as fh:
                ver = int(json.loads(fh.readline()).get("meta", {}).get("version", 0))
        except Exception:
            ver = 0
        if ver == END_TELEMETRY_VERSION and os.path.isfile(dets) and _rate_matches(out_path, cfg):
            print(f"[END-TELEM] Using cached end telemetry: {out_path}")
            return out_path
    return EndTelemetryExtractor(video_path, cfg=cfg).extract(
        out_path=out_path, progress_cb=progress_cb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Point-end telemetry: shared 540p proxy, 15 fps pose, "
                    "10 fps whole-court ball")
    ap.add_argument("video")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-proxy", action="store_true",
                    help="Decode the source directly instead of the 540p proxy")
    ap.add_argument("--rescue", action="store_true",
                    help="Re-run empty pose frames at imgsz 1920.  Off by "
                         "default because walking.predict does not do it "
                         "either, and off a 540p proxy there are no extra "
                         "pixels to find")
    ap.add_argument("--pose-fps", type=float, default=None)
    ap.add_argument("--pose-imgsz", type=int, default=None)
    ap.add_argument("--ball-fps", type=float, default=None)
    ap.add_argument("--ball-imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--sequential", action="store_true",
                    help="Run the two models in separate passes over their own "
                         "decode, instead of interleaved over one (measured "
                         "slower: 14.11 vs 12.84 ms/frame on Data/21)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c = EndExtractorConfig()
    if args.no_proxy:
        c.use_proxy = False
    if args.rescue:
        c.rescue_empty = True
    if args.pose_fps is not None:
        c.pose_fps = args.pose_fps
    if args.pose_imgsz is not None:
        c.pose_imgsz = args.pose_imgsz
    if args.ball_fps is not None:
        c.ball_fps = args.ball_fps
    if args.ball_imgsz is not None:
        c.ball_imgsz = args.ball_imgsz
    if args.batch is not None:
        c.batch_size = args.batch
    if args.sequential:
        c.interleave = False
    extract_end_telemetry(args.video, force=args.force, cfg=c, out_path=args.out)
