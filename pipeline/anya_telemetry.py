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

v3 dropped `fp`/`fpw` (the far player as seen by the full-frame pass).  No
consumer of THIS cache ever read them — anya_far_serve arms against `fprw`
from the native-resolution ROI pass, and anya_near_serve / rally_reel read
only the near player.  (point_segmenter.py does read `fp`/`fpw`, but from
match_telemetry.py's separate v4 cache, which is untouched.)  Dropping them
is what lets the full-frame player call shrink to the near-court band.

First line of the file is a meta header: {"meta": {...}}.  It carries
`exclusion_zones` — the auto-detected static false-ball boxes as
[x1, y1, x2, y2] in analysis coords — so a consumer can apply the filter
itself against the raw `all_balls` stream.

Optional sidecar: `<stem>_walk_pose.npz`, the near-player pose track the
walking classifier consumes, produced from this decode pass instead of
walking/'s own.  OFF by default — it did not pay for itself; the numbers are
on ExtractorConfig.walk_pose.

That attempt is worth reading before trying anything similar here.  Three
successive versions were each faster and each wrong, and none of the errors
were visible in review — only against walking/labels_snippet21.json:

    player model + 256px crop pose     F1 0.997 -> 0.820
    band pose, no high-res recovery    F1 0.997 -> 0.947
    band pose + banded recovery        F1 0.997 -> 0.959

The models downstream of this file are fitted to what it feeds them, so the
detector, its input scale, and even the presence of a recovery pass are tuned
hyperparameters, not efficiency knobs.  Score an input-side change on Data/21
before believing it.

Run:
    python -m pipeline.anya_telemetry match.mp4 [--force] [--stride N]
"""

import argparse
import json
import os
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
TELEMETRY_VERSION = 3   # v2 added source_size + the native-resolution far-player
                        # ROI pass (`fpr`/`fprw`); v1 files lack both.  v3 drops
                        # the unread `fp`/`fpw`, confines the player pass to the
                        # near-court band, and emits the walking pose sidecar.
WALK_POSE_SUFFIX  = "_walk_pose.npz"

# COCO-17 indices used by the near-player selection (ankles) — the same
# keypoints walking/select_near.py uses for its foot point.
L_ANKLE, R_ANKLE = 15, 16
N_KP = 17

# Near-player eligibility, mirroring walking/select_near.py's zone rules but
# expressed in FEET (this module's homography is in feet; walking/court.py's is
# in metres over the identical 27 x 78 ft singles rectangle).  Keeping the
# surrounding floor eligible matters: players walk off court to the ball carts
# and those walks are exactly the walking label, so a court-only gate would
# drop the positives it most needs to keep.
_M_TO_FT          = 1.0 / 0.3048
NEAR_ZONE_MAX_FT  = 12.5 * _M_TO_FT   # 41.0 ft — beyond this is the opponent
NEAR_MARGIN_FT    = 6.0  * _M_TO_FT   # 19.7 ft of slack around the near half
WALK_MIN_BOX_H_PX = 25.0              # smaller is a spectator on another court
WALK_MAX_SPEED_FS = 9.0  * _M_TO_FT   # 29.5 ft/s — faster is a different human
WALK_LOST_FRAMES  = 60                # keep the continuity anchor this long
WALK_CONT_SCALE_FT = 2.0 * _M_TO_FT   # continuity bonus falls off over 6.6 ft
WALK_KP_CONF      = 0.3


@dataclass
class ExtractorConfig:
    analysis_size: Tuple[int, int] = (960, 540)
    player_conf:    float = 0.2    # detection floor (far player needs the low floor)
    near_min_conf:  float = 0.5    # near-player candidates must clear this
    ball_conf:      float = Config.ACTIVE_BALL_CONF

    # Ball + player inference run RECTANGULAR.  A square imgsz letterboxes the
    # 16:9 analysis frame to 960x960, so ~44% of every forward pass was grey
    # padding.  Ultralytics accepts an (h, w) imgsz and pads only to the stride
    # of 32, so (544, 960) covers the same 960x540 pixels.
    #
    # Verified bit-identical, not just equivalent: over sampled frames of
    # Data/21 the ball model returned the same 113 detections with a maximum
    # centre displacement of 0.00 px, and the same 3682 detections over a
    # 400-frame run.  Measured 28.7 -> 21.6 ms/frame (1.33x) — well short of
    # the 1.76x the pixel count implies, because fixed per-call overhead
    # (preprocess, NMS, postprocess) dominates a nano model on MPS.
    ball_imgsz:  Tuple[int, int] = (544, 960)   # was 960 square; the 1920 pass
                                                # costs ~2.5x wall clock for
                                                # recall our consumers don't need
    player_imgsz_pad: int = 32     # stride the near-band crop height rounds up to

    # Near-player band (item 3).  The player pass used to run on the whole
    # frame to classify BOTH sides; nothing reads its far-player output any
    # more, so it is confined to the region that can hold a near player.  The
    # band top is derived from court geometry — the image row where world
    # y = NEAR_ZONE_MAX_FT projects — rather than a blind "bottom half", which
    # would cut off a player standing at the net (their feet are near the
    # halfway row but their head is well above it).
    near_band_head_frac: float = 0.45  # extra height above that row, as a
                                       # fraction of the band below it, to hold
                                       # the tallest plausible standing player
    near_band_min_frac:  float = 0.55  # floor on band height (frac of frame) so
                                       # an odd homography cannot starve it

    # Near-player pose for the walking classifier.  Same model and confidence
    # as walking/extract_pose.py, run on the near band instead of the whole
    # frame — so the pixels it sees are unrescaled, just fewer rows.  That
    # fidelity is the point: see _detect_walk_persons for what happened when
    # this was a cheaper small-crop call instead.
    # OFF by default, and the measurements are why.  Folding this in was meant
    # to delete walking/'s separate decode.  Done faithfully it does not pay:
    #
    #   model work   band pose 14.0 ms + rescue 53 ms on ~6.5% of frames
    #                ~= 17.4 ms, against 17.0 ms for the full-frame pass it
    #                replaces.  A controlled back-to-back run over 500 frames
    #                put new stage1 at 0.88x old stage1+stage5 — a wash at best.
    #   decode       the whole prize, and it is 6.0 ms/frame (measured), not
    #                the tens of ms the three-pass structure suggests.
    #   accuracy     walking F1 0.997 -> 0.959 on labels_snippet21 (recall
    #                0.997 -> 0.952), coverage 83.8% -> 82.5%.
    #
    # Paying ~4 pp of walking F1 for ~6 ms/frame is the wrong side of the
    # trade for a dead-time cutter, where a missed walk leaves dead time in
    # the cut.  Kept, documented and switchable rather than deleted: if the
    # walking model is ever retrained on band+rescue pose it becomes
    # self-consistent and this turns into a free decode.  Turning it on
    # without that retrain reintroduces the regression above.
    walk_pose:        bool  = False
    walk_pose_conf:   float = 0.20   # POSE_CONF from walking/extract_pose.py

    # High-resolution recovery, the inline equivalent of extract_pose.rescue().
    # A player walking to the ball carts is ~60 px tall in the analysis frame
    # and yolov8n-pose misses them outright there; at 1920 px they are found
    # reliably.  Those frames are not random — they are the mid-clip break,
    # which is almost entirely walking — so skipping them costs recall exactly
    # where it hurts.  Leaving this pass out cost F1 0.997 -> 0.947 on
    # labels_snippet21; with it, the band pass alone recovers 0/50 sampled
    # misses and this recovers 50/50.
    #
    # Fires only when the cheap band pass finds NOBODY (the same trigger the
    # standalone rescue used), so it costs ~53 ms on the ~20% of frames that
    # need it rather than on every frame.  Cropping the band before upscaling
    # rather than upscaling the whole frame halves it (53 vs 101 ms).
    walk_rescue:       bool = True
    walk_rescue_width: int  = 1920

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


def walk_pose_path_for(video_path: str) -> str:
    """The walking-classifier pose sidecar. Matches walking.select_near.pose_path
    exactly, so walking/predict.py finds this file with no change to its lookup."""
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_stem}{WALK_POSE_SUFFIX}")


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
        self.pose_model   = (YOLO(str(_MODELS_DIR / "yolov8n-pose.pt"))
                             if self.cfg.walk_pose else None)

        # Court geometry / homography — reuses the same disk caches as the
        # existing pipeline, prompting interactively only on first run.
        self.court_vertices, _ = init_court(video_path, analysis_size=self.cfg.analysis_size)
        self.H = self._compute_homography()
        self.H_inv = np.linalg.inv(self.H)

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

        self.near_band = self._near_band()
        nb_h = self.near_band[3] - self.near_band[1]
        self.player_imgsz = self._band_imgsz(nb_h)
        self.walk_imgsz   = self.player_imgsz
        print(f"[ANYA-TELEM] near-player band (analysis px): {list(self.near_band)} "
              f"-> imgsz {self.player_imgsz}  "
              f"({nb_h / self.cfg.analysis_size[1]:.0%} of frame height)")

        # Rolling state for the walking-pose near-player track (select_near.py's
        # continuity anchor, carried frame to frame).
        self._walk_prev_world: Optional[Tuple[float, float]] = None
        self._walk_since: int = 10 ** 9

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

    def _near_band(self) -> Tuple[int, int, int, int]:
        """The analysis-frame band that can contain a near-side person.

        Derived from the court, not from a fixed fraction of the frame.  The
        deepest eligible foot point is world y = NEAR_ZONE_MAX_FT (just past
        the net — beyond it the person is the opponent and is rejected by both
        the near-player rule and walking's zone rule).  Projecting that world
        line back through the homography gives the image row where such a
        player's FEET sit; their head is well above it, so the band is padded
        upward by near_band_head_frac of its own height.

        Full width is kept deliberately: walking's eligible region reaches
        NEAR_MARGIN_FT past both sidelines (the ball carts and walkways), which
        is already off the left and right edges of the frame.
        """
        aw, ah = self.cfg.analysis_size
        xs = np.linspace(-NEAR_MARGIN_FT, Config.COURT_WIDTH_FT + NEAR_MARGIN_FT, 9)
        pts = np.array([[[float(x), NEAR_ZONE_MAX_FT]] for x in xs], dtype=np.float32)
        img = cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

        # Only rows actually inside the frame say anything about where to cut;
        # a sideline that projects off-frame can land anywhere, including
        # behind the camera on a near-degenerate homography.
        ys = [y for _, y in img if -ah <= y <= 2 * ah]
        y_feet = min(ys) if ys else ah * 0.5

        y_top = y_feet - self.cfg.near_band_head_frac * (ah - y_feet)
        # Never let the band shrink past the configured floor.
        y_top = min(y_top, ah * (1.0 - self.cfg.near_band_min_frac))
        return (0, max(0, int(y_top)), aw, ah)

    def _band_imgsz(self, band_h: int) -> Tuple[int, int]:
        """Rectangular (h, w) inference size for the near band, stride-aligned."""
        pad = self.cfg.player_imgsz_pad
        h = int(np.ceil(band_h / pad) * pad)
        return (h, self.cfg.analysis_size[0])

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
    def _detect_near_persons(self, frame):
        """Every person in the near band, in ANALYSIS coordinates.

        One rectangular player-model call on the band crop.  This replaces the
        old full-frame square call: it no longer classifies the far side, whose
        `fp`/`fpw` output nothing read (see the module docstring), and the far
        player still has its own native-resolution ROI pass.

        Returns [(x1, y1, x2, y2, wx, wy, conf), ...] with feet-world position.
        """
        bx1, by1, bx2, by2 = self.near_band
        crop = frame[by1:by2, bx1:bx2]
        if crop.size == 0:
            return []

        results = self.player_model(crop, verbose=False,
                                    conf=self.cfg.player_conf,
                                    imgsz=self.player_imgsz,
                                    device=_DEVICE)
        if not (results and results[0].boxes):
            return []

        cands = []
        for b in results[0].boxes:
            if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
                continue
            cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
            x1, y1 = int(cx1 + bx1), int(cy1 + by1)
            x2, y2 = int(cx2 + bx1), int(cy2 + by1)
            wx, wy = self._world((x1 + x2) / 2.0, y2)
            cands.append((x1, y1, x2, y2, wx, wy, float(b.conf[0])))
        return cands

    def _pick_near_player(self, cands):
        """The near player under the ORIGINAL strict rule: closest to the near
        baseline, inside the sidelines (+pad), over near_min_conf.

        Unchanged semantics — anya_near_serve and rally_reel read `np`/`npw`
        and neither should notice the band crop.
        """
        L   = Config.COURT_LENGTH_FT
        pad = Config.NEAR_PLAYER_X_PAD_FT
        near_cands = [
            c for c in cands
            if (c[6] >= self.cfg.near_min_conf and
                abs(c[5]) < abs(c[5] - L) and
                -pad <= c[4] <= Config.COURT_WIDTH_FT + pad)
        ]
        if not near_cands:
            return None, None
        near = min(near_cands, key=lambda c: abs(c[5]))
        return near[:4], (near[4], near[5])

    def _detect_walk_persons(self, frame):
        """Every person in the near band, WITH keypoints, in analysis coords.

        Deliberately the same yolov8n-pose model, at the same confidence, on
        unrescaled pixels that walking/extract_pose.py used full-frame.  Only
        the rows above the band are missing, and a person entirely above it is
        in the far half, which the zone rule rejects anyway.

        An earlier version of this tried to be cleverer — detect with the
        (already-running) player model and then run pose on the winner's own
        small crop.  It was faster and it was wrong: scored against
        walking/labels_snippet21.json the classifier fell from F1 0.997 to
        0.820, recall 0.997 to 0.739, because the general-purpose detector at
        a different input scale loses the near player ~5 pp more often and the
        losses concentrate off court, which is precisely where the walking
        label lives.  Detector and input scale are part of what the classifier
        was fitted to, not free efficiency knobs.

        Returns [(x1, y1, x2, y2, conf, kp[17,3]), ...].
        """
        if self.pose_model is None:
            return []
        bx1, by1, bx2, by2 = self.near_band
        crop = frame[by1:by2, bx1:bx2]
        if crop.size == 0:
            return []

        res = self.pose_model(crop, verbose=False, conf=self.cfg.walk_pose_conf,
                              imgsz=list(self.walk_imgsz), classes=[0],
                              device=_DEVICE)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return []

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kpts  = res.keypoints.data.cpu().numpy()
        out = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            k = kpts[i].astype(np.float32).copy()
            k[:, 0] += bx1
            k[:, 1] += by1
            out.append((float(x1 + bx1), float(y1 + by1),
                        float(x2 + bx1), float(y2 + by1), float(confs[i]), k))
        return out

    def _rescue_walk_persons(self, orig_frame):
        """Second look at the near band, upscaled, for players the analysis
        frame is too coarse to resolve.  See `walk_rescue` in ExtractorConfig.

        Works from the SOURCE frame, so the band is cropped at full resolution
        and then scaled to walk_rescue_width — the detail is real, not
        interpolated back in.  Keypoints come home to analysis coordinates.
        """
        aw, ah = self.cfg.analysis_size
        sw, sh = self.source_size
        sby1 = int(self.near_band[1] * sh / float(ah))
        crop = orig_frame[sby1:sh, 0:sw]
        if crop.size == 0:
            return []

        rw = self.cfg.walk_rescue_width
        rh = max(32, int(round(crop.shape[0] * rw / float(crop.shape[1]))))
        big = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_AREA)
        imgsz = [int(np.ceil(rh / 32) * 32), rw]

        res = self.pose_model(big, verbose=False, conf=self.cfg.walk_pose_conf,
                              imgsz=imgsz, classes=[0], device=_DEVICE)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return []

        # Resized-band pixels -> source pixels -> analysis pixels.
        fx = (sw / float(rw)) * (aw / float(sw))
        fy = (crop.shape[0] / float(rh)) * (ah / float(sh))
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kpts  = res.keypoints.data.cpu().numpy()
        y0 = sby1 * ah / float(sh)

        out = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            k = kpts[i].astype(np.float32).copy()
            k[:, 0] *= fx
            k[:, 1] = k[:, 1] * fy + y0
            out.append((float(x1 * fx), float(y1 * fy + y0),
                        float(x2 * fx), float(y2 * fy + y0), float(confs[i]), k))
        return out

    @staticmethod
    def _walk_foot(kp, box):
        """Foot point in analysis pixels — port of walking/select_near._foot.

        Ankle midpoint when confident, else the box bottom centre.  Using the
        ankles rather than the box bottom matters for the court projection: a
        clipped or loose box moves the projected position by feet.
        """
        la, ra = kp[L_ANKLE], kp[R_ANKLE]
        if la[2] > WALK_KP_CONF and ra[2] > WALK_KP_CONF:
            return 0.5 * (la[0] + ra[0]), 0.5 * (la[1] + ra[1])
        if la[2] > WALK_KP_CONF:
            return float(la[0]), float(la[1])
        if ra[2] > WALK_KP_CONF:
            return float(ra[0]), float(ra[1])
        return 0.5 * (box[0] + box[2]), float(box[3])

    def _pick_walk_person(self, persons):
        """The near-side person for the WALKING track — a deliberately wider
        rule than `_pick_near_player`.

        Ports walking/select_near.py's zone + size + continuity scoring into
        this loop.  The width matters and is not incidental: a player who walks
        off court to the ball carts is off-court, out of the near-player rule's
        sideline pad, and is exactly the walking positive the classifier is
        trained to catch.  Selecting on the strict `np` box instead would drop
        those frames in a label-correlated way.

        Returns (box, kp, on_court) or (None, None, None).
        """
        best, best_s = None, -np.inf
        best_zone = best_world = None
        for p in persons:
            x1, y1, x2, y2, conf, kp = p
            h = float(y2 - y1)
            if h < WALK_MIN_BOX_H_PX:
                continue
            fx, fy = self._walk_foot(kp, (x1, y1, x2, y2))
            wx, wy = self._world(fx, fy)
            zone = self._walk_zone(wx, wy)
            if zone == "far":
                continue
            s = 0.6 * conf + min(h, 250.0) / 250.0
            if zone == "near":
                s += 1.2
            if self._walk_prev_world is not None and self._walk_since <= WALK_LOST_FRAMES:
                d = float(np.hypot(wx - self._walk_prev_world[0],
                                   wy - self._walk_prev_world[1]))
                # Nobody covers 9 m/s: reject outright rather than score down,
                # so a penalised-but-still-best bystander cannot become the
                # track and teleport the trace.
                if d / max(self._walk_since, 1) * self.fps > WALK_MAX_SPEED_FS:
                    continue
                s += 2.0 * np.exp(-d / WALK_CONT_SCALE_FT)
            if s > best_s:
                best_s, best, best_zone, best_world = s, p, zone, (wx, wy)

        if best is None:
            self._walk_since += 1     # keep the anchor; re-acquire after LOST_FRAMES
            return None, None, None

        self._walk_prev_world = best_world
        self._walk_since = 1
        return best[:4], best[5], 1.0 if best_zone == "near" else 0.0

    @staticmethod
    def _walk_zone(wx: float, wy: float) -> str:
        """'far' (ineligible), 'near' (preferred), or 'off' (eligible, no bonus).
        Feet-unit mirror of walking/select_near.py's `_zone`."""
        pad_x = 4.0 * _M_TO_FT
        if (-pad_x <= wx <= Config.COURT_WIDTH_FT + pad_x and
                NEAR_ZONE_MAX_FT <= wy <= Config.COURT_LENGTH_FT + 8.0 * _M_TO_FT):
            return "far"
        if (-NEAR_MARGIN_FT <= wy < NEAR_ZONE_MAX_FT and
                -NEAR_MARGIN_FT <= wx <= Config.COURT_WIDTH_FT + NEAR_MARGIN_FT):
            return "near"
        return "off"

    # ------------------------------------------------------------------
    def _detect_balls(self, frame) -> List[Tuple[float, float, float]]:
        """Every ball detection on the whole analysis frame, unfiltered."""
        res = self.ball_model(frame, verbose=False, conf=self.cfg.ball_conf,
                              imgsz=list(self.cfg.ball_imgsz), device=_DEVICE)
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

        # Walking-pose sidecar, indexed by SOURCE frame so a partial or strided
        # run still lands each sample at the right index (walking/features.py
        # reads these as a dense per-frame series).
        want_walk = self.cfg.walk_pose and start_frame == 0 and stride == 1
        n_walk = self.total_frames if want_walk else 0
        walk_kp   = np.full((n_walk, N_KP, 3), np.nan, dtype=np.float32)
        walk_bbox = np.full((n_walk, 4), np.nan, dtype=np.float32)
        walk_on   = np.full(n_walk, np.nan, dtype=np.float32)
        n_rescued = 0
        if self.cfg.walk_pose and not want_walk:
            print("[ANYA-TELEM] walking pose sidecar skipped "
                  "(windowed or strided run writes an incomplete track)")

        with open(tmp_path, "w") as fh:
            meta = {
                "version":        TELEMETRY_VERSION,
                "video":          os.path.basename(self.video_path),
                "fps":            self.fps,
                "total_frames":   self.total_frames,
                "stride":         stride,
                "analysis_size":  list(self.cfg.analysis_size),
                "source_size":    list(self.source_size),
                "ball_imgsz":     list(self.cfg.ball_imgsz),
                "ball_conf":      self.cfg.ball_conf,
                "player_imgsz":   list(self.player_imgsz),
                "court_length_ft": Config.COURT_LENGTH_FT,
                "court_width_ft":  Config.COURT_WIDTH_FT,
                "exclusion_zones": [list(z) for z in self.exclusion_zones],
                "far_roi":        list(self.far_roi),
                "near_band":      list(self.near_band),
                "walk_pose":      bool(want_walk),
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

                    # One player call on the near band serves both consumers:
                    # the strict `np`/`npw` record and the wider walking track.
                    cands = self._detect_near_persons(frame)
                    near_box, near_world = self._pick_near_player(cands)

                    if want_walk:
                        persons = self._detect_walk_persons(frame)
                        if not persons and self.cfg.walk_rescue:
                            # Same trigger the standalone rescue used: nobody
                            # found at all, not "found but rejected".
                            persons = self._rescue_walk_persons(orig_frame)
                            n_rescued += bool(persons)
                        wbox, wkp, w_on = self._pick_walk_person(persons)
                        if wbox is not None:
                            walk_kp[frame_idx]   = wkp
                            walk_bbox[frame_idx] = wbox
                            walk_on[frame_idx]   = w_on

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

        if want_walk:
            # Trim to what was actually read: probe_video's frame count can
            # overshoot the decodable tail, and a short array of NaN would be
            # read by walking/features.py as real missing detections.
            n_seen = frame_idx + 1
            walk_path = walk_pose_path_for(self.video_path)
            np.savez_compressed(walk_path,
                                kp=walk_kp[:n_seen], bbox=walk_bbox[:n_seen],
                                on_court=walk_on[:n_seen],
                                fps=np.float64(self.fps))
            cov = float(np.mean(np.isfinite(walk_bbox[:n_seen, 0]))) if n_seen else 0.0
            near_frac = (float(np.nansum(walk_on[:n_seen] == 1.0) / n_seen)
                         if n_seen else 0.0)
            print(f"[ANYA-TELEM] walk pose → {walk_path}: coverage {cov:.1%} "
                  f"(near-half {near_frac:.1%}, {n_rescued} frame(s) recovered "
                  f"by the high-res pass)")

        return out_path


def extract_anya_telemetry(video_path: str, force: bool = False, stride: int = 1,
                           max_frames: Optional[int] = None,
                           start_frame: int = 0,
                           progress_cb=None) -> str:
    """Extract (or reuse cached) telemetry for video_path. Returns JSONL path."""
    out_path = telemetry_path_for(video_path)
    if not force and os.path.isfile(out_path):
        cached_ver, had_walk = 0, False
        try:
            with open(out_path, "r") as fh:
                cached_meta = json.loads(fh.readline()).get("meta", {})
            cached_ver = int(cached_meta.get("version", 0))
            had_walk = bool(cached_meta.get("walk_pose", False))
        except Exception:
            pass
        if cached_ver == TELEMETRY_VERSION:
            # The walking sidecar is written by the same pass, so a cache whose
            # sidecar has been deleted is incomplete even though the JSONL is
            # current — stage 5 would otherwise find no pose file at all.
            if had_walk and not os.path.isfile(walk_pose_path_for(video_path)):
                print("[ANYA-TELEM] telemetry is current but its walk-pose "
                      "sidecar is missing — re-extracting.")
            else:
                print(f"[ANYA-TELEM] Using cached telemetry: {out_path}  "
                      "(--force to re-extract)")
                return out_path
        else:
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
