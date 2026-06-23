"""
Far-side tennis serve tracker — toss + racquet-up (+ optional audio).

A serve is declared for the FAR-SIDE player when, in order and within a short
window, the evidence shows:

    (1) BALL TOSS    — a ball tracklet with an UPWARD trajectory in the far
                       ROI (image cy DECREASES; near-vertical, ~constant area).
                       REQUIRED.
    (2) RACQUET UP   — a "tennis racket" detection in an UPWARD orientation
                       (tall/vertical bbox, raised overhead near the toss),
                       co-located in time with the toss apex.  REQUIRED.
    (3) AUDIO IMPACT — a racket-ball "pock" (spectral-flux onset) near the
                       apex.  OPTIONAL — it only boosts the score; its absence
                       never rejects a serve.

This differs from `farside_serve_detector_v2`, whose third gate was a
ball-descent-into-court signal and whose audio gate was mandatory.  Almost all
of the heavy machinery (Kalman ball tracker, static-ball exclusion, toss
kinematics, audio onsets, ROI calibration, video I/O) is REUSED from v2 by
import; the only genuinely new pieces here are the racquet-orientation detector
and the toss->racquet(->audio) gate.

Camera geometry (same as v2): behind the baseline, ~10 ft high; image +y is
DOWN.  The far player is near the TOP of the frame and small.  A toss makes the
ball's cy DECREASE; an overhead serve makes the racquet appear as a tall,
vertical box raised above the player's torso.

Dependencies: numpy (required).  opencv-python + ultralytics are needed only
for `run_on_video_toss`; the synthetic demo runs offline with pure numpy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Reuse everything that already works in v2 ---------------------------------- #
from farside_serve_detector_v2 import (
    Config,
    Detection,
    Track,
    KalmanBallTracker,
    find_static_exclusion_zones,
    apply_exclusion_zones,
    compute_features,
    _toss_segments,
    onset_times_from_samples,
    detect_audio_impacts,
    _load_calib,
    _calibrate_roi_interactive,
    _try_extract_audio,
    _put_outlined_text,
    _CALIB_CACHE,
    _CV2_AVAILABLE,
)

try:
    import cv2 as _cv2
    from ultralytics import YOLO as _YOLO
except ImportError:  # pragma: no cover - exercised only in video mode
    _cv2 = None       # type: ignore
    _YOLO = None      # type: ignore


# --------------------------------------------------------------------------- #
# Configuration (extends v2's Config with racquet-orientation parameters)
# --------------------------------------------------------------------------- #
@dataclass
class TossConfig(Config):
    # --- Racquet (COCO "tennis racket") detector ---
    pose_model_path: str = "yolo26n.pt"   # general COCO detector (person + racket)
    racket_conf: float = 0.20             # racket is small on the far side
    racket_imgsz: int = 1280              # bigger than ball pass: thin racket
    racket_class_names: tuple = ("tennis racket", "tennis racquet")
    person_class_name: str = "person"

    # --- "Upward orientation" definition ---
    racket_aspect_min: float = 1.15       # bbox h/w >= this => tall/vertical box
    require_overhead: bool = True         # racket center must sit in the TOP
                                          #   THIRD of the associated player box
    overhead_frac: float = 2.0 / 3.0      # racket center must be higher than this
                                          #   fraction up from the player box bottom
    person_link_px: float = 220.0         # racket<->player association radius
    racket_min_area: float = 8.0          # reject pinprick spurious boxes

    # --- Player-anchored racquet crops (combined path: no full-frame racquet) ---
    far_racket_search: bool = True        # crop around each player and run the
                                          #   racket detector zoomed in
    far_racket_window_px: int = 200       # square crop side, centered on player box
    far_racket_conf: float = 0.10         # lower conf: crop is upscaled, racket faint
    far_racket_dedup_px: float = 15.0     # merge crop dets at the same spot
    racket_crop_imgsz: int = 512          # a 200px crop needs nowhere near 1280

    # --- Toss -> racquet linkage ---
    racket_window_s: float = 0.8          # racket-up within +/- this of the apex
    racket_link_px: float = 260.0         # racket x near the toss apex x

    # --- Gating switches (legacy far-only path; combined path uses 2-of-3) ---
    require_racket: bool = True           # (2) is mandatory
    require_audio: bool = False           # (3) is OPTIONAL by design
    audio_window_s: float = 0.30          # audio onset this close to apex counts
    serve_cooldown_s: float = 2.5

    # ===================================================================== #
    # COMBINED (near + far) detector parameters
    # ===================================================================== #

    # --- Analysis resolution (all pixel constants below assume this) ---
    analysis_w: int = 1280
    analysis_h: int = 720

    # --- Court / world geometry (singles court) ---
    court_width_ft: float = 27.0          # singles width
    court_length_ft: float = 78.0         # baseline-to-baseline
    court_cache: str = "farside_combined_court.json"

    # --- Player classification ---
    player_conf: float = 0.30             # person-detection confidence
    player_imgsz: int = 960               # 960 is plenty for the (large) person
    player_stride: int = 3                # detect persons every N frames, hold
                                          #   boxes between (players move slowly)
    # The far singles baseline is narrow in pixels (perspective); servers stand
    # wider than it. Pad each baseline's pixel-x containment band so wide-court
    # servers still classify. pad = max(pad_min_px, pad_frac * band_width).
    xband_pad_frac: float = 0.35
    xband_pad_min_px: float = 60.0

    # --- Near-side serve: foot-position gate (world feet, near baseline = 0) ---
    near_pos_into_ft: float = 0.5         # foot may be up to this far INSIDE court
    near_pos_behind_ft: float = 3.5       # ...and this far BEHIND the baseline
    near_pos_lookback_s: float = 1.5      # (a) must hold within this window BEFORE
                                          #   the toss starts ("(a) happens first")

    # --- Far-side toss crop (player-anchored ball-toss search) ---
    far_toss_w: int = 60                  # crop width (px), centered on player x
    far_toss_h: int = 100                 # crop height (px)
    far_toss_bottom_off: int = 10         # crop bottom = player_box_top + this
    far_toss_imgsz: int = 928             # YOLO size for the upscaled toss crop
    far_toss_conf: float = 0.25

    # --- Near-side toss crop (mirrors far, scaled up: near ball is large) ---
    near_toss_w: int = 180
    near_toss_h: int = 320
    near_toss_bottom_off: int = 20        # crop bottom = player_box_top + this
    near_toss_imgsz: int = 928
    near_toss_conf: float = 0.25

    # --- Full-frame ball pass confidence (overrides v2 base default of 0.08) ---
    ball_conf: float = 0.25               # full-frame ball (Pass 2 / legacy mode)

    # --- Full-frame ball pass (diagonal post-contact trace) ---
    # GATED two-pass mode: Pass 1 runs only the cheap player-anchored toss/racket
    # crops; the expensive full-frame ball pass runs in Pass 2 ONLY in short
    # windows after a confirmed FAR candidate toss apex (where the diagonal trace
    # lives). Set gated_fullframe_ball=False to run full-frame ball every frame.
    gated_fullframe_ball: bool = True
    diag_pass_pre_s: float = 0.10         # Pass-2 window starts this before apex
    diag_pass_post_s: float = 1.20        # ...and extends this far after it
    ball_dedup_px: float = 12.0           # merge near-coincident dets across passes

    # --- Diagonal-trace detection (far criterion c) ---
    diag_min_frames: int = 4              # length of a descending+lateral run
    diag_vy_min_px_s: float = 40.0        # must be descending (vy>0) at least this
    diag_vx_min_px_s: float = 30.0        # ...and moving laterally at least this
    diag_link_px: float = 120.0           # trace start near the toss apex
    diag_link_window_s: float = 1.0       # ...and begins within this of the apex

    # --- Combined gate switches ---
    near_require_racket: bool = True      # near serve needs racquet-up (b2)
    far_min_signals: int = 2              # far serve: toss + N-of-3 corroborators


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class RacketDet:
    frame: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return self.h / (self.w + 1e-6)


@dataclass
class PersonDet:
    frame: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float
    side: Optional[str] = None         # "near" | "far" | None (unclassified)
    world: Optional[tuple] = None      # (wx, wy) ft of bottom-center foot

    @property
    def box_top(self) -> float:
        return self.cy - self.h / 2.0

    @property
    def foot_y(self) -> float:
        return self.cy + self.h / 2.0


@dataclass
class TossServeEvent:
    toss_start_frame: int
    apex_frame: int
    contact_frame: int        # racket-up frame, or audio impact if available
    fps: float
    track_id: int
    side: str = "far"         # "near" | "far"
    score: float = 1.0
    notes: dict = field(default_factory=dict)

    def t(self) -> float:
        return self.contact_frame / self.fps

    def hhmmss(self) -> str:
        s = self.t()
        return f"{int(s//3600):02d}:{int(s%3600//60):02d}:{s%60:06.3f}"


# --------------------------------------------------------------------------- #
# Court geometry — homography from 4 singles-court corners
# --------------------------------------------------------------------------- #
class CourtGeometry:
    """
    Maps image pixels to world feet via a homography from the four singles-court
    corners. World frame (matching far_anya_base.py):

        near baseline  -> wy = 0           (bottom of image)
        far  baseline  -> wy = court_len   (top of image)
        left sideline  -> wx = 0
        right sideline -> wx = court_width

    Corners are stored ordered BL, BR, TR, TL (bottom-left, bottom-right,
    top-right, top-left in IMAGE space).
    """

    def __init__(self, corners_clicked, cfg: TossConfig):
        self.cfg = cfg
        self.BL, self.BR, self.TR, self.TL = self._order_corners(corners_clicked)
        self.H = self._compute_homography()
        # Pixel-x ranges of each baseline (for the x-containment gate), PADDED
        # so wide-court servers still classify (the far baseline is narrow).
        def _pad(lo, hi):
            p = max(cfg.xband_pad_min_px, cfg.xband_pad_frac * (hi - lo))
            return lo - p, hi + p
        self.near_x_lo, self.near_x_hi = _pad(min(self.BL[0], self.BR[0]),
                                              max(self.BL[0], self.BR[0]))
        self.far_x_lo, self.far_x_hi = _pad(min(self.TL[0], self.TR[0]),
                                            max(self.TL[0], self.TR[0]))
        self.near_baseline_y = (self.BL[1] + self.BR[1]) / 2.0
        self.far_baseline_y  = (self.TL[1] + self.TR[1]) / 2.0

    @staticmethod
    def _order_corners(pts):
        """Sort 4 clicked points (any order) into BL, BR, TR, TL (image space)."""
        pts = [(float(x), float(y)) for x, y in pts]
        if len(pts) != 4:
            raise ValueError(f"Need exactly 4 court corners, got {len(pts)}.")
        ys = sorted(pts, key=lambda p: p[1])
        top2, bot2 = ys[:2], ys[2:]            # smaller y = higher in image = far
        TL, TR = sorted(top2, key=lambda p: p[0])
        BL, BR = sorted(bot2, key=lambda p: p[0])
        return BL, BR, TR, TL

    def _compute_homography(self):
        W, L = self.cfg.court_width_ft, self.cfg.court_length_ft
        dst = np.array([[0, 0], [W, 0], [W, L], [0, L]], dtype=np.float64)
        src = np.array([self.BL, self.BR, self.TR, self.TL], dtype=np.float64)
        if _cv2 is not None:
            H, _ = _cv2.findHomography(src, dst)
            return H
        return self._dlt_homography(src, dst)         # pure-numpy fallback

    @staticmethod
    def _dlt_homography(src, dst):
        """4-point Direct Linear Transform (no OpenCV needed)."""
        A = []
        for (x, y), (u, v) in zip(src, dst):
            A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        _, _, Vt = np.linalg.svd(np.array(A, dtype=np.float64))
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]

    def world_pos(self, px_x: float, px_y: float) -> tuple:
        v = self.H @ np.array([float(px_x), float(px_y), 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def classify(self, box_cx: float, foot_y: float) -> tuple:
        """Classify a player by the bottom-center of its box.

        Returns (side, (wx, wy)) where side in {"near","far",None}.
          near : foot world-y nearer to 0 than to court_len  AND box x within
                 the NEAR baseline's pixel-x span.
          far  : foot world-y nearer to court_len than to 0  AND box x within
                 the FAR baseline's pixel-x span.
        """
        wx, wy = self.world_pos(box_cx, foot_y)
        mid = self.cfg.court_length_ft / 2.0
        if wy < mid and self.near_x_lo <= box_cx <= self.near_x_hi:
            return "near", (wx, wy)
        if wy >= mid and self.far_x_lo <= box_cx <= self.far_x_hi:
            return "far", (wx, wy)
        return None, (wx, wy)

    def corners_px(self):
        return [self.BL, self.BR, self.TR, self.TL]


def init_court_geometry(video_path: str, cfg: TossConfig,
                        recalibrate: bool = False) -> CourtGeometry:
    """Calibrate the four singles-court corners (interactive, cached) and build
    the homography. Reuses `utilities.init_court` (lazy import keeps the offline
    demo free of cv2/ultralytics)."""
    from utilities import init_court        # lazy: heavy deps
    analysis_size = (cfg.analysis_w, cfg.analysis_h)
    if recalibrate:
        cp = None
        try:
            from utilities import _court_cache_path
            cp = _court_cache_path(video_path)
        except Exception:
            cp = None
        if cp and os.path.exists(cp):
            os.remove(cp)
            print(f"[COURT] Removed cached corners → recalibrating ({cp}).")
    corners, _shape = init_court(video_path, analysis_size=analysis_size)
    geom = CourtGeometry(corners, cfg)
    print(f"[COURT] Corners (BL,BR,TR,TL): "
          f"{[tuple(round(v,1) for v in c) for c in geom.corners_px()]}")
    print(f"[COURT] near baseline y≈{geom.near_baseline_y:.0f}px "
          f"x∈[{geom.near_x_lo:.0f},{geom.near_x_hi:.0f}]  |  "
          f"far baseline y≈{geom.far_baseline_y:.0f}px "
          f"x∈[{geom.far_x_lo:.0f},{geom.far_x_hi:.0f}]")
    return geom


# --------------------------------------------------------------------------- #
# Racquet-up evidence: is there an upward-oriented racquet near a point/time?
# --------------------------------------------------------------------------- #
def _racket_is_upward(
    r: RacketDet,
    persons_in_frame: list[PersonDet],
    cfg: TossConfig,
) -> tuple[bool, dict]:
    """A racquet counts as 'upward' when its bbox is tall/vertical and (if a
    player can be associated) its center sits above the player's center, i.e.
    the racquet is raised overhead rather than held low for a groundstroke."""
    info = {"aspect": round(r.aspect, 2), "conf": round(r.conf, 2)}
    if r.area < cfg.racket_min_area:
        return False, info
    if r.aspect < cfg.racket_aspect_min:
        return False, info

    if cfg.require_overhead and persons_in_frame:
        p = min(persons_in_frame,
                key=lambda q: np.hypot(q.cx - r.cx, q.cy - r.cy))
        if np.hypot(p.cx - r.cx, p.cy - r.cy) <= cfg.person_link_px:
            # The racket center must be HIGHER than the line `overhead_frac` of
            # the way up from the BOTTOM of the player box (default 2/3 -> the
            # racket must be in the top third of the player box). Image +y is
            # DOWN, so "higher" means a SMALLER cy.
            p_bottom = p.cy + p.h / 2.0
            thresh_cy = p_bottom - cfg.overhead_frac * p.h   # = p.cy - p.h/6 at 2/3
            info["player_dy"] = round(p.cy - r.cy, 1)        # >0 => racket above
            info["thresh_cy"] = round(thresh_cy, 1)
            if r.cy >= thresh_cy:                            # not high enough
                return False, info
    return True, info


def find_racket_up_evidence(
    apex_frame: int,
    apex_cx: float,
    rackets: dict[int, list[RacketDet]],
    persons: dict[int, list[PersonDet]],
    cfg: TossConfig,
) -> Optional[dict]:
    """Search a +/- window of frames around the toss apex for an upward-oriented
    racquet whose x is near the toss column.  Returns the best (closest-to-apex)
    match, or None."""
    half = int(round(cfg.racket_window_s * cfg.fps))
    best = None
    for f in range(apex_frame - half, apex_frame + half + 1):
        for r in rackets.get(f, []):
            if abs(r.cx - apex_cx) > cfg.racket_link_px:
                continue
            up, info = _racket_is_upward(r, persons.get(f, []), cfg)
            if not up:
                continue
            cand = {"frame": f, "cx": r.cx, "cy": r.cy,
                    "dt_apex": abs(f - apex_frame), **info}
            if best is None or cand["dt_apex"] < best["dt_apex"]:
                best = cand
    return best


# --------------------------------------------------------------------------- #
# The toss -> racquet(-up) (-> optional audio) serve gate
# --------------------------------------------------------------------------- #
def detect_serves_toss(
    tracks: list[Track],
    rackets: dict[int, list[RacketDet]],
    persons: dict[int, list[PersonDet]],
    audio_impacts: list[float],
    cfg: TossConfig,
) -> list[TossServeEvent]:
    # (1) Pool all upward-ball toss segments across tracklets (reuse v2 logic).
    tosses: list[dict] = []
    for tr in tracks:
        feat = compute_features(tr, cfg)
        if feat is None:
            continue
        tosses.extend(_toss_segments(feat, tr.id, cfg))

    impacts = sorted(audio_impacts)
    events: list[TossServeEvent] = []
    n_racket_ok = n_audio_ok = 0

    for tz in tosses:
        apex_f = tz["apex_f"]
        t_apex = apex_f / cfg.fps

        # (2) Racquet-up evidence near the apex — REQUIRED (unless disabled).
        rk = find_racket_up_evidence(apex_f, tz["apex_cx"], rackets, persons, cfg)
        if rk is not None:
            n_racket_ok += 1
        if rk is None and cfg.require_racket:
            continue

        # (3) Audio impact near the apex — OPTIONAL; boosts confidence only.
        cands = [a for a in impacts if abs(a - t_apex) <= cfg.audio_window_s]
        impact = min(cands, key=lambda a: abs(a - t_apex)) if cands else None
        if impact is not None:
            n_audio_ok += 1
        if impact is None and cfg.require_audio:
            continue

        # Contact timestamp: prefer the audio pock; else the racquet-up frame;
        # else fall back to the toss apex.
        if impact is not None:
            contact_f = int(round(impact * cfg.fps))
        elif rk is not None:
            contact_f = int(rk["frame"])
        else:
            contact_f = apex_f

        # Score: toss quality (low area CV) x racquet verticality x audio bonus.
        toss_q = max(0.0, 1.0 - tz["area_cv"])
        rk_q = min(1.0, rk["aspect"] / 2.0) if rk is not None else 0.4
        audio_bonus = 1.0 if impact is not None else 0.75
        score = float(min(1.0, toss_q * rk_q * audio_bonus))

        events.append(TossServeEvent(
            toss_start_frame=tz["toss_start_f"],
            apex_frame=apex_f,
            contact_frame=contact_f,
            fps=cfg.fps,
            track_id=tz["track_id"],
            score=score,
            notes={
                "area_cv": round(tz["area_cv"], 3),
                "racket": rk if rk is not None else None,
                "audio": impact is not None,
            },
        ))

    print(f"[TOSS] gate: {len(tosses)} toss segs -> "
          f"{n_racket_ok} with racquet-up -> {n_audio_ok} also with audio "
          f"-> {len(events)} pre-dedup events")

    # Order + cooldown dedup (same approach as v2).
    events.sort(key=lambda ev: ev.contact_frame)
    deduped: list[TossServeEvent] = []
    cd = cfg.serve_cooldown_s * cfg.fps
    for ev in events:
        if deduped and (ev.contact_frame - deduped[-1].contact_frame) < cd:
            if ev.score > deduped[-1].score:
                deduped[-1] = ev
            continue
        deduped.append(ev)
    return deduped


# --------------------------------------------------------------------------- #
# Combined (near + far) serve detection
# --------------------------------------------------------------------------- #
def _generic_toss_segments(feat, track_id: int, cfg: TossConfig) -> list[dict]:
    """Rising + vertical + ~constant-area runs ANYWHERE (no global ROI gate).

    Side is assigned later by which player the apex sits above. Mirrors v2's
    `_toss_segments` minus the far-ROI / court-x masks."""
    from farside_serve_detector_v2 import _contiguous_runs
    mask = (feat.vy < -cfg.rise_vy_px_s) & (feat.verticality > cfg.verticality_min)
    out = []
    for (s, e) in _contiguous_runs(mask, cfg.min_rise_frames):
        seg = feat.area[s:e + 1]
        cv = float(np.std(seg) / (np.mean(seg) + 1e-6))
        if cv > cfg.area_cv_max:
            continue
        apex = s + int(np.argmin(feat.cy[s:e + 1]))
        out.append({"track_id": track_id,
                    "toss_start_f": int(feat.frame[s]),
                    "apex_f": int(feat.frame[apex]),
                    "apex_cx": float(feat.cx[apex]),
                    "apex_cy": float(feat.cy[apex]),
                    "area_cv": cv})
    return out


def _diagonal_segments(feat, track_id: int, cfg: TossConfig) -> list[dict]:
    """Descending (vy>0) AND laterally-moving (|vx| large) runs — the
    post-contact diagonal ball trace (far criterion c)."""
    from farside_serve_detector_v2 import _contiguous_runs
    mask = (feat.vy > cfg.diag_vy_min_px_s) & (np.abs(feat.vx) > cfg.diag_vx_min_px_s)
    out = []
    for (s, e) in _contiguous_runs(mask, cfg.diag_min_frames):
        out.append({"track_id": track_id,
                    "start_f": int(feat.frame[s]),
                    "start_cx": float(feat.cx[s]),
                    "start_cy": float(feat.cy[s]),
                    "dir": "right" if feat.vx[s:e + 1].mean() > 0 else "left"})
    return out


def _player_at(frame_idx: int, players_by_frame: dict, max_gap: int = 5):
    """Nearest-in-time player box for a side at/around a frame."""
    if frame_idx in players_by_frame:
        return players_by_frame[frame_idx]
    for d in range(1, max_gap + 1):
        if frame_idx - d in players_by_frame:
            return players_by_frame[frame_idx - d]
        if frame_idx + d in players_by_frame:
            return players_by_frame[frame_idx + d]
    return None


def _assign_toss_side(tz: dict, near_by_frame: dict, far_by_frame: dict,
                      cfg: TossConfig) -> Optional[str]:
    """A toss belongs to whichever player's box it rises ABOVE: apex above the
    head (apex_cy < box_top + slack) and apex_cx within the player's box."""
    apex_f, ax, ay = tz["apex_f"], tz["apex_cx"], tz["apex_cy"]
    for side, by_frame in (("far", far_by_frame), ("near", near_by_frame)):
        p = _player_at(apex_f, by_frame)
        if p is None:
            continue
        slack = 0.25 * p.h
        if (ay < p.box_top + slack and
                p.cx - p.w / 2 - 30 <= ax <= p.cx + p.w / 2 + 30):
            return side
    return None


def _near_position_ok(toss_start_f: int, near_by_frame: dict,
                      cfg: TossConfig) -> Optional[dict]:
    """Near criterion (a): within the lookback window BEFORE the toss starts,
    the near player's foot (box bottom) must be between near_pos_into_ft inside
    and near_pos_behind_ft behind the near baseline (world wy in
    [-behind, +into]). Returns the satisfying sample or None."""
    look = int(round(cfg.near_pos_lookback_s * cfg.fps))
    best = None
    for f in range(max(0, toss_start_f - look), toss_start_f + 1):
        p = near_by_frame.get(f)
        if p is None or p.world is None:
            continue
        wy = p.world[1]
        if -cfg.near_pos_behind_ft <= wy <= cfg.near_pos_into_ft:
            cand = {"frame": f, "wy": round(wy, 2), "wx": round(p.world[0], 2)}
            if best is None or f > best["frame"]:   # closest to toss start
                best = cand
    return best


def compute_tosses_with_side(tracks: list[Track], near_by_frame: dict,
                             far_by_frame: dict, cfg: TossConfig) -> list[dict]:
    """Extract toss segments from toss tracks and tag each with its side."""
    out = []
    for tr in tracks:
        feat = compute_features(tr, cfg)
        if feat is None:
            continue
        for tz in _generic_toss_segments(feat, tr.id, cfg):
            tz["side"] = _assign_toss_side(tz, near_by_frame, far_by_frame, cfg)
            out.append(tz)
    return out


def detect_serves_combined(
    tosses: list[dict],
    diagonals: list[dict],
    rackets: dict,
    persons_all: dict,
    near_by_frame: dict,
    far_by_frame: dict,
    audio_impacts: list[float],
    cfg: TossConfig,
) -> list[TossServeEvent]:
    """Combined near + far serve gate over PRE-COMPUTED tosses + diagonals.

      NEAR : (a) foot-position-at-baseline FIRST, then toss + racquet-up
             (audio optional).
      FAR  : toss + at least `far_min_signals` of
             {audio impact, racquet-up, diagonal ball trace}.

    `tosses` carry a "side" key (from compute_tosses_with_side); `diagonals`
    come from the gated Pass-2 full-frame ball detection.
    """
    impacts = sorted(audio_impacts)
    events: list[TossServeEvent] = []
    stat = {"near_pos": 0, "far_diag": 0, "near": 0, "far": 0}

    for tz in tosses:
        side = tz.get("side") or _assign_toss_side(tz, near_by_frame, far_by_frame, cfg)
        if side is None:
            continue
        apex_f = tz["apex_f"]
        t_apex = apex_f / cfg.fps

        # Shared corroborators -------------------------------------------------
        rk = find_racket_up_evidence(apex_f, tz["apex_cx"], rackets, persons_all, cfg)
        audio_cands = [a for a in impacts if abs(a - t_apex) <= cfg.audio_window_s]
        audio = min(audio_cands, key=lambda a: abs(a - t_apex)) if audio_cands else None

        if side == "far":
            # Diagonal trace linked to this apex (criterion c).
            diag = None
            for d in diagonals:
                dt = d["start_f"] / cfg.fps - t_apex
                if (0 <= dt <= cfg.diag_link_window_s and
                        np.hypot(d["start_cx"] - tz["apex_cx"],
                                 d["start_cy"] - tz["apex_cy"]) <= cfg.diag_link_px):
                    if diag is None or d["start_f"] < diag["start_f"]:
                        diag = d
            signals = {"audio": audio is not None,
                       "racket": rk is not None,
                       "diagonal": diag is not None}
            n = sum(signals.values())
            if n < cfg.far_min_signals:
                continue
            if diag is not None:
                stat["far_diag"] += 1
            contact_f = (int(round(audio * cfg.fps)) if audio is not None
                         else (int(rk["frame"]) if rk is not None
                               else (diag["start_f"] if diag else apex_f)))
            toss_q = max(0.0, 1.0 - tz["area_cv"])
            score = float(min(1.0, toss_q * (0.5 + 0.25 * n)))
            events.append(TossServeEvent(
                toss_start_frame=tz["toss_start_f"], apex_frame=apex_f,
                contact_frame=contact_f, fps=cfg.fps, track_id=tz["track_id"],
                side="far", score=score,
                notes={"area_cv": round(tz["area_cv"], 3), "n_signals": n,
                       **signals, "racket": rk,
                       "diag_dir": diag["dir"] if diag else None}))
            stat["far"] += 1

        else:  # NEAR
            pos = _near_position_ok(tz["toss_start_f"], near_by_frame, cfg)
            if pos is None:
                continue
            stat["near_pos"] += 1
            if cfg.near_require_racket and rk is None:
                continue
            contact_f = (int(round(audio * cfg.fps)) if audio is not None
                         else (int(rk["frame"]) if rk is not None else apex_f))
            toss_q = max(0.0, 1.0 - tz["area_cv"])
            rk_q = min(1.0, rk["aspect"] / 2.0) if rk is not None else 0.4
            score = float(min(1.0, toss_q * rk_q * (1.0 if audio else 0.85)))
            events.append(TossServeEvent(
                toss_start_frame=tz["toss_start_f"], apex_frame=apex_f,
                contact_frame=contact_f, fps=cfg.fps, track_id=tz["track_id"],
                side="near", score=score,
                notes={"area_cv": round(tz["area_cv"], 3), "position": pos,
                       "racket": rk, "audio": audio is not None}))
            stat["near"] += 1

    print(f"[COMBINED] gate: {len(tosses)} tosses, {len(diagonals)} diagonals -> "
          f"near={stat['near']} (pos_ok={stat['near_pos']})  "
          f"far={stat['far']} (with_diag={stat['far_diag']})  "
          f"pre-dedup={len(events)}")

    # Per-side cooldown dedup.
    events.sort(key=lambda ev: ev.contact_frame)
    deduped: list[TossServeEvent] = []
    cd = cfg.serve_cooldown_s * cfg.fps
    for ev in events:
        prev = next((d for d in reversed(deduped) if d.side == ev.side), None)
        if prev and (ev.contact_frame - prev.contact_frame) < cd:
            if ev.score > prev.score:
                deduped[deduped.index(prev)] = ev
            continue
        deduped.append(ev)
    deduped.sort(key=lambda ev: ev.contact_frame)
    return deduped


# --------------------------------------------------------------------------- #
# Real-video entry point
# --------------------------------------------------------------------------- #
_DEFAULT_BALL_MODEL = (
    "/Users/tennis/Documents/Code/Laptop/weights/ball/weights/best.pt"
)


def _resolve_class_ids(model, cfg: TossConfig) -> tuple[set[int], Optional[int]]:
    """Map the racket/person class names to the loaded model's id space."""
    names = model.names if hasattr(model, "names") else {}
    if isinstance(names, dict):
        items = names.items()
    else:                                       # list-like
        items = enumerate(names)
    racket_ids, person_id = set(), None
    wanted = {n.lower() for n in cfg.racket_class_names}
    for i, nm in items:
        low = str(nm).lower()
        if low in wanted:
            racket_ids.add(int(i))
        if low == cfg.person_class_name:
            person_id = int(i)
    return racket_ids, person_id


def _batch_crops(model, frame, boxes, imgsz, conf, classes=None):
    """Batched inference over several crops of `frame` in ONE model call.

    `boxes` is a list of (x1, y1, x2, y2). Returns a list aligned with `boxes`,
    each element a list of (cx, cy, w, h, conf) in FULL-FRAME coords. Empty or
    out-of-bounds crops yield []. Batching cuts per-call Python/launch overhead
    (a big win on CPU/MPS) versus one model call per crop."""
    H, W = frame.shape[:2]
    crops, offs = [], []
    for (x1, y1, x2, y2) in boxes:
        ix1 = int(max(0, min(W - 1, x1))); iy1 = int(max(0, min(H - 1, y1)))
        ix2 = int(max(0, min(W, x2)));     iy2 = int(max(0, min(H, y2)))
        if ix2 - ix1 < 6 or iy2 - iy1 < 6:
            crops.append(None); offs.append(None)
        else:
            crops.append(frame[iy1:iy2, ix1:ix2]); offs.append((ix1, iy1))
    results = [[] for _ in boxes]
    valid = [(i, c) for i, c in enumerate(crops) if c is not None]
    if not valid:
        return results
    kw = dict(conf=conf, imgsz=imgsz, verbose=False)
    if classes is not None:
        kw["classes"] = list(classes)
    res = model([c for _, c in valid], **kw)
    for (i, _), r in zip(valid, res):
        ox, oy = offs[i]
        out = []
        if r.boxes:
            for b in r.boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                out.append((ox + (bx1 + bx2) / 2.0, oy + (by1 + by2) / 2.0,
                            bx2 - bx1, by2 - by1, float(b.conf[0])))
        results[i] = out
    return results


def _crop_ball_search(frame, x1, y1, x2, y2, ball_model, imgsz, conf):
    """Run the ball detector on an arbitrary crop; return boxes in FULL-FRAME
    coords (cx, cy, w, h, conf)."""
    H, W = frame.shape[:2]
    x1 = int(max(0, min(W - 1, x1))); y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W, x2)));     y2 = int(max(0, min(H, y2)))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return []
    res = ball_model(frame[y1:y2, x1:x2], conf=conf, imgsz=imgsz, verbose=False)
    out = []
    if res and res[0].boxes:
        for b in res[0].boxes:
            bx1, by1, bx2, by2 = b.xyxy[0].tolist()
            out.append((x1 + (bx1 + bx2) / 2.0, y1 + (by1 + by2) / 2.0,
                        bx2 - bx1, by2 - by1, float(b.conf[0])))
    return out


def _build_streams(
    video_path: str,
    ball_model,
    pose_model,
    cfg: TossConfig,
    geom: "CourtGeometry",
    excl_zones: list = None,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
):
    """Combined Pass 1 over the video — CHEAP per-frame work only.

    Per frame (no full-frame ball, no full-frame racquet):
      • persons via a STRIDED full-frame person pass (held between strides)
      • racquets via batched player-anchored crops (far + near)
      • ball-TOSS detections via batched player-anchored toss crops (far + near)

    The expensive full-frame ball pass is deferred to the gated Pass 2
    (`_pass2_diagonals`) when `cfg.gated_fullframe_ball` is True.

    Returns: ball_stream, rackets, persons_all, near_by_frame, far_by_frame.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("opencv-python is required for run_on_video_toss.")

    racket_ids, person_id = _resolve_class_ids(pose_model, cfg)
    if not racket_ids:
        print("[COMBINED] WARNING: no 'tennis racket' class in pose model.")
    if person_id is None:
        raise RuntimeError("Pose model has no 'person' class — required.")

    cap = _cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)

    W, H = cfg.frame_w, cfg.frame_h
    total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total - start_frame, max_frames) if max_frames else (total - start_frame)
    print(f"[COMBINED] Pass 1 (cheap): scanning {limit} frames at {W}x{H}  "
          f"(person stride={cfg.player_stride}, gated_ball={cfg.gated_fullframe_ball}) …")

    ball_stream: list[list[Detection]] = []
    rackets: dict[int, list[RacketDet]] = {}
    persons_all: dict[int, list[PersonDet]] = {}
    near_by_frame: dict[int, PersonDet] = {}
    far_by_frame: dict[int, PersonDet] = {}
    rk_half = cfg.far_racket_window_px // 2
    _zones = excl_zones or []

    def _not_excluded(cx, cy):
        for (x1, y1, x2, y2) in _zones:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return False
        return True

    held_near: Optional[PersonDet] = None     # held between strided detections
    held_far: Optional[PersonDet] = None

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, orig = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            frame = _cv2.resize(orig, (W, H), interpolation=_cv2.INTER_LINEAR)

            # --- 1. STRIDED person detection (person class only) ------------
            if frame_idx % cfg.player_stride == 0:
                r_pose = pose_model(frame, conf=cfg.player_conf,
                                    imgsz=cfg.player_imgsz, verbose=False,
                                    classes=[person_id])
                near_c, far_c = [], []
                if r_pose and r_pose[0].boxes:
                    for b in r_pose[0].boxes:
                        if int(b.cls[0]) != person_id:
                            continue
                        bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                        w_, h_ = bx2 - bx1, by2 - by1
                        side, world = geom.classify(cx, cy + h_ / 2.0)
                        pd = PersonDet(frame_idx, cx, cy, w_, h_,
                                       float(b.conf[0]), side, world)
                        if side == "near":
                            near_c.append(pd)
                        elif side == "far":
                            far_c.append(pd)
                held_near = max(near_c, key=lambda p: p.w * p.h) if near_c else None
                held_far = max(far_c, key=lambda p: p.world[1] if p.world else 0) \
                    if far_c else None

            near_p, far_p = held_near, held_far
            persons_all[frame_idx] = [p for p in (near_p, far_p) if p is not None]
            if near_p is not None:
                near_by_frame[frame_idx] = near_p
            if far_p is not None:
                far_by_frame[frame_idx] = far_p

            # --- 2. Racquet crops (batched, far + near) ---------------------
            if racket_ids:
                rk_boxes = [(p.cx - rk_half, p.cy - rk_half,
                             p.cx + rk_half, p.cy + rk_half)
                            for p in (far_p, near_p) if p is not None]
                if rk_boxes:
                    existing: list[RacketDet] = []
                    for blist in _batch_crops(pose_model, frame, rk_boxes,
                                              cfg.racket_crop_imgsz,
                                              cfg.far_racket_conf, classes=racket_ids):
                        for (cx, cy, w_, h_, cf) in blist:
                            if any(np.hypot(e.cx - cx, e.cy - cy) <=
                                   cfg.far_racket_dedup_px for e in existing):
                                continue
                            existing.append(RacketDet(frame_idx, cx, cy, w_, h_, cf))
                    if existing:
                        rackets[frame_idx] = existing

            # --- 3. Ball TOSS crops (batched, far + near) -------------------
            toss_boxes = []
            if far_p is not None:
                bot = far_p.box_top + cfg.far_toss_bottom_off
                toss_boxes.append((far_p.cx - cfg.far_toss_w / 2, bot - cfg.far_toss_h,
                                   far_p.cx + cfg.far_toss_w / 2, bot))
            if near_p is not None:
                bot = near_p.box_top + cfg.near_toss_bottom_off
                toss_boxes.append((near_p.cx - cfg.near_toss_w / 2, bot - cfg.near_toss_h,
                                   near_p.cx + cfg.near_toss_w / 2, bot))
            dets: list[Detection] = []
            if not cfg.gated_fullframe_ball:           # legacy: full-frame every frame
                for (cx, cy, w_, h_, cf) in _crop_ball_search(
                        frame, 0, 0, W, H, ball_model, cfg.ball_imgsz, cfg.ball_conf):
                    if _not_excluded(cx, cy):
                        dets.append(Detection(frame_idx, cx, cy, w_, h_, cf))
            if toss_boxes:
                for blist in _batch_crops(ball_model, frame, toss_boxes,
                                          cfg.far_toss_imgsz,
                                          min(cfg.far_toss_conf, cfg.near_toss_conf)):
                    for (cx, cy, w_, h_, cf) in blist:
                        if not _not_excluded(cx, cy):
                            continue
                        if any(np.hypot(d.cx - cx, d.cy - cy) <= cfg.ball_dedup_px
                               for d in dets):
                            continue
                        dets.append(Detection(frame_idx, cx, cy, w_, h_, cf))
            ball_stream.append(dets)

            frame_idx += 1
            if frame_idx % 300 == 0:
                pct = 100.0 * frame_idx / max(1, limit)
                print(f"[COMBINED]   P1 frame {frame_idx:>6} ({pct:.0f}%)  "
                      f"toss-balls(last300)={sum(len(d) for d in ball_stream[-300:])}  "
                      f"rackets={sum(len(v) for v in rackets.values())}  "
                      f"near/far={len(near_by_frame)}/{len(far_by_frame)}")
    finally:
        cap.release()

    print(f"[COMBINED] Pass 1 done: {len(ball_stream)} frames, "
          f"{sum(len(d) for d in ball_stream)} toss-ball dets, "
          f"{sum(len(v) for v in rackets.values())} racket dets, "
          f"near-frames={len(near_by_frame)} far-frames={len(far_by_frame)}")
    return ball_stream, rackets, persons_all, near_by_frame, far_by_frame


def _pass2_diagonals(video_path, far_apexes, ball_model, cfg,
                     n_frames, start_frame=0, excl_zones=None):
    """GATED Pass 2: run the full-frame ball detector ONLY in short windows
    after each confirmed FAR toss apex, track those balls, and return diagonal
    (descending + lateral) trace segments. A decode-only sweep elsewhere keeps
    this cheap — inference fires on a small fraction of frames."""
    if not far_apexes:
        return []
    pre = int(round(cfg.diag_pass_pre_s * cfg.fps))
    post = int(round(cfg.diag_pass_post_s * cfg.fps))
    wanted: set[int] = set()
    for a in far_apexes:
        wanted.update(range(max(0, a - pre), min(n_frames, a + post + 1)))
    print(f"[COMBINED] Pass 2 (gated): full-frame ball on {len(wanted)} / "
          f"{n_frames} frames ({100.0*len(wanted)/max(1,n_frames):.0f}%) "
          f"around {len(far_apexes)} far apexes …")

    W, H = cfg.frame_w, cfg.frame_h
    diag_stream: list[list[Detection]] = [[] for _ in range(n_frames)]
    cap = _cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)
    fi = 0
    try:
        while cap.isOpened() and fi < n_frames:
            ret, orig = cap.read()
            if not ret:
                break
            if fi in wanted:
                frame = _cv2.resize(orig, (W, H), interpolation=_cv2.INTER_LINEAR)
                for (cx, cy, w_, h_, cf) in _crop_ball_search(
                        frame, 0, 0, W, H, ball_model, cfg.ball_imgsz, cfg.ball_conf):
                    if not any(x1 <= cx <= x2 and y1 <= cy <= y2
                               for (x1, y1, x2, y2) in (excl_zones or [])):
                        diag_stream[fi].append(Detection(fi, cx, cy, w_, h_, cf))
            fi += 1
    finally:
        cap.release()

    tracks = KalmanBallTracker(cfg).run(diag_stream)
    diagonals = []
    for tr in tracks:
        feat = compute_features(tr, cfg)
        if feat is not None:
            diagonals.extend(_diagonal_segments(feat, tr.id, cfg))
    print(f"[COMBINED] Pass 2 done: {sum(len(d) for d in diag_stream)} ball dets, "
          f"{len(tracks)} tracks, {len(diagonals)} diagonal segments.")
    return diagonals


def run_on_video_toss(
    video_path: str,
    cfg: Optional[TossConfig] = None,
    ball_model_path: str = _DEFAULT_BALL_MODEL,
    audio_path: Optional[str] = None,
    recalibrate: bool = False,
    calib_cache: str = _CALIB_CACHE,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    viz_cfg: Optional[VizConfig] = None,
) -> list[TossServeEvent]:
    """Run the toss + racquet-up (+ optional audio) far-side serve tracker."""
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "opencv-python and ultralytics are required for run_on_video_toss.\n"
            "  pip install opencv-python ultralytics")

    if cfg is None:
        cfg = TossConfig()
    # Pin the analysis resolution used by all pixel-space constants.
    cfg.frame_w, cfg.frame_h = cfg.analysis_w, cfg.analysis_h
    # Auto-enable live visualization when no explicit VizConfig is supplied.
    # Saves to a file next to the video and opens a live display window.
    if viz_cfg is None:
        default_viz_path = os.path.splitext(video_path)[0] + "_serve_viz.mp4"
        viz_cfg = VizConfig(out_path=default_viz_path, show_live=True)
        print(f"[COMBINED] Live viz enabled → {default_viz_path}")
    # When visualizing, keep the full-frame ball every frame so the post-contact
    # diagonal trace is actually drawn (gated mode keeps only toss-crop balls).
    if viz_cfg is not None and cfg.gated_fullframe_ball:
        print("[COMBINED] --viz on → disabling gated ball pass so the full ball "
              "trace is visible (slower).")
        cfg.gated_fullframe_ball = False

    # 1. Court calibration → homography (4 singles-court corners)
    geom = init_court_geometry(video_path, cfg, recalibrate=recalibrate)

    # 2. Models: custom ball detector + general COCO detector (person + racket)
    print(f"[COMBINED] Loading ball model:  {ball_model_path}")
    ball_model = _YOLO(ball_model_path)
    print(f"[COMBINED] Loading pose model:  {cfg.pose_model_path}")
    pose_model = _YOLO(cfg.pose_model_path)

    # 3. FPS probe
    cap = _cv2.VideoCapture(video_path)
    raw_fps = cap.get(_cv2.CAP_PROP_FPS)
    cap.release()
    if 0 < raw_fps < 300:
        cfg.fps = raw_fps
    print(f"[COMBINED] Video FPS: {cfg.fps:.3f}")

    # 4. Startup exclusion-zone scan (run_anya style): sample random frames from
    #    the full video at low confidence and cluster static ball-like objects
    #    (ball baskets, court markers, etc.) via DBSCAN.  The resulting
    #    (x1,y1,x2,y2) zones are passed into the detection streams so any ball
    #    detection whose center falls inside a zone is discarded at source.
    print("[COMBINED] Scanning for static exclusion zones …")
    try:
        from utilities import create_auto_exclusion_zones
        excl_zones = create_auto_exclusion_zones(
            video_path, ball_model,
            num_frames=20, conf=0.05,
            analysis_size=(cfg.frame_w, cfg.frame_h),
        )
    except Exception as _e:
        print(f"[COMBINED] Exclusion zone scan skipped: {_e}")
        excl_zones = []
    print(f"[COMBINED] Static exclusion zones: {len(excl_zones)}")
    for z in excl_zones:
        print(f"    zone  x=[{z[0]},{z[2]}]  y=[{z[1]},{z[3]}]")

    # 5. Combined detection streams (players + multi-source ball + racquet)
    ball_stream, rackets, persons_all, near_by_frame, far_by_frame = _build_streams(
        video_path, ball_model, pose_model, cfg, geom, excl_zones,
        start_frame, max_frames)

    # 5. Audio onsets (OPTIONAL — absence never blocks detection)
    impacts: list[float] = []
    if audio_path is not None:
        print(f"[COMBINED] Loading audio: {audio_path}")
        impacts = detect_audio_impacts(audio_path, cfg)
    else:
        audio_data = _try_extract_audio(video_path)
        if audio_data is not None:
            impacts = onset_times_from_samples(audio_data[0], audio_data[1], cfg)
        else:
            print("[COMBINED] No audio — proceeding (audio is optional).")
    if start_frame > 0 and impacts:
        t_off = start_frame / cfg.fps
        impacts = [t - t_off for t in impacts if t >= t_off]
    print(f"[COMBINED] Audio impacts: {len(impacts)}")

    # 6. Static-ball exclusion + Kalman tracking of the toss balls (Pass 1)
    zones = find_static_exclusion_zones(ball_stream, cfg)
    print(f"[COMBINED] Static exclusion zones: {len(zones)}")
    clean = apply_exclusion_zones(ball_stream, zones, cfg)
    tracks = KalmanBallTracker(cfg).run(clean)
    print(f"[COMBINED] Toss-ball tracks: {len(tracks)}")

    # 7. Toss segments (with side) -> gated Pass 2 for FAR diagonals
    tosses = compute_tosses_with_side(tracks, near_by_frame, far_by_frame, cfg)
    n_frames = len(ball_stream)
    if cfg.gated_fullframe_ball:
        far_apexes = [tz["apex_f"] for tz in tosses if tz.get("side") == "far"]
        diagonals = _pass2_diagonals(video_path, far_apexes, ball_model, cfg,
                                     n_frames, start_frame, excl_zones)
    else:
        # Legacy: full-frame ball already in Pass-1 stream → diagonals from tracks.
        diagonals = []
        for tr in tracks:
            feat = compute_features(tr, cfg)
            if feat is not None:
                diagonals.extend(_diagonal_segments(feat, tr.id, cfg))

    # 8. Combined near + far serve gate
    serves = detect_serves_combined(
        tosses, diagonals, rackets, persons_all, near_by_frame, far_by_frame,
        impacts, cfg)

    n_near = sum(1 for s in serves if s.side == "near")
    n_far  = sum(1 for s in serves if s.side == "far")
    print(f"\n{'='*60}")
    print(f"  SERVES DETECTED: {len(serves)}  (near={n_near}  far={n_far})")
    print(f"{'='*60}")
    for i, ev in enumerate(serves, 1):
        abs_contact = (start_frame + ev.contact_frame) / cfg.fps
        mm = int(abs_contact // 60); ss = abs_contact % 60
        print(f"  #{i:>3}  [{ev.side.upper():>4}]  contact @ {mm:02d}:{ss:06.3f} (abs)  "
              f"track={ev.track_id}  apex_f={ev.apex_frame}  "
              f"score={ev.score:.2f}  {ev.notes}")
    print(f"{'='*60}\n")

    # 9. Visualization pass (always runs — live display + saved file)
    toss_segs = _all_toss_segments(tracks, cfg)
    render_visualization(
        video_path, cfg, geom,
        ball_stream, rackets, persons_all, near_by_frame, far_by_frame,
        tracks, toss_segs, serves, impacts,
        excl_zones=excl_zones,
        start_frame=start_frame, viz=viz_cfg)

    return serves


# --------------------------------------------------------------------------- #
# Visualization — annotated video output
# --------------------------------------------------------------------------- #
@dataclass
class VizConfig:
    """Controls the visualization rendering pass."""
    out_path: str = "farside_serve_viz.mp4"
    show_live: bool = False            # also open a display window while writing
    trail_frames: int = 25             # ball track trail length (frames)
    serve_banner_frames: int = 90      # how long the SERVE banner stays on screen
    fourcc: str = "mp4v"
    draw_court: bool = True            # singles-court polygon from homography
    draw_persons: bool = True
    draw_rackets: bool = True
    draw_crop_window: bool = True      # dashed far-racket + toss crop boxes
    draw_ball_trail: bool = True
    draw_raw_dets: bool = True
    draw_audio: bool = True
    draw_world: bool = True            # near/far label + world (wx,wy) per player
    draw_near_band: bool = True        # near-baseline serve-position band


# BGR colour palette ----------------------------------------------------------
_VC = dict(
    roi          = (160, 160, 160),   # light grey ROI border
    person       = ( 50, 160, 200),   # steel-blue person box
    thresh_line  = (180,  50, 180),   # magenta 2/3-height gate line
    racket_up    = ( 40, 220,  40),   # green — racquet passes upward test
    racket_no    = ( 40, 120, 220),   # orange — racquet fails upward test
    crop_box     = (220, 220,  40),   # cyan dashed crop-search window
    ball_raw     = (100, 220, 220),   # yellow circle — raw ball detection
    trail_norm   = ( 80, 200, 255),   # warm-yellow ball trail (normal)
    trail_toss   = ( 40, 255,  40),   # bright-green trail during a toss segment
    audio_bar    = ( 40, 220, 255),   # audio-impact flash bar
    hud_fg       = (230, 230, 230),
    serve_bg     = ( 30, 160,  30),   # green banner background
    serve_fg     = (255, 255, 255),
    gate_ok      = ( 40, 220,  40),
    gate_no      = (100, 100, 100),
    court_line   = ( 80, 200,  80),   # green court polygon
    near_side    = ( 60, 200, 255),   # amber — near player
    far_side     = (255, 170,  60),   # blue  — far player
    diag_trace   = (255,  80, 200),   # purple diagonal-trace highlight
    pos_band     = ( 80, 255, 160),   # near-baseline serve-position band
)


def _dashed_rect(img, pt1: tuple, pt2: tuple,
                 color: tuple, thickness: int = 1, dash: int = 7) -> None:
    """Draw a dashed rectangle (OpenCV has no built-in)."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
             ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for (ax, ay), (bx, by) in edges:
        dx, dy = bx - ax, by - ay
        length = max(1, int(np.hypot(dx, dy)))
        nx, ny = dx / length, dy / length
        t = 0
        draw = True
        while t < length:
            t2 = min(t + dash, length)
            if draw:
                p1 = (int(ax + nx * t),  int(ay + ny * t))
                p2 = (int(ax + nx * t2), int(ay + ny * t2))
                _cv2.line(img, p1, p2, color, thickness, _cv2.LINE_AA)
            t += dash
            draw = not draw


def _text_bg(img, text: str, org: tuple,
             scale: float = 0.45, fg: tuple = (230, 230, 230),
             bg: tuple = (0, 0, 0), thick: int = 1) -> None:
    """White text on a filled black rectangle."""
    (tw, th), bl = _cv2.getTextSize(text, _cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    _cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + bl + 1), bg, -1)
    _cv2.putText(img, text, (x, y), _cv2.FONT_HERSHEY_SIMPLEX,
                 scale, fg, thick, _cv2.LINE_AA)


def _cross(img, cx: int, cy: int, size: int, color: tuple, thick: int = 2) -> None:
    _cv2.line(img, (cx - size, cy), (cx + size, cy), color, thick, _cv2.LINE_AA)
    _cv2.line(img, (cx, cy - size), (cx, cy + size), color, thick, _cv2.LINE_AA)


def _all_toss_segments(tracks: list[Track], cfg: TossConfig) -> list[dict]:
    """Return every (generic) toss segment across all tracks, regardless of gate
    outcome — used by the visualizer to show candidate tosses."""
    segs: list[dict] = []
    for tr in tracks:
        feat = compute_features(tr, cfg)
        if feat is None:
            continue
        segs.extend(_generic_toss_segments(feat, tr.id, cfg))
    return segs


def render_visualization(
    video_path: str,
    cfg: TossConfig,
    geom: "CourtGeometry",
    ball_stream: list[list[Detection]],
    rackets: dict[int, list[RacketDet]],
    persons_all: dict[int, list[PersonDet]],
    near_by_frame: dict[int, PersonDet],
    far_by_frame: dict[int, PersonDet],
    tracks: list[Track],
    toss_segs: list[dict],
    serves: list[TossServeEvent],
    audio_impacts: list[float],
    excl_zones: list = None,
    start_frame: int = 0,
    viz: Optional[VizConfig] = None,
) -> None:
    """
    Annotated combined (near + far) renderer. Overlays (back → front):
      • Singles-court polygon (from the homography)
      • Near-baseline serve-position band (the [-3.5, +0.5] ft zone)
      • Ball track trails (green during a toss, yellow otherwise) + diagonal
        post-contact trace highlighted purple
      • Raw ball detection circles
      • Player boxes coloured by side (amber near / blue far) with the side +
        world (wx,wy) label and the magenta 2/3-height racquet-gate line
      • Dashed toss-crop and far-racquet-crop windows on the chosen players
      • Racquet boxes (green = upward PASS / orange = fail) with aspect + conf
      • Toss crosshair + label at the apex
      • Audio-impact flash bar
      • Per-side SERVE banner after contact
      • Gate-status HUD (POS/TOSS/RACKET/AUDIO/DIAG/SERVE) while a toss is active
      • Frame counter + absolute timestamp
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("opencv-python required for visualization.")
    if viz is None:
        viz = VizConfig()

    W, H = cfg.frame_w, cfg.frame_h
    n_frames = len(ball_stream)

    # ── Per-toss gate pre-computation (side-aware) ─────────────────────────── #
    serve_keys = {(ev.track_id, ev.apex_frame): ev for ev in serves}
    diag_track_ids = set()
    for ev in serves:
        if ev.side == "far" and ev.notes.get("diagonal"):
            diag_track_ids.add(ev.track_id)

    toss_at: dict[int, list[dict]] = {}
    toss_track_ids: set[int] = {s["track_id"] for s in toss_segs}
    hud_active: dict[int, dict] = {}
    for seg in toss_segs:
        for f in range(seg["toss_start_f"], seg["apex_f"] + 1):
            toss_at.setdefault(f, []).append(seg)
        side = _assign_toss_side(seg, near_by_frame, far_by_frame, cfg)
        key = (seg["track_id"], seg["apex_f"])
        rk = find_racket_up_evidence(seg["apex_f"], seg["apex_cx"],
                                     rackets, persons_all, cfg)
        audio_ok = any(abs(t - seg["apex_f"] / cfg.fps) <= cfg.audio_window_s
                       for t in audio_impacts)
        pos_ok = (side == "near" and
                  _near_position_ok(seg["toss_start_f"], near_by_frame, cfg) is not None)
        ev = serve_keys.get(key)
        info = dict(seg, side=side, rk_ok=rk is not None, audio_ok=audio_ok,
                    pos_ok=pos_ok,
                    diag_ok=(ev.notes.get("diagonal") if ev else False),
                    is_serve=ev is not None)
        for f in range(max(0, seg["toss_start_f"] - 10),
                       min(n_frames, seg["apex_f"] + 90)):
            if f not in hud_active or (abs(f - seg["apex_f"]) <
                                       abs(f - hud_active[f]["apex_f"])):
                hud_active[f] = info

    serve_at: dict[int, TossServeEvent] = {}
    for ev in serves:
        for f in range(ev.apex_frame,
                       min(n_frames, ev.contact_frame + viz.serve_banner_frames)):
            serve_at[f] = ev

    audio_frame_set = {int(round(t * cfg.fps)) for t in audio_impacts}
    half_rk = cfg.far_racket_window_px // 2

    # ── Video open + writer ────────────────────────────────────────────────── #
    cap = _cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = _cv2.VideoWriter(viz.out_path, _cv2.VideoWriter_fourcc(*viz.fourcc),
                              cfg.fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"[VIZ] Cannot open VideoWriter → {viz.out_path}")

    court_poly = np.array([[int(x), int(y)] for x, y in geom.corners_px()],
                          dtype=np.int32)

    print(f"[VIZ] Rendering {n_frames} frames → {viz.out_path} …")
    if viz.show_live:
        _cv2.namedWindow("Combined Serve Tracker", _cv2.WINDOW_NORMAL)
        _cv2.resizeWindow("Combined Serve Tracker", min(W, 1280), min(H, 720))

    def _toss_crop_box(p: PersonDet, w_, h_, bot_off):
        bot = p.box_top + bot_off
        return (int(p.cx - w_ / 2), int(bot - h_), int(p.cx + w_ / 2), int(bot))

    for fi in range(n_frames):
        ret, orig = cap.read()
        if not ret:
            break
        frame = _cv2.resize(orig, (W, H), interpolation=_cv2.INTER_LINEAR)

        # 1. Court polygon
        if viz.draw_court:
            _cv2.polylines(frame, [court_poly], True, _VC["court_line"], 2, _cv2.LINE_AA)

        # 1b. Static exclusion zones (red bounding boxes, per run_anya style)
        for (ex1, ey1, ex2, ey2) in (excl_zones or []):
            _cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 220), 2)
            _text_bg(frame, "EXCL", (ex1 + 2, ey1 + 14),
                     scale=0.35, fg=(0, 0, 220), bg=(0, 0, 0))

        # 1c. Near-baseline serve-position band (project the wy band to pixels)
        if viz.draw_near_band:
            try:
                Hinv = np.linalg.inv(geom.H)
                band = []
                for wy in (-cfg.near_pos_behind_ft, cfg.near_pos_into_ft):
                    row = []
                    for wx in (0, cfg.court_width_ft):
                        pt = np.array([[[wx, wy]]], dtype=np.float32)
                        q = _cv2.perspectiveTransform(pt, Hinv)[0][0]
                        row.append((int(q[0]), int(q[1])))
                    band.append(row)
                poly = np.array([band[0][0], band[0][1], band[1][1], band[1][0]],
                                dtype=np.int32)
                ov = frame.copy()
                _cv2.fillPoly(ov, [poly], _VC["pos_band"])
                _cv2.addWeighted(ov, 0.15, frame, 0.85, 0, frame)
                _cv2.polylines(frame, [poly], True, _VC["pos_band"], 1, _cv2.LINE_AA)
            except Exception:
                pass

        # 2. Ball trails (+ diagonal highlight)
        if viz.draw_ball_trail:
            for tr in tracks:
                if tr.id in diag_track_ids:
                    color = _VC["diag_trace"]
                elif tr.id in toss_track_ids:
                    color = _VC["trail_toss"]
                else:
                    color = _VC["trail_norm"]
                pts = [(int(p.cx), int(p.cy)) for p in tr.history
                       if fi - viz.trail_frames <= p.frame <= fi]
                for k in range(1, len(pts)):
                    a = k / max(1, len(pts))
                    _cv2.line(frame, pts[k - 1], pts[k],
                              tuple(int(c * a) for c in color), 2, _cv2.LINE_AA)
                if pts:
                    _cv2.circle(frame, pts[-1], 4, color, -1, _cv2.LINE_AA)

        # 3. Raw ball detections
        if viz.draw_raw_dets:
            for d in ball_stream[fi]:
                _cv2.circle(frame, (int(d.cx), int(d.cy)), 5,
                            _VC["ball_raw"], 1, _cv2.LINE_AA)

        # 4. Player boxes (coloured by side) + 2/3 line + world label
        if viz.draw_persons:
            near_p = near_by_frame.get(fi)
            far_p = far_by_frame.get(fi)
            for p in persons_all.get(fi, []):
                col = (_VC["near_side"] if p.side == "near" else
                       _VC["far_side"] if p.side == "far" else (140, 140, 140))
                chosen = (p is near_p) or (p is far_p)
                bx1 = int(p.cx - p.w / 2); by1 = int(p.cy - p.h / 2)
                bx2 = int(p.cx + p.w / 2); by2 = int(p.cy + p.h / 2)
                _cv2.rectangle(frame, (bx1, by1), (bx2, by2), col,
                               2 if chosen else 1)
                if viz.draw_world and p.side:
                    lbl = f"{p.side}"
                    if p.world is not None:
                        lbl += f" ({p.world[0]:.0f},{p.world[1]:.0f}ft)"
                    _text_bg(frame, lbl, (bx1, max(10, by1 - 4)),
                             scale=0.40, fg=col, bg=(0, 0, 0))
                if chosen:                      # 2/3 racquet-gate line
                    thresh_cy = int(p.cy + p.h / 2.0 - cfg.overhead_frac * p.h)
                    _cv2.line(frame, (bx1, thresh_cy), (bx2, thresh_cy),
                              _VC["thresh_line"], 1, _cv2.LINE_AA)
                    _text_bg(frame, "2/3", (bx1 + 3, thresh_cy - 3),
                             scale=0.33, fg=_VC["thresh_line"], bg=(0, 0, 0))

            # Toss-crop + far-racquet-crop windows on the chosen players
            if viz.draw_crop_window:
                if far_p is not None:               # far toss crop (60x100) + racquet crop
                    x1, y1, x2, y2 = _toss_crop_box(far_p, cfg.far_toss_w,
                                                    cfg.far_toss_h, cfg.far_toss_bottom_off)
                    _dashed_rect(frame, (x1, y1), (x2, y2), _VC["crop_box"], 1, 6)
                    _dashed_rect(frame,
                                 (int(far_p.cx - half_rk), int(far_p.cy - half_rk)),
                                 (int(far_p.cx + half_rk), int(far_p.cy + half_rk)),
                                 _VC["far_side"], 1, 8)
                if near_p is not None:              # near toss crop (mirrored)
                    x1, y1, x2, y2 = _toss_crop_box(near_p, cfg.near_toss_w,
                                                    cfg.near_toss_h, cfg.near_toss_bottom_off)
                    _dashed_rect(frame, (x1, y1), (x2, y2), _VC["crop_box"], 1, 6)

        # 5. Racquet boxes
        if viz.draw_rackets:
            for r in rackets.get(fi, []):
                up, _ = _racket_is_upward(r, persons_all.get(fi, []), cfg)
                col = _VC["racket_up"] if up else _VC["racket_no"]
                rx1 = int(r.cx - r.w / 2); ry1 = int(r.cy - r.h / 2)
                rx2 = int(r.cx + r.w / 2); ry2 = int(r.cy + r.h / 2)
                _cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col, 2)
                _text_bg(frame, f"{'UP' if up else '--'} {r.aspect:.1f} {r.conf:.2f}",
                         (rx1, max(10, ry1 - 4)), scale=0.38, fg=col, bg=(0, 0, 0))

        # 6. Toss crosshair
        for seg in toss_at.get(fi, []):
            ax, ay = int(seg["apex_cx"]), int(seg["apex_cy"])
            _cross(frame, ax, ay, 12, _VC["trail_toss"], 2)
            _cv2.circle(frame, (ax, ay), 14, _VC["trail_toss"], 1, _cv2.LINE_AA)
            _text_bg(frame, f"TOSS cv={seg['area_cv']:.2f}", (ax + 16, ay + 5),
                     scale=0.42, fg=_VC["trail_toss"], bg=(0, 0, 0))

        # 7. Audio flash bar
        if viz.draw_audio and fi in audio_frame_set:
            _cv2.rectangle(frame, (0, H - 8), (W, H), _VC["audio_bar"], -1)
            _text_bg(frame, "AUDIO IMPACT", (W // 2 - 55, H - 11),
                     scale=0.42, fg=(0, 0, 0), bg=_VC["audio_bar"])

        # 8. Serve banner (per side)
        if fi in serve_at:
            ev = serve_at[fi]
            by_ctr = (H // 2 - 70) if ev.side == "far" else (H // 2 + 70)
            ov = frame.copy()
            _cv2.rectangle(ov, (0, by_ctr - 20), (W, by_ctr + 20), _VC["serve_bg"], -1)
            _cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
            txt = (f"{ev.side.upper()} SERVE   score={ev.score:.2f}"
                   f"   t={(start_frame + ev.contact_frame) / cfg.fps:.2f}s")
            (tw, th), _ = _cv2.getTextSize(txt, _cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
            _cv2.putText(frame, txt, ((W - tw) // 2, by_ctr + th // 2),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.70, _VC["serve_fg"], 2, _cv2.LINE_AA)

        # 9. Gate HUD
        if fi in hud_active:
            info = hud_active[fi]
            side = info["side"] or "?"
            rows = [(f"SIDE: {side}", True, _VC["hud_fg"])]
            if side == "near":
                rows.append(("POS", info["pos_ok"],
                             _VC["gate_ok"] if info["pos_ok"] else _VC["gate_no"]))
            rows += [
                ("TOSS", True, _VC["trail_toss"]),
                ("RACKET", info["rk_ok"], _VC["gate_ok"] if info["rk_ok"] else _VC["gate_no"]),
                ("AUDIO", info["audio_ok"], _VC["audio_bar"] if info["audio_ok"] else _VC["gate_no"]),
            ]
            if side == "far":
                rows.append(("DIAG", info["diag_ok"],
                             _VC["diag_trace"] if info["diag_ok"] else _VC["gate_no"]))
            rows.append(("SERVE", info["is_serve"],
                         _VC["serve_bg"] if info["is_serve"] else _VC["gate_no"]))
            for i, (label, ok, col) in enumerate(rows):
                txt = label if label.startswith("SIDE") else f"{label}: {'OK' if ok else '--'}"
                _text_bg(frame, txt, (W - 132, 16 + i * 22), scale=0.46, fg=col, bg=(0, 0, 0))

        # 10. Frame counter
        abs_f = start_frame + fi
        t_abs = abs_f / cfg.fps
        _text_bg(frame, f"f={abs_f}  {int(t_abs // 60):02d}:{t_abs % 60:05.2f}",
                 (8, H - 10), scale=0.42, fg=_VC["hud_fg"], bg=(0, 0, 0))

        writer.write(frame)
        if viz.show_live:
            _cv2.imshow("Combined Serve Tracker", frame)
            if _cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if fi % 300 == 0 and fi > 0:
            print(f"[VIZ]   rendered {fi}/{n_frames} …")

    cap.release()
    writer.release()
    if viz.show_live:
        _cv2.destroyAllWindows()
    print(f"[VIZ] Done → {viz.out_path}")


# --------------------------------------------------------------------------- #
# Synthetic offline demo: exercises toss + racquet-up + optional audio
# --------------------------------------------------------------------------- #
def _synthetic_combined(cfg: TossConfig, geom: "CourtGeometry", rng):
    """Build a synthetic combined scene with one NEAR serve and one FAR serve
    (plus a far toss that lacks corroboration and must be rejected)."""
    N = 320
    frames: list[list[Detection]] = [[] for _ in range(N)]
    rackets: dict[int, list[RacketDet]] = {}
    persons_all: dict[int, list[PersonDet]] = {f: [] for f in range(N)}
    contacts: list[int] = []

    def far_serve(t0, with_racket, with_audio, with_diag):
        # far player near top baseline (small box)
        px, py, pw, ph = 640, 120, 34, 74           # box center; foot_y≈157
        top = py - ph / 2
        for f in range(t0 - 6, t0 + 40):
            if 0 <= f < N:
                persons_all[f].append(PersonDet(f, px, py, pw, ph, 0.9))
        rise = 12; apex = t0 + rise
        for k in range(rise + 1):                    # toss rises above head
            f = t0 + k
            cy = (top - 5) - 55 * np.sin((np.pi / 2) * k / rise)
            frames[f].append(Detection(f, px + rng.normal(0, 0.6), cy,
                                       6, 6, 0.7))
            if with_racket and k >= rise - 3:
                rackets.setdefault(f, []).append(
                    RacketDet(f, px + 5, top - 18, 8, 22, 0.5))
        if with_audio:
            contacts.append(apex)
        if with_diag:                                # diagonal down + right
            for k in range(1, 16):
                f = apex + k
                if f >= N:
                    break
                frames[f].append(Detection(f, px + 6 * k, (top + 5) + 9 * k,
                                           7, 7, 0.7))

    def near_serve(t0, in_position, with_racket, with_audio):
        # near player at bottom baseline (large box). foot_y≈650 -> wy≈0 (in pos)
        # if out of position, place foot well inside the court (wy large).
        py = 560 if in_position else 470
        px, pw, ph = 640, 80, 180
        top = py - ph / 2
        for f in range(t0 - 40, t0 + 40):
            if 0 <= f < N:
                persons_all[f].append(PersonDet(f, px, py, pw, ph, 0.95))
        rise = 16; apex = t0 + rise
        for k in range(rise + 1):                    # large near ball rises
            f = t0 + k
            cy = (top - 10) - 150 * np.sin((np.pi / 2) * k / rise)
            frames[f].append(Detection(f, px + rng.normal(0, 1.0), cy,
                                       14, 14, 0.85))
            if with_racket and k >= rise - 4:
                rackets.setdefault(f, []).append(
                    RacketDet(f, px + 10, top - 40, 18, 48, 0.7))
        if with_audio:
            contacts.append(apex)

    near_serve(40, in_position=True, with_racket=True, with_audio=True)   # valid near
    far_serve(150, with_racket=True, with_audio=True, with_diag=True)      # valid far
    far_serve(240, with_racket=False, with_audio=False, with_diag=False)  # reject (toss only)

    for f in range(N):                               # parked ball (static zone)
        frames[f].append(Detection(f, 300 + rng.normal(0, 0.4),
                                   400 + rng.normal(0, 0.4), 8, 8, 0.6))

    # Classify players + pick near/far per frame (mirror _build_streams logic).
    near_by_frame: dict[int, PersonDet] = {}
    far_by_frame: dict[int, PersonDet] = {}
    for f, plist in persons_all.items():
        for p in plist:
            p.side, p.world = geom.classify(p.cx, p.foot_y)
        near = [p for p in plist if p.side == "near"]
        far = [p for p in plist if p.side == "far"]
        if near:
            near_by_frame[f] = max(near, key=lambda q: q.w * q.h)
        if far:
            far_by_frame[f] = max(far, key=lambda q: q.world[1] if q.world else 0)
    return frames, rackets, persons_all, near_by_frame, far_by_frame, contacts


def _synthetic_audio(cfg: TossConfig, contact_frames, n_frames, rng):
    sr = 16000
    x = 0.01 * rng.standard_normal(int(sr * n_frames / cfg.fps))

    def pock(t):
        i0 = int(t * sr)
        L = int(0.006 * sr)
        x[i0:i0 + L] += np.exp(-np.linspace(0, 6, L)) * rng.standard_normal(L) * 0.9

    for cf in contact_frames:
        pock(cf / cfg.fps)
    return x, sr


def main() -> None:
    cfg = TossConfig()
    cfg.frame_w, cfg.frame_h = cfg.analysis_w, cfg.analysis_h
    rng = np.random.default_rng(7)

    # Synthetic singles court (trapezoid) for a 1280x720 frame.
    geom = CourtGeometry([(240, 650), (1040, 650), (840, 150), (440, 150)], cfg)

    stream, rackets, persons_all, near_bf, far_bf, contacts = \
        _synthetic_combined(cfg, geom, rng)
    audio, sr = _synthetic_audio(cfg, contacts, len(stream), rng)

    zones = find_static_exclusion_zones(stream, cfg)
    clean = apply_exclusion_zones(stream, zones, cfg)
    tracks = KalmanBallTracker(cfg).run(clean)
    impacts = onset_times_from_samples(audio, sr, cfg)
    print(f"Static zones: {len(zones)}  tracks: {len(tracks)}  "
          f"audio impacts: {len(impacts)}  "
          f"near-frames={len(near_bf)} far-frames={len(far_bf)}")

    # Offline demo provides a full ball stream directly, so tosses + diagonals
    # both come from the same tracks (no gated Pass 2 here).
    tosses = compute_tosses_with_side(tracks, near_bf, far_bf, cfg)
    diagonals = []
    for tr in tracks:
        feat = compute_features(tr, cfg)
        if feat is not None:
            diagonals.extend(_diagonal_segments(feat, tr.id, cfg))
    serves = detect_serves_combined(
        tosses, diagonals, rackets, persons_all, near_bf, far_bf, impacts, cfg)
    n_near = sum(1 for s in serves if s.side == "near")
    n_far = sum(1 for s in serves if s.side == "far")
    print(f"\nValid serves: {len(serves)}  (near={n_near} far={n_far})")
    for i, ev in enumerate(serves, 1):
        print(f"  #{i}  [{ev.side.upper():>4}]  contact @ {ev.hhmmss()}  "
              f"track={ev.track_id}  apex_f={ev.apex_frame}  "
              f"score={ev.score:.2f}  {ev.notes}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Combined near + far serve tracker (court homography, "
                    "toss + racquet-up + audio/diagonal corroboration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
On first run you click the 4 SINGLES-COURT corners (any order); the homography
is cached next to the video. NEAR serves require a baseline foot-position +
toss + racquet-up (audio optional). FAR serves require a toss + 2 of
{audio, racquet-up, diagonal ball trace}.

Examples:
  # Synthetic demo (no video, no cv2 needed):
  python farside_serve_tracker_toss.py

  # Real video, detection only (4-corner calibration on first run):
  python farside_serve_tracker_toss.py video.mp4

  # Real video + annotated output video:
  python farside_serve_tracker_toss.py video.mp4 --viz out.mp4

  # Real video + live display + annotated output, 1000-frame window:
  python farside_serve_tracker_toss.py video.mp4 --viz out.mp4 --show --max-frames 1000

  # Force court re-calibration:
  python farside_serve_tracker_toss.py video.mp4 --recalibrate
""")
    p.add_argument("video", nargs="?", default=None)
    p.add_argument("--ball-model", default=_DEFAULT_BALL_MODEL)
    p.add_argument("--pose-model", default=None,
                   help="General COCO detector for person+racket (default yolo26n.pt).")
    p.add_argument("--audio", default=None)
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--calib-cache", default=_CALIB_CACHE)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--conf", type=float, default=None,
                   help="YOLO ball confidence threshold.")
    p.add_argument("--require-audio", action="store_true",
                   help="Make the audio impact mandatory (off by default).")
    p.add_argument("--viz", default=None, metavar="OUT.mp4",
                   help="Write an annotated visualization video to this path.")
    p.add_argument("--show", action="store_true",
                   help="Display the visualization live while writing (requires --viz).")
    p.add_argument("--trail", type=int, default=25, metavar="N",
                   help="Ball trail length in frames for visualization (default 25).")
    p.add_argument("--stride", type=int, default=None, metavar="N",
                   help="Person-detection stride (frames). Higher = faster (default 3).")
    p.add_argument("--no-gate", action="store_true",
                   help="Disable the gated 2-pass ball detection (full-frame ball "
                        "every frame; slower but simpler).")
    p.add_argument("--no-live", action="store_true",
                   help="Suppress the automatic live display window (still saves "
                        "the annotated video to file).")
    args = p.parse_args()

    if args.video is None:
        main()
    else:
        _cfg = TossConfig()
        if args.conf is not None:
            _cfg.ball_conf = args.conf
        if args.pose_model is not None:
            _cfg.pose_model_path = args.pose_model
        if args.require_audio:
            _cfg.require_audio = True
        if args.stride is not None:
            _cfg.player_stride = max(1, args.stride)
        if args.no_gate:
            _cfg.gated_fullframe_ball = False
        # Explicit --viz / --show still work; otherwise the auto-default in
        # run_on_video_toss handles the live viz (auto-show + save-to-file).
        _viz = None
        if args.viz or args.show:
            _viz = VizConfig(
                out_path=args.viz or "farside_serve_viz.mp4",
                show_live=args.show,
                trail_frames=args.trail,
            )
        elif args.no_live:
            # Disable auto-show: pass an explicit VizConfig with show_live=False
            default_viz_path = os.path.splitext(args.video)[0] + "_serve_viz.mp4"
            _viz = VizConfig(out_path=default_viz_path, show_live=False,
                             trail_frames=args.trail)
        run_on_video_toss(
            args.video,
            cfg=_cfg,
            ball_model_path=args.ball_model,
            audio_path=args.audio,
            recalibrate=args.recalibrate,
            calib_cache=args.calib_cache,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            viz_cfg=_viz,
        )
