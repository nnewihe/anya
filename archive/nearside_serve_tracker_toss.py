"""
Near-side tennis serve tracker — toss + racquet-up (+ optional audio).

Detect serves by the NEAR-SIDE player (bottom of frame, large, close to camera).

Gate (in order within a short window):
  (a) FOOT POSITION  — player foot within serve-zone bounds near the near
                       baseline (world-y in [-behind, +into] ft) at some point
                       in the lookback window BEFORE the toss starts.  REQUIRED.
  (b1) BALL TOSS     — ball tracklet with upward trajectory above the near
                       player's head.  REQUIRED.
  (b2) RACQUET UP    — upward-oriented racquet near the apex (tall bbox,
                       overhead position relative to player box).  REQUIRED by
                       default (near_require_racket=True).
  (c)  AUDIO IMPACT  — spectral-flux onset near the apex.  OPTIONAL — boosts
                       score only; absence never rejects a serve.

Compared to farside_serve_tracker_toss.py this file:
  • Only tracks and gates the NEAR player — no far-side logic, no diagonal
    trace, no gated Pass-2 full-frame ball pass.
  • Runs a single cheap pass: strided person detection + player-anchored toss
    and racket crops (no full-frame ball at all by default).
  • Removes far_min_signals / diagonal-corroboration machinery.
  • ~40 % less per-frame compute than the combined tracker.

Camera geometry: behind the baseline, ~10 ft high; image +y is DOWN.  The near
player is near the BOTTOM of the frame and LARGE.  A toss makes the ball's cy
DECREASE; the racquet appears as a tall vertical box raised above the torso.

Dependencies: numpy (required).  opencv-python + ultralytics are needed only
for `run_on_video_nearside`; the synthetic demo runs offline with pure numpy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Reuse ball-tracking / feature primitives already validated in v2 ----------- #
from farside_serve_detector_v2 import (
    Config,
    Detection,
    Track,
    KalmanBallTracker,
    find_static_exclusion_zones,
    apply_exclusion_zones,
    compute_features,
    onset_times_from_samples,
    detect_audio_impacts,
    _load_calib,
    _try_extract_audio,
    _CALIB_CACHE,
    _CV2_AVAILABLE,
)

try:
    import cv2 as _cv2
    from ultralytics import YOLO as _YOLO
except ImportError:  # pragma: no cover
    _cv2 = None       # type: ignore
    _YOLO = None      # type: ignore


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class NearConfig(Config):
    # --- Racquet (COCO "tennis racket") detector ---
    pose_model_path: str = "yolo26n.pt"
    racket_conf: float = 0.20
    racket_imgsz: int = 1280
    racket_class_names: tuple = ("tennis racket", "tennis racquet")
    person_class_name: str = "person"

    # --- "Upward orientation" definition ---
    racket_aspect_min: float = 1.15       # bbox h/w >= this → tall/vertical
    require_overhead: bool = True         # racket center must be in top third of player box
    overhead_frac: float = 2.0 / 3.0
    person_link_px: float = 300.0         # larger: near player box is large
    racket_min_area: float = 8.0

    # --- Player-anchored racquet crop ---
    near_racket_window_px: int = 300      # crop centered on player (larger near box)
    near_racket_conf: float = 0.10
    near_racket_dedup_px: float = 20.0
    racket_crop_imgsz: int = 512

    # --- Toss → racquet linkage ---
    racket_window_s: float = 0.8
    racket_link_px: float = 350.0         # wider: near ball drifts laterally more

    # --- Gating switches ---
    near_require_racket: bool = True
    require_audio: bool = False           # audio optional; only boosts score
    audio_window_s: float = 0.30
    serve_cooldown_s: float = 2.5

    # --- Analysis resolution ---
    analysis_w: int = 1280
    analysis_h: int = 720

    # --- Court / world geometry (singles court) ---
    court_width_ft: float = 27.0
    court_length_ft: float = 78.0
    court_cache: str = "nearside_court.json"

    # --- Player detection ---
    player_conf: float = 0.30
    player_imgsz: int = 960
    player_stride: int = 3               # detect persons every N frames, hold between

    # --- Near-baseline foot-position gate (world feet, near baseline = wy 0) ---
    near_pos_into_ft: float = 0.5        # foot may be up to this far INSIDE court
    near_pos_behind_ft: float = 3.5      # ...and this far BEHIND the baseline
    near_pos_lookback_s: float = 1.5     # must hold within this window before the toss

    # --- Near-side toss crop (large: near ball is big) ---
    near_toss_w: int = 180
    near_toss_h: int = 320
    near_toss_bottom_off: int = 20       # crop bottom = player_box_top + this
    near_toss_imgsz: int = 928
    near_toss_conf: float = 0.25

    # --- Ball detection ---
    ball_conf: float = 0.25
    ball_dedup_px: float = 12.0

    # --- x-band padding for near-baseline player classification ---
    xband_pad_frac: float = 0.35
    xband_pad_min_px: float = 60.0


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
    world: Optional[tuple] = None        # (wx, wy) ft of bottom-center foot

    @property
    def box_top(self) -> float:
        return self.cy - self.h / 2.0

    @property
    def foot_y(self) -> float:
        return self.cy + self.h / 2.0


@dataclass
class NearServeEvent:
    toss_start_frame: int
    apex_frame: int
    contact_frame: int
    fps: float
    track_id: int
    score: float = 1.0
    notes: dict = field(default_factory=dict)

    def t(self) -> float:
        return self.contact_frame / self.fps

    def hhmmss(self) -> str:
        s = self.t()
        return f"{int(s//3600):02d}:{int(s%3600//60):02d}:{s%60:06.3f}"


# --------------------------------------------------------------------------- #
# JSON I/O  (imported by nearside_point_end_tracker as the single source)
# --------------------------------------------------------------------------- #
def save_events(events: list["NearServeEvent"], path: str) -> None:
    """Serialise a list of NearServeEvent objects to JSON."""
    data = [
        {
            "toss_start_frame": ev.toss_start_frame,
            "apex_frame":       ev.apex_frame,
            "contact_frame":    ev.contact_frame,
            "fps":              ev.fps,
            "track_id":         ev.track_id,
            "score":            ev.score,
            "notes":            ev.notes,
        }
        for ev in events
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[EVENTS] Saved {len(events)} serve events → {path}")


def load_events(path: str) -> list["NearServeEvent"]:
    """Load NearServeEvent objects from a JSON file written by save_events."""
    with open(path) as f:
        data = json.load(f)
    events = [
        NearServeEvent(
            toss_start_frame=int(d["toss_start_frame"]),
            apex_frame=int(d["apex_frame"]),
            contact_frame=int(d["contact_frame"]),
            fps=float(d["fps"]),
            track_id=int(d["track_id"]),
            score=float(d.get("score", 1.0)),
            notes=d.get("notes", {}),
        )
        for d in data
    ]
    print(f"[EVENTS] Loaded {len(events)} serve events ← {path}")
    return events


# --------------------------------------------------------------------------- #
# Court geometry — homography from 4 singles-court corners
# Used only for foot-position world-coordinate gate (near baseline = wy≈0).
# --------------------------------------------------------------------------- #
class CourtGeometry:
    """
    Maps image pixels → world feet via a homography from the four singles-court
    corners.  World frame: near baseline → wy=0, far baseline → wy=court_len,
    left sideline → wx=0, right sideline → wx=court_width.

    Corners are stored ordered BL, BR, TR, TL (bottom-left → bottom-right →
    top-right → top-left in IMAGE space).
    """

    def __init__(self, corners_clicked, cfg: NearConfig):
        self.cfg = cfg
        self.BL, self.BR, self.TR, self.TL = self._order_corners(corners_clicked)
        self.H = self._compute_homography()
        def _pad(lo, hi):
            p = max(cfg.xband_pad_min_px, cfg.xband_pad_frac * (hi - lo))
            return lo - p, hi + p
        self.near_x_lo, self.near_x_hi = _pad(min(self.BL[0], self.BR[0]),
                                               max(self.BL[0], self.BR[0]))
        self.near_baseline_y = (self.BL[1] + self.BR[1]) / 2.0

    @staticmethod
    def _order_corners(pts):
        pts = [(float(x), float(y)) for x, y in pts]
        if len(pts) != 4:
            raise ValueError(f"Need exactly 4 court corners, got {len(pts)}.")
        ys = sorted(pts, key=lambda p: p[1])
        top2, bot2 = ys[:2], ys[2:]
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
        return self._dlt_homography(src, dst)

    @staticmethod
    def _dlt_homography(src, dst):
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

    def is_near_player(self, box_cx: float, foot_y: float) -> tuple[bool, tuple]:
        """Return (is_near, (wx, wy)).  Near: world-y closer to 0 than court
        midpoint AND pixel-x within the padded near-baseline x-band."""
        wx, wy = self.world_pos(box_cx, foot_y)
        mid = self.cfg.court_length_ft / 2.0
        near = (wy < mid and self.near_x_lo <= box_cx <= self.near_x_hi)
        return near, (wx, wy)

    def corners_px(self):
        return [self.BL, self.BR, self.TR, self.TL]


def init_court_geometry(video_path: str, cfg: NearConfig,
                        recalibrate: bool = False) -> CourtGeometry:
    from utilities import init_court
    analysis_size = (cfg.analysis_w, cfg.analysis_h)
    if recalibrate:
        try:
            from utilities import _court_cache_path
            cp = _court_cache_path(video_path)
            if cp and os.path.exists(cp):
                os.remove(cp)
                print(f"[COURT] Removed cached corners → recalibrating ({cp}).")
        except Exception:
            pass
    corners, _shape = init_court(video_path, analysis_size=analysis_size)
    geom = CourtGeometry(corners, cfg)
    print(f"[COURT] Corners (BL,BR,TR,TL): "
          f"{[tuple(round(v, 1) for v in c) for c in geom.corners_px()]}")
    print(f"[COURT] Near baseline y≈{geom.near_baseline_y:.0f}px "
          f"x∈[{geom.near_x_lo:.0f},{geom.near_x_hi:.0f}]")
    return geom


# --------------------------------------------------------------------------- #
# Racquet-up evidence
# --------------------------------------------------------------------------- #
def _racket_is_upward(
    r: RacketDet,
    persons_in_frame: list[PersonDet],
    cfg: NearConfig,
) -> tuple[bool, dict]:
    """A racquet counts as 'upward' when its bbox is tall/vertical and (if a
    player is nearby) its center sits in the top third of that player's box."""
    info = {"aspect": round(r.aspect, 2), "conf": round(r.conf, 2)}
    if r.area < cfg.racket_min_area:
        return False, info
    if r.aspect < cfg.racket_aspect_min:
        return False, info

    if cfg.require_overhead and persons_in_frame:
        p = min(persons_in_frame,
                key=lambda q: np.hypot(q.cx - r.cx, q.cy - r.cy))
        if np.hypot(p.cx - r.cx, p.cy - r.cy) <= cfg.person_link_px:
            p_bottom = p.cy + p.h / 2.0
            thresh_cy = p_bottom - cfg.overhead_frac * p.h
            info["player_dy"] = round(p.cy - r.cy, 1)
            info["thresh_cy"] = round(thresh_cy, 1)
            if r.cy >= thresh_cy:
                return False, info
    return True, info


def find_racket_up_evidence(
    apex_frame: int,
    apex_cx: float,
    rackets: dict[int, list[RacketDet]],
    persons: dict[int, list[PersonDet]],
    cfg: NearConfig,
) -> Optional[dict]:
    """Search +/- window around the toss apex for an upward racquet near the
    toss column.  Returns the best (closest-to-apex) match, or None."""
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
# Toss segments — near player only (no ROI x-mask; apex must be above player)
# --------------------------------------------------------------------------- #
def _near_toss_segments(feat, track_id: int, near_player: PersonDet,
                        cfg: NearConfig) -> list[dict]:
    """Rising + vertical + ~constant-area runs that apex ABOVE the near player's
    head.  Player box is checked at the apex frame to anchor the toss to the
    correct player."""
    from farside_serve_detector_v2 import _contiguous_runs
    mask = (feat.vy < -cfg.rise_vy_px_s) & (feat.verticality > cfg.verticality_min)
    out = []
    for (s, e) in _contiguous_runs(mask, cfg.min_rise_frames):
        seg = feat.area[s:e + 1]
        cv = float(np.std(seg) / (np.mean(seg) + 1e-6))
        if cv > cfg.area_cv_max:
            continue
        apex_idx = s + int(np.argmin(feat.cy[s:e + 1]))
        ax = float(feat.cx[apex_idx])
        ay = float(feat.cy[apex_idx])
        # Apex must sit above the player's head with the usual slack.
        p = near_player
        slack = 0.25 * p.h
        if ay >= p.box_top + slack:
            continue
        if not (p.cx - p.w / 2 - 30 <= ax <= p.cx + p.w / 2 + 30):
            continue
        out.append({
            "track_id": track_id,
            "toss_start_f": int(feat.frame[s]),
            "apex_f": int(feat.frame[apex_idx]),
            "apex_cx": ax,
            "apex_cy": ay,
            "area_cv": cv,
        })
    return out


def _player_at(frame_idx: int, by_frame: dict, max_gap: int = 5) -> Optional[PersonDet]:
    if frame_idx in by_frame:
        return by_frame[frame_idx]
    for d in range(1, max_gap + 1):
        if frame_idx - d in by_frame:
            return by_frame[frame_idx - d]
        if frame_idx + d in by_frame:
            return by_frame[frame_idx + d]
    return None


def _near_position_ok(toss_start_f: int, near_by_frame: dict,
                      cfg: NearConfig) -> Optional[dict]:
    """Near criterion (a): within the lookback window before the toss starts,
    the near player's foot must be within the serve zone (world-y in
    [-behind, +into] ft relative to the near baseline)."""
    look = int(round(cfg.near_pos_lookback_s * cfg.fps))
    best = None
    for f in range(max(0, toss_start_f - look), toss_start_f + 1):
        p = near_by_frame.get(f)
        if p is None or p.world is None:
            continue
        wy = p.world[1]
        if -cfg.near_pos_behind_ft <= wy <= cfg.near_pos_into_ft:
            cand = {"frame": f, "wy": round(wy, 2), "wx": round(p.world[0], 2)}
            if best is None or f > best["frame"]:
                best = cand
    return best


# --------------------------------------------------------------------------- #
# Near-side serve gate
# --------------------------------------------------------------------------- #
def detect_serves_near(
    tracks: list[Track],
    rackets: dict[int, list[RacketDet]],
    persons: dict[int, list[PersonDet]],
    near_by_frame: dict[int, PersonDet],
    audio_impacts: list[float],
    cfg: NearConfig,
) -> list[NearServeEvent]:
    """
    Gate (in order):
      (a) foot-position near baseline  [lookback window before toss]
      (b1) toss tracklet above player head
      (b2) racquet-up near apex         [required if near_require_racket]
      (c)  audio impact near apex       [optional — boosts score only]
    """
    impacts = sorted(audio_impacts)
    events: list[NearServeEvent] = []
    stat = {"toss": 0, "pos_ok": 0, "rk_ok": 0, "audio_ok": 0}

    for tr in tracks:
        feat = compute_features(tr, cfg)
        if feat is None:
            continue
        # Pick the nearest-in-time near player for each toss detection.
        for tz_apex_f in _candidate_apex_frames(feat, cfg):
            p = _player_at(tz_apex_f, near_by_frame)
            if p is None:
                continue
            for tz in _near_toss_segments(feat, tr.id, p, cfg):
                stat["toss"] += 1
                apex_f = tz["apex_f"]
                t_apex = apex_f / cfg.fps

                # (a) Foot position
                pos = _near_position_ok(tz["toss_start_f"], near_by_frame, cfg)
                if pos is None:
                    continue
                stat["pos_ok"] += 1

                # (b2) Racquet-up
                rk = find_racket_up_evidence(apex_f, tz["apex_cx"],
                                             rackets, persons, cfg)
                if rk is not None:
                    stat["rk_ok"] += 1
                if rk is None and cfg.near_require_racket:
                    continue

                # (c) Audio (optional)
                audio_cands = [a for a in impacts
                               if abs(a - t_apex) <= cfg.audio_window_s]
                audio = (min(audio_cands, key=lambda a: abs(a - t_apex))
                         if audio_cands else None)
                if audio is not None:
                    stat["audio_ok"] += 1
                if audio is None and cfg.require_audio:
                    continue

                # Contact timestamp: prefer audio pock → racquet-up frame → apex
                if audio is not None:
                    contact_f = int(round(audio * cfg.fps))
                elif rk is not None:
                    contact_f = int(rk["frame"])
                else:
                    contact_f = apex_f

                toss_q = max(0.0, 1.0 - tz["area_cv"])
                rk_q = min(1.0, rk["aspect"] / 2.0) if rk is not None else 0.4
                audio_bonus = 1.0 if audio is not None else 0.85
                score = float(min(1.0, toss_q * rk_q * audio_bonus))

                events.append(NearServeEvent(
                    toss_start_frame=tz["toss_start_f"],
                    apex_frame=apex_f,
                    contact_frame=contact_f,
                    fps=cfg.fps,
                    track_id=tz["track_id"],
                    score=score,
                    notes={
                        "area_cv": round(tz["area_cv"], 3),
                        "position": pos,
                        "racket": rk,
                        "audio": audio is not None,
                    },
                ))

    print(f"[NEAR] gate: {stat['toss']} toss segs → "
          f"{stat['pos_ok']} pos-ok → {stat['rk_ok']} racket-ok → "
          f"{stat['audio_ok']} audio-ok → {len(events)} pre-dedup events")

    # Cooldown dedup: keep the highest-score event within each cooldown window.
    events.sort(key=lambda ev: ev.contact_frame)
    deduped: list[NearServeEvent] = []
    cd = cfg.serve_cooldown_s * cfg.fps
    for ev in events:
        if deduped and (ev.contact_frame - deduped[-1].contact_frame) < cd:
            if ev.score > deduped[-1].score:
                deduped[-1] = ev
            continue
        deduped.append(ev)
    return deduped


def _candidate_apex_frames(feat, cfg: NearConfig) -> list[int]:
    """Quick pre-filter: frame indices where the ball is rising and near-vertical.
    Used to look up the near player before running the full toss-segment search."""
    from farside_serve_detector_v2 import _contiguous_runs
    mask = (feat.vy < -cfg.rise_vy_px_s) & (feat.verticality > cfg.verticality_min)
    frames = []
    for (s, e) in _contiguous_runs(mask, cfg.min_rise_frames):
        apex_idx = s + int(np.argmin(feat.cy[s:e + 1]))
        frames.append(int(feat.frame[apex_idx]))
    return frames


# --------------------------------------------------------------------------- #
# Video processing — single cheap pass (near player only)
# --------------------------------------------------------------------------- #
_DEFAULT_BALL_MODEL = (
    "/Users/tennis/Documents/Code/Laptop/weights/ball/weights/best.pt"
)


def _resolve_class_ids(model, cfg: NearConfig) -> tuple[set[int], Optional[int]]:
    names = model.names if hasattr(model, "names") else {}
    items = names.items() if isinstance(names, dict) else enumerate(names)
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
    """Batched inference over several crops of `frame` in one model call.

    Returns a list aligned with `boxes`; each element is a list of
    (cx, cy, w, h, conf) in FULL-FRAME coordinates."""
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


def _build_streams_near(
    video_path: str,
    ball_model,
    pose_model,
    cfg: NearConfig,
    geom: CourtGeometry,
    excl_zones: list = None,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
):
    """Single cheap pass over the video — near side only.

    Per frame:
      1. Strided full-frame person detection (held between strides).
         Only the near player (largest person in the near half) is kept.
      2. Racquet crop batched on near player box.
      3. Ball-toss crop batched on near player box.

    No full-frame ball pass; no Pass 2 diagonals.

    Returns: ball_stream, rackets, persons_all, near_by_frame.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("opencv-python is required for run_on_video_nearside.")

    racket_ids, person_id = _resolve_class_ids(pose_model, cfg)
    if not racket_ids:
        print("[NEAR] WARNING: no 'tennis racket' class in pose model.")
    if person_id is None:
        raise RuntimeError("Pose model has no 'person' class — required.")

    cap = _cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)

    W, H = cfg.frame_w, cfg.frame_h
    total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    limit = min(total - start_frame, max_frames) if max_frames else (total - start_frame)
    rk_half = cfg.near_racket_window_px // 2
    _zones = excl_zones or []

    def _not_excluded(cx, cy):
        return not any(x1 <= cx <= x2 and y1 <= cy <= y2 for (x1, y1, x2, y2) in _zones)

    print(f"[NEAR] Pass 1: scanning {limit} frames at {W}x{H}  "
          f"(person stride={cfg.player_stride}) …")

    ball_stream: list[list[Detection]] = []
    rackets: dict[int, list[RacketDet]] = {}
    persons_all: dict[int, list[PersonDet]] = {}
    near_by_frame: dict[int, PersonDet] = {}
    held_near: Optional[PersonDet] = None

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, orig = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            frame = _cv2.resize(orig, (W, H), interpolation=_cv2.INTER_LINEAR)

            # 1. Strided person detection — keep only the near player ----------
            if frame_idx % cfg.player_stride == 0:
                r_pose = pose_model(frame, conf=cfg.player_conf,
                                    imgsz=cfg.player_imgsz, verbose=False,
                                    classes=[person_id])
                near_cands = []
                if r_pose and r_pose[0].boxes:
                    for b in r_pose[0].boxes:
                        if int(b.cls[0]) != person_id:
                            continue
                        bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                        cx_, cy_ = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                        w_, h_ = bx2 - bx1, by2 - by1
                        is_near, world = geom.is_near_player(cx_, cy_ + h_ / 2.0)
                        if is_near:
                            near_cands.append(PersonDet(frame_idx, cx_, cy_, w_, h_,
                                                        float(b.conf[0]), world))
                # Keep the largest (highest-confidence) near detection.
                held_near = (max(near_cands, key=lambda p: p.w * p.h)
                             if near_cands else None)

            near_p = held_near
            persons_all[frame_idx] = [near_p] if near_p is not None else []
            if near_p is not None:
                near_by_frame[frame_idx] = near_p

            # 2. Racquet crop (near player only) --------------------------------
            if racket_ids and near_p is not None:
                rk_box = [(near_p.cx - rk_half, near_p.cy - rk_half,
                           near_p.cx + rk_half, near_p.cy + rk_half)]
                existing: list[RacketDet] = []
                for blist in _batch_crops(pose_model, frame, rk_box,
                                          cfg.racket_crop_imgsz,
                                          cfg.near_racket_conf, classes=racket_ids):
                    for (cx, cy, w_, h_, cf) in blist:
                        if any(np.hypot(e.cx - cx, e.cy - cy) <=
                               cfg.near_racket_dedup_px for e in existing):
                            continue
                        existing.append(RacketDet(frame_idx, cx, cy, w_, h_, cf))
                if existing:
                    rackets[frame_idx] = existing

            # 3. Ball toss crop (near player only) ------------------------------
            dets: list[Detection] = []
            if near_p is not None:
                bot = near_p.box_top + cfg.near_toss_bottom_off
                toss_box = [(near_p.cx - cfg.near_toss_w / 2, bot - cfg.near_toss_h,
                             near_p.cx + cfg.near_toss_w / 2, bot)]
                for blist in _batch_crops(ball_model, frame, toss_box,
                                          cfg.near_toss_imgsz, cfg.near_toss_conf):
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
                print(f"[NEAR]   frame {frame_idx:>6} ({pct:.0f}%)  "
                      f"toss-balls(last300)={sum(len(d) for d in ball_stream[-300:])}  "
                      f"rackets={sum(len(v) for v in rackets.values())}  "
                      f"near-frames={len(near_by_frame)}")
    finally:
        cap.release()

    print(f"[NEAR] Pass 1 done: {len(ball_stream)} frames, "
          f"{sum(len(d) for d in ball_stream)} toss-ball dets, "
          f"{sum(len(v) for v in rackets.values())} racket dets, "
          f"near-frames={len(near_by_frame)}")
    return ball_stream, rackets, persons_all, near_by_frame


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_on_video_nearside(
    video_path: str,
    cfg: Optional[NearConfig] = None,
    ball_model_path: str = _DEFAULT_BALL_MODEL,
    audio_path: Optional[str] = None,
    recalibrate: bool = False,
    calib_cache: str = _CALIB_CACHE,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    viz_cfg: Optional["VizConfig"] = None,
) -> list[NearServeEvent]:
    """Run the near-side toss + racquet-up (+ optional audio) serve tracker."""
    if not _CV2_AVAILABLE:
        raise RuntimeError(
            "opencv-python and ultralytics are required.\n"
            "  pip install opencv-python ultralytics")

    if cfg is None:
        cfg = NearConfig()
    cfg.frame_w, cfg.frame_h = cfg.analysis_w, cfg.analysis_h

    if viz_cfg is None:
        default_viz_path = os.path.splitext(video_path)[0] + "_nearside_viz.mp4"
        viz_cfg = VizConfig(out_path=default_viz_path, show_live=True)
        print(f"[NEAR] Live viz enabled → {default_viz_path}")

    # 1. Court calibration → homography
    geom = init_court_geometry(video_path, cfg, recalibrate=recalibrate)

    # 2. Models
    print(f"[NEAR] Loading ball model:  {ball_model_path}")
    ball_model = _YOLO(ball_model_path)
    print(f"[NEAR] Loading pose model:  {cfg.pose_model_path}")
    pose_model = _YOLO(cfg.pose_model_path)

    # 3. FPS
    cap = _cv2.VideoCapture(video_path)
    raw_fps = cap.get(_cv2.CAP_PROP_FPS)
    cap.release()
    if 0 < raw_fps < 300:
        cfg.fps = raw_fps
    print(f"[NEAR] Video FPS: {cfg.fps:.3f}")

    # 4. Static exclusion zones
    print("[NEAR] Scanning for static exclusion zones …")
    try:
        from utilities import create_auto_exclusion_zones
        excl_zones = create_auto_exclusion_zones(
            video_path, ball_model,
            num_frames=20, conf=0.05,
            analysis_size=(cfg.frame_w, cfg.frame_h),
        )
    except Exception as _e:
        print(f"[NEAR] Exclusion zone scan skipped: {_e}")
        excl_zones = []
    print(f"[NEAR] Static exclusion zones: {len(excl_zones)}")
    for z in excl_zones:
        print(f"    zone  x=[{z[0]},{z[2]}]  y=[{z[1]},{z[3]}]")

    # 5. Single detection pass (near player only)
    ball_stream, rackets, persons_all, near_by_frame = _build_streams_near(
        video_path, ball_model, pose_model, cfg, geom, excl_zones,
        start_frame, max_frames)

    # 6. Audio onsets
    impacts: list[float] = []
    if audio_path is not None:
        print(f"[NEAR] Loading audio: {audio_path}")
        impacts = detect_audio_impacts(audio_path, cfg)
    else:
        audio_data = _try_extract_audio(video_path)
        if audio_data is not None:
            impacts = onset_times_from_samples(audio_data[0], audio_data[1], cfg)
        else:
            print("[NEAR] No audio — proceeding (audio is optional).")
    if start_frame > 0 and impacts:
        t_off = start_frame / cfg.fps
        impacts = [t - t_off for t in impacts if t >= t_off]
    print(f"[NEAR] Audio impacts: {len(impacts)}")

    # 7. Static-ball exclusion + Kalman tracking
    zones = find_static_exclusion_zones(ball_stream, cfg)
    print(f"[NEAR] Static exclusion zones (Kalman): {len(zones)}")
    clean = apply_exclusion_zones(ball_stream, zones, cfg)
    tracks = KalmanBallTracker(cfg).run(clean)
    print(f"[NEAR] Ball tracks: {len(tracks)}")

    # 8. Near-side serve gate
    serves = detect_serves_near(
        tracks, rackets, persons_all, near_by_frame, impacts, cfg)

    print(f"\n{'='*60}")
    print(f"  NEAR-SIDE SERVES DETECTED: {len(serves)}")
    print(f"{'='*60}")
    for i, ev in enumerate(serves, 1):
        abs_contact = (start_frame + ev.contact_frame) / cfg.fps
        mm = int(abs_contact // 60); ss = abs_contact % 60
        print(f"  #{i:>3}  contact @ {mm:02d}:{ss:06.3f} (abs)  "
              f"track={ev.track_id}  apex_f={ev.apex_frame}  "
              f"score={ev.score:.2f}  {ev.notes}")
    print(f"{'='*60}\n")

    # 9. Auto-save events → <video_stem>_serves.json  (consumed by point tracker)
    events_path = os.path.splitext(video_path)[0] + "_serves.json"
    save_events(serves, events_path)
    print(f"[NEAR] Pass events to point tracker with:  --events {events_path}")

    # 10. Visualization
    toss_segs = _all_toss_segs_near(tracks, near_by_frame, cfg)
    render_visualization(
        video_path, cfg, geom,
        ball_stream, rackets, persons_all, near_by_frame,
        tracks, toss_segs, serves, impacts,
        excl_zones=excl_zones,
        start_frame=start_frame, viz=viz_cfg)

    return serves


def _all_toss_segs_near(tracks: list[Track],
                        near_by_frame: dict[int, PersonDet],
                        cfg: NearConfig) -> list[dict]:
    """All toss segments across all tracks anchored to the near player.
    Used by the visualizer to draw candidate tosses regardless of gate outcome."""
    segs: list[dict] = []
    for tr in tracks:
        feat = compute_features(tr, cfg)
        if feat is None:
            continue
        for apex_f in _candidate_apex_frames(feat, cfg):
            p = _player_at(apex_f, near_by_frame)
            if p is None:
                continue
            segs.extend(_near_toss_segments(feat, tr.id, p, cfg))
    return segs


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
@dataclass
class VizConfig:
    out_path: str = "nearside_serve_viz.mp4"
    show_live: bool = False
    trail_frames: int = 25
    serve_banner_frames: int = 90
    fourcc: str = "mp4v"
    draw_court: bool = True
    draw_person: bool = True
    draw_rackets: bool = True
    draw_crop_window: bool = True
    draw_ball_trail: bool = True
    draw_raw_dets: bool = True
    draw_audio: bool = True
    draw_world: bool = True
    draw_pos_band: bool = True


_VC = dict(
    person      = ( 60, 200, 255),   # amber — near player
    thresh_line = (180,  50, 180),   # magenta 2/3-height gate line
    racket_up   = ( 40, 220,  40),   # green — racquet passes upward test
    racket_no   = ( 40, 120, 220),   # orange — racquet fails
    crop_box    = (220, 220,  40),   # cyan dashed crop window
    ball_raw    = (100, 220, 220),   # raw ball detection circle
    trail_toss  = ( 40, 255,  40),   # bright-green trail during toss
    trail_norm  = ( 80, 200, 255),   # warm-yellow trail otherwise
    audio_bar   = ( 40, 220, 255),   # audio-impact flash bar
    hud_fg      = (230, 230, 230),
    serve_bg    = ( 30, 160,  30),
    serve_fg    = (255, 255, 255),
    gate_ok     = ( 40, 220,  40),
    gate_no     = (100, 100, 100),
    court_line  = ( 80, 200,  80),
    pos_band    = ( 80, 255, 160),
)


def _dashed_rect(img, pt1, pt2, color, thickness=1, dash=7):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    edges = [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
             ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]
    for (ax, ay), (bx, by) in edges:
        dx, dy = bx - ax, by - ay
        length = max(1, int(np.hypot(dx, dy)))
        nx, ny = dx / length, dy / length
        t = 0; draw = True
        while t < length:
            t2 = min(t + dash, length)
            if draw:
                p1 = (int(ax + nx * t),  int(ay + ny * t))
                p2 = (int(ax + nx * t2), int(ay + ny * t2))
                _cv2.line(img, p1, p2, color, thickness, _cv2.LINE_AA)
            t += dash; draw = not draw


def _text_bg(img, text, org, scale=0.45, fg=(230, 230, 230),
             bg=(0, 0, 0), thick=1):
    (tw, th), bl = _cv2.getTextSize(text, _cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    _cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + bl + 1), bg, -1)
    _cv2.putText(img, text, (x, y), _cv2.FONT_HERSHEY_SIMPLEX,
                 scale, fg, thick, _cv2.LINE_AA)


def _cross(img, cx, cy, size, color, thick=2):
    _cv2.line(img, (cx - size, cy), (cx + size, cy), color, thick, _cv2.LINE_AA)
    _cv2.line(img, (cx, cy - size), (cx, cy + size), color, thick, _cv2.LINE_AA)


def render_visualization(
    video_path: str,
    cfg: NearConfig,
    geom: CourtGeometry,
    ball_stream: list[list[Detection]],
    rackets: dict[int, list[RacketDet]],
    persons_all: dict[int, list[PersonDet]],
    near_by_frame: dict[int, PersonDet],
    tracks: list[Track],
    toss_segs: list[dict],
    serves: list[NearServeEvent],
    audio_impacts: list[float],
    excl_zones: list = None,
    start_frame: int = 0,
    viz: Optional[VizConfig] = None,
) -> None:
    if not _CV2_AVAILABLE:
        raise RuntimeError("opencv-python required for visualization.")
    if viz is None:
        viz = VizConfig()

    W, H = cfg.frame_w, cfg.frame_h
    n_frames = len(ball_stream)
    court_poly = np.array([[int(x), int(y)] for x, y in geom.corners_px()],
                          dtype=np.int32)

    # Pre-compute per-frame gate info for HUD
    serve_keys = {(ev.track_id, ev.apex_frame): ev for ev in serves}
    toss_track_ids = {s["track_id"] for s in toss_segs}
    toss_at: dict[int, list[dict]] = {}
    hud_active: dict[int, dict] = {}
    for seg in toss_segs:
        for f in range(seg["toss_start_f"], seg["apex_f"] + 1):
            toss_at.setdefault(f, []).append(seg)
        key = (seg["track_id"], seg["apex_f"])
        rk = find_racket_up_evidence(seg["apex_f"], seg["apex_cx"],
                                     rackets, persons_all, cfg)
        audio_ok = any(abs(t - seg["apex_f"] / cfg.fps) <= cfg.audio_window_s
                       for t in audio_impacts)
        pos_ok = (_near_position_ok(seg["toss_start_f"], near_by_frame, cfg)
                  is not None)
        ev = serve_keys.get(key)
        info = dict(seg, rk_ok=rk is not None, audio_ok=audio_ok,
                    pos_ok=pos_ok, is_serve=ev is not None)
        for f in range(max(0, seg["toss_start_f"] - 10),
                       min(n_frames, seg["apex_f"] + 90)):
            if f not in hud_active or (abs(f - seg["apex_f"]) <
                                       abs(f - hud_active[f]["apex_f"])):
                hud_active[f] = info

    serve_at: dict[int, NearServeEvent] = {}
    for ev in serves:
        for f in range(ev.apex_frame,
                       min(n_frames, ev.contact_frame + viz.serve_banner_frames)):
            serve_at[f] = ev

    audio_frame_set = {int(round(t * cfg.fps)) for t in audio_impacts}

    cap = _cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = _cv2.VideoWriter(viz.out_path, _cv2.VideoWriter_fourcc(*viz.fourcc),
                              cfg.fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"[VIZ] Cannot open VideoWriter → {viz.out_path}")

    print(f"[VIZ] Rendering {n_frames} frames → {viz.out_path} …")
    if viz.show_live:
        _cv2.namedWindow("Near-Side Serve Tracker", _cv2.WINDOW_NORMAL)
        _cv2.resizeWindow("Near-Side Serve Tracker", min(W, 1280), min(H, 720))

    for fi in range(n_frames):
        ret, orig = cap.read()
        if not ret:
            break
        frame = _cv2.resize(orig, (W, H), interpolation=_cv2.INTER_LINEAR)

        # 1. Court polygon
        if viz.draw_court:
            _cv2.polylines(frame, [court_poly], True, _VC["court_line"], 2,
                           _cv2.LINE_AA)

        # 1b. Static exclusion zones
        for (ex1, ey1, ex2, ey2) in (excl_zones or []):
            _cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 220), 2)
            _text_bg(frame, "EXCL", (ex1 + 2, ey1 + 14),
                     scale=0.35, fg=(0, 0, 220), bg=(0, 0, 0))

        # 1c. Near-baseline serve-position band
        if viz.draw_pos_band:
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

        # 2. Ball trails
        if viz.draw_ball_trail:
            for tr in tracks:
                color = _VC["trail_toss"] if tr.id in toss_track_ids else _VC["trail_norm"]
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

        # 4. Near player box + 2/3 gate line + world label
        if viz.draw_person:
            near_p = near_by_frame.get(fi)
            if near_p is not None:
                bx1 = int(near_p.cx - near_p.w / 2)
                by1 = int(near_p.cy - near_p.h / 2)
                bx2 = int(near_p.cx + near_p.w / 2)
                by2 = int(near_p.cy + near_p.h / 2)
                _cv2.rectangle(frame, (bx1, by1), (bx2, by2), _VC["person"], 2)
                if viz.draw_world and near_p.world is not None:
                    lbl = f"near ({near_p.world[0]:.0f},{near_p.world[1]:.0f}ft)"
                    _text_bg(frame, lbl, (bx1, max(10, by1 - 4)),
                             scale=0.40, fg=_VC["person"], bg=(0, 0, 0))
                thresh_cy = int(near_p.cy + near_p.h / 2.0 - cfg.overhead_frac * near_p.h)
                _cv2.line(frame, (bx1, thresh_cy), (bx2, thresh_cy),
                          _VC["thresh_line"], 1, _cv2.LINE_AA)
                _text_bg(frame, "2/3", (bx1 + 3, thresh_cy - 3),
                         scale=0.33, fg=_VC["thresh_line"], bg=(0, 0, 0))

                if viz.draw_crop_window:
                    bot = int(near_p.box_top + cfg.near_toss_bottom_off)
                    _dashed_rect(frame,
                                 (int(near_p.cx - cfg.near_toss_w / 2), bot - cfg.near_toss_h),
                                 (int(near_p.cx + cfg.near_toss_w / 2), bot),
                                 _VC["crop_box"], 1, 6)
                    rk_half = cfg.near_racket_window_px // 2
                    _dashed_rect(frame,
                                 (int(near_p.cx - rk_half), int(near_p.cy - rk_half)),
                                 (int(near_p.cx + rk_half), int(near_p.cy + rk_half)),
                                 _VC["person"], 1, 8)

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

        # 8. Serve banner
        if fi in serve_at:
            ev = serve_at[fi]
            ov = frame.copy()
            _cv2.rectangle(ov, (0, H // 2 + 50), (W, H // 2 + 90), _VC["serve_bg"], -1)
            _cv2.addWeighted(ov, 0.80, frame, 0.20, 0, frame)
            txt = (f"NEAR SERVE   score={ev.score:.2f}"
                   f"   t={(start_frame + ev.contact_frame) / cfg.fps:.2f}s")
            (tw, th), _ = _cv2.getTextSize(txt, _cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
            _cv2.putText(frame, txt, ((W - tw) // 2, H // 2 + 78),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.70, _VC["serve_fg"], 2, _cv2.LINE_AA)

        # 9. Gate HUD
        if fi in hud_active:
            info = hud_active[fi]
            rows = [
                ("POS",   info["pos_ok"],  _VC["gate_ok"] if info["pos_ok"]  else _VC["gate_no"]),
                ("TOSS",  True,            _VC["trail_toss"]),
                ("RACKET",info["rk_ok"],   _VC["gate_ok"] if info["rk_ok"]   else _VC["gate_no"]),
                ("AUDIO", info["audio_ok"],_VC["audio_bar"] if info["audio_ok"] else _VC["gate_no"]),
                ("SERVE", info["is_serve"],_VC["serve_bg"] if info["is_serve"] else _VC["gate_no"]),
            ]
            for i, (label, ok, col) in enumerate(rows):
                txt = f"{label}: {'OK' if ok else '--'}"
                _text_bg(frame, txt, (W - 132, 16 + i * 22), scale=0.46,
                         fg=col, bg=(0, 0, 0))

        # 10. Frame counter
        abs_f = start_frame + fi
        t_abs = abs_f / cfg.fps
        _text_bg(frame, f"f={abs_f}  {int(t_abs // 60):02d}:{t_abs % 60:05.2f}",
                 (8, H - 10), scale=0.42, fg=_VC["hud_fg"], bg=(0, 0, 0))

        writer.write(frame)
        if viz.show_live:
            _cv2.imshow("Near-Side Serve Tracker", frame)
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
# Synthetic offline demo
# --------------------------------------------------------------------------- #
def _synthetic_scene(cfg: NearConfig, geom: CourtGeometry, rng):
    """Synthetic near-side scene: two valid serves + one that fails the position
    gate (player is inside the court, not at the baseline)."""
    N = 280
    frames: list[list[Detection]] = [[] for _ in range(N)]
    rackets: dict[int, list[RacketDet]] = {}
    persons_all: dict[int, list[PersonDet]] = {f: [] for f in range(N)}
    contacts: list[int] = []

    def near_serve(t0, in_position, with_racket, with_audio):
        py = 555 if in_position else 460   # 555 → wy≈0 (baseline); 460 → inside court
        px, pw, ph = 640, 80, 180
        top = py - ph / 2
        for f in range(t0 - 40, t0 + 40):
            if 0 <= f < N:
                persons_all[f].append(PersonDet(f, px, py, pw, ph, 0.95))
        rise = 16; apex = t0 + rise
        for k in range(rise + 1):
            f = t0 + k
            cy = (top - 10) - 150 * np.sin((np.pi / 2) * k / rise)
            frames[f].append(Detection(f, px + rng.normal(0, 1.0), cy, 14, 14, 0.85))
            if with_racket and k >= rise - 4:
                rackets.setdefault(f, []).append(
                    RacketDet(f, px + 10, top - 40, 18, 48, 0.7))
        if with_audio:
            contacts.append(apex)

    near_serve(30,  in_position=True,  with_racket=True,  with_audio=True)   # VALID
    near_serve(130, in_position=False, with_racket=True,  with_audio=True)   # REJECT (pos)
    near_serve(210, in_position=True,  with_racket=True,  with_audio=False)  # VALID (no audio)

    # Parked ball (static zone)
    for f in range(N):
        frames[f].append(Detection(f, 280 + rng.normal(0, 0.4),
                                   390 + rng.normal(0, 0.4), 8, 8, 0.6))

    near_by_frame: dict[int, PersonDet] = {}
    for f, plist in persons_all.items():
        near_cands = []
        for p in plist:
            is_near, world = geom.is_near_player(p.cx, p.foot_y)
            p.world = world
            if is_near:
                near_cands.append(p)
        if near_cands:
            near_by_frame[f] = max(near_cands, key=lambda q: q.w * q.h)
    return frames, rackets, persons_all, near_by_frame, contacts


def _synthetic_audio(cfg: NearConfig, contact_frames, n_frames, rng):
    sr = 16000
    x = 0.01 * rng.standard_normal(int(sr * n_frames / cfg.fps))
    for cf in contact_frames:
        t = cf / cfg.fps
        i0 = int(t * sr)
        L = int(0.006 * sr)
        if i0 + L <= len(x):
            x[i0:i0 + L] += (np.exp(-np.linspace(0, 6, L))
                              * rng.standard_normal(L) * 0.9)
    return x, sr


def main() -> None:
    cfg = NearConfig()
    cfg.frame_w, cfg.frame_h = cfg.analysis_w, cfg.analysis_h
    rng = np.random.default_rng(42)

    geom = CourtGeometry([(240, 650), (1040, 650), (840, 150), (440, 150)], cfg)

    stream, rackets, persons_all, near_bf, contacts = \
        _synthetic_scene(cfg, geom, rng)
    audio, sr = _synthetic_audio(cfg, contacts, len(stream), rng)

    zones = find_static_exclusion_zones(stream, cfg)
    clean = apply_exclusion_zones(stream, zones, cfg)
    tracks = KalmanBallTracker(cfg).run(clean)
    impacts = onset_times_from_samples(audio, sr, cfg)
    print(f"Static zones: {len(zones)}  tracks: {len(tracks)}  "
          f"audio impacts: {len(impacts)}  near-frames={len(near_bf)}")

    serves = detect_serves_near(tracks, rackets, persons_all, near_bf, impacts, cfg)
    print(f"\nNear-side serves: {len(serves)}")
    for i, ev in enumerate(serves, 1):
        print(f"  #{i}  contact @ {ev.hhmmss()}  track={ev.track_id}  "
              f"apex_f={ev.apex_frame}  score={ev.score:.2f}  {ev.notes}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Near-side tennis serve tracker (toss + racquet-up + audio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
On first run you click 4 singles-court corners; the homography is cached next
to the video. The near-side gate requires: foot at baseline (a), ball toss (b1),
racquet-up (b2), optional audio impact (c).

Examples:
  # Synthetic demo (no video needed):
  python nearside_serve_tracker_toss.py

  # Real video:
  python nearside_serve_tracker_toss.py video.mp4

  # Real video + annotated output + live display:
  python nearside_serve_tracker_toss.py video.mp4 --viz out.mp4 --show

  # Force court re-calibration:
  python nearside_serve_tracker_toss.py video.mp4 --recalibrate
""")
    p.add_argument("video", nargs="?", default=None)
    p.add_argument("--ball-model", default=_DEFAULT_BALL_MODEL)
    p.add_argument("--pose-model", default=None)
    p.add_argument("--audio", default=None)
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--calib-cache", default=_CALIB_CACHE)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--require-audio", action="store_true")
    p.add_argument("--viz", default=None, metavar="OUT.mp4")
    p.add_argument("--show", action="store_true")
    p.add_argument("--trail", type=int, default=25)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--no-live", action="store_true")
    args = p.parse_args()

    if args.video is None:
        main()
    else:
        _cfg = NearConfig()
        if args.conf is not None:
            _cfg.ball_conf = args.conf
        if args.pose_model is not None:
            _cfg.pose_model_path = args.pose_model
        if args.require_audio:
            _cfg.require_audio = True
        if args.stride is not None:
            _cfg.player_stride = max(1, args.stride)
        _viz = None
        if args.viz or args.show:
            _viz = VizConfig(
                out_path=args.viz or "nearside_serve_viz.mp4",
                show_live=args.show,
                trail_frames=args.trail,
            )
        elif args.no_live:
            default_viz_path = os.path.splitext(args.video)[0] + "_nearside_viz.mp4"
            _viz = VizConfig(out_path=default_viz_path, show_live=False,
                             trail_frames=args.trail)
        run_on_video_nearside(
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
