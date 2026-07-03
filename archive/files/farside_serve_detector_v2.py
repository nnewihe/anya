"""
Far-side tennis serve detector — v2.

Adds, on top of the v1 scaffold:
  1. A Kalman ball tracker (multi-track, Mahalanobis gating, coast-through-misses).
  2. A static-ball scan that marks resting balls as EXCLUSION ZONES and filters
     their detections out before tracking (parked balls, ball in pocket, etc.).
  3. Audio racket-impact ("pock") detection via spectral-flux onsets.
  4. A strict serve gate: a valid serve must show, IN ORDER and within 1.5 s,
        (1) a ball toss   ->   (2) an audio impact   ->   (3) ball moving
        downward / toward the near side (bbox grows, cy increases).

Camera geometry (unchanged): behind baseline, ~10 ft high. Image +y is DOWN.
Far player is near the TOP of the frame and small; a toss makes cy DECREASE,
a strike-and-travel-toward-camera makes area GROW and cy INCREASE.

Dependencies: numpy (required). Audio file loading tries soundfile, then
scipy.io.wavfile, but the onset detector itself is pure numpy, and the demo
feeds it a synthetic waveform so the whole thing runs offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    fps: float = 30.0
    frame_w: int = 1280
    frame_h: int = 720
    far_roi_y_max_frac: float = 0.55
    far_roi_x_min_frac: float = 0.0     # far-COURT x-band: rejects neighboring
    far_roi_x_max_frac: float = 1.0     #   courts whose serves are otherwise valid
    far_roi_x_min_frac: float = 0.0     # x-band of OUR court's far baseline;
    far_roi_x_max_frac: float = 1.0     #   narrowing rejects neighboring courts
    far_roi_y_min_frac: float = 0.10    # exclude the very-top background band
    court_x_min_frac: float = 0.28      # court left edge in the image (CALIBRATION)
    court_x_max_frac: float = 0.73      # court right edge (rejects neighbor courts)

    # --- Static exclusion scan ---
    static_grid_px: int = 24            # spatial quantization for the scan
    static_min_frames: int = 30         # cell must be occupied this many frames
    static_min_span_frames: int = 60    # ...spread over at least this many frames
    static_density_min: float = 0.5     # occupied / span: catches RESTING balls,
                                        #   not a toss apex that recurs sparsely
    static_pos_std_px: float = 6.0      # positional std inside the cell must be low
    static_zone_radius: float = 28.0
    static_area_tol: float = 0.6        # only exclude dets whose area ~ resting ball

    # --- Kalman model / tracking ---
    kf_meas_pos_std: float = 3.0
    kf_meas_area_std: float = 10.0
    kf_proc_pos_std: float = 2.0
    kf_proc_vel_std: float = 45.0
    kf_proc_area_std: float = 18.0
    gate_maha: float = 9.21             # chi-square, 2 dof, 99%
    min_hits: int = 3                   # tentative -> confirmed
    max_misses: int = 8                 # frames of coasting before a track dies
    spawn_suppress_px: float = 45.0     # don't birth a new track this close to one
    recovery_px: float = 120.0          # coasting track may re-grab a detection
                                        #   this close to its LAST MEASURED point
                                        #   (bridges the velocity reversal at apex)

    # --- Kinematic signals ---
    smooth_win: int = 5

    # --- Toss (hypothesis A + B) ---
    min_rise_frames: int = 4
    rise_vy_px_s: float = 60.0
    verticality_min: float = 1.5
    area_cv_max: float = 0.45

    # --- Descent into court (hypothesis C) ---
    strike_window_s: float = 0.9
    area_growth_min: float = 0.4
    min_desc_frames: int = 4            # length of a downward+growing run
    link_px: float = 90.0               # apex -> descent-start spatial link radius

    # --- Audio onset detection ---
    audio_win: int = 1024
    audio_hop: int = 512
    audio_peak_k: float = 2.5           # threshold = mean + k * std of flux
    audio_min_sep_s: float = 0.18
    audio_hf_frac: float = 0.5          # use upper fraction of spectrum (sharp pock)

    # --- Ordered serve gate ---
    serve_window_s: float = 1.5         # toss-start -> descent must fit in here
    audio_pre_apex_tol_s: float = 0.15  # impact may land slightly before apex
    audio_post_desc_tol_s: float = 0.25 # ...and slightly after the descent onset
                                        #   (contact and first area-growth frame
                                        #    coincide to within a few frames)
    require_audio: bool = True
    serve_cooldown_s: float = 2.5


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    frame: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float = 1.0

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class TrackPoint:
    frame: int
    cx: float
    cy: float
    area: float
    interpolated: bool = False


@dataclass
class Zone:
    cx: float
    cy: float
    radius: float
    area: float


@dataclass
class ServeEvent:
    toss_start_frame: int
    apex_frame: int
    impact_frame: int          # from AUDIO — the actual contact
    descent_frame: int
    fps: float
    track_id: int
    score: float = 1.0
    notes: dict = field(default_factory=dict)

    def t(self) -> float:                       # serve timestamp = audio contact
        return self.impact_frame / self.fps

    def hhmmss(self) -> str:
        s = self.t()
        return f"{int(s//3600):02d}:{int(s%3600//60):02d}:{s%60:06.3f}"


# --------------------------------------------------------------------------- #
# Stage 1a: static-ball scan -> exclusion zones
# --------------------------------------------------------------------------- #
def find_static_exclusion_zones(stream: list[list[Detection]], cfg: Config) -> list[Zone]:
    """
    A resting ball occupies one small image region CONTINUOUSLY for a long time.
    We bin detections into a grid and flag cells that are (a) occupied across a
    long span, (b) densely occupied within that span (so a toss apex that recurs
    in the same spot a few frames per serve is NOT flagged), and (c) positionally
    tight. The density test is the key guard against killing the serve location.
    """
    g = cfg.static_grid_px
    cells: dict[tuple[int, int], dict] = {}
    for f, dets in enumerate(stream):
        for d in dets:
            key = (int(d.cx // g), int(d.cy // g))
            c = cells.setdefault(key, {"frames": [], "xs": [], "ys": [], "areas": []})
            c["frames"].append(f)
            c["xs"].append(d.cx)
            c["ys"].append(d.cy)
            c["areas"].append(d.area)

    zones: list[Zone] = []
    for c in cells.values():
        frames = c["frames"]
        occ = len(frames)
        span = frames[-1] - frames[0] + 1
        if occ < cfg.static_min_frames or span < cfg.static_min_span_frames:
            continue
        if occ / span < cfg.static_density_min:
            continue
        if np.std(c["xs"]) > cfg.static_pos_std_px or np.std(c["ys"]) > cfg.static_pos_std_px:
            continue
        zones.append(Zone(float(np.mean(c["xs"])), float(np.mean(c["ys"])),
                          cfg.static_zone_radius, float(np.mean(c["areas"]))))

    # Merge zones whose centers nearly coincide (same ball spanning 2 cells).
    merged: list[Zone] = []
    for z in zones:
        hit = next((m for m in merged
                    if np.hypot(m.cx - z.cx, m.cy - z.cy) < cfg.static_zone_radius), None)
        if hit is None:
            merged.append(z)
    return merged


def apply_exclusion_zones(stream: list[list[Detection]], zones: list[Zone],
                          cfg: Config) -> list[list[Detection]]:
    """Drop detections that sit inside a zone AND match the resting ball's size."""
    if not zones:
        return stream
    out = []
    for dets in stream:
        keep = []
        for d in dets:
            excluded = False
            for z in zones:
                near = np.hypot(d.cx - z.cx, d.cy - z.cy) <= z.radius
                sized = abs(d.area - z.area) <= cfg.static_area_tol * z.area
                if near and sized:
                    excluded = True
                    break
            if not excluded:
                keep.append(d)
        out.append(keep)
    return out


# --------------------------------------------------------------------------- #
# Stage 1b: Kalman filter + multi-track tracker
# --------------------------------------------------------------------------- #
class KalmanFilter:
    """
    Constant-velocity model. State = [x, y, area, vx, vy, vArea]^T,
    measurement = [x, y, area]^T. Process noise absorbs gravity/perspective
    accelerations that the CV model doesn't represent explicitly.
    """

    def __init__(self, cfg: Config, x0: np.ndarray):
        dt = 1.0 / cfg.fps
        self.F = np.eye(6)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0
        self.Q = np.diag([cfg.kf_proc_pos_std**2, cfg.kf_proc_pos_std**2,
                          cfg.kf_proc_area_std**2, cfg.kf_proc_vel_std**2,
                          cfg.kf_proc_vel_std**2, cfg.kf_proc_area_std**2])
        self.R = np.diag([cfg.kf_meas_pos_std**2, cfg.kf_meas_pos_std**2,
                          cfg.kf_meas_area_std**2])
        self.x = x0.astype(float)
        self.P = np.diag([cfg.kf_meas_pos_std**2, cfg.kf_meas_pos_std**2,
                          cfg.kf_meas_area_std**2, 1e3, 1e3, 1e3])

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def innovation(self, z: np.ndarray):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def update(self, z: np.ndarray) -> None:
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


class Track:
    _next_id = 0

    def __init__(self, frame: int, det: Detection, cfg: Config):
        self.cfg = cfg
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = KalmanFilter(cfg, np.array([det.cx, det.cy, det.area, 0, 0, 0]))
        self.hits = 1
        self.misses = 0
        self.status = "tentative"
        self.last_meas = (det.cx, det.cy)
        self.history = [TrackPoint(frame, det.cx, det.cy, det.area, False)]

    def predict(self) -> None:
        self.kf.predict()

    def gating_d2(self, det: Detection) -> float:
        """Mahalanobis distance on the (x, y) position only, for association."""
        z = np.array([det.cx, det.cy, det.area])
        y, S = self.kf.innovation(z)
        yp, Sp = y[:2], S[:2, :2]
        return float(yp @ np.linalg.inv(Sp) @ yp)

    def assoc_cost(self, det: Detection, cfg: Config) -> float:
        """Normal Mahalanobis gate; if coasting, allow a position-based recovery
        grab near the last measured point so a track can survive the apex."""
        d2 = self.gating_d2(det)
        if d2 <= cfg.gate_maha:
            return d2
        if self.misses > 0:
            e = float(np.hypot(self.last_meas[0] - det.cx, self.last_meas[1] - det.cy))
            if e <= cfg.recovery_px:
                return cfg.gate_maha + e          # admissible, lower priority
        return float("inf")

    def update(self, frame: int, det: Detection) -> None:
        self.kf.update(np.array([det.cx, det.cy, det.area]))
        self.hits += 1
        self.misses = 0
        self.last_meas = (det.cx, det.cy)
        if self.status == "tentative" and self.hits >= self.cfg.min_hits:
            self.status = "confirmed"
        x = self.kf.x
        self.history.append(TrackPoint(frame, x[0], x[1], x[2], False))

    def coast(self, frame: int) -> None:
        self.misses += 1
        x = self.kf.x
        self.history.append(TrackPoint(frame, x[0], x[1], x[2], True))

    @property
    def dead(self) -> bool:
        return self.misses > self.cfg.max_misses


class KalmanBallTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.active: list[Track] = []
        self.finished: list[Track] = []

    def step(self, frame: int, dets: list[Detection]) -> None:
        for t in self.active:
            t.predict()

        # Greedy association by cost (Mahalanobis, or recovery grab if coasting).
        pairs = []
        for ti, t in enumerate(self.active):
            for di, d in enumerate(dets):
                c = t.assoc_cost(d, self.cfg)
                if np.isfinite(c):
                    pairs.append((c, ti, di))
        pairs.sort(key=lambda p: p[0])

        used_t: set[int] = set()
        used_d: set[int] = set()
        for c, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            self.active[ti].update(frame, dets[di])
            used_t.add(ti)
            used_d.add(di)

        for ti, t in enumerate(self.active):
            if ti not in used_t:
                t.coast(frame)

        for di, d in enumerate(dets):
            if di in used_d:
                continue
            # Suppress spawns near ANY active track; coasting tracks re-acquire
            # via the recovery gate rather than being shadowed by a duplicate.
            near = any(np.hypot(t.kf.x[0] - d.cx, t.kf.x[1] - d.cy)
                       < self.cfg.spawn_suppress_px for t in self.active)
            if not near:
                self.active.append(Track(frame, d, self.cfg))

        still: list[Track] = []
        for t in self.active:
            (self.finished if t.dead else still).append(t)
        self.active = still

    def run(self, stream: Iterable[list[Detection]]) -> list[Track]:
        for frame, dets in enumerate(stream):
            self.step(frame, dets)
        self.finished.extend(self.active)
        self.active = []
        # Trim trailing coasted points and keep only real tracks.
        out = []
        for t in self.finished:
            while t.history and t.history[-1].interpolated:
                t.history.pop()
            if t.hits >= self.cfg.min_hits and len(t.history) >= self.cfg.smooth_win:
                out.append(t)
        return out


# --------------------------------------------------------------------------- #
# Stage 2: kinematic features (per track)
# --------------------------------------------------------------------------- #
@dataclass
class Features:
    frame: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    area: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    verticality: np.ndarray


def _smooth(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or a.size < win:
        return a.astype(float)
    win += (win + 1) % 2
    pad = win // 2
    ap = np.pad(a.astype(float), pad, mode="edge")
    return np.convolve(ap, np.ones(win) / win, mode="valid")


def compute_features(track: Track, cfg: Config) -> Optional[Features]:
    h = track.history
    if len(h) < cfg.smooth_win:
        return None
    frame = np.array([p.frame for p in h])
    cx = _smooth(np.array([p.cx for p in h]), cfg.smooth_win)
    cy = _smooth(np.array([p.cy for p in h]), cfg.smooth_win)
    area = _smooth(np.array([p.area for p in h]), cfg.smooth_win)
    dt = 1.0 / cfg.fps
    vx = np.gradient(cx) / dt
    vy = np.gradient(cy) / dt
    verticality = np.abs(vy) / (np.abs(vx) + 1e-6)
    return Features(frame, cx, cy, area, vx, vy, verticality)


# --------------------------------------------------------------------------- #
# Stage 3a: audio impact detection (spectral-flux onsets)
# --------------------------------------------------------------------------- #
def audio_flux(samples: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    """High-frequency-weighted positive spectral flux (expensive part; cache it)."""
    x = np.asarray(samples, dtype=float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n_fft, hop = cfg.audio_win, cfg.audio_hop
    if x.size < n_fft:
        return np.zeros(0)
    window = np.hanning(n_fft)
    n_frames = 1 + (x.size - n_fft) // hop
    lo = int((1 - cfg.audio_hf_frac) * (n_fft // 2 + 1))
    prev = None
    flux = np.zeros(n_frames)
    for i in range(n_frames):
        mag = np.abs(np.fft.rfft(x[i * hop: i * hop + n_fft] * window))[lo:]
        if prev is not None:
            flux[i] = np.sum(np.maximum(0.0, mag - prev))
        prev = mag
    return flux


def pick_onsets(flux: np.ndarray, sr: int, cfg: Config) -> list[float]:
    """Adaptive peak-pick on a precomputed flux (cheap; depends on audio_peak_k)."""
    if flux.size == 0 or flux.max() <= 0:
        return []
    thr = flux.mean() + cfg.audio_peak_k * flux.std()
    min_sep = max(1, int(cfg.audio_min_sep_s * sr / cfg.audio_hop))
    peaks, last = [], -10**9
    for i in range(1, flux.size - 1):
        if flux[i] > thr and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            if i - last >= min_sep:
                peaks.append(i)
                last = i
    return [p * cfg.audio_hop / sr for p in peaks]


def onset_times_from_samples(samples: np.ndarray, sr: int, cfg: Config) -> list[float]:
    return pick_onsets(audio_flux(samples, sr, cfg), sr, cfg)


def detect_audio_impacts(audio_path: str, cfg: Config) -> list[float]:
    """Load audio (soundfile -> scipy fallback) and return impact times (sec)."""
    try:
        import soundfile as sf
        samples, sr = sf.read(audio_path)
    except Exception:
        from scipy.io import wavfile
        sr, samples = wavfile.read(audio_path)
        samples = samples.astype(float)
        if np.issubdtype(samples.dtype, np.integer):
            samples /= np.iinfo(samples.dtype).max
    return onset_times_from_samples(samples, sr, cfg)


# --------------------------------------------------------------------------- #
# Stage 3b: serve detection with the ordered toss -> impact -> descent gate
# --------------------------------------------------------------------------- #
def _contiguous_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs, i, n = [], 0, mask.size
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _toss_segments(feat: Features, track_id: int, cfg: Config) -> list[dict]:
    """Rising + vertical + constant-area runs in the far ROI (hypothesis A + B)."""
    H, W = cfg.frame_h, cfg.frame_w
    mask = (feat.cy < cfg.far_roi_y_max_frac * H) & \
           (feat.cy > cfg.far_roi_y_min_frac * H) & \
           (feat.cx > cfg.court_x_min_frac * W) & \
           (feat.cx < cfg.court_x_max_frac * W) & \
           (feat.vy < -cfg.rise_vy_px_s) & (feat.verticality > cfg.verticality_min)
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


def _descent_segments(feat: Features, track_id: int, cfg: Config) -> list[dict]:
    """Runs moving DOWN with GROWING bbox (hypothesis C: ball into near court)."""
    da = np.gradient(feat.area)
    mask = (feat.vy > 0) & (da > 0)
    out = []
    for (s, e) in _contiguous_runs(mask, cfg.min_desc_frames):
        a0 = float(feat.area[s]) + 1e-6
        growth = (float(feat.area[e]) - a0) / a0
        if growth < cfg.area_growth_min:
            continue
        out.append({"track_id": track_id,
                    "start_f": int(feat.frame[s]),
                    "start_cx": float(feat.cx[s]),
                    "start_cy": float(feat.cy[s]),
                    "growth": growth})
    return out


def detect_serves(tracks: list[Track], audio_impacts: list[float],
                  cfg: Config) -> list[ServeEvent]:
    # Pool toss and descent segments across ALL tracklets (the apex velocity
    # reversal usually splits a serve into two tracks; we link them here).
    tosses, descents = [], []
    for tr in tracks:
        feat = compute_features(tr, cfg)
        if feat is None:
            continue
        tosses.extend(_toss_segments(feat, tr.id, cfg))
        descents.extend(_descent_segments(feat, tr.id, cfg))

    impacts = sorted(audio_impacts)
    win = cfg.strike_window_s
    events: list[ServeEvent] = []

    for tz in tosses:
        t_start = tz["toss_start_f"] / cfg.fps
        t_apex = tz["apex_f"] / cfg.fps

        # (2->3) descent that begins just after the apex, spatially linked to it.
        link = None
        for d in descents:
            dt = d["start_f"] / cfg.fps - t_apex
            if 0 <= dt <= win and np.hypot(d["start_cx"] - tz["apex_cx"],
                                           d["start_cy"] - tz["apex_cy"]) <= cfg.link_px:
                if link is None or d["start_f"] < link["start_f"]:
                    link = d
        if link is None:
            continue
        t_desc = link["start_f"] / cfg.fps
        if (t_desc - t_start) > cfg.serve_window_s:
            continue

        # Ordered audio impact between (apex - tol) and (descent + tol).
        impact, lo, hi = None, t_apex - cfg.audio_pre_apex_tol_s, t_desc + cfg.audio_post_desc_tol_s
        cands = [a for a in impacts if lo <= a <= hi]
        if cands:
            impact = min(cands, key=lambda a: abs(a - t_apex))
        elif cfg.require_audio:
            continue

        impact_f = int(round((impact if impact is not None else t_apex) * cfg.fps))
        events.append(ServeEvent(
            toss_start_frame=tz["toss_start_f"], apex_frame=tz["apex_f"],
            impact_frame=impact_f, descent_frame=link["start_f"],
            fps=cfg.fps, track_id=tz["track_id"],
            score=float(min(1.0, (1 - tz["area_cv"]) * min(1.0, link["growth"]))),
            notes={"area_cv": round(tz["area_cv"], 3),
                   "growth": round(link["growth"], 3),
                   "desc_track": link["track_id"], "audio": impact is not None}))

    events.sort(key=lambda ev: ev.impact_frame)
    deduped: list[ServeEvent] = []
    cd = cfg.serve_cooldown_s * cfg.fps
    for ev in events:
        if deduped and (ev.impact_frame - deduped[-1].impact_frame) < cd:
            if ev.score > deduped[-1].score:
                deduped[-1] = ev
            continue
        deduped.append(ev)
    return deduped


# --------------------------------------------------------------------------- #
# Synthetic demo: parked ball + 2 serves + decoy sound, all gates exercised
# --------------------------------------------------------------------------- #
def _synthetic_video(cfg: Config, rng) -> tuple[list[list[Detection]], list[int]]:
    N = 230
    frames: list[list[Detection]] = [[] for _ in range(N)]
    contact_frames: list[int] = []

    def add_serve(t0: int, base_x: float):
        # Toss RISE: cy 300 -> ~150 with parabolic deceleration, area ~constant.
        rise = 13
        for k in range(rise + 1):
            f = t0 + k
            cy = 300 - 150 * np.sin((np.pi / 2) * k / rise)
            frames[f].append(Detection(f, base_x + rng.normal(0, 1.2),
                                       cy, 8 + rng.normal(0, 0.5),
                                       8 + rng.normal(0, 0.5), 0.8))
        contact_frames.append(t0 + rise)          # racket contact at the apex
        # DESCENT toward camera: cy increases, area grows; continuous from apex.
        for k in range(1, 32):
            f = t0 + rise + k
            if f >= N:
                break
            if rng.random() > 0.18:               # ~18% missed (motion blur)
                side = 8 + 0.55 * k
                frames[f].append(Detection(f, base_x + 4 * k + rng.normal(0, 2),
                                           150 + 11 * k, side, side, 0.7))

    add_serve(20, 600)
    add_serve(135, 640)

    for f in range(N):                            # parked ball, every frame (static)
        frames[f].append(Detection(f, 300 + rng.normal(0, 0.4),
                                   200 + rng.normal(0, 0.4), 8, 8, 0.6))
    for f in range(N):                            # scattered false positives
        if rng.random() < 0.10:
            frames[f].append(Detection(f, rng.uniform(0, cfg.frame_w),
                                       rng.uniform(0, cfg.frame_h), 7, 7, 0.4))
    return frames, contact_frames


def _synthetic_audio(cfg: Config, contact_frames: list[int],
                     n_frames: int, rng) -> tuple[np.ndarray, int]:
    sr = 16000
    dur = n_frames / cfg.fps
    x = 0.01 * rng.standard_normal(int(sr * dur))   # quiet background

    def pock(t_sec: float):
        i0 = int(t_sec * sr)
        L = int(0.006 * sr)                          # ~6 ms broadband transient
        env = np.exp(-np.linspace(0, 6, L))
        x[i0:i0 + L] += env * rng.standard_normal(L) * 0.9

    for cf in contact_frames:
        pock(cf / cfg.fps)
    pock(3.0)                                        # DECOY sound: no toss near it
    return x, sr


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(7)

    stream, contact_frames = _synthetic_video(cfg, rng)
    audio, sr = _synthetic_audio(cfg, contact_frames, len(stream), rng)

    zones = find_static_exclusion_zones(stream, cfg)
    print(f"Static exclusion zones found: {len(zones)}")
    for z in zones:
        print(f"  zone @ ({z.cx:.0f}, {z.cy:.0f}) r={z.radius:.0f} area~{z.area:.0f}")

    clean = apply_exclusion_zones(stream, zones, cfg)
    tracks = KalmanBallTracker(cfg).run(clean)
    print(f"Confirmed tracks after exclusion: {len(tracks)}")

    impacts = onset_times_from_samples(audio, sr, cfg)
    print(f"Audio impacts detected (s): {[round(t, 2) for t in impacts]}")

    serves = detect_serves(tracks, impacts, cfg)
    print(f"\nValid far-side serves (toss -> audio -> descent, <=1.5s): {len(serves)}")
    for i, ev in enumerate(serves, 1):
        print(f"  #{i}  contact @ {ev.hhmmss()}  track={ev.track_id}  "
              f"toss_f={ev.toss_start_frame} apex_f={ev.apex_frame} "
              f"descent_f={ev.descent_frame}  score={ev.score:.2f}  {ev.notes}")


if __name__ == "__main__":
    main()
