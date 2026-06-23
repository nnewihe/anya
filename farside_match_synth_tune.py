"""
Synthetic tennis-match generator + evaluation + tuning for the far-side serve
detector (uses farside_serve_detector_v2 as the detector under test).

What the generator produces, with ground-truth labels:
  - FAR serves (the TARGETS): toss rises (small bbox, cy down), contact at apex,
    then ball travels toward the camera (bbox GROWS, cy down/into near court).
  - NEAR serves (must be IGNORED): big bbox at the bottom, then travels AWAY ->
    bbox SHRINKS, cy up.  Tests the ROI + area-growth discrimination.
  - RALLY strokes incl. "far drives": a far-player groundstroke hit toward camera
    (area grows, downward) with NO preceding toss -> hypothesis C alone would fire,
    so this tests that the toss+ordering gate rejects it.
  - NEIGHBOR-COURT activity: balls and "pock" sounds off to the sides, some of
    them deliberately serve-like decoys (vertical-ish in the far band) to force
    the verticality / ROI / audio-window thresholds to do real work.
  - STATIC balls resting on court (should be excluded).
  - NOISE: position & size jitter, blur-correlated missed detections (faster ball
    => more misses), scattered YOLO false positives, audio background + crowd pocks.

Then we score detected serves against the FAR-serve ground truth (precision /
recall / F1, with a per-cause breakdown of false positives) and run a randomized
parameter search to fine-tune the detector's thresholds.
"""

from __future__ import annotations

import dataclasses
import json
import random as _random

import numpy as np

import farside_serve_detector_v2 as M
from farside_serve_detector_v2 import Config, Detection


SR = 16000


# --------------------------------------------------------------------------- #
# Ground-truth event record
# --------------------------------------------------------------------------- #
class Event:
    def __init__(self, kind: str, frame: int, x: float):
        self.kind = kind       # 'far_serve','near_serve','far_drive','rally','neighbor'
        self.frame = frame     # contact frame
        self.x = x


# --------------------------------------------------------------------------- #
# Low-level emitters (with noise baked in)
# --------------------------------------------------------------------------- #
def _blur_miss_prob(speed_px: float) -> float:
    # Fast balls motion-blur and get missed by YOLO more often.
    return min(0.55, 0.05 + speed_px / 60.0)


def _emit_point(frames, f, cx, cy, side_px, conf, rng):
    """Add one noisy detection; randomly drop it as a 'missed detection'."""
    if f < 0 or f >= len(frames):
        return
    cxj = cx + rng.normal(0, 1.4)
    cyj = cy + rng.normal(0, 1.4)
    s = max(3.0, side_px * (1 + rng.normal(0, 0.10)))
    frames[f].append(Detection(f, cxj, cyj, s, s, conf))


def _emit_path(frames, f0, xs, ys, sides, rng, conf=0.7, blur=True):
    """Place a moving ball along (xs,ys,sides); drop fast frames (blur)."""
    for k in range(len(xs)):
        f = f0 + k
        if f >= len(frames):
            break
        if blur and k > 0:
            sp = np.hypot(xs[k] - xs[k - 1], ys[k] - ys[k - 1])
            if rng.random() < _blur_miss_prob(sp):
                continue
        _emit_point(frames, f, xs[k], ys[k], sides[k], conf, rng)


def _pock(audio, t_sec, amp, rng, length_s=0.006):
    i0 = int(t_sec * SR)
    L = int(length_s * SR)
    if i0 < 0 or i0 + L > audio.size:
        return
    env = np.exp(-np.linspace(0, 6, L))
    audio[i0:i0 + L] += env * rng.standard_normal(L) * amp


# --------------------------------------------------------------------------- #
# Serve / stroke builders
# --------------------------------------------------------------------------- #
def _far_serve(frames, audio, f0, rng, events):
    base_x = rng.uniform(560, 760)
    rise = 13
    xs = [base_x + rng.normal(0, 1.0) for _ in range(rise + 1)]
    ys = [300 - 150 * np.sin((np.pi / 2) * k / rise) for k in range(rise + 1)]
    sides = [8.0 for _ in range(rise + 1)]
    _emit_path(frames, f0, xs, ys, sides, rng, conf=0.8, blur=False)
    apex_f = f0 + rise
    _pock(audio, apex_f / FPS, amp=rng.uniform(0.35, 0.5), rng=rng)  # quiet (far)
    dxs, dys, dsd = [], [], []
    for k in range(1, 30):
        dxs.append(base_x + 4 * k + rng.normal(0, 2))
        dys.append(150 + 11 * k)
        dsd.append(8 + 0.55 * k)
    _emit_path(frames, apex_f + 1, dxs, dys, dsd, rng, conf=0.7, blur=True)
    events.append(Event("far_serve", apex_f, base_x))
    return f0 + rise + 30


def _near_serve(frames, audio, f0, rng, events):
    base_x = rng.uniform(480, 820)
    rise = 13
    xs = [base_x + rng.normal(0, 1.5) for _ in range(rise + 1)]
    ys = [620 - 150 * np.sin((np.pi / 2) * k / rise) for k in range(rise + 1)]
    sides = [30.0 for _ in range(rise + 1)]            # big: close to camera
    _emit_path(frames, f0, xs, ys, sides, rng, conf=0.85, blur=False)
    apex_f = f0 + rise
    _pock(audio, apex_f / FPS, amp=rng.uniform(0.8, 1.0), rng=rng)   # loud (near)
    axs, ays, asd = [], [], []
    for k in range(1, 30):                              # travels AWAY: shrinks, up
        axs.append(base_x + 5 * k + rng.normal(0, 2))
        ays.append(470 - 8 * k)
        asd.append(max(6, 30 - 0.7 * k))
    _emit_path(frames, apex_f + 1, axs, ays, asd, rng, conf=0.75, blur=True)
    events.append(Event("near_serve", apex_f, base_x))
    return f0 + rise + 30


def _far_drive(frames, audio, f0, rng, events):
    """Far player groundstroke toward camera: area grows + downward, NO toss."""
    base_x = rng.uniform(500, 780)
    _pock(audio, f0 / FPS, amp=rng.uniform(0.4, 0.6), rng=rng)
    xs, ys, sd = [], [], []
    for k in range(28):
        xs.append(base_x + 6 * k + rng.normal(0, 2))
        ys.append(220 + 12 * k)                         # starts mid, goes down
        sd.append(9 + 0.5 * k)                          # grows
    _emit_path(frames, f0, xs, ys, sd, rng, conf=0.7, blur=True)
    events.append(Event("far_drive", f0, base_x))
    return f0 + 30


def _rally_stroke(frames, audio, f0, rng, events):
    """Generic cross-court ball + one contact pock."""
    x0, x1 = rng.uniform(300, 1000), rng.uniform(300, 1000)
    y0, y1 = rng.uniform(250, 500), rng.uniform(250, 500)
    n = rng.integers(18, 28)
    _pock(audio, f0 / FPS, amp=rng.uniform(0.3, 0.55), rng=rng)
    xs = np.linspace(x0, x1, n) + rng.normal(0, 2, n)
    ys = np.linspace(y0, y1, n) + 40 * np.sin(np.linspace(0, np.pi, n))
    depth = np.clip((ys - 200) / 350, 0.1, 1.0)
    sd = 8 + 22 * depth
    _emit_path(frames, f0, list(xs), list(ys), list(sd), rng, conf=0.65, blur=True)
    events.append(Event("rally", int(f0), float(x0)))
    return f0 + int(n)


# --------------------------------------------------------------------------- #
# Background processes (run across the whole timeline)
# --------------------------------------------------------------------------- #
def _add_static_balls(frames, rng, n_static):
    spots = [(rng.uniform(200, 1080), rng.uniform(120, 620)) for _ in range(n_static)]
    sides = [8 + 22 * np.clip((y - 200) / 350, 0.1, 1.0) for _, y in spots]
    for f in range(len(frames)):
        for (sx, sy), ss in zip(spots, sides):
            if rng.random() < 0.95:                     # occasionally missed
                _emit_point(frames, f, sx, sy, ss, 0.55, rng)


def _add_neighbor_court(frames, audio, rng, n_frames, events):
    """Side-of-frame activity: horizontal balls + pocks, plus serve-like decoys."""
    f = int(rng.uniform(0, 120))
    while f < n_frames:
        side_x0 = rng.choice([rng.uniform(40, 230), rng.uniform(1050, 1240)])
        if rng.random() < 0.35:
            # DECOY: vertical-ish toss-like motion in the far band + pock + mild grow
            rise = 12
            xs = [side_x0 + rng.normal(0, 3) for _ in range(rise + 1)]
            ys = [300 - 120 * np.sin((np.pi / 2) * k / rise) for k in range(rise + 1)]
            sd = [8.0] * (rise + 1)
            _emit_path(frames, f, xs, ys, sd, rng, conf=0.6, blur=False)
            _pock(audio, (f + rise) / FPS, amp=rng.uniform(0.4, 0.7), rng=rng)
            dxs = [side_x0 + 5 * k for k in range(1, 18)]
            dys = [180 + 10 * k for k in range(1, 18)]
            dsd = [8 + 0.35 * k for k in range(1, 18)]   # weaker growth than a serve
            _emit_path(frames, f + rise + 1, dxs, dys, dsd, rng, conf=0.55, blur=True)
            events.append(Event("neighbor", int(f + rise), float(side_x0)))
            f += rise + 20
        else:
            n = rng.integers(14, 24)
            xs = np.linspace(side_x0, side_x0 + rng.uniform(-120, 120), n)
            ys = np.linspace(rng.uniform(150, 600), rng.uniform(150, 600), n)
            sd = [7.0] * n
            _pock(audio, f / FPS, amp=rng.uniform(0.3, 0.6), rng=rng)
            _emit_path(frames, f, list(xs), list(ys), sd, rng, conf=0.5, blur=True)
            events.append(Event("neighbor", int(f), float(side_x0)))
            f += int(n)
        f += int(rng.uniform(20, 90))


def _add_yolo_fp(frames, rng, rate=0.06):
    for f in range(len(frames)):
        if rng.random() < rate:
            frames[f].append(Detection(f, rng.uniform(0, 1280), rng.uniform(0, 720),
                                       rng.uniform(6, 10), rng.uniform(6, 10), 0.35))


# --------------------------------------------------------------------------- #
# Full match
# --------------------------------------------------------------------------- #
FPS = 30.0


def generate_match(n_points=14, seed=0):
    rng = np.random.default_rng(seed)
    # Rough frame budget: each point ~ serve(45) + rally + gap.
    n_frames = 200 + n_points * 230
    frames = [[] for _ in range(n_frames)]
    audio = 0.01 * rng.standard_normal(int(n_frames / FPS * SR) + SR)
    events: list[Event] = []
    gt_far: list[int] = []

    f = 80
    for p in range(n_points):
        far = (p % 2 == 0)                              # alternate server
        if far:
            f = _far_serve(frames, audio, f, rng, events)
            gt_far.append(events[-1].frame)
        else:
            f = _near_serve(frames, audio, f, rng, events)
        for _ in range(int(rng.integers(2, 6))):        # rally
            if rng.random() < 0.35:
                f = _far_drive(frames, audio, f, rng, events)
            else:
                f = _rally_stroke(frames, audio, f, rng, events)
            f += int(rng.uniform(3, 12))
        f += int(rng.uniform(30, 70))                   # point-end gap
        if f >= n_frames - 120:
            break

    _add_static_balls(frames, rng, n_static=int(rng.integers(1, 4)))
    _add_neighbor_court(frames, audio, rng, n_frames, events)
    _add_yolo_fp(frames, rng)

    # crowd / misc background pocks
    for _ in range(int(n_frames / FPS * 0.4)):
        _pock(audio, rng.uniform(0, n_frames / FPS), amp=rng.uniform(0.15, 0.35), rng=rng)

    return {"frames": frames, "audio": audio, "sr": SR,
            "events": events, "gt_far_frames": sorted(gt_far), "n_frames": n_frames}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(cfg: Config, match, clean_stream, flux):
    tracks = M.KalmanBallTracker(cfg).run(clean_stream)
    impacts = M.pick_onsets(flux, match["sr"], cfg)
    serves = M.detect_serves(tracks, impacts, cfg)

    tol = int(0.4 * cfg.fps)
    gt = list(match["gt_far_frames"])
    det = sorted(ev.impact_frame for ev in serves)

    matched_gt, used = 0, [False] * len(det)
    fp_frames = []
    # Greedy one-to-one matching of detections to GT far serves.
    for g in gt:
        best, bi = tol + 1, -1
        for i, d in enumerate(det):
            if used[i]:
                continue
            if abs(d - g) <= tol and abs(d - g) < best:
                best, bi = abs(d - g), i
        if bi >= 0:
            used[bi] = True
            matched_gt += 1
    for i, d in enumerate(det):
        if not used[i]:
            fp_frames.append(d)

    tp = matched_gt
    fp = len(det) - tp
    fn = len(gt) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Attribute each false positive to the nearest non-far event type.
    causes = {}
    for d in fp_frames:
        near = min((e for e in match["events"] if e.kind != "far_serve"),
                   key=lambda e: abs(e.frame - d), default=None)
        k = near.kind if near and abs(near.frame - d) <= 1.5 * cfg.fps else "spurious"
        causes[k] = causes.get(k, 0) + 1
    return {"tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec,
            "f1": f1, "fp_causes": causes, "n_tracks": len(tracks),
            "n_impacts": len(impacts)}


# --------------------------------------------------------------------------- #
# Randomized fine-tuning
# --------------------------------------------------------------------------- #
SEARCH_SPACE = {
    "far_roi_y_max_frac": (0.40, 0.60),
    "rise_vy_px_s": (40.0, 110.0),
    "verticality_min": (1.0, 3.2),
    "area_cv_max": (0.20, 0.55),
    "area_growth_min": (0.4, 1.6),
    "link_px": (55.0, 140.0),
    "recovery_px": (80.0, 160.0),
    "gate_maha": (6.0, 15.0),
    "audio_peak_k": (1.6, 3.8),
    "audio_pre_apex_tol_s": (0.10, 0.30),
    "audio_post_desc_tol_s": (0.15, 0.40),
}


def sample_cfg(base: Config, rng) -> Config:
    over = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SEARCH_SPACE.items()}
    return dataclasses.replace(base, **over)


def score_key(m):
    # Maximize F1; tie-break: fewer FP, then higher recall.
    return (round(m["f1"], 4), -m["fp"], round(m["rec"], 4))


def tune(base: Config, matches, n_iter=70, seed=1):
    rng = np.random.default_rng(seed)
    # Pre-compute the parts that don't depend on tuned params.
    prepped = []
    for mt in matches:
        zones = M.find_static_exclusion_zones(mt["frames"], base)
        clean = M.apply_exclusion_zones(mt["frames"], zones, base)
        flux = M.audio_flux(mt["audio"], mt["sr"], base)
        prepped.append((mt, clean, flux, zones))

    def mean_eval(cfg):
        ms = [evaluate(cfg, mt, clean, flux) for (mt, clean, flux, _) in prepped]
        agg = {k: float(np.mean([m[k] for m in ms]))
               for k in ("tp", "fp", "fn", "prec", "rec", "f1")}
        return agg

    base_eval = mean_eval(base)
    best_cfg, best_eval = base, base_eval
    for _ in range(n_iter):
        cfg = sample_cfg(base, rng)
        ev = mean_eval(cfg)
        if score_key(ev) > score_key(best_eval):
            best_cfg, best_eval = cfg, ev
    return base_eval, best_cfg, best_eval, prepped


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _fmt(m):
    return (f"P={m['prec']:.2f} R={m['rec']:.2f} F1={m['f1']:.2f} "
            f"(TP={m['tp']:.1f} FP={m['fp']:.1f} FN={m['fn']:.1f})")


def main():
    base = Config()
    # A few independent matches so we don't overfit one random seed.
    matches = [generate_match(n_points=14, seed=s) for s in (11, 22, 33)]
    tot_far = sum(len(mt["gt_far_frames"]) for mt in matches)
    print(f"Generated {len(matches)} matches, {tot_far} far serves total, "
          f"{sum(len(mt['events']) for mt in matches)} labeled events.\n")

    base_eval, best_cfg, best_eval, prepped = tune(base, matches, n_iter=70)

    print("BASELINE (default config):  " + _fmt(base_eval))
    print("TUNED    (search best)   :  " + _fmt(best_eval))
    print()

    # Per-match FP cause breakdown with the tuned config.
    print("Tuned-config false-positive causes, per match:")
    for (mt, clean, flux, _) in prepped:
        m = evaluate(best_cfg, mt, clean, flux)
        print(f"  match: {_fmt(m)}  fp_causes={m['fp_causes']}")
    print()

    changed = {k: round(getattr(best_cfg, k), 3) for k in SEARCH_SPACE}
    print("Tuned parameters:")
    for k, v in changed.items():
        print(f"  {k:24s} {getattr(base, k):>7} -> {v}")

    with open("tuned_config.json", "w") as fh:
        json.dump(changed, fh, indent=2)
    print("\nSaved tuned overrides to tuned_config.json")


if __name__ == "__main__":
    main()
