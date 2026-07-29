"""
features.py
===========
Frame-rate signals and multi-scale window features for the walking classifier.

Two layers:

  * ``frame_signals``  — per-frame physical quantities that do not depend on any
    window: court position of the feet (metres), body scale, and bbox-normalized
    pose descriptors (ankle separation, knee bend, wrist motion, torso lean).
    Everything pose-derived is divided by the box height so a player at the net
    and a player behind the baseline produce the same numbers.

  * ``window_features`` — statistics of those signals over centred windows of
    0.5/1/2/4/8 s. Walking is a *sustained, moderate, rhythmic* translation, so
    a single instant can never separate it from a split-step or a recovery step;
    the short windows carry the gait rhythm and the long ones carry "is this
    person going somewhere".

Gaps: runs of missing detection up to ``MAX_GAP_S`` are linearly interpolated
(they are single-frame occlusions); longer runs stay NaN and flow through to the
features as NaN, which HistGradientBoosting consumes natively.
"""

import numpy as np

from walking.court import to_court

# COCO keypoint indices
NOSE = 0
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

MAX_GAP_S = 0.5          # interpolate detection gaps no longer than this
KP_CONF = 0.3
WINDOWS_S = (0.5, 1.0, 2.0, 4.0, 8.0)
CADENCE_BAND = (0.7, 4.0)   # Hz; human step rate at a walk through a sprint
SPEED_SMOOTH_S = 0.25


def _interp_gaps(x, max_gap):
    """Linearly fill NaN runs up to ``max_gap`` samples. x is [N] or [N, D]."""
    x = np.array(x, dtype=np.float64, copy=True)
    flat = x.reshape(len(x), -1)
    valid = ~np.isnan(flat[:, 0])
    if valid.sum() < 2:
        return x
    idx = np.flatnonzero(valid)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for d in range(flat.shape[1]):
                flat[a + 1:b, d] = np.interp(np.arange(a + 1, b), [a, b],
                                             [flat[a, d], flat[b, d]])
    return flat.reshape(x.shape)


def _kp_xy(kp, i):
    """[N,2] keypoint pixels, NaN where the keypoint is not confidently seen."""
    xy = kp[:, i, :2].astype(np.float64).copy()
    xy[kp[:, i, 2] < KP_CONF] = np.nan
    return xy


def _mid(a, b):
    """Midpoint that falls back to whichever of the two points exists."""
    m = 0.5 * (a + b)
    only_a = np.isnan(m[:, 0]) & ~np.isnan(a[:, 0])
    only_b = np.isnan(m[:, 0]) & ~np.isnan(b[:, 0])
    m[only_a] = a[only_a]
    m[only_b] = b[only_b]
    return m


def _movmean(x, w):
    """Centred moving mean that ignores NaN."""
    if w <= 1:
        return x
    v = np.isfinite(x)
    xf = np.where(v, x, 0.0)
    k = np.ones(w)
    num = np.convolve(xf, k, mode="same")
    den = np.convolve(v.astype(float), k, mode="same")
    out = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return out


def _angle(a, b, c):
    """Interior angle at b of the a-b-c chain, degrees; NaN if any point is NaN."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos = np.sum(v1 * v2, axis=1) / np.maximum(n1 * n2, 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def frame_signals(kp, bbox, H, fps, on_court=None):
    """Per-frame physical signals. Returns a dict of [N] (or [N,2]) arrays."""
    n = len(kp)
    kp = _interp_gaps(kp, int(MAX_GAP_S * fps))
    bbox = _interp_gaps(bbox, int(MAX_GAP_S * fps))

    h = bbox[:, 3] - bbox[:, 1]              # body scale in pixels
    h = np.where(h > 1.0, h, np.nan)

    l_ank, r_ank = _kp_xy(kp, L_ANK), _kp_xy(kp, R_ANK)
    l_hip, r_hip = _kp_xy(kp, L_HIP), _kp_xy(kp, R_HIP)
    l_sho, r_sho = _kp_xy(kp, L_SHO), _kp_xy(kp, R_SHO)
    l_kne, r_kne = _kp_xy(kp, L_KNE), _kp_xy(kp, R_KNE)
    l_wri, r_wri = _kp_xy(kp, L_WRI), _kp_xy(kp, R_WRI)

    foot = _mid(l_ank, r_ank)
    fallback = np.stack([0.5 * (bbox[:, 0] + bbox[:, 2]), bbox[:, 3]], axis=1)
    bad = np.isnan(foot[:, 0])
    foot[bad] = fallback[bad]

    # Court position of the ground contact point (metres).
    court = np.full((n, 2), np.nan)
    ok = np.isfinite(foot[:, 0]) & np.isfinite(foot[:, 1])
    if ok.any():
        court[ok] = to_court(H, foot[ok])

    hip = _mid(l_hip, r_hip)
    sho = _mid(l_sho, r_sho)

    # Everything below is scale-normalized by body height in pixels.
    ank_sep = np.linalg.norm(l_ank - r_ank, axis=1) / h
    ank_dx = np.abs(l_ank[:, 0] - r_ank[:, 0]) / h
    ank_dy = np.abs(l_ank[:, 1] - r_ank[:, 1]) / h
    hip_above_ank = (foot[:, 1] - hip[:, 1]) / h          # crouch depth
    knee_l = _angle(l_hip, l_kne, l_ank)
    knee_r = _angle(r_hip, r_kne, r_ank)
    knee = np.nanmean(np.stack([knee_l, knee_r], axis=1), axis=1)
    torso_lean = np.degrees(np.arctan2(sho[:, 0] - hip[:, 0],
                                       np.maximum(hip[:, 1] - sho[:, 1], 1e-6)))
    wri_spread = np.linalg.norm(l_wri - r_wri, axis=1) / h
    wri_above_hip = (hip[:, 1] - np.fmin(l_wri[:, 1], r_wri[:, 1])) / h
    sho_width = np.linalg.norm(l_sho - r_sho, axis=1) / h  # body orientation proxy
    aspect = (bbox[:, 2] - bbox[:, 0]) / h

    # Wrist / ankle motion relative to the hip, in body-height units per second:
    # a swing moves the arms fast against a slow torso, a walk does not.
    def rel_speed(pt):
        rel = (pt - hip) / h[:, None]
        d = np.full(n, np.nan)
        d[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps
        return d

    wri_rel_speed = np.fmax(rel_speed(l_wri), rel_speed(r_wri))
    ank_rel_speed = np.fmax(rel_speed(l_ank), rel_speed(r_ank))

    # Court-space velocity: smooth the position first, differentiating raw foot
    # positions turns ankle jitter into metres per second.
    w = max(1, int(round(SPEED_SMOOTH_S * fps)))
    cx = _movmean(court[:, 0], w)
    cy = _movmean(court[:, 1], w)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vx[1:] = np.diff(cx) * fps
    vy[1:] = np.diff(cy) * fps
    speed = np.hypot(vx, vy)
    acc = np.full(n, np.nan)
    acc[1:] = np.abs(np.diff(speed)) * fps

    return {
        "court_x": court[:, 0], "court_y": court[:, 1],
        "cx_s": cx, "cy_s": cy,
        "vx": vx, "vy": vy, "speed": speed, "acc": acc,
        "box_h": h, "aspect": aspect,
        "ank_sep": ank_sep, "ank_dx": ank_dx, "ank_dy": ank_dy,
        "hip_above_ank": hip_above_ank, "knee": knee, "torso_lean": torso_lean,
        "wri_spread": wri_spread, "wri_above_hip": wri_above_hip,
        "sho_width": sho_width,
        "wri_rel_speed": wri_rel_speed, "ank_rel_speed": ank_rel_speed,
        "valid": np.isfinite(court[:, 0]).astype(float),
        # 1 inside the near half, 0 on the surrounding floor (ball carts,
        # walkways), NaN when nobody was tracked at all.
        "on_court": (np.asarray(on_court, dtype=np.float64)
                     if on_court is not None else np.full(n, np.nan)),
    }


def _spectrum(sig, fps, lo, hi):
    """(dominant frequency, its share of in-band power, in-band power) of sig."""
    s = sig[np.isfinite(sig)]
    if len(s) < 8:
        return np.nan, np.nan, np.nan
    s = s - s.mean()
    if s.std() < 1e-6:
        return np.nan, np.nan, 0.0
    s = s * np.hanning(len(s))
    p = np.abs(np.fft.rfft(s, n=max(64, len(s)))) ** 2
    f = np.fft.rfftfreq(max(64, len(s)), d=1.0 / fps)
    band = (f >= lo) & (f <= hi)
    if not band.any() or p[band].sum() <= 0:
        return np.nan, np.nan, 0.0
    pb = p[band]
    i = int(np.argmax(pb))
    return float(f[band][i]), float(pb[i] / pb.sum()), float(pb.sum() / len(s))


def _stats(x, prefix, out):
    v = x[np.isfinite(x)]
    if len(v) == 0:
        for k in ("mean", "std", "p10", "p50", "p90", "max"):
            out[f"{prefix}_{k}"] = np.nan
        return
    out[f"{prefix}_mean"] = float(v.mean())
    out[f"{prefix}_std"] = float(v.std())
    out[f"{prefix}_p10"] = float(np.percentile(v, 10))
    out[f"{prefix}_p50"] = float(np.percentile(v, 50))
    out[f"{prefix}_p90"] = float(np.percentile(v, 90))
    out[f"{prefix}_max"] = float(v.max())


def window_features(sig, fps, idx, windows_s=WINDOWS_S):
    """Feature matrix [len(idx), D] and the matching feature-name list."""
    n = len(sig["speed"])
    rows, names = [], None
    half = {w: int(round(w * fps / 2)) for w in windows_s}

    for t in idx:
        out = {}
        # Instantaneous context: where on the court. Body scale in PIXELS is
        # deliberately absent — it identifies the camera and the depth of this
        # clip's walks, and the ablation showed it costs generalisation
        # (F1 0.728 -> 0.754 when dropped).
        out["court_x"] = sig["court_x"][t]
        out["court_y"] = sig["court_y"][t]

        for w in windows_s:
            h = half[w]
            a, b = max(0, t - h), min(n, t + h + 1)
            sl = slice(a, b)
            dur = (b - a) / fps
            p = f"w{w:g}"

            _stats(sig["speed"][sl], f"{p}_speed", out)
            _stats(np.abs(sig["vx"][sl]), f"{p}_vx", out)
            _stats(np.abs(sig["vy"][sl]), f"{p}_vy", out)
            _stats(sig["acc"][sl], f"{p}_acc", out)
            _stats(sig["ank_sep"][sl], f"{p}_anksep", out)
            _stats(sig["knee"][sl], f"{p}_knee", out)
            _stats(sig["wri_rel_speed"][sl], f"{p}_wrirel", out)
            _stats(sig["ank_rel_speed"][sl], f"{p}_ankrel", out)
            _stats(sig["hip_above_ank"][sl], f"{p}_hipank", out)

            # Net displacement vs path length: a walk goes somewhere, a rally
            # exchange shuffles back and forth around one spot.
            cx, cy = sig["cx_s"][sl], sig["cy_s"][sl]
            fin = np.isfinite(cx) & np.isfinite(cy)
            if fin.sum() >= 2:
                xs, ys = cx[fin], cy[fin]
                net = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
                path = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
                out[f"{p}_net_disp"] = net
                out[f"{p}_net_speed"] = net / dur
                out[f"{p}_path_speed"] = path / dur
                out[f"{p}_straightness"] = net / max(path, 1e-6)
                out[f"{p}_range_x"] = float(xs.max() - xs.min())
                out[f"{p}_range_y"] = float(ys.max() - ys.min())
                out[f"{p}_dy_signed"] = float(ys[-1] - ys[0]) / dur
                out[f"{p}_dx_abs"] = abs(float(xs[-1] - xs[0])) / dur
            else:
                for k in ("net_disp", "net_speed", "path_speed", "straightness",
                          "range_x", "range_y", "dy_signed", "dx_abs"):
                    out[f"{p}_{k}"] = np.nan

            sp = sig["speed"][sl]
            spf = sp[np.isfinite(sp)]
            if len(spf):
                # Duty cycle of movement: walking sits in a moderate band almost
                # all the time; a rally is bursty and dead time is near zero.
                out[f"{p}_frac_slow"] = float(np.mean(spf < 0.4))
                out[f"{p}_frac_walkband"] = float(np.mean((spf >= 0.4) & (spf < 2.5)))
                out[f"{p}_frac_fast"] = float(np.mean(spf >= 2.5))
            else:
                out[f"{p}_frac_slow"] = out[f"{p}_frac_walkband"] = np.nan
                out[f"{p}_frac_fast"] = np.nan

            out[f"{p}_valid"] = float(np.nanmean(sig["valid"][sl]))
            oc = sig["on_court"][sl]
            out[f"{p}_on_court"] = (float(np.nanmean(oc))
                                    if np.isfinite(oc).any() else np.nan)

            # Gait rhythm. Ankle separation peaks once per step; the vertical
            # ankle difference and the hip bob carry the same period.
            if w >= 1.0:
                for key, tag in (("ank_sep", "anksep"), ("ank_dy", "ankdy"),
                                 ("hip_above_ank", "hipbob")):
                    f0, share, pw = _spectrum(sig[key][sl], fps, *CADENCE_BAND)
                    out[f"{p}_{tag}_freq"] = f0
                    out[f"{p}_{tag}_share"] = share
                    out[f"{p}_{tag}_power"] = pw

        if names is None:
            names = list(out.keys())
        rows.append([out[k] for k in names])

    return np.asarray(rows, dtype=np.float32), names
