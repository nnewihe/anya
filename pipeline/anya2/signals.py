"""
signals.py
==========
Pure signal-processing helpers shared by the detectors.

MACHINERY IS SHARED HERE; THRESHOLDS ARE NOT.  Every function below is
parameter-free maths -- a ramp, a running max, a run-length decomposition.  The
numbers that decide what counts as a serve or an end stay in the detector that
owns them, because that is the premise of building the three separately: an
agent may retune anything in its own module without touching another's.

If something here ever needs a constant, it belongs in the caller instead.
"""

from typing import List

import numpy as np

# COCO keypoint indices, as in walking.features / near_end.
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16


def ramp(x, lo, hi):
    """Soft 0->1 ramp from lo to hi (either order); NaN in, NaN out."""
    with np.errstate(invalid="ignore"):
        return np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)


def movmax(x, w):
    """Centred running max over `w` samples, NaN-tolerant (NaN acts as -inf).

    Shifts rather than a stride trick, so the edges degrade to a shorter window
    instead of to NaN -- an event in the first second of a clip is still an
    event.

    This is what lets a CONJUNCTION describe a phase rather than an instant.  A
    product of per-sample terms requires them to hold on the same sample, and
    over a fast phase sampled at 15 Hz they routinely do not: measured on the
    near serve, all three trophy terms cleared their thresholds while the
    product stayed near zero, because the tossing arm extends a sample or two
    before the racket arm settles.  Dilating each term first asks "did this hold
    ANYWHERE NEAR now", which is what a phase means.
    """
    x = np.asarray(x, dtype=np.float64)
    if w <= 1:
        return x
    out = x.copy()
    for d in range(1, w // 2 + 1):
        out[:-d] = np.fmax(out[:-d], x[d:])
        out[d:] = np.fmax(out[d:], x[:-d])
    return out


def runs(mask) -> List[tuple]:
    """[(lo, hi)) index runs of True."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(m.astype(np.int8))
    lo = list(np.flatnonzero(d == 1) + 1)
    hi = list(np.flatnonzero(d == -1) + 1)
    if m[0]:
        lo.insert(0, 0)
    if m[-1]:
        hi.append(len(m))
    return list(zip(lo, hi))


def refractory(events, gap_s, key="t", strongest=True):
    """Thin `events` so no two survive within `gap_s`.

    `strongest=True` keeps the highest-scoring member of each cluster (the
    right choice when a cluster is one action detected several times);
    `False` keeps the EARLIEST (the right choice when a cluster is a real
    sequence whose first member is the one being timed).
    """
    if strongest:
        ordered = sorted(events, key=lambda e: (-e.get("p", 0.0), e[key]))
        kept = []
        for e in ordered:
            if all(abs(e[key] - k[key]) >= gap_s for k in kept):
                kept.append(e)
        return sorted(kept, key=lambda e: e[key])
    kept = []
    for e in sorted(events, key=lambda e: e[key]):
        if not kept or e[key] - kept[-1][key] >= gap_s:
            kept.append(e)
    return kept


def interp_gaps(a, max_gap):
    """Linearly fill runs of NaN no longer than `max_gap` samples.

    Short occlusions are filled; real holes are left as NaN, because a filled
    hole is a fabricated posture and every downstream conjunction would read it
    as evidence.
    """
    a = np.array(a, dtype=np.float64, copy=True)
    flat = a.reshape(len(a), -1)
    idx = np.arange(len(a))
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        filled = np.interp(idx, idx[ok], col[ok])
        for lo, hi in runs(~ok):
            if hi - lo <= max_gap and lo > 0 and hi < len(a):
                col[lo:hi] = filled[lo:hi]
    return flat.reshape(a.shape)


def kp_xy(kp, i, conf=0.0):
    """[N,2] of keypoint `i`, NaN where confidence is at or below `conf`."""
    k = np.asarray(kp, dtype=np.float64)
    xy = k[:, i, :2].copy()
    xy[k[:, i, 2] <= conf] = np.nan
    return xy


def mid(a, b):
    """Midpoint of two [N,2] tracks; NaN unless both are present."""
    return 0.5 * (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64))
