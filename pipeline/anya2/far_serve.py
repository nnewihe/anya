"""
far_serve.py
============
A far-side serve detector: the same phase-sequence idea as `near_serve`, with
the geometry re-derived for a player 30 pixels tall on the other side of the
net.

Why not just run the near detector on the far slots
---------------------------------------------------
Because two of its three trophy terms do not survive the change of viewpoint
and scale, and one of them is actively harmful.  Measured on Data/23's 15
labelled far serves, per pose sample, against every non-serve sample in the
clip:

    hi_head > 0.05      serve 13%   non-serve  0%   (non-serve p99 = +0.002)
    hi_elev > 0.10      serve 14%   non-serve  1%
    lo_elev > -0.05     serve 11%   non-serve  2%
    gap     > 0.18      serve 35%   non-serve 58%   <-- INVERTED

The elevation terms are close to perfectly discriminative: a far wrist above
the head essentially never happens outside a serve.  THE HAND SPLIT IS WORSE
THAN USELESS.  On the near player -- filmed from behind, 200 px tall -- the
hands coming apart off the grip is the sharpest transition in the action.  On
the far player it is not measurable: the arms are extended VERTICALLY at the
trophy, so their horizontal separation is small exactly when the near view
would have it large, while ordinary walking swings the arms apart.  So the far
trophy is elevation-only, and `gap` is not read anywhere in this module.

That asymmetry is the reason these are two detectors and not one with a flag.

What is kept from the near construction
---------------------------------------
  * THE ORDERING.  ready -> trophy -> swing, each within a bounded delay.  A
    lone thresholded shape is what the shipped `anya_far_serve` hand-raise gate
    is, and its measured cost is 39 false positives over 14 clips.

  * THE PHASE DILATION.  Terms are dilated before the conjunction, so the
    product describes a phase and not a single 67 ms sample.  This mattered
    more on the near side than anything else and the sampling argument is
    identical here.

  * THE SWING MULTIPLIES WITH A FLOOR, the shape terms add.  Same arbitration,
    same reason: a raise that never becomes a strike must be vetoable, but a
    serve caught mid-stroke must not be thrown away.

What is dropped, and why
------------------------
  * THE HAND SPLIT -- measured inverted, see above.

  * THE SERVE-ZONE BAND ON court_y.  The near detector gates on the server
    standing within a 1.4 m band about the near baseline, which is what
    separates a serve from a mid-court smash.  The same gate cannot be built
    here: a far player's ground point is 22-32 px tall in the analysis frame,
    and at that depth two pixels of box-bottom error is metres of court.
    Measured on Data/23 the far player's court_y spans 19.7-28.6 m (p5-p95)
    while really moving about three, and sits a median 4 m BEHIND the baseline
    it is standing on.  A 1.4 m band on that quantity is noise.  The far
    player is still required to be on the far side and inside the lateral
    doubles+3ft gate (`tracks.eligible`), because those are coarse enough to
    survive the noise -- 100% of far detections satisfy both.

BOX SANITY, which the near side never needed
--------------------------------------------
Every quantity here is normalised by box height, and at 30 px a bad box makes
those ratios meaningless rather than merely noisy -- a merged two-person box or
a fragment produced apparent wrist separations of 1.4-1.9 body heights, which
is anatomically impossible.  `_sane` rejects boxes by height and aspect before
anything is divided by them.  This is not a detection threshold; it is refusing
to divide by a number that is wrong.

What this detector structurally cannot do
-----------------------------------------
The same thing the near one cannot: know whether a point is already in
progress.  The shipped far detector's 39 false positives over 14 clips broke
down as 17 idle raises in dead time, 14 in-rally overheads and returns, and 8
the far player REACTING to a near serve -- 22 of 39 being "play was live and
the detector did not know it".  None of that is visible from inside a far-player
pose crop, and every far-side pose formulation aimed at it has failed.  This
module therefore does not try: it declares `windows="between_points"` and
leaves the arbitration to the composition layer, which will have the near
serves and the point ends that actually settle it.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.anya2 import signals as S
from pipeline.anya2 import tracks as T
from pipeline.anya2.contract import (FAR_SERVE, ROI_FAR, W_BETWEEN, Event,
                                     Requirement, dump_events)

EVENTS_SUFFIX = "_anya2_far_serve.json"
MAX_GAP_S = 0.5
KP_CONF = 0.20            # keypoint confidence floor.  Higher than the near
                          # side's implicit 0: at this scale a low-confidence
                          # keypoint is a guess, and a guessed wrist reads as a
                          # posture the player never held.

# ── box sanity ───────────────────────────────────────────────────────────
BOX_H_MIN_PX, BOX_H_MAX_PX = 12.0, 120.0
BOX_ASPECT_MAX = 1.6      # wider than this is two people merged or a fragment

# ── head reference ───────────────────────────────────────────────────────
# The far player FACES this camera, so nose/eyes/ears are the keypoints the
# model is most confident about -- the exact opposite of the near player, who
# has their back turned.  The geometric fallback is therefore rarely used here.
HEAD_ABOVE_SHO_BH = 0.14

# ── ready ────────────────────────────────────────────────────────────────
READY_HI_MAX_BH = 0.02    # both wrists carried at or below the shoulder line.
                          # Tighter than the near side's 0.10 because the far
                          # ready position is measured over a 30 px body and a
                          # loose bound admits half the clip.
READY_STILL_BH_S = 1.2    # limb speed relative to the hip, body heights/s.
                          # Looser than near's 0.9: at 30 px the same real
                          # motion is a smaller pixel motion but a NOISIER
                          # normalised one, so the quiet bar has to allow for
                          # keypoint jitter that the near view does not have.
READY_WIN_S = 0.7

# ── how the ready phase is READ ──────────────────────────────────────────
# The original reading -- the MAX of `ready` anywhere in the six seconds before
# the trophy -- is nearly useless, and measurably so: over 105 true far serves
# and 101 false positives its median is 1.00 on BOTH, separating them at AUC
# 62%.  Any six-second window contains some quiet moment, so a returner who
# stood still once five seconds ago scores a perfect ready.
#
# Read instead as the MEAN over a short window ENDING AT THE TROPHY, the
# separation is AUC 79% (true 0.90, false 0.62).  That is the difference
# between "was quiet at some point recently" and "was quiet right up until the
# racket went up", and only the second is what a service stance is.
READY_MEAN_FROM_S = 2.0   # window start, before the trophy
READY_MEAN_TO_S = 0.3     # ...and end, stopping short of the toss itself

# ── pre-serve stillness ──────────────────────────────────────────────────
# A SERVER IS STATIONARY BEFORE SERVING; A RETURNER HAS JUST BEEN RUNNING.
# That is the single largest thing separating a real far serve from the far
# player's return, and it is local to this detector -- no other agent needed.
#
# Measured as the median of the player's own translation (box centre, in body
# heights per second) over the window below: true far serves 0.15, false
# positives 0.31, separating at AUC 78%.  It is the mirror of the ready term
# and independent of it -- ready is about the ARMS, this is about the FEET.
#
# It enters multiplicatively but softened, so it can veto a candidate that is
# clearly mid-rally without deleting one whose track was noisy.
STILL_FROM_S, STILL_TO_S = 5.0, 1.0
STILL_FULL_BH_S = 0.20    # at or below this the player was stationary
STILL_NONE_BH_S = 0.55    # ...and at or above it they were travelling
W_STILL = 0.30            # weight of the stillness veto; see THRESHOLD

# ── trophy (elevation only -- see the module docstring) ──────────────────
TROPHY_HEAD_MIN_BH = -0.02   # higher wrist below head height = no credit
TROPHY_HEAD_FULL_BH = 0.06   # ...and this far above it = full credit.  Lower
                             # than the near side's 0.10: a 0.10 bh margin is
                             # 3 px on a 30 px body and is not resolvable.
TROPHY_LO_MIN_BH = -0.18     # lower wrist this far below the shoulder = none
TROPHY_LO_FULL_BH = -0.02    # ...and level with it = full credit
TROPHY_DILATE_S = 0.20
TROPHY_MIN = 0.30

# ── swing ────────────────────────────────────────────────────────────────
SWING_MIN_S, SWING_MAX_S = 0.06, 0.90
PEAK_ABOVE_HEAD_BH = 0.04

# ── ready lookback ───────────────────────────────────────────────────────
READY_BACK_MIN_S, READY_BACK_MAX_S = 0.20, 6.00

# ── combination ──────────────────────────────────────────────────────────
W_TROPHY, W_READY = 0.45, 0.20
SWING_FLOOR = 0.45
# Swept on the nine clips carrying a far serve.  Over the six FAR-DOMINANT
# clips (70 of the 77 labelled far serves) the curve reads:
#
#     thr    recall   precision
#     0.55   100.0%     65.4%
#     0.90   100.0%     71.4%      <-- here
#     0.95    98.6%     73.4%
#     0.999   94.3%     76.7%
#
# RE-SWEPT after the ready and stillness terms were added.  Those two multiply
# the score down, so the old 0.90 knee no longer sits in the same place; over
# all 129 labelled far serves on 13 clips:
#
#     ready  w_still  thr     recall  precision   F1
#     max      0.0    0.90     81.4%     51.0%   62.7   <- the previous default
#     max      0.3    0.85     80.6%     63.0%   70.7
#     mean     0.3    0.75     79.8%     65.2%   71.8   <- here
#     mean     0.3    0.85     70.5%     72.8%   71.7
#     mean     0.5    0.90     56.6%     72.3%   63.5
#
# 0.75 with both terms holds recall (-1.6 points) and buys FOURTEEN points of
# precision.  Pushing to 0.85 buys another 7.6 of precision for 9.3 of recall,
# which is the wrong direction for a point START -- a missed serve loses a
# whole point from the reel, an extra one is something the orchestrator can
# arbitrate.
THRESHOLD = 0.75
REFRACT_S = 3.0

# The label lead, as on the near side: `ground_truth.json`'s `start` is a point
# boundary marked before the server moves, so it is corrected as a constant.
# The far anchor is the TROPHY ONSET, not the hands-together instant the near
# detector uses -- the hand split is not measurable on a 30 px body (see the
# module docstring), so there is no earlier anchor to walk back to.  Swept over
# the far corpus, the lead is almost inert: 0.9, 1.2 and 1.6 all produce
# identical matches at a +/-2.0 s tolerance, and only below ~0.6 does it start
# costing hits.  0.90 sits in the middle of that flat region.
SERVE_LEAD_S = 0.90

REQUIREMENT = Requirement(roi=ROI_FAR, pose_fps=15.0, needs_ball=False,
                          windows=W_BETWEEN)


def events_path(video, suffix=EVENTS_SUFFIX):
    d = os.path.dirname(os.path.abspath(video))
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(d, f"{stem}{suffix}")


def _sane(bbox):
    """Box height, NaN where the box cannot be trusted to normalise by."""
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    with np.errstate(invalid="ignore"):
        ok = ((h > BOX_H_MIN_PX) & (h < BOX_H_MAX_PX)
              & (w / np.maximum(h, 1e-6) < BOX_ASPECT_MAX))
    return np.where(ok, h, np.nan)


def serve_primitives(kp, bbox, fps: float,
                     eligible: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Per-sample phase primitives for one far-player track.

    `kp` [N, 17, 3] and `bbox` [N, 4] are one far slot of `anya2.tracks`.
    `eligible` [N] is that module's lateral gate.  There is deliberately no
    `court_y` parameter -- see the module docstring on why a serve-zone band
    cannot be built on the far side.
    """
    kp = np.asarray(kp, dtype=np.float64)
    bbox = np.asarray(bbox, dtype=np.float64)
    n = len(kp)
    gap = int(MAX_GAP_S * fps)
    kp = S.interp_gaps(kp, gap)
    bbox = S.interp_gaps(bbox, gap)

    h = _sane(bbox)
    valid = np.isfinite(h)

    l_sho, r_sho = S.kp_xy(kp, S.L_SHO, KP_CONF), S.kp_xy(kp, S.R_SHO, KP_CONF)
    l_wri, r_wri = S.kp_xy(kp, S.L_WRI, KP_CONF), S.kp_xy(kp, S.R_WRI, KP_CONF)
    l_hip, r_hip = S.kp_xy(kp, S.L_HIP, KP_CONF), S.kp_xy(kp, S.R_HIP, KP_CONF)
    nose = S.kp_xy(kp, S.NOSE, KP_CONF)
    l_ear, r_ear = S.kp_xy(kp, S.L_EAR, KP_CONF), S.kp_xy(kp, S.R_EAR, KP_CONF)

    sho, hip = S.mid(l_sho, r_sho), S.mid(l_hip, r_hip)
    head = S.mid(S.mid(l_ear, r_ear), nose)
    head_y = np.where(np.isfinite(head[:, 1]),
                      head[:, 1], sho[:, 1] - HEAD_ABOVE_SHO_BH * h)

    elev_l = (sho[:, 1] - l_wri[:, 1]) / h
    elev_r = (sho[:, 1] - r_wri[:, 1]) / h
    head_l = (head_y - l_wri[:, 1]) / h
    head_r = (head_y - r_wri[:, 1]) / h

    both = np.isfinite(elev_l) & np.isfinite(elev_r)
    with np.errstate(invalid="ignore"):
        hi_elev = np.where(both, np.fmax(elev_l, elev_r), np.nan)
        lo_elev = np.where(both, np.fmin(elev_l, elev_r), np.nan)
        hi_head = np.where(both, np.fmax(head_l, head_r), np.nan)
    hi_side = np.where(both, np.where(elev_l >= elev_r, 1.0, -1.0), 0.0)

    def rel_speed(pt):
        rel = (pt - hip) / h[:, None]
        d = np.full(n, np.nan)
        d[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=1) * fps
        return d

    # The player's own translation across the ground, in body heights per
    # second.  Box centre rather than a keypoint: at 30 px an individual joint
    # is noisy, while the box as a whole is what moves when the player runs.
    cx = 0.5 * (bbox[:, 0] + bbox[:, 2])
    cy_px = bbox[:, 3]
    self_move = np.full(n, np.nan)
    self_move[1:] = np.hypot(np.diff(cx), np.diff(cy_px)) / h[1:] * fps

    limb = np.fmax(rel_speed(l_wri), rel_speed(r_wri))
    still = 1.0 - S.ramp(limb, READY_STILL_BH_S * 0.5, READY_STILL_BH_S * 1.6)
    carried = 1.0 - S.ramp(hi_elev, READY_HI_MAX_BH, READY_HI_MAX_BH + 0.14)
    ready_raw = carried * still
    w = max(1, int(round(READY_WIN_S * fps)))
    ready = ready_raw if w <= 1 else np.convolve(
        np.nan_to_num(ready_raw, nan=0.0), np.ones(w) / w, mode="same")

    dil = max(1, int(round(TROPHY_DILATE_S * fps)) * 2 + 1)
    trophy = (S.movmax(S.ramp(hi_head, TROPHY_HEAD_MIN_BH, TROPHY_HEAD_FULL_BH), dil)
              * S.movmax(S.ramp(lo_elev, TROPHY_LO_MIN_BH, TROPHY_LO_FULL_BH), dil))

    on_court = (np.asarray(eligible, dtype=bool) if eligible is not None
                else np.ones(n, dtype=bool))

    return {"valid": valid & both, "ready": ready, "trophy": trophy,
            "hi_elev": hi_elev, "lo_elev": lo_elev, "hi_head": hi_head,
            "head_l": head_l, "head_r": head_r, "hi_side": hi_side,
            "still": still, "self_move": self_move,
            "on_court": on_court, "fps": np.float64(fps)}


def detect_serves(prim, threshold: float = THRESHOLD,
                  require_court: bool = True,
                  track: Optional[int] = None) -> List[Dict]:
    """Sequence-match ready -> trophy -> swing, as `near_serve.detect_serves`."""
    fps = float(prim["fps"])
    trophy, ready = prim["trophy"], prim["ready"]
    n = len(trophy)
    tro = np.nan_to_num(trophy, nan=0.0)
    cand = tro >= TROPHY_MIN
    if require_court:
        cand = cand & prim["on_court"]

    back_lo = int(round(READY_BACK_MAX_S * fps))
    back_hi = int(round(READY_BACK_MIN_S * fps))
    fwd_lo = max(1, int(round(SWING_MIN_S * fps)))
    fwd_hi = int(round(SWING_MAX_S * fps))

    out: List[Dict] = []
    for lo, hi in S.runs(cand):
        k = lo + int(np.argmax(tro[lo:hi]))
        s_trophy = float(tro[k])

        # Ready as a MEAN over a short window ending at the trophy, not a max
        # over six seconds -- see READY_MEAN_FROM_S for why the max is inert.
        a = max(0, k - int(round(READY_MEAN_FROM_S * fps)))
        b = max(1, k - int(round(READY_MEAN_TO_S * fps)))
        rd = ready[a:b]
        rd = rd[np.isfinite(rd)]
        s_ready = float(rd.mean()) if rd.size else 0.0

        # Was this player standing still before the racket went up?
        sa = max(0, k - int(round(STILL_FROM_S * fps)))
        sb = max(1, k - int(round(STILL_TO_S * fps)))
        mv = prim["self_move"][sa:sb]
        mv = mv[np.isfinite(mv)]
        # An untracked stretch scores 0.5 rather than 0 or 1: not knowing where
        # the player was is not evidence either way, and this term must not
        # delete a serve just because the track had a hole before it.
        s_still = (1.0 - float(S.ramp(float(np.median(mv)),
                                      STILL_FULL_BH_S, STILL_NONE_BH_S))
                   if mv.size else 0.5)

        toss_left = prim["hi_side"][k] > 0
        rack_head = prim["head_r"] if toss_left else prim["head_l"]
        c, d = min(n, k + fwd_lo), min(n, k + fwd_hi + 1)
        s_swing, t_contact = 0.0, None
        if d > c:
            win = rack_head[c:d]
            ok = np.isfinite(win)
            if ok.any():
                j = int(np.nanargmax(np.where(ok, win, -np.inf)))
                s_swing = float(S.ramp(float(win[j]), 0.0, PEAK_ABOVE_HEAD_BH))
                t_contact = (c + j) / fps

        shape = (W_TROPHY * s_trophy + W_READY * s_ready) / (W_TROPHY + W_READY)
        p = shape * (SWING_FLOOR + (1.0 - SWING_FLOOR) * s_swing)
        p *= (1.0 - W_STILL) + W_STILL * s_still
        if p < threshold:
            continue
        out.append({
            "t": lo / fps - SERVE_LEAD_S,
            "p": round(p, 4),
            "trophy": round(s_trophy, 4), "swing": round(s_swing, 4),
            "ready": round(s_ready, 4), "still": round(s_still, 4),
            "t_trophy": round(k / fps, 3),
            "t_contact": round(t_contact, 3) if t_contact is not None else None,
            "track": track,
        })
    return S.refractory(out, REFRACT_S)


def detect_video(video, tracks_npz=None, threshold: float = THRESHOLD,
                 require_court: bool = True, verbose: bool = True):
    """Score every far slot; the server is whichever produced the candidate."""
    z = T.load(video, tracks_npz)
    fps = float(z["fps"])
    kp, bbox, el = z["kp"], z["bbox"], z["eligible"]

    raw: List[Dict] = []
    for slot in T.FAR_SLOTS:
        seen = np.isfinite(bbox[:, slot, 0])
        if not seen.any():
            if verbose:
                print(f"[far-serve] slot {slot}: never tracked, skipped")
            continue
        prim = serve_primitives(kp[:, slot], bbox[:, slot], fps,
                                eligible=el[:, slot])
        ev = detect_serves(prim, threshold, require_court, track=int(slot))
        if verbose:
            print(f"[far-serve] slot {slot}: tracked {100 * seen.mean():5.1f}%"
                  f"  sane boxes {100 * np.mean(prim['valid']):5.1f}%"
                  f"  -> {len(ev)} candidates")
        raw.extend(ev)

    kept = S.refractory(raw, REFRACT_S)
    return [Event(t=float(e["t"]), p=float(e["p"]), kind=FAR_SERVE,
                  track=e["track"],
                  detail={k: e[k] for k in ("trophy", "swing", "ready",
                                            "still", "t_trophy", "t_contact")})
            for e in kept]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("video")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--no-court", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    ev = detect_video(a.video, a.tracks, a.threshold, not a.no_court)
    print(f"[far-serve] {len(ev)} serves at p >= {a.threshold}")
    for e in ev[:40]:
        d = e.detail
        print(f"  {e.t:8.2f}s  p={e.p:.3f}  slot={e.track}  "
              f"trophy={d['trophy']:.2f} swing={d['swing']:.2f} ready={d['ready']:.2f}")
    out = a.json or events_path(a.video)
    dump_events(ev, out, threshold=a.threshold, requirement=REQUIREMENT.__dict__)
    print(f"[far-serve] wrote {out}")


if __name__ == "__main__":
    main()
