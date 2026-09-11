"""
rally.py
========
Rally confidence: one continuous value per frame, for the whole video, saying
whether a point is in play.  Agent 3.

This module emits NO events.  It replaces `point_end.detect_ends`, which took a
curve separating live from dead at AUC 86.7% and collapsed it into a
49.6%-recall event stream at its falling edge.  The curve was never the weak
part; the collapse was.  Finding the edges is the orchestrator's job, and it is
better placed to do it because it also holds the serve detections that bound
them.

See RALLY_CONFIDENCE.md for the design and the measurements behind every
constant here.

The construction
----------------
    activity = max(near, far)          body heights per second, per frame
    raw      = activity * (1 - union)  the non-rally union VETOES activity
    conf     = smooth(raw) * (1 - absence)

The first two lines are `point_end.live_score` unchanged, and they are what
earns the 86.7%.  A product and not a sum because the union's job is to veto:
a player walking to the ball is active and emphatically not playing, and only a
multiplicative term can say so.

THE ABSENCE TERM IS THE NEW PART, and it is applied AFTER the smoothing rather
than before.  Before the smoothing it would be a no-op: with nobody tracked,
activity is already zero, so `raw` is already zero and there is nothing left to
veto.  What the term actually removes is the SMOOTHING LEAK -- a 5 s window
sitting at the edge of a 30 s empty stretch spreads real activity into footage
where there is no one on court.  Killing that leak is what sharpens the edge
the orchestrator has to find, which is why it is worth a term of its own rather
than a wider window.

Why absence means dead, measured
--------------------------------
Absence is not a missing measurement to be abstained on.  It is evidence, and
over the corpus (minus clip 58) it is strong evidence, against a base live
prevalence of 35.9%:

    absence run     near absent    far absent    BOTH absent
      1-2 s            51.5%          42.4%         22.9%
      2-3 s            68.0%          23.5%         22.1%
      3-5 s            27.5%          25.6%          0.0%
      5-8 s            26.3%          26.7%          9.4%
      8-15 s           26.6%          32.9%          0.0%
      >15 s            15.6%           2.7%          0.0%

Three things in that table decide the construction:

  SHORT ABSENCE IS EVIDENCE OF LIVE, NOT DEAD.  Under 3 s, every column is at
  or ABOVE the 35.9% base rate -- a brief dropout happens while a rally is
  being played, because that is when players move fast, blur, and occlude each
  other.  So the ramp must start at zero penalty and stay there for seconds.
  A term that penalised absence from the first frame would be backwards.

  THE NEAR PLAYER IS THE ONE THAT MATTERS, AND THIS TABLE DOES NOT SAY SO.
  Read alone it points at a joint "both absent" term: that column is 0.0% live
  from 3 s on, while near-only absence is still 15.6% live at 15 s.  Built and
  measured, the joint term is worth +0.02 AUC and the NEAR term is worth +1.3.
  See W_ABSENCE_NEAR for the numbers.  Marginal prevalence is not incremental
  value: both sides absent already drives `raw` to zero over a wide window, so
  there is nothing left for a veto to remove there.  The case only the near
  term can see is the near player gone while the far player is still tracked
  and still moving -- which holds `max(near, far)` up over footage where the
  point is long over.

  THE PENALTY MUST STAY PARTIAL.  At a full near veto (weight 1.0) mean AUC
  collapses from 90.0% to 83.1%, because it deletes real live play wherever the
  near player is briefly untracked.

Doubles: the veto must describe the player who is playing
---------------------------------------------------------
`raw` takes the MAX activity over the two slots on a side -- the busiest player
on that side.  The union it is multiplied by did not: `run._end_signals` builds
one shim from whichever near slot has better coverage, and every consumer read
that single player.  In singles they are the same person.  In doubles they
routinely are not, and one partner standing at the net then supplies a
"between points" veto that cancels the activity of the partner hitting the ball.

That is measurable and it is the largest union error in the corpus.  Mean union
during LIVE play:

    clip 25 (doubles)   0.413        the two highest values in the corpus
    clip 40 (doubles)   0.388
    singles range       0.09 - 0.37

Both near slots are tracked on 36-38% of frames on those two clips, against
under 5% on most singles clips, so there really are two players to choose
between.  `team_union(mode="active")` picks, per frame, the union of the slot
that supplied the max activity -- so activity and veto describe one person.

The other obvious rule, `mode="min"` (a side is non-rally only if EVERY player
on it looks non-rally), is implemented and is worse: it weakens the veto
uniformly instead of re-aiming it, which helps where the wrong player was
picked and hurts where the right one was.  Per-clip AUC against the shim:

                        clip 25    clip 40    mean per-clip
    min                  +1.2       -1.4          -0.1
    active               +2.0       +1.9          +0.3
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import workdir as WD

from pipeline.anya2 import point_end as PE
from pipeline.anya2 import signals as S
from pipeline.anya2 import tracks as T
from pipeline.anya2.contract import ROI_BOTH, Requirement, W_ALWAYS

ARTIFACT_SUFFIX = "_anya2_rally.npz"
SLOT_WALK_SUFFIX = "_anya2_walk_s%d.npz"
SLOT_SIG_SUFFIX = "_anya2_endsig_s%d.npz"

# -- the window ----------------------------------------------------------
# Swept over all 13 clips: mean per-clip AUC 87.2% at 4 s, 88.2% at 5 s, 88.5%
# at 8 s, and 8 s wins on pooled best-F1 too (71.6% vs 69.7%).  There is no
# separation/threshold trade at the FRAME level, which is the level the
# orchestrator consumes.
#
# `point_end.py` chose 4 s because 8 s scored a worse EVENT F1 for the falling
# edge (40.3% vs 42.8%).  That is a different metric answering a different
# question, and moving the edge-finding into the orchestrator is what dissolves
# the conflict -- frame separation prefers the long window, edge timing
# preferred the short one, and nothing has to satisfy both any more.
#
# 5 s takes +1.0 of the +1.3 and stays nearer the edge.  `raw` is stored
# unsmoothed regardless, so a consumer that wants a different window does not
# need a re-run.
SMOOTH_S = 5.0
SCALE_PCT = 90            # per-clip normaliser; recorded as `scale` in the
                          # artifact.  Activity is in body heights per second,
                          # so its absolute level depends on how large the
                          # players are in frame -- a fixed threshold would mean
                          # a different thing on every camera.  Recorded rather
                          # than left implicit, because a published signal that
                          # is normalised per clip is not comparable across
                          # clips and every downstream threshold must know that.

# -- the absence term ----------------------------------------------------
# Ramp start: 3 s, because under 3 s absence is evidence of LIVE (see the table
# in the module docstring) and a penalty there would invert the signal.
# Ramp end: 8 s, by which point both-absent is 0.0% live in every bucket.
ABSENCE_LO_S = 2.0
ABSENCE_HI_S = 6.0
W_ABSENCE_NEAR = 0.25
W_ABSENCE_FAR = 0.25
# THERE IS NO w_both, AND THAT IS THE SURPRISE HERE.  A joint "nobody on court"
# term was built first, because the live-prevalence table in the docstring says
# it is the decisive one -- both-absent is 0.0% live from 3 s on, while
# near-only absence is still 15.6% live at 15 s.  Measured against the curve it
# is worth +0.02 AUC:
#
#     w_near   w_far   w_both      mean per-clip AUC
#       0       0        0              88.65%        (term off)
#       0       0        1.0            88.67%        <-- the "decisive" term
#       0       0.25     0              88.70%
#       0.25    0        0              89.66%        <-- the near term alone
#       0.25    0.25     0              89.71%        <-- shipped
#
# Marginal prevalence is not incremental value.  Both sides absent already
# drives `raw` to zero over a wide window, so the smoothed score is near the
# floor there before any veto applies and there is nothing left to remove.  The
# case only the near term can see is the NEAR player gone while the FAR player
# is still tracked and still moving, which holds `max(near, far)` up over
# footage where the point is long over.  The joint term is absent rather than
# merely small because it earns nothing at all.
#
# THE WEIGHT IS AN INTERIOR OPTIMUM, NOT A SLOPE.  At w_near 1.00 mean AUC
# COLLAPSES to 83.1%: a full veto deletes real live play wherever the near
# player is briefly untracked, which under 3 s is evidence of LIVE (docstring
# table).  0.25 / 0.50 / 0.75 / 1.00 scores 89.7 / 90.0 / 88.9 / 83.1.
#
# 0.50 SCORES HIGHER ON THE MEAN AND IS NOT SHIPPED.  Per clip, against the
# term switched off:
#
#                     mean     clips improved    clip 35      clip 58
#     0.25 / 0.25     +1.1        12 / 12         +0.1         +1.1
#     0.50 / 0.25     +1.4         7 / 12         -1.1         +2.0
#
# 0.50 buys +0.3 of mean AUC by regressing five clips, one of which is clip 35,
# the corpus's designated out-of-sample clip.  Clip 58 -- held out of every
# number here -- prefers 0.50, and that is the one piece of evidence pointing
# the other way, recorded rather than dropped.  A setting that improves every
# clip it is measured on transfers more credibly than one that is 0.3 better on
# average and worse in five places, and "per-clip is the result, pooled is a
# summary" is the corpus rule this repo already runs on.  Revisit it if the
# orchestrator turns out to want the sharper edge more than the safer curve.
#
# The ramp bounds barely matter: 2/6 s scores 89.71% and 3/8 s 89.66%.  2/6 is
# the earlier of two equivalent answers, and the orchestrator wants the edge as
# early as it can honestly have it.

REQUIREMENT = Requirement(roi=ROI_BOTH, pose_fps=15.0, needs_ball=False,
                          windows=W_ALWAYS)


def artifact_path(video, suffix=ARTIFACT_SUFFIX):
    d = WD.artifact_dir(video)
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(d, f"{stem}{suffix}")


def _slot_signals(video, slot: int, force: bool = False) -> Dict[str, np.ndarray]:
    """The walking probability and `near_end`'s four signals for ONE near slot.

    WHY PER SLOT.  `run._end_signals` builds a single shim from the near slot
    with the better coverage, gap-filled from the other, and every consumer of
    the non-rally union reads that one player.  In singles that is the only
    player there and the shim is exactly right.  In DOUBLES it is one of two,
    and the union it produces is then applied multiplicatively to the whole
    court -- so a partner standing still at the net vetoes their partner's
    activity while the point is being played.

    Measured over the corpus, the union during LIVE play on the two doubles
    clips is 0.413 (clip 25) and 0.388 (clip 40), the two highest values in the
    corpus against a singles range of 0.09-0.37.  The veto is firing hardest
    exactly where it is least entitled to.

    Cached per slot beside the video.  Untracked slots are cheap: the pose is
    already computed, and this only runs the walking classifier and `near_end`
    over it.
    """
    import sys as _sys
    d = WD.artifact_dir(video)
    stem = os.path.splitext(os.path.basename(video))[0]
    walk_p = os.path.join(d, stem + SLOT_WALK_SUFFIX % slot)
    sig_p = os.path.join(d, stem + SLOT_SIG_SUFFIX % slot)
    if not (os.path.isfile(walk_p) and os.path.isfile(sig_p)) or force:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        if root not in _sys.path:
            _sys.path.insert(0, root)
        from walking.predict import predict_video
        from pipeline import near_end as NE
        z = T.load(video)
        extra = {k: z[k] for k in ("stride", "src_fps", "n_src_frames") if k in z}
        shim = os.path.join(d, f"{stem}_anya2_pose_s{slot}.npz")
        # NO gap-fill from the partner here, unlike `run._end_signals`.  The
        # whole point is to describe THIS player; borrowing the other's pose to
        # cover a gap is what produced the conflated signal in the first place.
        np.savez_compressed(shim, kp=z["kp"][:, slot], bbox=z["bbox"][:, slot],
                            on_court=z["eligible"][:, slot].astype(np.float32),
                            fps=np.float64(z["fps"]), **extra)
        r = predict_video(video, pose_npz=shim)
        np.savez_compressed(walk_p, prob=r["prob"], fps=np.float64(r["fps"]))
        sig = NE.signals_for_video(video, pose_npz=shim)
        np.savez_compressed(sig_p, **{k: np.asarray(sig[k], dtype=np.float32)
                                      for k in NE.SIGNAL_NAMES})
    w, sg = np.load(walk_p), np.load(sig_p)
    n = min(len(w["prob"]), len(sg[PE.UNION_NAMES[0]]))
    parts = [np.nan_to_num(sg[k][:n], nan=0.0) for k in PE.UNION_NAMES]
    parts.append(np.asarray(w["prob"][:n], dtype=np.float64))
    return {"union": np.max(np.stack(parts), axis=0), "fps": float(w["fps"])}


def slot_unions(video, n: int, tracks_npz=None, force: bool = False):
    """Per-near-slot non-rally union, and which slots are tracked. ([2,n],[2,n])."""
    z = T.load(video, tracks_npz)
    bb = z["bbox"]
    U = np.zeros((len(T.NEAR_SLOTS), n))
    OK = np.zeros((len(T.NEAR_SLOTS), n), dtype=bool)
    for i, slot in enumerate(T.NEAR_SLOTS):
        seen = np.isfinite(bb[:, slot, 0])
        if not seen.any():
            continue
        u = _slot_signals(video, slot, force=force)["union"]
        m = min(n, len(u), len(seen))
        U[i, :m] = u[:m]
        OK[i, :m] = seen[:m]
    return U, OK


def team_union(video, n: int, act: np.ndarray, tracks_npz=None,
               mode: str = "active", force: bool = False) -> np.ndarray:
    """The non-rally veto for the near TEAM, read off the player who is playing.

    THE VETO MUST DESCRIBE THE SAME PERSON THE ACTIVITY DOES.  `raw` takes the
    MAX activity over the two near slots -- the busiest player on the side.
    The union it is multiplied by came from `run._end_signals`, which builds one
    shim from whichever slot has better coverage.  In singles those are the same
    player.  In doubles they are routinely not, and then a partner standing at
    the net supplies a "between points" veto that cancels the activity of the
    player actually hitting the ball.  Measured, the union during LIVE play is
    0.413 on clip 25 and 0.388 on clip 40 -- the two highest in the corpus
    against a singles range of 0.09-0.37.

    `mode="active"` therefore selects, per frame, the union of the slot that
    supplied the max activity.  Activity and veto then describe one person, and
    the pair reads as "how busy is the busiest player on this side, and does
    THAT player look like someone between points".

    `mode="min"` is the other obvious rule -- the side is non-rally only if
    every player on it looks non-rally -- and it is kept because it is cheap to
    compare, not because it is better.  It weakens the veto uniformly rather
    than re-aiming it, which helps where the wrong player was selected and
    hurts where the right one was: clip 25 +1.2 AUC, clip 40 -1.4.

    An UNTRACKED slot never wins the selection and is excluded from the min --
    absence is not a vote for "playing".  With neither slot tracked the veto is
    zero and the absence term carries the frame instead.
    """
    U, OK = slot_unions(video, n, tracks_npz, force=force)
    A = act[list(T.NEAR_SLOTS)][:, :n]
    if mode == "min":
        out = np.min(np.where(OK, U, np.inf), axis=0)
        return np.where(np.isfinite(out), out, 0.0)
    # active: the slot with the greatest activity, among those tracked
    A = np.where(OK & np.isfinite(A), A, -np.inf)
    best = np.argmax(A, axis=0)
    none = ~np.isfinite(A).any(axis=0)
    out = U[best, np.arange(n)]
    return np.where(none, 0.0, out)


def run_length_s(mask: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame length in seconds of the True run each frame belongs to.

    Zero where the mask is False.  This is what turns absence from a boolean
    into evidence: a 0.5 s dropout and a 40 s changeover are the same mask and
    opposite conclusions, and only the run length tells them apart.
    """
    out = np.zeros(len(mask), dtype=np.float64)
    for a, b in S.runs(mask):
        out[a:b] = (b - a) / fps
    return out


def absence_term(act: np.ndarray, fps: float) -> Dict[str, np.ndarray]:
    """Graded 0->1 "nobody is there, and has not been for a while"."""
    near = ~np.isfinite(act[list(T.NEAR_SLOTS)]).any(axis=0)
    far = ~np.isfinite(act[list(T.FAR_SLOTS)]).any(axis=0)
    both = near & far
    # Each side's ramp is computed on its OWN run lengths, not on the joint
    # mask sliced up: a near player absent for 20 s while the far player comes
    # and goes is one 20 s near absence, and reading it as several short ones
    # would discard exactly the duration that makes it informative.
    r_near = S.ramp(run_length_s(near, fps), ABSENCE_LO_S, ABSENCE_HI_S)
    r_far = S.ramp(run_length_s(far, fps), ABSENCE_LO_S, ABSENCE_HI_S)
    # max, not a sum: the two overlap whenever both are gone, and adding them
    # would count one absence twice.
    dead = np.fmax(W_ABSENCE_NEAR * r_near, W_ABSENCE_FAR * r_far)
    return {"absence": np.clip(dead, 0.0, 1.0),
            "absent_near": near.astype(np.float32),
            "absent_far": far.astype(np.float32),
            "valid": (~both).astype(np.float32)}


def compute(video, tracks_npz=None, smooth_s: Optional[float] = None,
            w_absence: Optional[float] = None,
            per_slot_union: bool = True) -> Dict[str, np.ndarray]:
    """The rally confidence curve and every channel behind it."""
    parts = PE.end_signal(video, tracks_npz)
    fps = float(parts["fps"])
    n = len(parts["union"])
    z = T.load(video, tracks_npz)
    act = PE.player_activity(z, fps)[:, :n]

    union = parts["union"]
    if per_slot_union:
        try:
            union = team_union(video, n, act, tracks_npz,
                               mode=("min" if per_slot_union == "min" else "active"))
        except Exception:
            union = parts["union"]      # fall back to the single-player shim

    with np.errstate(invalid="ignore"):
        near = np.nan_to_num(np.nanmax(act[list(T.NEAR_SLOTS)], axis=0), nan=0.0)
        far = np.nan_to_num(np.nanmax(act[list(T.FAR_SLOTS)], axis=0), nan=0.0)
    raw = np.fmax(near, far) * (1.0 - union)

    w = max(1, int(round((SMOOTH_S if smooth_s is None else float(smooth_s)) * fps)))
    sm = np.convolve(raw, np.ones(w) / w, mode="same")
    scale = max(float(np.percentile(sm, SCALE_PCT)), 1e-6)
    sm = sm / scale

    ab = absence_term(act, fps)
    k = 1.0 if w_absence is None else float(w_absence)
    conf = np.clip(sm * (1.0 - k * ab["absence"]), 0.0, None)

    return {"conf": conf, "raw": raw, "smoothed": sm,
            "near_act": near, "far_act": far, "union": union,
            "scale": np.float64(scale), "fps": np.float64(fps), **ab}


def save(video, out=None, **kw) -> str:
    r = compute(video, **kw)
    out = out or artifact_path(video)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in r.items()})
    return out


def load(video, path=None):
    return np.load(path or artifact_path(video))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("video")
    ap.add_argument("--tracks", default=None)
    ap.add_argument("--smooth", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = compute(a.video, a.tracks, smooth_s=a.smooth)
    print(f"[rally] {len(r['conf'])} samples at {float(r['fps']):.3f} fps  "
          f"scale={float(r['scale']):.4f}")
    print(f"[rally] conf mean {r['conf'].mean():.3f}  "
          f"absence>0 on {100 * (r['absence'] > 0).mean():.1f}% of frames  "
          f"unmeasurable {100 * (1 - r['valid'].mean()):.1f}%")
    p = save(a.video, a.out, tracks_npz=a.tracks, smooth_s=a.smooth)
    print(f"[rally] wrote {p}")


if __name__ == "__main__":
    main()
