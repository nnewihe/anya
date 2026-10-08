"""
serve_common.py
===============
The steps the near and far serve detectors share exactly.  Each detector keeps
its own features, constants, ready/stillness/toss scoring and timing; only the
skeleton lives here.

A serve is READY -> TROPHY -> SWING.  Both detectors:

  1. compute a per-sample `trophy` score in [0, 1]
  2. take every run of samples where trophy >= the detector's minimum (and,
     if required, the player is in the allowed zone) as ONE candidate, timed
     at the run's peak                                    [trophy_candidates]
  3. score the swing: does the RACKET wrist (the one that was lower at the
     trophy) rise over the head within a short window after it   [swing]
  4. combine:   shape = weighted mean of trophy and ready scores
                p     = shape * (SWING_FLOOR + (1 - SWING_FLOOR) * swing)
     so a missing swing can veto, but only softly                [combine]
  5. keep only the strongest candidate per refractory window, first within
     each player slot, then across slots (in doubles that also decides WHO
     served)                                    [signals.keep_one_per_window]

Image y grows DOWNWARD.  Every elevation in these detectors is written so that
POSITIVE = HIGHER, in body heights ("bh").
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pipeline.anya2 import signals


def samples(seconds, fps):
    """Seconds -> whole pose samples (rounded)."""
    return int(round(seconds * fps))


@dataclass
class TrophyCandidate:
    run_start: int        # first sample of the trophy run
    peak: int             # sample where the trophy score peaks
    trophy: float         # trophy score at the peak


def trophy_candidates(trophy, allowed, trophy_min, require_court) -> List[TrophyCandidate]:
    """One candidate per run of trophy >= trophy_min (AND `allowed`, if required)."""
    trophy_score = np.nan_to_num(trophy, nan=0.0)
    is_candidate = trophy_score >= trophy_min
    if require_court:
        is_candidate = is_candidate & allowed
    out = []
    for run_start, run_stop in signals.true_runs(is_candidate):
        peak = run_start + int(np.argmax(trophy_score[run_start:run_stop]))
        out.append(TrophyCandidate(run_start, peak, float(trophy_score[peak])))
    return out


@dataclass
class Swing:
    score: float                  # 0..1, how far the racket wrist cleared the head
    contact_s: Optional[float]    # time of the racket wrist's highest point
    toss_is_left: bool            # which arm was tossing (higher wrist at trophy)


def swing(features, peak, fps, swing_min_s, swing_max_s, peak_above_head_bh) -> Swing:
    """Did the racket wrist rise over the head shortly after the trophy?

    The tossing arm is the one carrying the higher wrist at the trophy peak;
    the racket arm is the other.  Contact is read as the racket wrist's
    highest point in [peak + swing_min_s, peak + swing_max_s] -- a position,
    never a velocity (at 15 Hz the strike spans about two samples).

    `features` needs `hi_side` (+1 left wrist higher, -1 right, 0 unknown) and
    `head_l` / `head_r` (each wrist's height above the head, bh).
    """
    toss_is_left = features.hi_side[peak] > 0
    racket_above_head = features.head_r if toss_is_left else features.head_l
    n = len(racket_above_head)
    first = max(1, samples(swing_min_s, fps))
    last = samples(swing_max_s, fps)
    window_start, window_stop = min(n, peak + first), min(n, peak + last + 1)
    if window_stop > window_start:
        window = racket_above_head[window_start:window_stop]
        seen = np.isfinite(window)
        if seen.any():
            highest = int(np.nanargmax(np.where(seen, window, -np.inf)))
            score = float(signals.ramp(float(window[highest]), 0.0, peak_above_head_bh))
            return Swing(score, (window_start + highest) / fps, toss_is_left)
    return Swing(0.0, None, toss_is_left)


def combine(trophy, ready, swing_score, w_trophy, w_ready, swing_floor):
    """Weighted shape score, then gated (softly) by the swing."""
    shape = (w_trophy * trophy + w_ready * ready) / (w_trophy + w_ready)
    return shape * (swing_floor + (1.0 - swing_floor) * swing_score)
