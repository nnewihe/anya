"""Tunables for the rally-reel builder."""

from dataclasses import dataclass


@dataclass
class ReelConfig:
    # ---- point starts -------------------------------------------------
    near_threshold: float = 0.8
    # anya_near_serve's P is documented as an uncalibrated ranking score
    # ("treat P as a ranking score until it has been tuned against ground
    # truth"), and earlier evaluation found it saturates toward dwell-only
    # with precision that does not separate.  Exposed here rather than
    # buried so it can be raised per clip.

    use_near: bool = True
    use_far: bool = True

    merge_window_s: float = 4.0
    # Near and far detectors can both fire on the same serve, and the far
    # gate fires on the toss while the near gate fires on the strike — so
    # the same physical serve can appear ~1-2s apart on the two timelines.

    # ---- service-run constraint ---------------------------------------
    enforce_service_runs: bool = True
    min_service_run: int = 8
    # One player serves a whole game, so point starts should arrive in runs
    # from the same side rather than alternating freely.  A lone "near"
    # among a stretch of "far" is a detector error, not a service change.
    #
    # NOTE 8 is a denoising threshold, not the rule: a love game is 4
    # points, so the rules-accurate minimum is 4, and a tiebreak alternates
    # every 2 points after the first.  At 8 a genuine service change will
    # be bridged whenever a game is short — set 4 to track the rule more
    # closely, at the cost of letting more isolated mislabels through.
    #
    # The runs at the very start and end of a clip are exempt from the
    # minimum: a recording almost never begins or ends on a game boundary.

    min_boundary_run: int = 2
    # ...but a truncated run still needs this many points.  A boundary run
    # of length 1 is indistinguishable from a single stray detection, and
    # exempting it lets exactly that through at the clip edge.  Set to 1 to
    # restore the fully-open exemption.

    conflict_cost_high: float = 2.0
    conflict_cost_default: float = 1.0
    # Relabelling is not equally plausible in both directions.  A far serve
    # marked HIGH has a confirmed far-to-near ball trace behind it, while a
    # near detection carries an explicitly uncalibrated score — so flipping
    # the former should cost more.  Without this the solver would happily
    # invent a short opening "near game" (exempt from the minimum) and drag
    # a confirmed far serve into it to absorb one stray near detection.

    drop_side_conflicts: bool = False
    # When a detection's side disagrees with the inferred serving side:
    # False relabels it and flags it, True discards it.  Discarding is the
    # aggressive read (a conflicting detection is probably a false
    # positive) and costs recall if it was a real point the other detector
    # missed.

    # ---- point ends ---------------------------------------------------
    point_min_s: float = 3.0     # a point cannot end before this
    point_max_s: float = 40.0    # hard cap when no walk interval is found
    next_serve_guard_s: float = 1.0   # always stop short of the next serve

    walk_min_duration_s: float = 0.4
    # Ignore walk blips shorter than this; the classifier emits brief
    # spurious runs mid-rally when a player's gait momentarily looks like
    # walking (e.g. a slow recovery step).

    walk_min_coverage: float = 0.5
    # walking/predict.py reports `detection_coverage` per interval — the
    # share of frames where a near player was actually tracked.  A low
    # coverage interval is a guess made across a hole in the input, and its
    # own docs say downstream code should be able to drop those.

    ball_quiet_mode: str = "gated"      # "off" | "gated" | "always"
    ball_quiet_s: float = 1.5
    # Walking stays the primary point-end signal.  Ball-quiet is a scoped
    # fallback for the one blind spot walking cannot cover: the near player
    # is off camera 59% of the time on Data/23, and the classifier has
    # nothing to say when it cannot see anyone.
    #
    #   "off"     walking only — the exact previous behaviour, and the
    #             revert path if this turns out worse on other clips.
    #   "gated"   ball-quiet counts only where the near player is untracked
    #             or stationary, i.e. where walking is uninformative.
    #   "always"  every ball-quiet onset counts (blanket union).
    #
    # Measured on Data/23, point-end error against the labelled end:
    #
    #   policy            median err   within 3s   truncations
    #   off                   +4.5s        5/15         2
    #   always                +0.1s        9/15         3
    #   gated                 +0.2s        9/15         2
    #
    # "gated" keeps essentially all of the accuracy of "always" while adding
    # no truncations beyond the two walking already causes on its own.
    # (A truncation is a point ended >2s early, i.e. a rally cut short.)

    near_untracked_s: float = 1.0
    # No near-player fix for this long -> walking is blind, trust ball-quiet.

    near_stationary_s: float = 3.0
    near_stationary_ft: float = 2.0
    # Tracked but parked in one spot -> also uninformative.  NOTE this
    # clause is close to inert on Data/23 (it fires on 0% of rally frames
    # and 3% of dead-time frames) because the near player is only tracked
    # 41% of the time, so there is rarely a position to call stationary.
    # The untracked clause is what earns the improvement.  Kept for cameras
    # that hold the near player more reliably.

    # ---- output segments ----------------------------------------------
    pre_roll_s: float = 1.5      # lead-in before the serve
    post_roll_s: float = 1.0     # tail after the point ends
    merge_gap_s: float = 1.5     # join segments closer than this
    min_segment_s: float = 2.0   # drop anything shorter after clamping
