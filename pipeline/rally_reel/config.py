"""Tunables for the rally-reel builder."""

from dataclasses import dataclass


@dataclass
class ReelConfig:
    # ---- point starts -------------------------------------------------
    near_threshold: float = 0.8
    # With fast_near this is no longer an uncalibrated guess: 0.80 is the
    # swept optimum for the additive P over 12 ground-truthed clips (59/59
    # recall, 59/88 precision, zero events on both far-serve-only controls),
    # and the threshold curve around it is gentle — 0.775 gives 33 false
    # positives, 0.800 gives 29, 0.825 trades one serve for 25.  Still
    # exposed so it can be raised per clip.
    #
    # With fast_near=False it reverts to the legacy product-form P, where the
    # original caveat stands: that score saturates toward dwell-only and no
    # threshold separates cleanly.

    use_near: bool = True
    use_far: bool = True

    fast_near: bool = True
    # Score near serves from anya_near_telemetry (540p proxy, 5 fps player,
    # upscaled toss-ROI ball) instead of the shared anya_telemetry pass.
    #
    # This is an ACCURACY change first and a compute change only sometimes.
    # Measured against ground truth: Data/38 went 4/8 -> 8/8, Data/21
    # 11/13 -> 11/12 precision at unchanged recall, and across 12 clips the
    # fast path reaches 59/59 recall at 59/88 precision.
    #
    # On compute it depends on what else the run needs.  Far serves (stage 3,
    # via the far-pose pass in stage 2) and ball-quiet dead time (stage 5)
    # both read the full telemetry, so with those enabled BOTH passes run and
    # the reel gets ~8% slower (34.4 -> ~37.2 ms/frame) in exchange for the
    # accuracy above.  With use_far=False and ball_quiet_mode="off" nothing
    # else needs the full pass, stages 1-2 are skipped entirely, and the run
    # is ~12x cheaper.

    fast_far: bool = True
    # Detect far serves from anya_far_telemetry (native-resolution band proxy,
    # 5 fps far player, pose while armed at imgsz 320, gated ball) instead of
    # the shared anya_telemetry pass plus extract_far_pose.
    #
    # Like fast_near this is an accuracy change as well as a compute one.  Over
    # the ten clips where both extractors have run (77 ground-truthed far
    # serves, DESIGN.md 8.5): full pass 41/77 recall with 37 FP, fast path
    # 54/77 with 29 FP — better on both axes.  The full pass's 15/15 reputation
    # was entirely Data/23, the clip its thresholds were fitted to; on clip 26
    # it finds 1 of 11 where the fast path finds 11.
    #
    # detect_far_serves picks the matching threshold preset itself from
    # meta.source (see anya_far_serve.config_for), so nothing downstream of
    # stage 3 changes.  --no-fast-far reverts.
    #
    # Compute only lands when the OTHER consumers of the full pass are gone
    # too: 48.7 -> 6.9 ms/frame steady state, but stage 1 keeps running while
    # ball-quiet dead time still reads it.

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

    fast_end: bool = True
    # Take BOTH point-end signals from anya_end_telemetry (shared 540p proxy,
    # 15 fps pose, 10 fps whole-court ball) instead of a full-rate walking pose
    # pass plus the shared anya_telemetry ball stream.
    #
    # This is the stage that decides whether the reel needs the full pass at
    # all: with fast_near and fast_far already on, ball-quiet was the last
    # consumer of it, so turning this on is what lets stages 1-2 be skipped.
    #
    # walking.predict was already scoring at 15 Hz — every 2nd frame of a 30
    # fps clip — so the pose beneath it ran at twice the rate anything read.
    # 15 Hz is also the floor: the features measure a 0.7-4.0 Hz cadence band,
    # and 7.5 Hz would be below Nyquist for it.
    #
    # --no-fast-end reverts.

    ball_quiet_min_looks: int = 6
    # How many frames the quiet window must actually have LOOKED at the ball
    # before its silence counts.  Per-frame ball recall varies from 7% to 92%
    # across the corpus, so a thinly-sampled window is silent by sampling
    # noise rather than by evidence — and a false quiet ends the point early,
    # which is the point-end error that loses tennis rather than just footage.
    # Non-binding on the full pass (a 1.5 s window there holds ~45 looks) and
    # still clear at the fast path's 10 fps (~15).

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
