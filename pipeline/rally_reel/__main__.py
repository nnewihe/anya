"""CLI:  python -m pipeline.rally_reel match.mp4 [flags]"""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "pipeline.rally_reel"

from .config import ReelConfig
from .reel import build_reel


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m pipeline.rally_reel",
        description="Cut a tennis video down to its active rallies.")
    ap.add_argument("video", help="Input match video")
    ap.add_argument("-o", "--output", default=None,
                    help="Reel path (default: <stem>_rally_reel.mp4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write segments JSON and report, but do not cut video")
    ap.add_argument("--force-telemetry", action="store_true",
                    help="Re-run the perception pass even if cached")
    ap.add_argument("--device", default="mps", help="Torch device for the walking pass")

    ap.add_argument("--near-threshold", type=float, default=None,
                    help="Near-serve probability threshold (default 0.80, the "
                         "swept optimum for the fast path)")
    ap.add_argument("--no-fast-near", action="store_true",
                    help="Score near serves from the shared full-resolution "
                         "telemetry using the legacy P, instead of the cheaper "
                         "and more accurate anya_near_telemetry pass")
    ap.add_argument("--no-fast-far", action="store_true",
                    help="Detect far serves from the shared full-resolution "
                         "telemetry plus extract_far_pose, instead of the "
                         "cheaper and more accurate anya_far_telemetry pass")
    ap.add_argument("--no-fast-end", action="store_true",
                    help="Take point ends from a full-rate walking pose pass "
                         "plus the shared telemetry's ball stream, instead of "
                         "the cheaper anya_end_telemetry pass (~32.6 vs 12.84 "
                         "ms/frame).  The full pass truncates live points less "
                         "often (11 vs 13 over 135 labelled ends) but is worse "
                         "on pooled recall, precision and timing — and it is "
                         "what keeps stages 1-2 alive, so this flag costs far "
                         "more than the point-end difference alone")
    ap.add_argument("--no-near", action="store_true", help="Far-side serves only")
    ap.add_argument("--no-far", action="store_true", help="Near-side serves only")
    ap.add_argument("--pre-roll", type=float, default=None, help="Lead-in seconds")
    ap.add_argument("--post-roll", type=float, default=None, help="Tail seconds")
    ap.add_argument("--point-max", type=float, default=None,
                    help="Cap on point length when no walk interval is found")
    ap.add_argument("--end-policy",
                    choices=("walk-ball", "trace", "confidence", "energy",
                             "legacy"),
                    default=None,
                    help="How point ends read the dead-time signals: "
                         "'energy' (default) integrates in-court ball trace, "
                         "walking and player motion into a per-point energy "
                         "bar and ends where it drains; 'trace' ends on a fixed "
                         "window of IMM-tracked in-court ball silence; "
                         "'walk-ball' makes walking primary and lets a visible "
                         "ball veto it; 'confidence' ends where the dead-time "
                         "accumulator crosses threshold; 'legacy' is the old "
                         "union of walk onsets and gated ball-quiet")
    ap.add_argument("--segments-suffix", default=None,
                    help="Write the segments JSON under this suffix instead of "
                         "_rally_segments.json, so eval_point_end.py --arm can "
                         "compare two runs without copying files by hand")
    ap.add_argument("--trace-ball-fps", type=float, default=None)
    ap.add_argument("--trace-ball-imgsz", type=int, default=None)
    ap.add_argument("--trace-quiet", type=float, default=None,
                    help="Gap length that ends a point with NO walking")
    ap.add_argument("--trace-gap-walk", type=float, default=None,
                    help="Gap length that ends a point when walking begins in it")
    ap.add_argument("--trace-stamp", type=float, default=None,
                    help="End offset from the last trace")
    ap.add_argument("--trace-point-min", type=float, default=None,
                    help="Minimum point length under the trace policy")
    ap.add_argument("--trace-merge-gap", type=float, default=None,
                    help="Bridge trace gaps up to this long")
    ap.add_argument("--trace-walk-lead", type=float, default=None)
    ap.add_argument("--trace-court-pad-ft", type=float, default=None)
    ap.add_argument("--no-trace-court-gate", action="store_true",
                    help="Ablate the in-court gate on the trace policy")
    ap.add_argument("--energy-ball-weight", type=float, default=None,
                    help="Drain per second under full ball-trace silence")
    ap.add_argument("--energy-walk-boost", type=float, default=None,
                    help="How much walking multiplies that silence drain")
    ap.add_argument("--energy-motion-weight", type=float, default=None,
                    help="Recharge per second at full non-walking motion")
    ap.add_argument("--energy-reversal-weight", type=float, default=None,
                    help="Recharge per second at a saturating reversal rate")
    ap.add_argument("--energy-near-weight", type=float, default=None,
                    help="Drain per second while the near player is missing")
    ap.add_argument("--energy-hold", type=float, default=None,
                    help="Quiet period after the serve, bar frozen")
    ap.add_argument("--energy-stamp", type=float, default=None,
                    help="End offset from the start of the drain")
    ap.add_argument("--energy-step", type=float, default=None,
                    help="Integration step (resolution, not strength)")
    ap.add_argument("--energy-max-drop", type=float, default=None,
                    help="Cap on how fast the bar can fall, per second")
    ap.add_argument("--energy-max-rise", type=float, default=None,
                    help="Cap on how fast the bar can recover, per second")
    ap.add_argument("--energy-confirm", type=float, default=None,
                    help="Seconds the bar must sit at the floor to end a point")
    # The four near_end signals.  All ship at zero, so these flags are how an
    # arm gets turned on at all; see ReelConfig for what each one measures.
    ap.add_argument("--energy-settle-weight", type=float, default=None,
                    help="Near-player stillness as a multiplier on the ball-"
                         "silence drain (0 = off, the shipped default)")
    ap.add_argument("--energy-turn-away-weight", type=float, default=None,
                    help="Near player turning to face the camera rather than "
                         "the net, same units")
    ap.add_argument("--energy-stance-drop-weight", type=float, default=None,
                    help="Loss of the ready stance (hands down, legs straight), "
                         "same units")
    ap.add_argument("--energy-idle-hands-weight", type=float, default=None,
                    help="Between-point hand rituals — pocket, face/cap, hands "
                         "on hips — same units")
    ap.add_argument("--energy-near-signal-cap", type=float, default=None,
                    help="Ceiling on the four near-player signals combined")
    ap.add_argument("--no-energy-debug-rows", action="store_true",
                    help="Omit the per-step energy samples from the segments JSON")
    ap.add_argument("--walk-ball-veto", type=float, default=None,
                    help="walk-ball rule A: seconds the ball must be unseen "
                         "before an active walk ends the point (default 1.0)")
    ap.add_argument("--no-walk-quiet", type=float, default=None,
                    help="walk-ball rule B: seconds of ball silence that end "
                         "the point where no walking is detected (default 5.0)")
    ap.add_argument("--no-walk-stamp", type=float, default=None,
                    help="walk-ball rule B: where the end is placed, measured "
                         "from the last ball sighting (default 1.5; set equal "
                         "to --no-walk-quiet to end at the confirmation)")
    ap.add_argument("--ball-quiet-mode", choices=("off", "gated", "always"),
                    default=None,
                    help="Ball-quiet as a point-end signal: 'off' is walking "
                         "only (revert path), 'gated' (default) only where the "
                         "near player is untracked or stationary, 'always' "
                         "unconditionally")
    ap.add_argument("--min-service-run", type=int, default=None,
                    help="Min consecutive points served by one side (default 8; "
                         "4 tracks the tennis rule more closely)")
    ap.add_argument("--no-service-runs", action="store_true",
                    help="Disable the service-run constraint entirely")
    ap.add_argument("--drop-side-conflicts", action="store_true",
                    help="Discard starts that disagree with the inferred serving side")
    args = ap.parse_args(argv)

    cfg = ReelConfig()
    if args.near_threshold is not None:
        cfg.near_threshold = args.near_threshold
    if args.no_near:
        cfg.use_near = False
    if args.no_far:
        cfg.use_far = False
    if args.pre_roll is not None:
        cfg.pre_roll_s = args.pre_roll
    if args.post_roll is not None:
        cfg.post_roll_s = args.post_roll
    if args.point_max is not None:
        cfg.point_max_s = args.point_max
    if args.ball_quiet_mode is not None:
        cfg.ball_quiet_mode = args.ball_quiet_mode
    if args.end_policy is not None:
        cfg.end_policy = args.end_policy
    for arg, field in (("trace_ball_fps", "trace_ball_fps"),
                       ("trace_ball_imgsz", "trace_ball_imgsz"),
                       ("trace_quiet", "trace_quiet_s"),
                       ("trace_gap_walk", "trace_gap_walk_s"),
                       ("trace_stamp", "trace_stamp_s"),
                       ("trace_point_min", "trace_point_min_s"),
                       ("trace_merge_gap", "trace_merge_gap_s"),
                       ("trace_walk_lead", "trace_walk_lead_s"),
                       ("trace_court_pad_ft", "trace_court_pad_ft"),
                       ("energy_ball_weight", "energy_ball_weight"),
                       ("energy_walk_boost", "energy_walk_boost"),
                       ("energy_motion_weight", "energy_motion_weight"),
                       ("energy_reversal_weight", "energy_reversal_weight"),
                       ("energy_near_weight", "energy_near_missing_weight"),
                       ("energy_hold", "energy_hold_s"),
                       ("energy_stamp", "energy_stamp_s"),
                       ("energy_step", "energy_step_s"),
                       ("energy_confirm", "energy_confirm_s"),
                       ("energy_max_drop", "energy_max_drop_per_s"),
                       ("energy_max_rise", "energy_max_rise_per_s"),
                       ("energy_settle_weight", "energy_settle_weight"),
                       ("energy_turn_away_weight", "energy_turn_away_weight"),
                       ("energy_stance_drop_weight", "energy_stance_drop_weight"),
                       ("energy_idle_hands_weight", "energy_idle_hands_weight"),
                       ("energy_near_signal_cap", "energy_near_signal_cap")):
        # setattr on a dataclass instance accepts anything, so a stale mapping
        # would be silently inert — the worst failure mode for an A/B arm.
        if not hasattr(cfg, field):
            ap.error(f"internal: ReelConfig has no field {field!r}")
        v = getattr(args, arg, None)
        if v is not None:
            setattr(cfg, field, v)
    if args.no_energy_debug_rows:
        cfg.energy_debug_rows = False
    if args.no_trace_court_gate:
        cfg.trace_court_gate = False
    if args.walk_ball_veto is not None:
        cfg.walk_ball_veto_s = args.walk_ball_veto
    if args.no_walk_quiet is not None:
        cfg.no_walk_quiet_s = args.no_walk_quiet
    if args.no_walk_stamp is not None:
        cfg.no_walk_stamp_s = args.no_walk_stamp
    if args.min_service_run is not None:
        cfg.min_service_run = args.min_service_run
    if args.no_service_runs:
        cfg.enforce_service_runs = False
    if args.drop_side_conflicts:
        cfg.drop_side_conflicts = True
    if args.no_fast_near:
        cfg.fast_near = False
    if args.no_fast_far:
        cfg.fast_far = False
    if args.no_fast_end:
        cfg.fast_end = False

    if not (cfg.use_near or cfg.use_far):
        ap.error("--no-near and --no-far together leave no serve detector")

    segments, out = build_reel(
        args.video, cfg=cfg, output_path=args.output,
        force_telemetry=args.force_telemetry, device=args.device,
        dry_run=args.dry_run, segments_suffix=args.segments_suffix,
    )
    for s in segments:
        print(f"  [{s.index:02d}] {s.side:>4}  {s.start:7.2f}s - {s.end:7.2f}s "
              f"({s.end - s.start:5.1f}s, end={s.end_method})")
    if out:
        print(f"[REEL] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
