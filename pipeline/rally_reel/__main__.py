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
                    choices=("walk-ball", "trace", "confidence", "legacy"),
                    default=None,
                    help="How point ends combine the two dead-time signals: "
                         "'walk-ball' (default) makes walking primary and lets "
                         "a visible ball veto it, with ball silence alone "
                         "ending the point where walking is silent; 'legacy' "
                         "is the old union of walk onsets and gated "
                         "ball-quiet; 'trace' ends on IMM-tracked in-court ball "
                         "trace instead of ball presence; 'confidence' ends "
                         "where the dead-time accumulator crosses threshold")
    ap.add_argument("--segments-suffix", default=None,
                    help="Write the segments JSON under this suffix instead of "
                         "_rally_segments.json, so eval_point_end.py --arm can "
                         "compare two runs without copying files by hand")
    ap.add_argument("--trace-ball-fps", type=float, default=None)
    ap.add_argument("--trace-walk-confirm", type=float, default=None)
    ap.add_argument("--trace-walk-stamp", type=float, default=None)
    ap.add_argument("--trace-quiet", type=float, default=None)
    ap.add_argument("--trace-quiet-stamp", type=float, default=None)
    ap.add_argument("--trace-court-pad-ft", type=float, default=None)
    ap.add_argument("--no-trace-court-gate", action="store_true",
                    help="Ablate the in-court gate on the trace policy")
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
                       ("trace_walk_confirm", "trace_walk_confirm_s"),
                       ("trace_walk_stamp", "trace_walk_stamp_s"),
                       ("trace_quiet", "trace_quiet_s"),
                       ("trace_quiet_stamp", "trace_quiet_stamp_s"),
                       ("trace_court_pad_ft", "trace_court_pad_ft")):
        v = getattr(args, arg, None)
        if v is not None:
            setattr(cfg, field, v)
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
