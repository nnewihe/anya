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
    ap.add_argument("--fast-end", action="store_true",
                    help="Take point ends from the cheaper anya_end_telemetry "
                         "pass (12.84 vs ~32.6 ms/frame).  Off by default: "
                         "pooled recall/precision/timing improve but it "
                         "truncates live points more often (13 vs 11 over 135 "
                         "labelled ends)")
    ap.add_argument("--no-near", action="store_true", help="Far-side serves only")
    ap.add_argument("--no-far", action="store_true", help="Near-side serves only")
    ap.add_argument("--pre-roll", type=float, default=None, help="Lead-in seconds")
    ap.add_argument("--post-roll", type=float, default=None, help="Tail seconds")
    ap.add_argument("--point-max", type=float, default=None,
                    help="Cap on point length when no walk interval is found")
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
    if args.fast_end:
        cfg.fast_end = True

    if not (cfg.use_near or cfg.use_far):
        ap.error("--no-near and --no-far together leave no serve detector")

    segments, out = build_reel(
        args.video, cfg=cfg, output_path=args.output,
        force_telemetry=args.force_telemetry, device=args.device,
        dry_run=args.dry_run,
    )
    for s in segments:
        print(f"  [{s.index:02d}] {s.side:>4}  {s.start:7.2f}s - {s.end:7.2f}s "
              f"({s.end - s.start:5.1f}s, end={s.end_method})")
    if out:
        print(f"[REEL] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
