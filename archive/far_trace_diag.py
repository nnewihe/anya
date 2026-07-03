"""
far_trace_diag.py
=================
Headless far-side diagnostic for the rally trace tracker.

Mirrors rally_detector's setup (AnyaTelemetryProvider forced ACTIVE + the same
BallTrackManager) but, instead of cutting clips, it quantifies WHY the trace is
lost on the far side by splitting each far-side frame into one of two failure
modes:

  • detection loss — no far-side ball was detected at all this frame (YOLO miss).
  • tracking loss  — a far-side ball WAS detected but no live trace covers it
                     (the tracker dropped / drifted off it).

"Far side" is court-aware: a point is far when its world_y (via the telemetry
homography) is past mid-court (COURT_LENGTH_FT / 2 = 39 ft).  This is the same
geometry the ready-position gate and player classifier use, so the split matches
how the rest of the pipeline reasons about sides.

Supports one or more frame windows (e.g. a set of far-side points).  Each window
gets a fresh tracker so a stale track can't leak across the seek gap; per-window
results plus an aggregate verdict are printed at the end.

Usage:
    python far_trace_diag.py /Volumes/Anya/Data/63/match.mp4 --windows 36837-37288 39076-39338
    python far_trace_diag.py match.mp4 --start-frame 9000 --max-frames 3000
"""

import argparse

import cv2

from anya_base import AnyaTelemetryProvider
from ball_tracker import BallTrackManager, make_image_row_perspective
from utilities import Config


FAR_WORLD_Y_MIN = Config.COURT_LENGTH_FT / 2.0   # 39 ft — past mid-court = far half


def _pct(n, d):
    return f"{(100.0 * n / d):.1f}%" if d else "n/a"


def diag_window(cap, provider, tracker, start_frame, end_frame):
    """
    Analyse one [start_frame, end_frame] window and return a stats dict.

    Assumes `cap` is positioned at start_frame and `tracker` is freshly reset.
    Reads exactly (end_frame - start_frame + 1) frames.
    """
    far_run_gap = int(provider.fps * 2.0)   # 2 s memory of a far ball

    st = {
        "frames": 0,
        "far_det_total": 0,
        "far_det_frames": 0,
        "far_alive_frames": 0,
        "track_loss": 0,
        "det_loss": 0,
        "state_hist": {},
    }
    last_far_det_frame = None

    frame_num = start_frame - 1
    while cap.isOpened() and frame_num < end_frame:
        ok, orig = cap.read()
        if not ok:
            break
        frame_num += 1
        st["frames"] += 1

        frame = cv2.resize(orig, (960, 540), interpolation=cv2.INTER_LINEAR)
        tel   = provider.process_frame(frame)

        cands = tel.active_ball_candidates or []
        dets  = [(c["pixel_center"][0], c["pixel_center"][1], c["conf"]) for c in cands]

        far_dets = 0
        for (px, py, _c) in dets:
            _wx, wy = provider.get_world_pos(px, py)
            if wy > FAR_WORLD_Y_MIN:
                far_dets += 1
        st["far_det_total"] += far_dets
        has_far_det = far_dets > 0
        if has_far_det:
            st["far_det_frames"] += 1
            last_far_det_frame = frame_num

        status = tracker.update(dets, tel.timestamp)

        trace_far_alive = False
        if status.has_moving_trace and status.position is not None:
            _wx, wy = provider.get_world_pos(status.position[0], status.position[1])
            trace_far_alive = wy > FAR_WORLD_Y_MIN
        if trace_far_alive:
            st["far_alive_frames"] += 1

        in_far_run = (last_far_det_frame is not None and
                      frame_num - last_far_det_frame <= far_run_gap)
        if has_far_det and not trace_far_alive:
            st["track_loss"] += 1
            st["state_hist"][status.state] = st["state_hist"].get(status.state, 0) + 1
        elif in_far_run and not has_far_det and not trace_far_alive:
            st["det_loss"] += 1

    return st


def run_diag(video_path, windows):
    provider = AnyaTelemetryProvider(video_path)
    provider.update_state("ACTIVE")

    cap = cv2.VideoCapture(video_path)

    per_window = []
    for (s, e) in windows:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        tracker = BallTrackManager(
            fps=provider.fps,
            perspective_scale=make_image_row_perspective(540),
        )
        print(f"[diag] window {s}-{e} …", flush=True)
        st = diag_window(cap, provider, tracker, s, e)
        per_window.append((s, e, st))

    cap.release()

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg = {"frames": 0, "far_det_total": 0, "far_det_frames": 0,
           "far_alive_frames": 0, "track_loss": 0, "det_loss": 0, "state_hist": {}}
    for _s, _e, st in per_window:
        for k in ("frames", "far_det_total", "far_det_frames",
                  "far_alive_frames", "track_loss", "det_loss"):
            agg[k] += st[k]
        for k, v in st["state_hist"].items():
            agg["state_hist"][k] = agg["state_hist"].get(k, 0) + v

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  FAR-SIDE TRACE DIAGNOSTIC — {video_path}")
    print("=" * 78)
    hdr = (f"  {'window':>16}  {'frames':>7}  {'far-det%':>8}  "
           f"{'far-alive%':>10}  {'trk-loss%':>9}  {'det-loss':>8}")
    print(hdr)
    print("  " + "-" * 74)
    for s, e, st in per_window:
        print(f"  {f'{s}-{e}':>16}  {st['frames']:>7}  "
              f"{_pct(st['far_det_frames'], st['frames']):>8}  "
              f"{_pct(st['far_alive_frames'], st['frames']):>10}  "
              f"{_pct(st['track_loss'], st['far_det_frames']):>9}  "
              f"{st['det_loss']:>8}")
    print("  " + "-" * 74)
    print(f"  {'AGGREGATE':>16}  {agg['frames']:>7}  "
          f"{_pct(agg['far_det_frames'], agg['frames']):>8}  "
          f"{_pct(agg['far_alive_frames'], agg['frames']):>10}  "
          f"{_pct(agg['track_loss'], agg['far_det_frames']):>9}  "
          f"{agg['det_loss']:>8}")
    print("=" * 78)

    print("\n  Column meanings:")
    print("    far-det%    : frames with ≥1 far-side ball detection (YOLO saw a far ball)")
    print("    far-alive%  : frames the live trace sits on the far side")
    print("    trk-loss%   : of far-det frames, how many had NO live far trace (TRACKING loss)")
    print("    det-loss    : frames inside a far run with NO far detection (DETECTION loss)")

    print("\n  Tracker state on TRACKING-loss frames (aggregate):")
    if agg["state_hist"]:
        for stt, n in sorted(agg["state_hist"].items(), key=lambda kv: -kv[1]):
            print(f"      {stt:>10}: {n}  ({_pct(n, agg['track_loss'])})")
    else:
        print("      (none)")

    # ── Verdict ───────────────────────────────────────────────────────────
    print("\n  " + "-" * 74)
    if agg["far_det_frames"] == 0:
        verdict = "No far-side detections at all → pure DETECTION problem (YOLO)."
    else:
        track_rate = agg["track_loss"] / agg["far_det_frames"]
        det_share  = agg["det_loss"]
        if track_rate > 0.4:
            verdict = ("Far balls ARE detected but are frequently uncovered by a live "
                       "trace → TRACKING-dominated. Tune the tracker knobs.")
        elif det_share > agg["track_loss"]:
            verdict = ("Far balls are mostly tracked when detected; losses are dominated "
                       "by missing detections → DETECTION-dominated (YOLO / conf / imgsz).")
        else:
            verdict = ("Mixed: both detection gaps and tracking dropouts contribute; "
                       "address detection first, then re-tune the tracker.")
    print(f"  VERDICT: {verdict}")
    print("  " + "=" * 74)


def _parse_window(tok):
    a, b = tok.split("-")
    return (int(a), int(b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Headless far-side trace diagnostic")
    ap.add_argument("video", help="Input tennis video (e.g. match.mp4)")
    ap.add_argument("--windows", nargs="+", metavar="START-END",
                    help="One or more frame windows, e.g. 36837-37288 39076-39338")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Single-window convenience: analyse start-frame .. start+max")
    args = ap.parse_args()

    if args.windows:
        wins = [_parse_window(w) for w in args.windows]
    else:
        s = args.start_frame
        e = (s + args.max_frames) if args.max_frames else (s + 3000)
        wins = [(s, e)]

    run_diag(args.video, wins)
