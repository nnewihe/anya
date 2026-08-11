"""
select_near.py
==============
Pick the near-side player out of the cached person detections, frame by frame.

The clip is not just a rally camera: players leave the court entirely to pick up
balls by the wall carts, and they change ends, so "near player" is neither a
fixed human nor always inside the court. A pure court-region gate loses a third
of the clip (and loses it in a label-correlated way — the walks off court are
exactly the walking label). A pure continuity tracker follows the wrong human
through a changeover. Selection therefore combines:

  zone      — a foot point projected into the far half of the court is the
              opponent and is never eligible; the near half is preferred; the
              surrounding floor (ball carts, walkways) stays eligible so a
              player who walks off court keeps being tracked.
  size      — near-side people are large in this camera.
  continuity— a bonus for being where the track was, which resolves ties among
              several people standing together at the carts.

Output ``<clip>/<stem>_walk_pose.npz``:
    kp    [N, 17, 3]  near-player keypoints, 960x540 pixels (NaN = no player)
    bbox  [N, 4]      near-player box xyxy
    on_court [N]      1.0 when the selected foot point is inside the near half
"""

import argparse
import os

import numpy as np

from walking.court import COURT_L, COURT_W, HALF_L, load_homography, to_court
from walking.extract_pose import dets_path

L_ANK, R_ANK = 15, 16
KP_CONF = 0.3
MIN_H = 25.0             # px; smaller than this is a spectator on another court
FAR_Y = 12.5             # court metres; beyond this is the opponent's half
NEAR_MARGIN = 6.0        # metres of slack around the near half for wide play
LOST_FRAMES = 60         # keep using the last position as a continuity anchor
MAX_SPEED = 9.0          # m/s; faster than a sprint means it is a different person


def pose_path(video_path):
    d = os.path.dirname(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}_walk_pose.npz")


def _foot(kp, box):
    la, ra = kp[L_ANK], kp[R_ANK]
    if la[2] > KP_CONF and ra[2] > KP_CONF:
        return 0.5 * (la[0] + ra[0]), 0.5 * (la[1] + ra[1])
    if la[2] > KP_CONF:
        return float(la[0]), float(la[1])
    if ra[2] > KP_CONF:
        return float(ra[0]), float(ra[1])
    return 0.5 * (box[0] + box[2]), float(box[3])


def _zone(cx, cy):
    """'far' (ineligible), 'near' (preferred), or 'off' (eligible, no bonus)."""
    on_court_x = -4.0 <= cx <= COURT_W + 4.0
    if on_court_x and FAR_Y <= cy <= COURT_L + 8.0:
        return "far"
    if -NEAR_MARGIN <= cy < FAR_Y and -NEAR_MARGIN <= cx <= COURT_W + NEAR_MARGIN:
        return "near"
    return "off"


def select(video, dets_npz=None, out=None, verbose=True):
    dets_npz = dets_npz or dets_path(video)
    z = np.load(dets_npz)
    kp_all, box_all, conf_all, fps = z["kp"], z["box"], z["conf"], float(z["fps"])
    n, k = conf_all.shape
    H = load_homography(video)

    kp_out = np.full((n, 17, 3), np.nan, dtype=np.float32)
    bb_out = np.full((n, 4), np.nan, dtype=np.float32)
    on_out = np.full(n, np.nan, dtype=np.float32)

    prev_court, since = None, 10 ** 9
    n_near = n_off = 0
    for f in range(n):
        best, best_s = -1, -np.inf
        best_zone, best_court = None, None
        for i in range(k):
            if not np.isfinite(conf_all[f, i]):
                continue
            box = box_all[f, i]
            h = float(box[3] - box[1])
            if h < MIN_H:
                continue
            fx, fy = _foot(kp_all[f, i], box)
            cx, cy = to_court(H, [[fx, fy]])[0]
            zone = _zone(cx, cy)
            if zone == "far":
                continue
            s = 0.6 * float(conf_all[f, i]) + min(h, 250.0) / 250.0
            if zone == "near":
                s += 1.2
            # Continuity is measured in court metres, not pixels: the same
            # pixel gap means very different distances near and far from the
            # camera, and it was pixel-space continuity that let the track flip
            # between the player and a bystander at the ball carts.
            if prev_court is not None and since <= LOST_FRAMES:
                d = float(np.hypot(cx - prev_court[0], cy - prev_court[1]))
                if d / max(since, 1) * fps > MAX_SPEED:
                    # Nobody covers 9 m/s: this is a different human. Reject it
                    # outright rather than scoring it down — on this clip the
                    # detector often finds only ONE of the two people present,
                    # so a penalised-but-still-best candidate silently becomes
                    # the track and the trace teleports. A missing frame is
                    # honest; a wrong frame poisons every window that spans it.
                    continue
                s += 2.0 * np.exp(-d / 2.0)
            if s > best_s:
                best_s, best, best_zone, best_court = s, i, zone, (cx, cy)

        if best < 0:
            since += 1          # keep the anchor; re-acquire after LOST_FRAMES
            continue
        box = box_all[f, best]
        kp_out[f] = kp_all[f, best]
        bb_out[f] = box
        on_out[f] = 1.0 if best_zone == "near" else 0.0
        n_near += best_zone == "near"
        n_off += best_zone == "off"
        prev_court = best_court
        since = 1

    out = out or pose_path(video)
    # `fps` is the rate of THESE rows, which is the source rate only when the
    # detections were extracted every frame.  A decimated pass (pipeline/
    # anya_end_telemetry) writes its effective rate as fps and carries the
    # stride alongside, so everything here — the MAX_SPEED gate above included
    # — works on a consistent timeline, and only the mapping back to source
    # frame numbers needs the stride.
    extra = {k: z[k] for k in ("stride", "src_fps", "n_src_frames") if k in z}
    np.savez_compressed(out, kp=kp_out, bbox=bb_out, on_court=on_out,
                        fps=np.float64(fps), **extra)
    if verbose:
        cov = float(np.mean(np.isfinite(bb_out[:, 0])))
        print(f"[select] {out}: coverage {cov:.1%} "
              f"(near-half {n_near / n:.1%}, off-court {n_off / n:.1%})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--dets", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    select(a.video, a.dets, a.out)


if __name__ == "__main__":
    main()
