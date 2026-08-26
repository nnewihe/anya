"""
far_toss_ball.py
================
See the toss BALL above the far player's hand, at native resolution, with
SAHI-style tiled inference.

STATUS: it works, it is measured, and it is OFF BY DEFAULT because it does not
currently pay for itself.  Read the numbers below before enabling it.

Two bugs made this look impossible at first
-------------------------------------------
An earlier attempt concluded the ball was undetectable at far-court range --
median 0 to 0.5 detections per serve, and one setting where false positives
scored HIGHER than true ones.  Both readings were artefacts of aiming:

  THE WINDOW WAS BEFORE THE TOSS.  `far_serve`'s event time is lead-corrected
  to ~0.9 s BEFORE the trophy, so a window of [t-0.2, t+1.0] covers
  trophy-1.1s to trophy+0.1s -- while the ball is still in the hand.  The ball
  is airborne from about trophy+0.0 to +0.6.

  THE ROI WAS ABOVE THE HEAD.  At this range the toss does not rise far above
  the head in IMAGE terms; it sits just above the tossing hand.  An ROI
  reaching 2.4 body heights above the box top put the ball at the very bottom
  of a mostly empty crop.

Re-aimed at the TOSSING WRIST and anchored on the TROPHY, the ball is plainly
visible -- a bright dot climbing away from the hand over consecutive frames --
in a crop 130-200 px wide at source resolution.

What fixed detection
--------------------
    head-centred ROI, wrong window      0-0.5 frames with a ball, per serve
    wrist-centred ROI, single shot      5.0
    wrist-centred ROI + SAHI tiling     8.0

SAHI earns its place: 160 px tiles at 30% overlap, each run at imgsz 320, merged
by distance NMS.  Native-resolution crops throughout -- nothing is downscaled.

Why it is off anyway
--------------------
BALL PRESENCE DOES NOT DISCRIMINATE.  A returner also has a ball near them --
the one they are about to hit.  Median frames-with-ball is 8.0 on true serves
against 6.0 on false ones, and "at least three frames with a ball" is 71% true
against 83% FALSE.  Counting the ball is useless.

THE ARC DOES, BUT ONLY THROUGH DIRECTION.  Of three trajectory features, height
says nothing and motion says a lot:

    total rise of the highest ball        AUC 49%
    peak height above the wrist          AUC 45%
    fraction of time the ball is CLIMBING AUC 73-75%

That is the physics: a toss rises away from the hand, an incoming ball does not.

AND IT IS REDUNDANT WITH THE POSE TOSS.  On the same 19 true and 12 false
sequences the ball arc scores AUC 75% and `far_serve.toss_score` scores 83%,
they correlate at +0.49, and the best blend (0.3 arc + 0.7 pose) reaches 83% --
NO BETTER THAN POSE ALONE.  Both are watching the same event; the arm is simply
easier to see than the ball.

So a pass costing a tiled ball inference over ~18 native-resolution frames per
candidate, with a video seek each, buys nothing over a signal that is already
computed for free.  Enable it only if that changes -- a closer camera, a better
small-object model, or a clip where far pose degrades.

The sample is small (19/12), so this is a reason not to spend the compute, not
proof the ball can never help.
"""

import numpy as np

from pipeline.anya2 import court as C
from pipeline.anya2 import signals as S

TILE_PX = 160
TILE_OVERLAP = 0.30
TILE_IMGSZ = 320
BALL_CONF = 0.05
NMS_R_PX = 6.0

WIN_FROM_S, WIN_TO_S = -0.2, 1.0    # anchored on the TROPHY, not the event time
ROI_W_BH, ROI_UP_BH, ROI_DOWN_BH = 1.1, 1.8, 0.7
KP_CONF = 0.15


def tiles(img, tile=TILE_PX, ov=TILE_OVERLAP):
    """SAHI-style overlapping slices as [(sub, ox, oy)]."""
    h, w = img.shape[:2]
    if h <= tile and w <= tile:
        return [(img, 0, 0)]
    st = max(16, int(tile * (1 - ov)))
    ys = list(range(0, max(1, h - tile + 1), st)) or [0]
    xs = list(range(0, max(1, w - tile + 1), st)) or [0]
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    return [(img[y:y + tile, x:x + tile], x, y) for y in ys for x in xs]


def detect_tiled(model, img, conf=BALL_CONF):
    """Ball centres in `img` coordinates, via tiled inference at native scale."""
    subs = tiles(img)
    res = model.predict([s for s, _, _ in subs], imgsz=TILE_IMGSZ,
                        conf=conf, verbose=False)
    out = []
    for (_, ox, oy), r in zip(subs, res):
        if r.boxes is None or not len(r.boxes):
            continue
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append(((b[0] + b[2]) / 2 + ox, (b[1] + b[3]) / 2 + oy, float(c)))
    out.sort(key=lambda t: -t[2])
    keep = []
    for x, y, c in out:
        if all((x - a) ** 2 + (y - b) ** 2 > NMS_R_PX ** 2 for a, b, _ in keep):
            keep.append((x, y, c))
    return keep


def arc_score(track: np.ndarray) -> float:
    """[0, 1] from a toss track of (dt, height_above_wrist_in_body_heights).

    Only the CLIMBING FRACTION is read.  Height and total rise were measured at
    AUC 45-49% -- chance -- while the share of steps where the highest ball is
    rising separates at 73-75%.  A toss goes up; a ball arriving to be hit does
    not.
    """
    if track is None or len(track) < 3:
        return 0.0
    ts = np.asarray(sorted(set(track[:, 0])))
    top = np.array([np.max(track[track[:, 0] == t][:, 1]) for t in ts])
    if len(top) < 3:
        return 0.0
    climbing = float(np.mean(np.diff(top) > 0))
    # 0.45 is the false-positive median, 0.64 the true-serve median.
    return float(S.ramp(climbing, 0.45, 0.64))
