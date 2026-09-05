"""
tracks.py
=========
Turn the all-persons pose pass into at most TWO near-side and TWO far-side
player tracks, with a per-frame eligibility flag.

This generalizes `walking.select_near`, which picks exactly one near player and
nothing else.  Two of its findings are load-bearing and are kept verbatim in
spirit:

  CONTINUITY IS SCORED IN COURT METRES, NOT PIXELS.  The same pixel gap is a
  different real distance near and far from the camera, and it was pixel-space
  continuity that let a track flip between the player and a bystander at the
  ball carts.

  A CANDIDATE IMPLYING MORE THAN 9 m/s IS REJECTED OUTRIGHT, not penalised.  A
  penalised-but-still-best candidate silently becomes the track and the trace
  teleports.  A missing frame is honest; a wrong frame poisons every window that
  spans it.

Tracking zone vs eligibility gate -- a deliberate split
------------------------------------------------------
`select_near` refuses to gate on the court on purpose: players walk off court to
the ball carts and change ends, and a court-region gate loses exactly the frames
that carry walking labels.  The user's rule for this redesign is the opposite --
a player's box x-centre must be inside the doubles court plus 3 ft.

Both are right, about different things, so they are separated:

  the TRACKING zone is loose (`X_TRACK_PAD`, `NEAR_BACK_M`, `FAR_BACK_M`).  A
      player who walks to the carts stays on the same slot, so continuity
      survives the trip and the track does not have to be re-acquired.

  `eligible[n, slot]` is the user's STRICT gate (`court.in_bounds`), stored per
      frame per slot.  Detectors read it to decide who is allowed to be serving
      or ending a point.

So a player off court is still TRACKED, just not ELIGIBLE.  Collapsing the two
would mean losing the track and then mis-reacquiring it, which is a worse error
than carrying an ineligible frame.

Slot layout
-----------
    0, 1  near side        2, 3  far side
Singles is not a special case -- it is the case where slots 1 and 3 are always
NaN.  Nothing downstream may assume a slot count.

Output `<stem>_anya2_tracks.npz`:
    kp       [N, 4, 17, 3]  keypoints in ANALYSIS_SIZE pixels (NaN = no player)
    bbox     [N, 4, 4]      xyxy
    court    [N, 4, 2]      ground-point court metres
    eligible [N, 4]         bool, the strict doubles+3ft lateral gate
    side     [4]            'near','near','far','far'
    fps, stride, src_fps, n_src_frames
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline.anya2 import court as C
from pipeline import workdir as WD

N_KP = 17
N_SLOTS = 4
NEAR_SLOTS = (0, 1)
FAR_SLOTS = (2, 3)
SIDE_OF_SLOT = np.array([C.NEAR, C.NEAR, C.FAR, C.FAR], dtype=object)

# ── tracking admissibility (loose; see the module docstring) ─────────────
MIN_H_PX = 12.0        # below this is a spectator on another court, or noise.
                       # Lower than select_near's 25: that threshold was tuned
                       # for the NEAR player, who is 100-300 px tall here, and
                       # it deletes the far player outright -- measured on
                       # Data/21, a far player at the far baseline is ~20-40 px
                       # in the 960x540 analysis frame.
X_TRACK_PAD = 6.0      # metres outside the singles lines that a tracked person
                       # may wander (ball carts, walkways, the bench)
NEAR_BACK_M = 6.0      # ...and behind the near baseline
# ...and behind the far one.  12 m is not "8 m plus slack" -- it is sized from
# the MEASURED projection error at far-court depth, and it is large because that
# error is large and mostly systematic.  A far player's ground point is 22-32 px
# up the frame, where two pixels of box-bottom error is metres of court, and the
# box bottom sits above the true feet more often than below.  Across Data/23, 24
# and 25 (34k far-band detections) the projected court_y of a player standing ON
# the far baseline exceeds it by a MEDIAN of 4 m, with p90 at 5.7 and p99 at
# 12.8.  At 8 m the bound was silently deleting a third of clip 23's far
# detections -- real players, rejected for standing where the homography said
# they were.  12 m keeps 98.7%.
#
# The looseness costs less than it looks: far detections come from the far-band
# crop, which already bounds the region physically, and the lateral doubles+3ft
# gate still applies.  This bound is a backstop against a spectator high in the
# frame, not the thing deciding who is a player.
FAR_BACK_M = 12.0

# ── height must agree with depth ─────────────────────────────────────────
# A person standing on the court has a PREDICTABLE pixel height: the court's own
# width at their depth gives px-per-metre there, and a player is ~1.75 m.  The
# ratio of observed to expected height is tight -- across four clips and both
# sides its median is 0.82-1.06 and its p90 never exceeds 1.24.
#
# A candidate whose box is far larger than its projected depth allows is not a
# player at that depth; it is a CLOSER player whose ground point has been
# mis-projected.  Found by rendering a frame: on Data/21 at t=60.3 s the near
# player, having moved forward, appeared inside the far-band crop and was
# tracked as a FAR player with a box of 69 px where the far baseline allows 26.
#
# The bound is one-sided.  A box SMALLER than expected is ordinary -- a crouched
# player, a partial detection, a crop clipping the head -- and rejecting those
# would cost real tracking.  Only "too big for where you claim to be standing"
# is a contradiction.
HEIGHT_MAX_RATIO = 1.40    # observed / expected; p90 is 1.24
PLAYER_HEIGHT_M = 1.75

# ── the far band's lower edge, enforced a second time ────────────────────
# `perceive._pose_pass` already refuses far-band detections whose box bottom
# sits ON the crop's lower edge, for the reason given there: the box bottom is
# the CROP BOUNDARY and not the person's feet, so projecting it through a
# ground-plane homography invents a place they never stood.
#
# THAT FILTER WAS TOO TIGHT BY ABOUT ONE PIXEL, and it cost six false far
# serves.  Its `edge_px` is 2.0, and the near SERVER -- who at the trophy
# extends up into the band and is cut by it -- lands just outside:
#
#     clip 22 @108.1s   box bottom 2.7 px from the edge   (cutoff 2.0)
#     clip 22 @111.3s                 4.0 px
#     clip 22 @165.3s                 2.5 px
#     clip 22 @196.1s                 2.7 px
#     clip 35 @139.0s                 2.6 px
#     clip 35 @153.5s                 2.5 px
#
# Every one clears the cutoff by 0.5 to 2.0 px.  A detector does not put a box
# edge exactly on the image boundary; it lands a few pixels inside, so a 2 px
# tolerance tests for something that does not happen.
#
# The consequence is not a slightly wrong position.  The cut box projects to
# court_y 16-19 m, `is_near` calls it FAR, and the fragment -- a head with a
# wrist above it -- is precisely the shape `far_serve` hunts.  The near
# player's own serve is scored as a far serve, twice per clip.
#
# `_height_plausible` cannot catch it either: it is one-sided by design and
# rejects only boxes TOO BIG for their depth, while truncation makes the box
# too SMALL -- 51 px where the intact player is 200 -- which is exactly what a
# real player at 18 m looks like.  Truncation is the one failure mode that
# defeats a one-sided height test.
#
# So the test is repeated here, wider.  It lives in `tracks` rather than as a
# raised `edge_px` because changing the perceive-side constant invalidates
# every cached far pose pass -- the most expensive stage in the pipeline --
# for a bound that wants to be measured against results.
#
# SWEPT over the twelve snippets, far serve at the shipped thresholds:
#
#     edge_px   hits/91   fp   recall  precision     F1
#       2.0       87      18    95.6%     82.9%     88.8   <- perceive's value
#       3.0       87      12    95.6%     87.9%     91.6   <- here
#       3.5       87      14    95.6%     86.1%     90.6
#       4.0       86      13    94.5%     86.9%     90.5
#       5.0       86      13    94.5%     86.9%     90.5
#       6.0       85      13    93.4%     86.7%     89.9
#
# 3.0 keeps every serve and removes a third of the false positives.  Clip 22
# goes from five false positives to two, clip 35 from four to one.
#
# IT DOES NOT RISE MONOTONICALLY, and that is worth knowing before anyone
# retunes it: 3.5 scores WORSE than 3.0.  Dropping a far-band candidate frees a
# slot, the freed slot is filled by somebody else, and that person brings their
# own detections.  Past 3.0 the number is being decided by slot-assignment
# churn rather than by the truncation criterion, so it should not be pushed
# further on the strength of this table alone.
#
# The cost above 3.0 is real and has a cause: the band is grown UPWARD from a
# ground strip about the far baseline, so a far player standing INSIDE the
# baseline has their feet near the band's lower edge too.  At 6.0 that deletes
# two of clip 23's labelled serves.  This bound separates a cut box from a
# player standing near the front of the band, and those two are only a few
# pixels apart.
#
# NOT A FAR-SIDE-ONLY CHANGE, measured on the same twelve clips:
#     near serve   88.9% / 86.2%  ->  unchanged, to the detection
#     point end    59.1% / 50.3%  ->  58.3% / 54.0%, truncations 0 either way
# Two clips (26, 50) see their NEAR slots change, because the band's lower edge
# legitimately catches a near player at the net and those detections are now
# dropped -- correctly, since a cut box has no ground point whichever side it
# lands on, and the whole-frame pass sees that player intact anyway.
FAR_EDGE_PX = 3.0          # source pixels; the six mis-projections are 2.5-4.0

# ── the box bottom must actually BE the feet ─────────────────────────────
# FAR_EDGE_PX tests one CAUSE of a box with no feet in it.  This tests the
# thing itself, and so catches the same failure however it arose -- the crop
# edge, the frame edge, an occluding player, the net post.
#
# The projection's whole assumption is that the box bottom is where the person
# meets the ground.  A pose model tells us directly whether it is: if it found
# no knee and no ankle, there is no lower body in the box and its bottom edge
# is wherever the person was cut off.  Measured at the trophy, over the six
# known mis-projections and 114 true far serves:
#
#     rule                                    rejects fragments   real far
#     no confident ankle                             83%             2%
#     no confident knee                              83%             1%
#     no ankle AND no knee                           83%             1%
#     fewer than 12 confident keypoints              33%             1%
#     head sits below 0.25 of box height             50%            13%
#     head below the hips (an upside-down box)        0%             0%
#
# THE ANATOMICAL ORDERING RULES DO NOT FIRE.  A box "with the head at the
# bottom" sounds like the signature of this failure and never once occurs: the
# band cuts the near player at the WAIST, so the fragment keeps the head at the
# top and looks perfectly well-formed.  What it loses is the FEET.  The
# invariant worth testing is the presence of the lower body, not the order of
# what is present.
#
# IT WORKS, AND IT IS OFF.  Measured end to end it removes the fragment class
# completely -- 2 of the 12 remaining far-serve false positives are still cut
# near players, and this takes them to 0 -- but it does not pay for itself:
#
#     rules on                    far serve        point end      fragments
#     edge 3.0                  96.7% / 88.0%   58.3% / 54.0%         2
#     edge 3.0 + lower body     95.6% / 87.9%   57.5% / 51.8%         0
#
# One far serve, one point end, and 2.2 points of point-end precision, to
# remove two false positives that the orchestrator's rule 2 -- a far serve
# surrounded by near serves is suspect -- is already aimed at.  The recall side
# of that trade is the wrong direction for a point START.
#
# The serve it costs is NOT one this rule rejects.  Clip 23's server at 377.8 s
# has every keypoint confident, both ankles, hips at 0.51 of the box, and sits
# 41 px clear of the band edge.  It is lost because dropping OTHER samples puts
# holes in its stillness window -- a second-order effect through `still`, not a
# judgement about that box.
#
# Kept because it is the more general statement.  FAR_EDGE_PX tests one cause of
# a footless box; this tests the box.  On footage where the truncation comes
# from an occluding player, the net post or the frame edge rather than the crop,
# this is the rule that catches it and FAR_EDGE_PX is not.
FAR_LOWER_BODY_CONF = 0.20
FAR_NEED_LOWER_BODY = False


FAR_HIP_MAX_FRAC = 0.85    # hips this far down the box means no legs below them


def _has_lower_body(kp_one, box=None):
    """Is there a lower body in this box, so its bottom edge is the feet?

    Two readings, because they fail in different places.  A knee or an ankle is
    direct evidence.  Failing that, WHERE THE HIPS SIT IN THE BOX answers the
    same question indirectly: on a whole player the hips are around 0.45-0.77 of
    the way down, and on a box cut at the waist they are at 0.88-1.02 -- there
    is nothing below them because there is no room below them.

    The hip reading is kept as a fallback rather than the primary test because
    it is the weaker of the two, and absent hips are not evidence either way.
    """
    if np.any(kp_one[[13, 14, 15, 16], 2] >= FAR_LOWER_BODY_CONF):
        return True
    if box is None:
        return False
    h = float(box[3] - box[1])
    hips = [(float(kp_one[i, 1]) - float(box[1])) / h
            for i in (11, 12) if kp_one[i, 2] >= FAR_LOWER_BODY_CONF and h > 0]
    if not hips:
        return False                    # nothing to read; no lower body found
    return float(np.mean(hips)) <= FAR_HIP_MAX_FRAC

LOST_FRAMES = 60       # keep a lost slot's anchor this long before freeing it
MAX_SPEED = 9.0        # m/s; faster than a sprint means a different human

# ── assignment weights ───────────────────────────────────────────────────
W_CONT = 2.0           # continuity bonus scale, exp(-d / CONT_SCALE)
CONT_SCALE = 2.0       # metres
W_CONF = 0.6
W_HOME = 1.0           # weight on proximity to the player's OWN baseline

# THE CLAIM SCORE PREFERS PLAYERS NEAR THEIR OWN BASELINE, replacing an earlier
# box-height term that was quietly side-specific.
#
# Box height is a DEPTH proxy, not a quality one.  On the near side bigger means
# closer to the camera, hence closer to the near baseline -- so preferring the
# biggest box happened to prefer the right people, and the near detector reached
# 100% recall with it.  On the far side the same term is exactly backwards:
# bigger means closer to the NET, and the server is by definition the deepest
# player on their side.
#
# Measured on Data/25 (doubles), at every one of the seven missed far serves
# three candidates competed for two far slots, and the server was every time the
# DEEPEST (court_y 22.1-27.6 against 17.7-19.7) and also the LOWEST-confidence
# (0.73-0.79 against 0.85-0.92).  The height term handed both slots to the two
# shallowest and dropped the server -- while that serve's trophy shape was
# present at full strength in the band, in all ten serves.  Ranking by
# confidence instead would have dropped the server too.
#
# Proximity to one's own baseline is the rule that is right on both sides at
# once: near players are nearest y=0, far players nearest y=COURT_L, and the
# candidate least likely to belong to a side is the one closest to mid-court --
# which is also where a mis-projected player from the OTHER side lands.


def tracks_path(video_path, suffix="_anya2_tracks.npz"):
    d = WD.artifact_dir(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{suffix}")


def _px_per_m(H_inv, cy):
    """Image px per court metre at depth `cy`, from the court's width there."""
    p = cv2.perspectiveTransform(
        np.array([[[0.0, cy]], [[C.COURT_W, cy]]], dtype=np.float64), H_inv
    ).reshape(-1, 2)
    return float(np.hypot(*(p[1] - p[0]))) / C.COURT_W


def _far_band_edge(video, far_npz):
    """The far band's lower edge in ANALYSIS-frame y, and the tolerance there.

    Returns (edge_y, tol) or None when the far pass carries no crop -- a file
    written before the crop was recorded, which must not be guessed at.
    """
    if not far_npz:
        return None
    z = np.load(far_npz)
    if "crop" not in z:
        return None
    crop = np.asarray(z["crop"], dtype=np.float64)
    cap = cv2.VideoCapture(video)
    src_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if not src_h:
        return None
    sy = float(src_h) / C.ANALYSIS_SIZE[1]     # source px per analysis px
    return float(crop[3]) / sy, FAR_EDGE_PX / sy


def _height_plausible(h, cy, H_inv):
    """Is a box of height `h` consistent with standing at depth `cy`?

    One-sided -- see HEIGHT_MAX_RATIO.  Unknown depth passes: this rule catches
    one specific contradiction, it is not another way to lose a player.
    """
    if not (np.isfinite(h) and np.isfinite(cy)):
        return True
    exp = PLAYER_HEIGHT_M * _px_per_m(H_inv, float(cy))
    if exp < 1.0:
        return True
    return h <= HEIGHT_MAX_RATIO * exp


def _homeness(cy, side):
    """1.0 at the side's own baseline, 0.0 at mid-court (and beyond). NaN -> 0.

    Clipped above 1.0 rather than rewarded: the far player's ground point
    routinely projects several metres PAST the baseline they are standing on
    (median +4 m -- see FAR_BACK_M), so "even deeper" is noise, not evidence.
    """
    if not np.isfinite(cy):
        return 0.0
    if side == C.NEAR:
        return float(np.clip(1.0 - cy / C.MID_Y, 0.0, 1.0))
    return float(np.clip((cy - C.MID_Y) / (C.COURT_L - C.MID_Y), 0.0, 1.0))


def _trackable(cx, cy, side):
    """Loose zone membership, per side. NaN -> False."""
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return False
    if not (-X_TRACK_PAD <= cx <= C.COURT_W + X_TRACK_PAD):
        return False
    if side == C.NEAR:
        return -NEAR_BACK_M <= cy < C.MID_Y
    return C.MID_Y <= cy <= C.COURT_L + FAR_BACK_M


class _Slot:
    """One player slot: its last known court position and how stale it is."""

    __slots__ = ("court", "since")

    def __init__(self):
        self.court = None
        self.since = 10 ** 9

    @property
    def anchored(self):
        return self.court is not None and self.since <= LOST_FRAMES

    def miss(self):
        self.since += 1
        # The anchor is NOT cleared when it goes stale.  A stale anchor stops
        # being usable for continuity (see `anchored`) but stays usable as a
        # RE-ACQUISITION HINT -- "this slot was last holding somebody here".
        #
        # Clearing it is what made a single player oscillate between slots: on
        # a singles clip the near player was split 47%/49% across the two near
        # slots with 38 identity switches in 420 s, because once both slots had
        # expired the next detection went to whichever slot index came first,
        # not to the slot that had been following that player.  Detection
        # survived it (the cross-slot refractory in near_serve merges the
        # halves) but the answer to WHICH PLAYER SERVED did not, and in doubles
        # that answer is the point.

    def hit(self, court):
        self.court, self.since = court, 1


def _shortlist(cands, slots, fps):
    """Cut a side's candidates down to at most `len(slots)`, by BELONGING.

    The user's rule is at most two players per side, and this is where it is
    enforced -- as a deliberate choice of WHICH two, rather than as whichever
    two happened to reach a slot first.

    That distinction is the whole fix.  Assignment alone cannot do it: anchored
    slots claim before free ones (rightly -- identity is worth more than a
    stranger's confidence), so once two people hold the slots they hold them
    forever, however little they belong to that side.  Measured on Data/40
    (doubles), three or four candidates competed for two far slots at every
    serve, and the two shallowest held them: the serve's trophy shape was
    present at FULL STRENGTH in the band for all 13 labelled serves while the
    tracked slots caught only 6.  Data/25 showed the same thing before the
    homeness term, and homeness alone fixed 25 but not 40, because on 40 the
    wrong pair was already anchored.

    Belonging combines three things, and continuity is one of them so that a
    player who is genuinely mid-court for a moment -- a far player at the net
    during a rally -- is not dropped in favour of a stranger who happens to be
    standing deeper.
    """
    if len(cands) <= len(slots):
        return cands
    scored = []
    for cd in cands:
        cont = 0.0
        for slot in slots:
            if not slot.anchored:
                continue
            d = float(np.hypot(cd["court"][0] - slot.court[0],
                               cd["court"][1] - slot.court[1]))
            if d / max(slot.since, 1) * fps > MAX_SPEED:
                continue
            cont = max(cont, W_CONT * np.exp(-d / CONT_SCALE))
        scored.append((-(W_HOME * cd["home"] + W_CONF * cd["conf"] + cont), cd))
    scored.sort(key=lambda x: x[0])
    return [cd for _, cd in scored[:len(slots)]]


def _assign_side(cands, slots, fps):
    """Assign candidates to this side's slots. Returns {slot_idx: cand_idx}.

    Two passes, and the order is the whole design:

      1. ANCHORED SLOTS CLAIM FIRST, by continuity.  An anchored slot is a
         player we were already following, and keeping that identity is worth
         more than any freshly-detected person's confidence -- otherwise a
         partner stepping in front briefly steals the slot and both tracks
         swap, which is the failure `select_near` describes.

      2. FREE SLOTS take what is left, best claim first.  `claim` deliberately
         has no continuity term (there is nothing to be continuous with) and no
         side preference (the side was already decided).

    `cands` is a list of dicts with 'court', 'conf', 'h'.
    """
    out = {}
    taken = set()

    pairs = []
    for si, slot in enumerate(slots):
        if not slot.anchored:
            continue
        for ci, cd in enumerate(cands):
            d = float(np.hypot(cd["court"][0] - slot.court[0],
                               cd["court"][1] - slot.court[1]))
            # The outright rejection, not a penalty. See the module docstring.
            if d / max(slot.since, 1) * fps > MAX_SPEED:
                continue
            pairs.append((-W_CONT * np.exp(-d / CONT_SCALE), si, ci))

    for _, si, ci in sorted(pairs):
        if si in out or ci in taken:
            continue
        out[si] = ci
        taken.add(ci)

    free = [si for si in range(len(slots)) if si not in out]
    if free:
        claims = sorted(
            ((-(W_CONF * cd["conf"] + W_HOME * cd["home"]), ci)
             for ci, cd in enumerate(cands) if ci not in taken))
        # Strongest claim first, and each one goes to the free slot whose STALE
        # anchor is nearest -- not to the lowest free index.  A slot that never
        # held anybody has no anchor and sorts last, so a genuinely new player
        # still lands somewhere.
        for _, ci in claims:
            if not free:
                break
            here = cands[ci]["court"]
            si = min(free, key=lambda s: (
                float(np.hypot(here[0] - slots[s].court[0],
                               here[1] - slots[s].court[1]))
                if slots[s].court is not None else np.inf, s))
            free.remove(si)
            out[si] = ci
            taken.add(ci)
    return out


def _stack(near_npz, far_npz):
    """Merge the two ROIs' detections into one per-sample candidate list.

    The near pass and the far pass are separate because ONE POSE PASS CANNOT
    SEE BOTH ENDS -- at 540p the far player is 20-40 px and the model does not
    find them at all (measured: 0 of 192 sampled detections on Data/21 project
    past mid-court).  See perceive.py.

    They are merged rather than kept apart because side membership is decided
    by the COURT, not by which pass found the person.  A pass is an ROI, not a
    label: the far band's lower edge catches a near player who has come to the
    net, and the whole-frame pass catches a far player who has come forward.
    Trusting the pass would mislabel both.  So everything is pooled here and
    `court.is_near` decides, exactly as it does for a single pass.
    """
    zs = [np.load(p) for p in (near_npz, far_npz) if p]
    if not zs:
        raise ValueError("no detections given")
    if len(zs) == 1:
        z = zs[0]
        # No far pass: nothing can be cut by a band that was never cropped.
        return z["kp"], z["box"], z["conf"], float(z["fps"]), z, z["kp"].shape[1]
    a, b = zs
    # The two passes must be on ONE timeline.  They are extracted at the same
    # pose_fps from the same source, so they should already agree; a mismatch
    # means one is stale and silently interleaving them would shift every event
    # by an unknown amount.  Refuse rather than guess.
    na, nb = len(a["conf"]), len(b["conf"])
    if abs(na - nb) > 2 or abs(float(a["fps"]) - float(b["fps"])) > 0.01:
        raise ValueError(
            f"near dets ({na} @ {float(a['fps']):.3f} fps) and far dets "
            f"({nb} @ {float(b['fps']):.3f} fps) are not the same timeline -- "
            "re-run perceive for whichever is stale")
    n = min(na, nb)
    kp = np.concatenate([a["kp"][:n], b["kp"][:n]], axis=1)
    box = np.concatenate([a["box"][:n], b["box"][:n]], axis=1)
    conf = np.concatenate([a["conf"][:n], b["conf"][:n]], axis=1)
    # Candidate indices below this came from the near (whole-frame) pass, which
    # has no crop and therefore no edge to be cut by.  At or above it they came
    # from the far band -- the only ones FAR_EDGE_PX applies to.
    return kp, box, conf, float(a["fps"]), a, int(a["kp"].shape[1])


def build(video, dets_npz=None, far_npz=None, out=None, verbose=True):
    from walking.extract_pose import dets_path

    dets_npz = dets_npz or dets_path(video)
    kp_all, box_all, conf_all, fps, z, n_near = _stack(dets_npz, far_npz)
    # The far band's lower edge, and how close a box bottom may come to it
    # before it stops being a ground point.  See FAR_EDGE_PX.
    band = _far_band_edge(video, far_npz)
    n_cut = n_nolegs = 0
    n, k = conf_all.shape
    # The court map is read PER FRAME, not once.  A bumped camera leaves the
    # cached corners describing a court that is no longer where they say, and
    # nothing here would notice: every projection still returns a number, just
    # the wrong one, for every frame after the bump.  `Geometry` composes the
    # clicked homography with the camera track's warp for that frame, and with
    # no track cached it IS the clicked homography -- see anya2/camera.py.
    geom = C.Geometry(video)
    # Pose sample f is source frame f * stride (`perceive._pose_pass` writes
    # `idx = range(0, total, stride)`), and the camera track is keyed by source
    # frame because that is the one clock the two decimated passes share.
    stride = int(z["stride"]) if "stride" in z else 1
    if verbose and not geom.is_static:
        print(f"[tracks] camera track active over {len(geom.track.frames)} samples")

    kp_out = np.full((n, N_SLOTS, N_KP, 3), np.nan, dtype=np.float32)
    bb_out = np.full((n, N_SLOTS, 4), np.nan, dtype=np.float32)
    ct_out = np.full((n, N_SLOTS, 2), np.nan, dtype=np.float32)
    el_out = np.zeros((n, N_SLOTS), dtype=bool)

    slots = {C.NEAR: [_Slot(), _Slot()], C.FAR: [_Slot(), _Slot()]}
    slot_ids = {C.NEAR: NEAR_SLOTS, C.FAR: FAR_SLOTS}

    for f in range(n):
        src_f = f * stride
        H = geom.H_at(src_f)
        H_inv = geom.H_inv_at(src_f)
        by_side = {C.NEAR: [], C.FAR: []}
        for i in range(k):
            cf = conf_all[f, i]
            if not np.isfinite(cf):
                continue
            box = box_all[f, i]
            h = float(box[3] - box[1])
            if not np.isfinite(h) or h < MIN_H_PX:
                continue
            # A far-pass box cut by the band's lower edge has no feet in it, so
            # it has no ground point to project.  Drop it rather than place it
            # somewhere nobody stood -- see FAR_EDGE_PX.
            if band is not None and i >= n_near and box[3] >= band[0] - band[1]:
                n_cut += 1
                continue
            # No lower body in the box means its bottom edge is not a ground
            # point, whatever cut it off -- see FAR_NEED_LOWER_BODY.
            if (FAR_NEED_LOWER_BODY and i >= n_near
                    and not _has_lower_body(kp_all[f, i], box)):
                n_nolegs += 1
                continue
            cx, cy = C.project(H, box)
            sd = C.NEAR if C.is_near(cy) else C.FAR
            if not _trackable(cx, cy, sd):
                continue
            if not _height_plausible(h, cy, H_inv):
                continue
            by_side[sd].append({"i": i, "court": (float(cx), float(cy)),
                                "conf": float(cf), "h": h,
                                "home": _homeness(cy, sd)})

        for sd in (C.NEAR, C.FAR):
            cands = _shortlist(by_side[sd], slots[sd], fps)
            picked = _assign_side(cands, slots[sd], fps) if cands else {}
            for local, slot in enumerate(slots[sd]):
                g = slot_ids[sd][local]
                if local not in picked:
                    slot.miss()
                    continue
                cd = cands[picked[local]]
                kp_out[f, g] = kp_all[f, cd["i"]]
                bb_out[f, g] = box_all[f, cd["i"]]
                ct_out[f, g] = cd["court"]
                el_out[f, g] = bool(C.in_bounds(cd["court"][0]))
                slot.hit(cd["court"])

    out = out or tracks_path(video)
    extra = {kk: z[kk] for kk in ("stride", "src_fps", "n_src_frames") if kk in z}
    np.savez_compressed(out, kp=kp_out, bbox=bb_out, court=ct_out,
                        eligible=el_out, side=SIDE_OF_SLOT.astype("U4"),
                        fps=np.float64(fps), **extra)
    if verbose:
        if band is not None:
            print(f"[tracks] dropped {n_cut} far-band detections cut by the "
                  f"band's lower edge (within {FAR_EDGE_PX:.0f} source px)"
                  + (f", {n_nolegs} with no knee or ankle in the box"
                     if FAR_NEED_LOWER_BODY else ""))
        _report(out, n)
    return out


def _report(path, n=None):
    z = np.load(path, allow_pickle=False)
    bb, el = z["bbox"], z["eligible"]
    n = n or len(bb)
    seen = np.isfinite(bb[:, :, 0])
    print(f"[tracks] {os.path.basename(path)}  {n} samples @ {float(z['fps']):.2f} fps")
    for sd, ss in (("near", NEAR_SLOTS), ("far", FAR_SLOTS)):
        ss = list(ss)
        occ = seen[:, ss].sum(axis=1)
        hist = {int(v): int(c) for v, c in zip(*np.unique(occ, return_counts=True))}
        share = ", ".join(f"{v}:{100.0 * c / n:.0f}%" for v, c in sorted(hist.items()))
        # Eligibility is reported over OCCUPIED slot-frames only. Over all slot
        # -frames it is dominated by empty slots, which says nothing about the
        # gate -- on a singles clip half the near slots are empty by definition.
        nocc = int(seen[:, ss].sum())
        elig = f"{100.0 * el[:, ss].sum() / nocc:.0f}%" if nocc else "n/a"
        print(f"  {sd:>4}: frames by #players [{share}]"
              f"   tracked slot-frames {nocc}, of which eligible {elig}")
    return z


def load(video, path=None):
    """Load a tracks npz as a dict of arrays."""
    z = np.load(path or tracks_path(video), allow_pickle=False)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("video")
    ap.add_argument("--dets", default=None, help="near-ROI detections npz")
    ap.add_argument("--far-dets", default=None,
                    help="far-ROI detections npz (perceive --roi far). Without "
                         "it the far slots stay empty -- the whole-frame pass "
                         "does not see the far player.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", action="store_true",
                    help="Re-print the occupancy report for an existing npz")
    a = ap.parse_args()
    if a.report and os.path.isfile(a.out or tracks_path(a.video)):
        _report(a.out or tracks_path(a.video))
        return
    build(a.video, a.dets, a.far_dets, a.out)


if __name__ == "__main__":
    main()
