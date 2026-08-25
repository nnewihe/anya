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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline.anya2 import court as C

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
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{suffix}")


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
        return z["kp"], z["box"], z["conf"], float(z["fps"]), z
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
    return kp, box, conf, float(a["fps"]), a


def build(video, dets_npz=None, far_npz=None, out=None, verbose=True):
    from walking.extract_pose import dets_path

    dets_npz = dets_npz or dets_path(video)
    kp_all, box_all, conf_all, fps, z = _stack(dets_npz, far_npz)
    n, k = conf_all.shape
    H = C.load_homography(video)

    kp_out = np.full((n, N_SLOTS, N_KP, 3), np.nan, dtype=np.float32)
    bb_out = np.full((n, N_SLOTS, 4), np.nan, dtype=np.float32)
    ct_out = np.full((n, N_SLOTS, 2), np.nan, dtype=np.float32)
    el_out = np.zeros((n, N_SLOTS), dtype=bool)

    slots = {C.NEAR: [_Slot(), _Slot()], C.FAR: [_Slot(), _Slot()]}
    slot_ids = {C.NEAR: NEAR_SLOTS, C.FAR: FAR_SLOTS}

    for f in range(n):
        by_side = {C.NEAR: [], C.FAR: []}
        for i in range(k):
            cf = conf_all[f, i]
            if not np.isfinite(cf):
                continue
            box = box_all[f, i]
            h = float(box[3] - box[1])
            if not np.isfinite(h) or h < MIN_H_PX:
                continue
            cx, cy = C.project(H, box)
            sd = C.NEAR if C.is_near(cy) else C.FAR
            if not _trackable(cx, cy, sd):
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
