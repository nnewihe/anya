"""
tracks.py
=========
Turn the two all-persons pose passes into at most FOUR player slots:

    slot 0, 1   near side
    slot 2, 3   far side

Singles is just the case where slots 1 and 3 stay empty (NaN).

Per pose sample, every detected person is projected onto the court and:

  1. FILTERED   too small, outside the loose tracking zone, or too tall for
                where they claim to stand -> dropped        (_candidates_at)
  2. SIDED      near or far, by which baseline their feet are closer to
  3. SHORTLISTED  at most two per side, chosen by how much they belong there:
                near their own baseline, confident, continuous  (_shortlist)
  4. ASSIGNED   slots already following someone claim first, by continuity;
                free slots take the rest, best claim first      (_assign_side)

A candidate that would require moving faster than MAX_SPEED_MPS from a slot's
last position is never given that slot: a missing frame is honest, a wrong
one poisons everything after it.

TRACKED vs ELIGIBLE
-------------------
The tracking zone is deliberately loose, so a player walking to the ball carts
keeps their slot.  Separately, `eligible[n, slot]` records the user's strict
rule (box x-centre inside the doubles court + 3 ft).  Detectors use `eligible`
to decide who may be serving.

Output `<stem>_anya2_tracks.npz`:
    kp       [N, 4, 17, 3]  keypoints, ANALYSIS_SIZE px (NaN = slot empty)
    bbox     [N, 4, 4]      xyxy
    court    [N, 4, 2]      ground point, court metres
    eligible [N, 4]         bool
    side     [4]            'near','near','far','far'
    fps, stride, src_fps, n_src_frames

Evidence and history: docs/anya2/NOTES.md, "tracks.py".
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from pipeline import workdir
from pipeline.anya2 import court

N_KEYPOINTS = 17
N_SLOTS = 4
NEAR_SLOTS = (0, 1)
FAR_SLOTS = (2, 3)
SIDE_OF_SLOT = np.array([court.NEAR, court.NEAR, court.FAR, court.FAR], dtype=object)

# ── who may be tracked at all (loose on purpose) ─────────────────────────
MIN_H_PX = 12.0            # smaller boxes are spectators on other courts / noise
X_TRACK_PAD = 6.0          # metres outside the singles lines a player may wander
NEAR_BACK_M = 6.0          # ...and behind the near baseline
FAR_BACK_M = 12.0          # ...and behind the far one (large: far projections
                           # land a median 4 m too deep -- see NOTES)

# ── box height must agree with depth ─────────────────────────────────────
# A box much taller than a 1.75 m person would be at its projected depth is a
# closer player mis-projected (e.g. the near player inside the far band).
HEIGHT_MAX_RATIO = 1.40    # observed / expected; normal p90 is 1.24
PLAYER_HEIGHT_M = 1.75

# ── slot continuity ──────────────────────────────────────────────────────
LOST_FRAMES = 60           # samples a slot keeps following an unseen player
MAX_SPEED = 9.0            # m/s; faster than a sprint means a different person

# ── how candidates are scored ────────────────────────────────────────────
W_CONT = 2.0               # continuity bonus: W_CONT * exp(-distance / CONT_SCALE)
CONT_SCALE = 2.0           # metres
W_CONF = 0.6               # weight on detection confidence
W_HOME = 1.0               # weight on closeness to the player's OWN baseline


def tracks_path(video_path, suffix="_anya2_tracks.npz"):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(workdir.artifact_dir(video_path), f"{stem}{suffix}")


def load(video, path=None):
    """The tracks npz as a plain dict of arrays."""
    z = np.load(path or tracks_path(video), allow_pickle=False)
    return {key: z[key] for key in z.files}


# ═════════════════════════════════════════════════════════════════════════
# Building the tracks
# ═════════════════════════════════════════════════════════════════════════

def build(video, dets_npz=None, far_npz=None, out=None, verbose=True):
    """Build `<stem>_anya2_tracks.npz` from the near and far pose passes."""
    from walking.extract_pose import dets_path

    dets_npz = dets_npz or dets_path(video)
    all_kp, all_boxes, all_conf, fps, source = _merge_passes(dets_npz, far_npz)
    n_samples, n_people = all_conf.shape

    # The court map is evaluated per frame (the camera may have moved).
    geometry = court.Geometry(video)
    # Pose sample i is source frame i * stride; the camera track is keyed by
    # source frame.
    stride = int(source["stride"]) if "stride" in source else 1
    if verbose and not geometry.is_static:
        print(f"[tracks] camera track active over {len(geometry.track.frames)} samples")

    out_kp = np.full((n_samples, N_SLOTS, N_KEYPOINTS, 3), np.nan, dtype=np.float32)
    out_boxes = np.full((n_samples, N_SLOTS, 4), np.nan, dtype=np.float32)
    out_court = np.full((n_samples, N_SLOTS, 2), np.nan, dtype=np.float32)
    out_eligible = np.zeros((n_samples, N_SLOTS), dtype=bool)

    slots = {court.NEAR: [PlayerSlot(), PlayerSlot()], court.FAR: [PlayerSlot(), PlayerSlot()]}
    slot_ids = {court.NEAR: NEAR_SLOTS, court.FAR: FAR_SLOTS}

    for sample in range(n_samples):
        src_frame = sample * stride
        candidates_by_side = _candidates_at(
            sample, all_boxes, all_conf, n_people,
            geometry.H_at(src_frame), geometry.H_inv_at(src_frame))

        for side in (court.NEAR, court.FAR):
            candidates = _shortlist(candidates_by_side[side], slots[side], fps)
            chosen = _assign_side(candidates, slots[side], fps) if candidates else {}
            for local_index, slot in enumerate(slots[side]):
                slot_id = slot_ids[side][local_index]
                if local_index not in chosen:
                    slot.miss()
                    continue
                candidate = candidates[chosen[local_index]]
                person = candidate["i"]
                out_kp[sample, slot_id] = all_kp[sample, person]
                out_boxes[sample, slot_id] = all_boxes[sample, person]
                out_court[sample, slot_id] = candidate["court"]
                out_eligible[sample, slot_id] = bool(court.in_bounds(candidate["court"][0]))
                slot.hit(candidate["court"])

    out = out or tracks_path(video)
    extra = {key: source[key] for key in ("stride", "src_fps", "n_src_frames") if key in source}
    np.savez_compressed(out, kp=out_kp, bbox=out_boxes, court=out_court,
                        eligible=out_eligible, side=SIDE_OF_SLOT.astype("U4"),
                        fps=np.float64(fps), **extra)
    if verbose:
        _report(out, n_samples)
    return out


def _merge_passes(near_npz, far_npz):
    """Pool the near and far pose passes into one person list per sample.

    Side is decided by the COURT, not by which pass found a person (the far
    band can catch a near player at the net, and vice versa), so both passes
    are pooled here.  They must be on one timeline; a mismatch means one file
    is stale, and this refuses rather than guess.

    Returns (kp [N,P,17,3], boxes [N,P,4], conf [N,P], fps, near-pass npz).
    """
    passes = [np.load(p) for p in (near_npz, far_npz) if p]
    if not passes:
        raise ValueError("no detections given")
    if len(passes) == 1:
        only = passes[0]
        return only["kp"], only["box"], only["conf"], float(only["fps"]), only
    near_pass, far_pass = passes
    n_near, n_far = len(near_pass["conf"]), len(far_pass["conf"])
    if abs(n_near - n_far) > 2 or abs(float(near_pass["fps"]) - float(far_pass["fps"])) > 0.01:
        raise ValueError(
            f"near dets ({n_near} @ {float(near_pass['fps']):.3f} fps) and far dets "
            f"({n_far} @ {float(far_pass['fps']):.3f} fps) are not the same timeline -- "
            "re-run perceive for whichever is stale")
    n = min(n_near, n_far)
    kp = np.concatenate([near_pass["kp"][:n], far_pass["kp"][:n]], axis=1)
    boxes = np.concatenate([near_pass["box"][:n], far_pass["box"][:n]], axis=1)
    conf = np.concatenate([near_pass["conf"][:n], far_pass["conf"][:n]], axis=1)
    return kp, boxes, conf, float(near_pass["fps"]), near_pass


def _candidates_at(sample, all_boxes, all_conf, n_people, H, H_inv):
    """Every person in this sample who may be tracked, split by side.

    Each candidate is a dict: i (person index), court (x, y), conf, h (box
    height px), home (closeness to own baseline, 0..1).
    """
    by_side = {court.NEAR: [], court.FAR: []}
    for person in range(n_people):
        confidence = all_conf[sample, person]
        if not np.isfinite(confidence):
            continue
        box = all_boxes[sample, person]
        height_px = float(box[3] - box[1])
        if not np.isfinite(height_px) or height_px < MIN_H_PX:
            continue
        court_x, court_y = court.project(H, box)
        side = court.NEAR if court.is_near(court_y) else court.FAR
        if not _trackable(court_x, court_y, side):
            continue
        if not _height_plausible(height_px, court_y, H_inv):
            continue
        by_side[side].append({"i": person, "court": (float(court_x), float(court_y)),
                              "conf": float(confidence), "h": height_px,
                              "home": _homeness(court_y, side)})
    return by_side


def _trackable(court_x, court_y, side):
    """Inside the loose tracking zone for this side.  NaN -> False."""
    if not (np.isfinite(court_x) and np.isfinite(court_y)):
        return False
    if not (-X_TRACK_PAD <= court_x <= court.COURT_W + X_TRACK_PAD):
        return False
    if side == court.NEAR:
        return -NEAR_BACK_M <= court_y < court.MID_Y
    return court.MID_Y <= court_y <= court.COURT_L + FAR_BACK_M


def _height_plausible(height_px, court_y, H_inv):
    """False only when the box is too TALL for its projected depth.

    One-sided: small boxes (crouching, partial, clipped) are normal.  Unknown
    depth passes.
    """
    if not (np.isfinite(height_px) and np.isfinite(court_y)):
        return True
    expected_px = PLAYER_HEIGHT_M * _px_per_m(H_inv, float(court_y))
    if expected_px < 1.0:
        return True
    return height_px <= HEIGHT_MAX_RATIO * expected_px


def _px_per_m(H_inv, court_y):
    """Image px per court metre at depth `court_y`, from the court's width there."""
    ends = cv2.perspectiveTransform(
        np.array([[[0.0, court_y]], [[court.COURT_W, court_y]]], dtype=np.float64), H_inv
    ).reshape(-1, 2)
    return float(np.hypot(*(ends[1] - ends[0]))) / court.COURT_W


def _homeness(court_y, side):
    """1.0 on the side's own baseline (or deeper), 0.0 at mid-court.  NaN -> 0."""
    if not np.isfinite(court_y):
        return 0.0
    if side == court.NEAR:
        return float(np.clip(1.0 - court_y / court.MID_Y, 0.0, 1.0))
    return float(np.clip((court_y - court.MID_Y) / (court.COURT_L - court.MID_Y), 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════════════
# Slots and assignment
# ═════════════════════════════════════════════════════════════════════════

class PlayerSlot:
    """One player slot: where it last saw its player, and how long ago."""

    __slots__ = ("court", "since")

    def __init__(self):
        self.court = None          # last (x, y) in court metres, or None
        self.since = 10 ** 9       # samples since last seen

    @property
    def anchored(self):
        """Following someone recently enough to use continuity."""
        return self.court is not None and self.since <= LOST_FRAMES

    def miss(self):
        # The last position is KEPT when it goes stale: it no longer counts for
        # continuity, but it is still the best hint for re-acquiring the same
        # player into the same slot (clearing it made players swap slots).
        self.since += 1

    def hit(self, court_xy):
        self.court, self.since = court_xy, 1


def _distance_m(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _too_fast(distance_m, slot, fps):
    """Would reaching this candidate from the slot's last position exceed MAX_SPEED?"""
    return distance_m / max(slot.since, 1) * fps > MAX_SPEED


def _shortlist(candidates, slots, fps):
    """Keep at most len(slots) candidates: the ones that most BELONG on this side.

    belonging = W_HOME * home + W_CONF * conf + best continuity bonus to any
    anchored slot.  Without this, two people who grabbed the slots first keep
    them forever however little they belong there.
    """
    if len(candidates) <= len(slots):
        return candidates
    scored = []
    for candidate in candidates:
        continuity = 0.0
        for slot in slots:
            if not slot.anchored:
                continue
            distance = _distance_m(candidate["court"], slot.court)
            if _too_fast(distance, slot, fps):
                continue
            continuity = max(continuity, W_CONT * np.exp(-distance / CONT_SCALE))
        belonging = W_HOME * candidate["home"] + W_CONF * candidate["conf"] + continuity
        scored.append((-belonging, candidate))
    scored.sort(key=lambda pair: pair[0])
    return [candidate for _, candidate in scored[:len(slots)]]


def _assign_side(candidates, slots, fps):
    """Assign candidates to this side's slots.  Returns {slot index: candidate index}.

      1. Anchored slots claim first, closest candidate first (continuity beats
         a stranger's confidence, so partners don't swap slots).
      2. Free slots take what is left, strongest claim (conf + home) first;
         each goes to the free slot whose stale last position is nearest.
    """
    chosen = {}
    taken = set()

    continuity_pairs = []
    for slot_index, slot in enumerate(slots):
        if not slot.anchored:
            continue
        for candidate_index, candidate in enumerate(candidates):
            distance = _distance_m(candidate["court"], slot.court)
            if _too_fast(distance, slot, fps):
                continue
            continuity_pairs.append((-W_CONT * np.exp(-distance / CONT_SCALE),
                                     slot_index, candidate_index))
    for _, slot_index, candidate_index in sorted(continuity_pairs):
        if slot_index in chosen or candidate_index in taken:
            continue
        chosen[slot_index] = candidate_index
        taken.add(candidate_index)

    free_slots = [s for s in range(len(slots)) if s not in chosen]
    if free_slots:
        claims = sorted(
            (-(W_CONF * candidate["conf"] + W_HOME * candidate["home"]), candidate_index)
            for candidate_index, candidate in enumerate(candidates)
            if candidate_index not in taken)
        for _, candidate_index in claims:
            if not free_slots:
                break
            position = candidates[candidate_index]["court"]
            # Nearest stale position wins; a slot that never held anyone sorts last.
            slot_index = min(free_slots, key=lambda s: (
                _distance_m(position, slots[s].court)
                if slots[s].court is not None else np.inf, s))
            free_slots.remove(slot_index)
            chosen[slot_index] = candidate_index
            taken.add(candidate_index)
    return chosen


# ═════════════════════════════════════════════════════════════════════════
# Reporting and CLI
# ═════════════════════════════════════════════════════════════════════════

def _report(path, n_samples=None):
    z = np.load(path, allow_pickle=False)
    boxes, eligible = z["bbox"], z["eligible"]
    n_samples = n_samples or len(boxes)
    seen = np.isfinite(boxes[:, :, 0])
    print(f"[tracks] {os.path.basename(path)}  {n_samples} samples @ {float(z['fps']):.2f} fps")
    for side_name, side_slots in (("near", NEAR_SLOTS), ("far", FAR_SLOTS)):
        side_slots = list(side_slots)
        players_per_frame = seen[:, side_slots].sum(axis=1)
        histogram = {int(v): int(c) for v, c in
                     zip(*np.unique(players_per_frame, return_counts=True))}
        share = ", ".join(f"{v}:{100.0 * c / n_samples:.0f}%" for v, c in sorted(histogram.items()))
        # Eligibility over OCCUPIED slot-frames only (empty slots say nothing).
        n_occupied = int(seen[:, side_slots].sum())
        eligible_share = (f"{100.0 * eligible[:, side_slots].sum() / n_occupied:.0f}%"
                          if n_occupied else "n/a")
        print(f"  {side_name:>4}: frames by #players [{share}]"
              f"   tracked slot-frames {n_occupied}, of which eligible {eligible_share}")
    return z


def main():
    ap = argparse.ArgumentParser(description="Turn the all-persons pose pass into at "
                                 "most TWO near-side and TWO far-side")
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
