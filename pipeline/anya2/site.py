"""
site.py
=======
A fixed camera's calibration, clicked once and carried to every later
recording.

WHY
---
`run.ensure_court` asks for four clicks per VIDEO, because the court cache is
keyed by the video's stem.  That is right for a phone propped on a bag and
wrong for a camera bolted to the back fence: the court is in the same place in
every recording, and a headless box (the Raspberry Pi service in `pi/`) has no
one to click.

A site profile is the clicked corners plus the frame they were clicked on:

    <profile>/court.json   corners in 960x540 analysis px, source size, band
    <profile>/ref.png      the calibration frame at 960x540

WHY THE NEW RECORDING IS REGISTERED, NOT TRUSTED
------------------------------------------------
A "fixed" mount is fixed until someone leans on the fence.  Copying the corners
blindly would reproduce exactly the failure `camera.py` exists to catch -- no
error, every projection a plausible number, the far baseline four metres off --
except across sessions instead of within one.  So `apply` registers the new
recording's reference frame against `ref.png` with the camera track's own
machinery (ORB + ground-plane RANSAC, `camera.register`), moves the corners by
that warp, and refuses outright when the warp fails or moves a corner further
than `MAX_SITE_SHIFT_PX`.  A small nudge is corrected; a knocked mount stops the
run and asks for a re-click instead of producing a confidently wrong reel.

The corners are written to the recording's court cache in exactly the format
`utilities.init_court` writes, and the reference-frame rule is the same one
(`camera.reference_index`), so everything downstream -- including the per-frame
camera track -- sees an ordinary calibrated video.
"""

import argparse
import json
import os

import cv2
import numpy as np

from pipeline.anya2 import camera as CAM
from pipeline.anya2 import court as C

COURT_FILE = "court.json"
REF_FILE = "ref.png"

# A clicked corner is good to ~2 px; the camera track's own jostle threshold
# is in the same range.  15 px at 960x540 is a visible move of the mount -- at
# the far baseline it is ~3 m of court -- and is past the point where silently
# correcting it is the honest thing to do.
MAX_SITE_SHIFT_PX = 15.0


class NeedsCalibration(RuntimeError):
    """The site profile cannot be carried onto this recording."""


def _ref_color(video):
    from pipeline.utilities import get_reference_frame
    img = get_reference_frame(video, target_idx=CAM.reference_index(video))
    return cv2.resize(img, C.ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)


def _source_size(video):
    from pipeline.utilities import probe_video
    p = probe_video(video)
    return [int(p["width"]), int(p["height"])]


def load(profile_dir):
    with open(os.path.join(profile_dir, COURT_FILE)) as fh:
        prof = json.load(fh)
    ref = cv2.imread(os.path.join(profile_dir, REF_FILE), cv2.IMREAD_COLOR)
    if ref is None:
        raise FileNotFoundError(f"no {REF_FILE} in {profile_dir}")
    return prof, ref


def save(video, profile_dir, prompt=True):
    """Write a profile from a calibrated video (clicking first if needed)."""
    cache = C.court_cache_path(video)
    if not os.path.isfile(cache):
        if not prompt:
            raise NeedsCalibration(f"{video} has no court calibration")
        from pipeline.utilities import init_court
        init_court(video, analysis_size=C.ANALYSIS_SIZE)
    with open(cache) as fh:
        court = json.load(fh)
    os.makedirs(profile_dir, exist_ok=True)
    ref = _ref_color(video)
    cv2.imwrite(os.path.join(profile_dir, REF_FILE), ref)

    from pipeline.anya2 import perceive as PC
    crop, _ = PC.far_band(video)
    prof = {"points": court["points"],
            "analysis_size": list(C.ANALYSIS_SIZE),
            "source_size": _source_size(video),
            "far_band": [int(v) for v in crop],
            "from_video": os.path.basename(video)}
    with open(os.path.join(profile_dir, COURT_FILE), "w") as fh:
        json.dump(prof, fh, indent=2)
    print(f"[SITE] profile -> {profile_dir}  (corners from {prof['from_video']}, "
          f"far band {prof['far_band']})")
    return prof


def register_corners(ref_gray, ref_pts, cur_gray):
    """Corners moved from `ref_gray` onto `cur_gray`, and how far they moved.

    Returns (corners [4,2] in cur_gray's pixels, max shift px); raises
    NeedsCalibration when the two frames do not register.
    """
    ref_pts = np.asarray(ref_pts, dtype=np.float64)
    h, w = ref_gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, ref_pts.astype(np.int32).reshape(-1, 1, 2), 255)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51)))
    mask = mask.astype(bool)

    def ground_mask_fn(dst):
        xi = np.clip(np.rint(dst[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(dst[:, 1]).astype(int), 0, h - 1)
        return mask[yi, xi]

    orb = cv2.ORB_create(nfeatures=CAM.ORB_FEATURES)
    kp_ref, des_ref = CAM._detect(orb, ref_gray)
    if des_ref is None:
        raise NeedsCalibration("the profile's reference frame has no features")
    # W maps the CURRENT frame onto the profile frame, the same direction the
    # camera track stores; the corners go the other way.
    W, n, _, _ = CAM.register(cur_gray, kp_ref, des_ref, ground_mask_fn, orb,
                              probe=ref_pts)
    if W is None:
        raise NeedsCalibration(
            "this recording's view does not match the site profile (the camera "
            "has moved or is pointing somewhere else) -- re-calibrate")
    cur = cv2.perspectiveTransform(ref_pts.reshape(-1, 1, 2),
                                   np.linalg.inv(W)).reshape(-1, 2)
    shift = float(np.hypot(*(cur - ref_pts).T).max())
    return cur, shift, n


def apply(profile_dir, video, max_shift_px=MAX_SITE_SHIFT_PX, force=False):
    """Seed `video`'s court cache from the profile. Returns the shift in px."""
    cache = C.court_cache_path(video)
    if os.path.isfile(cache) and not force:
        print(f"[SITE] {os.path.basename(cache)} already exists -- keeping it")
        return 0.0
    prof, ref = load(profile_dir)
    src = _source_size(video)
    if list(prof["source_size"]) != src:
        raise NeedsCalibration(
            f"the profile was made on {prof['source_size'][0]}x"
            f"{prof['source_size'][1]} footage but this is {src[0]}x{src[1]}")

    cur = _ref_color(video)
    corners, shift, n = register_corners(
        cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), prof["points"],
        cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY))
    if shift > max_shift_px:
        raise NeedsCalibration(
            f"the court has moved {shift:.0f} px since the site was calibrated "
            f"(limit {max_shift_px:.0f}) -- the mount was probably knocked; "
            f"re-calibrate")

    data = {"points": [[float(x), float(y)] for x, y in corners],
            "frame_shape": list(cur.shape),
            "analysis_size": list(C.ANALYSIS_SIZE),
            "video": os.path.basename(video),
            "site": {"profile": os.path.abspath(profile_dir),
                     "shift_px": round(shift, 2), "inliers": int(n)}}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"[SITE] corners carried onto {os.path.basename(video)} "
          f"(moved {shift:.1f} px, {n} inliers)")
    return shift


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fixed-camera site calibration.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save", help="make a profile from a video (clicks if "
                                    "the video is not calibrated yet)")
    s.add_argument("video")
    s.add_argument("profile_dir")
    s.add_argument("--export", choices=["ncnn", "onnx"], default=None,
                   help="also export the pose models this site needs")
    a = sub.add_parser("apply", help="seed a video's court cache from a profile")
    a.add_argument("profile_dir")
    a.add_argument("video")
    a.add_argument("--max-shift", type=float, default=MAX_SITE_SHIFT_PX)
    a.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "save":
        prof = save(args.video, args.profile_dir)
        if args.export:
            from pipeline.anya2 import export_pose as EP
            EP.export_for_site(prof, args.export)
    else:
        apply(args.profile_dir, args.video, args.max_shift, args.force)


if __name__ == "__main__":
    main()
