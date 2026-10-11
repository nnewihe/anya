"""
export_camera_check.py
======================
Frames and answers for the Swift camera-registration tests (CameraTests):

  synthetic/   camera.py's own self-test scene: the reference image and three
               jostled copies with the jostle A that made each (the Swift
               registration must undo A to within 1 px at the court corners)
  real/        a stretch of a real clip around its biggest camera jump: the
               calibration frame, every camera sample's 960x540 greyscale
               frame, main's cached warp W for each, and W_fresh: camera.py
               re-run on exactly these frames in a FRESH process

Why W_fresh: FLANN-LSH matching is randomised, so camera.py does not
reproduce its own cached track (3-6 px apart at the court corners between
runs on Data/78). Run in a fresh process, the random state is the same each
time -- and the same as the Swift app's, which links the same OpenCV.

    python ios/AnyaTennis/tools/export_camera_check.py OUT_DIR [CLIP_DIR] [--samples N]

CLIP_DIR defaults to Data/78 (bumped mid-match).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from pipeline import proxy                     # noqa: E402
from pipeline.anya2 import camera, court       # noqa: E402
from pipeline.anya2 import eval as anya2_eval  # noqa: E402


def synthetic(out):
    os.makedirs(out, exist_ok=True)
    w, h = court.ANALYSIS_SIZE
    ref, quad = camera._synthetic_court()
    cv2.imwrite(os.path.join(out, "reference.png"), ref)
    jostles = {
        "shift": [[1, 0, 12.0], [0, 1, -7.0], [0, 0, 1]],
        "roll": cv2.getRotationMatrix2D((w / 2, h / 2), 1.5, 1.0).tolist() + [[0, 0, 1]],
        "tilt": [[1.0, 0.0, 4.0], [0.0, 1.0, -9.0], [2e-5, 3e-5, 1.0]],
    }
    for name, A in jostles.items():
        jostled = cv2.warpPerspective(ref, np.asarray(A, np.float64), (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(os.path.join(out, f"{name}.png"), jostled)
    noise = np.random.default_rng(7).integers(0, 255, (h, w)).astype(np.uint8)
    cv2.imwrite(os.path.join(out, "noise.png"), noise)
    with open(os.path.join(out, "synthetic.json"), "w") as f:
        json.dump({"court_quad": quad.tolist(), "jostles": jostles}, f)


def real(out, clip_dir, n_samples):
    os.makedirs(out, exist_ok=True)
    video = anya2_eval.clip_video(clip_dir)
    track = camera.load(video, required=True)
    corners = camera.court_corners(video)
    shift = track.corner_shift(corners)
    jump = int(np.argmax(np.abs(np.diff(shift)))) + 1          # the biggest step between samples
    first = max(0, jump - n_samples // 2)
    rows = range(first, min(len(track.frames), first + n_samples))

    cv2.imwrite(os.path.join(out, "reference.png"), camera._reference_frame(video, track.ref_index))
    cap = cv2.VideoCapture(proxy.ensure_proxy(video, size=court.ANALYSIS_SIZE, crf=14, label="PROXY540"))
    samples = []
    for i in rows:
        frame = int(track.frames[i])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {frame}")
        name = f"frame_{frame:07d}.png"
        cv2.imwrite(os.path.join(out, name), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        samples.append({"file": name, "frame": frame, "W": track.W[i].reshape(-1).tolist(),
                        "ok": bool(track.ok[i])})
    cap.release()
    fresh = _fresh_track(out, samples, corners)
    for sample, W in zip(samples, fresh):
        sample["W_fresh"] = W.reshape(-1).tolist()
    with open(os.path.join(out, "real.json"), "w") as f:
        json.dump({"clip": os.path.basename(clip_dir.rstrip("/")), "court_corners": corners.tolist(),
                   "jump_sample": jump - first, "samples": samples}, f)
    print(f"real: {len(samples)} samples of {clip_dir}, biggest jump {np.abs(np.diff(shift)).max():.1f} px "
          f"at sample {jump - first}")


def _fresh_track(out, samples, corners):
    """camera.register + _fill_failed_samples + smooth over the exported frames
    (the track restricted to them), as the first registrations of this process."""
    w, h = court.ANALYSIS_SIZE
    reference = cv2.imread(os.path.join(out, "reference.png"), cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(nfeatures=camera.ORB_FEATURES)
    matcher = camera.make_matcher()
    points, descriptors = camera._detect(orb, reference)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32).reshape(-1, 1, 2), 255)
    k = 2 * 25 + 1
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))).astype(bool)

    def on_court(p):
        xi = np.clip(np.rint(p[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(p[:, 1]).astype(int), 0, h - 1)
        return mask[yi, xi]

    registered = camera._Samples([], [], [], [])
    for sample in samples:
        gray = cv2.imread(os.path.join(out, sample["file"]), cv2.IMREAD_GRAYSCALE)
        W, n, _, _ = camera.register(gray, points, descriptors, on_court, orb, matcher, probe=corners)
        registered.frames.append(sample["frame"])
        registered.warps.append(W)
        registered.ok.append(W is not None)
        registered.inliers.append(n)
    return camera.smooth(camera._fill_failed_samples(registered), corners)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("out")
    parser.add_argument("clip_dir", nargs="?", default=os.path.join(os.environ.get("ANYA_DATA", "/Volumes/Anya/Data"), "78"))
    parser.add_argument("--samples", type=int, default=40)
    args = parser.parse_args()
    synthetic(os.path.join(args.out, "synthetic"))
    real(os.path.join(args.out, "real"), args.clip_dir, args.samples)


if __name__ == "__main__":
    main()
