"""
camera.py
=========
The camera track: for each sampled frame, a warp W that maps that frame's
image back onto the calibration frame's image.  With it, court projections
stay right after the camera is bumped mid-match:

    court metres  =  H_ref  @  W_t  @  (image point at frame t)

H_ref comes from the four corners clicked once (court.py).  W_t is estimated
here by matching ORB features between frame t and the calibration frame.

How it works
------------
  1. Every SAMPLE_FPS-th frame of the 540p proxy is registered DIRECTLY against
     the calibration frame (never chained frame to frame, so errors cannot
     accumulate and a still camera stays exactly at identity).
  2. Each registration is a global RANSAC homography, optionally refined by a
     re-fit on the court-surface features alone ("ground-plane re-fit") when
     there are enough of them and the two fits agree.
  3. A sample that fails to register borrows its neighbour's warp.
  4. A running median over the track removes isolated bad samples while
     keeping a real jostle as a step.

Turn the whole thing off with the environment variable ANYA_CAMERA_TRACK=0:
nothing is estimated and nothing cached is read, so every projection uses
the clicked homography alone.

Output `<stem>_anya2_camera.npz`:
    frames      [N] int32    SOURCE frame index of each sample
    W           [N,3,3]      image(frame t) -> image(reference), analysis px
    ok          [N] bool     did this sample register on its own merit
    inliers     [N] int32    inliers behind the accepted warp
    ref_index   scalar int   the frame the corners were clicked on
    sample_step scalar int   source frames between samples
    src_fps, n_src_frames, analysis_size

Evidence and history: docs/anya2/NOTES.md, "camera.py".
"""

import argparse
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from pipeline import workdir
from pipeline.anya2 import court

# ── sampling ─────────────────────────────────────────────────────────────
SAMPLE_FPS = 5.0           # samples per second; a jostle is a step, not a signal

# ── feature matching ─────────────────────────────────────────────────────
ORB_FEATURES = 4000        # high, so the low-texture court still gets features
LOWE_RATIO = 0.75          # ratio test for a good match
# Approximate (FLANN-LSH) matching: 6x faster than brute force for the same
# answer.  Brute force is used as a retry when the approximate match fails.
FLANN_LSH_PARAMS = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)

# ── homography fit ───────────────────────────────────────────────────────
RANSAC_PX = 3.0
MIN_INLIERS = 30           # fewer than this and the fit is not evidence

# The ground-plane re-fit replaces the global fit only if it has at least this
# many inliers AND moves the court corners no more than GROUND_AGREE_PX from
# where the global fit puts them.
MIN_GROUND_INLIERS = 60
GROUND_AGREE_PX = 4.0

# ── sanity bounds on any accepted warp ───────────────────────────────────
MAX_SHIFT_PX = 240.0       # no frame corner may move more than this
AREA_LO, AREA_HI = 0.70, 1.45   # warped frame area / original area

# ── smoothing and reporting ──────────────────────────────────────────────
SMOOTH_SAMPLES = 5         # running-median width (1 s at 5 Hz); <3 disables
JOSTLE_PX = 3.0            # step in corner position reported as a "jostle"

_IDENTITY = np.eye(3, dtype=np.float64)


def enabled():
    """False when ANYA_CAMERA_TRACK is 0/false/no/off."""
    return os.environ.get("ANYA_CAMERA_TRACK", "1").strip().lower() not in (
        "0", "false", "no", "off")


def track_path(video_path, suffix="_anya2_camera.npz"):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(workdir.artifact_dir(video_path), f"{stem}{suffix}")


# ═════════════════════════════════════════════════════════════════════════
# Estimating the track
# ═════════════════════════════════════════════════════════════════════════

def estimate(video_path, force=False, sample_fps=SAMPLE_FPS, out=None,
             limit=None, verbose=True, on_progress=None):
    """Register the video against its calibration frame and cache the warps.

    Returns the cached npz path (None if tracking is turned off).  Reuses an
    existing file unless `force`.  `limit` stops after that many source frames.
    """
    from pipeline.utilities import probe_video

    if not enabled():
        if verbose:
            print("[camera] ANYA_CAMERA_TRACK=0 -- not tracking the camera")
        return None

    out = out or track_path(video_path)
    if os.path.isfile(out) and not force:
        if verbose:
            print(f"[camera] cached: {out}")
        return out

    info = probe_video(video_path)
    src_fps = float(info["fps"])
    n_src_frames = int(info["frame_count"])
    sample_step = max(1, int(round(src_fps / float(sample_fps))))

    reference = _load_reference(video_path)
    samples = _register_samples(video_path, reference, sample_step,
                                n_src_frames, limit, on_progress)
    W = _fill_failed_samples(samples)
    W = smooth(W, reference.court_corners)

    np.savez_compressed(
        out,
        frames=np.asarray(samples.frames, dtype=np.int32),
        W=W,
        ok=np.asarray(samples.ok, dtype=bool),
        inliers=np.asarray(samples.inliers, dtype=np.int32),
        ref_index=np.int32(reference.frame_index),
        sample_step=np.int32(sample_step),
        src_fps=np.float64(src_fps),
        n_src_frames=np.int32(n_src_frames),
        analysis_size=np.asarray(court.ANALYSIS_SIZE, dtype=np.int32),
    )
    if verbose:
        report(video_path, out)
    return out


@dataclass
class _Reference:
    """Everything about the calibration frame that registration needs."""
    frame_index: int
    court_corners: np.ndarray      # [4, 2] clicked corners, analysis px
    court_mask: np.ndarray         # [H, W] bool, court surface (dilated)
    points: np.ndarray             # ORB keypoint positions
    descriptors: np.ndarray        # ORB descriptors
    orb: object
    matcher: object

    def is_on_court(self, ref_points):
        """bool[M]: which reference-frame points fall on the court surface."""
        mask = self.court_mask
        xi = np.clip(np.rint(ref_points[:, 0]).astype(int), 0, mask.shape[1] - 1)
        yi = np.clip(np.rint(ref_points[:, 1]).astype(int), 0, mask.shape[0] - 1)
        return mask[yi, xi]


def _load_reference(video_path) -> _Reference:
    frame_index = reference_index(video_path)
    gray = _reference_frame(video_path, frame_index)
    court_mask = _court_mask(video_path)
    corners = court_corners(video_path)
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = make_matcher()
    points, descriptors = _detect(orb, gray)
    if descriptors is None:
        raise RuntimeError("no ORB features in the calibration frame; "
                           "the camera track cannot be anchored")
    return _Reference(frame_index, corners, court_mask, points, descriptors,
                      orb, matcher)


@dataclass
class _Samples:
    frames: list        # source frame index per sample
    warps: list         # W per sample, or None where registration failed
    ok: list
    inliers: list


def _register_samples(video_path, reference, sample_step, n_src_frames,
                      limit, on_progress) -> _Samples:
    """Read every `sample_step`-th proxy frame and register it."""
    from pipeline import cancel
    from pipeline import proxy

    # The same 540p proxy perceive.py uses.  If it could not be built
    # frame-exact, ensure_proxy returns the SOURCE; the resize below handles it.
    proxy_path = proxy.ensure_proxy(video_path, size=court.ANALYSIS_SIZE, crf=14,
                                    label="PROXY540")
    cap = cv2.VideoCapture(proxy_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {proxy_path}")

    samples = _Samples([], [], [], [])
    last_good_W, last_good_features = None, None
    n_wanted = (min(n_src_frames, limit) if limit else n_src_frames)
    frame = 0
    while True:
        # grab() skips decoding work for frames we do not sample.
        if not cap.grab():
            break
        if limit and frame >= limit:
            break
        if frame % sample_step:
            frame += 1
            continue
        ok, image = cap.retrieve()
        if not ok:
            frame += 1
            continue
        cancel.check()
        if image.shape[1] != court.ANALYSIS_SIZE[0] or image.shape[0] != court.ANALYSIS_SIZE[1]:
            image = cv2.resize(image, court.ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        W, n_inliers, points, descriptors = register(
            gray, reference.points, reference.descriptors, reference.is_on_court,
            reference.orb, reference.matcher, probe=reference.court_corners)

        if W is None and last_good_W is not None:
            # Fallback: match against the last sample that DID register, and
            # compose with its warp.  One link only, so nothing accumulates.
            W, n_inliers = _register_via_last_good(
                reference.matcher, points, descriptors, last_good_W, last_good_features)

        samples.frames.append(frame)
        if W is not None:
            samples.warps.append(W)
            samples.ok.append(True)
            samples.inliers.append(n_inliers)
            last_good_W = W
            last_good_features = (points, descriptors)
        else:
            samples.warps.append(None)
            samples.ok.append(False)
            samples.inliers.append(0)

        if on_progress and n_wanted:
            on_progress(min(1.0, frame / float(n_wanted)))
        frame += 1
    cap.release()

    if not samples.frames:
        raise RuntimeError("no frames sampled for the camera track")
    return samples


def _register_via_last_good(matcher, points, descriptors, last_good_W, last_good_features):
    """Warp to the reference by way of the last sample that registered:
    (last_good_W @ link, n_inliers), or (None, 0).

    Like `register`, the approximate match is retried with brute force if
    it fails.
    """
    good_points, good_descriptors = last_good_features
    everything = lambda d: np.ones(len(d), dtype=bool)
    for each_matcher in (matcher, make_matcher(approximate=False)):
        src, dst = _match(each_matcher, descriptors, good_descriptors, points, good_points)
        W_link, n_link = _fit(src, dst, everything)
        if W_link is not None:
            composed = last_good_W @ W_link
            if _sane(composed):
                return composed, n_link
            return None, 0
        if isinstance(matcher, cv2.BFMatcher):
            break
    return None, 0


def _fill_failed_samples(samples) -> np.ndarray:
    """[N,3,3] warps with every failed sample filled from a neighbour.

    Forward-fill from the last good warp; the leading failures (before any good
    one) take the FIRST good warp.  Identity is never used as a filler: after a
    jostle it would claim the camera is back where it started.
    """
    n = len(samples.frames)
    W = np.empty((n, 3, 3), dtype=np.float64)
    ok = np.asarray(samples.ok, dtype=bool)
    first_good = int(np.argmax(ok)) if ok.any() else n
    last = None
    for i in range(n):
        if samples.warps[i] is not None:
            last = samples.warps[i]
        W[i] = last if last is not None else _IDENTITY
    if 0 < first_good < n:
        W[:first_good] = samples.warps[first_good]
    return W


def smooth(W, court_corners_px, k=SMOOTH_SAMPLES):
    """Running median over the track, applied to point positions.

    A homography is fully defined by where a few points go, so: map 8 points
    (frame corners + court corners) through every warp, take the running
    median of their positions, and re-fit each warp from the medians.  The
    window "sticks" at the ends (indices 0 and 1 share window 0..k-1) instead
    of repeating the edge sample.
    """
    n = len(W)
    if k < 3 or n < k:
        return W
    if k % 2 == 0:
        k += 1
    points = np.concatenate([_frame_corners(), np.asarray(court_corners_px, dtype=np.float64)])
    W_inv = np.linalg.inv(W)
    located = np.stack([cv2.perspectiveTransform(points.reshape(-1, 1, 2), W_inv[i]).reshape(-1, 2)
                        for i in range(n)])                               # [n, P, 2]
    half = k // 2
    window_start = np.clip(np.arange(n) - half, 0, n - k)
    median = np.median(located[window_start[:, None] + np.arange(k)], axis=1)   # [n, P, 2]
    out = np.empty_like(W)
    for i in range(n):
        # method=0: plain least squares -- these points are already a consensus.
        M, _ = cv2.findHomography(median[i].astype(np.float64), points, 0)
        out[i] = M if (M is not None and _sane(M)) else W[i]
    return out


# ═════════════════════════════════════════════════════════════════════════
# Registering one frame
# ═════════════════════════════════════════════════════════════════════════

def make_matcher(approximate=True):
    if approximate:
        return cv2.FlannBasedMatcher(FLANN_LSH_PARAMS, dict(checks=32))
    return cv2.BFMatcher(cv2.NORM_HAMMING)


def register(gray, pts_ref, des_ref, ground_mask_fn, orb=None, matcher=None,
             probe=None):
    """Warp mapping `gray` onto the reference frame.

    Returns (W or None, n_inliers, this frame's keypoint positions, descriptors).
    Retries with exact (brute-force) matching if the approximate match fails.
    """
    orb = orb or cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = matcher or make_matcher()
    points, descriptors = _detect(orb, gray)
    src, dst = _match(matcher, descriptors, des_ref, points, pts_ref)
    W, n_inliers = _fit(src, dst, ground_mask_fn, probe)
    if W is None and not isinstance(matcher, cv2.BFMatcher):
        src, dst = _match(make_matcher(approximate=False), descriptors, des_ref,
                          points, pts_ref)
        W, n_inliers = _fit(src, dst, ground_mask_fn, probe)
    return W, n_inliers, points, descriptors


def _detect(orb, gray):
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 8:
        return np.zeros((0, 2), np.float32), None
    return np.array([k.pt for k in keypoints], dtype=np.float32), descriptors


def _match(matcher, des_cur, des_ref, pts_cur, pts_ref):
    """Ratio-tested correspondences: (points in this frame, points in reference)."""
    empty = (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32))
    if des_cur is None or des_ref is None or len(des_cur) < 2 or len(des_ref) < 2:
        return empty
    current, reference = [], []
    for pair in matcher.knnMatch(des_cur, des_ref, k=2):
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < LOWE_RATIO * second.distance:
            current.append(pts_cur[best.queryIdx])
            reference.append(pts_ref[best.trainIdx])
    if not current:
        return empty
    return np.asarray(current, np.float32), np.asarray(reference, np.float32)


def _fit(src, dst, ground_mask_fn, probe=None):
    """Global RANSAC homography, then the ground-plane re-fit if it earns it.

    `ground_mask_fn(dst) -> bool[M]` says which matches lie on the court in the
    reference frame.  `probe` ([K,2] court corners) is where the two fits are
    compared; without it the global fit is returned as is.
    Returns (W, n_inliers) or (None, 0).
    """
    if len(src) < MIN_INLIERS:
        return None, 0
    W_global, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX,
                                               maxIters=5000, confidence=0.999)
    if W_global is None or inlier_mask is None:
        return None, 0
    inliers = inlier_mask.ravel().astype(bool)
    n_global = int(inliers.sum())
    if n_global < MIN_INLIERS or not _sane(W_global):
        return None, 0
    if probe is None:
        return W_global, n_global

    on_court = inliers & ground_mask_fn(dst)
    if int(on_court.sum()) < MIN_GROUND_INLIERS:
        return W_global, n_global
    W_ground, ground_mask = cv2.findHomography(src[on_court], dst[on_court],
                                               cv2.RANSAC, RANSAC_PX,
                                               maxIters=5000, confidence=0.999)
    if W_ground is None or ground_mask is None or not _sane(W_ground):
        return W_global, n_global
    n_ground = int(ground_mask.ravel().sum())
    if n_ground < MIN_GROUND_INLIERS:
        return W_global, n_global
    # Agreement gate: the re-fit may refine the global fit, not replace it.
    corners = np.asarray(probe, dtype=np.float64).reshape(-1, 1, 2)
    by_global = cv2.perspectiveTransform(corners, W_global).reshape(-1, 2)
    by_ground = cv2.perspectiveTransform(corners, W_ground).reshape(-1, 2)
    if float(np.hypot(*(by_global - by_ground).T).max()) > GROUND_AGREE_PX:
        return W_global, n_global
    return W_ground, n_ground


def _sane(W):
    """Reject a warp that folds, collapses or flings the frame.

    Not a quality score: a wrong fit is wildly wrong, and this only has to
    catch that.
    """
    if W is None or not np.all(np.isfinite(W)):
        return False
    w, h = court.ANALYSIS_SIZE
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    warped = cv2.perspectiveTransform(frame.reshape(-1, 1, 2), W).reshape(-1, 2)
    if not np.all(np.isfinite(warped)):
        return False
    if float(np.abs(warped - frame).max()) > MAX_SHIFT_PX:
        return False
    # Signed (shoelace) area: negative means the quad was folded over.
    x, y = warped[:, 0], warped[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return AREA_LO <= area / float(w * h) <= AREA_HI


# ── the calibration frame ────────────────────────────────────────────────

def reference_index(video_path, target_idx=300):
    """The frame `utilities.init_court` showed the user for calibration.

    Must match `utilities.get_reference_frame`'s rule (frame 300, or the middle
    of a shorter clip).  If that rule changes, change this too.
    """
    from pipeline.utilities import probe_video
    total = int(probe_video(video_path)["frame_count"])
    if total <= 0:
        return 0
    return int(min(target_idx, total // 2))


def _reference_frame(video_path, frame_index):
    """The calibration frame from the SOURCE video, greyscale, at ANALYSIS_SIZE."""
    from pipeline.utilities import get_reference_frame
    image = get_reference_frame(video_path, target_idx=frame_index)
    image = cv2.resize(image, court.ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def court_corners(video_path):
    """The four clicked corners, analysis px.  [4, 2]."""
    with open(court.court_cache_path(video_path)) as fh:
        return np.array(json.load(fh)["points"], dtype=np.float64)


def _court_mask(video_path, pad_px=25):
    """[H, W] bool: the clicked court quad, dilated by `pad_px` (lines sit at its edge)."""
    corners = court_corners(video_path).astype(np.int32)
    w, h = court.ANALYSIS_SIZE
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, corners.reshape(-1, 1, 2), 255)
    k = 2 * int(pad_px) + 1
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return mask.astype(bool)


def _frame_corners():
    w, h = court.ANALYSIS_SIZE
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════
# Reading the track
# ═════════════════════════════════════════════════════════════════════════

class CameraTrack:
    """Sampled warps, looked up by SOURCE frame number."""

    def __init__(self, frames, W, ok, inliers, ref_index, src_fps,
                 sample_step=1):
        self.frames = np.asarray(frames, dtype=np.int64)
        self.W = np.asarray(W, dtype=np.float64)
        self.ok = np.asarray(ok, dtype=bool)
        self.inliers = np.asarray(inliers, dtype=np.int64)
        self.ref_index = int(ref_index)
        self.src_fps = float(src_fps)
        self.sample_step = int(sample_step)
        self._inv = None

    @classmethod
    def identity(cls, src_fps=30.0):
        """The track of a camera that never moved."""
        return cls(frames=[0], W=[_IDENTITY], ok=[True], inliers=[0],
                   ref_index=0, src_fps=src_fps, sample_step=1)

    @property
    def is_identity(self):
        return len(self.frames) == 1 and np.allclose(self.W[0], _IDENTITY)

    def index_at(self, src_frame):
        """Index of the sample NEAREST `src_frame` (ties go to the earlier one).

        Nearest rather than interpolated: interpolation would invent geometry
        the camera never had.
        """
        i = int(np.searchsorted(self.frames, src_frame))
        if i <= 0:
            return 0
        if i >= len(self.frames):
            return len(self.frames) - 1
        after, before = self.frames[i] - src_frame, src_frame - self.frames[i - 1]
        return i if after < before else i - 1

    def warp_at(self, src_frame):
        return self.W[self.index_at(src_frame)]

    def inv_at(self, src_frame):
        if self._inv is None:
            self._inv = np.linalg.inv(self.W)
        return self._inv[self.index_at(src_frame)]

    def corner_shift(self, corners):
        """Per sample, how far (px, max over corners) the court corners have moved."""
        c = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
        W_inv = np.linalg.inv(self.W)
        out = np.empty(len(self.W))
        for i in range(len(self.W)):
            moved = cv2.perspectiveTransform(c, W_inv[i]).reshape(-1, 2)
            out[i] = float(np.hypot(*(moved - c.reshape(-1, 2)).T).max())
        return out


def load(video_path, path=None, required=False):
    """The cached track, or the identity track when there is none (or it is off)."""
    p = path or track_path(video_path)
    if not enabled() or not os.path.isfile(p):
        if required:
            raise FileNotFoundError(
                f"no camera track at {p}"
                if enabled() else "camera tracking is off (ANYA_CAMERA_TRACK=0)")
        from pipeline.utilities import probe_video
        try:
            fps = float(probe_video(video_path)["fps"])
        except Exception:
            fps = 30.0
        return CameraTrack.identity(src_fps=fps)
    z = np.load(p, allow_pickle=False)
    return CameraTrack(frames=z["frames"], W=z["W"], ok=z["ok"],
                       inliers=z["inliers"], ref_index=int(z["ref_index"]),
                       src_fps=float(z["src_fps"]),
                       sample_step=int(z["sample_step"]))


# ── diagnostics ──────────────────────────────────────────────────────────

def jostles(track, corners, step_px=JOSTLE_PX):
    """Sample-to-sample jumps in court-corner position, biggest first."""
    shift = track.corner_shift(corners)
    step = np.abs(np.diff(shift))
    events = [{"frame": int(track.frames[i + 1]),
               "t": float(track.frames[i + 1] / track.src_fps),
               "step_px": float(step[i]),
               "shift_px": float(shift[i + 1])} for i in np.where(step > step_px)[0]]
    events.sort(key=lambda e: -e["step_px"])
    return events


def report(video_path, path=None):
    """Print what the track says: shift in px, and the court error in metres a
    FIXED homography would have made."""
    track = load(video_path, path)
    corners = court_corners(video_path)
    shift = track.corner_shift(corners)
    H = court.load_homography(video_path)
    truth = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    W_inv = np.linalg.inv(track.W)
    error_m = np.empty(len(track.W))
    for i in range(len(track.W)):
        moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), W_inv[i]).reshape(-1, 2)
        fixed = cv2.perspectiveTransform(moved.reshape(-1, 1, 2), H).reshape(-1, 2)
        error_m[i] = float(np.hypot(*(fixed - truth).T).max())

    print(f"[camera] {len(track.frames)} samples, anchored on frame {track.ref_index}, "
          f"registered {100.0 * track.ok.mean():.1f}%")
    print(f"[camera] court-corner shift px: median {np.median(shift):.1f}  "
          f"p95 {np.percentile(shift, 95):.1f}  max {shift.max():.1f}")
    print(f"[camera] court error under the FIXED homography, metres: "
          f"median {np.median(error_m):.2f}  p95 {np.percentile(error_m, 95):.2f}  "
          f"max {error_m.max():.2f}")
    events = jostles(track, corners)
    if not events:
        print("[camera] no jostle above "
              f"{JOSTLE_PX:.0f} px -- a fixed homography would have been fine")
    for e in events[:10]:
        print(f"[camera]   jostle at {e['t']:8.2f}s (frame {e['frame']}): "
              f"step {e['step_px']:.1f} px, now {e['shift_px']:.1f} px off")
    return track


# ═════════════════════════════════════════════════════════════════════════
# Self-test (synthetic, no video):  python -m pipeline.anya2.camera --self-test
# ═════════════════════════════════════════════════════════════════════════

def _synthetic_court(seed=0):
    """A 960x540 image with court lines in the middle and clutter at the edges."""
    rng = np.random.default_rng(seed)
    w, h = court.ANALYSIS_SIZE
    img = np.full((h, w), 70, np.uint8)
    img[:] = cv2.GaussianBlur(rng.integers(50, 90, (h, w)).astype(np.uint8), (5, 5), 0)
    quad = np.array([[210, 470], [760, 470], [610, 180], [350, 180]], np.int32)
    cv2.fillConvexPoly(img, quad, 95)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        cv2.line(img, tuple(quad[a]), tuple(quad[b]), 235, 2, cv2.LINE_AA)
    cv2.line(img, (280, 325), (690, 325), 235, 2, cv2.LINE_AA)   # service line
    cv2.line(img, (480, 470), (480, 180), 235, 1, cv2.LINE_AA)   # centre
    for _ in range(900):                                          # edge clutter
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        if cv2.pointPolygonTest(quad, (x, y), False) > 0 and rng.random() < 0.75:
            continue
        cv2.circle(img, (x, y), int(rng.integers(1, 4)), int(rng.integers(0, 255)), -1)
    return img, quad.astype(np.float64)


def _self_test():
    w, h = court.ANALYSIS_SIZE
    ref, quad = _synthetic_court()
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = make_matcher()
    pts_ref, des_ref = _detect(orb, ref)
    court_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(court_mask, quad.astype(np.int32), 255)
    court_mask = cv2.dilate(court_mask, np.ones((51, 51), np.uint8)).astype(bool)

    def on_court(dst):
        xi = np.clip(np.rint(dst[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(dst[:, 1]).astype(int), 0, h - 1)
        return court_mask[yi, xi]

    fails = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not cond:
            fails.append(name)

    print("[self-test] registration recovers a known jostle")
    for label, W_true in (
        ("pure shift  (12, -7) px",
         np.array([[1, 0, 12.0], [0, 1, -7.0], [0, 0, 1]])),
        ("shift + 1.5 deg roll",
         cv2.getRotationMatrix2D((w / 2, h / 2), 1.5, 1.0).tolist() + [[0, 0, 1]]),
        ("tilt (perspective)",
         np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -9.0], [2e-5, 3e-5, 1.0]])),
    ):
        # W_true maps reference -> jostled; the estimate must be its inverse.
        A = np.asarray(W_true, dtype=np.float64)
        jostled = cv2.warpPerspective(ref, A, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
        W, n, _, _ = register(jostled, pts_ref, des_ref, on_court, orb, matcher, probe=quad)
        if W is None:
            check(label, False, "did not register")
            continue
        moved = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), A)
        back = cv2.perspectiveTransform(moved, W).reshape(-1, 2)
        err = float(np.hypot(*(back - quad).T).max())
        check(label, err < 1.0, f"max court-corner error {err:.2f} px, {n} inliers")

    print("[self-test] a static camera is the identity")
    W, n, _, _ = register(ref, pts_ref, des_ref, on_court, orb, matcher, probe=quad)
    err = float(np.abs(cv2.perspectiveTransform(quad.reshape(-1, 1, 2), W)
                       .reshape(-1, 2) - quad).max())
    check("unmoved frame -> identity", err < 0.5, f"max {err:.3f} px")

    print("[self-test] garbage is rejected, not accepted quietly")
    noise = np.random.default_rng(7).integers(0, 255, (h, w)).astype(np.uint8)
    W, n, _, _ = register(noise, pts_ref, des_ref, on_court, orb, matcher, probe=quad)
    check("unrelated frame -> no warp", W is None, f"inliers {n}")
    check("_sane rejects a collapsed quad",
          not _sane(np.array([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1.0]])))
    check("_sane rejects a folded quad",
          not _sane(np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])))
    check("_sane accepts identity", _sane(_IDENTITY))

    print("[self-test] lookup takes the NEAREST sample")
    track = CameraTrack(frames=[0, 100, 200], W=[_IDENTITY] * 3, ok=[1, 1, 1],
                        inliers=[0, 0, 0], ref_index=0, src_fps=30.0, sample_step=100)
    check("index_at before the start", track.index_at(-5) == 0)
    check("index_at past the end", track.index_at(9999) == 2)
    check("index_at rounds down below the midpoint", track.index_at(149) == 1)
    check("index_at rounds up above the midpoint", track.index_at(151) == 2)

    print("[self-test] Geometry undoes a jostle end to end")
    H_ref = np.array([[0.02, 0.0, -4.0], [0.0, -0.05, 24.0], [0.0, -1e-3, 1.0]])
    A = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -7.0], [0.0, 0.0, 1.0]])
    W_t = np.linalg.inv(A)                       # frame -> reference
    p_ref = np.array([[480.0, 300.0]])
    p_now = cv2.perspectiveTransform(p_ref.reshape(-1, 1, 2), A).reshape(-1, 2)
    truth = court.to_court(H_ref, p_ref)[0]
    fixed = court.to_court(H_ref, p_now)[0]
    tracked = court.to_court(H_ref @ W_t, p_now)[0]
    e_fixed = float(np.hypot(*(fixed - truth)))
    e_tracked = float(np.hypot(*(tracked - truth)))
    # H_ref is invented, so only the RELATION is asserted: fixed is off, tracked is not.
    check("fixed homography carries the jostle into court metres",
          e_fixed > 0.1, f"off by {e_fixed:.2f} m")
    check("tracked homography does not", e_tracked < e_fixed / 1e6,
          f"off by {e_tracked:.2e} m")

    print(f"\n[self-test] {'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return not fails


def main():
    ap = argparse.ArgumentParser(description="The camera track: a per-frame "
                                 "image -> reference-image warp")
    ap.add_argument("video", nargs="?")
    ap.add_argument("--self-test", action="store_true",
                    help="synthetic check of the registration math; no video")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="report an existing track without re-estimating")
    ap.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many SOURCE frames")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(0 if _self_test() else 1)
    if not a.video:
        ap.error("a video is required unless --self-test")
    if a.report and os.path.isfile(track_path(a.video)) and not a.force:
        report(a.video)
        return
    estimate(a.video, force=a.force, sample_fps=a.sample_fps, limit=a.limit)


if __name__ == "__main__":
    main()
