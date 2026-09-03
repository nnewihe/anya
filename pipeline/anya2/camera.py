"""
camera.py
=========
The camera track: a per-frame image -> reference-image warp, so every court
projection in anya2 keeps working after the camera is bumped mid-match.

THE PROBLEM
-----------
Calibration is four corners clicked ONCE, on one reference frame, and cached
(`pipeline.utilities.init_court`).  `walking.court.load_homography` turns them
into a single homography that is then applied to every frame of the video.  That
is correct exactly as long as the camera does not move.

When it does move -- a bumped tripod, a knocked fence mount, someone leaning on
the netting -- nothing errors.  The corners still load, the homography still
inverts, every projection still returns a number.  The numbers are simply wrong
from that frame on, and they are wrong in the direction that breaks the most: at
far-court depth the 960x540 analysis frame is only ~4-5 px per court metre, so a
20 px jostle moves the far baseline by four metres.  `court.side()` starts
calling far players near, `court.in_bounds()` starts rejecting players who are
standing on the court, and the far-serve detector gates on a `court_y` that no
longer means what it meant at calibration time.

THE FIX
-------
Keep the single clicked calibration.  Make the MAP from image to that
calibration time-varying instead:

    court metres  =  H_ref  @  W_t  @  (image point at frame t)

`H_ref` is the existing cached homography, unchanged and still clicked once.
`W_t` warps frame t's image back onto the reference frame the corners were
clicked on, and is estimated here by registering frame t against that reference
frame.

DIRECTLY, not frame to frame.  Chained inter-frame warps accumulate drift over
tens of thousands of frames, and accumulated drift is indistinguishable from the
thing being corrected.  Registering every sample against the ONE reference means
an error in one sample cannot contaminate any other, and a camera that never
moves produces identity warps forever with no slow wander.  Chaining exists here
only as a fallback, for a sample that fails to match the reference at all.

WHY GROUND-PLANE INLIERS ARE RE-FIT SEPARATELY
----------------------------------------------
A homography maps one image of a scene to another exactly when the camera
rotates about its optical centre (which is what a tripod jostle mostly is: pan,
tilt, roll) OR when everything it sees lies on one plane.  A real bump usually
translates the camera a little as well, and then no single homography fits the
whole scene -- the fence, the stands and the court surface each want a different
one, and a global fit lands somewhere between them.

Downstream, `W_t` is only ever used to project GROUND points: a bounding box's
bottom centre, a ball's image position.  So after the global fit, the inliers
whose reference position falls inside the court polygon are re-fit on their own.
Those are the court-surface features, and the homography they agree on is the
ground plane's -- the one the projection actually needs.  When there are too few
of them the global fit is kept, which for a pure rotation is the same answer.

WHAT IT COSTS
-------------
It reads the 540p proxy `perceive` already builds, at `SAMPLE_FPS` rather than
every frame, so it adds no decode of the source video and no model call.  ORB
plus a RANSAC homography at 960x540 is roughly 10 ms a sample.

Sampling below the frame rate is not an approximation of a moving camera; it is
the right shape for a camera that is STILL except for a handful of instants.
The cost of landing a step edge up to `1 / SAMPLE_FPS` late is one fifth of a
second of stale geometry, against re-estimating a warp that is identity for
99.9% of the match.

Output `<stem>_anya2_camera.npz`:
    frames      [N] int32    SOURCE frame index of each sample
    W           [N,3,3]      image(frame t) -> image(reference), analysis px
    ok          [N] bool     did this sample register on its own merit
    inliers     [N] int32    ground-plane inliers behind the accepted warp
    ref_index   scalar int   the frame the corners were clicked on
    sample_step scalar int   source frames between samples
    src_fps, n_src_frames, analysis_size
"""

import argparse
import os

import cv2
import numpy as np

from pipeline import workdir as WD
from pipeline.anya2 import court as C

# ── sampling ─────────────────────────────────────────────────────────────
SAMPLE_FPS = 5.0        # see "WHAT IT COSTS" -- a jostle is a step, not a signal

# ── registration ─────────────────────────────────────────────────────────
ORB_FEATURES = 4000     # a tennis frame is texture-poor in the middle and rich
                        # at the edges (fence, stands, sponsor boards, lines).
                        # The count has to be high enough that the COURT still
                        # contributes after the edges have taken their share.
LOWE_RATIO = 0.75
RANSAC_PX = 3.0
MIN_INLIERS = 30        # below this the fit is not evidence of anything
MIN_GROUND_INLIERS = 20 # ...to prefer the ground-plane re-fit over the global

# ── sanity bounds on an accepted warp ────────────────────────────────────
# A homography fit to mismatched features is usually not subtly wrong, it is
# wildly wrong -- a collapsed or inverted quad.  These reject that without
# pretending to know how hard the camera was hit.
MAX_SHIFT_PX = 240.0    # a frame corner may not move more than a quarter frame
AREA_LO, AREA_HI = 0.70, 1.45

# ── what counts as a jostle, for reporting ───────────────────────────────
JOSTLE_PX = 3.0         # sample-to-sample step in court-corner position

_IDENTITY = np.eye(3, dtype=np.float64)

# ── the kill switch ──────────────────────────────────────────────────────
# ONE mechanism, checked in the two functions that can turn a track on: nothing
# estimates a track and nothing reads a cached one.  It is an environment
# variable rather than a config field because it has to reach code that never
# sees the config -- `tracks.build` and `perceive.far_band` are both callable
# straight from the CLI -- and because the state it disables lives on DISK: a
# config flag would leave a stale npz on disk still being read by every caller
# that did not get the flag.  `ANYA_ENGINE` in the desktop app is the same idiom.
#
#     ANYA_CAMERA_TRACK=0   every projection uses the clicked homography alone,
#                           which is exactly the behaviour before this existed.


def enabled():
    return os.environ.get("ANYA_CAMERA_TRACK", "1").strip().lower() not in (
        "0", "false", "no", "off")



def track_path(video_path, suffix="_anya2_camera.npz"):
    d = WD.artifact_dir(video_path)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}{suffix}")


def reference_index(video_path, target_idx=300):
    """The frame `init_court` showed the user.

    Duplicated from `utilities.get_reference_frame` rather than imported,
    because that function returns the IMAGE and not the index, and the index is
    what the whole track is anchored to.  If that rule ever changes, this is the
    other place it has to change.
    """
    from pipeline.utilities import probe_video
    total = int(probe_video(video_path)["frame_count"])
    if total <= 0:
        return 0
    return int(min(target_idx, total // 2))


def _reference_frame(video_path, ref_idx):
    """The calibration frame, greyscale, at ANALYSIS_SIZE.

    Read from the SOURCE, the same way `init_court` read it, so the pixels the
    warps are anchored to are the pixels the corners were clicked on.  One 4K
    seek, once per video.
    """
    from pipeline.utilities import get_reference_frame
    img = get_reference_frame(video_path, target_idx=ref_idx)
    img = cv2.resize(img, C.ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _court_mask(video_path, pad_px=25):
    """Boolean [H,W] mask of the court surface in the reference frame.

    The cached corners are already a quadrilateral in analysis pixels, which is
    exactly the ground-plane region -- there is nothing to detect.  It is
    dilated because a feature ON a line sits at the line's edge, and the corner
    clicks are human.
    """
    import json
    with open(C.court_cache_path(video_path)) as fh:
        pts = np.array(json.load(fh)["points"], dtype=np.int32)
    w, h = C.ANALYSIS_SIZE
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(m, pts.reshape(-1, 1, 2), 255)
    k = 2 * int(pad_px) + 1
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return m.astype(bool)


def _detect(orb, gray):
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 8:
        return np.zeros((0, 2), np.float32), None
    return np.array([k.pt for k in kp], dtype=np.float32), des


def _match(matcher, des_cur, des_ref, pts_cur, pts_ref):
    """Ratio-tested correspondences. Returns (src_cur [M,2], dst_ref [M,2])."""
    if des_cur is None or des_ref is None or len(des_cur) < 2 or len(des_ref) < 2:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pairs = matcher.knnMatch(des_cur, des_ref, k=2)
    a, b = [], []
    for p in pairs:
        if len(p) < 2:
            continue
        m, n = p
        if m.distance < LOWE_RATIO * n.distance:
            a.append(pts_cur[m.queryIdx])
            b.append(pts_ref[m.trainIdx])
    if not a:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.asarray(a, np.float32), np.asarray(b, np.float32)


def _sane(W):
    """Reject a fit that is not a plausible view of the same court.

    Not a quality measure -- a wrong fit here is a collapsed or folded quad, not
    a slightly-off one, and this only has to catch that.
    """
    if W is None or not np.all(np.isfinite(W)):
        return False
    w, h = C.ANALYSIS_SIZE
    box = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    out = cv2.perspectiveTransform(box.reshape(-1, 1, 2), W).reshape(-1, 2)
    if not np.all(np.isfinite(out)):
        return False
    if float(np.abs(out - box).max()) > MAX_SHIFT_PX:
        return False
    # Shoelace area, signed: a negative area means the quad was folded over.
    x, y = out[:, 0], out[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    ratio = area / float(w * h)
    return AREA_LO <= ratio <= AREA_HI


def _fit(src, dst, ground_mask_fn):
    """Global RANSAC homography, then the ground-plane re-fit.

    `ground_mask_fn(dst_points) -> bool[M]` says which correspondences land on
    the court in the REFERENCE frame.  Returns (W, n_inliers) or (None, 0).
    """
    if len(src) < MIN_INLIERS:
        return None, 0
    W, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX,
                                maxIters=5000, confidence=0.999)
    if W is None or inl is None:
        return None, 0
    inl = inl.ravel().astype(bool)
    n = int(inl.sum())
    if n < MIN_INLIERS or not _sane(W):
        return None, 0

    on_court = inl & ground_mask_fn(dst)
    if int(on_court.sum()) >= MIN_GROUND_INLIERS:
        Wg, inl_g = cv2.findHomography(src[on_court], dst[on_court],
                                       cv2.RANSAC, RANSAC_PX,
                                       maxIters=5000, confidence=0.999)
        if Wg is not None and inl_g is not None and _sane(Wg):
            ng = int(inl_g.ravel().sum())
            if ng >= MIN_GROUND_INLIERS:
                return Wg, ng
    return W, n


def register(gray, pts_ref, des_ref, ground_mask_fn, orb=None, matcher=None):
    """Warp mapping `gray` onto the reference frame, or (None, 0).

    Factored out of `estimate` so the registration can be exercised on a
    synthetic pair with a KNOWN answer -- see `_self_test`.  A geometry routine
    whose only test is "the reel looked better" is a routine nobody can change.
    """
    orb = orb or cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = matcher or cv2.BFMatcher(cv2.NORM_HAMMING)
    pts_cur, des_cur = _detect(orb, gray)
    src, dst = _match(matcher, des_cur, des_ref, pts_cur, pts_ref)
    W, n = _fit(src, dst, ground_mask_fn)
    return W, n, pts_cur, des_cur


def estimate(video_path, force=False, sample_fps=SAMPLE_FPS, out=None,
             limit=None, verbose=True, on_progress=None):
    """Register the video against its calibration frame; cache the warps.

    Idempotent: returns the cached path unless `force`.
    """
    from pipeline import cancel
    from pipeline import proxy as P
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
    n_src = int(info["frame_count"])
    step = max(1, int(round(src_fps / float(sample_fps))))

    ref_idx = reference_index(video_path)
    ref_gray = _reference_frame(video_path, ref_idx)
    court = _court_mask(video_path)

    def ground_mask_fn(dst):
        xi = np.clip(np.rint(dst[:, 0]).astype(int), 0, court.shape[1] - 1)
        yi = np.clip(np.rint(dst[:, 1]).astype(int), 0, court.shape[0] - 1)
        return court[yi, xi]

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pts_ref, des_ref = _detect(orb, ref_gray)
    if des_ref is None:
        raise RuntimeError("no ORB features in the calibration frame; "
                           "the camera track cannot be anchored")

    # The proxy `perceive` builds anyway. If it could not be made frame-exact,
    # ensure_proxy hands back the SOURCE, which is still correct here -- the
    # resize below is unconditional for exactly that case.
    prox = P.ensure_proxy(video_path, size=C.ANALYSIS_SIZE, crf=14,
                          label="PROXY540")
    cap = cv2.VideoCapture(prox)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {prox}")

    frames, Ws, oks, inls = [], [], [], []
    prev_ok_W, prev_kp = None, None
    f = 0
    n_want = (min(n_src, limit) if limit else n_src)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if limit and f >= limit:
            break
        if f % step:
            f += 1
            continue
        cancel.check()
        if img.shape[1] != C.ANALYSIS_SIZE[0] or img.shape[0] != C.ANALYSIS_SIZE[1]:
            img = cv2.resize(img, C.ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        W, n, pts_cur, des_cur = register(gray, pts_ref, des_ref,
                                          ground_mask_fn, orb, matcher)

        if W is None and prev_ok_W is not None:
            # Fallback: register against the last sample that DID match the
            # reference, and compose. One link, never a chain -- the anchor is
            # still a reference-registered frame, so nothing accumulates.
            src2, dst2 = _match(matcher, des_cur, prev_kp[1], pts_cur, prev_kp[0])
            W2, n2 = _fit(src2, dst2,
                          lambda d: np.ones(len(d), dtype=bool))
            if W2 is not None:
                cand = prev_ok_W @ W2
                if _sane(cand):
                    W, n = cand, n2

        if W is not None:
            frames.append(f)
            Ws.append(W)
            oks.append(True)
            inls.append(n)
            prev_ok_W = W
            prev_kp = (pts_cur, des_cur)
        else:
            frames.append(f)
            Ws.append(None)
            oks.append(False)
            inls.append(0)

        if on_progress and n_want:
            on_progress(min(1.0, f / float(n_want)))
        f += 1
    cap.release()

    if not frames:
        raise RuntimeError("no frames sampled for the camera track")

    # Hold the last good warp across failures, forward then backward.  A sample
    # that could not be registered gets its neighbour's geometry rather than
    # identity: identity would be a silent claim that the camera is back where
    # it started, which after a jostle is the one answer guaranteed wrong.
    W_arr = np.empty((len(frames), 3, 3), dtype=np.float64)
    ok_arr = np.asarray(oks, dtype=bool)
    first = int(np.argmax(ok_arr)) if ok_arr.any() else len(frames)
    last = None
    for i in range(len(frames)):
        if Ws[i] is not None:
            last = Ws[i]
        W_arr[i] = last if last is not None else _IDENTITY
    # The leading run has no previous good warp to hold, so it borrows the
    # FIRST one instead.  The reference frame is early (frame 300) and the
    # camera has not been hit yet there, so backwards is the safe direction.
    if 0 < first < len(frames):
        W_arr[:first] = Ws[first]

    np.savez_compressed(
        out,
        frames=np.asarray(frames, dtype=np.int32),
        W=W_arr,
        ok=ok_arr,
        inliers=np.asarray(inls, dtype=np.int32),
        ref_index=np.int32(ref_idx),
        sample_step=np.int32(step),
        src_fps=np.float64(src_fps),
        n_src_frames=np.int32(n_src),
        analysis_size=np.asarray(C.ANALYSIS_SIZE, dtype=np.int32),
    )
    if verbose:
        report(video_path, out)
    return out


class CameraTrack:
    """Sampled image -> reference-image warps, indexable by SOURCE frame."""

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

    # ── construction ─────────────────────────────────────────────────────
    @classmethod
    def identity(cls, src_fps=30.0):
        """The no-op track: what a camera that never moved would produce.

        Used when no track is cached and one is not wanted, so that callers have
        exactly ONE code path rather than a fixed-H branch and a tracked branch
        that can drift apart.
        """
        return cls(frames=[0], W=[_IDENTITY], ok=[True], inliers=[0],
                   ref_index=0, src_fps=src_fps, sample_step=1)

    @property
    def is_identity(self):
        return len(self.frames) == 1 and np.allclose(self.W[0], _IDENTITY)

    # ── lookup ───────────────────────────────────────────────────────────
    def index_at(self, src_frame):
        """Index of the sample nearest `src_frame`.

        Nearest, not previous.  A jostle lands between two samples and there is
        no way to know where; nearest halves the worst-case error and, unlike
        interpolation, never invents a geometry the camera never had.
        """
        i = int(np.searchsorted(self.frames, src_frame))
        if i <= 0:
            return 0
        if i >= len(self.frames):
            return len(self.frames) - 1
        return i if (self.frames[i] - src_frame) < (src_frame - self.frames[i - 1]) else i - 1

    def warp_at(self, src_frame):
        return self.W[self.index_at(src_frame)]

    def inv_at(self, src_frame):
        if self._inv is None:
            self._inv = np.linalg.inv(self.W)
        return self._inv[self.index_at(src_frame)]

    # ── diagnostics ──────────────────────────────────────────────────────
    def corner_shift(self, corners):
        """Per-sample max displacement, px, of the court corners.

        `corners` are the clicked corners in reference analysis pixels.  Where
        are they in each sampled frame?  At `W^-1 @ corner`.  How far that is
        from where the fixed homography still assumes they are IS the error.
        """
        c = np.asarray(corners, dtype=np.float64).reshape(-1, 1, 2)
        inv = np.linalg.inv(self.W)
        out = np.empty(len(self.W))
        for i in range(len(self.W)):
            p = cv2.perspectiveTransform(c, inv[i]).reshape(-1, 2)
            out[i] = float(np.hypot(*(p - c.reshape(-1, 2)).T).max())
        return out


def load(video_path, path=None, required=False):
    """The cached track, or the identity track when there is none.

    Missing is not an error by default: a video processed before this existed,
    or one the user chose not to track, must still run -- with exactly today's
    behaviour, which is what the identity track reproduces.
    """
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


def jostles(track, corners, step_px=JOSTLE_PX):
    """Sample-to-sample steps in court-corner position, biggest first.

    Reported rather than acted on.  Nothing downstream needs to know WHERE the
    camera moved -- every frame already carries its own warp -- but a human
    reading the log does, because "the camera was hit at 14:02" is the sentence
    that explains a run.
    """
    shift = track.corner_shift(corners)
    d = np.abs(np.diff(shift))
    idx = np.where(d > step_px)[0]
    ev = [{"frame": int(track.frames[i + 1]),
           "t": float(track.frames[i + 1] / track.src_fps),
           "step_px": float(d[i]),
           "shift_px": float(shift[i + 1])} for i in idx]
    ev.sort(key=lambda e: -e["step_px"])
    return ev


def report(video_path, path=None):
    """What the track says, in the units that decide whether it matters."""
    import json
    tr = load(video_path, path)
    with open(C.court_cache_path(video_path)) as fh:
        corners = np.array(json.load(fh)["points"], dtype=np.float64)

    shift = tr.corner_shift(corners)
    H = C.load_homography(video_path)
    # The court error the FIXED homography would have made: take each corner
    # where it actually is in frame t, project it with H_ref, and measure how
    # far that lands from the corner's true court position.  Metres, because
    # metres is what the detectors' gates are written in.
    truth = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    inv = np.linalg.inv(tr.W)
    err = np.empty(len(tr.W))
    for i in range(len(tr.W)):
        p = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), inv[i]).reshape(-1, 2)
        q = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)
        err[i] = float(np.hypot(*(q - truth).T).max())

    n = len(tr.frames)
    print(f"[camera] {n} samples, anchored on frame {tr.ref_index}, "
          f"registered {100.0 * tr.ok.mean():.1f}%")
    print(f"[camera] court-corner shift px: median {np.median(shift):.1f}  "
          f"p95 {np.percentile(shift, 95):.1f}  max {shift.max():.1f}")
    print(f"[camera] court error under the FIXED homography, metres: "
          f"median {np.median(err):.2f}  p95 {np.percentile(err, 95):.2f}  "
          f"max {err.max():.2f}")
    ev = jostles(tr, corners)
    if not ev:
        print("[camera] no jostle above "
              f"{JOSTLE_PX:.0f} px -- a fixed homography would have been fine")
    for e in ev[:10]:
        print(f"[camera]   jostle at {e['t']:8.2f}s (frame {e['frame']}): "
              f"step {e['step_px']:.1f} px, now {e['shift_px']:.1f} px off")
    return tr


# ── self-test ────────────────────────────────────────────────────────────
def _synthetic_court(seed=0):
    """A 960x540 frame with court lines and enough texture for ORB.

    Not a picture of tennis -- a picture with the STATISTICS that matter here:
    long straight high-contrast lines in the middle (the court, low feature
    density) and dense clutter around the edge (fence, stands), which is the
    distribution that makes the ground-plane re-fit worth doing at all.
    """
    rng = np.random.default_rng(seed)
    w, h = C.ANALYSIS_SIZE
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
        cv2.circle(img, (x, y), int(rng.integers(1, 4)),
                   int(rng.integers(0, 255)), -1)
    return img, quad.astype(np.float64)


def _self_test():
    w, h = C.ANALYSIS_SIZE
    ref, quad = _synthetic_court()
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pts_ref, des_ref = _detect(orb, ref)
    court = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(court, quad.astype(np.int32), 255)
    court = cv2.dilate(court, np.ones((51, 51), np.uint8)).astype(bool)

    def gmask(dst):
        xi = np.clip(np.rint(dst[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(dst[:, 1]).astype(int), 0, h - 1)
        return court[yi, xi]

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
        # W_true maps REFERENCE -> jostled frame; the estimator must return its
        # inverse, because that is the direction the pipeline consumes.
        A = np.asarray(W_true, dtype=np.float64)
        cur = cv2.warpPerspective(ref, A, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        W, n, _, _ = register(cur, pts_ref, des_ref, gmask, orb, matcher)
        if W is None:
            check(label, False, "did not register")
            continue
        # Round trip a court point: warp it into the jostled frame, then ask the
        # estimate to put it back.  Error in reference pixels is the number that
        # decides whether the court map is right.
        p = quad.reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(p, A)
        back = cv2.perspectiveTransform(moved, W).reshape(-1, 2)
        err = float(np.hypot(*(back - quad).T).max())
        check(label, err < 1.0, f"max court-corner error {err:.2f} px, {n} inliers")

    print("[self-test] a static camera is the identity")
    W, n, _, _ = register(ref, pts_ref, des_ref, gmask, orb, matcher)
    err = float(np.abs(cv2.perspectiveTransform(quad.reshape(-1, 1, 2), W)
                       .reshape(-1, 2) - quad).max())
    check("unmoved frame -> identity", err < 0.5, f"max {err:.3f} px")

    print("[self-test] garbage is rejected, not accepted quietly")
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (h, w)).astype(np.uint8)
    W, n, _, _ = register(noise, pts_ref, des_ref, gmask, orb, matcher)
    check("unrelated frame -> no warp", W is None, f"inliers {n}")
    check("_sane rejects a collapsed quad",
          not _sane(np.array([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1.0]])))
    check("_sane rejects a folded quad",
          not _sane(np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])))
    check("_sane accepts identity", _sane(_IDENTITY))

    print("[self-test] lookup takes the NEAREST sample")
    tr = CameraTrack(frames=[0, 100, 200], W=[_IDENTITY] * 3, ok=[1, 1, 1],
                     inliers=[0, 0, 0], ref_index=0, src_fps=30.0,
                     sample_step=100)
    check("index_at before the start", tr.index_at(-5) == 0)
    check("index_at past the end", tr.index_at(9999) == 2)
    check("index_at rounds down below the midpoint", tr.index_at(149) == 1)
    check("index_at rounds up above the midpoint", tr.index_at(151) == 2)

    print("[self-test] Geometry undoes a jostle end to end")
    # A ground point that the camera moved: with the fixed homography it
    # projects somewhere else on the court; through Geometry it does not.
    H_ref = np.array([[0.02, 0.0, -4.0], [0.0, -0.05, 24.0], [0.0, -1e-3, 1.0]])
    A = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -7.0], [0.0, 0.0, 1.0]])
    W_t = np.linalg.inv(A)                       # frame -> reference
    p_ref = np.array([[480.0, 300.0]])
    p_now = cv2.perspectiveTransform(p_ref.reshape(-1, 1, 2), A).reshape(-1, 2)
    truth = C.to_court(H_ref, p_ref)[0]
    fixed = C.to_court(H_ref, p_now)[0]
    tracked = C.to_court(H_ref @ W_t, p_now)[0]
    e_fixed = float(np.hypot(*(fixed - truth)))
    e_tracked = float(np.hypot(*(tracked - truth)))
    # The absolute metres here mean nothing -- H_ref is invented, and the real
    # magnitude is whatever the clip's own geometry makes it.  What is being
    # asserted is the RELATION: the fixed map carries the jostle into court
    # metres and the tracked map does not.
    check("fixed homography carries the jostle into court metres",
          e_fixed > 0.1, f"off by {e_fixed:.2f} m")
    check("tracked homography does not", e_tracked < e_fixed / 1e6,
          f"off by {e_tracked:.2e} m")

    print(f"\n[self-test] {'ALL PASS' if not fails else str(len(fails)) + ' FAILED: ' + ', '.join(fails)}")
    return not fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
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
