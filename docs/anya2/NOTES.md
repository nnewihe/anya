# anya2 — research notes and evidence

Everything below was moved out of the code in the October 2026 readability
rewrite.  It is the **verbatim** text of every docstring and comment in the
pre-rewrite modules (git `16ec715`), in source order, so no measurement, sweep
table or dead end was lost.  Each block is anchored to the line of code it
sat on (`> code:`), so a constant here can be found in today's module by name.

The code now carries short "what and why" comments; this file is the
"how do we know" behind them.  The second half is the former package README
(the results narrative), also verbatim.

Contents: [run.py](#runpy), [config.py](#configpy), [contract.py](#contractpy), [signals.py](#signalspy), [court.py](#courtpy), [camera.py](#camerapy), [perceive.py](#perceivepy), [tracks.py](#trackspy), [near_serve.py](#near_servepy), [far_serve.py](#far_servepy), [point_end.py](#point_endpy), [near_end.py](#near_endpy), [orchestrator.py](#orchestratorpy), [Former README](#former-readme)


---

## run.py

### Module docstring

```text
run.py
======
One call that takes a video and returns a cut reel -- the anya2 equivalent of
`pipeline.rally_reel.build_reel`, with the same signature so the desktop app
switches by changing an import.

    segments, out_path = build_reel(video, output, cfg=Anya2Config(),
                                    on_progress=cb)

`video` may be a single path or several: GoPro chapters of one recording are
joined into one file first (`pipeline.join`), and every stage after that sees
an ordinary single source video.  See that module for why the join happens
here rather than the pipeline learning to read a list.

Stages, and why they are in this order
--------------------------------------
  1 JOIN          several GoPro chapter files -> one video.  A no-op, and not
                  even a file, for the single-video case that is still the
                  normal one.
  2 CALIBRATION   the court corners, prompted once per video and cached.  This
                  opens a cv2 window, so it MUST run on the caller's main
                  thread -- `ensure_court` is separated out for exactly that,
                  and the desktop app calls it before starting the worker.
  3 CAMERA TRACK  where each frame sits relative to the frame those corners were
                  clicked on.  BEFORE perceive, and not merely for tidiness:
                  the far pose pass crops a fixed rectangle around the far
                  baseline, and that rectangle has to be sized to cover
                  everywhere the camera went, which is only knowable once the
                  track exists.  It also builds the 540p proxy that the near
                  pass then reuses, so it costs one decode, not two.
  4 PERCEIVE      the two pose passes.  The whole cost of the pipeline; see
                  perceive.py for why one ROI cannot serve both ends.
  5 TRACKS        <=2 near + <=2 far player slots on one timeline.
  6 END SIGNALS   the walking classifier and near_end's four pose signals, both
                  read off the near track.  Agent 3 needs them; nothing else
                  does.
  7 DETECTORS     the three agents, independently.
  8 ORCHESTRATE   structure, rules, recovery, smoothing -> segments.
  9 CUT           ffmpeg.

Progress is reported as (stage_index, n_stages, label, fraction), stages
numbered from ONE -- which is what the desktop app's bar already assumes when
it places a stage's fraction at `(i - 1 + frac) / n`.  It mattered little
while stage one was a cached calibration check; the join is minutes of I/O
with a real fraction to report, and off by one it would sit at zero for all
of them.
```

> code: `cancel.check()`

```text
Every stage announces itself through here exactly once, which makes this
the one place a per-stage cancel check can live without a line per stage
-- and a stage added later inherits it. The long stages check in far more
often than this from the inside (see pipeline.cancel).
```

> code: `def ensure_court(video, on_progress=None) -> None:`

```text
Prompt for the court corners if they are not cached.

Separated from `build_reel` because it opens an OpenCV window: it has to run
on the main thread, before any worker starts.  Idempotent once cached.

Accepts the same one-or-many `video` as `build_reel`, and joins first --
the corners are clicked on the JOINED file, because that is what every
later stage indexes into.  The join is cached, so `build_reel` calling
`resolve_input` again a moment later costs a sidecar read.
```

> code: `def _join(video, on_progress=None) -> str:`

```text
One source path from whatever the caller passed.

A single path comes back untouched; several are remuxed into one file in
the artifact dir.  Every stage below this line takes the result and knows
nothing about chapters.
```

> code: `def _stem(video: str) -> Tuple[str, str]:`

```text
The video's OWN directory and stem -- NOT the artifact dir.

The DIRECTORY half is used only for the output video's default location,
which must stay beside the input regardless of any work-dir override: the
app's tmp_anya holds calibration and interim files, never the finished
reel.  (With chapter inputs it is the FIRST CHAPTER that supplies it, not
the join -- the join itself lives in tmp_anya.)  Every cache or event file
uses `pipeline.workdir.artifact_dir` instead, paired with the stem half of
this -- see `_end_signals` and `build_reel` below for the split.
```

> code: `def _end_signals(video: str, force: bool = False) -> None:`

```text
Walking probabilities and near_end's four signals, cached beside the video.

Both are computed from the NEAR track through a shim npz shaped like
`walking.select_near`'s output, because that is what both consumers already
read.  The near slot with the better coverage leads and the other fills its
gaps -- on a changeover the same human moves between slots, and using one
slot alone loses half the clip.
```

> code: `from pipeline import near_end as NE`

```text
near_end.py lives in pipeline/, not the repo root -- every other caller
in this codebase imports it as pipeline.near_end (`from ..near_end import
...` inside pipeline.rally_reel, `from .near_end import ...` inside
pipeline/tune_energy.py).  `import near_end` alone only resolves when the
CALLER has also put pipeline/ itself on sys.path, which the CLI scripts
used to test this module did and this function did not -- the bug
surfaced as "No module named 'near_end'" the first time this ran from the
desktop app rather than from one of those scripts.
```

> code: `def cut(video: str, segments: List[dict], output: str,`

```text
ffmpeg: encode each segment, then concat.
```

> code: `wd = WD.get_work_dir()`

```text
A work-dir override (the desktop app's tmp_anya) gets the scratch
segment files too; the app decides whether to keep them as part of the
same "keep interim files" choice that covers everything else there, so
this function does not clean up after itself in that case.  With no
override -- every CLI caller -- this used to leak a system temp
directory on every call; it now self-cleans, matching the legacy
cutter's behaviour in pipeline.utilities.create_highlights_ffmpeg.
```

> code: `if cancel.run(cmd, capture_output=True).returncode == 0:`

```text
cancel.run rather than subprocess.run: one 4K segment can take tens
of seconds to encode, which would be the whole latency of Cancel.
```

> code: `from pipeline.utilities import concat_cmd, write_concat_list`

```text
The join re-encodes the audio (video is still copied) so sound cannot
walk away from picture over the length of the reel -- see the long note
above `concat_cmd` in pipeline.utilities for the measurements.
```

> code: `def build_reel(video_path, output_path: Optional[str] = None,`

```text
Video in, cut reel out.  Signature-compatible with rally_reel.build_reel.

`video_path` is one path, or several GoPro chapters of one recording that
are joined into one before anything else runs.
```

> code: `first = video_path if isinstance(video_path, (str, os.PathLike)) else video_path[0]`

```text
The output default comes from the FIRST INPUT, not the join: a joined
file lives in the artifact dir, which under the desktop app is the
tmp_anya the run deletes on its way out -- defaulting the reel into it
would throw away the only thing the run was for.
```

> code: `_emit(on_progress, 3, "Tracking the camera")`

```text
Before perceive: `PC.far`'s crop rectangle is sized from this track, and
a crop is a fixed ffmpeg rectangle that cannot follow a moving camera.
```

> code: `cfg.reel.use_near = cfg.near.enabled`

```text
A disabled agent is disabled for the ORCHESTRATOR too, not merely skipped
here -- its events are cached on disk and would otherwise still be read.
```


---

## config.py

### Module docstring

```text
config.py
=========
One knob-set per agent, so each role can be adjusted independently.

WHY THIS EXISTS SEPARATELY FROM THE DETECTORS
---------------------------------------------
Each detector keeps its constants at module level, and that is deliberate: the
constant sits next to the paragraph explaining what was measured to choose it,
which is the only reason those numbers are trustworthy.  Moving them wholesale
into a config object would separate every value from its evidence.

So this module does not replace them.  It exposes the handful that a CALLER
legitimately wants to vary per run -- thresholds, timing, and the few weights
that trade recall against precision -- and leaves the measured geometry where it
is documented.  A field of `None` means "use the module's own value", so a
default config reproduces the shipped behaviour exactly.

Anything not here is not meant to be tuned per run; it is meant to be re-derived
against the corpus and re-committed with its evidence.
```

> code: `class PerceiveConfig:`

```text
The two pose passes. See perceive.py for why the two ROIs differ.
```

> code: `pose_fps: float = 15.0`

```text
locked to ONE rate for both ROIs -- tracks
interleaves them sample for sample
```

> code: `near_imgsz: Optional[int] = None`

```text
default 640; near players have headroom
```

> code: `far_imgsz: Optional[int] = None`

```text
default 960; the far player has NONE --
768 costs 13 points of far recall
```

> code: `@dataclass`

```text
default 5.0 -- how often the camera
track registers a frame against the
calibration frame.  Raise it only to
place a jostle more precisely in time;
a jostle is a step, and the geometry
either side of it is already right.
Turn tracking OFF entirely with
ANYA_CAMERA_TRACK=0, not from here --
see pipeline/anya2/camera.py.
```

> code: `class NearServeConfig:`

```text
Agent 1. Pose-only ready -> trophy -> swing.
```

> code: `threshold: Optional[float] = None`

```text
default 0.70, top of a flat plateau
```

> code: `lead_s: Optional[float] = None`

```text
default 1.63, the label convention
```

> code: `require_court: bool = True`

```text
the serve-zone band; off = smashes
```

> code: `class FarServeConfig:`

```text
Agent 2. Elevation-only trophy, plus stillness and a separate toss score.
```

> code: `w_still: Optional[float] = None`

```text
default 0.30 -- weight of the
pre-serve stillness veto, the term
that took precision 51% -> 65%
```

> code: `class PointEndConfig:`

```text
Agent 3. Pose-only; the falling edge of a live score.
```

> code: `live_hi: Optional[float] = None`

```text
default 0.50, enter "live"
```

> code: `class Anya2Config:`

```text
Everything, per agent, plus the orchestrator.

`reel` is the orchestrator's own config and already carries its rules,
rolls and recovery settings -- see orchestrator.ReelConfig.
```


---

## contract.py

### Module docstring

```text
contract.py
===========
The two dataclasses that let three detectors be built independently and still
compose.

`Event` is the only thing a detector emits.  `Requirement` is the only thing it
says about what perception it needs.  Neither carries a threshold, a weight, or
anything else a detector might want to tune -- that is the point: an agent
working on point ends can change every constant it owns without a near-serve
agent having to know.

Why `Requirement` exists at all
-------------------------------
Compute optimization is deliberately the LAST phase of this redesign, not a
constraint during it.  But an optimization pass can only collapse perception if
it knows what each consumer actually reads, and asking three finished detectors
after the fact means reverse-engineering it from their code.  So each declares
it up front, and the collapse becomes a mechanical union over the declarations
rather than an archaeology exercise.

`windows` is the field that matters most there.  A near-serve detector can only
fire between points; a point-end detector can only fire after one has started.
Neither detector knows the other exists -- they just say when they are capable
of firing, and the scheduler intersects that with what the other detectors have
already found.
```

> code: `NEAR_SERVE = "near_serve"`

```text
Detector kinds.  Also the `--mode` values `eval.py` accepts, and the keys the
eventual fusion pass will read; kept as constants so a typo is an ImportError
rather than a silently empty event list.
```

> code: `ROI_NEAR = "near"`

```text
`Requirement.roi` values -- see perceive.py for why there are two and not one.
```

> code: `W_ALWAYS = "always"`

```text
`Requirement.windows` values.
```

> code: `W_ALWAYS = "always"`

```text
must run over the whole clip
```

> code: `W_BETWEEN = "between_points"`

```text
can only fire while no point is live
```

> code: `W_AFTER_SERVE = "after_serve"`

```text
can only fire once a point has started
```

> code: `class Event:`

```text
One detected point boundary.

`t` is SECONDS ON THE SOURCE TIMELINE, always -- not proxy frames, not
decimated pose samples.  Every detector runs on a decimated and/or rescaled
stream, and having each consumer undo a different stride is exactly how the
current pipeline accumulated three incompatible notions of "when".
```

> code: `track: Optional[int] = None`

```text
which player slot; doubles: who served
```

> code: `detail: Dict = field(default_factory=dict)`

```text
diagnostics; never read by
anything but a human
```

> code: `class Requirement:`

```text
What a detector needs from the perception layer.

Declarative and inert: nothing reads it during development.  Phase E's
scheduler unions these to decide what actually has to run where.
```

> code: `def dump_events(events: List[Event], path: str, **meta) -> str:`

```text
Write events as JSON, sorted by time.

The schema is `{"events": [...], **meta}` with each event a flat dict, which
is what `eval.py` reads.  `meta` is for whatever the detector wants on the
record -- its Requirement, its thresholds, the git sha -- so a scored run
can be traced back to the code that produced it.
```


---

## signals.py

### Module docstring

```text
signals.py
==========
Pure signal-processing helpers shared by the detectors.

MACHINERY IS SHARED HERE; THRESHOLDS ARE NOT.  Every function below is
parameter-free maths -- a ramp, a running max, a run-length decomposition.  The
numbers that decide what counts as a serve or an end stay in the detector that
owns them, because that is the premise of building the three separately: an
agent may retune anything in its own module without touching another's.

If something here ever needs a constant, it belongs in the caller instead.
```

> code: `NOSE = 0`

```text
COCO keypoint indices, as in walking.features / near_end.
```

> code: `def ramp(x, lo, hi):`

```text
Soft 0->1 ramp from lo to hi (either order); NaN in, NaN out.
```

> code: `def movmax(x, w):`

```text
Centred running max over `w` samples, NaN-tolerant (NaN acts as -inf).

Shifts rather than a stride trick, so the edges degrade to a shorter window
instead of to NaN -- an event in the first second of a clip is still an
event.

This is what lets a CONJUNCTION describe a phase rather than an instant.  A
product of per-sample terms requires them to hold on the same sample, and
over a fast phase sampled at 15 Hz they routinely do not: measured on the
near serve, all three trophy terms cleared their thresholds while the
product stayed near zero, because the tossing arm extends a sample or two
before the racket arm settles.  Dilating each term first asks "did this hold
ANYWHERE NEAR now", which is what a phase means.
```

> code: `def runs(mask) -> List[tuple]:`

```text
[(lo, hi)) index runs of True.
```

> code: `def refractory(events, gap_s, key="t", strongest=True):`

```text
Thin `events` so no two survive within `gap_s`.

`strongest=True` keeps the highest-scoring member of each cluster (the
right choice when a cluster is one action detected several times);
`False` keeps the EARLIEST (the right choice when a cluster is a real
sequence whose first member is the one being timed).
```

> code: `def interp_gaps(a, max_gap):`

```text
Linearly fill runs of NaN no longer than `max_gap` samples.

Short occlusions are filled; real holes are left as NaN, because a filled
hole is a fabricated posture and every downstream conjunction would read it
as evidence.
```

> code: `def kp_xy(kp, i, conf=0.0):`

```text
[N,2] of keypoint `i`, NaN where confidence is at or below `conf`.
```

> code: `def mid(a, b):`

```text
Midpoint of two [N,2] tracks; NaN unless both are present.
```


---

## court.py

### Module docstring

```text
court.py
========
Court geometry for anya2: the image -> court-metres map, and the two player
gates that every detector in this package shares.

The homography itself is NOT redefined here.  `walking.court.load_homography`
already maps the four cached corners onto a singles rectangle, and duplicating
that would let the two drift; this module imports it and adds only what anya2
needs on top.

Reference frame
---------------
Origin at the near-left SINGLES corner, +x along the near baseline, +y running
away from the camera, metres.  So:

    near baseline   y = 0
    far  baseline   y = COURT_L = 23.77
    mid court       y = MID_Y   = 11.885
    singles sidelines  x = 0 and x = COURT_W = 8.23

y goes NEGATIVE behind the near baseline, which is where a server stands.

Doubles without re-calibration
------------------------------
The cached corners are the SINGLES corners and the homography is a full plane
map, so the doubles sidelines need no clicking and no new cache format: the
alley is a fixed 1.37 m outside each singles line, in the same court frame.
That is why `ALLEY_W` is a constant here rather than a calibration input.

The two gates
-------------
Both are the user's rules, and both are evaluated on ONE projected point --
`ground_point`, the bounding box's bottom-centre:

  side()       near vs far by which baseline the box BOTTOM is closer to in y.
  in_bounds()  the box X-CENTRE inside the doubles court plus a 3 ft margin.

Bottom-centre is the only point that carries both quantities (it is the box
x-centre, and it is the box bottom) while still being a legitimate ground-plane
projection.  Projecting a mid-body point through a GROUND-plane homography puts
the player metres up the court, so the bottom is not a detail.

NaN is False, never True.  A player whose projection is unavailable is excluded
rather than admitted -- the conservative direction for a rule whose job is to
say who is allowed to be serving.
```

> code: `def court_cache_path(video_path):`

```text
Override-aware court cache path.

`walking.court.court_cache_path` computes the SAME filename convention
(`{stem}_court_cache.json`) beside the video, unconditionally -- it has no
knowledge of the app's work-dir override, because `walking/` is used by
CLI/eval callers that must never be redirected.  anya2 wraps it so the
court cache anya2 looks for is the same one `pipeline.utilities.init_court`
writes when a work dir IS set.
```

> code: `ALLEY_W = 1.37`

```text
── doubles geometry ─────────────────────────────────────────────────────
```

> code: `ALLEY_W = 1.37`

```text
doubles alley width, metres, each side of the singles
line.  Doubles court is 10.97 m wide against singles
8.23, so the alley is (10.97 - 8.23) / 2.
```

> code: `FT_TO_M = 0.3048`

```text
── the 3 ft margin ──────────────────────────────────────────────────────
```

> code: `X_LO = -(ALLEY_W + MARGIN_M)`

```text
Lateral eligibility band, in the singles-origin court frame.
```

> code: `def ground_point(bbox):`

```text
Bottom-centre of an xyxy box (or [..., 4] array of them), in image px.

Returns [..., 2].  NaN in, NaN out.
```

> code: `def project(H, bbox):`

```text
Court metres for the ground point of each box. [..., 4] -> [..., 2].
```

> code: `def side(court_y):`

```text
NEAR / FAR by proximity of the ground point to each baseline.

The user's rule stated literally: |y - 0| < |y - COURT_L|.  That reduces to
y < MID_Y, but it is written out because the literal form is the one that
stays correct if the court frame is ever re-origined.
Returns an object array of NEAR/FAR/None (None where y is NaN).
```

> code: `def is_near(court_y):`

```text
Boolean form of `side`. NaN -> False.
```

> code: `def in_bounds(court_x):`

```text
Box x-centre inside the doubles court plus the 3 ft margin. NaN -> False.
```

> code: `def eligible(court_xy):`

```text
The lateral gate over a [..., 2] court array. NaN -> False.

Deliberately NOT combined with a y gate: how far up or down the court a
player may be is a per-detector question (a server stands behind the
baseline, a receiver does not), whereas the lateral band is the user's
court-membership rule and is the same for everyone.
```

> code: `class Geometry:`

```text
── the map is a function of time, not a constant ────────────────────────
```

> code: `class Geometry:`

```text
Image -> court metres AT A GIVEN FRAME.

`load_homography` returns the one map implied by the four clicked corners,
and every caller used to apply it to every frame of the video.  That is only
right while the camera does not move.  This wraps it with the camera track
(`pipeline.anya2.camera`), which says where frame t's image sits relative to
the frame those corners were clicked on:

    H_at(t)  =  H_ref @ W_t

With no cached track the camera track is the identity and `H_at` returns
`H_ref` for every frame -- byte for byte the old behaviour, through the same
code path rather than through a branch beside it.

FRAME INDICES ARE SOURCE FRAMES.  Every pass in anya2 is decimated by its
own stride, and a sample index means nothing without knowing which; the
source frame number is the one clock they all share.
```

> code: `def is_static(self):`

```text
True when this is the plain fixed homography after all.
```

> code: `def H_at(self, src_frame):`

```text
3x3 image(src_frame) -> court metres.
```

> code: `H = self._H[i] = self.H_ref @ self.track.W[i]`

```text
Cached per SAMPLE, not per frame: at 15 Hz pose over a 30 minute
match this is 9000 lookups against ~9000 distinct warps, and the
matrix product is the only work that would otherwise repeat.
```

> code: `def project_at(self, src_frame, bbox):`

```text
`project`, at a frame. [..., 4] -> [..., 2] court metres.
```

> code: `def to_court_at(self, src_frame, pts):`

```text
`to_court`, at a frame. [N,2] image px -> [N,2] court metres.
```


---

## camera.py

### Module docstring

```text
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
```

> code: `SAMPLE_FPS = 5.0`

```text
── sampling ─────────────────────────────────────────────────────────────
```

> code: `SAMPLE_FPS = 5.0`

```text
see "WHAT IT COSTS" -- a jostle is a step, not a signal
```

> code: `ORB_FEATURES = 4000`

```text
── registration ─────────────────────────────────────────────────────────
```

> code: `ORB_FEATURES = 4000`

```text
a tennis frame is texture-poor in the middle and rich
at the edges (fence, stands, sponsor boards, lines).
The count has to be high enough that the COURT still
contributes after the edges have taken their share.
```

> code: `_FLANN_LSH = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)`

```text
Approximate nearest-neighbour matching, not brute force.  Matching dominates
this pass -- 4000 reference descriptors against 4000 per frame is 16 M Hamming
comparisons -- and on Data/77 it is where 80% of the time went:

    detect  match+fit        ground inliers (median)
    9.7 ms    72.9 ms  BF           175
    9.7 ms    11.4 ms  FLANN-LSH    176      <-- here

Six times faster for the same answer.  LSH is approximate and can in principle
return fewer matches on a frame; `_register` falls back to brute force when the
LSH fit fails, so the approximation can cost time on a hard frame but never an
answer.
```

> code: `MIN_INLIERS = 30`

```text
below this the fit is not evidence of anything
```

> code: `MIN_GROUND_INLIERS = 60`

```text
── when the ground-plane re-fit may override the global fit ─────────────
MEASURED, and it overturned the value this shipped with.  Over 414 samples
spread across Data/77's 69 minutes:

    global fit        median 1365 inliers (p05 567)
    on-court inliers  median   18

A tennis court is flat paint.  It carries almost no ORB features, so the
ground-plane re-fit -- which is right in principle, see the module docstring --
is almost never SUPPORTABLE on real footage.  At the 20 this shipped with it
fired anyway, replacing a 1365-inlier fit with a 20-point one:

    |ground - global| corner displacement, px
      >= 20 on-court inliers (n=178)   median  3.14   p95 33.73   max 86.20
      >= 50 on-court inliers (n= 26)   median  0.22   p95  1.84   max  2.67

Above 50 the two agree to a quarter of a pixel; below it the re-fit is noise
wearing the same interface.  That produced 47 phantom "jostles" of 100-213 px
on this clip, half of them reverting within 0.2 s -- which no tripod does, and
which is what gave them away.

60, and an agreement gate as well: the re-fit exists to REFINE the global fit
under a small camera translation, not to find a different camera.  One that
disagrees by more than a few pixels has not refined anything.
```

> code: `MAX_SHIFT_PX = 240.0`

```text
── sanity bounds on an accepted warp ────────────────────────────────────
A homography fit to mismatched features is usually not subtly wrong, it is
wildly wrong -- a collapsed or inverted quad.  These reject that without
pretending to know how hard the camera was hit.
```

> code: `MAX_SHIFT_PX = 240.0`

```text
a frame corner may not move more than a quarter frame
```

> code: `JOSTLE_PX = 3.0`

```text
── what counts as a jostle, for reporting ───────────────────────────────
```

> code: `JOSTLE_PX = 3.0`

```text
sample-to-sample step in court-corner position
```

> code: `SMOOTH_SAMPLES = 5`

```text
── isolated bad samples ─────────────────────────────────────────────────
A sample can register on far weaker evidence than its neighbours and still
clear MIN_INLIERS.  On Data/77 exactly two of 20,681 did -- both in the
opening seconds, where the picture is still fading up -- with 136 inliers
against the track's median of 1339, and they put the far baseline 14.9 px
out.  Every other sample was within 3.8 px.

A running MEDIAN over the track removes those and leaves a genuine jostle
intact, which is the whole reason it is a median and not a mean: a median
passes a step edge through with at most a couple of samples' delay, while a
mean would ramp across it and smear the one event this module exists to catch.

The filter runs on the corner POSITIONS, not on the matrices -- there is no
meaningful median of a homography, but a homography is fully determined by
where a handful of points go, so filtering the points and re-deriving the
matrix is both well defined and exactly equivalent.
```

> code: `SMOOTH_SAMPLES = 5`

```text
1.0 s at SAMPLE_FPS; 0 or 1 disables
```

> code: `def enabled():`

```text
── the kill switch ──────────────────────────────────────────────────────
ONE mechanism, checked in the two functions that can turn a track on: nothing
estimates a track and nothing reads a cached one.  It is an environment
variable rather than a config field because it has to reach code that never
sees the config -- `tracks.build` and `perceive.far_band` are both callable
straight from the CLI -- and because the state it disables lives on DISK: a
config flag would leave a stale npz on disk still being read by every caller
that did not get the flag.  `ANYA_ENGINE` in the desktop app is the same idiom.

    ANYA_CAMERA_TRACK=0   every projection uses the clicked homography alone,
                          which is exactly the behaviour before this existed.
```

> code: `def reference_index(video_path, target_idx=300):`

```text
The frame `init_court` showed the user.

Duplicated from `utilities.get_reference_frame` rather than imported,
because that function returns the IMAGE and not the index, and the index is
what the whole track is anchored to.  If that rule ever changes, this is the
other place it has to change.
```

> code: `def _reference_frame(video_path, ref_idx):`

```text
The calibration frame, greyscale, at ANALYSIS_SIZE.

Read from the SOURCE, the same way `init_court` read it, so the pixels the
warps are anchored to are the pixels the corners were clicked on.  One 4K
seek, once per video.
```

> code: `def court_corners(video_path):`

```text
The four clicked corners, in analysis pixels. [4,2].
```

> code: `def _court_mask(video_path, pad_px=25):`

```text
Boolean [H,W] mask of the court surface in the reference frame.

The cached corners are already a quadrilateral in analysis pixels, which is
exactly the ground-plane region -- there is nothing to detect.  It is
dilated because a feature ON a line sits at the line's edge, and the corner
clicks are human.
```

> code: `def _match(matcher, des_cur, des_ref, pts_cur, pts_ref):`

```text
Ratio-tested correspondences. Returns (src_cur [M,2], dst_ref [M,2]).
```

> code: `def _sane(W):`

```text
Reject a fit that is not a plausible view of the same court.

Not a quality measure -- a wrong fit here is a collapsed or folded quad, not
a slightly-off one, and this only has to catch that.
```

> code: `x, y = out[:, 0], out[:, 1]`

```text
Shoelace area, signed: a negative area means the quad was folded over.
```

> code: `def _fit(src, dst, ground_mask_fn, probe=None):`

```text
Global RANSAC homography, then the ground-plane re-fit if it earns it.

`ground_mask_fn(dst_points) -> bool[M]` says which correspondences land on
the court in the REFERENCE frame.  `probe` is [K,2] reference points -- the
court corners -- where the two fits are compared; without it the agreement
gate cannot be applied and the global fit is simply kept.

Returns (W, n_inliers) or (None, 0).
```

> code: `q = np.asarray(probe, dtype=np.float64).reshape(-1, 1, 2)`

```text
The agreement gate.  Both fits are asked where the court corners go; a
re-fit that moves them further than GROUND_AGREE_PX from what a fit with
an order of magnitude more support says is not a correction, and the
global fit is kept instead.
```

> code: `def register(gray, pts_ref, des_ref, ground_mask_fn, orb=None, matcher=None,`

```text
Warp mapping `gray` onto the reference frame, or (None, 0).

Factored out of `estimate` so the registration can be exercised on a
synthetic pair with a KNOWN answer -- see `_self_test`.  A geometry routine
whose only test is "the reel looked better" is a routine nobody can change.
```

> code: `src, dst = _match(make_matcher(approximate=False), des_cur, des_ref,`

```text
The approximate matcher missed; the exact one is 6x slower and this
only runs on frames that already failed, so it costs nothing on a
clip where it is never needed.
```

> code: `def estimate(video_path, force=False, sample_fps=SAMPLE_FPS, out=None,`

```text
Register the video against its calibration frame; cache the warps.

Idempotent: returns the cached path unless `force`.
```

> code: `prox = P.ensure_proxy(video_path, size=C.ANALYSIS_SIZE, crf=14,`

```text
The proxy `perceive` builds anyway. If it could not be made frame-exact,
ensure_proxy hands back the SOURCE, which is still correct here -- the
resize below is unconditional for exactly that case.
```

> code: `if not cap.grab():`

```text
grab() then retrieve(), rather than read().  Eleven of every twelve
frames are skipped here, and read() pays for a colour conversion and a
copy of each one before we throw it away; grab() advances the decoder
without either.  Measured on Data/77 it saves ~2% -- decode is NOT the
bottleneck, matching is (see MATCHER below).  Kept because it is
strictly less work, not because it bought anything.
```

> code: `src2, dst2 = _match(matcher, des_cur, prev_kp[1], pts_cur, prev_kp[0])`

```text
Fallback: register against the last sample that DID match the
reference, and compose. One link, never a chain -- the anchor is
still a reference-registered frame, so nothing accumulates.
```

> code: `W_arr = np.empty((len(frames), 3, 3), dtype=np.float64)`

```text
Hold the last good warp across failures, forward then backward.  A sample
that could not be registered gets its neighbour's geometry rather than
identity: identity would be a silent claim that the camera is back where
it started, which after a jostle is the one answer guaranteed wrong.
```

> code: `if 0 < first < len(frames):`

```text
The leading run has no previous good warp to hold, so it borrows the
FIRST one instead.  The reference frame is early (frame 300) and the
camera has not been hit yet there, so backwards is the safe direction.
```

> code: `def _probe_points():`

```text
Points the track is filtered and re-derived on.

The four court corners because they are what the map is used for, and the
four frame corners because a homography fitted to the court quad alone is
poorly conditioned outside it -- the far band reaches above the top of the
court, and the near player stands below its bottom.
```

> code: `def smooth(W, quad, k=SMOOTH_SAMPLES):`

```text
Running median over the track. See SMOOTH_SAMPLES for why a median.
```

> code: `half = k // 2`

```text
The window STICKS at the ends rather than being padded with a repeat of
the edge sample.  Repeating it makes a bad first sample outvote the good
ones beside it, which is precisely the case this filter exists for -- the
two outliers on Data/77 are the first two samples of the video.  Sticking
means index 0 and index 1 both take the median of samples 0..k-1, where
the outliers are outnumbered.
```

> code: `M, _ = cv2.findHomography(med[i].astype(np.float64), pts, 0)`

```text
method=0 -- a plain least-squares fit.  These eight points are already
the filtered consensus; running RANSAC over them again would be
re-deciding which of an agreed set to believe.
```

> code: `class CameraTrack:`

```text
Sampled image -> reference-image warps, indexable by SOURCE frame.
```

> code: `@classmethod`

```text
── construction ─────────────────────────────────────────────────────
```

> code: `def identity(cls, src_fps=30.0):`

```text
The no-op track: what a camera that never moved would produce.

Used when no track is cached and one is not wanted, so that callers have
exactly ONE code path rather than a fixed-H branch and a tracked branch
that can drift apart.
```

> code: `def index_at(self, src_frame):`

```text
── lookup ───────────────────────────────────────────────────────────
```

> code: `def index_at(self, src_frame):`

```text
Index of the sample nearest `src_frame`.

Nearest, not previous.  A jostle lands between two samples and there is
no way to know where; nearest halves the worst-case error and, unlike
interpolation, never invents a geometry the camera never had.
```

> code: `def corner_shift(self, corners):`

```text
── diagnostics ──────────────────────────────────────────────────────
```

> code: `def corner_shift(self, corners):`

```text
Per-sample max displacement, px, of the court corners.

`corners` are the clicked corners in reference analysis pixels.  Where
are they in each sampled frame?  At `W^-1 @ corner`.  How far that is
from where the fixed homography still assumes they are IS the error.
```

> code: `def load(video_path, path=None, required=False):`

```text
The cached track, or the identity track when there is none.

Missing is not an error by default: a video processed before this existed,
or one the user chose not to track, must still run -- with exactly today's
behaviour, which is what the identity track reproduces.
```

> code: `def jostles(track, corners, step_px=JOSTLE_PX):`

```text
Sample-to-sample steps in court-corner position, biggest first.

Reported rather than acted on.  Nothing downstream needs to know WHERE the
camera moved -- every frame already carries its own warp -- but a human
reading the log does, because "the camera was hit at 14:02" is the sentence
that explains a run.
```

> code: `def report(video_path, path=None):`

```text
What the track says, in the units that decide whether it matters.
```

> code: `truth = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)`

```text
The court error the FIXED homography would have made: take each corner
where it actually is in frame t, project it with H_ref, and measure how
far that lands from the corner's true court position.  Metres, because
metres is what the detectors' gates are written in.
```

> code: `def _synthetic_court(seed=0):`

```text
── self-test ────────────────────────────────────────────────────────────
```

> code: `def _synthetic_court(seed=0):`

```text
A 960x540 frame with court lines and enough texture for ORB.

Not a picture of tennis -- a picture with the STATISTICS that matter here:
long straight high-contrast lines in the middle (the court, low feature
density) and dense clutter around the edge (fence, stands), which is the
distribution that makes the ground-plane re-fit worth doing at all.
```

> code: `A = np.asarray(W_true, dtype=np.float64)`

```text
W_true maps REFERENCE -> jostled frame; the estimator must return its
inverse, because that is the direction the pipeline consumes.
```

> code: `p = quad.reshape(-1, 1, 2)`

```text
Round trip a court point: warp it into the jostled frame, then ask the
estimate to put it back.  Error in reference pixels is the number that
decides whether the court map is right.
```

> code: `H_ref = np.array([[0.02, 0.0, -4.0], [0.0, -0.05, 24.0], [0.0, -1e-3, 1.0]])`

```text
A ground point that the camera moved: with the fixed homography it
projects somewhere else on the court; through Geometry it does not.
```

> code: `check("fixed homography carries the jostle into court metres",`

```text
The absolute metres here mean nothing -- H_ref is invented, and the real
magnitude is whatever the clip's own geometry makes it.  What is being
asserted is the RELATION: the fixed map carries the jostle into court
metres and the tracked map does not.
```


---

## perceive.py

### Module docstring

```text
perceive.py
===========
The perception passes anya2's detectors read, cached beside the video.

TWO ROIs, BECAUSE ONE DOES NOT WORK
-----------------------------------
The obvious design is one all-persons pose pass over a downscaled proxy, and it
is the design the current pipeline nominally has: `anya_end_telemetry` runs pose
full-frame over a 540p proxy at 15 Hz, with no crop.  Measured on Data/21,
ZERO of 192 sampled detections from that pass project past mid-court.  The far
player is simply not there.

The cause is scale, not the projection: at 540p a player standing on the far
baseline is 20-40 px tall in the 960x540 analysis frame, and yolov8n-pose at
imgsz 960 does not find them.  `walking/extract_pose.py` already documents the
same effect for the near player walking to the ball carts, and works around it
with a `--rescue` pass at 1920 px.

So the far side gets its own ROI: a NATIVE-RESOLUTION CROP of the band around
the far baseline, which keeps source pixels on the subject while paying to
decode only the part of the frame it occupies.  This is the same conclusion
`anya_far_telemetry` reached independently, and it is recorded here as a
property of the perception layer rather than rediscovered by each detector.

    near   540p whole-frame proxy   15 Hz pose   near players, point ends
    far    native-res band crop     15 Hz pose   far players

RATES ARE INPUTS, NOT DECISIONS
-------------------------------
15 Hz is where `anya_end_telemetry` landed and the reasoning holds here: the
walking classifier scores at 15 Hz, and 15 Hz is above Nyquist for the 0.7-4.0
Hz gait cadence band, while 7.5 Hz is not.

THE TWO ROIs ARE LOCKED TO ONE RATE, and that is a real constraint rather than
an oversight.  `tracks._stack` requires both passes on one timeline, because
side membership is decided by the COURT and not by which pass found a person,
so the two candidate lists have to be interleaved sample for sample.  Dropping
the far pass to 10 Hz therefore cannot be done alone -- attempted, it makes
`tracks` refuse outright, which is the guard working.  (It also produced a
tempting false result: with tracks refusing, the far detector scored the STALE
15 Hz tracks and reported identical accuracy for 30% less time.  The number was
real and measured the wrong thing.)
Decoupling the rates means resampling the far stream onto the near timeline --
possible, but a change to the substrate rather than a parameter, and not worth
making speculatively while the far pass is the cheaper of the two.

Unsampled frames are DROPPED, never held.  A zero-order hold turns a
differentiated feature into spurious events; the npz is written decimated with
`fps` set to the effective rate and `stride` alongside, so every consumer sees
one consistent, slower clip and only the mapping back to source frames needs
the stride.
```

> code: `FAR_IMGSZ = 960`

```text
FAR_IMGSZ stays at 960, and the attempt to lower it is recorded because the
micro-benchmark was persuasive and WRONG.  Timed on 64 band frames, imgsz 768
looked free -- 35.2 ms/frame against 52.6, at 1.00 persons/frame against 1.02.
Run over the whole clip it is not: detections fall from 0.94 to 0.86 per
sample, empty frames rise 21.1% to 25.1%, and the far detector goes from
100% recall / 83.3% precision to 86.7% / 72.2% on Data/23.  An 8% detection
loss lands squarely on the serves.

The lesson generalises: the far player sits AT the model's detection size, so
persons-per-frame over a short sample cannot see the cost -- the frames that
break are the ones where the player is smallest, which is not most frames but
is disproportionately the ones a serve happens in.  Do not re-tune this on a
throughput benchmark; only an end-to-end recall number can price it.
```

> code: `NEAR_IMGSZ = 640`

```text
640, not 960.  The near frame is a 960x540 proxy, so 960 is native -- but the
near players are 72-275 px tall and survive the downscale, unlike the far
player.  Validated end to end rather than on throughput (see FAR_IMGSZ for why
that distinction matters):

    clip 35   near serve 100%/100% -> 100%/100%   pass 140s -> 57s
    clip 58   near serve 79.5%/61.4% -> 81.8%/62.1%   pass 784s -> 484s
    clip 58   point end  38.3%/25.0% -> 37.0%/26.1%, truncations 0 either way

Accuracy is unchanged to slightly better and the pass is 38-59% cheaper.  The
asymmetry with the far pass is the whole story of this pipeline: near players
are far above the detection size and have headroom, far players have none.
```

> code: `def _stride_for(src_fps, pose_fps):`

```text
Integer stride, never below 1. Returns (stride, effective_fps).
```

> code: `def _pose_pass(video, out_path, stride, imgsz, device, to_analysis=None,`

```text
All-persons pose over every `stride`-th frame, written decimated.

`offset`/`scale` map detections back into the 960x540 ANALYSIS frame, which
is the one frame every consumer and the homography agree on.  A crop proxy
produces coordinates in the crop; without this they would silently be a
different coordinate system wearing the same field names.

`drop_bottom_edge` discards detections whose box bottom sits ON the crop's
lower edge.  THIS IS NOT A QUALITY FILTER -- it is a refusal to invent a
ground point.  A person who straddles the bottom of the far band is cut off
by the crop, so their box bottom is the CROP BOUNDARY and not their feet,
and projecting it through a ground-plane homography puts them somewhere they
have never stood.

It matters far more than it sounds.  Measured across six clips, 23-61% of
far-band detections are truncated this way, and because they are the people
standing closest to the camera they carry SYSTEMATICALLY HIGHER confidence
than the real far player (0.82 vs 0.54 on Data/40).  So they won every
confidence-weighted competition for the two far slots and pushed the actual
server out of the tracks entirely -- which is why Data/40's serve trophy was
present at full strength in the band for all 13 labelled serves while the
tracked slots caught six.
```

> code: `cancel.check()`

```text
The pose passes are most of a run's wall time, so this is
where Cancel has to be able to land. Once per batch is a
sub-second check-in and costs an already-predicted branch.
```

> code: `assert_decode_complete(label, video, f - 1, total - 1, src_fps)`

```text
A mid-file decode failure and a clean EOF are the same `ok=False` to the
loop above, so without this a pass that saw the first two minutes of a
forty-minute match returns a perfectly well-formed npz and every stage
downstream reads the silence as "the player was never there".  Every
legacy telemetry pass has this check; this one did not, and a joined
multi-chapter file is exactly the shape of input that provokes it (see
pipeline/videoio.py for the GoPro case that shipped).  `limit` truncates
`total` on purpose, so the expectation is already the right one.
```

> code: `np.savez_compressed(out_path, kp=kp, box=bx, conf=cf,`

```text
The crop travels WITH the detections.  It is the coordinate frame they
were produced in, and `far()` needs it to tell a cached pass that still
covers the right pixels from one aimed at a band the camera has since
left -- a distinction the file's existence alone cannot make.
```

> code: `def near(video, device="mps", pose_fps=POSE_FPS, force=False, limit=None):`

```text
540p whole-frame proxy, all-persons pose at `pose_fps`.
```

> code: `prox = P.ensure_proxy(video, size=C.ANALYSIS_SIZE, crf=14, label="PROXY540")`

```text
CRF 14, not the default 20: the proxy is shared, and a detector that
wants the ball off it needs the ball to have survived the re-encode.
```

> code: `def _band_union(video, ref_pts):`

```text
Where the reference band's points land in EVERY sampled frame.

Returns a [M, 2] stack in analysis pixels: the reference points themselves
plus their position under each camera-track warp.  The caller takes one
bounding box around the lot.  With no cached track there is one warp, the
identity, and this returns `ref_pts` unchanged.
```

> code: `outs = [ref_pts]`

```text
Only the EXTREMES of the track can widen a bounding box, but which sample
is extreme differs per corner, so all of them are transformed.  It is 18
points against a few thousand 3x3s -- milliseconds, once per video.
```

> code: `FAR_BAND_UP_M = 2.8`

```text
How far above the ground the far band must reach, in metres.  A standing
player is ~1.8 m; a server at the trophy has the racket tip near 2.6 m, and
THAT is the far-serve signal -- a band that clips the raised arm deletes the
thing it exists to capture.
```

> code: `FAR_BAND_PAD_M = 4.0`

```text
along the court, either side of the far baseline
```

> code: `def far_band(video, pad_m=FAR_BAND_PAD_M, up_m=FAR_BAND_UP_M):`

```text
Native-res crop rectangle around the far baseline, in SOURCE pixels.

Derived from the court homography rather than clicked: the far baseline is
a known line in court metres, so its image position is already known, and
asking the user for a second rectangle would be asking for something the
calibration already contains.

HEIGHT IS NOT FORESHORTENING.  The homography is a GROUND-plane map, so it
can place the baseline but cannot project a standing player's head -- there
is no height in it.  Taking the band from the ground strip alone gives ~55
px on a 4K clip, which is a third of a far player and none of a raised arm.
So the vertical extent is derived instead from the image SCALE at that
depth: the band's own width spans a known number of court metres, giving
px-per-metre there, and the band grows upward by `up_m` of them.  That
scales with the camera instead of being a pixel constant tuned on one clip.

THE BAND IS THE UNION OVER THE WHOLE CAMERA TRACK, not the band at
calibration time.  This crop becomes a fixed ffmpeg rectangle, and ffmpeg
cannot be asked for a rectangle that moves; a far player who leaves it is
not mis-projected, they are absent, and no amount of correct geometry
downstream brings them back.  So every sampled warp is applied to the
reference band and the crop is taken around all of them at once.  Paying for
a slightly larger decode is the cheap side of that trade -- and on a camera
that never moved the union is the reference band exactly.
```

> code: `px_per_m = ((ref_img[:, 0].max() - ref_img[:, 0].min()) * sx) / (C.X_HI - C.X_LO)`

```text
px-per-metre comes from the REFERENCE band's width, not the union's: the
union is wider than the court by however far the camera moved, and using
it here would inflate the upward growth by that same error.
```

> code: `def _crop_of(dets_npz):`

```text
The crop a cached pose pass was run over, or None for an old file.
```

> code: `def far(video, device="mps", pose_fps=POSE_FPS, force=False, limit=None):`

```text
Native-resolution far-baseline band, all-persons pose at `pose_fps`.
```

> code: `cached_crop = _crop_of(out)`

```text
Existence is not freshness.  The band widens the first time a camera
track finds a jostle, and a cached pass over the OLD band is missing
exactly the frames the widening exists to recover -- silently, since
a missing far player looks the same as a far player who is not there.
```

> code: `from pipeline.anya2 import camera as CAM`

```text
A file written before the crop was recorded reports None.  That is
only ambiguous when the band could have MOVED: with a static camera
track the band is the calibration band, which is what any older file
was cut from, so the cache is kept rather than charging every existing
video a fresh far pose pass for a camera that never moved.
```

> code: `return _pose_pass(prox, out, stride, imgsz=FAR_IMGSZ, device=device,`

```text
imgsz 768, not 960.  The far band is ~730 px wide, so 768 is essentially
native and 960 only pays to letterbox padding.  Measured on Data/23's band:

    imgsz 960   52.6 ms/frame   1.02 persons/frame
    imgsz 768   35.2 ms/frame   1.00 persons/frame   <-- here
    imgsz 640   26.3 ms/frame   0.06 persons/frame   <-- collapses

A third off the cost for no lost detections, and a hard cliff immediately
below: at 640 the band is downscaled and the far player falls under the
model's detection size, which is the same scale limit that made a
whole-frame 540p pass useless for the far side in the first place.
Crop pixels are SOURCE pixels; map them back to the analysis frame.
```


---

## tracks.py

### Module docstring

```text
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
```

> code: `MIN_H_PX = 12.0`

```text
── tracking admissibility (loose; see the module docstring) ─────────────
```

> code: `MIN_H_PX = 12.0`

```text
below this is a spectator on another court, or noise.
Lower than select_near's 25: that threshold was tuned
for the NEAR player, who is 100-300 px tall here, and
it deletes the far player outright -- measured on
Data/21, a far player at the far baseline is ~20-40 px
in the 960x540 analysis frame.
```

> code: `X_TRACK_PAD = 6.0`

```text
metres outside the singles lines that a tracked person
may wander (ball carts, walkways, the bench)
```

> code: `NEAR_BACK_M = 6.0`

```text
...and behind the near baseline
```

> code: `FAR_BACK_M = 12.0`

```text
...and behind the far one.  12 m is not "8 m plus slack" -- it is sized from
the MEASURED projection error at far-court depth, and it is large because that
error is large and mostly systematic.  A far player's ground point is 22-32 px
up the frame, where two pixels of box-bottom error is metres of court, and the
box bottom sits above the true feet more often than below.  Across Data/23, 24
and 25 (34k far-band detections) the projected court_y of a player standing ON
the far baseline exceeds it by a MEDIAN of 4 m, with p90 at 5.7 and p99 at
12.8.  At 8 m the bound was silently deleting a third of clip 23's far
detections -- real players, rejected for standing where the homography said
they were.  12 m keeps 98.7%.

The looseness costs less than it looks: far detections come from the far-band
crop, which already bounds the region physically, and the lateral doubles+3ft
gate still applies.  This bound is a backstop against a spectator high in the
frame, not the thing deciding who is a player.
```

> code: `HEIGHT_MAX_RATIO = 1.40`

```text
── height must agree with depth ─────────────────────────────────────────
A person standing on the court has a PREDICTABLE pixel height: the court's own
width at their depth gives px-per-metre there, and a player is ~1.75 m.  The
ratio of observed to expected height is tight -- across four clips and both
sides its median is 0.82-1.06 and its p90 never exceeds 1.24.

A candidate whose box is far larger than its projected depth allows is not a
player at that depth; it is a CLOSER player whose ground point has been
mis-projected.  Found by rendering a frame: on Data/21 at t=60.3 s the near
player, having moved forward, appeared inside the far-band crop and was
tracked as a FAR player with a box of 69 px where the far baseline allows 26.

The bound is one-sided.  A box SMALLER than expected is ordinary -- a crouched
player, a partial detection, a crop clipping the head -- and rejecting those
would cost real tracking.  Only "too big for where you claim to be standing"
is a contradiction.
```

> code: `HEIGHT_MAX_RATIO = 1.40`

```text
observed / expected; p90 is 1.24
```

> code: `LOST_FRAMES = 60`

```text
keep a lost slot's anchor this long before freeing it
```

> code: `MAX_SPEED = 9.0`

```text
m/s; faster than a sprint means a different human
```

> code: `W_CONT = 2.0`

```text
── assignment weights ───────────────────────────────────────────────────
```

> code: `W_CONT = 2.0`

```text
continuity bonus scale, exp(-d / CONT_SCALE)
```

> code: `W_HOME = 1.0`

```text
weight on proximity to the player's OWN baseline
```

> code: `def tracks_path(video_path, suffix="_anya2_tracks.npz"):`

```text
THE CLAIM SCORE PREFERS PLAYERS NEAR THEIR OWN BASELINE, replacing an earlier
box-height term that was quietly side-specific.

Box height is a DEPTH proxy, not a quality one.  On the near side bigger means
closer to the camera, hence closer to the near baseline -- so preferring the
biggest box happened to prefer the right people, and the near detector reached
100% recall with it.  On the far side the same term is exactly backwards:
bigger means closer to the NET, and the server is by definition the deepest
player on their side.

Measured on Data/25 (doubles), at every one of the seven missed far serves
three candidates competed for two far slots, and the server was every time the
DEEPEST (court_y 22.1-27.6 against 17.7-19.7) and also the LOWEST-confidence
(0.73-0.79 against 0.85-0.92).  The height term handed both slots to the two
shallowest and dropped the server -- while that serve's trophy shape was
present at full strength in the band, in all ten serves.  Ranking by
confidence instead would have dropped the server too.

Proximity to one's own baseline is the rule that is right on both sides at
once: near players are nearest y=0, far players nearest y=COURT_L, and the
candidate least likely to belong to a side is the one closest to mid-court --
which is also where a mis-projected player from the OTHER side lands.
```

> code: `def _px_per_m(H_inv, cy):`

```text
Image px per court metre at depth `cy`, from the court's width there.
```

> code: `def _height_plausible(h, cy, H_inv):`

```text
Is a box of height `h` consistent with standing at depth `cy`?

One-sided -- see HEIGHT_MAX_RATIO.  Unknown depth passes: this rule catches
one specific contradiction, it is not another way to lose a player.
```

> code: `def _homeness(cy, side):`

```text
1.0 at the side's own baseline, 0.0 at mid-court (and beyond). NaN -> 0.

Clipped above 1.0 rather than rewarded: the far player's ground point
routinely projects several metres PAST the baseline they are standing on
(median +4 m -- see FAR_BACK_M), so "even deeper" is noise, not evidence.
```

> code: `def _trackable(cx, cy, side):`

```text
Loose zone membership, per side. NaN -> False.
```

> code: `class _Slot:`

```text
One player slot: its last known court position and how stale it is.
```

> code: `def hit(self, court):`

```text
The anchor is NOT cleared when it goes stale.  A stale anchor stops
being usable for continuity (see `anchored`) but stays usable as a
RE-ACQUISITION HINT -- "this slot was last holding somebody here".

Clearing it is what made a single player oscillate between slots: on
a singles clip the near player was split 47%/49% across the two near
slots with 38 identity switches in 420 s, because once both slots had
expired the next detection went to whichever slot index came first,
not to the slot that had been following that player.  Detection
survived it (the cross-slot refractory in near_serve merges the
halves) but the answer to WHICH PLAYER SERVED did not, and in doubles
that answer is the point.
```

> code: `def _shortlist(cands, slots, fps):`

```text
Cut a side's candidates down to at most `len(slots)`, by BELONGING.

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
```

> code: `def _assign_side(cands, slots, fps):`

```text
Assign candidates to this side's slots. Returns {slot_idx: cand_idx}.

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
```

> code: `if d / max(slot.since, 1) * fps > MAX_SPEED:`

```text
The outright rejection, not a penalty. See the module docstring.
```

> code: `for _, ci in claims:`

```text
Strongest claim first, and each one goes to the free slot whose STALE
anchor is nearest -- not to the lowest free index.  A slot that never
held anybody has no anchor and sorts last, so a genuinely new player
still lands somewhere.
```

> code: `def _stack(near_npz, far_npz):`

```text
Merge the two ROIs' detections into one per-sample candidate list.

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
```

> code: `na, nb = len(a["conf"]), len(b["conf"])`

```text
The two passes must be on ONE timeline.  They are extracted at the same
pose_fps from the same source, so they should already agree; a mismatch
means one is stale and silently interleaving them would shift every event
by an unknown amount.  Refuse rather than guess.
```

> code: `geom = C.Geometry(video)`

```text
The court map is read PER FRAME, not once.  A bumped camera leaves the
cached corners describing a court that is no longer where they say, and
nothing here would notice: every projection still returns a number, just
the wrong one, for every frame after the bump.  `Geometry` composes the
clicked homography with the camera track's warp for that frame, and with
no track cached it IS the clicked homography -- see anya2/camera.py.
```

> code: `stride = int(z["stride"]) if "stride" in z else 1`

```text
Pose sample f is source frame f * stride (`perceive._pose_pass` writes
`idx = range(0, total, stride)`), and the camera track is keyed by source
frame because that is the one clock the two decimated passes share.
```

> code: `nocc = int(seen[:, ss].sum())`

```text
Eligibility is reported over OCCUPIED slot-frames only. Over all slot
-frames it is dominated by empty slots, which says nothing about the
gate -- on a singles clip half the near slots are empty by definition.
```

> code: `def load(video, path=None):`

```text
Load a tracks npz as a dict of arrays.
```


---

## near_serve.py

### Module docstring

```text
near_serve.py
=============
A near-side serve detector built from POSE ALONE — no ball evidence anywhere.

Why a second near detector at all
---------------------------------
`anya_near_serve` scores three cues (dwell, toss, ratio-jerk) and two of them
saturate, so in practice its P is dwell-only: "the near player stood near the
baseline and held still".  That is a description of a player about to serve AND
of a player who has just picked up a ball, is arguing a call, or is waiting for
their opponent.  Measured against Data/77's tags, 33 of 44 false positives fell
in tagged DEAD TIME rather than mid-rally, which is exactly what a dwell-shaped
cue should be expected to do.

The fix is not a higher threshold on the same cue.  It is to require the thing
that actually distinguishes a serve from standing still: THE SERVE MOTION.

The motion, and what survives 15 Hz
-----------------------------------
The service action has three phases with distinct, pose-visible shapes:

  1. READY     hands together on the racket, both wrists carried between hip and
               shoulder, body quiet.  Lasts 1-4 s, so at the 15 Hz the pose npz
               is written at this is 15-60 samples.  Abundant, and cheap.

  2. TROPHY    the hands SPLIT and both go up.  The tossing wrist rises clear
               above the head; the racket wrist is above the shoulder with the
               elbow flexed.  Held ~0.2-0.5 s, so 3-7 samples.  Thin, but this
               is the discriminative core: "both wrists above the shoulder line,
               one of them above the head" is a shape that between-point
               behaviour essentially never produces.

  3. SWING     the racket wrist accelerates to a peak above the head while the
               tossing wrist DROPS.  The two wrists trade vertical order.  This
               takes 0.10-0.15 s — 1.5-2 samples at 15 Hz.

That last number sets a hard design rule: **the swing is read as a position
extreme, never as a velocity or an acceleration.**  At 15 Hz the contact spans
about two samples, so any derivative across it is dominated by sampling phase
rather than by the stroke — the same trap that made the ratio-jerk cue
meaningless at 5 fps (see NearServeConfig.jerk_mode).  What IS robust across two
samples is "the racket wrist got higher than it had been", because an extremum
does not care where in the motion the samples landed.

Three ideas carry the detector, and only the first is in the phase list above:

  * THE HAND SPLIT.  Ready has the hands together on the grip; trophy has them
    as far apart as they get all point.  The transition together -> apart is
    sharper and better sampled than anything inside the trophy itself.

  * THE WRIST-ORDER CROSSOVER.  At trophy the tossing wrist is the higher one;
    a fifth of a second later the racket wrist is.  Nothing in the between-point
    repertoire swaps which arm is on top while both are above the shoulders.

  * THE ORDERING ITSELF.  ready -> trophy -> peak, each within a bounded delay.
    A single thresholded shape is what `anya_far_serve`'s hand-raise gate is,
    and its measured cost is 29-37 false positives per corpus.  Requiring the
    three to arrive in sequence is what a lone raise cannot fake.

What pose alone genuinely cannot do
-----------------------------------
An OVERHEAD SMASH is the same motion.  Trophy shape, wrist crossover, peak above
the head — a smash has all of it, and no amount of keypoint work separates the
two, because they are not different at the joints.  The separators are
contextual, and one of them is still ball-free: a serve is struck at or behind
the baseline, a smash almost never is.  So `court_y` gates the detector when a
homography is available (`require_court=True`, the default when it is).  This is
pose + court geometry, not ball evidence.  Without the gate, expect smashes.

Handedness is never assumed.  Every measurement is over "the higher wrist" and
"the lower wrist", or over the two sides symmetrically, so a lefty and a righty
score identically and no per-clip configuration is needed.

Reference frames
----------------
Image y grows DOWNWARD, so "above" is a SMALLER y.  Every elevation below is
written as (reference_y - point_y) / body_height, i.e. positive = higher up, in
body heights — the same normalisation `near_end` uses, and for the same reason:
the near player's box ranges over tens of pixels of height as they move, so a
pixel threshold means different postures at different depths.

The head reference deserves its own note.  The near player has their BACK to
this camera, so NOSE/EYE/EAR keypoints are exactly the ones a pose model is
least sure of here.  `_head_y` therefore uses the detected head when it is
confidently seen and falls back to a fixed offset above the shoulder line when
it is not — never NaN, because "we could not see the face" must not be allowed
to read as "the wrist was not above the head".
```

> code: `NOSE = 0`

```text
COCO indices, as in near_end / walking.features.
```

> code: `MAX_GAP_S = 0.5`

```text
as near_end: fill short occlusions, leave real holes
```

> code: `HEAD_ABOVE_SHO_BH = 0.14`

```text
── head reference ───────────────────────────────────────────────────────
```

> code: `HEAD_ABOVE_SHO_BH = 0.14`

```text
fallback head height above the shoulder line, body
heights.  A head is ~0.13 of standing height and the
shoulder line sits just below it; erring LOW makes
"above the head" easier to satisfy, so the fallback
is the permissive direction and the gate leans on
the trophy shape rather than on this constant.
```

> code: `TOGETHER_BH = 0.10`

```text
── ready phase ──────────────────────────────────────────────────────────
```

> code: `TOGETHER_BH = 0.10`

```text
wrist separation at or under this = hands on the grip
```

> code: `SPLIT_BH    = 0.26`

```text
...and at or over this they are definitely apart
```

> code: `READY_HI_MAX_BH = 0.10`

```text
in ready, even the higher wrist stays under this far
above the shoulder line.  Not below the shoulder:
plenty of players carry the racket head-high while
waiting, and rejecting that costs real serves.
```

> code: `READY_STILL_BH_S = 0.9`

```text
limb speed relative to the hip, body heights/s,
under which the player counts as quiet.  Looser than
near_end.SETTLE_LIMB_BH_S (0.75) on purpose: a
server bouncing a ball is NOT still by that standard
but is unambiguously in the ready phase.
```

> code: `READY_WIN_S = 0.7`

```text
sustain window; a serve stance is held
```

> code: `TROPHY_ABOVE_HEAD_BH = 0.058`

```text
── trophy phase ─────────────────────────────────────────────────────────
── how strict is the trophy? ────────────────────────────────────────────
These three full-credit points were loosened by 35% of their range toward
their own no-credit floors, because SOME SERVERS SIMPLY DO NOT MAKE A TEXTBOOK
TROPHY -- a compact or abbreviated service motion carries the racket lower and
splits the hands less, and the original bands scored those at nearly zero
however clean the rest of the sequence was.

Swept over all 107 labelled near serves.  Loosening trades a little precision
for recall and the trade is worth taking for a point START, where a miss loses
a whole point from the reel and an extra start is something the composition
layer can arbitrate:

    loosen   recall   precision   F1
     0.00     86.9%     78.2%    82.3      (the original bands)
     0.20     88.8%     77.9%    83.0
     0.35     90.7%     76.4%    82.9      <-- here
     0.50     89.7%     74.4%    81.4

0.35 is where recall peaks; 0.50 gives it back, so this is a maximum and not
a slope being ridden. F1 is flat across 0.20-0.35, so the choice between them
is the recall/precision preference above, not a measurement.

TROPHY_MIN is deliberately NOT part of this. Swept from 0.35 down to 0.16 it
changes nothing at all -- the run threshold is not binding, because the final
probability threshold already dominates it. Lowering it would look like
loosening while doing nothing.
```

> code: `TROPHY_ABOVE_HEAD_BH = 0.058`

```text
higher wrist this far over the head = full credit
```

> code: `TROPHY_HEAD_MIN_BH   = -0.02`

```text
...and no credit at all below head height
```

> code: `TROPHY_LO_MIN_BH = -0.13`

```text
The lower wrist is measured against the SHOULDER LINE, and the band below is
set from data rather than from anatomy-by-description.  The original 0.00/0.14
encoded "the other wrist is also up", read literally as above the shoulder.
That is wrong: at trophy the racket ARM is up but the racket WRIST hangs just
under the shoulder with the elbow flexed — across Data/21's 11 tagged near
serves the lower wrist sits at -0.096..+0.079, median -0.028.  The original
band scored every one of those at or near zero.  Away from serves the same
quantity has p50 -0.276 and p95 -0.157, so the discriminative band is narrow
and sits just below the shoulder, not above it.
```

> code: `TROPHY_LO_MIN_BH = -0.13`

```text
lower wrist this far BELOW the shoulder = no credit
```

> code: `TROPHY_LO_FULL_BH = -0.052`

```text
...and near the shoulder = full credit.
This is the term that rejects the whole one-armed
family (hand to cap, hand to face, a wave, a
raised finger): those leave the other wrist down.
```

> code: `TROPHY_SPLIT_MIN_BH = 0.149`

```text
hands must have come apart to score at all
```

> code: `TROPHY_MIN = 0.35`

```text
run threshold for calling a sample "trophy"
```

> code: `TROPHY_DILATE_S = 0.20`

```text
THE TROPHY IS A PHASE, NOT AN INSTANT.

The three terms above are multiplied, so as written they must all hold ON THE
SAME SAMPLE.  Measured on Data/38's five missed serves, they do not: at each
one every term comfortably clears its own threshold somewhere in the window
(hi_head peaks at +0.19..+0.27 against a 0.10 full-credit line, lo_elev at
+0.16..+0.29 against -0.01, gap at 0.33..0.61 against 0.18) while the PRODUCT
never exceeds 0.30 and mostly sits at 0.00.  The terms are satisfied; they are
just not satisfied simultaneously.

That is a sampling artefact, not a shape failure.  The trophy is held for
0.2-0.5 s, which is 3-7 samples at 15 Hz, and within it the tossing arm
reaches full extension before the racket arm settles -- so the two elevation
terms peak a sample or two apart and the conjunction falls between them.  It
is the same class of problem as reading contact off a derivative, and it has
the same answer: measure the phase over a window rather than at a point.

So each term is dilated (running max) before the product.  The conjunction
then asks "did this hold ANYWHERE NEAR now", which is what a phase means,
instead of "did all three hold in this one 67 ms sample".
```

> code: `TROPHY_DILATE_S = 0.20`

```text
+/- this around each sample; ~3 samples at 15 Hz,
comfortably inside the 0.2-0.5 s the trophy is
held, so it cannot merge two separate actions
```

> code: `SWING_MIN_S = 0.06`

```text
── swing phase ──────────────────────────────────────────────────────────
```

> code: `SWING_MIN_S = 0.06`

```text
contact cannot precede the trophy by less than one
sample, so the search opens just past it
```

> code: `SWING_MAX_S = 0.90`

```text
...and a real trophy-to-contact is 0.15-0.45 s; the
slack is for a slow recreational action, not for
finding an unrelated arm raise a second later
```

> code: `PEAK_ABOVE_HEAD_BH = 0.06`

```text
the racket wrist must clear the head at contact
```

> code: `READY_BACK_MIN_S = 0.20`

```text
The tossing arm's descent is DELIBERATELY NOT REQUIRED.  The original design
read contact as a wrist-order crossover — racket wrist up AND toss wrist
falling away — but measured on Data/22 that second half is simply not true at
these sample rates: across 7 missed serves the racket wrist peaked above the
head on time (0.20-0.23 bh at +0.20..+0.53 s) while the tossing arm had not
yet come down, so the crossover term zeroed a swing that had plainly happened.
The toss arm drops AFTER contact, not by it.  Contact is therefore read as the
racket wrist clearing the head, full stop, and the tossing arm is free to stay
up through the whole swing.
```

> code: `READY_BACK_MIN_S = 0.20`

```text
── ready lookback ───────────────────────────────────────────────────────
```

> code: `READY_BACK_MIN_S = 0.20`

```text
the ready phase ends as the hands split, so the
window closes just before the trophy
```

> code: `READY_BACK_MAX_S = 6.00`

```text
and opens far enough back to cover a long ritual
```

> code: `FT_TO_M = 0.3048`

```text
── court gate ───────────────────────────────────────────────────────────
`walking.court.load_homography` maps the clicked corners so that the NEAR
baseline is court_y = 0 and court_y grows toward the far end — verified on
Data/21, where the near player's court_y sits at -0.07..+0.10 across all 11
tagged near serves and ranges to +12.8 during rallies.  A gate written the
other way round (">= 21.3", i.e. assuming the near baseline was at COURT_L)
rejects 100% of samples, which is how this was found.
The gate is a BAND on the server's FEET, not a one-sided test on the body.
`walking.features.frame_signals` already projects the ankle midpoint (falling
back to the bbox bottom-centre when both ankles are lost) through the court
homography, so `court_y` is the ground-contact point — which is what a
standing-position rule has to be measured on.

The band is the same idea as `anya_near_serve.NearServeConfig`'s ready zone
(zone_y_min_ft -3.5, zone_y_max_ft 0.5), sharing its lower edge so the two
near detectors agree about how far back a server may stand.  The court-side
edge is 1.0 ft rather than 0.5: the ready zone is scored on a player who is
still settling, whereas this gate is evaluated at the TROPHY, by which point
the front foot has rocked forward into the swing and a half-foot ceiling
clips real serves.
```

> code: `SERVE_ZONE_MIN_FT = -3.5`

```text
behind the baseline; a server stands back from it
```

> code: `SERVE_ZONE_MAX_FT =  1.0`

```text
...and no further INTO the court than this, which is
what excludes mid-court overheads and smashes
```

> code: `W_TROPHY, W_READY = 0.45, 0.20`

```text
── combination / events ─────────────────────────────────────────────────
The shape terms are additive; the SWING is a multiplicative gate with a floor.

That asymmetry is the whole arbitration, and it was chosen against a measured
failure rather than picked for elegance.  Under a flat additive form
(0.45/0.35/0.20) two behaviours scored 0.600 and fired: a player who stands
quiet with the hands together and THEN raises both arms to stretch or adjust
strings, and a server who tosses and catches an abandoned toss.  Both have a
real ready phase and a textbook trophy shape.  Neither has a racket wrist that
peaks above the head while the tossing wrist falls away, because in neither
case was a ball ever struck.

Additive weighting cannot fix that: trophy and ready are genuinely present, so
any threshold that rejects those two also rejects real serves scoring the same
0.65 on the same two terms.  The swing has to be able to VETO, which means it
multiplies.  The floor keeps the veto soft — at 15 Hz contact spans about two
samples, so a real serve can be caught mid-stroke with a partial swing score,
and a hard gate would throw those away.
```

> code: `SWING_FLOOR = 0.45`

```text
what a candidate retains with NO swing evidence at
all.  Chosen so the two adversarial cases above land
at 0.42 (rejected) while a serve seen with only half
its swing lands at 0.67 (kept).
```

> code: `THRESHOLD = 0.70`

```text
Swept over the nine labelled clips with a near serve on them.  The curve is
flat from 0.60 to 0.70 (recall 100%, 4 false positives) and only starts
costing recall at 0.75, so this is the top of a plateau rather than a point
fitted to the corpus -- which is the only reason a threshold chosen on the
same clips it is scored on is worth anything.  Clip 58's 44 near serves are
the holdout.
```

> code: `REFRACT_S = 3.0`

```text
as anya_near_serve.event_refract_s
```

> code: `SERVE_LEAD_S = 1.63`

```text
── when is the point start? ─────────────────────────────────────────────
The seed reported the TROPHY ONSET as the serve time, and measured against
`ground_truth.json` that is systematically LATE: over 46 matched near serves
on nine clips the trophy onset lands +1.63 s after the label, sd 0.66.

The first fix is physical.  The seed's own docstring names the hand split as
the sharpest, best-sampled transition in the whole action, but only ever uses
it as a SHAPE term inside the trophy product, never as a TIME anchor.  Walking
back from the trophy peak to the last sample at which the hands were still
together on the grip -- the last instant of the ready stance -- moves the
residual to +1.13 s and, more usefully, cuts the share of detections already
inside +/-2.0 s of the label from 63% to 84%.

The remaining +1.13 s is NOT physical and is not treated as though it were.
`ground_truth.json`'s `start` is a POINT boundary, not a stroke event: it is
marked before the server's hands ever move, so no definition taken from the
serve motion can reach it.  Every candidate onset tried (trophy onset, hands
together, ready>=0.5, wrists-still-low) left a residual of +1.10 to +1.63 s,
which is what a convention offset looks like and not what a mis-detection
looks like.  So it is corrected as one explicitly-fitted constant.

It is ONE parameter over 45 observations, and it was validated the only way
that means anything -- LEAVE-ONE-CLIP-OUT, refitting on the other eight clips
and scoring the held-out one: 44/45 (98%) land within +/-2.0 s.  Per-clip
medians span 0.93-2.16 s, so the constant is not equally right everywhere;
it is right within the tolerance the corpus is scored at.  A detector asked
to hit a tighter tolerance than 2 s would need a per-clip estimate, and this
constant should be refitted, not trusted, if the labelling convention changes.
1.63 is the MEDIAN OVER ALL TEN labelled clips carrying a near serve.  Fitted
on the nine short clips alone it was 1.13, and clip 58 -- a full 55-minute
match, held out until its perception pass finished -- wanted 2.4.  That is
the honest headline of this constant: the per-clip lead genuinely spans
0.9-2.4 s, and no single value satisfies both ends at a +/-2.0 s tolerance.
Measured directly: at 1.13 clip 21 scores 100% and clip 58 57%; at 2.00 clip
21 drops to 91% and clip 58 reaches 68%.  Clips 22-50 do not move at all.

So this is a compromise, not a fit, and it is a small effect: the lead is
worth about nine points of recall on clip 58 and the remaining third of that
clip's misses are a different problem (see the README).  A detector required
to hold a tighter tolerance would need a PER-CLIP lead, estimated from the
clip's own detections rather than from this constant.
```

> code: `SERVE_LEAD_S = 1.63`

```text
subtracted from the hands-together time
```

> code: `TROPHY_LEAD_S = 2.13`

```text
...and from the trophy onset, when the hands were
never seen together (occluded grip).  Fitted on the
same 46 detections, so both paths land on the same
convention rather than one silently drifting.
```

> code: `TOGETHER_BACK_S = 6.00`

```text
how far back from the trophy peak to look for the
ready stance; matches READY_BACK_MAX_S
```

> code: `def _kin():`

```text
`walking.features`' pose helpers, imported lazily — see near_end._kin.
```

> code: `def _ramp(x, lo, hi):`

```text
Soft 0->1 ramp from lo to hi (either order); NaN in, NaN out.
```

> code: `def _movmax(x, w):`

```text
Centred running max over `w` samples, NaN-tolerant (NaN acts as -inf).

Written with shifts rather than a stride trick so the edges degrade to a
shorter window instead of to NaN -- a serve in the first second of a clip
is still a serve.
```

> code: `def serve_primitives(kp, bbox, fps: float,`

```text
Per-sample phase primitives for one near-player pose track.

`kp` [N, 17, 3] and `bbox` [N, 4] are one near slot of `anya2.tracks`,
the same layout `near_end.near_signals` consumes.  `court_y` [N] in metres is
that slot's court position and enables the baseline gate.  `eligible` [N]
is `anya2.tracks`' strict lateral gate -- the user's "inside the doubles
court plus 3 ft" rule -- and is ANDed into it.

The two gates are separate on purpose and are not interchangeable.
`eligible` asks WHO IS A PLAYER ON THIS COURT; the serve-zone band below
asks WHO IS STANDING WHERE A SERVER STANDS.  A receiver passes the first
and fails the second, which is why both are kept.

Returns arrays of length N.  Scores are in [0, 1] with NaN where the player
was not tracked; geometry terms are in body heights and may be NaN freely.
```

> code: `head_y = np.where(np.isfinite(head[:, 1]),`

```text
Head reference: measured when seen, geometric when not.  Never NaN — see
the module docstring on why a missing face must not read as a low wrist.
```

> code: `elev_l = (sho[:, 1] - l_wri[:, 1]) / h`

```text
Elevations, positive = higher in the frame, in body heights.
```

> code: `both = np.isfinite(elev_l) & np.isfinite(elev_r)`

```text
Both wrists must actually be SEEN for a two-armed shape to mean anything:
np.fmax ignores NaN, so without this an occluded second arm would be
silently read as whatever the visible arm is doing.
```

> code: `hi_side = np.where(both, np.where(elev_l >= elev_r, 1.0, -1.0), 0.0)`

```text
Which side is carrying the high wrist: +1 left, -1 right, 0 unknown.
```

> code: `def rel_speed(pt):`

```text
── limb quiet, for the ready phase ──────────────────────────────────
Relative to the hip, so a player walking into position does not read as
limb activity and a racket take-back while standing still does — the same
construction as near_end's settle term.
```

> code: `together = 1.0 - _ramp(gap_bh, TOGETHER_BH, SPLIT_BH)`

```text
── ready ────────────────────────────────────────────────────────────
```

> code: `dil = max(1, int(round(TROPHY_DILATE_S * fps)) * 2 + 1)`

```text
── trophy ───────────────────────────────────────────────────────────
A product, not a sum: all three have to hold at once.  Both wrists up AND
one over the head AND the hands apart is the conjunction that between-point
behaviour does not produce; any two of the three regularly happen alone.
Each term is dilated over TROPHY_DILATE_S before the product -- see the
constant.  Dilating the RAMPS rather than the raw geometry keeps every
term bounded in [0, 1], so the product is still a score and not a
quantity that a noisy keypoint can drive arbitrarily high.
```

> code: `else:`

```text
A lost foot must not silently pass the gate: court_y is NaN there and
both comparisons are False, so an untracked player is excluded rather
than admitted.  That is the conservative direction for a position rule.
```

> code: `def serve_onset(prim: Dict[str, np.ndarray], k: int,`

```text
Point-start time for a trophy peaking at sample `k`, and how it was found.

Returns (t_seconds, basis) with basis "together" or "trophy" -- the basis is
carried into the event so a timing complaint can be traced to which of the
two calibrations produced it, rather than to an unlabelled number.
```

> code: `tro_lead = lead + (TROPHY_LEAD_S - SERVE_LEAD_S)`

```text
The trophy fallback keeps its offset RELATIVE to the hands-together lead,
so overriding one moves both onto the same convention.
```

> code: `def _runs(mask: np.ndarray) -> List[tuple]:`

```text
[(lo, hi)) index runs of True.
```

> code: `def detect_serves(prim: Dict[str, np.ndarray],`

```text
Sequence-match ready -> trophy -> swing over the primitives.

Each trophy run is one candidate.  The run's peak sample fixes the trophy
time and which arm is tossing; the ready term is searched BACKWARD and the
swing term FORWARD, both in bounded windows.  A candidate that cannot find
its swing scores zero on that term rather than being discarded outright, so
the arbitration stays graded — a strong trophy over a long, clean ready is
still allowed to carry a serve whose contact fell between two samples.

Returns dicts with `t` (trophy onset — see `serve_t` below), `p`, the three
component scores, and the phase timestamps, sorted by time.
```

> code: `a, b = max(0, k - back_lo), max(0, k - back_hi)`

```text
Ready: the best sustained ready score in the lookback window.
```

> code: `toss_left = prim["hi_side"][k] > 0`

```text
Swing: the tossing arm is whichever carried the high wrist at trophy;
the racket arm is the other one.  Contact wants the racket wrist over
the head AND the tossing wrist fallen away from its own trophy height.
```

> code: `shape = (W_TROPHY * s_trophy + W_READY * s_ready) / (W_TROPHY + W_READY)`

```text
Shape first, normalised so a perfect trophy over a perfect ready is
1.0, then gated by the swing — see the SWING_FLOOR comment.
```

> code: `"t": t_start,`

```text
POINT START: see SERVE_LEAD_S
```

> code: `gap = REFRACT_S if refract_s is None else float(refract_s)`

```text
Refractory: keep the strongest candidate in each window.  A serve produces
one trophy run, but a double fault produces two serves 15-25 s apart and
both are real point starts, so this only suppresses the immediate
re-detection of a single action, never a second serve.
```

> code: `from pipeline.anya2 import court as C`

```text
=========================================================================
anya2 plumbing
=========================================================================
Everything above this line is the validated seed, changed only to accept the
substrate's eligibility mask and to carry a slot id.  Everything below is the
redesign: it replaces the seed's single-player `signals_for_video` with a
per-slot pass over `anya2.tracks`.
```

> code: `REQUIREMENT = Requirement(roi=ROI_NEAR, pose_fps=15.0, needs_ball=False,`

```text
What this detector needs from perception.  Pose-only is not an accident: the
seed establishes that the serve MOTION is what separates a server from a
player standing still, and the motion is entirely in the joints.  Carrying
that into Phase E means the near path never has to run a ball model.
```

> code: `def detect_video(video, tracks_npz=None, threshold: float = THRESHOLD,`

```text
Score every near slot and return the serves, as contract `Event`s.

DOUBLES IS WHY THIS IS A LOOP.  Each near slot is scored independently and
on its own merits; the server is whichever slot produced the candidate, and
two slots are free to produce different serves at different times (they
alternate service games).  Singles is the case where slot 1 is always NaN
and the loop runs once for real -- there is no separate code path, and so no
second path to keep correct.

What the loop must NOT do is average or fuse the slots.  A serve is one
player's action; blending the partner's quiet pose into it would dilute
exactly the trophy shape the detector exists to find.
```

> code: `gap = REFRACT_S if refract_s is None else float(refract_s)`

```text
Cross-slot refractory.  Within a slot `detect_serves` has already applied
REFRACT_S; across slots it has not, and it must be applied here rather
than left out: the two near players stand close together at the baseline
and a partner's raised arm can produce a weak candidate alongside the
server's real one.  Keeping the STRONGER is the same arbitration used
within a slot, and it is what decides WHO SERVED in doubles.
```


---

## far_serve.py

### Module docstring

```text
far_serve.py
============
A far-side serve detector: the same phase-sequence idea as `near_serve`, with
the geometry re-derived for a player 30 pixels tall on the other side of the
net.

Why not just run the near detector on the far slots
---------------------------------------------------
Because two of its three trophy terms do not survive the change of viewpoint
and scale, and one of them is actively harmful.  Measured on Data/23's 15
labelled far serves, per pose sample, against every non-serve sample in the
clip:

    hi_head > 0.05      serve 13%   non-serve  0%   (non-serve p99 = +0.002)
    hi_elev > 0.10      serve 14%   non-serve  1%
    lo_elev > -0.05     serve 11%   non-serve  2%
    gap     > 0.18      serve 35%   non-serve 58%   <-- INVERTED

The elevation terms are close to perfectly discriminative: a far wrist above
the head essentially never happens outside a serve.  THE HAND SPLIT IS WORSE
THAN USELESS.  On the near player -- filmed from behind, 200 px tall -- the
hands coming apart off the grip is the sharpest transition in the action.  On
the far player it is not measurable: the arms are extended VERTICALLY at the
trophy, so their horizontal separation is small exactly when the near view
would have it large, while ordinary walking swings the arms apart.  So the far
trophy is elevation-only, and `gap` is not read anywhere in this module.

That asymmetry is the reason these are two detectors and not one with a flag.

What is kept from the near construction
---------------------------------------
  * THE ORDERING.  ready -> trophy -> swing, each within a bounded delay.  A
    lone thresholded shape is what the shipped `anya_far_serve` hand-raise gate
    is, and its measured cost is 39 false positives over 14 clips.

  * THE PHASE DILATION.  Terms are dilated before the conjunction, so the
    product describes a phase and not a single 67 ms sample.  This mattered
    more on the near side than anything else and the sampling argument is
    identical here.

  * THE SWING MULTIPLIES WITH A FLOOR, the shape terms add.  Same arbitration,
    same reason: a raise that never becomes a strike must be vetoable, but a
    serve caught mid-stroke must not be thrown away.

What is dropped, and why
------------------------
  * THE HAND SPLIT -- measured inverted, see above.

  * THE SERVE-ZONE BAND ON court_y.  The near detector gates on the server
    standing within a 1.4 m band about the near baseline, which is what
    separates a serve from a mid-court smash.  The same gate cannot be built
    here: a far player's ground point is 22-32 px tall in the analysis frame,
    and at that depth two pixels of box-bottom error is metres of court.
    Measured on Data/23 the far player's court_y spans 19.7-28.6 m (p5-p95)
    while really moving about three, and sits a median 4 m BEHIND the baseline
    it is standing on.  A 1.4 m band on that quantity is noise.  The far
    player is still required to be on the far side and inside the lateral
    doubles+3ft gate (`tracks.eligible`), because those are coarse enough to
    survive the noise -- 100% of far detections satisfy both.

BOX SANITY, which the near side never needed
--------------------------------------------
Every quantity here is normalised by box height, and at 30 px a bad box makes
those ratios meaningless rather than merely noisy -- a merged two-person box or
a fragment produced apparent wrist separations of 1.4-1.9 body heights, which
is anatomically impossible.  `_sane` rejects boxes by height and aspect before
anything is divided by them.  This is not a detection threshold; it is refusing
to divide by a number that is wrong.

What this detector structurally cannot do
-----------------------------------------
The same thing the near one cannot: know whether a point is already in
progress.  The shipped far detector's 39 false positives over 14 clips broke
down as 17 idle raises in dead time, 14 in-rally overheads and returns, and 8
the far player REACTING to a near serve -- 22 of 39 being "play was live and
the detector did not know it".  None of that is visible from inside a far-player
pose crop, and every far-side pose formulation aimed at it has failed.  This
module therefore does not try: it declares `windows="between_points"` and
leaves the arbitration to the composition layer, which will have the near
serves and the point ends that actually settle it.
```

> code: `KP_CONF = 0.20`

```text
keypoint confidence floor.  Higher than the near
side's implicit 0: at this scale a low-confidence
keypoint is a guess, and a guessed wrist reads as a
posture the player never held.
```

> code: `BOX_H_MIN_PX, BOX_H_MAX_PX = 12.0, 120.0`

```text
── box sanity ───────────────────────────────────────────────────────────
```

> code: `BOX_ASPECT_MAX = 1.6`

```text
wider than this is two people merged or a fragment
```

> code: `HEAD_ABOVE_SHO_BH = 0.14`

```text
── head reference ───────────────────────────────────────────────────────
The far player FACES this camera, so nose/eyes/ears are the keypoints the
model is most confident about -- the exact opposite of the near player, who
has their back turned.  The geometric fallback is therefore rarely used here.
```

> code: `READY_HI_MAX_BH = 0.02`

```text
── ready ────────────────────────────────────────────────────────────────
```

> code: `READY_HI_MAX_BH = 0.02`

```text
both wrists carried at or below the shoulder line.
Tighter than the near side's 0.10 because the far
ready position is measured over a 30 px body and a
loose bound admits half the clip.
```

> code: `READY_STILL_BH_S = 1.2`

```text
limb speed relative to the hip, body heights/s.
Looser than near's 0.9: at 30 px the same real
motion is a smaller pixel motion but a NOISIER
normalised one, so the quiet bar has to allow for
keypoint jitter that the near view does not have.
```

> code: `READY_MEAN_FROM_S = 2.0`

```text
── how the ready phase is READ ──────────────────────────────────────────
The original reading -- the MAX of `ready` anywhere in the six seconds before
the trophy -- is nearly useless, and measurably so: over 105 true far serves
and 101 false positives its median is 1.00 on BOTH, separating them at AUC
62%.  Any six-second window contains some quiet moment, so a returner who
stood still once five seconds ago scores a perfect ready.

Read instead as the MEAN over a short window ENDING AT THE TROPHY, the
separation is AUC 79% (true 0.90, false 0.62).  That is the difference
between "was quiet at some point recently" and "was quiet right up until the
racket went up", and only the second is what a service stance is.
```

> code: `READY_MEAN_FROM_S = 2.0`

```text
window start, before the trophy
```

> code: `READY_MEAN_TO_S = 0.3`

```text
...and end, stopping short of the toss itself
```

> code: `TOSS_PRE_FROM_S, TOSS_PRE_TO_S = -1.3, -0.2`

```text
── pre-serve stillness ──────────────────────────────────────────────────
A SERVER IS STATIONARY BEFORE SERVING; A RETURNER HAS JUST BEEN RUNNING.
That is the single largest thing separating a real far serve from the far
player's return, and it is local to this detector -- no other agent needed.

Measured as the median of the player's own translation (box centre, in body
heights per second) over the window below: true far serves 0.15, false
positives 0.31, separating at AUC 78%.  It is the mirror of the ready term
and independent of it -- ready is about the ARMS, this is about the FEET.

It enters multiplicatively but softened, so it can veto a candidate that is
clearly mid-rally without deleting one whose track was noisy.
── the toss ─────────────────────────────────────────────────────────────
A SEPARATE SCORE, deliberately never folded into `p`.  It answers a different
question -- "did the tossing arm do what a toss does" -- and the orchestrator
arbitrates the two, so keeping them apart is what lets it trade precision
against recall without this module choosing on its behalf.

Read from the TOSSING ARM ONLY, which is the arm carrying the higher wrist at
the trophy (`hi_side`).  Four components, each measured against the trophy
peak k, each ramped to [0, 1] and then AVERAGED:

  low    the wrist STARTED low, below the shoulder, before the rise
  lat    at the top the wrist is offset LATERALLY from its own shoulder
  hold   the wrist stays above head height for a sustained beat
  peak   how far above the head it gets

Every one is weak alone -- over 103 true far serves and 56 false positives
their individual AUCs are 66%, 66%, 64% and 65% -- and the average of the four
reaches AUC 74%, true serves at median 0.72 against 0.33 for false positives.

Two findings worth keeping:

  THE TOSS ARM IS NOT VERTICAL.  The obvious formulation -- the wrist travels
  straight up over its own shoulder -- is BACKWARDS.  True serves have a
  LARGER lateral offset than false positives (0.148 vs 0.090 body heights),
  because a real toss is released up and forward across the body while a
  return's arm stays nearer the shoulder line.  `lat` is scored in the
  direction the data gives, not the one anatomy-by-description suggests.

  AVERAGE, DO NOT MULTIPLY.  A product of the same four scores the same AUC
  (74%) but is degenerate -- most candidates land at exactly zero, so as a
  filter it costs 53% of true serves to remove 88% of false positives.  The
  mean keeps it graded, which is what a separate signal handed to an
  arbitrator has to be.
```

> code: `TOSS_PRE_FROM_S, TOSS_PRE_TO_S = -1.3, -0.2`

```text
window for the pre-rise low point
```

> code: `TOSS_TOP_FROM_S, TOSS_TOP_TO_S = -0.3, 0.6`

```text
window for the extended top
```

> code: `STILL_FULL_BH_S = 0.20`

```text
at or below this the player was stationary
```

> code: `STILL_NONE_BH_S = 0.55`

```text
...and at or above it they were travelling
```

> code: `W_STILL = 0.30`

```text
weight of the stillness veto; see THRESHOLD
```

> code: `TROPHY_HEAD_MIN_BH = -0.02`

```text
── trophy (elevation only -- see the module docstring) ──────────────────
```

> code: `TROPHY_HEAD_MIN_BH = -0.02`

```text
higher wrist below head height = no credit
```

> code: `TROPHY_HEAD_FULL_BH = 0.06`

```text
...and this far above it = full credit.  Lower
than the near side's 0.10: a 0.10 bh margin is
3 px on a 30 px body and is not resolvable.
```

> code: `TROPHY_LO_MIN_BH = -0.18`

```text
lower wrist this far below the shoulder = none
```

> code: `TROPHY_LO_FULL_BH = -0.02`

```text
...and level with it = full credit
```

> code: `SWING_MIN_S, SWING_MAX_S = 0.06, 0.90`

```text
── swing ────────────────────────────────────────────────────────────────
```

> code: `READY_BACK_MIN_S, READY_BACK_MAX_S = 0.20, 6.00`

```text
── ready lookback ───────────────────────────────────────────────────────
```

> code: `W_TROPHY, W_READY = 0.45, 0.20`

```text
── combination ──────────────────────────────────────────────────────────
```

> code: `THRESHOLD = 0.75`

```text
Swept on the nine clips carrying a far serve.  Over the six FAR-DOMINANT
clips (70 of the 77 labelled far serves) the curve reads:

    thr    recall   precision
    0.55   100.0%     65.4%
    0.90   100.0%     71.4%      <-- here
    0.95    98.6%     73.4%
    0.999   94.3%     76.7%

RE-SWEPT after the ready and stillness terms were added.  Those two multiply
the score down, so the old 0.90 knee no longer sits in the same place; over
all 129 labelled far serves on 13 clips:

    ready  w_still  thr     recall  precision   F1
    max      0.0    0.90     81.4%     51.0%   62.7   <- the previous default
    max      0.3    0.85     80.6%     63.0%   70.7
    mean     0.3    0.75     79.8%     65.2%   71.8   <- here
    mean     0.3    0.85     70.5%     72.8%   71.7
    mean     0.5    0.90     56.6%     72.3%   63.5

0.75 with both terms holds recall (-1.6 points) and buys FOURTEEN points of
precision.  Pushing to 0.85 buys another 7.6 of precision for 9.3 of recall,
which is the wrong direction for a point START -- a missed serve loses a
whole point from the reel, an extra one is something the orchestrator can
arbitrate.
```

> code: `SERVE_LEAD_S = 0.90`

```text
The label lead, as on the near side: `ground_truth.json`'s `start` is a point
boundary marked before the server moves, so it is corrected as a constant.
The far anchor is the TROPHY ONSET, not the hands-together instant the near
detector uses -- the hand split is not measurable on a 30 px body (see the
module docstring), so there is no earlier anchor to walk back to.  Swept over
the far corpus, the lead is almost inert: 0.9, 1.2 and 1.6 all produce
identical matches at a +/-2.0 s tolerance, and only below ~0.6 does it start
costing hits.  0.90 sits in the middle of that flat region.
```

> code: `def _sane(bbox):`

```text
Box height, NaN where the box cannot be trusted to normalise by.
```

> code: `def serve_primitives(kp, bbox, fps: float,`

```text
Per-sample phase primitives for one far-player track.

`kp` [N, 17, 3] and `bbox` [N, 4] are one far slot of `anya2.tracks`.
`eligible` [N] is that module's lateral gate.  There is deliberately no
`court_y` parameter -- see the module docstring on why a serve-zone band
cannot be built on the far side.
```

> code: `lat_l = np.abs(l_wri[:, 0] - l_sho[:, 0]) / h`

```text
Lateral offset of each wrist from ITS OWN shoulder (not the midpoint), in
body heights.  The toss score reads this; see the TOSS block for why it
points the opposite way to the obvious guess.
```

> code: `cx = 0.5 * (bbox[:, 0] + bbox[:, 2])`

```text
The player's own translation across the ground, in body heights per
second.  Box centre rather than a keypoint: at 30 px an individual joint
is noisy, while the box as a whole is what moves when the player runs.
```

> code: `def toss_score(prim: Dict[str, np.ndarray], k: int, toss_left: bool) -> Dict:`

```text
Did the tossing arm perform a toss?  Independent of the serve score.

Returns the mean of four ramped components plus the components themselves,
so a low score can be attributed rather than merely observed.  Missing
keypoints yield None, which a caller must NOT read as zero -- an untracked
arm is not evidence of a bad toss.
```

> code: `def detect_serves(prim, threshold: float = THRESHOLD,`

```text
Sequence-match ready -> trophy -> swing, as `near_serve.detect_serves`.
```

> code: `a = max(0, k - int(round(READY_MEAN_FROM_S * fps)))`

```text
Ready as a MEAN over a short window ending at the trophy, not a max
over six seconds -- see READY_MEAN_FROM_S for why the max is inert.
```

> code: `sa = max(0, k - int(round(STILL_FROM_S * fps)))`

```text
Was this player standing still before the racket went up?
```

> code: `s_still = (1.0 - float(S.ramp(float(np.median(mv)),`

```text
An untracked stretch scores 0.5 rather than 0 or 1: not knowing where
the player was is not evidence either way, and this term must not
delete a serve just because the track had a hole before it.
```

> code: `def detect_video(video, tracks_npz=None, threshold: float = THRESHOLD,`

```text
Score every far slot; the server is whichever produced the candidate.
```


---

## point_end.py

### Module docstring

```text
point_end.py
============
When did the point stop?  Pose only -- the ball is not read anywhere in this
module.

Why no ball
-----------
The shipped policy makes the ball trace the primary point-end signal, and it
works on hard courts.  It is not dependable on clay, where the ball is low
contrast against the surface for much of its flight, and a point-end policy
whose primary evidence disappears on one surface is not a policy.  So this
module is built from player pose, and the ball is left for a later, MINOR
corroborating role if one is ever shown to earn its cost.

Two measured facts shape everything here
----------------------------------------
FACT 1: INSTANTANEOUS ACTIVITY DOES NOT SEPARATE LIVE FROM DEAD.  Measured over
four clips, per-sample player motion and limb energy separate a live point from
the three seconds after it at an AUC of 38-75% -- at or BELOW chance on the two
hardest clips.  The reason is not subtle: a rally contains long quiet beats
while the opponent plays the ball, and dead time contains a player walking to
retrieve it.  Motion does not stop at the end of a point; it CHANGES CHARACTER.
Any construction that thresholds "how much are they moving" is doomed, however
the terms are weighted.

FACT 2: SUSTAINED QUIET IS ALMOST PERFECTLY SPECIFIC AND USELESS FOR TIMING.
A window in which both players stay quiet for 1.5 s covers 0.0-1.2% of live
play and 7-21% of dead time -- so it is nearly proof that the point is over.
But the FIRST such window after a labelled end arrives a median of +78 SECONDS
later, because players do not stand still after a point: they walk, bounce a
ball, and reposition.  Quiet marks changeovers, not point ends.  It can confirm
that a point ended; it cannot say when.

What is left is the character of the motion, and that is exactly what the
walking classifier already answers.

The construction
----------------
The point ends when the near player stops playing and starts TRAVELLING --
walking to the ball, to position, or to the towel.  `walking/predict.py` is a
HistGradientBoosting model over 373 gait-window features, trained on hand
labels from an indoor hard clip and an outdoor clay clip, and validated
LEAVE-ONE-CLIP-OUT at frame F1 0.82-0.84 across surface, camera and players.
It is the only learned component in anya2 and the only one that has ever been
shown to transfer across surfaces, which is precisely the property the ball
trace lacks.

Timing comes from the ONSET of a walking interval; corroborators only score it.
Measured on Data/21, walk onsets sit at a median +0.6 s from the labelled end
(p25 -0.9, p75 +6.4) and every one of the 12 labelled ends has one.
```

> code: `QUIET_BH_S = 0.45`

```text
── activity, for the quiet corroborator ─────────────────────────────────
Scale-free and homography-free: box travel and limb speed are both divided by
box height, giving body heights per second.  Court speed is deliberately NOT
used -- a far player's ground point carries metres of projection error (see
tracks.FAR_BACK_M), so any court-metre speed for them is noise, and a signal
that means different things on the two sides of the net cannot be combined.
```

> code: `QUIET_BH_S = 0.45`

```text
activity below this counts as quiet
```

> code: `QUIET_WIN_S = 1.5`

```text
...held this long to count as a quiet window
```

> code: `WALK_MIN_S = 1.0`

```text
── walking onsets ───────────────────────────────────────────────────────
```

> code: `WALK_MIN_S = 1.0`

```text
ignore walking intervals shorter than this
```

> code: `REFRACT_S = 6.0`

```text
one end per point; ends are seconds apart at worst
```

> code: `def player_activity(z, fps) -> np.ndarray:`

```text
Per-slot activity in body heights/second. [4, N], NaN where untracked.
```

> code: `def quiet_mask(act: np.ndarray, fps: float) -> np.ndarray:`

```text
True where EVERY tracked player has been quiet for QUIET_WIN_S.

An untracked player contributes nothing rather than counting as quiet:
absence of a player is not evidence that the point ended -- the same rule
near_end applies to the energy bar, and for the same reason.
```

> code: `UNION_NAMES = ("settle", "turn_away", "stance_drop", "idle_hands")`

```text
── the non-rally union ──────────────────────────────────────────────────
The point end is the onset of the first sustained NON-RALLY state.  Walking
alone is not that state: measured over all 216 labelled ends, a walk onset
follows 212 of them -- essentially every point end leads to the near player
travelling -- but at a median of +5.1 s and a p75 of +15.9, because the walk
is a CONSEQUENCE of the end rather than the end itself.  Between the last ball
and the first step the player stands, turns, or celebrates.

`near_end`'s four pose signals cover exactly that gap, and the union of all
five is much better than any of them:

    signal          p25     median   p75      (clip 58, 81 ends)
    walk           +2.1     +6.8    +24.8
    settle         +5.6    +10.7    +19.1
    turn_away      +5.3    +14.8    +36.9
    stance_drop   +26.2    +68.9   +118.1
    UNION          +0.3     +3.0     +6.7

Every corroborator is individually WORSE than walking, some grossly so, and
the union still beats walking by a factor of four at the p75.  That is not a
paradox: they are worse on average but they fire EARLIER on the points where
walking is late, which is the only thing a max() is asking of them.  It is the
same "individually noisy, collectively decisive" construction near_end was
written for, and the reason the combination is a max and not a weighted sum.
```

> code: `UNION_THR = 0.40`

```text
Swept over all 216 labelled ends.  Both parameters want to be LOW: at 0.40 /
0.6 s every end has an onset, median +0.5 s and p75 +3.7 s, and raising either
only pushes the tail out (0.70 / 1.5 s gives median +3.5, p75 +14.0).  A
lower bar finds the moment play stopped; a higher one waits for proof.
```

> code: `def end_signal(video, tracks_npz=None) -> Dict[str, np.ndarray]:`

```text
The non-rally union and its parts, off the cached pose passes.
```

> code: `LIVE_SMOOTH_S = 4.0`

```text
── the live score ───────────────────────────────────────────────────────
The union above is a DEAD-time indicator and, taken alone, a weak one: at the
frame level it separates live from dead at AUC 60%, and its parts are worse
still (settle 48% -- below chance, turn_away 53%, stance_drop 54%, walk 59%).
Those numbers are not a failure of the signals; they are a statement that the
question they were built for is not this one.  The walking classifier answers
"is this person travelling" at F1 0.82, and near_end's four answer "does this
posture look between-points" -- neither is "is the ball in play".

What does carry live/dead is PLAYER ACTIVITY INTEGRATED OVER SECONDS.  Per
sample it is useless (AUC 33%, and inverted against the three seconds after an
end, because that window is full of walking).  Smoothed over 8 s it separates
at 79.5% for the near player and 75.1% for the far one -- and the far number is
only available at all because anya2 tracks the far player, which no previous
point-end work here could do.

The two combine multiplicatively, and that is the whole construction:

    near activity, 8 s               79.5%
    far activity, 8 s                75.1%
    max(near, far), 8 s              78.7%
    max(near, far) - union, 8 s      82.6%
    max(near, far) * (1 - union)     86.7%     <-- this

A product, not a sum, because the union's job is to VETO activity rather than
to be traded off against it: a player walking to the ball is active and is
emphatically not playing, and only a multiplicative term can say so.  It is
the same arbitration shape the near serve detector's swing term has, for the
same reason.
```

> code: `LIVE_SMOOTH_S = 4.0`

```text
8 s separates live from dead slightly better but
blurs the EDGE, and the edge is what is being timed:
at 8 s the best F1 is 40.3% against 42.8% at 4 s.
```

> code: `LIVE_SCALE_PCT = 90`

```text
per-clip normaliser.  Activity is in body heights per
second, so its absolute level depends on how large
the players are in frame -- a fixed threshold would
mean a different thing on every camera.  The clip's
own 90th percentile is what makes the hysteresis
levels below portable.
```

> code: `LIVE_HI = 0.50`

```text
enter "live" above this fraction of that scale
```

> code: `LIVE_LO = 0.35`

```text
...and leave it below this
```

> code: `LIVE_MIN_S = 2.0`

```text
ignore live runs shorter than this
```

> code: `def live_score(parts: Dict[str, np.ndarray], video, tracks_npz=None,`

```text
Normalised [0, ~1] score for "a point is in play".
```

> code: `def detect_ends(parts: Dict[str, np.ndarray], live: np.ndarray,`

```text
A point end is the FALLING EDGE of the live score.

Not the onset of a dead state -- the falling edge.  Onsets of "looks dead"
fire all over dead time, because dead time is noisy and flickers: scored
that way the detector emitted 504 candidates for 216 labelled ends, and no
local feature separated the good ones (run duration AUC 53%, mean depth 64%,
quiet-window overlap 1%).  What DOES separate them is that a real end is
PRECEDED BY PLAY, which is exactly what a falling edge encodes and what an
onset does not.
```

> code: `"p": round(float(np.clip(1.0 - (after.mean() if after.size else 1.0), 0, 1)), 4),`

```text
Confidence is how far the live score falls and stays fallen: a
real end drops to the floor, a lull between shots does not.
```


---

## near_end.py

### Module docstring

```text
near_end.py
===========
Four near-player point-end signals, alongside the walking classifier that
already ships.

The walking classifier answers one question — "is this person going somewhere"
— and it is the only near-player evidence the energy policy reads.  It misses
the whole family of end-of-point behaviour that happens WITHOUT translation: the
player who wins the point standing still, turns to the back fence, drops the
racket hand, reaches into a pocket for the second ball, or stands with hands on
hips getting their breath back.  None of that moves the feet, so `walk_prob`
stays at zero through all of it.

This module computes four such signals, each per pose sample, each in [0, 1]:

    settle        court speed and limb energy both collapse and STAY collapsed.
                  Rally motion is bursty and never quiet for long; the settle
                  after the last ball is.

    turn_away     the shoulder line rotates so the player faces the CAMERA
                  rather than the net.  The near player plays with their back to
                  this camera, so turning toward it is turning away from the
                  point.

    stance_drop   loss of the ready stance: the wrists fall to or below the hip
                  line, the two-handed grip opens, and the knees straighten out
                  of the athletic crouch.  A player who has stopped expecting a
                  ball stands up.

    idle_hands    the between-point rituals, as alternatives under a max():
                  hand to the pocket for a second ball, hand to face / cap /
                  hair, or both hands on the hips with the elbows flared.

The design rule behind all four is the one the user's framing states outright:
INDIVIDUALLY NOISY, COLLECTIVELY NEAR-CERTAIN.  Each of these fires during live
tennis too — a player is briefly still between shots, turns their back chasing a
lob, straightens up after a short ball, wipes their face during a long point.
So none of them is a rule here.  Each is a graded score, and the arbitration
between them is `rally_reel/energy.py`'s: they enter the bar as weighted terms
on the ball-silence drain, which means that WITH THE BALL VISIBLY IN PLAY THEY
DRAIN NOTHING, however confident they are.  That is the same construction (and
the same reason) as `energy_walk_boost` — see that config comment for the
mid-rally-walking failure an additive term reintroduces.

Two things the user's list asked for are deliberately NOT here:

  * BALL BOUNCING (the player bouncing a ball before serving) needs ball
    evidence, not pose, and the energy bar already reads a ball stream —
    `ball_trace` in-court intervals.  A bounce is in-court ball motion, so it
    would read as rally activity and CHARGE the bar.  Separating a serve bounce
    from a rally requires the ball's position relative to the near player, which
    is a change to `ball_trace`, not a fifth pose signal.

  * SERVE-PREP RITUAL AS RETROACTIVE CONFIRMATION ("the next serve proves the
    last point ended") is already structural in the pipeline and does not want
    to be a signal: `points.find_point_end` carries `next_serve_guard_s`, and
    the serve detectors are what produce the next start.  Adding a backwards
    confirmation here would be a second, weaker copy of that guard.

Everything is MODEL-FREE and measured in body-height units or court metres —
no classifier, no weights fit off this corpus.  That is deliberate: stage 2 of
the dead-time cutter has to port to Dart on-device, and a trained sklearn model
would not.  The thresholds below are anatomical constants, not tuned
parameters; the tuning happens once, downstream, on the four weights in
ReelConfig.

Input is the near-player pose the walking classifier already selected
(`walking.select_near`, cached as `<stem>_end_walk_pose.npz`), so this costs one
pass over an array and no perception at all.

Run standalone to inspect the signals on a clip:
    python -m pipeline.near_end /Volumes/Anya/Data/21/snippet.mp4
    python -m pipeline.near_end /Volumes/Anya/Data/21/snippet.mp4 --profile
```

> code: `NOSE = 0`

```text
COCO indices, mirroring walking/features.py.  The geometry helpers are shared
with that module (see `_kin`) but the indices are named here so that importing
SIGNAL_NAMES — which `rally_reel/energy.py` does, at module scope — costs
nothing.  The shared helpers arrive lazily instead: `walking` is a sibling
top-level package that is not on the path until the repo root is, and paying
for that at import time would put a second package's import graph behind
every `from .energy import ...` in the reel.
```

> code: `MAX_GAP_S = 0.5`

```text
matches walking.features: fill single-frame occlusions
only.  A longer hole stays NaN and surfaces as `valid`
False, which the energy bar HOLDS on rather than
draining — absence of a player is not evidence that the
point ended.
```

> code: `SETTLE_SPEED_MPS = 0.55`

```text
── settle ───────────────────────────────────────────────────────────────
A rally is bursty: even the quiet beat while the opponent swings carries a
split-step.  These are the levels below which the near player is doing
nothing, not merely between things.
```

> code: `SETTLE_SPEED_MPS = 0.55`

```text
court speed that scores 0; a walk sits at 1.0-1.5
and walking.features' own "slow" band is < 0.4
```

> code: `SETTLE_LIMB_BH_S = 0.75`

```text
wrist/ankle speed RELATIVE TO THE HIP, in body
heights per second.  Relative, so a player being
carried across court by a run does not read as
limb activity, and a racket take-back while
standing still does.
```

> code: `SETTLE_WIN_S = 1.2`

```text
the sustain window.  Shorter than this and every
between-shot pause scores; much longer and the
signal arrives after the reel has already cut.
```

> code: `TURN_MIN_SHO_BH = 0.055`

```text
── turn_away ────────────────────────────────────────────────────────────
The camera sits behind the near baseline, so the near player faces AWAY from
it while the point is live.  Facing is read from the image-x order of the two
shoulders: with the back to the camera the anatomical right shoulder falls to
the image right of the left one, and the sign flips when they turn round.
```

> code: `TURN_MIN_SHO_BH = 0.055`

```text
shoulder separation below this is a player seen
edge-on; the facing sign is then pure noise and
the sample contributes nothing
```

> code: `TURN_FULL_BH = 0.14`

```text
separation at which the facing read is at full
confidence
```

> code: `TURN_WIN_S = 0.8`

```text
a rally turn (chasing a lob) is a fast pivot back;
an end-of-point turn is held
```

> code: `READY_WRIST_BH = 0.06`

```text
── stance_drop ──────────────────────────────────────────────────────────
Ready: wrists carried at or above the hip line, hands together on the racket,
knees bent.  Dropped: hands by the sides, legs straight.
```

> code: `READY_WRIST_BH = 0.06`

```text
highest wrist ABOVE the hip line, body heights.
Above this is a ready carry, below DROP is a hand
hanging by the side.
```

> code: `READY_KNEE_DEG = 152.0`

```text
mean knee interior angle: an athletic ready
position, against ~172 for standing upright
```

> code: `POCKET_R_BH = 0.13`

```text
── idle_hands ───────────────────────────────────────────────────────────
Radii are body-height fractions of the distance from the wrist to the target
landmark, so they hold at any depth in frame.
```

> code: `POCKET_R_BH = 0.13`

```text
wrist within this of the same-side hip, and below
the hip line: reaching for the second ball
```

> code: `HEAD_R_BH = 0.22`

```text
wrist within this of the head centre: cap, hair,
wiping the face
```

> code: `HIP_R_BH = 0.11`

```text
both wrists this close to their hips …
```

> code: `FLARE_BH = 0.10`

```text
… with both elbows this far LATERAL of the
shoulders is the hands-on-hips akimbo shape, which
the flare is what distinguishes from arms hanging
```

> code: `IDLE_WIN_S = 0.6`

```text
rituals are held for a beat; a single frame of a
wrist passing a hip is not one
```

> code: `def _kin():`

```text
The pose helpers `walking.features` already defines, imported lazily.

Shared rather than re-derived on purpose: gap interpolation, the
confidence-gated keypoint read, the midpoint fallback and the NaN-tolerant
moving mean all have to behave here EXACTLY as they do under the walking
classifier, or the two near-player signals disagree about what a tracked
frame even is.  A second copy would drift on the first bug fix.
```

> code: `def _ramp(x, lo, hi):`

```text
Soft 0->1 ramp from lo to hi (either order); NaN in, NaN out.
```

> code: `def _dist_bh(a, b, h):`

```text
Distance between two [N,2] keypoint tracks, in body heights.
```

> code: `def _smooth(x, fps, win_s, movmean):`

```text
Centred moving mean over `win_s`, NaN-tolerant.

Centred rather than causal because this is an offline cutter: the reel is
built after the clip exists, and a causal filter would lag every signal by
half a window into the dead time it is trying to time.  A future on-device
port that must run live gets a causal variant here and re-measures — the
lag is a real change to the timing, not an implementation detail.
```

> code: `def _nan_to_zero(x):`

```text
Score arrays leave here as evidence, and missing evidence is 0.
```

> code: `def near_signals(kp, bbox, fps: float,`

```text
The four signals for one near-player pose track.

`kp` [N, 17, 3] and `bbox` [N, 4] are `walking.select_near`'s output — the
near player only, NaN where they were not tracked.  `speed` is the court
speed in m/s that `walking.features.frame_signals` already computed off the
same track; passing it keeps this module from re-deriving a homography and
guarantees the settle signal and the walking classifier disagree about
nothing.  Without it, settle falls back to limb energy alone.

Returns each signal as [N] in [0, 1], plus `valid` [N] bool.
```

> code: `def rel_speed(pt):`

```text
── settle ───────────────────────────────────────────────────────────
```

> code: `quiet = np.nanmean(np.stack([quiet_foot, quiet_limb], axis=1), axis=1)`

```text
MEAN, not min — measured, against the opposite prior.  The argument
for min() is that a settle ought to be BOTH feet and hands quiet, and
it scores worse on every clip: post/live separation over
Data/21,22,23,24,43 is

    min   1.27  1.38  1.82  0.94  1.56     (negative on 24)
    mean  2.16  1.40  1.51  1.24  1.63     (positive on all five)

because at 15 Hz off a 540p proxy the two channels fail
independently: a lost ankle keypoint zeroes the foot term while the
hands are perfectly still, and min() then throws away the half of the
evidence that survived.  Averaging degrades toward whichever channel
is readable, which is the behaviour this input actually needs.
```

> code: `sho_dx = (r_sho[:, 0] - l_sho[:, 0]) / h`

```text
── turn_away ────────────────────────────────────────────────────────
Signed shoulder order.  Positive = anatomical right shoulder to the image
right = back to the camera = facing the net = playing.  Negative = turned
round.  Normalised by body height so it is a shape, not a pixel count.
```

> code: `conf = _ramp(sep, TURN_MIN_SHO_BH, TURN_FULL_BH)`

```text
Confidence in the SIGN, which is the only part being read.  Edge-on, the
two shoulders project on top of each other and the order is a coin flip.
```

> code: `turn_raw = np.clip(conf * facing_cam, 0.0, 1.0)`

```text
SHOULDER ORDER ALONE.  Nose visibility looks like free corroboration — the
model finds a face far more readily from the front than from behind — and
it is the single most damaging term measured in this module.  Blending it
in at weight w costs separation monotonically over Data/21,22,23,24,43:

    w = 0.00   11.83  4.62  1.48  3.49  88.67
    w = 0.25    4.56  2.61  1.49  2.57   4.73
    w = 0.50    3.11  2.00  1.50  2.17   2.79

The nose is found on a large share of frames from BEHIND as well (the
model hallucinates a plausible face onto the back of a head at this
resolution), so the term is close to a constant, and a constant added to a
discriminative signal can only dilute it.  Left out rather than
down-weighted: there is no weight at which it pays.
```

> code: `wri_above_hip = (hip[:, 1] - np.fmin(l_wri[:, 1], r_wri[:, 1])) / h`

```text
── stance_drop ──────────────────────────────────────────────────────
Image y grows downward, so hip_y - wrist_y is positive above the hip.
```

> code: `stance_drop = _smooth(np.fmin(hands_low, legs_straight),`

```text
MIN, not mean — the opposite of the choice `settle` makes ten lines up,
and for the opposite reason.  Straight legs on their own are not evidence
of anything (a player stands upright constantly while the ball is at the
far end), and it shows: as a lone cue the knee term is negative on one
clip and flat on the rest.  Conjunction is what makes it mean something.
Over Data/21,22,23,24,43:

    knees only  1.56  1.16  0.93  1.21  1.38
    hands only  1.96  1.34  2.42  6.44  2.36
    mean        1.65  1.20  1.27  1.69  1.59
    min         2.80  1.53  2.24  7.08  3.32

min() beats hands-alone on four clips of five, so the knee term is real —
it just has to VETO rather than vote.  Averaging it in makes the pair
worse than the better half of it, which is the signature of a weak cue
being given independent authority it has not earned.
```

> code: `l_hip_d, r_hip_d = _dist_bh(l_wri, l_hip, h), _dist_bh(r_wri, r_hip, h)`

```text
── idle_hands ───────────────────────────────────────────────────────
```

> code: `l_below = (l_wri[:, 1] - l_hip[:, 1]) / h`

```text
positive = wrist below hip
```

> code: `pocket = np.fmax(`

```text
(a) hand to pocket: at the hip AND dropped to it, which is what separates
a reach into a pocket from a hand resting on the hip bone.
```

> code: `l_head_d, r_head_d = _dist_bh(l_wri, head, h), _dist_bh(r_wri, head, h)`

```text
(b) hand to face / cap / hair: wrist at the head and raised above the
shoulder line, so a racket held up in front of the chest does not count.
```

> code: `l_flare = _ramp((l_sho[:, 0] - l_elb[:, 0]) / h, 0.0, FLARE_BH)`

```text
(c) hands on hips, elbows flared.  Both sides, or it is one hand on a hip
— which is (a) — and the flare is measured outward from each shoulder, so
arms hanging straight down score zero however close the wrists sit.
```

> code: `l_flare = np.fmax(l_flare, _ramp((l_elb[:, 0] - l_sho[:, 0]) / h, 0.0, FLARE_BH))`

```text
The shoulder order flips with the player, so take the flare either way
round rather than trusting the sign of a single frame's left/right call.
```

> code: `idle_hands = _smooth(pocket, fps, IDLE_WIN_S, _movmean)`

```text
POCKET ONLY.  The three rituals were built as alternatives under a max(),
on the reasoning that a player does one of them and summing would reward
only the rare player who does two at once.  The max() is right; the
membership was not.  Measured separately over Data/21,22,23,24,43
(post/live over tracked samples):

    pocket    3.09  3.16  2.29  10.05   7.71     every clip, strongly
    akimbo    1.76  0.77  2.83   5.74   1.00     negative on 22, flat on 43
    to_head   0.26  0.00  0.07   0.00   0.00     ANTI-correlated everywhere

to_head is not weak, it is backwards, and no radius fixes it (0.22 and
0.32 body heights score the same): at 540p the wrist that comes near the
head is the racket arm at the top of a swing or a serve toss, which is
live play, and a genuine cap adjust between points is a handful of frames
per clip.  akimbo is closer to real but its flare term is unreachable at
this resolution — loosening FLARE_BH to 0.05 raises its level 4x and moves
separation the wrong way on two clips, which is a threshold buying noise.

Both are still COMPUTED and returned under `cue_` keys.  They cost two
array ops, the measurement above is five clips of one camera height, and a
deleted cue cannot be re-measured on better footage.  They just do not
enter the signal the energy bar reads.
```

> code: `"cue_pocket": pocket, "cue_to_head": to_head, "cue_akimbo": akimbo}`

```text
Measured and excluded from idle_hands; kept for re-measurement.
`SIGNAL_NAMES` is what the energy bar reads, so nothing downstream
picks these up by accident.
```

> code: `for k in out:`

```text
Zeroed wherever there is no player at all: a signal computed across a hole
is a guess, and `valid` is what lets the energy bar hold instead.
```

> code: `def signals_for_video(video: str, pose_npz: Optional[str] = None,`

```text
`near_signals` off the cached near-player pose for `video`.

`sig` is `walking.features.frame_signals`' output when the caller already
has it (the reel and the tuner both do, inside `walk_result`); otherwise the
court speed is recomputed here from the same npz.
```

> code: `def _profile(video: str, sigs: Dict[str, np.ndarray]) -> None:`

```text
── CLI ──────────────────────────────────────────────────────────────────
```

> code: `def _profile(video: str, sigs: Dict[str, np.ndarray]) -> None:`

```text
Mean of each signal in bands around every labelled rally end.

The question this answers is the only one that matters before any weight is
fitted: does the signal RISE after the point ends, and by how much against
its own level during play.  A signal that reads the same either side of the
boundary cannot help the bar whatever weight it is given.

Means are taken over TRACKED SAMPLES ONLY, with the coverage printed
alongside.  Pooling the untracked ones — which the energy bar never sees,
because `near_held` makes it hold rather than drain — would deflate every
band by however much of it the player spent off camera, and the dead bands
are exactly where they leave.  On Data/23 that is 58% of the clip, so the
difference between the two readings is not a rounding detail.
```


---

## orchestrator.py

### Module docstring

```text
orchestrator.py
===============
Agent 4.  Takes the three detectors' event streams and produces the reel.

Its job is NOT to detect anything.  It is to impose the structure that tennis
has and the detectors cannot see, and to make the result watchable.

THE ONE MEASUREMENT THAT DECIDES THE ARCHITECTURE
-------------------------------------------------
Over the 13-clip corpus the three detectors do not have comparable recall:

    near serve   90.7%      far serve  82.2%      point end  49.6%

Serves are found; ends are found half the time.  So THE REEL IS BUILT FROM
STARTS, and ends only trim.  A point whose end was never detected still becomes
a segment -- it runs to just before the next serve instead.  Building from ends,
or requiring a start/end pair, would silently drop half the points, and the
brief is that every point is in the reel.

That asymmetry is also why a missed end is cheap here and a missed START is not:
a missed end costs some dead time at the tail of one segment, while a missed
start loses the whole point.

WHAT TENNIS KNOWS THAT THE DETECTORS DO NOT
-------------------------------------------
  SERVICE RUNS.  One player serves a whole game, so the side sequence looks
      like NNNNN FFFFF NNNNN, never NNFNN.  An isolated side flip is a
      mislabel, and relabelling it is free -- it changes which detector gets
      credit, not whether a segment exists.

  DEUCE/AD ALTERNATION.  Within a game the server alternates courts every
      point, always.  In the video it is far weaker than that: re-measured over
      all 14 clip-sides with six or more labelled serves, the server's court-x
      flips sign across consecutive serves a MEAN OF 63% of the time, spanning
      20% (clip 43 near) to 91% (clip 58 near).  An earlier note here quoted the
      91% as representative; it is the best case, not the corpus.  Median-
      filtering the court track does not improve it at all (63% either way),
      because the per-serve reading is already a median over three seconds.  At
      63% this is barely above chance, so it FLAGS a suspected missing point and
      nothing more -- it must never delete a detection, and the flag should be
      read as a hint rather than a finding.

  A POINT IS NOT LIVE TWICE.  A serve detected while a point is already in
      progress is spurious, and this is where most of the detectors' false
      positives live: 23 of 24 near and 79% of far false positives fall inside
      a live point.  Both detectors declare `windows="between_points"` for
      exactly this reason and neither can enforce it alone.  The orchestrator
      can, because it knows where the previous point ended.

  RHYTHM.  Points are seconds long and separated by tens of seconds; a serve
      3 s after the last one is not a new point.

SMOOTHNESS IS A FIRST-CLASS OUTPUT, NOT A SIDE EFFECT
-----------------------------------------------------
The brief puts viewing experience above precision, and that changes two
choices a pure accuracy metric would make differently:

  * WHEN IN DOUBT, KEEP FOOTAGE.  An extra second of a player walking is
    barely noticeable; a cut that lands mid-rally is jarring and loses the
    point.  Pre-roll and post-roll are generous, and a synthesized end runs
    long rather than short.

  * A CUT COSTS SOMETHING.  Two segments separated by a couple of seconds of
    dead time are worse to watch than one continuous segment that includes
    those seconds -- the cut draws attention to itself for no gain.  So
    segments closer than `merge_gap_s` are joined, and very short segments are
    dropped or absorbed rather than flashed on screen.
```

> code: `merge_window_s: float = 2.5`

```text
── merging the two serve streams ────────────────────────────────────
```

> code: `merge_window_s: float = 2.5`

```text
both detectors firing this close are one
serve.  Wider than it looks because the
two anchor on different events (hands
together vs trophy onset).
```

> code: `min_point_gap_s: float = 8.0`

```text
two starts closer than this are one point.
A real inter-point gap is 15-25 s (the
ATP shot clock is 25); 8 is deliberately
permissive so a genuinely quick point is
never merged away.
```

> code: `min_service_run: int = 3`

```text
── service structure ────────────────────────────────────────────────
```

> code: `min_service_run: int = 3`

```text
a game is 4+ points, but detections are
missed, so requiring 4 would force real
games to merge.  3 is the smallest run
that still rejects an isolated flip.
```

> code: `min_point_s: float = 2.5`

```text
Tested and rejected.  The idea was that on a near service game a lone far
detection is a phantom rather than a mislabel -- the far player is
RETURNING, and a return is a serve motion -- so it should be dropped, not
relabelled.  Measured over the corpus it changes almost nothing (whole
points 207 -> 206, reel 61.6% -> 60.9%) and it does not fix the case that
motivated it: on Data/21, where 9 of 21 merged starts are spurious far
detections, the reel is identical at 3 segments either way, because the
phantoms cluster into runs of 2-3 that any min_service_run of 3-5 accepts
as a legitimate game.  Left off, so the rule "structure relabels, it never
drops a point" holds without exception.
```

> code: `min_point_s: float = 2.5`

```text
── pairing starts with ends ─────────────────────────────────────────
```

> code: `min_point_s: float = 2.5`

```text
an end this soon after the serve is the
serve motion itself, not the point
```

> code: `max_point_s: float = 40.0`

```text
...and beyond this the "end" is a missed
end and a later, unrelated quiet
```

> code: `default_point_s: float = 9.0`

```text
fallback when a clip has no detected ends
at all to estimate from
```

> code: `est_duration_pct: float = 85.0`

```text
when an end is missing, assume a point
this percentile of the clip's OWN
detected point durations.  Per clip
because rally length is a property of the
players, and high because overrunning
costs dead time while underrunning cuts
live tennis.
```

> code: `next_start_guard_s: float = 4.0`

```text
never run a segment closer than this to
the next serve
```

> code: `pre_roll_s: float = 1.0`

```text
── smoothness ───────────────────────────────────────────────────────
Roll is the single biggest lever on how much tennis survives, and it is
far more effective than any end-estimation tuning.  Measured over all 236
labelled points -- whole points / live retained / reel as a share of the
source:

    pre 2.5 post 2.0    179    93.1%    54.9%
    pre 2.5 post 3.0    196    94.6%    57.3%
    pre 3.0 post 3.5    202    94.9%    59.5%
    pre 3.5 post 4.0    207    95.3%    61.6%     <-- here
    pre 3.5 post 5.0    212    95.8%    63.7%

For comparison, moving the end-duration percentile from 85 to 97 buys 2
whole points.  The ends are close to right; they were simply being cut a
second or two early, and roll is what fixes that.

3.5/4.0 is chosen for the brief's stated priority -- viewing experience
first, dead time second.  Lower both to tighten the reel; the table above
is the exchange rate.
Set to 1.0/1.0 at the user's direction, tightening the reel.  The table
above is what that costs against the corpus; re-measured at 1.0/1.0 below.
```

> code: `merge_gap_s: float = 6.0`

```text
segments closer than this are joined
rather than cut apart
```

> code: `min_segment_s: float = 4.0`

```text
anything shorter is a flash, not a point
```

> code: `near_threshold: float = 0.0`

```text
── thresholds on the incoming streams ───────────────────────────────
```

> code: `near_threshold: float = 0.0`

```text
the detectors already threshold; these
```

> code: `far_threshold: float = 0.0`

```text
exist so a caller can be stricter
```

> code: `use_near: bool = True`

```text
Which streams to read AT ALL.  A disabled agent must not contribute, and
that cannot be left to simply not regenerating its file: the events are
cached beside the video, so a stale run would silently keep feeding the
reel after the agent was switched off.
```

> code: `live_gate_far: float = 0.8`

```text
── in-rally suppression ─────────────────────────────────────────────
A serve struck while a point is already live is spurious, and agent 3
already computes the signal that says so.  Measured over the corpus, the
median live score in the 3 s BEFORE a detection separates true serves from
false ones far better on the far side than the near:

    far   live > 0.8 drops 26% of false positives for 4% of true serves
    near  live > 0.8 drops 17% of false positives for 10% of true serves

That asymmetry is the taxonomy showing through: 79% of far false
positives are the RETURNER mid-rally, so they sit on live play, while
near false positives are the server's own repeated motions in dead time
and look exactly like a serve from here.  So the gate is applied to the
far stream only -- on the near stream it would cost more than it buys.
```

> code: `live_gate_near: float = 0.0`

```text
off: measured AUC 55%, not worth it
```

> code: `recover_live_thr: float = 0.75`

```text
── recovering points with no serve detection ────────────────────────
```

> code: `recover_live_thr: float = 0.75`

```text
live score above which play is clearly on
```

> code: `recover_min_s: float = 3.0`

```text
...held this long to count as a point
```

> code: `rule1_enabled: bool = True`

```text
── rule 1: toss evidence adjusts a far serve's confidence ───────────
Two independent readings of the same event, neither of which the far
detector folds into its own score:

The evidence is `far_serve.toss_score` -- the tossing arm's own motion,
AUC 75% against far false positives and correlated only +0.04 with the
serve score, so it is genuinely new evidence.

A BALL toss detector was built and removed.  Re-aimed at the tossing wrist
with native-resolution SAHI tiling it detected the ball well (8 frames per
serve against 0-0.5 for a head-centred ROI), and its arc separated true
from false serves at AUC 75% -- but it correlated +0.49 with the pose toss
and the best blend of the two was no better than pose alone.  Two readings
of the same event, and the arm is far easier to see than a 3 px ball.  It
cost tiled inference over ~18 native frames per candidate for nothing.

This ADJUSTS confidence rather than filtering.  A hard gate on either one
costs real serves (dropping pose-toss below 0.40 removes 54% of false
positives but 18% of true ones), and the brief is precision WITHOUT
giving up recall -- so a weak toss reading demotes a candidate and lets
the rest of the pipeline decide, instead of deleting it outright.
```

> code: `toss_pose_lo: float = 0.30`

```text
pose-toss at or below this is unsupportive
```

> code: `toss_pose_hi: float = 0.60`

```text
...and at or above it, corroborating
```

> code: `rule1_max_boost: float = 0.15`

```text
most a strong toss may add
```

> code: `rule1_max_penalty: float = 0.35`

```text
...and most a missing one may subtract.
Asymmetric on purpose: seeing a toss is
strong evidence FOR a serve, while not
seeing one is weak evidence against --
the ball is a few pixels at that range
and the arm may be occluded.
```

> code: `rule2_enabled: bool = True`

```text
── rule 2: a far serve among near serves is suspect ─────────────────
Measured over the corpus: requiring 3 of a far candidate's 4 nearest
neighbouring starts to be NEAR-side flags 18% of the remaining far false
positives and ZERO true far serves.  At 2 of 4 it reaches 27% of false
positives but starts costing 7% of true ones.

So the 3-of-4 setting is free, and this too adjusts rather than deletes:
one player does serve a whole game, but detections are missed, and a real
far game whose first serves were not detected would look exactly like
this.  Demoting is recoverable; deleting is not.
```

> code: `rule2_neighbours: int = 4`

```text
how many surrounding starts to look at
```

> code: `rule2_min_near: int = 3`

```text
...of which this many must be near-side
```

> code: `toss_pose: Optional[float] = None`

```text
far_serve's pose toss score
```

> code: `conf_adj: float = 0.0`

```text
rules 1 and 2 accumulate here
```

> code: `def adjusted_p(self) -> float:`

```text
Confidence after the orchestrator's rules, clipped to [0, 1].
```

> code: `end_source: str`

```text
"detected" | "next-serve" | "default"
```

> code: `def merge_starts(near: Sequence[Event], far: Sequence[Event],`

```text
── 1. one timeline of serves ────────────────────────────────────────────
```

> code: `def merge_starts(near: Sequence[Event], far: Sequence[Event],`

```text
Union the two serve streams, deduped.

When both sides fire for the same serve the EARLIER timestamp wins and the
side becomes "both".  Earlier because the two detectors anchor on different
moments of the same action -- the near one on the hands leaving the grip,
the far one on the trophy -- so the earlier is closer to where the point
actually began, and erring early only costs pre-roll.
```

> code: `def drop_rapid_repeats(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:`

```text
Collapse starts closer together than a point can possibly be.

Keeps the FIRST, not the strongest.  A cluster of serve detections around
one point is usually the real serve plus an echo of the same motion, and the
first is the one that opens the point -- keeping a later, higher-scoring
echo would cut into the serve itself.
```

> code: `def apply_toss_rule(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:`

```text
RULE 1 -- adjust a far serve's confidence by its toss evidence.

Pose evidence only -- `far_serve`'s toss score.  The result shifts confidence
within [-rule1_max_penalty, +rule1_max_boost] and never removes a start; see
the config for why the bound is asymmetric, and for why the ball half of
this was measured and removed.
```

> code: `u = 2.0 * float(np.clip((t - cfg.toss_pose_lo)`

```text
-1 (no toss seen) .. +1 (clearly a toss)
```

> code: `def apply_neighbour_rule(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:`

```text
RULE 2 -- a far serve surrounded by near serves is suspect.

Uses the sides AS DETECTED, before the service-run DP relabels anything:
the DP's own output would make this circular, since it has already decided
the run each start belongs to.
```

> code: `def enforce_service_runs(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:`

```text
── 2. service runs ──────────────────────────────────────────────────────
```

> code: `def enforce_service_runs(starts: List[PointStart], cfg: ReelConfig) -> List[PointStart]:`

```text
Relabel sides so every service block runs at least `min_service_run`.

A segmental DP over (side, run-so-far, still-in-the-first-run), exact rather
than greedy: a greedy left-to-right pass commits to an early wrong side and
can never recover.  The first and last runs are exempt from the minimum,
because a clip rarely starts or ends on a game boundary.

This only ever CHANGES A LABEL, never drops a start.  Which side served is
metadata; that a point happened is not, and the brief is that every point is
in the reel.
```

> code: `return 1.0 + ps.p`

```text
a confident disagreement costs more to overrule
```

> code: `def annotate_serve_court(starts: List[PointStart], video: str,`

```text
Record which half of the court each serve was struck from.

Within a game the server alternates deuce/ad EVERY point, without
exception, so consecutive serves in one service run should flip sign.  In
the video they flip a mean of 63% of the time across 14 clip-sides, ranging
20% to 91% -- weak enough that this is a hint and not a finding.

So this is recorded and used to FLAG, never to delete: a pair of
consecutive same-court serves inside one service run suggests a point was
missed between them, which is a recall question, not a reason to throw away
a detection that is probably real.
```

> code: `def suppress_in_rally(starts: List[PointStart], live, fps: float,`

```text
── 3. pair starts with ends ─────────────────────────────────────────────
```

> code: `def suppress_in_rally(starts: List[PointStart], live, fps: float,`

```text
Drop serves struck while a point was already live. See ReelConfig.
```

> code: `continue`

```text
a point was already in progress
```

> code: `def estimate_point_s(segs: Sequence[Segment], cfg: ReelConfig) -> float:`

```text
How long to assume a point ran when its end was never detected.

Taken from THIS clip's detected point durations, because rally length is a
property of the players rather than of tennis, and at a high percentile
because the two errors are not symmetric: overrunning leaves dead time in
the reel, underrunning cuts live tennis out of it.
```

> code: `def pair_ends(starts: Sequence[PointStart], ends: Sequence[Event],`

```text
One segment per start.  Every start becomes a segment, always.

The end is chosen in this order:

  DETECTED   the first end event that falls in the plausible window for
             this point -- later than `min_point_s` (before that it is the
             serve motion itself), earlier than `max_point_s` (after that
             it is a missed end and some later, unrelated quiet), and before
             the next serve.

  ESTIMATED  no end was detected in the window, so the point is assumed to
             have run for `estimate_point_s` -- a high percentile of THIS
             clip's own detected point durations -- bounded by the next
             serve.  Running to the next serve instead would keep every
             inter-point gap on the ~50% of points whose end is missed.

Note what this does NOT do: it never drops a start for want of an end.
Point-end recall is 49.6% against 90.7%/82.2% for the serves, so requiring a
pair would discard half the points.
```

> code: `segs = _pair_once(starts, et_arr, cfg, duration, cfg.default_point_s)`

```text
Two passes: the first only to learn this clip's typical point length from
the ends that WERE detected, the second to use it for the ones that were
not.  One pass cannot do it -- the estimate is derived from the same
pairing it feeds.
```

> code: `est = ps.t + est_s`

```text
An UNDETECTED end must not mean "run to the next serve": with
point-end recall at 49.6% that would keep every inter-point gap on
half the points, which is the opposite of the brief.  Assume a
point of the clip's own typical length instead, bounded by the
next serve.
```

> code: `def flag_missing_points(starts: List[PointStart], segs: List[Segment],`

```text
Note where the deuce/ad alternation says a point is missing.

Advisory only -- it adds a note, changes no timing.  Two consecutive serves
from the same court inside one service run means an odd number of points
happened between them, and the simplest explanation is one we did not
detect.  Surfacing it is how a recall gap becomes visible in the output
instead of silently absent.
```

> code: `continue`

```text
too near the centre mark to call
```

> code: `def smooth(segs: List[Segment], cfg: ReelConfig,`

```text
── 4. smoothing ─────────────────────────────────────────────────────────
```

> code: `def smooth(segs: List[Segment], cfg: ReelConfig,`

```text
Make the sequence watchable: no overlaps, no flashes, no needless cuts.

Three passes, in this order, because each can create work for the next:

  1. CLAMP AND ORDER -- a segment may not start before the clip or overlap
     its neighbour.  Overlap is resolved by splitting the difference, which
     keeps both points rather than truncating one.
  2. JOIN NEAR NEIGHBOURS -- if the dead gap between two segments is under
     `merge_gap_s`, the cut costs more attention than the gap costs time.
     Keep it continuous.
  3. DROP FLASHES -- a segment under `min_segment_s` is a blink.  It is
     dropped only if it holds no detected end; one that does is a real short
     point and is PADDED to the minimum instead.
```

> code: `def build_reel(video: str, cfg: Optional[ReelConfig] = None,`

```text
── 5. the whole thing ───────────────────────────────────────────────────
```

> code: `starts = apply_toss_rule(starts, cfg)`

```text
Rules 1 and 2 run BEFORE the service-run DP: rule 2 reads the sides as
detected, and the DP would otherwise have already overwritten them.
```

> code: `ap.add_argument("video", nargs="+")`

```text
nargs="+": the same GoPro chapters build_reel takes.  Assembly reads
artifacts rather than pixels, so what this needs from pipeline.join is
the joined PATH -- whose stem those artifacts are keyed by -- and the
join itself is already cached from the detection run that wrote them.
```

> code: `def recover_missed(segs: List[Segment], live, fps: float, cfg: ReelConfig,`

```text
── 5b. recovering points nobody detected ────────────────────────────────
```

> code: `def recover_missed(segs: List[Segment], live, fps: float, cfg: ReelConfig,`

```text
Add segments for sustained live play that no serve detector caught.

The brief is that every point is in the reel, and the serve detectors miss
some -- 15 of 236 labelled points had no detection at all before this pass.
Nothing in the serve streams can recover those, but agent 3's LIVE SCORE is
a different measurement entirely: it separates live play from dead time at
AUC 86.7%, and it does not care whether a serve was seen.

So any run of sustained live play that no segment already covers is treated
as a point that was missed, and kept.  It is marked `recovered` and carries
no side, because there is no serve detection to say who served -- the reel
gains the tennis without inventing a fact about it.

This is deliberately the LAST pass: it fills gaps rather than competing with
the detectors, so a point that WAS detected keeps its proper serve-anchored
boundaries and only genuinely unexplained play is added.
```

> code: `def score_reel(res: Dict, clip_dir: str) -> Dict:`

```text
── 6. scoring a reel ────────────────────────────────────────────────────
Serve and end detectors are scored on event timing; a REEL is not. What
matters to a viewer is whether the tennis is all there, how much waiting was
left in, and how often the picture cuts. Those are the numbers below.
```

> code: `def score_reel(res: Dict, clip_dir: str) -> Dict:`

```text
Measure a reel against the labelled rallies.
```


---

## Former README

# anya2 — point boundaries, rebuilt as three independent detectors

Three questions, three detectors, one shared substrate. Nothing here imports
from `pipeline/rally_reel/` or `pipeline/anya_*.py`; those keep shipping
untouched so every number below is A/B-able against them.

| detector | question | status |
|---|---|---|
| `near_serve.py` | did a point start with a near-side serve? | **built** |
| `far_serve.py` | did a point start with a far-side serve? | **built** |
| `point_end.py` | did the point end? | **built** (pose only) |
| `orchestrator.py` | **agent 4** — turn the three streams into a watchable reel | **built** |

### Substrate

| module | what it owns |
|---|---|
| `court.py` | image→court-metres map, and the two player gates |
| `camera.py` | per-frame warp back onto the calibration frame — the map is a function of time |
| `tracks.py` | ≤2 near + ≤2 far player tracks, with a per-frame eligibility flag |
| `balls.py` | ball detection with exclusion zones applied at source |
| `perceive.py` | the two pose ROIs (near 540p whole-frame, far native-res band) |
| `contract.py` | `Event` and `Requirement` — the only things detectors exchange |
| `eval.py` | one event-matching harness, three modes |

The two **player gates** are the user's rules, evaluated on the bounding box's
bottom-centre projected through the ground-plane homography:

- **near vs far** — whichever baseline the box bottom is closer to in y.
- **in bounds** — the box x-centre inside the doubles court plus 3 ft
  (`-2.284 … 10.514` m in the singles-origin court frame).

Doubles needs no re-calibration: the cached corners are the singles corners and
the homography is a full plane map, so the alley is analytic.

Tracking uses a **looser** zone than the gate, on purpose. Players walk off
court to the ball carts; a court gate would break the track and then
mis-reacquire it. So a player off court is still *tracked*, just not *eligible*.

### The map is a function of time

Calibration is four corners clicked once, on one reference frame. That is one
homography, and applying it to every frame is correct exactly as long as the
camera does not move. When the camera *is* bumped, nothing errors: the corners
still load, every projection still returns a number, and the numbers are simply
wrong from that frame on.

They are wrong in the direction that costs most. At far-court depth the 960×540
analysis frame is only ~4–5 px per court metre, so a 20 px jostle moves the far
baseline by four metres — `side()` starts calling far players near, `in_bounds()`
starts rejecting players who are on the court, and the far-serve band gates on a
`court_y` that no longer means what it meant at calibration.

`camera.py` keeps the single clicked calibration and makes the map time-varying
instead:

    court metres  =  H_ref  @  W_t  @  (image point at frame t)

`W_t` registers frame t against the frame the corners were clicked on — ORB plus
a RANSAC homography over the 540p proxy `perceive` already builds, at 5 Hz, so it
costs no extra decode and no model call. Every sample is registered against the
**one** reference, never chained, so nothing accumulates and a static camera
gives identity warps forever. `court.Geometry` composes the two; with no cached
track it *is* `H_ref`, through the same code path rather than a branch beside it.

Two details that are not incidental:

- **Ground-plane inliers are re-fit separately.** A single homography fits the
  whole scene only under pure rotation; a real bump translates a little too, and
  then the fence, the stands and the court surface each want a different one.
  Downstream `W_t` is only ever used on ground points, so the inliers landing
  inside the court polygon are re-fit alone when there are ≥20 of them.
- **The far band is the union over the whole track.** `perceive.far_band` becomes
  a fixed ffmpeg crop, and a crop cannot follow a camera. A far player who leaves
  it is not mis-projected, they are *absent*. So the band is taken around the
  reference band under every sampled warp at once. The cached far pose pass now
  records its crop and re-runs when the band moves.

Measured on a synthetic 30 s clip with a known 21 px pan/tilt/roll at t=15 s
(`--self-test` covers the same math analytically):

| | max court error |
|---|---|
| fixed homography | **2.63 m** |
| tracked | **0.10 m** |

The jostle is reported at frame 450 — the frame it was injected at.

#### What Data/77 actually turned out to be

The synthetic clip validated a code path real footage cannot support, and the
69-minute match said so. Two corrections came out of it.

**A real court is flat paint.** Across 414 samples the global fit had a median
of **1365** inliers and the *on-court* subset a median of **18** — see
`MIN_GROUND_INLIERS` for the numbers and what the original 20 did with them.
The synthetic court had speckle texture inside the quad, so it never exercised
the case where the ground re-fit is unsupportable, which on real footage is
almost always. The floor is now 60 plus an agreement gate.

**Data/77 has no jostle.** Its largest *sustained* level change over 69 minutes
is 0.92 px. What it has is monotonic drift — a settling mount — of the far
baseline:

| | 0–10 min | 30–40 min | 60–70 min |
|---|---|---|---|
| far-baseline offset | 0.2 px | 1.0 px | **2.2 px** |

Two pixels sounds like nothing and is not. The court's whole 23.77 m length
occupies ~117 px in the analysis frame, and the far end is where the perspective
compression is worst, so the same drift reads as **0.02 m of error at the near
corners and 2.4 m at the far ones** by the end of the match. That asymmetry is
the entire story of why a fixed homography fails quietly rather than obviously.

#### Data/78 — the clip with an actual jostle

Data/77 has drift; Data/78 has a bump, and it is what the module was built for.
The camera is dead still for ten minutes, is knocked at 10:53, and is still
again for the next thirty:

| | corner shift |
|---|---|
| 0–10 min | 0.5–1.0 px |
| **10:53** | → **45.2 px** (dx +24.1, dy −9.4 at the far baseline) |
| 12–42 min | 45.8–46.1 px, flat |
| **~43:30** | → **52.1 px**, a second smaller knock |

Sub-pixel stable across fifteen consecutive 2-minute windows, which is the
strongest evidence available that the estimate is real: noise does not hold
46.0 for half an hour. Under the fixed homography that is a **median 18.8 m**
court error, and the far player lands a median **9.15 m** from where they are.

Scored against the clip's own `tags.json` point starts (±3 s), both arms run off
the SAME cached pose passes so the court map is the only variable:

| point recall | fixed | tracked |
|---|---|---|
| before the jostle | 88.2% | **88.2%** |
| after | 65.8% | **89.5%** |
| overall | 72.7% | **89.1%** |

The first row is the control and matters as much as the second: the correction
does nothing when there is nothing to correct, which is what a change of
coordinates that is the identity on a still camera has to look like. Far
slot-frames rise +21% / +45% / +60% across the three thirds, far serves 37 → 76,
and the reel goes 960.7 s → 1195.2 s — the broken run was dropping about four
minutes of real rally.

#### Both halves of the fix, separated

The table above reuses the cached far dets, cut from a band aimed before the
camera moved — post-jostle that band is missing 99 px on the right and 83 px at
the bottom of where the far player now is. So it exercises the corrected
geometry but not the union band. Re-running the far pose pass over the union
band separates the two:

| arm | recall pre | recall post | far events/pt pre | post |
|---|---|---|---|---|
| A — fixed H, old band | 88.2% | **65.8%** | 0.53 | 0.74 |
| B — tracked H, old band | 88.2% | 89.5% | 0.59 | 1.74 |
| C — tracked H, union band | 88.2% | **94.7%** | 0.71 | 1.58 |

Both halves are load-bearing and they fix different things. The geometry
recovers 24 points of recall (A→B): the detections existed, and the court map
was throwing them away. The band recovers 5 more (B→C): those detections did
not exist, because the crop no longer contained the player — persons/sample
0.99 → 1.11 and empty frames 22.7% → 14.2%. Nothing downstream can recover a
frame the crop never saw, which is why the band is sized from the track rather
than from the calibration.

**The unresolved part.** Arm C's far-serve rate after the jostle is 1.58/point
against 0.71 before it. Widening the band explains some of it — the same ratio
in arm A is 0.53 → 0.74, and a wider crop admits more people — but not a factor
of 2.2. Near-side density is roughly flat across the same boundary (1.24 → 0.87),
so this is specific to the far side after the camera moved. It may be real (more
far serves in the later part of the match) or it may be false positives that the
recall number cannot see. `tags.json` records point starts only and misses
faults, so serve-level precision cannot be scored from it, and this stays open
until there is serve-level truth on this clip.

**One honest negative.** On the two coarse far-side backstops the corrected
geometry is marginally *worse* — `_trackable` survival 95.69% → 95.42% over
16,072 far detections. The correction pushes the far player ~1.8 m deeper in the
last third, toward `FAR_BACK_M`'s 12 m cutoff, and that 12 was measured against
the *uncorrected* projection: it absorbed some of this error. Near-side
eligibility moves the other way, +1.7 points in the last third. Correcting the
geometry without revisiting the constants that were fitted around its absence is
a job only half done, and `FAR_BACK_M` is the one to revisit first.

Turn it off with `ANYA_CAMERA_TRACK=0`, which restores the previous behaviour
everywhere at once: nothing estimates a track and nothing reads a cached one.

### Near-side serve — results

Scored against `ground_truth.json` with `eval.py`, ±2.0 s, greedy one-to-one,
restricted to each clip's labelled span. **All 13 trusted clips, 107 near
serves.**

| clip | labelled | recall | precision |
|---|---|---|---|
| 21 | 11 | 90.9% | 90.9% |
| 22 | 11 | 100% | 91.7% |
| 23 | 0 | — | **0 fires** |
| 24 | 2 | 100% | 66.7% |
| 25 (doubles) | 5 | 100% | 100% |
| 26 | 2 | 100% | 100% |
| **35 (out-of-sample)** | **5** | **100%** | **100%** |
| 36 | 8 | 100% | 100% |
| 38 | 8 | 100% | 100% |
| 40 (doubles) | 0 | — | **0 fires** |
| 43 | 6 | 100% | 100% |
| 50 | 5 | 100% | 71.4% |
| **58 (holdout)** | **44** | **65.9%** | **55.8%** |
| pooled | **107** | **90.7%** | **76.4%** |

Nine of eleven scored clips are at 100% recall. **Clips 23 and 40 have no near
serves and the detector fires zero times on either** — 40 being doubles, so two
near players stand at the baseline through 13 far-serve rallies without
producing a start.

### How it got there

Three fixes, in order of how much they mattered. The first two are defects in
the seed that eyeball review could not have found, because both are invisible
without labels.

1. **The trophy is a phase, not an instant.** The seed multiplies three shape
   terms sample-by-sample. On Data/38's five missed serves every term clears its
   threshold comfortably — `hi_head` +0.19…+0.27 against a 0.10 line — while the
   product never exceeds 0.30, because the tossing arm reaches full extension a
   sample or two before the racket arm settles. Each term is now dilated ±0.20 s
   before the product. Recall 79% → 100% on the nine short clips.

2. **Trophy onset is not the point start**, and lands +1.63 s late. Walking back
   to the last instant of the ready stance gets to +1.13 s; the rest is a
   labelling convention. Two other anchors were tried and are worse — starting
   the ready run scores sd 2.46–2.68 against the hand split's 1.41.

3. **The eval was scoring unlabelled regions.** Clip 38 is labelled to 206 s of
   420 s, and all three of its apparent false positives lived in the tail.

### What clip 58 showed, and why it was worth holding out

Clip 58 is a 55-minute match; the other eleven are 7-minute snippets. Every
parameter was fixed before its perception pass finished. It scores 65.9% / 55.8%
against ~100% / ~93% on the clips used for tuning, and the gap decomposes into
two findings, neither of which is a threshold problem:

**The label lead does not transfer.** Fitted on the nine short clips the
hand-split lead is 1.13 s; clip 58 wants 2.4 s. Per-clip leads genuinely span
0.9–2.4 s, and at a ±2.0 s tolerance no constant satisfies both ends: at 1.13
clip 21 scores 100% and clip 58 57%; at 2.00 clip 21 falls to 91% and clip 58
reaches 68%. Clips 22–50 do not move at all. `SERVE_LEAD_S` is now the median
over all ten clips (1.63) — a compromise, and worth only ~9 points on clip 58.
**A tighter tolerance would need a per-clip lead estimated from the clip's own
detections, not this constant.**

**The rest is overheads, and pose cannot fix it.** Of clip 58's 24 false
positives, **23 fall inside a labelled rally** rather than in dead time — and
all 23 are struck from *inside the serve zone*, at the same court depth as real
serves (median court_y −0.62 m for both). These are baseline overheads and high
defensive balls during a point. The seed predicted this exactly: a smash and a
serve are not different at the joints, and the court gate is the only ball-free
separator there is. It has nothing left to give here, because the player really
is standing where a server stands.

That is not a tuning problem, and no threshold will move it. **The fix is
structural: do not look for a serve while a point is live.** The detector
already declares this — `Requirement(windows="between_points")` — and honouring
it is the composition layer's job, once the point-end detector exists. Expect
most of clip 58's precision gap to close then, and treat 55.8% as the floor a
detector reaches with no knowledge of whether a point is in progress.

### Far-side serve — results

Same scoring: ±2.0 s, greedy one-to-one, restricted to the labelled span.
**Eleven clips carry a far serve, 129 in all.** Clip 58 was held out until its
far pass finished; clip 35 was out of the corpus entirely until relabelled.

| clip | labelled | recall | precision |
|---|---|---|---|
| 23 | 15 | 100% | 78.9% |
| 24 | 12 | 100% | 92.3% |
| 25 (doubles) | 10 | 100% | 71.4% |
| 26 | 11 | 100% | 45.8% |
| **35 (out-of-sample)** | **15** | **93.3%** | **93.3%** |
| 36 | 9 | 100% | 75.0% |
| 40 (doubles) | 13 | 100% | 81.2% |
| **seven far-dominant clips** | **85** | **98.8%** | **74.3%** |
| 21 | 1 | 0% | 0% (14 fires) |
| 22 | 4 | 75% | 18.8% |
| 50 | 2 | 0% | 0 fires |
| 38 | 0 | — | 9 fires |
| 43 | 0 | — | 3 fires |
| **58 (holdout)** | **37** | **51.4%** | **27.5%** |
| pooled, all 13 clips | **129** | **79.8%** | **64.8%** |

Bias +0.07 s. **Every far serve on every far-dominant clip is found.** The clips
that look terrible are near-dominant — 1, 4 and 2 far serves against 11, 11 and 5
near ones, and clips 38 and 43 with NO far serves at all — where the far player
spends the whole clip RETURNING, and a return is a serve motion the detector has
no way to place inside a point.

Clips 38 and 43 are the cleanest statement of that: zero labelled far serves, and
the detector fires 9 and 3 times inside their labelled spans. There is no
threshold that fixes this, because the motion really is a serve motion.

### Clip 58, the holdout — and what it says about the scoring

Clip 58's far pass finished after the parameters were fixed. It scores **21.6%
recall / 11.6% precision** on 37 far serves, against 100% on the six clips used
to tune. That is a worse collapse than the near detector's, and it decomposes
into two separate things:

**Most of it is the label lead again, not the detector.** 30 of the 37 serves
have a detection within 12 s; the median nearest-detection error is +2.32 s with
sd 2.31. An oracle per-clip lead of +2.25 s would give 24/37 = 65%. The threshold
is not implicated at all — at threshold **zero**, recall is still only 32%.

**The rest is genuine signal absence.** The trophy shape clears its floor
anywhere in the far band for only 23 of 37 serves. On clips 25 and 40 the same
diagnostic said 10/10 and 13/13 and the fault was slot assignment; here the
signal simply is not there for a third of the serves.

### Detection quality vs label-convention alignment

The lead exists only to match a labelling convention — `ground_truth.json`'s
`start` is a point boundary marked before the server moves. It is an EVALUATION
alignment parameter, not a product one: a reel needs pre-roll, which is a
user-facing choice, not an estimate. So it is worth reading recall at a wider
tolerance and with a per-clip lead removed, which separates "did we find the
serve" from "did we time it to this labeller's convention":

| | R @±2 s | R @±4 s | R @±2 s, per-clip lead |
|---|---|---|---|
| **near**, 9 short clips | 100% | 100% | 100% |
| **near**, clip 58 | 66% | 86% | 80% |
| **near**, pooled (102) | 84% | 94% | 91% |
| **far**, 6 far-dominant clips | 100% | 100% | 100% |
| **far**, clip 58 | 22% | 54% | 59% |
| **far**, pooled (114) | 71% | 82% | 83% |

The per-clip-lead column is an upper bound a deployment cannot fit, since it
needs labels. It is here to attribute error, not to claim performance.

Read that way: **the near detector finds essentially every serve it is shown and
the residual is convention.** The far detector does the same on far-dominant
clips, but on a full-length match a third of its serves are genuinely invisible
to it even with perfect timing — that is the honest limit, and it is the next
thing to attack on the far side.

### Why it is not the near detector with a flag

Two of the near trophy's three terms do not survive the change of viewpoint and
scale. Measured on Data/23's 15 far serves, per pose sample, against every
non-serve sample in the clip:

| term | serve | non-serve |
|---|---|---|
| wrist above head | 13% | **0%** (p99 = +0.002) |
| lower wrist near the shoulder | 11% | 2% |
| **hand split** | 35% | **58% — inverted** |

The hand split is the sharpest cue the near detector has and is worse than
useless here: at the trophy the far player's arms extend VERTICALLY, so their
horizontal separation is small exactly when the near view has it large, while
ordinary walking swings the arms apart. The far trophy is elevation-only and
never reads `gap`.

The serve-zone court band is dropped too. A far player's ground point is 22-32
px up the frame, where two pixels of box-bottom error is metres of court: their
`court_y` spans 19.7-28.6 m while really moving about three, and sits a median
4 m behind the baseline they are standing on.

### Three substrate defects it exposed

Each was found by measurement, and each was costing recall outright:

1. **Crop-truncated boxes were projected as ground points.** A person straddling
   the bottom of the far band is cut off, so their box bottom is the *crop
   boundary*, not their feet. 23–61% of far-band detections are truncated this
   way — and being closest to the camera they carry systematically *higher*
   confidence than the real far player (0.82 vs 0.54 on Data/40). They won every
   competition for the two far slots and pushed the server out of the tracks:
   clip 40's serve trophy was present at full strength in the band for all 13
   labelled serves while the tracks caught six. Fixing it took clip 40 from
   46% to 100% recall.

2. **`FAR_BACK_M` was deleting a third of clip 23's far detections** — real
   players rejected for standing where the homography said they were. Clip 23
   went 66.7% → 100% on that alone. Re-measured after the truncation fix: the
   spread is genuine homography noise, so 12 m stands.

3. **The claim score preferred the biggest box**, which is a *depth* proxy and is
   side-specific. On the near side bigger means closer to the near baseline, so
   it was accidentally right; on the far side bigger means closer to the net, and
   the server is by definition the deepest player. Replaced with proximity to the
   player's own baseline — right on both sides at once. Clip 25 went 30% → 100%,
   with no near-side regression.

### Measured dead ends — do not retry these

Two local cues were tested for separating far serves from far false positives,
because if one worked it would fix the near-dominant clips without any context:

  box height / that slot's clip median   **AUC 51%** — no separation at all
  projected court_y at the trophy        AUC 66% — real, far too weak to gate on

So there is no local depth cue. This agrees with every previous far-side attempt
(DESIGN.md's velocity dead end, and the far-gate taxonomy's finding that the
far player's ordinary play looks like a serve from that viewpoint).

### The residual is structural, and it is the same one the near detector has

Over all 67 false positives inside labelled spans on eleven clips, **79% fall in
a live point** — 75% in-rally, 4% the far player reacting to a near serve —
against 21% idle raises in dead time. The shipped `anya_far_serve` measured the
same taxonomy at 56% over 14 clips; ours is more concentrated because recall is
higher.

None of that is visible from inside a far-player pose crop. The module declares
`Requirement(windows="between_points")` and leaves the arbitration to the
composition layer, which will have the near serves and the point ends that
settle it. **Treat 52% pooled precision as the floor for a serve detector with
no knowledge of whether a point is in progress** — and note that the near
detector bottomed out at 55.8% for exactly the same reason.

### Point end — results

**Pose only. The ball is not read anywhere in the module**, at the user's
request: the shipped policy makes the ball trace primary, and that is not
dependable on clay, where the ball is low-contrast against the surface for much
of its flight.

Scored ±2.0 s over all **236 labelled ends on 13 clips**:

| | recall | precision | bias | truncations |
|---|---|---|---|---|
| shipped **ball-trace** policy (clips 21/22/23) | 48% | 65% | +0.87 s | 0 |
| **anya2 pose-only** (same 3 clips) | **57.1%** | 55.8% | −0.24 s | **0** |
| **anya2 pose-only** (all 13 clips) | 49.6% | 40.2% | −0.27 s | **0** |
| **anya2 pose-only** (clip 35, out-of-sample) | 60.0% | 60.0% | −0.92 s | **0** |

Better recall than the ball-based policy with no ball at all; behind on
precision. **Zero truncations on every clip** — no detected end lands more than
2 s early, so the harmful error (deleting live tennis from the reel) does not
occur. Per-clip recall spans 30.8%–78.6%; clip 58 (the 55-minute match) and clip
40 (doubles) are the weak ones at ~30–38%.

### Four measurements, in the order they killed the obvious designs

1. **Instantaneous activity does not separate live from dead** — AUC 38–75%
   per sample, at or below chance on the hardest clips. A rally contains long
   quiet beats; dead time contains a player walking to the ball. Motion does not
   stop at the end of a point, it changes character.

2. **Sustained quiet is nearly perfect and useless for timing.** Both players
   quiet for 1.5 s covers 0.0–1.2% of live play and 7–21% of dead time — close
   to proof the point is over. But the first such window after a labelled end
   arrives a **median +78 s** later. Quiet marks changeovers, not point ends.

3. **Walking has the coverage but not the timing.** A walk onset follows 212 of
   216 ends — essentially every one — but at median +5.1 s, p75 +15.9 s, because
   the walk is a *consequence* of the end. `near_end`'s four pose signals cover
   the gap between the last ball and the first step, and their union with
   walking is far better than any part (clip 58: p75 +6.7 s vs +24.8 s for
   walking alone) — even though every corroborator is individually *worse* than
   walking. They are worse on average and earlier where walking is late, which
   is all a `max()` asks of them.

4. **But the union is a weak live/dead signal on its own** (frame AUC 60%; its
   parts 48–59%, `settle` below chance). That is not a failure of the signals —
   it says the question they were built for is not this one. What carries
   live/dead is **player activity integrated over seconds**:

   | | AUC (live > dead) |
   |---|---|
   | near activity, 8 s | 79.5% |
   | far activity, 8 s | 75.1% |
   | max(near, far) − union | 82.6% |
   | **max(near, far) × (1 − union)** | **86.7%** |

   A product, not a sum: the union's job is to **veto** activity, not to be
   traded against it. A player walking to the ball is active and emphatically
   not playing, and only a multiplicative term can say so.

   The far-activity term exists only because anya2 tracks the far player —
   no previous point-end work here could, and memory records "far-serve rallies
   read as dead" as the biggest error source of the dead/live GRU.

5. **The end is the FALLING EDGE of that score, not the onset of a dead state.**
   Scored as dead-state onsets the detector emitted 504 candidates for 216 ends,
   and no local feature separated the good ones (run duration AUC 53%, depth
   64%, quiet overlap 1%). What does separate them is that a real end is
   *preceded by play* — which a falling edge encodes and an onset does not.

### Known gaps

- **Precision, and clips 58 and 40.** Half the corpus' ends are on clip 58 and
  it scores 38.3%/25.0%. The pooled row is dominated by it.
- **The near-slot shim.** The walking classifier is fed from anya2's near tracks
  through a shim npz; near coverage varies 42%–88% by clip, and on clip 23 the
  `settle` and `stance_drop` signals produce no usable onsets at all.
- The ball has **not** been tried as a minor corroborator. It may be worth a
  bounded experiment for precision, but only if it can be shown to earn its cost
  on clay.

### Can far-player depth be made reliable, and does it help?

**Yes to the first, no to the second, and the second is the one that matters.**

*Reliable:* raw per-frame `court_y` for a far player is noise-dominated — 11.9%
of consecutive samples imply a speed above 9 m/s, which nobody can run, and the
p90 implied speed is 12.3 m/s. A 0.7 s median filter fixes it: **impossible
samples 11.9% to 4.2%, p90 implied speed 12.3 to 3.2 m/s.** So depth *is*
recoverable, and the box-bottom projection is the better of the two estimators
available — a box-height proxy is worse (20% impossible against 12%).

*But it does not strengthen the far serve detector.* Every depth-derived
quantity tested sits at chance for separating a true far serve from a far false
positive:

| quantity | AUC |
|---|---|
| depth at the trophy, relative to the clip median (raw) | 41% |
| the same, median-filtered 0.5 / 1.5 / 3.0 s | 41 / 46 / 47% |
| step into the court during the strike | 47% |
| depth change from −1 s to +2 s | 55% |
| pre-serve drift, −4 s to −1 s | 47% |

The reason is structural, not a measurement failure: **the false positives are
the far player RETURNING, and a returner stands at essentially the same depth as
a server.** Depth cannot separate two things that happen in the same place. What
*does* separate them is what the player was doing before the racket went up —
stillness (AUC 78%) and a sustained ready stance (79%) — which is what the
refinement below uses instead.

### Far-serve refinement: how the ready phase is read, and pre-serve stillness

Two changes, both from measuring what actually separates a far serve from the
far player's *return* — which is where the false positives were.

**1. The ready term was inert.** It read the **max** of `ready` anywhere in the
6 s before the trophy. Over 105 true far serves and 101 false positives its
median is **1.00 on both**, separating at AUC 62% — any 6-second window contains
some quiet moment, so a returner who stood still once five seconds ago scored a
perfect ready. Read instead as the **mean over `[k−2.0s, k−0.3s]`**, ending at
the trophy, it separates at **AUC 79%** (true 0.90, false 0.62). The difference
is "was quiet at some point recently" versus "was quiet right up until the racket
went up", and only the second is a service stance.

**2. A server is stationary before serving; a returner has just been running.**
New term, and the largest single discriminator found: the median of the player's
own translation (box centre, body heights/s) over `[k−5s, k−1s]` is **0.15 for
true serves and 0.31 for false positives, AUC 78%**. It is the mirror of ready
and independent of it — **ready is about the arms, this is about the feet** — and
it is local to this detector, needing no other agent. It enters multiplicatively
at weight 0.30 so it can veto a candidate that is clearly mid-rally without
deleting one whose track was noisy; an untracked stretch scores 0.5, because not
knowing where the player was is not evidence either way.

Threshold re-swept, since both terms multiply the score down:

| ready | stillness | thr | recall | precision | F1 |
|---|---|---|---|---|---|
| max-6s | — | 0.90 | 81.4% | 51.0% | 62.7 |
| max-6s | 0.3 | 0.85 | 80.6% | 63.0% | 70.7 |
| **mean** | **0.3** | **0.75** | **79.8%** | **64.8%** | **71.8** |
| mean | 0.3 | 0.85 | 70.5% | 72.8% | 71.7 |

**Corpus: 81.4%/51.0% → 79.8%/64.8%** — 1.6 points of recall for **13.8 of
precision**. Per clip:

| clip | before | after |
|---|---|---|
| **21** (1 far serve) | 0% / 0%, **14 fires** | 0% / —, **0 fires** |
| 26 | 100% / 45.8% | 100% / 64.7% |
| 36 | 100% / 75.0% | **100% / 100%** |
| 25 | 100% / 71.4% | 100% / 83.3% |
| 24 | 100% / 92.3% | 91.7% / **100%** |
| 38, 43 (no far serves) | 9 and 3 fires | **2 and 0** |

Clip 21's fourteen phantom far serves — the case this refinement was aimed at —
are **gone entirely**, and the eleven that were the far player returning are
exactly the ones the stillness term rejects.

Reel effect is small and in the tightening direction: whole points 143 → 137,
live retained 89.5% → 87.9%, reel 47.5% → 45.5% of source. Fewer far detections
means fewer anchors, so a few points lose their start.

### Roll tightened to 1 s, and a far-side leak found (2026-08-25)

**Roll is now 1.0 s / 1.0 s** at the user's direction. Corpus effect:

| | pre/post 3.5/4.0 | **1.0/1.0** |
|---|---|---|
| points whole | 207 | **143** |
| partial | 22 | **85** |
| live retained | 95.3% | **89.5%** |
| reel | 61.6% of source | **47.5% of source** |

### Debugging Data/21's far false positives

Clip 21 has **11 near serves and 1 far**, and the far detector fired **14 times
at confidence 0.92–1.00** — while missing the one real far serve entirely. The
14 split into two unrelated causes:

**Three were a near player occupying a far slot.** Rendering the frame settled
it: at t=60.3 s the near player had moved forward, appeared inside the far-band
crop, and was tracked as a far player with a **69 px box where the far baseline
allows 26**. The existing crop-truncation filter missed them because their boxes
sit just inside the crop edge rather than on it.

The fix is geometric, in `tracks._height_plausible`. A person on the court has a
predictable pixel height — the court's own width at their depth gives
px-per-metre, and a player is ~1.75 m. Measured across four clips and both sides
the observed/expected ratio has median 0.82–1.06 and **p90 never above 1.24**, so
a box beyond 1.40× is a contradiction: not a player at that depth, but a closer
one mis-projected. The bound is one-sided — a box *smaller* than expected is
ordinary (crouching, partial detection, a clipped head) and rejecting those would
cost real tracking.

**Corpus effect: far precision 47.3% → 51.0% for 0.8 points of recall.** Near is
untouched.

**The other eleven are the far player returning a near serve** — genuinely the
far player, genuinely a serve-like motion, in a live point. Those are contextual
and the orchestrator's live gate is the only thing that can see them. It is set
at 0.80 and catches 3 of the 11; their median live score is 0.72 against 0.17 for
the dead-time ones, so a gate at 0.35 would catch 9 of 14 on this clip — but
corpus-wide that costs far recall 82% → 54%. **Not worth it**, and recorded here
so it is not retried:

| live gate | far recall | far precision |
|---|---|---|
| 0.80 (current) | 79.1% | 54.0% |
| 0.60 | 76.0% | 56.3% |
| 0.50 | 71.3% | 59.0% |
| 0.40 | 62.8% | 60.0% |

Also tested and rejected on this clip: dropping starts that contradict the
service run, and raising `min_service_run` to 4 or 5 — **neither changed clip 21
at all**, because the phantoms cluster into runs of 2–3 that any game-length
minimum accepts as legitimate.

### Toss evidence, and two confidence rules (2026-08-25)

### The pose toss score (far_serve.toss_score)

A **separate** score, never folded into the serve `p`, so the orchestrator can
arbitrate. Read from the tossing arm only: the wrist started low, is offset
laterally at the top, is held above head height, and how far above it reaches —
four ramps, averaged.

| | AUC vs far false positives |
|---|---|
| each component alone | 64–66% |
| **the mean of the four** | **75%** |
| correlation with the serve score `p` | **+0.04** |

That last row is the point: it is independent evidence, not a restatement of the
trophy. Two findings worth keeping — the toss arm is **not** vertical (true
serves have a *larger* lateral offset, 0.148 vs 0.090 bh, because the ball goes
up and forward across the body), and the four components must be **averaged, not
multiplied** (the product scores the same AUC but is degenerate, costing 53% of
true serves to cut 88% of false ones).

### The ball toss tracker — re-aimed, now works, still not shipped

A first attempt concluded the ball was undetectable at far-court range. **That
was wrong, and for two aiming bugs rather than any limit of the ball model:**

- **The window was before the toss.** `far_serve`'s event time is lead-corrected
  to ~0.9 s *before* the trophy, so `[t−0.2, t+1.0]` covered `trophy−1.1s …
  trophy+0.1s` — while the ball is still in the hand. It is airborne from about
  `trophy+0.0` to `+0.6`.
- **The ROI was above the head.** At this range the toss does not rise far above
  the head in *image* terms; it sits just above the tossing hand.

Re-aimed at the **tossing wrist**, anchored on the **trophy**, at native
resolution with SAHI tiling (160 px tiles, 30% overlap, imgsz 320, distance NMS):

| | frames with a ball per serve |
|---|---|
| head-centred ROI, wrong window | 0–0.5 |
| wrist-centred ROI, single shot | 5.0 |
| **wrist-centred ROI + SAHI** | **8.0** |

**But ball presence does not discriminate.** A returner also has a ball near
them — the one they are about to hit. Median frames-with-ball is 8.0 true against
6.0 false, and "≥3 frames with a ball" is 71% true against **83% false**.

**The arc does, but only through direction:**

| trajectory feature | AUC |
|---|---|
| total rise of the highest ball | 49% |
| peak height above the wrist | 45% |
| **fraction of time the ball is climbing** | **73–75%** |

**And it is redundant with the pose toss.** On the same 19 true / 12 false
sequences the ball arc scores 75%, the pose toss 83%, they correlate at **+0.49**,
and the best blend (0.3 arc + 0.7 pose) reaches **83% — no better than pose
alone**. Both watch the same event; the arm is simply easier to see than the ball.

So a pass costing tiled ball inference over ~18 native-resolution frames per
candidate, with a video seek each, buys nothing over a signal already computed
for free. `far_toss_ball.py` is complete and tested; enable it only if that
changes — a closer camera, a better small-object model, or a clip where far pose
degrades. The sample is small (19/12), so this is a reason not to spend the
compute, not proof the ball can never help.

### Rule 1 — toss evidence adjusts far confidence
### Rule 2 — a far serve among near serves is suspect

Both **adjust confidence, never delete**. A hard gate on toss below 0.40 removes
54% of false positives but 18% of true serves, and the brief is precision without
giving up recall. Rule 2 fires when ≥3 of a far candidate's 4 neighbouring starts
are near-side — measured to flag 18% of remaining far false positives at **zero**
true-serve cost, where the 2-of-4 setting starts costing 7%.

Rule 1's bound is asymmetric (+0.15 boost, −0.35 penalty): seeing a toss is
strong evidence *for* a serve, while not seeing one is weak evidence against,
since the arm may be occluded. Both run **before** the service-run DP, because
rule 2 reads the sides as detected and the DP would otherwise have overwritten
them.

**Effect on far starts leaving the orchestrator** (87 matched, 33 unmatched):

| | TP median | FP median | AUC |
|---|---|---|---|
| raw `p` | 0.97 | 0.94 | 64% |
| **adjusted_p** | **1.00** | **0.68** | **74%** |

A consumer thresholding `adjusted_p ≥ 0.70` keeps 80% of true far serves and 45%
of false ones — where raw `p` could barely tell them apart.

### Acquisition and inference cost

Everything above the two pose passes is numpy; the passes ARE the cost.

| pass | what | cost, clip 58 (56 min) |
|---|---|---|
| near | 540p whole-frame proxy, pose @15 Hz, **imgsz 640** | 784s → **484s** |
| far | native-res baseline band, pose @15 Hz, **imgsz 960** | 534s |
| everything else | walking model, near_end, tracks, 3 detectors, reel | seconds |

**~0.30× realtime** end to end, down from 0.39×.

**The near pass had headroom; the far pass has none.** That asymmetry is the
whole story of this pipeline, and it is the same fact that forced two ROIs in
the first place — near players are 72–275 px and far players 22–32 px, right at
the model's detection size.

- **near 960 → 640**: clip 35 near serve 100%/100% unchanged, pass 140s → 57s;
  clip 58 near serve 79.5%→81.8% recall, point end within noise, pass 784s →
  484s. Adopted.
- **far 960 → 768**: a micro-benchmark said this was free (35.2 vs 52.6 ms/frame,
  1.00 vs 1.02 persons/frame). End-to-end it cost **recall 100% → 86.7%**.
  Rejected. Persons-per-frame over a short sample cannot price this, because the
  frames that break are the ones where the player is smallest — which is
  disproportionately the serves. **Only an end-to-end recall number can price a
  far-side change.**
- **far 15 Hz → 10 Hz**: cannot be done alone. `tracks._stack` requires both
  passes on one timeline, since side membership is decided by the court rather
  than by which pass found a person. Attempted, `tracks` refuses — and the
  refusal produced a tempting false positive: with tracks failing, the far
  detector scored the *stale* 15 Hz tracks and reported identical accuracy for
  30% less time. Decoupling means resampling the far stream onto the near
  timeline: a substrate change, not a parameter.

### Known gap

11 of 13 clips still take their near detections from the legacy
`_end_walk_dets.npz` (540p, imgsz 960) rather than `perceive.near`; only clips
35 and 58 use the new pass. The corpus is therefore mixed, and a full
regeneration at imgsz 640 would shift the near-serve and point-end numbers by a
little in either direction.

### Agent 4 — the orchestrator

Takes the three event streams and produces the reel. It detects nothing; it
imposes the structure tennis has and the detectors cannot see.

**Over 236 labelled points on 13 clips (133 min of source):**

| | |
|---|---|
| points wholly inside a segment | **207 / 236 (88%)** |
| partially cut | 22 |
| **missing entirely** | **7 (3.0%)** |
| **live tennis retained** | **95.3%** |
| reel length | **61.6% of source** |
| segments | 135 (≈1.2 cuts/min) |
| points recovered from live play alone | 14 |

### The measurement that decides the architecture

The three detectors do not have comparable recall — near serve 90.7%, far serve
82.2%, **point end 49.6%**. So **the reel is built from starts, and ends only
trim.** A point whose end was never detected still becomes a segment; it runs for
the clip's own typical point length instead. Requiring a start/end pair would
silently drop half the points.

That asymmetry also sets the error budget: a missed end costs some dead time at
one segment's tail, while a missed start loses the whole point.

### What tennis knows that the detectors don't

- **Service runs.** One player serves a whole game, so the side sequence is
  `NNNNN FFFFF`, never `NNFNN`. A segmental DP (exact, not greedy — a greedy pass
  commits to an early wrong side and never recovers) relabels isolated flips. It
  only ever changes a *label*, never drops a start.
- **Deuce/ad alternation.** Within a game the server alternates court every
  point, always. In the video it is much weaker: re-measured over all 14
  clip-sides with at least 6 labelled serves, court-x flips sign **a mean of 63%
  of the time**, from 20% (clip 43 near) to 91% (clip 58 near). An earlier
  version of this file quoted the 91% as representative — that is the best case,
  not the corpus. Median-filtering the court track changes it not at all. At 63%
  it is barely above chance, so it only ever *flags* a suspected missing point.
- **A point is not live twice.** A serve struck while a point is already running
  is spurious — and agent 3's live score says so. Gating on the median live score
  in the 3 s before a detection drops **26% of far false positives for 4% of true
  far serves**. It is applied to the far stream only: on the near stream the same
  gate is AUC 55% and costs more than it buys, which is the FP taxonomy showing
  through (79% of far false positives are the returner mid-rally; near ones are
  the server's own repeated motions in dead time).
- **Rhythm.** Real start-to-start spacing is median 27 s; only 5% fall under 8 s.

### Recovering points nobody detected

15 of 236 points had no serve detection at all. Nothing in the serve streams can
recover those, but the live score is a different measurement and does not care
whether a serve was seen: any sustained run of live play that no segment covers
is kept, marked `recovered`, with **no side** — there is no detection to say who
served, and the reel should gain the tennis without inventing a fact about it.
This runs last, so detected points keep their serve-anchored boundaries.

**Missing points 15 → 7; live retained 88.8% → 93.1%** before roll tuning.

### Smoothness is an output, not a side effect

Two choices a pure accuracy metric would make differently:

- **When in doubt, keep footage.** An extra second of a player walking is barely
  noticeable; a cut landing mid-rally is jarring and loses the point.
- **A cut costs something.** Two segments separated by a couple of seconds of
  dead time are worse to watch than one continuous segment spanning them, so
  segments closer than `merge_gap_s` are joined and sub-4 s flashes are dropped
  or padded.

Roll turned out to be the dominant lever — far more than any end-estimation
tuning (moving the duration percentile from 85 to 97 buys 2 whole points; +1 s of
post-roll buys 17). The ends are close to right, they were just being cut a
second early:

| pre / post | whole points | live kept | reel |
|---|---|---|---|
| 2.5 / 2.0 | 179 | 93.1% | 54.9% |
| 2.5 / 3.0 | 196 | 94.6% | 57.3% |
| 3.0 / 3.5 | 202 | 94.9% | 59.5% |
| **3.5 / 4.0** | **207** | **95.3%** | **61.6%** |
| 3.5 / 5.0 | 212 | 95.8% | 63.7% |

That table is the exchange rate — lower both to tighten the reel.

### Caveats

- **Clip 38 reads 189% of span** because its labels stop at 206 s of a 420 s clip
  while the tennis (and the detectors) continue. That is a labelling gap, not a
  reel that doubled the match.
- Clip 58 still carries most of the residue: 16 of the 22 partial points and 4 of
  the 7 missing.
- `score_reel` measures the reel, not the detectors — points whole/partial/
  missing, live retained, dead per point, cuts per minute.

### Two corrections applied 2026-08-25

**1. Clip 58's labels carry the lead already.** Its author confirmed it, and it
measures that way: the median signed error from a labelled serve to the nearest
detection is between −0.93 s and +0.87 s on **every other clip**, and +0.80 s
(near) / +2.63 s (far) on clip 58. Scoring it against the corpus convention was
charging the detector for a labelling choice. `eval.LABEL_CONVENTION_S` now
brings each clip's labels onto the corpus convention at the corpus's own choke
point — the detectors get no per-clip parameter, because a detector must not
know which clip it is looking at. `--raw-labels` scores without it.

Clip 58 alone: near **65.9% → 79.5%** recall, far **21.6% → 51.4%**, both with a
bias of ≈0.00 s.

Worth stating plainly rather than smoothing over: the two modes needed different
corrections (+0.80 vs +2.63). A single built-in lead would shift both equally.
The near and far detectors anchor on different events — hands-together versus
trophy onset — so part of that gap is detector-side anchor bias that is only
visible on the clip with the largest offset. **These numbers are measured, not
derived from a stated convention.**

**2. The near trophy bands are 35% looser.** Some servers simply do not make a
textbook trophy: a compact service motion carries the racket lower and splits the
hands less, and the original bands scored those near zero however clean the rest
of the sequence was. Swept over all 107 near serves:

| loosen | recall | precision | F1 |
|---|---|---|---|
| 0.00 (original) | 86.9% | 78.2% | 82.3 |
| 0.20 | 88.8% | 77.9% | 83.0 |
| **0.35** | **90.7%** | **76.4%** | 82.9 |
| 0.50 | 89.7% | 74.4% | 81.4 |

0.35 is where recall peaks — 0.50 gives it back, so it is a maximum, not a slope
being ridden. F1 is flat from 0.20–0.35, so the choice between them is the
recall/precision preference and not a measurement: for a point START a miss loses
a whole point from the reel, while an extra start is something the composition
layer can arbitrate.

Two things the sweep settled that are worth not re-testing:

- **`TROPHY_MIN` is inert.** Swept from 0.35 down to 0.16 it changes *nothing* —
  the run threshold is not binding because the final probability threshold
  already dominates it. Lowering it looks like loosening while doing nothing.
- **Loosening the FAR trophy does not help.** The same sweep moves far recall
  only 82.2% → 83.7% at heavy loosening while costing precision. The far limit is
  resolution — the trophy is absent from the band entirely for 14 of clip 58's 37
  serves — not the threshold. Far bands are left alone.

### Clip 35 — the one clip none of this was built on

Clip 35 sat in `parse_ground_truth.EXCLUDED` for the whole of this work (7.0%
of it was labelled live), so **not one threshold, weight or window in anya2 was
chosen with any knowledge of it**. It was relabelled on 2026-08-24 — 20 rallies,
34.8% live, denser than clip 38 which was never excluded — and rejoined the
corpus. It is also the fastest source in the corpus at 119.88 fps, twice
anything else, so it exercises the stride logic at 8 rather than the usual 2-4.

| detector | labelled | recall | precision | vs pooled |
|---|---|---|---|---|
| near serve | 5 | **100%** | **100%** | above |
| far serve | 15 | **93.3%** | **93.3%** | well above |
| point end | 20 | **60.0%** | **60.0%** (0 truncations) | above |

All three land at or above their corpus averages, and the far-serve precision is
the second-best on any clip. That is the strongest evidence available here that
the constructions are not fitted to the clips they were built on — with the
important caveat that 5 near serves is a small sample, and that clip 35 is a
7-minute snippet, which is the format the detectors do well on. The failure the
holdouts exposed is specific to FULL-LENGTH MATCHES (clip 58), and clip 35 does
not test that.

### Usage

```bash
python -m pipeline.anya2.camera   --self-test          # no video needed
python -m pipeline.anya2.camera   /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.camera   /Volumes/Anya/Data/21/snippet.mp4 --report
python -m pipeline.anya2.perceive /Volumes/Anya/Data/21/snippet.mp4 --roi near
python -m pipeline.anya2.perceive /Volumes/Anya/Data/21/snippet.mp4 --roi far
python -m pipeline.anya2.tracks   /Volumes/Anya/Data/21/snippet.mp4 \
    --dets   /Volumes/Anya/Data/21/snippet_anya2_near_dets.npz \
    --far-dets /Volumes/Anya/Data/21/snippet_anya2_far_dets.npz
python -m pipeline.anya2.near_serve /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.far_serve  /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.eval --mode near_serve --arm anya2:_anya2_near_serve.json
python -m pipeline.anya2.far_serve  /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.point_end  /Volumes/Anya/Data/21/snippet.mp4
python -m pipeline.anya2.eval --mode far_serve  --arm anya2:_anya2_far_serve.json
python -m pipeline.anya2.eval --mode point_end  --arm anya2:_anya2_point_end.json
python -m pipeline.anya2.orchestrator /Volumes/Anya/Data/21/snippet.mp4
```

### Several GoPro files for one match

A GoPro splits a continuous recording into ~4 GB chapters. `run.build_reel`
takes them directly — as does `orchestrator`, and the desktop app's file
picker — and joins them, losslessly and at disk speed, into the one video
every stage below the join then sees:

```python
from pipeline.anya2.run import build_reel
build_reel(["GX010123.MP4", "GX020123.MP4", "GX030123.MP4"])
```

Order is derived from the GoPro chapter field, not from the order they are
passed. The join lands in the artifact dir, so under the desktop app it is
part of the `tmp_anya` the run cleans up; the reel itself still lands beside
the FIRST chapter.

The per-stage CLIs above stay single-path. To drive one of them across
chapters, materialize the join first and point it at the result:

```bash
python -m pipeline.join GX010123.MP4 GX020123.MP4   # prints the joined path
```

See `pipeline/join.py` for why the pipeline joins rather than learning to read
a list of inputs.

### Known gaps

- **Precision on a full match is gated by the composition layer**, not by this
  detector — see clip 58 above. Until a point-end detector exists there is no
  way to suppress mid-rally overheads, and 55.8% is the floor.
- **The lead is a compromise across clips** and should become a per-clip
  estimate if the tolerance ever tightens below ±2 s.
- **The far ROI is built but unused.** `perceive.far` derives the band from the
  homography and grows it upward by 2.8 m of court scale (a ground-strip band is
  ~55 px on a 4K clip and clips the server's raised arm). Nothing reads it yet.
- **The end signals still project through the fixed homography.** `walking.predict`
  and `near_end.frame_signals` each call `load_homography` themselves, so after a
  jostle their `court_x`/`court_y` carry a constant offset and their `speed`/`acc`
  carry a one-sample spike at the jostle instant. anya2's own gates are corrected;
  these two feed agent 3 and are not. Threading `Geometry` through them means
  changing signatures in `walking/`, which is why it is listed here rather than
  done.
- **Slot flapping on clip 38** — 31 identity switches over 420 s on a singles
  clip. The doubles clips are stable (2 and 1), which is what the "who served"
  answer depends on, so this is not currently costing anything measurable.

---

## Near-serve trophy anchor (2026-09-05, merged 2026-10)

The evidence behind `near_serve.SERVE_LEAD_S = 1.75` and `serve_onset`. This is
the verbatim commit message of `faceeb7` from the branch
`claude/near-serve-detector-conditions-a58148`, which was ported into the
rewritten code (QUIRKS #5). Its README changes were not ported.

```text
Anchor the near serve on the trophy peak, not the hand split

The near detector timed a point start by walking BACK from the trophy peak to
the last sample where the hands were together on the grip, on the grounds that
the hand split is the sharpest, best-sampled transition in the service action.
That is true about the SHAPE and false about the TIME: where the split sits
relative to the trophy is not fixed.  Over 58 detections paired to a labelled
serve the interval ranges 0.07 s to 4.07 s, because a server who splits the
hands and then pauses -- grip, line call, a second look at the returner -- puts
seconds of unrelated delay into the timestamp.  The trophy peak has no such
freedom; it is the instant the shape is maximal.

    anchor            median    sd    within +/-2.0s after best constant
    trophy peak       +1.67    0.67          100.0%
    hands together    -0.36    0.75           98.3%

Worth two misses AND two false positives on the twelve short clips -- a mistimed
true positive is charged as both -- taking them 88.9%/86.2% -> 92.1%/89.2%, and
clip 58 65.9% -> 84.1% recall.  Data/38's serve labelled at 159.5 s is the
clearest case: trophy at 159.8 s, but the hands were last together 1.60 s
earlier, so the event landed at 156.6 s and scored as a miss and a false
positive at once.

SERVE_LEAD_S is 1.75 and sits on a plateau, not at a fitted point: swept in
0.05 s steps every value from 1.40 to 1.80 scores identically, and 1.75 is
picked from inside it because the signed bias crosses zero there.
TROPHY_LEAD_S and TOGETHER_BACK_S are gone; t_basis stays in the contract.

The README's near table was regenerated from freshly run detections on all 13
clips, because every clip's detection JSON was 8-10 days OLDER than the tracks
npz it was nominally derived from.  eval.py warns when a run file is missing but
not when it is stale, so the old table recorded a miss on clip 21 that does not
exist and 100% recall on clips 36, 38 and 43 that also does not.  The tell was
physical: three "false positives" carried ready scores of 0.60-0.73 while having
zero track coverage in the ready lookback, where s_ready is 0.0 by construction.

Also recorded, from user review of every false positive on the short clips: they
are reverse / buggy-whip forehands, and no band separates them.  The unit turn
puts both hands on the grip, the non-dominant arm goes up for balance, and the
buggy-whip finish carries the racket wrist above the head, so all three terms
score 1.00 honestly.  A ready >= 0.75 gate rejects six of seven but costs eight
true serves; no W_READY/threshold pair does better.  Clip 58 is unchanged in
kind -- 21 of its 22 false positives fall inside a labelled rally -- so the fix
remains structural: honour windows="between_points" now that point_end exists.
```
