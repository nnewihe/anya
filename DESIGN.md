# Anya Tennis — Design Spec

A design refresh for the **Anya Tennis** app. Same engine, same business logic —
this document only covers **look & feel**, **screen flow**, and one new
**background-processing** capability. No changes to the rally-detection /
dead-time engine (`mobile/lib/engine/`) are in scope.

> **Scope note.** This is a design/behaviour spec, not an implementation. It
> describes what each screen should look like and do. File paths are given so
> the eventual implementation lands in the right place.

---

## 1. Brand

### 1.1 Name & positioning

**Anya Tennis** — turn a full match recording into a tight rally reel. The app
watches the match so you don't have to sit through the dead time between points.

### 1.2 Tagline

Replace any existing tagline (currently baked into the logo mark and the
"Detect rallies on your device" home copy) with:

> **Primary:** "Watch your matches in minutes, not hours."
>
> **Sub-line (optional, smaller):** "We cut the dead time between points, so you
> only see the tennis."

Alternates, same idea, if a shorter form is needed for tight spaces:

- "All the rallies. None of the waiting."
- "Every point, minus the dead time."
- "The whole match in a fraction of the time."

### 1.3 Colors — black & yellow primary

Black and yellow are the two primary brand colors. The current theme
(`mobile/lib/theme.dart`) is already close; the refresh **promotes yellow to the
hero accent and black to the ground, and demotes the sky-blue** to a rare,
functional-only accent (or removes it entirely — see below).

| Token | Hex | Role |
|---|---|---|
| `black` | `#000000` | App background, primary buttons' text, deepest ground |
| `surface` | `#141412` | Cards, sheets, list backgrounds |
| `surfaceAlt` | `#1F1F1B` | Raised surfaces, input fields, chips |
| `yellow` | `#E8FF3D` | **Primary accent** — buttons, progress, active states, logo ball, key highlights |
| `outline` | `#3A3A36` | Borders, dividers |
| `white` | `#FFFFFF` | Primary text on dark |
| `textDim` | `#A0A099` | Secondary text, captions |
| `error` | `#FF6B6B` | Errors only |

**Sky-blue (`#49C5F1`) stays as the secondary accent.** Black and yellow are the
two primaries and carry the brand; sky-blue remains a supporting accent for
quiet, functional emphasis (chips, list-tile icons, secondary info) — never as a
headline color and never competing with yellow for the primary action. Yellow is
always the focal point; sky-blue is the calm second voice.

Rule of thumb: **the eye should land on yellow.** One yellow focal point per
screen (the primary action or the progress indicator). Everything else is black,
grey, and white.

### 1.4 Logo

The user is supplying a new logo. When provided:

- Drop the new asset into `mobile/assets/images/` (keep the existing filenames
  so `mobile/lib/widgets/anya_logo.dart` needs no change, **or** update that
  widget's asset paths to match the new filenames).
- Provide, ideally, two forms:
  - **Full wordmark** (ball + "ANYA TENNIS") — used on the Home screen.
  - **Compact mark** (ball only) — used in the app bar and tight spaces.
- Provide a **light-on-dark** variant (the app is dark-themed throughout). If the
  logo has color, the ball should read as the brand yellow on black.
- Replace the placeholder `Icons.sports_tennis` on the Home screen with the
  wordmark, and the app-bar text title with the compact mark (or mark + short
  "ANYA TENNIS" text).

Until the new logo lands, the existing SVG marks
(`assets/images/anya_logo_mark.svg`, `anya_ball_mark.svg`) remain as
placeholders.

---

## 2. Screen flow

```
Home  ──choose video──▶  Match Setup  ──(auto)──▶  Processing / Result
(pick a match)           (title, players,          (progress → reel →
                          privacy; processing        Save to Gallery /
                          already running)            Upload to YouTube)
```

Three screens. The middle **Match Setup** screen is new. The final
**Processing / Result** screen is the current Analysis screen, kept largely
as-is with a brand refresh.

---

### 3. Home screen

**File:** `mobile/lib/screens/home_screen.dart`

Purpose unchanged: pick a match video from the device; everything then runs
on-device.

**Refresh:**

- App-bar title → **"Anya Tennis"** (replacing the "Rally Predictor" text),
  optionally preceded by the compact logo mark.
- Center the **full wordmark** (replacing `Icons.sports_tennis`).
- Copy under the mark → the **tagline** ("Watch your matches in minutes, not
  hours.") plus the sub-line ("We cut the dead time between points, so you only
  see the tennis.").
- Primary CTA unchanged in function ("Choose a match video"), styled in
  **yellow** (already the filled-button color).

**Behaviour unchanged:** picking a video pushes to the next screen. The only
change is *where it goes*: instead of jumping straight into Analysis, it opens
the new **Match Setup** screen (below), passing the picked file path and name.

---

### 4. Match Setup screen  *(new)*

**File (suggested):** `mobile/lib/screens/match_setup_screen.dart`

This is the screen the user described: it appears **immediately after a video is
chosen**, and it does two things at once.

#### 4.1 Processing starts automatically

The moment this screen opens, the on-device engine begins analyzing the picked
video **in the background** — the user does **not** have to press "start." The
existing engine call (`Engine.shared().analyze(...)` from
`analysis_screen.dart`) is kicked off in `initState`, exactly as Analysis does
today.

While it runs, the user fills in the details below. A **slim yellow progress
bar** (or a small "Analyzing… NN%" line) sits at the top or bottom of the screen
so the user can see work is already happening.

#### 4.2 The details the user sets

A simple form, three inputs:

**a) Video title**

- A single text field, pre-filled with a **default title** the user can edit.
- Default format:

  ```
  {today's date} Tennis Match — {Player A} vs {Player B}
  ```

  e.g. on 2026-07-11 with empty names:
  `2026-07-11 Tennis Match — Player A vs Player B`

- The title field **stays live-linked to the player-name fields**: as the user
  types Player A / Player B names, the default title updates to use them —
  **unless** the user has manually edited the title, in which case their edit
  wins and is left alone.
- Date format: ISO `YYYY-MM-DD` (sorts cleanly; unambiguous). Locale-friendly
  alternative acceptable if preferred.

**b) Player A name / Player B name**

- Two text fields, "Player A" and "Player B".
- Empty by default; placeholders read "Player A" / "Player B".
- These feed the default title (see above).

**c) Upload privacy**

- A three-option selector (segmented control or radio group) for the **YouTube
  upload privacy**:
  - **Private** — *default*
  - **Unlisted**
  - **Public**
- This sets the `privacyStatus` used when the reel is later uploaded to YouTube.

#### 4.3 What the form feeds

The title and privacy chosen here replace the currently-hardcoded values in the
upload path:

- **Title** — replaces the current `'${widget.title} — Rally Reel'` string in
  `_onUploadToYouTube` (`analysis_screen.dart`).
- **Privacy** — replaces the hardcoded `'privacyStatus': 'private'` in
  `mobile/lib/services/youtube_upload.dart`. That service's
  `uploadPrivateVideo(...)` should be generalized to accept a
  `privacy` argument (`private` | `unlisted` | `public`) rather than being
  private-only.

The player names can also flow into the YouTube video **description** (e.g.
"Player A vs Player B — rally reel, generated by Anya Tennis").

#### 4.4 Transition to the result

Because processing runs while the form is filled, there is no separate "start"
step. When analysis finishes, the screen shows the result **in place** (reel
player + segment list + Save/Upload actions) — i.e. Match Setup and the result
view can be the **same screen** that swaps its body from "form + progress" to
"result," or it can navigate to the Processing/Result screen carrying the
user's title/players/privacy. Either is fine; keeping it one screen is simpler
and avoids losing the entered details.

**Layout sketch:**

```
┌─────────────────────────────────────┐
│  ◀  [Anya mark]                      │   app bar
├─────────────────────────────────────┤
│  Analyzing…                    37%   │   ← yellow progress, auto-running
│  ▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░     │
│                                      │
│  Title                               │
│  ┌─────────────────────────────────┐ │
│  │ 2026-07-11 Tennis Match — …     │ │
│  └─────────────────────────────────┘ │
│                                      │
│  Player A            Player B        │
│  ┌───────────────┐  ┌──────────────┐ │
│  │ Player A       │  │ Player B      │ │
│  └───────────────┘  └──────────────┘ │
│                                      │
│  Upload privacy                      │
│  ( Private ) ( Unlisted ) ( Public ) │   ← Private selected
│                                      │
│  …result (reel + actions) appears    │
│     here when analysis completes     │
└─────────────────────────────────────┘
```

---

### 5. Processing / Result screen

**File:** `mobile/lib/screens/analysis_screen.dart` (current Analysis screen)

**Kept essentially as-is** — the user is happy with it. Changes are limited to:

- **Brand refresh only:** yellow progress bar and accents (already the theme
  color), Anya mark in the app bar, sky-blue chips/icons → yellow or dim per
  §1.3.
- **Title** shown / used for upload comes from the Match Setup form, not the raw
  filename.
- **Privacy** for the YouTube upload comes from the Match Setup form (see §4.3),
  removing the "private only" limitation and the private-specific button copy.
- If Match Setup and this view are merged into one screen (per §4.4), this file's
  result-rendering body is reused directly.

Everything else — the progress %, the reel `VideoPlayer`, the segment list,
"Save to Gallery," "Upload to YouTube," the quotes-during-processing — stays.

---

## 6. Background processing

**Requirement:** analysis must keep running when the app is **not in the
foreground** — the user can switch away and come back to a finished (or
further-along) reel. Required on **iOS**, **Android**, and **desktop**.

This is a functionality addition (not a business-logic change): the *same*
engine runs; it just must not be suspended when the app is backgrounded.

**Chosen behaviour: best-effort + notify.** Keep analysing during whatever
background window the OS grants; if the OS suspends the work, it resumes when
the user returns, and a notification fires when the reel is ready. This is the
honest maximum within platform rules (iOS in particular does not permit
unbounded background CPU). Implemented via `flutter_foreground_task` on mobile
(`mobile/lib/services/background_analysis.dart`) and a stdlib sleep-blocker on
desktop (`desktop/background.py`).

### 6.1 Android

- Run the analysis inside a **foreground service** with an ongoing notification
  ("Analyzing your match — NN%"). Foreground services are the supported way to
  keep CPU-bound work alive when the app is not visible.
- Suggested package: `flutter_background` or a `foreground_service` /
  `flutter_foreground_task` plugin. The notification doubles as live progress
  and a tap-to-return affordance.
- Manifest: add `FOREGROUND_SERVICE` (and, on Android 14+, the
  `FOREGROUND_SERVICE_DATA_SYNC` / `mediaProcessing` type as appropriate)
  permissions in `mobile/android/app/src/main/AndroidManifest.xml`.

### 6.2 iOS

- iOS does not allow arbitrary long-running background CPU work. Use the
  realistic options in combination:
  - **`beginBackgroundTask` (background execution assertion)** to finish an
    in-progress analysis for the extra window iOS grants after the app
    backgrounds — often enough for short clips.
  - **`BGProcessingTask`** (BackgroundTasks framework) for longer jobs that the
    system schedules when conditions allow (charging / on Wi-Fi is typical for
    heavy processing).
  - A **local notification** when analysis completes so the user knows to return.
- Add the **Background Modes** capability (Background processing / Background
  fetch) and register the `BGProcessingTask` identifier in
  `mobile/ios/Runner/Info.plist`.
- Be explicit in UX that a very long match may pause when backgrounded and resume
  on return — iOS makes no guarantee of uninterrupted background CPU time.

### 6.3 Desktop (macOS / PyQt6)

- The Flutter macOS build and the **PyQt6 desktop app** (`desktop/app.py`) run
  processing on a **background worker thread/isolate**, decoupled from the UI
  event loop, so minimizing or losing focus never blocks the job. (The desktop
  app largely does off-thread work already; confirm the analysis is not tied to
  window/foreground state.)
- Prevent **app-nap / system sleep** while a job runs (e.g. a power-assertion on
  macOS) so a backgrounded window keeps processing.
- Optional: a system notification on completion, matching mobile.

### 6.4 Shared UX

- A single source of truth for job progress that the foreground UI, the
  background notification, and the completion notification all read from.
- On return to foreground, the UI reflects current progress or the finished
  result — no restart of a job that already completed in the background.

---

## 7. Summary of changes

| Area | Change | Where |
|---|---|---|
| Palette | Yellow = hero accent; demote/remove sky-blue | `mobile/lib/theme.dart` |
| Tagline | "Watch your matches in minutes, not hours." + sub-line | Home screen / logo |
| Logo | Swap in user-supplied wordmark + compact mark (light-on-dark) | `assets/images/`, `widgets/anya_logo.dart` |
| Home | Logo replaces text title & tennis icon; new tagline copy | `screens/home_screen.dart` |
| **Match Setup (new)** | Auto-start processing; title (date + players default, live-linked), Player A/B names, privacy selector (Private default / Unlisted / Public) | new `screens/match_setup_screen.dart` |
| Result screen | Brand refresh; title & privacy come from Match Setup | `screens/analysis_screen.dart` |
| Upload | Accept `private`/`unlisted`/`public` instead of private-only | `services/youtube_upload.dart` |
| Background processing | Keep analysis running when backgrounded | Android (foreground service), iOS (BGProcessingTask + assertion), desktop (off-thread + no-sleep) |

**Not in scope:** the rally-detection / dead-time engine and any other business
logic — unchanged.

---

## 8. Desktop inference performance

Notes on what actually makes the desktop pipeline fast, measured rather than
assumed. Recorded here because two plausible-sounding optimisations do **not**
work on this path, and both are the kind of idea that comes back around.

### 8.1 Batching is the win (adopted)

The per-frame model calls each pay a fixed cost — Python-side preprocess, the
MPS dispatch, postprocess — that batching amortises. Measured on an M4 over
Data/21 (12,594 frames, 7:00 of 4K):

| Stage | Before | After | |
|---|---|---|---|
| telemetry (3 calls/frame) | 954.7s | 411.9s | 2.32x |
| far pose | 189.8s | 202.1s | unchanged (see 8.4) |
| walking pose | 428.6s | 179.8s | 2.38x |
| **total perception** | **1573.1s** | **793.8s** | **1.98x** |

That is 3.75x realtime down to 1.89x. Two things carried the gain, and neither
is the one you would guess:

- **The ROI player call**, 10.09 -> 3.25 ms/frame (3.1x). A 556x540 crop at
  imgsz=384 was almost entirely fixed cost. The two full-frame 960px calls are
  genuinely compute-bound and only gave back ~15%.
- **Threaded decode.** 4K read+resize is ~6.4-6.9 ms/frame of pure CPU against
  ~12.6 ms of GPU, and both passes were running them strictly one after the
  other. Overlapping is free and lossless: telemetry's wall time fell further
  than its CPU time (757 -> 637s user against 954 -> 412s real), and adding
  the same reader thread to the walking pass alone took it 248.3 -> 179.8s
  (1.38x) with bit-identical output.

Beware when timing any of this: `time.time()` counts machine sleep, and an
unattended run that sleeps mid-pass reports absurd figures (one such run
reported 11,113s and 21,746s for passes that actually take ~250s) *and* can
appear to change detection counts. Run long benchmarks under `caffeinate -i`
and include a same-config control arm — this pipeline is deterministic, so
two identical runs must produce bit-identical `.npz` output.

Batching does not change results: ultralytics only switches letterbox mode for
mixed-shape batches (`pre_transform`: `auto=same_shapes and ...`), and every
batch built here is shape-uniform. Verified on Data/21 — far-serve and
near-serve event times identical, final segments byte-identical.

### 8.2 CoreML export is SLOWER here (rejected)

The intuition — a compiled graph collapses the Python-side preprocess/NMS/
postprocess, as `ios_tracker` does at ~5 ms/frame on the ANE — does not
survive measurement on the desktop Python path. ms/frame, M4:

| call | torch B=1 | torch B=16 | CoreML |
|---|---|---|---|
| ball (960px) | 12.9 | **11.0** | 18.8 |
| player full (960px) | 11.9 | **9.7** | 20.3 |
| player ROI (crop) | 9.7 | **3.3** | 20.9 |

This is not just wrapper overhead. A raw `coremltools.predict` forward, with
ultralytics stripped out entirely, is still **8.3 ms** against 5.0 ms for
batched torch-MPS — the compiled graph is not faster before any bridge cost.
`compute_units` made no difference (ALL 8.35, CPU_AND_NE 8.32, CPU_AND_GPU
8.83), so the ANE is not being meaningfully engaged from Python.

Why `ios_tracker`'s ~5 ms does not transfer: that is native Swift handing an
MLMultiArray straight to the ANE. Through Python the input marshalling costs
more than the graph saves.

Two structural problems on top of the timings: CoreML exports are
**shape-fixed** (the ROI pass at imgsz=384 needs its own separate export from
the 960px passes) and **batch-fixed** — mutually exclusive with the change in
8.1 that did work.

Accuracy, for the record, was fine (99.9% of torch detections matched within
5px, median centre shift 0.13px). Speed is the reason to say no.

### 8.3 Decoupled player striding (rejected)

Running the two player calls every 2nd frame while keeping the ball call on
every frame — motivated by the player consumers all integrating over 0.5-3s
while every ball consumer is detection-starved. Measured: total 862 -> 787s,
**only 1.10x**, because stage 1's wall clock is set by the ball call plus
decode, which the reader thread already overlapped.

For that 10% it changed the output: far serves 7 -> 6, near serves 8 -> 9
(a spurious detection at t=340.17s, p=0.982 — not a borderline flip),
segments 6 -> 8, footage kept 125.0s -> 160.7s (+29%). `all_balls` was
identical on 12,594/12,594 frames, isolating the cause entirely to the player
fields.

The mechanism is worth remembering: **zero-order hold is the wrong
interpolator for a differentiated signal.** `anya_near_serve` takes a second
derivative of the box aspect ratio, and a staircase has more curvature than
the smooth original — so holding *inflated* jerk rather than suppressing it.
A working stride-2 needs the consumers changed (rate-aware
`jerk_min_samples` / `ratio_smooth_n`, linear interpolation instead of hold),
not just the extractor.

### 8.4 Why the far-pose pass is not batched

`extract_far_pose` crops `fpr` + padding, giving **3,323 distinct crop shapes
across 7,302 frames** (mean consecutive equal-shape run: 1.04). A mixed-shape
batch flips ultralytics' letterbox mode, which would shift the very keypoints
the hand-raise gate reads. Making it batchable requires a fixed-size crop
window — a change to detection behaviour, needing its own validation.

That validation is 8.5, which does exactly this and measures the consequences.

### 8.5 The far fast path (`anya_far_telemetry`)

Stages 1 and 2 exist to serve three consumers; `anya_far_serve` is one of
them and reads only `fpr`, `fprw`, `all_balls` and the pixels inside `fpr`.
`anya_far_telemetry` serves that consumer alone, writing both a
schema-compatible telemetry and a v2 pose cache in one pass, so
`detect_far_serves` runs against it unmodified. Four levers, in descending
order of what they actually bought:

1. **A native-resolution band proxy.** The band around the far baseline is
   ~12% of a 4K frame, and both far passes read it instead of the source. No
   downscale — these are the same pixels the full pass crops out.
2. **Pose at imgsz 320, batched.** The crop is ~110x160, so ultralytics'
   default 640 was upscaling it fourfold and paying for the pixels. Halves
   the pass and the raise signal comes out *cleaner*, not coarser.
3. **Far player at 5 fps.** Arming is a 1 s stationarity test; it does not
   need 30 samples a second to decide someone is standing still.
4. **Ball at 10 fps, gated on whether a point could be open.**

Measured over the 13 ground-truthed clips (289,665 frames), against a
baseline of 48.7 ms/frame (Data/21: 411.9s telemetry + 202.1s far pose):

| | total | ms/frame |
|---|---|---|
| including one-time proxy builds | 2,719s | 9.4 |
| steady state | 1,997s | **6.9** (~7x) |

Of the steady state, pose is 57%, ball 34%, player 8%.

**What did NOT pay off, and why it is worth remembering:**

- **Gating pose on the armed windows** — the direct analogue of the near
  path's ready gate, which is where most of that path's win came from. Here
  the duty cycle is **91%**: the far player stands still for most of a match,
  so "armed" is almost always true. Kept because it bounds the pass and costs
  nothing, but it is not a lever.
- **Batching is not a correctness risk** (contra 8.4's caution): with
  shape-uniform crops, batch 1 and batch 16 produce *identical* raise-gate
  crossings. The letterbox concern is real only for mixed shapes.
- **Dropping the ball to 5 fps is free** on detections (0 recall change, +2
  FP across 83 serves) but is not the default, because precision — not speed
  — is this detector's weak point. 2.5 fps costs +6 FP.
- **The 5 fps player track is not why the keypoints differ.** Sampling the
  player at 60 fps instead reproduces the 5 fps result exactly.

**The keypoints are not the full pass's keypoints**, and no crop tuning made
them so. Raise-gate crossings on Data/23, where the full pass gives 44:

| variant | crossings |
|---|---|
| band proxy crf 14, crop resized to canonical | 72 |
| band proxy crf 14, crop padded into canvas (shipped) | 67 |
| band proxy crf 6 | 64 |
| source pixels, no proxy | 56 |
| source pixels, batch size 1 | 56 |

Roughly half the gap is the band's re-encode and the rest is crop geometry.
But the fast stream's signal is **different, not degraded**: it crosses more
often at a low threshold while holding on to true serves at a high one, where
the full pass starts losing them. Hence a separate preset
(`FarServeDetectorConfig.for_fast_path`), selected from the telemetry's
`meta.source` so the two streams cannot be scored with each other's
thresholds by accident.

**Trap — do not fit thresholds on absolute corpus F1.** Clip 25's 10 far
serves are invisible to *both* extractors (0/10 either way; the raise ratio
there peaks at p99=0.18 against a working clip's 0.88). A sweep scored on
corpus totals therefore rewards whichever setting simply detects less, and it
picked a preset that was mid-table in the comparison that matters. Score the
fast path against the full pass clip for clip.

**Accuracy.** Both extractors, 10 clips, 77 ground-truthed far serves:

| clip | GT | baseline | fast @ 0.20/0.65 |
|---|---|---|---|
| 21 | 1 | 0, 8 FP | 0, 5 FP |
| 22 | 4 | 3, 2 FP | 3, 2 FP |
| 23 | 15 | 15, 5 FP | 15, 3 FP |
| 24 | 12 | 12, 1 FP | 12, 2 FP |
| 25 | 10 | 0, 4 FP | 0, 2 FP |
| 26 | 11 | **1**, 7 FP | **11**, 8 FP |
| 36 | 9 | 8, 1 FP | 9, 1 FP |
| 40 | 13 | **2**, 3 FP | **3**, 3 FP |
| 43 | 0 | 0, 3 FP | 0, 0 FP |
| 50 | 2 | 0, 3 FP | 1, 3 FP |
| **total** | **77** | **41 (53%), 37 FP** | **54 (70%), 29 FP** |

The fast path is better on both axes. A recall-leaning alternative sits one
constant away — `FAST_RAISE_RATIO = 0.55` gives 56/77 at 38 FP, i.e. two more
serves for nine more false positives.

**The headline number here is not the fast path, it is the baseline.** The far
detector's reputation rests on Data/23, where it scores 15/15 — and 15 of its
41 corpus-wide detections are that one clip. On clips 26 and 40 the full pass
finds 1 of 11 and 2 of 13. Whatever the fast path is doing differently
(canonicalised crops, a re-encoded band, imgsz 320) recovers most of clip 26
outright, which says the full pass's keypoints are not the better ones in any
absolute sense — they are just the ones the current thresholds were fitted to.
Treat far-serve recall on a new clip as an open question, not a solved one.
