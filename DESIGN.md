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
