# Tennis Point Detection — Architectural Redesign

## Problem with Current 3-State System

The existing state machine (`Waiting → Armed → Active`) is brittle because state drives detection. If a point is incorrectly classified as active when it's actually dead, the player can enter the ready/serve zone and no new serve is detected — one bad state transition corrupts all subsequent point detection.

**Root cause:** Using state to *gate* detection means errors compound forward in time.

---

## New Architecture: Detect First, Reason Second

Two independent, decoupled layers:

1. **Serve Detector** — stateless, continuous, emits serve candidates
2. **Point Segmenter** — takes serve timestamps, segments them into points

Detection drives state — not the other way around.

---

## Layer 1: Serve Detector (Stateless)

- Runs continuously on a sliding window of sensor data
- No awareness of current game state — no gating on "armed" or "active"
- Asks only: *"does this window look like a serve?"*
- Emits `(timestamp, confidence)` for every serve candidate
- Accepts false positives — filtering happens downstream

**Output:** Raw list of `(timestamp, confidence)` serve candidates.

---

## Layer 2: Point Segmenter

Takes the raw serve candidate list and reasons over it to find real serves and point boundaries.

### Step 1 — Filter Real Serves from Candidates

- Apply a **minimum inter-serve gap** (e.g. 10–15 seconds) — no real point resolves faster
- Cluster candidates that fall within a short window; take the **highest-confidence** one
- Apply a **refractory period** after each confirmed serve to suppress echoes

### Step 2 — Find Point End for Each Confirmed Serve

For each confirmed serve at time `T`, scan forward to find the point-end event:

```
Point end candidates = {
  next confirmed serve timestamp,   ← hard upper bound (guaranteed fallback)
  inactivity / silence signal,      ← ball stopped moving
  audio / crowd signal,             ← point resolution cue
  explicit dead-ball marker         ← if available from external source
}
```

Pick the highest-probability point-end event before the next serve.  
The next serve is always a safe fallback — even without a clean point-end signal, the point is guaranteed to have ended somewhere in `[serve_N, serve_N+1)`.

---

## Before vs. After

```
BEFORE:                              AFTER:
Waiting → Armed → Active             Serve Stream → Point Segments
      ↑_________↓                    (t1, t2, t3, t4 ...)
  (state gates detection)                  ↓
                                     Segment [t1 → t2)
                                     Segment [t2 → t3)
                                     Segment [t3 → t4)
                                 (detection defines segments)
```

---

## Edge Case Handling

| Scenario | How it's handled |
|---|---|
| Double fault / let (two quick serves) | Clustering groups them; highest-confidence one wins |
| Long rallies | No issue — scan forward until next serve with no time cap |
| False positive serve mid-rally | Filtered by refractory period and minimum inter-serve gap |
| Missed serve detection | That point boundary is lost, but future points are unaffected — segments are independent |

---

## Implementation Notes

- The serve detector should be tunable independently of segmentation logic
- Confidence scores enable offline threshold tuning — replay a match, adjust, re-segment without rerunning detection
- Consider logging all raw candidates (not just confirmed serves) for debugging and model improvement
- The refractory period and minimum gap values should be configurable constants, not hardcoded

---

## Key Principle

> A bad point-end classification should never corrupt the next point's detection.  
> Each point segment must be independently recoverable.