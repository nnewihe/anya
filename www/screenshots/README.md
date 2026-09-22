# Screenshots used by `about.html`

The five images `../about.html` shows:

| file | step | what it shows |
|---|---|---|
| `court-framing.jpg` | 1 | a frame of a recorded match, whole court in view |
| `highlight-reel.png` | 2 | the Highlight Reel tab from the tab bar down: input path, output path, checkbox |
| `court-corners.jpg` | 3 | the same frame with the four corners marked BL/BR/TR/TL |
| `highlight-reel-done.png` | 4 | the end of a run: full progress bar, "Done — 18 rallies", the rally-count badge, Open Folder |
| `scoreboard.jpg` | — | the Scoreboard tab mid-review — video frame, live score panel, the point table, REVIEW POINTS controls |

`highlight-reel.png` and `highlight-reel-done.png` are the top and bottom
crops of one window screenshot — the empty middle of the window is dead space
on a web page. From a fresh grab, with `k = W / 2000`, step 2 is
`crop((0, 150*k, 1955*k, 615*k))` and step 4 is `crop((0, 1010*k, W, H))`.
`sips --cropOffset` silently ignores the offset; use PIL.

The two court frames are 1920x1080, re-extracted from
`/Users/tennis/Documents/match_play/data/langmead_1min.mov` at t≈5.0s — the
same frame as `spikes/fixtures/langmead/ref_frame.png`, which is a 960px
downscale and too soft to display at 1180px. The crosshairs were redrawn at
that scale from the marker centres measured in
`spikes/fixtures/langmead/corners_annotated.png` (BL 94,418 · BR 940,392 ·
TR 563,257 · TL 413,262, all doubled). Swap them for a court you would rather
advertise; nothing but `about.html` refers to them.

All five carry the class `shot-wide` in `about.html`: they break out of the
760 px prose column and render at one shared width, up to 1180 px, because at
column width the app windows' own UI text falls under 8 px. Keep any
replacement at least ~1900 px wide so none of them is upscaled, and under ~600 KB each so the page stays
quick — `scoreboard` is a JPEG (q88) because a 2.5 MB PNG of a photo is not
worth serving.

`about.html` hides a screenshot figure entirely if its file is missing, so a
missing shot degrades to no image rather than a broken one.
