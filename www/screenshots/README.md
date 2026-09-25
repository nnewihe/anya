# Screenshots used by `about.html`

The five images `../about.html` shows:

| file | card | what it shows |
|---|---|---|
| `court-framing.jpg` | 1 | a frame of a recorded match, whole court in view |
| `highlight-reel.png` | 2 | the Highlight Reel tab from the tab bar down: input path, output path, checkbox |
| `court-corners.jpg` | 3 | the same frame with the four corners marked BL/BR/TR/TL |
| `highlight-reel-done.png` | 4 | the end of a run: full progress bar, "Done — 18 rallies", the rally-count badge, Open Folder |
| `scoreboard.jpg` | 5 (optional) | the Scoreboard tab mid-review — video frame, live score panel, the point table, REVIEW POINTS controls |

`highlight-reel.png` and `highlight-reel-done.png` are the top and bottom
crops of one window screenshot — the empty middle of the window is dead space
on a web page. From a fresh grab, with `k = W / 2000`, step 2 is
`crop((0, 150*k, 1955*k, 615*k))` and step 4 is `crop((0, 1010*k, W, H))`.

Those strips are then tightened further, because at card width their UI text
was unreadable. Coordinates below are in the strip scaled to 2000 px wide:

- **step 2**: keep x 0–730 and x 1772–1975 side by side. The cut runs through
  the empty middle of both path fields, so the seam doesn't show and the Browse
  buttons stay.
- **step 4**: keep the centre, x 740–1256 / y 30–255 (button, progress bar,
  "Done — 18 rallies"), and set the badge row, x 1560–1890 / y 280–342
  (18 RALLIES, Open Folder), centred beneath it. That row moves from the
  window's right edge to the middle.
`sips --cropOffset` silently ignores the offset; use PIL.

The two court frames are 1920x1080, re-extracted from
`/Users/tennis/Documents/match_play/data/langmead_1min.mov` at t≈5.0s — the
same frame as `spikes/fixtures/langmead/ref_frame.png`, which is a 960px
downscale and too soft to display at 1180px. The crosshairs were redrawn at
that scale from the marker centres measured in
`spikes/fixtures/langmead/corners_annotated.png` (BL 94,418 · BR 940,392 ·
TR 563,257 · TL 413,262, all doubled). Swap them for a court you would rather
advertise; nothing but `about.html` refers to them.

Each one sits at the top of a slideshow card in a 16:9 frame, letterboxed
(`object-fit: contain`) so none is ever cropped. Keep any replacement at least
~1900 px wide and under ~600 KB so the page stays quick. `scoreboard` is a
JPEG (q88) because a 2.5 MB PNG of a photo is not worth serving.

`about.html` hides a card's image if its file is missing, so the card degrades
to text only rather than a broken image.
