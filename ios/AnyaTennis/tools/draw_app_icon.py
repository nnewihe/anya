# Anya Tennis app icon (placeholder until a designed one): a court seen from behind the baseline (the app's own
# tripod view) with a ball in flight. Drawn at 4x and downsampled.
# Usage: python tools/draw_app_icon.py AnyaTennis/Resources/Assets.xcassets/AppIcon.appiconset/AppIcon-1024.png
import sys, math
from PIL import Image, ImageDraw, ImageFilter
N = 4096; s = N / 1024
img = Image.new("RGB", (N, N))
d = ImageDraw.Draw(img)
# background: vertical gradient, deep navy -> court green
for y in range(N):
    t = y / N
    c = tuple(int(a + (b - a) * t) for a, b in zip((14, 32, 52), (18, 92, 64)))
    d.line([(0, y), (N, y)], fill=c)
# court in perspective (singles box + service line + centre line)
def P(x, y): return (x * s, y * s)
near_y, far_y = 900, 430
nl, nr, fl, fr = 150, 874, 372, 652
court = [P(nl, near_y), P(nr, near_y), P(fr, far_y), P(fl, far_y)]
d.polygon(court, fill=(36, 128, 84))
w = int(14 * s)
def lerp(a, b, t): return a + (b - a) * t
def edge(t):  # point on left/right sideline at depth t (0 near, 1 far)
    return (lerp(nl, fl, t), lerp(near_y, far_y, t)), (lerp(nr, fr, t), lerp(near_y, far_y, t))
for a, b in [((nl, near_y), (nr, near_y)), ((fl, far_y), (fr, far_y)),
             ((nl, near_y), (fl, far_y)), ((nr, near_y), (fr, far_y))]:
    d.line([P(*a), P(*b)], fill=(245, 248, 240), width=w)
# net (about 55% of the way back in screen space) and service lines
for t, width in [(0.30, w), (0.62, int(w * 0.8))]:
    l, r = edge(t)
    d.line([P(*l), P(*r)], fill=(245, 248, 240), width=width)
l1, r1 = edge(0.30); l2, r2 = edge(0.62)
d.line([P((l1[0] + r1[0]) / 2, l1[1]), P((l2[0] + r2[0]) / 2, l2[1])], fill=(245, 248, 240), width=int(w * 0.8))
ln, rn = edge(0.47)
d.line([P(*ln), P(*rn)], fill=(20, 40, 36), width=int(22 * s))   # net band
d.line([P(*ln), P(*rn)], fill=(235, 240, 235), width=int(6 * s))
# ball trail (arc), then the ball
cx, cy, r = 700, 300, 120
trail = Image.new("L", (N, N), 0); td = ImageDraw.Draw(trail)
for i in range(40):
    t = i / 39
    x = lerp(250, cx - 60, t); y = lerp(560, cy + 30, t) - math.sin(t * math.pi) * 120
    td.ellipse([P(x - 10 - 40 * t, y - 10 - 40 * t), P(x + 10 + 40 * t, y + 10 + 40 * t)], fill=int(25 + 70 * t))
trail = trail.filter(ImageFilter.GaussianBlur(18 * s))
img.paste((214, 240, 70), (0, 0), trail)
d = ImageDraw.Draw(img)
d.ellipse([P(cx - r, cy - r), P(cx + r, cy + r)], fill=(214, 240, 70))
# shading: darker lower-right crescent
shade = Image.new("L", (N, N), 0); sd = ImageDraw.Draw(shade)
sd.ellipse([P(cx - r, cy - r), P(cx + r, cy + r)], fill=70)
sd.ellipse([P(cx - r - 40, cy - r - 40), P(cx + r - 30, cy + r - 30)], fill=0)
shade = shade.filter(ImageFilter.GaussianBlur(10 * s))
img.paste((120, 140, 30), (0, 0), shade)
d = ImageDraw.Draw(img)
# seams: two arcs
sw = int(12 * s)
d.arc([P(cx - r - 150, cy - r + 10), P(cx - 30, cy + r - 10)], -48, 48, fill=(250, 252, 245), width=sw)
d.arc([P(cx + 30, cy - r + 10), P(cx + r + 150, cy + r - 10)], 132, 228, fill=(250, 252, 245), width=sw)
img = img.resize((1024, 1024), Image.LANCZOS)
img.save(sys.argv[1], optimize=True)
