"""
audit_windows.py
===============
Stage 3: quick audit tool to review the auto-seeded ACTIVE/DEAD labels.

Default (video) mode plays the actual near-player footage for each window — the
clip is cropped to the near player over the 2-second window so you can see
whether they're hitting/split-stepping (ACTIVE) or walking/standing (DEAD).
Pass --skeleton to instead animate the normalized pose the model sees (drive-
free). By default it walks only the `boundary` windows (the ambiguous ones near
the transition); pass --all to review every window.

Keys:
    a = ACTIVE     d = DEAD        (sets label, advances)
    n / →  next    p / ←  prev     r = replay
    u = mark unaudited (revert to auto)
    s = save       q = save & quit

Writes labels.json in place: sets label + audited=True for reviewed windows.

Usage:
    python pipeline/audit_windows.py                 # video, boundary windows
    python pipeline/audit_windows.py --all
    python pipeline/audit_windows.py --skeleton
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_energy import _video_path, TELEMETRY_CACHE, ANALYSIS_SIZE
from make_windows import WIN_SEC

CANVAS_W, CANVAS_H = 700, 520
EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def draw_skeleton(row, w=360, h=480):
    img = np.full((h, w, 3), 24, np.uint8)
    if np.isnan(row[0]):
        cv2.putText(img, "no pose", (w//2-40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,200), 2)
        return img
    pts = [(int(row[3*k]*w), int(row[3*k+1]*h), row[3*k+2]) for k in range(17)]
    for a, b in EDGES:
        if pts[a][2] > 0.2 and pts[b][2] > 0.2:
            cv2.line(img, pts[a][:2], pts[b][:2], (90, 200, 160), 2)
    for px, py, c in pts:
        if c > 0.2:
            cv2.circle(img, (px, py), 4, (60, 220, 255), -1)
    return img


def fit_canvas(img):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((CANVAS_H, CANVAS_W, 3), 24, np.uint8)
    s = min(CANVAS_W / w, CANVAS_H / h)
    r = cv2.resize(img, (max(1, int(w*s)), max(1, int(h*s))))
    canvas = np.full((CANVAS_H, CANVAS_W, 3), 24, np.uint8)
    y0 = (CANVAS_H - r.shape[0]) // 2; x0 = (CANVAS_W - r.shape[1]) // 2
    canvas[y0:y0+r.shape[0], x0:x0+r.shape[1]] = r
    return canvas


class VideoSource:
    """Lazily opens each clip's video + telemetry and renders a stable
    near-player crop for a window's 2s of frames."""
    def __init__(self, data_root):
        self.data_root = data_root
        self.caps, self.tels = {}, {}

    def _clip(self, clip):
        if clip not in self.tels:
            cdir = os.path.join(self.data_root, clip)
            self.tels[clip] = json.load(open(os.path.join(cdir, TELEMETRY_CACHE)))
            self.caps[clip] = cv2.VideoCapture(_video_path(cdir))
        return self.tels[clip], self.caps[clip]

    def frames(self, clip, E):
        tel, cap = self._clip(clip)
        fps = tel["fps"]
        wf = max(2, round(fps * WIN_SEC))
        f_by_id = {}
        for r in tel["rallies"]:
            for k, v in r["frames"].items():
                f_by_id[int(k)] = v
        start = E - wf + 1
        # stable crop = union of near boxes across the window (padded)
        boxes = [f_by_id[f]["near_bbox"] for f in range(start, E+1)
                 if f in f_by_id and f_by_id[f]["near_bbox"]]
        if boxes:
            x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
            x2 = max(b[0]+b[2] for b in boxes); y2 = max(b[1]+b[3] for b in boxes)
            pw, ph = (x2-x1)*0.25, (y2-y1)*0.15
            rx1, ry1 = int(max(0, x1-pw)), int(max(0, y1-ph))
            rx2, ry2 = int(min(ANALYSIS_SIZE[0], x2+pw)), int(min(ANALYSIS_SIZE[1], y2+ph))
        else:
            rx1, ry1, rx2, ry2 = 0, 0, ANALYSIS_SIZE[0], ANALYSIS_SIZE[1]
        out = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(wf):
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.resize(fr, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
            crop = fr[ry1:ry2, rx1:rx2]
            out.append(fit_canvas(crop if crop.size else fr))
        return out


def parse_E(wid):
    return int(wid.rsplit("_e", 1)[1])


def main():
    ap = argparse.ArgumentParser(description="Audit active/dead window labels")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--windows", default="/Volumes/Anya/Data/windows.npz")
    ap.add_argument("--labels", default="/Volumes/Anya/Data/labels.json")
    ap.add_argument("--all", action="store_true", help="Review every window, not just boundary")
    ap.add_argument("--skeleton", action="store_true", help="Show pose skeleton instead of video")
    ap.add_argument("--fps", type=int, default=30, help="Playback fps")
    args = ap.parse_args()

    d = np.load(args.windows, allow_pickle=True)
    X, wid, clip, boundary = d["X"], d["wid"].astype(str), d["clip"].astype(str), d["boundary"]
    labels = json.load(open(args.labels))
    vs = None if args.skeleton else VideoSource(args.data_root)

    order = [i for i in range(len(wid)) if args.all or boundary[i] == 1]
    if not order:
        print("Nothing to audit (no boundary windows). Use --all."); return
    print(f"Auditing {len(order)} windows ({'skeleton' if args.skeleton else 'video'}). "
          f"a=active d=dead n/p=nav r=replay u=unaudit s=save q=quit")

    win = "audit active/dead"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    delay = int(1000 / max(1, args.fps))

    cache = {}
    def get_frames(gi):
        if gi in cache:
            return cache[gi]
        if args.skeleton:
            fr = [fit_canvas(draw_skeleton(X[gi][t])) for t in range(X[gi].shape[0])]
        else:
            try:
                fr = vs.frames(clip[gi], parse_E(wid[gi]))
            except Exception as e:
                print(f"[warn] video load failed for {wid[gi]}: {e} — skeleton fallback")
                fr = [fit_canvas(draw_skeleton(X[gi][t])) for t in range(X[gi].shape[0])]
        if not fr:
            fr = [np.full((CANVAS_H, CANVAS_W, 3), 24, np.uint8)]
        cache[gi] = fr
        return fr

    i = 0
    while True:
        gi = order[i]; w = wid[gi]; lab = labels[w]
        frames = get_frames(gi)
        k = 255
        for img0 in frames:
            img = img0.copy()
            tag = "ACTIVE" if lab["label"] == 1 else "DEAD"
            col = (90, 220, 120) if lab["label"] == 1 else (90, 120, 240)
            cv2.rectangle(img, (0,0), (CANVAS_W, 30), (0,0,0), -1)
            cv2.putText(img, f"{i+1}/{len(order)}  {w}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)
            cv2.putText(img, f"{tag}{'*' if lab['audited'] else ' (auto)'}", (CANVAS_W-150, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.imshow(win, img)
            k = cv2.waitKey(delay) & 0xFF
            if k != 255:
                break
        if k in (ord('a'), ord('d')):
            lab["label"] = 1 if k == ord('a') else 0; lab["audited"] = True
            i = min(i + 1, len(order) - 1)
        elif k == ord('u'):
            lab["label"] = lab["auto"]; lab["audited"] = False
        elif k in (ord('n'), 83, ord(' ')):
            i = min(i + 1, len(order) - 1)
        elif k in (ord('p'), 81):
            i = max(i - 1, 0)
        elif k == ord('r'):
            continue
        elif k == ord('s'):
            json.dump(labels, open(args.labels, "w"), indent=0); print("saved")
        elif k in (ord('q'), 27):
            break

    json.dump(labels, open(args.labels, "w"), indent=0)
    n_aud = sum(1 for v in labels.values() if v["audited"])
    cv2.destroyAllWindows()
    print(f"Saved {args.labels}  ({n_aud} audited)")


if __name__ == "__main__":
    main()
