"""
mark_transitions.py
==================
Efficient relabeling: instead of labeling ~800 near-duplicate windows, mark the
single active->dead PLAYER transition frame per rally by scrubbing the actual
near-player footage around each point-end. make_windows.py --transitions then
derives every window's label from these frames (player-based, not ball-based).

Per rally you see the near-player crop over [gt_end - PRE, gt_end + POST]. Scrub
to the frame where the player stops active play (finishes the shot / starts
walking) and mark it.

Keys:
    . / →   next frame        , / ←   prev frame
    space   play/pause
    m       set transition = current frame
    n       next rally        p       prev rally
    r       reset to GT end    s       save        q       save & quit

Writes transitions.json: {"<clip>_r<rally>": transition_frame}.

Usage:
    python pipeline/mark_transitions.py
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_energy import _video_path, _near_rallies, TELEMETRY_CACHE, ANALYSIS_SIZE

CANVAS_W, CANVAS_H = 760, 560
PRE_SEC, POST_SEC = 2.0, 3.0


def fit_canvas(img):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((CANVAS_H, CANVAS_W, 3), 24, np.uint8)
    s = min(CANVAS_W / w, CANVAS_H / h)
    r = cv2.resize(img, (max(1, int(w*s)), max(1, int(h*s))))
    c = np.full((CANVAS_H, CANVAS_W, 3), 24, np.uint8)
    y0, x0 = (CANVAS_H - r.shape[0]) // 2, (CANVAS_W - r.shape[1]) // 2
    c[y0:y0+r.shape[0], x0:x0+r.shape[1]] = r
    return c


def load_region(cap, f_by_id, f0, f1):
    boxes = [f_by_id[f]["near_bbox"] for f in range(f0, f1+1)
             if f in f_by_id and f_by_id[f]["near_bbox"]]
    if boxes:
        x1 = min(b[0] for b in boxes); y1 = min(b[1] for b in boxes)
        x2 = max(b[0]+b[2] for b in boxes); y2 = max(b[1]+b[3] for b in boxes)
        pw, ph = (x2-x1)*0.25, (y2-y1)*0.15
        rx1, ry1 = int(max(0, x1-pw)), int(max(0, y1-ph))
        rx2, ry2 = int(min(ANALYSIS_SIZE[0], x2+pw)), int(min(ANALYSIS_SIZE[1], y2+ph))
    else:
        rx1, ry1, rx2, ry2 = 0, 0, *ANALYSIS_SIZE
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for _ in range(f1 - f0 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        fr = cv2.resize(fr, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        crop = fr[ry1:ry2, rx1:rx2]
        frames.append(fit_canvas(crop if crop.size else fr))
    return frames


def main():
    ap = argparse.ArgumentParser(description="Mark per-rally active->dead player transition frames")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--out", default="/Volumes/Anya/Data/transitions.json")
    args = ap.parse_args()

    clips = [d for d in sorted(os.listdir(args.data_root))
             if os.path.isfile(os.path.join(args.data_root, d, TELEMETRY_CACHE)) and _near_rallies(os.path.join(args.data_root, d))]

    trans = json.load(open(args.out)) if os.path.isfile(args.out) else {}

    # Flatten to a rally list
    items = []
    for name in clips:
        cdir = os.path.join(args.data_root, name)
        tel = json.load(open(os.path.join(cdir, TELEMETRY_CACHE)))
        f_by_id = {}
        for r in tel["rallies"]:
            for k, v in r["frames"].items():
                f_by_id[int(k)] = v
        fps = tel["rallies"] and tel.get("fps", 30)
        for ri, r in enumerate(tel["rallies"]):
            items.append((name, ri, r["start"], r["end"], r["span_end"], cdir, f_by_id, tel.get("fps", 30)))

    print(f"{len(items)} rallies across {len(clips)} clips. "
          ". / ,=step  space=play  m=mark  n/p=rally  r=reset  s=save  q=quit")

    win = "mark transition"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    caps = {}
    i = 0
    frames = None; f0 = 0; idx = 0
    def load(i):
        name, ri, start, end, span_end, cdir, f_by_id, fps = items[i]
        if name not in caps:
            caps[name] = cv2.VideoCapture(_video_path(cdir))
        pre, post = round(fps*PRE_SEC), round(fps*POST_SEC)
        a = max(start, end - pre); b = min(span_end, end + post)
        fr = load_region(caps[name], f_by_id, a, b)
        key = f"{name}_r{ri}"
        t = trans.get(key, end)
        return fr, a, max(0, min(len(fr)-1, t - a)), end
    frames, f0, idx, gt_end = load(i)
    playing = False

    while True:
        name, ri = items[i][0], items[i][1]
        img = frames[idx].copy() if frames else np.full((CANVAS_H, CANVAS_W, 3), 24, np.uint8)
        cur = f0 + idx
        key = f"{name}_r{ri}"
        T = trans.get(key, gt_end)
        cv2.rectangle(img, (0,0), (CANVAS_W, 30), (0,0,0), -1)
        cv2.putText(img, f"{i+1}/{len(items)} clip {name} r{ri}  frame {cur}", (8,21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
        rel = cur - gt_end
        cv2.putText(img, f"GT_end {gt_end} ({rel:+d})   T={T}", (8, CANVAS_H-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120,220,255) if cur==T else (200,200,200), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(30 if playing else 0) & 0xFF
        if playing:
            idx = min(idx+1, len(frames)-1)
            if idx == len(frames)-1:
                playing = False
        if k in (ord('.'), 83):          # . or right arrow: next frame
            idx = min(idx+1, len(frames)-1)
        elif k in (ord(','), 81):        # , or left arrow: prev frame
            idx = max(idx-1, 0)
        elif k == ord(' '):
            playing = not playing
        elif k == ord('m'):
            trans[key] = cur
        elif k == ord('r'):
            trans[key] = gt_end; idx = max(0, min(len(frames)-1, gt_end - f0))
        elif k == ord('n'):
            i = min(i+1, len(items)-1); frames, f0, idx, gt_end = load(i); playing = False
        elif k == ord('p'):
            i = max(i-1, 0); frames, f0, idx, gt_end = load(i); playing = False
        elif k == ord('s'):
            json.dump(trans, open(args.out, "w"), indent=0); print(f"saved {len(trans)}")
        elif k in (ord('q'), 27):
            break

    json.dump(trans, open(args.out, "w"), indent=0)
    cv2.destroyAllWindows()
    print(f"Saved {args.out}  ({len(trans)} transitions marked)")


if __name__ == "__main__":
    main()
