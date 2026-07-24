"""
audit_windows.py
===============
Stage 3: quick audit tool to review the auto-seeded ACTIVE/DEAD labels. Animates
the near-player pose skeleton for each window (exactly what the model sees) and
lets you confirm or flip the label. By default it walks only the `boundary`
windows (the ambiguous ones near the transition); pass --all to review every
window.

Keys:
    a = ACTIVE     d = DEAD        (sets label, advances)
    n / →  next    p / ←  prev     r = replay
    u = mark unaudited (revert to auto)
    s = save       q = save & quit

Writes labels.json in place: sets label + audited=True for reviewed windows.

Usage:
    python pipeline/audit_windows.py
    python pipeline/audit_windows.py --all
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np

CANVAS = (360, 480)   # w, h
EDGES = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def draw_skeleton(row, w, h):
    """row = [51] normalized keypoints; returns a canvas image."""
    img = np.full((h, w, 3), 24, np.uint8)
    if np.isnan(row[0]):
        cv2.putText(img, "no pose", (w//2-40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,200), 2)
        return img
    pts = []
    for k in range(17):
        nx, ny, c = row[3*k], row[3*k+1], row[3*k+2]
        px, py = int(nx * w), int(ny * h)
        pts.append((px, py, c))
    for a, b in EDGES:
        if pts[a][2] > 0.2 and pts[b][2] > 0.2:
            cv2.line(img, pts[a][:2], pts[b][:2], (90, 200, 160), 2)
    for (px, py, c) in pts:
        if c > 0.2:
            cv2.circle(img, (px, py), 4, (60, 220, 255), -1)
    return img


def main():
    ap = argparse.ArgumentParser(description="Audit active/dead window labels via pose skeleton")
    ap.add_argument("--windows", default="/Volumes/Anya/Data/windows.npz")
    ap.add_argument("--labels", default="/Volumes/Anya/Data/labels.json")
    ap.add_argument("--all", action="store_true", help="Review every window, not just boundary")
    ap.add_argument("--fps", type=int, default=30, help="Playback fps")
    args = ap.parse_args()

    d = np.load(args.windows, allow_pickle=True)
    X, wid, boundary = d["X"], d["wid"].astype(str), d["boundary"]
    labels = json.load(open(args.labels))

    order = [i for i in range(len(wid)) if args.all or boundary[i] == 1]
    if not order:
        print("Nothing to audit (no boundary windows). Use --all."); return
    print(f"Auditing {len(order)} windows. Keys: a=active d=dead n/p=nav r=replay u=unaudit s=save q=quit")

    win = "audit active/dead"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    delay = int(1000 / max(1, args.fps))
    i = 0
    while True:
        gi = order[i]; w = wid[gi]; seq = X[gi]
        lab = labels[w]
        for t in range(seq.shape[0]):
            img = draw_skeleton(seq[t], *CANVAS)
            tag = "ACTIVE" if lab["label"] == 1 else "DEAD"
            col = (90, 220, 120) if lab["label"] == 1 else (90, 120, 240)
            cv2.putText(img, f"{i+1}/{len(order)}  {w}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
            cv2.putText(img, f"{tag}{'*' if lab['audited'] else ' (auto)'}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.imshow(win, img)
            k = cv2.waitKey(delay) & 0xFF
            if k != 255:
                break
        if k in (ord('a'), ord('d')):
            lab["label"] = 1 if k == ord('a') else 0; lab["audited"] = True
            i = min(i + 1, len(order) - 1)
        elif k in (ord('u'),):
            lab["label"] = lab["auto"]; lab["audited"] = False
        elif k in (ord('n'), 83, ord(' ')):
            i = min(i + 1, len(order) - 1)
        elif k in (ord('p'), 81):
            i = max(i - 1, 0)
        elif k == ord('r'):
            continue
        elif k in (ord('s'),):
            json.dump(labels, open(args.labels, "w"), indent=0); print("saved")
        elif k in (ord('q'), 27):
            break

    json.dump(labels, open(args.labels, "w"), indent=0)
    n_aud = sum(1 for v in labels.values() if v["audited"])
    cv2.destroyAllWindows()
    print(f"Saved {args.labels}  ({n_aud} audited)")


if __name__ == "__main__":
    main()
