"""
label_ball.py
=============
Interactive ball labeler for tuning ``ball_detector.py``.

Steps through the video every ``--stride`` frames (default 3).  The cached YOLO
detections (from ``cache_detections.py``) are drawn as small dots so you can see
the candidate balls; a magnifier inset near the cursor makes the (few-pixel) ball
easy to place on a 4K frame.

Per visited frame you do exactly one of:
  * left-click on the ball  -> label its position (snaps to the nearest cached
                               detection within a few px), then auto-advance
  * press 'x'               -> mark NO BALL / dead time (trace should be empty
                               here), then auto-advance
  * press SPACE / 'd'       -> skip (leave unlabeled/unknown — not scored)

Labels are keyed by the true video frame index and saved to
``<stem>_labels.json``:  {frame: [x, y]}  for a ball, {frame: null} for no-ball.
Unlabeled frames are simply absent (the tuner does not score them).

Keys
----
  left-click  label ball at cursor (snap) + advance
  x           no-ball / dead time + advance
  space / d   skip forward (no change)
  a           back one label frame
  z / u       undo this frame's label
  f           toggle fullscreen
  m           toggle the magnifier on/off
  +/-         magnifier zoom
  s           save now
  q / ESC     save and quit

    python pipeline/ball_tuning/label_ball.py /Volumes/Anya/Data/69/snippet_1min.mp4
    python pipeline/ball_tuning/label_ball.py <video> [stride] [--fullscreen] [--width N]
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np


def _stem_path(video_path: str, suffix: str) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{stem}_{suffix}")


class BallLabeler:
    def __init__(self, video_path: str, stride: int = 3,
                 disp_max_w: int = 1600, snap_px: float = 30.0,
                 fullscreen: bool = False):
        self.video_path = video_path
        self.stride = max(1, int(stride))
        self.snap_px = float(snap_px)
        self.fullscreen = bool(fullscreen)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open {video_path}")
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Display scale so a 4K frame fits on screen; clicks are mapped back up.
        self.disp_scale = min(1.0, disp_max_w / float(self.width))
        self.disp_w = int(round(self.width * self.disp_scale))
        self.disp_h = int(round(self.height * self.disp_scale))

        # Per-frame cached detections (full-res px), used for dots + click snap.
        self.dets = self._load_dets()

        # Labels: {frame_idx: [x, y]} for a ball, {frame_idx: None} for no-ball.
        self.labels_path = _stem_path(video_path, "labels.json")
        self.labels = self._load_labels()

        # Visit list: every stride-th frame.
        self.frames = list(range(0, self.n_frames, self.stride))
        self.pos = 0  # index into self.frames
        # Resume at the first unlabeled frame for convenience.
        for i, f in enumerate(self.frames):
            if f not in self.labels:
                self.pos = i
                break

        # Magnifier: a small, fixed-size inset (constant on-screen footprint).
        # Higher zoom just samples a smaller source region — the box stays small.
        self.mag_inset = 190   # on-screen size of the inset, px (square)
        self.mag_zoom = 5      # magnification factor (+/- to change)
        self.show_mag = True   # 'm' toggles it off to see the whole court
        self.cursor = (self.disp_w // 2, self.disp_h // 2)  # display px
        self._cur_frame = None
        self._cur_idx = -1

    # ---- io -----------------------------------------------------------
    def _load_dets(self):
        p = _stem_path(self.video_path, "dets.json")
        if not os.path.isfile(p):
            print(f"[WARN] no detection cache at {p}; dots/snap disabled. "
                  f"Run cache_detections.py first for best results.")
            return {}
        with open(p) as f:
            data = json.load(f)
        return {i: d for i, d in enumerate(data["dets"])}

    def _load_labels(self):
        if not os.path.isfile(self.labels_path):
            return {}
        with open(self.labels_path) as f:
            raw = json.load(f)
        return {int(k): (tuple(v) if v is not None else None)
                for k, v in raw.items()}

    def save(self):
        out = {str(k): (list(v) if v is not None else None)
               for k, v in sorted(self.labels.items())}
        with open(self.labels_path, "w") as f:
            json.dump(out, f, indent=0)
        n_ball = sum(1 for v in self.labels.values() if v is not None)
        n_none = sum(1 for v in self.labels.values() if v is None)
        print(f"[INFO] Saved {len(self.labels)} labels "
              f"({n_ball} ball, {n_none} no-ball) -> {self.labels_path}")

    # ---- frame access -------------------------------------------------
    def _frame(self, idx: int):
        if idx == self._cur_idx and self._cur_frame is not None:
            return self._cur_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = self.cap.read()
        if not ok:
            fr = np.zeros((self.height, self.width, 3), np.uint8)
        self._cur_frame, self._cur_idx = fr, idx
        return fr

    # ---- interaction --------------------------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            fx, fy = x / self.disp_scale, y / self.disp_scale
            fx, fy = self._snap(fx, fy)
            self.labels[self.frames[self.pos]] = (round(fx, 2), round(fy, 2))
            self._advance(+1)

    def _snap(self, fx, fy):
        cand = self.dets.get(self.frames[self.pos], [])
        best, bd = None, self.snap_px
        for d in cand:
            dx, dy = d[0] - fx, d[1] - fy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= bd:
                bd, best = dist, (d[0], d[1])
        return best if best else (fx, fy)

    def _advance(self, step):
        self.pos = max(0, min(len(self.frames) - 1, self.pos + step))

    # ---- rendering ----------------------------------------------------
    def _render(self):
        f = self.frames[self.pos]
        frame = self._frame(f)
        disp = cv2.resize(frame, (self.disp_w, self.disp_h),
                          interpolation=cv2.INTER_AREA)

        # Cached detections as yellow dots (brighter = higher conf).
        for d in self.dets.get(f, []):
            px, py = int(d[0] * self.disp_scale), int(d[1] * self.disp_scale)
            c = int(80 + 175 * min(1.0, d[2] / 0.3))
            cv2.circle(disp, (px, py), 4, (0, c, c), 1)

        # Current label marker.
        lab = self.labels.get(f, "UNSET")
        if isinstance(lab, tuple):
            px, py = int(lab[0] * self.disp_scale), int(lab[1] * self.disp_scale)
            cv2.circle(disp, (px, py), 8, (0, 255, 0), 2)
            cv2.line(disp, (px - 12, py), (px + 12, py), (0, 255, 0), 1)
            cv2.line(disp, (px, py - 12), (px, py + 12), (0, 255, 0), 1)

        self._draw_magnifier(frame, disp)
        self._draw_hud(disp, f, lab)
        cv2.imshow(self.win, disp)

    def _draw_magnifier(self, frame, disp):
        if not self.show_mag:
            return
        cx, cy = self.cursor
        fx, fy = int(cx / self.disp_scale), int(cy / self.disp_scale)
        # Source region shrinks as zoom rises, so the on-screen inset is always
        # `mag_inset` px — small and constant, not a big overlay.
        src = max(8, int(round(self.mag_inset / self.mag_zoom)))
        h = src // 2
        x1, y1 = max(0, fx - h), max(0, fy - h)
        x2, y2 = min(self.width, fx + h), min(self.height, fy + h)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2]
        mag = cv2.resize(crop, (self.mag_inset, self.mag_inset),
                         interpolation=cv2.INTER_NEAREST)
        sx = self.mag_inset / float(x2 - x1)
        sy = self.mag_inset / float(y2 - y1)
        mh, mw = mag.shape[:2]
        # cross-hair at the true cursor position within the crop
        ccx = int((fx - x1) * sx)
        ccy = int((fy - y1) * sy)
        cv2.line(mag, (ccx, 0), (ccx, mh), (0, 0, 255), 1)
        cv2.line(mag, (0, ccy), (mw, ccy), (0, 0, 255), 1)
        # cached dets inside the crop
        for d in self.dets.get(self.frames[self.pos], []):
            if x1 <= d[0] <= x2 and y1 <= d[1] <= y2:
                px = int((d[0] - x1) * sx)
                py = int((d[1] - y1) * sy)
                cv2.circle(mag, (px, py), 5, (0, 255, 255), 1)
        # paste into the top-right corner of the display
        ph, pw = disp.shape[:2]
        ox, oy = pw - mw - 10, 10
        if ox > 0 and oy + mh < ph:
            disp[oy:oy + mh, ox:ox + mw] = mag
            cv2.rectangle(disp, (ox, oy), (ox + mw, oy + mh), (255, 255, 255), 1)

    def _draw_hud(self, disp, f, lab):
        done = len(self.labels)
        total = len(self.frames)
        state = ("NO-BALL" if lab is None else
                 ("BALL" if isinstance(lab, tuple) else "unset"))
        col = ((0, 165, 255) if lab is None else
               ((0, 255, 0) if isinstance(lab, tuple) else (200, 200, 200)))
        t = f / max(1.0, self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        lines = [
            f"frame {f}  ({self.pos + 1}/{total})  t={t:5.2f}s   [{state}]",
            f"labeled {done}   click=ball  x=no-ball  space=skip  a=back  z=undo"
            f"   f=fullscreen  m=magnifier  +/-=zoom  q=quit",
        ]
        y = 26
        for i, ln in enumerate(lines):
            cv2.putText(disp, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3)
            cv2.putText(disp, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        col if i == 0 else (255, 255, 255), 1)
            y += 26

    # ---- loop ---------------------------------------------------------
    def _apply_fullscreen(self):
        # WINDOW_NORMAL lets us stretch to fill the screen; OpenCV maps mouse
        # coords back to the image space, so click->full-res math is unchanged.
        cv2.setWindowProperty(
            self.win, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        if not self.fullscreen:
            cv2.resizeWindow(self.win, self.disp_w, self.disp_h)

    def run(self):
        self.win = "Ball Labeler"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.disp_w, self.disp_h)
        cv2.setMouseCallback(self.win, self._on_mouse)
        self._apply_fullscreen()
        print(__doc__)
        while True:
            self._render()
            k = cv2.waitKey(20) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('x'):
                self.labels[self.frames[self.pos]] = None
                self._advance(+1)
            elif k in (ord('d'), ord(' ')):
                self._advance(+1)
            elif k == ord('a'):
                self._advance(-1)
            elif k in (ord('z'), ord('u')):
                self.labels.pop(self.frames[self.pos], None)
            elif k == ord('s'):
                self.save()
            elif k in (ord('+'), ord('=')):
                self.mag_zoom = min(12, self.mag_zoom + 1)
            elif k == ord('-'):
                self.mag_zoom = max(2, self.mag_zoom - 1)
            elif k == ord('m'):
                self.show_mag = not self.show_mag
            elif k == ord('f'):
                self.fullscreen = not self.fullscreen
                self._apply_fullscreen()
        self.save()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("usage: python label_ball.py <video.mp4> [stride] "
              "[--fullscreen] [--width N]")
        sys.exit(1)
    video = args[0]
    fullscreen = "--fullscreen" in args
    disp_max_w = 1600
    if "--width" in args:
        disp_max_w = int(args[args.index("--width") + 1])
    stride = 3
    for a in args[1:]:
        if a.isdigit():
            stride = int(a)
            break
    BallLabeler(video, stride=stride, disp_max_w=disp_max_w,
                fullscreen=fullscreen).run()
