#!/usr/bin/env python3
"""Label the ball's position frame by frame, to score the tracker against truth.

The point of this tool is speed. A tennis ball is ~14-28 px in a 1920-wide
frame, so clicking it accurately is slow and error-prone. Instead we run the
real Core ML detector at a low threshold, draw its candidates as numbered
circles, and let you accept one with a digit key. You only click manually when
the detector missed the ball entirely. Frames you never touch stay unlabelled
and are skipped by the scorer, so partial passes are fine.

    python3 label_ball.py --video /tmp/clip.mov --out labels.json --start 900 --end 1150

Keys
    0-9        accept that candidate as the ball (auto-advances)
    click      label the ball at that point (auto-advances)
    n          no ball visible in this frame (auto-advances)
    u          clear this frame's label
    d / right  next frame            a / left   previous frame
    ] / [      jump 10 frames
    z          toggle the magnifier
    s          save          q          save and quit

Labels are written as {"<frame>": {"x":.., "y":..}} in SOURCE pixels, or null
for "no ball visible". Missing key == not labelled.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

CAND_CONF = 0.02   # deliberately low: we want to offer even faint balls
MAX_BOX_PX = 45    # mirrors BallDetector.defaultMaxBoxPx
W, H = 960, 544    # model input


def letterbox(frame):
    sh, sw = frame.shape[:2]
    r = min(W / sw, H / sh)
    nw, nh = round(sw * r), round(sh * r)
    px, py = (W - nw) // 2, (H - nh) // 2
    canvas = np.full((H, W, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, r, px, py


def detect_candidates(model, frame):
    """Model candidates for one frame, in SOURCE px, best confidence first."""
    from PIL import Image
    canvas, r, px, py = letterbox(frame)
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    det = np.array(model.predict({"image": img})["detections"]).reshape(5, -1)
    out = []
    for j in np.where(det[4] > CAND_CONF)[0]:
        cx, cy, bw, bh, cf = (float(v) for v in det[:, j])
        if max(bw, bh) > MAX_BOX_PX:
            continue
        out.append(((cx - px) / r, (cy - py) / r, cf))
    out.sort(key=lambda c: -c[2])
    # Collapse near-duplicates so the digit keys map to distinct objects.
    keep = []
    for c in out:
        if all((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2 > 20 ** 2 for k in keep):
            keep.append(c)
        if len(keep) == 10:
            break
    return keep


def cache_candidates(video, start, end, cache_path, step=1):
    key = {"video": video, "start": start, "end": end, "step": step}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        if all(data.get(k) == v for k, v in key.items()):
            print(f"using cached candidates: {cache_path}")
            return {int(k): v for k, v in data["cands"].items()}

    import coremltools as ct
    here = os.path.dirname(os.path.abspath(__file__))
    mlpkg = os.path.join(here, "..", "BallTracker", "Resources", "ball_best.mlpackage")
    print("loading model (ANE)…")
    model = ct.models.MLModel(mlpkg, compute_units=ct.ComputeUnit.ALL)

    # Sequential read and skip: seeking per frame is far slower on 4K.
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    cands = {}
    f = start
    while f <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if (f - start) % step == 0:          # only frames the labeller will land on
            cands[f] = detect_candidates(model, frame)
            if len(cands) % 20 == 0:
                pct = 100 * (f - start) / max(end - start, 1)
                print(f"\r  detecting candidates… {pct:3.0f}%", end="", flush=True)
        f += 1
    cap.release()
    print("\r  detecting candidates… done   ")
    with open(cache_path, "w") as f_:
        json.dump({**key, "cands": {str(k): v for k, v in cands.items()}}, f_)
    return cands


class Labeler:
    def __init__(self, video, out, start, end, step=1):
        self.video = video
        self.out = out
        self.start, self.end = start, end
        # Label every `step`-th frame. A 120 fps clip holds the same tennis as a
        # 30 fps one but would cost 4x the keystrokes; the scorer skips frames
        # that were never labelled, so sampling costs nothing but effort saved.
        self.step = max(1, step)
        self.cap = cv2.VideoCapture(video)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.end = min(self.end, self.total - 1)
        self.cands = cache_candidates(video, self.start, self.end,
                                      out + ".cands.json", self.step)
        self.labels = {}
        if os.path.exists(out):
            with open(out) as f:
                d = json.load(f)
            self.labels = {int(k): v for k, v in d.get("labels", {}).items()}
            print(f"loaded {len(self.labels)} existing labels from {out}")
        self.f = self.start
        self.frame = None
        self.mouse = (0, 0)
        self.magnify = True
        self.scale = 1.0
        self.dirty = False

    def read(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.f)
        ok, fr = self.cap.read()
        if ok:
            self.frame = fr

    def save(self):
        with open(self.out, "w") as f:
            json.dump({"video": self.video,
                       "labels": {str(k): v for k, v in sorted(self.labels.items())}},
                      f, indent=1)
        self.dirty = False
        n_ball = sum(1 for v in self.labels.values() if v)
        print(f"saved {len(self.labels)} labels ({n_ball} with ball) -> {self.out}")

    def on_mouse(self, event, x, y, flags, param):
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.labels[self.f] = {"x": x / self.scale, "y": y / self.scale}
            self.dirty = True
            self.advance(1)

    def advance(self, d):
        self.f = max(self.start, min(self.end, self.f + d * self.step))
        self.read()

    def render(self):
        disp = self.frame.copy()
        for i, (cx, cy, cf) in enumerate(self.cands.get(self.f, [])):
            p = (int(cx), int(cy))
            cv2.circle(disp, p, 16, (0, 200, 255), 2)
            cv2.putText(disp, f"{i}", (p[0] + 18, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)
            cv2.putText(disp, f"{cf:.2f}", (p[0] + 18, p[1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        lab = self.labels.get(self.f, "unset")
        if isinstance(lab, dict):
            cv2.drawMarker(disp, (int(lab["x"]), int(lab["y"])), (0, 0, 255),
                           cv2.MARKER_CROSS, 40, 3)
        txt = ("NO BALL" if lab is None else
               "unlabelled" if lab == "unset" else
               f"ball @ {lab['x']:.0f},{lab['y']:.0f}")
        done = len(self.labels)
        total = self.end - self.start + 1
        cv2.putText(disp, f"f{self.f}  [{self.start}-{self.end}]  {txt}   labelled {done}/{total}",
                    (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        cv2.putText(disp, "0-9 pick  click place  n none  u clear  a/d prev/next  [ ] +-10  z zoom  s save  q quit",
                    (20, disp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        sh, sw = disp.shape[:2]
        self.scale = min(1600 / sw, 900 / sh, 1.0)
        view = cv2.resize(disp, (int(sw * self.scale), int(sh * self.scale)))

        # Magnifier: the ball is small, so show a 4x inset around the cursor.
        if self.magnify:
            mx, my = int(self.mouse[0] / self.scale), int(self.mouse[1] / self.scale)
            r = 60
            x1, y1 = max(0, mx - r), max(0, my - r)
            x2, y2 = min(sw, mx + r), min(sh, my + r)
            if x2 > x1 and y2 > y1:
                patch = cv2.resize(disp[y1:y2, x1:x2], None, fx=4, fy=4,
                                   interpolation=cv2.INTER_NEAREST)
                ph, pw = patch.shape[:2]
                ph, pw = min(ph, 260), min(pw, 260)
                view[10:10 + ph, view.shape[1] - pw - 10:view.shape[1] - 10] = patch[:ph, :pw]
                cv2.rectangle(view, (view.shape[1] - pw - 10, 10),
                              (view.shape[1] - 10, 10 + ph), (0, 255, 255), 2)
        return view

    def run(self):
        cv2.namedWindow("label", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("label", self.on_mouse)
        self.read()
        while True:
            cv2.imshow("label", self.render())
            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if ord("0") <= k <= ord("9"):
                i = k - ord("0")
                cs = self.cands.get(self.f, [])
                if i < len(cs):
                    self.labels[self.f] = {"x": cs[i][0], "y": cs[i][1]}
                    self.dirty = True
                    self.advance(1)
            elif k == ord("n"):
                self.labels[self.f] = None
                self.dirty = True
                self.advance(1)
            elif k == ord("u"):
                self.labels.pop(self.f, None)
                self.dirty = True
            elif k in (ord("d"), 83):
                self.advance(1)
            elif k in (ord("a"), 81):
                self.advance(-1)
            elif k == ord("]"):
                self.advance(10)
            elif k == ord("["):
                self.advance(-10)
            elif k == ord("z"):
                self.magnify = not self.magnify
            elif k == ord("s"):
                self.save()
            elif k == ord("q"):
                if self.dirty:
                    self.save()
                break
        cv2.destroyAllWindows()
        self.cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="labels JSON (resumable)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10 ** 9)
    ap.add_argument("--step", type=int, default=1,
                    help="label every Nth frame (keeps effort constant across "
                         "frame rates; unlabelled frames are skipped by the scorer)")
    a = ap.parse_args()
    if not os.path.exists(a.video):
        sys.exit(f"no such video: {a.video}")
    Labeler(a.video, a.out, a.start, a.end, a.step).run()


if __name__ == "__main__":
    main()
