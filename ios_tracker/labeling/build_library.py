#!/usr/bin/env python3
"""Build a labelled rally library across many matches.

    python3 build_library.py --data /Volumes/Anya/Data --out library

For every match it finds rally windows, cuts each to a short clip, and
precomputes the detector candidates so labelling is instant. Then run
label_library.py.

Why it works the way it does
----------------------------
* Rally finding is **tracker-independent**. It looks for windows where
  non-clutter detections *move*, not for windows the tracker already handles.
  Seeding a training set from the tracker's own output teaches it only what it
  already knows, and hides its failures.
* Each clip is padded past the rally into dead time. Ghost rate (drawing a ball
  where there is none) is the tracker's real failure mode, so the library has to
  contain frames with no ball in them.
* Clips are short. A tuning sweep re-solves every clip on every trial; over 12 h
  of source that would be unusable, over ~5 min of clips it's seconds.
* Scanning goes through ffmpeg at 4 Hz, hardware-decoded and pre-scaled to
  960-wide — decoding 4K is the bottleneck, not inference.
* Stationary clutter is clustered per match: every venue has its own baskets and
  racks, so exclusion zones cannot be shared between courts.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

SAMPLE_HZ = 4.0        # rally scan rate
SCAN_W, SCAN_H = 960, 544
SCAN_CONF = 0.15       # conservative: we want confident balls to find rallies
CAND_CONF = 0.02       # generous: labelling wants every plausible candidate
MAX_BOX_PX = 45
DBSCAN_EPS = 12
DBSCAN_MIN = 15

RALLY_S = 5.0          # window length considered one rally
PAD_S = 1.5            # dead time kept either side
MIN_SPREAD_PX = 120    # analysis px the ball must travel to count as a rally


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe(path):
    r = sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames", "-show_entries",
            "format=duration", "-of", "json", path])
    if r.returncode != 0:
        return None
    d = json.loads(r.stdout)
    st = d["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    return {"w": st["width"], "h": st["height"], "fps": fps,
            "dur": float(d["format"]["duration"])}


def scan_frames(path, hz):
    """Yield 960-wide BGR frames at `hz`, hardware-decoded. ffmpeg is far faster
    at this than cv2 seeking, and 4K decode dominates the scan."""
    w = SCAN_W
    cmd = ["ffmpeg", "-hwaccel", "videotoolbox", "-i", path,
           "-vf", f"fps={hz},scale={w}:-2", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-loglevel", "error", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # frame height follows the source aspect; read the first frame to learn it
    # by probing a single scaled frame first.
    return p


def scaled_size(info):
    w = SCAN_W
    h = int(round(info["h"] * (w / info["w"])))
    h -= h % 2
    return w, h


def letterbox(frame):
    sh_, sw_ = frame.shape[:2]
    r = min(SCAN_W / sw_, SCAN_H / sh_)
    nw, nh = int(round(sw_ * r)), int(round(sh_ * r))
    px, py = (SCAN_W - nw) // 2, (SCAN_H - nh) // 2
    import cv2
    canvas = np.full((SCAN_H, SCAN_W, 3), 114, np.uint8)
    canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
    return canvas, r, px, py


def detect(model, frame, conf):
    """Detections in the frame's own pixel space (already ~960 wide)."""
    import cv2
    from PIL import Image
    canvas, r, px, py = letterbox(frame)
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    det = np.array(model.predict({"image": img})["detections"]).reshape(5, -1)
    out = []
    for j in np.where(det[4] > conf)[0]:
        cx, cy, bw, bh, cf = (float(v) for v in det[:, j])
        if max(bw, bh) > MAX_BOX_PX:
            continue
        out.append(((cx - px) / r, (cy - py) / r, cf))
    out.sort(key=lambda c: -c[2])
    return out


def dbscan_zones(points):
    """Stationary clutter, same recipe as the Swift ExclusionZones."""
    if len(points) < DBSCAN_MIN:
        return []
    from sklearn.cluster import DBSCAN
    X = np.array([(p[0], p[1]) for p in points])
    lab = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit(X).labels_
    zones = []
    for k in set(lab):
        if k < 0:
            continue
        pts = X[lab == k]
        zones.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    return zones


def in_zones(x, y, zones):
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in zones)


def find_rallies(samples, zones, hz, n_want):
    """Windows where non-clutter detections move. samples[i] = list of dets."""
    win = int(RALLY_S * hz)
    scored = []
    for s in range(0, max(len(samples) - win, 1)):
        pts = []
        hits = 0
        for i in range(s, min(s + win, len(samples))):
            live = [d for d in samples[i] if not in_zones(d[0], d[1], zones)]
            if live:
                hits += 1
                pts.extend([(d[0], d[1]) for d in live])
        if hits < win * 0.35 or len(pts) < 4:
            continue
        a = np.array(pts)
        spread = float(np.hypot(np.ptp(a[:, 0]), np.ptp(a[:, 1])))
        if spread < MIN_SPREAD_PX:
            continue                       # stationary clutter, not a rally
        scored.append((hits * min(spread, 600), s, hits, spread))
    scored.sort(reverse=True)

    picked = []
    for score, s, hits, spread in scored:
        if any(abs(s - p[1]) < win for p in picked):
            continue                       # non-overlapping
        picked.append((score, s, hits, spread))
        if len(picked) >= n_want:
            break
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/Volumes/Anya/Data")
    ap.add_argument("--out", default="library")
    ap.add_argument("--per-match", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="only first N matches (smoke test)")
    ap.add_argument("--match", action="append", default=[], help="only these match dirs")
    a = ap.parse_args()

    import cv2
    import coremltools as ct
    here = os.path.dirname(os.path.abspath(__file__))
    mlpkg = os.path.join(here, "..", "BallTracker", "Resources", "ball_best.mlpackage")
    print("loading model (ANE)…")
    model = ct.models.MLModel(mlpkg, compute_units=ct.ComputeUnit.ALL)

    clips_dir = os.path.join(a.out, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    man_path = os.path.join(a.out, "manifest.json")
    manifest = {"clips": []}
    if os.path.exists(man_path):
        manifest = json.load(open(man_path))
        print(f"resuming: {len(manifest['clips'])} clips already in manifest")
    done = {c["match"] for c in manifest["clips"]}

    matches = sorted(d for d in os.listdir(a.data)
                     if os.path.isdir(os.path.join(a.data, d)))
    if a.match:
        matches = [m for m in matches if m in a.match]
    if a.limit:
        matches = matches[:a.limit]

    for mi, m in enumerate(matches, 1):
        if m in done:
            print(f"[{mi}/{len(matches)}] {m}: already done, skipping")
            continue
        src = None
        for name in ("snippet.mp4", "match.mp4"):
            p = os.path.join(a.data, m, name)
            if os.path.exists(p):
                src = p
                break
        if not src:
            continue
        info = probe(src)
        if not info:
            print(f"[{mi}/{len(matches)}] {m}: unreadable, skipping")
            continue

        sw, sh_ = scaled_size(info)
        print(f"[{mi}/{len(matches)}] {m}: {info['w']}x{info['h']} @{info['fps']:.0f}fps "
              f"{info['dur']:.0f}s — scanning…", flush=True)

        proc = scan_frames(src, SAMPLE_HZ)
        fsize = sw * sh_ * 3
        samples, allpts = [], []
        while True:
            buf = proc.stdout.read(fsize)
            if len(buf) < fsize:
                break
            fr = np.frombuffer(buf, np.uint8).reshape(sh_, sw, 3)
            dets = detect(model, fr, SCAN_CONF)
            samples.append(dets)
            allpts.extend(dets)
        proc.stdout.close()
        proc.wait()

        zones = dbscan_zones(allpts)
        picks = find_rallies(samples, zones, SAMPLE_HZ, a.per_match)
        print(f"      {len(samples)} samples, {len(zones)} clutter zones, "
              f"{len(picks)} rally window(s)")

        for ri, (score, s, hits, spread) in enumerate(picks):
            t0 = max(0.0, s / SAMPLE_HZ - PAD_S)
            dur = RALLY_S + 2 * PAD_S
            name = f"{m}_{ri}"
            clip = os.path.join(clips_dir, name + ".mp4")
            if not os.path.exists(clip):
                r = sh(["ffmpeg", "-y", "-ss", f"{t0:.2f}", "-i", src, "-t", f"{dur:.2f}",
                        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                        "-loglevel", "error", clip])
                if r.returncode != 0:
                    print("      ffmpeg failed:", r.stderr.strip()[:160])
                    continue
            ci = probe(clip)
            # Keep labelling effort ~30 labels per second of video regardless of
            # source frame rate: a 120 fps clip otherwise costs 4x the keystrokes
            # for the same tennis.
            step = max(1, int(round(ci["fps"] / 30.0)))
            manifest["clips"].append({
                "name": name, "match": m, "source": src, "clip": clip,
                "src_t0": round(t0, 2), "dur": round(ci["dur"], 2),
                "fps": round(ci["fps"], 3), "w": ci["w"], "h": ci["h"],
                "step": step, "rally_score": round(score, 1),
                "spread_px": round(spread, 1),
            })
            print(f"      -> {name}.mp4  t={t0:.1f}s spread={spread:.0f}px step={step}")
        json.dump(manifest, open(man_path, "w"), indent=1)

    print(f"\n{len(manifest['clips'])} clips in {man_path}")
    print("next: python3 label_library.py --lib", a.out)


if __name__ == "__main__":
    main()
