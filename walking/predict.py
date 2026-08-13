"""
predict.py
==========
Run the trained walking classifier over a tennis video.

Extracts near-player pose if it is not already cached, builds the same window
features used in training, applies the model plus its tuned post-processing,
and
writes per-second JSONL and the walking intervals. ``--overlay`` renders a
review video with the near-player box, court speed, cadence and the WALKING
banner burned in.

Usage:
    python -m walking.predict /Volumes/Anya/Data/21/snippet.mp4 \
        --out walking/outputs/snippet_walking.json --overlay review.mp4
"""

import argparse
import json
import os

import numpy as np

from walking.court import ANALYSIS_SIZE, load_homography
from walking.evaluate import apply_post, to_intervals
from walking.extract_pose import DEFAULT_POSE_MODEL, extract
from walking.select_near import pose_path, select
from walking.features import frame_signals, window_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "outputs", "walking_model.joblib")


def predict_video(video, model_path=MODEL_PATH, pose_npz=None, device="mps",
                  pose_model=DEFAULT_POSE_MODEL):
    import joblib
    bundle = joblib.load(model_path)
    model, names, post = bundle["model"], bundle["feature_names"], bundle["post"]

    pose_npz = pose_npz or pose_path(video)
    if not os.path.isfile(pose_npz):
        extract(video, pose_model, device)
        select(video, out=pose_npz)
    z = np.load(pose_npz)
    kp, bbox, fps = z["kp"], z["bbox"], float(z["fps"])
    n = len(kp)

    sig = frame_signals(kp, bbox, load_homography(video), fps,
                        on_court=z.get("on_court"))

    # `post["stride"]` is a DECISION RATE, tuned on 30 fps clips extracted
    # every frame: stride 2 there meant scoring 15 times a second.  A pass that
    # already extracted pose at a lower rate (anya_end_telemetry samples at
    # 15 fps whatever the source does) must not decimate a second time, or the
    # decisions land at 7.5 Hz — below Nyquist for the 0.7-4.0 Hz cadence band
    # the features measure.  So the npz's own stride is divided out.
    pose_stride = int(z["stride"]) if "stride" in z else 1
    stride = max(1, int(round(int(post.get("stride", 2)) / pose_stride)))
    idx = np.arange(0, n, stride)
    X, got = window_features(sig, fps, idx)
    if got != names:
        raise RuntimeError("feature layout changed since training")

    prob_s = model.predict_proba(X)[:, 1]
    mask_s = apply_post(prob_s, fps / stride, post)

    prob = np.zeros(n)
    mask = np.zeros(n, bool)
    for i, t in enumerate(idx):
        prob[t:min(t + stride, n)] = prob_s[i]
        mask[t:min(t + stride, n)] = mask_s[i]
    # `fps` is the rate of the returned arrays, so a caller converting index to
    # seconds gets source time without knowing the stride; `pose_stride` is
    # there for anything that wants source FRAME numbers back.
    return {"fps": fps, "n_frames": n, "prob": prob, "is_walking": mask,
            "sig": sig, "pose_stride": pose_stride}


def write_outputs(res, out_json, jsonl=None):
    fps, mask, prob = res["fps"], res["is_walking"], res["prob"]
    valid = res["sig"]["valid"]
    # ``detection_coverage`` is the share of the interval in which a near player
    # was actually tracked. An interval with low coverage is a guess made across
    # a hole in the input, not a confident call, and downstream code should be
    # able to drop those without re-deriving them.
    st = int(res.get("pose_stride", 1))
    intervals = [{"start_frame": int(a) * st, "end_frame": int(b) * st,
                  "start_second": a / fps, "end_second": (b + 1) / fps,
                  "duration_s": (b - a + 1) / fps,
                  "mean_prob": float(np.mean(prob[a:b + 1])),
                  "detection_coverage": float(np.mean(valid[a:b + 1]))}
                 for a, b in to_intervals(mask)]
    payload = {"label": "near_player_walking", "fps": fps,
               "n_frames": int(res["n_frames"]),
               "walking_seconds": float(mask.sum() / fps),
               "intervals": intervals}
    json.dump(payload, open(out_json, "w"), indent=2)

    if jsonl:
        sig = res["sig"]
        with open(jsonl, "w") as fh:
            for s in range(int(np.ceil(res["n_frames"] / fps))):
                a, b = int(round(s * fps)), int(round((s + 1) * fps))
                b = min(b, res["n_frames"])
                if b <= a:
                    break
                fh.write(json.dumps({
                    "second": s,
                    "is_walking": bool(mask[a:b].mean() >= 0.5),
                    "confidence": float(np.mean(prob[a:b])),
                    "speed_mps": float(np.nanmean(sig["speed"][a:b]))
                    if np.isfinite(sig["speed"][a:b]).any() else None,
                    "court_x": float(np.nanmean(sig["court_x"][a:b]))
                    if np.isfinite(sig["court_x"][a:b]).any() else None,
                    "court_y": float(np.nanmean(sig["court_y"][a:b]))
                    if np.isfinite(sig["court_y"][a:b]).any() else None,
                }) + "\n")
    return payload


def render_overlay(video, res, out_path, labels=None, max_seconds=None,
                   start_second=0.0):
    """Review video: box, probability bar, and the ground-truth strip if given."""
    import cv2
    cap = cv2.VideoCapture(video)
    fps, n = res["fps"], res["n_frames"]
    z = np.load(pose_path(video))
    bbox = z["bbox"]
    w, h = ANALYSIS_SIZE
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    f0 = int(start_second * fps)
    total = n if max_seconds is None else min(n, f0 + int(max_seconds * fps))
    if f0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    gt = None
    if labels:
        gt = np.zeros(n, bool)
        for iv in json.load(open(labels))["intervals"]:
            gt[iv["start_frame"]:min(iv["end_frame"] + 1, n)] = True

    for f in range(f0, total):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
        b = bbox[f]
        walking = bool(res["is_walking"][f])
        if np.isfinite(b[0]):
            col = (0, 220, 0) if walking else (200, 200, 200)
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
        p = float(res["prob"][f])
        cv2.rectangle(frame, (20, 20), (20 + int(200 * p), 40), (0, 200, 255), -1)
        cv2.rectangle(frame, (20, 20), (220, 40), (255, 255, 255), 1)
        cv2.putText(frame, f"p(walk) {p:.2f}", (230, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if walking:
            cv2.putText(frame, "WALKING", (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
        sp = res["sig"]["speed"][f]
        if np.isfinite(sp):
            cv2.putText(frame, f"{sp:.2f} m/s", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if gt is not None:
            cv2.putText(frame, f"GT {'WALK' if gt[f] else 'stationary/active'}",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if gt[f] else (180, 180, 180), 2)
        vw.write(frame)
    cap.release()
    vw.release()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--out", default="walking/outputs/walking.json")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--overlay", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--overlay-seconds", type=float, default=None)
    ap.add_argument("--overlay-start", type=float, default=0.0)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    res = predict_video(a.video, a.model, device=a.device)
    payload = write_outputs(res, a.out, a.jsonl)
    print(f"{len(payload['intervals'])} walking intervals, "
          f"{payload['walking_seconds']:.1f}s of "
          f"{payload['n_frames'] / payload['fps']:.1f}s -> {a.out}")
    if a.overlay:
        render_overlay(a.video, res, a.overlay, a.labels, a.overlay_seconds,
                       a.overlay_start)
        print(f"overlay -> {a.overlay}")


if __name__ == "__main__":
    main()
