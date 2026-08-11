"""
train.py
========
Fit and honestly score the walking classifier.

Protocol
--------
Two levels of validation, because they answer different questions.

  within-clip   GroupKFold over contiguous ``--block-s`` second blocks. Samples
                are frames, so neighbouring samples are near-identical and a
                random split would leak a walk across train and test. This is
                the number the earlier hand-tuned rule reported against, but
                every fold still shares the clip's camera, lighting and players.

  leave-one-clip-out (LOCO)   Train on every clip but one, test on the held-out
                clip. With more than one clip this is the headline number: it is
                the only one that answers "does this work on footage it has
                never seen". Post-processing is tuned on out-of-fold
                probabilities *within the training clips* and then applied
                unchanged to the held-out clip.

Outputs (walking/outputs/):
    walking_model.joblib   model + post-processing config + feature names
    metrics.json           every score below, plus the baselines
    oof.npz                per-frame out-of-fold probabilities (within-clip)

Usage:
    python -m walking.train --clips 21 22
    python -m walking.train --pose ...npz --video ...mp4 --labels ...json
"""

import argparse
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

from walking.court import load_homography
from walking.evaluate import (boundary_mask, event_scores, fbeta, hysteresis,
                              prf, smooth_prob, to_intervals, to_seconds, viterbi)
from walking.features import frame_signals, window_features

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
LABEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/Volumes/Anya/Data"
# The grids reach well below 0.5 on purpose: with --beta 2 (dead-time cutting,
# where a missed walk leaves dead footage in the cut but a loose one only trims
# a little extra) the optimum sits at a permissive threshold, and a grid that
# stopped at 0.3 was clipping the search rather than finding an optimum.
HI_GRID = (0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.65)
LO_RATIO = (0.3, 0.5, 0.75, 1.0)
MIN_DUR_GRID = (0.0, 1.0, 2.0)
SMOOTH_GRID = (0.0, 1.0, 2.0, 3.0)
THR_GRID = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
SWITCH_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
BIAS_GRID = (-0.4, -0.2, 0.0, 0.2, 0.4, 0.8, 1.2, 1.8)
POST_KINDS = ("threshold", "hysteresis", "viterbi")
MIN_COVERAGE = 0.5   # labelled events below this are outside the detector's reach


def load_labels(path, n_frames):
    d = json.load(open(path))
    y = np.zeros(n_frames, bool)
    for iv in d["intervals"]:
        y[iv["start_frame"]:min(iv["end_frame"] + 1, n_frames)] = True
    return y, d


def load_clip(video, pose_npz, labels_path, stride, block_s, name):
    """Everything downstream needs about one clip, in one dict.

    A pose npz may already be decimated (pipeline/anya_end_telemetry extracts
    at 15 fps whatever the source runs at), in which case it carries its own
    `stride` and an `fps` that is the rate of ITS rows.  `stride` here is a
    decision rate on top of that, exactly as in walking/predict.py, so the
    npz's own stride is divided out rather than compounded.  Labels are always
    in SOURCE frames and are indexed back through the npz stride.
    """
    z = np.load(pose_npz)
    kp, bbox, fps = z["kp"], z["bbox"], float(z["fps"])
    pose_stride = int(z["stride"]) if "stride" in z else 1
    stride = max(1, int(round(stride / pose_stride)))
    n = len(kp)
    sig = frame_signals(kp, bbox, load_homography(video), fps,
                        on_court=z.get("on_court"))
    idx = np.arange(0, n, stride)
    X, names = window_features(sig, fps, idx)
    n_src = int(z["n_src_frames"]) if "n_src_frames" in z else n * pose_stride
    y_src, meta = load_labels(labels_path, n_src)
    # Everything below this point lives on the npz's OWN row grid, whose rate
    # is `fps`; seconds therefore still come out right, and only the label
    # lookup has to know about the source timeline.
    rows = np.clip(np.arange(n) * pose_stride, 0, n_src - 1)
    y_full = y_src[rows]
    y = y_full[idx]
    return {"name": name, "X": X, "y": y, "idx": idx, "names": names,
            "fps": fps, "fps_s": fps / stride, "n_frames": n, "y_full": y_full,
            "sig": sig, "meta": meta, "stride": stride,
            "pose_stride": pose_stride,
            "groups": (idx / fps // block_s).astype(int)}


def fit_model(X, y, seed=0):
    m = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=False, random_state=seed)
    m.fit(X, y)
    return m


def apply_post(prob, fps, cfg):
    """Turn a probability trace into a boolean mask under a post-proc config."""
    p = smooth_prob(prob, fps, cfg.get("smooth_s", 0.0))
    kind = cfg["kind"]
    if kind == "threshold":
        mask = p >= cfg["thr"]
    elif kind == "hysteresis":
        return hysteresis(p, cfg["hi"], cfg["lo"], fps,
                          min_dur_s=cfg["min_dur_s"], max_gap_s=0.5)
    elif kind == "viterbi":
        mask = viterbi(p, cfg["switch_cost"], cfg.get("bias", 0.0))
    else:
        raise ValueError(kind)
    md = int(round(cfg.get("min_dur_s", 0.0) * fps))
    if md > 1:
        for a, b in to_intervals(mask):
            if (b - a + 1) < md:
                mask[a:b + 1] = False
    return mask


def post_grid(kind):
    if kind == "threshold":
        return [{"kind": kind, "smooth_s": s, "thr": t, "min_dur_s": md}
                for s in SMOOTH_GRID for t in THR_GRID for md in MIN_DUR_GRID]
    if kind == "hysteresis":
        return [{"kind": kind, "smooth_s": s, "hi": hi, "lo": hi * r,
                 "min_dur_s": md}
                for s in SMOOTH_GRID for hi in HI_GRID for r in LO_RATIO
                for md in MIN_DUR_GRID]
    return [{"kind": kind, "smooth_s": s, "switch_cost": c, "bias": b}
            for s in SMOOTH_GRID for c in SWITCH_GRID for b in BIAS_GRID]


def tune_post(prob, y, fps, kind, beta=1.0):
    """Best config of ``kind`` by frame F-beta on the given (prob, y)."""
    best, best_s = None, -1.0
    for cfg in post_grid(kind):
        sc = fbeta(prf(y, apply_post(prob, fps, cfg)), beta)
        if sc > best_s:
            best_s, best = sc, cfg
    return best, best_s


def cross_validate(X, y, groups, n_splits=5, seed=0):
    """Out-of-fold probabilities over grouped folds."""
    gkf = GroupKFold(n_splits=n_splits)
    prob = np.full(len(y), np.nan)
    folds = np.full(len(y), -1)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = fit_model(X[tr], y[tr], seed)
        prob[te] = m.predict_proba(X[te])[:, 1]
        folds[te] = k
    return prob, folds


def nested_post(prob, y, folds, fps_s, kind, beta=1.0):
    """Apply post-processing tuned on the other folds to each fold in turn."""
    pred = np.zeros(len(y), bool)
    chosen = []
    for k in np.unique(folds):
        te = folds == k
        cfg, _ = tune_post(prob[~te], y[~te], fps_s, kind, beta)
        # Post-processing is temporal: run it on the whole strided sequence and
        # keep only this fold's slots, so the filter sees real neighbours.
        mask = apply_post(prob, fps_s, cfg)
        pred[te] = mask[te]
        chosen.append({"fold": int(k), **cfg})
    return pred, chosen


def speed_baseline(X, names, y, folds, fps_s):
    """Best single speed threshold, tuned per fold on the other folds."""
    col = names.index("w2_net_speed")
    v = X[:, col]
    pred = np.zeros(len(y), bool)
    grid = np.arange(0.1, 3.0, 0.05)
    for k in np.unique(folds):
        te = folds == k
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = prf(y[~te], np.nan_to_num(v[~te]) > t)["f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, t
        pred[te] = np.nan_to_num(v[te]) > best_t
    return pred


def frames_from_samples(pred, idx, n_frames, stride):
    """Expand a strided sample mask back to one value per video frame."""
    out = np.zeros(n_frames, bool)
    for i, t in enumerate(idx):
        out[t:min(t + stride, n_frames)] = pred[i]
    return out


def clip_report(clip, pred, fps=None):
    """Frame / second / event scores for one clip's prediction."""
    fps = fps or clip["fps"]
    pred_full = frames_from_samples(pred, clip["idx"], clip["n_frames"],
                                    clip["stride"])
    n_sec = int(np.ceil(clip["n_frames"] / fps))
    y_sec = to_seconds(clip["y_full"], fps, n_sec)

    # A walk during which the detector never saw a person is a detection miss,
    # not a classifier miss; those events are also scored separately.
    scored = np.ones(clip["n_frames"], bool)
    excluded = []
    ps = clip.get("pose_stride", 1)      # label frames are SOURCE frames
    for iv in clip["meta"]["intervals"]:
        s0 = iv["start_frame"] // ps
        s1 = min(iv["end_frame"] // ps + 1, clip["n_frames"])
        cov = float(clip["sig"]["valid"][s0:s1].mean()) if s1 > s0 else 0.0
        if cov < MIN_COVERAGE:
            scored[s0:s1] = False
            excluded.append({"start_second": iv["start_second"],
                             "end_second": iv["end_second"], "coverage": cov})
    return {
        "frame": prf(clip["y"], pred),
        "second": prf(y_sec, to_seconds(pred_full, fps, n_sec)),
        "event": event_scores(clip["y_full"], pred_full, fps),
        "event_detectable_subset": {
            "min_detection_coverage": MIN_COVERAGE,
            "excluded_labelled_events": excluded,
            **event_scores(clip["y_full"][scored], pred_full[scored], fps)},
        "detection_coverage": float(np.mean(clip["sig"]["valid"])),
        "positive_rate": float(clip["y"].mean()),
    }


def within_clip(clip, folds_n, guard_s, seed, beta=1.0):
    """Grouped-block CV inside a single clip."""
    prob, folds = cross_validate(clip["X"], clip["y"], clip["groups"],
                                 folds_n, seed)
    y, fps_s = clip["y"], clip["fps_s"]
    variants = {}
    for kind in POST_KINDS:
        p, ch = nested_post(prob, y, folds, fps_s, kind, beta)
        variants[kind] = {"pred": p, "chosen": ch, "score": prf(y, p)}
    best_kind = max(variants, key=lambda k: fbeta(variants[k]["score"], beta))
    pred = variants[best_kind]["pred"]
    keep = boundary_mask(y, fps_s, guard_s)
    per_fold = [prf(y[folds == k], pred[folds == k])["f1"] for k in np.unique(folds)]
    rep = clip_report(clip, pred)
    rep["post_processing_choice"] = best_kind
    rep["post_variants"] = {k: v["score"] for k, v in variants.items()}
    rep["frame_boundary_guarded"] = prf(y[keep], pred[keep])
    rep["frame_raw_0.5"] = prf(y, prob >= 0.5)
    rep["speed_threshold_baseline"] = prf(
        y, speed_baseline(clip["X"], clip["names"], y, folds, fps_s))
    rep["per_fold_f1"] = [float(x) for x in per_fold]
    rep["per_fold_f1_mean"] = float(np.mean(per_fold))
    rep["per_fold_f1_std"] = float(np.std(per_fold))
    return rep, prob, folds, pred, variants[best_kind]["chosen"]


def loco(clips, folds_n, seed, beta=1.0):
    """Leave-one-clip-out: train on the rest, test on the held-out clip."""
    out = {}
    for held in clips:
        tr = [c for c in clips if c["name"] != held["name"]]
        Xtr = np.vstack([c["X"] for c in tr])
        ytr = np.concatenate([c["y"] for c in tr])
        # Offset each training clip's blocks so folds never straddle clips.
        gtr, off = [], 0
        for c in tr:
            gtr.append(c["groups"] + off)
            off += c["groups"].max() + 1
        gtr = np.concatenate(gtr)

        # Post-processing is tuned on out-of-fold probabilities of the TRAINING
        # clips only, then frozen — the held-out clip never informs it.
        prob_tr, folds_tr = cross_validate(Xtr, ytr, gtr, folds_n, seed)
        fps_tr = tr[0]["fps_s"]
        cfgs = {k: tune_post(prob_tr, ytr, fps_tr, k, beta)[0] for k in POST_KINDS}

        model = fit_model(Xtr, ytr, seed)
        prob_te = model.predict_proba(held["X"])[:, 1]
        scored = {}
        for k, cfg in cfgs.items():
            pred = apply_post(prob_te, held["fps_s"], cfg)
            scored[k] = {"config": cfg, **clip_report(held, pred)}
        # Report the family that was best on the training clips, plus all of
        # them, so the choice itself is visible rather than cherry-picked.
        chosen = max(cfgs, key=lambda k: fbeta(prf(
            ytr, apply_post(prob_tr, fps_tr, cfgs[k])), beta))
        out[held["name"]] = {
            "trained_on": [c["name"] for c in tr],
            "n_train_samples": int(len(ytr)),
            "post_processing_choice": chosen,
            "raw_0.5": prf(held["y"], prob_te >= 0.5),
            "by_post_processing": scored,
            "headline": scored[chosen],
        }
        h = scored[chosen]
        print(f"  LOCO test={held['name']:<4} train={[c['name'] for c in tr]}  "
              f"frame P {h['frame']['precision']:.3f} R {h['frame']['recall']:.3f} "
              f"F1 {h['frame']['f1']:.3f}   second F1 {h['second']['f1']:.3f}   "
              f"events {h['event']['matched']}/{h['event']['n_true']} "
              f"prec {h['event']['precision']:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip ids under --data-root, e.g. 21 22")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--pose", default=None, help="single-clip mode")
    ap.add_argument("--video", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--block-s", type=float, default=30.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--guard-s", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="F-beta weight used to pick thresholds; >1 favours recall")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    specs = []
    if a.pose:
        specs.append((a.video, a.pose, a.labels, os.path.basename(
            os.path.dirname(a.video or ""))or "clip"))
    else:
        for cid in (a.clips or ["21"]):
            d = os.path.join(a.data_root, cid)
            specs.append((os.path.join(d, "snippet.mp4"),
                          os.path.join(d, "snippet_walk_pose.npz"),
                          os.path.join(LABEL_DIR, f"labels_snippet{cid}.json"),
                          cid))

    clips = [load_clip(v, p, l, a.stride, a.block_s, n) for v, p, l, n in specs]
    for c in clips:
        print(f"clip {c['name']}: samples {len(c['y'])}  positives "
              f"{c['y'].mean():.1%}  coverage {np.mean(c['sig']['valid']):.1%}  "
              f"blocks {len(np.unique(c['groups']))}  features {c['X'].shape[1]}")

    res = {"clips": [c["name"] for c in clips],
           "tuning_objective": f"F{a.beta:g}",
           "cv": {"within_clip": f"GroupKFold {a.folds} folds over "
                                 f"{a.block_s:g}s blocks",
                  "cross_clip": "leave-one-clip-out"},
           "within_clip": {}}

    oof_store = {}
    for c in clips:
        print(f"within-clip CV: {c['name']}")
        rep, prob, folds, pred, chosen = within_clip(c, a.folds, a.guard_s,
                                                     a.seed, a.beta)
        res["within_clip"][c["name"]] = rep
        oof_store[c["name"]] = (prob, folds, pred, c["idx"])
        print(f"  frame P {rep['frame']['precision']:.3f} "
              f"R {rep['frame']['recall']:.3f} F1 {rep['frame']['f1']:.3f}  "
              f"second F1 {rep['second']['f1']:.3f}  "
              f"events {rep['event']['matched']}/{rep['event']['n_true']}  "
              f"per-fold {rep['per_fold_f1_mean']:.3f}+/-"
              f"{rep['per_fold_f1_std']:.3f}")

    if len(clips) > 1:
        print("leave-one-clip-out:")
        res["loco"] = loco(clips, a.folds, a.seed, a.beta)

    # Ship a model trained on everything, with the post-processing that the
    # within-clip runs agreed on.
    X = np.vstack([c["X"] for c in clips])
    y = np.concatenate([c["y"] for c in clips])
    model = fit_model(X, y, a.seed)
    first = res["within_clip"][clips[0]["name"]]
    kind = first["post_processing_choice"]
    _, _, _, _, chosen = within_clip(clips[0], a.folds, a.guard_s, a.seed, a.beta)
    keys = [k for k in chosen[0] if k not in ("fold", "kind")]
    final_post = {"kind": kind, "stride": a.stride, "max_gap_s": 0.5}
    for k in keys:
        final_post[k] = float(np.median([c[k] for c in chosen]))
    final_post["tuning_objective"] = f"F{a.beta:g}"
    res["final_post_processing"] = final_post

    # Permutation importance from one within-clip fold of the first clip.
    c0 = clips[0]
    gkf = GroupKFold(n_splits=a.folds)
    tr, te = next(iter(gkf.split(c0["X"], c0["y"], c0["groups"])))
    m0 = fit_model(c0["X"][tr], c0["y"][tr], a.seed)
    pi = permutation_importance(m0, c0["X"][te], c0["y"][te], n_repeats=5,
                                random_state=a.seed, scoring="average_precision")
    order = np.argsort(pi.importances_mean)[::-1][:25]
    res["top_features"] = [{"name": c0["names"][i],
                            "importance": float(pi.importances_mean[i])}
                           for i in order]

    import joblib
    joblib.dump({"model": model, "feature_names": clips[0]["names"],
                 "post": final_post, "fps": clips[0]["fps"],
                 "trained_on": [c["name"] for c in clips]},
                os.path.join(OUT_DIR, "walking_model.joblib"))
    np.savez_compressed(
        os.path.join(OUT_DIR, "oof.npz"),
        **{f"{k}_{f}": v for k, arrs in oof_store.items()
           for f, v in zip(("prob", "folds", "pred", "idx"), arrs)})
    suffix = "" if a.beta == 1.0 else f"_f{a.beta:g}"
    json.dump(res, open(os.path.join(OUT_DIR, f"metrics{suffix}.json"), "w"), indent=2)
    print("top features:", ", ".join(t["name"] for t in res["top_features"][:8]))


if __name__ == "__main__":
    main()
