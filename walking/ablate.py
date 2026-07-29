"""
ablate.py
=========
Feature-group ablation on the cached feature matrix.

The per-fold F1 spread on the first fit was large (0.50 to 0.89), which is the
signature of a model leaning on features that identify *where in this clip* a
sample came from rather than what the player is doing. Absolute court position,
body scale in pixels and signed direction of travel are all such features: they
are perfectly predictive of the walks the model has seen and useless on a walk
in a block it has not. This script measures that directly by dropping groups of
features and re-running the same grouped CV.

Usage:
    python -m walking.ablate --pose ... --video ... --labels ...
"""

import argparse
import json
import os

import numpy as np

from walking.evaluate import prf, viterbi, smooth_prob
from walking.train import (OUT_DIR, build_dataset, cross_validate, nested_post)

# Feature-name prefixes/suffixes per group, matched against the column names.
GROUPS = {
    "absolute_position": ("court_x", "court_y"),
    "scale": ("box_h", "aspect"),
    "signed_direction": ("dy_signed",),
    "gait_spectrum": ("_freq", "_share", "_power"),
    "posture": ("_knee", "_hipank", "torso"),
    "coverage": ("_valid", "_on_court"),
}


def cols_for(names, keys):
    out = []
    for i, n in enumerate(names):
        if any(k in n for k in keys):
            out.append(i)
    return out


def run(X, y, groups, fps_s, folds_n, drop_cols, seed):
    keep = np.array([i for i in range(X.shape[1]) if i not in set(drop_cols)])
    prob, folds = cross_validate(X[:, keep], y, groups, fps_s, folds_n, seed)
    pred, _ = nested_post(prob, y, folds, fps_s, "viterbi")
    per_fold = [prf(y[folds == k], pred[folds == k])["f1"] for k in np.unique(folds)]
    s = prf(y, pred)
    return {"n_features": int(len(keep)), "precision": s["precision"],
            "recall": s["recall"], "f1": s["f1"],
            "per_fold_f1_mean": float(np.mean(per_fold)),
            "per_fold_f1_min": float(np.min(per_fold)),
            "per_fold_f1_std": float(np.std(per_fold))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--block-s", type=float, default=30.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X, y, idx, names, fps, n_frames, y_full, sig, meta = build_dataset(
        a.pose, a.video, a.labels, a.stride)
    fps_s = fps / a.stride
    groups = (idx / fps // a.block_s).astype(int)

    trials = {"all": []}
    for g, keys in GROUPS.items():
        trials[f"drop_{g}"] = cols_for(names, keys)
    # The combination the reasoning above predicts should generalise best.
    trials["drop_clip_specific"] = sorted(set(
        cols_for(names, GROUPS["absolute_position"])
        + cols_for(names, GROUPS["scale"])
        + cols_for(names, GROUPS["signed_direction"])))

    res = {}
    for tag, drop in trials.items():
        r = run(X, y, groups, fps_s, a.folds, drop, a.seed)
        res[tag] = r
        print(f"{tag:<24} feats {r['n_features']:>3}  P {r['precision']:.3f} "
              f"R {r['recall']:.3f} F1 {r['f1']:.3f}  "
              f"fold-mean {r['per_fold_f1_mean']:.3f} min {r['per_fold_f1_min']:.3f}",
              flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, "ablation.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
