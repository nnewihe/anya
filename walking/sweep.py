"""
sweep.py
========
Second round of model selection, on top of the ablation result (pixel body
scale is camera-specific and hurts, so it is dropped throughout).

Varies the two things the ablation left open: how much temporal context the
features carry (adding an 8 s window), and how hard the booster is regularised
— with 15 blocks of labelled clip, an under-regularised booster memorises the
block it is in. Scored with the same grouped CV as everything else.

Usage:
    python -m walking.sweep --pose ... --video ... --labels ...
"""

import argparse
import itertools
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from walking.ablate import cols_for, GROUPS
from walking.court import load_homography
from walking.evaluate import prf
from walking.features import frame_signals, window_features
from walking.train import OUT_DIR, load_labels, nested_post

MODELS = {
    "base":     dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                     min_samples_leaf=40, l2_regularization=1.0),
    "shallow":  dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=8,
                     min_samples_leaf=80, l2_regularization=5.0),
    "stumps":   dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=4,
                     min_samples_leaf=120, l2_regularization=10.0),
    "balanced": dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=8,
                     min_samples_leaf=80, l2_regularization=5.0,
                     class_weight="balanced"),
}
WINDOW_SETS = {
    "w4": (0.5, 1.0, 2.0, 4.0),
    "w8": (0.5, 1.0, 2.0, 4.0, 8.0),
}


def cv_score(X, y, groups, fps_s, params, folds_n=5, seed=0):
    gkf = GroupKFold(n_splits=folds_n)
    prob = np.full(len(y), np.nan)
    folds = np.full(len(y), -1)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = HistGradientBoostingClassifier(early_stopping=False,
                                           random_state=seed, **params)
        m.fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
        folds[te] = k
    pred, chosen = nested_post(prob, y, folds, fps_s, "viterbi")
    per_fold = [prf(y[folds == k], pred[folds == k])["f1"] for k in np.unique(folds)]
    s = prf(y, pred)
    return {"precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
            "per_fold_f1_mean": float(np.mean(per_fold)),
            "per_fold_f1_min": float(np.min(per_fold)),
            "per_fold_f1_std": float(np.std(per_fold))}, prob, folds, chosen


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

    z = np.load(a.pose)
    kp, bbox, fps = z["kp"], z["bbox"], float(z["fps"])
    sig = frame_signals(kp, bbox, load_homography(a.video), fps,
                        on_court=z.get("on_court"))
    y_full, _ = load_labels(a.labels, len(kp))
    idx = np.arange(0, len(kp), a.stride)
    y = y_full[idx]
    groups = (idx / fps // a.block_s).astype(int)
    fps_s = fps / a.stride

    res = {}
    for wtag, wins in WINDOW_SETS.items():
        X, names = window_features(sig, fps, idx, windows_s=wins)
        drop = set(cols_for(names, GROUPS["scale"]))
        keep = np.array([i for i in range(X.shape[1]) if i not in drop])
        Xk = X[:, keep]
        for mtag, params in MODELS.items():
            s, _, _, _ = cv_score(Xk, y, groups, fps_s, params, a.folds, a.seed)
            res[f"{wtag}_{mtag}"] = {"n_features": int(Xk.shape[1]), **s}
            print(f"{wtag}_{mtag:<10} feats {Xk.shape[1]:>3}  P {s['precision']:.3f} "
                  f"R {s['recall']:.3f} F1 {s['f1']:.3f}  "
                  f"fold-mean {s['per_fold_f1_mean']:.3f} "
                  f"min {s['per_fold_f1_min']:.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(OUT_DIR, "sweep.json"), "w"), indent=2)
    best = max(res, key=lambda k: res[k]["f1"])
    print("best:", best, res[best])


if __name__ == "__main__":
    main()
