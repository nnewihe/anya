"""
export_walking_model.py
=======================
The walking classifier (walking/outputs/walking_model.joblib, a sklearn
HistGradientBoostingClassifier) as plain JSON, so the Swift engine evaluates
the same 400 trees without any ML runtime.

sklearn predicts from RAW features: at each split `x <= threshold` goes left
and NaN follows `missing_go_to_left`; the probability is the logistic of the
baseline plus the sum of the leaf values. That is everything a node needs, so
each tree is stored as parallel arrays.

    ~/.pyenv/versions/anya-build/bin/python ios/AnyaTennis/tools/export_walking_model.py
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

OUT = os.path.join(REPO, "ios", "AnyaTennis", "Packages", "Anya2Engine", "Sources",
                   "Anya2Engine", "Resources", "walking_model.json")


def main():
    import joblib
    from walking.predict import MODEL_PATH, _alias_sklearn_loss
    _alias_sklearn_loss()
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    if model.n_trees_per_iteration_ != 1 or model.is_categorical_ is not None:
        raise RuntimeError("only a binary, numeric-feature model can be exported this way")
    trees = []
    for (predictor,) in model._predictors:
        nodes = predictor.nodes
        trees.append({
            "is_leaf": nodes["is_leaf"].astype(int).tolist(),
            "feature": nodes["feature_idx"].astype(int).tolist(),
            "threshold": [float(v) for v in nodes["num_threshold"]],
            "missing_go_left": nodes["missing_go_to_left"].astype(int).tolist(),
            "left": nodes["left"].astype(int).tolist(),
            "right": nodes["right"].astype(int).tolist(),
            "value": [float(v) for v in nodes["value"]],
        })
    out = {"source": os.path.relpath(MODEL_PATH, REPO),
           "feature_names": bundle["feature_names"],
           "post_stride": int(bundle["post"].get("stride", 2)),
           "baseline": float(np.ravel(model._baseline_prediction)[0]),
           "trees": trees}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {os.path.relpath(OUT, REPO)}: {len(trees)} trees, "
          f"{len(bundle['feature_names'])} features, {os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
