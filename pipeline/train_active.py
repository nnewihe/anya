"""
train_active.py
==============
Stage 4: train a small GRU that maps a 2-second near-player pose window to
P(active). Honest evaluation via leave-one-clip-out CV (windows from a clip
never split across train/test), plus a logistic-regression baseline on
aggregate pose features so we can see whether the sequence model earns its keep.

Inputs (from make_windows.py): windows.npz + labels.json.
  X [N,60,51] normalized keypoints (NaN where pose missing) -> replaced with 0
  and given a per-timestep present flag, so input dim = 52.
Training excludes boundary windows by default (genuinely ambiguous); they are
still scored at eval. Augmentation: horizontal flip (L/R joint swap) + keypoint
noise.

Usage:
    python pipeline/train_active.py                       # LOCO CV + final model
    python pipeline/train_active.py --include_boundary
"""

import os
import sys
import json
import argparse

import numpy as np

# COCO-17 left/right keypoint pairs for horizontal-flip augmentation.
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def load_dataset(npz_path, labels_path, include_boundary):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)                     # [N,60,51]
    wid = d["wid"]; clip = d["clip"]; boundary = d["boundary"]
    labels = json.load(open(labels_path))
    y = np.array([labels[w]["label"] for w in wid], dtype=np.float32)

    keep = np.ones(len(X), dtype=bool)
    if not include_boundary:
        keep &= (boundary == 0)
    return X[keep], y[keep], clip[keep].astype(str)


def featurize(X):
    """Mask NaN -> 0, append per-timestep present flag. -> [N,60,52]."""
    present = (~np.isnan(X[:, :, 0])).astype(np.float32)[:, :, None]  # [N,60,1]
    Xz = np.nan_to_num(X, nan=0.0)
    return np.concatenate([Xz, present], axis=2)


def augment(xb, yb):
    import torch
    b = xb.shape[0]
    out = xb.clone()
    # horizontal flip: negate normalized x (feature 3k+0), swap L/R joints
    flip = torch.rand(b) < 0.5
    xf = out[flip].clone()
    for k in range(17):
        xf[:, :, 3 * k + 0] *= -1.0
    for a, bb in LR_PAIRS:
        for ch in range(3):
            tmp = xf[:, :, 3 * a + ch].clone()
            xf[:, :, 3 * a + ch] = xf[:, :, 3 * bb + ch]
            xf[:, :, 3 * bb + ch] = tmp
    out[flip] = xf
    # keypoint noise on x/y channels only (not conf, not present flag)
    noise = torch.zeros_like(out)
    for k in range(17):
        noise[:, :, 3 * k + 0] = torch.randn(b, out.shape[1]) * 0.02
        noise[:, :, 3 * k + 1] = torch.randn(b, out.shape[1]) * 0.02
    return out + noise, yb


def make_model():
    import torch.nn as nn
    class GRUClf(nn.Module):
        def __init__(self, in_dim=52, hidden=48):
            super().__init__()
            self.gru = nn.GRU(in_dim, hidden, batch_first=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            _, h = self.gru(x)
            return self.fc(self.drop(h[-1])).squeeze(-1)
    return GRUClf()


def train_fold(Xtr, ytr, Xva, yva, epochs=40, seed=0):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva)
    model = make_model()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(Xtr_t); bs = 64
    best_state, best_va = None, -1
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = augment(Xtr_t[idx], ytr_t[idx])
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = (torch.sigmoid(model(Xva_t)).numpy() > 0.5).astype(float)
        acc = float((va == yva).mean())
        if acc > best_va:
            best_va, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def evaluate(model, X):
    import torch
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def baseline_features(X):
    """Aggregate pose-motion features for the logistic baseline."""
    present = (~np.isnan(X[:, :, 0])).astype(np.float32)
    Xz = np.nan_to_num(X, nan=0.0)
    # per-frame keypoint speed (mean over joints)
    dif = np.diff(Xz.reshape(Xz.shape[0], Xz.shape[1], 17, 3)[:, :, :, :2], axis=1)
    speed = np.linalg.norm(dif, axis=3).mean(axis=2)   # [N,59]
    feats = np.stack([
        speed.mean(axis=1), speed.std(axis=1), speed.max(axis=1),
        present.mean(axis=1),
        Xz[:, :, 1::3].std(axis=(1, 2)),   # vertical spread (bounce)
        Xz[:, :, 0::3].std(axis=(1, 2)),   # lateral spread
    ], axis=1)
    return np.nan_to_num(feats)


def main():
    ap = argparse.ArgumentParser(description="Train active/dead GRU on pose windows (LOCO CV)")
    ap.add_argument("--windows", default="/Volumes/Anya/Data/windows.npz")
    ap.add_argument("--labels", default="/Volumes/Anya/Data/labels.json")
    ap.add_argument("--include_boundary", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="/Volumes/Anya/Data/active_model.pt")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import torch

    X, y, clip = load_dataset(args.windows, args.labels, args.include_boundary)
    Xf = featurize(X)
    clips = sorted(set(clip))
    print(f"[data] {len(X)} windows, {y.mean():.0%} active, {len(clips)} clips")

    # ── Baseline (LOCO) ──
    Bfeat = baseline_features(X)
    b_acc, g_acc, g_auc = [], [], []
    for held in clips:
        tr, va = clip != held, clip == held
        if va.sum() == 0 or len(set(y[tr])) < 2:
            continue
        lr = LogisticRegression(max_iter=1000).fit(Bfeat[tr], y[tr])
        b_acc.append((lr.predict(Bfeat[va]) == y[va]).mean())

        model = train_fold(Xf[tr], y[tr], Xf[va], y[va], epochs=args.epochs)
        p = evaluate(model, Xf[va])
        g_acc.append(((p > 0.5) == y[va]).mean())
        if len(set(y[va])) == 2:
            g_auc.append(roc_auc_score(y[va], p))
        print(f"  hold {held:>4}: baseline={b_acc[-1]:.2f}  GRU acc={g_acc[-1]:.2f}"
              + (f"  AUC={g_auc[-1]:.2f}" if len(set(y[va])) == 2 else "  AUC=n/a"))

    print(f"\n[LOCO] baseline acc {np.mean(b_acc):.3f} | GRU acc {np.mean(g_acc):.3f}"
          f" | GRU AUC {np.mean(g_auc):.3f}")

    # ── Final model on all data ──
    final = train_fold(Xf, y, Xf, y, epochs=args.epochs)
    torch.save({"state": final.state_dict(), "in_dim": 52, "hidden": 48,
                "loco_acc": float(np.mean(g_acc)), "loco_auc": float(np.mean(g_auc))}, args.out)
    print(f"[done] final model -> {args.out}")


if __name__ == "__main__":
    main()
