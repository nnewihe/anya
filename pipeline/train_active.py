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


def load_dataset(npz_path, labels_path, include_boundary, drop_global=False):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)                     # [N,60,51] or [N,60,59]
    wid = d["wid"]; clip = d["clip"]; boundary = d["boundary"]
    n_pose = int(d["n_pose"]) if "n_pose" in d.files else 51
    n_global = int(d["n_global"]) if "n_global" in d.files else 0
    if drop_global and n_global:                      # pose-only ablation
        X, n_global = X[:, :, :n_pose], 0
    labels = json.load(open(labels_path))
    y = np.array([labels[w]["label"] for w in wid], dtype=np.float32)

    keep = np.ones(len(X), dtype=bool)
    if not include_boundary:
        keep &= (boundary == 0)
    return X[keep], y[keep], clip[keep].astype(str), n_pose, n_global


def featurize(X, n_pose=51):
    """Mask NaN -> 0, append per-timestep present flag. -> [N,60,D+1].

    The present flag is keyed off pose channel 0; Stream B can be present on a
    frame where the pose failed, so its NaNs are zero-filled under the same flag.
    """
    present = (~np.isnan(X[:, :, 0])).astype(np.float32)[:, :, None]  # [N,60,1]
    Xz = np.nan_to_num(X, nan=0.0)
    return np.concatenate([Xz, present], axis=2)


def augment(xb, yb, n_pose=51, n_global=0):
    """Horizontal-mirror + keypoint-noise augmentation, stream-aware.

    Pose x is bbox-relative in [0,1] ((x-bx)/bw), so the mirror is 1-x, NOT -x.
    Negating it — as this did before the two-stream change — maps augmented
    poses into [-1,0], a range no real sample occupies, so the "augmented" half
    of each batch was off-manifold noise rather than plausible mirrored play.

    Stream B mirrors in full-frame coords: cx -> 1-cx and dcx -> -dcx, while
    cy/bw/bh/dcy/speed/disp are mirror-invariant.

    Both transforms are gated on the present flag (last channel) so zero-filled
    missing frames are not turned into fabricated observations.
    """
    import torch
    b, T, D = xb.shape
    out = xb.clone()
    present = out[:, :, -1:]                       # [b,T,1], 1 where pose exists
    flip = torch.rand(b) < 0.5
    xf = out[flip].clone()
    pf = present[flip]

    for k in range(17):                            # mirror pose x within the box
        xf[:, :, 3 * k + 0] = torch.where(
            pf[:, :, 0] > 0, 1.0 - xf[:, :, 3 * k + 0], xf[:, :, 3 * k + 0])
    for a, bb in LR_PAIRS:                         # swap left/right joints
        for ch in range(3):
            tmp = xf[:, :, 3 * a + ch].clone()
            xf[:, :, 3 * a + ch] = xf[:, :, 3 * bb + ch]
            xf[:, :, 3 * bb + ch] = tmp
    if n_global:
        gi = n_pose                                # [cx,cy,bw,bh,dcx,dcy,speed,disp]
        xf[:, :, gi + 0] = 1.0 - xf[:, :, gi + 0]  # cx
        xf[:, :, gi + 4] = -xf[:, :, gi + 4]       # dcx
    out[flip] = xf

    # Keypoint jitter on pose x/y only — not conf, not Stream B, not the flag.
    noise = torch.zeros_like(out)
    for k in range(17):
        noise[:, :, 3 * k + 0] = torch.randn(b, T) * 0.02
        noise[:, :, 3 * k + 1] = torch.randn(b, T) * 0.02
    return out + noise * present, yb


def make_model(in_dim=52, hidden=48, n_out=1):
    """Unidirectional GRU classifier. n_out=1 -> binary logit (squeezed),
    n_out=k -> k-way logits for the dead/transition/active head."""
    import torch.nn as nn
    class GRUClf(nn.Module):
        def __init__(self, in_dim=52, hidden=48, n_out=1):
            super().__init__()
            self.gru = nn.GRU(in_dim, hidden, batch_first=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden, n_out)
            self.n_out = n_out
        def forward(self, x):
            _, h = self.gru(x)
            z = self.fc(self.drop(h[-1]))
            return z.squeeze(-1) if self.n_out == 1 else z
    return GRUClf(in_dim, hidden, n_out)


def train_fold(Xtr, ytr, Xva, yva, epochs=40, seed=0, n_pose=51, n_global=0):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xva_t = torch.tensor(Xva)
    model = make_model(in_dim=Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    # Class-weighted BCE: dead/live window counts are imbalanced by construction.
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pw = torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(Xtr_t); bs = 64
    best_state, best_va = None, -1
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = augment(Xtr_t[idx], ytr_t[idx], n_pose, n_global)
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


def baseline_features(X, n_pose=51, n_global=0):
    """Aggregate pose-motion features for the logistic baseline."""
    P = X[:, :, :n_pose]
    present = (~np.isnan(P[:, :, 0])).astype(np.float32)
    Pz = np.nan_to_num(P, nan=0.0)
    # per-frame keypoint speed (mean over joints)
    dif = np.diff(Pz.reshape(Pz.shape[0], Pz.shape[1], 17, 3)[:, :, :, :2], axis=1)
    speed = np.linalg.norm(dif, axis=3).mean(axis=2)   # [N,T-1]
    cols = [
        speed.mean(axis=1), speed.std(axis=1), speed.max(axis=1),
        present.mean(axis=1),
        Pz[:, :, 1::3].std(axis=(1, 2)),   # vertical spread (bounce)
        Pz[:, :, 0::3].std(axis=(1, 2)),   # lateral spread
    ]
    if n_global:                            # global motion summary (Stream B)
        G = np.nan_to_num(X[:, :, n_pose:n_pose + n_global], nan=0.0)
        cols += [G[:, :, 6].mean(axis=1), G[:, :, 6].max(axis=1),   # speed
                 G[:, :, 7].mean(axis=1),                           # disp
                 G[:, :, 1].mean(axis=1), G[:, :, 3].mean(axis=1)]  # cy, bh
    return np.nan_to_num(np.stack(cols, axis=1))


def main():
    ap = argparse.ArgumentParser(description="Train active/dead GRU on pose windows (LOCO CV)")
    ap.add_argument("--windows", default="/Volumes/Anya/Data/windows.npz")
    ap.add_argument("--labels", default="/Volumes/Anya/Data/labels.json")
    ap.add_argument("--include_boundary", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="/Volumes/Anya/Data/active_model.pt")
    ap.add_argument("--no_global", action="store_true",
                    help="pose-only ablation: drop Stream B even if present in the npz")
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="clips excluded from training entirely — use before running "
                         "evaluate_events.py on those clips, otherwise the timeline "
                         "evaluation is scored on data the model was fit to")
    ap.add_argument("--skip_loco", action="store_true",
                    help="skip leave-one-clip-out CV and just fit the final model")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import torch

    X, y, clip, n_pose, n_global = load_dataset(
        args.windows, args.labels, args.include_boundary, drop_global=args.no_global)
    if args.holdout:
        keep = ~np.isin(clip, np.array(args.holdout, dtype=clip.dtype))
        print(f"[holdout] excluding clips {sorted(args.holdout)}: "
              f"{len(X) - int(keep.sum())} of {len(X)} windows dropped")
        X, y, clip = X[keep], y[keep], clip[keep]
    Xf = featurize(X, n_pose)
    clips = sorted(set(clip))
    print(f"[data] {len(X)} windows, {y.mean():.0%} active, {len(clips)} clips, "
          f"in_dim={Xf.shape[2]} (pose {n_pose} + global {n_global} + present 1)")

    # ── Baseline (LOCO) ──
    Bfeat = baseline_features(X, n_pose, n_global)
    b_acc, g_acc, g_auc = [], [], []
    for held in (clips if not args.skip_loco else []):
        tr, va = clip != held, clip == held
        if va.sum() == 0 or len(set(y[tr])) < 2:
            continue
        lr = LogisticRegression(max_iter=1000).fit(Bfeat[tr], y[tr])
        b_acc.append((lr.predict(Bfeat[va]) == y[va]).mean())

        model = train_fold(Xf[tr], y[tr], Xf[va], y[va], epochs=args.epochs,
                           n_pose=n_pose, n_global=n_global)
        p = evaluate(model, Xf[va])
        g_acc.append(((p > 0.5) == y[va]).mean())
        if len(set(y[va])) == 2:
            g_auc.append(roc_auc_score(y[va], p))
        print(f"  hold {held:>4}: baseline={b_acc[-1]:.2f}  GRU acc={g_acc[-1]:.2f}"
              + (f"  AUC={g_auc[-1]:.2f}" if len(set(y[va])) == 2 else "  AUC=n/a"))

    if g_acc:
        print(f"\n[LOCO] baseline acc {np.mean(b_acc):.3f} | GRU acc {np.mean(g_acc):.3f}"
              f" | GRU AUC {np.mean(g_auc):.3f}")
    else:
        g_acc, g_auc = [float("nan")], [float("nan")]

    # ── Final model on all data ──
    final = train_fold(Xf, y, Xf, y, epochs=args.epochs,
                       n_pose=n_pose, n_global=n_global)
    torch.save({"state": final.state_dict(), "in_dim": Xf.shape[2], "hidden": 48,
                "n_pose": n_pose, "n_global": n_global, "win_sec": 2.0, "L": 60,
                "loco_acc": float(np.mean(g_acc)), "loco_auc": float(np.mean(g_auc))}, args.out)
    print(f"[done] final model -> {args.out}")


if __name__ == "__main__":
    main()
