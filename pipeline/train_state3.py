"""
train_state3.py
===============
Three-class GRU: {dead=0, transition=1, active=2} from near-player pose
(Stream A, 51 dims) + near-player bbox kinematics (Stream B, 8 dims).

Same backbone as the binary `train_active.py` — unidirectional GRU, scored at
the window's END frame, so the design stays streaming-compatible — with a 3-way
head and class-weighted cross-entropy.

The transition class is not cosmetic. Measured on the cached spans, mean bbox
speed 1s AFTER a point ends (0.123) exceeds 1s before it (0.091); the two
streams only separate ~3s out. A binary model is therefore forced to guess in
the band around every transition, and the old pipeline's response was to EXCLUDE
those windows from training. Making the band its own label lets the model
express "I am in the ambiguous region", which is what the downstream event rule
actually wants to know.

`transition` is an INTERNAL state: the product still emits binary point-end
timestamps (see evaluate_events.py --states 3), so no consumer needs a policy
for it.

Evaluation here is window-level and leave-one-clip-out, which is diagnostic
only. The number that decides anything is the event-level one from
evaluate_events.py.

Usage:
    python pipeline/train_state3.py
    python pipeline/train_state3.py --holdout 21 22 --skip_loco --out /tmp/s3.pt
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_active import make_model, featurize, augment

CLASS_NAMES = ["dead", "transition", "active"]


def load(npz_path, drop_global=False):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    clip = d["clip"].astype(str)
    n_pose = int(d["n_pose"]); n_global = int(d["n_global"])
    if drop_global and n_global:
        X, n_global = X[:, :, :n_pose], 0
    return X, y, clip, n_pose, n_global, d


def train_fold(Xtr, ytr, epochs=40, seed=0, n_pose=51, n_global=0, hidden=48):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    model = make_model(in_dim=Xtr.shape[2], hidden=hidden, n_out=3)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)

    # Inverse-frequency class weights: transition is rare by construction, and
    # the dead/active durations are imbalanced on top of that.
    cnt = np.bincount(ytr, minlength=3).astype(np.float64)
    w = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1) / 3.0, 0.0)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))

    n, bs = len(Xt), 64
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = augment(Xt[idx], yt[idx], n_pose, n_global)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step(tot / n)
    return model


def predict(model, X):
    import torch
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X)), dim=1).numpy()


def report(y, p, tag):
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    print(f"  [{tag}] acc={acc:.3f}")
    print(f"  {'':>11}" + "".join(f"{c:>12}" for c in CLASS_NAMES) + f"{'recall':>9}")
    for c in range(3):
        row = [int(((y == c) & (pred == k)).sum()) for k in range(3)]
        n = max(sum(row), 1)
        print(f"  {CLASS_NAMES[c]:>11}" + "".join(f"{v:>12}" for v in row)
              + f"{row[c]/n:>9.2f}")
    prec = [((pred == c) & (y == c)).sum() / max((pred == c).sum(), 1) for c in range(3)]
    print(f"  {'precision':>11}" + "".join(f"{v:>12.2f}" for v in prec))
    return acc


def main():
    ap = argparse.ArgumentParser(description="Train 3-class dead/transition/active GRU")
    ap.add_argument("--windows", default="/Volumes/Anya/Data/state_windows.npz")
    ap.add_argument("--out", default="/Volumes/Anya/Data/state3_model.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_global", action="store_true",
                    help="pose-only ablation (drops near-player bbox kinematics)")
    ap.add_argument("--holdout", nargs="*", default=None,
                    help="clips excluded from training — required before running "
                         "evaluate_events.py on them")
    ap.add_argument("--skip_loco", action="store_true")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    X, y, clip, n_pose, n_global, d = load(args.windows, args.no_global)
    if args.holdout:
        keep = ~np.isin(clip, np.array(args.holdout))
        print(f"[holdout] excluding {sorted(args.holdout)}: "
              f"{len(X)-int(keep.sum())} of {len(X)} windows dropped")
        X, y, clip = X[keep], y[keep], clip[keep]
    Xf = featurize(X, n_pose)
    clips = sorted(set(clip))
    cnt = np.bincount(y, minlength=3)
    print(f"[data] {len(X)} windows  in_dim={Xf.shape[2]} "
          f"(pose {n_pose} + global {n_global} + present 1)  {len(clips)} clips")
    print(f"[data] dead={cnt[0]} transition={cnt[1]} active={cnt[2]}  "
          f"band=-{float(d['band_before_sec'])}s/+{float(d['band_after_sec'])}s "
          f"seed={int(d['seed'])}")

    accs = []
    for held in (clips if not args.skip_loco else []):
        tr, va = clip != held, clip == held
        if va.sum() == 0 or len(set(y[tr])) < 2:
            continue
        m = train_fold(Xf[tr], y[tr], args.epochs, args.seed, n_pose, n_global, args.hidden)
        p = predict(m, Xf[va])
        accs.append(report(y[va], p, f"hold {held}"))
    if accs:
        print(f"\n[LOCO] mean acc {np.mean(accs):.3f} over {len(accs)} clips")

    final = train_fold(Xf, y, args.epochs, args.seed, n_pose, n_global, args.hidden)
    print("\n[final] in-sample (diagnostic only):")
    report(y, predict(final, Xf), "train")
    torch.save({"state": final.state_dict(), "in_dim": Xf.shape[2],
                "hidden": args.hidden, "n_out": 3, "n_classes": 3,
                "n_pose": n_pose, "n_global": n_global,
                "win_sec": 2.0, "L": 60,
                "band_before_sec": float(d["band_before_sec"]),
                "band_after_sec": float(d["band_after_sec"]),
                "loco_acc": float(np.mean(accs)) if accs else float("nan")}, args.out)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
