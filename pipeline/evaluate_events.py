"""
evaluate_events.py
==================
Event-level evaluation of the dead/live classifier on a match timeline.

Window accuracy is a vanity metric here: dead and live are both long, mostly
easy states, so a model can score well on windows while still firing spurious
point-ends mid-rally. Measured: clip 22 scores 0.777 window accuracy and 0.387
event F1. What the product needs is the live->dead TRANSITION, at the right
time, without false fires during play.

Handles both heads:
  * binary (train_active.py)   — P(live), debounced by hysteresis N/M.
  * 3-class (train_state3.py)  — {dead, transition, active}. The transition
    class acts as the buffer: it neither confirms dead nor re-arms live, so a
    point-end fires only after n_dead confirmed DEAD windows following an
    active state. `transition` stays internal — the output is still binary
    point-end timestamps.

Either way an event is credited to the window where the model FIRST stopped
saying active, not where the decision was confirmed. Otherwise every detection
is late by the debounce length by construction, and the timing error measures
the smoother rather than the model.

Coverage may be non-contiguous (extract_timeline --mode rallies), so windows are
built through the frame_idx lookup and only emitted where the full 2s span is
covered. Live-minutes for the false-fire rate count only COVERED live frames —
otherwise the denominator includes time that was never scored.

Usage:
    python pipeline/evaluate_events.py --clip 21 --model /tmp/s3.pt --sweep
"""

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ground_truth import load_rallies, live_mask, transitions
from extract_timeline import TIMELINE_NPZ
from make_state_windows import clip_features, L, WIN_SEC


# ── window construction ──────────────────────────────────────────────────────

def timeline_windows(clip_dir, n_global, stride):
    """(X, ends, feat, frame_idx, fps, covered) — windows over covered spans."""
    feat, frame_idx, fps = clip_features(clip_dir, want_global=bool(n_global))
    total = int(frame_idx[-1]) + 1
    win_frames = max(2, round(fps * WIN_SEC))
    pos = np.full(total, -1, dtype=np.int64)
    inr = frame_idx < total
    pos[frame_idx[inr]] = np.flatnonzero(inr)
    covered = pos >= 0

    grid = np.linspace(0, win_frames - 1, L).round().astype(int)
    ends, rows = [], []
    for E in range(win_frames - 1, total, stride):
        s = E - win_frames + 1
        if pos[s] < 0 or pos[E] < 0 or pos[E] - pos[s] != win_frames - 1:
            continue
        ends.append(E)
        rows.append(feat[pos[s]:pos[E] + 1][grid])
    if not ends:
        raise SystemExit(f"{os.path.basename(clip_dir)}: no fully-covered windows")
    return np.asarray(rows, dtype=np.float32), np.asarray(ends), fps, covered, total


def score(model_path, X):
    """Returns (p_live, pred_state or None, checkpoint)."""
    import torch
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    from train_active import make_model, featurize
    Xf = featurize(X, int(ck.get("n_pose", 51)))
    if Xf.shape[2] != ck["in_dim"]:
        raise SystemExit(
            f"feature/model mismatch: timeline gives in_dim={Xf.shape[2]}, model "
            f"expects {ck['in_dim']}. Rebuild the timeline with the same streams "
            f"the model was trained on (n_global={ck.get('n_global')}).")
    n_out = int(ck.get("n_out", 1))
    m = make_model(in_dim=ck["in_dim"], hidden=ck.get("hidden", 48), n_out=n_out)
    m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad():
        z = m(torch.tensor(Xf))
        if n_out == 1:
            return torch.sigmoid(z).numpy(), None, ck
        p = torch.softmax(z, dim=1).numpy()
        return p[:, 2], p.argmax(1), ck        # P(active), hard state


# ── event extraction ─────────────────────────────────────────────────────────

def events_binary(is_live, ends, n_dead, m_live):
    """Point-ends from a binary live/dead sequence with hysteresis N/M."""
    state = bool(is_live[0])
    run_d = run_l = 0
    first_non_live = None
    ev = []
    for i, live in enumerate(is_live):
        if live:
            run_l += 1; run_d = 0; first_non_live = None
        else:
            run_d += 1; run_l = 0
            if first_non_live is None:
                first_non_live = i
        if state and run_d >= n_dead:
            ev.append(int(ends[first_non_live if first_non_live is not None else i]))
            state = False; run_d = 0
        elif not state and run_l >= m_live:
            state = True; run_l = 0
    return ev


def events_3state(pred, ends, n_dead, m_live):
    """Point-ends from {0 dead, 1 transition, 2 active}.

    `transition` is a genuine buffer: it does not count toward confirming dead
    and does not re-arm active. Only confirmed DEAD windows advance the counter,
    so the ambiguous band around a boundary no longer forces a decision.
    """
    state = "active" if pred[0] == 2 else "dead"
    run_d = run_a = 0
    first_non_active = None
    ev = []
    for i, s in enumerate(pred):
        if s == 2:                       # active
            run_a += 1; run_d = 0; first_non_active = None
        else:
            run_a = 0
            if first_non_active is None:
                first_non_active = i
            if s == 0:                   # dead confirms; transition only holds
                run_d += 1
        if state == "active" and run_d >= n_dead:
            ev.append(int(ends[first_non_active if first_non_active is not None else i]))
            state = "dead"; run_d = 0
        elif state == "dead" and run_a >= m_live:
            state = "active"; run_a = 0
    return ev


def match_events(pred, truth, tol_frames):
    """Greedy nearest-first one-to-one matching. Returns (pairs, fp, fn)."""
    cand = sorted(((abs(p - t), p, t) for p in pred for t in truth
                   if abs(p - t) <= tol_frames))
    used_p, used_t, pairs = set(), set(), []
    for _, p, t in cand:
        if p in used_p or t in used_t:
            continue
        used_p.add(p); used_t.add(t); pairs.append((p, t))
    return pairs, [p for p in pred if p not in used_p], [t for t in truth if t not in used_t]


# ── main evaluation ──────────────────────────────────────────────────────────

def evaluate(clip_dir, model_path, args):
    name = os.path.basename(clip_dir)
    import torch
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    n_global = int(ck.get("n_global", 0))
    n_out = int(ck.get("n_out", 1))

    X, ends, fps, covered, total = timeline_windows(clip_dir, n_global, args.stride)
    p_active, pred_state, _ = score(model_path, X)

    rallies = [r for r in load_rallies(clip_dir, fps) if r["end"] >= 0 and r["start"] < total]
    lm = live_mask(rallies, total)
    truth_ends = [e["frame"] for e in transitions(rallies)["point_end"] if 0 <= e["frame"] < total]
    tol = int(round(args.tol * fps))
    y_win = lm[ends]

    # Only covered live frames were ever scored, so only they can host a false fire.
    live_min = float((lm & covered).sum()) / fps / 60.0
    cov_pct = float(covered.mean())

    print(f"\n=== {name} — {total} frames @ {fps:.2f}fps ({total/fps/60:.1f} min), "
          f"coverage {cov_pct:.0%}, {len(rallies)} rallies, {len(truth_ends)} point-ends ===")
    print(f"[model] {'3-class' if n_out == 3 else 'binary'} in_dim={ck['in_dim']} "
          f"n_global={n_global} windows={len(ends)}")

    # ── window level (report only) ──
    if n_out == 3:
        acc3 = float((pred_state == np.where(lm[ends], 2, 0)).mean())
        share = np.bincount(pred_state, minlength=3) / len(pred_state)
        print(f"[window] predicted dead={share[0]:.2f} transition={share[1]:.2f} "
              f"active={share[2]:.2f}  (vs binarized GT acc={acc3:.3f}, "
              f"transition windows count as wrong by construction)")
    pred_live = p_active > args.thresh
    tp = int(((~pred_live) & (~y_win)).sum()); fp_ = int(((~pred_live) & y_win).sum())
    fn_ = int((pred_live & (~y_win)).sum())
    prec = tp / max(tp + fp_, 1); rec = tp / max(tp + fn_, 1)
    print(f"[window] n={len(ends)} acc={(pred_live == y_win).mean():.3f}  "
          f"dead P={prec:.3f} R={rec:.3f} F1={2*prec*rec/max(prec+rec,1e-9):.3f}")

    # ── event level (primary) ──
    grid = ([(args.n_dead, args.m_live)] if not args.sweep else
            [(n, m) for n in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24)
             for m in (2, 3, 5, 8, 12)])
    rows, best = [], None
    for n_dead, m_live in grid:
        if n_out == 3:
            pred = events_3state(pred_state, ends, n_dead, m_live)
        else:
            pred = events_binary(pred_live, ends, n_dead, m_live)
        pairs, fp_e, _ = match_events(pred, truth_ends, tol)
        P = len(pairs) / max(len(pred), 1); R = len(pairs) / max(len(truth_ends), 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        errs = [(p - t) / fps for p, t in pairs]
        med = float(np.median(errs)) if errs else float("nan")
        ff = [p for p in fp_e if 0 <= p < total and lm[p]]
        rows.append((n_dead, m_live, len(pred), len(pairs), P, R, F1, med,
                     len(ff), len(ff) / max(live_min, 1e-9)))
        if best is None or F1 > best[6]:
            best = rows[-1]

    print(f"\n[event] tol=+/-{args.tol}s stride={args.stride}f")
    print(f"{'N':>3} {'M':>3} {'pred':>5} {'hit':>4} {'P':>6} {'R':>6} {'F1':>6} "
          f"{'medErr':>7} {'FF':>4} {'FF/live-min':>12}")
    for r in rows:
        med_s = "    n/a" if np.isnan(r[7]) else f"{r[7]:>+7.2f}"
        print(f"{r[0]:>3} {r[1]:>3} {r[2]:>5} {r[3]:>4} {r[4]:>6.3f} {r[5]:>6.3f} "
              f"{r[6]:>6.3f} {med_s} {r[8]:>4} {r[9]:>12.2f}" + (" *" if r is best else ""))

    med_s = "n/a" if np.isnan(best[7]) else f"{best[7]:+.2f}s"
    print(f"\n[best] N={best[0]} M={best[1]}  point-end P={best[4]:.3f} R={best[5]:.3f} "
          f"F1={best[6]:.3f}  median timing {med_s}  "
          f"false fires {best[8]} ({best[9]:.2f}/live-min over {live_min:.1f} live-min)")
    return {"clip": name, "n_dead": best[0], "m_live": best[1], "precision": best[4],
            "recall": best[5], "f1": best[6], "median_err_s": best[7],
            "false_fires": best[8], "ff_per_live_min": best[9],
            "live_minutes": live_min, "n_truth": len(truth_ends),
            "coverage": cov_pct, "head": "3class" if n_out == 3 else "binary"}


def main():
    ap = argparse.ArgumentParser(description="Event-level dead/live evaluation")
    ap.add_argument("--data_root", default="/Volumes/Anya/Data")
    ap.add_argument("--clip", nargs="+", required=True)
    ap.add_argument("--model", default="/Volumes/Anya/Data/state3_model.pt")
    ap.add_argument("--stride", type=int, default=5, help="frames between window ends")
    ap.add_argument("--thresh", type=float, default=0.5, help="binary head only")
    ap.add_argument("--tol", type=float, default=2.0, help="match tolerance, seconds")
    ap.add_argument("--n_dead", type=int, default=4)
    ap.add_argument("--m_live", type=int, default=3)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    out = []
    for c in args.clip:
        d = os.path.join(args.data_root, c)
        if not os.path.isfile(os.path.join(d, TIMELINE_NPZ)):
            print(f"[skip] {c}: no {TIMELINE_NPZ} — run extract_timeline.py --clips {c}")
            continue
        out.append(evaluate(d, args.model, args))

    if len(out) > 1:
        print(f"\n[overall] point-end F1 {np.mean([o['f1'] for o in out]):.3f}  "
              f"false fires {np.mean([o['ff_per_live_min'] for o in out]):.2f}/live-min "
              f"over {len(out)} clips")
    if args.json_out and out:
        json.dump(out, open(args.json_out, "w"), indent=1)
        print(f"[done] -> {args.json_out}")


if __name__ == "__main__":
    main()
