"""
evaluate.py
===========
Scoring helpers shared by training and inference.

Three views of the same prediction, because they answer different questions:

  frame  — per-sample P/R/F1. Sensitive to boundary jitter, and the labels were
           made live at playback speed, so a hand-labelled edge is worth about
           half a second of reaction time. ``boundary_guard`` optionally drops
           samples within that band from scoring.
  second — one prediction per elapsed second (majority vote). This is the unit
           the earlier hand-tuned detector reported, so it is the comparable
           number.
  event  — matched intervals. What a downstream consumer actually wants: did we
           find the walk, and how many walks did we invent.
"""

import numpy as np


def prf(y_true, y_pred):
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f1, "accuracy": acc}


def fbeta(score, beta=1.0):
    """F-beta from a ``prf`` result. beta>1 weights recall, which is what a
    dead-time cutter wants: a walk called late is trimmed footage, a walk missed
    is dead time left in the cut."""
    p, r = score["precision"], score["recall"]
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0


def boundary_mask(y, fps, guard_s):
    """False within ``guard_s`` of any label transition (live-labelling slack)."""
    y = np.asarray(y, bool)
    keep = np.ones(len(y), bool)
    if guard_s <= 0:
        return keep
    g = int(round(guard_s * fps))
    edges = np.flatnonzero(np.diff(y.astype(int)) != 0)
    for e in edges:
        keep[max(0, e - g):min(len(y), e + g + 1)] = False
    return keep


def to_intervals(mask):
    """Contiguous True runs of a boolean array as [(start, end_inclusive)]."""
    mask = np.asarray(mask, bool)
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def hysteresis(prob, hi, lo, fps, min_dur_s=1.0, max_gap_s=0.5):
    """Two-threshold gate plus gap-fill and minimum duration -> boolean mask."""
    prob = np.nan_to_num(np.asarray(prob, float), nan=0.0)
    on = prob >= hi
    keep = prob >= lo
    mask = np.zeros(len(prob), bool)
    for a, b in to_intervals(keep):
        if on[a:b + 1].any():
            mask[a:b + 1] = True
    # bridge short holes
    gap = int(round(max_gap_s * fps))
    for a, b in to_intervals(~mask):
        if a > 0 and b < len(mask) - 1 and (b - a + 1) <= gap:
            mask[a:b + 1] = True
    md = int(round(min_dur_s * fps))
    for a, b in to_intervals(mask):
        if (b - a + 1) < md:
            mask[a:b + 1] = False
    return mask


def smooth_prob(prob, fps, width_s):
    """Centred median filter on the probability trace (0 disables)."""
    prob = np.asarray(prob, float)
    w = int(round(width_s * fps))
    if w < 3:
        return prob.copy()
    w += 1 - w % 2
    pad = w // 2
    p = np.pad(prob, pad, mode="edge")
    out = np.empty_like(prob)
    for i in range(len(prob)):
        out[i] = np.median(p[i:i + w])
    return out


def viterbi(prob, switch_cost, bias=0.0, eps=1e-6):
    """Two-state HMM decode of the probability trace.

    The principled version of hysteresis: one knob (``switch_cost``, the log-odds
    penalty for changing state) instead of three, so it transfers between folds
    far more stably than a tuned threshold pair.
    """
    p = np.clip(np.nan_to_num(np.asarray(prob, float), nan=0.0), eps, 1 - eps)
    e1 = np.log(p) + bias
    e0 = np.log(1 - p)
    n = len(p)
    dp = np.zeros((n, 2))
    bk = np.zeros((n, 2), dtype=np.int8)
    dp[0] = [e0[0], e1[0]]
    for t in range(1, n):
        for s, em in ((0, e0[t]), (1, e1[t])):
            stay = dp[t - 1, s]
            swap = dp[t - 1, 1 - s] - switch_cost
            if stay >= swap:
                dp[t, s], bk[t, s] = stay + em, s
            else:
                dp[t, s], bk[t, s] = swap + em, 1 - s
    out = np.zeros(n, bool)
    s = int(np.argmax(dp[-1]))
    for t in range(n - 1, -1, -1):
        out[t] = bool(s)
        s = bk[t, s]
    return out


def apply_post(prob, fps, cfg):
    """Turn a probability trace into a boolean mask under a post-proc config.

    Lives here rather than in train.py because predict.py needs the same
    dispatch: a bundle's `post` may be any of the three kinds the sweep can
    pick, and a consumer that assumes one of them breaks the day the sweep
    picks another.
    """
    p = smooth_prob(prob, fps, cfg.get("smooth_s", 0.0))
    kind = cfg["kind"]
    if kind == "threshold":
        mask = p >= cfg["thr"]
    elif kind == "hysteresis":
        return hysteresis(p, cfg["hi"], cfg["lo"], fps,
                          min_dur_s=cfg["min_dur_s"],
                          max_gap_s=cfg.get("max_gap_s", 0.5))
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


def to_seconds(y, fps, n_seconds=None):
    """Majority vote of a frame mask into one label per elapsed second."""
    y = np.asarray(y, bool)
    n = n_seconds if n_seconds is not None else int(np.ceil(len(y) / fps))
    out = np.zeros(n, bool)
    for s in range(n):
        a, b = int(round(s * fps)), int(round((s + 1) * fps))
        seg = y[a:min(b, len(y))]
        if len(seg):
            out[s] = seg.mean() >= 0.5
    return out


def event_scores(y_true, y_pred, fps, iou_min=0.3):
    """Greedy one-to-one interval matching by IoU."""
    gt = to_intervals(y_true)
    pr = to_intervals(y_pred)
    pairs = []
    for i, (a1, b1) in enumerate(gt):
        for j, (a2, b2) in enumerate(pr):
            inter = max(0, min(b1, b2) - max(a1, a2) + 1)
            if inter <= 0:
                continue
            union = (b1 - a1 + 1) + (b2 - a2 + 1) - inter
            pairs.append((inter / union, i, j))
    pairs.sort(reverse=True)
    used_g, used_p, matched = set(), set(), []
    for iou, i, j in pairs:
        if iou < iou_min or i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched.append((i, j, iou))
    tp = len(matched)
    total_s = len(y_true) / fps
    onset = [abs(pr[j][0] - gt[i][0]) / fps for i, j, _ in matched]
    offset = [abs(pr[j][1] - gt[i][1]) / fps for i, j, _ in matched]
    return {
        "n_true": len(gt), "n_pred": len(pr), "matched": tp,
        "recall": tp / len(gt) if gt else 0.0,
        "precision": tp / len(pr) if pr else 0.0,
        "false_events_per_min": (len(pr) - tp) / (total_s / 60.0),
        "mean_iou": float(np.mean([m[2] for m in matched])) if matched else 0.0,
        "onset_mae_s": float(np.mean(onset)) if onset else float("nan"),
        "offset_mae_s": float(np.mean(offset)) if offset else float("nan"),
    }
