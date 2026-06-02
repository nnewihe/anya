"""
train_serve_rnn.py
==================
Side-aware serve vs. non-serve classifier.

Pipeline
--------
1. Load serve timestamps from snippet_serve_events.json; cross-reference
   ground_truth.json rally frame ranges to determine serving side
   ("near" or "far").  Events whose frame_id falls outside every rally
   range are skipped (incomplete ground-truth coverage).
2. Positive samples  : 2 s window starting at each serve timestamp.
   Near-side serves   → near player kinematics.
   Far-side serves    → far player kinematics.
3. Negative samples  : 2 s window starting at the midpoint between each
   consecutive serve pair (gap ≥ 6 s).  Each midpoint produces TWO
   negatives: one with near-player kinematics, one with far-player
   kinematics (both players are "not serving" at that moment).
4. For each sample: detect the appropriate player box, crop, run
   MediaPipe Pose, collect 33 landmark (x, y) per frame.
   Append frame-to-frame velocity (Δx, Δy) → 132 features/frame × 60 frames.
5. Train a 2-layer LSTM binary classifier; save to serve_lstm.pt.

Usage
-----
  python -m src.ai.train_serve_rnn
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT         = Path("/Volumes/Anya/Data")
CLIP_DURATION     = 2.0          # seconds after timestamp
MIN_SERVE_GAP_SEC = 6.0          # skip negative midpoints from gaps narrower than this
ANALYSIS_SIZE     = (960, 540)
NUM_LANDMARKS     = 33
POS_PER_FRAME     = NUM_LANDMARKS * 2   # 66  (x, y)
VEL_PER_FRAME     = NUM_LANDMARKS * 2   # 66  (Δx, Δy)
FEAT_PER_FRAME    = POS_PER_FRAME + VEL_PER_FRAME  # 132
POSE_MODEL        = str(Path(__file__).parent / "pose_landmarker_full.task")
FEATURE_CACHE     = Path(__file__).parent / "serve_features.npz"

# LSTM hyperparameters
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 16
EPOCHS       = 60
LR           = 1e-3
VAL_FRAC     = 0.20
RANDOM_SEED  = 42

# Court / player detection
COURT_WIDTH_FT   = 27.0
COURT_LENGTH_FT  = 78.0
NEAR_PAD_FT      = 3.0
PLAYER_CLASS_IDX = 0
PLAYER_CONF      = 0.5
PLAYER_IMGSZ     = 960
FAR_STRIP_PAD_PX = 10   # extra pixels below far baseline


# ─────────────────────────────────────────────────────────────────────────────
# Shared homography loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_court_data(video_path: str):
    """Returns (H, pts) where pts = [BL, BR, TR, TL] in pixel space."""
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(video_dir, f"{video_name}_court_cache.json")
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Court cache missing: {cache_path}")
    with open(cache_path) as f:
        cached = json.load(f)
    pts = [tuple(p) for p in cached["points"]]   # [BL, BR, TR, TL]
    BL, BR, TR, TL = pts
    dst = np.array([[0, 0], [COURT_WIDTH_FT, 0],
                    [COURT_WIDTH_FT, COURT_LENGTH_FT], [0, COURT_LENGTH_FT]],
                   dtype=np.float32)
    src = np.array([BL, BR, TR, TL], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H, pts


def _get_yolo():
    from ultralytics import YOLO
    if not hasattr(_get_yolo, "_model"):
        _get_yolo._model = YOLO("yolo26n.pt")
    return _get_yolo._model


# ─────────────────────────────────────────────────────────────────────────────
# Near-player detector
# ─────────────────────────────────────────────────────────────────────────────

class NearPlayerDetector:
    def __init__(self, video_path: str):
        self.H, _ = _load_court_data(video_path)
        self.player_model = _get_yolo()

    def _to_world(self, px_x, px_y):
        pt = np.array([[[px_x, px_y]]], dtype=np.float32)
        w  = cv2.perspectiveTransform(pt, self.H)
        return float(w[0][0][0]), float(w[0][0][1])

    def detect(self, frame):
        results = self.player_model(frame, verbose=False,
                                    conf=PLAYER_CONF, imgsz=PLAYER_IMGSZ)
        candidates = []
        if results and results[0].boxes:
            for b in results[0].boxes:
                if int(b.cls[0]) != PLAYER_CLASS_IDX:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                wx, wy = self._to_world((x1 + x2) / 2.0, float(y2))
                candidates.append((x1, y1, x2, y2, wx, wy))
        near = [c for c in candidates
                if abs(c[5]) < abs(c[5] - COURT_LENGTH_FT)
                and -NEAR_PAD_FT <= c[4] <= COURT_WIDTH_FT + NEAR_PAD_FT]
        if not near:
            return None
        best = min(near, key=lambda c: abs(c[5]))
        return best[:4]


# ─────────────────────────────────────────────────────────────────────────────
# Far-player detector  (strip-based, mirrors full_anya._track_far_player_strip)
# ─────────────────────────────────────────────────────────────────────────────

class FarPlayerDetector:
    """Detects the far-side player by running YOLO on the far-baseline strip."""

    def __init__(self, video_path: str):
        _, pts = _load_court_data(video_path)
        BL, BR, TR, TL = pts   # noqa: F841
        x1 = float(min(TL[0], TR[0]))
        x2 = float(max(TL[0], TR[0]))
        y_baseline = (TL[1] + TR[1]) / 2.0
        # Strip: from y_baseline-50 to y_baseline+FAR_STRIP_PAD_PX
        self._strip = (x1, y_baseline - 50.0, x2, y_baseline + FAR_STRIP_PAD_PX)
        self.player_model = _get_yolo()

    def detect(self, frame):
        fh, fw = frame.shape[:2]
        sx1, sy1, sx2, sy2 = self._strip
        rx1 = int(max(0, sx1))
        ry1 = int(max(0, sy1))
        rx2 = int(min(fw, sx2))
        ry2 = int(min(fh, sy2))
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None
        results = self.player_model(roi, verbose=False,
                                    conf=PLAYER_CONF, imgsz=PLAYER_IMGSZ)
        if not (results and results[0].boxes):
            return None
        best_conf, best_box = -1.0, None
        for b in results[0].boxes:
            if int(b.cls[0]) != PLAYER_CLASS_IDX:
                continue
            lx1, ly1, lx2, ly2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box  = (rx1 + lx1, ry1 + ly1, rx1 + lx2, ry1 + ly2)
        return best_box


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe pose extractor  (Tasks API)
# ─────────────────────────────────────────────────────────────────────────────

from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision

_BaseOptions        = _mp_tasks.BaseOptions
_PoseLandmarker     = _mp_vision.PoseLandmarker
_PoseLandmarkerOpts = _mp_vision.PoseLandmarkerOptions
_RunningMode        = _mp_vision.RunningMode


def make_pose():
    opts = _PoseLandmarkerOpts(
        base_options=_BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=_RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return _PoseLandmarker.create_from_options(opts)


def extract_landmarks(pose_instance, frame_bgr, box) -> np.ndarray:
    zeros = np.zeros(POS_PER_FRAME, dtype=np.float32)
    if box is None:
        return zeros
    fh, fw = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return zeros
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return zeros
    try:
        rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose_instance.detect(mp_img)
        if not result.pose_landmarks:
            return zeros
        feats = []
        for lm in result.pose_landmarks[0]:
            feats.extend([lm.x, lm.y])
        return np.array(feats, dtype=np.float32)
    except Exception:
        return zeros


# ─────────────────────────────────────────────────────────────────────────────
# Label generation  (side-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _get_side_from_ground_truth(frame_id: int, rallies: list) -> str:
    for r in rallies:
        if r["start"] <= frame_id <= r["end"]:
            return r["serve"]
    return "unknown"


def load_labels():
    """
    Returns list of (match_id, timestamp, label, side) tuples.
      label=1  serve positive  — side is "near" or "far"
      label=0  non-serve neg   — side is "near" or "far" (both generated per midpoint)
    Events whose side cannot be determined are skipped.
    """
    samples = []

    for match_dir in sorted(DATA_ROOT.iterdir()):
        if not match_dir.is_dir():
            continue
        events_path = match_dir / "snippet_serve_events.json"
        gt_path     = match_dir / "ground_truth.json"
        video_path  = match_dir / "snippet.mp4"
        if not events_path.exists() or not video_path.exists():
            continue
        try:
            match_id = int(match_dir.name)
        except ValueError:
            continue

        with open(events_path) as f:
            events = json.load(f)

        rallies = []
        if gt_path.exists():
            with open(gt_path) as f:
                rallies = json.load(f).get("rallies", [])

        # Sort events by timestamp; keep frame_id for side lookup
        events_sorted = sorted(events, key=lambda e: e["timestamp"])
        timestamps    = [e["timestamp"] for e in events_sorted]

        # Positives: skip events with unknown side
        for e in events_sorted:
            side = _get_side_from_ground_truth(e["frame_id"], rallies)
            if side == "unknown":
                continue
            samples.append((match_id, e["timestamp"], 1, side))

        # Negatives: one near + one far per valid midpoint
        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap < MIN_SERVE_GAP_SEC:
                continue
            mid = (timestamps[i] + timestamps[i + 1]) / 2.0
            samples.append((match_id, mid, 0, "near"))
            samples.append((match_id, mid, 0, "far"))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_sample(video_path: str, timestamp: float,
                   detector, pose, seq_len: int) -> np.ndarray:
    """detector is either NearPlayerDetector or FarPlayerDetector."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))

    positions, last_box, frames_read = [], None, 0
    while frames_read < seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, ANALYSIS_SIZE)
        box = detector.detect(frame)
        if box is not None:
            last_box = box
        positions.append(extract_landmarks(pose, frame,
                                           box if box is not None else last_box))
        frames_read += 1
    cap.release()

    zero = np.zeros(POS_PER_FRAME, dtype=np.float32)
    while len(positions) < seq_len:
        positions.append(zero.copy())

    pos_arr = np.stack(positions[:seq_len])          # (T, 66)
    vel_arr = np.zeros_like(pos_arr)
    vel_arr[1:] = pos_arr[1:] - pos_arr[:-1]
    vel_arr[(pos_arr == 0).all(axis=1)] = 0.0
    return np.concatenate([pos_arr, vel_arr], axis=1) # (T, 132)


def build_dataset(labels, seq_len: int):
    if FEATURE_CACHE.exists():
        cached = np.load(FEATURE_CACHE)
        X_c, y_c = cached["X"], cached["y"]
        if X_c.shape[1] == seq_len and X_c.shape[2] == FEAT_PER_FRAME:
            print(f"[INFO] Loaded feature cache  {X_c.shape}  from {FEATURE_CACHE}")
            return X_c.astype(np.float32), y_c.astype(np.float32)
        print("[INFO] Cache shape mismatch — re-extracting.")

    # Group by (match, side) so we open each video once per side
    by_match_side = defaultdict(list)
    for match, ts, label, side in labels:
        by_match_side[(match, side)].append((ts, label))

    X_list, y_list = [], []

    for (match, side), samples in sorted(by_match_side.items()):
        video_path = str(DATA_ROOT / f"{match:02d}" / "snippet.mp4")
        if not os.path.isfile(video_path):
            print(f"  [SKIP] video not found: {video_path}")
            continue
        print(f"\n[MATCH {match:02d} / {side}]  {len(samples)} samples")
        try:
            detector = (NearPlayerDetector(video_path) if side == "near"
                        else FarPlayerDetector(video_path))
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        pose = make_pose()
        for i, (ts, label) in enumerate(samples):
            t0      = time.time()
            seq     = extract_sample(video_path, ts, detector, pose, seq_len)
            elapsed = time.time() - t0
            tag     = "serve" if label == 1 else "neg  "
            nz      = int(np.any(seq[:, :POS_PER_FRAME] != 0, axis=1).sum())
            print(f"  [{i+1:3d}/{len(samples)}]  t={ts:7.2f}s  {tag}  "
                  f"({elapsed:.1f}s)  non-zero frames: {nz}/{seq_len}")
            X_list.append(seq)
            y_list.append(label)
        pose.close()

    if not X_list:
        raise RuntimeError("No samples extracted.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    np.savez_compressed(FEATURE_CACHE, X=X, y=y)
    print(f"[INFO] Feature cache saved to {FEATURE_CACHE}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────────

class ServeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :])).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray):
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    print(f"\n{'='*60}")
    print(f"Dataset  :  {len(y)} samples  ({n_pos} serve / {n_neg} non-serve)")
    print(f"Sequence :  {X.shape[1]} frames × {X.shape[2]} features")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_FRAC, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train    :  {len(y_tr)}  |  Val: {len(y_val)}")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   :  {device}")

    def to_loader(Xa, ya, shuffle=True):
        ds = TensorDataset(torch.tensor(Xa), torch.tensor(ya))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(X_tr, y_tr)
    val_loader   = to_loader(X_val, y_val, shuffle=False)

    model      = ServeLSTM(FEAT_PER_FRAME, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor([(1 - y_tr.mean()) / (y_tr.mean() + 1e-6)]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            tr_correct += ((logits.sigmoid() >= 0.5).float() == yb).sum().item()
            tr_total   += len(yb)

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits      = model(xb)
                val_loss    += criterion(logits, yb).item() * len(yb)
                val_correct += ((logits.sigmoid() >= 0.5).float() == yb).sum().item()
                val_total   += len(yb)

        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train loss={tr_loss/tr_total:.4f} acc={tr_correct/tr_total:.3f}  "
                  f"val loss={val_loss/val_total:.4f} acc={val_acc:.3f}")

    print(f"\nBest val accuracy : {best_val_acc:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb.to(device))
            all_preds.extend((logits.sigmoid() >= 0.5).long().cpu().tolist())
            all_true.extend(yb.long().tolist())

    print("\nValidation classification report:")
    print(classification_report(all_true, all_preds,
                                 target_names=["non-serve", "serve"], digits=3))

    save_path = Path(__file__).parent / "serve_lstm.pt"
    torch.save({
        "model_state": best_state,
        "input_size":  FEAT_PER_FRAME,
        "hidden_size": HIDDEN_SIZE,
        "num_layers":  NUM_LAYERS,
        "dropout":     DROPOUT,
        "seq_len":     X.shape[1],
    }, save_path)
    print(f"Model saved to {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(str(DATA_ROOT / "22" / "snippet.mp4"))
    fps_probe = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    seq_len = max(1, int(round(CLIP_DURATION * fps_probe)))
    print(f"[INFO] fps={fps_probe:.2f}  seq_len={seq_len} frames  "
          f"features/frame={FEAT_PER_FRAME}")

    labels = load_labels()
    n_pos  = sum(1 for _, _, l, _ in labels if l == 1)
    n_neg  = sum(1 for _, _, l, _ in labels if l == 0)
    near_pos = sum(1 for _, _, l, s in labels if l == 1 and s == "near")
    far_pos  = sum(1 for _, _, l, s in labels if l == 1 and s == "far")
    matches  = sorted(set(m for m, _, _, _ in labels))
    print(f"[INFO] {len(labels)} samples  "
          f"({n_pos} serve [{near_pos} near / {far_pos} far] / {n_neg} non-serve)  "
          f"across matches: {matches}")

    X, y = build_dataset(labels, seq_len)
    train(X, y)


if __name__ == "__main__":
    main()
