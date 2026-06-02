"""
train_walking_rnn.py
====================
Walking vs. non-walking classifier for tennis videos.

Pipeline
--------
1. Read walking_labels.csv  (match, timestamp, label)
2. For each sample open /Volumes/Anya/Data/{match:02d}/snippet.mp4
3. Detect the near-side player bounding box per frame using the court
   homography + YOLO (same logic as FullTelemetryProvider._track_players)
4. Crop to the near-side box, run MediaPipe Pose, collect 33 landmark
   (x, y) pairs → 66 features per frame
5. Extract 1.5 s of frames starting at the labelled timestamp (~45 frames)
6. Train a 2-layer LSTM binary classifier; report train / val accuracy.

Usage
-----
  python -m src.ai.train_walking_rnn
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT      = Path("/Volumes/Anya/Data")
LABELS_CSV     = Path(__file__).parent / "walking_labels.csv"
CLIP_DURATION  = 1.5          # seconds to extract after each timestamp
ANALYSIS_SIZE  = (960, 540)
NUM_LANDMARKS     = 33
POS_PER_FRAME     = NUM_LANDMARKS * 2   # x, y positions
VEL_PER_FRAME     = NUM_LANDMARKS * 2   # Δx, Δy velocities
FEAT_PER_FRAME    = POS_PER_FRAME + VEL_PER_FRAME  # 132 total
POSE_MODEL        = str(Path(__file__).parent / "pose_landmarker_full.task")
FEATURE_CACHE     = Path(__file__).parent / "walking_features.npz"

# LSTM hyperparameters
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 16
EPOCHS       = 60
LR           = 1e-3
VAL_FRAC     = 0.20
RANDOM_SEED  = 42

# Court / player detection (mirrors Config in utilities.py)
COURT_WIDTH_FT   = 27.0
COURT_LENGTH_FT  = 78.0
NEAR_PAD_FT      = 3.0
PLAYER_CLASS_IDX = 0
PLAYER_CONF      = 0.5
PLAYER_IMGSZ     = 960


# ─────────────────────────────────────────────────────────────────────────────
# Near-player detector  (lightweight — no active-zone, no far-player strip)
# ─────────────────────────────────────────────────────────────────────────────

class NearPlayerDetector:
    """
    Loads court geometry from the per-video cache written by init_court(),
    builds the homography, and detects the near-side player per frame using
    the same classification logic as FullTelemetryProvider._track_players().
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.H = self._load_homography(video_path)

        from ultralytics import YOLO
        # Reuse a single shared model across instances via a module-level cache
        if not hasattr(NearPlayerDetector, "_model"):
            NearPlayerDetector._model = YOLO("yolo26n.pt")
        self.player_model = NearPlayerDetector._model

    # ── homography ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_homography(video_path: str):
        video_dir  = os.path.dirname(os.path.abspath(video_path))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cache_path = os.path.join(video_dir, f"{video_name}_court_cache.json")
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"Court cache missing: {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        pts = [tuple(p) for p in cached["points"]]   # BL, BR, TR, TL
        BL, BR, TR, TL = pts
        dst = np.array([
            [0,               0],
            [COURT_WIDTH_FT,  0],
            [COURT_WIDTH_FT,  COURT_LENGTH_FT],
            [0,               COURT_LENGTH_FT],
        ], dtype=np.float32)
        src = np.array([BL, BR, TR, TL], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H

    def _to_world(self, px_x: float, px_y: float):
        pt = np.array([[[px_x, px_y]]], dtype=np.float32)
        w  = cv2.perspectiveTransform(pt, self.H)
        return float(w[0][0][0]), float(w[0][0][1])

    # ── detection ─────────────────────────────────────────────────────────────

    def detect(self, frame) -> tuple:
        """Return (x1,y1,x2,y2) of the near-side player, or None."""
        results = self.player_model(frame, verbose=False,
                                    conf=PLAYER_CONF, imgsz=PLAYER_IMGSZ)
        candidates = []
        if results and results[0].boxes:
            for b in results[0].boxes:
                if int(b.cls[0]) != PLAYER_CLASS_IDX:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cx = (x1 + x2) / 2.0
                wx, wy = self._to_world(cx, float(y2))
                candidates.append((x1, y1, x2, y2, wx, wy))

        # Near player is closer to world-y = 0 than to COURT_LENGTH_FT
        near_cands = [
            c for c in candidates
            if (abs(c[5]) < abs(c[5] - COURT_LENGTH_FT) and
                -NEAR_PAD_FT <= c[4] <= COURT_WIDTH_FT + NEAR_PAD_FT)
        ]
        if not near_cands:
            return None
        best = min(near_cands, key=lambda c: abs(c[5]))
        return best[:4]   # (x1, y1, x2, y2)


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe pose extractor  (Tasks API — mediapipe >= 0.10)
# ─────────────────────────────────────────────────────────────────────────────

from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision

_BaseOptions          = _mp_tasks.BaseOptions
_PoseLandmarker       = _mp_vision.PoseLandmarker
_PoseLandmarkerOpts   = _mp_vision.PoseLandmarkerOptions
_RunningMode          = _mp_vision.RunningMode


def make_pose():
    """Return a PoseLandmarker configured for video-frame-by-frame inference."""
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
    """
    Run MediaPipe Pose on the crop defined by box (x1,y1,x2,y2).
    Returns array of shape (NUM_LANDMARKS*2,) with (x,y) normalised to [0,1]
    within the crop, or zeros if detection fails.
    """
    zeros = np.zeros(POS_PER_FRAME, dtype=np.float32)
    if box is None:
        return zeros
    fh, fw = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1);  y1 = max(0, y1)
    x2 = min(fw, x2); y2 = min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return zeros
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return zeros
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
    result   = pose_instance.detect(mp_image)
    if not result.pose_landmarks:
        return zeros
    feats = []
    for lm in result.pose_landmarks[0]:
        feats.extend([lm.x, lm.y])
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def load_labels():
    rows = []
    with open(LABELS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["match"]), float(row["timestamp"]),
                         int(row["label"])))
    return rows


def extract_sample(video_path: str, timestamp: float,
                   detector: NearPlayerDetector,
                   pose, seq_len: int) -> np.ndarray:
    """
    Extract seq_len frames starting at timestamp from video_path.
    Returns array of shape (seq_len, FEAT_PER_FRAME) where each row is
    [pos_x0, pos_y0, ..., pos_x32, pos_y32, vel_x0, vel_y0, ..., vel_x32, vel_y32].
    Velocity at frame t = position[t] - position[t-1]; frame 0 gets zero velocity.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    start_frame = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    positions = []   # raw (x,y) landmark arrays, shape (T, POS_PER_FRAME)
    last_box  = None
    frames_read = 0

    while frames_read < seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, ANALYSIS_SIZE)
        box = detector.detect(frame)
        if box is not None:
            last_box = box
        pos = extract_landmarks(pose, frame, box if box is not None else last_box)
        positions.append(pos)
        frames_read += 1

    cap.release()

    zero_pos = np.zeros(POS_PER_FRAME, dtype=np.float32)
    while len(positions) < seq_len:
        positions.append(zero_pos.copy())

    pos_arr = np.stack(positions[:seq_len])   # (seq_len, POS_PER_FRAME)

    # Compute frame-to-frame velocity; frame 0 velocity = 0
    vel_arr = np.zeros_like(pos_arr)
    vel_arr[1:] = pos_arr[1:] - pos_arr[:-1]
    # Zero-out velocity when the position itself was zero (no detection)
    no_detect = (pos_arr == 0).all(axis=1, keepdims=True)
    vel_arr[no_detect.squeeze(1)] = 0.0

    return np.concatenate([pos_arr, vel_arr], axis=1)   # (seq_len, FEAT_PER_FRAME)


def build_dataset(labels, seq_len: int):
    """
    Returns X of shape (N, seq_len, FEAT_PER_FRAME) and y of shape (N,).
    Loads from FEATURE_CACHE when the cache shape matches; otherwise extracts
    from video and saves the cache for subsequent runs.
    """
    if FEATURE_CACHE.exists():
        cached = np.load(FEATURE_CACHE)
        X_c, y_c = cached["X"], cached["y"]
        if X_c.shape[1] == seq_len and X_c.shape[2] == FEAT_PER_FRAME:
            print(f"[INFO] Loaded feature cache  {X_c.shape}  from {FEATURE_CACHE}")
            return X_c.astype(np.float32), y_c.astype(np.float32)
        print(f"[INFO] Cache shape mismatch {X_c.shape} vs expected "
              f"(N, {seq_len}, {FEAT_PER_FRAME}) — re-extracting.")

    from collections import defaultdict
    by_match = defaultdict(list)
    for match, ts, label in labels:
        by_match[match].append((ts, label))

    X_list, y_list = [], []

    for match, samples in sorted(by_match.items()):
        video_path = str(DATA_ROOT / f"{match:02d}" / "snippet.mp4")
        if not os.path.isfile(video_path):
            print(f"  [SKIP] video not found: {video_path}")
            continue

        print(f"\n[MATCH {match:02d}]  {len(samples)} samples  — {video_path}")
        try:
            detector = NearPlayerDetector(video_path)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        pose = make_pose()
        for i, (ts, label) in enumerate(samples):
            t0 = time.time()
            seq = extract_sample(video_path, ts, detector, pose, seq_len)
            elapsed = time.time() - t0
            walking = "walk" if label == 1 else "idle"
            pos_nz = int(np.any(seq[:, :POS_PER_FRAME] != 0, axis=1).sum())
            print(f"  [{i+1:3d}/{len(samples)}]  t={ts:7.2f}s  {walking}  "
                  f"({elapsed:.1f}s)  non-zero frames: {pos_nz}/{seq_len}")
            X_list.append(seq)
            y_list.append(label)
        pose.close()

    if not X_list:
        raise RuntimeError("No samples extracted — check video paths and court caches.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    np.savez_compressed(FEATURE_CACHE, X=X, y=y)
    print(f"[INFO] Feature cache saved to {FEATURE_CACHE}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate features for logistic regression
# ─────────────────────────────────────────────────────────────────────────────

def make_aggregate_features(X: np.ndarray) -> np.ndarray:
    """
    Collapse (N, seq_len, FEAT_PER_FRAME) → (N, agg_dim) by computing
    mean and std of each feature over detected frames only (skip zero rows).

    Features used: velocity half only (FEAT_PER_FRAME // 2 : end) because
    velocity captures the walking rhythm more directly than raw position.
    Also includes mean/std of position for context.
    """
    N = X.shape[0]
    agg = []
    for seq in X:
        # Identify frames where the position was detected (non-zero)
        detected = np.any(seq[:, :POS_PER_FRAME] != 0, axis=1)
        rows = seq[detected] if detected.any() else seq
        agg.append(np.concatenate([rows.mean(axis=0), rows.std(axis=0)]))
    return np.stack(agg).astype(np.float32)   # (N, FEAT_PER_FRAME * 2)


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_logreg(X: np.ndarray, y: np.ndarray):
    print(f"\n{'='*60}")
    print("LOGISTIC REGRESSION  (aggregate pos + vel statistics)")
    print("=" * 60)

    X_agg = make_aggregate_features(X)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_agg, y, test_size=VAL_FRAC, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {len(y_tr)}  |  Val: {len(y_val)}  |  Features: {X_agg.shape[1]}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=0.1, max_iter=1000,
                                      class_weight="balanced",
                                      random_state=RANDOM_SEED)),
    ])
    pipe.fit(X_tr, y_tr)

    val_preds = pipe.predict(X_val)
    val_acc   = (val_preds == y_val).mean()
    print(f"Val accuracy : {val_acc:.3f}")
    print("\nValidation classification report:")
    print(classification_report(y_val.astype(int), val_preds.astype(int),
                                 target_names=["non-walking", "walking"],
                                 digits=3))
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# LSTM model
# ─────────────────────────────────────────────────────────────────────────────

class WalkingLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        # use last timestep
        last = out[:, -1, :]
        return self.head(self.drop(last)).squeeze(-1)   # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray):
    print(f"\n{'='*60}")
    print(f"Dataset  :  {len(y)} samples  "
          f"({int(y.sum())} walking / {int((1-y).sum())} non-walking)")
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

    model = WalkingLSTM(FEAT_PER_FRAME, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # Class imbalance weight
    pos_weight = torch.tensor([(1 - y_tr.mean()) / (y_tr.mean() + 1e-6)]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            preds       = (logits.sigmoid() >= 0.5).float()
            tr_correct += (preds == yb).sum().item()
            tr_total   += len(yb)

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits  = model(xb)
                loss    = criterion(logits, yb)
                val_loss    += loss.item() * len(yb)
                preds        = (logits.sigmoid() >= 0.5).float()
                val_correct += (preds == yb).sum().item()
                val_total   += len(yb)

        tr_acc  = tr_correct  / tr_total
        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train loss={tr_loss/tr_total:.4f} acc={tr_acc:.3f}  "
                  f"val loss={val_loss/val_total:.4f} acc={val_acc:.3f}")

    print(f"\nBest val accuracy : {best_val_acc:.3f}")

    # ── final report ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds  = (logits.sigmoid() >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(yb.long().tolist())

    print("\nValidation classification report:")
    print(classification_report(all_true, all_preds,
                                 target_names=["non-walking", "walking"],
                                 digits=3))

    # Save model
    save_path = Path(__file__).parent / "walking_lstm.pt"
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
    first_video = str(DATA_ROOT / "21" / "snippet.mp4")
    cap = cv2.VideoCapture(first_video)
    fps_probe = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    seq_len = max(1, int(round(CLIP_DURATION * fps_probe)))
    print(f"[INFO] fps={fps_probe:.2f}  seq_len={seq_len} frames  "
          f"features/frame={FEAT_PER_FRAME} (pos+vel)")

    print("[INFO] Loading labels …")
    labels = load_labels()
    print(f"[INFO] {len(labels)} labelled samples across "
          f"{len(set(m for m,_,_ in labels))} videos")

    print("[INFO] Extracting / loading features …")
    X, y = build_dataset(labels, seq_len)

    # ── Model 1: Logistic regression on aggregate statistics ─────────────────
    train_logreg(X, y)

    # ── Model 2: LSTM with position + velocity features ──────────────────────
    print(f"\n{'='*60}")
    print("LSTM  (position + velocity features, 132 features/frame)")
    print("=" * 60)
    train(X, y)


if __name__ == "__main__":
    main()
