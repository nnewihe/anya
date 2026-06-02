"""
train_serve_gcn.py
==================
Serve vs. non-serve binary classifier using Spatio-Temporal Graph
Convolutional Networks (ST-GCN).

Upper-body skeleton graph — 9 nodes relevant to serve biomechanics:

  node  MediaPipe  joint
    0       0      nose
    1      11      left shoulder
    2      12      right shoulder
    3      13      left elbow
    4      14      right elbow
    5      15      left wrist
    6      16      right wrist
    7      23      left hip
    8      24      right hip

Anatomical edges (bidirectional):
  0-1, 0-2          nose → shoulders
  1-2               shoulder bridge
  1-3, 3-5          left arm
  2-4, 4-6          right arm
  1-7, 2-8          torso sides
  7-8               hip bridge

Features per node per frame: x, y, Δx, Δy  (C = 4)

Architecture
------------
  STGCNBlock(4→64) → STGCNBlock(64→64) → STGCNBlock(64→128)
  → global average pool over T and V
  → Linear(128, 1) + BCEWithLogitsLoss

Requires torch-geometric:
  pip install torch-geometric

Differences from train_serve_rnn.py
------------------------------------
  - Feature cache : serve_gcn_features.npz   shape [N, T, 9, 4]
  - Skips matches : 40, 44–48 (no ground truth / serve events)
  - Model output  : serve_stgcn.pt
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
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT         = Path("/Volumes/Anya/Data")
CLIP_DURATION     = 2.0           # seconds after each serve timestamp
MIN_SERVE_GAP_SEC = 6.0
ANALYSIS_SIZE     = (960, 540)
POSE_MODEL        = str(Path(__file__).parent / "pose_landmarker_full.task")
FEATURE_CACHE     = Path(__file__).parent / "serve_gcn_features.npz"
SKIP_MATCHES      = {40, 44, 45, 46, 47, 48}

# ── Skeleton graph ─────────────────────────────────────────────────────────────

MP_JOINT_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]   # MediaPipe → local node
NUM_JOINTS   = len(MP_JOINT_IDX)                       # 9

_EDGES_HALF = [
    (0, 1), (0, 2),
    (1, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8),
    (7, 8),
]
_EDGES_FULL = _EDGES_HALF + [(b, a) for a, b in _EDGES_HALF]
EDGE_INDEX  = torch.tensor(_EDGES_FULL, dtype=torch.long).t().contiguous()  # [2, 20]

NUM_CHANNELS = 4   # x, y, Δx, Δy

# ── Player detection (mirrors train_serve_rnn) ─────────────────────────────────

COURT_WIDTH_FT   = 27.0
COURT_LENGTH_FT  = 78.0
NEAR_PAD_FT      = 3.0
PLAYER_CLASS_IDX = 0
PLAYER_CONF      = 0.5
PLAYER_IMGSZ     = 960
FAR_STRIP_PAD_PX = 10

# ── ST-GCN hyperparameters ─────────────────────────────────────────────────────

CHANNELS    = [NUM_CHANNELS, 64, 64, 128]
T_KERNEL    = 9
DROPOUT     = 0.3
BATCH_SIZE  = 16
EPOCHS      = 60
LR          = 1e-3
VAL_FRAC    = 0.20
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Court / player detection
# ──────────────────────────────────────────────────────────────────────────────

def _load_court_data(video_path: str):
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(video_dir, f"{video_name}_court_cache.json")
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(f"Court cache missing: {cache_path}")
    with open(cache_path) as f:
        cached = json.load(f)
    pts = [tuple(p) for p in cached["points"]]
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
        return min(near, key=lambda c: abs(c[5]))[:4]


class FarPlayerDetector:
    def __init__(self, video_path: str):
        _, pts = _load_court_data(video_path)
        BL, BR, TR, TL = pts  # noqa: F841
        x1 = float(min(TL[0], TR[0]))
        x2 = float(max(TL[0], TR[0]))
        y_baseline = (TL[1] + TR[1]) / 2.0
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

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe pose extractor
# ──────────────────────────────────────────────────────────────────────────────

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


def extract_gcn_landmarks(pose_instance, frame_bgr, box) -> np.ndarray:
    """Returns [NUM_JOINTS, 2] (x, y normalised to crop) or zeros on failure."""
    zeros = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
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
        all_lms = result.pose_landmarks[0]
        return np.array([[all_lms[i].x, all_lms[i].y] for i in MP_JOINT_IDX],
                        dtype=np.float32)
    except Exception:
        return zeros

# ──────────────────────────────────────────────────────────────────────────────
# Label generation
# ──────────────────────────────────────────────────────────────────────────────

def _get_side(frame_id: int, rallies: list) -> str:
    for r in rallies:
        if r["start"] <= frame_id <= r["end"]:
            return r["serve"]
    return "unknown"


def load_labels():
    """Returns list of (match_id, timestamp, label, side)."""
    samples = []
    for match_dir in sorted(DATA_ROOT.iterdir()):
        if not match_dir.is_dir():
            continue
        try:
            match_id = int(match_dir.name)
        except ValueError:
            continue
        if match_id in SKIP_MATCHES:
            continue

        events_path = match_dir / "snippet_serve_events.json"
        gt_path     = match_dir / "ground_truth.json"
        video_path  = match_dir / "snippet.mp4"
        if not events_path.exists() or not video_path.exists():
            continue

        with open(events_path) as f:
            events = json.load(f)

        rallies = []
        if gt_path.exists():
            with open(gt_path) as f:
                rallies = json.load(f).get("rallies", [])

        events_sorted = sorted(events, key=lambda e: e["timestamp"])
        timestamps    = [e["timestamp"] for e in events_sorted]

        for e in events_sorted:
            side = _get_side(e["frame_id"], rallies)
            if side == "unknown":
                continue
            samples.append((match_id, e["timestamp"], 1, side))

        for i in range(len(timestamps) - 1):
            gap = timestamps[i + 1] - timestamps[i]
            if gap < MIN_SERVE_GAP_SEC:
                continue
            mid = (timestamps[i] + timestamps[i + 1]) / 2.0
            samples.append((match_id, mid, 0, "near"))
            samples.append((match_id, mid, 0, "far"))

    return samples

# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction  →  [T, V, C] per sample
# ──────────────────────────────────────────────────────────────────────────────

def extract_gcn_sample(video_path: str, timestamp: float,
                       detector, pose, seq_len: int) -> np.ndarray:
    """Returns [seq_len, NUM_JOINTS, NUM_CHANNELS] with x, y, Δx, Δy."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))

    pos_list, last_box, frames_read = [], None, 0
    while frames_read < seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, ANALYSIS_SIZE)
        box = detector.detect(frame)
        if box is not None:
            last_box = box
        pos_list.append(extract_gcn_landmarks(pose, frame,
                                              box if box is not None else last_box))
        frames_read += 1
    cap.release()

    zero = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
    while len(pos_list) < seq_len:
        pos_list.append(zero.copy())

    pos = np.stack(pos_list[:seq_len])      # [T, V, 2]
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    # Zero out velocity on frames where position detection failed
    vel[(pos == 0).all(axis=-1)] = 0.0

    return np.concatenate([pos, vel], axis=-1)  # [T, V, 4]


def build_dataset(labels, seq_len: int):
    if FEATURE_CACHE.exists():
        cached = np.load(FEATURE_CACHE)
        X_c, y_c = cached["X"], cached["y"]
        if (X_c.shape[1] == seq_len
                and X_c.shape[2] == NUM_JOINTS
                and X_c.shape[3] == NUM_CHANNELS):
            print(f"[INFO] Loaded GCN feature cache  {X_c.shape}  from {FEATURE_CACHE}")
            return X_c.astype(np.float32), y_c.astype(np.float32)
        print("[INFO] Cache shape mismatch — re-extracting.")

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
            t0  = time.time()
            seq = extract_gcn_sample(video_path, ts, detector, pose, seq_len)
            elapsed = time.time() - t0
            tag = "serve" if label == 1 else "neg  "
            nz  = int((seq[:, :, :2] != 0).any(axis=(1, 2)).sum())
            print(f"  [{i+1:3d}/{len(samples)}]  t={ts:7.2f}s  {tag}  "
                  f"({elapsed:.1f}s)  non-zero frames: {nz}/{seq_len}")
            X_list.append(seq)
            y_list.append(label)
        pose.close()

    if not X_list:
        raise RuntimeError("No samples extracted.")

    X = np.stack(X_list).astype(np.float32)   # [N, T, V, C]
    y = np.array(y_list, dtype=np.float32)
    np.savez_compressed(FEATURE_CACHE, X=X, y=y)
    print(f"[INFO] GCN feature cache saved to {FEATURE_CACHE}")
    return X, y

# ──────────────────────────────────────────────────────────────────────────────
# ST-GCN model
# ──────────────────────────────────────────────────────────────────────────────

class STGCNBlock(nn.Module):
    """Spatial graph conv (per-frame) + temporal conv along T + residual."""

    def __init__(self, in_ch: int, out_ch: int, edge_index: torch.Tensor,
                 num_nodes: int, t_kernel: int = 9, dropout: float = 0.3):
        super().__init__()
        self.num_nodes  = num_nodes
        self._num_edges = edge_index.shape[1]
        self.register_buffer("edge_index", edge_index)

        self.gcn  = GCNConv(in_ch, out_ch)
        self.bn_s = nn.BatchNorm1d(out_ch)

        pad = t_kernel // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(t_kernel, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.residual = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, V, in_ch]
        N, T, V, C = x.shape
        NT = N * T

        # ── Spatial GCN: apply to all N*T frames simultaneously ───────────────
        x_nodes = x.reshape(NT * V, C)

        # Tile edge_index NT times, shifting node indices by V per graph
        offsets = (torch.arange(NT, device=x.device)
                   .repeat_interleave(self._num_edges) * V)          # [NT*E]
        ei = self.edge_index.repeat(1, NT) + offsets.unsqueeze(0)    # [2, NT*E]

        x_nodes   = F.relu(self.bn_s(self.gcn(x_nodes, ei)))         # [NT*V, out_ch]
        x_spatial = x_nodes.reshape(N, T, V, -1)                     # [N, T, V, out_ch]

        # ── Temporal conv: Conv2d along T ─────────────────────────────────────
        x_t   = self.tcn(x_spatial.permute(0, 3, 1, 2))              # [N, out_ch, T, V]

        # ── Residual ──────────────────────────────────────────────────────────
        x_res = self.residual(x.permute(0, 3, 1, 2))                 # [N, out_ch, T, V]

        return F.relu(x_t + x_res).permute(0, 2, 3, 1)               # [N, T, V, out_ch]


class ServeSTGCN(nn.Module):
    """3-block ST-GCN → global average pool → binary head."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int = NUM_JOINTS,
                 channels=None, t_kernel: int = T_KERNEL, dropout: float = DROPOUT):
        super().__init__()
        if channels is None:
            channels = CHANNELS
        self.blocks = nn.ModuleList([
            STGCNBlock(channels[i], channels[i + 1], edge_index, num_nodes,
                       t_kernel, dropout)
            for i in range(len(channels) - 1)
        ])
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, T, V, C]
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[1, 2])           # global avg pool → [N, channels[-1]]
        return self.head(x).squeeze(-1)

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray):
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    print(f"\n{'='*60}")
    print(f"Dataset  :  {len(y)} samples  ({n_pos} serve / {n_neg} non-serve)")
    print(f"Sequence :  {X.shape[1]} frames × {X.shape[2]} nodes × {X.shape[3]} ch")

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

    model     = ServeSTGCN(EDGE_INDEX).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
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
        scheduler.step()

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

    save_path = Path(__file__).parent / "serve_stgcn.pt"
    torch.save({
        "model_state":  best_state,
        "channels":     CHANNELS,
        "t_kernel":     T_KERNEL,
        "dropout":      DROPOUT,
        "edge_index":   EDGE_INDEX,
        "num_nodes":    NUM_JOINTS,
        "mp_joint_idx": MP_JOINT_IDX,
        "seq_len":      X.shape[1],
    }, save_path)
    print(f"Model saved to {save_path}")
    return model

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(str(DATA_ROOT / "22" / "snippet.mp4"))
    fps_probe = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    seq_len = max(1, int(round(CLIP_DURATION * fps_probe)))
    print(f"[INFO] fps={fps_probe:.2f}  seq_len={seq_len} frames  "
          f"nodes={NUM_JOINTS}  channels={NUM_CHANNELS}")

    labels = load_labels()
    n_pos    = sum(1 for _, _, l, _ in labels if l == 1)
    n_neg    = sum(1 for _, _, l, _ in labels if l == 0)
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
