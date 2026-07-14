"""
serve_stgcn.py
==============
Far-side serve classifier: a Spatio-Temporal Graph Convolutional Network
(ST-GCN) over 9 upper-body MediaPipe joints (nose, shoulders, elbows, wrists,
hips), matching the architecture/checkpoint trained by
archive/train_serve_gcn.py.

Ported from archive/serve_stgcn.py with model weights relocated to
pipeline/models/ so the dead-time cutter can run far-side serve detection
without reaching into the archive.

Interface: update(crop) / reset() / score().
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from collections import deque
from typing import Optional

from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
from torch_geometric.nn import GCNConv

_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_MODEL_PATH      = os.path.join(_MODELS_DIR, "serve_stgcn.pt")
DEFAULT_POSE_MODEL_PATH = os.path.join(_MODELS_DIR, "pose_landmarker_full.task")

# MediaPipe landmark indices for the 9-node upper-body graph (nose, shoulders,
# elbows, wrists, hips), and the anatomical edges connecting them — must match
# archive/train_serve_gcn.py exactly so the checkpoint's GCNConv weights line
# up with the right node positions.
MP_JOINT_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
NUM_JOINTS   = 9
_EDGES_HALF  = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 5),
                (2, 4), (4, 6), (1, 7), (2, 8), (7, 8)]
_EDGES_FULL  = _EDGES_HALF + [(b, a) for a, b in _EDGES_HALF]


class _STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, edge_index, num_nodes, t_kernel, dropout):
        super().__init__()
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

    def forward(self, x):
        N, T, V, C = x.shape
        NT = N * T
        x_nodes = x.reshape(NT * V, C)
        offsets = (torch.arange(NT, device=x.device)
                   .repeat_interleave(self._num_edges) * V)
        ei        = self.edge_index.repeat(1, NT) + offsets.unsqueeze(0)
        x_nodes   = F.relu(self.bn_s(self.gcn(x_nodes, ei)))
        x_spatial = x_nodes.reshape(N, T, V, -1)
        x_t       = self.tcn(x_spatial.permute(0, 3, 1, 2))
        x_res     = self.residual(x.permute(0, 3, 1, 2))
        return F.relu(x_t + x_res).permute(0, 2, 3, 1)


class ServeSTGCN(nn.Module):
    """Architecture must match archive/train_serve_gcn.py exactly to load its checkpoint."""

    def __init__(self, edge_index, channels, num_nodes, t_kernel, dropout):
        super().__init__()
        self.blocks = nn.ModuleList([
            _STGCNBlock(channels[i], channels[i + 1], edge_index,
                        num_nodes, t_kernel, dropout)
            for i in range(len(channels) - 1)
        ])
        self.head = nn.Linear(channels[-1], 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=[1, 2])).squeeze(-1)


class ServeSTGCNDetector:
    """
    Rolling-window serve classifier driven by upper-body joint-graph
    kinematics inside a player's padded box crop.

    Call update(crop) once per frame with the box+padding crop (native
    resolution, no resize); pose landmarks for the 9 upper-body joints are
    extracted internally and appended to a seq_len-frame rolling window.
    score() returns the model's serve probability once the window is full
    (0.0 before that).
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 pose_model_path: str = DEFAULT_POSE_MODEL_PATH,
                 device: Optional[str] = None):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.seq_len = ckpt["seq_len"]
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device  = torch.device(device)

        edge_index = torch.tensor(_EDGES_FULL, dtype=torch.long).t().contiguous()
        self.model = ServeSTGCN(edge_index, ckpt["channels"], ckpt["num_nodes"],
                                 ckpt["t_kernel"], ckpt["dropout"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        opts = _mp_vision.PoseLandmarkerOptions(
            base_options=_mp_tasks.BaseOptions(model_asset_path=pose_model_path),
            running_mode=_mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._pose = _mp_vision.PoseLandmarker.create_from_options(opts)

        self._positions: deque = deque(maxlen=self.seq_len)

    def _extract_joints(self, crop_bgr) -> np.ndarray:
        zeros = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
        if crop_bgr is None or crop_bgr.size == 0:
            return zeros
        try:
            rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._pose.detect(mp_img)
            if not result.pose_landmarks:
                return zeros
            lms = result.pose_landmarks[0]
            return np.array([[lms[i].x, lms[i].y] for i in MP_JOINT_IDX], dtype=np.float32)
        except Exception:
            return zeros

    def update(self, crop_bgr) -> None:
        """Push one frame's 9-joint positions (from the padded player crop) into the rolling window."""
        self._positions.append(self._extract_joints(crop_bgr))

    def reset(self) -> None:
        """Clear the rolling window — call when the far player leaves the serve zone."""
        self._positions.clear()

    def score(self) -> float:
        """Serve probability for the current window, or 0.0 until seq_len frames have been collected."""
        if len(self._positions) < self.seq_len:
            return 0.0

        pos_arr = np.stack(self._positions)                  # (T, 9, 2)
        vel_arr = np.zeros_like(pos_arr)
        vel_arr[1:] = pos_arr[1:] - pos_arr[:-1]
        vel_arr[(pos_arr == 0).all(axis=-1)] = 0.0
        feat = np.concatenate([pos_arr, vel_arr], axis=-1)   # (T, 9, 4)

        x = torch.tensor(feat[np.newaxis], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logit = self.model(x)
            prob  = torch.sigmoid(logit).item()
        return float(prob)
