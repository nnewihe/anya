"""
serve_lstm.py
=============
Far-side serve classifier: a 2-layer LSTM over MediaPipe pose-landmark
kinematics (position + frame-to-frame velocity) extracted from a player's
(padded) box, matching the feature pipeline used to train archive/serve_lstm.pt
(see archive/train_serve_rnn.py). This is the sole far-side ARMED -> ACTIVE
signal — there is no separate toss or trophy-pose score on the far side.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque
from typing import Optional

from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision

_ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")
DEFAULT_MODEL_PATH      = os.path.join(_ARCHIVE_DIR, "serve_lstm.pt")
DEFAULT_POSE_MODEL_PATH = os.path.join(_ARCHIVE_DIR, "pose_landmarker_full.task")

NUM_LANDMARKS  = 33
POS_PER_FRAME  = NUM_LANDMARKS * 2   # 66  (x, y)


class ServeLSTM(nn.Module):
    """Architecture must match archive/train_serve_rnn.py exactly to load its checkpoint."""

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


class ServeLSTMDetector:
    """
    Rolling-window serve classifier driven by pose-landmark kinematics inside
    a player's padded box crop.

    Call update(crop) once per ARMED frame with the box+padding crop (native
    resolution, no resize); pose landmarks are extracted internally and
    appended to a seq_len-frame rolling window. score() returns the model's
    serve probability once the window is full (0.0 before that).
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 pose_model_path: str = DEFAULT_POSE_MODEL_PATH,
                 device: Optional[str] = None):
        ckpt = torch.load(model_path, map_location="cpu")
        self.seq_len = ckpt["seq_len"]
        self.device  = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = ServeLSTM(ckpt["input_size"], ckpt["hidden_size"],
                                ckpt["num_layers"], ckpt["dropout"]).to(self.device)
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

    def _extract_landmarks(self, crop_bgr) -> np.ndarray:
        zeros = np.zeros(POS_PER_FRAME, dtype=np.float32)
        if crop_bgr is None or crop_bgr.size == 0:
            return zeros
        try:
            rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._pose.detect(mp_img)
            if not result.pose_landmarks:
                return zeros
            feats = []
            for lm in result.pose_landmarks[0]:
                feats.extend([lm.x, lm.y])
            return np.array(feats, dtype=np.float32)
        except Exception:
            return zeros

    def update(self, crop_bgr) -> None:
        """Push one frame's pose landmarks (from the padded player crop) into the rolling window."""
        self._positions.append(self._extract_landmarks(crop_bgr))

    def reset(self) -> None:
        """Clear the rolling window — call on ARMED entry so stale frames from a prior arm don't leak in."""
        self._positions.clear()

    def score(self) -> float:
        """Serve probability for the current window, or 0.0 until seq_len frames have been collected."""
        if len(self._positions) < self.seq_len:
            return 0.0

        pos_arr = np.stack(self._positions)              # (T, 66)
        vel_arr = np.zeros_like(pos_arr)
        vel_arr[1:] = pos_arr[1:] - pos_arr[:-1]
        vel_arr[(pos_arr == 0).all(axis=1)] = 0.0
        feat = np.concatenate([pos_arr, vel_arr], axis=1)  # (T, 132)

        x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logit = self.model(x)
            prob  = torch.sigmoid(logit).item()
        return float(prob)
