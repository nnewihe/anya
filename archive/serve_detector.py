"""
serve_detector.py
=================
Stateless serve detection pipeline.

Each step saves its output next to the video; re-running is a no-op if the
file already exists.  Delete a file to force that step to re-run.

  Step 1 — Init             : <stem>_court_cache.json  (via init_court)
                               <stem>_exclusion_zones.json
  Step 2 — Players          : <stem>_player_telemetry.json
  Step 3 — Toss             : <stem>_toss_telemetry.json
  Step 4 — Scores           : <stem>_serve_scores.csv
  Step 5 — Raw candidates   : <stem>_serve_candidates.json
  Step 6 — Final candidates : <stem>_final_candidates.json

Usage:
    python -m src.ai.serve_detector path/to/video.mp4
"""

import argparse
import csv
import json
import os
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.ai.utilities import (
    Config,
    _is_in_exclusion_zone,
    create_auto_exclusion_zones,
    init_court,
)

# ── ST-GCN serve detector ─────────────────────────────────────────────────────

class STGCNDetector:
    """
    ST-GCN serve classifier.  Extracts 9 upper-body joints via MediaPipe and
    runs serve_stgcn.pt over a rolling 60-frame window.

    Input tensor: (1, T, V, C) = (1, 60, 9, 4)  —  x, y, Δx, Δy per joint.
    Joints (MediaPipe indices): [0, 11, 12, 13, 14, 15, 16, 23, 24]
    """

    _MP_JOINT_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    _NUM_JOINTS   = 9
    _NUM_CHANNELS = 4

    def __init__(self, model_path: str):
        self._ready = False
        self._pose  = None
        self._model = None
        self._torch = None
        self._mp    = None

        # ── MediaPipe ─────────────────────────────────────────────────────
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as _mpt
            from mediapipe.tasks.python import vision as _mpv
            pose_model = model_path.replace("serve_stgcn.pt", "pose_landmarker_full.task")
            opts = _mpv.PoseLandmarkerOptions(
                base_options=_mpt.BaseOptions(model_asset_path=pose_model),
                running_mode=_mpv.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self._pose = _mpv.PoseLandmarker.create_from_options(opts)
            self._mp   = mp
        except Exception as e:
            print(f"[STGCN] MediaPipe init failed — serve detector disabled: {e}")
            return

        # ── Model ─────────────────────────────────────────────────────────
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch_geometric.nn import GCNConv

            ckpt      = torch.load(model_path, map_location="cpu", weights_only=False)
            channels  = ckpt["channels"]
            t_kernel  = ckpt["t_kernel"]
            dropout   = ckpt["dropout"]
            num_nodes = ckpt["num_nodes"]
            seq_len   = ckpt["seq_len"]

            _edges_half = [(0,1),(0,2),(1,2),(1,3),(3,5),(2,4),(4,6),(1,7),(2,8),(7,8)]
            _edges_full = _edges_half + [(b, a) for a, b in _edges_half]
            edge_index  = torch.tensor(_edges_full, dtype=torch.long).t().contiguous()

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

            class _ServeSTGCN(nn.Module):
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

            model = _ServeSTGCN(edge_index, channels, num_nodes, t_kernel, dropout)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            self._model   = model
            self._seq_len = seq_len
            self._torch   = torch
        except Exception as e:
            print(f"[STGCN] Model load failed — serve detector disabled: {e}")
            return

        self._pos_buf: deque = deque(maxlen=self._seq_len)
        self._ready = True
        print(f"[STGCN] Serve ST-GCN ready  "
              f"(seq_len={self._seq_len}  nodes={self._NUM_JOINTS}  model={model_path})")

    def predict_proba(self, frame_bgr, near_box) -> float:
        """Feed one frame; return serve probability (0.0–1.0)."""
        if not self._ready:
            return 0.0
        self._pos_buf.append(self._extract_pos(frame_bgr, near_box))
        if len(self._pos_buf) < self._seq_len:
            return 0.0
        return self._proba()

    def reset(self):
        if self._ready:
            self._pos_buf.clear()

    def _extract_pos(self, frame_bgr, box) -> np.ndarray:
        zeros = np.zeros((self._NUM_JOINTS, 2), dtype=np.float32)
        if box is None or self._pose is None:
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
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            res    = self._pose.detect(mp_img)
            if not res.pose_landmarks:
                return zeros
            lms = res.pose_landmarks[0]
            return np.array([[lms[i].x, lms[i].y] for i in self._MP_JOINT_IDX],
                            dtype=np.float32)
        except Exception:
            return zeros

    def _proba(self) -> float:
        pos_arr = np.stack(self._pos_buf)           # (T, 9, 2)
        vel_arr = np.zeros_like(pos_arr)
        vel_arr[1:] = pos_arr[1:] - pos_arr[:-1]
        vel_arr[(pos_arr == 0).all(axis=-1)] = 0.0
        x_np = np.concatenate([pos_arr, vel_arr], axis=-1)   # (T, 9, 4)
        x    = self._torch.tensor(x_np[np.newaxis], dtype=self._torch.float32)
        with self._torch.no_grad():
            return float(self._model(x).sigmoid().item())


# ── Pipeline constants ────────────────────────────────────────────────────────

ANALYSIS_SIZE         = (960, 540)
PLAYER_IMGSZ          = 320
TOSS_BALL_IMGSZ       = 320
TOSS_BALL_CONF           = Config.TOSS_BALL_CONF   # 0.10
TOSS_MAX_LATERAL_DRIFT_PX = 25  # px — max x-shift between consecutive toss detections

READY_MIN_DIST_FT     = -0.5
READY_MAX_DIST_FT     =  3.5
READY_WAIT_TIME_SEC   =  1.0
READY_EXIT_BUFFER_SEC =  2.0   # keep scoring this long after player leaves ready zone

EVENT_WINDOW_SEC      = 5.0
SERVE_SCORE_THRESHOLD = 0.7
CANDIDATE_CLUSTER_SEC = 0.5
TROPHY_AFTER_TOSS_SEC = 2.0   # trophy LSTM must peak within this window after toss fires


# ── Path helpers ──────────────────────────────────────────────────────────────

def _out(video_path: str, suffix: str) -> str:
    d = os.path.dirname(os.path.abspath(video_path))
    s = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(d, f"{s}_{suffix}")


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _compute_homography(court_vertices) -> np.ndarray:
    BL, BR, TR, TL = court_vertices
    src = np.array([BL, BR, TR, TL], dtype=np.float32)
    dst = np.array([
        [0,                      0                       ],
        [Config.COURT_WIDTH_FT,  0                       ],
        [Config.COURT_WIDTH_FT,  Config.COURT_LENGTH_FT  ],
        [0,                      Config.COURT_LENGTH_FT  ],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def _world_pos(H: np.ndarray, px_x: float, px_y: float) -> Tuple[float, float]:
    pt  = np.array([[[px_x, px_y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def _create_z_box(player_box) -> Optional[Tuple[int, int, int, int]]:
    if player_box is None:
        return None
    x1, y1, x2, y2 = player_box
    pw, ph   = x2 - x1, y2 - y1
    pcx, pcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # 1.2× player width keeps the search zone tight above the player's hand;
    # the original 2.0× was wide enough to catch background foliage on clay courts.
    zw, zh   = pw * 1.2, ph * 1.5
    return (int(pcx - zw / 2), max(0, int(pcy - zh)), int(pcx + zw / 2), int(pcy))


def _in_z_box(cx: float, cy: float, z_box) -> bool:
    if z_box is None:
        return False
    x1, y1, x2, y2 = z_box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _in_player_box(cx: float, cy: float, box, padding: int = 15) -> bool:
    if box is None:
        return False
    x1, y1, x2, y2 = box
    return x1 - padding <= cx <= x2 + padding and y1 - padding <= cy <= y2 + padding


# ── Step 1 — Initialization ───────────────────────────────────────────────────

def step1_init(video_path: str, ball_model):
    """
    Court corners + static exclusion zones.
    Returns (court_vertices, H, exclusion_zones).
    """
    print("\n[STEP 1] Court initialization")

    court_vertices, _ = init_court(video_path, analysis_size=ANALYSIS_SIZE)
    H = _compute_homography(court_vertices)

    excl_path = _out(video_path, "exclusion_zones.json")
    if os.path.exists(excl_path):
        with open(excl_path) as f:
            zones = [tuple(z) for z in json.load(f)]
        print(f"[STEP 1] Loaded {len(zones)} exclusion zone(s) from {os.path.basename(excl_path)}")
    else:
        print("[STEP 1] Scanning for static exclusion zones (50 frames) …")
        zones = create_auto_exclusion_zones(
            video_path, ball_model,
            num_frames=50, conf=0.04, eps=12, padding=5,
            ball_class_index=Config.DEFAULT_BALL_CLASS_INDEX,
            analysis_size=ANALYSIS_SIZE,
        )
        with open(excl_path, "w") as f:
            json.dump([list(z) for z in zones], f)
        print(f"[STEP 1] {len(zones)} exclusion zone(s) → {os.path.basename(excl_path)}")

    return court_vertices, H, zones


# ── Step 2 — Near-side player telemetry ──────────────────────────────────────

def step2_player_telemetry(video_path: str, H: np.ndarray, player_model):
    """
    Scan full video for near-side player positions at PLAYER_IMGSZ=320.
    Returns list of {frame_id, timestamp, near_player_box, near_player_world}.
    """
    path = _out(video_path, "player_telemetry.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"[STEP 2] Loaded {len(data)} frames from {os.path.basename(path)}")
        return data

    print(f"\n[STEP 2] Scanning full video for near-side player (imgsz={PLAYER_IMGSZ}) …")
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    records  = []
    frame_id = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        frame = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_LINEAR)
        ts    = round(frame_id / fps, 4)

        near_box, near_world = _detect_near_player(frame, player_model, H)
        records.append({
            "frame_id":          frame_id,
            "timestamp":         ts,
            "near_player_box":   list(near_box)   if near_box   else None,
            "near_player_world": list(near_world) if near_world else None,
        })

        if frame_id % 1000 == 0:
            print(f"  … {frame_id}/{total} ({frame_id / total * 100:.0f}%)")

    cap.release()

    with open(path, "w") as f:
        json.dump(records, f)
    print(f"[STEP 2] {len(records)} frames → {os.path.basename(path)}")
    return records


def _detect_near_player(frame, player_model, H):
    results = player_model(frame, verbose=False, conf=0.5, imgsz=PLAYER_IMGSZ)
    if not (results and results[0].boxes):
        return None, None

    pad        = Config.NEAR_PLAYER_X_PAD_FT
    candidates = []
    for b in results[0].boxes:
        if int(b.cls[0]) != Config.DEFAULT_PLAYER_CLASS_INDEX:
            continue
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cx = (x1 + x2) / 2.0
        wx, wy = _world_pos(H, cx, y2)
        candidates.append((x1, y1, x2, y2, wx, wy))

    near_cands = [
        c for c in candidates
        if abs(c[5]) < abs(c[5] - Config.COURT_LENGTH_FT)   # closer to near baseline
        and -pad <= c[4] <= Config.COURT_WIDTH_FT + pad       # within lateral bounds
    ]
    if not near_cands:
        return None, None

    best = min(near_cands, key=lambda c: abs(c[5]))
    return best[:4], (best[4], best[5])


def _ready_window_intervals(player_telemetry) -> List[Tuple[float, float]]:
    """
    Return (start_ts, end_ts) for each contiguous period where the near player
    was in the ready zone for at least READY_WAIT_TIME_SEC.
    end_ts is extended by READY_EXIT_BUFFER_SEC so the LSTM can score the serve
    motion after the player steps forward off the baseline.
    """
    intervals   = []
    ready_start = None
    armed       = False
    armed_start = None
    last_ts     = None

    for rec in player_telemetry:
        world = rec.get("near_player_world")
        ts    = rec["timestamp"]

        in_zone = (
            world is not None
            and world[1] < 0
            and READY_MIN_DIST_FT <= abs(world[1]) <= READY_MAX_DIST_FT
        )

        if in_zone:
            if ready_start is None:
                ready_start = ts
                armed       = False
                armed_start = None
            elif not armed and (ts - ready_start) >= READY_WAIT_TIME_SEC:
                armed       = True
                armed_start = ts
            last_ts = ts
        else:
            if armed and armed_start is not None:
                intervals.append((armed_start, last_ts + READY_EXIT_BUFFER_SEC))
            ready_start = None
            armed       = False
            armed_start = None
            last_ts     = None

    if armed and armed_start is not None:
        intervals.append((armed_start, last_ts + READY_EXIT_BUFFER_SEC))

    return intervals


def _scored_frame_ids(player_telemetry) -> set:
    """Frame IDs whose timestamps fall inside any buffered ready-zone interval."""
    intervals = _ready_window_intervals(player_telemetry)
    return {
        r["frame_id"] for r in player_telemetry
        if any(start <= r["timestamp"] <= end for start, end in intervals)
    }


# ── Step 3 — Toss ball telemetry ──────────────────────────────────────────────

def step3_toss_telemetry(video_path: str, player_telemetry, exclusion_zones, ball_model):
    """
    For each frame inside the buffered ready-zone windows, detect balls in the
    toss z_box above the near player.
    Returns list of {frame_id, timestamp, toss_ball_candidates}.
    """
    path = _out(video_path, "toss_telemetry.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"[STEP 3] Loaded toss telemetry from {os.path.basename(path)}")
        return data

    scored_ids   = _scored_frame_ids(player_telemetry)
    player_by_id = {r["frame_id"]: r for r in player_telemetry}
    print(f"\n[STEP 3] Scanning {len(scored_ids)} ready-zone frames for toss balls …")

    if not scored_ids:
        print("[STEP 3] No ready-zone windows found.")
        with open(path, "w") as f:
            json.dump([], f)
        return []

    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0
    records  = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id not in scored_ids:
            continue

        frame    = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_LINEAR)
        ts       = round(frame_id / fps, 4)
        pdata    = player_by_id.get(frame_id, {})
        near_box = tuple(pdata["near_player_box"]) if pdata.get("near_player_box") else None
        if near_box is None:
            continue

        z_box      = _create_z_box(near_box)
        candidates = _detect_toss_balls(frame, near_box, z_box, exclusion_zones, ball_model)
        records.append({
            "frame_id":             frame_id,
            "timestamp":            ts,
            "toss_ball_candidates": candidates,
        })

    cap.release()

    with open(path, "w") as f:
        json.dump(records, f)
    print(f"[STEP 3] {len(records)} toss frames → {os.path.basename(path)}")
    return records


def _detect_toss_balls(frame, near_box, z_box, exclusion_zones, ball_model) -> List[dict]:
    nx1, ny1, nx2, ny2 = near_box
    pw, ph = nx2 - nx1, ny2 - ny1
    fh, fw = frame.shape[:2]

    rx1 = max(0,  int(nx1 - pw / 2))
    ry1 = max(0,  int(ny1 - ph))
    rx2 = min(fw, int(nx2 + pw / 2))
    ry2 = min(fh, int(ny1 + ph / 2))
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return []

    results = ball_model(roi, verbose=False, conf=TOSS_BALL_CONF, imgsz=TOSS_BALL_IMGSZ)
    if not (results and results[0].boxes):
        return []

    out = []
    for b in results[0].boxes:
        cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
        full_cx = rx1 + (cx1 + cx2) / 2.0
        full_cy = ry1 + (cy1 + cy2) / 2.0
        if (not _in_z_box(full_cx, full_cy, z_box)
                or _is_in_exclusion_zone(full_cx, full_cy, exclusion_zones)
                or _in_player_box(full_cx, full_cy, near_box)):
            continue
        out.append({
            "box":          [rx1 + cx1, ry1 + cy1, rx1 + cx2, ry1 + cy2],
            "conf":         round(float(b.conf[0]), 4),
            "pixel_center": [full_cx, full_cy],
        })
    return out


def _toss_score(candidates, ny1: float, now: float,
                consec: int, gap: int, above: bool, last) -> tuple:
    """
    Pure-function port of TransitionEngine._update_toss_detection.
    Returns (toss_score, consec, gap, above, last_ball).
    """
    if not candidates:
        gap += 1
        if gap > 3:
            consec = 0
            above  = False
        return 0.0, consec, gap, above, None

    best = max(candidates, key=lambda x: x["conf"])
    cx   = (best["box"][0] + best["box"][2]) / 2.0
    cy   = (best["box"][1] + best["box"][3]) / 2.0

    moving_up = False
    if last is not None:
        dy  = cy - last["y"]
        dx  = abs(cx - last["x"])
        dtt = now - last["time"]
        # Reject if the ball jumps laterally — real toss travels nearly straight
        # up; foliage detections scatter across the z_box between frames.
        if dy < 0 and dtt > 0 and dx <= TOSS_MAX_LATERAL_DRIFT_PX:
            moving_up = True

    new_last = {"x": cx, "y": cy, "time": now}
    is_above = cy < ny1

    if moving_up and is_above:
        gap    = 0
        consec += 1
        above  = True
    else:
        gap += 1
        if gap > 3:
            consec = 0
            above  = False

    if not above:
        return 0.0, consec, gap, above, new_last
    if consec >= 3:
        return 1.0, consec, gap, above, new_last
    if consec >= 2:
        return 0.7, consec, gap, above, new_last
    return 0.0, consec, gap, above, new_last


# ── Step 4 — Serve scoring (trophy LSTM + toss) ───────────────────────────────

def step4_serve_scores(video_path: str, player_telemetry, toss_telemetry, model_path: str):
    """
    serve_score = ST-GCN score (100% weight); toss score logged but weighted 0.0.

    The ST-GCN runs on ALL player-visible frames for pre-warming; toss scores come
    from the pre-computed Step 3 telemetry and are saved in the CSV for reference.
    All per-window state resets on each new ready-zone window.

    Saves CSV: frame_id, timestamp, trophy_score (ST-GCN), toss_score, serve_score.
    Returns list of dicts.
    """
    path = _out(video_path, "serve_scores.csv")
    if os.path.exists(path):
        rows = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "frame_id":     int(row["frame_id"]),
                    "timestamp":    float(row["timestamp"]),
                    "trophy_score": float(row["trophy_score"]),
                    "toss_score":   float(row["toss_score"]),
                    "serve_score":  float(row["serve_score"]),
                })
        print(f"[STEP 4] Loaded {len(rows)} score rows from {os.path.basename(path)}")
        return rows

    print(f"\n[STEP 4] Computing serve scores "
          f"(trophy + toss, ready-zone + {READY_EXIT_BUFFER_SEC:.0f}s buffer) …")

    player_by_id = {r["frame_id"]: r for r in player_telemetry}
    toss_by_id   = {r["frame_id"]: r for r in toss_telemetry}
    scored_ids   = _scored_frame_ids(player_telemetry)
    print(f"[STEP 4] {len(scored_ids)} frames in scoring windows")

    detector = STGCNDetector(model_path)

    # Per-window state
    trophy_buf:  deque = deque()
    toss_buf:    deque = deque()
    toss_consec   = 0
    toss_gap      = 0
    toss_above    = False
    toss_latched  = False
    toss_fire_ts: Optional[float] = None
    last_toss     = None
    prev_in_window = False

    scores   = []
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_id = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        pdata     = player_by_id.get(frame_id)
        near_box  = tuple(pdata["near_player_box"]) if pdata and pdata.get("near_player_box") else None
        in_window = frame_id in scored_ids

        # Reset all per-window state on entry to a new window.
        # The ST-GCN sequence buffer is intentionally NOT reset — pre-warms naturally.
        if in_window and not prev_in_window:
            trophy_buf.clear()
            toss_buf.clear()
            toss_consec   = 0
            toss_gap      = 0
            toss_above    = False
            toss_latched  = False
            toss_fire_ts  = None
            last_toss     = None
        prev_in_window = in_window

        # LSTM runs on all player-visible frames to keep the buffer pre-warmed.
        frame_r      = cv2.resize(frame, ANALYSIS_SIZE, interpolation=cv2.INTER_LINEAR)
        trophy_score = detector.predict_proba(frame_r, near_box) if near_box else 0.0

        if not in_window or near_box is None:
            continue

        ts  = round(frame_id / fps, 4)
        ny1 = near_box[1]

        # Toss score from Step 3 telemetry
        toss_cands = toss_by_id.get(frame_id, {}).get("toss_ball_candidates", [])
        toss_score, toss_consec, toss_gap, toss_above, last_toss = \
            _toss_score(toss_cands, ny1, ts, toss_consec, toss_gap, toss_above, last_toss)
        if toss_score >= 1.0 and not toss_latched:
            toss_latched = True
            toss_fire_ts = ts

        # Update rolling trophy buffer (toss buffer kept for CSV visibility only)
        if trophy_score > 0:
            trophy_buf.append((trophy_score, ts))
        if toss_score > 0:
            toss_buf.append((toss_score, ts))

        cutoff = ts - EVENT_WINDOW_SEC
        while trophy_buf and trophy_buf[0][1] < cutoff:
            trophy_buf.popleft()
        while toss_buf and toss_buf[0][1] < cutoff:
            toss_buf.popleft()

        # Toss is a hard gate: if no toss fired in this window, score is 0.
        # When toss has latched, blend 20% trophy + 80% toss (matches live pipeline).
        serve_score = (0.4 * trophy_score + 0.6 * toss_score) if toss_latched else 0.0

        scores.append({
            "frame_id":     frame_id,
            "timestamp":    ts,
            "trophy_score": round(trophy_score, 4),
            "toss_score":   round(toss_score,   4),
            "serve_score":  round(serve_score,  4),
        })

    cap.release()

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["frame_id", "timestamp", "trophy_score", "toss_score", "serve_score"]
        )
        w.writeheader()
        w.writerows(scores)
    print(f"[STEP 4] {len(scores)} score rows → {os.path.basename(path)}")
    return scores


# ── Step 5 — Raw serve candidates ────────────────────────────────────────────

def step5_serve_candidates(video_path: str, serve_scores) -> List[Tuple[float, float]]:
    """
    Filter by SERVE_SCORE_THRESHOLD, then cluster frames within CANDIDATE_CLUSTER_SEC
    — keeping the highest-confidence frame per cluster.
    Returns list of (timestamp, confidence) pairs.
    """
    path = _out(video_path, "serve_candidates.json")
    if os.path.exists(path):
        with open(path) as f:
            cands = [tuple(c) for c in json.load(f)]
        print(f"[STEP 5] Loaded {len(cands)} raw candidates from {os.path.basename(path)}")
        return cands

    print("\n[STEP 5] Clustering serve candidates …")

    above = [
        (r["timestamp"], r["serve_score"])
        for r in serve_scores
        if r["serve_score"] >= SERVE_SCORE_THRESHOLD
    ]

    if not above:
        print("[STEP 5] No frames exceeded the serve score threshold.")
        with open(path, "w") as f:
            json.dump([], f)
        return []

    cands   = []
    cl_ts   = [above[0][0]]
    cl_conf = [above[0][1]]

    for ts, conf in above[1:]:
        if ts - cl_ts[-1] <= CANDIDATE_CLUSTER_SEC:
            cl_ts.append(ts)
            cl_conf.append(conf)
        else:
            best_i = int(np.argmax(cl_conf))
            cands.append((round(cl_ts[best_i], 4), round(cl_conf[best_i], 4)))
            cl_ts   = [ts]
            cl_conf = [conf]

    best_i = int(np.argmax(cl_conf))
    cands.append((round(cl_ts[best_i], 4), round(cl_conf[best_i], 4)))

    with open(path, "w") as f:
        json.dump(cands, f)

    print(f"[STEP 5] {len(cands)} raw candidate(s):")
    for ts, conf in cands:
        m, s = divmod(ts, 60)
        print(f"  {int(m):02d}:{s:05.2f}  conf={conf:.3f}")

    return cands


# ── Step 6 — Final candidates (ready-zone filter) ────────────────────────────

def step6_final_candidates(
    video_path: str,
    raw_candidates: List[Tuple[float, float]],
    player_telemetry,
) -> List[Tuple[float, float]]:
    """
    Keep only raw candidates whose timestamp falls inside a buffered ready-zone
    window.  Saves to <stem>_final_candidates.json.
    Returns list of (timestamp, confidence) pairs.
    """
    path = _out(video_path, "final_candidates.json")
    if os.path.exists(path):
        with open(path) as f:
            cands = [tuple(c) for c in json.load(f)]
        print(f"[STEP 6] Loaded {len(cands)} final candidates from {os.path.basename(path)}")
        return cands

    print("\n[STEP 6] Filtering by ready-zone windows …")

    intervals = _ready_window_intervals(player_telemetry)
    final     = [
        (ts, conf)
        for ts, conf in raw_candidates
        if any(start <= ts <= end for start, end in intervals)
    ]

    with open(path, "w") as f:
        json.dump(final, f)

    n_dropped = len(raw_candidates) - len(final)
    print(f"[STEP 6] {len(final)} final candidate(s)  ({n_dropped} dropped by ready-zone filter):")
    for ts, conf in final:
        m, s = divmod(ts, 60)
        print(f"  {int(m):02d}:{s:05.2f}  conf={conf:.3f}")

    return final


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run_pipeline(video_path: str) -> List[Tuple[float, float]]:
    _ai_dir = os.path.dirname(os.path.abspath(__file__))

    player_model = YOLO("yolo26n.pt")
    ball_model   = YOLO("weights/ball/weights/best.pt")
    model_path   = os.path.join(_ai_dir, "serve_stgcn.pt")

    court_vertices, H, excl_zones = step1_init(video_path, ball_model)
    player_tel  = step2_player_telemetry(video_path, H, player_model)
    toss_tel    = step3_toss_telemetry(video_path, player_tel, excl_zones, ball_model)
    scores      = step4_serve_scores(video_path, player_tel, toss_tel, model_path)
    raw_cands   = step5_serve_candidates(video_path, scores)
    final_cands = step6_final_candidates(video_path, raw_cands, player_tel)

    print(f"\n[DONE] {len(final_cands)} serve(s) detected in {os.path.basename(video_path)}")
    return final_cands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stateless serve detection pipeline")
    parser.add_argument("video", help="Path to input video file")
    args = parser.parse_args()
    run_pipeline(args.video)
