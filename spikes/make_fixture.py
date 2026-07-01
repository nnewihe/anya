"""
Build a self-contained golden fixture for the Dart inference spike (Spike B).

For each model we:
  1. Re-export with nms=True so the ONNX graph emits a uniform [1, N, 6]
     (x1,y1,x2,y2,conf,cls) tensor — no hand-written NMS needed in Dart.
  2. Reproduce Ultralytics' letterbox preprocessing BY HAND with numpy/cv2
     (this is exactly the recipe the Dart port must implement).
  3. Run the ONNX via onnxruntime directly (not Ultralytics) — mirrors Dart.
  4. Un-letterbox the boxes back to 960x540 analysis-frame coordinates.
  5. Cross-check our hand-rolled result against Ultralytics on the same frame.

Outputs (spikes/fixtures/):
  frame_960x540.png          the analysis frame
  input_960.f32              raw float32 [1,3,960,960] tensor fed to ORT (row-major)
  <model>_meta.json          letterbox params + expected boxes + ort output shape
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "pipeline" / "models"
OUT = Path(__file__).resolve().parent / "models"
FIX = Path(__file__).resolve().parent / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)
FIX.mkdir(parents=True, exist_ok=True)

IMGSZ = 960
FRAME = FIX / "frame_960x540.png"


def letterbox(img, new=960, color=(114, 114, 114)):
    """Ultralytics-style letterbox to a square, returning padded img + params."""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    dw, dh = (new - nw) // 2, (new - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def to_tensor(bgr):
    """BGR uint8 HWC -> float32 [1,3,H,W] RGB, /255, contiguous."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])  # 1,3,H,W


def unletterbox(box, r, dw, dh):
    x1, y1, x2, y2 = box
    return [(x1 - dw) / r, (y1 - dh) / r, (x2 - dw) / r, (y2 - dh) / r]


def nms(boxes, scores, iou_thr=0.45):
    """Plain greedy NMS. Returns kept indices. (This is the ~30 lines of Dart
    the ball model needs; players are NMS-free end2end.)"""
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])
    keep = []
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter/ua if ua > 0 else 0.0
    while idx:
        i = idx.pop(0); keep.append(i)
        idx = [j for j in idx if iou(boxes[i], boxes[j]) < iou_thr]
    return keep


def decode_raw(out, conf, r, dw, dh):
    """Decode raw YOLO [1, 4+nc, N] (cx,cy,w,h + class scores) -> boxes.
    ball_best is single-class so 5 rows. This is the Dart recipe for the ball."""
    arr = out[0]                      # [4+nc, N]
    xywh = arr[:4].T                  # [N,4]
    scores_all = arr[4:].T            # [N, nc]
    cls = scores_all.argmax(1)
    sc = scores_all.max(1)
    keep0 = sc >= conf
    xywh, cls, sc = xywh[keep0], cls[keep0], sc[keep0]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    keep = nms(xyxy.tolist(), sc.tolist())
    boxes = []
    for i in keep:
        bx = unletterbox(xyxy[i].tolist(), r, dw, dh)
        boxes.append({"cls": int(cls[i]), "conf": round(float(sc[i]), 5),
                      "xyxy": [round(float(v), 2) for v in bx]})
    boxes.sort(key=lambda d: -d["conf"])
    return boxes


def main():
    from ultralytics import YOLO

    frame = cv2.imread(str(FRAME))
    if frame is None:
        raise SystemExit(f"missing {FRAME} — run export_onnx.py first")

    padded, r, dw, dh = letterbox(frame, IMGSZ)
    tensor = to_tensor(padded)
    tensor.tofile(FIX / "input_960.f32")  # shared by both models (same size)
    print(f"[input] tensor {tensor.shape} dtype={tensor.dtype} "
          f"letterbox r={r:.5f} pad=({dw},{dh})  -> input_960.f32")

    # ball_best: raw export (keeps low-conf detections tunable in Dart).
    # yolo26n:   natively end2end (NMS-free) -> [1,300,6].
    for name, conf, use_nms in (("ball_best", 0.05, False), ("yolo26n", 0.25, True)):
        pt = YOLO(str(SRC / f"{name}.pt"))
        onnx_path = Path(pt.export(format="onnx", imgsz=IMGSZ, opset=12,
                                   dynamic=False, simplify=True, nms=use_nms))
        dest = OUT / f"{name}.onnx"
        onnx_path.replace(dest)

        sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]

        if out.shape[-1] == 6 and out.ndim == 3:      # end2end [1,N,6]
            boxes = []
            for row in out[0]:
                x1, y1, x2, y2, c, cls = row[:6]
                if c < conf:
                    continue
                bx = unletterbox([x1, y1, x2, y2], r, dw, dh)
                boxes.append({"cls": int(cls), "conf": round(float(c), 5),
                              "xyxy": [round(float(v), 2) for v in bx]})
            boxes.sort(key=lambda d: -d["conf"])
        else:                                          # raw [1,4+nc,N]
            boxes = decode_raw(out, conf, r, dw, dh)

        # Cross-check against Ultralytics on the raw frame.
        ul = pt(frame, verbose=False, conf=conf, imgsz=IMGSZ)[0]
        ul_boxes = [] if ul.boxes is None else [
            {"cls": int(b.cls[0]), "conf": round(float(b.conf[0]), 5),
             "xyxy": [round(float(v), 2) for v in b.xyxy[0].tolist()]}
            for b in ul.boxes]

        (FIX / f"{name}_meta.json").write_text(json.dumps({
            "onnx_out_shape": list(out.shape),
            "letterbox": {"r": r, "dw": dw, "dh": dh, "imgsz": IMGSZ},
            "conf_threshold": conf,
            "expected_boxes_manual_ort": boxes,
            "ultralytics_boxes": ul_boxes,
        }, indent=2))
        print(f"[{name}] ort_out={list(out.shape)}  manual={len(boxes)} box  "
              f"ultralytics={len(ul_boxes)} box  -> {name}_meta.json")
        for a in boxes[:3]:
            print(f"    manual {a}")
        for a in ul_boxes[:3]:
            print(f"    ultra  {a}")


if __name__ == "__main__":
    main()
