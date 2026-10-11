"""
export_pose_coreml.py
=====================
yolov8n-pose (pipeline/models/yolov8n-pose.pt, the model perceive.py runs) as
a Core ML model for the app, plus a check that it still finds the same people.

WHAT IS EXPORTED, AND WHY
-------------------------
The convolutional network only: the graph ends at the head's three RAW feature
maps (strides 8, 16, 32), each [1, 116, H/s, W/s] =
    64  box distribution (4 sides x 16 bins, "DFL")
     1  person score (logit)
    51  17 keypoints x (x offset, y offset, visibility logit)
Ultralytics' own decode is left out on purpose: it builds the anchor grid from
the input size, and a traced graph would freeze that grid at ONE size. The
app decodes in Swift (PoseModel.swift) -- the same arithmetic as
`decode()` below -- so one model serves every input shape.

Input: an RGB image scaled by 1/255, at one of a set of ENUMERATED shapes
(flexible ranges fall off the Neural Engine):
    640 x 384          the near pass (a 960x540 frame letterboxed like Ultralytics)
    960 x H, H = 96..960 step 32   the far pass (the far band, letterboxed)

THE CHECK
---------
On real frames (tools/make_pose_check_frames.py), an fp32 copy of the model,
run on the CPU, decoded with `decode()`, must find the same people as
Ultralytics' own predict at perceive.py's settings (conf 0.20, IoU 0.70):
same count, box IoU >= 0.99, confidence within 0.005, keypoints within 1 px.
That gates the GRAPH and the DECODE. Measured: IoU 1.000, conf 0.000, 0.55 px.

The shipped model is fp16 (the Neural Engine runs fp16), which drifts:
on the same frames, fp16 on the CPU reaches IoU 0.990 / conf 0.016 / 2.7 px,
and on the Neural Engine IoU 0.963 / conf 0.036 / 9.6 px. That drift is
REPORTED, not gated: whether it matters is decided end to end, by scoring the
Swift pipeline on the corpus against ground truth (README, "Perception").

Needs the pinned converter env (torch 2.7, coremltools 9.0, numpy < 2, see
memory "iOS tracker Swift port"):
    python ios/AnyaTennis/tools/export_pose_coreml.py [--check-frames DIR] [--fp32-out PATH]
"""

import argparse
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WEIGHTS = os.path.join(REPO, "pipeline", "models", "yolov8n-pose.pt")
OUT = os.path.join(REPO, "ios", "AnyaTennis", "AnyaTennis", "Resources", "PoseModel.mlpackage")

NEAR_SHAPE = (384, 640)                              # (H, W)
FAR_SHAPES = [(h, 960) for h in range(96, 961, 32)]  # every far band height
STRIDES = (8, 16, 32)
CONF = 0.20
IOU = 0.70


def raw_head_model():
    """The pose network with its head returning the three raw maps."""
    import torch
    from ultralytics import YOLO
    model = YOLO(WEIGHTS).model.eval().float()
    model.fuse()
    head = model.model[-1]

    def raw_maps(features):
        return [torch.cat((head.cv2[i](features[i]), head.cv3[i](features[i]), head.cv4[i](features[i])), 1)
                for i in range(head.nl)]
    head.forward = raw_maps
    return model


def convert(precision="fp16"):
    import coremltools as ct
    import torch
    model = raw_head_model()
    example = torch.zeros(1, 3, *NEAR_SHAPE)
    traced = torch.jit.trace(model, example, strict=False)
    shapes = [(1, 3, *NEAR_SHAPE)] + [(1, 3, h, w) for h, w in FAR_SHAPES]
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=ct.EnumeratedShapes(shapes=shapes, default=shapes[0]),
                             scale=1 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name=f"p{s}") for s in STRIDES],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
    )
    return mlmodel


def export():
    mlmodel = convert("fp16")
    mlmodel.short_description = ("yolov8n-pose, raw head maps (decode in PoseModel.swift); "
                                 "exported by ios/AnyaTennis/tools/export_pose_coreml.py")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    mlmodel.save(OUT)
    print(f"wrote {os.path.relpath(OUT, REPO)}: {1 + len(FAR_SHAPES)} input shapes")
    return mlmodel


# ── the decode, mirrored by PoseModel.swift ──────────────────────────────

def letterbox(image, imgsz):
    """Ultralytics LetterBox(auto=True, stride=32): (canvas, gain, (pad_x, pad_y))."""
    import cv2
    h, w = image.shape[:2]
    gain = min(imgsz / h, imgsz / w)
    new_w, new_h = int(round(w * gain)), int(round(h * gain))
    dw, dh = (imgsz - new_w) % 32 / 2, (imgsz - new_h) % 32 / 2
    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    canvas = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return canvas, gain, (left, top)


def decode(maps, gain, pad, image_shape):
    """Raw maps -> people (boxes xyxy, confs, keypoints [17,3]) in IMAGE px, after NMS."""
    boxes, scores, keypoints = [], [], []
    for stride, m in zip(STRIDES, maps):
        m = m[0]                                   # [116, h, w]
        _, gh, gw = m.shape
        ys, xs = np.meshgrid(np.arange(gh) + 0.5, np.arange(gw) + 0.5, indexing="ij")
        dist = m[:64].reshape(4, 16, gh, gw)
        dist = np.exp(dist - dist.max(1, keepdims=True))
        dist = (dist / dist.sum(1, keepdims=True) * np.arange(16)[None, :, None, None]).sum(1)  # [4, h, w]
        x1, y1 = (xs - dist[0]) * stride, (ys - dist[1]) * stride
        x2, y2 = (xs + dist[2]) * stride, (ys + dist[3]) * stride
        score = 1 / (1 + np.exp(-m[64]))
        k = m[65:].reshape(17, 3, gh, gw)
        kx = (k[:, 0] * 2.0 + (xs - 0.5)) * stride
        ky = (k[:, 1] * 2.0 + (ys - 0.5)) * stride
        kv = 1 / (1 + np.exp(-k[:, 2]))
        boxes.append(np.stack([x1, y1, x2, y2], -1).reshape(-1, 4))
        scores.append(score.reshape(-1))
        keypoints.append(np.stack([kx, ky, kv], -1).transpose(1, 2, 0, 3).reshape(-1, 17, 3))
    boxes, scores, keypoints = np.concatenate(boxes), np.concatenate(scores), np.concatenate(keypoints)
    keep = scores > CONF
    boxes, scores, keypoints = boxes[keep], scores[keep], keypoints[keep]
    order = np.argsort(-scores)
    chosen = []
    for i in order:                                # greedy NMS, suppress IoU > 0.70
        if all(_iou(boxes[i], boxes[j]) <= IOU for j in chosen):
            chosen.append(i)
    boxes, scores, keypoints = boxes[chosen][:300], scores[chosen][:300], keypoints[chosen][:300]
    h, w = image_shape
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad[0]) / gain, 0, w)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad[1]) / gain, 0, h)
    keypoints = keypoints.copy()
    keypoints[:, :, 0] = np.clip((keypoints[:, :, 0] - pad[0]) / gain, 0, w)
    keypoints[:, :, 1] = np.clip((keypoints[:, :, 1] - pad[1]) / gain, 0, h)
    return boxes, scores, keypoints


def _iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def check(mlmodel, frames_dir, gate=True):
    """Core ML + decode() vs Ultralytics predict on the same frames. With
    `gate`, the strict fp32 tolerances apply; otherwise the numbers are only
    reported."""
    import cv2
    from PIL import Image
    from ultralytics import YOLO
    ultralytics = YOLO(WEIGHTS)
    worst = {"conf": 0.0, "kp": 0.0, "iou": 1.0}
    people = mismatched = 0
    for path in sorted(glob.glob(os.path.join(frames_dir, "*.png"))):
        image = cv2.imread(path)
        imgsz = 640 if os.path.basename(path).startswith("near") else 960
        want = ultralytics.predict(image, imgsz=imgsz, conf=CONF, iou=IOU, verbose=False)[0]
        canvas, gain, pad = letterbox(image, imgsz)
        rgb = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        out = mlmodel.predict({"image": rgb})
        boxes, scores, keypoints = decode([out[f"p{s}"] for s in STRIDES], gain, pad, image.shape[:2])
        want_boxes = want.boxes.xyxy.cpu().numpy()
        want_scores = want.boxes.conf.cpu().numpy()
        want_kp = want.keypoints.data.cpu().numpy() if want.keypoints is not None else np.zeros((0, 17, 3))
        people += len(want_boxes)
        if len(boxes) != len(want_boxes):
            mismatched += 1
            print(f"  {os.path.basename(path)}: Core ML {len(boxes)} people, Ultralytics {len(want_boxes)}")
        for i, wb in enumerate(want_boxes):
            if not len(boxes):
                break
            j = int(np.argmax([_iou(wb, b) for b in boxes]))
            worst["iou"] = min(worst["iou"], _iou(wb, boxes[j]))
            worst["conf"] = max(worst["conf"], abs(want_scores[i] - scores[j]))
            seen = (want_kp[i, :, 2] > 0.5) & (keypoints[j, :, 2] > 0.5)
            if seen.any():
                worst["kp"] = max(worst["kp"], float(np.abs(want_kp[i, seen, :2] - keypoints[j, seen, :2]).max()))
    print(f"check: {people} reference people; frames with a count mismatch: {mismatched}; "
          f"worst box IoU {worst['iou']:.3f}, conf diff {worst['conf']:.3f}, keypoint diff {worst['kp']:.2f} px")
    ok = mismatched == 0 and worst["iou"] >= 0.99 and worst["conf"] <= 0.005 and worst["kp"] <= 1.0
    if gate:
        print("PASS" if ok else "FAIL")
    return ok or not gate


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--check-frames", default=None, help="directory of .png frames to check on")
    parser.add_argument("--fp32-out", default=None,
                        help="also save the fp32 copy here (the Swift PoseModelTests decode it)")
    args = parser.parse_args()
    export()
    if args.check_frames or args.fp32_out:
        reference = convert("fp32")
        if args.fp32_out:
            reference.save(args.fp32_out)
            print(f"wrote {args.fp32_out}")
    if args.check_frames:
        import coremltools as ct
        print("graph + decode (fp32 copy, CPU):")
        passed = check(reference, args.check_frames)
        print("shipped fp16 model, Neural Engine allowed (reported, not gated):")
        check(ct.models.MLModel(OUT, compute_units=ct.ComputeUnit.ALL), args.check_frames, gate=False)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
