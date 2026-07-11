"""
YOLOv8 post-processing on the ARM CPU (ZCU104), pure NumPy.
===========================================================
The DPU returns the 3 RAW feature maps (P3/P4/P5). The Detect head — DFL
decode, box assembly, sigmoid on class scores, and NMS — runs here on the
ARM core. This mirrors Ultralytics' decode but with no torch dependency so it
runs on a bare PYNQ image.

Feature-map layout per scale (YOLOv8):
    channels = 4 * reg_max + num_classes      (reg_max = 16 by default)
    first  4*reg_max channels -> DFL box distribution
    last   num_classes        channels -> class logits
"""

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _dfl(reg, reg_max=16):
    """
    Distribution Focal Loss decode: turn a (N, 4, reg_max) distribution into
    4 expected distances (left, top, right, bottom) in stride units.
    """
    prob = _softmax(reg.reshape(-1, 4, reg_max), axis=-1)   # (N,4,reg_max)
    bins = np.arange(reg_max, dtype=np.float32)
    return (prob * bins).sum(-1)                            # (N,4)


def decode_feature_maps(feats, strides=(8, 16, 32),
                        num_classes=80, reg_max=16, conf_thres=0.25):
    """
    feats   : list of 3 arrays, each (C, H, W) with C = 4*reg_max + num_classes
    returns : boxes (M,4) xyxy in input-image pixels, scores (M,), classes (M,)
    """
    all_boxes, all_scores, all_cls = [], [], []
    nreg = 4 * reg_max

    for feat, stride in zip(feats, strides):
        C, H, W = feat.shape
        feat = feat.reshape(C, -1).T                        # (H*W, C)
        reg, cls = feat[:, :nreg], feat[:, nreg:]           # split

        scores = _sigmoid(cls)                              # (N, num_classes)
        cls_id = scores.argmax(1)
        conf = scores.max(1)
        keep = conf >= conf_thres
        if not keep.any():
            continue

        # grid centers for the kept cells
        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cx = (gx.reshape(-1) + 0.5)[keep]
        cy = (gy.reshape(-1) + 0.5)[keep]

        d = _dfl(reg[keep], reg_max)                        # (K,4) l,t,r,b
        x1 = (cx - d[:, 0]) * stride
        y1 = (cy - d[:, 1]) * stride
        x2 = (cx + d[:, 2]) * stride
        y2 = (cy + d[:, 3]) * stride

        all_boxes.append(np.stack([x1, y1, x2, y2], 1))
        all_scores.append(conf[keep])
        all_cls.append(cls_id[keep])

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)
    return (np.concatenate(all_boxes), np.concatenate(all_scores),
            np.concatenate(all_cls))


def nms(boxes, scores, iou_thres=0.45):
    """Standard greedy NMS (class-agnostic). Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def postprocess(feats, num_classes=80, conf_thres=0.25, iou_thres=0.45):
    """One call: raw DPU feature maps -> final (boxes, scores, classes)."""
    boxes, scores, cls = decode_feature_maps(
        feats, num_classes=num_classes, conf_thres=conf_thres)
    keep = nms(boxes, scores, iou_thres)
    return boxes[keep], scores[keep], cls[keep]
