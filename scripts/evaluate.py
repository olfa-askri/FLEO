"""Detection-cast FER accuracy / macro-F1, and the decisive deltas.

Each image carries exactly one full-frame emotion box, so we score a *top-1
detection* rule: the emotion is the class of the highest-scoring anchor after the
Detect head (NMS-free, since there is a single object per frame).  The same rule
runs over an FP32 torch model, a quantized ONNX graph, or -- on target -- a VART
xmodel, so FP32/INT8 and full/folded are compared apples-to-apples.

Derived quantities:
    Delta_q    = Acc(FP32) - Acc(INT8)                (Eq. (8))
    Delta_fold = Acc(full graph) - Acc(folded graph)  (Eq. (6))

Examples:
    python -m scripts.evaluate --weights runs/fleo/fleo_seed0/weights/best.pt \
        --data datasets/fer2013/data.yaml --route full --imgsz 640

    python -m scripts.evaluate --onnx export/r1_folded.onnx \
        --data datasets/fer2013/data.yaml --imgsz 640 --tag "R1 INT8-sim"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


# ----------------------------------------------------------------------------- I/O
def _list_val(data_yaml: Path):
    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d["path"])
    val_dir = root / d["val"]
    imgs = sorted([p for p in val_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    labels_dir = root / d["val"].replace("images", "labels")
    gts = []
    kept = []
    for p in imgs:
        lp = labels_dir / f"{p.stem}.txt"
        if not lp.exists():
            continue
        first = lp.read_text().strip().splitlines()
        if not first:
            continue
        gts.append(int(first[0].split()[0]))
        kept.append(p)
    return kept, gts, d["names"]


def _letterbox(img, imgsz=640, stride=32):
    from PIL import Image

    im = Image.open(img).convert("RGB")
    w, h = im.size
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    im_r = im.resize((nw, nh))
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(im_r, ((imgsz - nw) // 2, (imgsz - nh) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]  # 1x3xHxW


def _topclass_from_head(out, nc):
    """out: (1, 4+nc, N) decoded Detect output. Return predicted class id.

    Single-object rule: for each emotion class take its maximum confidence over
    all anchors, then pick the class with the highest such confidence.  This is
    more robust than trusting one anchor's argmax (a single spurious anchor can't
    flip the decision) and matches "is emotion X present in this frame".
    """
    out = np.asarray(out)
    if out.ndim == 3:
        out = out[0]
    if out.shape[0] == 4 + nc:  # (4+nc, N)
        cls_scores = out[4 : 4 + nc, :]  # (nc, N)
    elif out.shape[1] == 4 + nc:  # (N, 4+nc)
        cls_scores = out[:, 4 : 4 + nc].T
    else:
        raise ValueError(f"Unexpected head output shape {out.shape} for nc={nc}")
    return int(cls_scores.max(axis=1).argmax())  # per-class max over anchors -> argmax


# ----------------------------------------------------------------------------- predictors
class TorchPredictor:
    def __init__(self, weights=None, det_model=None, route="full", device="cpu"):
        import torch
        from fleo.yolo_integration import set_route, register_torch_safe_globals

        self.torch = torch
        if det_model is None:
            from ultralytics import YOLO

            register_torch_safe_globals()  # so a FLEO checkpoint unpickles cleanly
            yolo = YOLO(weights)
            det_model = yolo.model
        self.model = det_model.to(device).eval()
        self.device = device
        set_route(self.model, route)
        self.nc = self.model.model[-1].nc

    def predict(self, img_path, imgsz):
        x = self.torch.from_numpy(_letterbox(img_path, imgsz)).to(self.device)
        with self.torch.no_grad():
            out = self.model(x)
        out = out[0] if isinstance(out, (list, tuple)) else out
        return _topclass_from_head(out.cpu().numpy(), self.nc)


class OnnxPredictor:
    def __init__(self, onnx_path, nc=7):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.nc = nc

    def predict(self, img_path, imgsz):
        x = _letterbox(img_path, imgsz)
        out = self.sess.run(None, {self.inp: x})[0]
        return _topclass_from_head(out, self.nc)


# ----------------------------------------------------------------------------- metrics
def evaluate(predictor, data_yaml, imgsz):
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

    imgs, gts, names = _list_val(data_yaml)
    nc = len(names)
    preds = [predictor.predict(p, imgsz) for p in imgs]
    gts = np.array(gts)
    preds = np.array(preds)
    acc = float(accuracy_score(gts, preds))
    macro_f1 = float(f1_score(gts, preds, average="macro", labels=list(range(nc)), zero_division=0))
    per_class = f1_score(gts, preds, average=None, labels=list(range(nc)), zero_division=0).tolist()
    cm = confusion_matrix(gts, preds, labels=list(range(nc))).tolist()
    return {
        "n": int(len(gts)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_f1": per_class,
        "confusion": cm,
        "names": names,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", help="ultralytics .pt checkpoint")
    ap.add_argument("--onnx", help="quantized/exported ONNX graph")
    ap.add_argument("--route", choices=["full", "folded", "householder"], default="full")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None, help="write metrics json here")
    args = ap.parse_args()

    if args.onnx:
        names = yaml.safe_load(Path(args.data).read_text())["names"]
        pred = OnnxPredictor(args.onnx, nc=len(names))
    elif args.weights:
        pred = TorchPredictor(weights=args.weights, route=args.route, device=args.device)
    else:
        raise SystemExit("Provide --weights or --onnx")

    m = evaluate(pred, args.data, args.imgsz)
    m["tag"] = args.tag or (args.onnx or f"{args.weights}:{args.route}")
    print(json.dumps({k: v for k, v in m.items() if k != "confusion"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(m, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
