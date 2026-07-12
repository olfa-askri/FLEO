"""Vitis AI INT8 quantization (vai_q_pytorch) for the FLEO detector.

Runs *inside the Vitis AI 3.5 pytorch docker*, NOT in the host venv used for
training. It performs power-of-two PTQ over a calibration set drawn from the
training distribution; if Delta_q exceeds the pre-registered threshold, re-run
with --qat for the fast-finetune/QAT fallback.

The route rewrite must already be applied: the quantizer only ever sees the
inference graph. For FPGA (DPU) use route r1 (folded) so Gram-Schmidt is folded
out and only DPU-native Conv/SE ops remain.

Usage (in the Vitis AI docker):
    # calibration (PTQ)
    python quantize_vai.py --weights runs/FLEO/fer2013/fleo_seed0/weights/best.pt \
        --route r1 --cfg yolov8s.yaml \
        --data datasets/fer2013/data.yaml --imgsz 640 \
        --mode calib --out quantized/r1

    # export xmodel after calibration
    python quantize_vai.py --weights runs/FLEO/fer2013/fleo_seed0/weights/best.pt \
        --route r1 --cfg yolov8s.yaml --data ... --imgsz 640 \
        --mode test --out quantized/r1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the repo root importable so `fleo` / `scripts` resolve regardless of the
# directory this script is launched from (e.g. `python deploy/quantize_vai.py`).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch


# ---------------------------------------------------------------------------
# Checkpoint loading: Ultralytics .pt files are CHECKPOINT DICTS
# ({'model': DetectionModel, 'optimizer':..., 'epoch':...}), NOT raw
# state_dicts. Extract the state_dict properly, whichever form we get.
# ---------------------------------------------------------------------------
def _extract_state_dict(weights: str) -> dict:
    try:
        from fleo.yolo_integration import register_torch_safe_globals
        register_torch_safe_globals()
    except Exception:
        pass

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model_obj = ckpt["model"]
        # ultralytics stores the whole nn.Module; pull float() weights out of it.
        return model_obj.float().state_dict()
    if isinstance(ckpt, dict):
        return ckpt                      # already a state_dict
    return ckpt.float().state_dict()     # a bare nn.Module was pickled


def _swap_silu_to_relu(module):
    """Replace every nn.SiLU with nn.ReLU in place (DPU has no SiLU).

    For HARDWARE metrics (FPS/LUT/DSP/power) this is enough on already-trained
    weights: those metrics depend on architecture, not weight values. Detection
    accuracy WILL drop (activation mismatch) — retrain with ReLU for real
    accuracy (deploy/train_relu.py).
    """
    import torch.nn as nn
    n = 0
    for m in module.modules():
        for name, child in list(m.named_children()):
            if isinstance(child, nn.SiLU):
                setattr(m, name, nn.ReLU(inplace=True))
                n += 1
    print(f"[force-relu] swapped {n} SiLU -> ReLU")
    return module


def build_model(weights: str, route: str, nc: int, imgsz: int,
                cfg: str = "yolov8s.yaml", force_relu: bool = False):
    """Rebuild the FLEO detector and load the route-rewritten weights.

    ``cfg`` selects the backbone family. Use ``yolov8s.yaml`` for a fully
    DPU-native (convolution-only) backbone: YOLOv12's area-attention (A2C2f/AAttn)
    and YOLOv11's PSA are not compilable by the DPUCZDX8G XIR flow, so the
    single-DPU-subgraph deployment requires an attention-free backbone.
    """
    from fleo.yolo_integration import build_fleo_detection_model, set_route

    mode = {"r1": "folded", "r2": "householder", "r3": "full"}[route]
    det = build_fleo_detection_model(cfg, nc=nc, mode="full", imgsz=imgsz)

    sd = _extract_state_dict(weights)
    missing, unexpected = det.load_state_dict(sd, strict=False)
    print(f"loaded weights: {len(sd)} tensors "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")

    set_route(det, mode)                 # r1 -> fold Gram-Schmidt out for the DPU

    # Cut the Detect head so the model returns the raw per-scale feature maps
    # (pure conv -> DPU-native). The DPU cannot run the Detect head's DFL decode
    # / eltwise SUB / DIV / strided_slice, so those move to the ARM CPU
    # (postprocess_arm.py). Without this, vai_c_xir aborts with
    # "Op_type 18 is invalid for xcompiler".
    import types
    detect = det.model[-1]

    def _raw_head(self, x):
        return [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
                for i in range(self.nl)]

    detect.forward = types.MethodType(_raw_head, detect)

    if force_relu:                       # DPU-compilable (hardware-metrics path)
        _swap_silu_to_relu(det)

    det.eval()
    return det


def _letterbox(img_path, imgsz: int, color=(114, 114, 114)):
    """Self-contained letterbox: read image -> resize keeping aspect ratio ->
    pad to square -> BGR2RGB, CHW, /255. Returns numpy (3, imgsz, imgsz) float32.
    Matches YOLOv8 default preprocessing so INT8 ranges reflect training."""
    import cv2
    import numpy as np

    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas = np.full((imgsz, imgsz, 3), color, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    arr = canvas[:, :, ::-1].transpose(2, 0, 1)          # BGR->RGB, HWC->CHW
    return np.ascontiguousarray(arr, dtype=np.float32) / 255.0


def calib_loader(data_yaml: str, imgsz: int, n: int = 200, batch: int = 8):
    """Yield calibration batches from the training split (self-contained)."""
    import yaml

    d = yaml.safe_load(Path(data_yaml).read_text())
    yaml_dir = Path(data_yaml).resolve().parent
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    # Try several ways to resolve the train images dir, robust to how the
    # data.yaml was written (absolute Windows path, relative to yaml, etc.).
    candidates = []
    train_rel = d.get("train", "images/train")
    if "path" in d:
        candidates.append(Path(d["path"]) / train_rel)
        candidates.append(yaml_dir / d["path"] / train_rel)
    candidates.append(yaml_dir / train_rel)
    candidates.append(Path(train_rel))
    candidates.append(yaml_dir / "images")
    candidates.append(yaml_dir)

    imgs = []
    used = None
    for c in candidates:
        if c.exists():
            found = []
            for e in exts:
                found += sorted(c.rglob(e))
            if found:
                imgs, used = found[:n], c
                break

    print(f"[calib] resolved images dir: {used}")
    print(f"[calib] image files found  : {len(imgs)}")
    assert imgs, ("no calibration images found. Checked:\n  " +
                  "\n  ".join(str(c) for c in candidates))

    buf, loaded = [], 0
    for p in imgs:
        arr = _letterbox(p, imgsz)
        if arr is None:
            continue
        loaded += 1
        buf.append(torch.from_numpy(arr))
        if len(buf) == batch:
            yield torch.stack(buf, 0)
            buf = []
    if buf:
        yield torch.stack(buf, 0)
    print(f"[calib] images actually read: {loaded}")
    assert loaded > 0, "found files but cv2 could not read any (corrupt/paths?)"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--route", choices=["r1", "r2", "r3"], default="r1",
                    help="r1=folded (FPGA), r2=householder, r3=full")
    ap.add_argument("--data", required=True)
    ap.add_argument("--nc", type=int, default=7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--mode", choices=["calib", "test"], default="calib")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--cfg", default="yolov8s.yaml",
                    help="backbone cfg; yolov8s.yaml is DPU-native (attention-free)")
    ap.add_argument("--out", default="quantized")
    ap.add_argument("--qat", action="store_true", help="use QAT processor (fallback)")
    ap.add_argument("--force-relu", action="store_true",
                    help="swap SiLU->ReLU on trained weights so it compiles on "
                         "the DPU (hardware metrics valid; accuracy drops)")
    args = ap.parse_args()

    try:
        from pytorch_nndct.apis import torch_quantizer
    except ImportError:
        raise SystemExit(
            "pytorch_nndct not found. Run this INSIDE the Vitis AI pytorch docker "
            "(xilinx/vitis-ai-pytorch-cpu:latest)."
        )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model = build_model(args.weights, args.route, args.nc, args.imgsz, args.cfg,
                        force_relu=args.force_relu)
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)

    quantizer = torch_quantizer(
        quant_mode=args.mode, module=model, input_args=(dummy,),
        output_dir=args.out, device=torch.device("cpu"),
    )
    q_model = quantizer.quant_model

    # xmodel export requires batch size 1; calibration can batch for speed.
    calib_batch = 1 if args.mode == "test" else 8
    with torch.no_grad():
        for xb in calib_loader(args.data, args.imgsz, args.calib_n, batch=calib_batch):
            q_model(xb)

    if args.mode == "calib":
        quantizer.export_quant_config()
        print(f"Quantization config written to {args.out}")
    else:
        quantizer.export_xmodel(output_dir=args.out, deploy_check=True)
        print(f"xmodel exported to {args.out} (ready for vai_c_xir)")


if __name__ == "__main__":
    main()
