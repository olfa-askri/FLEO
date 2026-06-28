"""
INT8 deployment SIMULATION without an FPGA board or Vitis AI.
============================================================
Runs anywhere PyTorch runs (Kaggle, a laptop CPU/GPU). It estimates the
numbers the paper's deployment table needs WITHOUT the ZCU104:

  * FP32 accuracy of the Route-1 (Gram-Schmidt folded out) model
  * INT8-simulated accuracy  -> the FP32 -> INT8 accuracy drop
  * latency (ms/image) and FPS on the current device
  * parameter count (and FLOPs if `thop` is installed)

How the INT8 simulation works: per-output-channel symmetric 8-bit
quantization is applied to every Conv/Linear weight (round to [-127,127] and
scale back). This captures the dominant source of INT8 accuracy loss and is a
faithful *software* proxy. It is NOT the AMD DPU: real DPU latency and the
official Vitis quantizer numbers still require Vitis AI / the board
(see deploy/VITIS.md). Report this as "simulated INT8" in the paper.

Usage
-----
    python deploy/simulate_int8.py --data /kaggle/input/fer2013 \
        --ckpt checkpoints/best_fleo.pt --backbone yolov12s --mode fleo_full
"""

import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_fer2013 import build_loaders, build_model, find_dataset
from deploy.export_fpga import evaluate


def fake_quant_int8_(model):
    """In-place per-output-channel symmetric INT8 fake-quantization of all
    Conv/Linear weights (simulates what the DPU does to the weights)."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, "weight"):
            w = m.weight.data
            wf = w.reshape(w.shape[0], -1)
            scale = (wf.abs().amax(dim=1) / 127.0).clamp(min=1e-12)
            q = torch.round(wf / scale[:, None]).clamp(-127, 127)
            m.weight.data = (q * scale[:, None]).reshape(w.shape)
            n += 1
    print(f"[int8-sim] quantized {n} weight tensors to 8-bit")
    return model


@torch.no_grad()
def measure_latency(model, device, img_size=128, iters=50, warmup=10):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000.0
    return ms, 1000.0 / ms


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def try_flops(model, img_size, device):
    try:
        from thop import profile
        x = torch.randn(1, 3, img_size, img_size, device=device)
        macs, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
        return macs * 2 / 1e9                          # GFLOPs
    except Exception as e:
        print(f"[int8-sim] FLOPs skipped ({type(e).__name__}); pip install thop")
        return None


def main():
    ap = argparse.ArgumentParser(description="INT8 simulation (no FPGA needed)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", default="checkpoints/best_fleo.pt")
    ap.add_argument("--backbone", default="yolov12s")
    ap.add_argument("--mode", default="fleo_full")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--route1", action="store_true",
                    help="fold out Gram-Schmidt (deployment graph) before sim")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = find_dataset(args.data)
    use_norm = args.backbone.startswith("resnet")
    _, val_loader = build_loaders(root, args.img_size, args.batch_size,
                                  num_workers=2, normalize=use_norm, balanced=False)

    model = build_model(backbone=args.backbone, mode=args.mode, pretrained=False)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    # Route-1 fold-out: the deployment graph drops Gram-Schmidt.
    if args.route1 and getattr(model, "has_neck", False):
        for m in (model.fleo_p3, model.fleo_p4):
            m.train_only = True
    model.to(args.device).eval()

    fp32_acc = evaluate(model, val_loader, args.device)
    params = count_params(model)
    gflops = try_flops(model, args.img_size, args.device)
    lat_ms, fps = measure_latency(model, args.device, args.img_size)

    int8_model = fake_quant_int8_(copy.deepcopy(model)).to(args.device)
    int8_acc = evaluate(int8_model, val_loader, args.device)

    print("\n" + "=" * 56)
    print("  INT8 SIMULATION (software proxy — not the DPU)")
    print("=" * 56)
    print(f"  backbone / mode : {args.backbone} / {args.mode}"
          f"{'  (Route-1)' if args.route1 else ''}")
    print(f"  params          : {params:.2f} M")
    if gflops is not None:
        print(f"  GFLOPs          : {gflops:.2f}")
    print(f"  latency (1 img) : {lat_ms:.2f} ms  ({fps:.1f} FPS, {args.device})")
    print(f"  FP32 accuracy   : {fp32_acc:.2f} %")
    print(f"  INT8 accuracy   : {int8_acc:.2f} %")
    print(f"  INT8 drop       : {fp32_acc - int8_acc:+.2f} pts")
    print("=" * 56)
    print("Note: real DPU latency/FPS and official Vitis INT8 need the ZCU104 /"
          " Vitis AI (see deploy/VITIS.md).")


if __name__ == "__main__":
    main()
