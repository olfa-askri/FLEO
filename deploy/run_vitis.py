"""
End-to-end Vitis AI export + INT8 simulation for a trained FLEOClassifier.
=========================================================================
Run this INSIDE the Vitis AI PyTorch Docker container (it provides
`pytorch_nndct`). It is NOT runnable on Kaggle.

    docker run -it --rm -v $(pwd):/work xilinx/vitis-ai-pytorch-cpu:latest
    cd /work
    python deploy/run_vitis.py --data data/fer2013 --ckpt checkpoints/best_fleo.pt

What it does (the paper's deployment experiment):
  1. Load checkpoint, fold out Gram-Schmidt (Route 1).
  2. Measure FP32 accuracy.
  3. Quantize to INT8 (calib + test) with the Vitis AI quantizer.
  4. Measure INT8 accuracy  -> report the FP32->INT8 drop.
  5. Dump the .xmodel and print the vai_c_xir compile command.

Outside the Docker (no pytorch_nndct) it still reports FP32 accuracy and the
compile instructions, so you can dry-run the pipeline.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_fer2013 import build_loaders, find_dataset
from deploy.export_fpga import (
    prepare_classifier_for_export, quantize_model, evaluate,
    print_compile_instructions,
)


def main():
    ap = argparse.ArgumentParser(description="Vitis AI export + INT8 simulation")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", default="checkpoints/best_fleo.pt")
    ap.add_argument("--backbone", default="yolov12s")
    ap.add_argument("--mode", default="fleo_full")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--calib-batches", type=int, default=8,
                    help="number of batches used for INT8 calibration")
    ap.add_argument("--output-dir", default="quantized")
    args = ap.parse_args()

    root = find_dataset(args.data)
    use_norm = args.backbone.startswith("resnet")
    _, val_loader = build_loaders(root, args.img_size, args.batch_size,
                                  num_workers=2, normalize=use_norm, balanced=False)

    print("[vitis] loading + folding out Gram-Schmidt (Route 1) ...")
    model = prepare_classifier_for_export(args.ckpt, args.backbone, args.mode)

    fp32 = evaluate(model, val_loader, device="cpu")
    print(f"[vitis] FP32 (Route-1) accuracy: {fp32:.2f}%")

    # A small calibration subset.
    calib = [b for i, b in zip(range(args.calib_batches), val_loader)]
    qmodel = quantize_model(model, calib, args.output_dir, img_size=args.img_size)

    if qmodel is not model:                          # quantizer was available
        int8 = evaluate(qmodel, val_loader, device="cpu")
        print(f"[vitis] INT8 accuracy: {int8:.2f}%   (drop {fp32 - int8:+.2f} pts)")

    print_compile_instructions(quant_dir=args.output_dir)


if __name__ == "__main__":
    main()
