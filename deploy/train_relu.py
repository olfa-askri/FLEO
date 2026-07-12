"""Retrain FLEO+YOLOv8 with ReLU activations for DPU deployment.
=================================================================
The ZCU104 DPU (DPUCZDX8G) does NOT support SiLU. YOLOv8 uses SiLU by default,
which makes vai_c_xir abort ("Op_type 18 invalid" / SiLU not defined in XIR).
Swapping SiLU -> ReLU BEFORE building the model makes every conv DPU-native.

Run this on your GPU machine (host venv, NOT the Vitis AI docker):
    python deploy/train_relu.py --data datasets/fer2013/data.yaml --epochs 50

Output: runs/FLEO/fer2013/fleo_relu/weights/best.pt
Then quantize + compile that checkpoint exactly like before.
"""

from __future__ import annotations

import argparse
import os
import sys

# repo root importable (so `fleo` resolves)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/fer2013/data.yaml")
    ap.add_argument("--cfg", default="yolov8s.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pretrained", default="yolov8s.pt",
                    help="COCO warm-start; set '' to train from scratch")
    ap.add_argument("--project", default="runs/FLEO/fer2013")
    ap.add_argument("--name", default="fleo_relu")
    args = ap.parse_args()

    # ---- THE KEY LINE: swap SiLU -> ReLU for the whole model --------------
    # Must run BEFORE any model is built so parse_model picks ReLU as the
    # default conv activation.
    from ultralytics.nn.modules.conv import Conv
    Conv.default_act = nn.ReLU()
    print("[relu] Conv.default_act set to", Conv.default_act)

    from fleo.yolo_integration import make_fleo_trainer

    TrainerCls = make_fleo_trainer(pretrained=args.pretrained or None)
    overrides = dict(
        model=args.cfg,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    trainer = TrainerCls(overrides=overrides)
    trainer.train()
    print(f"\n[relu] done -> {args.project}/{args.name}/weights/best.pt")
    print("[relu] now quantize+compile THIS checkpoint (DPU-ready).")


if __name__ == "__main__":
    main()
