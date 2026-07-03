"""Vitis AI INT8 quantization (vai_q_pytorch) for the FLEO detector.

Runs *inside the Vitis AI 3.5 pytorch docker* on the host, NOT in the host venv
used for training.  It performs power-of-two PTQ over a calibration set drawn from
the training distribution (Section IV-E); if Delta_q exceeds the pre-registered
threshold, re-run with --qat for the fast-finetune/QAT fallback.

The route rewrite must already be applied (export the r1/r2/r3 checkpoint first):
the quantizer only ever sees the inference graph (Fig. 5).

Usage (in the Vitis AI docker):
    # calibration (PTQ)
    python quantize_vai.py --weights /workspace/export/r1.pt --route r1 \
        --data /workspace/datasets/fer2013/data.yaml --imgsz 640 \
        --mode calib --out /workspace/quantized/r1

    # export xmodel after calibration
    python quantize_vai.py --weights /workspace/export/r1.pt --route r1 \
        --data ... --imgsz 640 --mode test --out /workspace/quantized/r1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def build_model(weights: str, route: str, nc: int, imgsz: int):
    """Rebuild the FLEO detector and load the route-rewritten weights."""
    from fleo.yolo_integration import build_fleo_detection_model, set_route

    mode = {"r1": "folded", "r2": "householder", "r3": "full"}[route]
    det = build_fleo_detection_model("yolo12s.yaml", nc=nc, mode="full", imgsz=imgsz)
    if route == "r2":
        set_route(det, "householder")
    det.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    set_route(det, mode)
    det.eval()
    return det


def calib_loader(data_yaml: str, imgsz: int, n: int = 200, batch: int = 8):
    """Yield calibration batches from the training split."""
    import yaml
    from scripts.evaluate import _letterbox

    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d["path"])
    imgs = sorted((root / d["train"]).glob("*"))[:n]
    buf = []
    for p in imgs:
        buf.append(torch.from_numpy(_letterbox(p, imgsz)))
        if len(buf) == batch:
            yield torch.cat(buf, 0)
            buf = []
    if buf:
        yield torch.cat(buf, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--route", choices=["r1", "r2", "r3"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--nc", type=int, default=7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--mode", choices=["calib", "test"], default="calib")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--out", default="quantized")
    ap.add_argument("--qat", action="store_true", help="use QAT processor (fallback)")
    args = ap.parse_args()

    try:
        from pytorch_nndct.apis import torch_quantizer
    except ImportError:
        raise SystemExit(
            "pytorch_nndct not found. Run this INSIDE the Vitis AI pytorch docker "
            "(xilinx/vitis-ai-pytorch-cpu:latest)."
        )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model = build_model(args.weights, args.route, args.nc, args.imgsz)
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)

    quantizer = torch_quantizer(
        quant_mode=args.mode, module=model, input_args=(dummy,),
        output_dir=args.out, device=torch.device("cpu"),
    )
    q_model = quantizer.quant_model

    # Feed calibration / evaluation data through the quantized model.
    with torch.no_grad():
        for xb in calib_loader(args.data, args.imgsz, args.calib_n):
            q_model(xb)

    if args.mode == "calib":
        quantizer.export_quant_config()
        print(f"Quantization config written to {args.out}")
    else:
        quantizer.export_xmodel(output_dir=args.out, deploy_check=True)
        print(f"xmodel exported to {args.out} (ready for vai_c_xir)")


if __name__ == "__main__":
    main()
