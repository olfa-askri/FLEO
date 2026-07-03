"""Compute the model workload W (Eq. (11)) and traffic Q (Eq. (14)) per route.

W is model-intrinsic, so it is counted here on the host by hooking every Conv2d and
summing 2*Kh*Kw*Cin*Cout*Hout*Wout -- identical to the compiler's per-layer op
count.  Q (off-chip bytes) is estimated as INT8 weight traffic plus non-resident
activation traffic; the DPU's on-chip buffering makes the true Q run-dependent, so
this is a documented estimate used to place the roofline operating point.

Usage:
    python -m deploy.measure.workload --weights export/r1.pt --route r1 \
        --imgsz 640 --out results/workload_r1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


def count(model, imgsz: int):
    layers = []
    handles = []

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            _, _, hout, wout = out.shape
            layers.append({
                "kh": m.kernel_size[0], "kw": m.kernel_size[1],
                "cin": m.in_channels, "cout": m.out_channels,
                "hout": hout, "wout": wout,
                "groups": m.groups,
                "weight_bytes": m.weight.numel(),  # INT8 -> 1 byte/param
                "act_bytes": out.numel(),          # INT8 activation bytes
            })

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, imgsz, imgsz))
    for h in handles:
        h.remove()
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="route-rewritten torch state_dict (export/<route>.pt)")
    ap.add_argument("--route", choices=["r1", "r2", "r3"], required=True)
    ap.add_argument("--nc", type=int, default=7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from fleo.yolo_integration import build_fleo_detection_model, set_route
    from ..metrics import model_workload_ops, arithmetic_intensity, roofline_tops, ridge_point

    mode = {"r1": "folded", "r2": "householder", "r3": "full"}[args.route]
    det = build_fleo_detection_model("yolo12s.yaml", nc=args.nc, mode="full", imgsz=args.imgsz)
    if args.route == "r2":
        set_route(det, "householder")
    det.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=False)
    set_route(det, mode)

    layers = count(det, args.imgsz)
    W = model_workload_ops(layers)
    weight_bytes = sum(L["weight_bytes"] for L in layers)
    act_bytes = sum(L["act_bytes"] for L in layers)
    Q = weight_bytes + act_bytes  # estimate (documented)
    I = arithmetic_intensity(W, Q)

    report = {
        "route": args.route, "imgsz": args.imgsz,
        "num_conv_layers": len(layers),
        "W_ops": int(W),
        "W_GFLOPs_madd": W / 2 / 1e9,   # MACs in GFLOP
        "weight_bytes_int8": int(weight_bytes),
        "act_bytes_int8": int(act_bytes),
        "Q_bytes_estimate": int(Q),
        "arithmetic_intensity_op_byte": I,
        "ridge_I_star": ridge_point(),
        "roofline_bound_TOPS": roofline_tops(I),
        "memory_or_compute_bound": "compute" if I >= ridge_point() else "memory",
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
