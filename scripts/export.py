"""Export the trained detector along one of the three deployment routes.

The route rewrite is applied to the *trained checkpoint before quantization*
(Fig. 5), so the quantizer/compiler only ever see the inference graph:

    R1  fold-out          -- delete Gram-Schmidt (identity); single DPU subgraph.
    R2  Householder       -- constant-fold Q into a 1x1 conv; single DPU subgraph.
    R3  hybrid PS/PL       -- keep Gram-Schmidt; compiler partitions DPU-CPU-DPU.

Outputs, per route:
    export/<route>.onnx     -- static-shape FP32 ONNX (onnxruntime eval + Vitis AI
                               ONNX quantizer input)
    export/<route>.pt        -- torch weights for vai_q_pytorch
    export/<route>.meta.json -- shapes, route, opset, op inventory

Examples:
    python -m scripts.export --weights runs/fleo/fleo_seed0/weights/best.pt \
        --route r1 --imgsz 640
    python -m scripts.export --weights ... --route r2 --verify
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fleo  # noqa: F401  (ensure FLEO classes are importable for unpickling)

ROUTE_TO_MODE = {"r1": "folded", "r2": "householder", "r3": "full", "full": "full"}


def _register_safe_globals():
    import torch

    from fleo.fleo_block import FLEO, ConvBNAct, BindingHead
    from fleo.orthogonal import GramSchmidt, HouseholderOrtho
    from fleo.yolo_integration import FLEOWrap

    try:
        torch.serialization.add_safe_globals(
            [FLEO, ConvBNAct, BindingHead, GramSchmidt, HouseholderOrtho, FLEOWrap]
        )
    except Exception:
        pass


def load_model(weights: str):
    from ultralytics import YOLO

    _register_safe_globals()  # so a FLEO checkpoint unpickles cleanly
    yolo = YOLO(weights)
    return yolo.model


def apply_route(det_model, route: str):
    from fleo.yolo_integration import set_route

    mode = ROUTE_TO_MODE[route]
    n = set_route(det_model, mode)
    return mode, n


def export_onnx(det_model, out_path: Path, imgsz: int, opset: int = 13):
    import torch

    det_model.eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    kwargs = dict(
        input_names=["images"], output_names=["output"],
        opset_version=opset, do_constant_folding=True,
        dynamic_axes=None,  # static shapes -- required by the Vitis AI compiler
    )
    try:
        # Legacy TorchScript exporter: robust for these static graphs and avoids
        # the dynamo path's onnxscript dependency.
        torch.onnx.export(det_model, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(det_model, dummy, str(out_path), **kwargs)
    return out_path


def op_inventory(onnx_path: Path):
    import onnx

    m = onnx.load(str(onnx_path))
    ops = {}
    for node in m.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    return ops


# Operators that unambiguously signal the Gram-Schmidt neck island (a reciprocal
# sqrt / L2 norm).  Unlike a bare Div -- which also appears in the Detect head's
# box decode that runs on the PS post-processing stage (Section III-D) -- these
# never occur in a folded/Householder graph, so their presence in the *neck* is
# what forces the DPU-CPU-DPU partition.
GS_SIGNATURE_OPS = {"Sqrt", "Reciprocal", "ReduceL2", "ReduceL1", "LpNormalization"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--route", choices=list(ROUTE_TO_MODE), required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--outdir", default="export")
    ap.add_argument("--verify", action="store_true", help="check onnxruntime vs torch parity")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.route

    det = load_model(args.weights)
    mode, n_sites = apply_route(det, args.route)
    print(f"route={args.route} mode={mode} FLEO sites rewritten={n_sites}")

    import torch

    torch.save(det.state_dict(), outdir / f"{stem}.pt")
    onnx_path = export_onnx(det, outdir / f"{stem}.onnx", args.imgsz, args.opset)

    ops = op_inventory(onnx_path)
    gs_ops = {k: v for k, v in ops.items() if k in GS_SIGNATURE_OPS}
    single_subgraph = len(gs_ops) == 0
    meta = {
        "route": args.route, "mode": mode, "imgsz": args.imgsz, "opset": args.opset,
        "fleo_sites": n_sites, "op_inventory": ops,
        "gram_schmidt_signature_ops": gs_ops,
        "expect_single_dpu_subgraph": single_subgraph,
        "note": (
            "Head-decode Div/Sub ops run on the PS post-processing stage (Section "
            "III-D); only neck GS-signature ops force a mid-graph CPU island."
        ),
    }
    (outdir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))

    print(f"ONNX  -> {onnx_path}")
    print(f"torch -> {outdir / f'{stem}.pt'}")
    print(f"Gram-Schmidt signature ops in neck: {gs_ops or 'none'}")
    print(
        "acceptance: "
        + ("single DPU subgraph expected (R1/R2); head decode on PS."
           if single_subgraph
           else "DPU-CPU-DPU partition expected (R3, GS island on PS).")
    )

    if args.verify:
        _verify(det, onnx_path, args.imgsz)


def _verify(det, onnx_path, imgsz):
    import numpy as np
    import torch
    import onnxruntime as ort

    det.eval()
    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        ty = det(x)
    ty = ty[0] if isinstance(ty, (list, tuple)) else ty
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    oy = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.abs(ty.numpy() - oy).max())
    print(f"onnxruntime vs torch max|diff| = {diff:.3e}  ({'OK' if diff < 1e-3 else 'CHECK'})")


if __name__ == "__main__":
    main()
