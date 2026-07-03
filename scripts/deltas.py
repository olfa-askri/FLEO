"""Compute the decisive deltas (Table VIII) from trained checkpoints.

For each dataset and each seed we evaluate the graph variants and precisions and
read off:

    Delta_fold = Acc(full graph, R3) - Acc(folded graph, R1)     (Eq. (6))
    Delta_q    = Acc(FP32)          - Acc(INT8)                   (Eq. (8))

FP32 comes from the torch checkpoint at the given route; INT8 comes from a
quantized ONNX (produced by the Vitis AI flow, or an onnxruntime-quantized proxy
for host-side sanity).  Results are aggregated as mean +/- s.d. over seeds.

Example:
    python -m scripts.deltas \
        --data datasets/fer2013/data.yaml --dataset FER2013 --imgsz 640 \
        --weights runs/fleo/fleo_seed0/weights/best.pt \
                  runs/fleo/fleo_seed1/weights/best.pt \
                  runs/fleo/fleo_seed2/weights/best.pt \
        --out results/deltas_fer2013.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate import TorchPredictor, OnnxPredictor, evaluate

# variant label -> route mode for the torch predictor
VARIANTS = {
    "full_R3": "full",
    "folded_R1": "folded",
    "householder_R2": "householder",
}


def _mean_std(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--weights", nargs="+", required=True, help="one .pt per seed")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--int8-onnx", nargs="*", default=None,
                    help="optional quantized ONNX per seed (same order) for Delta_q")
    ap.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    per_variant = {v: [] for v in VARIANTS}   # FP32 metric per seed
    fold_deltas, q_deltas = [], []

    for si, w in enumerate(args.weights):
        seed_scores = {}
        for v, mode in VARIANTS.items():
            pred = TorchPredictor(weights=w, route=mode, device=args.device)
            m = evaluate(pred, args.data, args.imgsz)
            seed_scores[v] = m[args.metric]
            per_variant[v].append(m[args.metric])
        fold_deltas.append(seed_scores["full_R3"] - seed_scores["folded_R1"])

        if args.int8_onnx:
            onnx_path = args.int8_onnx[si]
            names = json.loads(json.dumps(m["names"]))
            p8 = OnnxPredictor(onnx_path, nc=len(names))
            m8 = evaluate(p8, args.data, args.imgsz)
            # Delta_q against the folded FP32 (R1 is the deployed condition).
            q_deltas.append(seed_scores["folded_R1"] - m8[args.metric])

    report = {
        "dataset": args.dataset,
        "metric": args.metric,
        "seeds": len(args.weights),
        "variants_fp32": {v: dict(zip(("mean", "std"), _mean_std(s))) for v, s in per_variant.items()},
        "delta_fold": dict(zip(("mean", "std"), _mean_std(fold_deltas))),
    }
    if q_deltas:
        report["delta_q"] = dict(zip(("mean", "std"), _mean_std(q_deltas)))

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
