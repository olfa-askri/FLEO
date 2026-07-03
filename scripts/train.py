"""Train the FLEO-augmented YOLOv12-S facial-emotion detector.

The orthogonalization is a *training-time regularizer* (Section IV-B): we train the
full graph (mode='full', Gram-Schmidt active + auxiliary orthogonality/binding
losses) so the constraint is baked into the weights.  At export the GS stage is
folded out (Route 1) or constant-folded (Route 2); see scripts/export.py.

Examples:
    # baseline (no FLEO), FER2013, 3 seeds
    python -m scripts.train --data datasets/fer2013/data.yaml --variant baseline --seeds 0 1 2

    # FLEO full graph, RAF-DB
    python -m scripts.train --data datasets/rafdb/data.yaml --variant fleo --epochs 100

    # quick CPU smoke
    python -m scripts.train --data datasets/synthetic/data.yaml --variant fleo \
        --epochs 2 --imgsz 96 --batch 8 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def train_baseline(args, seed: int):
    from ultralytics import YOLO

    model = YOLO(args.cfg)
    res = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, seed=seed, project=args.project,
        name=f"baseline_seed{seed}", exist_ok=True, verbose=True,
        **_extra(args),
    )
    return model, res


def train_fleo(args, seed: int):
    """Train with FLEO via a custom trainer so the wrapping survives setup_model."""
    from fleo.yolo_integration import make_fleo_trainer

    Trainer = make_fleo_trainer(
        k=args.k, d=args.d, mode="full",
        lambda_ortho=args.lambda_ortho, lambda_bind=args.lambda_bind,
    )
    overrides = dict(
        model=args.cfg, data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, seed=seed, project=args.project,
        name=f"fleo_seed{seed}", exist_ok=True, verbose=True, **_extra(args),
    )
    trainer = Trainer(overrides=overrides)
    trainer.train()
    return trainer


def _extra(args):
    extra = {}
    if args.workers is not None:
        extra["workers"] = args.workers
    if args.patience is not None:
        extra["patience"] = args.patience
    if args.lr0 is not None:
        extra["lr0"] = args.lr0
    return extra


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="ultralytics data.yaml")
    ap.add_argument("--variant", choices=["baseline", "fleo"], default="fleo")
    ap.add_argument("--cfg", default="yolo12s.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--project", default="runs/fleo")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--lr0", type=float, default=None)
    # FLEO hyper-parameters.
    ap.add_argument("--k", type=int, default=7, help="emotion subspaces K")
    ap.add_argument("--d", type=int, default=8, help="per-emotion width d")
    ap.add_argument("--lambda-ortho", type=float, default=0.01)
    ap.add_argument("--lambda-bind", type=float, default=0.01)
    args = ap.parse_args()

    Path(args.project).mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"\n=== Training variant={args.variant} seed={seed} ===")
        if args.variant == "baseline":
            train_baseline(args, seed)
        else:
            train_fleo(args, seed)


if __name__ == "__main__":
    main()
