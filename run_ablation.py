"""
Ablation study for FLEO on FER2013 (and RAF-DB) -> Table 2 of the paper.
========================================================================
Runs the five configurations the reviewers will ask for, each over several
seeds, and reports accuracy / macro-F1 as mean +/- std:

    Method                       Acc.        macro-F1
    -----------------------------------------------------
    YOLOv12-S (baseline)         ...         ...
    + SE neck                    ...         ...
    + FLEO (no binding)          ...         ...
    + FLEO (full)                ...         ...
    + FLEO (full) + fuzzy loss   ...         ...

Each row isolates one claim (paper Sec. V-E):
  baseline -> +SE          : gain of channel gating alone
  +SE -> +FLEO(no binding) : gain of orthogonality
  no-binding -> full       : whether binding makes orthogonality meaningful
  full -> full+fuzzy       : gain of the confusion-aware loss

Usage
-----
    python run_ablation.py --data /kaggle/input/fer2013 --epochs 40 --seeds 0 1 2
    # quick smoke test:
    python run_ablation.py --data data/fer2013 --epochs 2 --seeds 0
"""

import argparse
import json
import os
import statistics as st

import torch

from train_fer2013 import build_loaders, build_model, find_dataset, set_seed
from train.trainer import FLEOTrainer

# (label, model mode, use_fuzzy)
ABLATION_ROWS = [
    ("baseline",                 "baseline",    False),
    ("+ SE neck",                "se",          False),
    ("+ FLEO (no binding)",      "fleo_nobind", False),
    ("+ FLEO (full)",            "fleo_full",   False),
    ("+ FLEO (full) + fuzzy",    "fleo_full",   True),
]


def run_one(args, mode, use_fuzzy, seed, loaders):
    set_seed(seed)
    train_loader, val_loader = loaders
    model = build_model(backbone=args.backbone, mode=mode,
                        pretrained=not args.no_pretrained)
    cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "num_emotions": 7,
        "device": args.device,
        "use_fuzzy": use_fuzzy,
        "verbose": args.verbose,
        "ckpt": f"checkpoints/ablation_{mode}_fuzzy{int(use_fuzzy)}_s{seed}.pt",
    }
    report = FLEOTrainer(model, train_loader, val_loader, cfg).run()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def agg(values):
    mean = st.mean(values)
    sd = st.pstdev(values) if len(values) > 1 else 0.0
    return mean, sd


def main():
    ap = argparse.ArgumentParser(description="FLEO ablation study (Table 2)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--backbone", default="yolov12s")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/ablation.json")
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = find_dataset(args.data)
    use_norm = args.backbone.startswith("resnet")
    # Build the loaders once; the WeightedRandomSampler reshuffles every epoch.
    loaders = build_loaders(root, args.img_size, args.batch_size,
                            args.num_workers, normalize=use_norm, balanced=True)

    print(f"\n=== FLEO ablation on FER2013 ===")
    print(f"backbone={args.backbone}  epochs={args.epochs}  seeds={args.seeds}\n")

    results = {}
    for label, mode, use_fuzzy in ABLATION_ROWS:
        accs, f1s = [], []
        for seed in args.seeds:
            print(f"--- {label}  (seed {seed}) ---")
            rep = run_one(args, mode, use_fuzzy, seed, loaders)
            accs.append(rep["accuracy"]); f1s.append(rep["macro_f1"])
            print(f"    acc={rep['accuracy']:.2f}  macro-F1={rep['macro_f1']:.2f}")
        results[label] = {
            "mode": mode, "fuzzy": use_fuzzy,
            "acc": accs, "macro_f1": f1s,
            "acc_mean_std": agg(accs), "f1_mean_std": agg(f1s),
        }

    # ---- pretty table -------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'Method':<28}{'Acc.':>14}{'macro-F1':>16}")
    print("-" * 60)
    for label, r in results.items():
        am, asd = r["acc_mean_std"]; fm, fsd = r["f1_mean_std"]
        print(f"{label:<28}{am:>8.2f}±{asd:<4.2f}{fm:>10.2f}±{fsd:<4.2f}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
