"""
Evaluate a trained FLEO checkpoint and produce the paper's qualitative figures.
==============================================================================
Outputs (under --out, default results/):
  * printed per-class precision / recall / F1 table
  * confusion_matrix.png   -- 7x7 heatmap; inspect the anger/sad & fear/surprise
                              cells (paper Sec. V-E)
  * tsne.png               -- t-SNE of the penultimate features, colored by
                              emotion (shows class separation from FLEO)
  * report.json            -- accuracy, macro-F1, per-class metrics, confusion

Usage
-----
    python evaluate.py --data /kaggle/input/fer2013 \
        --ckpt checkpoints/best_fleo.pt --backbone yolov12s --mode fleo_full
"""

import argparse
import json
import os

import torch

from train_fer2013 import build_loaders, build_model, find_dataset
from utils.metrics import full_report
from train.trainer import EMOTIONS_7


@torch.no_grad()
def collect(model, loader, device, max_feat=4000):
    model.eval()
    preds, targets, feats, fy = [], [], [], []
    for imgs, labels in loader:
        logits, f = model(imgs.to(device), return_feat=True)
        preds.append(logits.argmax(-1).cpu())
        targets.append(labels)
        if sum(len(t) for t in fy) < max_feat:         # cap features for t-SNE
            feats.append(f.cpu()); fy.append(labels)
    return (torch.cat(preds), torch.cat(targets),
            torch.cat(feats), torch.cat(fy))


def plot_confusion(cm, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(cm, dtype=float)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)     # row-normalized
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("FER2013 confusion matrix (row-normalized)")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_tsne(feats, labels, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    emb = TSNE(n_components=2, init="pca", perplexity=30,
               learning_rate="auto").fit_transform(feats.numpy())
    fig, ax = plt.subplots(figsize=(7, 6))
    for c, name in enumerate(names):
        m = labels.numpy() == c
        ax.scatter(emb[m, 0], emb[m, 1], s=6, label=name, alpha=0.6)
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_title("t-SNE of penultimate features (colored by emotion)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluate FLEO + make figures")
    ap.add_argument("--data", default=None)
    ap.add_argument("--ckpt", default="checkpoints/best_fleo.pt")
    ap.add_argument("--backbone", default="yolov12s")
    ap.add_argument("--mode", default="fleo_full",
                    choices=["baseline", "se", "fleo_nobind", "fleo_full"])
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-tsne", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    root = find_dataset(args.data)
    use_norm = args.backbone.startswith("resnet")
    _, val_loader = build_loaders(root, args.img_size, args.batch_size,
                                  args.num_workers, normalize=use_norm, balanced=False)

    model = build_model(backbone=args.backbone, mode=args.mode, pretrained=False)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state)
    model.to(args.device)
    print(f"[eval] loaded {args.ckpt}  (backbone={args.backbone}, mode={args.mode})")

    preds, targets, feats, fy = collect(model, val_loader, args.device)
    rep = full_report(preds, targets, 7, EMOTIONS_7)

    print(f"\nAccuracy {rep['accuracy']:.2f}%   macro-F1 {rep['macro_f1']:.2f}%")
    print(f"{'emotion':<10}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}")
    for name, m in rep["per_class"].items():
        print(f"{name:<10}{m['precision']:>8.1f}{m['recall']:>8.1f}"
              f"{m['f1']:>8.1f}{m['support']:>9}")

    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    plot_confusion(rep["confusion_matrix"], rep["class_names"],
                   os.path.join(args.out, "confusion_matrix.png"))
    print(f"[eval] wrote {args.out}/confusion_matrix.png")
    if not args.no_tsne:
        plot_tsne(feats, fy, rep["class_names"], os.path.join(args.out, "tsne.png"))
        print(f"[eval] wrote {args.out}/tsne.png")


if __name__ == "__main__":
    main()
