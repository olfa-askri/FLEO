"""
Train FLEO on the FER2013 dataset.
==================================
FER2013 (Kaggle: msambare/fer2013) is 48x48 **grayscale** face images,
7 emotions, already split into `train/` and `test/` folders:

    fer2013/
    ├── train/
    │   ├── angry/      *.jpg
    │   ├── disgust/    *.jpg
    │   ├── fear/       *.jpg
    │   ├── happy/      *.jpg
    │   ├── neutral/    *.jpg
    │   ├── sad/        *.jpg
    │   └── surprise/   *.jpg
    └── test/   (same 7 sub-folders)

Two things this script handles that the generic skeleton does not:
  1. Grayscale -> 3-channel RGB  (the model's stem expects 3 input channels).
  2. Label remap so the class index order matches the FACS confusion prior
     used by FuzzyConfusionLoss:
        FACS order : anger, disgust, fear, happy, sad, surprise, neutral
        ImageFolder: angry, disgust, fear, happy, neutral, sad, surprise (alphabetical)

Usage
-----
    # Local (after downloading/unzipping FER2013 into ./data/fer2013):
    python train_fer2013.py --data data/fer2013 --epochs 50 --batch-size 64

    # Kaggle (add the FER2013 dataset via "+ Add Input"):
    python train_fer2013.py --data /kaggle/input/fer2013 --epochs 50

The script auto-detects a few common dataset locations if --data is omitted.
Best checkpoint (by macro-F1) is written to checkpoints/best_fleo.pt
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import FLEOClassifier
from train.trainer import FLEOTrainer


# FACS confusion-prior order expected by FuzzyConfusionLoss / FACS_PRIOR_7.
FACS_ORDER = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
# FER2013 folder names map onto the FACS names like so ("angry" == "anger").
FOLDER_TO_FACS = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprise",
}

COMMON_PATHS = [
    "data/fer2013",
    "/kaggle/input/fer2013",
    "/kaggle/input/fer2013/fer2013",
    "/kaggle/input/face-expression-recognition-dataset/images",
]


def find_dataset(user_path: str | None) -> str:
    candidates = [user_path] if user_path else COMMON_PATHS
    for root in candidates:
        if root and os.path.isdir(os.path.join(root, "train")):
            return root
    tried = "\n  ".join(c for c in candidates if c)
    sys.exit(
        f"[FER2013] Could not find a 'train/' folder under any of:\n  {tried}\n"
        f"Pass the dataset root explicitly:  python train_fer2013.py --data <path>"
    )


def make_target_transform(class_to_idx: dict) -> callable:
    """Remap ImageFolder's alphabetical label index to the FACS index order."""
    facs_index = {name: i for i, name in enumerate(FACS_ORDER)}
    # imagefolder_idx -> facs_idx
    remap = {}
    for folder_name, folder_idx in class_to_idx.items():
        facs_name = FOLDER_TO_FACS.get(folder_name.lower(), folder_name.lower())
        if facs_name not in facs_index:
            sys.exit(f"[FER2013] Unexpected emotion folder: '{folder_name}'")
        remap[folder_idx] = facs_index[facs_name]
    remap_t = torch.tensor([remap[i] for i in range(len(remap))])
    return lambda y: int(remap_t[y])


# ImageNet stats — required when using a pretrained ResNet backbone.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _make_sampler(dataset):
    """WeightedRandomSampler that balances FER2013's skewed classes
    (e.g. 'disgust' has ~15x fewer samples than 'happy')."""
    from torch.utils.data import WeightedRandomSampler

    targets = dataset.targets if hasattr(dataset, "targets") else None
    if targets is None:
        return None
    targets = torch.tensor(targets)
    counts = torch.bincount(targets)
    class_w = 1.0 / counts.clamp(min=1).float()
    sample_w = class_w[targets]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def build_loaders(root: str, img_size: int, batch_size: int, num_workers: int,
                  normalize: bool = True, balanced: bool = True):
    # FER2013 normally has train/ + test/. Some mirrors use 'val', and some
    # have the emotion folders directly at the root (no split at all).
    train_dir = os.path.join(root, "train")
    if not os.path.isdir(train_dir):
        train_dir = root                       # emotion folders directly at root
    val_dir = os.path.join(root, "test")
    if not os.path.isdir(val_dir):
        val_dir = os.path.join(root, "val")

    norm = [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)] if normalize else []

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),   # 48x48 gray -> 3ch
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        *norm,
    ])
    # Strong augmentation to fight the overfitting seen with weak augmentation.
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), antialias=True),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        *norm,
        transforms.RandomErasing(p=0.25),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    tt = make_target_transform(train_ds.class_to_idx)
    train_ds.target_transform = tt

    if os.path.isdir(val_dir):
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
        val_ds.target_transform = make_target_transform(val_ds.class_to_idx)
    else:
        # No held-out split: carve 10% off train for validation.
        print("[FER2013] no test/val folder found — splitting 90/10 from train.")
        full = datasets.ImageFolder(train_dir, transform=eval_tf)
        full.target_transform = tt
        n_val = max(1, int(0.1 * len(full)))
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(full), generator=g).tolist()
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        val_ds = torch.utils.data.Subset(full, val_idx)
        train_ds = torch.utils.data.Subset(train_ds, train_idx)

    print(f"[FER2013] train={len(train_ds)} imgs  val={len(val_ds)} imgs")

    sampler = _make_sampler(train_ds) if balanced else None
    if sampler is not None:
        print("[FER2013] using class-balanced WeightedRandomSampler")
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=(sampler is None), sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    ap = argparse.ArgumentParser(description="Train FLEO on FER2013")
    ap.add_argument("--data", default=None, help="FER2013 root (contains train/ and test/)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--backbone", default="yolov12s",
                    choices=["tiny", "resnet18", "resnet34", "resnet50",
                             "yolov12n", "yolov12s", "yolov12m", "yolov12l", "yolov12x"],
                    help="feature extractor (yolov12s = the paper's backbone)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="train the backbone from scratch (no COCO/ImageNet weights)")
    ap.add_argument("--no-balanced", action="store_true",
                    help="disable the class-balanced sampler")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    root = find_dataset(args.data)
    print(f"[FER2013] dataset root: {root}")
    print(f"[FER2013] device: {args.device}  backbone: {args.backbone}")

    # ResNet uses ImageNet mean/std; YOLOv12 (and tiny) expect plain [0,1].
    use_norm = args.backbone.startswith("resnet")
    train_loader, val_loader = build_loaders(
        root, args.img_size, args.batch_size, args.num_workers,
        normalize=use_norm, balanced=not args.no_balanced,
    )

    # train_only=False -> Gram-Schmidt runs in BOTH train and eval, so the
    # validation forward matches training. (train_only=True is only for the
    # FPGA export route, where GS is folded out *after* training.)
    model = FLEOClassifier(
        num_emotions=7,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        train_only=False,
    )
    cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "num_emotions": 7,
        "device": args.device,
    }
    FLEOTrainer(model, train_loader, val_loader, cfg).run()
    print("[FER2013] done. Best checkpoint: checkpoints/best_fleo.pt")


if __name__ == "__main__":
    main()
