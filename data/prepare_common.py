"""Shared helpers for casting classification FER datasets into YOLO detection.

The FER task is *cast as detection* (Section V-F): each image contributes a single
full-frame box whose class is the emotion.  This keeps the detector/label pipeline
identical to a real face-detection deployment while letting classification datasets
drive training and evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# A full-image, axis-aligned box in normalized xywh (class prepended by caller).
FULL_IMAGE_BOX = "0.500000 0.500000 1.000000 1.000000"


def ensure_dirs(root: Path):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_label(root: Path, split: str, stem: str, cls: int):
    (root / "labels" / split / f"{stem}.txt").write_text(f"{cls} {FULL_IMAGE_BOX}\n")


def write_data_yaml(root: Path, names):
    d = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    return root / "data.yaml"


def save_gray_as_jpg(arr, path: Path):
    """Save a HxW (grayscale) or HxWx3 uint8 array to a 3-channel jpg."""
    from PIL import Image
    import numpy as np

    a = np.asarray(arr)
    if a.ndim == 2:
        img = Image.fromarray(a.astype("uint8"), mode="L").convert("RGB")
    else:
        img = Image.fromarray(a.astype("uint8"))
    img.save(path, quality=95)
