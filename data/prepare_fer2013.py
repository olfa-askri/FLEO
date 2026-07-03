"""Prepare FER2013 as a YOLO detection dataset (7 classes, full-image boxes).

Supports both common distributions:

* CSV form  ``fer2013.csv`` (columns: ``emotion``, ``pixels``, ``Usage``), and
* image-folder form ``<src>/{train,test}/<emotion_name>/*.jpg``.

Usage:
    python -m data.prepare_fer2013 --src /path/to/fer2013.csv --out datasets/fer2013
    python -m data.prepare_fer2013 --src /path/to/fer2013_images --out datasets/fer2013

FER2013 native ids already match the canonical order (see emotions.py).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .emotions import CANON, NAME_TO_ID, FER2013_TO_CANON
from .prepare_common import ensure_dirs, write_label, write_data_yaml, save_gray_as_jpg

# FER2013 "Usage" -> our split.  PublicTest + PrivateTest both map to val.
USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "val", "PrivateTest": "val"}


def from_csv(src: Path, out: Path, limit: int | None = None):
    ensure_dirs(out)
    counts = {"train": 0, "val": 0}
    with open(src, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            split = USAGE_TO_SPLIT.get(row.get("Usage", "Training"), "train")
            cls = FER2013_TO_CANON[int(row["emotion"])]
            pix = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
            side = int(round(len(pix) ** 0.5))
            img = pix.reshape(side, side)
            stem = f"fer_{i:06d}"
            save_gray_as_jpg(img, out / "images" / split / f"{stem}.jpg")
            write_label(out, split, stem, cls)
            counts[split] += 1
    return counts


def from_folder(src: Path, out: Path, limit: int | None = None):
    ensure_dirs(out)
    # Map folder split names -> our split.
    split_map = {"train": "train", "training": "train", "test": "val",
                 "val": "val", "validation": "val", "public_test": "val", "private_test": "val"}
    counts = {"train": 0, "val": 0}
    n = 0
    for split_dir in sorted(src.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_map.get(split_dir.name.lower())
        if split is None:
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            name = cls_dir.name.lower()
            if name not in NAME_TO_ID:
                # tolerate alternative folder names
                alias = {"anger": "angry", "happiness": "happy", "sadness": "sad"}.get(name)
                if alias is None:
                    continue
                name = alias
            cls = NAME_TO_ID[name]
            for img_path in sorted(cls_dir.glob("*")):
                if limit and n >= limit:
                    break
                stem = f"fer_{n:06d}"
                _copy_as_jpg(img_path, out / "images" / split / f"{stem}.jpg")
                write_label(out, split, stem, cls)
                counts[split] += 1
                n += 1
    return counts


def _copy_as_jpg(src_img: Path, dst: Path):
    from PIL import Image

    Image.open(src_img).convert("RGB").save(dst, quality=95)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="fer2013.csv OR image-folder root")
    ap.add_argument("--out", default="datasets/fer2013")
    ap.add_argument("--limit", type=int, default=None, help="cap images (debug)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if src.is_file() and src.suffix.lower() == ".csv":
        counts = from_csv(src, out, args.limit)
    elif src.is_dir():
        counts = from_folder(src, out, args.limit)
    else:
        raise SystemExit(f"Unrecognized --src: {src}")

    yml = write_data_yaml(out, CANON)
    print(f"FER2013 prepared -> {out}  counts={counts}")
    print(f"data.yaml: {yml}")


if __name__ == "__main__":
    main()
