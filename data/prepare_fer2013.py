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
from .prepare_common import (
    ensure_dirs, write_label, write_data_yaml, save_gray_as_jpg,
    is_image, has_images, split_from_path, print_tree,
)

# FER2013 "Usage" -> our split.  PublicTest + PrivateTest both map to val.
USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "val", "PrivateTest": "val"}

# Emotion folder-name aliases -> canonical FER name.
EMOTION_ALIASES = {
    "anger": "angry", "angry": "angry",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "afraid": "fear", "fearful": "fear",
    "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad",
    "surprise": "surprise", "surprised": "surprise", "surprising": "surprise",
    "neutral": "neutral", "neutrality": "neutral",
}


def _canon_emotion(name: str) -> str | None:
    n = name.strip().lower()
    if n in NAME_TO_ID:
        return n
    return EMOTION_ALIASES.get(n)


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
    """Auto-discover emotion-named class folders anywhere under ``src``.

    Handles ``<root>/{train,test}/<emotion>/*.jpg`` as well as nested
    (``<root>/DATASET/train/<emotion>/``) or split-less (``<root>/<emotion>/``)
    layouts.  Split is inferred from the path; if no val split is present, ~15%
    of the data is carved off deterministically.
    """
    ensure_dirs(out)

    # Gather (image_path, split_or_None, class_id) over all emotion class dirs.
    items: list[tuple[Path, str | None, int]] = []
    for d in sorted(src.rglob("*")):
        if not d.is_dir():
            continue
        canon = _canon_emotion(d.name)
        if canon is None or not has_images(d):
            continue
        split = split_from_path(d)
        cls = NAME_TO_ID[canon]
        for f in sorted(d.iterdir()):
            if is_image(f):
                items.append((f, split, cls))

    if not items:
        print_tree(src)
        raise SystemExit(
            f"No emotion-named image folders found under {src}.\n"
            f"Expected folders like train/happy, test/angry, ... (see tree above)."
        )

    # If nothing was labelled 'val', deterministically carve ~15% off for val.
    has_val = any(s == "val" for _, s, _ in items)
    counts = {"train": 0, "val": 0}
    n = 0
    for i, (img_path, split, cls) in enumerate(items):
        if limit and n >= limit:
            break
        if split is None:
            split = "val" if (not has_val and i % 7 == 0) else "train"
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

    if not src.exists():
        parent = src.parent if src.parent.exists() else Path("/kaggle/input")
        print_tree(parent, max_depth=2)
        raise SystemExit(
            f"--src does not exist: {src}\n"
            f"Check the mounted dataset name above and pass the correct path "
            f"(e.g. --src /kaggle/input/<dataset-folder>)."
        )

    if src.is_file() and src.suffix.lower() == ".csv":
        counts = from_csv(src, out, args.limit)
    else:
        # A directory: prefer an embedded fer2013.csv, else discover image folders.
        csvs = list(src.rglob("fer2013.csv")) + list(src.rglob("*.csv"))
        if csvs and any(c.name.lower() == "fer2013.csv" for c in csvs):
            csv = next(c for c in csvs if c.name.lower() == "fer2013.csv")
            print(f"[fer2013] using CSV {csv}")
            counts = from_csv(csv, out, args.limit)
        else:
            counts = from_folder(src, out, args.limit)

    yml = write_data_yaml(out, CANON)
    print(f"FER2013 prepared -> {out}  counts={counts}")
    print(f"data.yaml: {yml}")


if __name__ == "__main__":
    main()
