"""Rebuild a raw class-folder FER dataset from a YOLO-format prepared dataset.

The Vitis AI quantization entry point (``deploy/run_vitis.py`` ->
``train_fer2013.find_dataset``) expects the raw image-folder layout::

    <out>/train/<emotion>/*.jpg
    <out>/test/<emotion>/*.jpg

but the local dataset is in YOLO detection form (``images/{train,val}`` +
``labels/{train,val}/<stem>.txt`` whose first token is the class id). This
script reconstructs the class-folder layout by reading each label's class id
and copying the image into the matching emotion folder. The YOLO ``val`` split
maps to ``test`` (the split ``find_dataset`` uses for evaluation).

Usage (inside the Vitis AI docker, at /work):
    python deploy/yolo_to_folders.py --src datasets/fer2013 --out data/fer2013_raw
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

SPLIT_MAP = {"train": "train", "val": "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_names(src: Path) -> dict:
    dy = src / "data.yaml"
    names = yaml.safe_load(dy.read_text())["names"]
    # names may be a dict {0: 'angry', ...} or a list.
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def convert(src: Path, out: Path) -> None:
    names = load_names(src)
    print(f"[convert] classes: {names}")
    total = 0
    for ysplit, osplit in SPLIT_MAP.items():
        img_dir = src / "images" / ysplit
        lbl_dir = src / "labels" / ysplit
        if not img_dir.is_dir():
            print(f"[convert] skip missing split: {img_dir}")
            continue
        n = 0
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            first = lbl.read_text().strip().splitlines()
            if not first:
                continue
            cls = int(first[0].split()[0])
            emotion = names.get(cls, f"class_{cls}")
            dst_dir = out / osplit / emotion
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst_dir / img.name)
            n += 1
        print(f"[convert] {ysplit} -> {osplit}: {n} images")
        total += n
    print(f"[convert] done: {total} images -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="datasets/fer2013",
                    help="YOLO-format dataset root (images/ + labels/ + data.yaml)")
    ap.add_argument("--out", default="data/fer2013_raw",
                    help="output raw class-folder root (train/ + test/)")
    args = ap.parse_args()
    convert(Path(args.src), Path(args.out))


if __name__ == "__main__":
    main()
