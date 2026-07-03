"""Generate a tiny synthetic 7-class detection dataset for CPU smoke tests.

Not for accuracy -- purely to exercise the full ultralytics train/val/export path
without downloading FER2013 or RAF-DB.

    python -m data.make_synthetic --out datasets/synthetic --n 24 --imgsz 96
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .emotions import CANON
from .prepare_common import ensure_dirs, write_label, write_data_yaml


def _make_image(cls: int, imgsz: int, rng) -> np.ndarray:
    """A per-class colored blob so the task is learnable-in-principle but trivial."""
    img = rng.integers(0, 60, size=(imgsz, imgsz, 3), dtype=np.uint8)
    color = np.array([(cls * 37) % 256, (cls * 91) % 256, (cls * 53) % 256], dtype=np.uint8)
    cx, cy = imgsz // 2, imgsz // 2
    r = imgsz // 3
    yy, xx = np.ogrid[:imgsz, :imgsz]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    img[mask] = color
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="datasets/synthetic")
    ap.add_argument("--n", type=int, default=24, help="images per split")
    ap.add_argument("--imgsz", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    ensure_dirs(out)
    rng = np.random.default_rng(args.seed)
    for split in ("train", "val"):
        for i in range(args.n):
            cls = i % len(CANON)
            img = _make_image(cls, args.imgsz, rng)
            stem = f"syn_{split}_{i:04d}"
            Image.fromarray(img).save(out / "images" / split / f"{stem}.jpg", quality=90)
            write_label(out, split, stem, cls)
    yml = write_data_yaml(out, CANON)
    print(f"Synthetic dataset -> {out}  ({args.n}/split)  data.yaml={yml}")


if __name__ == "__main__":
    main()
