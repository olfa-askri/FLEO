"""Prepare RAF-DB (basic) as a YOLO detection dataset (7 classes, full-image boxes).

Expected RAF-DB layout (the standard Kaggle/official "basic" release):

    <src>/
      Image/aligned/               # train_00001_aligned.jpg, test_0001_aligned.jpg
      EmoLabel/list_patition_label.txt   # "train_00001.jpg 5" per line (labels 1..7)

The label file's split prefix (train_/test_) drives our train/val split; RAF-DB's
1..7 labels are remapped to the canonical FER order (see emotions.py).

Usage:
    python -m data.prepare_rafdb --src /path/to/RAF-DB --out datasets/rafdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .emotions import CANON, RAFDB_TO_CANON
from .prepare_common import ensure_dirs, write_label, write_data_yaml


def _find(src: Path, *cands: str) -> Path | None:
    for c in cands:
        p = src / c
        if p.exists():
            return p
    # fall back to a recursive search
    for c in cands:
        hits = list(src.rglob(Path(c).name))
        if hits:
            return hits[0]
    return None


def _resolve_image(aligned_dir: Path, base: str) -> Path | None:
    """Map a label line stem (e.g. 'train_00001') to its aligned image file."""
    for name in (f"{base}_aligned.jpg", f"{base}.jpg", f"{base}_aligned.png", f"{base}.png"):
        p = aligned_dir / name
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="RAF-DB root")
    ap.add_argument("--out", default="datasets/rafdb")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    label_file = _find(src, "EmoLabel/list_patition_label.txt", "list_patition_label.txt")
    aligned = _find(src, "Image/aligned", "aligned", "Image/original", "original")
    if label_file is None or aligned is None:
        raise SystemExit(
            f"Could not locate label file / aligned images under {src}\n"
            f"  label_file={label_file}  aligned_dir={aligned}"
        )
    aligned = aligned if aligned.is_dir() else aligned.parent

    from PIL import Image

    ensure_dirs(out)
    counts = {"train": 0, "val": 0}
    n = 0
    for line in label_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        fname, raw = line.split()
        base = Path(fname).stem  # train_00001
        split = "train" if base.startswith("train") else "val"
        cls = RAFDB_TO_CANON[int(raw)]
        img_path = _resolve_image(aligned, base)
        if img_path is None:
            continue
        if args.limit and n >= args.limit:
            break
        stem = f"raf_{n:06d}"
        Image.open(img_path).convert("RGB").save(out / "images" / split / f"{stem}.jpg", quality=95)
        write_label(out, split, stem, cls)
        counts[split] += 1
        n += 1

    yml = write_data_yaml(out, CANON)
    print(f"RAF-DB prepared -> {out}  counts={counts}")
    print(f"data.yaml: {yml}")


if __name__ == "__main__":
    main()
