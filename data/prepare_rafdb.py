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

from .emotions import CANON, NAME_TO_ID, RAFDB_TO_CANON
from .prepare_common import (
    ensure_dirs, write_label, write_data_yaml,
    is_image, has_images, split_from_path, print_tree,
)

# Emotion-name aliases (some RAF-DB releases use names instead of 1..7 ids).
_EMOTION_ALIASES = {
    "anger": "angry", "angry": "angry",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear",
    "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral",
}


def _canon_name(name: str) -> str | None:
    n = name.strip().lower()
    if n in NAME_TO_ID:
        return n
    return _EMOTION_ALIASES.get(n)


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


def _class_dirs(d: Path):
    """Return {canonical_class_id: class_dir} for a dir of class subfolders.

    Recognises numeric RAF-DB ids (1..7, remapped via RAFDB_TO_CANON) and
    emotion-named folders (mapped via the canonical FER order).
    """
    out: dict[int, Path] = {}
    try:
        children = sorted(d.iterdir())
    except OSError:
        return out
    for c in children:
        if not c.is_dir() or not has_images(c):
            continue
        if c.name.isdigit() and 1 <= int(c.name) <= 7:
            out[RAFDB_TO_CANON[int(c.name)]] = c
        else:
            canon = _canon_name(c.name)
            if canon is not None:
                out[NAME_TO_ID[canon]] = c
    return out


def _class_groups(src: Path):
    """Find every dir that holds >=3 class subfolders; tag each with a split.

    Layout-agnostic: matches ``.../train/1..7``, ``.../DATASET/test/<emotion>``,
    or a split-less ``.../<class>`` root, inferring the split from the path.
    """
    groups = []
    for d in [src, *[x for x in src.rglob("*") if x.is_dir()]]:
        cdirs = _class_dirs(d)
        if len(cdirs) >= 3:
            groups.append((split_from_path(d), d, cdirs))
    return groups


def from_groups(groups, out: Path, limit: int | None = None):
    """Convert discovered class-folder groups into YOLO detection form."""
    from PIL import Image

    ensure_dirs(out)
    counts = {"train": 0, "val": 0}
    known_split = any(s for s, _, _ in groups)
    n = 0
    idx = 0
    for split, _parent, cdirs in groups:
        for cls, cls_dir in cdirs.items():
            for img_path in sorted(cls_dir.iterdir()):
                if not is_image(img_path):
                    continue
                if limit and n >= limit:
                    break
                sp = split or ("val" if (not known_split and idx % 7 == 0) else "train")
                idx += 1
                stem = f"raf_{n:06d}"
                Image.open(img_path).convert("RGB").save(
                    out / "images" / sp / f"{stem}.jpg", quality=95
                )
                write_label(out, sp, stem, cls)
                counts[sp] += 1
                n += 1
    return counts


def _image_index(src: Path) -> dict[str, Path]:
    """Index all images by filename and by stem for O(1) CSV lookups."""
    idx: dict[str, Path] = {}
    for f in src.rglob("*"):
        if is_image(f):
            idx.setdefault(f.name, f)
            idx.setdefault(f.stem, f)
    return idx


def _from_csv_labels(src: Path, out: Path, limit: int | None = None):
    """Convert a ``*labels*.csv`` (image name + 1..7 label) RAF-DB release."""
    import csv as _csv

    from PIL import Image

    label_csvs = [p for p in src.rglob("*.csv") if "label" in p.name.lower()]
    if not label_csvs:
        return None
    index = _image_index(src)
    if not index:
        return None

    ensure_dirs(out)
    counts = {"train": 0, "val": 0}
    n = 0
    for csvp in label_csvs:
        split = split_from_path(csvp)
        if split is None:
            low = csvp.name.lower()
            split = "train" if "train" in low else "val" if "test" in low else "train"
        rows = list(_csv.reader(csvp.read_text().splitlines()))
        if not rows:
            continue
        # Locate the image-name column and the label column from the header.
        header = [h.strip().lower() for h in rows[0]]
        data_start = 1
        name_col, label_col = 0, 1
        if any(h in ("image", "name", "filename", "file", "img") for h in header):
            name_col = next(i for i, h in enumerate(header)
                            if h in ("image", "name", "filename", "file", "img"))
            label_col = next((i for i, h in enumerate(header)
                              if h in ("label", "emotion", "class", "expression")), 1)
        else:
            data_start = 0  # headerless
        for row in rows[data_start:]:
            if len(row) <= max(name_col, label_col):
                continue
            key = Path(row[name_col].strip()).name
            img = index.get(key) or index.get(Path(key).stem)
            try:
                raw = int(float(row[label_col]))
            except ValueError:
                continue
            if img is None or raw not in RAFDB_TO_CANON:
                continue
            if limit and n >= limit:
                break
            cls = RAFDB_TO_CANON[raw]
            stem = f"raf_{n:06d}"
            Image.open(img).convert("RGB").save(
                out / "images" / split / f"{stem}.jpg", quality=95
            )
            write_label(out, split, stem, cls)
            counts[split] += 1
            n += 1
    return counts if (counts["train"] + counts["val"]) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="RAF-DB root")
    ap.add_argument("--out", default="datasets/rafdb")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        parent = src.parent if src.parent.exists() else Path("/kaggle/input")
        print_tree(parent, max_depth=2)
        raise SystemExit(f"--src does not exist: {src} (see mounted datasets above)")

    label_file = _find(src, "EmoLabel/list_patition_label.txt", "list_patition_label.txt")
    aligned = _find(src, "Image/aligned", "aligned", "Image/original", "original")
    if label_file is None or aligned is None:
        # (1) class-folder form (numeric 1..7 or emotion-named), any nesting.
        groups = _class_groups(src)
        if groups:
            counts = from_groups(groups, out, args.limit)
            yml = write_data_yaml(out, CANON)
            print(f"RAF-DB (folder form) prepared -> {out}  counts={counts}")
            print(f"data.yaml: {yml}")
            return
        # (2) CSV-label form (e.g. train_labels.csv/test_labels.csv + image dirs).
        counts = _from_csv_labels(src, out, args.limit)
        if counts is not None:
            yml = write_data_yaml(out, CANON)
            print(f"RAF-DB (csv form) prepared -> {out}  counts={counts}")
            print(f"data.yaml: {yml}")
            return
        print_tree(src)
        raise SystemExit(
            f"Could not locate RAF-DB under {src} in any known layout "
            f"(official aligned+txt, class folders, or *labels*.csv). See tree above."
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
