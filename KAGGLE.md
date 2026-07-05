# Running FLEO-FER on Kaggle (free GPU, ~30 h/week)

This trains the FLEO-augmented YOLOv12-S emotion detector on **FER2013** and
**RAF-DB**, evaluates the fold-out delta (Δ_fold), and exports the three
deployment routes (R1 fold-out / R2 Householder / R3 hybrid) as ONNX for the
Vitis AI → ZCU104 flow. The GPU half fits comfortably in Kaggle's 30 h/week quota.

---

## 0. One-time: create the notebook

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. Right sidebar → **Settings**:
   - **Accelerator** → **GPU P100** (preferred; single 16 GB card, simplest) or **GPU T4 ×2**.
   - **Internet** → **On** (needed for `git clone` + `pip install`).
   - **Persistence** → *Files only* (optional, keeps `/kaggle/working` between sessions).
3. Right sidebar → **Input → Add Input**, search and add:
   - `msambare/fer2013` → mounts at `/kaggle/input/fer2013`
   - `shuvoalok/raf-db-dataset` → mounts at `/kaggle/input/raf-db-dataset`

> Session limit is 12 h; the full 2-dataset × 3-seed matrix is ~5–10 h at
> `imgsz=160`, so it fits in one session. Start with **1 seed** to validate, then
> scale to 3.

---

## 1. Clone the repo + install deps

Kaggle already ships CUDA PyTorch — **do not reinstall torch**. Only add the missing packages:

```python
!git clone https://github.com/olfa-askri/FLEO.git
%cd FLEO
!pip install -q ultralytics onnx onnxruntime onnxscript
import torch; print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
```

---

## 2. Prepare the datasets → YOLO detection format

The prep scripts auto-detect the on-disk layout (FER2013 emotion folders; RAF-DB
either the official `list_patition_label.txt`+`aligned` form **or** the numeric
`train/test/1..7/` folder form used by `shuvoalok/raf-db-dataset`).

```python
!python -m data.prepare_fer2013 --src /kaggle/input/fer2013        --out datasets/fer2013
!python -m data.prepare_rafdb   --src /kaggle/input/raf-db-dataset --out datasets/rafdb
```

Each writes `datasets/<name>/data.yaml` (7 classes, canonical FER order). Sanity check:

```python
!ls datasets/fer2013/images/train | head -3 ; !cat datasets/fer2013/data.yaml
```

---

## 3A. Quick path — the whole matrix in one command (per dataset)

`run_matrix` trains baseline + FLEO for the given seeds, exports R1/R2/R3, and
computes Δ_fold. Start with a single seed:

```python
# FER2013 (start with 1 seed to validate the full pipeline)
!python -m scripts.run_matrix --data datasets/fer2013/data.yaml --dataset fer2013 \
    --seeds 0 --epochs 100 --imgsz 160 --batch 64 --device 0

# RAF-DB
!python -m scripts.run_matrix --data datasets/rafdb/data.yaml --dataset rafdb \
    --seeds 0 --epochs 100 --imgsz 160 --batch 64 --device 0
```

When happy, rerun with `--seeds 0 1 2` for the paper's mean±sd (Table VIII).

## 3B. Granular path — step by step

```python
# 1) Train FLEO (full graph: Gram-Schmidt + aux losses active)
!python -m scripts.train --data datasets/fer2013/data.yaml --variant fleo \
    --epochs 100 --imgsz 160 --batch 64 --device 0 --seeds 0 --workers 2

# 2) Train the baseline (unaugmented YOLOv12-S) for comparison
!python -m scripts.train --data datasets/fer2013/data.yaml --variant baseline \
    --epochs 100 --imgsz 160 --batch 64 --device 0 --seeds 0 --workers 2

# 3) Δ_fold: evaluate the SAME weights as full graph vs folded graph
!python -m scripts.evaluate --data datasets/fer2013/data.yaml \
    --weights runs/fleo/fleo_seed0/weights/best.pt --route full   \
    --imgsz 160 --device 0 --out results/fer_full.json
!python -m scripts.evaluate --data datasets/fer2013/data.yaml \
    --weights runs/fleo/fleo_seed0/weights/best.pt --route folded \
    --imgsz 160 --device 0 --out results/fer_folded.json

# 4) Δ_fold / macro-F1 aggregated across seeds
!python -m scripts.deltas --data datasets/fer2013/data.yaml --dataset fer2013 \
    --weights runs/fleo/fleo_seed0/weights/best.pt --imgsz 160 --device 0 \
    --out results/fer_deltas.json
```

---

## 4. Export the three routes for Vitis AI (FP32 ONNX)

```python
W=runs/fleo/fleo_seed0/weights/best.pt
!python -m scripts.export --weights runs/fleo/fleo_seed0/weights/best.pt --route r1 --imgsz 160 --verify
!python -m scripts.export --weights runs/fleo/fleo_seed0/weights/best.pt --route r2 --imgsz 160 --verify
!python -m scripts.export --weights runs/fleo/fleo_seed0/weights/best.pt --route r3 --imgsz 160
```

Outputs land in `export/`:
- `r1.onnx` — fold-out (Gram-Schmidt deleted; single DPU subgraph). **This is the one you deploy first.**
- `r2.onnx` — Householder constant-folded into a 1×1 conv (single DPU subgraph).
- `r3.onnx` — hybrid (Gram-Schmidt kept; compiler will partition DPU–CPU–DPU).
- matching `*.pt` (for `vai_q_pytorch`) and `*.meta.json`.

---

## 5. Save / download artifacts

Everything under `/kaggle/working/FLEO` is your output. Zip what you need:

```python
!cd /kaggle/working/FLEO && zip -qr /kaggle/working/fleo_artifacts.zip \
    export results runs/fleo/*/weights/best.pt
print("done -> /kaggle/working/fleo_artifacts.zip")
```

Then **Save Version** (commit) the notebook, or use the **Output** tab to download
`fleo_artifacts.zip`. Take `export/r1.onnx` (+ `.pt`) into the Vitis AI 3.5
PyTorch docker for quantization/compile — see `deploy/` and `README.md`.

---

## Tips & gotchas

- **Device**: use `--device 0`. If you picked T4 ×2, still use `--device 0`
  (single-GPU is simplest in notebooks); lower `--batch` to 32 on a single T4.
- **imgsz**: `160` is a fast, quota-friendly default (FER2013 is 48 px native,
  RAF-DB aligned ~100 px). Bump to `320`/`640` for higher fidelity if quota allows;
  keep `--imgsz` identical in train, evaluate, and export.
- **Training is from scratch** (random init from `yolo12s.yaml`); no COCO
  pretrain is downloaded. Expect this to converge over ~100 epochs.
- **Workers**: Kaggle gives ~4 vCPUs; `--workers 2` is safe.
- **Wasted quota guard**: run the CPU smoke first to confirm the code path:
  `!python -m data.make_synthetic --out datasets/synthetic && python -m scripts.train --data datasets/synthetic/data.yaml --variant fleo --epochs 2 --imgsz 96 --batch 8 --device 0`
- **Multi-session**: if you split FER2013 and RAF-DB across sessions, enable
  *Persistence → Files only* so `datasets/` and `runs/` survive.
