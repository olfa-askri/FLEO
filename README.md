# FLEO — Fuzzy Label and Emotion Orthogonalization for YOLOv12-based FER

> **Goal:** un-cropped face → YOLOv12 detector with a FLEO neck → seven-class
> emotion, trained on FER2013 / RAF-DB, exported INT8 for the Xilinx ZCU104 DPU
> with ≤ a small accuracy drop versus the FP32 model.

FLEO is a drop-in neck module that splits the feature tensor into per-emotion
subspaces and forces them apart with a differentiable Gram–Schmidt operator, so
that morphologically similar expressions (anger/sadness, fear/surprise) stop
sharing latent directions. A binding head ties each subspace to one emotion,
and a confusion-aware fuzzy loss spreads label mass along a measured confusion
prior. The orthogonality is a property of the forward pass — no external
penalty to tune.

This repository takes the idea from PyTorch all the way to an FPGA-ready INT8
graph, with a full ablation harness and a journal manuscript.

---

## End-to-end pipeline

```
            FER2013 / RAF-DB (grayscale→RGB, FACS label remap, balanced sampler)
                                   │
                                   ▼
      YOLOv12 backbone ── P3 ──► FLEO block ─┐
                       └─ P4 ──► FLEO block ─┤  (Gram–Schmidt + SE gate + binding)
                                             ▼
                                  MLP emotion head ──► 7 logits
                                             │
                  L = L_Fuzzy(confusion-aware) + λ·L_bind
                                   │
        ┌──────────────────────────┼───────────────────────────┐
        ▼                          ▼                            ▼
  run_ablation.py            evaluate.py                 deploy/ (Route 1)
  Table 2 (5 configs)   confusion matrix + t-SNE   fold out Gram–Schmidt
  mean ± std / 3 seeds   per-class P/R/F1           → Vitis AI INT8 → .xmodel
                                                    → ZCU104 DPU (PYNQ)
```

---

## Status

| Stage | State | Entry point |
|---|---|---|
| FLEO block + losses (math verified) | ✅ done | `models/`, `tests/` |
| Training on FER2013 (Kaggle-ready) | ✅ done | `train_fer2013.py` |
| Ablation harness (Table 2) | ✅ done | `run_ablation.py` |
| Figures (confusion matrix, t-SNE) | ✅ done | `evaluate.py` |
| INT8 simulation (no board needed) | ✅ done | `deploy/simulate_int8.py` |
| Vitis AI INT8 + `.xmodel` export | ✅ scripted (needs Vitis Docker) | `deploy/run_vitis.py`, `deploy/VITIS.md` |
| ZCU104 on-board inference | ⏳ pending hardware | `deploy/export_fpga.py` template |
| Journal manuscript | ✅ draft, results pending runs | `paper/fleo_journal.tex` |

> **Backbone note.** The code targets YOLOv12. When the installed `ultralytics`
> ships no YOLOv12 config, the loader falls back to `yolo11`/`yolov8` so the
> pipeline still runs; the FLEO neck is backbone-agnostic. State which backbone
> you actually used when reporting results.

---

## Quickstart

### A. Train + evaluate (Kaggle / any GPU)
```bash
# 1) install
pip install -r requirements.txt

# 2) train (FER2013 root contains train/ and test/)
python train_fer2013.py --data data/fer2013 --backbone yolov12s \
    --epochs 80 --lr 3e-4 --img-size 224 --device cuda

# 3) ablation -> Table 2 (baseline → +SE → +FLEO no-bind → full → +fuzzy)
python run_ablation.py --data data/fer2013 --epochs 40 --seeds 0 1 2

# 4) figures + per-class report
python evaluate.py --data data/fer2013 --ckpt checkpoints/best_fleo.pt
```
On Kaggle, open `notebooks/fleo_kaggle.ipynb`, set GPU + Internet, add the
`fer2013` dataset, and use **Save & Run All (Commit)** for uninterrupted runs.

### B. INT8 deployment simulation (no FPGA, no Vitis)
```bash
python deploy/simulate_int8.py --data data/fer2013 \
    --ckpt checkpoints/best_fleo.pt --route1
# prints params, GFLOPs, latency/FPS, and FP32 vs simulated-INT8 accuracy
```

### C. Official Vitis AI export (Linux + Docker, board optional)
```bash
docker run -it --rm -v $(pwd):/work xilinx/vitis-ai-pytorch-cpu:latest
cd /work && python deploy/run_vitis.py --data data/fer2013 \
    --ckpt checkpoints/best_fleo.pt
# then vai_c_xir → compiled/fleo.xmodel   (see deploy/VITIS.md)
```

---

## Repository layout

```
FLEO/
├── models/
│   ├── fleo_module.py      # FLEO block: Gram–Schmidt + SE gate + binding head
│   ├── fer_classifier.py   # backbone (YOLOv12/ResNet/tiny) + FLEO + MLP head, ablation modes
│   ├── loss.py             # BindingLoss + FuzzyConfusionLoss + FLEOTotalLoss
│   ├── yolov12_fleo.py     # YOLOv12 neck wrapper (FLEO at P3/P4)
│   └── __init__.py
├── utils/
│   ├── confusion_matrix.py # FACS prior C + running EMA update
│   └── metrics.py          # accuracy, macro-F1, per-class report, confusion matrix
├── train/
│   └── trainer.py          # training loop (fuzzy/CE toggle, multi-seed, best report)
├── train_fer2013.py        # FER2013 entry point (aug, balanced sampler, --mode)
├── run_ablation.py         # Table 2: 5 configs × N seeds, mean ± std
├── evaluate.py             # confusion matrix + t-SNE + per-class report
├── deploy/
│   ├── export_fpga.py      # Route-1 fold-out + Vitis quantizer + PYNQ template
│   ├── run_vitis.py        # end-to-end Vitis export + INT8 accuracy
│   ├── simulate_int8.py    # FPGA-free INT8 simulation
│   ├── fpga_design.md      # hardware design notes
│   └── VITIS.md            # Vitis AI / ZCU104 walkthrough
├── notebooks/fleo_kaggle.ipynb   # one-click Kaggle pipeline
├── reference/              # NumPy reference + full simulation (no torch needed)
├── tests/test_fleo.py      # unit tests (math, ablation modes, losses)
├── paper/                  # IEEEtran journal manuscript + references.bib
├── configs/fleo_config.yaml
└── requirements.txt
```

---

## Key math

| Formula | Role |
|---|---|
| `u_k = ṽ_k − Σ proj_{u_j}(ṽ_k)` | Gram–Schmidt step |
| `e_k = u_k / (‖u_k‖ + ε)` | Normalization |
| `⟨e_i, e_j⟩ = δ_{ij}` | Orthonormality guarantee |
| `‖e_i − e_j‖² = 2` | Fixed margin |
| `T̂_{i,k} = 1−α_i` if `k=y_i` else `α_i·C[y_i,k]/Σ` | Confusion-structured fuzzy target |
| `α_i = clip(α_0·H(P_i)/log K, α_min, α_max)` | Adaptive smoothing |
| `L = L_Fuzzy + λ·L_bind` | Total objective (no orthogonality penalty) |

---

## Results

All result tables in `paper/fleo_journal.tex` are placeholders to be filled
from your own runs (`run_ablation.py`, `evaluate.py`, `deploy/simulate_int8.py`).
We do not report unmeasured numbers.

---

## Background & citations

Built on YOLOv12 [Tian et al., 2025] and the squeeze-and-excitation gate
[Hu et al., 2018]; the label-ambiguity line (SCN, DMUE, RUL) and orthogonality
regularization [Bansal et al., 2018] motivate the design. Benchmarks: FER2013
[Goodfellow et al., 2013] and RAF-DB [Li et al., 2017]. Deployment targets the
AMD/Xilinx Vitis AI stack and the ZCU104 DPU. Full reference list in
`paper/references.bib`.

## License
Research code for academic use. Vendor tools (Vitis AI) follow their own
licenses.
