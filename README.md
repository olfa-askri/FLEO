# FLEO – Fuzzy Label and Emotion Orthogonalization for YOLOv12

## Project Structure

```
FLEO/
├── models/
│   ├── fleo_module.py      # Core FLEO block (GS orthogonalizer + SE gate + binding head)
│   ├── loss.py             # BindingLoss + FuzzyConfusionLoss + FLEOTotalLoss
│   ├── yolov12_fleo.py     # YOLOv12 Neck wrapper with FLEO at P3 and P4
│   └── __init__.py
├── utils/
│   ├── confusion_matrix.py # FACS-based prior C + running update
│   └── metrics.py          # Accuracy, macro-F1, orthogonality check
├── train/
│   └── trainer.py          # Full training loop (RAF-DB → AffectNet eval)
├── deploy/
│   └── export_fpga.py      # Route 1 export → Vitis AI quantization → .xmodel
├── configs/
│   └── fleo_config.yaml    # All hyperparameters
├── notebooks/
│   └── 01_fleo_demo.ipynb  # Interactive forward pass + verification
├── scripts/
│   └── verify_orthogonality.py
└── requirements.txt
```

## Key Math

| Formula | Role |
|---|---|
| `u_k = ṽ_k − Σ proj_{u_j}(ṽ_k)` | Gram-Schmidt step |
| `e_k = u_k / (‖u_k‖ + ε)` | Normalization |
| `⟨e_i, e_j⟩ = δ_{ij}` | Orthonormality guarantee |
| `‖e_i − e_j‖² = 2` | Fixed margin (Proposition 1) |
| `T̂_{i,k} = 1−α_i` if k=y_i else `α_i · C[y_i,k]` | Fuzzy target |

## FPGA Deployment (Route 1 – Python only)

```
Training (PyTorch + FLEO full)
    ↓
export_inference_graph()   # removes Gram-Schmidt ops
    ↓
Vitis AI Quantizer (INT8)  # pytorch_nndct inside Docker
    ↓
vai_c_xir  →  .xmodel      # compiled for DPU B4096 / ZCU104
    ↓
PYNQ Jupyter on ZCU104     # vitis-ai-library Python API
```
