# How to Run FLEO

Two levels of code:
- **Level A** — math/results verification → needs only **NumPy** (runs anywhere, even here)
- **Level B** — real model training/testing → needs **PyTorch**
- **Level C** — FPGA deployment → needs **Vitis AI + ZCU104 board**

---

## Level A — Verify the math & see all results (needs only NumPy)

No GPU, no PyTorch. Works on any machine with Python 3.8+.

```bash
cd FLEO
pip install numpy

# 1) Proves Proposition 1 (||e_i - e_j||^2 = 2) + fuzzy targets
python reference/numpy_reference.py

# 2) Orthogonality check (auto-uses numpy if torch missing)
python scripts/verify_orthogonality.py

# 3) FULL simulation — all 14 result tables (training curve,
#    confusion matrix, per-emotion F1, FPGA performance, etc.)
python reference/full_simulation.py
```

---

## Level B — Train / test the real model (needs PyTorch)

### Step 1: Install dependencies
```bash
cd FLEO
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt    # torch, torchvision, numpy, sklearn, opencv, pyyaml
```

If `pip install torch` fails, use the official selector:
```bash
# CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Run the test suite (verifies everything works)
```bash
python tests/test_fleo.py
# expected: "All 8 tests passed."
```

### Step 3: Quick end-to-end demo (synthetic data, no dataset needed)
```bash
python -c "
import torch
from models import FLEOClassifier, FLEOTotalLoss
from utils.confusion_matrix import FACS_PRIOR_7, normalize_confusion

model = FLEOClassifier(num_emotions=7)
crit  = FLEOTotalLoss(normalize_confusion(FACS_PRIOR_7))
opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)

x = torch.randn(4, 3, 128, 128)      # fake batch of 4 images
y = torch.randint(0, 7, (4,))        # fake emotion labels

model.train()
for step in range(5):
    logits, z = model(x)
    loss = crit(logits, z, y)['total']
    opt.zero_grad(); loss.backward(); opt.step()
    print(f'step {step}  loss={loss.item():.4f}')
print('OK - training loop works')
"
```

### Step 4: Train on a real dataset (RAF-DB / AffectNet)
You need image folders + a `DataLoader`. Skeleton:
```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import FLEOClassifier
from train.trainer import FLEOTrainer

tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
# expects: data/rafdb/train/<emotion>/*.jpg
train_ds = datasets.ImageFolder("data/rafdb/train",   transform=tf)
val_ds   = datasets.ImageFolder("data/affectnet/val", transform=tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

model = FLEOClassifier(num_emotions=7)
cfg = {"epochs": 50, "lr": 1e-4, "num_emotions": 7}
FLEOTrainer(model, train_loader, val_loader, cfg).run()
```
Run it: `python train_rafdb.py` → checkpoints saved to `checkpoints/best_fleo.pt`.

### Step 4b: Train on FER2013 (ready-made script)
FER2013 (Kaggle `msambare/fer2013`) is 48×48 **grayscale**, 7 emotions, already
split into `train/` and `test/`. A ready-to-run script handles the grayscale→RGB
conversion and the label remap to the FACS confusion-prior order:

```bash
# Local: download/unzip FER2013 into ./data/fer2013 (so data/fer2013/train/... exists)
python train_fer2013.py --data data/fer2013 --epochs 50 --batch-size 64

# Kaggle: add the FER2013 dataset via "+ Add Input", then:
python train_fer2013.py --data /kaggle/input/fer2013 --epochs 50
```
If `--data` is omitted it auto-probes common paths (`data/fer2013`,
`/kaggle/input/fer2013`, …). Best checkpoint → `checkpoints/best_fleo.pt`.

Useful flags: `--lr`, `--img-size` (default 128), `--num-workers`, `--device`.

### Step 4c: Publication pipeline (ablation + figures)
The paper's decisive result is the **ablation** (Table 2) and the
**confusion-matrix / t-SNE** figures. Two scripts produce them:

```bash
# Table 2: baseline -> +SE -> +FLEO(no binding) -> +FLEO(full) -> +fuzzy,
# each over 3 seeds, reported as mean +/- std.
python run_ablation.py --data data/fer2013 --backbone yolov12s \
    --epochs 40 --seeds 0 1 2 --out results/ablation.json

# Figures from the best checkpoint: confusion matrix + t-SNE + per-class report.
python evaluate.py --data data/fer2013 --backbone yolov12s --mode fleo_full \
    --ckpt checkpoints/best_fleo.pt --out results
```

Ablation modes (also usable directly via `train_fer2013.py --mode ... [--no-fuzzy]`):

| `--mode` | orthogonalization | binding head | Table 2 row |
|---|---|---|---|
| `baseline` | – | – | YOLOv12-S baseline |
| `se` | – | – (SE gate only) | + SE neck |
| `fleo_nobind` | ✓ | – | + FLEO (no binding) |
| `fleo_full` | ✓ | ✓ | + FLEO (full) |
| `fleo_full` + fuzzy loss | ✓ | ✓ | + FLEO (full) + fuzzy |

### Step 5: Interactive notebook
```bash
pip install jupyter
jupyter notebook notebooks/01_fleo_demo.ipynb
```

---

## Level C — Deploy to FPGA (ZCU104 board required)

Needs: a ZCU104 board + the AMD Vitis AI Docker image. See `deploy/fpga_design.md` for the full hardware design.

### Step 1: Export (removes Gram-Schmidt — Route 1)
```python
from deploy.export_fpga import prepare_model_for_export
# loads checkpoint, sets train_only=True, eval mode
```

### Step 2: Quantize INT8 (inside Vitis AI Docker)
```bash
docker run -it --rm -v $(pwd):/work xilinx/vitis-ai-pytorch-cpu:latest
cd /work
python deploy/export_fpga.py        # runs quantizer, prints next command
```

### Step 3: Compile to .xmodel (inside Docker)
```bash
vai_c_xir \
  --xmodel  ./quantized/quantize_result/YOLOv12_FLEO_int.xir \
  --arch    /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  --output_dir ./compiled \
  --net_name   yolov12_fleo
```

### Step 4: Run on the board (PYNQ Jupyter on ZCU104)
Copy `compiled/yolov12_fleo.xmodel` to the board, then in Jupyter:
```python
from vitis_ai_library import FaceDetect
model = FaceDetect.create("yolov12_fleo")
# ... see PYNQ_INFERENCE_TEMPLATE in deploy/export_fpga.py for camera loop
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'torch'` | You're running Level B code without PyTorch → install it (Step 1) or use Level A scripts |
| `No module named 'models'` | Run from the `FLEO/` root dir, not from inside subfolders |
| `No module named 'sklearn'` | `pip install scikit-learn` (or metrics auto-fallback to torch) |
| `pip install torch` fails | Use the `--index-url` command in Level B Step 1 |
| CUDA out of memory | Lower `batch_size` in cfg, or use CPU wheel |
