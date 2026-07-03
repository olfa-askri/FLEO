# FLEO-FER: DPU-native FPGA deployment of an orthogonalization-regularized facial-emotion detector

Reference implementation of the methodology in *"Folding Structure Out of the
Datapath: A DPU-Native FPGA Deployment Methodology for a Real-Time
Orthogonalization-Regularized Facial-Emotion Detector on the Zynq UltraScale+
MPSoC"* (Askri, Massoud, Hajjaji).

A **YOLOv12-S** facial-emotion detector whose P3/P4 neck embeds **FLEO** — a block
that partitions features into per-emotion subspaces and enforces their mutual
orthogonality with a differentiable **Gram–Schmidt** pass. Gram–Schmidt is useful
at training time and hostile to the **DPUCZDX8G B4096** accelerator at inference
(division, reciprocal-sqrt, sequential recurrence). This repo implements the three
deployment routes that resolve that collision, plus the complete metric framework
for the **ZCU104 (XCZU7EV)**.

| Route | Idea | Inference graph | DPU partition |
|---|---|---|---|
| **R1 — fold-out** | treat GS as a training-time regularizer and delete it at export (Eq. 5) | conv/pool/gate only | **single DPU subgraph** |
| **R2 — Householder** | re-parameterize GS as a product of reflections; constant-fold `Q` into a 1×1 conv (Eq. 7) | + one dense 1×1 per site | **single DPU subgraph** |
| **R3 — hybrid PS/PL** | keep exact GS on the ARM cores between two DPU subgraphs | full graph | DPU–CPU–DPU (price of exactness) |

The decisive quantity is **Δfold = Acc(full) − Acc(folded)** (Eq. 6): does training
under the constraint bake its benefit into the weights so removal is nearly free?

---

## Install (host: training / export)

```bash
pip install -r requirements.txt        # CPU
# for GPU, install the matching torch build from pytorch.org first
```

Quantization and on-target execution use **Vitis AI 3.5** (`vai_q_pytorch`,
`vai_c_xir`) and the **ZCU104 PetaLinux** runtime (`vart`, `xir`) — not pip
packages; see *FPGA deployment* below.

## Repository layout

```
fleo/                     FLEO block + orthogonalization + YOLOv12 integration
  orthogonal.py           gram_schmidt (GS) and HouseholderOrtho (+ export_conv1x1)
  fleo_block.py           FLEO neck block; modes full|folded|householder; fold_out()
  losses.py               orthogonality penalty + binding loss (make GS a regularizer)
  yolo_integration.py     wrap P3/P4 necks; FLEODetectionModel; FLEO trainer
data/                     FER2013 / RAF-DB -> YOLO detection (7 classes, full-image box)
scripts/                  train / evaluate / export / deltas / run_matrix
deploy/                   Vitis AI kit: quantize, compile, VART pipeline, metrics, report
  constants.py            Table I + B4096 geometry (P_peak=1.229 TOPS, beta=19.2 GB/s)
  metrics.py              Section V formulas (Eqs. 1,3,9-20) as executable functions
  app/                    on-target 3-thread VART pipeline + PS-side GS (R3)
  measure/                Vivado report parsers, INA226 power, workload counter
sota/                     literature SOTA tables + comparison report generator
tests/                    unit tests (orthogonality, fold-out invariants, pickling)
```

Manuscript → code map: Sec. IV-B fold-out → `FLEO.fold_out`; IV-C Householder →
`HouseholderOrtho.export_conv1x1`; IV-D hybrid → `deploy/app/gs_ps.py`; Sec. V
metrics → `deploy/metrics.py`; Tables IV–VIII → `deploy/report.py`.

## 1. Datasets

```bash
# FER2013 (CSV or image-folder form both supported)
python -m data.prepare_fer2013 --src /path/to/fer2013.csv --out datasets/fer2013

# RAF-DB (aligned images + EmoLabel/list_patition_label.txt)
python -m data.prepare_rafdb --src /path/to/RAF-DB --out datasets/rafdb

# tiny synthetic set for a no-download smoke test
python -m data.make_synthetic --out datasets/synthetic --n 24 --imgsz 96
```

Both datasets are remapped to one canonical 7-emotion order
(angry, disgust, fear, happy, sad, surprise, neutral) so a single `nc=7` head
serves both (see `data/emotions.py`).

## 2. Train (full graph = GS + auxiliary regularizers)

```bash
# baseline YOLOv12-S (no FLEO)
python -m scripts.train --data datasets/fer2013/data.yaml --variant baseline --seeds 0 1 2

# FLEO full graph, 3 seeds
python -m scripts.train --data datasets/fer2013/data.yaml --variant fleo \
    --epochs 100 --imgsz 640 --batch 16 --device 0 --seeds 0 1 2
```

FLEO training uses a custom trainer so the P3/P4 wrapping survives ultralytics'
model rebuild; the orthogonality + binding losses live on `FLEODetectionModel`
(no unpicklable hooks/closures — checkpoints save cleanly).

## 3. Export a deployment route (rewrite before quantization)

```bash
python -m scripts.export --weights runs/fleo/fleo_seed0/weights/best.pt --route r1 --imgsz 640 --verify
python -m scripts.export --weights ... --route r2 --imgsz 640
python -m scripts.export --weights ... --route r3 --imgsz 640
```

Each writes `export/<route>.onnx` (+ `.pt`, `.meta.json`) and checks the neck for
Gram–Schmidt-signature ops (`ReduceL2`/`Sqrt`): **absent for R1/R2** (single DPU
subgraph), **present for R3** (DPU–CPU–DPU). `--verify` confirms onnxruntime↔torch
parity (~1e-4).

## 4. Evaluate + the decisive deltas

```bash
# top-1 detection-cast accuracy / macro-F1 at a given route
python -m scripts.evaluate --weights runs/fleo/fleo_seed0/weights/best.pt \
    --data datasets/fer2013/data.yaml --route full --imgsz 640

# Delta_fold (and Delta_q if INT8 ONNX given), mean +/- s.d. over seeds -> Table VIII
python -m scripts.deltas --data datasets/fer2013/data.yaml --dataset fer2013 --imgsz 640 \
    --weights runs/fleo/fleo_seed{0,1,2}/weights/best.pt --out results/deltas_fer2013.json
```

Or run the whole host matrix (train → export → workload → deltas):

```bash
python -m scripts.run_matrix --data datasets/fer2013/data.yaml --dataset fer2013 \
    --seeds 0 1 2 --epochs 100 --imgsz 640 --device 0
```

## 5. FPGA deployment (Vitis AI 3.5 + ZCU104)

**Quantize** (inside `xilinx/vitis-ai-pytorch-cpu` docker; power-of-two PTQ, QAT
fallback):

```bash
python deploy/quantize_vai.py --weights export/r1.pt --route r1 \
    --data datasets/fer2013/data.yaml --imgsz 640 --mode calib --out quantized/r1
python deploy/quantize_vai.py --weights export/r1.pt --route r1 \
    --data datasets/fer2013/data.yaml --imgsz 640 --mode test  --out quantized/r1
```

**Compile** to the B4096 instruction stream and inspect the partition:

```bash
bash deploy/compile_b4096.sh r1 quantized/r1 compiled/r1
xdputil xmodel compiled/r1/fleo_yolov12s_r1.xmodel -l   # R1/R2: 1 DPU subgraph
```

**Run** the 3-thread pipeline on the board (100-frame warm-up, 1000 timed frames)
and sample rail power at 10 Hz:

```bash
python3 deploy/app/vart_pipeline.py --xmodel compiled/r1/fleo_yolov12s_r1.xmodel \
    --images /run/media/frames --imgsz 640 --route r1 --out results/latency_r1.json
python3 deploy/measure/power_ina226.py --hz 10 --seconds 120 --tag load --out results/power_r1.json
```

**Assemble** the results tables (IV–VIII) — every derived number via
`deploy/metrics.py`, unmeasured cells rendered as pending dashes:

```bash
python -m deploy.measure.parse_vivado --util post_impl_util.rpt --timing timing.rpt --out results/impl_r1.json
python -m deploy.report --results-dir results --routes r1 r2 r3 --out results/RESULTS.md
python -m sota.compare --results-dir results --out sota/SOTA_COMPARISON.md
```

## 6. Tests

```bash
pytest -q          # fast unit tests
pytest -q -m slow  # + full-detector build, route switch, checkpoint pickle roundtrip
```

## Notes / honesty

- Accuracy vs published FER methods is **contextual**: those are GPU classifiers;
  this is a detection-cast detector at INT8 on an FPGA overlay. The load-bearing
  comparison is **Δfold**, read from one table.
- Board power from INA226 includes infrastructure → a conservative upper bound;
  idle and load are reported separately to expose the dynamic component.
- All device/accelerator constants are vendor-documented (`deploy/constants.py`);
  results templates ship complete and empty until real runs populate them.
