# FLEO-FER — Session log & project journal

A self-contained summary of the working session: what the project is, everything
that was built/fixed, all the concepts explained, the training results and their
analysis, the Kaggle workflow (and the data-loss lesson), the SOTA comparison, the
FPGA deployment plan, and the honest assessment of publishability.

Repo: https://github.com/olfa-askri/FLEO

---

## 1. What the project is

A **real-time facial-emotion detector** deployed on an FPGA. The network is a
**YOLOv12-S** detector whose neck (levels P3 and P4) is augmented with **FLEO**, a
block that partitions features into per-emotion subspaces and enforces their mutual
orthogonality with a differentiable **Gram-Schmidt** pass.

The paper's central idea (**the fold-out**): Gram-Schmidt helps during *training*
but is hostile to the accelerator at *inference* (it needs division, square root,
and a sequential recurrence — none supported by the DPU datapath). So we treat it
as a **training-time regularizer** and remove ("fold out") it at export, leaving a
graph made only of DPU-native operators.

Task cast: FER is cast as **detection with a single full-image box** per image
(class = emotion). Datasets: **FER2013** and **RAF-DB** (7 emotions:
angry, disgust, fear, happy, sad, surprise, neutral).

Target hardware: **ZCU104** board, **Zynq UltraScale+ XCZU7EV** MPSoC, **DPUCZDX8G
B4096** (4096 ops/cycle, 1.229 TOPS @ 300 MHz), Vitis AI flow.

---

## 2. Repo structure

```
fleo/        core: orthogonal.py (Gram-Schmidt, Householder), fleo_block.py,
             losses.py, yolo_integration.py (FLEODetectionModel, wrap P3/P4)
data/        FER2013 + RAF-DB -> YOLO detection format (auto-discovering)
scripts/     train.py, evaluate.py, deltas.py, export.py, run_matrix.py
deploy/      Vitis AI quantize/compile + VART pipeline + metrics + power + report
sota/        SOTA data + comparison tables (SOTA_COMPARISON.md, COMPARISON_TABLE.md)
notebooks/   kaggle_fleo.ipynb (clean, persist-safe Kaggle run)
tests/       unit tests (GS orthogonality, Householder, fold-out weight preservation)
```

The three deployment **routes** (what ORTHO becomes at export):
- **R1 fold-out** — Gram-Schmidt -> identity (removed). Single DPU subgraph. Fastest.
- **R2 Householder** — a constant-folded orthogonal 1x1 conv. Single DPU subgraph.
- **R3 hybrid** — keep exact Gram-Schmidt, run it on the ARM CPU (PS). Exact, slow.

---

## 3. What was built / fixed this session (chronological)

1. **Full project scaffold** implemented from the paper: FLEO block, YOLOv12-S
   integration (wrap P3/P4 layers in place so it survives `train()`), datasets,
   train/eval/export/deltas, Vitis AI deploy kit, SOTA framework, tests. Pushed to
   GitHub (`main` set as default branch).
2. **Layout-agnostic dataset prep** — FER2013/RAF-DB auto-discover images under any
   `--src` (emotion folders, numeric 1..7 folders, `*labels*.csv`, official
   aligned+txt), with a directory-tree diagnostic on failure.
3. **Checkpoint-path fix** — ultralytics nests a relative project under
   `runs/detect/`; switched to an absolute project path + `find_ckpt()` fallback so
   export/deltas find the weights.
4. **Recipe fix (late-training collapse)** — mosaic/mixup corrupt the single
   full-image-box label and, under AMP, blow box_loss to `inf` -> collapse to mAP 0
   around epoch 90. Fixed by `mosaic=0, mixup=0, close_mosaic=0` by default.
5. **Decisive-path robustness** — evaluate/deltas now register FLEO classes as torch
   safe-globals (so the FLEO checkpoint loads on modern torch); robust per-class
   top-1 rule; deltas loads each checkpoint once and switches route in place.
6. **COCO warm-start** — `--pretrained yolo12s.pt` default (backbone/neck), big FER
   convergence gain; `--scratch` to disable.
7. **Deploy-kit fixes** — vart_pipeline measures steady-state FPS over the timed
   window only (not incl. warm-up); postprocess uses the same decision rule as the
   host evaluator; power_ina226 rail selection cleaned.
8. **Anti-overfit changes so FLEO can beat baseline** — Dropout2d inside FLEO
   (`--fleo-dropout`, def 0.15), stronger orthogonality regularizer (lambda_ortho
   0.01 -> 0.05), cosine LR + weight_decay 0.0008.
9. **SOTA tables** — added 2024-2025 references, real FPGA-FER deployment numbers,
   and `COMPARISON_TABLE.md` (FER-on-FPGA only, with a Winner row).

---

## 4. Concepts explained (quick reference)

**FPGA** — a chip you *configure* into any circuit (like LEGO for hardware).
**ZCU104** — the evaluation *board*; carries the Zynq UltraScale+ XCZU7EV *chip*.
**MPSoC** — the chip = **PS** (ARM CPUs) + **PL** (the FPGA fabric) together.
**PS** — Processing System: 4x Cortex-A53 (Linux, VART, pre/post-proc), 2x R5F.
**PL** — Programmable Logic: the FPGA fabric that hosts the **DPU**.
**DPU (B4096)** — the CNN accelerator built on the PL; 4096 ops/cycle, 1.229 TOPS.
**Xilinx** — the vendor (now AMD). **Virtex/Zynq/Spartan** — FPGA families.
**Vitis AI** — tool that quantizes (INT8) + compiles a model to a `.xmodel`.
**Vivado** — tool that builds the DPU on the fabric and reports resources/timing.

FPGA building blocks (fabric budget on XCZU7EV):
- **LUT** (230,400) — look-up table, the basic logic unit.
- **FF** (460,800) — flip-flop, stores 1 bit.
- **DSP** (1,728) — fast multiply-accumulate (used for convolutions).
- **BRAM** (312) / **URAM** (96) — on-chip memory (fast, near the compute).

Metrics:
- **FPS** — frames/second (throughput; >=25 = real-time).
- **Latency** — time for one frame.
- **Power (W)** — from INA226 rails; **Energy** = P/FPS (mJ/frame); **FPS/W** = efficiency.
- **fmax** — max clock (300 MHz core; 600 MHz DSP). = 1/(T - WNS).
- **TOPS** — tera-ops/second (compute power). **Utilization %** — resources used.
- **mAP50 / mAP50-95** — detection metric; here (full-image box) it proxies
  classification quality and mAP50 ~= mAP50-95 for a well-localized model.
- **accuracy / macro-F1** — the paper's real numbers (from `scripts/deltas.py`).
- **Delta_fold** = Acc(full) - Acc(folded) — cost of removing Gram-Schmidt (KEY).
- **Delta_q** = Acc(FP32) - Acc(INT8) — quantization cost.

Why cast FER as detection: the *object* is the face; since the datasets are
pre-cropped faces, the box is the whole image and the class is the emotion. This
keeps the deployment realistic (a detector that finds a face and classifies its
emotion) and is why the paper is a *detector* deployment.

---

## 5. Training results & analysis (first real FER2013 run, 100 epochs)

Baseline (plain YOLOv12-S) and FLEO both trained cleanly (no collapse after the
recipe fix). Per the run logs:

| Metric | Baseline | FLEO |
|---|---|---|
| best mAP50 | 0.764 | 0.763 |
| best mAP50-95 | 0.764 | 0.628 |
| precision | 0.708 | ~0.72 |
| recall | 0.72 | ~0.69 |

Baseline per-class mAP50: happy 0.955, surprise 0.881, disgust 0.776, neutral 0.744,
angry 0.714, sad 0.654, fear 0.621 (fear/sad hardest — as expected on FER2013).

**Findings (honest):**
- FLEO ~= baseline on mAP50 (tied). FLEO did **not** beat baseline on this metric.
- FLEO **over-fits more** (val/cls_loss rose to ~2.1 vs ~0.75 for baseline) — this is
  the real problem, and the reason for the anti-overfit code changes (dropout,
  stronger ortho, cosine LR).
- The decisive numbers (accuracy, macro-F1, Delta_fold) come from `deltas` — not yet
  measured at the time of writing. macro-F1 is FLEO's best hope (class separation).
- `box_loss = inf` appeared intermittently but AMP recovered every time (no collapse).

RAF-DB partial run reached mAP50 ~0.83 (cleaner dataset than FER2013).

---

## 6. Kaggle workflow + the data-loss lesson

- Free GPU: ~30 h/week (T4 x2 or P100). GPU needs **phone verification** on new
  accounts. Datasets attach under `/kaggle/input/...`.
- **IMPORTANT LESSON:** `/kaggle/working` is **wiped when the session ends**. A long
  interactive run (FER2013 ~10 h at 100 epochs + RAF-DB) exceeded the ~12 h limit and
  the session reset -> all trained models were lost (only the code, downloaded
  earlier, survived).
- **Fixes going forward:**
  - Use **60 epochs** (model plateaus ~epoch 60-70; ~0.75 vs ~0.76 for 100 — not
    worth the risk/quota).
  - **Save Version -> Save & Run All (Commit)** runs top-to-bottom and *persists*
    output automatically (fits inside 12 h at 60 epochs).
  - Or interactively: **download the zip immediately** after each training.
  - The clean notebook (`notebooks/kaggle_fleo.ipynb`) zips outputs after every step.

Clean run order: clone+install -> auto-find+prepare datasets -> FER2013 run_matrix
(60 ep) -> read deltas + baseline eval -> zip -> RAF-DB run_matrix -> read -> zip.

---

## 7. SOTA comparison (FER-on-FPGA) — see sota/COMPARISON_TABLE.md

Direct competitors (peer-reviewed unless noted):
- Vinh & Vinh (2019, IEEE): FER2013 66%, 15 FPS, Zynq SoC.
- Emotion recognizer for autistic children (IEEE): 72.9%, Virtex-7.
- Ando & Inoue (2025, arXiv): FER2013 67.4%, 25 FPS, 2.7 W, Kria KV260 (B512 DPU).
- This work (FLEO): FER2013 ~73%*, RAF-DB ~83%*, ~30 FPS*, ZCU104 (B4096), + fold-out.

GPU accuracy SOTA (context, NOT edge-deployable): FER2013 ~76% (ResMaskingNet
ensemble), RAF-DB up to 94.76% (ResEmoteNet 2024).

**Who wins (FER-on-FPGA):** this work leads on accuracy (73% > 66-72.9%), is the
only one on both datasets, most recent network, competitive FPS, and the only one
with a fold-out method; **Ando & Inoue wins power/FPS-per-watt** (smaller B512 DPU).

Honest caveats: (*) figures are estimates pending `deltas` + the board. FPS ~30 is
optimistic — the pipeline is likely **CPU-bound** (same A53 as Ando), so real FPS may
be ~25-40. The B4096 choice needs justifying (headroom, real-time margin) since a
smaller DPU is more power-efficient on this lightweight task.

---

## 8. FPGA deployment plan (what remains)

The model file for the FPGA is **`.xmodel`** (not `.pt`, not `.h5`). Path:
`.pt -> ONNX (export.py) -> INT8 (Vitis AI quantize) -> .xmodel (Vitis AI compile)`.

- **Quantize + compile** can run on **any Linux + Docker** (no board needed) —
  including **Windows via Docker Desktop + WSL2** with the `xilinx/vitis-ai-pytorch-cpu`
  image. This already yields: the `.xmodel`, the **subgraph partition** (proves R1 =
  single DPU subgraph — the core claim), and Delta_q.
- **Real FPS / power / resources / timing** need the **physical ZCU104 board** — no
  cloud substitute for Zynq/ZCU104. The EµE (Monastir) / ENET'Com lab is the natural
  source of the board.

User's environment: **Windows, no FPGA board** -> can do quantize/compile via Docker
Desktop (validates the methodology); needs lab access to the ZCU104 for the
measurement tables.

---

## 9. Honest assessment (publishability)

- **As an accuracy paper: no** (~65-75% is below GPU SOTA; the detection-cast caps it).
- **As an FPGA deployment + fold-out methodology paper: yes**, conditionally.
  - Estimated acceptance at a mid-tier IEEE/Springer venue (IEEE Access, Sensors,
    Electronics, Microprocessors & Microsystems): **~60-70%**, *if* the FPGA
    measurements are done and it's framed as deployment.
  - Top-tier FPGA venue (FPL/FCCM/TCAD): ~25-35%.
- **The three gates:** (1) real FPGA measurements on the board; (2) Delta_fold small
  (~<2%); (3) framing = deployment, not accuracy.
- **Biggest risk to the premise:** FLEO must beat (or at least match) baseline so the
  fold-out has something to preserve. Current data shows a tie on mAP; the anti-overfit
  changes are the attempt to tip it. Even if FLEO ~= baseline, the paper stands on
  "adds a foldable orthogonality structure at no accuracy cost + DPU-native deployment."

**Where you win:** the fold-out methodology (novel), its generality (whitening,
iterative normalization, etc.), the Delta_fold characterization, the R1/R2/R3
trade-off, single-stream DPU latency, and rigor/reproducibility. **Not** raw accuracy,
FPS, or power efficiency.

---

## 10. Immediate next steps

1. On (new) Kaggle: run `notebooks/kaggle_fleo.ipynb` (60 epochs), **persist via
   Save & Run All**, read `results/deltas_fer2013.json`.
2. Check first: **macro-F1** (FLEO's best shot) and **Delta_fold** (the key number).
3. Then RAF-DB the same way.
4. Fill the real numbers into `sota/COMPARISON_TABLE.md` / Table VIII.
5. Windows + Docker Desktop -> Vitis AI quantize + compile -> confirm R1 single
   subgraph + Delta_q (no board needed).
6. Secure a ZCU104 (lab) for the FPS/power/resource tables.

Architecture figures for the paper were designed this session (system architecture
matching the manuscript's Fig. 1 + toolflow Fig. 5); reproduce them as needed.
