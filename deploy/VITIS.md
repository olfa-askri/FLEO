# Vitis AI — INT8 simulation & ZCU104 deployment (Route 1)

> **Important:** Vitis AI does **not** run on Kaggle. It is an AMD/Xilinx
> toolchain that runs in a **Docker container on your own machine** (Linux,
> x86). Kaggle is only for training. You train on Kaggle, download
> `best_fleo.pt`, then do the Vitis steps locally.

## No FPGA and no Vitis? Simulate INT8 in software (Kaggle-friendly)
If you have neither the board nor a Vitis AI install, you can still produce the
decisive number — the **FP32 → INT8 accuracy drop** — with a pure-PyTorch
simulation that runs anywhere:
```bash
python deploy/simulate_int8.py --data data/fer2013 \
    --ckpt checkpoints/best_fleo.pt --backbone yolov12s --mode fleo_full --route1
```
It prints params, GFLOPs, latency/FPS (on your CPU/GPU), and FP32 vs simulated
INT8 accuracy. Report it as **"simulated INT8"** in the paper — it is a faithful
software proxy (per-channel 8-bit weight quantization), **not** the AMD DPU.
Real DPU latency/FPS and the official Vitis quantizer numbers still need the
steps below.

---

The "simulation" people mean is usually one of two things:
1. **INT8 quantization + accuracy simulation** (no board needed) — the FP32→INT8
   accuracy drop. This is the decisive deployment number in the paper.
2. **DPU inference** — needs the ZCU104 board (or the QEMU/VART emulator).

This guide covers (1) fully and gives the commands for (2).

---

## 0. Prerequisites (local machine)
- Linux x86 with Docker installed
- The trained checkpoint `checkpoints/best_fleo.pt` (downloaded from Kaggle)
- This repo on disk

## 1. Pull and enter the Vitis AI Docker
```bash
docker pull xilinx/vitis-ai-pytorch-cpu:latest
cd /path/to/FLEO
docker run -it --rm -v $(pwd):/work xilinx/vitis-ai-pytorch-cpu:latest
cd /work
```

## 2. INT8 quantization + accuracy simulation
Inside the container (it provides `pytorch_nndct`):
```bash
python deploy/run_vitis.py \
    --data data/fer2013 \
    --ckpt checkpoints/best_fleo.pt \
    --backbone yolov12s --mode fleo_full --img-size 128
```
This prints:
```
[vitis] FP32 (Route-1) accuracy: XX.XX%
[vitis] INT8 accuracy:           YY.YY%   (drop -Z.ZZ pts)
```
→ The FP32 vs INT8 numbers fill the **deployment experiment** (paper Sec. V-F).
The quantized artifacts land in `quantized/quantize_result/`.

## 3. Compile to .xmodel for the ZCU104 DPU
Still inside the container, run the `vai_c_xir` command that `run_vitis.py`
printed (DPU arch DPUCZDX8G/ZCU104):
```bash
vai_c_xir \
  --xmodel  quantized/quantize_result/FLEOClassifier_int.xmodel \
  --arch    /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  --output_dir compiled \
  --net_name  fleo
```
→ produces `compiled/fleo.xmodel`.

## 4. Run on the ZCU104 board (needs hardware)
Copy `compiled/fleo.xmodel` to the board (PYNQ image) and run inference with
the Vitis AI Library / VART. See `PYNQ_INFERENCE_TEMPLATE` in
`deploy/export_fpga.py` for a camera loop.

---

## Notes specific to FLEO (Route 1)
- **Route 1 fold-out:** `prepare_classifier_for_export(..., train_only=True)`
  drops the Gram-Schmidt recurrence (square-root + division + sequential
  dependency — none of which the DPU runs). Only the projection conv, SE gate,
  and 3×3 conv remain, all DPU-native.
- The accuracy you must report is **FP32-with-GS (training)** vs
  **FP32-Route1 (GS folded out)** vs **INT8-Route1**. The drop from folding out
  GS is exactly the hypothesis the paper says must be tested.
- If a YOLOv12 op (area-attention) is not DPU-supported, it falls back to the
  ARM PS; profile with `vai_c_xir`'s report and consider the Householder/Cayley
  alternative (paper Sec. IV) if needed.
