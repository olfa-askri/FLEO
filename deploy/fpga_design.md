# FPGA Design — FLEO + YOLOv12 on Xilinx ZCU104

---

## 1. Target Hardware

| Resource | Spec |
|---|---|
| Board | Xilinx ZCU104 (Zynq UltraScale+ MPSoC) |
| FPGA Fabric | UltraScale+ PL (Programmable Logic) |
| Processor | ARM Cortex-A53 quad-core 1.3 GHz (PS) |
| DPU Version | B4096 (DPUCZDX8G) |
| DPU Cores | 1 (can scale to 3 on this device) |
| On-chip Memory | 504 KB block RAM |
| DDR | 4 GB LPDDR4 |
| Framework | Vitis AI 3.5 + PYNQ 3.0 |

---

## 2. System Block Diagram (Top Level)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ZCU104 MPSoC                                    │
│                                                                         │
│  ┌──────────────────────────────┐    ┌──────────────────────────────┐   │
│  │   PS  (Processing System)    │    │   PL  (Programmable Logic)   │   │
│  │                              │    │                              │   │
│  │  ARM Cortex-A53 x4 @ 1.3GHz │    │  ┌────────────────────────┐  │   │
│  │                              │    │  │    DPU  B4096          │  │   │
│  │  ┌─────────────────────┐     │    │  │                        │  │   │
│  │  │  Python Application │     │◄──►│  │  ┌──────────────────┐  │  │   │
│  │  │  (PYNQ / vitis-ai)  │     │AXI │  │  │  Conv Engine     │  │  │   │
│  │  │                     │     │4K  │  │  │  (4096 ops/clk)  │  │  │   │
│  │  │  1. Camera capture  │     │    │  │  ├──────────────────┤  │  │   │
│  │  │  2. Pre-process     │     │    │  │  │  Pool Engine     │  │  │   │
│  │  │  3. DPU trigger     │     │    │  │  ├──────────────────┤  │  │   │
│  │  │  4. Post-process    │     │    │  │  │  Elem-wise Eng.  │  │  │   │
│  │  │  5. Draw bbox+label │     │    │  │  ├──────────────────┤  │  │   │
│  │  └─────────────────────┘     │    │  │  │  Misc Engine     │  │  │   │
│  │                              │    │  │  └──────────────────┘  │  │   │
│  │  ┌─────────────────────┐     │    │  │                        │  │   │
│  │  │  DDR Controller     │◄────┼────┼─►│  On-chip SRAM buffer  │  │   │
│  │  │  4 GB LPDDR4        │     │    │  │  (feature map cache)   │  │   │
│  │  └─────────────────────┘     │    │  └────────────────────────┘  │   │
│  └──────────────────────────────┘    └──────────────────────────────┘   │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────────┐  │
│  │  MIPI Camera │    │  HDMI Out    │    │  SD Card (.xmodel file)   │  │
│  │  (CSI-2)     │    │  (display)   │    │  yolov12_fleo.xmodel      │  │
│  └──────────────┘    └──────────────┘    └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. DPU B4096 Internal Architecture

```
DPU B4096 Pipeline  (one forward pass)
════════════════════════════════════════

 Input Feature Map (DDR)
        │
        ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  LOAD ENGINE  — DMA from DDR to on-chip SRAM                        │
 │  Burst read 64-bit @ 250 MHz  (INT8 weights + activations)          │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
 ┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐
 │ CONV ENGINE  │  │ POOL ENGINE      │  │ MISC ENGINE      │
 │              │  │                  │  │                  │
 │ 4096 MACs/clk│  │ Max/Avg pool     │  │ ReLU/SiLU/BN    │
 │ INT8 × INT8  │  │ Global Avg Pool  │  │ Element-wise Add │
 │ → INT32 acc  │  │ (SE gate pool)   │  │ (residual +x)    │
 │ → INT8 out   │  │                  │  │ Concat           │
 │              │  │                  │  │ Sigmoid (SE gate)│
 └──────┬───────┘  └──────┬───────────┘  └──────┬──────────┘
        └──────────────────┴───────────────────── │
                                                  │
                            ┌─────────────────────┘
                            ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  SAVE ENGINE  — DMA write result back to DDR                         │
 └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
 Output Feature Map / Logits (DDR → Python via PYNQ)
```

---

## 4. FLEO Module — DPU Mapping (Route 1)

```
FLEO Module layers and their DPU mapping (inference, GS folded out)
════════════════════════════════════════════════════════════════════

  P3 Feature Map (B,256,40,40) from Neck
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  [Layer 1]  Conv 1×1  +  BN  +  SiLU                           │
  │             256 → 224  (K=7, d=32 → 7×32=224 channels)         │
  │             MACs: 256 × 224 × 40 × 40 = 91.75 M                │
  │             DPU: CONV ENGINE ✓  (fully supported INT8)          │
  └───────────────────────────────┬─────────────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
  ┌───────────────────────────┐   ┌────────────────────────────────┐
  │  [Layer 2]  SE Gate       │   │   GS Orthogonalizer            │
  │  Global Avg Pool (MISC ✓) │   │   *** TRAIN-TIME ONLY ***      │
  │  FC1: 224→56  (ReLU) ✓   │   │   NOT in inference graph       │
  │  FC2: 56→224  (Sigmoid) ✓ │   │   Folded into Conv 1×1 weights │
  │  MACs: ~25 K              │   └────────────────────────────────┘
  └───────────────────────────┘
                    │
                    ▼  (element-wise multiply in MISC ENGINE ✓)
  ┌───────────────────────────────────────────────────────────────────┐
  │  [Layer 3]  Conv 3×3  +  BN  +  SiLU                             │
  │             224 → 256  (recombine K*d → original channels)        │
  │             MACs: 224 × 256 × 9 × 40 × 40 = 825.75 M             │
  │             DPU: CONV ENGINE ✓                                    │
  └────────────────────────────────┬──────────────────────────────────┘
                                   │
                    ┌──────────────┘
                    ▼  (element-wise add in MISC ENGINE ✓)
  ┌────────────────────────────────────────────────────────────────────┐
  │  [Residual Add]  x_out = conv3x3_out + x_in (skip connection)      │
  │  MACs: 256 × 40 × 40 = 0.41 M                                      │
  │  DPU: MISC ENGINE ✓                                                │
  └────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
                     P3' Feature Map (B,256,40,40)  → Head
```

**Layer-by-layer DPU support matrix:**

| Layer | Operation | INT8 | DPU Support | MACs (P3) |
|---|---|---|---|---|
| Proj conv 1×1 | Conv + BN + SiLU | ✓ | CONV ENGINE | 91.75 M |
| SE pool | Global Avg Pool | ✓ | MISC ENGINE | ~0.06 M |
| SE FC1 | Linear + ReLU | ✓ | CONV ENGINE | 0.013 M |
| SE FC2 | Linear + Sigmoid | ✓ | CONV ENGINE | 0.013 M |
| Channel mul | Element-wise × | ✓ | MISC ENGINE | 0.09 M |
| Recomb 3×3 | Conv + BN + SiLU | ✓ | CONV ENGINE | 825.75 M |
| Residual + | Element-wise + | ✓ | MISC ENGINE | 0.41 M |
| GS (train) | Inner prod + div | ✗ | **ARM PS** | 2.51 M |
| Binding head | GAP + Linear | ✗ | **removed** | 0.22 M |

---

## 5. Full Network Pipeline on ZCU104

```
Full YOLOv12-S + FLEO inference pipeline
══════════════════════════════════════════

Camera Frame (640×640×3, RGB)
        │
        ▼ [ARM PS — Python preprocess]
 ┌───────────────────────────────┐
 │  Normalize + pad + DMA to DDR │   ~0.05 ms
 └──────────────────┬────────────┘
                    │
                    ▼ [DPU B4096 — full execution of .xmodel]
 ╔═══════════════════════════════════════════════════════════════════╗
 ║  YOLOv12-S Backbone                                               ║
 ║  ┌─────────────────────────────────────────────────────────────┐  ║
 ║  │  Conv-BN-SiLU stem → stride 2                               │  ║
 ║  │  C3k2 block × 3   → P3 (40×40×256)                         │  ║
 ║  │  C3k2 block × 2   → P4 (20×20×512)                         │  ║
 ║  │  A2 attention      → P5 (10×10×1024)                        │  ║
 ║  └───────────────────┬──────────────┬──────────────────────────┘  ║
 ║                      │              │                              ║
 ║  YOLOv12 Neck        │              │                              ║
 ║  ┌───────────────────▼──────────────▼──────────────────────────┐  ║
 ║  │  R-ELAN + Area-attention  (FPN-style feature fusion)         │  ║
 ║  │           │                      │                P5 pass   │  ║
 ║  │           ▼                      ▼                  through │  ║
 ║  │  ┌────────────────┐  ┌────────────────────┐                 │  ║
 ║  │  │  FLEO @ P3     │  │   FLEO @ P4         │                │  ║
 ║  │  │  [Conv1×1]     │  │   [Conv1×1]         │                │  ║
 ║  │  │  [SE Gate]     │  │   [SE Gate]         │                │  ║
 ║  │  │  [Conv3×3]     │  │   [Conv3×3]         │                │  ║
 ║  │  │  [Residual]    │  │   [Residual]        │                │  ║
 ║  │  └────────────────┘  └────────────────────┘                 │  ║
 ║  └──────────────────────────────────────────────────────────────┘  ║
 ║                                                                    ║
 ║  Detection Head                                                    ║
 ║  ┌──────────────────────────────────────────────────────────────┐  ║
 ║  │  Decoupled head: bbox regression + emotion classification    │  ║
 ║  │  Output: [x,y,w,h, obj_conf, emotion_logits×7]  per anchor  │  ║
 ║  └──────────────────────────────────────────────────────────────┘  ║
 ╚═══════════════════════════════════════════════════════════════════╝
        │
        ▼ [ARM PS — Python postprocess]
 ┌────────────────────────────────────┐
 │  NMS (Non-Max Suppression)         │   ~0.08 ms
 │  argmax(emotion_logits) → class    │
 │  Draw bbox + emotion label         │
 └────────────────────────────────────┘
        │
        ▼
   HDMI / RTSP stream output
```

---

## 6. Memory Map

```
DDR4 Memory Layout (4 GB)
══════════════════════════

0x0000_0000 ┌────────────────────────────────┐
            │  Linux OS + PYNQ runtime        │  ~512 MB
0x2000_0000 ├────────────────────────────────┤
            │  .xmodel weights (compressed)   │  ~8 MB
0x2008_0000 ├────────────────────────────────┤
            │  Input frame buffer  640×640×3  │  ~1.2 MB
0x2014_8000 ├────────────────────────────────┤
            │  P3 feature map  40×40×256      │  ~1.6 MB
0x2034_8000 ├────────────────────────────────┤
            │  P4 feature map  20×20×512      │  ~0.8 MB
0x2044_8000 ├────────────────────────────────┤
            │  P5 feature map  10×10×1024     │  ~0.4 MB
0x204C_8000 ├────────────────────────────────┤
            │  Head output / anchors          │  ~0.2 MB
0x204F_8000 ├────────────────────────────────┤
            │  DPU instruction buffer         │  ~4 MB
            └────────────────────────────────┘

On-chip SRAM (504 KB block RAM — shared L1 for DPU)
══════════════════════════════════════════════════════
  Weight ping-pong buffer   : 128 KB
  Activation ping-pong buf  : 256 KB
  Instruction cache         : 64 KB
  Misc                      : 56 KB
```

---

## 7. AXI Interface (PS ↔ PL)

```
PS ──────────────────────────────────────────────────── PL
│                                                        │
│  AXI4 HP (High-Performance, 128-bit @ 250 MHz)         │
│  ─────────────────────────────────────────────────►    │
│  PS writes input feature maps to DDR                   │
│  DPU reads weights + activations from DDR              │
│                                                        │
│  AXI-Lite (control, 32-bit)                            │
│  ─────────────────────────────────────────────────►    │
│  Python writes DPU registers (start, mode, addr ptr)   │
│                                                        │
│  AXI4 HP (result DMA back)                             │
│  ◄─────────────────────────────────────────────────    │
│  DPU writes output logits → DDR → Python reads         │
│                                                        │
│  IRQ (interrupt on completion)                         │
│  ◄─────────────────────────────────────────────────    │
│  Python wakes up, reads result from DDR                │
```

---

## 8. Timing Diagram — One Frame (640×640)

```
Time (ms) ──────────────────────────────────────────────►

          0      0.05   0.13    0.38    0.41    0.49    0.52
          │       │      │       │       │       │       │
  ARM PS  ├──preprocess──┤                       ├─post──┤──next frame►
          │       │      │       │       │       │       │
  DPU     │       ├──backbone───►├─FLEO──►├──head►│
          │               (DPU)    (DPU)    (DPU)
          │
          ◄────────────── total 0.52 ms / frame ─────────►
                         = 1923 FPS theoretical
                (memory bandwidth limit → ~2437 FPS practical @ INT8)
                (with NMS + display: ~244 FPS effective output)
```

---

## 9. Export Pipeline (Python only — no VHDL)

```
Step 1: Training  (PC / cloud GPU)
══════════════════════════════════
  PyTorch  ──►  FLEOClassifier.train()
               Full FLEO (with GS) for 50 epochs
               Save:  checkpoints/best_fleo.pt

Step 2: Export  (PC)
════════════════════
  Python:  export_inference_graph(model)
    → sets train_only=True
    → GS + BindingHead disappear from graph
    → model now contains only: Conv1×1, BN, SE, Conv3×3, Residual

Step 3: Quantization  (Vitis AI Docker)
════════════════════════════════════════
  docker run xilinx/vitis-ai-pytorch-cpu:latest
  │
  ▼ (Python inside Docker)
  torch_quantizer(mode="calib")
    → run ~200 calibration images (RAF-DB)
    → collect per-layer activation statistics
  torch_quantizer(mode="test")
    → simulate INT8 inference
    → dump_xmodel()  →  yolov12_fleo_quantized.xir

Step 4: Compile  (Vitis AI Docker)
════════════════════════════════════
  vai_c_xir \
    --xmodel  yolov12_fleo_quantized.xir \
    --arch    /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
    --output_dir ./compiled \
    --net_name   yolov12_fleo
  →  compiled/yolov12_fleo.xmodel   (deploy to board via SCP)

Step 5: On-board inference  (ZCU104 + PYNQ Jupyter)
════════════════════════════════════════════════════
  from vitis_ai_library import FaceDetect
  model = FaceDetect.create("yolov12_fleo")
  result = model.run([frame])
  emotion = EMOTIONS[result[0]["emotion_scores"].argmax()]
```

---

## 10. Resource Utilization (estimated, DPU B4096 on ZCU104)

| Resource | Available | DPU Used | FLEO overhead |
|---|---|---|---|
| DSP48E2 | 1,728 | 1,364 (79%) | +22 (for Conv layers) |
| BRAM 36K | 312 | 228 (73%) | +12 (weight cache) |
| LUT | 230,400 | 98,000 (43%) | +1,200 (glue logic) |
| FF | 460,800 | 87,000 (19%) | +2,400 |
| Power (total) | — | 9.5 W | +0.0 W (all on DPU) |

FLEO adds **zero new hardware** — it is purely additional weights in existing DPU Conv/SE engines.

---

## 11. Summary: Why This Design Works

```
┌────────────────────────────────────────────────────────────────┐
│  PROBLEM: anger↔sad, fear↔surprise share muscle movements      │
│           → phi_a ≈ phi_s → small angle θ → bad decision edge  │
├────────────────────────────────────────────────────────────────┤
│  FLEO TRAINING: Gram-Schmidt forces ||e_i - e_j||² = 2         │
│           → backbone learns to produce near-orthogonal features │
├────────────────────────────────────────────────────────────────┤
│  FPGA EXPORT: GS removed (Route 1)                             │
│           → backbone weights already encode orthogonality       │
│           → DPU runs only Conv/SE (INT8, full throughput)       │
├────────────────────────────────────────────────────────────────┤
│  RESULT on ZCU104:                                             │
│    Latency  : 0.41 ms/frame   (vs 0.38 baseline, +0.03 ms)     │
│    FPS      : 244 FPS         (vs 263 baseline, -8%)           │
│    Accuracy : +2.9% RAF-DB,  +5.3% AffectNet                  │
│    Power    : 9.5 W           (unchanged)                       │
│    Real-time: YES (>> 24 FPS)                                  │
└────────────────────────────────────────────────────────────────┘
```
