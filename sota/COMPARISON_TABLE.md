# Table IX — FER-on-FPGA systems comparison

Peer-reviewed / reported figures for facial-expression-recognition systems that
**train a network and then deploy it on an FPGA**. GPU-only accuracy SOTA is
intentionally excluded here (see `SOTA_COMPARISON.md` for that context) because
those models are not edge-deployable.

`*` = estimate / pending real measurement (fill from `results/deltas_*.json`,
the ZCU104 board, and Vivado reports).

| # | System (Ref) | Year | Dataset | Acc (%) | Network | FPGA Device | Accel. | Precision | Freq | FPS | Latency | Power | FPS/W | Contribution |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Phan-Xuan et al. | — | FER | — | CNN | FPGA | bespoke | fixed-pt | — | — | — | — | — | hand-mapped CNN |
| 2 | Vinh & Vinh | 2019 | FER2013 | 66.0 | CNN | Zynq SoC | bespoke | fixed-pt | 130 MHz | 15 | — | — | — | det+cls on Zynq |
| 3 | Emotion recognizer (autistic) | — | custom | 72.9 | CNN | Virtex-7 | bespoke | 12-bit | — | — | — | — | — | portable recognizer |
| 4 | Ando & Inoue | 2025 | FER2013 | 67.4 | CNN + DenseBox | Kria KV260 | B512 DPU | INT8 | 300 MHz | 25 | 7.34 ms | 2.7 W | ~9.3 | multi-thread DPU |
| 5 | **This work (FLEO)** | 2025 | FER2013 + RAF-DB | ~73* / ~83* | YOLOv12-S + FLEO | ZCU104 | B4096 DPU | INT8 | 300 MHz | ~30* | ~14 ms* | pending | pending | **fold-out methodology** |

## Where each number comes from (this work)
| Column | Source |
|---|---|
| Acc (%) | `scripts/deltas.py` -> `results/deltas_{fer2013,rafdb}.json` (top-1) |
| FPS / Latency | `deploy/app/vart_pipeline.py` on the ZCU104 |
| Power / FPS/W | `deploy/measure/power_ina226.py` (INA226 rails) |
| Resources (LUT/DSP/BRAM) | `deploy/measure/parse_vivado.py` (Vivado report) |

## Who wins (FER-on-FPGA only)
- **Accuracy:** this work (~73% > 66-72.9% of prior FPGA systems).
- **Datasets:** this work (only one reporting FER2013 **and** RAF-DB).
- **Throughput:** this work (~30 FPS, if it holds vs the CPU-bound ceiling).
- **Power efficiency (FPS/W):** Ando & Inoue (smaller B512 DPU draws less).
- **Method:** this work (fold-out of an accelerator-hostile training-time operator).

## Framing for the paper
Among FER-on-FPGA systems this work reaches the highest accuracy and is the only
one evaluated on both FER2013 and RAF-DB, at competitive throughput, with a novel
fold-out deployment methodology; the smaller-DPU baseline (Ando & Inoue) remains
more power-efficient, which motivates reporting device-level (not board-level)
power and, optionally, a smaller-DPU variant.
