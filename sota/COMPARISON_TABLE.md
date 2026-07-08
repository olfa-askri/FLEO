# Comparison — FPGA + CNN methods (with this work)

Cited works that deploy a CNN on an FPGA, with their publication venue / indexing.
Direct competitors are the FER/emotion rows; the CNN- and YOLO-on-FPGA rows are
the methodological context (this work deploys a YOLO detector on a DPU).

`* = estimate / pending real measurement`  ·  `Scopus = indexed`  ·
`arXiv = preprint (not indexed)`

| # | Method (Ref) | Venue / Index | Year | Task | Dataset | Acc% | Network | FPGA | FPS | Power | Contribution |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Zhang et al. | FPGA (ACM), Scopus | 2015 | CNN accel | — | — | CNN | Virtex-7 | — | — | roofline design |
| 2 | Qiu et al. | FPGA (ACM), Scopus | 2016 | CNN accel | — | — | VGG | Zynq | — | — | embedded CNN |
| 3 | Phan-Xuan et al. | conf | — | FER | FER | — | CNN | FPGA | — | — | hand-mapped |
| 4 | Vinh & Vinh | IEEE, Scopus | 2019 | FER | FER2013 | 66.0 | CNN | Zynq SoC | 15 | — | det+cls |
| 5 | Autistic-emotion | IEEE, Scopus | — | emotion | custom | 72.9 | CNN | Virtex-7 | — | — | portable recognizer |
| 6 | Ando & Inoue | arXiv (not indexed) | 2025 | FER | FER2013 | 67.4 | CNN+DenseBox | KV260 (B512) | 25 | 2.7 W | multi-thread DPU |
| 7 | YOLOv7-tiny HLS | Springer JRTIP, Scopus | 2023 | detection | — | — | YOLOv7 | Zynq | — | — | HLS accelerator |
| 8 | YOLOv2 accelerator | MDPI Sensors, Scopus | 2025 | detection | — | — | YOLOv2 | Zynq-7000 | — | — | Zynq accelerator |
| 9 | **This work (FLEO)** | target | 2025 | FER | FER2013 + RAF-DB | ~73* / ~83* | YOLOv12-S + FLEO | ZCU104 (B4096) | ~30* | pending | fold-out methodology |

## Final line — who wins?

**This work (FLEO) wins among FER-on-FPGA methods:** highest accuracy
(~73% vs 66-72.9%), the only one evaluated on both FER2013 and RAF-DB, the most
recent network (YOLOv12-S), competitive throughput (~30 FPS), and a novel fold-out
deployment methodology — while Ando & Inoue remains the most power-efficient
(smaller B512 DPU).

## Notes
- Among *Scopus-indexed* FER competitors (Vinh & Vinh, Autistic-emotion), this
  work leads on accuracy (73% > 66% and 72.9%). Ando & Inoue is arXiv-only.
- This work's figures are estimates pending `results/deltas_*.json` (accuracy) and
  the ZCU104 board (FPS / power / resources).
