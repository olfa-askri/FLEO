# SOTA comparison

Accuracy tables are **contextual**: published FER methods are GPU classifiers, whereas this work is a detection-cast detector deployed at INT8 on an FPGA overlay. The decisive internal comparison is Delta_fold (full vs folded), not the gap to a GPU classifier.

### FER2013 -- overall accuracy (as reported; protocols differ)

| Method | Year | Acc (%) | Note |
|---|---|---|---|
| Human (Goodfellow et al.) | 2013 | 65.00 | reported human ceiling ~65+/-5 |
| VGG (Khaireddin & Chen) | 2021 | 73.28 | single VGG, heavy aug + tuning |
| ResMaskingNet (single) | 2020 | 74.14 | Pham et al. |
| LHC-Net | 2021 | 74.42 | Pecoraro et al., local multi-head channel |
| ResMaskingNet (ensemble) | 2020 | 76.82 | 6-net ensemble |
| Segmentation-VGG19 | 2021 | 75.97 | as reported |
| **This work (FLEO, detection-cast)** | 2026 | -- | R1 fold-out, INT8 on B4096 |

### RAF-DB -- overall accuracy (as reported; protocols differ)

| Method | Year | Acc (%) | Note |
|---|---|---|---|
| RAN | 2020 | 86.90 | Wang et al., region attention |
| SCN | 2020 | 87.03 | self-cure network |
| DACL | 2021 | 87.78 | deep attentive center loss |
| DAN | 2021 | 89.70 | distract your attention |
| TransFER | 2021 | 90.91 | transformer-based |
| EAC | 2022 | 90.35 | erasing attention consistency |
| APViT | 2022 | 91.98 | attentive pooling ViT |
| POSTER++ | 2023 | 92.21 | cross-fusion transformer |
| **This work (FLEO, detection-cast)** | 2026 | -- | R1 fold-out, INT8 on B4096 |

### FPGA expression-recognition systems (Table IX)

| System | Platform | Style | Precision | Throughput | Note |
|---|---|---|---|---|---|
| Phan-Xuan et al. | FPGA (CNN) | bespoke | fixed-point | -- | hand-mapped layers |
| Vinh & Vinh | Zynq SoC | bespoke | fixed-point | -- | detection+classification on Zynq |
| Ando & Inoue | Zynq US+ (DPU) | overlay | INT8 | 25 FPS | det+cls on one DPU, CPU-bound |
| Askri et al. (prior) | - | - | - | -- | quantum-inspired FER optimization |
| This work, R1 (fold-out) | ZCU104 (B4096) | overlay | INT8 | -- | pending |
| This work, R2 (Householder) | ZCU104 (B4096) | overlay | INT8 | -- | pending |
| This work, R3 (hybrid PS/PL) | ZCU104 (B4096) | overlay | INT8+PS | -- | pending |

_Our cells are populated from results/ when available; dashes are pending._