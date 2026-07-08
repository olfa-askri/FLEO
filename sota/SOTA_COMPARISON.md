# SOTA comparison

Accuracy tables are **contextual**: published FER methods are GPU classifiers, whereas this work is a detection-cast detector deployed at INT8 on an FPGA overlay. The decisive internal comparison is Delta_fold (full vs folded), not the gap to a GPU classifier.

### FER2013 -- overall accuracy (as reported; protocols differ)

| Method | Year | Acc (%) | Note |
|---|---|---|---|
| Human (Goodfellow et al.) | 2013 | 65.00 | reported human ceiling ~65+/-5 |
| VGG (Khaireddin & Chen) | 2021 | 73.28 | single VGG, heavy aug + tuning |
| ResMaskingNet (single) | 2020 | 74.14 | Pham et al. |
| LHC-Net | 2021 | 74.42 | Pecoraro et al., local multi-head channel |
| Segmentation-VGG19 | 2021 | 75.97 | as reported |
| EmoNeXt | 2023 | 76.50 | ConvNeXt-based, single model |
| ResMaskingNet (ensemble) | 2020 | 76.82 | 6-net ensemble |
| ConvNeXt + TripSE | 2025 | 77.50 | attention + feature fusion (approx 76-78) |
| EfficientNet-B0 + GFPGAN | 2025 | 86.44 | NON-STANDARD: uses face restoration |
| Diffusion synthetic aug. | 2024 | 96.47 | NON-STANDARD: extra synthetic training data |
| **This work (FLEO, detection-cast)** | 2026 | -- | R1 fold-out, INT8 on B4096 |

### RAF-DB -- overall accuracy (as reported; protocols differ)

| Method | Year | Acc (%) | Note |
|---|---|---|---|
| RAN | 2020 | 86.90 | Wang et al., region attention |
| SCN | 2020 | 87.03 | self-cure network |
| DACL | 2021 | 87.78 | deep attentive center loss |
| EfficientFace | 2021 | 88.28 | lightweight, 1.28M params |
| DAN | 2021 | 89.70 | distract your attention |
| EAC | 2022 | 90.35 | erasing attention consistency |
| TransFER | 2021 | 90.91 | transformer-based |
| DDAMFN | 2023 | 91.35 | dual-direction attention, 8.2M params |
| APViT | 2022 | 91.98 | attentive pooling ViT |
| TriCAFFNet | 2024 | 92.17 | tri-cross-attention transformer |
| POSTER++ | 2023 | 92.21 | cross-fusion transformer |
| FMAE-IAT | 2024 | 93.54 | ViT-Large, 304.5M params |
| ResEmoteNet | 2024 | 94.76 | Roy et al., 80.24M params, current SOTA |
| **This work (FLEO, detection-cast)** | 2026 | -- | R1 fold-out, INT8 on B4096 |

### FPGA expression-recognition systems (Table IX)

| System | Platform | Style | Precision | Throughput | Note |
|---|---|---|---|---|---|
| Phan-Xuan et al. (2019) | FPGA (CNN) | bespoke | fixed-point | -- | hand-mapped layers |
| Vinh & Vinh (2019) | Zynq SoC | bespoke | fixed-point | -- | detection+classification on Zynq |
| CNN-FPGA (2020) | FPGA @130 MHz | bespoke | fixed-point | 15 FPS | FER2013 ~66%; 6.37 ms/face |
| Ando & Inoue (2025) | Zynq US+ Kria KV260 (B512) | overlay | INT8 | 25 FPS (2-thread) | arXiv 2511.02408; FER2013 acc=67.4%, power=2.7 W, L_dpu=7.34 ms, det DenseBox AP=0.917; ~9.3 FPS/W -- the closest competitor |
| Askri et al. (prior) | - | - | - | -- | quantum-inspired FER optimization |
| This work, R1 (fold-out) | ZCU104 (B4096) | overlay | INT8 | -- | pending |
| This work, R2 (Householder) | ZCU104 (B4096) | overlay | INT8 | -- | pending |
| This work, R3 (hybrid PS/PL) | ZCU104 (B4096) | overlay | INT8+PS | -- | pending |

_Our cells are populated from results/ when available; dashes are pending._