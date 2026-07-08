"""Reference SOTA numbers for context (as reported by the cited works).

Two comparison contexts, matching the manuscript:

1. Software (GPU) accuracy SOTA on FER2013 and RAF-DB -- to situate the detector's
   *accuracy*.  Protocols differ (classification vs our detection-cast top-1), so
   this is contextual, not a head-to-head claim.

2. FPGA expression-recognition *systems* (manuscript Table IX) -- to situate the
   *deployment* (platform, style, precision, throughput).

Numbers are the single-model figures most commonly cited; verify against the
original papers before publication.  Fields left None are intentionally open.
"""

# --- 1. Software accuracy SOTA (overall test accuracy, %) --------------------
FER2013_SOTA = [
    # (method, year, accuracy%, note)
    ("Human (Goodfellow et al.)", 2013, 65.0, "reported human ceiling ~65+/-5"),
    ("VGG (Khaireddin & Chen)", 2021, 73.28, "single VGG, heavy aug + tuning"),
    ("ResMaskingNet (single)", 2020, 74.14, "Pham et al."),
    ("LHC-Net", 2021, 74.42, "Pecoraro et al., local multi-head channel"),
    ("Segmentation-VGG19", 2021, 75.97, "as reported"),
    ("EmoNeXt", 2023, 76.5, "ConvNeXt-based, single model"),
    ("ResMaskingNet (ensemble)", 2020, 76.82, "6-net ensemble"),
    ("ConvNeXt + TripSE", 2025, 77.5, "attention + feature fusion (approx 76-78)"),
    ("EfficientNet-B0 + GFPGAN", 2025, 86.44, "NON-STANDARD: uses face restoration"),
    ("Diffusion synthetic aug.", 2024, 96.47, "NON-STANDARD: extra synthetic training data"),
]

RAFDB_SOTA = [
    # 7-class "basic" overall accuracy, %
    ("RAN", 2020, 86.90, "Wang et al., region attention"),
    ("SCN", 2020, 87.03, "self-cure network"),
    ("DACL", 2021, 87.78, "deep attentive center loss"),
    ("EfficientFace", 2021, 88.28, "lightweight, 1.28M params"),
    ("DAN", 2021, 89.70, "distract your attention"),
    ("EAC", 2022, 90.35, "erasing attention consistency"),
    ("TransFER", 2021, 90.91, "transformer-based"),
    ("DDAMFN", 2023, 91.35, "dual-direction attention, 8.2M params"),
    ("APViT", 2022, 91.98, "attentive pooling ViT"),
    ("TriCAFFNet", 2024, 92.17, "tri-cross-attention transformer"),
    ("POSTER++", 2023, 92.21, "cross-fusion transformer"),
    ("FMAE-IAT", 2024, 93.54, "ViT-Large, 304.5M params"),
    ("ResEmoteNet", 2024, 94.76, "Roy et al., 80.24M params, current SOTA"),
]

# --- 2. FPGA expression-recognition systems (manuscript Table IX) ------------
FPGA_FER_SYSTEMS = [
    # (system, platform, style, precision, throughput, note)
    # Fields carry the *deployment* figures that matter at the edge: FER accuracy,
    # FPS, power. These are the real head-to-head numbers (accuracy on FPGA is far
    # below GPU SOTA for everyone -- small edge models).
    ("Phan-Xuan et al. (2019)", "FPGA (CNN)", "bespoke", "fixed-point", None,
     "hand-mapped layers"),
    ("Vinh & Vinh (2019)", "Zynq SoC", "bespoke", "fixed-point", None,
     "detection+classification on Zynq"),
    ("CNN-FPGA (2020)", "FPGA @130 MHz", "bespoke", "fixed-point", "15 FPS",
     "FER2013 ~66%; 6.37 ms/face"),
    ("Ando & Inoue (2025)", "Zynq US+ Kria KV260 (B512)", "overlay", "INT8",
     "25 FPS (2-thread)",
     "arXiv 2511.02408; FER2013 acc=67.4%, power=2.7 W, L_dpu=7.34 ms, "
     "det DenseBox AP=0.917; ~9.3 FPS/W -- the closest competitor"),
    ("Askri et al. (prior)", "-", "-", "-", None, "quantum-inspired FER optimization"),
    # Our rows -- filled from results at report time. B4096 is 8x the DPU of the
    # 2025 KV260/B512 baseline: more throughput headroom, but watch FPS/W since a
    # larger DPU costs more power on a lightweight task (justify the choice).
    ("This work, R1 (fold-out)", "ZCU104 (B4096)", "overlay", "INT8", None, "pending"),
    ("This work, R2 (Householder)", "ZCU104 (B4096)", "overlay", "INT8", None, "pending"),
    ("This work, R3 (hybrid PS/PL)", "ZCU104 (B4096)", "overlay", "INT8+PS", None, "pending"),
]
