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
    ("ResMaskingNet (ensemble)", 2020, 76.82, "6-net ensemble"),
    ("Segmentation-VGG19", 2021, 75.97, "as reported"),
]

RAFDB_SOTA = [
    # 7-class "basic" overall accuracy, %
    ("RAN", 2020, 86.90, "Wang et al., region attention"),
    ("SCN", 2020, 87.03, "self-cure network"),
    ("DACL", 2021, 87.78, "deep attentive center loss"),
    ("DAN", 2021, 89.70, "distract your attention"),
    ("TransFER", 2021, 90.91, "transformer-based"),
    ("EAC", 2022, 90.35, "erasing attention consistency"),
    ("APViT", 2022, 91.98, "attentive pooling ViT"),
    ("POSTER++", 2023, 92.21, "cross-fusion transformer"),
]

# --- 2. FPGA expression-recognition systems (manuscript Table IX) ------------
FPGA_FER_SYSTEMS = [
    # (system, platform, style, precision, throughput, note)
    ("Phan-Xuan et al.", "FPGA (CNN)", "bespoke", "fixed-point", None, "hand-mapped layers"),
    ("Vinh & Vinh", "Zynq SoC", "bespoke", "fixed-point", None, "detection+classification on Zynq"),
    ("Ando & Inoue", "Zynq US+ (DPU)", "overlay", "INT8", "25 FPS", "det+cls on one DPU, CPU-bound"),
    ("Askri et al. (prior)", "-", "-", "-", None, "quantum-inspired FER optimization"),
    # Our rows -- filled from results at report time.
    ("This work, R1 (fold-out)", "ZCU104 (B4096)", "overlay", "INT8", None, "pending"),
    ("This work, R2 (Householder)", "ZCU104 (B4096)", "overlay", "INT8", None, "pending"),
    ("This work, R3 (hybrid PS/PL)", "ZCU104 (B4096)", "overlay", "INT8+PS", None, "pending"),
]
