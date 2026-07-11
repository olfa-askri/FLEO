"""
Quantize a trained YOLOv8 (Ultralytics .pt) to INT8 for Xilinx DPU (ZCU104).
============================================================================
Flow:  .pt  --split-->  backbone+neck  --calib/test-->  .xmodel

IMPORTANT — why we split the model:
    YOLOv8's Detect head (DFL + anchor decode + sigmoid) does NOT quantize
    cleanly on the DPU. The standard fix (same as AMD Model Zoo) is to run the
    backbone + neck on the DPU and do the decode + NMS on the ARM CPU. So we
    quantize ONLY up to the 3 raw feature maps and hand those to
    postprocess_arm.py at inference time.

Run this INSIDE the Vitis AI pytorch docker:
    docker run -it --rm -v $(pwd):/work xilinx/vitis-ai-pytorch-cpu:3.5.0
    cd /work && python deploy/yolov8/quantize_yolov8.py \
        --weights best.pt --calib_dir /work/calib_images --imgsz 640
"""

import argparse
import glob
import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Load the trained YOLOv8 and expose the raw multi-scale feature maps
# ---------------------------------------------------------------------------
class YOLOv8BackboneNeck(torch.nn.Module):
    """
    Wraps an Ultralytics DetectionModel and returns the 3 feature maps that
    FEED the Detect head (P3/P4/P5), i.e. everything the DPU can run. The
    Detect head itself is skipped here and re-implemented on the ARM CPU
    (see postprocess_arm.py).
    """

    def __init__(self, weights: str):
        super().__init__()
        from ultralytics import YOLO
        full = YOLO(weights).model.eval()          # DetectionModel
        self.model = full.model                    # nn.Sequential of layers
        self.save = full.save                      # indices whose output is reused
        self.detect_idx = len(self.model) - 1      # last layer == Detect

    def forward(self, x):
        y = []                                     # cached layer outputs
        for i, m in enumerate(self.model):
            if m.f != -1:                          # layer takes non-previous input
                x = (y[m.f] if isinstance(m.f, int)
                     else [x if j == -1 else y[j] for j in m.f])
            if i == self.detect_idx:
                # `x` here is the list [P3, P4, P5] going INTO Detect.
                return x                           # raw feature maps (DPU output)
            x = m(x)
            y.append(x if i in self.save else None)
        return x


# ---------------------------------------------------------------------------
# 2. Calibration data loader (real images, preprocessed like training)
# ---------------------------------------------------------------------------
def load_calib_images(calib_dir: str, imgsz: int, limit: int = 300):
    """Letterbox + normalize, exactly like YOLOv8 training preprocessing."""
    import cv2

    paths = sorted(glob.glob(os.path.join(calib_dir, "*")))[:limit]
    assert paths, f"no calibration images in {calib_dir}"
    batch = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = letterbox(img, imgsz)                # (imgsz, imgsz, 3) BGR
        img = img[:, :, ::-1].transpose(2, 0, 1)   # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        batch.append(torch.from_numpy(img))
    print(f"loaded {len(batch)} calibration images")
    return batch


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to a square — YOLOv8 default."""
    import cv2

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)


# ---------------------------------------------------------------------------
# 3. Quantize (calib -> test -> export xmodel)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="trained YOLOv8 .pt")
    ap.add_argument("--calib_dir", required=True, help="folder of real images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", default="quantize_result")
    args = ap.parse_args()

    from pytorch_nndct.apis import torch_quantizer   # Vitis AI only

    model = YOLOv8BackboneNeck(args.weights).eval()
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    calib = load_calib_images(args.calib_dir, args.imgsz, args.limit)

    # ---- CALIB pass: gather activation ranges -----------------------------
    q = torch_quantizer("calib", model, (dummy), output_dir=args.out)
    qm = q.quant_model
    with torch.no_grad():
        for img in calib:
            qm(img.unsqueeze(0))
    q.export_quant_config()
    print("calibration done")

    # ---- TEST pass: export the INT8 xmodel --------------------------------
    q = torch_quantizer("test", model, (dummy), output_dir=args.out)
    qm = q.quant_model
    with torch.no_grad():
        qm(dummy)
    q.export_xmodel(deploy_check=True)
    print(f"xmodel written to {args.out}/  -> next: run compile_zcu104.sh")


if __name__ == "__main__":
    main()
