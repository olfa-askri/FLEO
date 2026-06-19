"""
FPGA Export Pipeline – Route 1 (Train-time-only FLEO)
Target: Xilinx ZCU104  |  DPU: B4096  |  Framework: Vitis AI 3.5

Steps:
  1. Load trained checkpoint
  2. Switch to inference mode (disables Gram-Schmidt)
  3. Quantize weights FP32 → INT8 using Vitis AI Quantizer
  4. Compile to .xmodel for DPU B4096

IMPORTANT: Run inside the Vitis AI Docker container:
  docker run -it --rm xilinx/vitis-ai-pytorch-cpu:latest
"""

import torch
from models import YOLOv12FLEONeck, export_inference_graph


# ── Step 1: Load model and switch to Route 1 inference mode ──────────

def prepare_model_for_export(checkpoint_path: str, model_fn, cfg: dict):
    """
    Load a trained YOLOv12+FLEO model and prepare it for FPGA export.

    Args:
        checkpoint_path : path to best_fleo.pt
        model_fn        : callable that returns the full model
        cfg             : model configuration dict
    """
    model = model_fn(cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)

    # Route 1: disable GS ops, set eval mode
    model = export_inference_graph(model.neck)
    return model


# ── Step 2: Vitis AI Quantization (INT8) ─────────────────────────────

def quantize_model(model: torch.nn.Module, calib_loader, output_dir: str):
    """
    Quantize the export-ready model using Vitis AI PyTorch Quantizer.
    Requires: pip install pytorch-nndct  (inside Vitis AI Docker)

    Calibration data: ~100–200 representative images from RAF-DB.
    """
    try:
        from pytorch_nndct.apis import torch_quantizer, dump_xmodel

        # Build quantizer (calib mode)
        dummy_input = torch.randn(1, 3, 640, 640)
        quantizer = torch_quantizer(
            quant_mode="calib",
            module=model,
            input_args=dummy_input,
            output_dir=output_dir,
            device=torch.device("cpu"),
        )
        quant_model = quantizer.quant_model

        # Run calibration forward passes
        quant_model.eval()
        with torch.no_grad():
            for imgs, _ in calib_loader:
                quant_model(imgs)

        quantizer.export_quant_config()

        # Test mode to get final INT8 model
        quantizer_test = torch_quantizer(
            quant_mode="test",
            module=model,
            input_args=dummy_input,
            output_dir=output_dir,
            device=torch.device("cpu"),
        )
        quant_model_test = quantizer_test.quant_model
        dump_xmodel(output_dir=output_dir, deploy_check=True)

        print(f"[✓] INT8 quantized model saved to {output_dir}/quantize_result/")
        return quant_model_test

    except ImportError:
        print("[!] pytorch_nndct not found – run inside Vitis AI Docker container.")
        return model


# ── Step 3: Compile to .xmodel for DPU B4096 ─────────────────────────

COMPILE_CMD = """
# Run this command inside Vitis AI Docker:
vai_c_xir \\
  --xmodel {quant_dir}/quantize_result/YOLOv12_FLEO_int.xir \\
  --arch    /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \\
  --output_dir {output_dir} \\
  --net_name   yolov12_fleo
"""


def print_compile_instructions(quant_dir: str, output_dir: str = "./compiled"):
    print("\n" + "=" * 60)
    print("  STEP 3 – Compile to .xmodel for ZCU104 DPU B4096")
    print("=" * 60)
    print(COMPILE_CMD.format(quant_dir=quant_dir, output_dir=output_dir))
    print("Output: yolov12_fleo.xmodel  →  copy to ZCU104 board")


# ── Step 4: On-board inference with PYNQ / Vitis AI Library ──────────

PYNQ_INFERENCE_TEMPLATE = '''
# ── Run on ZCU104 board in Jupyter Notebook (PYNQ environment) ───────
from vitis_ai_library import FaceDetect
import cv2, numpy as np

# Load compiled model
detector = FaceDetect.create("yolov12_fleo")

# Open camera
cap = cv2.VideoCapture(0)

EMOTION_LABELS = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.run([frame])

    for r in results[0]:
        bbox  = r["bbox"]          # [x1, y1, x2, y2]
        scores = r["emotion_scores"]  # (7,) softmax output
        emotion_id = int(np.argmax(scores))
        label = EMOTION_LABELS[emotion_id]
        conf  = float(scores[emotion_id])

        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("FLEO @ ZCU104", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
'''


if __name__ == "__main__":
    print_compile_instructions(quant_dir="./quantized")
    print("\n── PYNQ On-board Inference Template ──")
    print(PYNQ_INFERENCE_TEMPLATE)
