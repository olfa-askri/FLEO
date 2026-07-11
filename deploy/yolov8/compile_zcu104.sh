#!/usr/bin/env bash
# Compile the quantized YOLOv8 xmodel for the ZCU104 DPU (DPUCZDX8G / B4096).
# Run INSIDE the Vitis AI docker, AFTER quantize_yolov8.py.
set -e

XMODEL="${1:-quantize_result/YOLOv8BackboneNeck_int.xmodel}"
ARCH="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json"
OUT="compiled"
NET="yolov8_zcu104"

echo ">> compiling $XMODEL for ZCU104 ..."
vai_c_xir \
  --xmodel     "$XMODEL" \
  --arch       "$ARCH" \
  --output_dir "$OUT" \
  --net_name   "$NET"

echo ">> done. deploy this to the board:  $OUT/$NET.xmodel"
