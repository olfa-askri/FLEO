#!/usr/bin/env bash
# Compile a quantized xmodel to the B4096 DPU instruction stream (Fig. 5).
#
# Run inside the Vitis AI docker after quantize_vai.py --mode test has produced
# <quant_dir>/<ModelName>_int.xmodel.
#
# The B4096 arch fingerprint for the ZCU104 reference design ships with Vitis AI:
#   /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json
# (PP=8, ICP=OCP=16 -> 4096 ops/cycle; peak 1.229 TOPS at 300 MHz, Eq. (2)-(3)).
#
# Usage:
#   ./compile_b4096.sh <route> <quant_dir> <out_dir>
# Example:
#   ./compile_b4096.sh r1 quantized/r1 compiled/r1

set -euo pipefail

ROUTE="${1:?route (r1|r2|r3)}"
QUANT_DIR="${2:?quantized dir}"
OUT_DIR="${3:-compiled/$ROUTE}"
ARCH="${ARCH:-/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json}"
NET="fleo_yolov12s_${ROUTE}"

mkdir -p "$OUT_DIR"

XMODEL="$(ls "$QUANT_DIR"/*_int.xmodel 2>/dev/null | head -1 || true)"
if [[ -z "$XMODEL" ]]; then
  echo "No *_int.xmodel in $QUANT_DIR. Run quantize_vai.py --mode test first." >&2
  exit 1
fi

echo "Compiling $XMODEL for B4096 (arch=$ARCH)"
vai_c_xir \
  --xmodel   "$XMODEL" \
  --arch     "$ARCH" \
  --net_name "$NET" \
  --output_dir "$OUT_DIR"

echo
echo "Compiled artifacts in $OUT_DIR:"
ls -la "$OUT_DIR"
echo
echo "Inspect the subgraph partition (acceptance criterion, Section IV-E):"
echo "  xdputil xmodel $OUT_DIR/${NET}.xmodel -l   # list subgraphs/devices"
echo "  R1/R2: expect ONE DPU subgraph (+ CPU input/output)."
echo "  R3:    expect DPU - CPU(GS) - DPU per FLEO site."
