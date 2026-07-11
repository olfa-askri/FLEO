# Deploy trained YOLOv8 on ZCU104 FPGA (Vitis AI)

FLEO stays a **standalone FER research module**. For on-board detection we
deploy a **plain YOLOv8** (no FLEO) because it quantizes cleanly on the DPU.

## Pipeline

```
best.pt  ──split──►  backbone+neck  ──INT8──►  .xmodel  ──compile──►  DPU
   │                      │                        │                    │
Ultralytics         quantize_yolov8.py       compile_zcu104.sh    run_on_board.py
                    (Vitis AI docker)        (Vitis AI docker)   + postprocess_arm.py
                                                                      (ARM CPU)
```

**Why split the model:** YOLOv8's Detect head (DFL decode + anchor assembly +
sigmoid) does not quantize well on the DPU. Standard practice (AMD Model Zoo):
run **backbone + neck on the DPU**, run **decode + NMS on the ARM CPU**.

## Files

| file | where it runs | what it does |
|---|---|---|
| `quantize_yolov8.py` | Vitis AI docker | splits `.pt`, INT8 calib/test, exports `.xmodel` |
| `compile_zcu104.sh` | Vitis AI docker | `vai_c_xir` → compiled `.xmodel` for ZCU104 |
| `postprocess_arm.py` | ZCU104 ARM | DFL decode + sigmoid + NMS (pure NumPy) |
| `run_on_board.py` | ZCU104 ARM | VART runner: image → DPU → postproc → boxes |

## Steps

### 1. Quantize (inside Vitis AI docker)
```bash
docker run -it --rm -v $(pwd):/work xilinx/vitis-ai-pytorch-cpu:3.5.0
cd /work
pip install ultralytics opencv-python          # if not present
python deploy/yolov8/quantize_yolov8.py \
    --weights best.pt \
    --calib_dir /work/calib_images \            # 100–1000 REAL images
    --imgsz 640
```

### 2. Compile for ZCU104 (inside docker)
```bash
bash deploy/yolov8/compile_zcu104.sh
# -> compiled/yolov8_zcu104.xmodel
```

### 3. Run on the board (PYNQ on ZCU104)
```bash
# copy compiled/yolov8_zcu104.xmodel + postprocess_arm.py + run_on_board.py
python run_on_board.py yolov8_zcu104.xmodel test.jpg
# -> result.jpg
```

## Before you run — set these to YOUR model

- `NUM_CLASSES` in `run_on_board.py` (default 80) → your trained class count.
- `reg_max` in `postprocess_arm.py` (default 16) → YOLOv8 default, rarely changes.
- `--imgsz` must match training image size (640 default).

## Common mistakes

1. **Not splitting the Detect head** → compile fails / accuracy collapses.
2. **Too few / off-distribution calib images** → weak INT8 accuracy. Use ≥100
   real validation images.
3. **On-board preprocessing ≠ training** → wrong boxes. Keep letterbox 640,
   `/255`, BGR→RGB identical to training.
4. **Wrong `strides`** in `postprocess_arm.py` — YOLOv8 uses (8, 16, 32) for
   P3/P4/P5. Change only if you altered the neck.
