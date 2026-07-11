"""
Run the compiled YOLOv8 xmodel on the ZCU104 board (VART + PYNQ).
================================================================
Copy compiled/yolov8_zcu104.xmodel to the board, then run this on-board.
DPU does backbone+neck; postprocess_arm.py does DFL decode + NMS on the ARM.
"""

import numpy as np
import cv2
import vart
import xir

from postprocess_arm import postprocess

IMGSZ = 640
NUM_CLASSES = 80          # set to YOUR trained class count


def letterbox_np(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    out = np.full((new_shape, new_shape, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


def make_runner(xmodel_path):
    g = xir.Graph.deserialize(xmodel_path)
    subs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_sub = [s for s in subs
               if s.has_attr("device") and s.get_attr("device") == "DPU"][0]
    return vart.Runner.create_runner(dpu_sub, "run")


def infer(runner, img_chw_int8):
    in_t = runner.get_input_tensors()
    out_t = runner.get_output_tensors()
    inp = [np.expand_dims(img_chw_int8, 0)]
    out = [np.empty(tuple(t.dims), np.int8) for t in out_t]
    jid = runner.execute_async(inp, out)
    runner.wait(jid)
    return out                       # list of raw feature maps (int8)


def main(xmodel="yolov8_zcu104.xmodel", image="test.jpg"):
    runner = make_runner(xmodel)
    raw = cv2.imread(image)
    lb, r, dx, dy = letterbox_np(raw, IMGSZ)
    x = lb[:, :, ::-1].transpose(2, 0, 1)              # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x).astype(np.int8)        # DPU expects int8

    feats = infer(runner, x)
    feats = [f[0] for f in feats]                      # drop batch dim -> (C,H,W)

    boxes, scores, cls = postprocess(feats, num_classes=NUM_CLASSES)

    # map boxes back to the original image (undo letterbox)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / r

    for (x1, y1, x2, y2), s, c in zip(boxes, scores, cls):
        cv2.rectangle(raw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(raw, f"{int(c)}:{s:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("result.jpg", raw)
    print(f"{len(boxes)} detections -> result.jpg")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
