"""Post-processing for the VART pipeline: dequantize + top-1 emotion decode.

The detection is cast to a single dominant emotion per frame (Section V-F), so we
skip full box regression + NMS and take the class of the highest-scoring anchor --
the same NMS-free rule used by the host evaluator, keeping FP32/INT8 comparable.
The DPU emits INT8; we dequantize with the output fixed-point scale.
"""

from __future__ import annotations

import numpy as np


def dequantize(out_int8: np.ndarray, fix_point: int) -> np.ndarray:
    return out_int8.astype(np.float32) / float(2 ** fix_point)


def decode_topclass(out, nc: int, fix_point: int = 0) -> int:
    """Return the predicted emotion id from a decoded head tensor.

    Accepts the head output in any of the common layouts:
        (1, 4+nc, N)  |  (1, N, 4+nc)  |  (4+nc, N)  |  (N, 4+nc)
    Integer inputs are dequantized first.
    """
    a = np.asarray(out)
    if np.issubdtype(a.dtype, np.integer):
        a = dequantize(a, fix_point)
    a = np.squeeze(a)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1) if a.shape[0] == 4 + nc else a.reshape(-1, 4 + nc)
    if a.shape[0] == 4 + nc:
        cls = a[4 : 4 + nc, :]           # (nc, N)
    elif a.shape[1] == 4 + nc:
        cls = a[:, 4 : 4 + nc].T          # (nc, N)
    else:
        raise ValueError(f"cannot locate {nc} class channels in shape {a.shape}")
    # Per-class max confidence over anchors, then argmax -- identical to the host
    # evaluator (scripts.evaluate._topclass_from_head), so FP32 and on-target INT8
    # are compared under the exact same decision rule.
    return int(cls.max(axis=1).argmax())
