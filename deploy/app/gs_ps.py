"""Route-3 processing-system Gram-Schmidt (the FP32 island on the A53 cluster).

For Route 3 the compiler splits each FLEO site at the GS boundary; the DPU runs
the two convolution subgraphs and this operator runs on the CPU in between,
reproducing Eq. (4) exactly.  It is the O(K^2 D) serialized recurrence whose cost
Route 3 exists to price (Section IV-D); D = d*H*W.

numpy-only so it runs on the target without torch.
"""

from __future__ import annotations

import numpy as np


def gram_schmidt_np(x: np.ndarray, k: int, eps: float = 1e-6) -> np.ndarray:
    """Per-sample modified Gram-Schmidt over K flattened subspaces (NCHW float32).

    Mirrors fleo.orthogonal.gram_schmidt so R3's on-target result matches the host
    reference bit-for-bit up to float error.
    """
    b, c, h, w = x.shape
    assert c % k == 0
    v = x.reshape(b, k, -1).astype(np.float32)  # (B, K, D)
    us = np.empty_like(v)
    for i in range(k):
        u_i = v[:, i, :].copy()
        for j in range(i):
            u_j = us[:, j, :]
            num = np.sum(v[:, i, :] * u_j, axis=1, keepdims=True)
            den = np.sum(u_j * u_j, axis=1, keepdims=True) + eps
            u_i -= (num / den) * u_j
        us[:, i, :] = u_i
    norm = np.linalg.norm(us, axis=2, keepdims=True) + eps
    e = us / norm
    return e.reshape(b, c, h, w)


if __name__ == "__main__":
    x = np.random.randn(2, 7 * 8, 4, 4).astype(np.float32)
    e = gram_schmidt_np(x, 7)
    v = e.reshape(2, 7, -1)
    gram = np.einsum("bkd,bld->bkl", v, v)
    err = np.abs(gram - np.eye(7)[None]).max()  # orthonormality error (diag->1, off->0)
    print("orthonormality error:", float(err))
