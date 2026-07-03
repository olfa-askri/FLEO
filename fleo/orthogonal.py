"""Orthogonalization operators for FLEO.

This module implements the two orthogonalization structures discussed in the
manuscript:

* ``gram_schmidt`` / ``GramSchmidt`` -- the per-sample, differentiable
  Gram-Schmidt map used *during training* (Eq. (4)).  It contains exactly the
  three properties that disqualify it from the DPUCZDX8G datapath: a division by
  a data-dependent quantity, a reciprocal square root, and a sequential
  recurrence across the K subspaces (Section IV-A).

* ``HouseholderOrtho`` -- the division-free, constant-foldable alternative of
  Route 2 (Section IV-C).  After training its reflection vectors are fixed, so
  the map collapses to a single dense 1x1 convolution whose per-channel scales
  are absorbed by the quantizer.  ``export_conv1x1`` performs that fold.

Both operate on NCHW activations that have already been projected to ``K*d``
channels; the K subspaces are the flattened vectors u_k in R^D, D = d*H*W.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def gram_schmidt(x: torch.Tensor, k: int, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample modified Gram-Schmidt over K flattened subspaces.

    Args:
        x: activations of shape ``(B, K*d, H, W)``.
        k: number of subspaces ``K``.
        eps: numerical floor for the division and the reciprocal sqrt.

    Returns:
        Tensor of the same shape whose K subspaces are mutually orthonormal
        (per sample).  The operation is the one folded out at export by Route 1.
    """
    b, c, h, w = x.shape
    assert c % k == 0, f"channel count {c} not divisible by K={k}"
    v = x.reshape(b, k, -1)  # (B, K, D), D = d*H*W

    us: list[torch.Tensor] = []
    for i in range(k):
        u_i = v[:, i, :]
        for u_j in us:
            # projection coefficient <v_i, u_j> / (||u_j||^2 + eps)   -- a division
            coef = (v[:, i, :] * u_j).sum(dim=1, keepdim=True) / (
                u_j.pow(2).sum(dim=1, keepdim=True) + eps
            )
            u_i = u_i - coef * u_j  # sequential recurrence over k
        us.append(u_i)

    u = torch.stack(us, dim=1)  # (B, K, D)
    # normalization e_k = u_k / (||u_k|| + eps)  -- a reciprocal square root
    e = u / (u.norm(dim=2, keepdim=True) + eps)
    return e.reshape(b, c, h, w)


class GramSchmidt(nn.Module):
    """Module wrapper around :func:`gram_schmidt` (Route 1 / Route 3 operator)."""

    def __init__(self, k: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.k = k
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gram_schmidt(x, self.k, self.eps)

    def extra_repr(self) -> str:
        return f"k={self.k}, eps={self.eps}"


class HouseholderOrtho(nn.Module):
    """Product of Householder reflections Q = H_1 H_2 ... H_m  (Eq. (7)).

    During training ``forward`` recomputes Q from the learned reflection vectors
    so gradients flow.  Because the vectors are *fixed* after training, the whole
    map is a constant orthogonal matrix; :meth:`export_conv1x1` materialises it as
    a 1x1 convolution so the deployed operator stays on the accelerator.
    """

    def __init__(self, dim: int, m: int | None = None, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.m = dim if m is None else m
        self.eps = eps
        # small reflections -> Q starts near identity, a benign init.
        self.w = nn.Parameter(torch.randn(self.m, dim) * 0.01)

    def matrix(self) -> torch.Tensor:
        """Return the orthogonal matrix Q (dim x dim)."""
        device = self.w.device
        q = torch.eye(self.dim, device=device, dtype=self.w.dtype)
        for j in range(self.m):
            wj = self.w[j]
            denom = wj.dot(wj) + self.eps  # fixed after training -> constant-folded
            hj = torch.eye(self.dim, device=device, dtype=self.w.dtype) - (
                2.0 / denom
            ) * torch.outer(wj, wj)
            q = q @ hj
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.matrix()  # (dim, dim)
        # channel-mixing 1x1 conv:  y[b,o,h,w] = sum_i Q[o,i] x[b,i,h,w]
        return torch.einsum("oi,bihw->bohw", q, x)

    @torch.no_grad()
    def export_conv1x1(self) -> nn.Conv2d:
        """Fold Q into a bias-free 1x1 convolution (the DPU-native form)."""
        q = self.matrix()
        conv = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False)
        conv.weight.copy_(q.reshape(self.dim, self.dim, 1, 1))
        return conv

    def orthogonality_error(self) -> torch.Tensor:
        q = self.matrix()
        return (q @ q.t() - torch.eye(self.dim, device=q.device, dtype=q.dtype)).abs().max()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, m={self.m}"
