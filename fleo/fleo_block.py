"""The FLEO neck block and its three deployment routes.

FLEO (Facial-emotion Latent Emotion Orthogonalization) is inserted at the P3 and
P4 neck levels of the YOLOv12-S detector.  Per insertion site it performs
(Section III-C, Eq. (4)):

    Y = Conv3x3( ORTHO(Conv1x1(X)) (*) s ) + X ,   s = sigma(Conv1x1(AvgPool(X')))

where X' = Conv1x1(X).  ``ORTHO`` is the piece that determines the deployment
route:

* ``mode="full"``  -> per-sample Gram-Schmidt  (the trained graph; PS fallback if
  exported literally -- this is also the Route 3 operator).
* ``mode="folded"`` -> identity  (Route 1, the fold-out rewrite R of Eq. (5)).
* ``mode="householder"`` -> constant-folded Householder product (Route 2).

The block *always preserves its channel count* (residual add), so it is a drop-in
neck module: ``c_out == c_in``.  Switching ``mode`` never touches a learned
weight, which is exactly what makes the fold-out a pure graph rewrite.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .orthogonal import GramSchmidt, HouseholderOrtho

MODES = ("full", "folded", "householder")


def _autopad(k: int) -> int:
    return k // 2


class ConvBNAct(nn.Module):
    """Conv-BN-SiLU, the DPU-native convolution primitive used throughout."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class BindingHead(nn.Module):
    """Training-only per-subspace binding head (Section III-C).

    Produces one logit per emotion subspace from the orthogonalized activations;
    its output feeds an auxiliary binding loss during training and is *absent from
    the inference graph* (never called when ``self.training`` is False and the
    caller drops it).  It exists so the orthogonality constraint has a supervised
    target to shape the upstream weights around.
    """

    def __init__(self, k: int, d: int, num_emotions: int = 7) -> None:
        super().__init__()
        self.k = k
        self.d = d
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(d, num_emotions)

    def forward(self, ortho: torch.Tensor) -> torch.Tensor:
        b, c, h, w = ortho.shape
        # (B, K, d, H, W) -> per-subspace GAP -> (B, K, d) -> (B, K, num_emotions)
        s = ortho.view(b, self.k, self.d, h, w).mean(dim=(3, 4))
        return self.fc(s)  # (B, K, num_emotions)


class FLEO(nn.Module):
    """Facial-emotion orthogonalization neck block.

    Args:
        c1: input (== output) channel count.
        k: number of emotion subspaces (``K``, default 7).
        d: per-emotion subspace width.  ``mid = K*d`` is the projection width.
        num_emotions: classes for the training-only binding head.
        mode: one of :data:`MODES`.
    """

    def __init__(
        self,
        c1: int,
        k: int = 7,
        d: int = 8,
        num_emotions: int = 7,
        mode: str = "full",
    ) -> None:
        super().__init__()
        assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
        self.c1 = c1
        self.k = k
        self.d = d
        self.mid = k * d
        self.mode = mode

        # 1x1 projection to K*d channels (the tensor that gets orthogonalized).
        self.proj = ConvBNAct(c1, self.mid, k=1)

        # SE gate branch: X' = Conv1x1(X); s = sigma(Conv1x1(AvgPool(X'))).
        self.gate_in = ConvBNAct(c1, self.mid, k=1)
        self.gate_pool = nn.AdaptiveAvgPool2d(1)
        self.gate_fc = nn.Conv2d(self.mid, self.mid, kernel_size=1, bias=True)

        # Orthogonalization operators (route-dependent; both are cheap to hold).
        self.gs = GramSchmidt(k)
        self.householder = HouseholderOrtho(self.mid)
        # Route 2 export target (populated by fold_householder()).
        self.householder_conv: nn.Conv2d | None = None

        # 3x3 fusion back to c1, then residual.
        self.fuse = ConvBNAct(self.mid, c1, k=3)

        # Training-only auxiliary head.
        self.binding = BindingHead(k, d, num_emotions)
        # Transient training state read by the auxiliary loss; always None outside
        # a training forward so the module pickles cleanly at checkpoint time.
        self._proj_out: torch.Tensor | None = None
        self._binding: torch.Tensor | None = None

    # -- orthogonalization dispatch --------------------------------------
    def _ortho(self, p: torch.Tensor) -> torch.Tensor:
        if self.mode == "full":
            return self.gs(p)
        if self.mode == "folded":
            return p  # identity -- the R rewrite of Eq. (5)
        if self.mode == "householder":
            if self.householder_conv is not None:
                return self.householder_conv(p)  # constant-folded 1x1
            return self.householder(p)
        raise RuntimeError(self.mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.proj(x)
        o = self._ortho(p)

        # SE gate.
        xg = self.gate_in(x)
        s = torch.sigmoid(self.gate_fc(self.gate_pool(xg)))  # (B, mid, 1, 1)
        gated = o * s

        # Training-only signals for the auxiliary losses, stashed on the module
        # (plain attrs, not hooks/closures, so the model stays picklable).  Cleared
        # by the loss and never set outside a training forward -> None at export.
        if self.training:
            self._proj_out = p
            self._binding = self.binding(o)
        else:
            self._proj_out = None
            self._binding = None

        return self.fuse(gated) + x

    # -- route rewrites (export time) ------------------------------------
    def fold_out(self) -> "FLEO":
        """Route 1: replace ORTHO with identity in place. Weights untouched."""
        self.mode = "folded"
        self._proj_out = None
        self._binding = None
        return self

    @torch.no_grad()
    def fold_householder(self) -> "FLEO":
        """Route 2: materialise Q as a fixed 1x1 conv (constant folding)."""
        self.mode = "householder"
        self.householder_conv = self.householder.export_conv1x1().to(
            self.proj.conv.weight.device
        )
        return self

    def set_full(self) -> "FLEO":
        """Route 3 / reference: keep the exact Gram-Schmidt operator."""
        self.mode = "full"
        return self

    def extra_repr(self) -> str:
        return f"c1={self.c1}, K={self.k}, d={self.d}, mid={self.mid}, mode={self.mode}"
