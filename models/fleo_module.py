"""
FLEO: Fuzzy Label and Emotion Orthogonalization Module
Designed to plug into YOLOv12 Neck at levels P3 and P4.

Mathematical guarantee: ||e_i - e_j||^2 = 2 for all i != j (Proposition 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramSchmidtOrthogonalizer(nn.Module):
    """
    Batched differentiable Gram-Schmidt orthonormalization.
    Operates on K subspace vectors of dimension D.

    Input:  V_tilde  shape (B, K, D)
    Output: E        shape (B, K, D)  where <e_i, e_j> = delta_ij
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, v_tilde: torch.Tensor) -> torch.Tensor:
        # v_tilde: (B, K, D)
        B, K, D = v_tilde.shape
        e = []

        u_1 = v_tilde[:, 0, :]                          # (B, D)
        e_1 = u_1 / (u_1.norm(dim=-1, keepdim=True) + self.eps)
        e.append(e_1)

        for k in range(1, K):
            v_k = v_tilde[:, k, :]                       # (B, D)
            u_k = v_k
            for j in range(k):
                u_j = e[j] * (e[j].norm(dim=-1, keepdim=True) + self.eps)
                proj = (
                    (v_k * u_j).sum(dim=-1, keepdim=True)
                    / (u_j.norm(dim=-1, keepdim=True) ** 2 + self.eps)
                ) * u_j
                u_k = u_k - proj
            e_k = u_k / (u_k.norm(dim=-1, keepdim=True) + self.eps)
            e.append(e_k)

        return torch.stack(e, dim=1)                      # (B, K, D)


class SEFuzzyGate(nn.Module):
    """
    Squeeze-and-Excitation gate producing per-channel fuzzy weights.
    Allows partial membership across emotion subspaces.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.fc(x)                                    # (B, C)
        return s.unsqueeze(-1).unsqueeze(-1)              # (B, C, 1, 1)


class BindingHead(nn.Module):
    """
    Auxiliary head used ONLY during training.
    Forces subspace k to predict emotion k  (L_bind).
    Removed at export / inference time.
    """

    def __init__(self, subspace_dim: int, num_emotions: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(subspace_dim, num_emotions)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        # e: (B, K, D)
        logits = self.classifier(e)                       # (B, K, num_emotions)
        return logits


class FLEOModule(nn.Module):
    """
    Full FLEO block.

    Args:
        in_channels  : channels from YOLOv12 Neck (P3 or P4)
        num_emotions : K  (e.g. 7 for RAF-DB / AffectNet)
        subspace_dim : d  per-emotion latent width
        eps          : numerical stability constant (default 1e-8)
        train_only   : if True, GS orthogonalization is skipped at inference
                       (Route 1 – recommended for FPGA/DPU deployment)
    """

    def __init__(
        self,
        in_channels: int,
        num_emotions: int = 7,
        subspace_dim: int = 32,
        eps: float = 1e-8,
        train_only: bool = True,
    ):
        super().__init__()
        self.K = num_emotions
        self.d = subspace_dim
        self.eps = eps
        self.train_only = train_only

        # ── Projection: C → K*d  (Conv 1×1, DPU-compatible)
        self.proj = nn.Conv2d(in_channels, num_emotions * subspace_dim, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(num_emotions * subspace_dim)

        # ── Gram-Schmidt orthogonalizer (used at train time)
        self.gs = GramSchmidtOrthogonalizer(eps=eps)

        # ── Fuzzy gate (SE, DPU-compatible)
        self.gate = SEFuzzyGate(num_emotions * subspace_dim)

        # ── Recombination: K*d → in_channels  (Conv 3×3, DPU-compatible)
        self.recomb = nn.Sequential(
            nn.Conv2d(num_emotions * subspace_dim, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # ── Binding head (training only, removed at export)
        self.binding_head = BindingHead(subspace_dim, num_emotions)

    # ------------------------------------------------------------------
    def _orthogonalize(self, x_proj: torch.Tensor) -> torch.Tensor:
        """Apply Gram-Schmidt in the spatial mean feature space."""
        B, C, H, W = x_proj.shape                        # C = K*d
        # Flatten spatial dims and split into K subspaces
        v = x_proj.view(B, self.K, self.d, H * W)        # (B, K, d, H*W)
        v_tilde = v.mean(dim=-1)                          # (B, K, d)  – spatial mean
        e = self.gs(v_tilde)                              # (B, K, d)

        # Broadcast back to spatial resolution
        e_spatial = e.unsqueeze(-1).expand(-1, -1, -1, H * W)  # (B, K, d, H*W)
        x_ortho = e_spatial.reshape(B, C, H, W)
        return x_ortho, e

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x : (B, in_channels, H, W)

        Returns (during training):
            out        : (B, in_channels, H, W)   – residual-added output
            bind_logits: (B, K, K)                 – for L_bind
        Returns (during inference / train_only=True after export):
            out        : (B, in_channels, H, W)
        """
        # 1. Projection
        x_proj = F.silu(self.proj_bn(self.proj(x)))      # (B, K*d, H, W)

        # 2. Orthogonalization (train-time; skipped if train_only and not training)
        if self.training or not self.train_only:
            x_ortho, e = self._orthogonalize(x_proj)
        else:
            x_ortho = x_proj
            e = None

        # 3. Fuzzy gate  (SE – always active)
        gate_weights = self.gate(x_ortho)                 # (B, K*d, 1, 1)
        x_gated = x_ortho * gate_weights

        # 4. Recombination + residual
        out = self.recomb(x_gated) + x               # residual connection

        # 5. Binding head (training only)
        if self.training and e is not None:
            bind_logits = self.binding_head(e)            # (B, K, K)
            return out, bind_logits

        return out
