"""
FLEO: Fuzzy Label and Emotion Orthogonalization Module
=======================================================
Drop-in block for the YOLOv12 Neck (inserted at P3 and P4).

Geometric contract (Proposition 1):
    After orthonormalization,  <e_i, e_j> = delta_ij , hence
        ||e_i - e_j||^2 = ||e_i||^2 + ||e_j||^2 - 2<e_i,e_j> = 1 + 1 - 0 = 2
    => every pair of emotions sits at a FIXED margin of sqrt(2), independent of
       how correlated the raw features phi_a, phi_s were.

Notation (matches the design document):
    X        input tensor               (B, C, H, W)
    K        number of emotion subspaces
    d        per-emotion subspace width
    D        flattened dim   = d * H * W
    v_k      raw subspace                channels [(k-1)d : kd]
    v~_k     flattened raw vector        (B, D)
    u_k, e_k orthogonal / orthonormal    (B, D)
    eps      stability constant          1e-8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Differentiable, batched Gram-Schmidt
# ---------------------------------------------------------------------------
class GramSchmidtOrthogonalizer(nn.Module):
    """
    Batched, differentiable Gram-Schmidt (modified / numerically-stable form).

    Input :  V  (B, K, D)   -- K raw vectors per sample
    Output:  E  (B, K, D)   -- orthonormal basis,  <e_i,e_j> = delta_ij

    Because the running basis e_1..e_{k-1} is already orthonormal, the
    projection of v_k onto span(e_1..e_{k-1}) collapses to a sum of simple
    inner products:   proj = sum_j <v_k, e_j> e_j
    (no division by ||u_j||^2 is needed once e_j is unit-norm -> DPU-friendlier).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        B, K, D = v.shape
        e_list = []                                   # holds e_1 .. e_K

        for k in range(K):
            u_k = v[:, k, :]                           # (B, D) start from raw v_k
            # Subtract the projection onto every previously built basis vector.
            for e_j in e_list:
                coef = (u_k * e_j).sum(dim=-1, keepdim=True)   # <v_k, e_j>  (B,1)
                u_k = u_k - coef * e_j                          # remove leakage
            # Normalize to unit length -> e_k.
            norm = u_k.norm(dim=-1, keepdim=True)              # (B,1)
            e_k = u_k / (norm + self.eps)
            e_list.append(e_k)

        return torch.stack(e_list, dim=1)              # (B, K, D)


# ---------------------------------------------------------------------------
# 2. Squeeze-and-Excitation fuzzy gate (DPU-supported, kept at inference)
# ---------------------------------------------------------------------------
class SEFuzzyGate(nn.Module):
    """
    Channel-wise soft gate s in [0,1]^C. Lets a spatial location belong
    *partially* to several emotion subspaces at once (fuzzy membership) -
    ideal for ambiguous expressions that lie between two classes.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)            # squeeze  (B,C,1,1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),                              # excitation -> (0,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        s = self.pool(x).reshape(B, C)                 # (B, C)
        s = self.fc(s)                                 # (B, C)
        return s.reshape(B, C, 1, 1)                   # broadcastable gate


# ---------------------------------------------------------------------------
# 3. Binding head  (training only -> folded out for FPGA)
# ---------------------------------------------------------------------------
class BindingHead(nn.Module):
    """
    Anchors geometry to semantics: subspace k must answer for emotion k.

    Each orthonormal subspace e_k is spatially pooled to a d-vector, then
    scored by its OWN linear scorer to a single scalar z_{i,k}. Collected
    over k this yields z_i in R^K, supervised by L_bind = CE(z_i, y_i).
    (Matches the document's formula where z_{i,k} is "the logit produced by
    subspace k for sample i".)
    """

    def __init__(self, num_emotions: int, subspace_dim: int):
        super().__init__()
        self.K = num_emotions
        self.d = subspace_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        # One independent scorer per subspace:  z_{b,k} = <pooled_{b,k}, W_k> + b_k
        self.weight = nn.Parameter(torch.randn(num_emotions, subspace_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_emotions))

    def forward(self, x_ortho: torch.Tensor) -> torch.Tensor:
        B = x_ortho.shape[0]
        pooled = self.pool(x_ortho).reshape(B, self.K, self.d)   # (B, K, d)
        z = (pooled * self.weight.unsqueeze(0)).sum(-1) + self.bias  # (B, K)
        return z


# ---------------------------------------------------------------------------
# 4. Full FLEO block
# ---------------------------------------------------------------------------
class FLEOModule(nn.Module):
    """
    Pipeline:
        X --conv1x1--> proj --split K--> GramSchmidt --> x_ortho
          --SE gate--> x_gated --conv3x3--> (+ residual X) --> out
        (training) x_ortho --BindingHead--> z  (B, K) for L_bind

    Args:
        in_channels  : channels of the incoming P3/P4 feature map (C)
        num_emotions : K
        subspace_dim : d
        eps          : numerical floor in Gram-Schmidt
        train_only   : Route 1. If True, the GS op is skipped at inference so
                       only Conv/SE (DPU-supported ops) remain in the graph.
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
        self.C = in_channels
        self.K = num_emotions
        self.d = subspace_dim
        self.eps = eps
        self.train_only = train_only
        mid = num_emotions * subspace_dim              # K*d

        # 4.1 Projection C -> K*d  (1x1 conv, fully DPU-supported)
        self.proj = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(mid)

        # 4.2 Geometry
        self.gs = GramSchmidtOrthogonalizer(eps=eps)

        # 4.3 Fuzzy gate
        self.gate = SEFuzzyGate(mid)

        # 4.4 Recombination K*d -> C  (3x3 conv) + residual
        self.recomb = nn.Sequential(
            nn.Conv2d(mid, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        # 4.5 Binding head (train-time only)
        self.binding_head = BindingHead(num_emotions, subspace_dim)

    # ---- geometry helper -------------------------------------------------
    def _orthogonalize(self, x_proj: torch.Tensor) -> torch.Tensor:
        """
        Flatten each of the K subspaces to a single D=d*H*W vector, run
        Gram-Schmidt across the K vectors, reshape back to a feature map.
        """
        B, C, H, W = x_proj.shape                      # C == K*d
        D = self.d * H * W
        v = x_proj.reshape(B, self.K, D)               # (B, K, D)  flatten subspaces
        e = self.gs(v)                                 # (B, K, D)  orthonormal
        x_ortho = e.reshape(B, C, H, W)                # back to (B, K*d, H, W)
        return x_ortho

    # ---- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor):
        # (1) projection into K subspaces
        x_proj = F.silu(self.proj_bn(self.proj(x)))    # (B, K*d, H, W)

        # (2) orthogonalize  (skipped at inference when train_only=True)
        if self.training or not self.train_only:
            x_ortho = self._orthogonalize(x_proj)
        else:
            x_ortho = x_proj                           # Route 1 fold-out

        # (3) fuzzy gating
        x_gated = x_ortho * self.gate(x_ortho)

        # (4) recombine + residual (preserves YOLOv12 training stability)
        out = self.recomb(x_gated) + x

        # (5) binding logits during training
        if self.training:
            z = self.binding_head(x_ortho)             # (B, K)
            return out, z
        return out
