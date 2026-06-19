"""
FLEO Loss Functions:
  1. BindingLoss  – L_bind (cross-entropy to anchor subspace k to emotion k)
  2. FuzzyConfusionLoss – confusion-aware adaptive fuzzy label loss
  3. FLEOTotalLoss – combined wrapper used during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BindingLoss(nn.Module):
    """
    L_bind = -(1/B) sum_i log [ exp(z_{i, y_i}) / sum_m exp(z_{i,m}) ]

    Forces subspace k to specialize in emotion k.

    Args:
        lam : weight (lambda) balancing binding vs. task loss
    """

    def __init__(self, lam: float = 0.1):
        super().__init__()
        self.lam = lam

    def forward(self, bind_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        bind_logits : (B, K, K) – logit from subspace k for sample i
        targets     : (B,)      – ground-truth emotion index y_i
        """
        B, K, _ = bind_logits.shape
        loss = 0.0
        for k in range(K):
            logits_k = bind_logits[:, k, :]              # (B, K)
            loss += F.cross_entropy(logits_k, targets)
        return self.lam * loss / K


class FuzzyConfusionLoss(nn.Module):
    """
    Confusion-aware adaptive fuzzy label loss.

    Instead of one-hot targets, uses a structured fuzzy distribution:
        T_hat_{i,k} =
            (1 - alpha_i)          if k == y_i
            alpha_i * C[y_i, k] / sum_{m!=y_i} C[y_i, m]   otherwise

    alpha_i is computed from predictive entropy H(P_i), clamped to [0, alpha_max].

    The gradient is equivalent to cross-entropy against a stable target
    (Proposition 2), with added value from structured confusion guidance.

    Args:
        confusion_matrix : (K, K) torch.Tensor  – prior confusion probabilities
        alpha_max        : upper bound for smoothing strength
        entropy_temp     : temperature scaling for alpha computation
    """

    def __init__(
        self,
        confusion_matrix: torch.Tensor,
        alpha_max: float = 0.3,
        entropy_temp: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("C", confusion_matrix)
        self.alpha_max = alpha_max
        self.entropy_temp = entropy_temp

    @torch.no_grad()
    def _compute_alpha(self, probs: torch.Tensor) -> torch.Tensor:
        """alpha_i based on predictive entropy H(P_i), stop-gradient applied."""
        # probs: (B, K)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)   # (B,)
        max_entropy = torch.log(torch.tensor(probs.size(1), dtype=probs.dtype))
        alpha = (entropy / (max_entropy + 1e-8)) * self.alpha_max
        return alpha.clamp(0.0, self.alpha_max)

    def _build_fuzzy_target(
        self, probs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Construct T_hat for each sample."""
        B, K = probs.shape
        alpha = self._compute_alpha(probs)                # (B,)  stop-grad

        T = torch.zeros(B, K, device=probs.device, dtype=probs.dtype)
        for i in range(B):
            y = targets[i].item()
            row = self.C[y].clone()
            row[y] = 0.0
            row_sum = row.sum()
            if row_sum > 1e-8:
                row = row / row_sum
            T[i] = alpha[i] * row
            T[i, y] = 1.0 - alpha[i]
        return T

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, K)
        targets : (B,)
        """
        probs = logits.softmax(dim=-1).detach()           # stop-gradient for alpha
        T = self._build_fuzzy_target(probs, targets)      # (B, K)

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(T * log_probs).sum(dim=-1).mean()
        return loss


class FLEOTotalLoss(nn.Module):
    """
    Combined loss:
        L_total = L_fuzzy + lambda * L_bind

    Args:
        confusion_matrix : (K, K) prior confusion
        lam_bind         : weight for binding loss
        alpha_max        : max fuzzy smoothing coefficient
    """

    def __init__(
        self,
        confusion_matrix: torch.Tensor,
        lam_bind: float = 0.1,
        alpha_max: float = 0.3,
    ):
        super().__init__()
        self.fuzzy_loss = FuzzyConfusionLoss(confusion_matrix, alpha_max=alpha_max)
        self.bind_loss = BindingLoss(lam=lam_bind)

    def forward(
        self,
        logits: torch.Tensor,
        bind_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        lf = self.fuzzy_loss(logits, targets)
        lb = self.bind_loss(bind_logits, targets)
        return {"fuzzy": lf, "bind": lb, "total": lf + lb}
