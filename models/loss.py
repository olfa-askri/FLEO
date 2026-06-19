"""
FLEO Loss Functions
===================
  1. BindingLoss        -- L_bind : anchors subspace k to emotion k
  2. FuzzyConfusionLoss -- confusion-aware adaptive fuzzy-label loss
  3. FLEOTotalLoss      -- L_total = L_fuzzy + lambda * L_bind

Proposition 2: the gradient of the fuzzy loss equals the gradient of
cross-entropy against a *fixed* target. The added value is the STRUCTURE of
that target (confusion-guided), not a new formula. Hence alpha is treated as a
constant (stop-gradient) when building the target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Binding loss
# ---------------------------------------------------------------------------
class BindingLoss(nn.Module):
    """
    z  : (B, K)  -- z[i,k] is the scalar logit produced by subspace k.
    y  : (B,)    -- ground-truth emotion index.

        L_bind = lambda * CE(z, y)
               = -lambda * (1/B) sum_i log softmax(z_i)[y_i]

    Pushing softmax(z_i) toward y_i forces subspace y_i to score highest,
    i.e. it binds "subspace k <-> emotion k".
    """

    def __init__(self, lam: float = 0.1):
        super().__init__()
        self.lam = lam

    def forward(self, z: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.lam * F.cross_entropy(z, targets)


# ---------------------------------------------------------------------------
# 2. Confusion-aware adaptive fuzzy loss
# ---------------------------------------------------------------------------
class FuzzyConfusionLoss(nn.Module):
    """
    Builds a structured soft target T_hat and minimizes KL(T_hat || softmax(logits))
    (implemented as cross-entropy with soft labels):

        T_hat[i, k] = 1 - alpha_i                      if k == y_i
                    = alpha_i * Cnorm[y_i, k]          otherwise

    where Cnorm[y_i, :] is the confusion row of the true class, re-normalized
    over the off-diagonal so that sum_k T_hat[i,k] = 1.

    alpha_i (smoothing strength) adapts to per-sample predictive entropy:
        alpha_i = alpha_max * H(P_i) / log K        (clamped to [0, alpha_max])
    Confident samples -> small alpha (near one-hot); ambiguous samples ->
    larger alpha (more probability leaks toward confusable classes).
    """

    def __init__(
        self,
        confusion_matrix: torch.Tensor,
        alpha_max: float = 0.3,
    ):
        super().__init__()
        self.register_buffer("C", confusion_matrix.clone())
        self.alpha_max = alpha_max

    @torch.no_grad()                                   # stop-gradient (Prop. 2)
    def _compute_alpha(self, probs: torch.Tensor) -> torch.Tensor:
        K = probs.size(1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)        # (B,)
        max_entropy = torch.log(torch.tensor(float(K), device=probs.device))
        alpha = (entropy / (max_entropy + 1e-8)) * self.alpha_max
        return alpha.clamp(0.0, self.alpha_max)                      # (B,)

    @torch.no_grad()
    def _build_fuzzy_target(self, probs, targets):
        B, K = probs.shape
        idx = torch.arange(B, device=probs.device)
        alpha = self._compute_alpha(probs)                          # (B,)

        rows = self.C[targets].clone()                              # (B, K)
        rows[idx, targets] = 0.0                                    # drop diagonal
        rows = rows / rows.sum(dim=1, keepdim=True).clamp(min=1e-8) # renormalize

        T = alpha.unsqueeze(1) * rows                               # off-diagonal mass
        T[idx, targets] = 1.0 - alpha                              # true-class mass
        return T

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=-1).detach()        # detach: alpha is constant
        T = self._build_fuzzy_target(probs, targets)   # (B, K) soft labels
        log_probs = F.log_softmax(logits, dim=-1)
        return -(T * log_probs).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# 3. Combined loss
# ---------------------------------------------------------------------------
class FLEOTotalLoss(nn.Module):
    def __init__(self, confusion_matrix, lam_bind=0.1, alpha_max=0.3):
        super().__init__()
        self.fuzzy_loss = FuzzyConfusionLoss(confusion_matrix, alpha_max=alpha_max)
        self.bind_loss = BindingLoss(lam=lam_bind)

    def forward(self, logits, z, targets) -> dict:
        lf = self.fuzzy_loss(logits, targets)
        lb = self.bind_loss(z, targets)
        return {"fuzzy": lf, "bind": lb, "total": lf + lb}
