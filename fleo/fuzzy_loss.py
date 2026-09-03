"""Fuzzy (soft-label) classification target for the FLEO detector — Eq. (2).

The benchmarks are single-label, so the annotator vote count degenerates to
``n_c = 1[c = y]`` and Eq. (2) reduces to additively-smoothed soft targets

    mu_c = ( 1[c = y] + alpha * pi_c ) / (1 + alpha),

generated algorithmically from the ground-truth label, the smoothing strength
``alpha`` and the prior ``pi``.  We realise it by smoothing the task-aligned
assigner's ``target_scores`` toward ``pi`` with strength ``alpha``:

    target_scores <- (1 - alpha) * target_scores + alpha * pi * weight,

where ``weight`` is the per-anchor alignment metric already carried by
``target_scores`` (so only matched anchors are supervised, unchanged from the
baseline).  ``alpha = 0`` recovers the standard hard-label detection loss
exactly, so this is a safe, opt-in change.
"""
from __future__ import annotations

import torch
from ultralytics.utils.loss import v8DetectionLoss


class _FuzzyAssigner:
    """Wraps a TaskAlignedAssigner and smooths its class targets toward ``pi``."""

    def __init__(self, base, alpha: float, pi: torch.Tensor) -> None:
        self.base = base
        self.alpha = float(alpha)
        self.pi = pi  # (nc,), sums to 1

    def __call__(self, *args, **kwargs):
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.base(
            *args, **kwargs
        )
        # target_scores: (B, A, nc) == one_hot(class) * alignment_metric on positives.
        weight = target_scores.sum(-1, keepdim=True)  # per-anchor alignment weight
        pi = self.pi.to(target_scores.device, target_scores.dtype).view(1, 1, -1)
        target_scores = (1.0 - self.alpha) * target_scores + self.alpha * pi * weight
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx


class FuzzyDetectionLoss(v8DetectionLoss):
    """v8 detection loss whose classification target is the fuzzy vector mu (Eq. 2)."""

    def __init__(self, model, alpha: float = 0.1, pi=None) -> None:
        super().__init__(model)
        nc = self.nc
        if pi is None:
            pi = torch.full((nc,), 1.0 / nc)  # uniform prior
        else:
            pi = torch.as_tensor(pi, dtype=torch.float32)
            pi = pi / pi.sum().clamp_min(1e-8)
        assert pi.numel() == nc, f"prior pi must have {nc} entries, got {pi.numel()}"
        self.assigner = _FuzzyAssigner(self.assigner, alpha, pi)
