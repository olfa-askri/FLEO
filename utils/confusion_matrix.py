"""
Utilities for building and updating the Confusion Prior matrix C.

C[i, j] = probability that emotion i is mis-classified as emotion j.
Used by FuzzyConfusionLoss to guide label smoothing toward semantically
similar emotions (e.g., anger → sadness) based on FACS Action-Unit overlap.
"""

import torch
import numpy as np


# FACS-based prior confusion (7 basic emotions: anger, disgust, fear,
# happy, sad, surprise, neutral) – AU-overlap initialisation.
# Rows = true class, Cols = confused class.  Diagonal = 0 by convention.
FACS_PRIOR_7 = torch.tensor([
    # ang   dis   fea   hap   sad   sur   neu
    [0.00, 0.10, 0.05, 0.00, 0.35, 0.05, 0.05],  # anger
    [0.10, 0.00, 0.05, 0.05, 0.05, 0.00, 0.05],  # disgust
    [0.05, 0.05, 0.00, 0.00, 0.10, 0.35, 0.05],  # fear
    [0.00, 0.05, 0.00, 0.00, 0.00, 0.05, 0.05],  # happy
    [0.35, 0.05, 0.10, 0.00, 0.00, 0.00, 0.05],  # sad
    [0.05, 0.00, 0.35, 0.05, 0.00, 0.00, 0.05],  # surprise
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00],  # neutral
], dtype=torch.float32)


def normalize_confusion(C: torch.Tensor) -> torch.Tensor:
    """Row-normalize so off-diagonal entries sum to 1 per row."""
    C = C.clone()
    C.fill_diagonal_(0.0)
    row_sums = C.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return C / row_sums


def build_running_confusion(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    prev_C: torch.Tensor = None,
    momentum: float = 0.9,
) -> torch.Tensor:
    """
    Update confusion matrix from a batch using exponential moving average.

    Args:
        preds       : (B,)  predicted class indices
        targets     : (B,)  ground-truth class indices
        num_classes : K
        prev_C      : previous matrix (K, K), None = start from zeros
        momentum    : EMA factor for running update
    """
    batch_C = torch.zeros(num_classes, num_classes, dtype=torch.float32)
    for t, p in zip(targets.cpu(), preds.cpu()):
        if t != p:
            batch_C[t, p] += 1.0

    batch_C = normalize_confusion(batch_C)

    if prev_C is None:
        return batch_C

    return momentum * prev_C + (1.0 - momentum) * batch_C
