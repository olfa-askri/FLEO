"""
Evaluation metrics for FER (Facial Emotion Recognition).
Computes Accuracy and macro-F1 consistent with RAF-DB / AffectNet protocols.
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Args:
        preds   : (N,) predicted class indices
        targets : (N,) ground-truth class indices
    Returns:
        dict with 'accuracy' and 'macro_f1'
    """
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()
    acc = accuracy_score(t, p) * 100.0
    f1 = f1_score(t, p, average="macro") * 100.0
    return {"accuracy": acc, "macro_f1": f1}


def orthogonality_check(e: torch.Tensor, eps: float = 1e-3) -> float:
    """
    Verify Gram-Schmidt guarantee: <e_i, e_j> ~ 0 for i != j.

    Args:
        e : (B, K, D) orthonormalized subspace vectors
    Returns:
        mean absolute off-diagonal inner product (should be << eps)
    """
    B, K, D = e.shape
    G = torch.bmm(e, e.transpose(1, 2))                 # (B, K, K)
    mask = ~torch.eye(K, dtype=torch.bool, device=e.device).unsqueeze(0)
    off_diag = G[mask.expand(B, -1, -1)]
    return off_diag.abs().mean().item()
