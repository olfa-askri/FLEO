"""
Evaluation metrics for FER (Facial Emotion Recognition).
Computes Accuracy and macro-F1 (RAF-DB / AffectNet protocol).

Falls back to a pure-torch implementation when scikit-learn is unavailable,
so the codebase runs in minimal environments.
"""

import torch

try:
    from sklearn.metrics import f1_score, accuracy_score
    _HAS_SKLEARN = True
except ImportError:                                    # pragma: no cover
    _HAS_SKLEARN = False


def _macro_f1_torch(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Pure-torch macro-F1 (unweighted mean of per-class F1)."""
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else torch.tensor(0.0))
    return float(torch.stack(f1s).mean()) * 100.0


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Args:
        preds   : (N,) predicted class indices
        targets : (N,) ground-truth class indices
    Returns:
        {'accuracy': %, 'macro_f1': %}
    """
    if _HAS_SKLEARN:
        p, t = preds.cpu().numpy(), targets.cpu().numpy()
        return {
            "accuracy": accuracy_score(t, p) * 100.0,
            "macro_f1": f1_score(t, p, average="macro") * 100.0,
        }
    acc = (preds == targets).float().mean().item() * 100.0
    num_classes = int(max(preds.max(), targets.max()).item()) + 1
    return {"accuracy": acc, "macro_f1": _macro_f1_torch(preds, targets, num_classes)}


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Return the K x K confusion matrix (rows = true, cols = predicted)."""
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.view(-1).cpu(), preds.view(-1).cpu()):
        cm[t.long(), p.long()] += 1
    return cm


def full_report(preds: torch.Tensor, targets: torch.Tensor, num_classes: int,
                class_names=None) -> dict:
    """Accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix.

    This is what feeds Table 2 (acc, macro-F1) and the per-class /
    confusion-matrix analysis (anger/sad, fear/surprise) in the paper.
    """
    cm = confusion_matrix(preds, targets, num_classes).float()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(tp))

    names = class_names or [str(i) for i in range(num_classes)]
    per_class = {
        names[c]: {
            "precision": float(precision[c]) * 100.0,
            "recall": float(recall[c]) * 100.0,
            "f1": float(f1[c]) * 100.0,
            "support": int(cm[c].sum()),
        }
        for c in range(num_classes)
    }
    return {
        "accuracy": float(tp.sum() / cm.sum()) * 100.0,
        "macro_f1": float(f1.mean()) * 100.0,
        "per_class": per_class,
        "confusion_matrix": cm.long().tolist(),
        "class_names": names,
    }


def orthogonality_check(e: torch.Tensor, eps: float = 1e-3) -> float:
    """
    Verify the Gram-Schmidt guarantee  <e_i, e_j> ~ 0 for i != j.

    Args:
        e : (B, K, D) orthonormalized subspace vectors
    Returns:
        mean absolute off-diagonal inner product (should be << eps)
    """
    B, K, D = e.shape
    G = torch.bmm(e, e.transpose(1, 2))                 # (B, K, K) Gram matrices
    mask = ~torch.eye(K, dtype=torch.bool, device=e.device).unsqueeze(0)
    off_diag = G[mask.expand(B, -1, -1)]
    return off_diag.abs().mean().item()


def margin_check(e: torch.Tensor) -> float:
    """
    Verify Proposition 1: ||e_i - e_j||^2 == 2 for all i != j.
    Returns the mean squared pairwise distance over off-diagonal pairs.
    """
    B, K, D = e.shape
    # ||e_i - e_j||^2 = 2 - 2<e_i,e_j> for unit vectors
    G = torch.bmm(e, e.transpose(1, 2))                 # (B, K, K)
    dist2 = 2.0 - 2.0 * G
    mask = ~torch.eye(K, dtype=torch.bool, device=e.device).unsqueeze(0)
    return dist2[mask.expand(B, -1, -1)].mean().item()
