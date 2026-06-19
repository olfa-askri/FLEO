"""
End-to-end tests for the FLEO pipeline (requires torch).
Run:  python -m pytest tests/test_fleo.py -v
  or: python tests/test_fleo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.fleo_module import GramSchmidtOrthogonalizer, FLEOModule
from models.loss import FuzzyConfusionLoss, BindingLoss, FLEOTotalLoss
from models.fer_classifier import FLEOClassifier
from utils.metrics import orthogonality_check, margin_check
from utils.confusion_matrix import FACS_PRIOR_7, normalize_confusion


def test_gram_schmidt_orthonormal():
    gs = GramSchmidtOrthogonalizer()
    v = torch.randn(8, 7, 64)
    e = gs(v)
    assert orthogonality_check(e) < 1e-4, "off-diagonal not zero"


def test_proposition_1_margin():
    """||e_i - e_j||^2 == 2 for all i != j."""
    gs = GramSchmidtOrthogonalizer()
    e = gs(torch.randn(8, 7, 64))
    assert abs(margin_check(e) - 2.0) < 1e-3, "fixed margin of 2 violated"


def test_gram_schmidt_differentiable():
    gs = GramSchmidtOrthogonalizer()
    v = torch.randn(4, 5, 32, requires_grad=True)
    e = gs(v)
    e.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


def test_fleo_shape_preserved():
    """FLEO is a drop-in: output shape == input shape."""
    fleo = FLEOModule(in_channels=64, num_emotions=7, subspace_dim=16)
    fleo.train()
    x = torch.randn(2, 64, 20, 20)
    out, z = fleo(x)
    assert out.shape == x.shape
    assert z.shape == (2, 7)


def test_route1_foldout():
    """train_only=True -> GS skipped at eval, graph holds only Conv/SE."""
    fleo = FLEOModule(64, 7, 16, train_only=True).eval()
    x = torch.randn(2, 64, 20, 20)
    out = fleo(x)                                   # no binding logits at eval
    assert out.shape == x.shape


def test_fuzzy_target_rows_sum_to_one():
    C = normalize_confusion(FACS_PRIOR_7)
    loss = FuzzyConfusionLoss(C)
    probs = torch.softmax(torch.randn(16, 7), dim=-1)
    targets = torch.randint(0, 7, (16,))
    T = loss._build_fuzzy_target(probs, targets)
    assert torch.allclose(T.sum(-1), torch.ones(16), atol=1e-5)


def test_total_loss_backward():
    C = normalize_confusion(FACS_PRIOR_7)
    crit = FLEOTotalLoss(C)
    logits = torch.randn(8, 7, requires_grad=True)
    z = torch.randn(8, 7, requires_grad=True)
    targets = torch.randint(0, 7, (8,))
    out = crit(logits, z, targets)
    out["total"].backward()
    assert torch.isfinite(logits.grad).all()


def test_end_to_end_train_step():
    """One optimizer step on synthetic data must reduce the loss."""
    torch.manual_seed(0)
    model = FLEOClassifier(num_emotions=7, subspace_dim=8)
    model.train()
    C = normalize_confusion(FACS_PRIOR_7)
    crit = FLEOTotalLoss(C)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(4, 3, 128, 128)
    y = torch.randint(0, 7, (4,))

    logits, z = model(x)
    l0 = crit(logits, z, y)["total"]
    opt.zero_grad(); l0.backward(); opt.step()

    logits, z = model(x)
    l1 = crit(logits, z, y)["total"]
    assert l1.item() < l0.item(), "loss did not decrease after one step"


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
