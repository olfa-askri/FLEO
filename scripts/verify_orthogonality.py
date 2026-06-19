"""
Quick sanity-check: verifies the Gram-Schmidt guarantee on a random batch.
Uses torch if available, otherwise falls back to the NumPy reference so it
always runs.

Run: python scripts/verify_orthogonality.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _torch_check():
    import torch
    from models.fleo_module import GramSchmidtOrthogonalizer
    from utils.metrics import orthogonality_check, margin_check

    gs = GramSchmidtOrthogonalizer()
    e = gs(torch.randn(8, 7, 64))
    G = torch.bmm(e, e.transpose(1, 2))
    print("[torch] mean diagonal   :", G.diagonal(dim1=1, dim2=2).mean().item())
    print("[torch] mean |off-diag| :", orthogonality_check(e))
    print("[torch] mean ||ei-ej||^2:", margin_check(e), "(target 2.0)")
    assert orthogonality_check(e) < 1e-4
    assert abs(margin_check(e) - 2.0) < 1e-3
    print("\n[torch] OK: Proposition 1 holds.")


def _numpy_check():
    import numpy as np
    from reference.numpy_reference import gram_schmidt

    e = gram_schmidt(np.random.default_rng(0).standard_normal((8, 7, 64)))
    G = e[0] @ e[0].T
    off = G[~np.eye(7, dtype=bool)]
    dist2 = np.array([[np.sum((e[0, i] - e[0, j]) ** 2) for j in range(7)]
                      for i in range(7)])[~np.eye(7, dtype=bool)]
    print("[numpy] mean diagonal   :", np.diag(G).mean())
    print("[numpy] mean |off-diag| :", np.abs(off).mean())
    print("[numpy] mean ||ei-ej||^2:", dist2.mean(), "(target 2.0)")
    assert np.abs(off).mean() < 1e-4
    assert abs(dist2.mean() - 2.0) < 1e-3
    print("\n[numpy] OK: Proposition 1 holds.")


if __name__ == "__main__":
    try:
        import torch  # noqa: F401
        _torch_check()
    except ImportError:
        print("torch not found -> using NumPy reference\n")
        _numpy_check()
