"""
Quick sanity-check script: verifies Gram-Schmidt guarantee on a random batch.
Run: python scripts/verify_orthogonality.py
"""

import sys, torch
sys.path.insert(0, '.')
from models.fleo_module import GramSchmidtOrthogonalizer
from utils.metrics import orthogonality_check

def main():
    gs = GramSchmidtOrthogonalizer()
    B, K, d = 8, 7, 64

    v = torch.randn(B, K, d)
    e = gs(v)

    score = orthogonality_check(e)
    Gram = torch.bmm(e, e.transpose(1, 2))              # (B, K, K)
    mean_diag = Gram.diagonal(dim1=1, dim2=2).mean().item()

    print(f"Mean diagonal  (should be ~1.0): {mean_diag:.6f}")
    print(f"Mean |off-diag| (should be ~0.0): {score:.6f}")

    assert abs(mean_diag - 1.0) < 1e-4, "Diagonal not unity!"
    assert score < 1e-5, "Off-diagonal not zero!"
    print("\n✓  Proposition 1 holds: ||e_i - e_j||^2 = 2  for all i != j")

if __name__ == "__main__":
    main()
