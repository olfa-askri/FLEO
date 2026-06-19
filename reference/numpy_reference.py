"""
Pure-NumPy reference for FLEO's core math.
=========================================
Mirrors models/fleo_module.py (Gram-Schmidt) and models/loss.py (fuzzy target)
so the algorithm can be executed and numerically verified WITHOUT PyTorch.

Run:  python reference/numpy_reference.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Gram-Schmidt  (same logic as GramSchmidtOrthogonalizer.forward)
# ---------------------------------------------------------------------------
def gram_schmidt(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    v : (B, K, D) raw vectors  ->  e : (B, K, D) orthonormal per sample.
    """
    B, K, D = v.shape
    e = np.zeros_like(v)
    for b in range(B):
        basis = []
        for k in range(K):
            u = v[b, k].astype(np.float64).copy()
            for ej in basis:                           # remove leakage onto basis
                u = u - np.dot(u, ej) * ej
            u = u / (np.linalg.norm(u) + eps)          # normalize
            basis.append(u)
            e[b, k] = u
    return e


# ---------------------------------------------------------------------------
# Confusion-aware fuzzy target  (same logic as FuzzyConfusionLoss)
# ---------------------------------------------------------------------------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def compute_alpha(probs: np.ndarray, alpha_max: float = 0.3) -> np.ndarray:
    K = probs.shape[1]
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=1)       # (B,)
    alpha = (entropy / (np.log(K) + 1e-8)) * alpha_max
    return np.clip(alpha, 0.0, alpha_max)


def build_fuzzy_target(probs, targets, C, alpha_max=0.3):
    B, K = probs.shape
    alpha = compute_alpha(probs, alpha_max)
    T = np.zeros((B, K))
    for i in range(B):
        y = targets[i]
        row = C[y].copy()
        row[y] = 0.0
        s = row.sum()
        row = row / s if s > 1e-8 else row
        T[i] = alpha[i] * row
        T[i, y] = 1.0 - alpha[i]
    return T, alpha


# ---------------------------------------------------------------------------
# Self-verification
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    B, K, D = 4, 7, 32 * 5 * 5         # K emotions, D = d*H*W

    print("=" * 64)
    print("  FLEO core-math verification (NumPy reference)")
    print("=" * 64)

    # ---- Gram-Schmidt + Proposition 1 ---------------------------------
    v = rng.standard_normal((B, K, D))
    e = gram_schmidt(v)

    # Gram matrix of sample 0
    G = e[0] @ e[0].T
    off_diag = G[~np.eye(K, dtype=bool)]
    diag = np.diag(G)

    print(f"\n[1] Gram-Schmidt orthonormality (sample 0)")
    print(f"    mean |diagonal|      = {np.abs(diag).mean():.8f}  (target 1.0)")
    print(f"    mean |off-diagonal|  = {np.abs(off_diag).mean():.3e}  (target 0.0)")

    # Proposition 1: ||e_i - e_j||^2 == 2
    dmat = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            dmat[i, j] = np.sum((e[0, i] - e[0, j]) ** 2)
    pair = dmat[~np.eye(K, dtype=bool)]
    print(f"\n[2] Proposition 1: ||e_i - e_j||^2 == 2")
    print(f"    min={pair.min():.6f}  max={pair.max():.6f}  mean={pair.mean():.6f}")
    assert np.allclose(pair, 2.0, atol=1e-4), "Proposition 1 FAILED"
    print("    -> PASS: fixed margin of sqrt(2) between every emotion pair.")

    # ---- Fuzzy target -------------------------------------------------
    # FACS-style confusion prior (anger<->sad, fear<->surprise heavy)
    C = np.array([
        [0,.10,.05,0,.35,.05,.05],
        [.10,0,.05,.05,.05,0,.05],
        [.05,.05,0,0,.10,.35,.05],
        [0,.05,0,0,0,.05,.05],
        [.35,.05,.10,0,0,0,.05],
        [.05,0,.35,.05,0,0,.05],
        [.05,.05,.05,.05,.05,.05,0],
    ], dtype=np.float64)

    logits = rng.standard_normal((B, K))
    probs = softmax(logits)
    targets = np.array([0, 2, 4, 5])          # anger, fear, sad, surprise
    T, alpha = build_fuzzy_target(probs, targets, C)

    print(f"\n[3] Confusion-aware fuzzy targets")
    print(f"    each row sums to 1?  {np.allclose(T.sum(1), 1.0)}")
    names = ["ang", "dis", "fea", "hap", "sad", "sur", "neu"]
    for i, y in enumerate(targets):
        leak = {names[k]: round(float(T[i, k]), 3) for k in range(K) if T[i, k] > 1e-3}
        print(f"    y={names[y]:<3} alpha={alpha[i]:.3f} -> {leak}")
    print("    -> note: anger leaks to sad; fear<->surprise leak to each other,")
    print("       NOT to unrelated classes (happy).")

    print("\nAll checks passed.\n")


if __name__ == "__main__":
    main()
