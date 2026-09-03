# Comment 1.4 — Define every symbol immediately below each equation

The manuscript has 11 numbered equations (Sections 3.1.1–3.2, pp. 8–12). Below each
equation add a "where …" line defining **every** symbol at its first use. This is the
concrete list — copy the definitions into the paper under each equation.

---

### Eq. (1) — fuzzy membership vector (Sec. 3.1.1)
`μ = (μ_1, …, μ_K),  μ_c ∈ [0,1],  Σ_c μ_c = 1,  μ ∈ Δ^{K−1}`
> where **μ** is the fuzzy membership (soft-label) vector, **μ_c** its component for
> emotion class c, **K** the number of emotion classes (K = 7), and **Δ^{K−1}** the
> (K−1)-dimensional probability simplex.

### Eq. (2) — additively-smoothed soft label (Sec. 3.1.1)
`μ_c = (n_c + α·π_c) / Σ_j (n_j + α·π_j)`
> where **n_c** is the vote count for class c (n_c = 𝟙[c = y] for single-label data),
> **α > 0** the smoothing strength, **π = (π_1,…,π_K)** the prior distribution over
> classes, and **j** the summation index over the K classes.

### Eq. (3) — fuzzy classification loss (Sec. 3.1.1)
`L_cls = − Σ_c μ_c log p_c`
> where **L_cls** is the classification loss, **p = (p_1,…,p_K)** the predicted class
> posterior, **p_c** its component for class c, and **μ_c** the fuzzy target of Eq. (2).

### Eq. (4) — subspace projection (Sec. 3.1.2)
`X_sub = split(Conv1×1(X)) = {X_1, …, X_K}`
> where **X** is the input feature map, **Conv1×1** the 1×1 projection to K·d channels,
> **split(·)** the partition into K groups, and **X_k** the raw sub-feature of emotion k.

### Eq. (5) — Gram–Schmidt orthogonalization (Sec. 3.1.2)
`Ũ_1 = X_1,  Ũ_k = X_k − Σ_{j<k} (⟨X_k, Ũ_j⟩ / ⟨Ũ_j, Ũ_j⟩) Ũ_j`
> where **Ũ_k** is the k-th orthogonalized sub-feature, **⟨·,·⟩** the inner product,
> **k, j** subspace indices, and the sum runs over the already-built vectors j < k.

### Eq. (6) — FLEO block output (Sec. 3.1.2)
`Y = X + Conv3×3( ‖_k (Ũ_k ⊙ s) )`
> where **Y** is the block output, **X** the residual input, **Conv3×3** the fusion
> convolution, **‖_k** channel-wise concatenation over k, **Ũ_k** the orthogonalized
> sub-feature, **s** the SE fuzzy gate, and **⊙** element-wise (channel) multiplication.

### Eq. (7) — total training loss (Sec. 3.1.3)
`L = L_cls + λ_o L_orth + λ_b L_bind = −Σ_c μ_c log p_c + λ_o ‖Û⊤Û − I‖_F² + λ_b KL(μ ‖ σ(b))`
> where **L** is the total loss, **λ_o, λ_b** the orthogonality and binding weights,
> **L_orth, L_bind** those loss terms, **Û** the stacked orthonormalized bases,
> **I** the identity matrix, **‖·‖_F** the Frobenius norm, **σ** the softmax,
> **b** the binding-head logits, and **KL(·‖·)** the Kullback–Leibler divergence.

### Eq. (8) — full inference graph (Sec. 3.2)
`Y = Conv3×3( GS(Conv1×1(X)) ⊙ s ) + X`
> where **GS(·)** is the Gram–Schmidt operator, **s** the SE gate, **⊙** element-wise
> multiplication, and X, Y, Conv1×1, Conv3×3 as above.

### Eq. (9) — folded (Route-1) inference graph (Sec. 3.2)
`R : Y_fold = Conv3×3( Conv1×1(X) ⊙ s ) + X`
> where **R** denotes the fold-out rewrite and **Y_fold** the output of the streamlined
> graph in which GS is replaced by the identity; other symbols as in Eq. (8).

### Eq. (10) — fold-out accuracy delta (Sec. 3.2)
`Δ_fold = Acc(full graph, Eq. 8) − Acc(folded graph, Eq. 9)`
> where **Δ_fold** is the accuracy drop from folding out GS and **Acc(·)** the top-1
> accuracy of the indicated graph.

### Eq. (11) — quantization accuracy delta (Sec. 3.2)
`Δ_q = Acc_FP32 − Acc_INT8`
> where **Δ_q** is the accuracy drop from INT8 quantization, **Acc_FP32** the
> full-precision accuracy and **Acc_INT8** the quantized accuracy.

---

## What to do in the paper (Author Action for 1.4)

1. Under **each** of Eq. (1)–(11), insert the corresponding "where …" sentence above.
2. Make sure every symbol appears with the **same** notation everywhere (e.g. always
   K for the number of classes, always λ_o / λ_b for the loss weights).
3. Add a one-line **Notation** note or a small symbol table at the start of Section 3 if
   the reviewer prefers a single reference (optional but clean).
