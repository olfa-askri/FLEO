# FLEO — Function-by-Function Study

This document walks through **every function** in the codebase: what it does, the
math behind it, tensor shapes, and the design reasoning. Read it top-to-bottom to
understand the whole system.

---

## 0. The big picture (data flow)

```
image (B,3,H,W)
   │
   ▼ backbone
P3 (B,C3,H/8 ,W/8)   P4 (B,C4,H/16,W/16)
   │                   │
   ▼ FLEOModule        ▼ FLEOModule          ◄── geometric bottleneck
P3'(B,C3,·,·)        P4'(B,C4,·,·)
   │                   │
   ▼ head ─────────────┘
emotion logits (B,K)   +   binding logits z (B,K)   (training only)
   │                       │
   ▼ FuzzyConfusionLoss    ▼ BindingLoss
        └──── L_total = L_fuzzy + λ·L_bind ────┘
```

FLEO converts an emotion-**classification** problem into a **geometry** problem:
force each emotion onto its own orthogonal axis so confusable pairs (anger/sad,
fear/surprise) can no longer collapse onto each other.

---

## 1. `models/fleo_module.py`

### 1.1 `GramSchmidtOrthogonalizer.forward(v) -> e`
**Shapes:** `v (B,K,D) → e (B,K,D)`, with `D = d·H·W`.

**What it does.** Turns `K` raw, possibly-correlated subspace vectors into an
**orthonormal** set, per sample, differentiably.

**Math (per sample b):**
```
e_1 = v_1 / ‖v_1‖
for k = 2..K:
    u_k = v_k − Σ_{j<k} ⟨v_k, e_j⟩ e_j      # remove components along earlier axes
    e_k = u_k / (‖u_k‖ + ε)
```
Because the running basis `e_j` is already **unit-norm**, the projection
simplifies from `⟨v_k,u_j⟩/‖u_j‖² · u_j` to just `⟨v_k,e_j⟩ · e_j`. This removes
a division — friendlier for later INT8 / DPU reasoning, and numerically cleaner.

**Why it matters.** Guarantees `⟨e_i,e_j⟩ = δ_ij`. Therefore the pairwise
distance is *constant*:
```
‖e_i − e_j‖² = ‖e_i‖² + ‖e_j‖² − 2⟨e_i,e_j⟩ = 1 + 1 − 0 = 2     (Proposition 1)
```
So every emotion pair has a **fixed margin √2**, regardless of how correlated the
raw features were. The decision boundary stops being a tiny, noise-sensitive
residual `φ_a − φ_s`.

**Differentiable?** Yes — only `+,−,×,÷,√` over tensors, so gradients flow back
into the projection conv and the backbone. `ε` (1e-8) keeps `÷` and `√` stable
when a vector is near-zero.

> 🐞 **Bug fixed vs. first draft.** The original rebuilt `u_j` from `e_j` via
> `e_j·(‖e_j‖+ε)` (a no-op that re-introduced a needless division) — replaced by
> the clean orthonormal-projection form above.

### 1.2 `SEFuzzyGate.forward(x) -> gate`
**Shapes:** `x (B,C,H,W) → (B,C,1,1)` with `C = K·d`.

Squeeze-and-Excitation: global-average-pool → `Linear → ReLU → Linear → Sigmoid`.
Produces a per-channel weight `s ∈ (0,1)`. Multiplying `x·s` lets a location hold
**partial membership** in several emotion subspaces at once — exactly what you
want for ambiguous faces that sit *between* two emotions. SE is fully DPU-
supported, so it **stays in the inference graph** (unlike Gram-Schmidt).

### 1.3 `BindingHead.forward(x_ortho) -> z`
**Shapes:** `x_ortho (B,K·d,H,W) → z (B,K)`.

Pools each subspace to a `d`-vector, then each subspace `k` has its **own** scorer
`(W_k, b_k)`:
```
z[b,k] = ⟨pooled[b,k], W_k⟩ + b_k
```
`z` is a length-`K` logit vector where entry `k` is "how strongly subspace `k`
fired". This is the `z_{i,k}` from the design doc, consumed by `L_bind`. It is a
**training-only** head — folded out at export.

> 🐞 **Bug fixed.** First draft produced `(B,K,K)` and applied CE per-subspace,
> which didn't match the doc's scalar-per-subspace `z_{i,k}`. Now `(B,K)`.

### 1.4 `FLEOModule.__init__`
Wires the block: `proj` (Conv1×1 `C→K·d`) + BN, the `gs` orthogonalizer, the
`gate`, `recomb` (Conv3×3 `K·d→C`) + BN + SiLU, and the `binding_head`.
`train_only=True` selects **Route 1** (FPGA): skip GS at inference.

### 1.5 `FLEOModule._orthogonalize(x_proj) -> x_ortho`
**Shapes:** `x_proj (B,K·d,H,W) → (B,K,D) → GS → (B,K·d,H,W)`, `D=d·H·W`.

Flattens **each** subspace (its `d` channels × all `H·W` pixels) into one
`D`-dim vector, orthogonalizes the `K` of them, reshapes back. This matches the
doc's `D = d·H·W` and the `O(B·K²·D)` cost.

> 🐞 **Bug fixed.** First draft took the spatial **mean** then broadcast one
> vector to every pixel — that destroyed all spatial information. Now the full
> `d·H·W` content is orthogonalized and spatial structure is preserved.

### 1.6 `FLEOModule.forward(x)`
**Shape contract:** `x (B,C,H,W) → out (B,C,H,W)` (drop-in: in == out).

Steps: (1) project+split → (2) orthogonalize *(skipped at eval if `train_only`)*
→ (3) fuzzy gate → (4) `recomb + x` residual (keeps YOLOv12 training stable) →
(5) training also returns `z` for `L_bind`. The residual + BN in `recomb` also
re-scale the tiny orthonormal values back to a useful range.

---

## 2. `models/loss.py`

### 2.1 `BindingLoss.forward(z, targets)`
`λ · CE(z, y)`. Pushing `softmax(z)` toward `y` makes subspace `y` score highest,
so **subspace k ↔ emotion k**. Without this, orthogonal axes would be
geometrically separated but **semantically arbitrary**.

### 2.2 `FuzzyConfusionLoss._compute_alpha(probs) -> α`
`α_i = α_max · H(P_i)/log K`, clamped to `[0, α_max]`, where
`H(P_i) = −Σ_k p_k log p_k` is predictive entropy. Confident sample → low
entropy → small `α` (almost one-hot); ambiguous sample → high entropy → larger
`α` (more smoothing). Wrapped in `@torch.no_grad()` — this is the
**stop-gradient on α** the doc requires, so the smoothing strength is treated as
a constant and weights don't diverge at high entropy (Proposition 2).

### 2.3 `FuzzyConfusionLoss._build_fuzzy_target(probs, targets) -> T`
Builds the structured soft label:
```
T[i,k] = 1 − α_i                         if k = y_i
       = α_i · Cnorm[y_i,k]              otherwise   (Cnorm = off-diag-normalized C)
```
Each row sums to 1 (verified in tests). The leaked mass goes **only** to
confusable classes: an *anger* sample leaks to *sad*, a *fear* sample to
*surprise* — never to *happy*. Fully vectorized (gather + masked fill).

> 🐞 **Improved.** First draft used a Python per-sample loop; now batched.

### 2.4 `FuzzyConfusionLoss.forward(logits, targets)`
`−Σ_k T[i,k]·log softmax(logits)[i,k]`, mean over batch (soft-label
cross-entropy = `KL(T‖p)` up to a constant). `probs` is `.detach()`ed before
building `T` (again, α is constant).

### 2.5 `FLEOTotalLoss.forward(logits, z, targets)`
Returns `{fuzzy, bind, total}` with `total = L_fuzzy + λ·L_bind`. `logits` come
from the detector head; `z` from the binding head.

---

## 3. `utils/confusion_matrix.py`

### 3.1 `normalize_confusion(C)`
Zeros the diagonal, row-normalizes the off-diagonal so each row is a valid
"leak" distribution. Diagonal is dropped because the *true* class mass is handled
separately by `1−α`.

### 3.2 `build_running_confusion(preds, targets, K, prev_C, momentum)`
Counts this batch's **mistakes** into a `K×K` matrix, normalizes, then EMA-blends
with the previous estimate: `C ← m·C_prev + (1−m)·C_batch`. This addresses
**Risk #2** (a weak prior would degrade fuzzy loss into plain label smoothing):
the prior is continuously refreshed from the model's *real* confusions.
`FACS_PRIOR_7` seeds it from FACS Action-Unit overlap (anger↔sad share AU4,
fear↔surprise share AU1+2/AU5).

---

## 4. `utils/metrics.py`

- **`compute_metrics`** → accuracy + macro-F1 (RAF-DB/AffectNet protocol). Uses
  scikit-learn if present, else a pure-torch fallback so it runs anywhere.
- **`orthogonality_check(e)`** → mean |off-diagonal| of the Gram matrix
  `E·Eᵀ`; should be ≈ 0. The runtime proof that GS worked.
- **`margin_check(e)`** → mean `‖e_i−e_j‖² = 2−2⟨e_i,e_j⟩`; should be ≈ 2
  (Proposition 1).

---

## 5. Models that host FLEO

### 5.1 `models/fer_classifier.py` — runnable stand-in
- **`TinyBackbone.forward`** → returns `P3` (stride 8) and `P4` (stride 16).
- **`TinyHead.forward`** → pools+concats P3,P4 → `Linear → (B,K)` logits.
- **`FLEOClassifier.forward`** → backbone → FLEO@P3 + FLEO@P4 → head. Training
  returns `(logits, z)` with `z = (z3+z4)/2`; eval returns `logits`. Lets you
  train/test/export the *whole* pipeline without the heavy ultralytics dep.

### 5.2 `models/yolov12_fleo.py` — real integration
- **`YOLOv12FLEONeck.forward`** → runs the stock neck, then applies FLEO at P3
  and P4, leaving **P5 untouched** (large-object context). Returns the binding
  logits per level during training.
- **`export_inference_graph(model)`** → sets `eval()` + `train_only=True` so the
  GS ops disappear (Route 1) and only DPU-supported ops remain.

---

## 6. `train/trainer.py`

- **`_train_epoch`** → forward → `FLEOTotalLoss` → backward → grad-clip(10) →
  step → cosine LR. After the epoch, refreshes the confusion prior via
  `build_running_confusion`.
- **`_val_epoch`** → eval path (GS folded out), computes accuracy + macro-F1.
  Cross-dataset protocol: train RAF-DB, validate AffectNet.
- **`run`** → loops epochs, checkpoints the best macro-F1.

---

## 7. `deploy/export_fpga.py` (Route 1)

- **`prepare_model_for_export`** → load checkpoint, `export_inference_graph`
  (drop GS).
- **`quantize_model`** → Vitis AI `torch_quantizer` calib→test → INT8, then
  `dump_xmodel`. Calibrate on ~100–200 RAF-DB images.
- **`print_compile_instructions`** → `vai_c_xir` for **DPU B4096 / ZCU104** →
  `yolov12_fleo.xmodel`.
- **`PYNQ_INFERENCE_TEMPLATE`** → on-board Python (vitis-ai-library) reading the
  camera and drawing emotion labels.

**Why Route 1?** The DPU lacks the `÷`/`√` units Gram-Schmidt needs. We pay that
cost only at **train time** (GS shapes the conv weights to *already* produce
near-orthogonal features); at inference we keep just Conv/SE — full INT8
throughput, with QAT fine-tuning to recover any drop (Risk #3).

---

## 8. How to verify (runs now, no torch needed)

```bash
python reference/numpy_reference.py        # proves Proposition 1 + fuzzy targets
python scripts/verify_orthogonality.py     # torch if present, else numpy
python tests/test_fleo.py                  # full torch test-suite (needs torch)
```
