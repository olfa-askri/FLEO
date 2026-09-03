# Response to Reviewers — YOLOv8-FLEO (MDPI Algorithms)

Working document. Each point: the reviewer's ask, our status, and the paste-ready
reply. **[ACTION]** marks work that must be done before the reply is truthful.

Status legend: ✅ ready · 🟠 needs a GPU run · 🔴 needs a code fix + rerun · 🎨 figure work

---

## Reviewer 1

### R1-1 — Justify subspace width d = 8   🟠
Hardware-cost half is computed and final (see `docs/reviewer_response_d_ablation.md`):
d=4/8/16/32 → 0.50× / 1.00× / 2.05× / 4.30× FLEO footprint. Accuracy half needs the
four RAF-DB runs (`notebooks/ablation_d.ipynb`). Reply paragraph is drafted there.

### R1-2 / R1-3 (and R3-1) — How are the fuzzy labels built? α and π?   🔴

**Reviewer:** RAF-DB/FER2013 are single-label; where do the annotator votes n_c in
Eq. (2) come from — multi-annotator metadata, or algorithmic soft labels? Justify α
and the prior π.

**Honest status.** The benchmarks are single-label; there is **no per-image vote
metadata**. In the released training path the classification term currently uses the
standard (hard-label) detector loss plus the orthogonality and binding auxiliaries —
i.e. **Eq. (2) is described but not yet exercised in the runs**. This must be fixed
before the reply below is true.

**[ACTION] Make the code match the paper:** enable additively-smoothed soft targets
during training (e.g. ultralytics `label_smoothing = α`, optionally with a
class-frequency / confusion prior π) and re-run. This folds into the retraining
already required by R1-1 and R3-4.

**Reply (valid once the above is done):**
> The benchmarks are single-label and we use **no** multi-annotator metadata. Eq. (2)
> is the *general* membership definition, where n_c is the vote count for class c. For
> a single-label corpus n_c = 𝟙[c = y], so Eq. (2) reduces to additively-smoothed soft
> targets **μ_c = (𝟙[c=y] + α·π_c) / (1 + α)** — generated **algorithmically and
> deterministically** from the ground-truth label, the smoothing strength α, and the
> prior π, with no annotator votes required. The construction is a few lines and is
> released as part of the training script, so the fuzzy targets are fully reproducible.
> We set **α = [FILL]** and **π = [FILL, e.g. the empirical class-frequency prior]**,
> so probability leaks toward frequently-confused neighbours while confident,
> well-represented classes stay near one-hot. We have clarified this in Sec. 3.1.1 and
> corrected the wording so Eq. (2) is presented as the general form that degenerates to
> smoothed labels on single-label data.

### R1-5 — English + reference formatting   ✅ (copy-edit pass, do last)

---

## Reviewer 2  (mostly cosmetic — do in one editing pass)

- **Title ≤ 2 lines** ✅ — shorten to e.g. *"DPU-Native YOLOv8-FLEO: Fold-Out
  Orthogonalization for Real-Time FPGA Facial-Expression Recognition"*.
- **Delete** the "remainder of this paper …" paragraph (lines 100-103) ✅.
- **Unify terminology** → pick one, "YOLOv8-FLEO framework", everywhere (not
  module/model) ✅.
- **Figures 1-4 X/Y symbols** — check and disambiguate ✅.
- **Redraw Fig 1** per the itemised edits; merge Fig 2 into Fig 1; vector graphics;
  enlarge small text; drop training flows from architecture panels 🎨.
- **Redraw Fig 3 & Fig 5** as academic (more imagery, less text, not colour-only) 🎨.
- **Trim Table 2/3 "Value" column** (e.g. "SGD (Momentum=0.937)" → "SGD") ✅.
- **New "Evaluation Metrics" section + a heatmap figure** ✅/🎨.
- **Compare with YOLO11 and YOLO26** 🟠 — needs training runs.
- **Spelling**: `bottleneckssuch`→`bottlenecks such`, `recurrenceand`→`recurrence and`,
  `workarounds` check ✅.
- **Affiliations** of authors 1 and 3 — split combined affiliations ✅.

---

## Reviewer 3

### R3-1 — reproducibility of fuzzy targets   🔴
Same as R1-2/R1-3 above — one shared reply.

### R3-2 — "GS is a training-time regularizer" contradicts Table 7 (0.858 → 0.073)   🔴 CRITICAL

**Reviewer:** Removing GS collapses RAF-DB accuracy 0.858 → 0.073; recovered to 0.849
only after 10 fine-tuning epochs. So GS is load-bearing in the forward path, not a mere
regularizer. Clarify.

**Honest reframe (the paper's wording must change):** GS is *not* a no-op at inference.
The correct claim is a **two-step deployment**: (i) fold-out removes the DPU-hostile GS
operator, which *does* break the forward path (→ 0.073); (ii) a short **post-fold
fine-tuning** (10 epochs, no orthogonality) lets the DPU-native convolutions re-absorb
the function GS was performing (→ 0.849). So the benefit is *internalized into the
weights by fine-tuning*, not "already free". We will rewrite Sec. 3.1.3 and the abstract
to state this explicitly and stop calling GS a pure training-time regularizer.

### R3-3 — add suggested citations   ✅ (add the 4 references)

### R3-4 — control: does baseline gain from the same 10 epochs?   🟠 CRITICAL

**Reviewer:** the 0.073→0.849 recovery may be extra optimization, not orthogonalization.
Give the control: standard YOLOv8 + the same 10 epochs / schedule.

**[ACTION]** Run: baseline YOLOv8n + 10 extra epochs at η0 = 5e-4 cosine, no
orthogonality, same data. Report Δ vs the fine-tuned FLEO. **This is decisive**: if the
baseline gains as much, the FLEO contribution is not supported and the framing must be
honest about it.

---

## Work queue (what unblocks the most replies)

1. 🔴 **Add label-smoothing (Eq. 2) to training** → unblocks R1-2/3, R3-1.
2. 🟠 **Retrain the RAF-DB matrix** (baseline, FLEO, d∈{4,8,16,32}, +10-epoch controls)
   → unblocks R1-1, R3-4, and provides the numbers R3-2 relies on.
3. 🟠 **Train YOLO11 + YOLO26** on RAF-DB → R2 comparison.
4. ✅ **One editing pass** for all R2 cosmetic + R1-5 + R3-3.
5. 🎨 **Redraw Figures 1/2/3/5 + heatmap.**
