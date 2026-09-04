# Où corriger chaque point dans le manuscrit

Localisation exacte (page / section / ligne) de chaque édition. Numéros de page = PDF fourni.

## ⚠️ AVERTISSEMENT PRIORITAIRE — Section 4.3 (p.22–25)

Le manuscrit contient **Section 4.3 "Hardware Deployment on the ZCU104"** avec
**4.3.3 Resource Utilization & Timing Closure** (p.24) et **4.3.4 Power Efficiency &
Quantization** (p.25). **VÉRIFIE d'urgence :** ces tables contiennent-elles des chiffres
présentés comme **mesurés sur une carte physique** (LUT/DSP/BRAM, puissance en W, latence,
FPS) ? **Tu n'as pas de carte ZCU104.**
- Si les valeurs sont des **estimations / templates / "protocol"** → écris-le explicitement.
- Si elles sont présentées comme **mesurées** → **retire-les ou marque-les "estimated"**.
  C'est la seule vraie ligne rouge d'intégrité (voir mon évaluation précédente).

---

| # | Correction | Où exactement (page / section / ligne) |
|---|---|---|
| **2.1** titre | raccourcir à 2 lignes | **p.1**, titre |
| **2.11** affiliations | séparer auteurs 1 et 3 | **p.1**, bloc auteurs (avant "Correspondence") |
| **3.2** GS (abstract) | reformuler "fold-out + fine-tuning" | **p.1**, Abstract |
| **3.2** GS (claim) | corriger "regularization mechanism is indispensable" | **p.2**, ~ligne 65 |
| **2.10** orthographe | `bottleneckssuch`→`bottlenecks such` | **p.3**, ligne 85 |
| **2.10** orthographe | `recurrenceand`→`recurrence and` ; `workarounds` | **p.3**, lignes 86–87 |
| **2.2** paragraphe | supprimer "The remainder of this paper…" | **p.3**, lignes 99–103 |
| **3.3** citations | ajouter les 4 références | **Section 2** (related work) + liste des références |
| **2.5** Figure 1 | redessiner (vectoriel) | **p.6**, Figure 1 |
| **2.6** Figure 2 | fusionner dans Fig 1 | **p.7**, Figure 2 |
| **1.2 / 3.1** fuzzy | ajouter paragraphe "single-label → μ lissé" | **p.7–8**, Section 3.1.1 |
| **1.4** notations | définir μ, n_c, α, π, K sous Eq(2) | **p.8**, sous Éq.(2) ligne 267 |
| **1.3** α et π | ajouter valeurs + justification (α=0.1, π fréquence) | **p.8**, Section 3.1.1 (après Éq.2) |
| **1.4** notations | "where…" sous Éq.(1)–(6) | **p.8** |
| **1.4** notations | "where…" sous Éq.(7) | **p.9**, ligne 307 |
| **3.2** GS (texte) | réécrire en "2 étapes" (fold + fine-tune) | **p.9**, Section 3.1.3 ligne 299 |
| **1.4** notations | "where…" sous Éq.(8) | **p.10** |
| **1.4** notations | "where…" sous Éq.(9) | **p.11** |
| **2.7** Figures 3 & 5 | redessiner académique | **p.11** (Fig 3), **p.12** (Fig 5) |
| **2.4** symboles X/Y | vérifier | **Figures 1–4** (p.6, 7, 11) |
| **1.4** notations | "where…" sous Éq.(10)(11) (Δ_fold, Δ_q) | **p.12** |
| **2.8** Table 3 | raccourcir colonne "Value" (SGD…) | **p.14**, Table 3 ligne 446 |
| **2.9** metrics | nouvelle sous-section métriques déjà présente (4.1.4) — compléter | **p.15**, Section 4.1.4 |
| **3.4** control + disgust | ajouter ligne control + finding ambiguïté | **p.17**, Section 4.2.2 "Emotion Ambiguity" (disgust +4.5) |
| **1.1** d-ablation | ajouter Table d∈{4,8,16,32} | **p.20**, Section 4.2.3 "Subspace Weight Ablation" ligne 568 |
| **3.4** control | ajouter ligne "baseline +10ep" | **p.20**, Table 7 |
| **2.9** YOLO11/26 + heatmap | ajouter comparaison + confusion figure | **p.26**, Section 4.4 "Comparative Study" |
| **1.5** anglais + refs | relecture globale + format MDPI | tout le manuscrit + références |
| **2.3** unifier termes | "YOLOv8-FLEO framework" partout | tout le manuscrit |

---

## Ordre conseillé (du plus rapide au plus lourd)
1. **Rapide (texte)** : 2.1, 2.2, 2.10, 2.11, 2.3, 3.2-abstract → p.1–3
2. **Méthodo** : 1.2, 1.3, 1.4, 3.1, 3.2-texte → Section 3 (p.7–12)
3. **Résultats** : 1.1, 3.4, 2.9 → Section 4 (p.15–26)
4. **⚠️ Vérifier 4.3 ZCU104** (p.22–25) — mesuré vs estimé
5. **Figures** : 2.4, 2.5, 2.6, 2.7 (p.6, 7, 11, 12)
6. **Global** : 1.5, 2.3, 3.3

Les textes prêts à coller sont dans `rebuttal.tex` / le PDF du rebuttal.
