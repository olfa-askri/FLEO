# HANDOFF — FLEO-FER (reprise de session, 2026-07-09)

Synthèse pour reprendre le travail. Sources : `docs/SESSION_LOG.md` (journal committé)
+ la session en cours (run Kaggle live + bugs notebook découverts cette nuit).

## État actuel du code

- Repo : `C:\Users\BASSEM GZIGUEZ\fleo-fer` ↔ https://github.com/olfa-askri/FLEO (branche `main`).
- Pipeline complet en place : `fleo/` (bloc FLEO, Gram-Schmidt/Householder, fold-out),
  `data/` (FER2013 + RAF-DB → format détection YOLO), `scripts/` (train / evaluate /
  deltas / export / run_matrix), `deploy/` (kit Vitis AI), `sota/` (tables), `tests/`.
- CLI vérifiées : `scripts/deltas.py` supporte `--metric {accuracy,macro_f1}` et `--out` ;
  `scripts/evaluate.py` supporte `--out` ; `run_matrix` supporte `--project`
  (défaut `runs/fleo`, noms `baseline_seed{N}` / `fleo_seed{N}`,
  deltas écrits dans `results/deltas_{dataset}.json`).
- **Bug corrigé cette session** : `notebooks/kaggle_fleo.ipynb` était cassé à 4 endroits
  (échappements mangés à sa création) — `\n` littéral au milieu des commandes shell des
  cellules d'entraînement (crash argparse immédiat) et vrais sauts de ligne dans des
  chaînes `print('...')` (SyntaxError → échec de tout le commit → perte totale des outputs).
- Notebook régénéré proprement (via `json.dump`) avec en plus :
  - horodatage `t0` en cellule 1 ;
  - **garde-fou temporel** avant RAF-DB : ne démarre que s'il reste > 3,5 h sur un budget
    de 11,5 h (marge 0,5 h avant le mur des 12 h), sinon skip propre ;
  - cellules de résultats robustes (aucun `json.load` sur fichier manquant ne peut plus
    faire échouer le commit) ;
  - `--project` séparés (`runs/fer2013`, `runs/rafdb`) pour éviter la collision des noms
    `baseline_seed0` entre datasets ;
  - zip FER2013 dès qu'il existe → un run partiel persiste toujours au moins FER2013.

## Résultats obtenus (par modèle)

| Run | Modèle | mAP50 | mAP50-95 | Notes |
|---|---|---|---|---|
| FER2013, 100 ep (1er run, **perdu** — wipe Kaggle) | Baseline YOLOv12-S | 0,764 | 0,764 | P 0,708 / R 0,72 |
| idem | FLEO | 0,763 | 0,628 | **égalité** ; sur-apprend (val cls_loss ~2,1 vs ~0,75) |
| RAF-DB partiel (perdu aussi) | — | ~0,83 | — | dataset plus propre |
| FER2013, 100 ep (**run en cours**, version 333623534) | Baseline puis FLEO | 0,724 @ ep 62, en montée | 0,622 | ~197 s/epoch |

- Per-class FER2013 (baseline, 1er run) : happy 0,955 > surprise 0,881 > disgust 0,776 >
  neutral 0,744 > angry 0,714 > sad 0,654 > fear 0,621.
- **Pas encore mesurés** : accuracy, macro-F1, Δ_fold, Δ_q (sortent de `scripts/deltas.py`) —
  ce sont les chiffres décisifs du papier.
- Run Kaggle en cours : démarré ~16 h 05 le 08/07 ; fin FER2013 estimée ~3 h-3 h 30 ;
  mur des 12 h à ~4 h 05. Si cette version contient l'ancienne cellule RAF-DB
  (probable, ancienne version du notebook), elle sera tuée à 4 h 05 → tout perdu.

## Décisions prises

- 60 epochs (plateau ~60-70 ; 100 ep = +0,01 mAP pour 4 h de risque en plus).
- Persistance via **Save & Run All (Commit)** uniquement ; jamais de long run interactif.
- `mosaic=0, mixup=0, close_mosaic=0` (mosaic/mixup corrompent le label boîte plein-image
  → box_loss inf sous AMP → collapse epoch ~90).
- Warm-start COCO (`yolo12s.pt`) par défaut.
- Anti-overfit FLEO : Dropout2d 0,15, lambda_ortho 0,01 → 0,05, cosine LR, wd 8e-4.
- FER cast en détection boîte plein-image (réalisme déploiement).
- Framing papier : **déploiement FPGA + méthodologie fold-out**, pas papier accuracy.
- Nouveau (cette session) : garde-fou temporel + cellules infaillibles dans le notebook.

## Approches écartées

- mosaic/mixup pour ce task (voir ci-dessus).
- Runs interactifs > 12 h sur Kaggle (perte totale au reset — vécu).
- 100 epochs en production (gain marginal).
- Viser le SOTA GPU (ResMaskingNet ~76 % FER2013, ResEmoteNet 94,76 % RAF-DB) : hors
  scope edge, non déployable DPU.
- Annuler le commit en cours pour « récupérer » les poids : impossible, un commit
  annulé/échoué ne publie aucun output.
- Subagent sur `fer2013-history.md` : le fichier n'existe pas ; l'historique réel est
  `docs/SESSION_LOG.md` (déjà en contexte).

## Prochaine étape

1. ✅ (fait cette session) Notebook corrigé + protégé, committé et poussé sur `main`.
2. **Au matin** : vérifier la version 333623534 sur Kaggle →
   - si « Successfully ran » : onglet Output → télécharger `fleo_fer2013.zip` →
     lire `results/deltas_fer2013.json` (**macro-F1** et **Δ_fold** d'abord) ;
   - si « Failed » (mur des 12 h) : relancer le notebook corrigé (60 ep) via
     Save & Run All — il est maintenant à l'épreuve du mur.
3. Remplir `sota/COMPARISON_TABLE.md` / Table VIII avec les vrais chiffres.
4. Vitis AI quantize + compile sous Docker Desktop (Windows) → prouver le sous-graphe
   DPU unique (R1) + mesurer Δ_q — pas besoin de carte.
5. Obtenir une ZCU104 (labo EµE Monastir / ENET'Com) pour FPS / puissance / ressources.
