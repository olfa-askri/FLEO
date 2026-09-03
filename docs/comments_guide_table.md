# Guide des 20 commentaires — comment corriger, explication, où dans le papier

| # | Commentaire | Comment corriger | Explication simple (exemple) | Où dans le papier / quoi ajouter–enlever |
|---|---|---|---|---|
| 1.1 | Justifier d=8 | Ajouter tableau ablation d=4,8,16,32 (déjà calculé) | On teste plusieurs largeurs d. Ex: d=8 → 56 canaux. La précision bouge peu (~3 pts) mais le coût monte fort (d=32 = 4.3×) → on garde d=8 | **§4 Résultats** : AJOUTER Table d-ablation. **§3.1.2** : AJOUTER 1 phrase « d=8 = équilibre précision/coût » |
| 1.2 | D'où viennent les fuzzy labels ? | Écrire : générés par calcul, pas de votes | Ex : « happy » → happy 92 %, autres 1.3 %. Aucun vote, juste une formule de lissage | **§3.1.1 sous Eq(2)** : AJOUTER le paragraphe « single-label → μ lissé » |
| 1.3 | Justifier α et π | α=0.1 (valeur standard), π=fréquence des classes | α = combien on adoucit (0.1 = léger) ; π = comment répartir le reste | **§3.1.1** : AJOUTER les valeurs α, π + justification |
| 1.4 | Définir les symboles | Sous chaque équation, ligne « where … » | Ex : sous Eq(2) : « où n_c = votes, α = lissage, π = prior » | **Sous Eq(1)–(11)** : AJOUTER une ligne de définitions (voir checklist 1.4) |
| 1.5 | Anglais + refs | Relecture langue + format MDPI des références | — | **Tout le manuscrit** : corriger langue ; refs au format MDPI |
| 2.1 | Titre trop long | Raccourcir à ≤ 2 lignes | Titre court, ex : « DPU-Native YOLOv8-FLEO: Fold-Out … FER » | **Page 1** : REMPLACER le titre |
| 2.2 | Paragraphe redondant | Supprimer « The remainder of this paper … » | Le paragraphe qui annonce le plan est inutile | **Fin §1 (lignes 100-103)** : ENLEVER |
| 2.3 | Termes non unifiés | Un seul terme : « YOLOv8-FLEO framework » | Ne pas mélanger framework / module / model | **Tout** : REMPLACER par un terme unique |
| 2.4 | Symboles X/Y des figures | Vérifier Fig 1-4 ; symboles distincts si sens différent | Même lettre = même chose partout | **Fig 1–4** : corriger les axes/symboles |
| 2.5 | Figure 1 à refaire | Redessiner en vectoriel ; enlever labels listés ; ajouter « Output » | Figure claire, gros texte, sans les flux d'entraînement | **Fig 1** : REDESSINER |
| 2.6 | Figure 2 | Fusionner dans Figure 1 ; enlever doublons | Un seul schéma au lieu de deux | **Fig 2 → Fig 1** |
| 2.7 | Fig 3 et 5 | Redessiner « académique » : plus d'images, moins de texte, pas que la couleur | Ajouter formes/motifs, pas seulement des couleurs | **Fig 3, 5** : REDESSINER |
| 2.8 | Tables 2 et 3 trop chargées | Raccourcir la colonne « Value » | « SGD (Momentum=0.937) » → « SGD » | **Table 2, 3** : SIMPLIFIER la colonne Value |
| 2.9 | Comparer + backbones | Entraîner YOLO11/26 (notebook) + heatmap | Tester d'autres modèles pour situer le nôtre | **§4** : AJOUTER Table comparaison + figure heatmap |
| 2.10 | Fautes d'orthographe | Corriger les mots collés | « bottleneckssuch » → « bottlenecks such » | **Lignes 86-87** + spell-check global |
| 2.11 | Affiliations | Séparer les affiliations des auteurs 1 et 3 | Deux institutions combinées → deux entrées | **Page 1 (auteurs)** : SÉPARER |
| 3.1 | Reproductibilité fuzzy | Identique à 1.2 | Même réponse : μ généré par calcul, code publié | **§3.1.1** : même ajout que 1.2 |
| 3.2 | GS « juste régularizer » ? | Reformuler « fold-out + post-fold fine-tuning » | On enlève GS → ça casse (0.073) → 10 epochs réparent (0.849). Le bénéfice s'absorbe dans les poids par le fine-tuning | **§3.1.3 + Abstract** : RÉÉCRIRE ; ANNOTER Table 7 (2 étapes) |
| 3.3 | Citations suggérées | Ajouter les 4 références | — | **§2 (related work) + références** : AJOUTER |
| 3.4 | Control manquant | Entraîner baseline +10 ep (notebook) | Prouver que le gain vient de FLEO, pas juste de +10 epochs d'entraînement | **§4 / Table 7** : AJOUTER la ligne « baseline +10 ep » |

---

**Légende actions :** AJOUTER = nouveau texte/figure · ENLEVER = supprimer · REMPLACER = modifier l'existant · REDESSINER = refaire la figure.
