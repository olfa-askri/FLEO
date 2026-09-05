# Reviewer-style assessment — YOLOv8-FLEO (MDPI Algorithms)

## Recommendation: MAJOR REVISION (not acceptable as-is; not outright reject)

Conceptual responses are strong and honest; however the Results section is internally
inconsistent and contradicts the response letter. Must be reconciled before acceptance.

## CRITICAL (blocking)
- C1 Backbone undecided: YOLOv8s (p.15 Table 3, p.17 results, p.23 Table 7 caption)
     vs YOLOv8n (p.29, backbone table, response). The DPU-native argument needs ONE backbone.
- C2 d-ablation table differs between manuscript and response:
     Manuscript Table 7 (p.23): RAF-DB Acc. 0.845/0.849/0.851/0.848 (d=8 selected).
     Response Table 1: mAP@0.5 0.807/0.790/0.806/0.820 (d=8=0.790 = WORST; d=32 best).
     Irreconcilable (different metric + values); d=8 is the minimum in the response table.
- C3 Headline RAF-DB number inconsistent: 0.8732 (p.17) vs 0.858 acc (p.23) vs 0.817
     (resp 3.4) vs ~0.79 (backbone table). 8-point spread. Need one labeled main table.
- C4 "improves every class" (p.17) contradicts response 3.4 "fear and surprise decrease".

## MEDIUM
- C5 Doubled caption "Table 7. Table 1:" (p.23).
- C6 Table numbering/cross-refs off (Table 8 cited before Table 7; backbone = Table 15
     in ms but Table 2 in response).
- C7 Disgust reported two ways: 0.562->0.638 (per-class) vs 0.510->0.554 (resp 3.4).
- C8 Remove any remaining [bracket]/TODO placeholders before submission.

## WELL ADDRESSED
- 1.2/1.3/3.1 fuzzy single-label; 3.2 GS load-bearing; 3.4 control; 1.4 notations;
  2.9(i) metrics section 4.1.5 (p.16); 2.9(ii) confusion + Grad-CAM (Fig 18);
  2.9(iii) YOLO11/YOLO26 backbone table; 2.1/2.2/2.3/2.8/2.10.

## TO VERIFY / MISSING
- Figures 1,2,3,5 actually redrawn vector + larger text + shape (not colour-only)?
- R3's 4 references actually cited in 2.6 + bibliography?
- Affiliations 1 & 3 actually separated (2.11)?
- 4.1.5 claims "mean +/- std over 3 seeds" but tables show no std -> add or remove claim.

## Verdict
Repairable WITHOUT new experiments: fix one backbone, one consistent main results table,
reconcile the ablation between letter and manuscript. Then Accept-track.
