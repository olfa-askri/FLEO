# FLEO — journal manuscript

`fleo_journal.tex` is a complete IEEEtran journal manuscript built from the
released code and the project drafts. Every results cell is a red placeholder
(`\pending`) — fill it from your own runs. **Do not report unmeasured values.**

## Compile

**Overleaf (easiest):** create a project, upload `fleo_journal.tex` and
`references.bib`, set the compiler to *pdfLaTeX*, hit Recompile.

**Local:**
```bash
pdflatex fleo_journal
bibtex   fleo_journal
pdflatex fleo_journal
pdflatex fleo_journal
```

## Which script fills which number

| Table / figure | Produced by | Command |
|---|---|---|
| Table II (Acc, macro-F1, ablation) | `run_ablation.py` | `python run_ablation.py --data <fer2013> --epochs 40 --seeds 0 1 2` |
| Confusion matrix + t-SNE (Sec. VII-E) | `evaluate.py` | `python evaluate.py --data <fer2013> --ckpt checkpoints/best_fleo.pt` |
| Per-class precision/recall | `evaluate.py` (`results/report.json`) | same as above |
| Table III (INT8 vs FP32, ZCU104) | `deploy/run_vitis.py` | inside Vitis AI Docker — see `deploy/VITIS.md` |
| Table I (params/FLOPs/latency) | profiling | `thop`/`fvcore` on the model + `vai_c_xir` report |

## Before submission checklist
- [ ] Replace every `\pending` with a measured number.
- [ ] Run each row of Table II over 3 seeds; report mean ± std.
- [ ] Add Grad-CAM panels if you want the visual evidence from Sec. VII-E.
- [ ] Complete author emails/ORCIDs, funding acknowledgment, and any missing
      citations (e.g. the contrastive-FER reference).
- [ ] State plainly whether you used true YOLOv12 or a YOLO fallback backbone
      (the code falls back to yolo11/yolov8 when YOLOv12 configs are absent).
