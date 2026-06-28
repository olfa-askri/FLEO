"""
Training loop for a FLEO-equipped FER model.
===========================================
Works with `models.FLEOClassifier` (any ablation mode) and, with the same
interface, with a YOLOv12+FLEO model whose forward returns:
    train : (logits (B,K), z (B,K) or None)
    eval  : logits (B,K)

Training strategy (paper Eq. 15, classification setting):
  - L_total = L_cls + lambda * L_bind
      L_cls  = confusion-aware fuzzy loss (use_fuzzy=True) or plain CE (False)
      L_bind = subspace-emotion binding loss (skipped when z is None)
  - Confusion prior C seeded from a FACS AU-overlap table, then refreshed each
    epoch from the model's real mistakes (running EMA) when fuzzy is on.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import FLEOTotalLoss
from utils.confusion_matrix import FACS_PRIOR_7, normalize_confusion, build_running_confusion
from utils.metrics import compute_metrics, full_report

EMOTIONS_7 = ["anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class FLEOTrainer:
    def __init__(self, model, train_loader, val_loader, cfg: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.K = cfg.get("num_emotions", 7)
        self.use_fuzzy = cfg.get("use_fuzzy", True)
        self.verbose = cfg.get("verbose", True)
        self.ckpt = cfg.get("ckpt", "checkpoints/best_fleo.pt")
        self.class_names = cfg.get("class_names", EMOTIONS_7[: self.K])
        self.model.to(self.device)

        C = normalize_confusion(FACS_PRIOR_7).to(self.device)
        self.loss_fn = FLEOTotalLoss(
            confusion_matrix=C,
            lam_bind=cfg.get("lam_bind", 0.1),
            alpha_max=cfg.get("alpha_max", 0.3),
            use_fuzzy=self.use_fuzzy,
        )
        self.optimizer = AdamW(model.parameters(),
                               lr=cfg.get("lr", 1e-4),
                               weight_decay=cfg.get("wd", 1e-4))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.get("epochs", 50))
        self.running_C = C.clone()

    # ------------------------------------------------------------------
    def _train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits, z = self.model(imgs)                 # (B,K), (B,K)|None
            losses = self.loss_fn(logits, z, labels)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            total_loss += losses["total"].item()
            all_preds.append(logits.argmax(-1).detach())
            all_targets.append(labels.detach())

        self.scheduler.step()
        preds = torch.cat(all_preds); targets = torch.cat(all_targets)

        # Refresh the confusion prior from real mistakes (only used by fuzzy loss).
        if self.use_fuzzy:
            self.running_C = build_running_confusion(
                preds, targets, num_classes=self.K, prev_C=self.running_C
            ).to(self.device)
            self.loss_fn.fuzzy_loss.C = self.running_C

        m = compute_metrics(preds, targets)
        m["loss"] = total_loss / max(len(self.train_loader), 1)
        return m

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        all_preds, all_targets = [], []
        for imgs, labels in self.val_loader:
            imgs = imgs.to(self.device)
            logits = self.model(imgs)                    # eval path: logits only
            all_preds.append(logits.argmax(-1).cpu())
            all_targets.append(labels)
        preds, targets = torch.cat(all_preds), torch.cat(all_targets)
        m = compute_metrics(preds, targets)
        return m, preds, targets

    # ------------------------------------------------------------------
    def run(self) -> dict:
        os.makedirs(os.path.dirname(self.ckpt) or ".", exist_ok=True)
        best_f1, best = 0.0, None
        for epoch in range(1, self.cfg.get("epochs", 50) + 1):
            tr = self._train_epoch()
            va, preds, targets = self._val_epoch()
            if self.verbose:
                print(f"Epoch {epoch:03d} | loss {tr['loss']:.4f} | "
                      f"train acc {tr['accuracy']:.2f}% | "
                      f"val acc {va['accuracy']:.2f}% | val F1 {va['macro_f1']:.2f}%")
            if va["macro_f1"] > best_f1:
                best_f1 = va["macro_f1"]
                best = full_report(preds, targets, self.K, self.class_names)
                torch.save(self.model.state_dict(), self.ckpt)
                if self.verbose:
                    print(f"  saved best (macro-F1 = {best_f1:.2f}%) -> {self.ckpt}")
        return best or full_report(preds, targets, self.K, self.class_names)
