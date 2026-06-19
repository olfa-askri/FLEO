"""
Training script for YOLOv12 + FLEO.

Training strategy:
  - Phase 1 : warm-up (freeze backbone, train FLEO + head)
  - Phase 2 : full fine-tune with FLEOTotalLoss
  - Confusion matrix C is updated every epoch (running mean)

Cross-dataset generalization protocol:
  - Train  : RAF-DB
  - Test   : AffectNet (zero-shot cross-dataset evaluation)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import FLEOTotalLoss
from utils.confusion_matrix import FACS_PRIOR_7, normalize_confusion, build_running_confusion
from utils.metrics import compute_metrics, orthogonality_check


class FLEOTrainer:
    def __init__(self, model, train_loader, val_loader, cfg: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Initialise confusion prior from FACS AU-overlap table
        C = normalize_confusion(FACS_PRIOR_7).to(self.device)
        self.loss_fn = FLEOTotalLoss(
            confusion_matrix=C,
            lam_bind=cfg.get("lam_bind", 0.1),
            alpha_max=cfg.get("alpha_max", 0.3),
        )

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("wd", 1e-4),
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.get("epochs", 50)
        )

        self.running_C = C.clone()

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for batch in self.train_loader:
            imgs, emotion_labels = batch
            imgs = imgs.to(self.device)
            emotion_labels = emotion_labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward through YOLOv12 + FLEO neck
            features, bind_dict = self.model.neck(self.model.backbone(imgs))
            logits = self.model.head(features)           # (B, K)

            # Combine binding logits from P3 and P4
            bind_logits = (bind_dict["bind_p3"] + bind_dict["bind_p4"]) / 2

            losses = self.loss_fn(logits, bind_logits, emotion_labels)
            losses["total"].backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            total_loss += losses["total"].item()
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.detach())
            all_targets.append(emotion_labels.detach())

        self.scheduler.step()

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Update running confusion matrix
        self.running_C = build_running_confusion(
            all_preds, all_targets,
            num_classes=self.cfg.get("num_emotions", 7),
            prev_C=self.running_C,
        ).to(self.device)
        self.loss_fn.fuzzy_loss.C = self.running_C

        metrics = compute_metrics(all_preds, all_targets)
        metrics["loss"] = total_loss / len(self.train_loader)
        return metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _val_epoch(self) -> dict:
        self.model.eval()
        all_preds, all_targets = [], []

        for batch in self.val_loader:
            imgs, emotion_labels = batch
            imgs = imgs.to(self.device)
            logits = self.model(imgs)                    # inference path
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_targets.append(emotion_labels)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        return compute_metrics(all_preds, all_targets)

    # ------------------------------------------------------------------
    def run(self):
        best_f1 = 0.0
        for epoch in range(1, self.cfg.get("epochs", 50) + 1):
            train_m = self._train_epoch(epoch)
            val_m = self._val_epoch()

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {train_m['loss']:.4f} | "
                f"Train Acc {train_m['accuracy']:.2f}% | "
                f"Val Acc {val_m['accuracy']:.2f}% | "
                f"Val macro-F1 {val_m['macro_f1']:.2f}%"
            )

            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                torch.save(self.model.state_dict(), "checkpoints/best_fleo.pt")
                print(f"  ✓ Saved best model (macro-F1 = {best_f1:.2f}%)")
