"""
Self-contained FER classifier built around FLEO.
================================================
This is a *minimal, runnable* stand-in for the full YOLOv12 model. It lets you
train / test / export the complete FLEO pipeline end-to-end without pulling in
the heavyweight ultralytics codebase. The real project swaps `TinyBackbone`
and `TinyHead` for YOLOv12's backbone/neck/head, keeping the FLEO blocks
identical.

Structure mirrors a detector:
    backbone -> {P3, P4} feature maps -> FLEO on each -> head -> emotion logits
"""

import torch
import torch.nn as nn
from .fleo_module import FLEOModule


def _conv(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class TinyBackbone(nn.Module):
    """Produces two pyramid levels P3 (stride 8) and P4 (stride 16)."""

    def __init__(self, p3_ch=64, p4_ch=128):
        super().__init__()
        self.stem = _conv(3, 32, stride=2)             # /2
        self.layer1 = _conv(32, p3_ch, stride=2)       # /4
        self.layer2 = _conv(p3_ch, p3_ch, stride=2)    # /8  -> P3
        self.layer3 = _conv(p3_ch, p4_ch, stride=2)    # /16 -> P4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        return p3, p4


class TinyHead(nn.Module):
    """Global-pools P3 & P4, concatenates, predicts K emotion logits."""

    def __init__(self, p3_ch, p4_ch, num_emotions):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(p3_ch + p4_ch, num_emotions)

    def forward(self, p3, p4):
        f = torch.cat([self.pool(p3).flatten(1), self.pool(p4).flatten(1)], dim=1)
        return self.fc(f)


class FLEOClassifier(nn.Module):
    """
    End-to-end FER model with FLEO at P3 and P4.

    forward(x) returns:
        training : (logits (B,K), z (B,K))   z = averaged binding logits
        eval     : logits (B,K)
    """

    def __init__(self, num_emotions=7, p3_ch=64, p4_ch=128,
                 subspace_dim=16, train_only=True):
        super().__init__()
        self.backbone = TinyBackbone(p3_ch, p4_ch)
        self.fleo_p3 = FLEOModule(p3_ch, num_emotions, subspace_dim, train_only=train_only)
        self.fleo_p4 = FLEOModule(p4_ch, num_emotions, subspace_dim, train_only=train_only)
        self.head = TinyHead(p3_ch, p4_ch, num_emotions)

    def forward(self, x):
        p3, p4 = self.backbone(x)
        if self.training:
            p3, z3 = self.fleo_p3(p3)
            p4, z4 = self.fleo_p4(p4)
            logits = self.head(p3, p4)
            z = (z3 + z4) / 2                           # combine binding logits
            return logits, z
        else:
            p3 = self.fleo_p3(p3)
            p4 = self.fleo_p4(p4)
            return self.head(p3, p4)
