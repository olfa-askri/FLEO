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


class ResNetBackbone(nn.Module):
    """ImageNet-pretrained ResNet trunk -> P3 (/16) and P4 (/32) feature maps.

    Far stronger than TinyBackbone for real FER datasets: the ImageNet
    features give the FLEO blocks a much better signal to orthogonalize.
    layer3 -> P3, layer4 -> P4. Channels depend on the depth:
        resnet18/34 : P3=256, P4=512
        resnet50    : P3=1024, P4=2048
    """

    def __init__(self, name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm

        ctor = {
            "resnet18": tvm.resnet18,
            "resnet34": tvm.resnet34,
            "resnet50": tvm.resnet50,
        }[name]
        try:
            weights = "IMAGENET1K_V1" if pretrained else None
            m = ctor(weights=weights)
        except TypeError:                                  # older torchvision
            m = ctor(pretrained=pretrained)

        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4
        self.p3_ch = 1024 if name == "resnet50" else 256
        self.p4_ch = 2048 if name == "resnet50" else 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        p3 = self.layer3(x)
        p4 = self.layer4(p3)
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
                 subspace_dim=16, train_only=True,
                 backbone="tiny", pretrained=True):
        super().__init__()
        if backbone in ("resnet18", "resnet34", "resnet50"):
            self.backbone = ResNetBackbone(backbone, pretrained=pretrained)
            p3_ch, p4_ch = self.backbone.p3_ch, self.backbone.p4_ch
        else:
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
