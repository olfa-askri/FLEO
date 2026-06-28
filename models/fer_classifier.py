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


class YOLOv12Backbone(nn.Module):
    """Real YOLOv12 backbone (ultralytics) tapped at P3 (/8) and P4 (/16).

    This is the backbone the paper is about: FLEO is inserted on the YOLOv12
    P3/P4 feature maps. Requires `pip install ultralytics`. On first use it
    downloads pretrained COCO weights for the chosen variant
    (yolov12n/s/m/l/x) unless pretrained=False.

    NOTE: YOLOv12 (ultralytics) expects inputs in [0,1] (plain ToTensor),
    NOT ImageNet mean/std normalization.
    """

    def __init__(self, variant: str = "yolov12s", pretrained: bool = True):
        super().__init__()
        from ultralytics import YOLO

        # Prefer pretrained COCO weights, but fall back to building the
        # architecture from the YAML config if the .pt cannot be downloaded
        # (offline runners, missing release asset, network blocked, ...).
        det = None
        if pretrained:
            try:
                det = YOLO(f"{variant}.pt").model
            except Exception as e:                 # download/load failed
                print(f"[YOLOv12Backbone] could not load {variant}.pt "
                      f"({type(e).__name__}); building from {variant}.yaml instead.")
        if det is None:
            det = YOLO(f"{variant}.yaml").model    # architecture only, no weights
        self.layers = det.model                    # ModuleList (each has .f, .i)
        self._p3_idx = self._p4_idx = None
        self.p3_ch = self.p4_ch = None
        self._probe()

    def _run(self, x):
        """Replicates ultralytics' layer routing, stopping before the head."""
        y, feats = [], {}
        for m in self.layers:
            if getattr(m, "f", -1) != -1:          # gather inputs from earlier layers
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if m.i == len(self.layers) - 1:        # stop before Detect head
                break
            x = m(x)
            y.append(x)
            feats[m.i] = x
        return feats

    @torch.no_grad()
    def _probe(self, size: int = 128):
        self.eval()
        feats = self._run(torch.zeros(1, 3, size, size))
        for i in sorted(feats):
            o = feats[i]
            if not torch.is_tensor(o) or o.dim() != 4:
                continue
            stride = size / o.shape[-1]
            if abs(stride - 8) < 0.5:              # /8  -> P3 (keep last = nearest head)
                self._p3_idx, self.p3_ch = i, o.shape[1]
            elif abs(stride - 16) < 0.5:           # /16 -> P4
                self._p4_idx, self.p4_ch = i, o.shape[1]
        if self._p3_idx is None or self._p4_idx is None:
            raise RuntimeError("Could not locate P3 (/8) and P4 (/16) in YOLOv12 backbone")

    def forward(self, x):
        feats = self._run(x)
        return feats[self._p3_idx], feats[self._p4_idx]


class TinyHead(nn.Module):
    """Global-pools P3 & P4, concatenates, predicts K emotion logits.

    `forward(..., return_feat=True)` also returns the pooled penultimate
    feature vector (used for the t-SNE visualization in the paper)."""

    def __init__(self, p3_ch, p4_ch, num_emotions):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(p3_ch + p4_ch, num_emotions)

    def forward(self, p3, p4, return_feat=False):
        f = torch.cat([self.pool(p3).flatten(1), self.pool(p4).flatten(1)], dim=1)
        logits = self.fc(f)
        return (logits, f) if return_feat else logits


# Ablation modes (Table 2 of the paper). The fuzzy loss is a separate,
# loss-level switch handled by the trainer, not by the architecture.
ABLATION_MODES = {
    "baseline":    dict(enabled=False),                              # backbone + head
    "se":          dict(enabled=True, use_ortho=False, use_binding=False),
    "fleo_nobind": dict(enabled=True, use_ortho=True,  use_binding=False),
    "fleo_full":   dict(enabled=True, use_ortho=True,  use_binding=True),
}


class FLEOClassifier(nn.Module):
    """
    End-to-end FER model: backbone -> (FLEO @ P3,P4) -> head.

    mode (ablation, see ABLATION_MODES):
        baseline    : no neck module (backbone + head only)
        se          : SE gate only (no orthogonalization, no binding)
        fleo_nobind : Gram-Schmidt + SE, no binding head
        fleo_full   : full FLEO block (default)

    forward(x) returns:
        training : (logits (B,K), z (B,K) or None)   z = averaged binding logits
        eval     : logits (B,K)
    """

    def __init__(self, num_emotions=7, p3_ch=64, p4_ch=128,
                 subspace_dim=16, train_only=True,
                 backbone="tiny", pretrained=True, mode="fleo_full"):
        super().__init__()
        if mode not in ABLATION_MODES:
            raise ValueError(f"mode must be one of {list(ABLATION_MODES)}, got {mode!r}")
        self.mode = mode
        spec = ABLATION_MODES[mode]

        if backbone in ("resnet18", "resnet34", "resnet50"):
            self.backbone = ResNetBackbone(backbone, pretrained=pretrained)
            p3_ch, p4_ch = self.backbone.p3_ch, self.backbone.p4_ch
        elif backbone.startswith("yolov12"):
            self.backbone = YOLOv12Backbone(backbone, pretrained=pretrained)
            p3_ch, p4_ch = self.backbone.p3_ch, self.backbone.p4_ch
        else:
            self.backbone = TinyBackbone(p3_ch, p4_ch)

        self.has_neck = spec["enabled"]
        if self.has_neck:
            self.fleo_p3 = FLEOModule(p3_ch, num_emotions, subspace_dim,
                                      train_only=train_only,
                                      use_ortho=spec["use_ortho"],
                                      use_binding=spec["use_binding"])
            self.fleo_p4 = FLEOModule(p4_ch, num_emotions, subspace_dim,
                                      train_only=train_only,
                                      use_ortho=spec["use_ortho"],
                                      use_binding=spec["use_binding"])
        self.head = TinyHead(p3_ch, p4_ch, num_emotions)

    def forward(self, x, return_feat=False):
        p3, p4 = self.backbone(x)
        z = None
        if self.has_neck:
            if self.training:
                p3, z3 = self.fleo_p3(p3)
                p4, z4 = self.fleo_p4(p4)
                if z3 is not None and z4 is not None:
                    z = (z3 + z4) / 2                   # combine binding logits
            else:
                p3 = self.fleo_p3(p3)
                p4 = self.fleo_p4(p4)
        out = self.head(p3, p4, return_feat=return_feat)
        if self.training:
            logits = out[0] if return_feat else out
            return (logits, z, out[1]) if return_feat else (logits, z)
        return out                                     # logits, or (logits, feat)

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
