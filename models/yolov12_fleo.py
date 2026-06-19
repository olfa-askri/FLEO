"""
YOLOv12 + FLEO integration wrapper.
Inserts FLEOModule at Neck levels P3 and P4.

This file shows the conceptual integration point.
Real training requires the full YOLOv12 codebase (ultralytics/ultralytics).
"""

import torch
import torch.nn as nn
from .fleo_module import FLEOModule


class YOLOv12FLEONeck(nn.Module):
    """
    Drop-in replacement for the YOLOv12 Neck that injects FLEO at P3 and P4.

    Args:
        base_neck     : original YOLOv12 neck module
        p3_channels   : channel width at P3 feature map
        p4_channels   : channel width at P4 feature map
        num_emotions  : K  (default 7)
        subspace_dim  : d  per-emotion latent width
        train_only    : Route 1 – remove GS ops at inference for DPU
    """

    def __init__(
        self,
        base_neck: nn.Module,
        p3_channels: int = 256,
        p4_channels: int = 512,
        num_emotions: int = 7,
        subspace_dim: int = 32,
        train_only: bool = True,
    ):
        super().__init__()
        self.base_neck = base_neck

        self.fleo_p3 = FLEOModule(
            in_channels=p3_channels,
            num_emotions=num_emotions,
            subspace_dim=subspace_dim,
            train_only=train_only,
        )
        self.fleo_p4 = FLEOModule(
            in_channels=p4_channels,
            num_emotions=num_emotions,
            subspace_dim=subspace_dim,
            train_only=train_only,
        )

    def forward(self, features: list):
        """
        features : list of backbone feature maps [P3, P4, P5]
        Returns  : list of enhanced feature maps with FLEO applied at P3, P4.
                   During training also returns binding logits z (B, K) for each
                   FLEO level so the trainer can compute L_bind.
        """
        p3, p4, p5 = self.base_neck(features)

        if self.training:
            p3_out, z3 = self.fleo_p3(p3)              # z3: (B, K)
            p4_out, z4 = self.fleo_p4(p4)              # z4: (B, K)
            return [p3_out, p4_out, p5], {"bind_p3": z3, "bind_p4": z4}
        else:
            p3_out = self.fleo_p3(p3)
            p4_out = self.fleo_p4(p4)
            return [p3_out, p4_out, p5]


def export_inference_graph(model: YOLOv12FLEONeck) -> nn.Module:
    """
    Route 1 export helper.
    Sets train_only=True and switches to eval mode so GS ops are bypassed.
    The resulting model contains only Conv 1x1, SE Gate, Conv 3x3 – all
    fully supported by Xilinx DPU B4096.
    """
    model.eval()
    model.fleo_p3.train_only = True
    model.fleo_p4.train_only = True
    return model
