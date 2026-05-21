from __future__ import annotations

import timm
import torch.nn as nn


def create_classifier(backbone: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

