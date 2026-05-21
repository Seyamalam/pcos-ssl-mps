from __future__ import annotations

import timm
import torch
import torch.nn as nn


class EncoderClassifier(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int = 2,
        pretrained: bool = False,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.classifier = nn.Linear(self.encoder.num_features, num_classes)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.classifier(features)


def load_simclr_encoder(model: EncoderClassifier, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected encoder keys in checkpoint: {unexpected[:10]}")
    critical_missing = [key for key in missing if not key.startswith("head.")]
    if critical_missing:
        raise ValueError(f"Missing encoder keys in checkpoint: {critical_missing[:10]}")

