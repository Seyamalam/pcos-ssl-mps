from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 128,
        hidden_dim: int = 512,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feature_dim = self.encoder.num_features
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        projections = self.projector(features)
        return F.normalize(projections, dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    similarity = similarity.masked_fill(mask, float("-inf"))

    positives = torch.cat(
        [
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ]
    )
    return F.cross_entropy(similarity, positives)

