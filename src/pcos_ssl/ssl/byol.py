from __future__ import annotations

import copy

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class BYOLModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 256,
        hidden_dim: int = 1024,
        predictor_hidden_dim: int = 512,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.online_encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        feature_dim = self.online_encoder.num_features
        self.online_projector = mlp(feature_dim, hidden_dim, projection_dim)
        self.target_projector = copy.deepcopy(self.online_projector)
        self.predictor = mlp(projection_dim, predictor_hidden_dim, projection_dim)
        self._freeze_target()

    def _freeze_target(self) -> None:
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.target_projector.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        self._update_target_module(self.online_encoder, self.target_encoder, momentum)
        self._update_target_module(self.online_projector, self.target_projector, momentum)

    @staticmethod
    @torch.no_grad()
    def _update_target_module(
        online_module: nn.Module,
        target_module: nn.Module,
        momentum: float,
    ) -> None:
        online_state = online_module.state_dict()
        target_state = target_module.state_dict()
        for key, online_value in online_state.items():
            target_value = target_state[key]
            if torch.is_floating_point(target_value):
                target_value.mul_(momentum).add_(online_value, alpha=1.0 - momentum)
            else:
                target_value.copy_(online_value)

    def online(self, images: torch.Tensor) -> torch.Tensor:
        projections = self.online_projector(self.online_encoder(images))
        return self.predictor(projections)

    @torch.no_grad()
    def target(self, images: torch.Tensor) -> torch.Tensor:
        projections = self.target_projector(self.target_encoder(images))
        return projections.detach()


def byol_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = F.normalize(predictions, dim=1)
    targets = F.normalize(targets, dim=1)
    return 2.0 - 2.0 * (predictions * targets).sum(dim=1).mean()
