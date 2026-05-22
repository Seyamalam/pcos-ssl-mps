from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import ImageFilter
from rich.console import Console
from torch.utils.data import DataLoader
from torchvision import transforms

from pcos_ssl.data.dataset import PCOSImageDataset
from pcos_ssl.models.factory import create_classifier
from pcos_ssl.models.ssl_classifier import EncoderClassifier
from pcos_ssl.training.metrics import binary_classification_metrics
from pcos_ssl.utils.runtime import configure_runtime


class AddGaussianNoise:
    def __init__(self, std: float = 0.05) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, min=0.0, max=1.0)


class PILBlur:
    def __init__(self, radius: float = 1.5) -> None:
        self.radius = radius

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(radius=self.radius))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robustness of a supervised checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=Path("metadata/splits_duplicate_aware.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/robustness.json"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--severity-sweep",
        action="store_true",
        help="Evaluate multiple corruption severities for manuscript robustness curves.",
    )
    return parser.parse_args()


def normalize():
    return transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def robustness_transforms(image_size: int) -> dict[str, transforms.Compose]:
    return {
        "clean_resize": transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize()]
        ),
        "center_crop": transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize(),
            ]
        ),
        "low_contrast": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(contrast=(0.55, 0.55)),
                transforms.ToTensor(),
                normalize(),
            ]
        ),
        "high_contrast": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(contrast=(1.5, 1.5)),
                transforms.ToTensor(),
                normalize(),
            ]
        ),
        "blur": transforms.Compose(
            [transforms.Resize((image_size, image_size)), PILBlur(radius=1.5), transforms.ToTensor(), normalize()]
        ),
        "downsample": transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize(),
            ]
        ),
        "gaussian_noise": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                AddGaussianNoise(std=0.05),
                normalize(),
            ]
        ),
    }


def robustness_severity_transforms(image_size: int) -> dict[str, tuple[str, float, transforms.Compose]]:
    cases: dict[str, tuple[str, float, transforms.Compose]] = {
        "clean_resize": (
            "clean_resize",
            0.0,
            transforms.Compose(
                [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize()]
            ),
        )
    }

    for factor in [1.05, 1.15, 1.30]:
        cases[f"center_crop_zoom_{factor:.2f}"] = (
            "center_crop",
            factor,
            transforms.Compose(
                [
                    transforms.Resize(int(image_size * factor)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize(),
                ]
            ),
        )

    for contrast in [0.75, 0.55, 0.35]:
        cases[f"low_contrast_{contrast:.2f}"] = (
            "low_contrast",
            contrast,
            transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ColorJitter(contrast=(contrast, contrast)),
                    transforms.ToTensor(),
                    normalize(),
                ]
            ),
        )

    for contrast in [1.25, 1.50, 2.00]:
        cases[f"high_contrast_{contrast:.2f}"] = (
            "high_contrast",
            contrast,
            transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ColorJitter(contrast=(contrast, contrast)),
                    transforms.ToTensor(),
                    normalize(),
                ]
            ),
        )

    for radius in [0.5, 1.5, 3.0]:
        cases[f"blur_radius_{radius:.1f}"] = (
            "blur",
            radius,
            transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    PILBlur(radius=radius),
                    transforms.ToTensor(),
                    normalize(),
                ]
            ),
        )

    for size in [160, 96, 48]:
        cases[f"downsample_{size}"] = (
            "downsample",
            float(size),
            transforms.Compose(
                [
                    transforms.Resize((size, size)),
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    normalize(),
                ]
            ),
        )

    for std in [0.02, 0.05, 0.10]:
        cases[f"gaussian_noise_{std:.2f}"] = (
            "gaussian_noise",
            std,
            transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    AddGaussianNoise(std=std),
                    normalize(),
                ]
            ),
        )

    return cases


def load_model_from_checkpoint(checkpoint_path: Path, device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    backbone = checkpoint.get("backbone") or config.get("model", {}).get("backbone", "resnet18")
    state_dict = checkpoint["model"]
    if any(key.startswith("encoder.") for key in state_dict):
        model = EncoderClassifier(backbone=backbone, num_classes=2, pretrained=False)
    else:
        model = create_classifier(backbone=backbone, num_classes=2, pretrained=False)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.inference_mode()
def evaluate_loader(model, loader, device) -> dict:
    y_true: list[int] = []
    y_prob: list[float] = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(labels.tolist())
        y_prob.extend(probs.cpu().tolist())
    return binary_classification_metrics(y_true, y_prob)


def main() -> None:
    args = parse_args()
    runtime = configure_runtime(seed=42, cpu_threads=18)
    model = load_model_from_checkpoint(args.checkpoint, runtime.device)

    if args.severity_sweep:
        transform_items = robustness_severity_transforms(args.image_size)
    else:
        transform_items = {
            name: (name, 0.0, transform)
            for name, transform in robustness_transforms(args.image_size).items()
        }

    results = {}
    for name, (condition, severity, transform) in transform_items.items():
        dataset = PCOSImageDataset.from_split_csv(args.split_csv, "test", transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        results[name] = {
            "condition": condition,
            "severity": severity,
            **evaluate_loader(model, loader, runtime.device),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "split_csv": str(args.split_csv),
                "severity_sweep": bool(args.severity_sweep),
                "results": results,
            },
            handle,
            indent=2,
        )
    Console().print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
