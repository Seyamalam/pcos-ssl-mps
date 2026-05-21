from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rich.console import Console
from torch.utils.data import DataLoader

from pcos_ssl.data.dataset import PCOSImageDataset
from pcos_ssl.data.transforms import build_supervised_transforms
from pcos_ssl.models.factory import create_classifier
from pcos_ssl.models.ssl_classifier import EncoderClassifier
from pcos_ssl.training.metrics import binary_classification_metrics, select_thresholds
from pcos_ssl.utils.runtime import configure_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select thresholds on validation data and evaluate locked thresholds on test."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=Path("metadata/splits_duplicate_aware.csv"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-sensitivity", type=float, default=0.90)
    return parser.parse_args()


def make_loader(split_csv: Path, split: str, image_size: int, batch_size: int, num_workers: int):
    _, eval_transform = build_supervised_transforms(image_size)
    dataset = PCOSImageDataset.from_split_csv(split_csv, split, transform=eval_transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )


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
def predict_probabilities(model, loader, device) -> tuple[list[int], list[float]]:
    y_true: list[int] = []
    y_prob: list[float] = []
    for images, labels in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1]
        y_true.extend(labels.tolist())
        y_prob.extend(probs.cpu().tolist())
    return y_true, y_prob


def evaluate_selected_thresholds(
    y_true: list[int],
    y_prob: list[float],
    selections: dict[str, dict],
) -> dict[str, dict]:
    results = {}
    for objective, selection in selections.items():
        threshold = float(selection["selected_threshold"])
        results[objective] = binary_classification_metrics(y_true, y_prob, threshold=threshold)
    return results


def main() -> None:
    args = parse_args()
    runtime = configure_runtime(seed=42, cpu_threads=18)
    model = load_model_from_checkpoint(args.checkpoint, runtime.device)

    val_loader = make_loader(
        args.split_csv, "val", args.image_size, args.batch_size, args.num_workers
    )
    test_loader = make_loader(
        args.split_csv, "test", args.image_size, args.batch_size, args.num_workers
    )
    val_true, val_prob = predict_probabilities(model, val_loader, runtime.device)
    test_true, test_prob = predict_probabilities(model, test_loader, runtime.device)

    val_default = binary_classification_metrics(val_true, val_prob)
    test_default = binary_classification_metrics(test_true, test_prob)
    val_selected = select_thresholds(
        val_true, val_prob, min_sensitivity=float(args.min_sensitivity)
    )
    test_selected = evaluate_selected_thresholds(test_true, test_prob, val_selected)
    result = {
        "checkpoint": str(args.checkpoint),
        "split_csv": str(args.split_csv),
        "validation_default": val_default,
        "test_default": test_default,
        "validation_selected": val_selected,
        "test_at_validation_selected_thresholds": test_selected,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    Console().print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
