from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from rich.console import Console
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from pcos_ssl.data.dataset import PCOSImageDataset
from pcos_ssl.models.factory import create_classifier
from pcos_ssl.training.metrics import binary_classification_metrics
from pcos_ssl.utils.runtime import configure_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised PCOS classifier baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/supervised_smoke"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transform, eval_transform


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches: int | None) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    progress = tqdm(loader, desc="train", leave=False)
    for batch_idx, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
        progress.set_postfix(loss=f"{total_loss / seen:.4f}")
        if max_batches is not None and batch_idx >= max_batches:
            break
    return total_loss / max(seen, 1)


@torch.inference_mode()
def evaluate(model, loader, criterion, device, max_batches: int | None) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    seen = 0
    y_true: list[int] = []
    y_prob: list[float] = []

    progress = tqdm(loader, desc="eval", leave=False)
    for batch_idx, (images, labels) in enumerate(progress, start=1):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        seen += batch_size
        y_true.extend(labels.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
        if max_batches is not None and batch_idx >= max_batches:
            break

    metrics = binary_classification_metrics(y_true, y_prob)
    return total_loss / max(seen, 1), metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = configure_runtime(
        seed=int(config["seed"]),
        cpu_threads=int(config["runtime"].get("cpu_threads", 18)),
    )

    epochs = args.epochs if args.epochs is not None else int(config["training"]["epochs"])
    pretrained = (
        args.pretrained
        if args.pretrained is not None
        else bool(config["model"].get("pretrained", True))
    )

    train_transform, eval_transform = build_transforms(int(config["data"]["image_size"]))
    split_csv = Path(config["data"]["split_csv"])
    train_dataset = PCOSImageDataset.from_split_csv(split_csv, "train", transform=train_transform)
    val_dataset = PCOSImageDataset.from_split_csv(split_csv, "val", transform=eval_transform)

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)

    model = create_classifier(
        backbone=str(config["model"]["backbone"]),
        num_classes=2,
        pretrained=pretrained,
    ).to(runtime.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    console = Console()
    console.print(
        f"Runtime: torch={runtime.torch_version}, device={runtime.device_name}, "
        f"cpu_threads={runtime.cpu_threads}, train={len(train_dataset):,}, val={len(val_dataset):,}"
    )

    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            runtime.device,
            args.max_train_batches,
        )
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            runtime.device,
            args.max_val_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        }
        history.append(row)
        console.print(row)

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    console.print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()

