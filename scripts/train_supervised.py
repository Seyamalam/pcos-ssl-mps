from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from rich.console import Console
from torch.utils.data import DataLoader

from pcos_ssl.data.dataset import PCOSImageDataset
from pcos_ssl.data.transforms import build_supervised_transforms
from pcos_ssl.models.factory import create_classifier
from pcos_ssl.training.loops import evaluate, train_one_epoch
from pcos_ssl.utils.runtime import configure_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised PCOS classifier baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/supervised_smoke"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    config = copy.deepcopy(load_config(args.config))
    if args.split_csv is not None:
        config["data"]["split_csv"] = str(args.split_csv)
    if args.backbone is not None:
        config["model"]["backbone"] = args.backbone
    if args.image_size is not None:
        config["data"]["image_size"] = args.image_size
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.seed is not None:
        config["seed"] = args.seed
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

    train_transform, eval_transform = build_supervised_transforms(int(config["data"]["image_size"]))
    split_csv = Path(config["data"]["split_csv"])
    train_dataset = PCOSImageDataset.from_split_csv(
        split_csv,
        "train",
        transform=train_transform,
        label_fraction=args.label_fraction,
        seed=int(config["seed"]),
    )
    val_dataset = PCOSImageDataset.from_split_csv(split_csv, "val", transform=eval_transform)
    test_dataset = PCOSImageDataset.from_split_csv(split_csv, "test", transform=eval_transform)

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = make_loader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

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
        f"cpu_threads={runtime.cpu_threads}, train={len(train_dataset):,}, "
        f"val={len(val_dataset):,}, test={len(test_dataset):,}, "
        f"label_fraction={args.label_fraction}"
    )

    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    best_epoch = None
    best_checkpoint_path = args.output_dir / "best_model.pt"
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
        score = val_metrics.get("auroc")
        if score is None:
            score = val_metrics.get("balanced_accuracy")
        if score is None:
            score = -val_loss
        if score is not None and float(score) > best_score:
            best_score = float(score)
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "best_score": best_score,
                    "label_fraction": args.label_fraction,
                    "backbone": str(config["model"]["backbone"]),
                },
                best_checkpoint_path,
            )

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=runtime.device)
        model.load_state_dict(checkpoint["model"])
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        runtime.device,
        args.max_test_batches,
    )
    test_result = {"best_epoch": best_epoch, "test_loss": test_loss, **test_metrics}

    metrics_path = args.output_dir / "metrics.json"
    test_metrics_path = args.output_dir / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with test_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(test_result, handle, indent=2)
    console.print(f"Wrote {metrics_path}")
    console.print(f"Wrote {test_metrics_path}")
    console.print(f"Wrote {best_checkpoint_path}")


if __name__ == "__main__":
    main()
