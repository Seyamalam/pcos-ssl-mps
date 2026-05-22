from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from rich.console import Console
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from pcos_ssl.data.ssl_dataset import ContrastivePCOSDataset
from pcos_ssl.ssl.byol import BYOLModel, byol_loss
from pcos_ssl.utils.runtime import configure_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a BYOL encoder on PCOS ultrasound images.")
    parser.add_argument("--config", type=Path, default=Path("configs/byol.yaml"))
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/byol_smoke"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_byol_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=12)], p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.0)],
                p=0.8,
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def export_encoder_state(model: BYOLModel) -> dict[str, torch.Tensor]:
    return {
        f"encoder.{key}": value.detach().cpu()
        for key, value in model.online_encoder.state_dict().items()
    }


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

    dataset = ContrastivePCOSDataset.from_split_csv(
        Path(config["data"]["split_csv"]),
        split="train",
        transform=build_byol_transform(int(config["data"]["image_size"])),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=False,
        persistent_workers=int(config["training"]["num_workers"]) > 0,
        drop_last=True,
    )

    model = BYOLModel(
        backbone=str(config["model"]["backbone"]),
        projection_dim=int(config["model"]["projection_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        predictor_hidden_dim=int(config["model"]["predictor_hidden_dim"]),
        pretrained=bool(config["model"].get("pretrained", False)),
    ).to(runtime.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    momentum = float(config["training"]["momentum"])

    console = Console()
    console.print(
        f"Runtime: torch={runtime.torch_version}, device={runtime.device_name}, "
        f"cpu_threads={runtime.cpu_threads}, train={len(dataset):,}"
    )

    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        progress = tqdm(loader, desc=f"byol epoch {epoch}", leave=False)
        for batch_idx, (view1, view2) in enumerate(progress, start=1):
            view1 = view1.to(runtime.device)
            view2 = view2.to(runtime.device)

            optimizer.zero_grad(set_to_none=True)
            pred1 = model.online(view1)
            pred2 = model.online(view2)
            target1 = model.target(view1)
            target2 = model.target(view2)
            loss = 0.5 * (byol_loss(pred1, target2) + byol_loss(pred2, target1))
            loss.backward()
            optimizer.step()
            model.update_target(momentum)

            batch_size = view1.size(0)
            total_loss += loss.item() * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{total_loss / seen:.4f}")
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

        row = {"epoch": epoch, "train_loss": total_loss / max(seen, 1)}
        history.append(row)
        console.print(row)

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    checkpoint_path = args.output_dir / "encoder_last.pt"
    torch.save(
        {"model": export_encoder_state(model), "config": config, "ssl_method": "byol"},
        checkpoint_path,
    )
    console.print(f"Wrote {metrics_path}")
    console.print(f"Wrote {checkpoint_path}")


if __name__ == "__main__":
    main()
