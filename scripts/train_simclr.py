from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from rich.console import Console
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from pcos_ssl.data.ssl_dataset import ContrastivePCOSDataset
from pcos_ssl.ssl.simclr import SimCLRModel, nt_xent_loss
from pcos_ssl.utils.runtime import configure_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a SimCLR encoder on PCOS ultrasound images.")
    parser.add_argument("--config", type=Path, default=Path("configs/simclr.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/simclr_smoke"))
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_simclr_transform(image_size: int):
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = configure_runtime(
        seed=int(config["seed"]),
        cpu_threads=int(config["runtime"].get("cpu_threads", 18)),
    )
    epochs = args.epochs if args.epochs is not None else int(config["training"]["epochs"])

    dataset = ContrastivePCOSDataset.from_split_csv(
        Path(config["data"]["split_csv"]),
        split="train",
        transform=build_simclr_transform(int(config["data"]["image_size"])),
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

    model = SimCLRModel(
        backbone=str(config["model"]["backbone"]),
        projection_dim=int(config["model"]["projection_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        pretrained=bool(config["model"].get("pretrained", False)),
    ).to(runtime.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    temperature = float(config["training"]["temperature"])

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
        progress = tqdm(loader, desc=f"simclr epoch {epoch}", leave=False)
        for batch_idx, (view1, view2) in enumerate(progress, start=1):
            view1 = view1.to(runtime.device)
            view2 = view2.to(runtime.device)

            optimizer.zero_grad(set_to_none=True)
            z1 = model(view1)
            z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            loss.backward()
            optimizer.step()

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
    torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
    console.print(f"Wrote {metrics_path}")
    console.print(f"Wrote {checkpoint_path}")


if __name__ == "__main__":
    main()

