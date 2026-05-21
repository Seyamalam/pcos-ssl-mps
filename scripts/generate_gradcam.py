from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from pcos_ssl.models.factory import create_classifier
from pcos_ssl.utils.runtime import configure_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM overlays for a classifier checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, default=Path("metadata/splits_duplicate_aware.csv"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gradcam"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--samples-per-class", type=int, default=8)
    return parser.parse_args()


def find_last_conv(module: nn.Module) -> nn.Module:
    last_conv = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise ValueError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        logits[:, class_idx].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy()


def main() -> None:
    args = parse_args()
    runtime = configure_runtime(seed=42, cpu_threads=18)
    checkpoint = torch.load(args.checkpoint, map_location=runtime.device)
    config = checkpoint.get("config", {})
    backbone = checkpoint.get("backbone") or config.get("model", {}).get("backbone", "resnet18")
    model = create_classifier(backbone=backbone, num_classes=2, pretrained=False).to(runtime.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    cam = GradCAM(model, find_last_conv(model))

    frame = pd.read_csv(args.split_csv)
    frame = frame[frame["split"] == args.split].copy()
    samples = (
        frame.groupby("label", group_keys=False)
        .sample(n=min(args.samples_per_class, frame["label"].value_counts().min()), random_state=42)
        .reset_index(drop=True)
    )

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in samples.iterrows():
        image = Image.open(row["rel_path"]).convert("RGB")
        resized = image.resize((args.image_size, args.image_size))
        tensor = preprocess(image).unsqueeze(0).to(runtime.device)
        with torch.no_grad():
            predicted = int(torch.softmax(model(tensor), dim=1).argmax(dim=1).item())
        heatmap = cam(tensor, class_idx=predicted)

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        axes[0].imshow(resized, cmap="gray")
        axes[0].set_title(f"label={int(row['label'])}")
        axes[0].axis("off")
        axes[1].imshow(resized, cmap="gray")
        axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[1].set_title(f"pred={predicted}")
        axes[1].axis("off")
        out_path = args.output_dir / f"gradcam_{idx:03d}_label{int(row['label'])}_pred{predicted}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    print(f"Wrote Grad-CAM images to {args.output_dir}")


if __name__ == "__main__":
    main()
