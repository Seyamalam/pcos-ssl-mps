from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from rich.console import Console
from torchvision import transforms

from pcos_ssl.models.factory import create_classifier
from pcos_ssl.models.ssl_classifier import EncoderClassifier
from pcos_ssl.utils.runtime import configure_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate side-by-side Grad-CAM XAI comparison panels.")
    parser.add_argument("--split-csv", type=Path, default=Path("metadata/splits_near_duplicate_aware_phash.csv"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec in name=checkpoint.pt format. Repeat for multiple models.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gradcam_xai_comparison"))
    parser.add_argument("--manifest-csv", type=Path, default=Path("reports/xai_comparison_manifest.csv"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--samples-per-class", type=int, default=4)
    return parser.parse_args()


def parse_model_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be name=checkpoint.pt, got {spec}")
        name, path = spec.split("=", 1)
        parsed.append((name, Path(path)))
    return parsed


def find_last_conv(module: nn.Module) -> nn.Module:
    last_conv = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    if last_conv is None:
        raise ValueError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv


def load_model(checkpoint_path: Path, device) -> tuple[nn.Module, str]:
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
    return model, backbone


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: int):
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
        )[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probability = torch.softmax(logits, dim=1)[0, 1].detach().cpu().item()
        predicted = int(torch.argmax(logits, dim=1).detach().cpu().item())
        return cam.detach().cpu().numpy(), predicted, float(probability)


def main() -> None:
    args = parse_args()
    runtime = configure_runtime(seed=42, cpu_threads=18)
    model_specs = parse_model_specs(args.model)
    models = []
    for name, checkpoint_path in model_specs:
        model, backbone = load_model(checkpoint_path, runtime.device)
        models.append((name, backbone, model, GradCAM(model, find_last_conv(model))))

    frame = pd.read_csv(args.split_csv)
    frame = frame[frame["split"] == args.split].copy()
    n = min(args.samples_per_class, frame["label"].value_counts().min())
    samples = (
        frame.groupby("label", group_keys=False)
        .sample(n=n, random_state=42)
        .sort_values(["label", "rel_path"])
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
    manifest = []
    for idx, row in samples.iterrows():
        image = Image.open(row["rel_path"]).convert("RGB")
        resized = image.resize((args.image_size, args.image_size))
        tensor = preprocess(image).unsqueeze(0).to(runtime.device)
        fig, axes = plt.subplots(1, len(models) + 1, figsize=(3.2 * (len(models) + 1), 3.2))
        axes[0].imshow(resized)
        axes[0].set_title(f"label={int(row['label'])}")
        axes[0].axis("off")
        panel_records = []
        for ax, (name, backbone, _model, cam_runner) in zip(axes[1:], models):
            heatmap, predicted, probability = cam_runner(tensor, class_idx=1)
            ax.imshow(resized)
            ax.imshow(heatmap, cmap="jet", alpha=0.45)
            ax.set_title(f"{name}\npred={predicted} p={probability:.3f}")
            ax.axis("off")
            panel_records.append(
                {
                    "model_name": name,
                    "backbone": backbone,
                    "predicted": predicted,
                    "pcos_probability": probability,
                }
            )
        out_path = args.output_dir / f"xai_{idx:03d}_label{int(row['label'])}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        for record in panel_records:
            manifest.append(
                {
                    "image_id": row["image_id"],
                    "rel_path": row["rel_path"],
                    "label": int(row["label"]),
                    "class_name": row["class_name"],
                    "output_path": str(out_path),
                    **record,
                }
            )

    manifest_frame = pd.DataFrame(manifest)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_frame.to_csv(args.manifest_csv, index=False)
    Console().print(f"Wrote XAI panels to {args.output_dir}")
    Console().print(f"Wrote {args.manifest_csv}")


if __name__ == "__main__":
    main()
