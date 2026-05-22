from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    parser = argparse.ArgumentParser(description="Run Grad-CAM sanity checks.")
    parser.add_argument("--split-csv", type=Path, default=Path("metadata/splits_near_duplicate_aware_phash.csv"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec in name=checkpoint.pt format. Repeat for multiple models.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("reports/xai_sanity_checks.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/gradcam_xai_sanity"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
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


def randomize_model(model: nn.Module) -> nn.Module:
    randomized = copy.deepcopy(model)
    for module in randomized.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    randomized.eval()
    return randomized


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

    def __call__(self, image_tensor: torch.Tensor, class_idx: int = 1):
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


def central_mask(size: int, margin_fraction: float = 0.18) -> torch.Tensor:
    mask = torch.ones((3, size, size), dtype=torch.float32)
    margin = int(size * margin_fraction)
    mask[:, :margin, :] = 0
    mask[:, -margin:, :] = 0
    mask[:, :, :margin] = 0
    mask[:, :, -margin:] = 0
    return mask


def border_fraction(cam: np.ndarray, margin_fraction: float = 0.18) -> float:
    size = cam.shape[0]
    margin = int(size * margin_fraction)
    border = np.zeros_like(cam, dtype=bool)
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    total = float(cam.sum() + 1e-8)
    return float(cam[border].sum() / total)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = left.reshape(-1)
    y = right.reshape(-1)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def topk_overlap(left: np.ndarray, right: np.ndarray, fraction: float = 0.10) -> float:
    k = max(1, int(left.size * fraction))
    left_idx = set(np.argpartition(left.reshape(-1), -k)[-k:].tolist())
    right_idx = set(np.argpartition(right.reshape(-1), -k)[-k:].tolist())
    return float(len(left_idx & right_idx) / k)


def main() -> None:
    args = parse_args()
    runtime = configure_runtime(seed=args.seed, cpu_threads=18)
    frame = pd.read_csv(args.split_csv)
    frame = frame[frame["split"] == args.split].copy()
    n = min(args.samples_per_class, frame["label"].value_counts().min())
    samples = (
        frame.groupby("label", group_keys=False)
        .sample(n=n, random_state=args.seed)
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
    mask = central_mask(args.image_size).to(runtime.device)
    rows = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, checkpoint_path in parse_model_specs(args.model):
        model, backbone = load_model(checkpoint_path, runtime.device)
        random_model = randomize_model(model).to(runtime.device)
        cam = GradCAM(model, find_last_conv(model))
        random_cam = GradCAM(random_model, find_last_conv(random_model))

        for sample_idx, row in samples.iterrows():
            image = Image.open(row["rel_path"]).convert("RGB")
            resized = image.resize((args.image_size, args.image_size))
            tensor = preprocess(image).unsqueeze(0).to(runtime.device)
            masked_tensor = tensor * mask.unsqueeze(0)

            trained_heatmap, pred, prob = cam(tensor, class_idx=1)
            randomized_heatmap, random_pred, random_prob = random_cam(tensor, class_idx=1)
            masked_heatmap, masked_pred, masked_prob = cam(masked_tensor, class_idx=1)

            similarity = pearson(trained_heatmap, randomized_heatmap)
            masked_similarity = pearson(trained_heatmap, masked_heatmap)
            trained_border = border_fraction(trained_heatmap)
            masked_border = border_fraction(masked_heatmap)
            rows.append(
                {
                    "model_name": model_name,
                    "backbone": backbone,
                    "image_id": row["image_id"],
                    "rel_path": row["rel_path"],
                    "label": int(row["label"]),
                    "predicted": pred,
                    "pcos_probability": prob,
                    "randomized_predicted": random_pred,
                    "randomized_pcos_probability": random_prob,
                    "masked_predicted": masked_pred,
                    "masked_pcos_probability": masked_prob,
                    "probability_drop_after_border_mask": prob - masked_prob,
                    "trained_vs_random_cam_pearson": similarity,
                    "trained_vs_random_cam_top10_overlap": topk_overlap(
                        trained_heatmap, randomized_heatmap
                    ),
                    "original_vs_masked_cam_pearson": masked_similarity,
                    "trained_cam_border_fraction": trained_border,
                    "masked_cam_border_fraction": masked_border,
                }
            )

            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            axes[0].imshow(resized)
            axes[0].set_title(f"label={int(row['label'])}")
            axes[0].axis("off")
            for ax, heatmap, title in [
                (axes[1], trained_heatmap, f"trained p={prob:.3f}"),
                (axes[2], randomized_heatmap, "randomized"),
                (axes[3], masked_heatmap, f"border masked p={masked_prob:.3f}"),
            ]:
                ax.imshow(resized)
                ax.imshow(heatmap, cmap="jet", alpha=0.45)
                ax.set_title(title)
                ax.axis("off")
            out_path = args.output_dir / f"{model_name}_sanity_{sample_idx:03d}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=160)
            plt.close(fig)

    output = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    Console().print(f"Wrote {args.output_csv}")
    Console().print(f"Wrote sanity panels to {args.output_dir}")
    if not output.empty:
        summary = output.groupby("model_name").agg(
            trained_vs_random_cam_pearson_mean=("trained_vs_random_cam_pearson", "mean"),
            trained_vs_random_cam_top10_overlap_mean=(
                "trained_vs_random_cam_top10_overlap",
                "mean",
            ),
            original_vs_masked_cam_pearson_mean=("original_vs_masked_cam_pearson", "mean"),
            probability_drop_after_border_mask_mean=("probability_drop_after_border_mask", "mean"),
            trained_cam_border_fraction_mean=("trained_cam_border_fraction", "mean"),
        )
        Console().print(summary.to_string())


if __name__ == "__main__":
    main()
