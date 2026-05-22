from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript-facing summary figures.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument(
        "--manifest-csv", type=Path, default=Path("reports/final_figures_manifest.csv")
    )
    return parser.parse_args()


def save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def clean_run_label(run_id: str) -> str:
    label = run_id
    replacements = {
        "resnet18_supervised_phash": "Supervised ResNet-18",
        "simclr_resnet18_phash_e25_finetune": "SimCLR e25",
        "byol_resnet18_phash_e25_finetune": "BYOL e25",
        "efficientnet_b0_phash": "EfficientNet-B0",
        "vit_tiny_patch16_224_phash": "ViT-Tiny",
        "convnext_tiny_phash": "ConvNeXt-Tiny",
        "_seed123": ", seed 123",
        "_seed7": ", seed 7",
        "_10pct": ", 10%",
        "_25pct": ", 25%",
        "_50pct": ", 50%",
        "_e25": ", e25",
        "_e10": ", e10",
        "_e5": ", e5",
        "_e1": ", e1",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.replace("_", " ")


def plot_seed_summary(seed_summary: pd.DataFrame, output_dir: Path) -> Path:
    frame = seed_summary[seed_summary["epochs"].isin([1, 10])].copy()
    frame["label_percent"] = frame["label_fraction"].str.replace("pct", "").astype(int)
    frame["method_label"] = frame["method"].map(
        {
            "supervised_resnet18": "Supervised ResNet-18",
            "simclr_e25_finetune": "SimCLR e25 fine-tune",
            "byol_e25_finetune": "BYOL e25 fine-tune",
        }
    )
    path = output_dir / "seed_summary_accuracy.png"
    plt.figure(figsize=(8, 4.8))
    for (method, epochs), group in frame.groupby(["method_label", "epochs"]):
        group = group.sort_values("label_percent")
        label = f"{method}, {epochs} epoch{'s' if epochs != 1 else ''}"
        plt.errorbar(
            group["label_percent"],
            group["accuracy_mean"],
            yerr=group["accuracy_std"].fillna(0),
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=label,
        )
    plt.xlabel("Label fraction (%)")
    plt.ylabel("Test accuracy")
    plt.ylim(0.85, 1.01)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    save_current(path)
    return path


def plot_label_efficiency(seed_summary: pd.DataFrame, output_dir: Path) -> Path:
    frame = seed_summary[
        (seed_summary["epochs"] == 10)
        & (seed_summary["label_fraction"].isin(["10pct", "50pct"]))
    ].copy()
    frame["label_percent"] = frame["label_fraction"].str.replace("pct", "").astype(int)
    frame["method_label"] = frame["method"].map(
        {
            "supervised_resnet18": "Supervised ResNet-18",
            "simclr_e25_finetune": "SimCLR e25 fine-tune",
            "byol_e25_finetune": "BYOL e25 fine-tune",
        }
    )
    path = output_dir / "label_efficiency_tuned_vs_default.png"
    plt.figure(figsize=(8.2, 4.8))
    for method, group in frame.groupby("method_label"):
        group = group.sort_values("label_percent")
        plt.plot(
            group["label_percent"],
            group["accuracy_mean"],
            marker="o",
            linewidth=1.9,
            label=f"{method}: default",
        )
        plt.plot(
            group["label_percent"],
            group["selected_accuracy_mean"],
            marker="s",
            linestyle="--",
            linewidth=1.9,
            label=f"{method}: tuned",
        )
    plt.xlabel("Label fraction (%)")
    plt.ylabel("Test accuracy")
    plt.ylim(0.93, 1.005)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7.5, ncol=2)
    save_current(path)
    return path


def plot_threshold_ci(threshold_ci: pd.DataFrame, output_dir: Path) -> Path:
    wanted = [
        "resnet18_supervised_phash_10pct_e10",
        "simclr_resnet18_phash_e25_finetune_10pct_e10",
        "byol_resnet18_phash_e25_finetune_10pct_e10",
        "resnet18_supervised_phash_50pct_e10",
        "simclr_resnet18_phash_e25_finetune_50pct_e10",
        "byol_resnet18_phash_e25_finetune_50pct_e10",
        "simclr_resnet18_phash_e25_finetune_50pct_e25",
    ]
    frame = threshold_ci[threshold_ci["run_id"].isin(wanted)].copy()
    frame["label"] = frame["run_id"].str.replace("_", "\n")
    y = range(len(frame))
    x = frame["selected_accuracy"]
    xerr = [
        x - frame["selected_accuracy_ci_low"],
        frame["selected_accuracy_ci_high"] - x,
    ]
    path = output_dir / "threshold_accuracy_ci.png"
    plt.figure(figsize=(8.5, 4.8))
    plt.errorbar(x, list(y), xerr=xerr, fmt="o", capsize=3)
    plt.yticks(list(y), frame["label"], fontsize=7)
    plt.xlabel("Threshold-selected test accuracy with 95% CI")
    plt.xlim(0.93, 1.005)
    plt.grid(axis="x", alpha=0.25)
    save_current(path)
    return path


def plot_bootstrap_auroc_ci(bootstrap: pd.DataFrame, output_dir: Path) -> Path:
    wanted = [
        "resnet18_supervised_phash_10pct_e10",
        "simclr_resnet18_phash_e25_finetune_10pct_e10",
        "byol_resnet18_phash_e25_finetune_10pct_e10",
        "resnet18_supervised_phash_50pct_e10",
        "simclr_resnet18_phash_e25_finetune_50pct_e10",
        "byol_resnet18_phash_e25_finetune_50pct_e10",
        "simclr_resnet18_phash_e25_finetune_50pct_e25",
    ]
    frame = bootstrap[bootstrap["run_id"].isin(wanted)].copy()
    frame["label"] = frame["run_id"].map(clean_run_label)
    frame = frame.sort_values("auroc")
    y = range(len(frame))
    x = frame["auroc"]
    xerr = [x - frame["auroc_ci_low"], frame["auroc_ci_high"] - x]
    path = output_dir / "bootstrap_auroc_ci.png"
    plt.figure(figsize=(8.4, 4.8))
    plt.errorbar(x, list(y), xerr=xerr, fmt="o", capsize=3)
    plt.yticks(list(y), frame["label"], fontsize=7.5)
    plt.xlabel("Bootstrap AUROC with 95% CI")
    plt.xlim(0.985, 1.001)
    plt.grid(axis="x", alpha=0.25)
    save_current(path)
    return path


def plot_calibration_improvement(calibration: pd.DataFrame, output_dir: Path) -> Path:
    frame = calibration.copy()
    frame["short_run"] = frame["run_id"].str.replace("simclr_resnet18_phash_e25_finetune", "simclr")
    frame["short_run"] = frame["short_run"].str.replace("byol_resnet18_phash_e25_finetune", "byol")
    frame["short_run"] = frame["short_run"].str.replace("resnet18_supervised_phash", "resnet")
    path = output_dir / "calibration_brier_improvement.png"
    plt.figure(figsize=(9, 4.8))
    for i, run_id in enumerate(frame["short_run"].unique()):
        group = frame[frame["short_run"] == run_id]
        for method, marker in [("uncalibrated", "o"), ("platt", "s"), ("temperature", "^")]:
            subset = group[group["method"] == method]
            if not subset.empty:
                plt.scatter(i, subset["brier_score"].iloc[0], marker=marker, label=method if i == 0 else None)
    plt.xticks(range(frame["short_run"].nunique()), frame["short_run"].unique(), rotation=35, ha="right")
    plt.ylabel("Brier score")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    save_current(path)
    return path


def plot_xai_sanity(xai: pd.DataFrame, output_dir: Path) -> Path:
    frame = (
        xai.groupby("model_name")
        .agg(
            cam_similarity=("trained_vs_random_cam_pearson", "mean"),
            border_fraction=("trained_cam_border_fraction", "mean"),
            border_mask_probability_increase=(
                "probability_drop_after_border_mask",
                lambda s: (-s).clip(lower=0).mean(),
            ),
        )
        .reset_index()
    )
    frame["label"] = frame["model_name"].str.replace("_", "\n")
    x = range(len(frame))
    width = 0.25
    path = output_dir / "xai_sanity_summary.png"
    plt.figure(figsize=(8.3, 4.8))
    plt.bar([i - width for i in x], frame["cam_similarity"], width, label="Trained-random CAM r")
    plt.bar([i for i in x], frame["border_fraction"], width, label="CAM border fraction")
    plt.bar(
        [i + width for i in x],
        frame["border_mask_probability_increase"],
        width,
        label="Border-mask prob. increase",
    )
    plt.xticks(list(x), frame["label"], fontsize=8)
    plt.ylabel("Mean value")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    save_current(path)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    seed_summary = pd.read_csv(args.reports_root / "seed_summary.csv")
    figures.append(
        {
            "figure": "seed_summary_accuracy",
            "path": str(plot_seed_summary(seed_summary, args.output_dir)),
            "source": "reports/seed_summary.csv",
        }
    )
    figures.append(
        {
            "figure": "label_efficiency_tuned_vs_default",
            "path": str(plot_label_efficiency(seed_summary, args.output_dir)),
            "source": "reports/seed_summary.csv",
        }
    )

    threshold_ci = pd.read_csv(args.reports_root / "threshold_confidence_intervals.csv")
    figures.append(
        {
            "figure": "threshold_accuracy_ci",
            "path": str(plot_threshold_ci(threshold_ci, args.output_dir)),
            "source": "reports/threshold_confidence_intervals.csv",
        }
    )

    bootstrap = pd.read_csv(args.reports_root / "bootstrap_auc_ci.csv")
    figures.append(
        {
            "figure": "bootstrap_auroc_ci",
            "path": str(plot_bootstrap_auroc_ci(bootstrap, args.output_dir)),
            "source": "reports/bootstrap_auc_ci.csv",
        }
    )

    calibration = pd.read_csv(args.reports_root / "calibration_improvement.csv")
    figures.append(
        {
            "figure": "calibration_brier_improvement",
            "path": str(plot_calibration_improvement(calibration, args.output_dir)),
            "source": "reports/calibration_improvement.csv",
        }
    )

    xai = pd.read_csv(args.reports_root / "xai_sanity_checks.csv")
    figures.append(
        {
            "figure": "xai_sanity_summary",
            "path": str(plot_xai_sanity(xai, args.output_dir)),
            "source": "reports/xai_sanity_checks.csv",
        }
    )

    existing = [
        "dataset_sample_panel.png",
        "duplicate_conflict_panel.png",
        "gradcam_comparison_panel.png",
        "gradcam_sanity_panel.png",
        "robustness_severity_accuracy.png",
        "robustness_severity_summary.png",
        "robustness_condition_heatmap.png",
        "calibration_curves.png",
        "cross_label_phash_examples.png",
    ]
    for name in existing:
        path = args.output_dir / name
        if path.exists():
            figures.append({"figure": path.stem, "path": str(path), "source": "existing report"})

    manifest = pd.DataFrame(figures)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest_csv, index=False)
    Console().print(f"Wrote {args.manifest_csv}")
    Console().print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
