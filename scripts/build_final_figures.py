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

    threshold_ci = pd.read_csv(args.reports_root / "threshold_confidence_intervals.csv")
    figures.append(
        {
            "figure": "threshold_accuracy_ci",
            "path": str(plot_threshold_ci(threshold_ci, args.output_dir)),
            "source": "reports/threshold_confidence_intervals.csv",
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

    existing = [
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
