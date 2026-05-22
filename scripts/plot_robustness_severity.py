from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot robustness severity results.")
    parser.add_argument("--long-csv", type=Path, default=Path("reports/robustness_severity_long.csv"))
    parser.add_argument(
        "--summary-csv", type=Path, default=Path("reports/robustness_severity_summary.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    long = pd.read_csv(args.long_csv)
    summary = pd.read_csv(args.summary_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_frame = long[long["condition"] != "clean_resize"].copy()
    sns.lineplot(
        data=plot_frame,
        x="severity",
        y="accuracy",
        hue="run_id",
        style="condition",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_title("Robustness severity sweep: accuracy")
    ax.set_xlabel("Severity value")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    severity_path = args.output_dir / "robustness_severity_accuracy.png"
    fig.savefig(severity_path, dpi=180)
    plt.close(fig)

    melted = summary.melt(
        id_vars=["run_id"],
        value_vars=["clean_accuracy", "mean_corruption_accuracy", "worst_corruption_accuracy"],
        var_name="metric",
        value_name="accuracy",
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=melted, x="run_id", y="accuracy", hue="metric", ax=ax)
    ax.set_title("Clean vs mean/worst corruption accuracy")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    summary_path = args.output_dir / "robustness_severity_summary.png"
    fig.savefig(summary_path, dpi=180)
    plt.close(fig)

    condition_summary = (
        plot_frame.groupby(["run_id", "condition"], as_index=False)["accuracy"].mean()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        condition_summary.pivot(index="run_id", columns="condition", values="accuracy"),
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.45,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title("Mean accuracy by corruption type")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    heatmap_path = args.output_dir / "robustness_condition_heatmap.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)

    Console().print(f"Wrote {severity_path}")
    Console().print(f"Wrote {summary_path}")
    Console().print(f"Wrote {heatmap_path}")


if __name__ == "__main__":
    main()
