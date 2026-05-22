from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final manuscript-ready tables from reports.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--metadata-root", type=Path, default=Path("metadata"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/manuscript_tables"))
    parser.add_argument(
        "--output-md", type=Path, default=Path("reports/manuscript_tables.md")
    )
    return parser.parse_args()


def write_table(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def markdown_table(title: str, frame: pd.DataFrame) -> str:
    text_frame = frame.fillna("").astype(str)
    columns = list(text_frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in text_frame.iterrows():
        values = [str(row[column]).replace("|", "/") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return f"## {title}\n\n" + "\n".join(lines) + "\n\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = pd.read_csv(args.metadata_root / "images.csv")
    splits = pd.read_csv(args.metadata_root / "splits_near_duplicate_aware_phash.csv")
    near = pd.read_csv(args.metadata_root / "near_duplicate_groups_phash.csv")

    dataset_table = pd.DataFrame(
        [
            {"metric": "Total readable images", "value": len(images)},
            {"metric": "PCOS-positive images", "value": int((images["label"] == 1).sum())},
            {"metric": "Healthy/non-PCOS images", "value": int((images["label"] == 0).sum())},
            {
                "metric": "Exact duplicate groups",
                "value": int(images.groupby("exact_group_id").size().gt(1).sum()),
            },
            {
                "metric": "Exact duplicate files beyond first",
                "value": int(len(images) - images["exact_group_id"].nunique()),
            },
            {
                "metric": "pHash near-duplicate groups",
                "value": int(near.groupby("near_group_id").size().gt(1).sum()),
            },
            {
                "metric": "pHash near-duplicate files beyond first",
                "value": int(len(near) - near["near_group_id"].nunique()),
            },
            {
                "metric": "Cross-label pHash near-duplicate groups",
                "value": int(near.groupby("near_group_id")["label"].nunique().gt(1).sum()),
            },
            {"metric": "pHash-aware train images", "value": int((splits["split"] == "train").sum())},
            {"metric": "pHash-aware validation images", "value": int((splits["split"] == "val").sum())},
            {"metric": "pHash-aware test images", "value": int((splits["split"] == "test").sum())},
        ]
    )

    results = pd.read_csv(args.reports_root / "all_test_results.csv")
    baseline_ids = [
        "resnet18_supervised_phash_e1",
        "efficientnet_b0_supervised_phash_e1",
        "vit_tiny_patch16_224_supervised_phash_e1",
        "convnext_tiny_supervised_phash_e1",
    ]
    baseline_table = results[results["run_id"].isin(baseline_ids)][
        ["run_id", "accuracy", "auroc", "auprc", "f1", "ece"]
    ].sort_values("accuracy", ascending=False)

    downstream = pd.read_csv(args.reports_root / "downstream_epoch_summary.csv")
    epoch_table = downstream[
        [
            "method",
            "label_fraction",
            "downstream_epochs",
            "default_accuracy",
            "auroc",
            "selected_accuracy",
            "selected_sensitivity",
            "selected_specificity",
        ]
    ]

    threshold_ci = pd.read_csv(args.reports_root / "threshold_confidence_intervals.csv")
    main_threshold_ids = [
        "resnet18_supervised_phash_50pct_e5",
        "simclr_resnet18_phash_e25_finetune_50pct_e10",
        "byol_resnet18_phash_e25_finetune_50pct_e10",
        "simclr_resnet18_phash_e25_finetune_50pct_e25",
    ]
    threshold_table = threshold_ci[threshold_ci["run_id"].isin(main_threshold_ids)][
        [
            "run_id",
            "selected_accuracy",
            "selected_accuracy_ci_low",
            "selected_accuracy_ci_high",
            "selected_sensitivity",
            "selected_sensitivity_ci_low",
            "selected_sensitivity_ci_high",
            "selected_specificity",
            "selected_specificity_ci_low",
            "selected_specificity_ci_high",
        ]
    ]

    bootstrap = pd.read_csv(args.reports_root / "bootstrap_auc_ci.csv")
    bootstrap_table = bootstrap[
        ["run_id", "auroc", "auroc_ci_low", "auroc_ci_high", "auprc", "auprc_ci_low", "auprc_ci_high"]
    ]

    robustness = pd.read_csv(args.reports_root / "robustness_severity_summary.csv")
    robustness_table = robustness[
        [
            "run_id",
            "clean_accuracy",
            "mean_corruption_accuracy",
            "worst_corruption_accuracy",
            "mean_accuracy_degradation",
            "worst_accuracy_degradation",
        ]
    ].sort_values("mean_corruption_accuracy", ascending=False)

    calibration = pd.read_csv(args.reports_root / "calibration_summary.csv")
    calibration_table = calibration[["run_id", "brier_score", "ece"]].sort_values("brier_score")

    seed_summary = pd.read_csv(args.reports_root / "seed_summary.csv")
    seed_table = seed_summary[
        [
            "method",
            "label_fraction",
            "epochs",
            "n_seeds",
            "seeds",
            "accuracy_mean",
            "accuracy_std",
            "auroc_mean",
            "auroc_std",
            "selected_accuracy_mean",
            "selected_accuracy_std",
            "selected_sensitivity_mean",
            "selected_specificity_mean",
        ]
    ]

    cross_label = pd.read_csv(args.reports_root / "cross_label_phash_groups.csv")
    cross_label_table = cross_label[
        [
            "near_group_id",
            "group_size",
            "infected_count",
            "noninfected_count",
            "min_cross_label_phash_distance",
            "representative_infected",
            "representative_noninfected",
        ]
    ].head(10)

    tables = {
        "dataset_audit": dataset_table,
        "baseline_architectures": baseline_table,
        "downstream_epoch_sensitivity": epoch_table,
        "threshold_operating_points_ci": threshold_table,
        "bootstrap_ranking_ci": bootstrap_table,
        "calibration": calibration_table,
        "multi_seed_summary": seed_table,
        "robustness_severity": robustness_table,
        "cross_label_phash_examples": cross_label_table,
    }
    for name, frame in tables.items():
        write_table(frame, args.output_dir / f"{name}.csv")

    sections = [
        markdown_table("Dataset Audit", dataset_table),
        markdown_table("Baseline Architectures", baseline_table),
        markdown_table("Downstream Epoch Sensitivity", epoch_table),
        markdown_table("Threshold Operating Points With Wilson CIs", threshold_table),
        markdown_table("Bootstrap AUROC/AUPRC CIs", bootstrap_table),
        markdown_table("Calibration", calibration_table),
        markdown_table("Multi-Seed Summary", seed_table),
        markdown_table("Robustness Severity", robustness_table),
        markdown_table("Cross-Label pHash Examples", cross_label_table),
    ]
    args.output_md.write_text("# Manuscript Tables\n\n" + "".join(sections), encoding="utf-8")
    Console().print(f"Wrote {args.output_dir}")
    Console().print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
