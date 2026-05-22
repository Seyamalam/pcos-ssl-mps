from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from rich.console import Console


RUN_PATTERN = re.compile(
    r"^(?P<family>resnet18_supervised_phash|simclr_resnet18_phash_e25_finetune|byol_resnet18_phash_e25_finetune)_"
    r"(?P<label>\d+pct)(?:_seed(?P<seed>\d+))?_e(?P<epochs>\d+)$"
)

METHODS = {
    "resnet18_supervised_phash": "supervised_resnet18",
    "simclr_resnet18_phash_e25_finetune": "simclr_e25_finetune",
    "byol_resnet18_phash_e25_finetune": "byol_e25_finetune",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated-seed label-efficiency runs.")
    parser.add_argument("--results-csv", type=Path, default=Path("reports/all_test_results.csv"))
    parser.add_argument(
        "--threshold-csv", type=Path, default=Path("reports/threshold_summary.csv")
    )
    parser.add_argument("--output-csv", type=Path, default=Path("reports/seed_summary.csv"))
    return parser.parse_args()


def add_run_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        match = RUN_PATTERN.match(str(row["run_id"]))
        if not match:
            continue
        label = match.group("label")
        if label not in {"10pct", "25pct", "50pct"}:
            continue
        method = METHODS[match.group("family")]
        seed = int(match.group("seed") or 42)
        enriched = row.to_dict()
        enriched.update(
            {
                "method": method,
                "label_fraction": label,
                "seed": seed,
                "epochs": int(match.group("epochs")),
            }
        )
        rows.append(enriched)
    return pd.DataFrame(rows)


def summarize(group: pd.DataFrame, metric: str) -> dict[str, float]:
    values = group[metric].dropna()
    return {
        f"{metric}_mean": float(values.mean()),
        f"{metric}_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    results = add_run_metadata(pd.read_csv(args.results_csv))
    thresholds = add_run_metadata(
        pd.read_csv(args.threshold_csv).query("objective == 'balanced_accuracy'")
    )

    rows = []
    for (method, label_fraction, epochs), group in results.groupby(
        ["method", "label_fraction", "epochs"]
    ):
        threshold_group = thresholds[
            (thresholds["method"] == method)
            & (thresholds["label_fraction"] == label_fraction)
            & (thresholds["epochs"] == epochs)
        ]
        row = {
            "method": method,
            "label_fraction": label_fraction,
            "epochs": int(epochs),
            "n_seeds": int(group["seed"].nunique()),
            "seeds": " ".join(str(seed) for seed in sorted(group["seed"].unique())),
        }
        for metric in ["accuracy", "balanced_accuracy", "recall_sensitivity", "auroc", "f1", "ece"]:
            row.update(summarize(group, metric))
        for metric in [
            "selected_accuracy",
            "selected_balanced_accuracy",
            "selected_sensitivity",
            "selected_specificity",
        ]:
            row.update(summarize(threshold_group, metric))
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["label_fraction", "epochs", "method"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    Console().print(f"Wrote {args.output_csv}")
    if not summary.empty:
        columns = [
            "method",
            "label_fraction",
            "epochs",
            "n_seeds",
            "accuracy_mean",
            "accuracy_std",
            "auroc_mean",
            "auroc_std",
            "selected_accuracy_mean",
            "selected_accuracy_std",
            "selected_sensitivity_mean",
        ]
        Console().print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
