from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect robustness severity-sweep JSON files.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument(
        "--output-long-csv",
        type=Path,
        default=Path("reports/robustness_severity_long.csv"),
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=Path("reports/robustness_severity_summary.csv"),
    )
    return parser.parse_args()


def run_id_from_path(path: Path) -> str:
    return path.stem.removeprefix("robustness_severity_")


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", payload)
    rows = []
    for transform_name, metrics in results.items():
        rows.append(
            {
                "run_id": run_id_from_path(path),
                "transform": transform_name,
                "condition": metrics.get("condition", transform_name),
                "severity": metrics.get("severity", 0.0),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "recall_sensitivity": metrics["recall_sensitivity"],
                "specificity": metrics["specificity"],
                "ece": metrics["ece"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
            }
        )
    return rows


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_id, group in frame.groupby("run_id"):
        clean_rows = group[group["condition"] == "clean_resize"]
        clean_accuracy = float(clean_rows["accuracy"].iloc[0]) if not clean_rows.empty else float("nan")
        corrupt = group[group["condition"] != "clean_resize"]
        rows.append(
            {
                "run_id": run_id,
                "clean_accuracy": clean_accuracy,
                "mean_corruption_accuracy": float(corrupt["accuracy"].mean()),
                "worst_corruption_accuracy": float(corrupt["accuracy"].min()),
                "mean_accuracy_degradation": float(clean_accuracy - corrupt["accuracy"].mean()),
                "worst_accuracy_degradation": float(clean_accuracy - corrupt["accuracy"].min()),
                "mean_corruption_auroc": float(corrupt["auroc"].mean()),
                "worst_corruption_auroc": float(corrupt["auroc"].min()),
            }
        )
        for condition, condition_group in corrupt.groupby("condition"):
            rows[-1][f"{condition}_mean_accuracy"] = float(condition_group["accuracy"].mean())
            rows[-1][f"{condition}_worst_accuracy"] = float(condition_group["accuracy"].min())
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.reports_root.glob("robustness_severity_*.json")):
        rows.extend(load_rows(path))

    long_frame = pd.DataFrame(rows)
    args.output_long_csv.parent.mkdir(parents=True, exist_ok=True)
    long_frame.to_csv(args.output_long_csv, index=False)

    summary = summarize(long_frame) if not long_frame.empty else pd.DataFrame()
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)

    Console().print(f"Wrote {args.output_long_csv}")
    Console().print(f"Wrote {args.output_summary_csv}")
    if not summary.empty:
        Console().print(
            summary[
                [
                    "run_id",
                    "clean_accuracy",
                    "mean_corruption_accuracy",
                    "worst_corruption_accuracy",
                    "mean_accuracy_degradation",
                    "worst_accuracy_degradation",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
