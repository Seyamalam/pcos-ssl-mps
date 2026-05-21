from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect validation-selected threshold results.")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("reports/threshold_summary.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.reports_root.glob("thresholds_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        run_id = path.stem.removeprefix("thresholds_")
        default_metrics = payload["test_default"]
        for objective, selected_metrics in payload[
            "test_at_validation_selected_thresholds"
        ].items():
            selection = payload["validation_selected"][objective]
            rows.append(
                {
                    "run_id": run_id,
                    "objective": objective,
                    "selected_threshold": selection["selected_threshold"],
                    "default_accuracy": default_metrics["accuracy"],
                    "default_balanced_accuracy": default_metrics["balanced_accuracy"],
                    "default_sensitivity": default_metrics["recall_sensitivity"],
                    "default_specificity": default_metrics["specificity"],
                    "default_f1": default_metrics["f1"],
                    "default_auroc": default_metrics["auroc"],
                    "selected_accuracy": selected_metrics["accuracy"],
                    "selected_balanced_accuracy": selected_metrics["balanced_accuracy"],
                    "selected_sensitivity": selected_metrics["recall_sensitivity"],
                    "selected_specificity": selected_metrics["specificity"],
                    "selected_f1": selected_metrics["f1"],
                    "selected_auroc": selected_metrics["auroc"],
                }
            )

    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    Console().print(f"Wrote {args.output_csv}")
    if not frame.empty:
        best = frame[frame["objective"] == "balanced_accuracy"]
        columns = [
            "run_id",
            "selected_threshold",
            "default_accuracy",
            "selected_accuracy",
            "default_sensitivity",
            "selected_sensitivity",
            "selected_specificity",
            "selected_auroc",
        ]
        Console().print(best[columns].to_string(index=False))


if __name__ == "__main__":
    main()
