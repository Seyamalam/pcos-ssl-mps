from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Wilson confidence intervals for threshold-selected test metrics."
    )
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/threshold_confidence_intervals.csv"),
    )
    parser.add_argument("--objective", type=str, default="balanced_accuracy")
    parser.add_argument("--z", type=float, default=1.96)
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    phat = successes / total
    denominator = 1.0 + z**2 / total
    center = (phat + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * total)) / total)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def add_metric_interval(
    row: dict[str, float | int | str],
    prefix: str,
    estimate: float,
    successes: int,
    total: int,
    z: float,
) -> None:
    lower, upper = wilson_interval(successes, total, z)
    row[prefix] = estimate
    row[f"{prefix}_ci_low"] = lower
    row[f"{prefix}_ci_high"] = upper
    row[f"{prefix}_n"] = total


def run_label_from_id(run_id: str) -> float | None:
    for token, value in [("05pct", 0.05), ("5pct", 0.05), ("10pct", 0.10), ("25pct", 0.25), ("50pct", 0.50)]:
        if token in run_id:
            return value
    return None


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.reports_root.glob("thresholds_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        run_id = path.stem.removeprefix("thresholds_")
        selected_by_objective = payload["test_at_validation_selected_thresholds"]
        if args.objective not in selected_by_objective:
            continue

        selected = selected_by_objective[args.objective]
        default = payload["test_default"]
        tn = int(selected["tn"])
        fp = int(selected["fp"])
        fn = int(selected["fn"])
        tp = int(selected["tp"])
        total = tn + fp + fn + tp

        row: dict[str, float | int | str | None] = {
            "run_id": run_id,
            "objective": args.objective,
            "label_fraction": run_label_from_id(run_id),
            "selected_threshold": payload["validation_selected"][args.objective][
                "selected_threshold"
            ],
            "default_accuracy": default["accuracy"],
            "default_sensitivity": default["recall_sensitivity"],
            "default_specificity": default["specificity"],
            "auroc": selected["auroc"],
            "auprc": selected["auprc"],
        }
        add_metric_interval(
            row,
            "selected_accuracy",
            float(selected["accuracy"]),
            successes=tp + tn,
            total=total,
            z=args.z,
        )
        add_metric_interval(
            row,
            "selected_sensitivity",
            float(selected["recall_sensitivity"]),
            successes=tp,
            total=tp + fn,
            z=args.z,
        )
        add_metric_interval(
            row,
            "selected_specificity",
            float(selected["specificity"]),
            successes=tn,
            total=tn + fp,
            z=args.z,
        )
        rows.append(row)

    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    Console().print(f"Wrote {args.output_csv}")
    if not frame.empty:
        columns = [
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
        Console().print(frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
