from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect test metrics from run directories.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-csv", type=Path, default=Path("reports/all_test_results.csv"))
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in sorted(args.runs_root.glob("*/test_metrics.json")):
        run_dir = metrics_path.parent
        if not args.include_smoke and "smoke" in run_dir.name:
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        checkpoint_path = run_dir / "best_model.pt"
        row = {"run_id": run_dir.name, "metrics_path": str(metrics_path)}
        row.update(metrics)
        if checkpoint_path.exists():
            row["checkpoint_path"] = str(checkpoint_path)
        rows.append(row)

    frame = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    Console().print(f"Wrote {args.output_csv}")
    if not frame.empty:
        columns = [
            column
            for column in ["run_id", "accuracy", "auroc", "f1", "ece", "best_epoch"]
            if column in frame.columns
        ]
        Console().print(frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
