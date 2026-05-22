from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap AUROC/AUPRC confidence intervals from exported prediction CSVs."
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("reports/predictions"))
    parser.add_argument("--output-csv", type=Path, default=Path("reports/bootstrap_auc_ci.csv"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def run_id_from_path(path: Path) -> str:
    return path.stem.removeprefix("predictions_")


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> tuple[float, float, float, int]:
    rng = np.random.default_rng(seed)
    point = float(metric_fn(y_true, y_prob))
    values = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sampled_true = y_true[indices]
        if len(np.unique(sampled_true)) < 2:
            continue
        values.append(float(metric_fn(sampled_true, y_prob[indices])))
    if not values:
        return point, float("nan"), float("nan"), 0
    lower, upper = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lower), float(upper), len(values)


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.predictions_root.glob("predictions_*.csv")):
        frame = pd.read_csv(path)
        frame = frame[frame["split"] == args.split].reset_index(drop=True)
        if frame.empty:
            continue
        y_true = frame["y_true"].to_numpy(dtype=int)
        y_prob = frame["y_prob"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            continue

        auroc, auroc_low, auroc_high, auroc_samples = bootstrap_metric(
            y_true,
            y_prob,
            roc_auc_score,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            alpha=args.alpha,
        )
        auprc, auprc_low, auprc_high, auprc_samples = bootstrap_metric(
            y_true,
            y_prob,
            average_precision_score,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + 1,
            alpha=args.alpha,
        )
        rows.append(
            {
                "run_id": run_id_from_path(path),
                "split": args.split,
                "n": len(frame),
                "positive_n": int(y_true.sum()),
                "negative_n": int(len(y_true) - y_true.sum()),
                "auroc": auroc,
                "auroc_ci_low": auroc_low,
                "auroc_ci_high": auroc_high,
                "auroc_bootstrap_samples": auroc_samples,
                "auprc": auprc,
                "auprc_ci_low": auprc_low,
                "auprc_ci_high": auprc_high,
                "auprc_bootstrap_samples": auprc_samples,
            }
        )

    output = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    Console().print(f"Wrote {args.output_csv}")
    if not output.empty:
        Console().print(
            output[
                [
                    "run_id",
                    "auroc",
                    "auroc_ci_low",
                    "auroc_ci_high",
                    "auprc",
                    "auprc_ci_low",
                    "auprc_ci_high",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
