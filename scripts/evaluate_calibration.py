from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from pcos_ssl.training.metrics import expected_calibration_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Brier score and calibration plots from exported prediction CSVs."
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("reports/predictions"))
    parser.add_argument("--output-csv", type=Path, default=Path("reports/calibration_summary.csv"))
    parser.add_argument("--output-figure", type=Path, default=Path("reports/figures/calibration_curves.png"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n-bins", type=int, default=10)
    return parser.parse_args()


def run_id_from_path(path: Path) -> str:
    return path.stem.removeprefix("predictions_")


def main() -> None:
    args = parse_args()
    rows = []
    curves = []
    for path in sorted(args.predictions_root.glob("predictions_*.csv")):
        frame = pd.read_csv(path)
        frame = frame[frame["split"] == args.split].reset_index(drop=True)
        if frame.empty:
            continue
        y_true = frame["y_true"].to_numpy(dtype=int)
        y_prob = frame["y_prob"].to_numpy(dtype=float)
        run_id = run_id_from_path(path)
        brier = float(brier_score_loss(y_true, y_prob))
        ece = float(expected_calibration_error(y_true, y_prob, n_bins=args.n_bins))
        rows.append(
            {
                "run_id": run_id,
                "split": args.split,
                "n": len(frame),
                "positive_n": int(y_true.sum()),
                "negative_n": int(len(frame) - y_true.sum()),
                "brier_score": brier,
                "ece": ece,
                "mean_probability": float(np.mean(y_prob)),
                "positive_prevalence": float(np.mean(y_true)),
            }
        )
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=args.n_bins, strategy="quantile"
        )
        curves.append((run_id, prob_pred, prob_true))

    summary = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Perfect calibration")
    for run_id, prob_pred, prob_true in curves:
        ax.plot(prob_pred, prob_true, marker="o", linewidth=1.6, label=run_id)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed PCOS fraction")
    ax.set_title("Calibration curves on pHash-aware test split")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)

    Console().print(f"Wrote {args.output_csv}")
    Console().print(f"Wrote {args.output_figure}")
    if not summary.empty:
        Console().print(summary[["run_id", "brier_score", "ece"]].to_string(index=False))


if __name__ == "__main__":
    main()
