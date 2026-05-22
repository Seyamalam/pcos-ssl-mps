from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from pcos_ssl.training.metrics import expected_calibration_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit validation-only probability calibration and evaluate on test predictions."
    )
    parser.add_argument("--predictions-root", type=Path, default=Path("reports/predictions"))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("reports/calibration_improvement.csv")
    )
    parser.add_argument(
        "--output-figure", type=Path, default=Path("reports/figures/calibration_improvement.png")
    )
    parser.add_argument("--n-bins", type=int, default=10)
    return parser.parse_args()


def run_id_from_path(path: Path) -> str:
    return path.stem.removeprefix("predictions_")


def clip_prob(prob: np.ndarray) -> np.ndarray:
    return np.clip(prob, 1e-6, 1.0 - 1e-6)


def logit(prob: np.ndarray) -> np.ndarray:
    prob = clip_prob(prob)
    return np.log(prob / (1.0 - prob))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> dict[str, float]:
    y_prob = clip_prob(y_prob)
    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob, n_bins=n_bins)),
        "nll": float(log_loss(y_true, y_prob)),
    }


def fit_temperature(val_true: np.ndarray, val_prob: np.ndarray) -> float:
    logits = logit(val_prob)
    candidates = np.geomspace(0.05, 10.0, 240)
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in candidates:
        calibrated = sigmoid(logits / temperature)
        nll = log_loss(val_true, clip_prob(calibrated))
        if nll < best_nll:
            best_nll = float(nll)
            best_temperature = float(temperature)
    return best_temperature


def main() -> None:
    args = parse_args()
    rows = []
    curves = []
    for path in sorted(args.predictions_root.glob("predictions_*.csv")):
        frame = pd.read_csv(path)
        run_id = run_id_from_path(path)
        val = frame[frame["split"] == "val"].reset_index(drop=True)
        test = frame[frame["split"] == "test"].reset_index(drop=True)
        if val.empty or test.empty:
            continue

        val_true = val["y_true"].to_numpy(dtype=int)
        val_prob = val["y_prob"].to_numpy(dtype=float)
        test_true = test["y_true"].to_numpy(dtype=int)
        test_prob = test["y_prob"].to_numpy(dtype=float)
        test_logit = logit(test_prob)

        platt = LogisticRegression(C=1e6, solver="lbfgs")
        platt.fit(logit(val_prob).reshape(-1, 1), val_true)
        platt_prob = platt.predict_proba(test_logit.reshape(-1, 1))[:, 1]

        temperature = fit_temperature(val_true, val_prob)
        temperature_prob = sigmoid(test_logit / temperature)

        methods = {
            "uncalibrated": (test_prob, None),
            "platt": (platt_prob, None),
            "temperature": (temperature_prob, temperature),
        }
        for method, (prob, fitted_temperature) in methods.items():
            row = {
                "run_id": run_id,
                "method": method,
                "temperature": fitted_temperature,
                "platt_intercept": (
                    float(platt.intercept_[0]) if method == "platt" else None
                ),
                "platt_coef": float(platt.coef_[0][0]) if method == "platt" else None,
                **metrics(test_true, prob, args.n_bins),
            }
            rows.append(row)
            prob_true, prob_pred = calibration_curve(
                test_true, prob, n_bins=args.n_bins, strategy="quantile"
            )
            curves.append((run_id, method, prob_pred, prob_true))

    output = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)

    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    for run_id, method, prob_pred, prob_true in curves:
        if method in {"uncalibrated", "platt"}:
            axes[0].plot(prob_pred, prob_true, marker="o", linewidth=1.2, label=f"{run_id} {method}")
    axes[0].set_title("Calibration curves: uncalibrated vs Platt")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed PCOS fraction")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.25)

    pivot = output.pivot(index="run_id", columns="method", values="brier_score")
    pivot.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Brier score after validation-fitted calibration")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Brier score")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)

    Console().print(f"Wrote {args.output_csv}")
    Console().print(f"Wrote {args.output_figure}")
    if not output.empty:
        Console().print(
            output[["run_id", "method", "brier_score", "ece", "nll"]].to_string(index=False)
        )


if __name__ == "__main__":
    main()
