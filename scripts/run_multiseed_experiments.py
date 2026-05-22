from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or plan key repeated-seed experiments.")
    parser.add_argument("--split-csv", type=Path, default=Path("metadata/splits_near_duplicate_aware_phash.csv"))
    parser.add_argument("--simclr-checkpoint", type=Path, default=Path("runs/simclr_resnet18_phash_e25/encoder_last.pt"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123])
    parser.add_argument("--label-fractions", type=float, nargs="+", default=[0.10, 0.50])
    parser.add_argument("--epochs", type=int, nargs="+", default=[10])
    parser.add_argument("--methods", type=str, nargs="+", default=["supervised", "simclr"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-plan-csv", type=Path, default=Path("reports/multiseed_plan.csv"))
    return parser.parse_args()


def fraction_token(fraction: float) -> str:
    return f"{int(round(fraction * 100)):02d}pct"


def run_id(method: str, fraction: float, epochs: int, seed: int) -> str:
    token = fraction_token(fraction)
    seed_suffix = "" if seed == 42 else f"_seed{seed}"
    if method == "supervised":
        return f"resnet18_supervised_phash_{token}{seed_suffix}_e{epochs}"
    if method == "simclr":
        return f"simclr_resnet18_phash_e25_finetune_{token}{seed_suffix}_e{epochs}"
    raise ValueError(f"Unknown method: {method}")


def command_for(args: argparse.Namespace, method: str, fraction: float, epochs: int, seed: int, out: Path) -> list[str]:
    if method == "supervised":
        return [
            "uv",
            "run",
            "python",
            "scripts/train_supervised.py",
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--pretrained",
            "--split-csv",
            str(args.split_csv),
            "--backbone",
            "resnet18",
            "--label-fraction",
            str(fraction),
            "--num-workers",
            str(args.num_workers),
            "--output-dir",
            str(out),
        ]
    if method == "simclr":
        return [
            "uv",
            "run",
            "python",
            "scripts/finetune_simclr.py",
            "--epochs",
            str(epochs),
            "--seed",
            str(seed),
            "--checkpoint",
            str(args.simclr_checkpoint),
            "--split-csv",
            str(args.split_csv),
            "--backbone",
            "resnet18",
            "--label-fraction",
            str(fraction),
            "--no-freeze-encoder",
            "--num-workers",
            str(args.num_workers),
            "--output-dir",
            str(out),
        ]
    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    args = parse_args()
    console = Console()
    rows = []
    for method in args.methods:
        for fraction in args.label_fractions:
            for epochs in args.epochs:
                for seed in args.seeds:
                    rid = run_id(method, fraction, epochs, seed)
                    out = Path("runs") / rid
                    metrics = out / "test_metrics.json"
                    command = command_for(args, method, fraction, epochs, seed, out)
                    status = "done" if metrics.exists() else "pending"
                    if args.execute and not metrics.exists():
                        console.print(f"Running {rid}")
                        subprocess.run(command, check=True)
                        status = "done" if metrics.exists() else "failed"
                    elif args.execute:
                        console.print(f"Skipping {rid}: already has {metrics}")
                    rows.append(
                        {
                            "run_id": rid,
                            "method": method,
                            "label_fraction": fraction,
                            "epochs": epochs,
                            "seed": seed,
                            "output_dir": str(out),
                            "status": status,
                            "command": " ".join(command),
                        }
                    )

    plan = pd.DataFrame(rows)
    args.output_plan_csv.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.output_plan_csv, index=False)
    console.print(f"Wrote {args.output_plan_csv}")
    console.print(plan[["run_id", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
