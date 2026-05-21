from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from pcos_ssl.data.splits import make_near_duplicate_aware_split, summarize_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pHash near-duplicate-aware splits.")
    parser.add_argument(
        "--near-groups-csv",
        type=Path,
        default=Path("metadata/near_duplicate_groups_phash.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("metadata/splits_near_duplicate_aware_phash.csv"),
    )
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-cross-label-groups", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_images = make_near_duplicate_aware_split(
        near_groups_csv=args.near_groups_csv,
        output_csv=args.output_csv,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        exclude_cross_label_groups=not args.keep_cross_label_groups,
    )
    console = Console()
    console.print(f"Wrote {args.output_csv}")
    console.print(summarize_split(split_images).to_string(index=False))


if __name__ == "__main__":
    main()

