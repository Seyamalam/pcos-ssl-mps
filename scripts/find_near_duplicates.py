from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console

from pcos_ssl.data.near_duplicates import build_phash_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group near-duplicate images with perceptual hashes.")
    parser.add_argument("--images-csv", type=Path, default=Path("metadata/images.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("metadata/near_duplicate_groups_phash.csv"))
    parser.add_argument("--threshold", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = pd.read_csv(args.images_csv)
    groups = build_phash_groups(images, threshold=args.threshold)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    groups.to_csv(args.output_csv, index=False)

    summary = groups.drop_duplicates("near_group_id")
    duplicate_summary = summary[summary["group_size"] > 1]
    cross_class = duplicate_summary[duplicate_summary["labels"] > 1]

    console = Console()
    console.print(f"Wrote {args.output_csv}")
    console.print(f"Near-duplicate groups: {len(duplicate_summary):,}")
    console.print(
        f"Near-duplicate files beyond first: "
        f"{int((duplicate_summary['group_size'] - 1).sum()):,}"
    )
    console.print(f"Cross-class near-duplicate groups: {len(cross_class):,}")


if __name__ == "__main__":
    main()

