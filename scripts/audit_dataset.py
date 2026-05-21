from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from pcos_ssl.data.audit import write_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the PCOS ultrasound image dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("PCOS"))
    parser.add_argument("--output-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--workers", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    images_path, duplicate_groups_path = write_audit(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        workers=args.workers,
    )

    images = pd.read_csv(images_path)
    duplicates = pd.read_csv(duplicate_groups_path)

    table = Table(title="Dataset Audit Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Images", f"{len(images):,}")
    table.add_row("Readable images", f"{int(images['readable'].sum()):,}")
    table.add_row("Unreadable images", f"{int((~images['readable']).sum()):,}")
    table.add_row("Exact duplicate groups", f"{len(duplicates):,}")
    table.add_row("Duplicate files beyond first", f"{int((duplicates['group_size'] - 1).sum()):,}")
    table.add_row("Images CSV", str(images_path))
    table.add_row("Duplicate groups CSV", str(duplicate_groups_path))
    console.print(table)


if __name__ == "__main__":
    main()

