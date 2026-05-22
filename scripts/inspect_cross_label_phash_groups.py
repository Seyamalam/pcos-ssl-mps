from __future__ import annotations

import argparse
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from rich.console import Console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect cross-label pHash near-duplicate groups.")
    parser.add_argument(
        "--near-groups-csv",
        type=Path,
        default=Path("metadata/near_duplicate_groups_phash.csv"),
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("metadata/splits_near_duplicate_aware_phash.csv"),
    )
    parser.add_argument("--output-csv", type=Path, default=Path("reports/cross_label_phash_groups.csv"))
    parser.add_argument(
        "--output-examples-csv",
        type=Path,
        default=Path("reports/cross_label_phash_examples.csv"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("reports/figures/cross_label_phash_examples.png"),
    )
    parser.add_argument("--max-groups-to-plot", type=int, default=8)
    return parser.parse_args()


def hamming_distance(left: str, right: str) -> int:
    return int(imagehash.hex_to_hash(left) - imagehash.hex_to_hash(right))


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.near_groups_csv)
    split = pd.read_csv(args.split_csv)[["image_id", "split"]]
    frame = frame.drop(columns=["split"], errors="ignore").merge(split, on="image_id", how="left")
    cross = frame.groupby("near_group_id").filter(lambda group: group["label"].nunique() > 1)

    rows = []
    examples = []
    for near_group_id, group in cross.groupby("near_group_id"):
        infected = group[group["label"] == 1]
        noninfected = group[group["label"] == 0]
        distances = []
        for _, left in infected.iterrows():
            for _, right in noninfected.iterrows():
                distances.append(hamming_distance(left["phash"], right["phash"]))
        rows.append(
            {
                "near_group_id": near_group_id,
                "group_size": len(group),
                "infected_count": len(infected),
                "noninfected_count": len(noninfected),
                "exact_group_count": group["exact_group_id"].nunique(),
                "split_values": (
                    ",".join(sorted(group["split"].dropna().unique()))
                    if group["split"].notna().any()
                    else "excluded_from_strict_split"
                ),
                "min_cross_label_phash_distance": min(distances) if distances else None,
                "max_cross_label_phash_distance": max(distances) if distances else None,
                "representative_infected": infected["rel_path"].iloc[0],
                "representative_noninfected": noninfected["rel_path"].iloc[0],
            }
        )
        for _, row in group.sort_values(["label", "rel_path"]).iterrows():
            examples.append(
                {
                    "near_group_id": near_group_id,
                    "image_id": row["image_id"],
                    "rel_path": row["rel_path"],
                    "label": row["label"],
                    "class_name": row["class_name"],
                    "phash": row["phash"],
                    "exact_group_id": row["exact_group_id"],
                    "split": row.get("split"),
                }
            )

    summary = pd.DataFrame(rows).sort_values(
        ["min_cross_label_phash_distance", "group_size"], ascending=[True, False]
    )
    example_frame = pd.DataFrame(examples)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    example_frame.to_csv(args.output_examples_csv, index=False)

    plot_groups = summary.head(args.max_groups_to_plot)
    fig, axes = plt.subplots(len(plot_groups), 2, figsize=(7, max(2, 2.5 * len(plot_groups))))
    if len(plot_groups) == 1:
        axes = [axes]
    for axis_row, (_, row) in zip(axes, plot_groups.iterrows()):
        for ax, path_key, title in [
            (axis_row[0], "representative_infected", "infected"),
            (axis_row[1], "representative_noninfected", "noninfected"),
        ]:
            image = Image.open(row[path_key]).convert("RGB")
            ax.imshow(image)
            ax.set_title(
                f"{row['near_group_id']} {title}\nd={row['min_cross_label_phash_distance']}"
            )
            ax.axis("off")
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)

    Console().print(f"Wrote {args.output_csv}")
    Console().print(f"Wrote {args.output_examples_csv}")
    Console().print(f"Wrote {args.output_figure}")
    Console().print(f"Cross-label pHash groups: {len(summary)}")
    if not summary.empty:
        Console().print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
