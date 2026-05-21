from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _split_groups(
    groups: pd.DataFrame,
    test_size: float,
    val_size: float,
    seed: int,
) -> pd.DataFrame:
    train_val, test = train_test_split(
        groups,
        test_size=test_size,
        random_state=seed,
        stratify=groups["label"],
    )
    adjusted_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=train_val["label"],
    )

    split_groups = pd.concat(
        [
            train.assign(split="train"),
            val.assign(split="val"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    return split_groups[["exact_group_id", "split"]]


def make_duplicate_aware_split(
    images_csv: Path,
    output_csv: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    images = pd.read_csv(images_csv)
    readable = images[images["readable"] & images["exact_group_id"].notna()].copy()

    group_labels = readable.groupby("exact_group_id").agg(
        label=("label", "first"),
        class_name=("class_name", "first"),
        group_size=("image_id", "size"),
    )
    cross_label_groups = readable.groupby("exact_group_id")["label"].nunique()
    bad_groups = cross_label_groups[cross_label_groups > 1]
    if not bad_groups.empty:
        raise ValueError(f"Found cross-label duplicate groups: {bad_groups.index.tolist()[:10]}")

    split_groups = _split_groups(
        groups=group_labels.reset_index(),
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    split_images = readable.merge(split_groups, on="exact_group_id", how="left")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_images.to_csv(output_csv, index=False)
    return split_images


def make_near_duplicate_aware_split(
    near_groups_csv: Path,
    output_csv: Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    exclude_cross_label_groups: bool = True,
) -> pd.DataFrame:
    images = pd.read_csv(near_groups_csv)
    if exclude_cross_label_groups:
        label_counts = images.groupby("near_group_id")["label"].nunique()
        valid_groups = label_counts[label_counts == 1].index
        images = images[images["near_group_id"].isin(valid_groups)].reset_index(drop=True)

    group_labels = images.groupby("near_group_id").agg(
        label=("label", "first"),
        class_name=("class_name", "first"),
        group_size=("image_id", "size"),
    )
    cross_label_groups = images.groupby("near_group_id")["label"].nunique()
    bad_groups = cross_label_groups[cross_label_groups > 1]
    if not bad_groups.empty:
        raise ValueError(f"Found cross-label near-duplicate groups: {bad_groups.index.tolist()[:10]}")

    groups = group_labels.reset_index().rename(columns={"near_group_id": "exact_group_id"})
    split_groups = _split_groups(
        groups=groups,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    ).rename(columns={"exact_group_id": "near_group_id"})

    split_images = images.merge(split_groups, on="near_group_id", how="left")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_images.to_csv(output_csv, index=False)
    return split_images


def summarize_split(split_images: pd.DataFrame) -> pd.DataFrame:
    group_column = "near_group_id" if "near_group_id" in split_images.columns else "exact_group_id"
    return (
        split_images.groupby(["split", "class_name"])
        .agg(images=("image_id", "size"), duplicate_groups=(group_column, "nunique"))
        .reset_index()
        .sort_values(["split", "class_name"])
    )
