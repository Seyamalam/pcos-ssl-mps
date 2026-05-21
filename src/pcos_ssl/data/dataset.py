from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PCOSImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    @classmethod
    def from_split_csv(
        cls,
        split_csv: Path,
        split: str,
        transform=None,
        label_fraction: float = 1.0,
        seed: int = 42,
    ) -> "PCOSImageDataset":
        frame = pd.read_csv(split_csv)
        frame = frame[frame["split"] == split].reset_index(drop=True)
        if split == "train" and label_fraction < 1.0:
            if not 0.0 < label_fraction <= 1.0:
                raise ValueError("label_fraction must be in (0, 1].")
            groups = (
                frame.groupby("exact_group_id")
                .agg(label=("label", "first"), group_size=("image_id", "size"))
                .reset_index()
            )
            sampled_groups = (
                groups.groupby("label", group_keys=False)
                .sample(frac=label_fraction, random_state=seed)
                .reset_index(drop=True)
            )
            frame = frame[frame["exact_group_id"].isin(sampled_groups["exact_group_id"])].reset_index(
                drop=True
            )
        return cls(frame=frame, transform=transform)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(Path(row["rel_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])
        return image, label
