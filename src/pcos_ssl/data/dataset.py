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
    def from_split_csv(cls, split_csv: Path, split: str, transform=None) -> "PCOSImageDataset":
        frame = pd.read_csv(split_csv)
        frame = frame[frame["split"] == split].reset_index(drop=True)
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
