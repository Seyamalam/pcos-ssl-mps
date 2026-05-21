from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ContrastivePCOSDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    @classmethod
    def from_split_csv(cls, split_csv: Path, split: str, transform) -> "ContrastivePCOSDataset":
        frame = pd.read_csv(split_csv)
        frame = frame[frame["split"] == split].reset_index(drop=True)
        return cls(frame=frame, transform=transform)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(Path(row["rel_path"])).convert("RGB")
        return self.transform(image), self.transform(image)
