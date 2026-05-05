from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset


class MultiHotDataset(Dataset):
    def __init__(
        self,
        multi_hot_data_frame: pl.DataFrame,
        *,
        target_col: str,
        feature_cols: list[str],
        dtype: torch.dtype = torch.float32,
    ):
        self.data_frame = multi_hot_data_frame
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.dtype = dtype

    def __len__(self) -> int:
        return self.data_frame.height

    def __getitem__(self, idx: int):
        row = self.data_frame.row(idx, named=True)

        features = self._flatten_values([row[col] for col in self.feature_cols])
        target = self._flatten_values([row[self.target_col]])

        x = torch.tensor(features, dtype=self.dtype)
        y = torch.tensor(target, dtype=self.dtype)

        return x, y
    
    @staticmethod
    def _flatten_values(values: list[Any]) -> list[float]:
        flattened = []

        for value in values:
            if isinstance(value, (list, tuple)):
                flattened.extend(value)
            else:
                flattened.append(value)

        return flattened