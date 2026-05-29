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
    
class MultiHotDatasetWithPatientLookBack(Dataset):
    def __init__(
        self,
        multi_hot_data_frame: pl.DataFrame,
        *,
        target_col: str,
        feature_cols: list[str],
        patient_id_col: str,
        time_col: str,
        look_back: int = 3,
        dtype: torch.dtype = torch.float32,
    ):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.patient_id_col = patient_id_col
        self.time_col = time_col
        self.look_back = look_back
        self.dtype = dtype

        self.data_frame = multi_hot_data_frame.sort(
            [patient_id_col, time_col]
        )

        self.samples = []

        for patient_id, patient_df in self.data_frame.group_by(patient_id_col):
            n_visits = patient_df.height

            for visit_idx in range(n_visits):
                start_idx = max(0, visit_idx - look_back)

                self.samples.append(
                    {
                        "patient_df": patient_df,
                        "start_idx": start_idx,
                        "end_idx": visit_idx + 1,
                        "target_idx": visit_idx,
                    }
                )
                
    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        patient_df = sample["patient_df"]
        start_idx = sample["start_idx"]
        end_idx = sample["end_idx"]
        target_idx = sample["target_idx"]

        history_df = patient_df.slice(start_idx, end_idx - start_idx)
        target_row = patient_df.row(target_idx, named=True)

        x = {
            "diagnoses": torch.tensor(
                history_df["diagnosis_multihot"].to_list(),
                dtype=self.dtype,
            ),
            "procedures": torch.tensor(
                history_df["procedure_multihot"].to_list(),
                dtype=self.dtype,
            ),
            "medication_history": torch.tensor(
                history_df[self.target_col].to_list(),
                dtype=self.dtype,
            ),
        }

        y = torch.tensor(
            target_row[self.target_col],
            dtype=self.dtype,
        )

        return x, y
    
    def __len__(self):
        return len(self.samples)